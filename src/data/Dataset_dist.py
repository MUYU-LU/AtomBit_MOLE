import mindspore as ms
import pickle
import os
import time
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from mindspore.dataset import Dataset
import h5py
import numpy as np
from sharker.data import Graph


class _CompactMetadataView:
    """轻量级 metadata 视图，兼容 BinPackingSampler 的 dict-like 访问（item['num_atoms'] 等）。
    底层数据存储在共享的 numpy 数组中，单条访问返回一个轻量 dict。"""

    def __init__(self, file_ids, index_in_file, num_atoms, num_edges, unique_files):
        self._file_ids = file_ids
        self._index_in_file = index_in_file
        self._num_atoms = num_atoms
        self._num_edges = num_edges
        self._unique_files = unique_files

    def __len__(self):
        return len(self._file_ids)

    def __getitem__(self, idx):
        return {
            'file_path': self._unique_files[self._file_ids[idx]],
            'index_in_file': int(self._index_in_file[idx]),
            'num_atoms': int(self._num_atoms[idx]),
            'num_edges': int(self._num_edges[idx]),
        }

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class ChunkedSmartDataset_h5:
    def __init__(self, data_dir, metadata_file, rank=0, world_size=1, max_open_h5_per_thread=8):
        """
        :param data_dir: 数据目录
        :param metadata_file: 元数据文件名 (e.g., 'train_metadata.pt')

        内存优化：将 Python dict 列表转换为紧凑的 numpy 数组存储。
        原始 metadata 每条 ~500 bytes (Python dict overhead)，
        转换后每条 ~12 bytes (3 个 int32) + 文件名去重共享。
        """
        self.data_dir = data_dir

        # 优先加载 npz（秒级），fallback 到 pickle（分钟级）
        npz_file = metadata_file.rsplit('.', 1)[0] + '.npz'
        npz_path = os.path.join(data_dir, npz_file)
        pkl_path = os.path.join(data_dir, metadata_file)

        t0 = time.time()
        if os.path.exists(npz_path):
            if rank == 0:
                print(f"[Dataset] Loading npz: {npz_path} ...")
            data = np.load(npz_path, allow_pickle=False)
            file_ids = data['file_ids']
            index_in_file = data['index_in_file']
            num_atoms = data['num_atoms']
            num_edges = data['num_edges']
            unique_files = data['unique_files'].tolist()
            n = len(file_ids)
            if rank == 0:
                print(f"[Dataset] npz loaded: {n} samples in {time.time() - t0:.1f}s")
        else:
            if rank == 0:
                print(f"[Dataset] npz not found, loading pickle: {pkl_path} ...")
                print(f"[Dataset] TIP: python scripts/convert_metadata_to_npz.py {data_dir} {metadata_file}")
            with open(pkl_path, 'rb') as f:
                raw_metadata = pickle.load(f)
            if rank == 0:
                print(f"[Dataset] Pickle loaded: {len(raw_metadata)} samples in {time.time() - t0:.1f}s")

            n = len(raw_metadata)
            unique_files = []
            file_to_id = {}
            file_ids = np.empty(n, dtype=np.int32)
            index_in_file = np.empty(n, dtype=np.int32)
            num_atoms = np.empty(n, dtype=np.int32)
            num_edges = np.empty(n, dtype=np.int32)

            t0 = time.time()
            log_interval = max(n // 10, 1)
            for i, item in enumerate(raw_metadata):
                fp = item['file_path']
                if fp not in file_to_id:
                    file_to_id[fp] = len(unique_files)
                    unique_files.append(fp)
                file_ids[i] = file_to_id[fp]
                index_in_file[i] = item['index_in_file']
                num_atoms[i] = item['num_atoms']
                num_edges[i] = item['num_edges']
                if rank == 0 and (i + 1) % log_interval == 0:
                    print(f"[Dataset] Parsing: {i + 1}/{n} ({100*(i+1)/n:.0f}%) - {time.time()-t0:.1f}s")

            if rank == 0:
                print(f"[Dataset] Parsed: {n} samples, {len(unique_files)} files in {time.time()-t0:.1f}s")
            del raw_metadata, file_to_id

        self._file_ids = file_ids
        self._index_in_file = index_in_file
        self._num_atoms = num_atoms
        self._num_edges = num_edges
        self._unique_files = unique_files

        self.metadata = _CompactMetadataView(file_ids, index_in_file, num_atoms, num_edges, unique_files)
        self._local = threading.local()
        self._max_open_h5_per_thread = max(1, int(max_open_h5_per_thread))

        if rank == 0:
            mem_mb = (file_ids.nbytes + index_in_file.nbytes + num_atoms.nbytes + num_edges.nbytes) / 1024**2
            print(f"[Dataset] Ready: {n} samples, {len(unique_files)} files, {mem_mb:.1f} MB")

    def _get_h5_handle(self, full_path):
        """Thread-local h5py 文件句柄缓存，避免每次 __getitem__ 都 open/close。
        每个 worker 进程/线程持有独立的句柄，不会死锁。"""
        local = self._local
        cache = getattr(local, 'h5_cache', None)
        if cache is None:
            local.h5_cache = OrderedDict()
            cache = local.h5_cache
        f = cache.get(full_path)
        if f is None or not f.id.valid:
            if f is not None:
                cache.pop(full_path, None)
            f = h5py.File(full_path, 'r', swmr=True)
            cache[full_path] = f
            while len(cache) > self._max_open_h5_per_thread:
                old_path, old_file = cache.popitem(last=False)
                if old_path == full_path:
                    cache[old_path] = old_file
                    break
                try:
                    old_file.close()
                except Exception:
                    pass
        else:
            cache.move_to_end(full_path)
        return f

    def _read_one(self, idx):
        """读取单条数据，供并行调用。"""
        return self[idx]

    def __getitem__(self, idx):
        file_name = self._unique_files[self._file_ids[idx]]

        # 兼容性处理：如果你没重新生成 metadata，这里强制修正后缀
        if file_name.endswith('.pt'):
            file_name = file_name.replace('.pt', '.h5')

        inner_idx = int(self._index_in_file[idx])
        full_path = os.path.join(self.data_dir, file_name)

        try:
            f = self._get_h5_handle(full_path)
            # 获取指针位置
            a_start = f['atom_ptr'][inner_idx]
            a_end = f['atom_ptr'][inner_idx + 1]

            e_start = f['edge_ptr'][inner_idx]
            e_end = f['edge_ptr'][inner_idx + 1]

            # 读取数据 (Numpy Slicing)
            z = f['z'][a_start:a_end].astype(np.int64)
            pos = f['pos'][a_start:a_end]
            force = f['force'][a_start:a_end]

            edge_index = f['edge_index'][:, e_start:e_end].astype(np.int64)
            shifts_int = f['shifts_int'][e_start:e_end].astype(np.float32)

            # Graph 级属性
            # Use slice so h5py returns a 1-D ndarray (shape (1,)) instead of a
            # numpy scalar, which the sharker collator cannot batch into a tensor.
            y = f['y'][inner_idx:inner_idx + 1]
            # Use slice to preserve the leading dim so collation stacks to (N,3,3)
            # instead of concatenating to (N*3,3).
            cell = f['cell'][inner_idx:inner_idx + 1]
            stress = f['stress'][inner_idx:inner_idx + 1]

            spin = f['spin'][inner_idx]
            charge = f['charge'][inner_idx]
            dataset = f['dataset'][inner_idx].decode('utf-8')

            data = Graph(
                z=z, pos=pos, cell=cell,
                edge_index=edge_index, shifts_int=shifts_int,
                y=y, force=force,
                spin=spin, charge=charge, dataset=dataset,
                stress=stress,
            )

            if data.pos.dtype != np.float32: data.pos = data.pos.astype(np.float32)
            if data.y.dtype != np.float32: data.y = data.y.astype(np.float32)

            return data

        except Exception as e:
            print(f"Error reading {full_path} at idx {inner_idx}: {e}")
            return Graph()

    def __len__(self):
        return len(self._file_ids)

class MultiSourceGraphDataset:
    """
    把多个 indexable dataset/list 合成一个虚拟大 dataset。
    global_idx -> (src_id, local_idx)
    """
    def __init__(self, datasets, num_io_threads=4):
        """
        datasets: list 或 dict
          - list: [dsA, dsB, dsC]，src_id=0,1,2
          - dict: {'A': dsA, 'B': dsB, 'C': dsC}，src_id='A','B','C'
        num_io_threads: 批量读取时的并行线程数 (h5py 释放 GIL，线程并行有效)
        """
        self.is_dict = isinstance(datasets, dict)
        if self.is_dict:
            self.keys = list(datasets.keys())
            self.ds_list = [datasets[k] for k in self.keys]
            self.src_ids = self.keys
        else:
            self.ds_list = list(datasets)
            self.src_ids = list(range(len(self.ds_list)))

        # 用 numpy 数组存 offsets，支持 searchsorted 二分查找
        self.offsets = []
        off = 0
        for ds in self.ds_list:
            self.offsets.append(off)
            off += len(ds)
        self.total_len = off
        self._offsets_arr = np.array(self.offsets + [off], dtype=np.int64)

        self._num_io_threads = num_io_threads
        self._thread_pool = None

    def _get_pool(self):
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(max_workers=self._num_io_threads)
        return self._thread_pool

    def close(self):
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=False, cancel_futures=True)
            self._thread_pool = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self):
        return self.total_len

    def _locate(self, global_idx: int):
        # 二分查找，O(log k) 其中 k 是数据集数量
        src_i = int(np.searchsorted(self._offsets_arr[1:], global_idx, side='right'))
        local_idx = global_idx - self.offsets[src_i]
        return src_i, local_idx

    def _get_one(self, global_idx: int):
        src_i, local_idx = self._locate(global_idx)
        return self.ds_list[src_i][local_idx]

    def __getitem__(self, idx):
        # sampler 会传 list[int]（一个 batch）
        if isinstance(idx, list):
            if len(idx) == 1:
                return [self._get_one(idx[0])]
            # h5py 释放 GIL，线程并行读取有效
            pool = self._get_pool()
            return list(pool.map(self._get_one, idx))
        return self._get_one(idx)

