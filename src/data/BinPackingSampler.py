import bisect
import random
import time
import numpy as np
from tqdm import tqdm
from mindspore.dataset import Sampler


class MultiDatasetBinPackingSampler(Sampler):
    def __init__(
        self,
        metadatas,          # list[meta_list] 或 dict[name->meta_list]
        max_cost=3000,
        edge_weight='auto',
        shuffle=True,
        world_size=1,
        rank=0,
        seed=42,
        max_graph_per_batch=None,
    ):
        self.shuffle = shuffle
        self.max_cost = float(max_cost)
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.max_graph_per_batch = max_graph_per_batch

        # ---- 统一 key/顺序 ----
        if isinstance(metadatas, dict):
            self.keys = list(metadatas.keys())
        else:
            self.keys = list(range(len(metadatas)))
            metadatas = {k: metadatas[k] for k in self.keys}

        # ---- offsets ----
        self.offsets = {}
        off = 0
        for k in self.keys:
            self.offsets[k] = off
            off += len(metadatas[k])
        self.total_samples = off

        # ---- 向量化: 收集所有 num_atoms / num_edges 到 numpy 数组 ----
        if self.rank == 0:
            print(f"[Sampler] Collecting metadata for {self.total_samples} samples ...")
        t0 = time.time()
        all_atoms = np.empty(self.total_samples, dtype=np.float32)
        all_edges = np.empty(self.total_samples, dtype=np.float32)
        for k in self.keys:
            off_k = self.offsets[k]
            meta = metadatas[k]
            n_k = len(meta)
            if self.rank == 0:
                print(f"[Sampler]   source {k}: {n_k} samples (offset {off_k})")
            # 优先直接访问底层 numpy 数组 (_CompactMetadataView)
            if hasattr(meta, '_num_atoms'):
                all_atoms[off_k:off_k + n_k] = meta._num_atoms
                all_edges[off_k:off_k + n_k] = meta._num_edges
            else:
                for i, item in enumerate(meta):
                    all_atoms[off_k + i] = item['num_atoms']
                    all_edges[off_k + i] = item['num_edges']
        if self.rank == 0:
            print(f"[Sampler] Metadata collected in {time.time() - t0:.1f}s")

        # ---- 计算 edge_weight (向量化) ----
        if edge_weight == 'auto':
            total_a = float(all_atoms.sum())
            total_e = float(all_edges.sum())
            self.edge_weight = (total_a / total_e) if total_e > 0 else 0.0
            if self.rank == 0:
                print(f"[Auto-Balance] edge_weight={self.edge_weight:.6f}")
        else:
            self.edge_weight = float(edge_weight)

        # ---- 向量化计算 cost，按数据集分别排序 ----
        if self.rank == 0:
            print(f"[Sampler] Computing costs and sorting per dataset ...")
        t0 = time.time()
        ew = self.edge_weight
        costs = all_atoms + ew * all_edges  # shape (N,), float32

        # 保存每个数据集自己的升序排列，用于 dataset-aware bin-packing
        self._per_ds_costs = {}
        self._per_ds_gidxs = {}
        for k in self.keys:
            off_k = self.offsets[k]
            n_k = len(metadatas[k])
            ds_costs = costs[off_k:off_k + n_k]
            ds_order = np.argsort(ds_costs)
            self._per_ds_costs[k] = ds_costs[ds_order].astype(np.float32)
            self._per_ds_gidxs[k] = (off_k + ds_order).astype(np.int64)

        # 保留全局排序供兼容（_batch_source_stats 等不依赖它打包）
        order = np.argsort(costs)
        self._sorted_costs = costs[order]
        self._sorted_gidxs = order.astype(np.int64)
        if self.rank == 0:
            print(f"[Sampler] Sorted {self.total_samples} samples in {time.time() - t0:.1f}s")

        del all_atoms, all_edges, costs, order

        self._threshold = 0.95 * self.max_cost
        self._max_graph_int = int(max_graph_per_batch) if max_graph_per_batch is not None else None

        # 预计算排好序的 offset 列表，供 _batch_source_stats 二分查找
        sorted_pairs = sorted((self.offsets[k], k) for k in self.keys)
        self._sorted_offsets = [p[0] for p in sorted_pairs]
        self._sorted_offset_keys = [p[1] for p in sorted_pairs]

        self._base_batches = None
        self._cached_epoch = None
        self._cached_batches = None
        self._cached_step_counts = {}

        if self.rank == 0:
            print(f"[BinPackingSampler] {self.total_samples} samples indexed")

        super().__init__()

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self._cached_batches = self._generate_batches(self.epoch)
        self._cached_epoch = self.epoch
        self._cached_step_counts[self.epoch] = len(self._cached_batches)

    def _get_base_batches(self):
        if self._base_batches is None:
            self._base_batches = self._compute_base_batches()
        return self._base_batches

    def _compute_base_batches(self):
        """Dataset-aware bin-packing:
        按数据集大小比例加权随机选数据集取 seed（最大 item），
        然后按同样比例随机填充（最小 item），使 batch 尽量混合多个数据集。
        随机序列由 numpy 批量预生成，避免逐条调用 Python random，减少开销。
        """
        if self.rank == 0:
            print(f"[Sampler] Dataset-aware bin-packing {self.total_samples} samples ...")
        t0 = time.time()

        keys = self.keys
        max_cost = self.max_cost
        threshold = self._threshold
        max_graph_int = self._max_graph_int

        # 每个数据集的左右指针（左=小cost端，右=大cost端）
        lefts  = {k: 0 for k in keys}
        rights = {k: len(self._per_ds_costs[k]) - 1 for k in keys}

        def _has_items(k):
            return lefts[k] <= rights[k]

        # 按初始数据集大小比例，批量预生成随机数据集选择序列
        # 3x 冗余量足以覆盖所有跳过（exhausted / cost 超限）
        ds_sizes = np.array([len(self._per_ds_costs[k]) for k in keys], dtype=np.float64)
        ds_probs = ds_sizes / ds_sizes.sum()
        np_rng = np.random.default_rng(self.seed)
        pool = np_rng.choice(len(keys), size=self.total_samples * 3, p=ds_probs)
        pool_pos = 0

        total_remaining = self.total_samples
        batches = []
        pbar = tqdm(total=total_remaining, desc="Bin-packing", unit="samples", disable=(self.rank != 0))

        while total_remaining > 0:
            # 1. Seed：从 pool 找下一个有剩余 item 的数据集，取其最大 item
            seed_k = None
            while pool_pos < len(pool):
                k = keys[int(pool[pool_pos])]; pool_pos += 1
                if _has_items(k):
                    seed_k = k; break
            if seed_k is None:
                break

            batch = [int(self._per_ds_gidxs[seed_k][rights[seed_k]])]
            batch_cost = float(self._per_ds_costs[seed_k][rights[seed_k]])
            rights[seed_k] -= 1
            total_remaining -= 1

            # 2. 填充：继续消费 pool，跳过不合适的条目
            # 每次 pool 条目无效时做全量检查，确认是否还有任何 item 能放入
            while batch_cost < threshold and pool_pos < len(pool):
                if max_graph_int is not None and len(batch) >= max_graph_int:
                    break
                k = keys[int(pool[pool_pos])]; pool_pos += 1
                if not _has_items(k) or self._per_ds_costs[k][lefts[k]] > max_cost - batch_cost:
                    budget = max_cost - batch_cost
                    if not any(_has_items(kk) and self._per_ds_costs[kk][lefts[kk]] <= budget
                               for kk in keys):
                        break
                    continue
                batch.append(int(self._per_ds_gidxs[k][lefts[k]]))
                batch_cost += float(self._per_ds_costs[k][lefts[k]])
                lefts[k] += 1
                total_remaining -= 1

            batches.append(batch)
            pbar.update(len(batch))

        pbar.close()
        if self.rank == 0:
            print(f"[Sampler] Bin-packing done: {len(batches)} batches in {time.time() - t0:.1f}s")

        return batches

    def _generate_batches(self, epoch_idx):
        base = self._get_base_batches()
        total_batches = (len(base) // self.world_size) * self.world_size
        batches = base[:total_batches]

        if self.shuffle:
            rng = random.Random(self.seed + epoch_idx)
            batches = list(batches)
            rng.shuffle(batches)

        return batches[self.rank::self.world_size]

    def _get_cached_batches(self):
        if self._cached_epoch != self.epoch or self._cached_batches is None:
            self._cached_batches = self._generate_batches(self.epoch)
            self._cached_epoch = self.epoch
            self._cached_step_counts[self.epoch] = len(self._cached_batches)
        return self._cached_batches

    def _batch_source_stats(self, batch):
        stats = {k: 0 for k in self.keys}
        for gidx in batch:
            pos = bisect.bisect_right(self._sorted_offsets, gidx) - 1
            stats[self._sorted_offset_keys[pos]] += 1
        return stats

    def __iter__(self):
        for batch in self._get_cached_batches():
            yield batch

    def __len__(self):
        return len(self._get_cached_batches())

    def precompute_total_steps(self, total_epochs):
        if self.rank == 0:
            print(f"Pre-computing exact steps for {total_epochs} epochs...")

        base = self._get_base_batches()
        steps_per_epoch = len(base) // self.world_size
        total_steps = steps_per_epoch * total_epochs

        for ep in range(1, total_epochs + 1):
            self._cached_step_counts[ep] = steps_per_epoch

        if self.rank == 0:
            print(f"Exact total steps: {total_steps} (Per epoch: {steps_per_epoch})")

        return total_steps

