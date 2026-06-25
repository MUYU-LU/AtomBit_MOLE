import warnings
import json
import os
import pickle
import re
import time
import mindspore as ms
from mindspore import nn, mint, ops
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
try:
    from matscipy.neighbours import neighbour_list
except ModuleNotFoundError:
    from ase.neighborlist import neighbor_list as neighbour_list
from sharker.data import Graph
from src.utils.Utils import scatter_mean  # 直接从子模块导入，不经过 __init__ 避免循环
# MOLE 在 fuse_experts / unfuse_experts 内部懒加载，同样避免循环导入

_TRANSPOSE_GATHER_IDX_3X3 = ms.Tensor([0, 3, 6, 1, 4, 7, 2, 5, 8], dtype=ms.int32)


def _profile_enabled() -> bool:
    return os.environ.get("ATOMBIT_PROFILE", "").lower() in {"1", "true", "yes", "on"}


def _tensor_to_numpy(value):
    if hasattr(value, "asnumpy"):
        return value.asnumpy()
    if hasattr(value, "data") and hasattr(value.data, "asnumpy"):
        return value.data.asnumpy()
    return np.asarray(value)


def _decode_config_from_param_dict(param_dict: dict):
    """Read HTGPConfig metadata embedded in newer checkpoints."""
    from src.utils.Utils import HTGPConfig

    meta = param_dict.get("_htgp_config_json")
    if meta is None:
        return None

    try:
        raw = bytes(int(x) for x in _tensor_to_numpy(meta).reshape(-1).tolist())
        payload = json.loads(raw.decode("utf-8"))
        cfg_dict = payload.get("config", payload)
        active_paths = cfg_dict.get("active_paths")
        if isinstance(active_paths, dict):
            parsed_paths = {}
            for key, value in active_paths.items():
                if isinstance(key, str) and "|" in key:
                    l_in, l_edge, l_out, op_type = key.split("|", 3)
                    parsed_paths[(int(l_in), int(l_edge), int(l_out), op_type)] = bool(value)
                else:
                    parsed_paths[key] = value
            cfg_dict["active_paths"] = parsed_paths

        return HTGPConfig(**{
            k: v for k, v in cfg_dict.items()
            if k in HTGPConfig.__dataclass_fields__
        })
    except Exception as exc:
        warnings.warn(
            f"Checkpoint contains HTGP config metadata but it could not be decoded: {exc}. "
            "Falling back to shape/buffer inference.",
            UserWarning,
        )
        return None


# ===========================================================================
# 从 checkpoint 参数形状自动推断 HTGPConfig 关键字段
# ===========================================================================
def _infer_config_from_param_dict(param_dict: dict):
    """
    从 MindSpore checkpoint 的参数名称和形状推断 HTGPConfig 关键超参。

    可推断字段（直接从形状读取，无歧义）：
        hidden_dim          embedding.embedding_table  [num_types, F]        → F
        num_rbf             geom_basis.rbf_mlp.0.weight [F, num_rbf]         → num_rbf
        num_layers          layers.{i}.norm.weight 中最大 i+1
        use_mole            h1_init.expert_weights 是否存在
        mole_num_experts    h1_init.expert_weights [K, F, F]                 → K
        use_L1              h1_init.* 是否存在
        use_L2              h2_init.* 是否存在
        use_direct_force    readout_force_0.* 是否存在

    旧 checkpoint 没有显式 config 元数据时，尽量只依赖 checkpoint 中的
    参数形状和 buffer 值；确实没有语义信息的字段（例如 dataset 名称、charge
    最小值）才回退到 HTGPConfig 默认约定。
    """
    from src.utils.Utils import HTGPConfig

    shapes = {name: tuple(p.shape) for name, p in param_dict.items()}

    # ---- hidden_dim ----
    hidden_dim = HTGPConfig.hidden_dim  # 默认
    if 'embedding.embedding_table' in shapes:
        hidden_dim = shapes['embedding.embedding_table'][1]

    # ---- num_rbf ----
    num_rbf = HTGPConfig.num_rbf
    if 'geom_basis.rbf_mlp.0.weight' in shapes:
        num_rbf = shapes['geom_basis.rbf_mlp.0.weight'][1]

    # ---- cutoff ----
    cutoff = HTGPConfig.cutoff
    if 'geom_basis.envelope.inv_r_cutoff' in param_dict:
        inv_cutoff = float(_tensor_to_numpy(param_dict['geom_basis.envelope.inv_r_cutoff']).reshape(-1)[0])
        if inv_cutoff != 0.0:
            cutoff = 1.0 / inv_cutoff
    elif 'geom_basis.rbf.prefactor' in param_dict:
        prefactor = float(_tensor_to_numpy(param_dict['geom_basis.rbf.prefactor']).reshape(-1)[0])
        if prefactor != 0.0:
            cutoff = 2.0 / (prefactor * prefactor)

    # ---- num_layers（遍历 layers.{i} / readout_energy_{i}，允许 checkpoint 带前缀） ----
    layer_indices = set()
    layer_re = re.compile(r"(?:^|\.)layers\.(\d+)\.")
    readout_re = re.compile(r"(?:^|\.)readout_(?:energy|force)_(\d+)(?:\.|$)")
    for name in shapes:
        layer_match = layer_re.search(name)
        if layer_match:
            layer_indices.add(int(layer_match.group(1)))
        readout_match = readout_re.search(name)
        if readout_match:
            layer_indices.add(int(readout_match.group(1)))
    num_layers = max(layer_indices) + 1 if layer_indices else HTGPConfig.num_layers

    # ---- active_paths（从 CellDict 中实际保存的路径权重恢复） ----
    active_paths = {}
    path_re = re.compile(r"(?:^|\.)(?:path_weights)\.(\d+)_(\d+)_(\d+)_(.+?)\.")
    for name in shapes:
        match = path_re.search(name)
        if match:
            l_in, l_edge, l_out, op_type = match.groups()
            active_paths[(int(l_in), int(l_edge), int(l_out), op_type)] = True
    if not active_paths:
        active_paths = HTGPConfig.__dataclass_fields__['active_paths'].default_factory()

    # ---- use_mole / mole_num_experts ----
    use_mole = any(
        n.endswith('.expert_weights') or
        n == 'h1_init.expert_weights' or
        n == 'h2_init.expert_weights' or
        n.startswith('routing_mlp.')
        for n in shapes
    )
    mole_num_experts = HTGPConfig.mole_num_experts
    for key, shape in shapes.items():
        if key.endswith('.expert_weights') or key in ('h1_init.expert_weights', 'h2_init.expert_weights'):
            mole_num_experts = shape[0]   # [K, F_out, F_in]
            break

    # ---- use_L1 / use_L2 ----
    use_L1 = any(n.startswith('h1_init.') for n in shapes) or any(1 in key[:3] for key in active_paths)
    use_L2 = any(n.startswith('h2_init.') for n in shapes) or any(2 in key[:3] for key in active_paths)

    # ---- use_direct_force ----
    use_direct_force = any(n.startswith('readout_force_0.') for n in shapes)

    # ---- charge / spin / dataset ----
    charge_rows = shapes.get('charge_embedding.embedding_table', (HTGPConfig.max_charge - HTGPConfig.min_charge + 1,))[0]
    min_charge = HTGPConfig.min_charge
    max_charge = min_charge + charge_rows - 1

    max_spin = shapes.get('spin_embedding.embedding_table', (HTGPConfig.max_spin + 1,))[0] - 1
    num_dataset = shapes.get('dataset_embedding.embedding_table', (HTGPConfig.num_dataset,))[0]

    default_dataset_types = HTGPConfig.__dataclass_fields__['dataset_types'].default_factory()
    default_names_by_idx = [name for name, _ in sorted(default_dataset_types.items(), key=lambda x: x[1])]
    dataset_names = [
        default_names_by_idx[i] if i < len(default_names_by_idx) else f"Dataset{i}"
        for i in range(num_dataset)
    ]
    dataset_types = {name: i for i, name in enumerate(dataset_names)}

    default_force_scale = HTGPConfig.__dataclass_fields__['force_scale'].default_factory()
    if 'force_scale_buf' in param_dict:
        scale_values = [float(x) for x in _tensor_to_numpy(param_dict['force_scale_buf']).reshape(-1).tolist()]
    else:
        scale_values = [float(default_force_scale.get(name, 1.0)) for name in dataset_names]
    force_scale = {
        name: (scale_values[i] if i < len(scale_values) else 1.0)
        for i, name in enumerate(dataset_names)
    }

    default_stress = HTGPConfig.__dataclass_fields__['stress_datasets'].default_factory()
    stress_datasets = {name: bool(default_stress.get(name, False)) for name in dataset_names}

    # ---- atom_types_map（从 z_mapper buffer 精确还原） ----
    # z_mapper[z] == -1 表示不支持，否则为该元素在 embedding 中的索引；
    # 读取实际值可精确重建支持的原子序数列表，无需任何"连续元素"假设。
    atom_types_map = HTGPConfig.__dataclass_fields__['atom_types_map'].default_factory()
    if 'z_mapper' in param_dict:
        try:
            z_mapper_np = param_dict['z_mapper'].asnumpy()
            atom_types_map = [int(z) for z in range(len(z_mapper_np)) if z_mapper_np[z] != -1]
        except Exception:
            # fallback: 仅用 shape 近似（要求元素集合从 H 开始连续）
            if 'embedding.embedding_table' in shapes:
                num_types = shapes['embedding.embedding_table'][0]
                atom_types_map = list(range(1, num_types + 1))

    return HTGPConfig(
        hidden_dim=hidden_dim,
        cutoff=cutoff,
        num_rbf=num_rbf,
        num_layers=num_layers,
        min_charge=min_charge,
        max_charge=max_charge,
        max_spin=max_spin,
        num_dataset=num_dataset,
        dataset_types=dataset_types,
        force_scale=force_scale,
        stress_datasets=stress_datasets,
        use_mole=use_mole,
        mole_num_experts=mole_num_experts,
        use_L1=use_L1,
        use_L2=use_L2,
        use_direct_force=use_direct_force,
        atom_types_map=atom_types_map,
        active_paths=active_paths,
    )


class HTGP_Calculator(Calculator):
    """
    适配 HTGPModel 训练逻辑的 ASE Calculator（MindSpore 版）

    支持：
    - 能量 / 力（自动微分或直接预测）
    - 应力（周期体系，Voigt 6 分量）
    - 权重 / 描述符捕获（用于可解释性分析）
    """
    implemented_properties = ['energy', 'forces', 'stress', 'descriptors', 'weights']

    def __init__(self, model, cutoff=6.0, capture_weights=False, capture_descriptors=False,
                 add_e0_baseline=True, e0_dir=None, dataset_name=None,
                 default_charge=0.0, default_spin=0.0, **kwargs):
        """
        Args:
            model: HTGPModel 实例（已加载权重）
            cutoff: 截断半径，须与训练时一致
            capture_weights: 是否捕获 PhysicsGating 权重（可解释性）
            capture_descriptors: 是否捕获各层原子描述符
            add_e0_baseline: 是否在最终能量中加回对应数据集 .pt 的基准能量
            e0_dir: 数据集 .pt 文件所在目录；默认依次查找当前工作目录和仓库根目录
            dataset_name: 默认数据集/头名称；atoms.info["dataset"] 可覆盖它
            default_charge: atoms.info["charge"] 缺失时使用的总电荷
            default_spin: atoms.info["spin"] 缺失时使用的自旋
        """
        Calculator.__init__(self, **kwargs)
        self.model = model
        self.cutoff = cutoff
        self.capture_weights = capture_weights
        self.capture_descriptors = capture_descriptors
        self.add_e0_baseline = bool(add_e0_baseline)
        self.e0_dir = e0_dir
        self.dataset_name = self._validate_dataset_name(dataset_name)
        self.default_charge = float(default_charge)
        self.default_spin = float(default_spin)
        self._e0_cache = {}
        self._missing_e0_warned = set()

        # 推理模式（关闭 dropout / BN 统计更新）
        self.model.set_train(False)

        # 冻结所有参数梯度（推理时不需要参数梯度）
        for param in self.model.trainable_params():
            param.requires_grad = False

        # MOLE 专家合并缓存：路由依赖组成、dataset、charge、spin。
        # 这些上下文不变（MD / 优化）则重用已合并权重。
        self._fused_context_key: tuple = ()
        model._mole_is_fused = False   # fuse 前 construct 照常走路由 MLP

    # ------------------------------------------------------------------
    # 从 checkpoint 文件直接构建 Calculator
    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, ckpt_path: str, config=None, cutoff: float = None,
                        capture_weights: bool = False,
                        capture_descriptors: bool = False,
                        strict: bool = False,
                        **kwargs) -> "HTGP_Calculator":
        """
        从 MindSpore checkpoint (.ckpt) 文件一步构建 Calculator。

        训练器以 EMA 权重保存推理 .ckpt，直接用 ms.load_checkpoint 加载即可。
        若同目录存在同名 _training_state.pkl，只加载 .ckpt（推理不需要优化器状态）。

        Args:
            ckpt_path: .ckpt 文件路径
            config: HTGPConfig 实例。为 None 时完全从 checkpoint 读取：
                    新 ckpt 使用内嵌 config 元数据，旧 ckpt 使用参数形状和
                    buffer 值推断。
            cutoff: 截断半径（Å）。为 None 时取 config.cutoff。
            capture_weights: 是否捕获 PhysicsGating 权重。
            capture_descriptors: 是否捕获各层描述符。
            strict: 是否严格要求所有参数都被加载（默认 False，与训练器行为一致）。
            **kwargs: 透传给 HTGP_Calculator.__init__。

        Returns:
            HTGP_Calculator 实例（已加载权重，推理模式）。

        Example::

            calc = HTGP_Calculator.from_checkpoint(
                ckpt_path="checkpoints/best_model.ckpt",
                config=HTGPConfig(hidden_dim=256, num_layers=4),
            )
            atoms.calc = calc
            print(atoms.get_potential_energy())
        """
        import os
        from src.models.Model import HTGPModel

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # ---- 先读取 checkpoint 参数字典，用于形状推断 ----
        print(f"[Calculator] Reading checkpoint: {ckpt_path}")
        param_dict = ms.load_checkpoint(ckpt_path)

        # ---- 1. 确定 config ----
        if config is None:
            config = _decode_config_from_param_dict(param_dict)
            if config is not None:
                print("[Calculator] Config loaded from checkpoint metadata")
            else:
                # 旧 ckpt 没有显式 config 元数据时，从参数形状和 buffer 值推断，
                # 避免因默认值与训练时不一致导致形状不匹配的 RuntimeError。
                config = _infer_config_from_param_dict(param_dict)
                print(f"[Calculator] Config inferred from checkpoint shapes: "
                      f"hidden_dim={config.hidden_dim}, cutoff={config.cutoff:.6g}, "
                      f"num_rbf={config.num_rbf}, "
                      f"num_layers={config.num_layers}, "
                      f"num_dataset={config.num_dataset}, "
                      f"use_mole={config.use_mole}, mole_num_experts={config.mole_num_experts}, "
                      f"use_L1={config.use_L1}, use_L2={config.use_L2}, "
                      f"use_direct_force={config.use_direct_force}")
                warnings.warn(
                    "Checkpoint has no embedded HTGP config metadata, so config was inferred "
                    "from checkpoint tensors. Dataset names and charge offset are not encoded "
                    "in old checkpoints; default naming/offset conventions were used for those "
                    "semantic fields.",
                    UserWarning,
                )
        else:
            print("[Calculator] Using explicit config argument")

        # ---- 2. 构建模型骨架 ----
        model = HTGPModel(config)

        # ---- 3. 加载权重 ----
        print(f"[Calculator] Loading weights into model ...")
        not_loaded, _ = ms.load_param_into_net(model, param_dict, strict_load=strict)
        if not_loaded:
            msg = (f"{len(not_loaded)} parameter(s) not loaded "
                   f"(first 5: {not_loaded[:5]}). "
                   "Pass strict=True to raise an error instead of this warning.")
            if strict:
                raise RuntimeError(msg)
            warnings.warn(msg, UserWarning)
        else:
            print(f"[Calculator] All parameters loaded successfully.")

        n_params = sum(p.size for p in model.get_parameters())
        print(f"[Calculator] Model parameters: {n_params:,}")

        # ---- 4. 确定 cutoff ----
        effective_cutoff = cutoff if cutoff is not None else config.cutoff

        return cls(
            model=model,
            cutoff=effective_cutoff,
            capture_weights=capture_weights,
            capture_descriptors=capture_descriptors,
            **kwargs,
        )

    def _validate_dataset_name(self, dataset_name):
        if dataset_name is None:
            return None
        dataset_name = str(dataset_name)
        if dataset_name not in self.model.cfg.dataset_types:
            known = ", ".join(self.model.cfg.dataset_types.keys())
            raise ValueError(
                f"Unknown dataset_name {dataset_name!r}. Available dataset names: {known}"
            )
        return dataset_name

    def set_inference_context(self, dataset_name=None, charge=None, spin=None):
        """Update calculator-level defaults used when atoms.info does not override them."""
        if dataset_name is not None:
            self.dataset_name = self._validate_dataset_name(dataset_name)
        if charge is not None:
            self.default_charge = float(charge)
        if spin is not None:
            self.default_spin = float(spin)
        self.unfuse_experts()
        # ASE caches results based on geometry, not model context. Changing the
        # dataset/head or charge/spin must force a fresh calculation even when
        # atom positions and cell are unchanged.
        self.reset()
        return self

    # ------------------------------------------------------------------
    # 主计算接口
    # ------------------------------------------------------------------
    def calculate(self, atoms=None, properties=('energy', 'forces', 'stress'),
                  system_changes=all_changes):
        profile = _profile_enabled()
        t_total = time.perf_counter() if profile else None
        Calculator.calculate(self, atoms, properties, system_changes)

        # ---- 1. 构建图数据（非可微预处理） ----
        t0 = time.perf_counter() if profile else None
        data = self._atoms_to_graph(atoms)
        graph_s = time.perf_counter() - t0 if profile else 0.0
        graph_profile = getattr(self, "_last_graph_profile", {})

        # ---- 1b. MOLE 专家合并（上下文不变时复用缓存，提升 MD/优化速度） ----
        t0 = time.perf_counter() if profile else None
        fused_now = False
        if getattr(self.model.cfg, 'use_mole', False):
            context_key = self._fuse_context_key(data)
            if context_key != self._fused_context_key:
                self.fuse_experts(data)
                self._fused_context_key = context_key
                fused_now = True
        fuse_s = time.perf_counter() - t0 if profile else 0.0

        is_periodic = atoms.pbc.any()
        calc_stress = 'stress' in properties and is_periodic
        use_direct_force = getattr(self.model.cfg, 'use_direct_force', False)

        pos0 = data.pos                          # (N, 3) float32，作为微分变量基准
        cell0 = getattr(data, 'cell', None)  # (1, 3, 3) 或 None（非周期体系属性不存在）

        # ---- 2. 根据需求选择前向计算路径 ----
        t0 = time.perf_counter() if profile else None
        grad_path = "direct_force"
        if use_direct_force and not calc_stress:
            # 模型直接输出力，无需 autodiff
            result = self.model(data,
                                capture_weights=self.capture_weights,
                                capture_descriptors=self.capture_descriptors)
            energy_val = result['energy']       # (num_graphs,)
            forces_val = result['force']        # (N, 3)

            self.results['energy'] = float(energy_val.sum().asnumpy())
            self.results['forces'] = forces_val.asnumpy()

        elif calc_stress:
            # 需要应力：对位移场 (pos, strain) 同时求导
            grad_path = "stress_ad"
            energy_val, grad_pos, grad_strain = self._forward_with_stress(
                data, pos0, cell0
            )
            self.results['energy'] = float(energy_val.asnumpy())
            self.results['forces'] = (-grad_pos).asnumpy()

            volume = float(mint.abs(mint.exp(ops.logdet(cell0[0]).astype(ms.float32))).asnumpy())
            if volume > 1e-8:
                stress_3x3 = (grad_strain.squeeze(0).asnumpy() / volume)
                self.results['stress'] = full_3x3_to_voigt_6_stress(stress_3x3)
            else:
                self.results['stress'] = np.zeros(6)

            if use_direct_force:
                # 若模型同时直接预测力，用直接力覆盖 autodiff 力（更精确）
                result_full = self.model(data,
                                         capture_weights=self.capture_weights,
                                         capture_descriptors=self.capture_descriptors)
                self.results['forces'] = result_full['force'].asnumpy()

        else:
            # 纯 autodiff：仅对 pos 求导
            grad_path = "force_ad"
            energy_val, grad_pos = self._forward_energy_and_force(data, pos0)
            self.results['energy'] = float(energy_val.asnumpy())
            self.results['forces'] = (-grad_pos).asnumpy()
        grad_s = time.perf_counter() - t0 if profile else 0.0

        t0 = time.perf_counter() if profile else None
        self.results['energy'] += self._get_e0_baseline_energy(atoms, data)
        e0_s = time.perf_counter() - t0 if profile else 0.0

        if profile:
            edges = int(data.edge_index.shape[1]) if hasattr(data, "edge_index") else -1
            total_s = time.perf_counter() - t_total
            print(
                "CALC_PROFILE "
                f"props={','.join(properties)} natoms={len(atoms)} edges={edges} "
                f"stress={int(calc_stress)} path={grad_path} fused_now={int(fused_now)} "
                f"graph_s={graph_s:.6f} "
                f"prep_s={graph_profile.get('prep_s', 0.0):.6f} "
                f"neighbor_s={graph_profile.get('neighbor_s', 0.0):.6f} "
                f"tensor_s={graph_profile.get('tensor_s', 0.0):.6f} "
                f"fuse_s={fuse_s:.6f} grad_s={grad_s:.6f} "
                f"e0_s={e0_s:.6f} total_s={total_s:.6f}",
                flush=True,
            )

        # ---- 3. 可选捕获 ----
        if self.capture_weights:
            self.results['weights'] = self._get_weights()

        if self.capture_descriptors:
            self.results['descriptors'] = self._get_descriptors()

    # ------------------------------------------------------------------
    # 梯度计算辅助（使用 ms.value_and_grad）
    # ------------------------------------------------------------------
    def _forward_energy_and_force(self, data, pos0):
        """仅对位置求导，返回 (energy_scalar, grad_pos)。"""
        model = self.model
        cw = self.capture_weights
        cd = self.capture_descriptors

        def energy_fn(pos):
            data.pos = pos
            result = model(data, capture_weights=cw, capture_descriptors=cd)
            if isinstance(result, dict):
                return result['energy'].sum()
            return result.sum()

        val_and_grad = ms.value_and_grad(energy_fn, grad_position=0)
        energy_val, grad_pos = val_and_grad(pos0)
        return energy_val, grad_pos

    def _forward_with_stress(self, data, pos0, cell0):
        """对位置和应变同时求导，返回 (energy_scalar, grad_pos, grad_strain)。"""
        model = self.model
        cw = self.capture_weights
        cd = self.capture_descriptors

        strain0 = mint.zeros((1, 3, 3), dtype=ms.float32)

        def energy_fn(pos, strain):
            # 对称化应变张量
            strain_flat = strain.reshape(strain.shape[0], 9)
            sym_strain = 0.5 * (strain_flat + ops.gather(strain_flat, _TRANSPOSE_GATHER_IDX_3X3, 1)).reshape(strain.shape)
            # 变形位置
            data.pos = pos + mint.matmul(pos, sym_strain[0])
            # 变形晶胞
            if cell0 is not None:
                data.cell = cell0 + mint.matmul(cell0, sym_strain)
            result = model(data, capture_weights=cw, capture_descriptors=cd)
            if isinstance(result, dict):
                return result['energy'].sum()
            return result.sum()

        val_and_grad = ms.value_and_grad(energy_fn, grad_position=(0, 1))
        energy_val, (grad_pos, grad_strain) = val_and_grad(pos0, strain0)
        return energy_val, grad_pos, grad_strain

    # ------------------------------------------------------------------
    # MOLE 专家合并（推理加速）
    # ------------------------------------------------------------------
    def _values_key(self, value, cast):
        if value is None:
            return ()
        arr = np.asarray(_tensor_to_numpy(value)).reshape(-1)
        return tuple(cast(x.item() if hasattr(x, "item") else x) for x in arr)

    def _dataset_key(self, dataset):
        if dataset is None:
            return (self.dataset_name,) if self.dataset_name is not None else ()
        if isinstance(dataset, str):
            return (dataset,)
        return tuple(str(x) for x in dataset)

    def _fuse_context_key(self, data) -> tuple:
        z_key = self._values_key(data.z, int)
        dataset_key = self._dataset_key(getattr(data, "dataset", None))
        charge_key = self._values_key(
            getattr(data, "charge", None), float
        ) or (self.default_charge,)
        spin_key = self._values_key(
            getattr(data, "spin", None), float
        ) or (self.default_spin,)
        return z_key, dataset_key, charge_key, spin_key

    def fuse_experts(self, data) -> None:
        """
        计算当前分子的 MOLE 路由系数，将模型中所有 MOLE 层的 K 个专家
        权重矩阵合并为单个等效线性层。

        对同一组成、dataset、charge、spin 上下文只需调用一次。
        Calculator.calculate() 会在检测到上下文变化时自动调用。

        数学等价性：
            原始路径  output = Σ_e coeffs[e] * (W_e @ x)
            合并路径  output = W_merged @ x，其中 W_merged = Σ_e coeffs[e] * W_e
        """
        model = self.model
        cfg = model.cfg

        # ---- 复现 HTGPModel.construct 的路由系数计算部分 ----
        num_graphs = 1  # Calculator 单次只处理一个结构
        z_idx = model.z_mapper[data.z]

        # charge / spin
        z_charge = (
            (data.charge - cfg.min_charge).to(dtype=ms.int32)
            if hasattr(data, 'charge') and data.charge is not None
            else ms.Tensor([-cfg.min_charge] * num_graphs, dtype=ms.int32)
        )
        z_spin = (
            data.spin.to(dtype=ms.int32)
            if hasattr(data, 'spin') and data.spin is not None
            else mint.zeros(num_graphs, dtype=ms.int32)
        )
        # dataset id
        if hasattr(data, 'dataset') and data.dataset is not None:
            z_dataset = ms.Tensor(
                [int(cfg.dataset_types[s]) for s in data.dataset], dtype=ms.int32
            )
        else:
            z_dataset = mint.zeros(num_graphs, dtype=ms.int32)

        # 原子/分子 embedding
        h_atom = model.embedding_norm(model.embedding(z_idx))
        h_charge = model.charge_embedding_norm(model.charge_embedding(z_charge))
        h_spin = model.spin_embedding_norm(model.spin_embedding(z_spin))
        h_dataset = model.dataset_embedding_norm(model.dataset_embedding(z_dataset))
        h_csd = model.csd_embedding_norm(
            model.csd_embedding(mint.cat([h_charge, h_spin, h_dataset], dim=-1))
        )

        # 分子级原子特征均值 + routing MLP → softmax 路由系数
        h_atom_mol = scatter_mean(h_atom, data.batch, dim=0, dim_size=num_graphs)
        h_mol = mint.cat([h_atom_mol, h_csd], dim=-1)
        logits = model.routing_mlp(h_mol)                          # [1, K]
        coeffs = mint.softmax(logits.astype(ms.float32), dim=-1)   # [1, K]
        coeffs_1g = coeffs[0]                                       # [K]

        # ---- 遍历模型所有 MOLE 实例并 fuse ----
        from src.models.Modules import MOLE  # 懒加载，避免循环导入
        for _, cell in model.cells_and_names():
            if isinstance(cell, MOLE):
                cell.fuse(coeffs_1g)
        # 通知 construct 跳过路由 MLP（系数已烧入权重）
        model._mole_is_fused = True

    def unfuse_experts(self) -> None:
        """清除所有 MOLE 层的合并权重，恢复原始 K-专家路径。"""
        from src.models.Modules import MOLE  # 懒加载，避免循环导入
        for _, cell in self.model.cells_and_names():
            if isinstance(cell, MOLE):
                cell.unfuse()
        self._fused_context_key = ()
        self.model._mole_is_fused = False

    # ------------------------------------------------------------------
    # 数据集 E0 基准能量
    # ------------------------------------------------------------------
    def _get_e0_baseline_energy(self, atoms, data) -> float:
        """返回当前结构的 e0 基准能量（eV）。

        优先级：
        1. self.e0_fit（微调任务在数据预处理时拟合的 per-element 参考能量）
           — 设置时直接用它，不再叠加预训练 e0，防止重复加两次
        2. 预训练 e0（从数据集 .pt 文件读取）
        """
        # Fine-tuned model: use the e0 fitted during data conversion
        e0_fit = getattr(self, "e0_fit", None)
        if e0_fit:
            return sum(e0_fit.get(int(z), 0.0) for z in atoms.get_atomic_numbers())

        if not self.add_e0_baseline:
            return 0.0

        dataset = None
        if hasattr(data, 'dataset') and data.dataset:
            dataset = str(data.dataset[0])
        elif atoms is not None:
            dataset = atoms.info.get("dataset")
            if dataset is not None:
                dataset = str(dataset)

        if not dataset:
            return 0.0

        e0_dict = self._load_e0_dict(dataset)
        if not e0_dict:
            return 0.0

        baseline = 0.0
        missing_z = set()
        for z in atoms.get_atomic_numbers():
            z_int = int(z)
            if z_int in e0_dict:
                baseline += float(e0_dict[z_int])
            else:
                missing_z.add(z_int)

        if missing_z:
            warn_key = (dataset, tuple(sorted(missing_z)))
            if warn_key not in self._missing_e0_warned:
                warnings.warn(
                    f"E0 baseline for dataset {dataset!r} does not contain atomic number(s) "
                    f"{sorted(missing_z)}; those atoms were skipped.",
                    UserWarning,
                )
                self._missing_e0_warned.add(warn_key)

        return baseline

    def _load_e0_dict(self, dataset: str):
        if dataset in self._e0_cache:
            return self._e0_cache[dataset]

        pt_path = self._resolve_e0_path(dataset)
        if not os.path.exists(pt_path):
            if dataset not in self._missing_e0_warned:
                warnings.warn(
                    f"E0 baseline file not found for dataset {dataset!r}: {pt_path}. "
                    "Energy will be returned without the dataset baseline.",
                    UserWarning,
                )
                self._missing_e0_warned.add(dataset)
            self._e0_cache[dataset] = None
            return None

        with open(pt_path, "rb") as f:
            payload = pickle.load(f)

        e0_dict = payload.get("e0_dict") if isinstance(payload, dict) else None
        if not isinstance(e0_dict, dict):
            raise ValueError(f"E0 baseline file {pt_path} does not contain an e0_dict dictionary.")

        self._e0_cache[dataset] = {int(k): float(v) for k, v in e0_dict.items()}
        return self._e0_cache[dataset]

    def _resolve_e0_path(self, dataset: str) -> str:
        filename = f"{dataset}.pt"
        if self.e0_dir is not None:
            return os.path.join(self.e0_dir, filename)

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        for base_dir in (os.getcwd(), repo_root):
            candidate = os.path.join(base_dir, filename)
            if os.path.exists(candidate):
                return candidate
        return os.path.join(os.getcwd(), filename)

    # ------------------------------------------------------------------
    # 数据转换：ASE Atoms -> sharker Graph
    # ------------------------------------------------------------------
    def _prepare_atoms_for_neighbor_list(self, atoms):
        """
        为邻居搜索准备 ASE Atoms。

        matscipy.neighbour_list 即使在非周期体系下，也会对 cell 求逆；
        因此对 .xyz 这类没有有效晶胞的分子，需要临时补一个非奇异盒子。
        """
        atoms_nl = atoms.copy()
        cell_np = atoms_nl.get_cell().array
        has_valid_cell = np.abs(np.linalg.det(cell_np)) > 1e-6

        if has_valid_cell:
            return atoms_nl

        if atoms_nl.pbc.any():
            raise ValueError(
                "Periodic structure has a singular cell matrix. "
                "Please provide a valid non-singular lattice."
            )

        # 非周期分子：居中并添加足够真空，避免 neighbour_list 因奇异 cell 崩溃。
        # 平移不会改变能量/力或邻接关系，但能保证内部线性代数稳定。
        vacuum = max(float(self.cutoff) + 2.0, 8.0)
        atoms_nl.center(vacuum=vacuum)
        return atoms_nl

    def _atoms_to_graph(self, atoms):
        """将 ASE Atoms 转换为模型所需的 Graph 对象。"""
        profile = _profile_enabled()
        t_graph0 = time.perf_counter() if profile else None
        t0 = time.perf_counter() if profile else None
        atoms_nl = self._prepare_atoms_for_neighbor_list(atoms)
        prep_s = time.perf_counter() - t0 if profile else 0.0

        t0 = time.perf_counter() if profile else None
        z = ms.Tensor(atoms_nl.get_atomic_numbers(), dtype=ms.int32)
        pos = ms.Tensor(atoms_nl.get_positions(), dtype=ms.float32)

        # 晶胞处理
        cell = None
        if atoms_nl.pbc.any():
            cell_np = atoms_nl.get_cell().array
            if np.abs(np.linalg.det(cell_np)) > 1e-6:
                cell = ms.Tensor(cell_np, dtype=ms.float32).unsqueeze(0)  # (1,3,3)

        # 邻居列表（非可微，matscipy 实现，比 ASE 快约 10x）
        pre_neighbor_tensor_s = time.perf_counter() - t0 if profile else 0.0
        t0 = time.perf_counter() if profile else None
        i_idx, j_idx, S_integers = neighbour_list('ijS', atoms_nl, self.cutoff)
        neighbor_s = time.perf_counter() - t0 if profile else 0.0
        t0 = time.perf_counter() if profile else None
        edge_index = ms.Tensor(np.vstack((i_idx, j_idx)), dtype=ms.int32)
        shifts_int = ms.Tensor(S_integers, dtype=ms.float32)

        num_atoms = len(atoms)
        batch = mint.zeros(num_atoms, dtype=ms.int32)

        data = Graph(
            z=z,
            pos=pos,
            cell=cell,
            edge_index=edge_index,
            shifts_int=shifts_int,
            batch=batch,
        )
        data.num_graphs = 1

        dataset = atoms.info.get("dataset", self.dataset_name)
        if dataset is not None:
            data.dataset = [self._validate_dataset_name(dataset)]

        charge = atoms.info.get("charge", self.default_charge)
        if charge is None:
            charge = self.default_charge
        data.charge = ms.Tensor([charge], dtype=ms.float32)

        spin = atoms.info.get("spin", self.default_spin)
        if spin is None:
            spin = self.default_spin
        data.spin = ms.Tensor([spin], dtype=ms.float32)
        if profile:
            self._last_graph_profile = {
                "prep_s": prep_s,
                "pre_neighbor_tensor_s": pre_neighbor_tensor_s,
                "neighbor_s": neighbor_s,
                "tensor_s": time.perf_counter() - t0,
                "graph_total_s": time.perf_counter() - t_graph0,
            }
        return data

    # ------------------------------------------------------------------
    # 权重 / 描述符提取
    # ------------------------------------------------------------------
    def _get_weights(self):
        """从各层 PhysicsGating 中提取捕获的物理权重。"""
        weights_per_layer = []
        for layer in self.model.layers:
            gating = layer['gating']

            def _extract(attr):
                val = getattr(gating, attr, None)
                return val.asnumpy() if val is not None else None

            weights_per_layer.append({
                'g0': _extract('g0_captured'),
                'g1': _extract('g1_captured'),
                'g2': _extract('g2_captured'),
                'chem_logits': _extract('chem_logits_captured'),
                'phys_logits': _extract('phys_logits_captured'),
                'scalar_basis': _extract('scalar_basis_captured'),
                'p_ij': _extract('p_ij_captured'),
            })
        return weights_per_layer

    def _get_descriptors(self):
        """从模型中提取各层原子描述符 (h0, h1, h2)。"""
        if not hasattr(self.model, 'all_layer_descriptors'):
            return None
        result = []
        for layer_feats in self.model.all_layer_descriptors:
            result.append({
                k: (v.asnumpy() if v is not None else None)
                for k, v in layer_feats.items()
            })
        return result
