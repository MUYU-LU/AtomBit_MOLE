from typing import Dict, Optional, Tuple
from dataclasses import asdict, dataclass, field, is_dataclass
import json
import mindspore as ms
from mindspore import nn, mint, ops
from mindspore.common.initializer import Normal
from .Modules import (
    GeometricBasis, LeibnizCoupling, PhysicsGating, CartesianDensityBlock, RMSNorm,
    SafeEquivariantRMSNorm, MOLE, MOLEEnergyReadout, MOLEForceReadout,
    _init_linear,
)
from src.utils import scatter_add, scatter_mean, HTGPConfig


def _encode_htgp_config(config: HTGPConfig) -> bytes:
    """Encode config as uint8 checkpoint metadata."""
    if is_dataclass(config):
        cfg = asdict(config)
    else:
        cfg = {
            k: getattr(config, k)
            for k in HTGPConfig.__dataclass_fields__
            if hasattr(config, k)
        }

    active_paths = cfg.get("active_paths")
    if isinstance(active_paths, dict):
        cfg["active_paths"] = {
            "|".join(map(str, key)) if isinstance(key, tuple) else str(key): value
            for key, value in active_paths.items()
        }

    payload = {
        "format": "HTGPConfig",
        "version": 1,
        "config": cfg,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ==========================================
# 7. 主模型 (Main Model)
# ==========================================
class HTGPModel(nn.Cell):
    def __init__(self, config: HTGPConfig):
        super().__init__()
        self.cfg = config
        self._recompute_applied = []
        self.register_buffer(
            "_htgp_config_json",
            ms.Tensor(list(_encode_htgp_config(config)), dtype=ms.uint8),
        )
        
        # ============================================================
        # 🔥 修改 1: 构建原子序数映射表 (Z-Mapper)
        # ============================================================
        # 优先从 config 获取原子列表，如果没有则使用默认的常用有机元素列表
        # 对应: H(1), B(5), C(6), N(7), O(8), F(9), P(15), S(16), Cl(17), Br(35), I(53)
        if hasattr(config, 'atom_types_map'):
            self.used_atomic_numbers = config.atom_types_map
        else:
            self.used_atomic_numbers = [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53]
            
        num_actual_types = len(self.used_atomic_numbers) # 通常为 11
        max_z = max(self.used_atomic_numbers)            # 通常为 53

        self.num_actual_types = num_actual_types
        num_actual_charges = config.max_charge - config.min_charge + 1
        num_actual_spins = config.max_spin + 1
        self.dataset_names = [name for name, _ in sorted(config.dataset_types.items(), key=lambda x: x[1])]
        self.num_datasets = len(self.dataset_names)

        # 注册映射表 buffer (会自动转到 GPU，但不更新梯度)
        # 初始化为 -1，方便后续检查非法原子
        self.register_buffer('z_mapper', mint.full((max_z + 1,), -1, dtype=ms.int32))
        
        # 填充映射: z -> idx (例如 53 -> 10)
        for idx, z in enumerate(self.used_atomic_numbers):
            self.z_mapper[z] = idx

        # ============================================================
        # 🔥 修改 2: Embedding 尺寸缩小
        # ============================================================
        # Embedding: 只分配 11 行参数，而不是 60 行
        self.embedding = nn.Embedding(num_actual_types, config.hidden_dim, embedding_table=Normal(sigma=0.1))
        self.charge_embedding = nn.Embedding(num_actual_charges, config.hidden_dim, embedding_table=Normal(sigma=0.1))
        self.spin_embedding = nn.Embedding(num_actual_spins, config.hidden_dim, embedding_table=Normal(sigma=0.1))
        self.dataset_embedding = nn.Embedding(config.num_dataset, config.hidden_dim, embedding_table=Normal(sigma=0.1))
        self.csd_embedding = nn.SequentialCell(
            nn.Linear(3 * config.hidden_dim, config.hidden_dim, bias=True),
            nn.SiLU(),
        )
        _init_linear(self.csd_embedding[0])
        self.embedding_norm = RMSNorm(config.hidden_dim)
        self.charge_embedding_norm = RMSNorm(config.hidden_dim)
        self.spin_embedding_norm = RMSNorm(config.hidden_dim)
        self.dataset_embedding_norm = RMSNorm(config.hidden_dim)
        self.csd_embedding_norm = RMSNorm(config.hidden_dim)

        if config.use_mole:
            self.routing_mlp = nn.SequentialCell(
                nn.Linear(2 * config.hidden_dim, config.mole_num_experts * 2, bias=True),
                nn.SiLU(),
                nn.Linear(config.mole_num_experts * 2, config.mole_num_experts * 2, bias=True),
                nn.SiLU(),
                nn.Linear(config.mole_num_experts * 2, config.mole_num_experts, bias=True),
            )
            _init_linear(self.routing_mlp[0])
            _init_linear(self.routing_mlp[2])
            _init_linear(self.routing_mlp[4])
            # self.mole_dropout = nn.Dropout(p=config.mole_dropout)
        
        # Components (保持不变)
        self.geom_basis = GeometricBasis(config)

        # h1/h2 几何初始化投影：将 scatter 后的 basis_edges[1/2] 投影为初始 L1/L2 特征
        # 使 Layer 0 的 l_in>=1 路径从第一步起就有非零输入，避免这些参数永久死亡
        if config.use_L1:
            if config.use_mole:
                self.h1_init = MOLE(config.hidden_dim, config.hidden_dim,
                                    num_experts=config.mole_num_experts)
            else:
                self.h1_init = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
                _init_linear(self.h1_init)
        if config.use_L2:
            if config.use_mole:
                self.h2_init = MOLE(config.hidden_dim, config.hidden_dim,
                                    num_experts=config.mole_num_experts)
            else:
                self.h2_init = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
                _init_linear(self.h2_init)
        
        self.layers = nn.CellList()
        for _ in range(config.num_layers):
            self.layers.append(nn.CellDict({
                'norm': RMSNorm(config.hidden_dim),
                'norm_L1': SafeEquivariantRMSNorm(config.hidden_dim, min_rms=0.05),
                'norm_L2': SafeEquivariantRMSNorm(config.hidden_dim, min_rms=0.05),
                'coupling': LeibnizCoupling(config),
                'gating': PhysicsGating(config),
                'density': CartesianDensityBlock(config),
            }))
        
        # 每层独立的 energy readout；use_mole 时用 MOLE 扩大参数量（节点级安全）
        self._num_readout_layers = config.num_layers
        for i in range(config.num_layers):
            if config.use_mole:
                block = MOLEEnergyReadout(config)
                block.mole3.expert_weights.set_data(
                    block.mole3.expert_weights.data * (1.0 / config.num_layers)
                )
                setattr(self, f'readout_energy_{i}', block)
            else:
                block = nn.SequentialCell(
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(config.hidden_dim, 1),
                )
                _init_linear(block[0])
                _init_linear(block[2])
                _init_linear(block[4])
                block[4].weight.set_data(block[4].weight.data * (1.0 / config.num_layers))
                setattr(self, f'readout_energy_{i}', block)

        if config.use_direct_force and config.use_L1:
            for i in range(config.num_layers):
                if config.use_mole:
                    block = MOLEForceReadout(config)
                    block.mole.expert_weights.set_data(
                        block.mole.expert_weights.data * (1.0 / config.num_layers)
                    )
                    setattr(self, f'readout_force_{i}', block)
                else:
                    block = nn.SequentialCell(
                        nn.Linear(config.hidden_dim, 1, bias=False),
                    )
                    _init_linear(block[0])
                    block[0].weight.set_data(block[0].weight.data * (1.0 / config.num_layers))
                    setattr(self, f'readout_force_{i}', block)

        # if config.use_long_range:
        #     self.long_range = LatentLongRange(config)
        # else:
        #     print("not used long_range")
            
        # Atomic Ref:
        # - 支持多 dataset：为每个 dataset 维护一份 E0 表
        # - 通过 flat_idx = dataset_id * num_actual_types + z_idx 索引
        # self.energy_scale = nn.Embedding(config.num_dataset * num_actual_types, 1)
        # self.energy_scale.embedding_table = ms.Parameter(self.energy_scale.embedding_table * 0)

        # self.force_scale = nn.Embedding(config.num_dataset * num_actual_types, 1)
        # self.force_scale.embedding_table = ms.Parameter(self.force_scale.embedding_table * 0)

        # 按 dataset_types 值排序，保证索引顺序与 z_dataset 一致
        scale_list = [config.force_scale[name] for name, _ in sorted(config.dataset_types.items(), key=lambda x: x[1])]
        self.register_buffer('force_scale_buf', ms.Tensor(scale_list, dtype=ms.float32))
        self._configure_recompute()

    def _mark_for_recompute(self, name: str, cell: Optional[nn.Cell], seen_cells: set) -> None:
        if cell is None:
            return

        cell_id = id(cell)
        if cell_id in seen_cells:
            return
        seen_cells.add(cell_id)

        recompute_fn = getattr(cell, "recompute", None)
        if recompute_fn is None:
            return

        applied = False
        try:
            recompute_fn(use_reentrant=False)
            applied = True
        except TypeError:
            # 兼容不同 MindSpore 版本里 recompute 的参数签名差异。
            for kwargs in (
                {},
                {"use_reentrant": False, "mp_comm_recompute": False},
                {"use_reentrant": False, "parallel_optimizer_comm_recompute": False},
                {"mp_comm_recompute": False},
                {"parallel_optimizer_comm_recompute": False},
            ):
                try:
                    recompute_fn(**kwargs)
                    applied = True
                    break
                except TypeError:
                    continue

        if applied:
            self._recompute_applied.append(name)

    def _configure_recompute(self) -> None:
        if not bool(getattr(self.cfg, "use_recompute", False)):
            return

        seen_cells = set()

        # 仅对显存占用最高的主干模块开启重计算，尽量控制吞吐损失。
        if getattr(self.cfg, "use_mole", False):
            self._mark_for_recompute("geom_basis", self.geom_basis, seen_cells)
        self._mark_for_recompute("h1_init", getattr(self, "h1_init", None), seen_cells)
        self._mark_for_recompute("h2_init", getattr(self, "h2_init", None), seen_cells)

        for idx, layer in enumerate(self.layers):
            self._mark_for_recompute(f"layers.{idx}.coupling", layer["coupling"], seen_cells)
            self._mark_for_recompute(f"layers.{idx}.gating", layer["gating"], seen_cells)
            self._mark_for_recompute(f"layers.{idx}.density", layer["density"], seen_cells)

        for idx in range(self._num_readout_layers):
            energy_readout = getattr(self, f"readout_energy_{idx}", None)
            force_readout = getattr(self, f"readout_force_{idx}", None)

            if isinstance(energy_readout, MOLEEnergyReadout):
                self._mark_for_recompute(f"readout_energy_{idx}", energy_readout, seen_cells)
            if isinstance(force_readout, MOLEForceReadout):
                self._mark_for_recompute(f"readout_force_{idx}", force_readout, seen_cells)

    def construct(self, data, capture_weights=False, capture_descriptors=False):
        if capture_descriptors:
            self.all_layer_descriptors = []
        # ============================================================
        # 🔥 修改 3: Forward 中应用映射
        # ============================================================
        # 获取分子数量（确保data.num_graphs存在，否则从batch推断）
        if hasattr(data, 'num_graphs'):
            num_graphs = data.num_graphs
        else:
            num_graphs = int(data.batch.max().item()) + 1
        # 获取原始原子序数 (N,)
        z_raw = data.z
        
        # 转换为稠密索引 (N,) -> [0, 2, 10, ...]
        z_idx = self.z_mapper[z_raw]
        if hasattr(data, 'charge') and data.charge is not None:
            z_charge = (data.charge - self.cfg.min_charge).to(dtype=ms.int32)
        else:
            # Missing charge means neutral, not embedding row 0/min_charge.
            z_charge = ms.Tensor([-self.cfg.min_charge] * num_graphs, dtype=ms.int32)
        z_spin = data.spin.to(dtype=ms.int32) if hasattr(data, 'spin') and data.spin is not None else ms.mint.zeros(num_graphs).to(dtype=ms.int32)
        # dataset id:
        # - 可能是单个字符串（整个 batch 同一个数据集）
        # - 也可能在 collate 后变成 list[str]（每个 graph 一个字符串）
        # - 或者已经是数值 id（Tensor/ndarray/int）
        if hasattr(data, 'dataset') and data.dataset is not None:
            raw_ds = data.dataset
            z_dataset = ms.Tensor([int(self.cfg.dataset_types[s]) for s in raw_ds], dtype=ms.int32)

        else:
            z_dataset = ms.mint.zeros(num_graphs).to(dtype=ms.int32)
        
        # (可选) 安全检查: 如果数据里混入了未定义的原子 (如 Fe=26)，这里会是 -1
        # if (z_idx == -1).any():
        #    raise ValueError(f"Input contains undefined atomic numbers! Supported: {self.used_atomic_numbers}")

        # 1. 几何计算
        row, col = data.edge_index
        # 处理 shifts_int (PBC)
        cell = getattr(data, 'cell', None)
        if cell is not None and len(cell.shape) == 4 and cell.shape[1] == 1:
            cell = cell.reshape((cell.shape[0], cell.shape[2], cell.shape[3]))
            data.cell = cell
        if hasattr(data, 'shifts_int') and data.shifts_int is not None and cell is not None:
            batch_cell = data.cell[data.batch[row]]          # (E, 3, 3)
            current_shifts = mint.bmm(
                data.shifts_int.unsqueeze(1), batch_cell
            ).squeeze(1)                                     # (E, 3)

            # OMol/SPICE periodic inference: keep periodic shifts for periodic
            # CSP crystals. Zeroing these shifts is only valid for nonperiodic
            # molecular data with folded coordinates.
        else:
            current_shifts = mint.zeros(
                (row.shape[0], 3),
                dtype=data.pos.dtype
            )

        # ============================================================
        # bf16 数值稳定性：几何相关的 norm/div 等强制用 fp32
        # - 大模型 + bf16 下最容易在距离/归一化链路上出现 overflow/underflow
        # ============================================================
        pos_f32 = data.pos.astype(ms.float32)
        shifts_f32 = current_shifts.astype(ms.float32)
        vec_ij = pos_f32[col] - pos_f32[row] + shifts_f32
        d_ij = mint.norm(vec_ij, dim=-1).clamp(min=1e-8)

        h_atom = self.embedding(z_idx)  # (N, F)
        h_atom = self.embedding_norm(h_atom)
        h_charge = self.charge_embedding(z_charge) # (M, F)
        h_charge = self.charge_embedding_norm(h_charge)
        # breakpoint()
        h_spin = self.spin_embedding(z_spin) # (M, F)
        h_spin = self.spin_embedding_norm(h_spin)
        h_dataset = self.dataset_embedding(z_dataset) # (M, F)
        h_dataset = self.dataset_embedding_norm(h_dataset)
        h_csd = self.csd_embedding(mint.cat([h_charge, h_spin, h_dataset], dim=-1)) # (M, F)
        h_csd = self.csd_embedding_norm(h_csd)
        h_csd_atom = h_csd[data.batch] # (N, F)
        # print("h0 shape:", h0.shape)
        # 2. 状态初始化 (使用 z_idx)
        # h0 = self.embedding(z_idx) # (N, F) -> 使用映射后的索引
        h0 = h_atom + h_csd_atom
        h1 = None  # 在 inv_sqrt_deg 计算后初始化（见下方）
        h2 = None

        # 提前计算 MOLE 路由系数，供后续节点级 MOLE 层使用
        if self.cfg.use_mole:
            # 在分子级别聚合原子序数embedding
            h_atom_mol = scatter_mean(h_atom, data.batch, dim=0, dim_size=num_graphs)  # [num_graphs, hidden_dim]
            h_mol = mint.cat([h_atom_mol, h_csd], dim=-1)
            # 通过路由网络计算混合系数
            expert_mixing_logits = self.routing_mlp(h_mol)  # [num_graphs, mole_num_experts]

            # expert_mixing_logits = self.mole_dropout(expert_mixing_logits)
            expert_mixing_coefficients = mint.softmax(expert_mixing_logits.astype(ms.float32), dim=-1)

        # 几何基中的边级投影已统一为普通 Linear
        basis_edges, r_hat = self.geom_basis(vec_ij, d_ij)
        
        total_energy = 0.0

        if self.cfg.use_direct_force:
            total_force = mint.zeros((data.z.shape[0], 3), dtype=data.pos.dtype)
        
        # 3. 层级传递
        # per-atom degree normalization（入度，与 col 聚合一致）
        # 无向图中 in-degree == out-degree，数值不变，仅语义对齐
        _ones = mint.ones(col.shape, dtype=ms.float32)
        _deg = scatter_add(_ones, col, dim=0, dim_size=data.z.shape[0])  # [N]
        _deg.clamp_(min=1.0)
        inv_sqrt_deg = 1.0 / mint.sqrt(_deg)  # [N]

        # h1/h2 几何初始化：scatter basis_edges 到目标节点（col），得到"邻居→自身"的初始矢量场
        # 无向图中边 (B,A) 的 col=A, row=B，故 h1_raw[A] = Σ_B basis(r_{B→A})
        if self.cfg.use_L1 and 1 in basis_edges:
            h1_raw = scatter_add(basis_edges[1], col, dim=0, dim_size=data.z.shape[0])
            h1_raw = h1_raw * inv_sqrt_deg.view(-1, 1, 1).astype(h1_raw.dtype)
            if self.cfg.use_mole:
                h1 = self.h1_init(h1_raw, expert_mixing_coefficients, data.batch)
            else:
                h1 = self.h1_init(h1_raw)

        if self.cfg.use_L2 and 2 in basis_edges:
            h2_raw = scatter_add(basis_edges[2], col, dim=0, dim_size=data.z.shape[0])
            h2_raw = h2_raw * inv_sqrt_deg.view(-1, 1, 1, 1).astype(h2_raw.dtype)
            if self.cfg.use_mole:
                h2 = self.h2_init(h2_raw, expert_mixing_coefficients, data.batch)
            else:
                h2 = self.h2_init(h2_raw)

        for i, layer in enumerate(self.layers):

            h0_norm = layer['norm'](h0)
            h1_in = layer['norm_L1'](h1) if (self.cfg.use_L1 and h1 is not None) else h1
            h2_in = layer['norm_L2'](h2) if (self.cfg.use_L2 and h2 is not None) else h2
            # A. 莱布尼茨消息生成（使用归一化特征，Pre-LN 保证数值稳定）
            node_feats = {0: h0_norm, 1: h1_in, 2: h2_in}
            if self.cfg.use_mole:
                raw_msgs = layer['coupling'](
                    node_feats, basis_edges, data.edge_index,
                    expert_mixing_coefficients=expert_mixing_coefficients,
                    batch=data.batch
                )
            else:
                raw_msgs = layer['coupling'](node_feats, basis_edges, data.edge_index)
            
            # B. 物理门控（与 coupling 一致，使用归一化特征）
            if self.cfg.use_mole:
                gated_msgs = layer['gating'](
                    raw_msgs, h0_norm, basis_edges[0], r_hat, h1_in, data.edge_index,
                    capture_weights=capture_weights,
                    expert_mixing_coefficients=expert_mixing_coefficients,
                    batch=data.batch,
                )
            else:
                gated_msgs = layer['gating'](raw_msgs, h0_norm, basis_edges[0], r_hat, h1_in, data.edge_index, capture_weights=capture_weights)

            # C. 密度聚合与更新：scatter 到 col（目标节点）
            # 无向图中，对节点 A 贡献的边为 (B,A)，其 row=B（邻居），col=A（自身）
            # 因此 h_trans=W(h[row=B]) 自然取到邻居特征，实现标准 MPNN
            if self.cfg.use_mole:
                delta_h0, delta_h1, delta_h2 = layer['density'](
                    gated_msgs, col, data.z.shape[0],
                    inv_sqrt_deg, expert_mixing_coefficients, data.batch,
                )
            else:
                delta_h0, delta_h1, delta_h2 = layer['density'](gated_msgs, col, data.z.shape[0], inv_sqrt_deg)

            # D. 残差更新 (Residual Update)
            h0 = h0 + delta_h0

            if self.cfg.use_L1 and delta_h1 is not None:
                h1 = delta_h1 if h1 is None else h1 + delta_h1

            if self.cfg.use_L2 and delta_h2 is not None:
                h2 = delta_h2 if h2 is None else h2 + delta_h2

            # h0 h1 h2保存
            if capture_descriptors:
                current_layer_feats = {
                    'h0': h0.detach().cpu(), # ⚠️ 必须 detach 并转到 cpu，否则显存爆炸
                }
                if self.cfg.use_L1 and h1 is not None:
                    current_layer_feats['h1'] = h1.detach().cpu()
                if self.cfg.use_L2 and h2 is not None:
                    current_layer_feats['h2'] = h2.detach().cpu()

                self.all_layer_descriptors.append(current_layer_feats)

            # E. 能量读出：post-update 归一化，确保每层 coupling/gating/density 参数有梯度
            h0_readout = layer['norm'](h0)
            h1_readout = layer['norm_L1'](h1) if (self.cfg.use_L1 and h1 is not None) else None

            readout_energy_i = getattr(self, f'readout_energy_{i}')
            if self.cfg.use_mole:
                atomic_energy = readout_energy_i(h0_readout, expert_mixing_coefficients, data.batch)
            else:
                atomic_energy = readout_energy_i(h0_readout)
            total_energy = total_energy + scatter_add(atomic_energy, data.batch, dim=0, dim_size=num_graphs)

            if self.cfg.use_direct_force and h1_readout is not None:
                readout_force_i = getattr(self, f'readout_force_{i}')
                if self.cfg.use_mole:
                    atomic_force = readout_force_i(h1_readout, expert_mixing_coefficients, data.batch).squeeze(-1)
                else:
                    atomic_force = readout_force_i(h1_readout).squeeze(-1)
                total_force = total_force + atomic_force

        # 按 dataset id 顺序索引预注册的 scale buffer，避免每次前向重建 Tensor
        scale_per_graph = self.force_scale_buf.astype(total_energy.dtype)[z_dataset].unsqueeze(-1)  # (num_graphs, 1)

        total_energy = total_energy * scale_per_graph
        if self.cfg.use_direct_force:
            scale_per_atom = scale_per_graph[data.batch]  # (num_atoms, 1)
            total_force = total_force * scale_per_atom

        # 根据配置返回结果
        if self.cfg.use_direct_force:
            result = {'energy': total_energy}
            result['force'] = total_force
            return result
        else:
            return total_energy
