import mindspore as ms
from mindspore import nn, mint, ops
from mindspore.common.initializer import initializer, HeUniform, Zero, Constant, Normal
import numpy as np


def _init_linear(linear_cell, weight_init=None, bias_init=None):
    """在模块定义时对 Linear 做权重/偏置初始化。默认 Kaiming 权重、零偏置。"""
    if weight_init is None:
        weight_init = HeUniform()
    if bias_init is None:
        bias_init = Zero()
    linear_cell.weight.set_data(initializer(weight_init, linear_cell.weight.shape))
    if hasattr(linear_cell, 'bias') and linear_cell.bias is not None:
        linear_cell.bias.set_data(initializer(bias_init, linear_cell.bias.shape))
import math
from typing import Dict, Optional, Tuple, List
from src.utils import scatter_add, scatter_mean, HTGPConfig

_EYE3 = ms.Tensor(
    [[1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0],
     [0.0, 0.0, 1.0]],
    dtype=ms.float32,
).view(1, 3, 3, 1)

_MAT_MUL_SYM_H_IDX = ms.Tensor(
    [0, 1, 2, 0, 1, 2, 0, 1, 2,
     3, 4, 5, 3, 4, 5, 3, 4, 5,
     6, 7, 8, 6, 7, 8, 6, 7, 8],
    dtype=ms.int32,
)
_MAT_MUL_SYM_G_IDX = ms.Tensor(
    [0, 3, 6, 1, 4, 7, 2, 5, 8,
     0, 3, 6, 1, 4, 7, 2, 5, 8,
     0, 3, 6, 1, 4, 7, 2, 5, 8],
    dtype=ms.int32,
)
_DIAG_GATHER_IDX = ms.Tensor([0, 4, 8], dtype=ms.int32)
_TRANSPOSE_GATHER_IDX = ms.Tensor([0, 3, 6, 1, 4, 7, 2, 5, 8], dtype=ms.int32)
_CROSS_A_IDX = ms.Tensor([1, 2, 2, 0, 0, 1], dtype=ms.int32)
_CROSS_B_IDX = ms.Tensor([2, 1, 0, 2, 1, 0], dtype=ms.int32)
_CROSS_COEFF = ms.Tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=ms.float32).view(1, 3, 2, 1)
_VEC_CROSS_TENSOR_V_IDX = ms.Tensor(
    [1, 2, 1, 2, 1, 2,
     2, 0, 2, 0, 2, 0,
     0, 1, 0, 1, 0, 1],
    dtype=ms.int32,
)
_VEC_CROSS_TENSOR_T_IDX = ms.Tensor(
    [6, 3, 7, 4, 8, 5,
     0, 6, 1, 7, 2, 8,
     3, 0, 4, 1, 5, 2],
    dtype=ms.int32,
)
_VEC_CROSS_TENSOR_COEFF = ms.Tensor(
    [1.0, -1.0] * 9,
    dtype=ms.float32,
).view(1, 9, 2, 1)
_TENSOR_CROSS_VECTOR_V_IDX = ms.Tensor(
    [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
     2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0,
     0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    dtype=ms.int32,
)
_TENSOR_CROSS_VECTOR_H_IDX = ms.Tensor(
    [6, 3, 2, 1, 7, 4, 5, 4, 8, 5, 8, 7,
     0, 6, 0, 2, 1, 7, 3, 5, 2, 8, 6, 8,
     3, 0, 1, 0, 4, 1, 4, 3, 5, 2, 7, 6],
    dtype=ms.int32,
)
_TENSOR_CROSS_VECTOR_COEFF = ms.Tensor(
    [1.0, -1.0, 1.0, -1.0] * 9,
    dtype=ms.float32,
).view(1, 9, 4, 1)
_COMMUTATOR_DUAL_IDX = ms.Tensor([7, 2, 3], dtype=ms.int32)

# ==========================================
# 🔥 核心 JIT 数学引擎 (安全加速区)
# ==========================================

# @ms.jit
def compute_bessel_math(d: ms.Tensor, prefactor: ms.Tensor, freq_scaled: ms.Tensor) -> ms.Tensor:
    # bf16 数值稳定性：sin/div 等强制用 fp32
    # prefactor = sqrt(2/r_max), freq_scaled = freq/r_max 均在 BesselBasis.__init__ 预计算
    d_f32 = d.astype(ms.float32)
    return prefactor * mint.sin(freq_scaled * d_f32) / (d_f32 + 1e-6)

# @ms.jit
def compute_envelope_math(d: ms.Tensor, inv_r_cut: ms.Tensor) -> ms.Tensor:
    # bf16 数值稳定性：多项式/幂次强制用 fp32
    # inv_r_cut = 1/r_cut 在 PolynomialEnvelope.__init__ 预计算
    # Horner 展开：1 - 10x³ + 15x⁴ - 6x⁵ = 1 + x³(-10 + x(15 - 6x))
    # 减少独立 kernel launch：从 3 次幂运算 → 2 次乘法 + 融合多项式
    d_f32 = d.astype(ms.float32)
    x = mint.clamp(d_f32 * inv_r_cut, min=0.0, max=1.0)
    x2 = x * x
    x3 = x2 * x
    return 1.0 + x3 * (-10.0 + x * (15.0 - 6.0 * x))

# @ms.jit
def compute_l2_basis(rbf_feat: ms.Tensor, r_hat: ms.Tensor) -> ms.Tensor:
    # bf16 数值稳定性：外积/减法强制用 fp32
    r_hat_f32 = r_hat.astype(ms.float32)
    rbf_f32 = rbf_feat.astype(ms.float32)
    outer = r_hat_f32.unsqueeze(2) * r_hat_f32.unsqueeze(1) 
    # eye = mint.eye(3, dtype=r_hat.dtype).unsqueeze(0)
    eye = ms.Tensor([[1, 0, 0],[0, 1, 0],[0, 0, 1]]).unsqueeze(0)
    trace_less = outer - (1.0/3.0) * eye
    return rbf_f32.unsqueeze(1).unsqueeze(1) * trace_less.unsqueeze(-1)

# @ms.jit
def compute_invariants(den0: Optional[ms.Tensor], 
                       den1: Optional[ms.Tensor], 
                       den2: Optional[ms.Tensor]) -> ms.Tensor:
    # ✅ 修复：使用标准类型标注
    invariants: List[ms.Tensor] = []
    
    if den0 is not None:
        invariants.append(den0)
        
    if den1 is not None:
        # bf16 兼容性：平方和/开方用 fp32 计算，再转回输入 dtype
        den1_f32 = den1.astype(ms.float32)
        sq_sum = mint.sum(den1_f32 * den1_f32, dim=1)
        norm = mint.sqrt(sq_sum + 1e-8).astype(den1.dtype)
        invariants.append(norm)
        
    if den2 is not None:
        den2_f32 = den2.astype(ms.float32)
        sq_sum = mint.sum(den2_f32 * den2_f32, dim=(1, 2))
        norm = mint.sqrt(sq_sum + 1e-8).astype(den2.dtype)
        invariants.append(norm)
        
    if len(invariants) > 0:
        return mint.cat(invariants, dim=-1)
    else:
        # 返回空 Tensor (注意处理 device 问题，最好由外部保证 invariants 不为空)
        return mint.zeros(0) 

# @ms.jit
def compute_gating_projections(h_node1: ms.Tensor, 
                               r_hat: ms.Tensor, 
                               scalar_basis: ms.Tensor,
                               src: ms.Tensor, 
                               dst: ms.Tensor) -> ms.Tensor:
    r_hat_uns = r_hat.unsqueeze(-1)
    p_src = mint.sum(h_node1[src] * r_hat_uns, dim=1)
    p_dst = mint.sum(h_node1[dst] * r_hat_uns, dim=1)
    return mint.cat([scalar_basis, p_src, p_dst], dim=-1)


# ==========================================
# 🧩 模块定义 (普通 nn.Module 区)
# ==========================================

class RMSNorm(nn.Cell):
    """
    最简 RMSNorm：
    - 只做按最后一维的 RMS 归一化（不减均值）
    - 仅保留 weight 和 eps（无 bias）
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive int, got {dim!r}")
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = ms.Parameter(mint.ones((self.dim,), dtype=ms.float32), name="weight")

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        orig_dtype = x.dtype
        x_f32 = x.astype(ms.float32)
        # Normalize over all dims except the first (sample) dim:
        # - (N, F)       -> reduce over (1,)
        # - (N, 3, F)    -> reduce over (1, 2)
        # - (N, 3, 3, F) -> reduce over (1, 2, 3)
        # This keeps equivariance for L1/L2 features by using a single scalar
        # scale per sample (no per-component normalization over xyz).
        reduce_axes = tuple(range(1, len(x.shape)))
        rms = mint.sqrt(mint.mean(x_f32 * x_f32, dim=reduce_axes, keepdim=True) + self.eps)
        y = x_f32 / rms
        w = self.weight.astype(ms.float32).reshape((1,) * (len(x.shape) - 1) + (self.dim,))
        return (y * w).astype(orig_dtype)


class SafeEquivariantRMSNorm(nn.Cell):
    """Bounded RMS normalization for low-norm equivariant L1/L2 channels.

    Uses y = x * rms / (rms^2 + min_rms^2). For ordinary nonzero features this
    approaches RMSNorm, while near symmetry-induced zeros it avoids amplifying
    numerical residual directions.
    """

    def __init__(self, dim: int, min_rms: float = 0.05, eps: float = 1e-8):
        super().__init__()
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive int, got {dim!r}")
        if min_rms <= 0:
            raise ValueError(f"min_rms must be positive, got {min_rms!r}")
        self.dim = int(dim)
        self.eps = float(eps)
        self.min_rms_sq = float(min_rms * min_rms)
        self.weight = ms.Parameter(mint.ones((self.dim,), dtype=ms.float32), name="weight")

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        orig_dtype = x.dtype
        x_f32 = x.astype(ms.float32)
        reduce_axes = tuple(range(1, len(x.shape)))
        rms_sq = mint.mean(x_f32 * x_f32, dim=reduce_axes, keepdim=True)
        rms = mint.sqrt(rms_sq + self.eps)
        scale = rms / (rms_sq + self.min_rms_sq)
        y = x_f32 * scale
        w = self.weight.astype(ms.float32).reshape((1,) * (len(x.shape) - 1) + (self.dim,))
        return (y * w).astype(orig_dtype)


class BesselBasis(nn.Cell):
    def __init__(self, r_max: float, num_basis: int = 8):
        super().__init__()
        self.r_max = float(r_max)
        self.num_basis = int(num_basis)
        freq = mint.arange(1, num_basis + 1).float() * np.pi
        # 预计算：消除 construct 中每次的除法和 Python float 运算
        self.register_buffer("freq_scaled", (freq / r_max).astype(ms.float32))
        self.register_buffer("prefactor", ms.Tensor([(2.0 / r_max) ** 0.5], dtype=ms.float32))

    def construct(self, d: ms.Tensor) -> ms.Tensor:
        return compute_bessel_math(d, self.prefactor, self.freq_scaled)

class PolynomialEnvelope(nn.Cell):
    def __init__(self, r_cut: float, p: int = 5):
        super().__init__()
        self.r_cutoff = float(r_cut)
        self.p = int(p)
        # 预计算倒数，消除 construct 中的运行时除法
        self.register_buffer("inv_r_cutoff", ms.Tensor([1.0 / r_cut], dtype=ms.float32))

    def construct(self, d_ij: ms.Tensor) -> ms.Tensor:
        return compute_envelope_math(d_ij, self.inv_r_cutoff)

class GeometricBasis(nn.Cell):
    def __init__(self, config: HTGPConfig):
        super().__init__()
        self.cfg = config
        self.rbf = BesselBasis(config.cutoff, config.num_rbf)
        self.envelope = PolynomialEnvelope(r_cut=config.cutoff)
        self.rbf_mlp = nn.SequentialCell(
            nn.Linear(config.num_rbf, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        _init_linear(self.rbf_mlp[0])
        _init_linear(self.rbf_mlp[2])

    def construct(self, vec_ij, d_ij, batch_row=None, expert_mixing_coefficients=None):
        # bf16 数值稳定性：几何基展开强制 fp32 计算
        vec_f32 = vec_ij.astype(ms.float32)
        d_f32 = d_ij.astype(ms.float32)
        rbf_out = self.rbf(d_f32.unsqueeze(-1))
        raw_rbf = self.rbf_mlp(rbf_out)
        env = self.envelope(d_f32)
        rbf_feat = raw_rbf * env.unsqueeze(-1)

        # ⚠️ r_hat 计算必须在 Python 层保留，确保梯度传导
        r_hat = vec_f32 / (d_f32.unsqueeze(-1) + 1e-6)
        
        basis = {}
        basis[0] = rbf_feat
        
        if self.cfg.use_L1 or self.cfg.use_L2:
            basis[1] = rbf_feat.unsqueeze(1) * r_hat.unsqueeze(-1)
            
        if self.cfg.use_L2:
            basis[2] = compute_l2_basis(rbf_feat, r_hat)
            
        return basis, r_hat

# ==========================================
# 🔥 MOLE (Mixture of Linear Experts) Layer
# ==========================================
# @ms.jit
class MOLE(nn.Cell):
    """
    MOLE (Mixture of Linear Experts) Layer
    
    用多个专家权重矩阵替换单个线性层，路由系数通过原子序数embedding计算。
    对同一个分子保持不变。
    
    Args:
        in_features: 输入特征维度
        out_features: 输出特征维度
        num_experts: 专家数量（默认8）
        routing_dim: 路由网络的输入维度（必需，通常等于embedding维度）
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int = 8,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.has_bias = bias
        self.expert_weights = ms.Parameter(
            initializer(HeUniform(), [num_experts, out_features, in_features]),
            name='expert_weights'
        )
        if bias:
            self.expert_biases = ms.Parameter(
                initializer(Zero(), [num_experts, out_features]),
                name='expert_biases'
            )
        # 推理加速用：fused 后的合并权重（非 Parameter，不参与训练）
        self._is_fused: bool = False
        self._fused_weight = None   # [out_features, in_features]
        self._fused_bias = None     # [out_features] or None

    # ------------------------------------------------------------------
    # 推理加速：专家权重合并
    # ------------------------------------------------------------------
    def fuse(self, coeffs: ms.Tensor) -> None:
        """
        将 K 个专家权重按路由系数加权合并为单个等效线性层权重，
        后续 construct 自动走快速路径（标准单次 matmul）。

        仅用于推理。对同一分子组成只需调用一次，在结构优化 / MD 全程有效。

        Args:
            coeffs: [K] 或 [1, K]，当前分子的路由系数（由模型 routing_mlp 输出）
        """
        coeffs_1d = coeffs.view(self.num_experts).astype(self.expert_weights.dtype)
        # merged[o, i] = Σ_e coeffs[e] * expert_weights[e, o, i]
        # expert_weights: [K, F_out, F_in] → 广播乘后在 dim=0 求和
        self._fused_weight = (
            coeffs_1d.view(self.num_experts, 1, 1) * self.expert_weights
        ).sum(axis=0)                                       # [F_out, F_in]
        if self.has_bias:
            # [F_out] = [K] @ [K, F_out]
            self._fused_bias = mint.matmul(coeffs_1d, self.expert_biases)
        self._is_fused = True

    def unfuse(self) -> None:
        """清除合并权重，恢复原始 K-专家路径。"""
        self._fused_weight = None
        self._fused_bias = None
        self._is_fused = False

    def construct(
        self,
        x: ms.Tensor,
        expert_mixing_coefficients: Optional[ms.Tensor] = None,
        batch: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        """
        前向传播。fuse() 调用后自动走快速路径（1 次 matmul），
        否则走原始 K-专家路径。

        Args:
            x: 输入特征 [N, in_features] 或 [N, spatial, in_features]
            expert_mixing_coefficients: 专家混合系数 [num_graphs, K]，fuse 后可省略
            batch: [N]，将原子映射到所属分子，fuse 后可省略
        """
        # ---- 快速路径：fuse() 已将专家合并为单个权重矩阵 ----
        if self._is_fused:
            x_shape = x.shape
            in_features = x_shape[-1]
            # 展平空间维度，做单次 matmul
            out = mint.matmul(x.view(-1, in_features), self._fused_weight.T)
            if self.has_bias:
                out = out + self._fused_bias
            return out.view(x_shape[:-1] + (self.out_features,))

        # ---- 标准路径：K 专家加权求和 ----
        coeffs_for_samples = expert_mixing_coefficients[batch]  # [N, K]

        N = coeffs_for_samples.shape[0]
        x_shape = x.shape
        in_features = x_shape[-1]

        x_reshaped = x.view(N, -1, in_features)
        spatial_size = x_reshaped.shape[1]

        # 所有专家一次性 matmul: [N, spatial, K*F_out]
        expert_weights_flat = self.expert_weights.view(self.num_experts * self.out_features, in_features)
        all_expert_outputs = mint.matmul(x_reshaped, expert_weights_flat.T)
        all_expert_outputs = all_expert_outputs.view(N, spatial_size, self.num_experts, self.out_features)

        # 加权求和专家维度
        coeffs_expanded = coeffs_for_samples.unsqueeze(1).unsqueeze(-1)  # [N, 1, K, 1]
        output_reshaped = (all_expert_outputs * coeffs_expanded).sum(dim=2)  # [N, spatial, F_out]

        if self.has_bias:
            mixed_bias = mint.matmul(coeffs_for_samples, self.expert_biases)
            output_reshaped = output_reshaped + mixed_bias.unsqueeze(1)

        output_shape = x_shape[:-1] + (self.out_features,)
        return output_reshaped.view(output_shape)


class MOLEEnergyReadout(nn.Cell):
    """使用 MOLE 的每层能量读出，节点级操作，可安全扩大参数量。"""
    def __init__(self, config: HTGPConfig):
        super().__init__()
        self.F = config.hidden_dim
        K = config.mole_num_experts
        self.mole1 = MOLE(self.F, self.F, num_experts=K, bias=True)
        self.act1 = nn.SiLU()
        self.mole2 = MOLE(self.F, self.F, num_experts=K, bias=True)
        self.act2 = nn.SiLU()
        self.mole3 = MOLE(self.F, 1, num_experts=K, bias=True)

    def construct(self, h0: ms.Tensor, expert_mixing_coefficients: ms.Tensor, batch: ms.Tensor) -> ms.Tensor:
        x = self.mole1(h0, expert_mixing_coefficients, batch)
        x = self.act1(x)
        x = self.mole2(x, expert_mixing_coefficients, batch)
        x = self.act2(x)
        return self.mole3(x, expert_mixing_coefficients, batch)


class MOLEForceReadout(nn.Cell):
    """使用 MOLE 的每层力读出，节点级，输出 1 维标量再在 L1 上组合为 3D 力。"""
    def __init__(self, config: HTGPConfig):
        super().__init__()
        self.F = config.hidden_dim
        K = config.mole_num_experts
        self.mole = MOLE(self.F, 1, num_experts=K, bias=False)

    def construct(self, h1: ms.Tensor, expert_mixing_coefficients: ms.Tensor, batch: ms.Tensor) -> ms.Tensor:
        return self.mole(h1, expert_mixing_coefficients, batch)


def optimized_cross(a, b):
    # 直接在分量维 (dim=1) 上 gather，避免 transpose 产生非连续视图。
    a_terms = ops.gather(a, _CROSS_A_IDX, 1).reshape(a.shape[0], 3, 2, a.shape[2])
    b_terms = ops.gather(b, _CROSS_B_IDX, 1).reshape(b.shape[0], 3, 2, b.shape[2])
    coeff = _CROSS_COEFF.astype(a.dtype)
    return mint.sum(a_terms * b_terms * coeff, dim=2)

class LeibnizCoupling(nn.Cell): 
    def __init__(self, config: HTGPConfig):
        super().__init__()
        self.cfg = config
        self.F = config.hidden_dim
        self.path_weights = nn.CellDict()
        out_path_counts = {0: 0, 1: 0, 2: 0}

        for path_key, active in config.active_paths.items():
            if not active:
                continue
            l_in, l_edge, l_out, _ = path_key
            if (l_in == 2 or l_edge == 2 or l_out == 2) and not config.use_L2:
                continue
            if (l_in == 1 or l_edge == 1 or l_out == 1) and not config.use_L1:
                continue
            out_path_counts[l_out] += 1
        
        for path_key, active in config.active_paths.items():
            if not active: continue
            l_in, l_edge, l_out, _ = path_key
            if (l_in == 2 or l_edge == 2 or l_out == 2) and not config.use_L2: continue
            if (l_in == 1 or l_edge == 1 or l_out == 1) and not config.use_L1: continue

            name = f"{l_in}_{l_edge}_{l_out}_{path_key[3]}"
            out_scale = out_path_counts[l_out] ** -0.5 if out_path_counts[l_out] > 0 else 1.0
            if config.use_mole:
                block = MOLE(self.F, self.F, num_experts=config.mole_num_experts)
                block.expert_weights.set_data(block.expert_weights.data * out_scale)
                self.path_weights[name] = block
                # 每条路径一个标量 scale，MOLE 时 K 组由路由混合，初始化 1（中性）
                self.insert_param_to_cell(
                    f'path_scale_{name}',
                    ms.Parameter(
                        initializer(Constant(1.0), [config.mole_num_experts]),
                        name=f'path_scale_{name}'
                    )
                )
            else:
                lin = nn.Linear(self.F, self.F, bias=False)
                _init_linear(lin)
                lin.weight.set_data(lin.weight.data * out_scale)
                self.path_weights[name] = lin
                # 非 MOLE：单个可学习标量 scale
                self.insert_param_to_cell(
                    f'path_scale_{name}',
                    ms.Parameter(
                        initializer(Constant(1.0), [1]),
                        name=f'path_scale_{name}'
                    )
                )

        # self.inv_sqrt_f = self.F ** -0.5

    def construct(
        self, 
        h_nodes: Dict[int, ms.Tensor], 
        basis_edges: Dict[int, ms.Tensor], 
        edge_index,
        expert_mixing_coefficients: Optional[ms.Tensor] = None,
        batch: Optional[ms.Tensor] = None,
    ):
        src, _ = edge_index
        messages: Dict[int, List[ms.Tensor]] = {0: [], 1: [], 2: []}
        
        for path_key, active in self.cfg.active_paths.items():
            if not active: continue
            l_in, l_edge, l_out, op_type = path_key
            
            if basis_edges.get(l_edge) is None: continue
            
            layer_name = f"{l_in}_{l_edge}_{l_out}_{op_type}"
            if layer_name not in self.path_weights: continue
            
            if h_nodes.get(l_in) is None: continue 
            else: inp = h_nodes[l_in]
            
            if self.cfg.use_mole and expert_mixing_coefficients is not None and batch is not None:
                # ★ 节点维度先变换，再 gather 到边：MOLE(inp)[src]  ==  MOLE(inp[src])
                #   因为路由系数 coeffs[batch[n]] 对同一节点的所有出边相同，gather 与线性变换可交换。
                #   这样中间张量从 [E, spatial, K*F] 缩减到 [N, spatial, K*F]（N << E），
                #   L0: ~1MB,  L1: ~3MB,  L2: ~9MB（E≈44500, N≈500, K=8）
                h_trans = self.path_weights[layer_name](inp, expert_mixing_coefficients, batch)[src]
            else:
                # ★ 同 MOLE 路径：Linear 也在节点维度先变换，再 gather 到边
                #   Linear(inp)[src] == Linear(inp[src])，但前者峰值张量是 [N,...,F] 而非 [E,...,F]
                #   L1 路径节省 ~89x（[E,3,F] → [N,3,F]），L2 路径节省 ~89x（[E,3,3,F] → [N,3,3,F]）
                h_trans = self.path_weights[layer_name](inp)[src]
            geom = basis_edges[l_edge]
            res = None
            
            # --- Operation Logic ---
            if op_type == 'prod':
                if l_in == 0 and l_edge == 0: res = h_trans * geom
                elif l_in == 0 and l_edge == 1: res = h_trans.unsqueeze(1) * geom
                elif l_in == 0 and l_edge == 2: res = h_trans.unsqueeze(1).unsqueeze(1) * geom
                elif l_in == 1 and l_edge == 0: res = h_trans * geom.unsqueeze(1)
                elif l_in == 2 and l_edge == 0: res = h_trans * geom.unsqueeze(1).unsqueeze(1)
            elif op_type == 'dot':
                res = mint.sum(h_trans * geom, dim=1)
            elif op_type == 'cross':
                g = geom
                if g.dim() == 2: g = g.unsqueeze(-1)
                # res = ops.cross(h_trans, g, dim=1)
                res = optimized_cross(h_trans, g)
            elif op_type == 'outer':
                outer = h_trans.unsqueeze(2) * geom.unsqueeze(1)
                trace = sum_diag_gather(outer)
                outer_flat = outer.reshape(outer.shape[0], 9, outer.shape[3])
                sym = 0.5 * (outer_flat + ops.gather(outer_flat, _TRANSPOSE_GATHER_IDX, 1)).reshape(outer.shape[0], 3, 3, outer.shape[3])
                eye = _EYE3.astype(h_trans.dtype)
                res = sym - (1.0/3.0) * trace.unsqueeze(1).unsqueeze(1) * eye
            elif op_type == 'mat_vec':
                # res = torch.einsum('eijf, ejf -> eif', h_trans, geom)
                res = (h_trans * geom.unsqueeze(1)).sum(dim=2)
            elif op_type == 'vec_mat':
                # res = torch.einsum('eif, eijf -> ejf', h_trans, geom)
                res = (h_trans.unsqueeze(2) * geom).sum(dim=1)
            elif op_type == 'double_dot':
                res = mint.sum(h_trans * geom, dim=(1, 2))
            elif op_type == 'mat_mul_sym':
                e, i, k, f = h_trans.shape
                _, _, j, _ = geom.shape
                h_flat = h_trans.reshape(e, i * k, f)
                g_flat = geom.reshape(e, k * j, f)
                h_terms = ops.gather(h_flat, _MAT_MUL_SYM_H_IDX, 1).reshape(e, i * j, k, f)
                g_terms = ops.gather(g_flat, _MAT_MUL_SYM_G_IDX, 1).reshape(e, i * j, k, f)
                raw_flat = mint.sum(h_terms * g_terms, dim=2)
                sym = 0.5 * (raw_flat + ops.gather(raw_flat, _TRANSPOSE_GATHER_IDX, 1)).reshape(e, i, j, f)
                trace = sum_diag_gather(sym)
                eye = _EYE3.astype(h_trans.dtype)
                res = sym - (1.0/3.0) * trace.unsqueeze(1).unsqueeze(1) * eye
            elif op_type == 'vec_cross_tensor':
                # h_trans: (E, 3, F)  [Vector v]
                # geom:    (E, 3, 3, F) [Tensor T]
                # 预编码每个输出位置的 2 个乘法项，避免沿 dim1 切矢量、沿 dim1 切张量。
                e, _, _, f = geom.shape
                geom_flat = geom.reshape(e, 9, f)
                v_terms = ops.gather(h_trans, _VEC_CROSS_TENSOR_V_IDX, 1).reshape(e, 9, 2, f)
                t_terms = ops.gather(geom_flat, _VEC_CROSS_TENSOR_T_IDX, 1).reshape(e, 9, 2, f)
                coeff = _VEC_CROSS_TENSOR_COEFF.astype(h_trans.dtype)
                res_raw_flat = mint.sum(v_terms * t_terms * coeff, dim=2)
                res_sym = 0.5 * (res_raw_flat + ops.gather(res_raw_flat, _TRANSPOSE_GATHER_IDX, 1)).reshape(e, 3, 3, f)
                trace = sum_diag_gather(res_sym).unsqueeze(1).unsqueeze(1)
                eye = _EYE3.astype(h_trans.dtype)
                res = res_sym - (1.0/3.0) * trace * eye

            elif op_type == 'tensor_cross_vector':
                # h_trans: (E, 3, 3, F) [Tensor H]
                # geom:    (E, 3, F)    [Vector v]
                # 物理含义: 旋转换位子 term1 + term2
                #   term1 = v × H (对 H 每列做叉积)
                #   term2 = (v × H^T)^T (对 H 每行做叉积后转置)
                # 预编码索引一次 gather 出 9 个输出位置所需的 4 个乘法项，
                # 避免对 dim1/dim2 的逐行逐列切片触发 ViewCopy。
                e, _, _, f = h_trans.shape
                h_flat = h_trans.reshape(e, 9, f)
                v_terms = ops.gather(geom, _TENSOR_CROSS_VECTOR_V_IDX, 1).reshape(e, 9, 4, f)
                h_terms = ops.gather(h_flat, _TENSOR_CROSS_VECTOR_H_IDX, 1).reshape(e, 9, 4, f)
                coeff = _TENSOR_CROSS_VECTOR_COEFF.astype(h_trans.dtype)
                res_raw_flat = mint.sum(v_terms * h_terms * coeff, dim=2)
                res_sym = 0.5 * (res_raw_flat + ops.gather(res_raw_flat, _TRANSPOSE_GATHER_IDX, 1)).reshape(e, 3, 3, f)
                trace = sum_diag_gather(res_sym).unsqueeze(1).unsqueeze(1)
                eye = _EYE3.astype(h_trans.dtype)
                res = res_sym - (1.0/3.0) * trace * eye

            elif op_type == 'tensor_commutator':
                # h_trans, geom: (E, 3, 3, F)
                # 计算 M = AB - BA，提取反对称对偶向量 (E, 3, F)
                E_tc, _, _, F_tc = h_trans.shape
                h_flat = h_trans.reshape(E_tc, 9, F_tc)
                g_flat = geom.reshape(E_tc, 9, F_tc)
                ab_terms = (
                    ops.gather(h_flat, _MAT_MUL_SYM_H_IDX, 1).reshape(E_tc, 9, 3, F_tc) *
                    ops.gather(g_flat, _MAT_MUL_SYM_G_IDX, 1).reshape(E_tc, 9, 3, F_tc)
                )
                ba_terms = (
                    ops.gather(g_flat, _MAT_MUL_SYM_H_IDX, 1).reshape(E_tc, 9, 3, F_tc) *
                    ops.gather(h_flat, _MAT_MUL_SYM_G_IDX, 1).reshape(E_tc, 9, 3, F_tc)
                )
                M_flat = mint.sum(ab_terms - ba_terms, dim=2)
                res = ops.gather(M_flat, _COMMUTATOR_DUAL_IDX, 1)

            if res is not None:
                # 每条路径的可学习标量 scale
                path_scale = getattr(self, f'path_scale_{layer_name}')
                if self.cfg.use_mole and expert_mixing_coefficients is not None:
                    # [num_graphs] = [num_graphs, K] @ [K] → gather 到边
                    mol_scale = mint.matmul(expert_mixing_coefficients, path_scale)  # [num_graphs]
                    edge_scale = mol_scale[batch[src]]  # [E]
                else:
                    edge_scale = path_scale[0]  # 标量
                # broadcast 到任意 l_out 形状: [E], [E,3,F], [E,3,3,F]
                for _ in range(res.dim() - 1):
                    edge_scale = edge_scale.unsqueeze(-1)
                res = res * edge_scale
                messages[l_out].append(res)

        final_msgs: Dict[int, Optional[ms.Tensor]] = {}
        for l in [0, 1, 2]:
            if len(messages[l]) > 0:
                final_msgs[l] = sum(messages[l])
            else:
                final_msgs[l] = None
        return final_msgs


def sum_diag_gather(x: ms.Tensor) -> ms.Tensor:
    """
    Equivalent to: y = einsum('eiif->ef', x)
    x: (E, I, I, F)
    y: (E, F) where y[e, f] = sum_i x[e, i, i, f]
    """
    E, I, _, F = x.shape
    if E == 0:
        # Guard: ops.gather_nd backward uses ScatterNd which rejects shape (0, I*I, F).
        return mint.zeros((0, F), dtype=x.dtype)

    if I != 3 or x.shape[2] != 3:
        raise ValueError(f"sum_diag_gather expects shape [E, 3, 3, F], got {tuple(x.shape)}")

    diag = ops.gather(x.reshape(E, 9, F), _DIAG_GATHER_IDX, 1)
    return mint.sum(diag, dim=1)


class PhysicsGating(nn.Cell):
    def __init__(self, config: HTGPConfig):
        super().__init__()
        self.cfg = config
        self.F = config.hidden_dim
        self.use_mole = config.use_mole
        K = config.mole_num_experts if config.use_mole else 0

        if config.use_mole:
            self.W_query = MOLE(self.F, self.F, num_experts=K)
            self.W_key = MOLE(self.F, self.F, num_experts=K)
        else:
            self.W_query = nn.Linear(self.F, self.F, bias=False)
            self.W_key = nn.Linear(self.F, self.F, bias=False)
            _init_linear(self.W_query)
            _init_linear(self.W_key)

        self.phys_bias_mlp = nn.SequentialCell(
            nn.Linear(3 * self.F, self.F, bias=False),
            nn.SiLU(),
            nn.Linear(self.F, 3 * self.F, bias=False)
        )
        _init_linear(self.phys_bias_mlp[0])
        self.phys_bias_mlp[-1].weight.set_data(
            initializer(Zero(), self.phys_bias_mlp[-1].weight.shape))
        self.channel_mixer = nn.Linear(self.F, 3 * self.F, bias=True)
        self.channel_mixer.weight.set_data(
            initializer(Normal(sigma=0.01), self.channel_mixer.weight.shape))
        self.channel_mixer.bias.set_data(
            initializer(Zero(), self.channel_mixer.bias.shape))
        # gate_scale 固定为 2.0 的 buffer（不可学习）：
        # sigmoid(0) × 2.0 = 1.0，初始 gate 精确全开，且梯度在 sigmoid 拐点最强
        self.gate_scale = ms.Parameter(mint.ones(1) * 2.0)

    def construct(self, msgs, h_node0, scalar_basis, r_hat, h_node1, edge_index,
                  capture_weights=False, expert_mixing_coefficients=None, batch=None):
        if not self.cfg.use_gating: return msgs

        src, dst = edge_index

        if h_node1 is not None:
            phys_input = compute_gating_projections(h_node1, r_hat, scalar_basis, src, dst)
            split_idx = scalar_basis.shape[-1]
            p_ij = phys_input[:, split_idx:]
        else:
            p_ij = mint.zeros((scalar_basis.shape[0], 2 * self.F))
            phys_input = mint.cat([scalar_basis, p_ij], dim=-1)

        if self.use_mole and expert_mixing_coefficients is not None and batch is not None:
            q_nodes = self.W_query(h_node0, expert_mixing_coefficients, batch)
            k_nodes = self.W_key(h_node0, expert_mixing_coefficients, batch)
        else:
            q_nodes = self.W_query(h_node0)
            k_nodes = self.W_key(h_node0)

        q = q_nodes[dst]
        k = k_nodes[src]
        chem_score = q * k
        chem_logits = self.channel_mixer(chem_score)
        phys_logits = self.phys_bias_mlp(phys_input)

        raw_gates = chem_logits + phys_logits
        gates = mint.sigmoid(raw_gates) * self.gate_scale
        
        if capture_weights: self.scalar_basis_captured = scalar_basis.detach()
        if capture_weights: self.p_ij_captured = p_ij.detach()
        if capture_weights: self.chem_logits_captured = chem_logits.detach()
        if capture_weights: self.phys_logits_captured = phys_logits.detach()

        g_list = mint.split(gates, self.F, dim=-1)
        g0, g1, g2 = [g.contiguous() for g in g_list]

        if capture_weights: self.g0_captured = g0.detach()
        if capture_weights: self.g1_captured = g1.detach()
        if capture_weights: self.g2_captured = g2.detach()
        
        out_msgs: Dict[int, ms.Tensor] = {}
        if 0 in msgs and msgs[0] is not None: out_msgs[0] = msgs[0] * g0
        if 1 in msgs and msgs[1] is not None: out_msgs[1] = msgs[1] * g1.unsqueeze(1)
        if 2 in msgs and msgs[2] is not None: out_msgs[2] = msgs[2] * g2.unsqueeze(1).unsqueeze(1)
            
        return out_msgs

class CartesianDensityBlock(nn.Cell):
    def __init__(self, config: HTGPConfig):
        super().__init__()
        self.F = config.hidden_dim
        self.cfg = config
        self.use_mole = config.use_mole
        K = config.mole_num_experts if config.use_mole else 0

        in_dim = 0
        if config.use_L0: in_dim += self.F
        if config.use_L1: in_dim += self.F
        if config.use_L2: in_dim += self.F

        expansion_factor = 3
        hidden_width = self.F * expansion_factor

        if config.use_mole:
            self.scalar_fc1 = MOLE(in_dim, hidden_width, num_experts=K, bias=True)
            self.scalar_norm = RMSNorm(hidden_width)
            self.scalar_act = nn.SiLU()
            self.scalar_fc2 = MOLE(hidden_width, self.F, num_experts=K, bias=True)
        else:
            self.scalar_update_mlp = nn.SequentialCell(
                nn.Linear(in_dim, hidden_width),
                RMSNorm(hidden_width),
                nn.SiLU(),
                nn.Linear(hidden_width, self.F)
            )
            _init_linear(self.scalar_update_mlp[0])
            _init_linear(self.scalar_update_mlp[3])

        if config.use_L1:
            if config.use_mole:
                self.L1_linear = MOLE(self.F, self.F, num_experts=K)
            else:
                self.L1_linear = nn.Linear(self.F, self.F, bias=False)
                _init_linear(self.L1_linear)
           
        if config.use_L2:
            if config.use_mole:
                self.L2_linear = MOLE(self.F, self.F, num_experts=K)
            else:
                self.L2_linear = nn.Linear(self.F, self.F, bias=False)
                _init_linear(self.L2_linear)

        scale_out_dim = 0
        if config.use_L1: scale_out_dim += self.F
        if config.use_L2: scale_out_dim += self.F
        self.has_scale = scale_out_dim > 0

        # 输入拼接 delta_h0 + ||den1|| + ||den2||，给 L1/L2 提供直接梯度通道。
        scale_input_dim = self.F  # delta_h0
        if config.use_L1: scale_input_dim += self.F  # ||den1||
        if config.use_L2: scale_input_dim += self.F  # ||den2||

        if scale_out_dim > 0:
            if config.use_mole:
                self.scale_fc1 = MOLE(scale_input_dim, self.F, num_experts=K, bias=True)
                self.scale_act = nn.SiLU()
                self.scale_fc2 = MOLE(self.F, scale_out_dim, num_experts=K, bias=True)
                # alpha 初始化 ≈ 1，防止冷启动陷阱
                # self.scale_fc2.expert_weights.set_data(
                #     self.scale_fc2.expert_weights.data * 0.1)
                self.scale_fc2.expert_biases.set_data(
                    initializer(Constant(1.0), self.scale_fc2.expert_biases.shape))
                self.scale_mlp = None
            else:
                self.scale_mlp = nn.SequentialCell(
                    nn.Linear(scale_input_dim, self.F),
                    nn.SiLU(),
                    nn.Linear(self.F, scale_out_dim)
                )
                _init_linear(self.scale_mlp[0])
                _init_linear(self.scale_mlp[2])
                # self.scale_mlp[2].weight.set_data(
                #     self.scale_mlp[2].weight.data * 0.1)
                self.scale_mlp[2].bias.set_data(
                    initializer(Constant(1.0), self.scale_mlp[2].bias.shape))
                self.scale_fc1 = None
        else:
            self.scale_mlp = None
            self.scale_fc1 = None

    def construct(self, msgs: Dict[int, ms.Tensor], index: ms.Tensor, num_nodes: int,
                  inv_sqrt_deg: ms.Tensor,
                  expert_mixing_coefficients: Optional[ms.Tensor] = None,
                  batch: Optional[ms.Tensor] = None):
        # 1. 密度聚合
        densities: Dict[int, Optional[ms.Tensor]] = {}
        densities[0], densities[1], densities[2] = None, None, None

        for l in [0, 1, 2]:
            if l in msgs and msgs[l] is not None:
                agg = scatter_add(msgs[l], index, dim=0, dim_size=num_nodes)
                if agg.dim() > 1:
                    view_shape = (num_nodes,) + (1,) * (agg.dim() - 1)
                    scale = inv_sqrt_deg.view(view_shape).astype(agg.dtype)
                else:
                    scale = inv_sqrt_deg.astype(agg.dtype)
                densities[l] = agg * scale
            else:
                densities[l] = None

        # 2. 提取不变量
        concat = compute_invariants(densities[0], densities[1], densities[2])

        # 3. 标量更新
        if concat.numel() > 0:
            if self.use_mole and expert_mixing_coefficients is not None and batch is not None:
                x = self.scalar_fc1(concat, expert_mixing_coefficients, batch)
                x = self.scalar_norm(x)
                x = self.scalar_act(x)
                delta_h0 = self.scalar_fc2(x, expert_mixing_coefficients, batch)
            elif self.use_mole:
                # fused MOLE path — experts already baked into single weight matrix
                x = self.scalar_fc1(concat)
                x = self.scalar_norm(x)
                x = self.scalar_act(x)
                delta_h0 = self.scalar_fc2(x)
            else:
                delta_h0 = self.scalar_update_mlp(concat)
        else:
            delta_h0 = mint.zeros((num_nodes, self.F))

        # 4. 矢量更新
        delta_h1 = None
        delta_h2 = None

        # scale_input = [delta_h0, ||den1||, ||den2||]，给 L1/L2 提供直接梯度通道
        has_scale = self.scale_fc1 is not None or self.scale_mlp is not None
        if has_scale:
            scale_parts = [delta_h0]
            if self.cfg.use_L1:
                if densities[1] is not None:
                    den1_f32 = densities[1].astype(ms.float32)
                    l1_norm = mint.sqrt(mint.sum(den1_f32 * den1_f32, dim=1) + 1e-8).astype(delta_h0.dtype)
                else:
                    l1_norm = mint.zeros((num_nodes, self.F), dtype=delta_h0.dtype)
                scale_parts.append(l1_norm)
            if self.cfg.use_L2:
                if densities[2] is not None:
                    den2_f32 = densities[2].astype(ms.float32)
                    l2_norm = mint.sqrt(mint.sum(den2_f32 * den2_f32, dim=(1, 2)) + 1e-8).astype(delta_h0.dtype)
                else:
                    l2_norm = mint.zeros((num_nodes, self.F), dtype=delta_h0.dtype)
                scale_parts.append(l2_norm)

            scale_input = mint.cat(scale_parts, dim=-1) if len(scale_parts) > 1 else scale_parts[0]

            if self.use_mole and expert_mixing_coefficients is not None and batch is not None:
                x = self.scale_fc1(scale_input, expert_mixing_coefficients, batch)
                x = self.scale_act(x)
                scales = self.scale_fc2(x, expert_mixing_coefficients, batch)
            elif self.use_mole:
                # fused MOLE path
                x = self.scale_fc1(scale_input)
                x = self.scale_act(x)
                scales = self.scale_fc2(x)
            else:
                scales = self.scale_mlp(scale_input)
            curr_dim = 0

            if self.cfg.use_L1 and densities[1] is not None:
                alpha1 = scales[:, curr_dim : curr_dim + self.F]
                if self.use_mole and expert_mixing_coefficients is not None and batch is not None:
                    h1_mixed = self.L1_linear(densities[1], expert_mixing_coefficients, batch)
                elif self.use_mole:
                    h1_mixed = self.L1_linear(densities[1])
                else:
                    h1_mixed = self.L1_linear(densities[1])
                delta_h1 = h1_mixed * alpha1.unsqueeze(1)
                curr_dim += self.F

            if self.cfg.use_L2 and densities[2] is not None:
                alpha2 = scales[:, curr_dim : curr_dim + self.F]
                if self.use_mole and expert_mixing_coefficients is not None and batch is not None:
                    h2_mixed = self.L2_linear(densities[2], expert_mixing_coefficients, batch)
                elif self.use_mole:
                    h2_mixed = self.L2_linear(densities[2])
                else:
                    h2_mixed = self.L2_linear(densities[2])
                delta_h2 = h2_mixed * alpha2.unsqueeze(1).unsqueeze(1)

        return delta_h0, delta_h1, delta_h2
