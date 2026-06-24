from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
import mindspore as ms
from mindspore import ops, mint
from typing import Optional

# ==========================================
# 1. 配置与消融控制 (Configuration & Ablation)
# ==========================================
@dataclass
class HTGPConfig:
    # --- 基础超参 ---
    hidden_dim: int = 128
    num_layers: int = 2
    cutoff: float = 6.0
    num_rbf: int = 12
    atom_types_map: list = field(default_factory=lambda: [x for x in range(1, 101)])

    min_charge: int = -10
    max_charge: int = 10
    max_spin: int = 11
    num_dataset: int = 6
    dataset_types: dict = field(default_factory=lambda: {'OMol25': 0, 'OMat24': 1, 'OMC': 2, 'OC20': 3, 'OC22': 4, 'OC25': 5})
    force_scale: dict = field(default_factory=lambda: {'OMol25': 1.4433652738511566, 'OMat24': 2.836595316534465, 'OMC': 1.7988458808331595, 'OC20': 0.20358776776484094, 'OC22': 0.25427970140320844, 'OC25': 1.14235185126804})
    stress_datasets: dict = field(default_factory=lambda: {'OMol25': False, 'OMat24': True, 'OMC': True, 'OC20': False, 'OC22': False, 'OC25': False})  # 各数据集是否计算 stress loss

    # --- 模块开关 (Global Flags) ---
    use_L0: bool = True   # 标量通道 (必需)
    use_L1: bool = True   # 矢量通道 (偶极/力)
    use_L2: bool = True   # 张量通道 (四极/应力)
    use_gating: bool = True      # 是否开启物理投影门控 (Physics Gating)
    avg_neighborhood: float = 89
    use_long_range: bool = False  # 是否开启隐式长程场 (Latent Long-Range)
    use_charge: bool = False
    use_vdw: bool = False
    use_dipole: bool = False

    use_mole: bool = True  # 是否使用MOLE层
    mole_num_experts: int = 32  # MOLE层的专家数量
    mole_dropout: float = 0.05
    
    use_direct_force: bool = False
    # 混合精度 / 低精度训练（显存优化：激活/部分权重用 bf16，约可省 30–50% 激活显存）
    # 注意：需要硬件支持（如 Ascend / 部分 GPU）。开启后通常需要将 loss 计算保持在 fp32 更稳定。
    use_bf16: bool = False
    # 梯度重计算（显存优化：用计算换显存，反向时重算部分层的前向激活）
    # 例如 (5, 10) 表示第 5～9 层在反向时重算，不存激活。None 或 () 表示不开启。
    use_recompute: bool = False

    # 微调
    steps_per_epoch: Optional[int] = None # 控制要不要数据都用上
    long_range_scale: float = 1

    # --- 莱布尼茨耦合路径字典 (Coupling Matrix) ---
    # 格式: (Node_L_in, Edge_L_in, Target_L_out, Operation_Type)
    # 通过设置 True/False 精确控制每一条物理路径
    active_paths: Dict[Tuple[int, int, int, str], bool] = field(default_factory=lambda: {
        # === A. 基础生成 (Generation) ===
        (0, 0, 0, 'prod'): True,   # s * s -> s
        (0, 1, 1, 'prod'): True,   # s * v -> v (极化)
        (0, 2, 2, 'prod'): True,   # s * t -> t (各向异性)

        # === B. 几何反馈 (Feedback) ===
        (1, 0, 1, 'prod'): True,   # v * s -> v (缩放)
        (1, 1, 0, 'dot'):  True,   # v . v -> s (投影/角度能量)
        
        # === [新增] 补全 1x1 的最后一块拼图 ===
        # v x v -> v (叉积: 捕捉扭转/手性)
        (1, 1, 1, 'cross'): True,  

        # === C. 高阶交互 ===
        (1, 1, 2, 'outer'): True,      # v x v -> t (偶极生成四极)
        (2, 0, 2, 'prod'): True,       # t * s -> t
        
        # 你的疑虑点：完全正确，必须保留
        (2, 1, 1, 'mat_vec'): True,    # t . v -> v (形状导向力)
        (1, 2, 1, 'vec_mat'): True,    # v . t -> v (反向)
        
        (2, 2, 0, 'double_dot'): True, # t : t -> s (形状匹配)
        
        # 可选：如果你想省显存，可以把下面这个关掉，影响不大
        (2, 2, 2, 'mat_mul_sym'): True,

        # # 矢量节点 x 张量边 -> 张量更新
        # (1, 2, 2, 'vec_cross_tensor'): True,
        # # (Tensor, Vector) -> Tensor: 旋转/极化响应
        # (2, 1, 2, 'tensor_cross_vector'): True, 
        # # (Tensor, Tensor) -> Vector: 换位子/涡旋流
        # (2, 2, 1, 'tensor_commutator' ): True,
    })


# # ==========================================
# # 0. 自定义 scatter_add (替代 torch_scatter)
# # ==========================================
# def scatter_add(src: ms.Tensor, index: ms.Tensor, dim: int = 0, dim_size: int = None) -> ms.Tensor:
#     """
#     自定义 scatter_add 实现，无需安装 torch_scatter 库。
#     利用 torch.index_add_ 实现高性能聚合。
#     """
#     if dim_size is None:
#         if index.numel() == 0:
#             dim_size = 0
#         else:
#             dim_size = int(index.max().item()) + 1
#     else:
#         dim_size = int(dim_size)

#     # 构建输出张量
#     out_size = list(src.size())
#     out_size[dim] = dim_size
#     out = torch.zeros(out_size, dtype=src.dtype, device=src.device)
    
#     # 针对 GNN 最常用的 dim=0 且 index 为 1D 的情况进行优化
#     # index_add_ 会自动处理 src 后面的维度 (e.g. [E, 3, F] -> [N, 3, F])
#     if dim == 0 and index.dim() == 1:
#         return out.index_add_(0, index, src)
        
#     # 通用路径 (处理非 dim=0 或 index 为多维的情况)
#     if index.dim() != src.dim():
#         view_shape = [1] * src.dim()
#         view_shape[dim] = -1
#         index = index.view(view_shape).expand_as(src)
        
#     return out.scatter_add_(dim, index, src)


def scatter_add(src: ms.Tensor, index: ms.Tensor, dim: int = 0, dim_size: Optional[int] = None) -> ms.Tensor:
    """
    自定义 scatter_add 实现 (Python 版)
    
    Args:
        src: 源数据 [E, F]
        index: 索引 [E]
        dim: 聚合维度 (通常为 0)
        dim_size: 目标节点数 (必须传入以获得最佳性能)
    """
    # 1. 确定 dim_size
    if dim_size is None:
        if index.numel() == 0:
            d_size = 0
        else:
            # ⚠️ 注意: 这里会有 CPU-GPU 同步，尽量在外部传入 dim_size
            d_size = int(index.max()) + 1
    else:
        d_size = dim_size

    # 2. 构建输出张量
    out_size = list(src.shape)
    out_size[dim] = d_size
    out = mint.zeros(out_size, dtype=src.dtype)
    
    # 3. 极速优化 path (GNN 常用)
    if dim == 0 and index.dim() == 1:
        return out.index_add_(0, index, src)
        
    # 4. 通用路径
    if index.dim() != src.dim():
        view_shape = [1] * src.dim()
        view_shape[dim] = -1
        index_expand = index.view(view_shape).expand_as(src)
        return out.scatter_add_(dim, index_expand, src)
    
    return out.scatter_add_(dim, index, src)


def scatter_mean(src: ms.Tensor, index: ms.Tensor, dim: int = 0, dim_size: Optional[int] = None) -> ms.Tensor:
    """
    自定义 scatter_mean 实现 (基于 scatter_add)
    原理: out = sum(src) / count(index)
    
    Args:
        src: 源数据 [E, F]
        index: 索引 [E]
        dim: 聚合维度 (通常为 0)
        dim_size: 目标节点数
    """
    # 1. 计算分子: Sum
    # 直接调用你现有的 scatter_add
    out = scatter_add(src, index, dim, dim_size)
    
    # 获取实际的输出大小 (避免在 count 步骤再次进行 CPU-GPU 同步)
    d_size = out.shape[dim]
    
    # 2. 计算分母: Count
    # 创建一个与 index 形状相同的全 1 张量
    ones = mint.ones(index.shape, dtype=src.dtype)
    # 聚合全 1 张量得到每个索引的出现次数
    # 注意: index 是 1D 的，所以这里 dim 始终传 0
    count = scatter_add(ones, index, dim=0, dim_size=d_size)
    
    # 3. 数值稳定性处理
    # 将计数为 0 的位置设为 1，防止除以 0 产生 NaN (这些位置分子也是 0，结果应为 0)
    count.clamp_(min=1.0)
    
    # 4. 广播处理 (Broadcasting)
    # 如果 src 是 [N, F] 而 count 是 [N]，直接除会报错
    # 我们需要把 count 变成 [N, 1, ..., 1] 以便广播
    if src.dim() > 1:
        # 构建视图形状: [1, 1, ..., -1, ..., 1]
        view_shape = [1] * src.dim()
        view_shape[dim] = -1 
        count = count.view(view_shape)
        
    # 5. 执行除法
    return out / count
