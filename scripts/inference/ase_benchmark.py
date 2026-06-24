"""
ase_benchmark.py
================
读取 HTGPModel 权重，使用 ASE Calculator 进行：
  1. 结构优化（BFGS / FIRE）
  2. 分子动力学（Langevin NVT）
  3. 不同原子数的推理速度 Benchmark

用法:
    python scripts/inference/ase_benchmark.py \
        --ckpt /path/to/model.ckpt \
        [--cutoff 6.0] \
        [--device CPU|GPU|Ascend] \
        [--dataset OMat24] \
        [--structure /path/to/your_structure.xyz] \
        [--atom-counts 10 50 100 200 500] \
        [--n-warmup 2] [--n-trials 5] \
        [--opt-atoms 64] [--opt-steps 200] \
        [--md-atoms 64] [--md-steps 200] [--md-temp 300]

权重加载说明:
    训练器以 EMA 权重保存 .ckpt 文件（推理专用）。
    脚本通过 HTGP_Calculator.from_checkpoint() 自动完成模型构建 + 权重加载，
    无需手动 import HTGPModel 或调用 ms.load_checkpoint。
    若 .ckpt 同目录存在同名 .json 文件，则自动读取 HTGPConfig；否则使用默认配置。
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from mindspore import context

# 项目根目录加入 Python 路径
ROOT = Path(__file__).resolve().parents[2]
for extra_path in (ROOT, ROOT / "sharker"):
    extra_path = str(extra_path)
    if extra_path not in sys.path:
        sys.path.insert(0, extra_path)

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import all_changes
from ase.io import write as ase_write, read as ase_read
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS, FIRE
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.langevin import Langevin
import ase.units as units

from src.utils.Calculator import HTGP_Calculator


# ===========================================================
# 显存 / 内存监控
# ===========================================================
def _get_memory_mb() -> dict:
    """
    返回当前进程的显存（GPU/Ascend）或内存（CPU）用量，单位 MB。

    返回字段：
        allocated_MB : 当前已分配的显存
        peak_MB      : 本次运行峰值显存
        rss_MB       : 进程 RSS（所有设备均可用）
    """
    result = {'allocated_MB': 0.0, 'peak_MB': 0.0, 'rss_MB': 0.0}

    # ---- 设备显存（GPU / Ascend）----
    # 优先尝试标量接口（MindSpore 2.x 新 API），再降级到 dict 接口，最后降级到 hal
    def _try_scalar_mem():
        for mod_name in ('mindspore.runtime', 'mindspore.hal'):
            try:
                mod = __import__(mod_name, fromlist=['memory_allocated'])
                alloc = mod.memory_allocated() / 1024 ** 2
                peak  = mod.max_memory_allocated() / 1024 ** 2
                return alloc, peak
            except Exception:
                pass
        return None, None

    def _try_dict_mem():
        for mod_name in ('mindspore.runtime', 'mindspore.hal'):
            try:
                mod = __import__(mod_name, fromlist=['memory_stats'])
                stats = mod.memory_stats()
                alloc = stats.get('allocated_bytes.all.current',
                          stats.get('Allocated Memory(MB)', 0) * 1024 ** 2
                        ) / 1024 ** 2
                peak  = stats.get('allocated_bytes.all.peak',
                          stats.get('Peak Allocated Memory(MB)', 0) * 1024 ** 2
                        ) / 1024 ** 2
                if alloc > 0 or peak > 0:
                    return alloc, peak
            except Exception:
                pass
        return None, None

    alloc, peak = _try_scalar_mem()
    if alloc is None:
        alloc, peak = _try_dict_mem()
    if alloc is not None:
        result['allocated_MB'] = alloc
        result['peak_MB'] = peak if peak is not None else 0.0

    # ---- 进程 RSS（CPU 也有效，作为补充）----
    try:
        import psutil
        result['rss_MB'] = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    except ImportError:
        pass

    return result


def _reset_peak_memory():
    """重置 MindSpore 峰值显存统计（若支持）。"""
    for mod_name in ('mindspore.runtime', 'mindspore.hal'):
        try:
            mod = __import__(mod_name, fromlist=['reset_peak_memory_stats'])
            mod.reset_peak_memory_stats()
            return
        except Exception:
            pass


def _traj_to_extxyz(traj_path: str) -> str:
    """将 ASE .traj 文件转换为 OVITO 可直接打开的 .extxyz 文件，返回输出路径。"""
    out_path = traj_path.replace('.traj', '.extxyz')
    frames = list(Trajectory(traj_path))
    ase_write(out_path, frames, format='extxyz')
    return out_path


def _fmt_mem(mem: dict) -> str:
    """格式化显存信息为单行字符串。"""
    parts = []
    if mem['allocated_MB'] > 0:
        parts.append(f"alloc={mem['allocated_MB']:.0f} MB")
    if mem['peak_MB'] > 0:
        parts.append(f"peak={mem['peak_MB']:.0f} MB")
    if mem['rss_MB'] > 0:
        parts.append(f"RSS={mem['rss_MB']:.0f} MB")
    return "  |  ".join(parts) if parts else "N/A"


# ===========================================================
# 辅助：测试结构生成
# ===========================================================

# 小分子直接从 ASE G2 库读取（≤14 原子，直接使用内置键合结构）
_OMOL_G2_POOL = [
    ("CH4",          5),
    ("CH3OH",        6),
    ("C2H4",         6),
    ("CH3CHO",       7),
    ("C2H6",         8),
    ("CH3COOH",      8),
    ("CH3CH2OH",     9),
    ("CH3OCH3",      9),
    ("CH3COCH3",    10),
    ("C6H6",        12),   # benzene
    ("trans-butane", 14),
    ("isobutane",   14),
]


def _alkane_backbone(n_c: int, rng: np.random.Generator) -> np.ndarray:
    """
    构建烷烃碳链的 3D 坐标（NeRF 法，随机二面角，全键合连通）。

    使用标准 sp3 几何：
      C-C 键长 1.54 Å，C-C-C 键角 109.47°，
      二面角从 trans(180°)/gauche(±60°) 中随机选取（近似旋转异构分布）。

    Args:
        n_c: 碳原子数 (≥2)
        rng: 随机数生成器

    Returns:
        (n_c, 3) float64 数组，单位 Å
    """
    CC = 1.54
    # cos/sin of tetrahedral angle (109.47°)
    COS_TET = -1.0 / 3.0
    SIN_TET = np.sqrt(8.0 / 9.0)   # sqrt(1 - 1/9)

    pos = np.zeros((n_c, 3))
    if n_c == 1:
        return pos

    pos[1] = [CC, 0., 0.]
    if n_c == 2:
        return pos

    # C(2): place in xz-plane (dihedral arbitrary → 0)
    cd_hat = np.array([1., 0., 0.])                 # C(0)→C(1)
    pos[2] = pos[1] + CC * (COS_TET * cd_hat         # forward component
                             + SIN_TET * np.array([0., 0., 1.]))  # perp: z-axis

    # C(3) onward: NeRF placement with random dihedral
    for i in range(3, n_c):
        b, c, d = pos[i-3], pos[i-2], pos[i-1]

        bc_hat = (c - b); bc_hat /= np.linalg.norm(bc_hat)
        cd     = (d - c)
        cd_hat = cd / np.linalg.norm(cd)

        # Normal to b-c-d plane
        n_vec = np.cross(bc_hat, cd_hat)
        n_len = np.linalg.norm(n_vec)
        if n_len < 1e-8:
            ref = np.array([0., 1., 0.]) if abs(cd_hat[0]) < 0.9 else np.array([0., 0., 1.])
            n_vec = np.cross(cd_hat, ref)
            n_vec /= np.linalg.norm(n_vec)
        else:
            n_vec /= n_len

        m_vec = np.cross(n_vec, cd_hat)             # in b-c-d plane, perp to cd

        # 二面角：60% trans(π), 20% gauche+(π/3), 20% gauche-(−π/3)
        phi_base = rng.choice([np.pi, np.pi / 3.0, -np.pi / 3.0],
                              p=[0.6, 0.2, 0.2])
        phi = phi_base + rng.normal(0.0, 0.12)      # ±~7° wiggle

        pos[i] = d + CC * (
            -COS_TET * cd_hat                        # = +1/3 * cd_hat (forward)
            + SIN_TET * np.cos(phi) * m_vec
            + SIN_TET * np.sin(phi) * n_vec
        )

    return pos


def _add_hydrogens(c_pos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    给碳链每个碳添加氢原子（sp3 四面体几何）。

    内部碳 → 2H，末端碳 → 3H。H-C 键长 1.09 Å。

    Returns:
        (n_h, 3) float64 数组，氢原子位置
    """
    CH = 1.09
    COS_TET = -1.0 / 3.0
    SIN_TET = np.sqrt(8.0 / 9.0)
    n_c = len(c_pos)
    h_pos = []

    for i in range(n_c):
        ci = c_pos[i]

        # 归一化键向量（指向邻近碳）
        nbr = []
        if i > 0:
            v = c_pos[i-1] - ci; nbr.append(v / np.linalg.norm(v))
        if i < n_c - 1:
            v = c_pos[i+1] - ci; nbr.append(v / np.linalg.norm(v))

        if len(nbr) == 2:
            # CH₂：H 在两个 C-C 键的对称平面两侧
            s_half = (nbr[0] + nbr[1]) * 0.5          # 指向"内侧"
            perp = np.cross(nbr[0], nbr[1])
            p_len = np.linalg.norm(perp)
            if p_len < 1e-8:
                # 两键共线（直链边缘情况）：选任意垂直方向
                ref = np.array([0., 1., 0.]) if abs(nbr[0][0]) < 0.9 else np.array([0., 0., 1.])
                perp = np.cross(nbr[0], ref)
                perp /= np.linalg.norm(perp)
            else:
                perp /= p_len

            t = np.sqrt(max(0., 1. - float(np.dot(s_half, s_half))))
            h_pos.append(ci + CH * (-s_half + t * perp))
            h_pos.append(ci + CH * (-s_half - t * perp))

        else:
            # CH₃（末端）：3 个 H 均匀分布在 C-C 轴周围
            axis = nbr[0]                              # 朝向唯一邻碳
            ref = np.array([0., 1., 0.]) if abs(axis[1]) < 0.9 else np.array([1., 0., 0.])
            p1 = np.cross(axis, ref); p1 /= np.linalg.norm(p1)
            p2 = np.cross(axis, p1)
            # 随机初始旋转角，避免两端 CH₃ 完全重叠
            stagger = rng.uniform(0., 2. * np.pi)
            for k in range(3):
                phi = 2. * np.pi * k / 3. + stagger
                h_unit = (COS_TET * (-axis)
                          + SIN_TET * (np.cos(phi) * p1 + np.sin(phi) * p2))
                h_pos.append(ci + CH * h_unit)

    return np.array(h_pos, dtype=float)


def _build_omol_molecule(n_atoms: int,
                         rng: np.random.Generator = None) -> Atoms:
    """
    生成一个单连通有机分子（所有原子键合在同一条链上），原子数约等于 n_atoms。

    策略：
    - n_atoms ≤ 14：直接从 ASE G2 库取最接近的分子（已内置键合）。
    - n_atoms > 14 ：构建 C_n H_{2n+2} 线性烷烃链
        n_c = round((n_atoms − 2) / 3)，实际原子数 = 3*n_c + 2。
        使用随机二面角（trans/gauche 混合）生成紧凑 3D 构象。

    Returns:
        ASE Atoms 对象（非周期，已 center 添加 8 Å vacuum）
    """
    from ase.build import molecule as ase_molecule

    if rng is None:
        rng = np.random.default_rng(42)

    # --- 小分子：直接使用 ASE G2 内置键合结构 ---
    if n_atoms <= 14:
        best_name, best_size = min(_OMOL_G2_POOL, key=lambda x: abs(x[1] - n_atoms))
        mol = ase_molecule(best_name)
        mol.center(vacuum=8.0)
        print(f"  [OMol] target={n_atoms} → G2 '{best_name}' ({best_size} atoms)")
        return mol

    # --- 大分子：线性烷烃 C_n H_{2n+2}（全键合单分子） ---
    # 3*n_c + 2 = n_atoms  →  n_c ≈ (n_atoms−2)/3
    n_c = max(2, round((n_atoms - 2) / 3.0))

    c_pos = _alkane_backbone(n_c, rng)
    h_pos = _add_hydrogens(c_pos, rng)

    n_h = len(h_pos)
    actual = n_c + n_h
    print(f"  [OMol] target={n_atoms} → C{n_c}H{n_h} = {actual} atoms "
          f"(connected alkane, random-dihedral conformation)")

    symbols = ['C'] * n_c + ['H'] * n_h
    positions = np.vstack([c_pos, h_pos])
    mol = Atoms(symbols, positions=positions)
    mol.center(vacuum=8.0)
    return mol


def make_atoms(n_atoms: int, dataset: str = "OMat24",
               cutoff: float = 6.0, target_neighbors: int = 50) -> Atoms:
    """
    生成用于测试的 ASE Atoms 对象。

    - OMol 系列 → 从 ASE 内置分子库贪心拼装真实有机分子，非周期
    - 其余      → FCC Cu 超胞，晶格常数自动调整使每原子邻居数 ≈ target_neighbors
    """
    if "OMol" in dataset:
        return _build_omol_molecule(n_atoms)
    else:
        # 自动搜索 FCC 晶格常数，使每原子邻居数 ≈ target_neighbors
        a_fcc, actual_neigh = _find_best_param(
            _count_fcc_neighbors,
            lo=cutoff * 0.35, hi=cutoff * 0.85,
            cutoff=cutoff, target=target_neighbors,
        )
        print(f"  [FCC] a={a_fcc:.3f} Å → {actual_neigh} neighbors/atom "
              f"(target={target_neighbors}, cutoff={cutoff:.1f} Å)")

        # cubic=True：返回 4 原子的立方惯用胞，而非 1 原子原胞
        primitive = bulk('Cu', 'fcc', a=a_fcc, cubic=True)

        # 四舍五入到最近的 4 的倍数，保证能整除原胞
        n_rounded = max(4, round(n_atoms / 4) * 4)
        n_cells = n_rounded // 4

        nx, ny, nz = _factorize_cells_exact(n_cells)
        atoms = primitive.repeat([nx, ny, nz])

        actual = len(atoms)
        if actual != n_atoms:
            print(f"  [NOTE] Requested {n_atoms} → {actual} atoms "
                  f"(FCC {nx}×{ny}×{nz} supercell, nearest multiple of 4).")

    return atoms


def _factorize_cells_exact(n: int):
    """
    将整数 n 精确分解为三个因子 (nx, ny, nz)，满足 nx*ny*nz == n，
    且超胞形状尽量接近立方体（最小化 max/min 边长比）。

    只枚举 n 的真因子（不做 ceiling），确保原子数精确等于 n*4，
    晶胞密度与完整 FCC Cu 一致，适用于几何优化和 MD。
    """
    # 枚举 n 的所有因子
    divs = []
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    divs.sort()

    best = (1, 1, n)
    best_ratio = float(n)

    for nx in divs:
        rem = n // nx
        # 枚举 rem 的因子作为 ny
        for ny in divs:
            if rem % ny != 0:
                continue
            nz = rem // ny
            if nx * ny * nz != n:
                continue
            ratio = max(nx, ny, nz) / min(nx, ny, nz)
            if ratio < best_ratio:
                best_ratio = ratio
                best = (nx, ny, nz)

    return best


# ------------------------------------------------------------------
# 邻居数辅助：给定截断半径，反推让每原子有 ~target 个邻居的结构参数
# ------------------------------------------------------------------
def _count_sc_neighbors(d: float, cutoff: float) -> int:
    """简单立方网格（间距 d）在截断半径 cutoff 内的邻居数（体相原子）。"""
    max_n = int(cutoff / d) + 1
    r = np.arange(-max_n, max_n + 1)
    ii, jj, kk = np.meshgrid(r, r, r, indexing='ij')
    r2 = (ii ** 2 + jj ** 2 + kk ** 2).ravel().astype(float) * d * d
    return int(np.sum((r2 > 0) & (r2 <= cutoff * cutoff)))


def _count_fcc_neighbors(a: float, cutoff: float) -> int:
    """FCC 晶格（晶格常数 a）在截断半径 cutoff 内的邻居数。"""
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0],
                      [0.5, 0, 0.5], [0, 0.5, 0.5]]) * a
    max_n = int(cutoff / a) + 2
    r = np.arange(-max_n, max_n + 1)
    ii, jj, kk = np.meshgrid(r, r, r, indexing='ij')
    offsets = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1).astype(float) * a
    r_c2 = cutoff * cutoff
    count = 0
    for b in basis:
        r2 = np.sum((offsets + b) ** 2, axis=1)
        count += int(np.sum((r2 > 1e-10) & (r2 <= r_c2)))
    return count


def _find_best_param(count_fn, lo: float, hi: float,
                     cutoff: float, target: int, n: int = 400):
    """在 [lo, hi] 内均匀搜索，返回使邻居数最接近 target 的参数值和实际邻居数。"""
    vals = np.linspace(lo, hi, n)
    counts = np.array([count_fn(v, cutoff) for v in vals])
    idx = int(np.argmin(np.abs(counts - target)))
    return float(vals[idx]), int(counts[idx])


def generate_structures(atom_counts: list, dataset: str, cutoff: float,
                        target_neighbors: int = 50,
                        save_dir: str = "benchmark_structures") -> dict:
    """
    为所有目标原子数生成测试结构，保存为 .xyz 文件，返回 {actual_n: atoms} 字典。

    save_dir 目录不存在时自动创建。后续 inference / geo_opt / MD 均使用
    此字典，保证三者在完全相同的结构上测试。
    """
    os.makedirs(save_dir, exist_ok=True)
    structures = {}

    print(f"\n[Structures] Generating {len(atom_counts)} structure(s) → {save_dir}/")
    for n in atom_counts:
        atoms = make_atoms(n, dataset, cutoff=cutoff, target_neighbors=target_neighbors)
        actual_n = len(atoms)
        path = os.path.join(save_dir, f"struct_{actual_n}atoms.xyz")
        ase_write(path, atoms)
        structures[actual_n] = atoms
        print(f"  Saved: {path}  ({actual_n} atoms)")

    return structures


def _sanitize_structure_name(name: str) -> str:
    """将文件名转换为适合输出文件的安全标签。"""
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    sanitized = "".join(keep).strip("_")
    return sanitized or "custom_structure"


def load_custom_structure(structure_path: str,
                          structure_index: int = 0,
                          save_dir: str = "benchmark_structures") -> dict:
    """
    读取用户提供的单个结构文件，返回 {n_atoms: atoms} 字典。

    支持任意 ASE 可读格式，例如 .xyz / .extxyz / .cif / .traj。
    为了便于后续检查，同时会在 save_dir 下导出一份 .extxyz 副本。
    """
    path = Path(structure_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Custom structure not found: {path}")

    atoms = ase_read(str(path), index=structure_index)
    if isinstance(atoms, list):
        raise ValueError(
            f"{path} with index={structure_index!r} returned multiple frames. "
            "Please provide a single frame index."
        )
    if not isinstance(atoms, Atoms):
        raise TypeError(f"Unsupported ASE read result type: {type(atoms)}")

    os.makedirs(save_dir, exist_ok=True)
    actual_n = len(atoms)
    safe_stem = _sanitize_structure_name(path.stem)
    export_path = os.path.join(
        save_dir,
        f"{safe_stem}_frame{structure_index}_{actual_n}atoms.extxyz",
    )
    ase_write(export_path, atoms, format="extxyz")

    print(f"\n[Structures] Loaded custom structure → {path}")
    print(f"  Frame index : {structure_index}")
    print(f"  Atom count  : {actual_n}")
    print(f"  Export copy : {export_path}")

    return {actual_n: atoms}


def _make_calc(base_calc: HTGP_Calculator, dataset: str) -> HTGP_Calculator:
    """
    基于 base_calc 中已加载的模型，创建一个新的 Calculator 实例，
    并设置 Calculator 级别的推理上下文。
    """
    calc = HTGP_Calculator(
        model=base_calc.model,
        cutoff=base_calc.cutoff,
        capture_weights=base_calc.capture_weights,
        capture_descriptors=base_calc.capture_descriptors,
        add_e0_baseline=base_calc.add_e0_baseline,
        e0_dir=base_calc.e0_dir,
        dataset_name=dataset,
        default_charge=0.0,
        default_spin=0.0,
    )
    calc.e0_fit = getattr(base_calc, "e0_fit", None)
    return calc


# ===========================================================
# 0. 单点推理
# ===========================================================
def run_single_point_inference(
    base_calc: HTGP_Calculator,
    structures: dict,
    dataset: str,
    out_dir: str = ".",
) -> dict:
    """
    对结构做一次能量+力推理，并导出带预测信息的 .extxyz 文件。

    返回:
        {n_atoms: {'energy_eV', 'fmax_eV_A', 'output_path'}}
    """
    W = 60
    print("\n" + "=" * W)
    print("  Single-Point Inference")
    print("=" * W)

    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for actual_n, atoms_orig in sorted(structures.items()):
        atoms = atoms_orig.copy()
        calc = _make_calc(base_calc, dataset)
        atoms.calc = calc

        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces())
        fmax = float(np.max(np.linalg.norm(forces, axis=1)))

        atoms.info["pred_energy_eV"] = energy
        atoms.arrays["pred_forces_eV_A"] = forces

        out_path = os.path.join(out_dir, f"single_point_{actual_n}atoms.extxyz")
        ase_write(out_path, atoms, format="extxyz")

        print(f"  {actual_n:>8d} atoms  |  E = {energy:>12.6f} eV  |  Fmax = {fmax:>10.6f} eV/Å")
        print(f"  Saved: {out_path}")

        results[actual_n] = {
            "energy_eV": energy,
            "fmax_eV_A": fmax,
            "output_path": out_path,
        }

    print("=" * W)
    return results


# ===========================================================
# 1. 推理速度 Benchmark
# ===========================================================
def run_inference_benchmark(
    base_calc: HTGP_Calculator,
    structures: dict,
    dataset: str,
    n_warmup: int = 2,
    n_trials: int = 5,
    profile_dir: str = None,
) -> dict:
    """
    对 structures 字典中每个结构测量单步（能量 + 力）推理耗时。

    Args:
        structures:   {n_atoms: Atoms}，由 generate_structures() 生成
        profile_dir:  非 None 时对最大原子数结构启用 MindSpore Profiler，
                      结果写入该目录，benchmark 结束后自动调用 analyse()。
    Returns:
        {n_atoms: {'mean_s', 'std_s', 'steps_per_s', 'atoms_per_s', ...}}
    """
    W = 80
    print("\n" + "=" * W)
    print("  Inference Speed Benchmark")
    print("=" * W)
    print(f"  Dataset : {dataset}  |  Warmup: {n_warmup}  |  Trials: {n_trials}")
    if profile_dir:
        print(f"  Profiling : enabled → {profile_dir}  (largest structure only)")
    hdr = (f"  {'N_atoms':>8}  {'mean(s)':>9}  {'std(s)':>7}  "
           f"{'steps/s':>9}  {'atoms/s':>10}  {'alloc(MB)':>10}  {'peak(MB)':>9}  {'RSS(MB)':>8}")
    print(hdr)
    print("-" * W)

    # Profiler 只对最大原子数的结构启用（避免多次初始化冲突）
    profile_target_n = max(structures.keys()) if profile_dir else None
    profiler = None

    results = {}
    for actual_n, atoms in sorted(structures.items()):
        calc = _make_calc(base_calc, dataset)
        atoms.calc = calc

        for _ in range(n_warmup):
            calc.results = {}
            calc.calculate(atoms, ['energy', 'forces'], all_changes)

        calc.results = {}
        calc.calculate(atoms, ['energy', 'forces'], all_changes)
        energy = float(np.asarray(calc.results['energy']).reshape(-1)[0])
        forces = np.asarray(calc.results['forces'])
        fmax = float(np.max(np.linalg.norm(forces, axis=1)))

        _reset_peak_memory()

        # ---- 按需启动 Profiler ----
        if profile_dir and actual_n == profile_target_n:
            from mindspore import Profiler as _Profiler
            from mindspore.profiler import ProfilerLevel, AicoreMetrics
            profiler = _Profiler(
                output_path=profile_dir,
                profile_memory=True,
                profiler_level=ProfilerLevel.Level2,
                aic_metrics = AicoreMetrics.ArithmeticUtilization,
                start_profile=True,
            )
            print(f"  [Profiler] started for {actual_n}-atom structure")

        timings = []
        for _ in range(n_trials):
            calc.results = {}
            t0 = time.perf_counter()
            calc.calculate(atoms, ['energy', 'forces'], all_changes)
            timings.append(time.perf_counter() - t0)

        # ---- Profiler analyse（生成可读报告）----
        if profiler is not None:
            print(f"\n[Profiler] Analysing... output → {profile_dir}")
            profiler.analyse()
            print(f"[Profiler] Done. Open {profile_dir} with MindSpore Insight.")

        mem = _get_memory_mb()
        mean_t = float(np.mean(timings))
        std_t = float(np.std(timings))
        sps = 1.0 / mean_t
        aps = actual_n / mean_t

        results[actual_n] = {
            'energy_eV': energy,
            'fmax_eV_A': fmax,
            'mean_s': mean_t, 'std_s': std_t,
            'steps_per_s': sps, 'atoms_per_s': aps,
            'mem_allocated_MB': mem['allocated_MB'],
            'mem_peak_MB': mem['peak_MB'],
            'mem_rss_MB': mem['rss_MB'],
        }
        print(f"  {actual_n:>8d}  {mean_t:>9.4f}  {std_t:>7.4f}  "
              f"{sps:>9.2f}  {aps:>10.1f}  "
              f"{mem['allocated_MB']:>10.1f}  {mem['peak_MB']:>9.1f}  {mem['rss_MB']:>8.1f}")
        print(f"            E = {energy:.6f} eV  |  Fmax = {fmax:.6f} eV/Å")

    print("=" * W)

    return results


# ===========================================================
# 2. 结构优化
# ===========================================================
def run_geometry_optimization(
    base_calc: HTGP_Calculator,
    structures: dict,
    dataset: str,
    optimizer: str = "BFGS",
    fmax: float = 0.05,
    max_steps: int = 200,
    out_dir: str = ".",
) -> tuple:
    """
    对 structures 中每个结构运行几何优化。

    Returns:
        (results, relaxed_structures)
            results:           {n_atoms: {...}} 统计结果
            relaxed_structures: {n_atoms: Atoms} 优化后的结构（供 MD 直接使用）
    """
    W = 60
    print("\n" + "=" * W)
    print(f"  Geometry Optimization ({optimizer})")
    print("=" * W)

    results = {}
    relaxed_structures = {}
    for actual_n, atoms_orig in sorted(structures.items()):
        print(f"\n  --- {actual_n} atoms ---")
        atoms = atoms_orig.copy()
        print(f"  fmax: {fmax} eV/Å  |  max_steps: {max_steps}")

        calc = _make_calc(base_calc, dataset)
        atoms.calc = calc

        rng = np.random.default_rng(0)
        atoms.positions += rng.normal(0, 0.05, atoms.positions.shape)

        traj_file = os.path.join(out_dir, f"relax_{actual_n}atoms.traj")
        log_file  = os.path.join(out_dir, f"relax_{actual_n}atoms.log")
        opt_cls = FIRE if optimizer == "FIRE" else BFGS
        opt = opt_cls(atoms, trajectory=traj_file, logfile=log_file)

        e0 = atoms.get_potential_energy()
        f0 = float(np.max(np.linalg.norm(atoms.get_forces(), axis=1)))
        print(f"  Initial: E = {e0:.4f} eV  |  Fmax = {f0:.4f} eV/Å")

        _reset_peak_memory()
        t0 = time.perf_counter()
        converged = opt.run(fmax=fmax, steps=max_steps)
        elapsed = time.perf_counter() - t0
        mem = _get_memory_mb()

        e1 = atoms.get_potential_energy()
        f1 = float(np.max(np.linalg.norm(atoms.get_forces(), axis=1)))
        nsteps = opt.nsteps
        sps = nsteps / elapsed if elapsed > 0 else 0.0

        print(f"  Final  : E = {e1:.4f} eV  |  Fmax = {f1:.4f} eV/Å")
        print(f"  Steps: {nsteps}  |  Converged: {converged}  |  "
              f"Time: {elapsed:.2f}s  |  {sps:.2f} steps/s  "
              f"({elapsed/max(nsteps,1)*1000:.1f} ms/step)")
        print(f"  Memory: {_fmt_mem(mem)}")

        xyz_file = _traj_to_extxyz(traj_file)
        print(f"  Saved: {xyz_file}  (OVITO)")

        results[actual_n] = {
            'n_atoms': actual_n, 'converged': converged,
            'n_steps': nsteps, 'steps_per_s': sps,
            'energy_initial_eV': e0, 'energy_final_eV': e1,
            'fmax_final_eV_A': f1, 'elapsed_s': elapsed,
            'mem_allocated_MB': mem['allocated_MB'],
            'mem_peak_MB': mem['peak_MB'],
            'mem_rss_MB': mem['rss_MB'],
        }
        # 保留优化后的结构（去掉 calculator 引用，避免残留状态干扰 MD）
        atoms.calc = None
        relaxed_structures[actual_n] = atoms

    print("=" * W)
    return results, relaxed_structures


# ===========================================================
# 3. 分子动力学
# ===========================================================
def run_molecular_dynamics(
    base_calc: HTGP_Calculator,
    structures: dict,
    dataset: str,
    n_steps: int = 200,
    temperature_K: float = 300.0,
    dt_fs: float = 1.0,
    friction: float = 0.01,
    out_dir: str = ".",
) -> dict:
    """对 structures 中每个结构运行 Langevin NVT MD，返回 {n_atoms: result}。"""
    W = 60
    print("\n" + "=" * W)
    print("  Molecular Dynamics (Langevin NVT)")
    print("=" * W)

    results = {}
    for actual_n, atoms_orig in sorted(structures.items()):
        print(f"\n  --- {actual_n} atoms ---")
        atoms = atoms_orig.copy()
        print(f"  T: {temperature_K} K  |  dt: {dt_fs} fs  |  steps: {n_steps}")

        calc = _make_calc(base_calc, dataset)
        atoms.calc = calc

        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K,
                                     rng=np.random.default_rng(42))
        Stationary(atoms)

        traj_file = os.path.join(out_dir, f"md_{actual_n}atoms.traj")
        log_file  = os.path.join(out_dir, f"md_{actual_n}atoms.log")
        dyn = Langevin(
            atoms,
            timestep=dt_fs * units.fs,
            temperature_K=temperature_K,
            friction=friction / units.fs,
            trajectory=traj_file,
            logfile=log_file,
        )

        energies, temps, step_times = [], [], []
        dyn.attach(lambda: (energies.append(atoms.get_potential_energy()),
                            temps.append(atoms.get_temperature())), interval=10)
        dyn.attach(lambda: step_times.append(time.perf_counter()), interval=1)

        e0 = atoms.get_potential_energy()
        print(f"  Initial energy: {e0:.4f} eV")

        _reset_peak_memory()
        t0 = time.perf_counter()
        dyn.run(n_steps)
        elapsed = time.perf_counter() - t0
        mem = _get_memory_mb()

        steps_per_s = n_steps / elapsed
        if len(step_times) >= 2:
            per_step_ms = np.diff(step_times) * 1000
            step_mean_ms = float(np.mean(per_step_ms))
            step_std_ms  = float(np.std(per_step_ms))
        else:
            step_mean_ms = elapsed / n_steps * 1000
            step_std_ms  = 0.0

        print(f"  Done: {elapsed:.2f}s  |  {steps_per_s:.2f} steps/s  "
              f"|  {step_mean_ms:.1f} ± {step_std_ms:.1f} ms/step")
        print(f"  Mean E: {np.mean(energies):.4f} eV  |  Mean T: {np.mean(temps):.1f} K")
        print(f"  Memory: {_fmt_mem(mem)}")

        xyz_file = _traj_to_extxyz(traj_file)
        print(f"  Saved: {xyz_file}  (OVITO)")

        results[actual_n] = {
            'n_atoms': actual_n, 'n_steps': n_steps,
            'elapsed_s': elapsed, 'steps_per_s': steps_per_s,
            'step_mean_ms': step_mean_ms, 'step_std_ms': step_std_ms,
            'mean_energy_eV': float(np.mean(energies)),
            'mean_temperature_K': float(np.mean(temps)),
            'mem_allocated_MB': mem['allocated_MB'],
            'mem_peak_MB': mem['peak_MB'],
            'mem_rss_MB': mem['rss_MB'],
            'energy_trajectory': [float(e) for e in energies],
            'temperature_trajectory': [float(t) for t in temps],
        }

    print("=" * W)
    return results


# ===========================================================
# CLI
# ===========================================================
def parse_args():
    p = argparse.ArgumentParser(description="HTGPModel ASE Benchmark")
    p.add_argument("--ckpt", required=True,
                   help="MindSpore checkpoint (.ckpt) 路径")
    p.add_argument("--cutoff", type=float, default=None,
                   help="截断半径（Å）。默认读取 config.cutoff")
    p.add_argument("--device", default="CPU", choices=["CPU", "GPU", "Ascend"])
    p.add_argument("--dataset", default="OMat24",
                   help="数据集名称，决定体系类型和 force_scale")
    p.add_argument("--structure", default=None,
                   help=("用户自定义结构文件路径。提供后将跳过自动生成结构，"
                         "直接读取该文件做推理/优化/MD；支持 ASE 可读格式，如 "
                         ".xyz/.extxyz/.cif/.traj"))
    p.add_argument("--structure-index", type=int, default=0,
                   help="当 --structure 是多帧文件（如 .traj）时，读取的帧编号，默认 0")
    p.add_argument("--atom-counts", type=int, nargs="+",
                   default=[10, 50, 100, 200, 500],
                   help="目标原子数列表；推理 / 结构优化 / MD 均使用同一批结构")
    p.add_argument("--target-neighbors", type=int, default=None,
                   help=("每原子目标邻居数（通过调整晶格/网格间距实现）。"
                         "默认 None：OMol 系列取 30，其余（OMat24 等无机晶体）取 50。"
                         "如需精确模拟训练分布，参考数据集实际密度手动指定。"))
    p.add_argument("--struct-dir", default="benchmark_structures",
                   help="生成结构的保存目录")
    p.add_argument("--n-warmup", type=int, default=2)
    p.add_argument("--n-trials", type=int, default=5)
    p.add_argument("--opt-steps", type=int, default=200)
    p.add_argument("--opt-fmax", type=float, default=0.05)
    p.add_argument("--optimizer", default="BFGS", choices=["BFGS", "FIRE"])
    p.add_argument("--md-steps", type=int, default=200)
    p.add_argument("--md-temp", type=float, default=300.0)
    p.add_argument("--md-dt", type=float, default=1.0)
    p.add_argument("--output", default="benchmark_results.json")
    p.add_argument("--skip-benchmark", action="store_true")
    p.add_argument("--skip-opt", action="store_true")
    p.add_argument("--skip-md", action="store_true")
    p.add_argument("--profile", action="store_true",
                   help="启用 MindSpore Profiler（仅对推理 benchmark 最大规模结构生效）")
    p.add_argument("--profile-dir", default="profiling_output",
                   help="Profiler 输出目录（默认 profiling_output/）")
    return p.parse_args()


def main():
    args = parse_args()

    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device)

    base_calc = HTGP_Calculator.from_checkpoint(
        ckpt_path=args.ckpt,
        cutoff=args.cutoff,
    )

    # ---- 统一准备结构（推理 / 优化 / MD 共用）----
    if args.structure:
        structures = load_custom_structure(
            structure_path=args.structure,
            structure_index=args.structure_index,
            save_dir=args.struct_dir,
        )
    else:
        # 默认邻居数：有机分子（OMol）~30，无机晶体（OMat24 等）~50
        if args.target_neighbors is not None:
            target_neighbors = args.target_neighbors
        elif "OMol" in args.dataset:
            target_neighbors = 30
        else:
            target_neighbors = 50

        structures = generate_structures(
            atom_counts=args.atom_counts,
            dataset=args.dataset,
            cutoff=base_calc.cutoff,
            target_neighbors=target_neighbors,
            save_dir=args.struct_dir,
        )

    all_results = {}

    if args.structure:
        all_results['single_point_inference'] = run_single_point_inference(
            base_calc=base_calc,
            structures=structures,
            dataset=args.dataset,
            out_dir=args.struct_dir,
        )

    # ---- 推理速度 Benchmark ----
    if not args.skip_benchmark:
        all_results['inference_benchmark'] = run_inference_benchmark(
            base_calc=base_calc,
            structures=structures,
            dataset=args.dataset,
            n_warmup=args.n_warmup,
            n_trials=args.n_trials,
            profile_dir=args.profile_dir if args.profile else None,
        )

    # ---- 结构优化 → 产出 relaxed_structures 供 MD 使用 ----
    relaxed_structures = None
    if not args.skip_opt:
        opt_results, relaxed_structures = run_geometry_optimization(
            base_calc=base_calc,
            structures=structures,
            dataset=args.dataset,
            optimizer=args.optimizer,
            fmax=args.opt_fmax,
            max_steps=args.opt_steps,
            out_dir=args.struct_dir,
        )
        all_results['geometry_optimization'] = opt_results

    # ---- 分子动力学：优先使用优化后结构，否则用原始结构 ----
    md_structures = relaxed_structures if relaxed_structures is not None else structures
    if not args.skip_md:
        if relaxed_structures is not None:
            print("\n[MD] Using relaxed structures from geometry optimization.")
        else:
            print("\n[MD] No relaxed structures available; using raw generated structures.")
        all_results['molecular_dynamics'] = run_molecular_dynamics(
            base_calc=base_calc,
            structures=md_structures,
            dataset=args.dataset,
            n_steps=args.md_steps,
            temperature_K=args.md_temp,
            dt_fs=args.md_dt,
            out_dir=args.struct_dir,
        )

    # ---- 保存结果 ----
    def _serial(obj):
        if isinstance(obj, (float, int, str, bool)):
            return obj
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        raise TypeError(f"Not JSON-serializable: {type(obj)}")

    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2, default=_serial)
    print(f"\n[INFO] Results saved → {args.output}")


if __name__ == "__main__":
    main()
