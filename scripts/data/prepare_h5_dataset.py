"""Convert user-uploaded extxyz data (from ZIP) to H5 + metadata.pkl format
compatible with ChunkedSmartDataset_h5 and the training pipeline.

Usage (standalone):
    python scripts/data/prepare_h5_dataset.py --zip <path> --output_dir <dir> \
        [--dataset_context OMat24] [--cutoff 6.0]
"""

import argparse
import io
import json
import multiprocessing
import os
import pickle
import sys
import time
import zipfile
from pathlib import Path

try:
    from tqdm import tqdm as _tqdm_cls
    def _progress(iterable, total=None, desc="", unit="it"):
        return _tqdm_cls(iterable, total=total, desc=desc, unit=unit, dynamic_ncols=True)
except ImportError:
    def _progress(iterable, total=None, desc="", unit="it"):
        items = list(iterable) if not hasattr(iterable, '__len__') else iterable
        n = total or len(items)
        print(f"{desc}: 0/{n}", flush=True)
        for i, item in enumerate(items, 1):
            if i % max(1, n // 20) == 0 or i == n:
                print(f"{desc}: {i}/{n}", flush=True)
            yield item

# Resolve neighbour-list backend once at import time so _process_one_frame
# threads never pay the try/except overhead per call.
try:
    from matscipy.neighbours import neighbour_list as _matscipy_nl
    def _neighbour_list(atoms, cutoff):
        return _matscipy_nl("ijS", atoms, cutoff)
except ImportError:
    from ase.neighborlist import neighbor_list as _ase_nl
    def _neighbour_list(atoms, cutoff):
        return _ase_nl("ijS", atoms, cutoff, self_interaction=False)


def _ase_available():
    try:
        import ase  # noqa: F401
        import h5py  # noqa: F401
        return True
    except ImportError:
        return False


def _parse_extxyz_bytes(args: tuple):
    """Parse one extxyz file given its (name, raw_bytes). Top-level for pickling."""
    from ase.io import read
    name, raw = args
    try:
        text = raw.decode("utf-8", errors="replace")
        frames = read(io.StringIO(text), index=":", format="extxyz")
        return name, (frames if isinstance(frames, list) else [frames]), None
    except Exception as exc:
        return name, [], str(exc)


def parse_extxyz_zip(zip_path: str, cutoff: float = 6.0, progress_callback=None) -> tuple:
    """Read all extxyz frames from a ZIP file.

    Returns:
        (atoms_list, errors): list of ASE Atoms objects, list of error strings
    """
    from concurrent.futures import ProcessPoolExecutor

    atoms_list = []
    errors = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        candidates = sorted(
            name
            for name in zf.namelist()
            if any(name.lower().endswith(e) for e in (".xyz", ".extxyz"))
            and not name.startswith("__MACOSX")
            and not os.path.basename(name).startswith(".")
        )

        if not candidates:
            errors.append("ZIP contains no .xyz / .extxyz files")
            return atoms_list, errors

        # Read bytes sequentially (ZipFile is not thread-safe for concurrent reads)
        file_bytes = []
        for name in candidates:
            try:
                file_bytes.append((name, zf.read(name)))
            except Exception as exc:
                errors.append(f"{name}: {exc}")

    if not file_bytes:
        return atoms_list, errors

    n_files = len(file_bytes)
    n_parse_workers = min(n_files, os.cpu_count() or 4, 64)
    print(f"[1/3] Parsing {n_files} extxyz file(s) with {n_parse_workers} workers ...", flush=True)
    t0 = time.time()
    # Send 0% immediately so frontend knows parsing has started
    if progress_callback:
        progress_callback("parsing", 0, n_files)
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_parse_workers, mp_context=ctx) as exe:
        for i, (_name, frames, err) in enumerate(_progress(
            exe.map(_parse_extxyz_bytes, file_bytes),
            total=n_files, desc="  parse", unit="file",
        )):
            if err:
                errors.append(f"{_name}: {err}")
            else:
                atoms_list.extend(frames)
            if progress_callback:
                progress_callback("parsing", i + 1, n_files)
    print(f"  -> {len(atoms_list)} frames in {time.time()-t0:.1f}s", flush=True)

    return atoms_list, errors


def validate_atoms(atoms) -> tuple:
    """Check that an Atoms object has energy and forces attached via a calculator.

    Returns (ok: bool, reason: str).
    """
    if atoms.calc is None:
        return False, "no calculator attached (energy/forces not in file)"
    try:
        energy = atoms.get_potential_energy()
        if not isinstance(energy, (int, float)):
            return False, f"energy is not a scalar: {type(energy)}"
        forces = atoms.get_forces()
        if forces.shape != (len(atoms), 3):
            return False, f"forces shape {forces.shape} != ({len(atoms)}, 3)"
        return True, ""
    except Exception as exc:
        return False, str(exc)


def fit_linear_references(valid_atoms: list) -> dict:
    """Fit per-element reference energies via least squares on pre-validated atoms.

    Solves: min ||N @ e0 - E||²
    where N[i, z] = count of element z in frame i, E[i] = DFT total energy.

    Returns:
        dict mapping atomic number (int) -> reference energy (float, eV)
    """
    import numpy as np

    if not valid_atoms:
        return {}

    all_z_nums = sorted({int(z) for a in valid_atoms for z in a.get_atomic_numbers()})
    z_to_col = {z: i for i, z in enumerate(all_z_nums)}
    n = len(valid_atoms)
    m = len(all_z_nums)

    N = np.zeros((n, m), dtype="float64")
    E = np.zeros(n, dtype="float64")
    for i, atoms in enumerate(valid_atoms):
        E[i] = atoms.get_potential_energy()
        for z in atoms.get_atomic_numbers():
            N[i, z_to_col[int(z)]] += 1

    coeffs, _, _, _ = np.linalg.lstsq(N, E, rcond=None)
    return {z: float(coeffs[j]) for z, j in z_to_col.items()}


def _process_one_frame(args: tuple) -> dict:
    """Validate, compute neighbor list, and extract arrays for a single ASE Atoms frame.

    e0_fit may be None on the first pass; energy is then stored raw and z_counts
    is returned so the caller can fit linear references after collecting all results.
    Returns a dict with all per-frame arrays, or {'ok': False, 'error': ...} on failure.
    """
    import numpy as np

    atoms, cutoff, e0_fit = args
    try:
        # Inline validation — avoids a separate sequential pass
        if atoms.calc is None:
            return {"ok": False, "error": "no calculator attached"}
        n_atoms = len(atoms)
        energy_raw = float(atoms.get_potential_energy())
        forces = atoms.get_forces().astype("float32")
        if forces.shape != (n_atoms, 3):
            return {"ok": False, "error": f"forces shape {forces.shape}"}

        z_arr = atoms.get_atomic_numbers().astype("int64")
        # z_counts used for e0 fitting when e0_fit is None
        uniq, cnts = np.unique(z_arr, return_counts=True)
        z_counts = {int(z): int(c) for z, c in zip(uniq, cnts)}

        if e0_fit is not None:
            e0_sum = sum(e0_fit.get(int(z), 0.0) * c for z, c in z_counts.items())
            energy = float(energy_raw - e0_sum)
        else:
            energy = energy_raw

        pos = atoms.get_positions().astype("float32")
        z_arr = atoms.get_atomic_numbers().astype("int64")

        cell_raw = atoms.get_cell()
        cell_arr = np.array(cell_raw, dtype="float32")
        if cell_arr.shape == (3,):
            cell_arr = np.diag(cell_arr)
        if cell_arr.shape != (3, 3):
            cell_arr = np.zeros((3, 3), dtype="float32")

        try:
            i_idx, j_idx, shifts = _neighbour_list(atoms, cutoff)
            i_idx = i_idx.astype("int64")
            j_idx = j_idx.astype("int64")
            shifts = shifts.astype("float32")
        except Exception:
            i_idx = np.array([], dtype="int64")
            j_idx = np.array([], dtype="int64")
            shifts = np.zeros((0, 3), dtype="float32")

        try:
            voigt = atoms.get_stress(voigt=False)
            stress_arr = np.array(voigt, dtype="float32").reshape(3, 3)
        except Exception:
            stress_arr = np.zeros((3, 3), dtype="float32")

        return {
            "ok": True,
            "z": z_arr, "pos": pos, "force": forces,
            "edge_src": i_idx, "edge_dst": j_idx, "shifts_int": shifts,
            "y": energy, "y_raw": energy_raw, "z_counts": z_counts,
            "cell": cell_arr, "stress": stress_arr,
            "n_atoms": n_atoms, "n_edges": len(i_idx),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def write_h5_dataset(
    atoms_list: list,
    output_dir: str,
    dataset_context: str = "OMat24",
    cutoff: float = 6.0,
    h5_name: str = "data.h5",
    dir_name: str = "UserData",
    n_workers: int = 0,
    progress_callback=None,
) -> dict:
    """Write atoms_list to an H5 file + train/test metadata pickles.

    Energies are stored as E_DFT - sum(e0_fit[Z_i]) where e0_fit is obtained
    from a per-element linear regression on the dataset.  This aligns the
    training targets with the pre-trained model's interaction-energy convention.
    The fitted references are saved to ``e0_fit.json`` for later inspection.

    The H5 schema matches ChunkedSmartDataset_h5:
        atom_ptr   int64 (n+1,)        cumulative atom counts
        edge_ptr   int64 (n+1,)        cumulative edge counts
        z          int64 (total_atoms,)
        pos        float32 (total_atoms, 3)
        force      float32 (total_atoms, 3)
        edge_index int64 (2, total_edges)
        shifts_int float32 (total_edges, 3)  fractional periodic shift vectors
        y          float32 (n,)        per-structure interaction energy (eV)
        cell       float32 (n, 3, 3)   unit cell
        spin       float32 (n,)        0 when unknown
        charge     float32 (n,)        0 when unknown
        dataset    vlen str (n,)       dataset context name
        stress     float32 (n, 3, 3)   zeros when not available

    Args:
        atoms_list: list of ASE Atoms objects (with calculator for energy/forces)
        output_dir: directory to write files into
        dataset_context: dataset label stored inside the H5 (e.g. "OMat24")
        cutoff: neighbor list cutoff in Angstrom
        h5_name: filename for the H5 file
        dir_name: prefix used for metadata file names (e.g. "UserData")
        n_workers: parallel threads for neighbor-list computation (0 = auto)

    Returns:
        dict with keys: n_samples, n_skipped, h5_path, meta_train_path,
        meta_test_path, e0_fit_path, e0_fit
    """
    import h5py
    from concurrent.futures import ProcessPoolExecutor

    os.makedirs(output_dir, exist_ok=True)
    h5_path = os.path.join(output_dir, h5_name)
    meta_train_path = os.path.join(output_dir, f"{dir_name}_train_metadata.pkl")
    meta_test_path = os.path.join(output_dir, f"{dir_name}_test_metadata.pkl")
    e0_fit_path = os.path.join(output_dir, "e0_fit.json")

    # Parallel validate + neighbor-list in one pass (e0_fit=None → store raw energy).
    # After collecting results, fit e0 and apply the correction in-memory.
    n_total = len(atoms_list)
    workers = n_workers if n_workers > 0 else min(os.cpu_count() or 4, n_total, 64)
    print(
        f"[2/3] Validating & computing neighbor lists for {n_total} frames "
        f"with {workers} workers ...", flush=True
    )
    t0 = time.time()
    if progress_callback:
        progress_callback("neighbor_list", 0, n_total)
    args_iter = ((atoms, cutoff, None) for atoms in atoms_list)
    ctx = multiprocessing.get_context("spawn")
    raw_results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as exe:
        for i, r in enumerate(_progress(
            exe.map(_process_one_frame, args_iter),
            total=n_total, desc="  process", unit="frame",
        )):
            raw_results.append(r)
            if progress_callback and (i % max(1, n_total // 100) == 0 or i == n_total - 1):
                progress_callback("neighbor_list", i + 1, n_total)
    n_skipped = sum(1 for r in raw_results if not r.get("ok"))
    n_valid = n_total - n_skipped
    print(
        f"  -> {n_valid} valid, {n_skipped} skipped in {time.time()-t0:.1f}s", flush=True
    )

    if n_valid == 0:
        raise ValueError(
            f"No valid samples found after processing {n_total} frames "
            f"({n_skipped} skipped). Each frame must have energy and forces."
        )

    # ── Stage: fitting ───────────────────────────────────────────────────────
    print("[3/3] Fitting linear references ...", flush=True)
    import numpy as np
    valid_results = [r for r in raw_results if r.get("ok")]
    if progress_callback:
        progress_callback("fitting", 0, 3)

    all_z_nums = sorted({z for r in valid_results for z in r["z_counts"]})
    z_to_col = {z: i for i, z in enumerate(all_z_nums)}
    n_v, m = len(valid_results), len(all_z_nums)
    N_mat = np.zeros((n_v, m), dtype="float64")
    E_vec = np.zeros(n_v, dtype="float64")
    for i, r in enumerate(valid_results):
        E_vec[i] = r["y_raw"]
        for z, c in r["z_counts"].items():
            N_mat[i, z_to_col[z]] += c
    if progress_callback:
        progress_callback("fitting", 1, 3)

    if m > 0:
        coeffs, _, _, _ = np.linalg.lstsq(N_mat, E_vec, rcond=None)
        e0_fit = {z: float(coeffs[j]) for z, j in z_to_col.items()}
    else:
        e0_fit = {}
    for r in valid_results:
        e0_sum = sum(e0_fit.get(z, 0.0) * c for z, c in r["z_counts"].items())
        r["y"] = float(r["y_raw"] - e0_sum)
    with open(e0_fit_path, "w") as fh:
        json.dump({str(k): v for k, v in e0_fit.items()}, fh, indent=2)
    if progress_callback:
        progress_callback("fitting", 3, 3)
    print(f"  -> {len(e0_fit)} elements fitted", flush=True)

    # ── Stage: assembling ─────────────────────────────────────────────────────
    print(f"Assembling {n_valid} frames ...", flush=True)
    if progress_callback:
        progress_callback("assembling", 0, n_valid)
    all_z, all_pos, all_force = [], [], []
    all_edge_src, all_edge_dst, all_shifts_int = [], [], []
    all_y, all_cell, all_stress = [], [], []
    atom_ptr, edge_ptr = [0], [0]
    metadata = []

    for r in raw_results:
        if not r.get("ok"):
            continue
        all_z.append(r["z"])
        all_pos.append(r["pos"])
        all_force.append(r["force"])
        all_edge_src.append(r["edge_src"])
        all_edge_dst.append(r["edge_dst"])
        all_shifts_int.append(r["shifts_int"])
        all_y.append(r["y"])
        all_cell.append(r["cell"])
        all_stress.append(r["stress"])
        atom_ptr.append(atom_ptr[-1] + r["n_atoms"])
        edge_ptr.append(edge_ptr[-1] + r["n_edges"])
        cur = len(metadata)
        metadata.append({
            "file_path": h5_name,
            "index_in_file": cur,
            "num_atoms": r["n_atoms"],
            "num_edges": r["n_edges"],
        })
        if progress_callback and (cur % max(1, n_valid // 50) == 0 or cur == n_valid - 1):
            progress_callback("assembling", cur + 1, n_valid)

    n_samples = len(metadata)
    if n_samples == 0:
        raise ValueError("All frames failed during processing.")

    z_all = np.concatenate(all_z)
    pos_all = np.concatenate(all_pos)
    force_all = np.concatenate(all_force)
    src_all = np.concatenate(all_edge_src) if any(len(x) for x in all_edge_src) else np.array([], dtype="int64")
    dst_all = np.concatenate(all_edge_dst) if any(len(x) for x in all_edge_dst) else np.array([], dtype="int64")
    shift_all = np.concatenate(all_shifts_int) if any(len(x) for x in all_shifts_int) else np.zeros((0, 3), dtype="float32")
    edge_index_full = np.stack([src_all, dst_all], axis=0).astype("int64")

    # ── Stage: writing ────────────────────────────────────────────────────────
    H5_DATASETS = [
        ("atom_ptr", lambda: np.array(atom_ptr, dtype="int64")),
        ("edge_ptr", lambda: np.array(edge_ptr, dtype="int64")),
        ("z",        lambda: z_all),
        ("pos",      lambda: pos_all),
        ("force",    lambda: force_all),
        ("edge_index", lambda: edge_index_full),
        ("shifts_int", lambda: shift_all),
        ("y",        lambda: np.array(all_y, dtype="float32")),
        ("cell",     lambda: np.stack(all_cell).astype("float32")),
        ("stress",   lambda: np.stack(all_stress).astype("float32")),
        ("spin",     lambda: np.zeros(n_samples, dtype="float32")),
        ("charge",   lambda: np.zeros(n_samples, dtype="float32")),
    ]
    n_ds = len(H5_DATASETS) + 1  # +1 for dataset strings
    print(f"Writing H5 dataset ({n_samples} samples) -> {h5_path}", flush=True)
    t0 = time.time()
    if progress_callback:
        progress_callback("writing", 0, n_ds)
    with h5py.File(h5_path, "w") as f:
        for i, (name, make_data) in enumerate(H5_DATASETS):
            f.create_dataset(name, data=make_data())
            if progress_callback:
                progress_callback("writing", i + 1, n_ds)
        dt = h5py.string_dtype()
        f.create_dataset("dataset", data=np.array([dataset_context] * n_samples, dtype=object), dtype=dt)
        if progress_callback:
            progress_callback("writing", n_ds, n_ds)

    # Train/test split: last 10% (min 1) held out; train gets the rest.
    n_test = max(1, n_samples // 10)
    meta_train = metadata[:-n_test]
    meta_test  = metadata[-n_test:]

    def _save_meta(meta, pkl_path):
        with open(pkl_path, "wb") as fh:
            pickle.dump(meta, fh)
        # Also write npz so ChunkedSmartDataset_h5 uses the fast path.
        n = len(meta)
        unique_files = [h5_name]
        np.savez(
            pkl_path.replace(".pkl", ".npz"),
            file_ids=np.zeros(n, dtype=np.int32),
            index_in_file=np.array([m["index_in_file"] for m in meta], dtype=np.int32),
            num_atoms=np.array([m["num_atoms"] for m in meta], dtype=np.int32),
            num_edges=np.array([m["num_edges"] for m in meta], dtype=np.int32),
            unique_files=np.array(unique_files),
        )

    _save_meta(meta_train, meta_train_path)
    _save_meta(meta_test,  meta_test_path)
    print(f"  -> H5 written in {time.time()-t0:.1f}s  (train={len(meta_train)}, test={len(meta_test)})", flush=True)

    return {
        "n_samples": n_samples,
        "n_skipped": n_skipped,
        "h5_path": h5_path,
        "meta_train_path": meta_train_path,
        "meta_test_path": meta_test_path,
        "e0_fit_path": e0_fit_path,
        "e0_fit": e0_fit,
    }


def convert_zip(
    zip_path: str,
    output_dir: str,
    dataset_context: str = "OMat24",
    cutoff: float = 6.0,
    dir_name: str = "UserData",
    n_workers: int = 0,
    progress_callback=None,
) -> dict:
    """High-level entry point: parse ZIP and write dataset.

    Returns info dict suitable for JSON serialisation.
    """
    if not _ase_available():
        raise ImportError("ase and h5py are required: pip install ase h5py")

    atoms_list, parse_errors = parse_extxyz_zip(zip_path, cutoff=cutoff, progress_callback=progress_callback)
    if not atoms_list:
        raise ValueError(
            f"No extxyz frames found in ZIP. Errors: {parse_errors}"
        )

    result = write_h5_dataset(
        atoms_list,
        output_dir,
        dataset_context=dataset_context,
        cutoff=cutoff,
        dir_name=dir_name,
        n_workers=n_workers,
        progress_callback=progress_callback,
    )
    result["parse_errors"] = parse_errors
    result["n_frames_read"] = len(atoms_list)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert extxyz ZIP to H5 dataset")
    parser.add_argument("--zip", required=True, help="Path to input ZIP file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--dataset_context", default="OMat24",
                        help="Dataset context label (default: OMat24)")
    parser.add_argument("--cutoff", type=float, default=6.0,
                        help="Neighbor list cutoff in Angstrom")
    parser.add_argument("--dir_name", default="UserData",
                        help="Prefix for metadata file names")
    args = parser.parse_args()

    result = convert_zip(
        zip_path=args.zip,
        output_dir=args.output_dir,
        dataset_context=args.dataset_context,
        cutoff=args.cutoff,
        dir_name=args.dir_name,
    )
    print(json.dumps(result, indent=2))
