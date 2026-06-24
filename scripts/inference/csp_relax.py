#!/usr/bin/env python3
"""Run CSP relaxations with the official AtomBit ASE calculator.

This runner deliberately uses `HTGP_Calculator.from_checkpoint()` from the
safe inference tree. The model implementation in that tree must already use
SafeEquivariantRMSNorm for L1/L2 channels.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from ase.filters import FrechetCellFilter
from ase.io import read
from ase.neighborlist import neighbor_list
from ase.optimize import BFGS
from joblib import Parallel, delayed


EV_TO_KJ_MOL = 96.48533212331002
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def setup_mindspore(device_target: str, mode: str):
    import mindspore as ms

    ms_mode = ms.PYNATIVE_MODE if mode.lower() == "pynative" else ms.GRAPH_MODE
    ms.set_context(mode=ms_mode, device_target=device_target)


def load_e0_fit(path: str | None):
    if not path:
        return None
    e0_path = Path(path).resolve()
    with open(e0_path) as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"e0 fit file must contain a dictionary: {e0_path}")
    return {int(z): float(e0) for z, e0 in payload.items()}


def build_official_calc(args):
    inference_root = Path(args.inference_root).resolve()
    sys.path.insert(0, str(inference_root))

    from src.utils.Calculator import HTGP_Calculator

    packaged_e0 = inference_root / "e0"
    e0_dir = args.e0_dir if args.e0_dir else str(packaged_e0 if packaged_e0.exists() else inference_root)
    e0_fit = load_e0_fit(args.e0_fit_json)
    calc = HTGP_Calculator.from_checkpoint(
        ckpt_path=str(Path(args.checkpoint).resolve()),
        cutoff=args.cutoff,
        e0_dir=e0_dir,
        add_e0_baseline=not args.no_e0,
        dataset_name=args.dataset_name,
        default_charge=0.0,
        default_spin=0.0,
    )
    if e0_fit:
        calc.e0_fit = e0_fit
        print(f"Using fitted e0 from {Path(args.e0_fit_json).resolve()}", flush=True)
    return calc


def calculate_density(atoms) -> float:
    total_mass = sum(atoms.get_masses())
    volume = atoms.get_volume()
    return total_mass / (volume * 1e-24) / (6.022140857e23)


def geom_stats(atoms, cutoff: float) -> dict:
    i, _, d = neighbor_list("ijd", atoms, cutoff)
    return {
        "volume_A3": float(atoms.get_volume()),
        "density": calculate_density(atoms),
        "neighbor_edges": int(len(i)),
        "min_distance_A": float(np.min(d)) if len(d) else None,
    }


def molecule_count_from_atoms(atoms, molecule_single: int) -> float:
    if molecule_single <= 0:
        raise ValueError("molecule_single must be positive")
    return len(atoms.get_atomic_numbers()) / molecule_single


def reset_calculator_cache(atoms) -> None:
    """Force ASE to recompute properties for the current atom/cell state."""
    calc = getattr(atoms, "calc", None)
    if calc is None:
        return
    if hasattr(calc, "results"):
        calc.results.clear()
    if hasattr(calc, "atoms"):
        calc.atoms = None


def residual_metrics(atoms, cutoff: float) -> dict:
    """Compute final residuals from a fresh energy/force/stress evaluation."""
    reset_calculator_cache(atoms)
    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    stress = atoms.get_stress() if atoms.pbc.any() else np.zeros(6)
    filter_forces = FrechetCellFilter(atoms).get_forces() if atoms.pbc.any() else forces

    atom_force_norm = np.linalg.norm(forces, axis=1)
    filter_force_norm = np.linalg.norm(filter_forces, axis=1)
    metrics = {
        "energy_eV_total": energy,
        "max_atom_force_eVA": float(atom_force_norm.max()) if len(atom_force_norm) else 0.0,
        "rms_atom_force_eVA": float(np.sqrt((forces * forces).sum(axis=1).mean())) if len(forces) else 0.0,
        "stress_voigt_eVA3": [float(x) for x in np.asarray(stress).reshape(-1).tolist()],
        "stress_norm_eVA3": float(np.linalg.norm(stress)),
        "filter_fmax": float(filter_force_norm.max()) if len(filter_force_norm) else 0.0,
    }
    metrics.update({f"final_{k}": v for k, v in geom_stats(atoms, cutoff).items()})
    return metrics


def run_optimizer(filter_obj, fmax: float, steps: int, logfile, timing_name: str, timing_phase: str):
    opt = BFGS(filter_obj, logfile=logfile)
    t_start = time.perf_counter()
    t_last = [t_start]

    def log_step_timing():
        now = time.perf_counter()
        step = int(opt.get_number_of_steps())
        print(
            f"TIMING name={timing_name} phase={timing_phase} "
            f"step={step} dt_s={now - t_last[0]:.6f} elapsed_s={now - t_start:.6f}",
            flush=True,
        )
        t_last[0] = now

    opt.attach(log_step_timing, interval=1)
    opt.run(fmax=fmax, steps=steps)
    return opt


def run_one(file_name: str, args, calc) -> int:
    target_path = Path(args.structures_dir) / file_name
    press_path = Path(args.press_dir) / f"{Path(file_name).stem}_press.cif"
    final_path = Path(args.final_dir) / f"{Path(file_name).stem}_opt.cif"
    json_path = Path(args.json_dir) / f"{Path(file_name).stem}.json"
    lock_path = Path(args.lock_dir) / Path(file_name).stem

    if json_path.exists() and json_path.stat().st_size > 0:
        return 0
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return 0

    row = None
    t0 = time.perf_counter()
    try:
        print(f"START {file_name}", flush=True)
        atoms = read(str(target_path))
        initial = geom_stats(atoms, args.cutoff)
        mol_count = molecule_count_from_atoms(atoms, args.molecule_single)
        atoms.calc = calc

        opt_log = "-" if args.optimizer_log else None
        filt = FrechetCellFilter(atoms, scalar_pressure=args.scalar_pressure)
        opt_press = run_optimizer(
            filt,
            args.fmax,
            args.steps,
            opt_log,
            Path(file_name).stem,
            "press",
        )
        atoms.write(str(press_path))

        atoms = read(str(press_path))
        atoms.calc = calc
        filt = FrechetCellFilter(atoms)
        opt_final = run_optimizer(
            filt,
            args.fmax,
            args.steps,
            opt_log,
            Path(file_name).stem,
            "final",
        )
        atoms.write(str(final_path))

        metrics = residual_metrics(atoms, args.cutoff)
        energy = metrics["energy_eV_total"]
        density = metrics["final_density"]
        press_steps = int(opt_press.get_number_of_steps())
        final_steps = int(opt_final.get_number_of_steps())
        status = "success" if metrics["filter_fmax"] <= args.fmax else "not_converged"
        row = {
            "name": Path(file_name).stem,
            "density": density,
            "energy_eV_total": energy,
            "energy_kj_per_mol": energy / mol_count * EV_TO_KJ_MOL,
            "press_steps": press_steps,
            "final_steps": final_steps,
            "status": status,
            "wall_time_s": time.perf_counter() - t0,
            "calculator": "HTGP_Calculator.from_checkpoint",
            "dataset": args.dataset_name,
            "safe_norm": True,
            "charge_spin_fix": True,
            "add_e0_baseline": not args.no_e0,
            "e0_fit_json": args.e0_fit_json,
            **{f"initial_{k}": v for k, v in initial.items()},
            **metrics,
        }
        print(
            f"DONE {file_name} status={status} press_steps={press_steps} "
            f"final_steps={final_steps} filter_fmax={metrics['filter_fmax']:.6g} "
            f"wall={row['wall_time_s']:.1f}s",
            flush=True,
        )
    except Exception as exc:
        row = {
            "name": Path(file_name).stem,
            "density": 100000.0,
            "energy_eV_total": 100000.0,
            "energy_kj_per_mol": 100000.0,
            "press_steps": 100000,
            "final_steps": 100000,
            "status": repr(exc),
            "traceback": traceback.format_exc(limit=12),
            "wall_time_s": time.perf_counter() - t0,
            "calculator": "HTGP_Calculator.from_checkpoint",
            "dataset": args.dataset_name,
            "safe_norm": True,
            "charge_spin_fix": True,
            "add_e0_baseline": not args.no_e0,
            "e0_fit_json": args.e0_fit_json,
        }
        print(f"ERROR {file_name} {row['status']}", flush=True)
    finally:
        with open(json_path, "w") as fh:
            json.dump(row, fh, indent=2)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
    return 1


def aggregate(args) -> None:
    files = sorted(Path(args.structures_dir).glob("*.cif"))
    json_files = sorted(Path(args.json_dir).glob("*.json"))
    if len(json_files) < len(files):
        return

    data_rows = []
    err_rows = []
    for jf in json_files:
        with open(jf) as fh:
            row = json.load(fh)
        if row.get("status") == "success":
            data_rows.append(row)
        else:
            err_rows.append(row)
    if data_rows:
        df = pd.DataFrame(data_rows).sort_values("energy_kj_per_mol").reset_index(drop=True)
        df["relative_energy_kj_per_mol"] = df["energy_kj_per_mol"] - df.loc[0, "energy_kj_per_mol"]
        df.to_csv(args.result_csv, index=False)
    if err_rows:
        pd.DataFrame(err_rows).to_csv(args.error_csv, index=False)


def worker(worker_idx: int, args) -> int:
    time.sleep(worker_idx * args.start_stagger)
    setup_mindspore(args.device_target, args.ms_mode)
    calc = build_official_calc(args)

    rng = random.Random(args.seed + worker_idx)
    old_files = set()
    while True:
        all_files = [p.name for p in Path(args.structures_dir).glob("*.cif")]
        new_files = list(set(all_files) - old_files)
        if not new_files:
            break
        rng.shuffle(new_files)
        for fn in new_files:
            run_one(fn, args, calc)
        old_files.update(new_files)
    return 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default=".")
    parser.add_argument("--inference-root", default=str(PROJECT_ROOT))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-name", default="OMC")
    parser.add_argument("--e0-dir", default=None)
    parser.add_argument("--e0-fit-json", default=None)
    parser.add_argument("--no-e0", action="store_true")
    parser.add_argument("--molecule-single", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--scalar-pressure", type=float, default=0.0006)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--device-target", default="Ascend")
    parser.add_argument("--ms-mode", choices=["pynative", "graph"], default="pynative")
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--start-stagger", type=float, default=2.0)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--optimizer-log", action="store_true")
    args = parser.parse_args()

    wd = Path(args.work_dir).resolve()
    args.structures_dir = str(wd / "structures")
    args.press_dir = str(wd / "cif_result_press")
    args.final_dir = str(wd / "cif_result_final")
    args.json_dir = str(wd / "json_result")
    args.lock_dir = str(wd / "lock")
    args.result_csv = str(wd / "result.csv")
    args.error_csv = str(wd / "error.csv")
    for directory in [args.press_dir, args.final_dir, args.json_dir, args.lock_dir]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    return args


def main():
    args = parse_args()
    if args.smoke > 0:
        setup_mindspore(args.device_target, args.ms_mode)
        calc = build_official_calc(args)
        files = sorted(p.name for p in Path(args.structures_dir).glob("*.cif"))[: args.smoke]
        for fn in files:
            print(f"SMOKE {fn}", flush=True)
            run_one(fn, args, calc)
        aggregate(args)
        return

    Parallel(n_jobs=args.n_jobs)(delayed(worker)(idx, args) for idx in range(args.n_jobs))
    aggregate(args)


if __name__ == "__main__":
    main()
