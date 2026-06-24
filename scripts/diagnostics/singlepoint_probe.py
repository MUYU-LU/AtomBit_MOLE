#!/usr/bin/env python3
"""Single-point CSP evaluation for norm/checkpoint ablations.

This intentionally does not relax structures. It evaluates energy, forces and
stress for the same CIF set while controlling dataset/charge/spin.
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
from ase.io import read


EV_TO_KJ_MOL = 96.48533212331002
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def setup_mindspore(device_target: str, mode: str) -> None:
    import mindspore as ms

    ms_mode = ms.PYNATIVE_MODE if mode.lower() == "pynative" else ms.GRAPH_MODE
    ms.set_context(mode=ms_mode, device_target=device_target)


def build_calc(args):
    inference_root = Path(args.inference_root).resolve()
    sys.path.insert(0, str(inference_root))

    from src.utils.Calculator import HTGP_Calculator

    e0_dir = args.e0_dir or str(inference_root / "e0")
    calc = HTGP_Calculator.from_checkpoint(
        ckpt_path=str(Path(args.checkpoint).resolve()),
        cutoff=args.cutoff,
        e0_dir=e0_dir,
        add_e0_baseline=not args.no_e0,
        dataset_name=args.dataset_name,
        default_charge=args.charge,
        default_spin=args.spin,
    )
    return calc


def calculate_density(atoms) -> float:
    total_mass = sum(atoms.get_masses())
    volume = atoms.get_volume()
    return total_mass / (volume * 1e-24) / (6.022140857e23)


def molecule_count_from_atoms(atoms, molecule_single: int) -> float:
    if molecule_single <= 0:
        raise ValueError("molecule_single must be positive")
    return len(atoms.get_atomic_numbers()) / molecule_single


def row_for_file(file_name: str, args, calc) -> dict:
    t0 = time.perf_counter()
    path = Path(args.structures_dir) / file_name
    row = {
        "name": Path(file_name).stem,
        "file_name": file_name,
        "checkpoint_label": args.checkpoint_label,
        "norm_label": args.norm_label,
        "dataset": args.dataset_name,
        "charge": args.charge,
        "spin": args.spin,
        "add_e0_baseline": not args.no_e0,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "inference_root": str(Path(args.inference_root).resolve()),
        "status": "unknown",
    }
    try:
        atoms = read(str(path))
        atoms.calc = calc
        mol_count = molecule_count_from_atoms(atoms, args.molecule_single)
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)
        if atoms.pbc.any():
            stress = np.asarray(atoms.get_stress(), dtype=float)
        else:
            stress = np.zeros(6, dtype=float)

        force_norms = np.linalg.norm(forces, axis=1)
        row.update(
            {
                "status": "success",
                "num_atoms": int(len(atoms)),
                "formula": atoms.get_chemical_formula(),
                "density": float(calculate_density(atoms)),
                "energy_eV_total": energy,
                "energy_kj_per_mol": float(energy / mol_count * EV_TO_KJ_MOL),
                "force_component_mae": float(np.mean(np.abs(forces))),
                "force_component_rms": float(np.sqrt(np.mean(forces * forces))),
                "force_atom_norm_mean": float(np.mean(force_norms)),
                "force_atom_norm_max": float(np.max(force_norms)),
                "stress_voigt": stress.tolist(),
                "stress_abs_mean": float(np.mean(np.abs(stress))),
                "stress_l2": float(np.linalg.norm(stress)),
                "wall_time_s": float(time.perf_counter() - t0),
            }
        )
    except Exception as exc:
        row.update(
            {
                "status": repr(exc),
                "traceback": traceback.format_exc(limit=12),
                "wall_time_s": float(time.perf_counter() - t0),
            }
        )
    return row


def run_worker(args) -> None:
    setup_mindspore(args.device_target, args.ms_mode)
    calc = build_calc(args)
    rng = random.Random(args.seed + args.worker_index)
    structures_dir = Path(args.structures_dir)
    json_dir = Path(args.json_dir)
    lock_dir = Path(args.lock_dir)
    json_dir.mkdir(parents=True, exist_ok=True)
    lock_dir.mkdir(parents=True, exist_ok=True)

    while True:
        files = [p.name for p in structures_dir.glob("*.cif")]
        rng.shuffle(files)
        progressed = False
        for file_name in files:
            out = json_dir / f"{Path(file_name).stem}.json"
            if out.exists() and out.stat().st_size > 0:
                continue
            lock = lock_dir / Path(file_name).stem
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
            except FileExistsError:
                continue
            print(f"START {args.checkpoint_label}/{args.norm_label} {file_name}", flush=True)
            row = row_for_file(file_name, args, calc)
            with open(out, "w") as fh:
                json.dump(row, fh, indent=2)
            print(
                f"DONE {args.checkpoint_label}/{args.norm_label} {file_name} "
                f"status={row.get('status')} wall={row.get('wall_time_s'):.3f}s",
                flush=True,
            )
            try:
                lock.unlink()
            except FileNotFoundError:
                pass
            progressed = True
        if not progressed:
            return


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--structures-dir", required=True)
    p.add_argument("--json-dir", required=True)
    p.add_argument("--lock-dir", required=True)
    p.add_argument("--inference-root", default=str(PROJECT_ROOT))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--checkpoint-label", required=True)
    p.add_argument("--norm-label", required=True)
    p.add_argument("--e0-dir", default=None)
    p.add_argument("--dataset-name", default="OMC")
    p.add_argument("--charge", type=float, default=0.0)
    p.add_argument("--spin", type=float, default=0.0)
    p.add_argument("--molecule-single", type=int, default=1)
    p.add_argument("--cutoff", type=float, default=None)
    p.add_argument("--no-e0", action="store_true")
    p.add_argument("--device-target", default="Ascend")
    p.add_argument("--ms-mode", default="pynative", choices=["pynative", "graph"])
    p.add_argument("--seed", type=int, default=20260611)
    p.add_argument("--worker-index", type=int, default=0)
    return p.parse_args()


def main() -> None:
    run_worker(parse_args())


if __name__ == "__main__":
    main()
