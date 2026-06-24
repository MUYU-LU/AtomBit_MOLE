#!/usr/bin/env python3
"""Probe charge/spin default sensitivity for the SPICE-finetuned checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase.io import read


EV_TO_KJ_MOL = 96.48533212331002
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def setup_ms(device_id: str):
    import mindspore as ms

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")


def load_e0_fit(path: str | None):
    if not path:
        return None
    payload = json.load(open(path))
    return {int(k): float(v) for k, v in payload.items()}


def build_calc(args):
    sys.path.insert(0, str(Path(args.inference_root).resolve()))
    from src.utils.Calculator import HTGP_Calculator

    e0_dir = args.e0_dir or str(PROJECT_ROOT / "e0")
    calc = HTGP_Calculator.from_checkpoint(
        ckpt_path=str(Path(args.checkpoint).resolve()),
        e0_dir=e0_dir,
        add_e0_baseline=True,
        dataset_name=args.dataset_name,
        default_charge=0.0,
        default_spin=0.0,
    )
    e0_fit = load_e0_fit(args.e0_fit_json)
    if e0_fit:
        calc.e0_fit = e0_fit
    return calc


def mol_count(atoms, molecule_single: int) -> float:
    return len(atoms.get_atomic_numbers()) / molecule_single


def density(atoms) -> float:
    return sum(atoms.get_masses()) / (atoms.get_volume() * 1e-24) / (6.022140857e23)


def eval_one(path: Path, calc, molecule_single: int, explicit_neutral: bool = False):
    atoms = read(str(path))
    if explicit_neutral:
        atoms.info["charge"] = 0.0
        atoms.info["spin"] = 0.0
    atoms.calc = calc
    e = float(atoms.get_potential_energy())
    f = np.asarray(atoms.get_forces(), dtype=float)
    try:
        s = np.asarray(atoms.get_stress(), dtype=float)
    except Exception:
        s = np.zeros(6)
    return {
        "energy_eV_total": e,
        "energy_kj_per_mol": e / mol_count(atoms, molecule_single) * EV_TO_KJ_MOL,
        "force_component_rms": float(np.sqrt(np.mean(f * f))),
        "force_atom_max": float(np.max(np.linalg.norm(f, axis=1))),
        "stress_l2": float(np.linalg.norm(s)),
        "density": float(density(atoms)),
        "formula": atoms.get_chemical_formula(),
        "num_atoms": len(atoms),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--structures-dir", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--inference-root", default=str(PROJECT_ROOT))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--e0-dir", default=None)
    p.add_argument("--e0-fit-json", default=None)
    p.add_argument("--dataset-name", default="OMol25")
    p.add_argument("--molecule-single", type=int, default=1)
    p.add_argument("--limit", type=int, default=64)
    args = p.parse_args()

    setup_ms("0")
    files = sorted(Path(args.structures_dir).glob("*.cif"))[: args.limit]
    calc_missing = build_calc(args)
    calc_fixed = build_calc(args)

    rows = []
    for path in files:
        print(f"START {path.name}", flush=True)
        missing = eval_one(path, calc_missing, args.molecule_single)
        fixed = eval_one(path, calc_fixed, args.molecule_single, explicit_neutral=True)
        row = {
            "name": path.stem,
            "file": str(path),
            "missing": missing,
            "fixed": fixed,
            "delta_fixed_minus_missing_E_kJmol": fixed["energy_kj_per_mol"] - missing["energy_kj_per_mol"],
            "delta_fixed_minus_missing_F_rms": fixed["force_component_rms"] - missing["force_component_rms"],
            "delta_fixed_minus_missing_stress_l2": fixed["stress_l2"] - missing["stress_l2"],
        }
        rows.append(row)
        print(
            f"DONE {path.name} dE={row['delta_fixed_minus_missing_E_kJmol']:.6f} "
            f"dF={row['delta_fixed_minus_missing_F_rms']:.6f}",
            flush=True,
        )

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(args.out_json, "w"), indent=2)


if __name__ == "__main__":
    main()
