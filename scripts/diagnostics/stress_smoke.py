#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from ase.io import read


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runner-root", default=str(PROJECT_ROOT / "scripts" / "inference"))
    p.add_argument("--inference-root", default=str(PROJECT_ROOT))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--structures-dir", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--dataset-name", default="OMol25")
    p.add_argument("--e0-dir", default=None)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--device-target", default="Ascend")
    p.add_argument("--ms-mode", choices=["pynative", "graph"], default="pynative")
    p.add_argument("--cutoff", type=float, default=6.0)
    return p.parse_args()


def build_calc(args):
    sys.path.insert(0, str(Path(args.runner_root).resolve()))
    from csp_relax import build_official_calc, setup_mindspore

    setup_mindspore(args.device_target, args.ms_mode)
    calc_args = argparse.Namespace(
        inference_root=args.inference_root,
        checkpoint=args.checkpoint,
        cutoff=args.cutoff,
        e0_dir=args.e0_dir or str(Path(args.inference_root).resolve() / "e0"),
        e0_fit_json=None,
        no_e0=False,
        dataset_name=args.dataset_name,
    )
    return build_official_calc(calc_args)


def density_g_cm3(atoms):
    return sum(atoms.get_masses()) / (atoms.get_volume() * 1e-24) / 6.022140857e23


def main():
    args = parse_args()
    calc = build_calc(args)
    rows = []
    files = sorted(Path(args.structures_dir).glob("*.cif"))[: args.n]
    for path in files:
        t0 = time.perf_counter()
        row = {"name": path.stem, "path": str(path)}
        try:
            atoms = read(str(path))
            atoms.calc = calc
            e = float(atoms.get_potential_energy())
            f = atoms.get_forces()
            s = atoms.get_stress()
            row.update(
                status="success",
                natoms=len(atoms),
                density_g_cm3=float(density_g_cm3(atoms)),
                energy_eV_total=e,
                max_atom_force_eVA=float(np.linalg.norm(f, axis=1).max()),
                rms_atom_force_eVA=float(np.sqrt((f * f).sum(axis=1).mean())),
                stress_voigt_eVA3=[float(x) for x in np.asarray(s).reshape(-1).tolist()],
                stress_norm_eVA3=float(np.linalg.norm(s)),
                wall_time_s=time.perf_counter() - t0,
            )
        except Exception as exc:
            row.update(
                status=repr(exc),
                traceback=traceback.format_exc(limit=10),
                wall_time_s=time.perf_counter() - t0,
            )
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
