import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import mindspore as ms
import numpy as np
from mindspore import mint, ops
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
TRAINING_SCRIPTS = ROOT / "scripts" / "training"
for _p in [str(ROOT), str(TRAINING_SCRIPTS)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import train_multidataset as train_dist
from src.utils import scatter_add


def parse_args():
    default_model_params = json.dumps(train_dist.Config.MODEL_PARAMS, ensure_ascii=True)

    parser = argparse.ArgumentParser(
        description="Validate a checkpoint and report energy/force MAE grouped by batch.dataset."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to the MindSpore checkpoint.")
    parser.add_argument(
        "--meta-file",
        default=train_dist.Config.TEST_META,
        help="Metadata filename suffix used by validation datasets.",
    )
    parser.add_argument(
        "--data-dir",
        nargs="+",
        default=list(train_dist.Config.DATA_DIR),
        help="Dataset root directories.",
    )
    parser.add_argument(
        "--data-dir-names",
        nargs="+",
        default=list(train_dist.Config.DATA_DIR_NAMES),
        help="Dataset names aligned with --data-dir.",
    )
    parser.add_argument(
        "--model-params",
        default=default_model_params,
        help="JSON dict used to override train_multidataset.Config.MODEL_PARAMS.",
    )
    parser.add_argument(
        "--max-cost-per-batch",
        type=float,
        default=train_dist.Config.MAX_COST_PER_BATCH,
        help="Bin packing max cost per batch.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=train_dist.Config.NUM_WORKERS,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=train_dist.Config.PREFETCH_FACTOR,
        help="Prefetch factor for the dataloader.",
    )
    parser.add_argument(
        "--device-target",
        default=os.environ.get("DEVICE_TARGET"),
        help="Optional MindSpore device target, e.g. CPU/GPU/Ascend.",
    )
    parser.add_argument(
        "--ms-mode",
        choices=["auto", "pynative", "graph"],
        default="auto",
        help="MindSpore execution mode. 'auto' keeps the runtime default used by the environment.",
    )
    parser.add_argument(
        "--distributed",
        choices=["auto", "off"],
        default="auto",
        help="Use DATA_PARALLEL when the environment is launched in distributed mode.",
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="Optional debug limit on the number of validation batches.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to save the grouped metrics as CSV.",
    )
    return parser.parse_args()


def configure_runtime(args):
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    if args.device_target:
        if hasattr(ms, "set_device"):
            ms.set_device(device_target=args.device_target)
        else:
            ms.set_context(device_target=args.device_target)

    if args.ms_mode == "pynative":
        ms.set_context(mode=ms.PYNATIVE_MODE)
    elif args.ms_mode == "graph":
        ms.set_context(mode=ms.GRAPH_MODE)


def _kill_port(port: int) -> bool:
    """Kill any process listening on the given TCP port. Returns True if a process was found."""
    try:
        result = subprocess.run(
            ["fuser", "-k", f"{port}/tcp"],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
        pids = result.stdout.strip().split()
        if pids:
            subprocess.run(["kill", "-9"] + pids, timeout=5)
            return True
    except Exception:
        pass
    return False


def init_parallel(args):
    if args.distributed == "off":
        return 0, 1

    parallel_mode = os.environ.get("PARALLEL_MODE", "NONE").upper()
    if parallel_mode == "DATA_PARALLEL":
        try:
            return train_dist.init_distributed_mode()
        except RuntimeError as exc:
            msg = str(exc)
            if "Repeated registration" in msg or "18082" in msg:
                port = 18082
                print(
                    f"[WARNING] Stale scheduler detected on port {port}. "
                    "Killing it and retrying..."
                )
                _kill_port(port)
                time.sleep(3)
                return train_dist.init_distributed_mode()
            raise

    return 0, 1


def apply_overrides(args):
    model_params = json.loads(args.model_params)

    if len(args.data_dir) != len(args.data_dir_names):
        raise ValueError("--data-dir and --data-dir-names must have the same length.")

    train_dist.Config.DATA_DIR = list(args.data_dir)
    train_dist.Config.DATA_DIR_NAMES = list(args.data_dir_names)
    train_dist.Config.MAX_COST_PER_BATCH = args.max_cost_per_batch
    train_dist.Config.NUM_WORKERS = args.num_workers
    train_dist.Config.PREFETCH_FACTOR = args.prefetch_factor
    train_dist.Config.MODEL_PARAMS = model_params


def infer_num_graphs(batch) -> int:
    if hasattr(batch, "num_graphs"):
        return int(batch.num_graphs)
    return int(ops.reduce_max(batch.batch).asnumpy().item()) + 1


def normalize_dataset_names(batch_dataset, num_graphs: int, idx_to_name: Dict[int, str]) -> List[str]:
    if batch_dataset is None:
        return ["unknown"] * num_graphs

    if isinstance(batch_dataset, str):
        return [batch_dataset] * num_graphs

    if isinstance(batch_dataset, bytes):
        return [batch_dataset.decode("utf-8")] * num_graphs

    if isinstance(batch_dataset, ms.Tensor):
        values = batch_dataset.asnumpy().reshape(-1).tolist()
        return [idx_to_name.get(int(v), str(int(v))) for v in values]

    values = list(batch_dataset)
    if len(values) == 1 and num_graphs > 1:
        values = values * num_graphs
    if len(values) != num_graphs:
        raise ValueError(
            f"batch.dataset has length {len(values)}, but num_graphs is {num_graphs}."
        )

    names = []
    for value in values:
        if isinstance(value, bytes):
            names.append(value.decode("utf-8"))
        elif isinstance(value, (int, float)):
            names.append(idx_to_name.get(int(value), str(int(value))))
        else:
            names.append(str(value))
    return names


EV_A3_TO_GPA = 160.21766


def predict_energy_force(model, batch):
    use_direct_force = model.cfg.use_direct_force
    num_graphs = infer_num_graphs(batch)

    original_pos = batch.pos
    original_cell = getattr(batch, "cell", None)

    if use_direct_force:
        out = model(batch)
        if isinstance(out, dict):
            pred_e = out.get("energy")
            if pred_e is None:
                raise ValueError("Model output is missing key 'energy'.")
            pred_e = pred_e.view(-1)
            pred_f = out.get("force", mint.zeros_like(batch.pos))
        else:
            pred_e = out.view(-1)
            pred_f = mint.zeros_like(batch.pos)
        return pred_e.astype(ms.float32), pred_f.astype(ms.float32), None

    displacement = mint.zeros((num_graphs, 3, 3)).astype(ms.float32)

    def get_energy(pos, disp):
        symmetric_strain = 0.5 * (disp + ops.transpose(disp, (0, 2, 1)))
        strain_per_atom = symmetric_strain[batch.batch]
        pos_deformed = pos + mint.einsum("ni,nij->nj", pos, strain_per_atom)

        batch.pos = pos_deformed
        if original_cell is not None and len(original_cell.shape) == 3:
            batch.cell = original_cell + ops.matmul(original_cell, symmetric_strain)

        out = model(batch)
        if isinstance(out, dict):
            pred_e_inner = out["energy"].view(-1).astype(ms.float32)
        else:
            pred_e_inner = out.view(-1).astype(ms.float32)
        return pred_e_inner, original_cell

    grads_fn = ms.value_and_grad(get_energy, grad_position=(0, 1), has_aux=True)
    (pred_e, _), grads = grads_fn(original_pos, displacement)

    batch.pos = original_pos
    if original_cell is not None:
        batch.cell = original_cell

    pred_f = -grads[0] if grads[0] is not None else mint.zeros_like(batch.pos)

    dE_dStrain = grads[1]
    if dE_dStrain is not None:
        if original_cell is not None:
            vol = mint.abs(mint.exp(ops.logdet(original_cell).astype(ms.float32))).view(-1, 1, 1)
        else:
            vol = mint.ones((num_graphs, 1, 1), ms.float32)
        pred_stress = dE_dStrain.astype(ms.float32) / vol
    else:
        pred_stress = mint.zeros((num_graphs, 3, 3), ms.float32)

    return pred_e.astype(ms.float32), pred_f.astype(ms.float32), pred_stress


def init_group_state():
    return {
        "energy_abs_sum": 0.0,
        "energy_count": 0,
        "force_abs_sum": 0.0,
        "force_count": 0,
        "stress_abs_sum": 0.0,
        "stress_count": 0,
        "num_graphs": 0,
        "num_atoms": 0,
    }


def accumulate_batch(stats, batch, pred_e, pred_f, pred_stress, cfg):
    if not hasattr(batch, "force") or batch.force is None:
        raise ValueError("Validation batch does not contain force labels.")

    num_graphs = infer_num_graphs(batch)

    ones = mint.ones(batch.batch.shape, dtype=ms.float32)
    num_atoms = scatter_add(ones, batch.batch, dim=0, dim_size=num_graphs).view(-1).clamp(min=1)

    target_e = batch.y.view(-1).astype(ms.float32)
    target_f = batch.force.astype(ms.float32)

    energy_abs = ops.abs(pred_e / num_atoms - target_e / num_atoms)
    force_abs_per_atom = ops.abs(pred_f.astype(ms.float32) - target_f).sum(axis=1)
    force_abs_sum_per_graph = scatter_add(
        force_abs_per_atom,
        batch.batch,
        dim=0,
        dim_size=num_graphs,
    ).view(-1)

    # Stress: per-graph mask matching trainer logic (norm > 1e-6 and stress_datasets config)
    dataset_names = list(batch.dataset)
    stress_mask_list = [False] * num_graphs
    stress_abs_per_graph = None
    if (
        pred_stress is not None
        and hasattr(batch, "stress")
        and batch.stress is not None
    ):
        target_s = batch.stress.astype(ms.float32)
        stress_norm = mint.norm(target_s.view(num_graphs, -1), dim=1)
        stress_mask = (stress_norm > 1e-6)

        if hasattr(cfg, "stress_datasets"):
            ds_flags = [cfg.stress_datasets.get(name, False) for name in dataset_names]
            stress_mask = stress_mask & ms.Tensor(ds_flags, dtype=ms.bool_)

        stress_abs_per_graph = ops.abs(
            pred_stress.view(num_graphs, -1) - target_s.view(num_graphs, -1)
        ).sum(axis=1)
        stress_mask_list = stress_mask.asnumpy().tolist()

    for graph_idx, dataset_name in enumerate(dataset_names):
        entry = stats[dataset_name]
        entry["energy_abs_sum"] += float(energy_abs[graph_idx].asnumpy().item())
        entry["energy_count"] += 1
        entry["force_abs_sum"] += float(force_abs_sum_per_graph[graph_idx].asnumpy().item())
        entry["force_count"] += int(num_atoms[graph_idx].asnumpy().item()) * 3
        entry["num_graphs"] += 1
        entry["num_atoms"] += int(num_atoms[graph_idx].asnumpy().item())
        if stress_mask_list[graph_idx] and stress_abs_per_graph is not None:
            entry["stress_abs_sum"] += float(stress_abs_per_graph[graph_idx].asnumpy().item())
            entry["stress_count"] += 9  # 3x3 components


def build_dataset_order(idx_to_name: Dict[int, str]) -> List[str]:
    names = [name for _, name in sorted(idx_to_name.items(), key=lambda item: item[0])]
    if "unknown" not in names:
        names.append("unknown")
    return names


def stats_to_tensors(stats, dataset_order: List[str]):
    sums = np.zeros((len(dataset_order), 3), dtype=np.float32)
    counts = np.zeros((len(dataset_order), 5), dtype=np.int32)
    for row_idx, dataset_name in enumerate(dataset_order):
        entry = stats.get(dataset_name, init_group_state())
        sums[row_idx, 0] = entry["energy_abs_sum"]
        sums[row_idx, 1] = entry["force_abs_sum"]
        sums[row_idx, 2] = entry["stress_abs_sum"]
        counts[row_idx, 0] = entry["energy_count"]
        counts[row_idx, 1] = entry["force_count"]
        counts[row_idx, 2] = entry["stress_count"]
        counts[row_idx, 3] = entry["num_graphs"]
        counts[row_idx, 4] = entry["num_atoms"]
    return ms.Tensor(sums, dtype=ms.float32), ms.Tensor(counts, dtype=ms.int32)


def tensors_to_stats(sum_tensor: ms.Tensor, count_tensor: ms.Tensor, dataset_order: List[str]):
    sum_array = sum_tensor.asnumpy()
    count_array = count_tensor.asnumpy()
    stats = defaultdict(init_group_state)
    for row_idx, dataset_name in enumerate(dataset_order):
        stats[dataset_name] = {
            "energy_abs_sum": float(sum_array[row_idx, 0]),
            "energy_count": int(count_array[row_idx, 0]),
            "force_abs_sum": float(sum_array[row_idx, 1]),
            "force_count": int(count_array[row_idx, 1]),
            "stress_abs_sum": float(sum_array[row_idx, 2]),
            "stress_count": int(count_array[row_idx, 2]),
            "num_graphs": int(count_array[row_idx, 3]),
            "num_atoms": int(count_array[row_idx, 4]),
        }
    return stats


def reduce_stats_across_ranks(stats, dataset_order: List[str], world_size: int):
    if world_size == 1:
        return stats

    reduce_op = ops.ReduceOp.SUM if hasattr(ops, "ReduceOp") else ms.ops.ReduceOp.SUM
    reducer = ops.AllReduce(op=reduce_op)
    sum_tensor, count_tensor = stats_to_tensors(stats, dataset_order)
    sum_tensor = reducer(sum_tensor)
    count_tensor = reducer(count_tensor)
    return tensors_to_stats(sum_tensor, count_tensor, dataset_order)


def finalize_stats(stats):
    rows = []
    total = init_group_state()

    for dataset_name in sorted(stats.keys()):
        entry = stats[dataset_name]
        total["energy_abs_sum"] += entry["energy_abs_sum"]
        total["energy_count"] += entry["energy_count"]
        total["force_abs_sum"] += entry["force_abs_sum"]
        total["force_count"] += entry["force_count"]
        total["stress_abs_sum"] += entry["stress_abs_sum"]
        total["stress_count"] += entry["stress_count"]
        total["num_graphs"] += entry["num_graphs"]
        total["num_atoms"] += entry["num_atoms"]

        stress_mae = entry["stress_abs_sum"] / max(entry["stress_count"], 1)
        rows.append(
            {
                "dataset": dataset_name,
                "num_graphs": entry["num_graphs"],
                "num_atoms": entry["num_atoms"],
                "energy_mae": entry["energy_abs_sum"] / max(entry["energy_count"], 1),
                "force_mae": entry["force_abs_sum"] / max(entry["force_count"], 1),
                "stress_mae_gpa": stress_mae * EV_A3_TO_GPA,
            }
        )

    total_stress_mae = total["stress_abs_sum"] / max(total["stress_count"], 1)
    summary = {
        "dataset": "ALL",
        "num_graphs": total["num_graphs"],
        "num_atoms": total["num_atoms"],
        "energy_mae": total["energy_abs_sum"] / max(total["energy_count"], 1),
        "force_mae": total["force_abs_sum"] / max(total["force_count"], 1),
        "stress_mae_gpa": total_stress_mae * EV_A3_TO_GPA,
    }
    return summary, rows


def print_rows(summary, rows):
    all_rows = [summary] + rows
    header = (
        f"{'dataset':<12} {'graphs':>10} {'atoms':>10}"
        f" {'energy_mae':>16} {'force_mae':>16} {'stress_mae_gpa':>16}"
    )
    print(header)
    print("-" * len(header))
    for row in all_rows:
        print(
            f"{row['dataset']:<12} "
            f"{row['num_graphs']:>10d} "
            f"{row['num_atoms']:>10d} "
            f"{row['energy_mae']:>16.8f} "
            f"{row['force_mae']:>16.8f} "
            f"{row['stress_mae_gpa']:>16.8f}"
        )


def write_csv(path, summary, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset", "num_graphs", "num_atoms",
                "energy_mae", "force_mae", "stress_mae_gpa",
            ],
        )
        writer.writeheader()
        writer.writerow(summary)
        writer.writerows(rows)


def main():
    args = parse_args()
    configure_runtime(args)
    apply_overrides(args)

    rank, world_size = init_parallel(args)

    loader, sampler = train_dist.get_dataloader(
        train_dist.Config.DATA_DIR,
        args.meta_file,
        rank,
        world_size,
        is_train=False,
    )

    avg_neighborhood = 1 / sampler.edge_weight
    checkpoint_weights = ms.load_checkpoint(args.checkpoint)
    model = train_dist.build_model(rank, avg_neighborhood, restart=True, state_dict=checkpoint_weights)
    model.set_train(False)

    stats = defaultdict(init_group_state)
    idx_to_name = {v: k for k, v in model.cfg.dataset_types.items()}
    dataset_order = build_dataset_order(idx_to_name)
    total_batches = args.limit_batches if args.limit_batches is not None else len(loader)

    if rank == 0:
        mode_name = "distributed" if world_size > 1 else "single-device"
        print(f"Running {mode_name} validation on {world_size} device(s)")

    cfg = model.cfg

    log_interval = max(1, total_batches // 20)  # flush CSV every ~5% of batches

    for batch_idx, batch in enumerate(
        tqdm(
            loader,
            total=total_batches,
            desc="Validating",
            leave=False,
            disable=(rank != 0),
        )
    ):
        pred_e, pred_f, pred_stress = predict_energy_force(model, batch)
        batch.dataset = normalize_dataset_names(
            getattr(batch, "dataset", None),
            infer_num_graphs(batch),
            idx_to_name,
        )
        accumulate_batch(stats, batch, pred_e, pred_f, pred_stress, cfg)

        if args.output_csv and rank == 0 and (batch_idx + 1) % log_interval == 0:
            _summary, _rows = finalize_stats(stats)
            write_csv(args.output_csv, _summary, _rows)

        if args.limit_batches is not None and (batch_idx + 1) >= args.limit_batches:
            break

    stats = reduce_stats_across_ranks(stats, dataset_order, world_size)
    summary, rows = finalize_stats(stats)
    if rank == 0:
        print_rows(summary, rows)

    if args.output_csv and rank == 0:
        write_csv(args.output_csv, summary, rows)
        print(f"\nSaved CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
