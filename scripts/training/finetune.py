"""Single-card or multi-card fine-tuning script for AtomBit (HTGP) model.

The script is launched in two ways:

Single card:
    python scripts/training/finetune.py --ckpt model.ckpt --data_dir ./train_data \
        --dataset_name UserData --epochs 5 --lr 1e-4 \
        --batch_cost 2000 --output_dir ./finetune_out \
        --progress_file ./progress.jsonl --device_id 0

Multi card (launched by msrun externally, PARALLEL_MODE injected via env):
    PARALLEL_MODE=DATA_PARALLEL msrun --worker-num=N \
        scripts/training/finetune.py --ckpt model.ckpt ... --num_devices N

Progress events written to --progress_file (one JSON per line):
    {"type": "finetune_epoch", "epoch": 1, "total_epochs": 5,
     "train_loss": 0.234, "val_loss": 0.198, "mae_f": 0.012,
     "elapsed_s": 120, "eta_s": 480}
    {"type": "completed", "ckpt_path": "/path/to/finetune_final.ckpt",
     "final_loss": 0.198, "epochs_trained": 5}
    {"type": "failed", "error": "..."}
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so src.* imports work
_SCRIPT_DIR = Path(__file__).resolve().parents[2]
for _p in [str(_SCRIPT_DIR), str(_SCRIPT_DIR / "sharker")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np


# ---------------------------------------------------------------------------
# Progress writer
# ---------------------------------------------------------------------------

_progress_file: str | None = None


def _write_progress(event: dict) -> None:
    if _progress_file is None:
        return
    try:
        with open(_progress_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            fh.flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Distributed init
# ---------------------------------------------------------------------------

def init_distributed(num_devices: int) -> tuple:
    """Initialise MindSpore distributed if num_devices > 1.

    Returns (rank, world_size).
    """
    import mindspore as ms
    from mindspore.communication import init, get_rank, get_group_size

    parallel_mode = os.environ.get("PARALLEL_MODE", "").upper()

    if num_devices > 1 or parallel_mode == "DATA_PARALLEL":
        init()
        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True
        )
        rank = get_rank()
        world_size = get_group_size()
        return rank, world_size
    else:
        return 0, 1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_dataloader(data_dir: str, meta_file: str, dataset_name: str,
                     rank: int, world_size: int,
                     batch_cost: int, is_train: bool = True):
    """Build a dataloader from user-converted H5 dataset."""
    from src.data import ChunkedSmartDataset_h5, MultiDatasetBinPackingSampler
    from src.data.Dataset_dist import MultiSourceGraphDataset
    from sharker.loader.dataloader import Dataloader

    full_path = os.path.join(data_dir, meta_file)
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"Metadata file not found: {full_path}\n"
            "Run scripts/data/prepare_h5_dataset.py first."
        )

    dataset = ChunkedSmartDataset_h5(
        data_dir,
        metadata_file=meta_file,
        rank=rank,
        world_size=world_size,
    )
    router_ds = MultiSourceGraphDataset({dataset_name: dataset})

    sampler = MultiDatasetBinPackingSampler(
        {dataset_name: dataset.metadata},
        max_cost=batch_cost,
        edge_weight="auto",
        shuffle=is_train,
        world_size=world_size,
        rank=rank,
    )
    loader = Dataloader(
        router_ds,
        sampler=sampler,
        num_parallel_workers=4,
        prefetch_factor=2,
    )
    return loader, sampler


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_model(ckpt_path: str, avg_neighborhood: float, rank: int):
    """Load HTGP model from checkpoint.

    Architecture config is read from the checkpoint itself (preferred) or
    inferred from parameter shapes, so the model always matches the ckpt.
    """
    import mindspore as ms
    from src.models import HTGPModel

    checkpoint_weights = ms.load_checkpoint(ckpt_path)
    new_state = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in checkpoint_weights.items()
    }

    # Try to recover config from checkpoint metadata, then fall back to shape inference.
    try:
        from src.utils.Calculator import (
            _decode_config_from_param_dict,
            _infer_config_from_param_dict,
        )
        cfg = _decode_config_from_param_dict(new_state) or _infer_config_from_param_dict(new_state)
    except ImportError:
        cfg = None

    if cfg is None:
        # Last-resort fallback: generic defaults (may not match ckpt)
        from src.utils import HTGPConfig
        cfg = HTGPConfig(hidden_dim=128, num_layers=2, cutoff=6.0, num_rbf=32)
        if rank == 0:
            print("Warning: could not read config from checkpoint; using default HTGPConfig.")

    cfg.avg_neighborhood = avg_neighborhood
    model = HTGPModel(cfg)

    not_loaded, _ = ms.load_param_into_net(model, new_state)
    if rank == 0:
        if not_loaded:
            print(f"Warning: {len(not_loaded)} params not loaded: {not_loaded[:5]}")
        else:
            print(f"Checkpoint loaded: {ckpt_path}")

    return model


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AtomBit fine-tuning")
    parser.add_argument("--ckpt", required=True, help="Base checkpoint path")
    parser.add_argument("--data_dir", required=True, help="User data directory (H5 + metadata)")
    parser.add_argument("--dataset_name", default="UserData",
                        help="Metadata file prefix (e.g. UserData)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_cost", type=int, default=2000,
                        help="Max atoms per batch")
    parser.add_argument("--output_dir", required=True, help="Output directory for checkpoints")
    parser.add_argument("--progress_file", default=None,
                        help="File to write per-epoch JSON progress lines")
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--device_id", type=int, default=0,
                        help="Device ID for single-card mode")
    args = parser.parse_args()

    global _progress_file
    _progress_file = args.progress_file

    import mindspore as ms
    from mindspore import context

    # --- Set device context ---
    if args.num_devices == 1:
        # Single-card: set device_id explicitly
        try:
            context.set_context(
                mode=context.PYNATIVE_MODE,
                device_target="Ascend",
                device_id=args.device_id,
            )
        except Exception:
            try:
                context.set_context(
                    mode=context.PYNATIVE_MODE,
                    device_target="GPU",
                    device_id=args.device_id,
                )
            except Exception:
                context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    rank, world_size = init_distributed(args.num_devices)

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Fine-tuning: rank={rank} world_size={world_size}")
        print(f"  ckpt={args.ckpt}")
        print(f"  data_dir={args.data_dir}")
        print(f"  epochs={args.epochs}, lr={args.lr}, batch_cost={args.batch_cost}")

    t_start = time.time()

    # --- Data ---
    train_meta = f"{args.dataset_name}_train_metadata.pkl"
    test_meta = f"{args.dataset_name}_test_metadata.pkl"

    train_loader, train_sampler = build_dataloader(
        args.data_dir, train_meta, args.dataset_name,
        rank, world_size, args.batch_cost, is_train=True,
    )

    has_test = os.path.exists(os.path.join(args.data_dir, test_meta))
    test_loader = test_sampler = None
    if has_test:
        try:
            test_loader, test_sampler = build_dataloader(
                args.data_dir, test_meta, args.dataset_name,
                rank, world_size, args.batch_cost, is_train=False,
            )
        except Exception as exc:
            if rank == 0:
                print(f"Warning: test loader failed ({exc}), skipping validation")

    # --- Model ---
    avg_neighborhood = 1.0 / train_sampler.edge_weight if train_sampler.edge_weight > 0 else 89.0
    model = build_model(args.ckpt, avg_neighborhood, rank)

    # --- Trainer ---
    from src.engine import PotentialTrainer

    train_total_steps = train_sampler.precompute_total_steps(args.epochs)
    if rank == 0:
        print(f"Estimated total steps: {train_total_steps}")

    trainer = PotentialTrainer(
        model,
        total_steps=train_total_steps,
        max_lr=args.lr,
        checkpoint_dir=args.output_dir,
        epochs=args.epochs,
        finetune_mode=True,
        saves_per_epoch=1,
    )

    steps_per_epoch = max(1, train_total_steps // args.epochs)
    final_loss = float("inf")

    # --- Per-step progress callback (rank 0 only) ---
    def _make_step_cb(total_epochs, spe):
        def _cb(epoch, step, metrics):
            _write_progress({
                "type": "finetune_step",
                "epoch": epoch,
                "total_epochs": total_epochs,
                "step": step,
                "steps_per_epoch": spe,
                "train_loss": round(float(metrics.get("total_loss", 0.0)), 6),
                "mae_f": round(float(metrics.get("mae_f", 0.0)), 6),
                "elapsed_s": round(time.time() - t_start, 1),
            })
        return _cb

    step_cb = _make_step_cb(args.epochs, steps_per_epoch) if rank == 0 else None

    # --- Training loop ---
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        t_epoch_start = time.time()

        train_metrics = trainer.train_epoch(
            train_loader, epoch_idx=epoch, skip_steps=0,
            steps_per_epoch=steps_per_epoch,
            on_step_fn=step_cb,
        )

        val_metrics = {"total_loss": 0.0, "mae_f": 0.0}
        if test_loader is not None:
            try:
                val_metrics = trainer.validate(test_loader, epoch_idx=epoch)
            except Exception:
                pass

        elapsed = time.time() - t_start
        epoch_time = time.time() - t_epoch_start
        remaining_epochs = args.epochs - epoch
        eta = epoch_time * remaining_epochs

        train_loss = train_metrics.get("total_loss", 0.0)
        final_loss = val_metrics.get("total_loss", train_loss)

        if rank == 0:
            msg = (
                f"Ep {epoch:03d}/{args.epochs} | "
                f"T_Loss: {train_loss:.4f} | "
                f"V_Loss: {val_metrics['total_loss']:.4f} | "
                f"MAE_F: {train_metrics.get('mae_f', 0)*1000:.1f} meV/A"
            )
            print(msg)

            _write_progress({
                "type": "finetune_epoch",
                "epoch": epoch,
                "total_epochs": args.epochs,
                "train_loss": round(float(train_loss), 6),
                "val_loss": round(float(val_metrics["total_loss"]), 6),
                "mae_f": round(float(train_metrics.get("mae_f", 0.0)), 6),
                "elapsed_s": round(elapsed, 1),
                "eta_s": round(eta, 1),
                "epoch_time_s": round(epoch_time, 1),
            })

        if rank == 0:
            ckpt_name = f"finetune_epoch_{epoch}.ckpt"
            trainer.save(
                ckpt_name,
                epoch=epoch,
                val_metrics=val_metrics,
                steps_per_epoch=steps_per_epoch,
            )

    # --- Copy final checkpoint ---
    final_ckpt = None
    if rank == 0:
        import shutil

        src = os.path.join(args.output_dir, f"finetune_epoch_{args.epochs}.ckpt")
        dst = os.path.join(args.output_dir, "finetune_final.ckpt")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            final_ckpt = dst
        else:
            # Find any ckpt written by trainer
            candidates = sorted(Path(args.output_dir).glob("*.ckpt"))
            if candidates:
                shutil.copy2(str(candidates[-1]), dst)
                final_ckpt = dst

        summary = {
            "n_epochs": args.epochs,
            "lr": args.lr,
            "batch_cost": args.batch_cost,
            "dataset_name": args.dataset_name,
            "base_ckpt": args.ckpt,
            "final_ckpt": final_ckpt,
            "final_loss": round(float(final_loss), 6),
            "elapsed_s": round(time.time() - t_start, 1),
        }
        summary_path = os.path.join(args.output_dir, "finetune_summary.json")
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2)

        _write_progress({
            "type": "completed",
            "ckpt_path": final_ckpt,
            "final_loss": round(float(final_loss), 6),
            "epochs_trained": args.epochs,
        })
        print(f"Fine-tuning complete. Checkpoint: {final_ckpt}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        main()
    except Exception as exc:
        import traceback
        _write_progress({"type": "failed", "error": str(exc)})
        traceback.print_exc()
        sys.exit(1)
