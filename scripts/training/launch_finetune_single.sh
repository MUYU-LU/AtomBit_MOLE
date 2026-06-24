#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

python scripts/training/finetune.py \
  --ckpt "${CKPT:?set CKPT=/path/to/base.ckpt}" \
  --data_dir "${DATA_DIR:?set DATA_DIR=/path/to/train_data}" \
  --dataset_name "${DATASET_NAME:-mpa}" \
  --epochs "${EPOCHS:-1}" \
  --lr "${LR:-1e-4}" \
  --batch_cost "${BATCH_COST:-1000}" \
  --output_dir "${OUTPUT_DIR:-./finetune_out}" \
  --progress_file "${PROGRESS_FILE:-./finetune_out/progress.jsonl}" \
  --num_devices 1 \
  --device_id "${DEVICE_ID:-0}"
