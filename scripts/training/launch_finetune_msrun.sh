#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

WORKER_NUM="${WORKER_NUM:-8}"
LOCAL_WORKER_NUM="${LOCAL_WORKER_NUM:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-18081}"
NODE_RANK="${NODE_RANK:-0}"
LOG_DIR="${LOG_DIR:-msrun_log_finetune}"
OUTPUT_DIR="${OUTPUT_DIR:-./finetune_out}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

export PARALLEL_MODE=DATA_PARALLEL
export PYTHONPATH="$PROJECT_ROOT/sharker:$PROJECT_ROOT:${PYTHONPATH:-}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1200}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-4096}"

msrun --worker_num="$WORKER_NUM" \
      --local_worker_num="$LOCAL_WORKER_NUM" \
      --master_addr="$MASTER_ADDR" \
      --master_port="$MASTER_PORT" \
      --node_rank="$NODE_RANK" \
      --log_dir="$LOG_DIR" \
      --join=True \
      --cluster_time_out="${CLUSTER_TIMEOUT:-1200}" \
      scripts/training/finetune.py \
      --ckpt "${CKPT:?set CKPT=/path/to/base.ckpt}" \
      --data_dir "${DATA_DIR:?set DATA_DIR=/path/to/train_data}" \
      --dataset_name "${DATASET_NAME:-mpa}" \
      --epochs "${EPOCHS:-1}" \
      --lr "${LR:-1e-4}" \
      --batch_cost "${BATCH_COST:-1000}" \
      --num_devices "$WORKER_NUM" \
      --output_dir "$OUTPUT_DIR" \
      --progress_file "${PROGRESS_FILE:-$OUTPUT_DIR/progress.jsonl}"
