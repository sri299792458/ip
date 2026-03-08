#!/bin/bash

# Sequential H100 probe sweep:
# - one Slurm allocation
# - one H100
# - run each batch-size probe back-to-back

#SBATCH -A kdesingh
#SBATCH -p msigpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G
#SBATCH --tmp=300G
#SBATCH -t 06:00:00
#SBATCH --job-name=ip_h100_sweep
#SBATCH --output=/scratch.global/%u/ips/logs/ip_h100_sweep_%j.out
#SBATCH --error=/scratch.global/%u/ips/logs/ip_h100_sweep_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROBE_SCRIPT="${PROBE_SCRIPT:-$SCRIPT_DIR/train_h100_streaming.slurm}"

BATCH_SIZES=(${BATCH_SIZES:-16 24 32 48})
PROBE_STEPS="${PROBE_STEPS:-2000}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-ip_probe}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-16}"

if [ ! -f "$PROBE_SCRIPT" ]; then
  echo "ERROR: probe script not found: $PROBE_SCRIPT"
  exit 1
fi

echo "Sequential H100 probe sweep"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "batch_sizes=${BATCH_SIZES[*]}"
echo "probe_steps=$PROBE_STEPS train_num_workers=$TRAIN_NUM_WORKERS"

for batch_size in "${BATCH_SIZES[@]}"; do
  run_name="${RUN_NAME_PREFIX}_b${batch_size}"
  train_buffer_size=$((256 * batch_size))
  save_every=$((PROBE_STEPS + 1))

  echo
  echo "=== probe run_name=$run_name batch_size=$batch_size ==="
  echo "buffer_size=$train_buffer_size save_every=$save_every"

  RUN_NAME="$run_name" \
  BATCH_SIZE="$batch_size" \
  TRAIN_NUM_WORKERS="$TRAIN_NUM_WORKERS" \
  TRAIN_BUFFER_SIZE="$train_buffer_size" \
  SAVE_EVERY="$save_every" \
  NUM_ITERS_OVERRIDE="$PROBE_STEPS" \
  bash "$PROBE_SCRIPT"
done
