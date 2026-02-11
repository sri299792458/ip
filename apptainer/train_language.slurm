#!/bin/bash
#SBATCH -A kdesingh
#SBATCH -p msigpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH -t 06:00:00
#SBATCH --job-name=ip_lang_train
#SBATCH --output=/scratch.global/%u/ips/logs/ip_lang_train_%j.out
#SBATCH --error=/scratch.global/%u/ips/logs/ip_lang_train_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
resolve_runner() {
  if [ -n "${RUNNER:-}" ] && [ -x "$RUNNER" ]; then
    printf "%s\n" "$RUNNER"
    return 0
  fi

  local candidates=()
  if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    candidates+=("$SLURM_SUBMIT_DIR/apptainer/run_instant_policy.sh")
    candidates+=("$SLURM_SUBMIT_DIR/run_instant_policy.sh")
  fi
  candidates+=("$PWD/apptainer/run_instant_policy.sh")
  candidates+=("$PWD/run_instant_policy.sh")
  candidates+=("$SCRIPT_DIR/run_instant_policy.sh")

  local c
  for c in "${candidates[@]}"; do
    if [ -x "$c" ]; then
      printf "%s\n" "$c"
      return 0
    fi
  done
  return 1
}

RUNNER="$(resolve_runner || true)"
DATA_DIR=${DATA_DIR:-}
if [ -z "${DATA_DIR:-}" ]; then
  if [ -d "/scratch.global/$USER/ip" ]; then
    DATA_DIR="/scratch.global/$USER/ip"
  else
    DATA_DIR="/scratch.global/$USER/ips"
  fi
fi

MAX_STEPS=${MAX_STEPS:-50000}
BATCH_SIZE=${BATCH_SIZE:-16}
LOG_EVERY=${LOG_EVERY:-200}
SAVE_EVERY=${SAVE_EVERY:-5000}
USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-Instant Policy}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-lang_train}

mkdir -p "$DATA_DIR/logs"

if [ ! -x "$RUNNER" ]; then
  echo "ERROR: runner not found or not executable: ${RUNNER:-<unset>}"
  echo "Tried candidates under:"
  echo "  SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
  echo "  PWD=$PWD"
  echo "  SCRIPT_DIR=$SCRIPT_DIR"
  exit 1
fi

WANDB_ARGS=""
if [ "$USE_WANDB" -eq 1 ]; then
  WANDB_ARGS="--use_wandb 1 --wandb_project \"$WANDB_PROJECT\" --wandb_run_name \"$WANDB_RUN_NAME\""
  if [ -n "$WANDB_ENTITY" ]; then
    WANDB_ARGS="$WANDB_ARGS --wandb_entity \"$WANDB_ENTITY\""
  fi
fi

"$RUNNER" "python /workspace/instant_policy/ip/train_language.py \
  --model_path /workspace/data/checkpoints \
  --data_path_train /workspace/data/train_all \
  --batch_size $BATCH_SIZE \
  --max_steps $MAX_STEPS \
  --log_every $LOG_EVERY \
  --save_every $SAVE_EVERY \
  --save_dir /workspace/data/runs_lang \
  --device cuda \
  $WANDB_ARGS"
