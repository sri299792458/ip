#!/bin/bash
# Trainer-only H100 probe with precomputed scene embeddings in the prefill dataset.

#SBATCH -A kdesingh
#SBATCH -p msigpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G
#SBATCH --tmp=300G
#SBATCH -t 02:00:00
#SBATCH --job-name=ip_h100_train_only_emb
#SBATCH --output=ip_h100_train_only_emb_%j.out
#SBATCH --error=ip_h100_train_only_emb_%j.err

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: ${BASH_SOURCE[0]}:${LINENO}: ${BASH_COMMAND}" >&2; exit "$rc"' ERR

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd -P)}"
PROBE_SCRIPT="${PROBE_SCRIPT:-$SUBMIT_DIR/probe_h100_train_only.slurm}"

if [ ! -f "$PROBE_SCRIPT" ]; then
  echo "ERROR: probe script not found: $PROBE_SCRIPT" >&2
  exit 1
fi

export COMPUTE_EMBEDDINGS=1
export RUN_NAME="${RUN_NAME:-ip_train_only_emb_b${BATCH_SIZE:-16}_${SLURM_JOB_ID:-manual}}"

bash "$PROBE_SCRIPT"
