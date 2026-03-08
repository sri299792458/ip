#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SWEEP_SLURM_SCRIPT="${SWEEP_SLURM_SCRIPT:-$SCRIPT_DIR/sweep_h100_batch.slurm}"
SWEEP_TIME="${SWEEP_TIME:-06:00:00}"

if [ ! -f "$SWEEP_SLURM_SCRIPT" ]; then
  echo "ERROR: sweep slurm script not found: $SWEEP_SLURM_SCRIPT"
  exit 1
fi

SBATCH_EXTRA_ARR=()
if [ -n "${SBATCH_EXTRA_ARGS:-}" ]; then
  read -r -a SBATCH_EXTRA_ARR <<< "${SBATCH_EXTRA_ARGS}"
fi

job_id="$(
  sbatch --parsable \
    --time="$SWEEP_TIME" \
    "${SBATCH_EXTRA_ARR[@]}" \
    "$SWEEP_SLURM_SCRIPT"
)"

echo "submitted sweep job_id=${job_id} script=${SWEEP_SLURM_SCRIPT}"
