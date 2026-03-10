#!/bin/bash

# Rebuild the Instant Policy Apptainer image cleanly on MSI.
# Submit from: ~/ip/apptainer

#SBATCH -A kdesingh
#SBATCH --job-name=ip_build
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH --output=ip_build_%j.out
#SBATCH --error=ip_build_%j.err

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: ${BASH_SOURCE[0]}:${LINENO}: ${BASH_COMMAND}" >&2; exit "$rc"' ERR

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd -P)}"
BUILD_SCRIPT="${BUILD_SCRIPT:-$SUBMIT_DIR/build_instant_policy.sh}"

if [ ! -f "$BUILD_SCRIPT" ]; then
  echo "ERROR: build script not found: $BUILD_SCRIPT" >&2
  exit 1
fi

echo "Build job starting on $(hostname)"
echo "submit_dir=$SUBMIT_DIR"
echo "build_script=$BUILD_SCRIPT"
echo "output_dir=${OUTPUT_DIR:-/scratch.global/$USER/ips}"

bash "$BUILD_SCRIPT"
