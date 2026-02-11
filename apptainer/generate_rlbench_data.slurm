#!/bin/bash
#SBATCH -A kdesingh
#SBATCH -p msigpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH -t 02:00:00
#SBATCH --array=0-16%4
#SBATCH --job-name=ip_data_gen
#SBATCH --output=/scratch.global/%u/ips/logs/ip_data_gen_%A_%a.out
#SBATCH --error=/scratch.global/%u/ips/logs/ip_data_gen_%A_%a.err

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

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INSTANT_POLICY_DIR=${INSTANT_POLICY_DIR:-$REPO_ROOT/instant_policy}
PROJECT_DIR=${PROJECT_DIR:-$REPO_ROOT}
DATA_DIR=${DATA_DIR:-}
RUNNER="$(resolve_runner || true)"

if [ -z "${DATA_DIR:-}" ]; then
  if [ -d "/scratch.global/$USER/ip" ]; then
    DATA_DIR="/scratch.global/$USER/ip"
  else
    DATA_DIR="/scratch.global/$USER/ips"
  fi
fi

EPISODES=${EPISODES:-50}

TASKS=(
  lift_lid
  phone_on_base
  open_box
  slide_block
  close_box
  basketball
  buzz
  close_microwave
  plate_out
  toilet_seat_down
  toilet_seat_up
  toilet_roll_off
  open_microwave
  lamp_on
  umbrella_out
  push_button
  put_rubbish
)

if [ ! -x "$RUNNER" ]; then
  echo "ERROR: runner not found or not executable: ${RUNNER:-<unset>}"
  echo "Tried candidates under:"
  echo "  SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
  echo "  PWD=$PWD"
  echo "  SCRIPT_DIR=$SCRIPT_DIR"
  exit 1
fi

if [ ! -f "$DATA_DIR/checkpoints/scene_encoder.pt" ]; then
  echo "ERROR: scene_encoder.pt not found at $DATA_DIR/checkpoints/scene_encoder.pt"
  exit 1
fi

mkdir -p "$DATA_DIR/logs"

TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

DISPLAY_BASE=${DISPLAY_BASE:-$((500 + (${SLURM_JOB_ID:-0} % 200)))}
VNC_BASE=${VNC_BASE:-$((26000 + (${SLURM_JOB_ID:-0} % 10000)))}
export RLBENCH_DISPLAY=${RLBENCH_DISPLAY:-$((DISPLAY_BASE + ${SLURM_ARRAY_TASK_ID:-0}))}
export RLBENCH_VNC_PORT=${RLBENCH_VNC_PORT:-$((VNC_BASE + ${SLURM_ARRAY_TASK_ID:-0}))}
export PROJECT_DIR INSTANT_POLICY_DIR DATA_DIR

start=$(date +%s)
echo "Starting task $TASK with $EPISODES episodes on $(hostname)"

"$RUNNER" "python /workspace/instant_policy/ip/scripts/generate_rlbench_data.py \
  --task_name $TASK \
  --num_episodes $EPISODES \
  --save_dir /workspace/data/train/$TASK \
  --headless \
  --restrict_rot \
  --compute_embeddings \
  --scene_encoder_path /workspace/data/checkpoints/scene_encoder.pt"

end=$(date +%s)
echo "Finished task $TASK in $((end - start)) seconds"
