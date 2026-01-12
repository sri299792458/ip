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

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  if [ -d "$SLURM_SUBMIT_DIR/apptainer" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR/apptainer"
  else
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
  fi
else
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INSTANT_POLICY_DIR=${INSTANT_POLICY_DIR:-$REPO_ROOT/instant_policy}
PROJECT_DIR=${PROJECT_DIR:-$REPO_ROOT}
DATA_DIR=${DATA_DIR:-/scratch.global/$USER/ips}
RUNNER=${RUNNER:-$SCRIPT_DIR/run_instant_policy_vnc.sh}

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
  echo "ERROR: run_instant_policy_vnc.sh not found at $RUNNER"
  exit 1
fi

if [ ! -f "$DATA_DIR/checkpoints/scene_encoder.pt" ]; then
  echo "ERROR: scene_encoder.pt not found at $DATA_DIR/checkpoints/scene_encoder.pt"
  exit 1
fi

mkdir -p "$DATA_DIR/logs"

TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

export RLBENCH_DISPLAY=$((1 + SLURM_ARRAY_TASK_ID))
export RLBENCH_VNC_PORT=$((5900 + SLURM_ARRAY_TASK_ID))
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
