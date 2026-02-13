#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Generate bimanual pseudo-debug videos by primitive family (skill-like view).

This script loops over the 7 bimanual primitive families and, for each one,
forces one representative RLBench2 task in the existing bimanual generator.

Required:
  --shapenet_path PATH
  --gripper_mesh_path PATH

Optional:
  --save_root DIR                (default: ./pseudo_bimanual_debug_by_skill)
  --shapenet_index_path PATH
  --num_samples_per_skill N      (default: 8)
  --seed N                       (default: 0; incremented per skill)
  --pred_horizon N               (default: 8)
  --num_points N                 (default: 2048)
  --pcd_storage_dtype DTYPE      (default: float32; choices: float32,float16)
  --render_visual_width N        (default: 800)
  --render_visual_height N       (default: 800)
  --render_video_fps N           (default: 15)

Example:
  bash instant_policy/ip/scripts/generate_bimanual_pseudo_debug_by_skill.sh \
    --shapenet_path /scratch.global/$USER/ips/shapenet \
    --gripper_mesh_path /scratch.global/$USER/ips/assets/robotiq_2f85_collision_open.obj \
    --save_root /scratch.global/$USER/ips/pseudo_bimanual_debug
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PKG_ROOT}:${PYTHONPATH:-}"

SHAPENET_PATH=""
GRIPPER_MESH_PATH=""
SAVE_ROOT="./pseudo_bimanual_debug_by_skill"
SHAPENET_INDEX_PATH=""
NUM_SAMPLES_PER_SKILL=8
SEED=0
PRED_HORIZON=8
NUM_POINTS=2048
PCD_STORAGE_DTYPE="float32"
RENDER_VISUAL_WIDTH=800
RENDER_VISUAL_HEIGHT=800
RENDER_VIDEO_FPS=15

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shapenet_path)
      SHAPENET_PATH="$2"; shift 2 ;;
    --gripper_mesh_path)
      GRIPPER_MESH_PATH="$2"; shift 2 ;;
    --save_root)
      SAVE_ROOT="$2"; shift 2 ;;
    --shapenet_index_path)
      SHAPENET_INDEX_PATH="$2"; shift 2 ;;
    --num_samples_per_skill)
      NUM_SAMPLES_PER_SKILL="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --pred_horizon)
      PRED_HORIZON="$2"; shift 2 ;;
    --num_points)
      NUM_POINTS="$2"; shift 2 ;;
    --pcd_storage_dtype)
      PCD_STORAGE_DTYPE="$2"; shift 2 ;;
    --render_visual_width)
      RENDER_VISUAL_WIDTH="$2"; shift 2 ;;
    --render_visual_height)
      RENDER_VISUAL_HEIGHT="$2"; shift 2 ;;
    --render_video_fps)
      RENDER_VIDEO_FPS="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "${SHAPENET_PATH}" || -z "${GRIPPER_MESH_PATH}" ]]; then
  echo "ERROR: --shapenet_path and --gripper_mesh_path are required." >&2
  usage
  exit 2
fi

if [[ "${PCD_STORAGE_DTYPE}" != "float32" && "${PCD_STORAGE_DTYPE}" != "float16" ]]; then
  echo "ERROR: --pcd_storage_dtype must be float32 or float16." >&2
  exit 2
fi

mkdir -p "${SAVE_ROOT}"

# Representative task per primitive family.
declare -A TASK_BY_PRIMITIVE=(
  [cooperative_lift]="bimanual_lift_ball"
  [dual_push_sync]="bimanual_dual_push_buttons"
  [dual_push_transport]="bimanual_push_box"
  [container_open_place_remove]="bimanual_put_item_in_drawer"
  [handover]="bimanual_handover_item"
  [two_endpoint_tension]="bimanual_straighten_rope"
  [tool_plus_receptacle]="bimanual_sweep_to_dustpan"
)

PRIMITIVE_ORDER=(
  cooperative_lift
  dual_push_sync
  dual_push_transport
  container_open_place_remove
  handover
  two_endpoint_tension
  tool_plus_receptacle
)

echo "[by_skill] save_root=${SAVE_ROOT}"
echo "[by_skill] num_samples_per_skill=${NUM_SAMPLES_PER_SKILL}"

current_seed="${SEED}"
for primitive in "${PRIMITIVE_ORDER[@]}"; do
  forced_task="${TASK_BY_PRIMITIVE[$primitive]}"
  save_dir="${SAVE_ROOT}/${primitive}/tasks"
  video_dir="${SAVE_ROOT}/${primitive}/videos"
  mkdir -p "${save_dir}" "${video_dir}"

  cmd=(
    python -m ip.scripts.generate_bimanual_pseudo_demos
    --shapenet_path "${SHAPENET_PATH}"
    --gripper_mesh_path "${GRIPPER_MESH_PATH}"
    --save_dir "${save_dir}"
    --num_samples "${NUM_SAMPLES_PER_SKILL}"
    --seed "${current_seed}"
    --pred_horizon "${PRED_HORIZON}"
    --num_points "${NUM_POINTS}"
    --pcd_storage_dtype "${PCD_STORAGE_DTYPE}"
    --forced_task "${forced_task}"
    --render_make_videos
    --render_video_dir "${video_dir}"
    --render_video_fps "${RENDER_VIDEO_FPS}"
    --render_visual_width "${RENDER_VISUAL_WIDTH}"
    --render_visual_height "${RENDER_VISUAL_HEIGHT}"
  )

  if [[ -n "${SHAPENET_INDEX_PATH}" ]]; then
    cmd+=(--shapenet_index_path "${SHAPENET_INDEX_PATH}")
  fi

  echo "[by_skill] primitive=${primitive} forced_task=${forced_task} seed=${current_seed}"
  "${cmd[@]}"
  current_seed=$((current_seed + 1))
done

echo "[by_skill] done. outputs under: ${SAVE_ROOT}"
