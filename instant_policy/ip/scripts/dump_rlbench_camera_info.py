import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig

from ip.utils.rl_bench_tasks import TASK_NAMES


CAMERAS = ("left_shoulder", "right_shoulder", "front", "wrist", "overhead")


def _to_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _to_list(value: Any) -> list:
    return _to_array(value).tolist()


def _camera_payload(intrinsics: Any, extrinsics: Any, near: Any, far: Any) -> Dict[str, Any]:
    extr = _to_array(extrinsics)
    return {
        "intrinsics": _to_list(intrinsics),
        "extrinsics": _to_list(extr),
        "near_plane": float(near),
        "far_plane": float(far),
        "position_world_m": _to_list(extr[:3, 3]),
    }


def _build_env(headless: bool) -> Environment:
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaIK(),
        gripper_action_mode=Discrete(),
    )
    return Environment(
        action_mode=action_mode,
        dataset_root="./",
        obs_config=obs_config,
        headless=headless,
    )


def _collect_scene_data(env: Environment) -> Dict[str, Any]:
    scene = env.get_scene_data()
    out: Dict[str, Any] = {}
    for cam in CAMERAS:
        key = f"{cam}_camera"
        entry = scene.get(key)
        if entry is None:
            continue
        out[cam] = _camera_payload(
            intrinsics=entry["intrinsics"],
            extrinsics=entry["extrinsics"],
            near=entry["near_plane"],
            far=entry["far_plane"],
        )
    return out


def _collect_obs_misc_data(env: Environment, task_name: str) -> Dict[str, Any]:
    if task_name not in TASK_NAMES:
        raise ValueError(f"Unknown task_name '{task_name}'. Choices: {sorted(TASK_NAMES.keys())}")

    env.launch()
    try:
        task = env.get_task(TASK_NAMES[task_name])
        _, obs = task.reset()
        out: Dict[str, Any] = {}
        for cam in CAMERAS:
            out[cam] = _camera_payload(
                intrinsics=obs.misc[f"{cam}_camera_intrinsics"],
                extrinsics=obs.misc[f"{cam}_camera_extrinsics"],
                near=obs.misc[f"{cam}_camera_near"],
                far=obs.misc[f"{cam}_camera_far"],
            )
        return out
    finally:
        env.shutdown()


def _compute_diffs(scene_data: Dict[str, Any], obs_data: Dict[str, Any]) -> Dict[str, Any]:
    diffs: Dict[str, Any] = {}
    for cam in CAMERAS:
        if cam not in scene_data or cam not in obs_data:
            continue
        s = scene_data[cam]
        o = obs_data[cam]
        diffs[cam] = {
            "intrinsics_max_abs": float(np.max(np.abs(_to_array(s["intrinsics"]) - _to_array(o["intrinsics"])))),
            "extrinsics_max_abs": float(np.max(np.abs(_to_array(s["extrinsics"]) - _to_array(o["extrinsics"])))),
            "near_abs": abs(float(s["near_plane"]) - float(o["near_plane"])),
            "far_abs": abs(float(s["far_plane"]) - float(o["far_plane"])),
        }
    return diffs


def _print_summary(scene_data: Dict[str, Any], obs_data: Dict[str, Any], diffs: Dict[str, Any]) -> None:
    print("RLBench camera summary:")
    for cam in CAMERAS:
        if cam not in scene_data:
            continue
        s = scene_data[cam]
        pos = s["position_world_m"]
        print(
            f"- {cam:>14}: near={s['near_plane']:.4f} far={s['far_plane']:.4f} "
            f"pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})"
        )
        if cam in diffs:
            d = diffs[cam]
            print(
                f"  reset-diff intr={d['intrinsics_max_abs']:.3e} "
                f"extr={d['extrinsics_max_abs']:.3e} "
                f"near={d['near_abs']:.3e} far={d['far_abs']:.3e}"
            )
        elif obs_data:
            print("  reset-diff unavailable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump RLBench camera intrinsics/extrinsics at runtime.")
    parser.add_argument(
        "--task_name",
        type=str,
        default="lift_lid",
        help="Task used for reset-time obs.misc camera extraction.",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run RLBench headless (default: true).",
    )
    parser.add_argument(
        "--skip_obs_reset",
        action="store_true",
        help="Only dump env.get_scene_data() and skip task reset.",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default="",
        help="Optional output JSON path.",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation.")
    args = parser.parse_args()

    env = _build_env(headless=args.headless)
    scene_data = _collect_scene_data(env)
    obs_data: Dict[str, Any] = {}
    if not args.skip_obs_reset:
        obs_data = _collect_obs_misc_data(env, args.task_name)

    diffs = _compute_diffs(scene_data, obs_data) if obs_data else {}
    payload = {
        "task_name": args.task_name,
        "headless": args.headless,
        "scene_data": scene_data,
        "obs_reset_data": obs_data,
        "scene_vs_obs_diffs": diffs,
    }

    _print_summary(scene_data, obs_data, diffs)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=args.indent), encoding="utf-8")
        print(f"\nWrote: {out_path}")
    else:
        print("\nJSON dump:")
        print(json.dumps(payload, indent=args.indent))


if __name__ == "__main__":
    main()
