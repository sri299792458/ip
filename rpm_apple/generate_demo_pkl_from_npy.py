#!/usr/bin/env python3
"""Generate an Instant-Policy demo.pkl from rpm_apple .npy pose/status dumps.

Assumptions for this MVP converter:
- Wrist translation is used as proxy for end-effector translation.
- End-effector orientation is fixed top-down (constant rotation).
- Gripper labels are converted to RLBench convention (1=open, 0=closed).
- Optional HAT->BASE transform is applied to wrist/object points.
"""

from __future__ import annotations

import argparse
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def hat_to_base_matrix() -> np.ndarray:
    theta = np.deg2rad(-135.0)
    E = np.eye(4, dtype=np.float64)
    E[1, 1], E[1, 2] = np.cos(theta), -np.sin(theta)
    E[2, 1], E[2, 2] = np.sin(theta), np.cos(theta)
    E[2, 3] = 1.0
    return E


def resolve_hat_transform(mode: str, matrix_txt_path: Path | None = None) -> np.ndarray:
    if matrix_txt_path is not None:
        mat = np.asarray(np.loadtxt(matrix_txt_path, dtype=np.float64), dtype=np.float64)
        if mat.shape != (4, 4):
            raise ValueError(
                f"--hat-to-base-matrix-txt must be a 4x4 numeric matrix, got shape {mat.shape} from {matrix_txt_path}"
            )
        return mat

    mode = str(mode).strip().lower()
    if mode == "forward":
        return hat_to_base_matrix()
    if mode == "inverse":
        return np.linalg.inv(hat_to_base_matrix())
    if mode == "none":
        return np.eye(4, dtype=np.float64)
    raise ValueError(f"Unsupported --hat-to-base-mode={mode!r}. Expected one of: none, forward, inverse.")


def top_down_rotation() -> np.ndarray:
    # R = Rx(pi): tool z axis points down in base frame.
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )


def load_records(path: Path) -> list[dict[str, Any]]:
    arr = np.load(path, allow_pickle=True)
    recs = arr.tolist()
    if not isinstance(recs, list):
        raise RuntimeError(f"Expected list-like records in {path}, got {type(recs)}")
    return recs


def transform_points(T: np.ndarray, pts_xyz: np.ndarray) -> np.ndarray:
    out = np.empty_like(pts_xyz, dtype=np.float64)
    out[:, :] = (T[:3, :3] @ pts_xyz.T).T + T[:3, 3]
    return out


def transform_point(T: np.ndarray, p_xyz: np.ndarray) -> np.ndarray:
    return (T[:3, :3] @ p_xyz.reshape(3)) + T[:3, 3]


def build_demo(
    pose_records: list[dict[str, Any]],
    grip_records: list[dict[str, Any]],
    hat_to_base_mode: str,
    hat_to_base_matrix_txt: Path | None,
    raw_one_means_closed: bool,
    top_down_z_offset_m: float,
) -> dict[str, Any]:
    status_by_frame: dict[int, int] = {}
    for rec in grip_records:
        if not isinstance(rec, dict) or "frame_idx" not in rec:
            continue
        g = rec.get("grasp_status")
        if isinstance(g, np.ndarray):
            g = np.asarray(g).reshape(-1)[0]
        status_by_frame[int(rec["frame_idx"])] = int(g)

    T_hat_base = resolve_hat_transform(hat_to_base_mode, matrix_txt_path=hat_to_base_matrix_txt)
    R_top = top_down_rotation()

    pcds = []
    T_w_es = []
    grips = []
    kept_frame_idx = []
    dropped = {
        "missing_status": 0,
        "missing_wrist": 0,
        "nonfinite_wrist": 0,
        "missing_object": 0,
        "bad_object_shape": 0,
        "empty_object_after_filter": 0,
        "nonbinary_grasp": 0,
    }

    for rec in sorted((r for r in pose_records if isinstance(r, dict)), key=lambda r: int(r.get("frame_idx", -1))):
        frame_idx = int(rec.get("frame_idx", -1))
        if frame_idx < 0:
            continue
        if frame_idx not in status_by_frame:
            dropped["missing_status"] += 1
            continue

        raw_g = status_by_frame[frame_idx]
        if raw_g not in (0, 1):
            dropped["nonbinary_grasp"] += 1
            continue
        grip_rlbench = 1 - raw_g if raw_one_means_closed else raw_g

        wrist_raw = rec.get("wrist_pose")
        if wrist_raw is None:
            dropped["missing_wrist"] += 1
            continue
        wrist = np.asarray(wrist_raw, dtype=np.float64).reshape(-1)
        if wrist.size < 3 or not np.isfinite(wrist[:3]).all():
            dropped["nonfinite_wrist"] += 1
            continue
        wrist = wrist[:3]

        obj_raw = rec.get("object_pose")
        if obj_raw is None:
            dropped["missing_object"] += 1
            continue
        obj = np.asarray(obj_raw, dtype=np.float64)
        if obj.ndim != 2 or obj.shape[1] < 3:
            dropped["bad_object_shape"] += 1
            continue
        obj_xyz = obj[:, :3]
        finite_rows = np.isfinite(obj_xyz).all(axis=1)
        obj_xyz = obj_xyz[finite_rows]
        if len(obj_xyz) == 0:
            dropped["empty_object_after_filter"] += 1
            continue

        wrist_base = transform_point(T_hat_base, wrist)
        obj_base = transform_points(T_hat_base, obj_xyz)
        wrist_base[2] += float(top_down_z_offset_m)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_top
        T[:3, 3] = wrist_base

        pcds.append(obj_base.astype(np.float32))
        T_w_es.append(T.astype(np.float32))
        grips.append(float(grip_rlbench))
        kept_frame_idx.append(frame_idx)

    if len(pcds) < 10:
        raise RuntimeError(
            f"Only {len(pcds)} valid frames after filtering; need at least 10 for Instant Policy context."
        )

    demo = {
        "pcds": pcds,
        "T_w_es": T_w_es,
        "grips": grips,
        "frame_spec": {
            "robot_tcp_frame": "flange",
            "flange_to_policy_origin_m": [0.0, 0.0, 0.088],
        },
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_meta": {
            "source_kind": "npy_conversion_proxy",
            "assumptions": {
                "top_down_fixed_orientation": True,
                "raw_one_means_closed": bool(raw_one_means_closed),
                "hat_to_base_mode": str(hat_to_base_mode),
                "hat_to_base_matrix_txt": None if hat_to_base_matrix_txt is None else str(hat_to_base_matrix_txt),
                "top_down_z_offset_m": float(top_down_z_offset_m),
            },
            "kept_frame_idx": kept_frame_idx,
            "dropped_counts": dropped,
            "num_frames_kept": int(len(kept_frame_idx)),
        },
    }
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo.pkl from out_poses_in_base.npy + out_gripper_status.npy")
    parser.add_argument(
        "--poses",
        type=Path,
        default=Path("/home/srinivas/Desktop/ip/rpm_apple/out_poses_in_base.npy"),
        help="Path to out_poses_in_base.npy",
    )
    parser.add_argument(
        "--gripper",
        type=Path,
        default=Path("/home/srinivas/Desktop/ip/rpm_apple/out_gripper_status.npy"),
        help="Path to out_gripper_status.npy",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/srinivas/Desktop/ip/rpm_apple/demo.pkl"),
        help="Output demo.pkl path",
    )
    parser.add_argument(
        "--hat-to-base-mode",
        choices=["none", "forward", "inverse"],
        default="forward",
        help=(
            "How to apply HAT_TO_BASE to wrist/object points: "
            "'forward'=HAT->BASE, 'inverse'=BASE->HAT inverse, 'none'=identity."
        ),
    )
    parser.add_argument(
        "--hat-to-base-matrix-txt",
        type=Path,
        default=None,
        help=(
            "Optional path to custom 4x4 transform matrix (plain text, whitespace separated). "
            "If set, this overrides --hat-to-base-mode."
        ),
    )
    parser.add_argument(
        "--raw-one-means-open",
        action="store_true",
        help="Treat raw grasp_status=1 as OPEN (skip inversion to RLBench convention).",
    )
    parser.add_argument(
        "--top-down-z-offset-m",
        type=float,
        default=0.0,
        help="Extra offset added to wrist z (meters) before writing T_w_e translation.",
    )
    args = parser.parse_args()

    poses = load_records(args.poses)
    gripper = load_records(args.gripper)
    demo = build_demo(
        pose_records=poses,
        grip_records=gripper,
        hat_to_base_mode=str(args.hat_to_base_mode),
        hat_to_base_matrix_txt=args.hat_to_base_matrix_txt,
        raw_one_means_closed=not args.raw_one_means_open,
        top_down_z_offset_m=float(args.top_down_z_offset_m),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump(demo, f)

    print(f"Saved demo to: {args.out}")
    print(f"frames kept: {demo['source_meta']['num_frames_kept']}")
    print(f"first/last kept frame: {demo['source_meta']['kept_frame_idx'][0]} / {demo['source_meta']['kept_frame_idx'][-1]}")
    print(f"dropped: {demo['source_meta']['dropped_counts']}")


if __name__ == "__main__":
    main()

