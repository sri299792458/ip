#!/usr/bin/env python3
"""Inspect Instant-Policy-style context dumps stored as .npy object arrays."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Row:
    frame_idx: int
    grasp_status_raw: int
    wrist: np.ndarray | None
    hand: np.ndarray | None
    object_xyz: np.ndarray | None
    object_channels: int | None


def _load_object_npy(path: Path) -> list[dict[str, Any]]:
    arr = np.load(path, allow_pickle=True)
    if not isinstance(arr, np.ndarray):
        raise RuntimeError(f"Expected ndarray in {path}, got {type(arr)}")
    obj = arr.tolist()
    if not isinstance(obj, list):
        raise RuntimeError(f"Expected list-like payload in {path}, got {type(obj)}")
    return obj


def _vec3_or_none(x: Any) -> np.ndarray | None:
    if x is None:
        return None
    a = np.asarray(x, dtype=float).reshape(-1)
    if a.size < 3:
        return None
    return a[:3]


def _array2_or_none(x: Any) -> np.ndarray | None:
    if x is None:
        return None
    a = np.asarray(x, dtype=float)
    if a.ndim != 2:
        return None
    return a


def _segments(values: list[int]) -> list[dict[str, int]]:
    if not values:
        return []
    out = []
    start = 0
    for i in range(1, len(values)):
        if values[i] != values[i - 1]:
            out.append({"start": start, "end": i - 1, "value": int(values[i - 1])})
            start = i
    out.append({"start": start, "end": len(values) - 1, "value": int(values[-1])})
    return out


def build_rows(
    pose_records: list[dict[str, Any]], gripper_records: list[dict[str, Any]]
) -> tuple[list[Row], dict[str, Any]]:
    pose_by_frame = {}
    for rec in pose_records:
        if not isinstance(rec, dict) or "frame_idx" not in rec:
            continue
        pose_by_frame[int(rec["frame_idx"])] = rec

    grip_by_frame = {}
    for rec in gripper_records:
        if not isinstance(rec, dict) or "frame_idx" not in rec:
            continue
        g = rec.get("grasp_status")
        if isinstance(g, np.ndarray):
            g = np.asarray(g).reshape(-1)[0]
        grip_by_frame[int(rec["frame_idx"])] = int(g)

    all_frames = sorted(set(pose_by_frame.keys()) | set(grip_by_frame.keys()))
    rows = []
    missing_pose = []
    missing_grip = []
    for frame in all_frames:
        pose = pose_by_frame.get(frame)
        grip = grip_by_frame.get(frame)
        if pose is None:
            missing_pose.append(frame)
            continue
        if grip is None:
            missing_grip.append(frame)
            continue

        hand = _array2_or_none(pose.get("hand_pose"))
        obj = _array2_or_none(pose.get("object_pose"))
        obj_xyz = None
        obj_channels = None
        if obj is not None and obj.shape[1] >= 3:
            obj_xyz = obj[:, :3]
            obj_channels = int(obj.shape[1])

        rows.append(
            Row(
                frame_idx=int(frame),
                grasp_status_raw=int(grip),
                wrist=_vec3_or_none(pose.get("wrist_pose")),
                hand=hand,
                object_xyz=obj_xyz,
                object_channels=obj_channels,
            )
        )

    meta = {
        "num_pose_records": len(pose_records),
        "num_gripper_records": len(gripper_records),
        "num_union_frames": len(all_frames),
        "missing_pose_frames": missing_pose,
        "missing_gripper_frames": missing_grip,
    }
    return rows, meta


def summarize(rows: list[Row], one_means_closed: bool) -> dict[str, Any]:
    frame_idxs = [r.frame_idx for r in rows]
    grasp_raw = [int(r.grasp_status_raw) for r in rows]
    grasp_rlbench = [1 - g if one_means_closed else g for g in grasp_raw]

    wrist_shape_counts = Counter("None" if r.wrist is None else str(np.asarray(r.wrist).shape) for r in rows)
    hand_shape_counts = Counter("None" if r.hand is None else str(r.hand.shape) for r in rows)
    object_shape_counts = Counter("None" if r.object_xyz is None else str(r.object_xyz.shape) for r in rows)
    object_channel_counts = Counter(
        "None" if r.object_channels is None else str(r.object_channels) for r in rows
    )

    wrist_nan_frames = [
        r.frame_idx for r in rows if r.wrist is not None and (not np.isfinite(r.wrist).all())
    ]
    hand_nan_frames = [
        r.frame_idx for r in rows if r.hand is not None and np.isnan(r.hand).any()
    ]
    object_nan_frames = [
        r.frame_idx for r in rows if r.object_xyz is not None and np.isnan(r.object_xyz).any()
    ]

    hand_per_joint_avail = None
    hand_stack = [r.hand for r in rows if r.hand is not None and r.hand.ndim == 2 and r.hand.shape == (21, 3)]
    if hand_stack:
        H = np.stack(hand_stack, axis=0)
        hand_per_joint_avail = np.isfinite(H).all(axis=2).mean(axis=0).tolist()

    # Check whether wrist equals hand joint-0 (common in hand keypoint sets).
    wrist_joint0_distance = None
    d = []
    for r in rows:
        if r.wrist is None or r.hand is None:
            continue
        if r.hand.shape != (21, 3):
            continue
        h0 = r.hand[0]
        if np.isfinite(h0).all() and np.isfinite(r.wrist).all():
            d.append(float(np.linalg.norm(h0 - r.wrist)))
    if d:
        wrist_joint0_distance = {
            "count": len(d),
            "mean": float(np.mean(d)),
            "median": float(np.median(d)),
            "max": float(np.max(d)),
        }

    # Object-to-wrist distances.
    d_rows = []
    for r, g_raw, g_rl in zip(rows, grasp_raw, grasp_rlbench):
        if r.wrist is None or r.object_xyz is None:
            continue
        if not np.isfinite(r.wrist).all():
            continue
        valid = np.isfinite(r.object_xyz).all(axis=1)
        xyz = r.object_xyz[valid]
        if len(xyz) == 0:
            continue
        centroid = xyz.mean(axis=0)
        d_cent = float(np.linalg.norm(centroid - r.wrist))
        d_nn = float(np.min(np.linalg.norm(xyz - r.wrist[None, :], axis=1)))
        d_rows.append((g_raw, g_rl, d_cent, d_nn, len(xyz), r.frame_idx))

    by_state = {}
    for label, idx in [("raw_grasp_status", 0), ("rlbench_grip", 1)]:
        group = {}
        for state in [0, 1]:
            vals = [r for r in d_rows if r[idx] == state]
            if not vals:
                continue
            cent = np.array([v[2] for v in vals], dtype=float)
            nn = np.array([v[3] for v in vals], dtype=float)
            pts = np.array([v[4] for v in vals], dtype=float)
            group[str(state)] = {
                "count": int(len(vals)),
                "centroid_dist_mean": float(np.mean(cent)),
                "nearest_dist_mean": float(np.mean(nn)),
                "point_count_mean": float(np.mean(pts)),
            }
        by_state[label] = group

    transitions = []
    for i in range(1, len(grasp_raw)):
        if grasp_raw[i] != grasp_raw[i - 1]:
            transitions.append(
                {
                    "index": i,
                    "frame_idx": int(frame_idxs[i]),
                    "from": int(grasp_raw[i - 1]),
                    "to": int(grasp_raw[i]),
                }
            )

    return {
        "num_rows": len(rows),
        "frame_range": None if not frame_idxs else [int(min(frame_idxs)), int(max(frame_idxs))],
        "frame_count_unique": len(set(frame_idxs)),
        "wrist_shape_counts": dict(wrist_shape_counts),
        "hand_shape_counts": dict(hand_shape_counts),
        "object_shape_counts_xyz": dict(object_shape_counts),
        "object_channel_counts_raw": dict(object_channel_counts),
        "wrist_nan_frames": wrist_nan_frames,
        "hand_any_nan_frames_count": len(hand_nan_frames),
        "object_any_nan_frames_count": len(object_nan_frames),
        "hand_joint_availability_ratio": hand_per_joint_avail,
        "wrist_minus_hand_joint0_distance": wrist_joint0_distance,
        "grasp_values_raw_unique": sorted(set(grasp_raw)),
        "grasp_values_rlbench_unique": sorted(set(grasp_rlbench)),
        "grasp_segments_raw": _segments(grasp_raw),
        "grasp_segments_rlbench": _segments(grasp_rlbench),
        "grasp_transitions_raw": transitions,
        "distance_stats": by_state,
        "one_means_closed_interpretation": bool(one_means_closed),
    }


def _pretty_print(summary: dict[str, Any], meta: dict[str, Any]) -> None:
    print("=== Data Inventory ===")
    print(f"pose records: {meta['num_pose_records']}")
    print(f"gripper records: {meta['num_gripper_records']}")
    print(f"union frame count: {meta['num_union_frames']}")
    print(f"missing pose frames: {len(meta['missing_pose_frames'])}")
    print(f"missing gripper frames: {len(meta['missing_gripper_frames'])}")
    print("")
    print("=== Core Shapes ===")
    print(f"frame range: {summary['frame_range']}")
    print(f"unique frames: {summary['frame_count_unique']}")
    print(f"wrist shapes: {summary['wrist_shape_counts']}")
    print(f"hand shapes: {summary['hand_shape_counts']}")
    print(f"object xyz shapes: {summary['object_shape_counts_xyz']}")
    print(f"object raw channel counts: {summary['object_channel_counts_raw']}")
    print("")
    print("=== Missing / NaNs ===")
    print(f"wrist nan frames: {summary['wrist_nan_frames']}")
    print(f"hand frames with any NaN: {summary['hand_any_nan_frames_count']}")
    print(f"object frames with any NaN: {summary['object_any_nan_frames_count']}")
    if summary["wrist_minus_hand_joint0_distance"] is not None:
        d = summary["wrist_minus_hand_joint0_distance"]
        print(
            "wrist vs hand[0] distance: "
            f"mean={d['mean']:.6f}, median={d['median']:.6f}, max={d['max']:.6f}, n={d['count']}"
        )
    if summary["hand_joint_availability_ratio"] is not None:
        ratios = np.array(summary["hand_joint_availability_ratio"], dtype=float)
        print(f"hand joint availability min/mean/max: {ratios.min():.3f}/{ratios.mean():.3f}/{ratios.max():.3f}")
    print("")
    print("=== Gripper Timeline ===")
    print(f"raw unique values: {summary['grasp_values_raw_unique']}")
    print(f"rlbench interpreted unique values: {summary['grasp_values_rlbench_unique']}")
    print(f"raw segments: {summary['grasp_segments_raw']}")
    print(f"raw transitions: {summary['grasp_transitions_raw']}")
    print("")
    print("=== Object-Wrist Distance Stats ===")
    print(json.dumps(summary["distance_stats"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect out_poses_in_base.npy + out_gripper_status.npy")
    parser.add_argument(
        "--poses",
        type=Path,
        default=Path("rpm_apple/out_poses_in_base.npy"),
        help="Path to pose records .npy",
    )
    parser.add_argument(
        "--gripper",
        type=Path,
        default=Path("rpm_apple/out_gripper_status.npy"),
        help="Path to gripper status .npy",
    )
    parser.add_argument(
        "--one-means-closed",
        action="store_true",
        help="Interpret raw grasp_status==1 as closed and convert to RLBench grip semantics (1=open).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON summary (otherwise pretty text).",
    )
    args = parser.parse_args()

    pose_records = _load_object_npy(args.poses)
    gripper_records = _load_object_npy(args.gripper)
    rows, meta = build_rows(pose_records, gripper_records)
    summary = summarize(rows, one_means_closed=args.one_means_closed)

    payload = {"meta": meta, "summary": summary}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _pretty_print(summary, meta)


if __name__ == "__main__":
    main()
