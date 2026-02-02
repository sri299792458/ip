#!/usr/bin/env python3
"""
Compute T_world_tag from three measured ArUco corner points (TL, TR, BL).
Inputs can be either:
  - 3 values: x y z (assumed to already be gripper-tip positions), or
  - 6 values: x y z rx ry rz (flange pose; tcp offset applied by default).

Tag frame convention matches OpenCV ArUco:
  - Origin at tag center
  - X axis: left -> right (TL -> TR)
  - Y axis: bottom -> top (BL -> TL)
  - Z axis: right-handed (X x Y), out of tag plane
"""
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _parse_vec(values: Tuple[float, ...], name: str) -> np.ndarray:
    if len(values) not in {3, 6}:
        raise ValueError(f"{name} must have 3 or 6 values")
    return np.array(values, dtype=np.float64)


def _normalize(v: np.ndarray, name: str) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= 0:
        raise ValueError(f"{name} has zero length")
    return v / n


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta <= 0:
        return np.eye(3, dtype=np.float64)
    k = rotvec / theta
    kx, ky, kz = k
    K = np.array(
        [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _to_tip_position(values: np.ndarray, tcp_offset_m: Optional[np.ndarray]) -> np.ndarray:
    if values.size == 3:
        return values
    pos = values[:3]
    rotvec = values[3:]
    if tcp_offset_m is None:
        return pos
    R = _rotvec_to_matrix(rotvec)
    return pos + (R @ tcp_offset_m)


def compute_T_world_tag(tl: np.ndarray, tr: np.ndarray, bl: np.ndarray) -> np.ndarray:
    x_raw = tr - tl
    y_raw = tl - bl
    x_axis = _normalize(x_raw, "x_axis")
    y_axis = _normalize(y_raw, "y_axis")
    z_axis = _normalize(np.cross(x_axis, y_axis), "z_axis")
    # Re-orthogonalize y to ensure a proper rotation matrix.
    y_axis = np.cross(z_axis, x_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    center = (tr + bl) * 0.5

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = center
    return T


def main():
    parser = argparse.ArgumentParser(description="Compute T_world_tag from three ArUco corners.")
    parser.add_argument(
        "--tl",
        type=float,
        nargs="+",
        required=True,
        metavar="VAL",
        help="Top-left corner: 3 values (x y z) or 6 values (x y z rx ry rz).",
    )
    parser.add_argument(
        "--tr",
        type=float,
        nargs="+",
        required=True,
        metavar="VAL",
        help="Top-right corner: 3 values (x y z) or 6 values (x y z rx ry rz).",
    )
    parser.add_argument(
        "--bl",
        type=float,
        nargs="+",
        required=True,
        metavar="VAL",
        help="Bottom-left corner: 3 values (x y z) or 6 values (x y z rx ry rz).",
    )
    parser.add_argument(
        "--tcp-offset-m",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.162),
        metavar=("X", "Y", "Z"),
        help="TCP offset in meters (applied when 6D poses are provided).",
    )
    parser.add_argument(
        "--no-tcp-offset",
        action="store_true",
        help="Disable TCP offset even if 6D poses are provided.",
    )
    parser.add_argument(
        "--tag-size",
        type=float,
        default=None,
        help="Optional: tag edge length (meters) for sanity check.",
    )
    parser.add_argument(
        "--warn-threshold",
        type=float,
        default=0.005,
        help="Warn if measured edge differs from tag size by more than this (meters).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent / "world_tag.json"),
        help="Output JSON path for T_world_tag.",
    )
    parser.add_argument(
        "--arm",
        choices=["left", "right"],
        default=None,
        help="Convenience suffix for output file name (e.g. world_tag_left.json).",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Only print the matrix; do not write a file.",
    )
    args = parser.parse_args()

    tl_raw = _parse_vec(tuple(args.tl), "tl")
    tr_raw = _parse_vec(tuple(args.tr), "tr")
    bl_raw = _parse_vec(tuple(args.bl), "bl")
    if not (tl_raw.size == tr_raw.size == bl_raw.size):
        raise ValueError("tl/tr/bl must all have the same number of values (3 or 6).")

    tcp_offset = None if args.no_tcp_offset else np.array(args.tcp_offset_m, dtype=np.float64)
    if tl_raw.size == 6 and tcp_offset is not None:
        print(f"Applying TCP offset (m): {tcp_offset.tolist()}")
    if tl_raw.size == 3:
        print("Interpreting inputs as gripper-tip positions (no TCP offset applied).")
    else:
        print("Interpreting inputs as flange poses (x y z rx ry rz).")
    tl = _to_tip_position(tl_raw, tcp_offset)
    tr = _to_tip_position(tr_raw, tcp_offset)
    bl = _to_tip_position(bl_raw, tcp_offset)

    T = compute_T_world_tag(tl, tr, bl)

    edge_x = float(np.linalg.norm(tr - tl))
    edge_y = float(np.linalg.norm(bl - tl))
    print(f"Measured edge lengths: x={edge_x:.6f} m, y={edge_y:.6f} m")
    if args.tag_size is not None:
        err_x = abs(edge_x - args.tag_size)
        err_y = abs(edge_y - args.tag_size)
        if err_x > args.warn_threshold or err_y > args.warn_threshold:
            print(
                f"WARNING: Edge mismatch vs tag_size={args.tag_size:.6f} m "
                f"(dx={err_x:.6f}, dy={err_y:.6f})."
            )

    print("T_world_tag:")
    for row in T:
        print(row.tolist())

    if args.print_only:
        return

    if args.arm and args.out == str(Path(__file__).resolve().parent / "world_tag.json"):
        out_path = Path(__file__).resolve().parent / f"world_tag_{args.arm}.json"
    else:
        out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(T.tolist(), f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
