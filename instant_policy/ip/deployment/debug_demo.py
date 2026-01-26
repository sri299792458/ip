#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np


def _parse_indices(value: Optional[str], total: int) -> List[int]:
    if not value:
        if total == 0:
            return []
        return sorted(set([0, total // 2, total - 1]))
    indices = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            start_i = int(start)
            end_i = int(end)
            indices.extend(list(range(start_i, end_i + 1)))
        else:
            indices.append(int(part))
    indices = [i for i in indices if 0 <= i < total]
    return sorted(set(indices))


def _write_ply(path: Path, points: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for x, y, z in points:
            f.write(f"{x} {y} {z}\n")


def _stats_summary(pcds: Sequence[np.ndarray]) -> None:
    sizes = [len(p) for p in pcds if p is not None and len(p) > 0]
    if not sizes:
        print("pcd points: none")
        return
    print("pcd points min/mean/max:", min(sizes), sum(sizes) / len(sizes), max(sizes))


def _pose_sanity(T_w_es: Sequence[np.ndarray]) -> None:
    if not T_w_es:
        return
    T = np.asarray(T_w_es[0])
    det = np.linalg.det(T[:3, :3])
    ortho = np.linalg.norm(T[:3, :3].T @ T[:3, :3] - np.eye(3))
    print("pose[0] det:", det, "ortho_err:", ortho)


def _compute_bounds(points: np.ndarray) -> None:
    print("pcd bounds min:", points.min(axis=0), "max:", points.max(axis=0))
    for i, axis in enumerate("xyz"):
        q = np.quantile(points[:, i], [0.01, 0.05, 0.5, 0.95, 0.99])
        print(f"{axis} quantiles 1/5/50/95/99%:", q)


def _check_workspace(points: np.ndarray, ws_min: np.ndarray, ws_max: np.ndarray) -> None:
    outside = ((points < ws_min) | (points > ws_max)).any(axis=1)
    ratio = outside.sum() / len(outside) if len(outside) else 0.0
    print("points outside workspace:", outside.sum(), "/", len(outside), f"({ratio*100:.2f}%)")
    low = (points < ws_min)
    high = (points > ws_max)
    print("exceed low counts:", low.sum(axis=0))
    print("exceed high counts:", high.sum(axis=0))


def main():
    parser = argparse.ArgumentParser(description="Inspect a collected demo for sanity and export PLYs.")
    parser.add_argument("--demo", required=True, help="Path to demo .pkl")
    parser.add_argument(
        "--frame-idx",
        default=None,
        help="Comma-separated indices (e.g., 0,10,20 or 5-15). Defaults to 0,mid,last.",
    )
    parser.add_argument("--out-dir", default="ip/deployment/debug_outputs", help="Output folder for PLYs")
    parser.add_argument("--sample-size", type=int, default=120000, help="Points to save per frame")
    parser.add_argument("--save-frames", action="store_true", help="Save world-frame PLYs")
    parser.add_argument("--save-ee", action="store_true", help="Save EE-frame PLYs")
    parser.add_argument("--workspace-min", type=float, nargs=3, default=None)
    parser.add_argument("--workspace-max", type=float, nargs=3, default=None)
    args = parser.parse_args()

    demo_path = Path(args.demo)
    with demo_path.open("rb") as f:
        data = pickle.load(f)

    pcds = data.get("pcds", [])
    T_w_es = data.get("T_w_es", [])
    grips = data.get("grips", [])

    print("keys:", data.keys())
    print("frames:", len(pcds), "poses:", len(T_w_es), "grips:", len(grips))
    _stats_summary(pcds)
    _pose_sanity(T_w_es)

    valid_pcds = [np.asarray(p) for p in pcds if p is not None and len(p) > 0]
    if not valid_pcds:
        print("No valid point clouds found.")
        return

    all_points = np.concatenate(valid_pcds, axis=0)
    all_points = all_points[np.isfinite(all_points).all(axis=1)]
    print("empty pcd frames:", sum(1 for p in pcds if p is None or len(p) == 0))
    print("nan frames:", sum(1 for p in pcds if p is not None and not np.isfinite(p).all()))
    _compute_bounds(all_points)

    if args.workspace_min is not None and args.workspace_max is not None:
        ws_min = np.array(args.workspace_min, dtype=float)
        ws_max = np.array(args.workspace_max, dtype=float)
        _check_workspace(all_points, ws_min, ws_max)

    if args.save_frames or args.save_ee:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        indices = _parse_indices(args.frame_idx, len(pcds))
        for idx in indices:
            pcd = np.asarray(pcds[idx])
            if pcd is None or len(pcd) == 0:
                continue
            if args.sample_size and len(pcd) > args.sample_size:
                sel = np.random.choice(len(pcd), size=args.sample_size, replace=False)
                pcd = pcd[sel]

            if args.save_frames:
                _write_ply(out_dir / f"frame_{idx:03d}.ply", pcd)

            if args.save_ee and idx < len(T_w_es):
                T = np.asarray(T_w_es[idx])
                R = T[:3, :3]
                t = T[:3, 3]
                pcd_ee = (R.T @ (pcd - t).T).T
                _write_ply(out_dir / f"frame_{idx:03d}_ee.ply", pcd_ee)

        print(f"Saved debug PLYs to {out_dir}")


if __name__ == "__main__":
    main()
