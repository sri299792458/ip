import argparse
import os
import tempfile
from typing import Optional, Tuple

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None


def to_world(points: np.ndarray, T_w_e: np.ndarray) -> np.ndarray:
    pts_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    world = (T_w_e @ pts_h.T).T
    return world[:, :3]


def load_frame(path: str, frame: str, max_points: int, rng: np.random.Generator):
    data = torch.load(path, map_location="cpu")
    if not hasattr(data, "pos_obs") or not hasattr(data, "T_w_e"):
        raise RuntimeError(f"{path} is missing required fields pos_obs/T_w_e")

    points = data.pos_obs.detach().cpu().numpy()
    if len(points) > max_points:
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]

    T_w_e = data.T_w_e.squeeze(0).detach().cpu().numpy()
    if frame == "world":
        points = to_world(points, T_w_e)
        grip = T_w_e[:3, 3]
    else:
        grip = np.zeros(3, dtype=np.float32)
    return points, grip


def apply_crop(points: np.ndarray, grip: np.ndarray, crop_radius: Optional[float]):
    if crop_radius is None:
        return points
    d = np.linalg.norm(points - grip[None, :], axis=1)
    return points[d <= crop_radius]


def set_equal_axes(ax, points: np.ndarray, grip: np.ndarray):
    if len(points) == 0:
        points = grip[None, :]
    all_pts = np.concatenate([points, grip[None, :]], axis=0)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    centers = (mins + maxs) * 0.5
    span = max((maxs - mins).max(), 0.1)
    r = span * 0.5
    ax.set_xlim(centers[0] - r, centers[0] + r)
    ax.set_ylim(centers[1] - r, centers[1] + r)
    ax.set_zlim(centers[2] - r, centers[2] + r)


def render(points: np.ndarray, grip: np.ndarray, view: Tuple[float, float], out_path: str):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    if len(points) > 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, alpha=0.6, c="gray")
    ax.scatter([grip[0]], [grip[1]], [grip[2]], s=35, c="red")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=view[0], azim=view[1])
    set_equal_axes(ax, points, grip)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Minimal pseudo-demo debug utility for data_*.pt visualization."
    )
    parser.add_argument("--data_dir", required=True, help="Directory containing data_*.pt files")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting data index")
    parser.add_argument("--count", type=int, default=1, help="Number of frames to render")
    parser.add_argument("--step", type=int, default=1, help="Frame stride in data indices")
    parser.add_argument("--frame", choices=["world", "ee"], default="world")
    parser.add_argument("--max_points", type=int, default=3000)
    parser.add_argument("--crop_radius", type=float, default=None)
    parser.add_argument("--view", type=float, nargs=2, default=[30.0, 45.0], metavar=("ELEV", "AZIM"))
    parser.add_argument("--fps", type=int, default=15, help="GIF FPS when count > 1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default=None,
        help="Output path. Defaults to debug_frame_<idx>.png (count=1) or debug_clip_<idx>.gif (count>1).",
    )
    args = parser.parse_args()

    if args.count < 1:
        raise RuntimeError("--count must be >= 1")
    if args.step < 1:
        raise RuntimeError("--step must be >= 1")

    rng = np.random.default_rng(args.seed)
    view = (float(args.view[0]), float(args.view[1]))

    frame_indices = [args.start_idx + i * args.step for i in range(args.count)]
    data_paths = [os.path.join(args.data_dir, f"data_{idx}.pt") for idx in frame_indices]
    existing = [p for p in data_paths if os.path.exists(p)]
    if not existing:
        raise RuntimeError("No matching data_*.pt files found for requested range.")

    if args.out is None:
        if args.count == 1:
            out_path = os.path.join(args.data_dir, f"debug_frame_{args.start_idx}.png")
        else:
            out_path = os.path.join(args.data_dir, f"debug_clip_{args.start_idx}.gif")
    else:
        out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if args.count == 1:
        points, grip = load_frame(existing[0], args.frame, args.max_points, rng)
        points = apply_crop(points, grip, args.crop_radius)
        render(points, grip, view, out_path)
        print(f"Saved {out_path}")
        return

    temp_dir = tempfile.mkdtemp(prefix="pseudo_debug_")
    pngs = []
    for i, path in enumerate(existing):
        points, grip = load_frame(path, args.frame, args.max_points, rng)
        points = apply_crop(points, grip, args.crop_radius)
        png = os.path.join(temp_dir, f"frame_{i:04d}.png")
        render(points, grip, view, png)
        pngs.append(png)

    if imageio is None:
        print(f"imageio not available; rendered PNG frames in {temp_dir}")
        return

    images = [imageio.imread(p) for p in pngs]
    imageio.mimsave(out_path, images, fps=args.fps)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
