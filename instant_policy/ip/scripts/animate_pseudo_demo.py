import argparse
import os
import tempfile

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None


def to_world(points, T):
    pts = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    world = (T @ pts.T).T
    return world[:, :3]


def load_frame(path, crop_radius=None):
    data = torch.load(path, map_location="cpu")
    points = data.pos_obs.numpy()
    T_w_e = data.T_w_e.squeeze(0).numpy()
    points = to_world(points, T_w_e)
    grip = T_w_e[:3, 3]
    if crop_radius is not None:
        d = np.linalg.norm(points - grip[None, :], axis=1)
        points = points[d <= crop_radius]
    return points, grip


def render_frame(points, grip, view, out_path):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    if len(points) > 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, alpha=0.6, c="gray")
    ax.scatter([grip[0]], [grip[1]], [grip[2]], s=30, c="red")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=view[0], azim=view[1])
    if len(points) > 0:
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        centers = (mins + maxs) * 0.5
        span = (maxs - mins).max()
        if span <= 0:
            span = 0.1
        radius = span * 0.6
        ax.set_xlim(centers[0] - radius, centers[0] + radius)
        ax.set_ylim(centers[1] - radius, centers[1] + radius)
        ax.set_zlim(centers[2] - radius, centers[2] + radius)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Animate pseudo-demo samples from data_*.pt.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=120)
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument("--crop_radius", type=float, default=0.4)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--view", type=float, nargs=2, default=[30.0, 45.0])
    args = parser.parse_args()

    frame_indices = [args.start_idx + i * args.frame_step for i in range(args.num_frames)]
    tmp_dir = tempfile.mkdtemp(prefix="pseudo_frames_")
    frame_paths = []
    for i, idx in enumerate(frame_indices):
        path = os.path.join(args.data_dir, f"data_{idx}.pt")
        if not os.path.exists(path):
            break
        points, grip = load_frame(path, crop_radius=args.crop_radius)
        frame_path = os.path.join(tmp_dir, f"frame_{i:04d}.png")
        render_frame(points, grip, args.view, frame_path)
        frame_paths.append(frame_path)

    if not frame_paths:
        raise RuntimeError("No frames rendered. Check data_dir/start_idx.")

    if imageio is None:
        print(f"Frames saved to {tmp_dir}; install imageio to write {args.out_path}")
        return

    images = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(args.out_path, images, fps=15)
    print(f"Saved animation to {args.out_path}")


if __name__ == "__main__":
    main()
