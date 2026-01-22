import argparse
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_sample(path):
    data = torch.load(path, map_location="cpu")
    return data


def subsample_points(points, max_points, rng):
    if max_points is None or len(points) <= max_points:
        return points
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def to_world(points, T):
    pts = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    world = (T @ pts.T).T
    return world[:, :3]


def collect_demo_trajs(data):
    demo_T = data.demo_T_w_es.squeeze(0).numpy()
    demos = []
    for d in range(demo_T.shape[0]):
        traj = demo_T[d, :, :3, 3]
        demos.append(traj)
    return demos


def collect_live_traj(data):
    T_w_e = data.T_w_e.squeeze(0).numpy()
    actions = data.actions.squeeze(0).numpy()
    points = [T_w_e[:3, 3]]
    for a in actions:
        Tw = T_w_e @ a
        points.append(Tw[:3, 3])
    return np.array(points)


def set_equal_axes(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) * 0.5
    span = (maxs - mins).max()
    if span <= 0:
        span = 0.1
    radius = span * 0.5
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def main():
    parser = argparse.ArgumentParser(description="Plot a saved pseudo-demo sample (data_*.pt).")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--out_path", default=None)
    parser.add_argument("--max_points", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame", choices=["world", "ee"], default="world")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    data = load_sample(args.data_path)

    points = data.pos_obs.numpy()
    points = subsample_points(points, args.max_points, rng)

    T_w_e = data.T_w_e.squeeze(0).numpy()
    if args.frame == "world":
        points = to_world(points, T_w_e)

    demo_trajs = collect_demo_trajs(data)
    live_traj = collect_live_traj(data)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, alpha=0.6, c="gray")
    for traj in demo_trajs:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=2)
    ax.plot(live_traj[:, 0], live_traj[:, 1], live_traj[:, 2], linewidth=2, c="red")
    all_points = np.concatenate([points, live_traj] + demo_trajs, axis=0)
    set_equal_axes(ax, all_points)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Pseudo-demo sample")

    out_path = args.out_path
    if out_path is None:
        out_path = args.data_path + ".png"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
