"""Smoke test for the bimanual reboot stack.

Runs one synthetic forward + one scaffold training step.
"""

from __future__ import annotations

import argparse

import torch

from ip.bimanual.data_adapter import BimanualWorldBatch
from ip.bimanual.diffusion import BimanualGraphDiffusionScaffold, BimanualTrainingConfig
from ip.bimanual.graph_rep import BimanualGraphConfig
from ip.bimanual.model import BimanualBackbone, BimanualModelConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-points", type=int, default=64)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--hidden-dim", type=int, default=64)
    return p.parse_args()


def make_synth_batch(batch_size: int, num_points: int, horizon: int, device: torch.device):
    points = torch.randn(batch_size, num_points, 3, device=device)
    T_w_left_cur = torch.eye(4, device=device)[None, :, :].repeat(batch_size, 1, 1)
    T_w_right_cur = torch.eye(4, device=device)[None, :, :].repeat(batch_size, 1, 1)
    T_w_right_cur[:, 0, 3] = 0.30

    T_w_left_fut = torch.eye(4, device=device)[None, None, :, :].repeat(batch_size, horizon, 1, 1)
    T_w_right_fut = torch.eye(4, device=device)[None, None, :, :].repeat(batch_size, horizon, 1, 1)
    T_w_left_fut[:, :, 1, 3] = torch.linspace(0.00, 0.08, horizon, device=device)
    T_w_right_fut[:, :, 1, 3] = torch.linspace(0.00, -0.08, horizon, device=device)

    grip_left_cur = torch.zeros(batch_size, device=device)
    grip_right_cur = torch.ones(batch_size, device=device)
    grip_left_fut = torch.zeros(batch_size, horizon, device=device)
    grip_right_fut = torch.ones(batch_size, horizon, device=device)

    return BimanualWorldBatch(
        points_world=points,
        T_w_left_current=T_w_left_cur,
        T_w_right_current=T_w_right_cur,
        T_w_left_future=T_w_left_fut,
        T_w_right_future=T_w_right_fut,
        grip_left_current=grip_left_cur,
        grip_right_current=grip_right_cur,
        grip_left_future=grip_left_fut,
        grip_right_future=grip_right_fut,
    )


def main():
    args = parse_args()
    device = torch.device(args.device)

    batch = make_synth_batch(args.batch_size, args.num_points, args.horizon, device)

    backbone = BimanualBackbone(
        BimanualGraphConfig(
            hidden_dim=args.hidden_dim,
            k_scene_scene=8,
            k_scene_gripper=4,
            device=args.device,
        ),
        BimanualModelConfig(
            hidden_dim=args.hidden_dim,
            num_layers=2,
            heads=4,
            pred_horizon=args.horizon,
            device=args.device,
        ),
    )

    trainer_mod = BimanualGraphDiffusionScaffold(
        backbone,
        BimanualTrainingConfig(
            pred_horizon=args.horizon,
            num_diffusion_iters_train=32,
            num_diffusion_iters_test=4,
        ),
    )
    trainer_mod.to(device)

    with torch.no_grad():
        # Training step smoke covers full adapter + model path.
        loss = trainer_mod.training_step(batch, 0)

    print("smoke_ok")
    print(f"loss={float(loss.detach().cpu()):.6f}")


if __name__ == "__main__":
    main()
