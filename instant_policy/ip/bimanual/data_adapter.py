"""World->relative data adapter for bimanual training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .contracts import BimanualObservation, BimanualTargets, build_observation_from_world
from .frame_ops import compose_transforms, relative_transform


@dataclass
class BimanualWorldBatch:
    """Canonical world-frame batch contract before conversion."""

    points_world: torch.Tensor
    T_w_left_current: torch.Tensor
    T_w_right_current: torch.Tensor
    T_w_left_future: torch.Tensor
    T_w_right_future: torch.Tensor
    grip_left_current: torch.Tensor
    grip_right_current: torch.Tensor
    grip_left_future: torch.Tensor
    grip_right_future: torch.Tensor

    def validate(self) -> None:
        if self.points_world.ndim != 3 or self.points_world.shape[-1] != 3:
            raise ValueError(
                f"points_world must be [B, N, 3], got {tuple(self.points_world.shape)}"
            )
        if self.T_w_left_current.shape[-2:] != (4, 4):
            raise ValueError("T_w_left_current must be [..., 4, 4]")
        if self.T_w_right_current.shape[-2:] != (4, 4):
            raise ValueError("T_w_right_current must be [..., 4, 4]")
        if self.T_w_left_future.ndim != 4 or self.T_w_left_future.shape[-2:] != (4, 4):
            raise ValueError("T_w_left_future must be [B, P, 4, 4]")
        if self.T_w_right_future.ndim != 4 or self.T_w_right_future.shape[-2:] != (4, 4):
            raise ValueError("T_w_right_future must be [B, P, 4, 4]")

        b = self.points_world.shape[0]
        if self.T_w_left_current.shape[0] != b or self.T_w_right_current.shape[0] != b:
            raise ValueError("current transform batch mismatch")
        if self.T_w_left_future.shape[0] != b or self.T_w_right_future.shape[0] != b:
            raise ValueError("future transform batch mismatch")
        if self.grip_left_current.shape[0] != b or self.grip_right_current.shape[0] != b:
            raise ValueError("current gripper batch mismatch")
        if self.grip_left_future.shape[0] != b or self.grip_right_future.shape[0] != b:
            raise ValueError("future gripper batch mismatch")
        if self.T_w_left_future.shape[1] != self.T_w_right_future.shape[1]:
            raise ValueError("left/right future horizons must match")


def build_obs_targets(batch: BimanualWorldBatch) -> Tuple[BimanualObservation, BimanualTargets]:
    """Convert world-frame batch to local observation + relative targets."""
    batch.validate()

    obs = build_observation_from_world(
        points_world=batch.points_world,
        T_w_left=batch.T_w_left_current,
        T_w_right=batch.T_w_right_current,
        grip_left=batch.grip_left_current,
        grip_right=batch.grip_right_current,
    )

    T_w_left_current = batch.T_w_left_current[:, None, :, :]
    T_w_right_current = batch.T_w_right_current[:, None, :, :]

    delta_T_left = relative_transform(T_w_left_current, batch.T_w_left_future)
    delta_T_right = relative_transform(T_w_right_current, batch.T_w_right_future)

    targets = BimanualTargets(
        delta_T_left=delta_T_left,
        delta_T_right=delta_T_right,
        target_grip_left=batch.grip_left_future,
        target_grip_right=batch.grip_right_future,
    )
    targets.validate()
    return obs, targets


def relabel_world(batch: BimanualWorldBatch, T_new_w_old: torch.Tensor) -> BimanualWorldBatch:
    """Apply a global frame relabeling: T'_W_X = T_new_w_old @ T_W_X."""
    if T_new_w_old.shape[-2:] != (4, 4):
        raise ValueError("T_new_w_old must be [..., 4, 4]")

    points_w = batch.points_world
    R = T_new_w_old[:, :3, :3]
    t = T_new_w_old[:, :3, 3]
    points_new = torch.matmul(R, points_w.transpose(-1, -2)).transpose(-1, -2) + t[:, None, :]

    return BimanualWorldBatch(
        points_world=points_new,
        T_w_left_current=compose_transforms(T_new_w_old, batch.T_w_left_current),
        T_w_right_current=compose_transforms(T_new_w_old, batch.T_w_right_current),
        T_w_left_future=compose_transforms(T_new_w_old[:, None, :, :], batch.T_w_left_future),
        T_w_right_future=compose_transforms(T_new_w_old[:, None, :, :], batch.T_w_right_future),
        grip_left_current=batch.grip_left_current,
        grip_right_current=batch.grip_right_current,
        grip_left_future=batch.grip_left_future,
        grip_right_future=batch.grip_right_future,
    )

