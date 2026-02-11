"""Data contracts for the clean bimanual representation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .frame_ops import relative_transform, world_to_local_points


@dataclass
class BimanualObservation:
    """Model input in relative/local form.

    Fields:
    - points_left: scene points in left frame, shape [B, Nl, 3]
    - points_right: scene points in right frame, shape [B, Nr, 3]
    - T_left_right: transform from right frame to left frame, shape [B, 4, 4]
    - grip_left: left gripper scalar, shape [B] or [B, 1]
    - grip_right: right gripper scalar, shape [B] or [B, 1]
    """

    points_left: torch.Tensor
    points_right: torch.Tensor
    T_left_right: torch.Tensor
    grip_left: torch.Tensor
    grip_right: torch.Tensor

    def validate(self) -> None:
        if self.points_left.ndim != 3 or self.points_left.shape[-1] != 3:
            raise ValueError(
                f"points_left must be [B, N, 3], got {tuple(self.points_left.shape)}"
            )
        if self.points_right.ndim != 3 or self.points_right.shape[-1] != 3:
            raise ValueError(
                f"points_right must be [B, N, 3], got {tuple(self.points_right.shape)}"
            )
        if self.T_left_right.shape[-2:] != (4, 4):
            raise ValueError(
                f"T_left_right must be [..., 4, 4], got {tuple(self.T_left_right.shape)}"
            )

        batch = self.points_left.shape[0]
        if self.points_right.shape[0] != batch:
            raise ValueError("points_left and points_right must have same batch size")
        if self.T_left_right.shape[0] != batch:
            raise ValueError("T_left_right batch size must match point batches")
        if self.grip_left.shape[0] != batch or self.grip_right.shape[0] != batch:
            raise ValueError("gripper batch sizes must match point batches")


@dataclass
class BimanualTargets:
    """Per-arm relative action training targets.

    Fields:
    - delta_T_left: shape [B, P, 4, 4]
    - delta_T_right: shape [B, P, 4, 4]
    - target_grip_left: shape [B, P] or [B, P, 1]
    - target_grip_right: shape [B, P] or [B, P, 1]
    """

    delta_T_left: torch.Tensor
    delta_T_right: torch.Tensor
    target_grip_left: torch.Tensor
    target_grip_right: torch.Tensor

    def validate(self) -> None:
        if self.delta_T_left.ndim != 4 or self.delta_T_left.shape[-2:] != (4, 4):
            raise ValueError(
                f"delta_T_left must be [B, P, 4, 4], got {tuple(self.delta_T_left.shape)}"
            )
        if self.delta_T_right.ndim != 4 or self.delta_T_right.shape[-2:] != (4, 4):
            raise ValueError(
                f"delta_T_right must be [B, P, 4, 4], got {tuple(self.delta_T_right.shape)}"
            )
        if self.delta_T_left.shape[:2] != self.delta_T_right.shape[:2]:
            raise ValueError("left/right action horizons must match")


def build_observation_from_world(
    points_world: torch.Tensor,
    T_w_left: torch.Tensor,
    T_w_right: torch.Tensor,
    grip_left: torch.Tensor,
    grip_right: torch.Tensor,
) -> BimanualObservation:
    """Build relative/local observation tensors from world-frame inputs.

    Args:
    - points_world: [B, N, 3]
    - T_w_left: [B, 4, 4]
    - T_w_right: [B, 4, 4]
    """
    points_left = world_to_local_points(T_w_left, points_world)
    points_right = world_to_local_points(T_w_right, points_world)
    T_left_right = relative_transform(T_w_left, T_w_right)

    obs = BimanualObservation(
        points_left=points_left,
        points_right=points_right,
        T_left_right=T_left_right,
        grip_left=grip_left,
        grip_right=grip_right,
    )
    obs.validate()
    return obs

