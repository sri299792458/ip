"""SE(3) frame utilities for bimanual Instant Policy.

Convention:
- `T_A_B` maps coordinates from frame B -> frame A.
- All transforms are homogeneous 4x4 matrices.
"""

from __future__ import annotations

from typing import Dict

import torch


def _check_transform(name: str, T: torch.Tensor) -> None:
    if T.shape[-2:] != (4, 4):
        raise ValueError(f"{name} must have shape [..., 4, 4], got {tuple(T.shape)}")


def compose_transforms(T_a_b: torch.Tensor, T_b_c: torch.Tensor) -> torch.Tensor:
    """Compose transforms: T_a_c = T_a_b @ T_b_c."""
    _check_transform("T_a_b", T_a_b)
    _check_transform("T_b_c", T_b_c)
    return torch.matmul(T_a_b, T_b_c)


def invert_transform(T_a_b: torch.Tensor) -> torch.Tensor:
    """Invert transform: returns T_b_a.

    Uses rigid-body inverse (R^T, -R^T t) for numerical stability.
    """
    _check_transform("T_a_b", T_a_b)
    R = T_a_b[..., :3, :3]
    t = T_a_b[..., :3, 3]

    R_t = R.transpose(-1, -2)
    t_inv = -torch.matmul(R_t, t[..., None]).squeeze(-1)

    T_b_a = torch.zeros_like(T_a_b)
    T_b_a[..., :3, :3] = R_t
    T_b_a[..., :3, 3] = t_inv
    T_b_a[..., 3, 3] = 1.0
    return T_b_a


def relative_transform(T_w_a: torch.Tensor, T_w_b: torch.Tensor) -> torch.Tensor:
    """Compute T_a_b from world-frame poses T_w_a and T_w_b."""
    _check_transform("T_w_a", T_w_a)
    _check_transform("T_w_b", T_w_b)
    return compose_transforms(invert_transform(T_w_a), T_w_b)


def transform_points(T_a_b: torch.Tensor, p_b: torch.Tensor) -> torch.Tensor:
    """Transform 3D points from frame B to frame A.

    Args:
        T_a_b: [..., 4, 4]
        p_b:   [..., N, 3]
    Returns:
        p_a:   [..., N, 3]
    """
    _check_transform("T_a_b", T_a_b)
    if p_b.shape[-1] != 3:
        raise ValueError(f"p_b must have shape [..., N, 3], got {tuple(p_b.shape)}")

    R = T_a_b[..., :3, :3]
    t = T_a_b[..., :3, 3]
    p_a = torch.matmul(R, p_b.transpose(-1, -2)).transpose(-1, -2) + t[..., None, :]
    return p_a


def world_to_local_points(T_w_f: torch.Tensor, p_w: torch.Tensor) -> torch.Tensor:
    """Convert world points to local frame F points."""
    T_f_w = invert_transform(T_w_f)
    return transform_points(T_f_w, p_w)


def check_global_relabel_invariance(
    T_w_left: torch.Tensor,
    T_w_right: torch.Tensor,
    T_w_obj: torch.Tensor,
    T_new_w_old: torch.Tensor,
) -> Dict[str, float]:
    """Numerically verify invariance to global relabeling.

    Let T'_W_X = T_new_w_old @ T_W_X.
    Then:
      T'_L_O == T_L_O
      T'_R_O == T_R_O
      T'_L_R == T_L_R
    """
    _check_transform("T_w_left", T_w_left)
    _check_transform("T_w_right", T_w_right)
    _check_transform("T_w_obj", T_w_obj)
    _check_transform("T_new_w_old", T_new_w_old)

    T_l_o = relative_transform(T_w_left, T_w_obj)
    T_r_o = relative_transform(T_w_right, T_w_obj)
    T_l_r = relative_transform(T_w_left, T_w_right)

    T_wp_left = compose_transforms(T_new_w_old, T_w_left)
    T_wp_right = compose_transforms(T_new_w_old, T_w_right)
    T_wp_obj = compose_transforms(T_new_w_old, T_w_obj)

    T_lp_o = relative_transform(T_wp_left, T_wp_obj)
    T_rp_o = relative_transform(T_wp_right, T_wp_obj)
    T_lp_r = relative_transform(T_wp_left, T_wp_right)

    return {
        "max_abs_err_T_l_o": float((T_l_o - T_lp_o).abs().max().item()),
        "max_abs_err_T_r_o": float((T_r_o - T_rp_o).abs().max().item()),
        "max_abs_err_T_l_r": float((T_l_r - T_lp_r).abs().max().item()),
    }

