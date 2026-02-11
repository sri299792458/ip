"""Bimanual model scaffold (M2 pre-diffusion backbone)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from ip.models.graph_transformer import GraphTransformer
from ip.utils.common_utils import SinusoidalPosEmb

from .contracts import BimanualObservation
from .graph_rep import BimanualGraphConfig, BimanualGraphRep


@dataclass
class BimanualModelConfig:
    hidden_dim: int = 256
    num_layers: int = 3
    heads: int = 4
    pred_horizon: int = 8
    edge_dropout: float = 0.0
    device: str = "cuda"


class BimanualBackbone(nn.Module):
    """Relative bimanual action backbone.

    This stage intentionally stays simple:
    - builds local/relative graph from observation,
    - runs one hetero transformer stack,
    - predicts per-arm relative deltas for a fixed horizon.
    """

    def __init__(self, graph_cfg: BimanualGraphConfig, model_cfg: BimanualModelConfig):
        super().__init__()
        self.model_cfg = model_cfg
        self.device = torch.device(model_cfg.device)

        # Keep graph hidden dim aligned with model hidden dim.
        graph_cfg = BimanualGraphConfig(**{**graph_cfg.__dict__, "hidden_dim": model_cfg.hidden_dim})
        self.graph_rep = BimanualGraphRep(graph_cfg)
        self.num_gripper_nodes = self.graph_rep.num_gripper_nodes

        metadata = (self.graph_rep.node_types, self.graph_rep.edge_types)
        self.encoder = GraphTransformer(
            in_channels=model_cfg.hidden_dim,
            hidden_channels=model_cfg.hidden_dim,
            heads=model_cfg.heads,
            edge_dim=self.graph_rep.edge_dim,
            num_layers=model_cfg.num_layers,
            metadata=metadata,
            dropout=model_cfg.edge_dropout,
            norm="layer",
        ).to(self.device)

        self.time_emb = SinusoidalPosEmb(64)
        self.time_proj = nn.Linear(64, model_cfg.hidden_dim).to(self.device)
        self.node_emb_left = nn.Embedding(self.num_gripper_nodes, model_cfg.hidden_dim).to(self.device)
        self.node_emb_right = nn.Embedding(self.num_gripper_nodes, model_cfg.hidden_dim).to(self.device)

        self.trans_head_left = self._head(3)
        self.rot_head_left = self._head(3)
        self.grip_head_left = self._head(1)

        self.trans_head_right = self._head(3)
        self.rot_head_right = self._head(3)
        self.grip_head_right = self._head(1)

    def _head(self, out_dim: int) -> nn.Module:
        h = self.model_cfg.hidden_dim
        return nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, out_dim)).to(self.device)

    @staticmethod
    def _mean_by_batch(x: torch.Tensor, batch: torch.Tensor, batch_size: int) -> torch.Tensor:
        out = torch.zeros((batch_size, x.shape[-1]), device=x.device, dtype=x.dtype)
        cnt = torch.zeros((batch_size, 1), device=x.device, dtype=x.dtype)
        out.index_add_(0, batch, x)
        one = torch.ones((batch.shape[0], 1), device=x.device, dtype=x.dtype)
        cnt.index_add_(0, batch, one)
        return out / cnt.clamp_min(1.0)

    def get_transformed_node_pos(
        self, actions: torch.Tensor, arm: Optional[str] = None, transform: bool = True
    ) -> torch.Tensor:
        """Return gripper keypoints, optionally transformed by relative actions.

        Args:
            actions: [B, P, 4, 4]
            arm: kept for API parity; both arms use same keypoint template.
            transform: if False, returns canonical keypoints in current local frame.
        """
        b, p = actions.shape[:2]
        kp = self.graph_rep.gripper_keypoints[None, None, :, :].expand(b, p, -1, -1)
        if not transform:
            return kp

        R = actions[..., :3, :3]
        t = actions[..., :3, 3]
        return torch.matmul(R, kp.transpose(-1, -2)).transpose(-1, -2) + t[:, :, None, :]

    def get_labels(
        self,
        gt_actions: torch.Tensor,
        noisy_actions: torch.Tensor,
        gt_grips: torch.Tensor,
        noisy_grips: torch.Tensor,
        arm: str,
    ) -> torch.Tensor:
        """Compute per-node relative labels, same formulation as single-arm Instant Policy."""
        del arm  # both arms share the same keypoint template
        gripper_points = self.graph_rep.gripper_keypoints[None, None, :, :].repeat(
            gt_actions.shape[0], gt_actions.shape[1], 1, 1
        )

        T_w_n = noisy_actions.view(-1, 4, 4)
        T_n_w = torch.inverse(T_w_n)
        T_w_g = gt_actions.view(-1, 4, 4)
        T_n_g = torch.bmm(T_n_w, T_w_g).view(gt_actions.shape[0], gt_actions.shape[1], 4, 4)

        labels_trans = T_n_g[..., :3, 3][:, :, None, :].repeat(1, 1, gripper_points.shape[-2], 1)

        T_n_g_rot = T_n_g.clone()
        T_n_g_rot[..., :3, 3] = 0
        R = T_n_g_rot[..., :3, :3]
        labels_rot = torch.matmul(R, gripper_points.transpose(-1, -2)).transpose(-1, -2) - gripper_points

        labels_grip = gt_grips[:, :, None, :].repeat(1, 1, gripper_points.shape[-2], 1)
        return torch.cat([labels_trans, labels_rot, labels_grip], dim=-1)

    def forward(
        self, obs: BimanualObservation, diff_time: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        obs.validate()
        graph = self.graph_rep.build_graph(obs)

        x_dict = self.encoder(graph.x_dict, graph.edge_index_dict, graph.edge_attr_dict)

        batch_size = int(obs.points_left.shape[0])
        horizon = int(self.model_cfg.pred_horizon)

        left_ctx = self._mean_by_batch(
            x_dict["gripper_left"],
            graph["gripper_left"].batch.long(),
            batch_size,
        )
        right_ctx = self._mean_by_batch(
            x_dict["gripper_right"],
            graph["gripper_right"].batch.long(),
            batch_size,
        )

        if diff_time is not None:
            dt = diff_time.view(batch_size).float()
            dt_emb = self.time_proj(self.time_emb(dt))
            left_ctx = left_ctx + dt_emb
            right_ctx = right_ctx + dt_emb

        node_ids = torch.arange(self.num_gripper_nodes, device=left_ctx.device).long()
        left_node = self.node_emb_left(node_ids)[None, None, :, :].expand(
            batch_size, horizon, self.num_gripper_nodes, -1
        )
        right_node = self.node_emb_right(node_ids)[None, None, :, :].expand(
            batch_size, horizon, self.num_gripper_nodes, -1
        )

        left_h = left_ctx[:, None, None, :].expand(
            batch_size, horizon, self.num_gripper_nodes, left_ctx.shape[-1]
        ) + left_node
        right_h = right_ctx[:, None, None, :].expand(
            batch_size, horizon, self.num_gripper_nodes, right_ctx.shape[-1]
        ) + right_node

        delta_trans_left = self.trans_head_left(left_h)
        delta_rot_left = self.rot_head_left(left_h)
        delta_grip_left = self.grip_head_left(left_h)

        delta_trans_right = self.trans_head_right(right_h)
        delta_rot_right = self.rot_head_right(right_h)
        delta_grip_right = self.grip_head_right(right_h)

        return {
            "delta_trans_left": delta_trans_left,
            "delta_rot_left": delta_rot_left,
            "delta_grip_left": delta_grip_left,
            "delta_trans_right": delta_trans_right,
            "delta_rot_right": delta_rot_right,
            "delta_grip_right": delta_grip_right,
            "delta_left": torch.cat([delta_trans_left, delta_rot_left, delta_grip_left], dim=-1),
            "delta_right": torch.cat([delta_trans_right, delta_rot_right, delta_grip_right], dim=-1),
        }
