"""Bimanual model scaffold (M2 pre-diffusion backbone)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from ip.models.graph_transformer import GraphTransformer

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

    def forward(self, obs: BimanualObservation) -> Dict[str, torch.Tensor]:
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

        left_h = left_ctx[:, None, :].expand(batch_size, horizon, left_ctx.shape[-1])
        right_h = right_ctx[:, None, :].expand(batch_size, horizon, right_ctx.shape[-1])

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

