"""Bimanual graph representation (M1).

This module builds a clean heterogenous graph from the relative/local bimanual contract:
- scene points in left and right local frames,
- per-arm gripper state,
- cross-arm transform T_left_right.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from ip.utils.common_utils import PositionalEncoder

from .contracts import BimanualObservation
from .frame_ops import invert_transform, transform_points


@dataclass
class BimanualGraphConfig:
    """Configuration for M1 graph construction."""

    hidden_dim: int = 256
    local_num_freq: int = 10
    k_scene_scene: int = 16
    k_scene_gripper: int = 6
    include_gripper_self_edges: bool = True
    use_cross_edges: bool = True
    device: str = "cuda"
    gripper_keypoints: Optional[torch.Tensor] = None


class BimanualGraphRep(nn.Module):
    """Construct bimanual hetero graph from local-frame observation."""

    def __init__(self, cfg: BimanualGraphConfig):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.gripper_keypoints = self._resolve_gripper_keypoints(cfg).to(self.device)
        self.num_gripper_nodes = int(self.gripper_keypoints.shape[0])

        self.pos_enc = PositionalEncoder(
            3, cfg.local_num_freq, log_space=True, add_original_x=True, scale=1.0
        )
        self.pos_dim = self.pos_enc.d_output
        self.edge_dim = self.pos_dim * 2

        self.scene_proj = nn.Linear(self.pos_dim, cfg.hidden_dim)
        self.gripper_proj = nn.Linear(self.pos_dim + 1, cfg.hidden_dim)

        self.node_types = ["scene_left", "scene_right", "gripper_left", "gripper_right"]
        self.edge_types = [
            ("scene_left", "rel", "scene_left"),
            ("scene_right", "rel", "scene_right"),
            ("scene_left", "rel", "gripper_left"),
            ("scene_right", "rel", "gripper_right"),
            ("gripper_left", "rel", "gripper_left"),
            ("gripper_right", "rel", "gripper_right"),
        ]
        if cfg.use_cross_edges:
            self.edge_types.extend(
                [
                    ("gripper_left", "cross", "gripper_right"),
                    ("gripper_right", "cross", "gripper_left"),
                ]
            )

    @staticmethod
    def _resolve_gripper_keypoints(cfg: BimanualGraphConfig) -> torch.Tensor:
        if cfg.gripper_keypoints is not None:
            if cfg.gripper_keypoints.ndim != 2 or cfg.gripper_keypoints.shape[-1] != 3:
                raise ValueError(
                    "gripper_keypoints must be [G, 3], got "
                    f"{tuple(cfg.gripper_keypoints.shape)}"
                )
            return cfg.gripper_keypoints.float()

        # Instant Policy style 6-node template (60 mm spacing).
        return torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, -0.06],
                [0.0, 0.06, 0.0],
                [0.0, -0.06, 0.0],
                [0.0, 0.06, 0.06],
                [0.0, -0.06, 0.06],
            ],
            dtype=torch.float32,
        )

    @staticmethod
    def _batch_index(batch_size: int, n_per_batch: int, device: torch.device) -> torch.Tensor:
        return torch.arange(batch_size, device=device).repeat_interleave(n_per_batch)

    def _dense_edges_same_batch(
        self,
        src_batch: torch.Tensor,
        dst_batch: torch.Tensor,
        exclude_self: bool = False,
    ) -> torch.Tensor:
        edges = []
        for b in torch.unique(src_batch):
            src_ids = torch.nonzero(src_batch == b, as_tuple=False).squeeze(-1)
            dst_ids = torch.nonzero(dst_batch == b, as_tuple=False).squeeze(-1)
            if src_ids.numel() == 0 or dst_ids.numel() == 0:
                continue
            prod = torch.cartesian_prod(src_ids, dst_ids)
            if exclude_self and src_ids.data_ptr() == dst_ids.data_ptr():
                prod = prod[prod[:, 0] != prod[:, 1]]
            edges.append(prod.t().contiguous())
        if not edges:
            return torch.zeros((2, 0), dtype=torch.long, device=self.device)
        return torch.cat(edges, dim=1)

    def _knn_edges_same_batch(
        self,
        src_pos: torch.Tensor,
        src_batch: torch.Tensor,
        dst_pos: torch.Tensor,
        dst_batch: torch.Tensor,
        k: int,
        exclude_self: bool = False,
    ) -> torch.Tensor:
        edges = []
        for b in torch.unique(src_batch):
            src_ids = torch.nonzero(src_batch == b, as_tuple=False).squeeze(-1)
            dst_ids = torch.nonzero(dst_batch == b, as_tuple=False).squeeze(-1)
            if src_ids.numel() == 0 or dst_ids.numel() == 0:
                continue

            src_b = src_pos[src_ids]
            dst_b = dst_pos[dst_ids]
            dist = torch.cdist(src_b, dst_b, p=2)

            if exclude_self and src_ids.shape == dst_ids.shape and torch.equal(src_ids, dst_ids):
                dist = dist + torch.eye(dist.shape[0], device=dist.device) * 1e9

            k_eff = int(min(max(k, 1), dst_b.shape[0]))
            nn_idx = torch.topk(dist, k=k_eff, largest=False, dim=1).indices
            src_rep = src_ids[:, None].repeat(1, k_eff).reshape(-1)
            dst_sel = dst_ids[nn_idx.reshape(-1)]
            edges.append(torch.stack([src_rep, dst_sel], dim=0))

        if not edges:
            return torch.zeros((2, 0), dtype=torch.long, device=self.device)
        return torch.cat(edges, dim=1)

    def _encode_rel(self, rel: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.pos_enc(rel), self.pos_enc(rel)], dim=-1)

    def _edge_attr_local(
        self, src_pos: torch.Tensor, dst_pos: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        if edge_index.shape[1] == 0:
            return torch.zeros((0, self.edge_dim), dtype=src_pos.dtype, device=self.device)
        rel = dst_pos[edge_index[1]] - src_pos[edge_index[0]]
        return self._encode_rel(rel)

    def _edge_attr_cross(
        self,
        src_pos: torch.Tensor,
        dst_pos: torch.Tensor,
        src_batch: torch.Tensor,
        edge_index: torch.Tensor,
        T_src_from_dst: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.shape[1] == 0:
            return torch.zeros((0, self.edge_dim), dtype=src_pos.dtype, device=self.device)

        src = src_pos[edge_index[0]]
        dst = dst_pos[edge_index[1]]
        batch = src_batch[edge_index[0]].long()
        T = T_src_from_dst[batch]

        dst_in_src = transform_points(T, dst[:, None, :]).squeeze(1)
        rel = dst_in_src - src
        rot_src = torch.matmul(T[:, :3, :3], src[..., None]).squeeze(-1)
        rot_delta = rot_src - src
        return torch.cat([self.pos_enc(rel), self.pos_enc(rot_delta)], dim=-1)

    def build_graph(self, obs: BimanualObservation) -> HeteroData:
        """Build a heterogenous graph from local-frame observation."""
        obs.validate()

        points_left = obs.points_left.to(self.device)
        points_right = obs.points_right.to(self.device)
        T_left_right = obs.T_left_right.to(self.device)
        grip_left = obs.grip_left.to(self.device)
        grip_right = obs.grip_right.to(self.device)

        batch_size, n_left, _ = points_left.shape
        _, n_right, _ = points_right.shape
        g = self.num_gripper_nodes

        graph = HeteroData()

        # Node positions + batch ids
        left_scene_pos = points_left.reshape(-1, 3)
        right_scene_pos = points_right.reshape(-1, 3)
        left_scene_batch = self._batch_index(batch_size, n_left, self.device)
        right_scene_batch = self._batch_index(batch_size, n_right, self.device)

        left_gripper_pos = self.gripper_keypoints[None, :, :].expand(batch_size, g, 3).reshape(-1, 3)
        right_gripper_pos = self.gripper_keypoints[None, :, :].expand(batch_size, g, 3).reshape(-1, 3)
        left_gripper_batch = self._batch_index(batch_size, g, self.device)
        right_gripper_batch = self._batch_index(batch_size, g, self.device)

        # Node features
        graph["scene_left"].pos = left_scene_pos
        graph["scene_right"].pos = right_scene_pos
        graph["gripper_left"].pos = left_gripper_pos
        graph["gripper_right"].pos = right_gripper_pos

        graph["scene_left"].batch = left_scene_batch
        graph["scene_right"].batch = right_scene_batch
        graph["gripper_left"].batch = left_gripper_batch
        graph["gripper_right"].batch = right_gripper_batch

        graph["scene_left"].x = self.scene_proj(self.pos_enc(left_scene_pos))
        graph["scene_right"].x = self.scene_proj(self.pos_enc(right_scene_pos))

        grip_left_scalar = grip_left.reshape(batch_size, -1)[:, :1].expand(batch_size, g).reshape(-1, 1)
        grip_right_scalar = grip_right.reshape(batch_size, -1)[:, :1].expand(batch_size, g).reshape(-1, 1)
        graph["gripper_left"].x = self.gripper_proj(
            torch.cat([self.pos_enc(left_gripper_pos), grip_left_scalar], dim=-1)
        )
        graph["gripper_right"].x = self.gripper_proj(
            torch.cat([self.pos_enc(right_gripper_pos), grip_right_scalar], dim=-1)
        )

        # Local scene-scene edges
        e_ss_l = self._knn_edges_same_batch(
            left_scene_pos,
            left_scene_batch,
            left_scene_pos,
            left_scene_batch,
            k=self.cfg.k_scene_scene,
            exclude_self=True,
        )
        e_ss_r = self._knn_edges_same_batch(
            right_scene_pos,
            right_scene_batch,
            right_scene_pos,
            right_scene_batch,
            k=self.cfg.k_scene_scene,
            exclude_self=True,
        )
        graph[("scene_left", "rel", "scene_left")].edge_index = e_ss_l
        graph[("scene_left", "rel", "scene_left")].edge_attr = self._edge_attr_local(
            left_scene_pos, left_scene_pos, e_ss_l
        )
        graph[("scene_right", "rel", "scene_right")].edge_index = e_ss_r
        graph[("scene_right", "rel", "scene_right")].edge_attr = self._edge_attr_local(
            right_scene_pos, right_scene_pos, e_ss_r
        )

        # Local scene-gripper edges
        e_sg_l = self._knn_edges_same_batch(
            left_scene_pos,
            left_scene_batch,
            left_gripper_pos,
            left_gripper_batch,
            k=self.cfg.k_scene_gripper,
            exclude_self=False,
        )
        e_sg_r = self._knn_edges_same_batch(
            right_scene_pos,
            right_scene_batch,
            right_gripper_pos,
            right_gripper_batch,
            k=self.cfg.k_scene_gripper,
            exclude_self=False,
        )
        graph[("scene_left", "rel", "gripper_left")].edge_index = e_sg_l
        graph[("scene_left", "rel", "gripper_left")].edge_attr = self._edge_attr_local(
            left_scene_pos, left_gripper_pos, e_sg_l
        )
        graph[("scene_right", "rel", "gripper_right")].edge_index = e_sg_r
        graph[("scene_right", "rel", "gripper_right")].edge_attr = self._edge_attr_local(
            right_scene_pos, right_gripper_pos, e_sg_r
        )

        # Local gripper-gripper edges
        e_gg_l = self._dense_edges_same_batch(
            left_gripper_batch, left_gripper_batch, exclude_self=not self.cfg.include_gripper_self_edges
        )
        e_gg_r = self._dense_edges_same_batch(
            right_gripper_batch, right_gripper_batch, exclude_self=not self.cfg.include_gripper_self_edges
        )
        graph[("gripper_left", "rel", "gripper_left")].edge_index = e_gg_l
        graph[("gripper_left", "rel", "gripper_left")].edge_attr = self._edge_attr_local(
            left_gripper_pos, left_gripper_pos, e_gg_l
        )
        graph[("gripper_right", "rel", "gripper_right")].edge_index = e_gg_r
        graph[("gripper_right", "rel", "gripper_right")].edge_attr = self._edge_attr_local(
            right_gripper_pos, right_gripper_pos, e_gg_r
        )

        # Cross-arm edges and attrs from T_left_right / T_right_left
        if self.cfg.use_cross_edges:
            e_lr = self._dense_edges_same_batch(left_gripper_batch, right_gripper_batch, exclude_self=False)
            e_rl = self._dense_edges_same_batch(right_gripper_batch, left_gripper_batch, exclude_self=False)

            T_right_left = invert_transform(T_left_right)
            graph[("gripper_left", "cross", "gripper_right")].edge_index = e_lr
            graph[("gripper_left", "cross", "gripper_right")].edge_attr = self._edge_attr_cross(
                left_gripper_pos, right_gripper_pos, left_gripper_batch, e_lr, T_left_right
            )

            graph[("gripper_right", "cross", "gripper_left")].edge_index = e_rl
            graph[("gripper_right", "cross", "gripper_left")].edge_attr = self._edge_attr_cross(
                right_gripper_pos, left_gripper_pos, right_gripper_batch, e_rl, T_right_left
            )

        graph.T_left_right = T_left_right
        return graph

