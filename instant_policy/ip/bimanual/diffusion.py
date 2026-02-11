"""Bimanual diffusion-stage scaffold.

This is a bridge module:
- consumes world-frame batches via the new adapter,
- trains the bimanual backbone against relative delta targets.

It does not yet implement DDIM sampling; this stage exists to lock the
data/model/training plumbing before adding full diffusion loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightning as L
import torch

from ip.utils.common_utils import transforms_to_actions

from .data_adapter import BimanualWorldBatch, build_obs_targets
from .model import BimanualBackbone


@dataclass
class BimanualTrainingConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-6


class BimanualGraphDiffusionScaffold(L.LightningModule):
    """Pre-diffusion trainer over relative deltas."""

    def __init__(self, backbone: BimanualBackbone, cfg: BimanualTrainingConfig):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        self.loss_fn = torch.nn.L1Loss()

    @staticmethod
    def _ensure_world_batch(batch: Any) -> BimanualWorldBatch:
        if isinstance(batch, BimanualWorldBatch):
            return batch
        if isinstance(batch, dict):
            return BimanualWorldBatch(**batch)
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    @staticmethod
    def _targets_to_supervision(targets):
        b, p = targets.delta_T_left.shape[:2]
        left_6d = transforms_to_actions(targets.delta_T_left.reshape(-1, 4, 4)).reshape(b, p, 6)
        right_6d = transforms_to_actions(targets.delta_T_right.reshape(-1, 4, 4)).reshape(b, p, 6)

        g_left = targets.target_grip_left
        g_right = targets.target_grip_right
        if g_left.ndim == 2:
            g_left = g_left[..., None]
        if g_right.ndim == 2:
            g_right = g_right[..., None]

        gt_left = torch.cat([left_6d, g_left], dim=-1)
        gt_right = torch.cat([right_6d, g_right], dim=-1)
        return gt_left, gt_right

    def training_step(self, batch: Any, batch_idx: int):
        wb = self._ensure_world_batch(batch)
        obs, targets = build_obs_targets(wb)

        preds = self.backbone(obs)
        gt_left, gt_right = self._targets_to_supervision(targets)

        loss_left = self.loss_fn(preds["delta_left"], gt_left)
        loss_right = self.loss_fn(preds["delta_right"], gt_right)
        loss = 0.5 * (loss_left + loss_right)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_left", loss_left, on_step=True, on_epoch=True)
        self.log("train/loss_right", loss_right, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        wb = self._ensure_world_batch(batch)
        obs, targets = build_obs_targets(wb)
        preds = self.backbone(obs)
        gt_left, gt_right = self._targets_to_supervision(targets)

        loss_left = self.loss_fn(preds["delta_left"], gt_left)
        loss_right = self.loss_fn(preds["delta_right"], gt_right)
        loss = 0.5 * (loss_left + loss_right)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss_left", loss_left, on_step=False, on_epoch=True)
        self.log("val/loss_right", loss_right, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

