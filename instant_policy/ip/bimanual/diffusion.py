"""Bimanual diffusion module (DDIM, dual-arm adaptation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from contextlib import nullcontext

import lightning as L
import torch
import numpy as np
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from ip.utils.common_utils import (
    actions_to_transforms,
    get_rigid_transforms,
    rotation_matrix_to_angle_axis,
    transforms_to_actions,
)
from ip.utils.normalizer import Normalizer

from .data_adapter import BimanualWorldBatch, build_obs_targets
from .model import BimanualBackbone


@dataclass
class BimanualTrainingConfig:
    lr: float = 1e-5
    weight_decay: float = 1e-2
    pred_horizon: int = 8
    num_diffusion_iters_train: int = 100
    num_diffusion_iters_test: int = 8
    use_lr_scheduler: bool = False
    num_warmup_steps: int = 1000
    lr_cooldown_steps: int = 0
    num_iters: int = 2550000
    min_actions: torch.Tensor = torch.tensor(
        [-0.01, -0.01, -0.01, -np.deg2rad(3), -np.deg2rad(3), -np.deg2rad(3)],
        dtype=torch.float32,
    )
    max_actions: torch.Tensor = torch.tensor(
        [0.01, 0.01, 0.01, np.deg2rad(3), np.deg2rad(3), np.deg2rad(3)],
        dtype=torch.float32,
    )


class BimanualGraphDiffusion(L.LightningModule):
    """DDIM diffusion trainer/inference for bimanual relative actions."""

    def __init__(self, backbone: BimanualBackbone, cfg: BimanualTrainingConfig):
        super().__init__()
        self.model = backbone
        self.cfg = cfg
        self.loss_fn = torch.nn.L1Loss()
        normalizer_device = str(backbone.device)

        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=cfg.num_diffusion_iters_train,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=False,
            prediction_type="sample",
        )

        self.normalizer_left = Normalizer(
            pred_horizon=cfg.pred_horizon,
            min_action=cfg.min_actions.to(backbone.device),
            max_action=cfg.max_actions.to(backbone.device),
            device=normalizer_device,
        )
        self.normalizer_right = Normalizer(
            pred_horizon=cfg.pred_horizon,
            min_action=cfg.min_actions.to(backbone.device),
            max_action=cfg.max_actions.to(backbone.device),
            device=normalizer_device,
        )

    def _ensure_world_batch(self, batch: Any) -> BimanualWorldBatch:
        if isinstance(batch, BimanualWorldBatch):
            wb = batch
        elif isinstance(batch, dict):
            wb = BimanualWorldBatch(**batch)
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")
        return wb.to(self.device)

    def add_noise(
        self,
        actions: torch.Tensor,
        grip_actions: torch.Tensor,
        timesteps: torch.Tensor,
        normalizer: Normalizer,
    ):
        """Add diffusion noise to SE(3)+gripper actions."""
        b, p = actions.shape[:2]
        actions_6d = transforms_to_actions(actions.view(-1, 4, 4)).view(b, p, 6)
        actions_6d = normalizer.normalize_actions(actions_6d)

        noise = torch.randn(actions_6d.shape, device=actions.device, dtype=actions_6d.dtype)
        noisy_actions = self.noise_scheduler.add_noise(actions_6d, noise, timesteps)
        noisy_actions = torch.clamp(noisy_actions, -1, 1)
        noisy_actions = normalizer.denormalize_actions(noisy_actions)
        noisy_actions = actions_to_transforms(noisy_actions.view(-1, 6)).view(b, p, 4, 4)

        noise_g = torch.randn(grip_actions.shape, device=grip_actions.device, dtype=grip_actions.dtype)
        noisy_grips = self.noise_scheduler.add_noise(grip_actions, noise_g, timesteps)
        noisy_grips = torch.clamp(noisy_grips, -1, 1)
        return noisy_actions, noisy_grips

    @staticmethod
    def se3_loss(pred: torch.Tensor, gt: torch.Tensor):
        """Translation + rotation error in degrees."""
        trans_err = torch.norm(pred[..., :3, 3] - gt[..., :3, 3], dim=-1).mean()

        rot_error = torch.eye(4, device=pred.device, dtype=pred.dtype).repeat(pred.shape[0], pred.shape[1], 1, 1)
        rot_error[..., :3, :3] = pred[..., :3, :3].transpose(-1, -2) @ gt[..., :3, :3]
        rot_error = rot_error.view(-1, 4, 4)
        angle_axis = rotation_matrix_to_angle_axis(rot_error[:, :3, :])
        rot_err_deg = angle_axis.norm(dim=-1).mean() * 180 / np.pi
        return trans_err, rot_err_deg

    def training_step(self, batch: Any, batch_idx: int):
        wb = self._ensure_world_batch(batch)
        obs, targets = build_obs_targets(wb)
        bsz = targets.delta_T_left.shape[0]

        if targets.delta_T_left.shape[1] != self.cfg.pred_horizon:
            raise ValueError(
                f"Horizon mismatch: targets={targets.delta_T_left.shape[1]} cfg={self.cfg.pred_horizon}"
            )

        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=targets.delta_T_left.device,
        ).long()

        noisy_left, noisy_grip_left = self.add_noise(
            targets.delta_T_left,
            targets.target_grip_left,
            timesteps,
            self.normalizer_left,
        )
        noisy_right, noisy_grip_right = self.add_noise(
            targets.delta_T_right,
            targets.target_grip_right,
            timesteps,
            self.normalizer_right,
        )

        labels_left = self.model.get_labels(
            targets.delta_T_left,
            noisy_left,
            targets.target_grip_left.unsqueeze(-1),
            noisy_grip_left.unsqueeze(-1),
            arm="left",
        )
        labels_right = self.model.get_labels(
            targets.delta_T_right,
            noisy_right,
            targets.target_grip_right.unsqueeze(-1),
            noisy_grip_right.unsqueeze(-1),
            arm="right",
        )
        labels_left[..., :6] = self.normalizer_left.normalize_labels(labels_left[..., :6])
        labels_right[..., :6] = self.normalizer_right.normalize_labels(labels_right[..., :6])

        preds = self.model(obs, diff_time=timesteps.view(-1, 1))
        loss_left = self.loss_fn(preds["delta_left"], labels_left)
        loss_right = self.loss_fn(preds["delta_right"], labels_right)
        loss = 0.5 * (loss_left + loss_right)

        if self._trainer is not None:
            self.log("Train_Loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("Train_Loss_Left", loss_left, on_step=False, on_epoch=True)
            self.log("Train_Loss_Right", loss_right, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int, vis: bool = False, ret_actions: bool = False):
        wb = self._ensure_world_batch(batch)
        obs, targets = build_obs_targets(wb)

        device_type = "cuda" if targets.delta_T_left.device.type == "cuda" else "cpu"
        # Keep CUDA path explicit; CPU autocast with float32 triggers warnings.
        cast_ctx = (
            torch.autocast(dtype=torch.float32, device_type=device_type)
            if device_type == "cuda"
            else nullcontext()
        )
        with cast_ctx:  # SVD in rigid fit prefers float32
            actions_left, grips_left, actions_right, grips_right = self.test_step(obs, batch_idx, vis=vis)

        trans_left, rot_left = self.se3_loss(actions_left, targets.delta_T_left)
        trans_right, rot_right = self.se3_loss(actions_right, targets.delta_T_right)
        grip_left = (grips_left.squeeze(-1) - targets.target_grip_left).abs().mean()
        grip_right = (grips_right.squeeze(-1) - targets.target_grip_right).abs().mean()
        loss = 0.5 * ((trans_left + trans_right) + (grip_left + grip_right))

        if self._trainer is not None:
            self.log("Val_Trans_Left", trans_left, on_step=False, on_epoch=True, prog_bar=True)
            self.log("Val_Rot_Left", rot_left, on_step=False, on_epoch=True)
            self.log("Val_Grip_Left", grip_left, on_step=False, on_epoch=True)
            self.log("Val_Trans_Right", trans_right, on_step=False, on_epoch=True)
            self.log("Val_Rot_Right", rot_right, on_step=False, on_epoch=True)
            self.log("Val_Grip_Right", grip_right, on_step=False, on_epoch=True)
            self.log("Val_Trans_Mean", 0.5 * (trans_left + trans_right), on_step=False, on_epoch=True, prog_bar=True)

        if ret_actions:
            return actions_left, grips_left, actions_right, grips_right
        return loss

    def _diffusion_step(
        self,
        noisy_actions: torch.Tensor,
        noisy_grips: torch.Tensor,
        preds: torch.Tensor,
        k: int,
        normalizer: Normalizer,
        arm: str,
    ):
        bsz = noisy_actions.shape[0]
        current_gripper_pos = self.model.get_transformed_node_pos(noisy_actions, arm=arm, transform=False)
        mode_output = preds[..., 3:6] + current_gripper_pos + torch.mean(preds[..., :3], dim=-2, keepdim=True)

        pred_gripper_pos = self.noise_scheduler.step(
            model_output=mode_output,
            sample=current_gripper_pos,
            timestep=k,
        ).prev_sample

        T_e_e = get_rigid_transforms(
            current_gripper_pos.view(-1, pred_gripper_pos.shape[-2], 3),
            pred_gripper_pos.view(-1, pred_gripper_pos.shape[-2], 3),
        ).view(bsz, -1, 4, 4)
        noisy_actions = torch.matmul(noisy_actions, T_e_e)

        noisy_grips = self.noise_scheduler.step(
            model_output=preds[..., -1:].mean(dim=-2),
            sample=noisy_grips,
            timestep=k,
        ).prev_sample
        noisy_grips = torch.clamp(noisy_grips, -1, 1)

        noisy_actions_6d = transforms_to_actions(noisy_actions.view(-1, 4, 4)).view(bsz, -1, 6)
        noisy_actions_6d = normalizer.normalize_actions(noisy_actions_6d)
        noisy_actions_6d = torch.clamp(noisy_actions_6d, -1, 1)
        noisy_actions_6d = normalizer.denormalize_actions(noisy_actions_6d)
        noisy_actions = actions_to_transforms(noisy_actions_6d.view(-1, 6)).view(bsz, -1, 4, 4)
        return noisy_actions, noisy_grips

    def test_step(self, obs, batch_idx: int, vis: bool = False):
        del batch_idx, vis
        bsz = obs.points_left.shape[0]
        p = self.cfg.pred_horizon
        local_device = obs.points_left.device

        noisy_left = torch.randn((bsz, p, 6), device=local_device)
        noisy_left = torch.clamp(noisy_left, -1, 1)
        noisy_left = self.normalizer_left.denormalize_actions(noisy_left)
        noisy_left = actions_to_transforms(noisy_left.view(-1, 6)).view(bsz, p, 4, 4)

        noisy_right = torch.randn((bsz, p, 6), device=local_device)
        noisy_right = torch.clamp(noisy_right, -1, 1)
        noisy_right = self.normalizer_right.denormalize_actions(noisy_right)
        noisy_right = actions_to_transforms(noisy_right.view(-1, 6)).view(bsz, p, 4, 4)

        noisy_grips_left = torch.randn((bsz, p, 1), device=local_device)
        noisy_grips_right = torch.randn((bsz, p, 1), device=local_device)
        noisy_grips_left = torch.clamp(noisy_grips_left, -1, 1)
        noisy_grips_right = torch.clamp(noisy_grips_right, -1, 1)

        self.noise_scheduler.set_timesteps(self.cfg.num_diffusion_iters_test)
        for k in range(self.cfg.num_diffusion_iters_test - 1, -1, -1):
            dt = torch.tensor(
                [[k if k != self.cfg.num_diffusion_iters_test - 1 else self.cfg.num_diffusion_iters_train]]
                * bsz,
                device=local_device,
            )

            preds = self.model(obs, diff_time=dt)
            preds_left = preds["delta_left"].clone()
            preds_right = preds["delta_right"].clone()
            preds_left[..., :6] = self.normalizer_left.denormalize_labels(preds_left[..., :6])
            preds_right[..., :6] = self.normalizer_right.denormalize_labels(preds_right[..., :6])

            noisy_left, noisy_grips_left = self._diffusion_step(
                noisy_left, noisy_grips_left, preds_left, k, self.normalizer_left, arm="left"
            )
            noisy_right, noisy_grips_right = self._diffusion_step(
                noisy_right, noisy_grips_right, preds_right, k, self.normalizer_right, arm="right"
            )

        return noisy_left, torch.sign(noisy_grips_left), noisy_right, torch.sign(noisy_grips_right)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        cooldown_steps = int(self.cfg.lr_cooldown_steps)
        total_steps = int(self.cfg.num_iters)
        if cooldown_steps > 0:
            steady_steps = max(total_steps - cooldown_steps, 1)

            def lr_lambda(step: int):
                if step < steady_steps:
                    return 1.0
                progress = (step - steady_steps + 1) / max(cooldown_steps, 1)
                return max(0.0, 1.0 - progress)

            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", "frequency": 1}]

        if self.cfg.use_lr_scheduler:
            lr_scheduler = get_scheduler(
                name="cosine",
                optimizer=optimizer,
                num_warmup_steps=self.cfg.num_warmup_steps,
                num_training_steps=self.cfg.num_iters,
            )
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", "frequency": 1}]
        return optimizer


# Backward-compatible alias used by the current smoke script/imports.
BimanualGraphDiffusionScaffold = BimanualGraphDiffusion
