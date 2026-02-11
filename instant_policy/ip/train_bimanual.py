import argparse
import os
import pickle
from glob import glob

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from ip.bimanual import (
    BimanualBackbone,
    BimanualFileDataset,
    BimanualGraphConfig,
    BimanualGraphDiffusion,
    BimanualModelConfig,
    BimanualTrainingConfig,
    collate_bimanual_world_batch,
)
from ip.configs.bimanual_config import config


def _latest_resume_checkpoint(save_dir: str):
    if not os.path.isdir(save_dir):
        return None

    last_ckpt = os.path.join(save_dir, "last.ckpt")
    if os.path.isfile(last_ckpt):
        return last_ckpt

    ckpts = sorted(glob(os.path.join(save_dir, "*.ckpt")), key=os.path.getmtime)
    if ckpts:
        return ckpts[-1]

    return None


def _str2bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", type=str, default="bimanual")
    p.add_argument("--record", type=int, default=1, help="Save checkpoints/config [0,1].")
    p.add_argument("--use_wandb", type=int, default=0, help="Enable W&B logger [0,1].")
    p.add_argument("--save_path", type=str, default="./runs_bimanual")

    p.add_argument("--data_path_train", type=str, required=True)
    p.add_argument("--data_path_val", type=str, required=True)
    p.add_argument("--train_pattern", type=str, default=None)
    p.add_argument("--val_pattern", type=str, default=None)

    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--batch_size_val", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)

    p.add_argument("--device", type=str, default=None, help="cuda or cpu")
    p.add_argument("--precision", type=str, default=None, help="e.g. 16-mixed")

    p.add_argument("--num_iters_override", type=int, default=None)
    p.add_argument("--val_check_interval", type=int, default=None)
    p.add_argument("--log_every_n_steps", type=int, default=None)
    p.add_argument("--checkpoint_every", type=int, default=None)

    p.add_argument("--resume_ckpt_path", type=str, default=None)
    p.add_argument("--auto_resume", action="store_true")

    p.add_argument("--wandb_id", type=str, default=None)
    p.add_argument("--wandb_resume", type=str, default="allow", choices=["allow", "must", "never"])

    p.add_argument("--pin_memory", type=str, default=None, help="Override dataloader pin_memory [true/false].")
    p.add_argument(
        "--persistent_workers",
        type=str,
        default=None,
        help="Override dataloader persistent_workers [true/false].",
    )
    return p.parse_args()


def main():
    args = build_args()

    cfg = dict(config)
    cfg["record"] = bool(args.record)

    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.batch_size_val is not None:
        cfg["batch_size_val"] = int(args.batch_size_val)
    if args.num_workers is not None:
        cfg["num_workers"] = int(args.num_workers)

    if args.device is not None:
        cfg["device"] = args.device
    if args.precision is not None:
        cfg["precision"] = args.precision

    if args.num_iters_override is not None:
        if int(args.num_iters_override) < 1:
            raise ValueError("--num_iters_override must be >= 1")
        cfg["num_iters"] = int(args.num_iters_override)

    if args.val_check_interval is not None:
        cfg["val_check_interval"] = int(args.val_check_interval)
    if args.log_every_n_steps is not None:
        cfg["log_every_n_steps"] = int(args.log_every_n_steps)
    if args.checkpoint_every is not None:
        cfg["checkpoint_every"] = int(args.checkpoint_every)

    if args.pin_memory is not None:
        cfg["pin_memory"] = _str2bool(args.pin_memory)
    if args.persistent_workers is not None:
        cfg["persistent_workers"] = _str2bool(args.persistent_workers)

    run_name = args.run_name
    save_dir = os.path.join(args.save_path, run_name)
    cfg["save_dir"] = save_dir if cfg["record"] else None

    if cfg["record"] or bool(args.use_wandb):
        os.makedirs(save_dir, exist_ok=True)

    resume_ckpt_path = args.resume_ckpt_path
    if args.auto_resume and resume_ckpt_path is None:
        resume_ckpt_path = _latest_resume_checkpoint(save_dir)
        if resume_ckpt_path is None:
            raise RuntimeError(f"--auto_resume requested, but no checkpoint found in {save_dir}")

    if resume_ckpt_path is not None and not os.path.isfile(resume_ckpt_path):
        raise RuntimeError(f"Resume checkpoint not found: {resume_ckpt_path}")

    graph_cfg = BimanualGraphConfig(
        hidden_dim=cfg["hidden_dim"],
        local_num_freq=cfg["local_num_freq"],
        k_scene_scene=cfg["k_scene_scene"],
        k_scene_gripper=cfg["k_scene_gripper"],
        include_gripper_self_edges=cfg["include_gripper_self_edges"],
        use_cross_edges=cfg["use_cross_edges"],
        device=cfg["device"],
    )
    model_cfg = BimanualModelConfig(
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        heads=cfg["heads"],
        pred_horizon=cfg["pred_horizon"],
        edge_dropout=cfg["edge_dropout"],
        device=cfg["device"],
    )
    train_cfg = BimanualTrainingConfig(
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        pred_horizon=cfg["pred_horizon"],
        num_diffusion_iters_train=cfg["num_diffusion_iters_train"],
        num_diffusion_iters_test=cfg["num_diffusion_iters_test"],
        use_lr_scheduler=cfg["use_lr_scheduler"],
        num_warmup_steps=cfg["num_warmup_steps"],
        lr_cooldown_steps=cfg["lr_cooldown_steps"],
        num_iters=cfg["num_iters"],
        min_actions=cfg["min_actions"],
        max_actions=cfg["max_actions"],
    )

    backbone = BimanualBackbone(graph_cfg, model_cfg)
    model = BimanualGraphDiffusion(backbone, train_cfg).to(cfg["device"])

    dset_train = BimanualFileDataset(args.data_path_train, pattern=args.train_pattern)
    dset_val = BimanualFileDataset(args.data_path_val, pattern=args.val_pattern)

    persistent_workers = bool(cfg["persistent_workers"]) and int(cfg["num_workers"]) > 0

    train_loader = DataLoader(
        dset_train,
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=cfg["num_workers"],
        pin_memory=bool(cfg["pin_memory"]),
        persistent_workers=persistent_workers,
        collate_fn=collate_bimanual_world_batch,
    )
    val_loader = DataLoader(
        dset_val,
        batch_size=cfg["batch_size_val"],
        shuffle=False,
        drop_last=False,
        num_workers=cfg["num_workers"],
        pin_memory=bool(cfg["pin_memory"]),
        persistent_workers=persistent_workers,
        collate_fn=collate_bimanual_world_batch,
    )

    logger = None
    if bool(args.use_wandb):
        wandb_kwargs = {
            "project": "Instant Policy Bimanual",
            "name": run_name,
            "save_dir": save_dir,
            "log_model": False,
            "resume": args.wandb_resume,
        }
        if args.wandb_id is not None:
            wandb_kwargs["id"] = args.wandb_id
        logger = WandbLogger(**wandb_kwargs)

    callbacks = [LearningRateMonitor(logging_interval="step")]
    if cfg["record"]:
        callbacks.append(
            ModelCheckpoint(
                dirpath=save_dir,
                filename="step_{step}",
                save_last=True,
                save_top_k=-1,
                every_n_train_steps=cfg["checkpoint_every"],
            )
        )
        with open(os.path.join(save_dir, "config.pkl"), "wb") as f:
            pickle.dump(cfg, f)

    trainer = L.Trainer(
        enable_checkpointing=cfg["record"],
        accelerator="gpu" if str(cfg["device"]).startswith("cuda") else "cpu",
        devices=1,
        max_steps=cfg["num_iters"],
        enable_progress_bar=True,
        precision=cfg["precision"],
        val_check_interval=cfg["val_check_interval"],
        num_sanity_val_steps=2,
        check_val_every_n_epoch=None,
        logger=logger,
        log_every_n_steps=cfg["log_every_n_steps"],
        gradient_clip_val=cfg["gradient_clip_val"],
        gradient_clip_algorithm="norm",
        callbacks=callbacks,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_ckpt_path,
    )


if __name__ == "__main__":
    main()
