import numpy as np
import torch

config = {
    # Runtime / trainer
    "device": "cuda",
    "precision": "16-mixed",
    "batch_size": 16,
    "batch_size_val": 1,
    "num_workers": 8,
    "pin_memory": True,
    "persistent_workers": True,
    "record": False,
    "save_dir": None,

    # Bimanual graph/model
    "local_num_freq": 10,
    "hidden_dim": 512,
    "num_layers": 3,
    "heads": 4,
    "pred_horizon": 8,
    "edge_dropout": 0.0,
    "k_scene_scene": 16,
    "k_scene_gripper": 6,
    "include_gripper_self_edges": True,
    "use_cross_edges": True,

    # Diffusion training
    "lr": 1e-5,
    "weight_decay": 1e-2,
    "use_lr_scheduler": False,
    "num_warmup_steps": 1000,
    "lr_cooldown_steps": 50000,
    "num_diffusion_iters_train": 100,
    "num_diffusion_iters_test": 8,
    "num_iters": 2550000,

    # Logging / validation / checkpoints
    "val_check_interval": 20000,
    "log_every_n_steps": 500,
    "gradient_clip_val": 1.0,
    "checkpoint_every": 100000,

    # Action limits (same as single-arm Instant Policy)
    "min_actions": torch.tensor(
        [-0.01, -0.01, -0.01, -np.deg2rad(3), -np.deg2rad(3), -np.deg2rad(3)],
        dtype=torch.float32,
    ),
    "max_actions": torch.tensor(
        [0.01, 0.01, 0.01, np.deg2rad(3), np.deg2rad(3), np.deg2rad(3)],
        dtype=torch.float32,
    ),
}
