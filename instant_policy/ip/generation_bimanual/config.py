from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


PERACT2_BIMANUAL_TASKS: List[str] = [
    "bimanual_push_box",
    "bimanual_lift_ball",
    "bimanual_dual_push_buttons",
    "bimanual_pick_plate",
    "bimanual_put_item_in_drawer",
    "bimanual_put_bottle_in_fridge",
    "bimanual_handover_item",
    "bimanual_pick_laptop",
    "bimanual_straighten_rope",
    "bimanual_sweep_to_dustpan",
    "bimanual_lift_tray",
    "bimanual_handover_item_easy",
    "bimanual_take_tray_out_of_oven",
]

# Benchmark-level variation counts (sum=23).
PERACT2_BIMANUAL_VARIATIONS: Dict[str, int] = {
    "bimanual_push_box": 1,
    "bimanual_lift_ball": 1,
    "bimanual_dual_push_buttons": 5,
    "bimanual_pick_plate": 1,
    "bimanual_put_item_in_drawer": 3,
    "bimanual_put_bottle_in_fridge": 1,
    "bimanual_handover_item": 5,
    "bimanual_pick_laptop": 1,
    "bimanual_straighten_rope": 1,
    "bimanual_sweep_to_dustpan": 1,
    "bimanual_lift_tray": 1,
    "bimanual_handover_item_easy": 1,
    "bimanual_take_tray_out_of_oven": 1,
}


@dataclass
class BimanualGenerationConfig:
    save_dir: str = "./data/pseudo_bimanual"
    num_samples: int = 10000
    seed: int = 0

    # Trajectory/sample layout.
    pred_horizon: int = 8
    min_steps: int = 14
    max_steps: int = 24
    num_points: int = 2048
    pcd_storage_dtype: str = "float32"  # float32|float16

    # Workspace and home poses.
    workspace_bounds: np.ndarray = field(
        default_factory=lambda: np.array(
            [[-0.35, 0.35], [-0.35, 0.35], [0.0, 0.45]], dtype=np.float32
        )
    )
    table_height: float = 0.0
    left_home: np.ndarray = field(default_factory=lambda: np.array([-0.20, 0.22, 0.27], dtype=np.float32))
    right_home: np.ndarray = field(default_factory=lambda: np.array([0.20, 0.22, 0.27], dtype=np.float32))

    # Task sampling.
    task_names: List[str] = field(default_factory=lambda: list(PERACT2_BIMANUAL_TASKS))
    task_weights: Optional[List[float]] = None
    forced_task: Optional[str] = None

    # Save behavior.
    task_start: int = 0
    append: bool = False
    buffer_size: Optional[int] = None
    fill_buffer: bool = False
    shard_id: int = 0
    num_shards: int = 1

    def validate(self) -> None:
        if self.pred_horizon < 1:
            raise ValueError("pred_horizon must be >= 1")
        if self.min_steps < self.pred_horizon + 1:
            raise ValueError("min_steps must be at least pred_horizon + 1")
        if self.max_steps < self.min_steps:
            raise ValueError("max_steps must be >= min_steps")
        if self.num_points < 128:
            raise ValueError("num_points is too small for stable scene coverage")
        if self.pcd_storage_dtype not in ("float32", "float16"):
            raise ValueError("pcd_storage_dtype must be float32 or float16")
        if self.forced_task is not None and self.forced_task not in PERACT2_BIMANUAL_TASKS:
            raise ValueError(f"Unknown forced_task={self.forced_task}")
        if self.task_weights is not None:
            if len(self.task_weights) != len(self.task_names):
                raise ValueError("task_weights length must match task_names")
            if float(np.sum(self.task_weights)) <= 0.0:
                raise ValueError("task_weights must sum to > 0")
        if self.num_shards < 1:
            raise ValueError("num_shards must be >= 1")
        if not (0 <= self.shard_id < self.num_shards):
            raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards")
        if self.fill_buffer and self.buffer_size is None:
            raise ValueError("fill_buffer requires buffer_size")
