from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ip.generation.config import CameraConfig, default_cameras


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
    # Core assets (single-arm parity): ShapeNet scene + Robotiq mesh.
    shapenet_path: str = "./data/shapenet"
    shapenet_index_path: Optional[str] = None
    gripper_mesh_path: Optional[str] = None

    save_dir: str = "./data/pseudo_bimanual"
    num_samples: int = 10000
    seed: int = 0

    # Trajectory/sample layout.
    pred_horizon: int = 8
    min_steps: int = 14
    max_steps: int = 24
    num_points: int = 2048
    pcd_storage_dtype: str = "float32"  # float32|float16

    # Scene sampling (ShapeNet scene builder).
    workspace_bounds: np.ndarray = field(
        default_factory=lambda: np.array(
            [[-0.3, 0.3], [-0.3, 0.3], [0.0, 0.5]], dtype=np.float32
        )
    )
    table_height: float = 0.0
    object_scale_range: Tuple[float, float] = (0.2, 0.3)
    num_objects_range: Tuple[int, int] = (3, 5)
    max_meshes: Optional[int] = None
    cache_meshes: bool = False
    surface_sample_count: int = 512

    # Camera rendering (same philosophy as single-arm pseudo pipeline).
    cameras: List[CameraConfig] = field(default_factory=default_cameras)
    render_downsample_voxel: float = 0.01
    max_points_per_obs: Optional[int] = None

    save_renders: bool = False
    render_dir: Optional[str] = None
    render_stride: int = 1
    render_visual_camera: int = 0
    render_visual_width: int = 640
    render_visual_height: int = 640
    render_save_depth: bool = False
    render_make_videos: bool = False
    render_video_dir: Optional[str] = None
    render_video_fps: int = 15
    render_video_ext: str = "mp4"

    # Motion resolution.
    trans_spacing: float = 0.01
    rot_spacing_deg: float = 3.0

    # Dual-arm attachment thresholds using gripper mesh capture region.
    attach_capture_min_points: int = 3

    # Approximate arm homes in world frame.
    left_home: np.ndarray = field(
        default_factory=lambda: np.array([-0.20, 0.22, 0.27], dtype=np.float32)
    )
    right_home: np.ndarray = field(
        default_factory=lambda: np.array([0.20, 0.22, 0.27], dtype=np.float32)
    )

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
        if not self.shapenet_path:
            raise ValueError("shapenet_path is required")
        if not self.gripper_mesh_path:
            raise ValueError("gripper_mesh_path is required")

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

        if self.object_scale_range[0] <= 0.0 or self.object_scale_range[1] < self.object_scale_range[0]:
            raise ValueError("object_scale_range must be positive and ordered")
        if self.num_objects_range[0] < 1 or self.num_objects_range[1] < self.num_objects_range[0]:
            raise ValueError("num_objects_range must be ordered and >= 1")
        if self.surface_sample_count < 64:
            raise ValueError("surface_sample_count must be >= 64")

        if self.render_stride < 1:
            raise ValueError("render_stride must be >= 1")
        if self.render_visual_width < 64 or self.render_visual_height < 64:
            raise ValueError("render visual size is too small")
        if self.render_video_fps < 1:
            raise ValueError("render_video_fps must be >= 1")

        if self.attach_capture_min_points < 1:
            raise ValueError("attach_capture_min_points must be >= 1")

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
