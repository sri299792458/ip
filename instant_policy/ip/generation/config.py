from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from ip.generation.geometry import look_at


@dataclass
class CameraConfig:
    pose: np.ndarray
    fx: float = 525.0
    fy: float = 525.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480
    z_near: float = 0.01
    z_far: float = 2.5


def default_cameras():
    target = np.array([0.0, 0.0, 0.15], dtype=np.float32)
    cams = [
        CameraConfig(pose=look_at([0.7, 0.0, 0.6], target)),
        CameraConfig(pose=look_at([0.0, -0.7, 0.6], target)),
        CameraConfig(pose=look_at([0.0, 0.7, 0.6], target)),
    ]
    return cams


@dataclass
class GenerationConfig:
    shapenet_path: str = "./data/shapenet"
    shapenet_index_path: Optional[str] = None
    save_dir: str = "./data/pseudo_demos"

    num_tasks: int = 100000
    num_demos_per_task: Tuple[int, int] = (3, 5)
    num_context_demos: int = 2
    randomize_num_demos: bool = True
    num_context_range: Tuple[int, int] = (1, 5)
    num_waypoints_range: Tuple[int, int] = (2, 6)
    bias_prob: float = 0.5
    num_objects_range: Tuple[int, int] = (2, 2)

    workspace_bounds: np.ndarray = field(default_factory=lambda: np.array([
        [-0.3, 0.3],
        [-0.3, 0.3],
        [0.0, 0.5],
    ], dtype=np.float32))
    gripper_bounds: np.ndarray = field(default_factory=lambda: np.array([
        [-0.25, 0.25],
        [-0.25, 0.25],
        [0.05, 0.4],
    ], dtype=np.float32))
    table_height: float = 0.0
    object_scale_range: Tuple[float, float] = (0.05, 0.15)
    max_meshes: Optional[int] = None
    cache_meshes: bool = False

    trans_spacing: float = 0.01
    rot_spacing_deg: float = 3.0
    interpolation_methods: Tuple[str, ...] = ("linear", "cubic", "spherical")

    disturbance_prob: float = 0.3
    gripper_noise_prob: float = 0.1
    pose_noise_std: float = 0.002
    rot_noise_deg: float = 1.0
    object_pose_noise_std: float = 0.05
    object_yaw_noise_deg: float = 45.0

    attach_on_grasp: bool = True
    attach_radius: float = 0.06

    cameras: List[CameraConfig] = field(default_factory=default_cameras)
    render_downsample_voxel: float = 0.01
    max_points_per_obs: Optional[int] = None

    num_waypoints_demo: int = 10
    pred_horizon: int = 8
    num_points: int = 2048
    live_spacing_trans: float = 0.01
    live_spacing_rot: float = 3.0
    subsample_live: bool = False

    seed: int = 0
