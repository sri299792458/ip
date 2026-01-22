from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ip.deployment.control.action_executor import SafetyLimits


@dataclass
class CameraConfig:
    serial: str
    T_world_camera: np.ndarray
    width: int = 640
    height: int = 480
    fps: int = 30
    align_to_color: bool = True


@dataclass
class SegmentationConfig:
    enable: bool = True
    backend: str = "xmem"
    model_type: str = "vit_b"
    checkpoint_path: Optional[str] = None
    sam_checkpoint_path: Optional[str] = None
    xmem_checkpoint_path: Optional[str] = None
    xmem_init_with_sam: bool = True
    xmem_config_overrides: Optional[dict] = None
    points_per_side: int = 32
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    min_mask_region_area: int = 256
    select_largest: bool = True
    mask_topics: Optional[List[str]] = None
    mask_timeout_s: float = 0.5
    mask_threshold: float = 0.5


@dataclass
class DeploymentConfig:
    camera_configs: List[CameraConfig] = field(default_factory=list)
    model_path: str = "./checkpoints"
    num_demos: int = 2
    num_traj_wp: int = 10
    max_execution_steps: int = 100
    num_diffusion_iters: int = 4
    pcd_num_points: int = 2048
    pcd_voxel_size: Optional[float] = None
    safety: Optional[SafetyLimits] = None
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    arm: str = "lightning"
    device: Optional[str] = None
    execute_until_grip_change: bool = True
