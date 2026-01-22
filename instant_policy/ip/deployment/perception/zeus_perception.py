import logging
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

try:
    import open3d as o3d
except Exception:  # pragma: no cover - optional dependency
    o3d = None

try:
    import pyrealsense2 as rs
except Exception as exc:  # pragma: no cover - optional dependency
    rs = None
    _RS_IMPORT_ERROR = exc
else:
    _RS_IMPORT_ERROR = None

from ip.deployment.perception.sam_segmentation import SAMSegmenter
from ip.deployment.zeus_env import ensure_zeus_on_path

ensure_zeus_on_path()
from common.ros_pointcloud import _get_xyz
from perception.scripts.publish_d405 import get_intrinsics, get_K


@dataclass
class _CameraHandle:
    pipeline: "rs.pipeline"
    align: Optional["rs.align"]
    K: np.ndarray
    depth_scale: float
    T_world_camera: np.ndarray


class ZeusPerception:
    def __init__(
        self,
        camera_configs: Iterable,
        segmenter: Optional[SAMSegmenter] = None,
        voxel_size: Optional[float] = None,
    ):
        if rs is None:
            raise ImportError(f"pyrealsense2 is required: {_RS_IMPORT_ERROR}")
        self._segmenter = segmenter
        self._voxel_size = voxel_size
        self._cameras = []

        for cam in camera_configs:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(cam.serial)
            config.enable_stream(rs.stream.depth, cam.width, cam.height, rs.format.z16, cam.fps)
            config.enable_stream(rs.stream.color, cam.width, cam.height, rs.format.rgb8, cam.fps)
            profile = pipeline.start(config)

            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            if cam.align_to_color:
                align = rs.align(rs.stream.color)
                color_profile = profile.get_stream(rs.stream.color)
                intr = get_intrinsics(color_profile)
            else:
                align = None
                depth_profile = profile.get_stream(rs.stream.depth)
                intr = get_intrinsics(depth_profile)
            K = get_K(intr)

            self._cameras.append(
                _CameraHandle(
                    pipeline=pipeline,
                    align=align,
                    K=K,
                    depth_scale=depth_scale,
                    T_world_camera=cam.T_world_camera,
                )
            )

    def stop(self):
        for cam in self._cameras:
            try:
                cam.pipeline.stop()
            except Exception:
                pass

    def _segment(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        if self._segmenter is None:
            return None
        return self._segmenter.segment(rgb)

    def capture_pcd_world(
        self,
        segmentation_masks: Optional[Iterable[np.ndarray]] = None,
        use_segmentation: bool = False,
    ) -> np.ndarray:
        all_points = []
        segmenter_masks = None
        if use_segmentation and segmentation_masks is None and self._segmenter is not None:
            if hasattr(self._segmenter, "get_masks"):
                segmenter_masks = self._segmenter.get_masks()
            else:
                segmenter_masks = None

        masks_iter = iter(segmentation_masks) if segmentation_masks is not None else None

        for idx, cam in enumerate(self._cameras):
            frames = cam.pipeline.wait_for_frames()
            if cam.align is not None:
                frames = cam.align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * cam.depth_scale
            color = np.asanyarray(color_frame.get_data())

            mask = next(masks_iter, None) if masks_iter is not None else None
            if mask is None and segmenter_masks is not None:
                if idx < len(segmenter_masks):
                    mask = segmenter_masks[idx]
            if mask is None and use_segmentation and self._segmenter is not None:
                if hasattr(self._segmenter, "segment_camera"):
                    mask = self._segmenter.segment_camera(color, idx)
                else:
                    mask = self._segment(color)
            if mask is not None and mask.shape != depth.shape:
                logging.warning("Segmentation mask shape mismatch, ignoring")
                mask = None
            if mask is not None:
                depth = depth * mask.astype(np.float32)

            xyz_cam = _get_xyz(depth, cam.K).T
            valid = np.isfinite(xyz_cam).all(axis=1) & (xyz_cam[:, 2] > 0)
            xyz_cam = xyz_cam[valid]
            xyz_world = (cam.T_world_camera[:3, :3] @ xyz_cam.T).T + cam.T_world_camera[:3, 3]
            all_points.append(xyz_world)

        if not all_points:
            logging.warning("No valid point clouds captured")
            return np.zeros((0, 3), dtype=np.float32)

        pcd = np.concatenate(all_points, axis=0)
        if self._voxel_size and o3d is not None:
            pcd = self._voxel_downsample(pcd, self._voxel_size)
        return pcd.astype(np.float32)

    @staticmethod
    def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
        if o3d is None:
            return points
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        down = cloud.voxel_down_sample(voxel_size)
        return np.asarray(down.points)
