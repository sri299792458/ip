from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np
import pyrealsense2 as rs

from ip.deployment.spark_teleop.config import CameraStreamConfig


@dataclass
class CameraFrame:
    rgb: np.ndarray
    depth: Optional[np.ndarray]


class RealSenseCamera:
    def __init__(self, config: CameraStreamConfig):
        self.config = config
        self._pipeline = rs.pipeline()
        self._cfg = rs.config()
        self._cfg.enable_device(config.serial)
        self._cfg.enable_stream(
            rs.stream.depth,
            config.width,
            config.height,
            rs.format.z16,
            config.fps,
        )
        self._cfg.enable_stream(
            rs.stream.color,
            config.width,
            config.height,
            rs.format.bgr8,
            config.fps,
        )
        self._pipeline.start(self._cfg)
        self._align = rs.align(rs.stream.color)

    def get_frame(self, include_depth: bool = True) -> Optional[CameraFrame]:
        frames = self._pipeline.wait_for_frames()
        aligned = self._align.process(frames)
        color = aligned.get_color_frame()
        if not color:
            return None
        color_bgr = np.asanyarray(color.get_data())
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        depth_arr = None
        if include_depth:
            depth = aligned.get_depth_frame()
            if depth:
                depth_arr = np.asanyarray(depth.get_data())
        return CameraFrame(rgb=rgb, depth=depth_arr)

    def stop(self) -> None:
        self._pipeline.stop()


class CameraManager:
    def __init__(self, cameras: list[CameraStreamConfig]):
        self._cams: Dict[str, RealSenseCamera] = {}
        for cam_cfg in cameras:
            if not cam_cfg.enabled:
                continue
            self._cams[cam_cfg.role] = RealSenseCamera(cam_cfg)

    def capture_all(self, include_depth: bool) -> dict:
        out = {}
        for role, cam in self._cams.items():
            frame = cam.get_frame(include_depth=include_depth)
            if frame is None:
                continue
            out[role] = {"rgb": frame.rgb, "depth": frame.depth}
        return out

    def stop(self) -> None:
        for cam in self._cams.values():
            try:
                cam.stop()
            except Exception:
                pass
