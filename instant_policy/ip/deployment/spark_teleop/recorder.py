from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import pickle
import threading
import time
from typing import Any, Optional

import numpy as np

from ip.deployment.spark_teleop.config import SparkTeleopConfig
from ip.deployment.spark_teleop.controller import SparkTeleopRuntime


@dataclass
class RecorderStats:
    num_frames: int = 0
    start_time: float = 0.0
    end_time: float = 0.0


class SparkDemoRecorder:
    def __init__(
        self,
        runtime: SparkTeleopRuntime,
        config: SparkTeleopConfig,
        out_path: str,
        lang_instruction: str = "",
        camera_manager: Optional[Any] = None,
    ):
        self.runtime = runtime
        self.config = config
        self.out_path = Path(out_path)
        self.lang_instruction = lang_instruction
        self.camera_manager = camera_manager
        self._frames = []
        self._stats = RecorderStats()
        self._thread = None
        self._stop = threading.Event()

    def _capture_once(self) -> None:
        t = time.time()
        snapshot = self.runtime.get_snapshot()
        cameras = {}
        if self.camera_manager is not None and self.config.recorder.include_cameras:
            cameras = self.camera_manager.capture_all(include_depth=self.config.recorder.include_depth)

        frame = {
            "timestamp": t,
            "lang_instruction": self.lang_instruction,
            "arms": {},
            "cameras": cameras,
        }
        for arm, telem in snapshot.items():
            arm_entry = {
                "enabled": bool(telem.enabled),
                "spark_enable": bool(telem.spark_sample.enable_switch) if telem.spark_sample is not None else None,
                "spark_angles_rad": (
                    np.asarray(telem.spark_sample.angles_rad, dtype=np.float64)
                    if telem.spark_sample is not None
                    else None
                ),
                "spark_raw_values": (
                    np.asarray(telem.spark_sample.raw_values, dtype=np.int64)
                    if telem.spark_sample is not None
                    else None
                ),
                "spark_device_id": telem.spark_sample.device_id if telem.spark_sample is not None else None,
                "command_joints_rad": (
                    np.asarray(telem.command_joints, dtype=np.float64)
                    if telem.command_joints is not None
                    else None
                ),
                "command_gripper_closed": (
                    float(telem.command_gripper_closed)
                    if telem.command_gripper_closed is not None
                    else None
                ),
                "stale": bool(telem.stale),
                "error": telem.last_error,
            }
            if telem.robot_snapshot is not None:
                arm_entry["joint_positions"] = np.asarray(
                    telem.robot_snapshot.joint_positions, dtype=np.float64
                )
                arm_entry["tcp_pose"] = np.asarray(telem.robot_snapshot.tcp_pose, dtype=np.float64)
                arm_entry["ft"] = np.asarray(telem.robot_snapshot.ft, dtype=np.float64)
                arm_entry["gripper_pos"] = (
                    None if telem.robot_snapshot.gripper_pos is None else float(telem.robot_snapshot.gripper_pos)
                )
            frame["arms"][arm] = arm_entry

        self._frames.append(frame)
        self._stats.num_frames += 1

    def _run(self) -> None:
        self._stats.start_time = time.time()
        period = 1.0 / max(self.config.recorder.frame_rate_hz, 1e-6)
        while not self._stop.is_set():
            t0 = time.time()
            self._capture_once()
            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)
        self._stats.end_time = time.time()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> RecorderStats:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.save()
        return self._stats

    def save(self) -> None:
        payload = {
            "meta": {
                "schema_version": "spark_teleop_v1",
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "arms": list(self.runtime.get_arm_names()),
                "lang_instruction": self.lang_instruction,
                "num_frames": self._stats.num_frames,
            },
            "frames": self._frames,
        }
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
