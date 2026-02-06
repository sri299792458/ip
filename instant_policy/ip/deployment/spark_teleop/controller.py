from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Dict, Optional

import numpy as np

from ip.deployment.spark_teleop.config import ArmConfig, SparkTeleopConfig
from ip.deployment.spark_teleop.spark_serial import SparkDevice, SparkSample
from ip.deployment.spark_teleop.ur_runtime import URArmRuntime, URArmSnapshot


def _map_value(x: float, in_min: float, in_max: float, out_min: float = 0.0, out_max: float = 1.0) -> float:
    if in_max == in_min:
        return out_min
    return out_min + (x - in_min) * (out_max - out_min) / (in_max - in_min)


@dataclass
class ArmTelemetry:
    timestamp: float
    arm: str
    enabled: bool
    spark_sample: Optional[SparkSample]
    command_joints: Optional[np.ndarray]
    command_gripper_closed: Optional[float]
    robot_snapshot: Optional[URArmSnapshot]
    stale: bool
    last_error: Optional[str]


class ArmTeleopLoop:
    def __init__(self, arm_config: ArmConfig, spark: SparkDevice, runtime: URArmRuntime):
        self.arm_config = arm_config
        self.spark = spark
        self.runtime = runtime
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._enabled = True
        self._telemetry = ArmTelemetry(
            timestamp=time.time(),
            arm=arm_config.name,
            enabled=True,
            spark_sample=None,
            command_joints=None,
            command_gripper_closed=None,
            robot_snapshot=None,
            stale=True,
            last_error=None,
        )
        self._base_wrap_adjust_rad = 0.0
        self._wrap_initialized = False
        self._was_commanding = False

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = bool(enabled)

    def get_enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def get_telemetry(self) -> ArmTelemetry:
        with self._lock:
            return self._telemetry

    def _set_telemetry(self, telemetry: ArmTelemetry) -> None:
        with self._lock:
            self._telemetry = telemetry

    def _run(self) -> None:
        period = 1.0 / max(self.arm_config.control.command_rate_hz, 1e-6)
        offset = np.asarray(self.arm_config.spark_joint_offset_rad, dtype=np.float64).reshape(7)
        while not self._stop.is_set():
            t0 = time.time()
            try:
                enabled = self.get_enabled()
                spark_sample = self.spark.get_latest()
                stale = True
                command_joints = None
                command_gripper_closed = None
                freedrive = self.runtime.is_freedrive_enabled()
                if spark_sample is not None:
                    stale = (t0 - float(spark_sample.timestamp)) > self.arm_config.stale_timeout_s

                can_command = (
                    enabled
                    and not freedrive
                    and spark_sample is not None
                    and not stale
                    and spark_sample.enable_switch
                )

                if not can_command and self._was_commanding and not freedrive:
                    self.runtime.stop_motion()
                    self._was_commanding = False

                if can_command:
                    spark_angles = np.asarray(spark_sample.angles_rad, dtype=np.float64).reshape(7)
                    if not self._wrap_initialized:
                        actual_q = self.runtime.get_actual_q()
                        dq0 = float(spark_angles[0] - actual_q[0] + offset[0])
                        if dq0 > np.pi:
                            self._base_wrap_adjust_rad = -2.0 * np.pi
                        elif dq0 < -np.pi:
                            self._base_wrap_adjust_rad = 2.0 * np.pi
                        self._wrap_initialized = True

                    command_joints = spark_angles[:6] + offset[:6]
                    command_joints[0] += self._base_wrap_adjust_rad
                    grip_raw = float(spark_angles[6] + offset[6])
                    command_gripper_closed = float(
                        np.round(
                            np.clip(
                                _map_value(
                                    grip_raw,
                                    in_min=self.arm_config.gripper_map.raw_min,
                                    in_max=self.arm_config.gripper_map.raw_max,
                                    out_min=0.0,
                                    out_max=1.0,
                                ),
                                0.0,
                                1.0,
                            )
                            * 10.0
                        )
                        / 10.0
                    )
                    ok = self.runtime.servo_j(command_joints.tolist())
                    if not ok:
                        raise RuntimeError(f"{self.arm_config.name}: servoJ returned failure")
                    if self.arm_config.use_gripper:
                        self.runtime.set_gripper_closed_norm(command_gripper_closed)
                    self._was_commanding = True

                robot_snapshot = self.runtime.get_snapshot()
                self._set_telemetry(
                    ArmTelemetry(
                        timestamp=t0,
                        arm=self.arm_config.name,
                        enabled=enabled,
                        spark_sample=spark_sample,
                        command_joints=command_joints,
                        command_gripper_closed=command_gripper_closed,
                        robot_snapshot=robot_snapshot,
                        stale=stale,
                        last_error=self.spark.get_last_error(),
                    )
                )
            except Exception as exc:
                self._set_telemetry(
                    ArmTelemetry(
                        timestamp=t0,
                        arm=self.arm_config.name,
                        enabled=self.get_enabled(),
                        spark_sample=self.spark.get_latest(),
                        command_joints=None,
                        command_gripper_closed=None,
                        robot_snapshot=None,
                        stale=True,
                        last_error=str(exc),
                    )
                )
                try:
                    self.runtime.stop_motion()
                except Exception:
                    pass
                self._was_commanding = False

            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


class SparkTeleopRuntime:
    def __init__(self, config: SparkTeleopConfig):
        self.config = config
        self.spark_devices: Dict[str, SparkDevice] = {}
        self.arm_runtimes: Dict[str, URArmRuntime] = {}
        self.loops: Dict[str, ArmTeleopLoop] = {}

        spark_by_arm = {d.arm: d for d in config.spark_devices}
        for arm_name, arm_cfg in config.arms.items():
            if not arm_cfg.enabled:
                continue
            if arm_name not in spark_by_arm:
                raise ValueError(f"Missing Spark device config for arm '{arm_name}'")
            spark = SparkDevice(spark_by_arm[arm_name])
            ur = URArmRuntime(arm_cfg)
            loop = ArmTeleopLoop(arm_cfg, spark, ur)
            self.spark_devices[arm_name] = spark
            self.arm_runtimes[arm_name] = ur
            self.loops[arm_name] = loop

    def start(self) -> None:
        for runtime in self.arm_runtimes.values():
            runtime.connect()
        for spark in self.spark_devices.values():
            spark.start()
        for loop in self.loops.values():
            loop.start()

    def stop(self) -> None:
        for loop in self.loops.values():
            loop.stop()
        for runtime in self.arm_runtimes.values():
            try:
                runtime.stop_motion()
            except Exception:
                pass
        for spark in self.spark_devices.values():
            spark.stop()
        for runtime in self.arm_runtimes.values():
            runtime.disconnect()

    def get_arm_names(self) -> list[str]:
        return list(self.loops.keys())

    def set_arm_enabled(self, arm: str, enabled: bool) -> None:
        self.loops[arm].set_enabled(enabled)

    def get_arm_enabled(self, arm: str) -> bool:
        return self.loops[arm].get_enabled()

    def get_telemetry(self, arm: str) -> ArmTelemetry:
        return self.loops[arm].get_telemetry()

    def get_snapshot(self) -> Dict[str, ArmTelemetry]:
        return {arm: loop.get_telemetry() for arm, loop in self.loops.items()}
