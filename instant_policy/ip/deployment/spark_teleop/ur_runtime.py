from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Optional

import numpy as np
import rtde_control
import rtde_receive

from ip.deployment.control.robotiq_gripper import RobotiqGripper
from ip.deployment.spark_teleop.config import ArmConfig


@dataclass
class URArmSnapshot:
    timestamp: float
    joint_positions: np.ndarray
    tcp_pose: np.ndarray
    ft: np.ndarray
    gripper_pos: Optional[float]


class URArmRuntime:
    def __init__(self, config: ArmConfig):
        self.config = config
        self._rtde_control = None
        self._rtde_receive = None
        self._gripper = None
        self._lock = threading.Lock()
        self._freedrive_enabled = False

    def connect(self) -> None:
        self._rtde_control = rtde_control.RTDEControlInterface(self.config.robot_ip, 500)
        self._rtde_receive = rtde_receive.RTDEReceiveInterface(self.config.robot_ip)
        if self.config.use_gripper:
            self._gripper = RobotiqGripper(
                host=self.config.robot_ip,
                open_position=self.config.gripper_map.open_position,
                closed_position=self.config.gripper_map.closed_position,
            )
            self._gripper.connect()
            self._gripper.activate()

    def disconnect(self) -> None:
        try:
            self.stop_motion()
        except Exception:
            pass
        with self._lock:
            if self._rtde_control is not None:
                try:
                    self._rtde_control.stopScript()
                except Exception:
                    pass
                try:
                    self._rtde_control.disconnect()
                except Exception:
                    pass
                self._rtde_control = None
            if self._rtde_receive is not None:
                try:
                    self._rtde_receive.disconnect()
                except Exception:
                    pass
                self._rtde_receive = None
            if self._gripper is not None:
                try:
                    self._gripper.disconnect()
                except Exception:
                    pass
                self._gripper = None

    def _require_control(self):
        if self._rtde_control is None:
            raise RuntimeError(f"{self.config.name}: RTDE control is not connected.")
        return self._rtde_control

    def _require_receive(self):
        if self._rtde_receive is None:
            raise RuntimeError(f"{self.config.name}: RTDE receive is not connected.")
        return self._rtde_receive

    def get_actual_q(self) -> np.ndarray:
        rx = self._require_receive()
        return np.asarray(rx.getActualQ(), dtype=np.float64).reshape(6)

    def get_snapshot(self) -> URArmSnapshot:
        rx = self._require_receive()
        joint_positions = np.asarray(rx.getActualQ(), dtype=np.float64).reshape(6)
        tcp_pose = np.asarray(rx.getActualTCPPose(), dtype=np.float64).reshape(6)
        ft = np.asarray(rx.getActualTCPForce(), dtype=np.float64).reshape(6)
        gripper_pos = None
        if self._gripper is not None:
            gripper_pos = float(self._gripper.get_position())
        return URArmSnapshot(
            timestamp=time.time(),
            joint_positions=joint_positions,
            tcp_pose=tcp_pose,
            ft=ft,
            gripper_pos=gripper_pos,
        )

    def servo_j(self, joints_rad: list[float]) -> bool:
        if len(joints_rad) != 6:
            raise ValueError(f"{self.config.name}: expected 6 joints, got {len(joints_rad)}")
        rtde = self._require_control()
        with self._lock:
            if self._freedrive_enabled:
                rtde.endFreedriveMode()
                self._freedrive_enabled = False
            result = rtde.servoJ(
                joints_rad,
                0.0,
                0.0,
                self.config.control.servo_dt_s,
                self.config.control.servo_lookahead_s,
                self.config.control.servo_gain,
            )
        if self.config.control.servo_dt_s > 0:
            time.sleep(self.config.control.servo_dt_s)
        return result is not False

    def move_home(self) -> None:
        rtde = self._require_control()
        with self._lock:
            rtde.moveJ(
                [float(x) for x in self.config.home_joint_rad],
                1.05,
                1.4,
            )

    def set_gripper_closed_norm(self, closed_norm: float) -> None:
        if self._gripper is None:
            return
        closed_norm = float(np.clip(closed_norm, 0.0, 1.0))
        lo = self.config.gripper_map.open_position
        hi = self.config.gripper_map.closed_position
        target = int(round(lo + closed_norm * (hi - lo)))
        self._gripper.move(target)

    def open_gripper(self) -> None:
        if self._gripper is not None:
            self._gripper.open()

    def close_gripper(self) -> None:
        if self._gripper is not None:
            self._gripper.close()

    def enable_freedrive(self) -> None:
        rtde = self._require_control()
        with self._lock:
            try:
                rtde.servoStop()
            except Exception:
                pass
            rtde.freedriveMode()
            self._freedrive_enabled = True

    def disable_freedrive(self) -> None:
        rtde = self._require_control()
        with self._lock:
            rtde.endFreedriveMode()
            self._freedrive_enabled = False

    def is_freedrive_enabled(self) -> bool:
        with self._lock:
            return self._freedrive_enabled

    def emergency_stop(self) -> None:
        rtde = self._require_control()
        with self._lock:
            if hasattr(rtde, "triggerProtectiveStop"):
                rtde.triggerProtectiveStop()
            else:
                rtde.stopJ(2.0)

    def stop_motion(self) -> None:
        if self._rtde_control is None:
            return
        with self._lock:
            try:
                self._rtde_control.servoStop()
            except Exception:
                pass
            try:
                self._rtde_control.speedStop()
            except Exception:
                pass
            if self._freedrive_enabled:
                try:
                    self._rtde_control.endFreedriveMode()
                except Exception:
                    pass
                self._freedrive_enabled = False
