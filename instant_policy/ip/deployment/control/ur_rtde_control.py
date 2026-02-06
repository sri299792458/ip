import time
from typing import Optional

import numpy as np
import rtde_control
from scipy.spatial.transform import Rotation

from ip.deployment.config import GripperConfig, RTDEControlConfig
from ip.deployment.control.robotiq_gripper import RobotiqGripper


class URRTDEControl:
    def __init__(
        self,
        rtde: "rtde_control.RTDEControlInterface",
        control_config: RTDEControlConfig,
        gripper: Optional[RobotiqGripper] = None,
        gripper_config: Optional[GripperConfig] = None,
        tcp_offset_in_code: bool = False,
        tcp_offset_m: Optional[np.ndarray] = None,
    ):
        self._rtde = rtde
        self._cfg = control_config
        self._gripper = gripper
        self._gripper_cfg = gripper_config or GripperConfig(enable=gripper is not None)
        if self._gripper_cfg.enable and self._gripper is None:
            raise ValueError("Gripper is enabled but no RobotiqGripper instance was provided.")
        self._tcp_offset_in_code = tcp_offset_in_code
        self._tcp_offset_m = tcp_offset_m

    @staticmethod
    def connect(robot_ip: str, control_config: RTDEControlConfig) -> "rtde_control.RTDEControlInterface":
        return rtde_control.RTDEControlInterface(robot_ip, control_config.frequency_hz)

    def execute_pose(self, T_w_e: np.ndarray) -> bool:
        if self._tcp_offset_in_code and self._tcp_offset_m is not None:
            T_offset = np.eye(4)
            T_offset[:3, 3] = self._tcp_offset_m
            T_w_e = T_w_e @ np.linalg.inv(T_offset)
        position = T_w_e[:3, 3]
        rotvec = Rotation.from_matrix(T_w_e[:3, :3]).as_rotvec()
        pose = list(position) + list(rotvec)

        mode = self._cfg.control_mode.lower()
        if mode == "servol":
            self._rtde.servoL(
                pose,
                self._cfg.servo_speed,
                self._cfg.servo_acceleration,
                self._cfg.servo_time,
                self._cfg.servo_lookahead,
                self._cfg.servo_gain,
            )
            if self._cfg.servo_time > 0:
                time.sleep(self._cfg.servo_time)
        else:
            self._rtde.moveL(
                pose,
                self._cfg.move_speed,
                self._cfg.move_acceleration,
            )
        return True

    def execute_gripper(self, command: float) -> None:
        if not self._gripper_cfg.enable:
            return
        if self._gripper is None:
            raise RuntimeError("Gripper command requested but RobotiqGripper is not available.")
        # Convention: command >= 0.5 means OPEN, < 0.5 means CLOSED.
        if command > 0.5:
            self._gripper.open(speed=self._gripper_cfg.speed, force=self._gripper_cfg.force)
        else:
            self._gripper.close(speed=self._gripper_cfg.speed, force=self._gripper_cfg.force)

    def enable_freedrive(self) -> None:
        self._rtde.freedriveMode()

    def disable_freedrive(self) -> None:
        self._rtde.endFreedriveMode()
