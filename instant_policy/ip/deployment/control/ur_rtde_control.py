import time
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import rtde_control
except Exception as exc:  # pragma: no cover - optional dependency
    rtde_control = None
    _RTDE_IMPORT_ERROR = exc
else:
    _RTDE_IMPORT_ERROR = None

from ip.deployment.config import GripperConfig, RTDEControlConfig
from ip.deployment.ur.robotiq_gripper import RobotiqGripper


class URRTDEControl:
    def __init__(
        self,
        rtde: "rtde_control.RTDEControlInterface",
        control_config: RTDEControlConfig,
        gripper: Optional[RobotiqGripper] = None,
        gripper_config: Optional[GripperConfig] = None,
    ):
        if rtde_control is None:
            raise ImportError(f"ur_rtde is required: {_RTDE_IMPORT_ERROR}")
        self._rtde = rtde
        self._cfg = control_config
        self._gripper = gripper
        self._gripper_cfg = gripper_config or GripperConfig()

    @staticmethod
    def connect(robot_ip: str, control_config: RTDEControlConfig) -> "rtde_control.RTDEControlInterface":
        if rtde_control is None:
            raise ImportError(f"ur_rtde is required: {_RTDE_IMPORT_ERROR}")
        return rtde_control.RTDEControlInterface(robot_ip, control_config.frequency_hz)

    def execute_pose(self, T_w_e: np.ndarray) -> bool:
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
        if self._gripper is None or not self._gripper_cfg.enable:
            return
        if command > 0.5:
            self._gripper.open(speed=self._gripper_cfg.speed, force=self._gripper_cfg.force)
        else:
            self._gripper.close(speed=self._gripper_cfg.speed, force=self._gripper_cfg.force)

    def enable_freedrive(self) -> None:
        self._rtde.freedriveMode()

    def disable_freedrive(self) -> None:
        self._rtde.endFreedriveMode()
