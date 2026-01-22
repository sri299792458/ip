from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import rtde_receive
except Exception as exc:  # pragma: no cover - optional dependency
    rtde_receive = None
    _RTDE_IMPORT_ERROR = exc
else:
    _RTDE_IMPORT_ERROR = None

from ip.deployment.ur.robotiq_gripper import RobotiqGripper


class URRTDEState:
    def __init__(self, rtde: "rtde_receive.RTDEReceiveInterface", gripper: Optional[RobotiqGripper] = None):
        if rtde_receive is None:
            raise ImportError(f"ur_rtde is required: {_RTDE_IMPORT_ERROR}")
        self._rtde = rtde
        self._gripper = gripper

    @staticmethod
    def connect(robot_ip: str) -> "rtde_receive.RTDEReceiveInterface":
        if rtde_receive is None:
            raise ImportError(f"ur_rtde is required: {_RTDE_IMPORT_ERROR}")
        return rtde_receive.RTDEReceiveInterface(robot_ip)

    def get_T_w_e(self) -> np.ndarray:
        pose = self._rtde.getActualTCPPose()
        T = np.eye(4)
        T[:3, 3] = pose[:3]
        T[:3, :3] = Rotation.from_rotvec(pose[3:]).as_matrix()
        return T

    def get_gripper_state(self, default: float = 0.5) -> float:
        if self._gripper is None:
            return default
        pos = self._gripper.get_position_normalized()
        if pos is None:
            return default
        return float(pos)
