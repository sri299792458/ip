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
    def __init__(
        self,
        rtde: "rtde_receive.RTDEReceiveInterface",
        gripper: Optional[RobotiqGripper] = None,
        tcp_offset_in_code: bool = False,
        tcp_offset_m: Optional[np.ndarray] = None,
    ):
        if rtde_receive is None:
            raise ImportError(f"ur_rtde is required: {_RTDE_IMPORT_ERROR}")
        self._rtde = rtde
        self._gripper = gripper
        self._tcp_offset_in_code = tcp_offset_in_code
        self._tcp_offset_m = tcp_offset_m

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
        if self._tcp_offset_in_code and self._tcp_offset_m is not None:
            T_offset = np.eye(4)
            T_offset[:3, 3] = self._tcp_offset_m
            T = T @ T_offset
        return T

    def get_gripper_state(self, default: float = 0.5) -> float:
        if self._gripper is None:
            return default
        pos = self._gripper.get_position_normalized()
        if pos is None:
            return default
        # Map Robotiq normalized position (0=open, 1=closed) to model convention (1=open, 0=closed).
        return float(1.0 - pos)

    def get_gripper_obj_state(self, require: bool = False) -> Optional[int]:
        if self._gripper is None:
            if require:
                raise RuntimeError("Robotiq gripper is not available for OBJ feedback.")
            return None
        try:
            obj = self._gripper.get_object_status()
        except Exception:
            obj = None
        if obj is None and require:
            raise RuntimeError("Robotiq OBJ feedback is required but missing.")
        return None if obj is None else int(obj)
