import time
from typing import Optional

import numpy as np

from ip.deployment.zeus_env import ensure_zeus_on_path

ensure_zeus_on_path()
from common.move_group_interface import MoveGroupInterface, transform_from_pose

try:
    import rospy
    from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_input as inputMsg
except Exception as exc:  # pragma: no cover - ROS runtime dependency
    rospy = None
    inputMsg = None
    _ROS_IMPORT_ERROR = exc
else:
    _ROS_IMPORT_ERROR = None


class ZeusState:
    def __init__(self, move_group: MoveGroupInterface, arm: str = "lightning"):
        if rospy is None:
            raise ImportError(f"rospy is required: {_ROS_IMPORT_ERROR}")
        self.mg = move_group
        self.arm = arm
        self.move_group = self.mg.mg_lightning if arm == "lightning" else self.mg.mg_thunder
        self._gripper_pos = None
        self._gripper_stamp = None
        topic = "Robotiq2FGripperRobotInput" if arm == "lightning" else "/right/Robotiq2FGripperRobotInput"
        self._gripper_sub = rospy.Subscriber(topic, inputMsg.Robotiq2FGripper_robot_input, self._on_gripper)

    def _on_gripper(self, msg):
        self._gripper_pos = msg.gPO
        self._gripper_stamp = time.time()

    def get_T_w_e(self) -> np.ndarray:
        pose = self.move_group.get_current_pose().pose
        return transform_from_pose(pose)

    def get_gripper_state(self, default: float = 0.5, max_age_s: float = 0.5) -> float:
        if self._gripper_pos is None or self._gripper_stamp is None:
            return default
        if (time.time() - self._gripper_stamp) > max_age_s:
            return default
        return float(self._gripper_pos) / 255.0
