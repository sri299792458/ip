import numpy as np

from ip.deployment.zeus_env import ensure_zeus_on_path

ensure_zeus_on_path()
from common.move_group_interface import MoveGroupInterface, pose_from_transform


class ZeusControl:
    def __init__(self, move_group: MoveGroupInterface, arm: str = "lightning"):
        self.mg = move_group
        self.arm = arm
        self.move_group_cmd = self.mg.mg_lightning if arm == "lightning" else self.mg.mg_thunder

    def execute_pose(
        self,
        T_w_e: np.ndarray,
        speed: float = 0.1,
        min_fraction: float = 0.9,
        max_tries: int = 5,
    ) -> bool:
        pose_goal = pose_from_transform(T_w_e)
        if speed > 0:
            eef = self.move_group_cmd.get_end_effector_link()
            self.move_group_cmd.limit_max_cartesian_link_speed(speed, eef)
        try:
            for _ in range(max_tries):
                plan, fraction = self.mg.plan_cartesian_path([pose_goal], self.move_group_cmd)
                if fraction >= min_fraction:
                    self.mg.execute_plan(plan, self.move_group_cmd, wait=True)
                    return True
            return False
        finally:
            try:
                self.move_group_cmd.clear_max_cartesian_link_speed()
            except Exception:
                pass

    def execute_gripper(self, command: float, wait: bool = False) -> None:
        if command > 0.5:
            self.mg.open_gripper(left=(self.arm == "lightning"))
        else:
            self.mg.close_gripper(left=(self.arm == "lightning"))
        if wait:
            self.mg.gripper_wait()
