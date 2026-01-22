from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class SafetyLimits:
    workspace_min: np.ndarray = field(default_factory=lambda: np.array([0.2, -0.4, 0.05]))
    workspace_max: np.ndarray = field(default_factory=lambda: np.array([0.7, 0.4, 0.5]))
    max_translation: float = 0.01
    max_rotation: float = np.deg2rad(3.0)


class ActionExecutor:
    def __init__(self, control, state, safety: SafetyLimits = None):
        self.control = control
        self.state = state
        self.safety = safety or SafetyLimits()

    def execute_actions(
        self,
        actions: np.ndarray,
        grips: np.ndarray,
        T_w_e_initial: np.ndarray,
        horizon: int = 8,
    ) -> Tuple[bool, int, str]:
        # Each action is relative to the pose at inference time (T_w_e_initial).
        T_w_e_base = T_w_e_initial.copy()
        T_w_e = T_w_e_base.copy()
        steps = min(horizon, len(actions))

        for j in range(steps):
            T_w_e_target = T_w_e_base @ actions[j]
            ok, reason = self._check_safety(T_w_e, T_w_e_target)
            if not ok:
                return False, j, reason

            if not self.control.execute_pose(T_w_e_target):
                return False, j, "Motion execution failed"

            grip_cmd = (grips[j] + 1) / 2
            self.control.execute_gripper(grip_cmd)
            T_w_e = T_w_e_target

        return True, steps, "Success"

    def _check_safety(self, T_prev: np.ndarray, T_next: np.ndarray) -> Tuple[bool, str]:
        pos = T_next[:3, 3]
        if not np.all((pos >= self.safety.workspace_min) & (pos <= self.safety.workspace_max)):
            return False, "Target outside workspace bounds"

        trans = np.linalg.norm(T_next[:3, 3] - T_prev[:3, 3])
        rot = Rotation.from_matrix(T_prev[:3, :3].T @ T_next[:3, :3]).magnitude()
        if trans > self.safety.max_translation:
            return False, "Translation exceeds per-step limit"
        if rot > self.safety.max_rotation:
            return False, "Rotation exceeds per-step limit"
        return True, ""
