from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class SafetyLimits:
    max_translation: float = 0.01
    max_rotation: float = np.deg2rad(3.0)


class ActionExecutor:
    def __init__(self, control, state, safety: SafetyLimits = None, debug_gripper: bool = False):
        self.control = control
        self.state = state
        self.safety = safety or SafetyLimits()
        self._debug_gripper = debug_gripper

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
        steps_executed = 0
        last_grip_state = None

        for j in range(steps):
            T_w_e_target = T_w_e_base @ actions[j]
            grip_cmd = (grips[j] + 1) / 2
            desired = 1 if grip_cmd >= 0.5 else 0
            if self._debug_gripper and (last_grip_state is None or desired != last_grip_state):
                state_label = "open" if desired == 1 else "close"
                print(f"[gripper] step {j}: cmd={grip_cmd:.3f} -> {state_label}")
            last_grip_state = desired
            self.control.execute_gripper(grip_cmd)
            trans_err = T_w_e_target[:3, 3] - T_w_e[:3, 3]
            rotvec = Rotation.from_matrix(T_w_e[:3, :3].T @ T_w_e_target[:3, :3]).as_rotvec()
            trans = np.linalg.norm(trans_err)
            rot = np.linalg.norm(rotvec)

            max_trans = self.safety.max_translation
            max_rot = self.safety.max_rotation
            n_steps = 1
            if trans > max_trans or rot > max_rot:
                trans_ratio = trans / max_trans if max_trans > 0 else 1.0
                rot_ratio = rot / max_rot if max_rot > 0 else 1.0
                n_steps = int(np.ceil(max(trans_ratio, rot_ratio)))

            step_trans = trans_err / n_steps
            step_rot = rotvec / n_steps

            for _ in range(n_steps):
                T_w_e_next = np.eye(4, dtype=np.float64)
                T_w_e_next[:3, :3] = T_w_e[:3, :3] @ Rotation.from_rotvec(step_rot).as_matrix()
                T_w_e_next[:3, 3] = T_w_e[:3, 3] + step_trans

                ok, reason = self._check_safety(T_w_e, T_w_e_next)
                if not ok:
                    return False, steps_executed, reason

                if not self.control.execute_pose(T_w_e_next):
                    return False, steps_executed, "Motion execution failed"

                steps_executed += 1
                T_w_e = T_w_e_next

        return True, steps_executed, "Success"

    def _check_safety(self, T_prev: np.ndarray, T_next: np.ndarray) -> Tuple[bool, str]:
        trans = np.linalg.norm(T_next[:3, 3] - T_prev[:3, 3])
        rot = Rotation.from_matrix(T_prev[:3, :3].T @ T_next[:3, :3]).magnitude()
        if trans > self.safety.max_translation:
            return False, "Translation exceeds per-step limit"
        if rot > self.safety.max_rotation:
            return False, "Rotation exceeds per-step limit"
        return True, ""

    def _bounded_step(self, T_prev: np.ndarray, T_target: np.ndarray) -> Tuple[np.ndarray, bool]:
        raise NotImplementedError("Use execute_actions() splitting instead of _bounded_step().")
