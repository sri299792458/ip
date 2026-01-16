from typing import List

import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

from ip.generation.waypoint_sampler import Waypoint


class TrajectoryInterpolator:
    def __init__(self, trans_spacing: float, rot_spacing_deg: float):
        self.trans_spacing = float(trans_spacing)
        self.rot_spacing = np.deg2rad(rot_spacing_deg)

    def _slerp_vectors(self, v0, v1, t):
        v0 = v0 / (np.linalg.norm(v0) + 1e-8)
        v1 = v1 / (np.linalg.norm(v1) + 1e-8)
        dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
        if dot > 0.9995:
            v = (1.0 - t) * v0 + t * v1
            return v / (np.linalg.norm(v) + 1e-8)
        theta = np.arccos(dot)
        sin_t = np.sin(theta)
        w0 = np.sin((1.0 - t) * theta) / sin_t
        w1 = np.sin(t * theta) / sin_t
        return w0 * v0 + w1 * v1

    def _interp_pos(self, p0, p1, t, method):
        if method == "cubic":
            t = 3.0 * t * t - 2.0 * t * t * t
            return (1.0 - t) * p0 + t * p1
        if method == "spherical":
            center = (p0 + p1) * 0.5
            v0 = p0 - center
            v1 = p1 - center
            if np.linalg.norm(v0) < 1e-6 or np.linalg.norm(v1) < 1e-6:
                return (1.0 - t) * p0 + t * p1
            u = self._slerp_vectors(v0, v1, t)
            r = (1.0 - t) * np.linalg.norm(v0) + t * np.linalg.norm(v1)
            return center + r * u
        return (1.0 - t) * p0 + t * p1

    def _interp_rot(self, R0, R1, t):
        slerp = Slerp([0.0, 1.0], Rot.from_matrix([R0, R1]))
        return slerp([t]).as_matrix()[0]

    def _segment(self, w0: Waypoint, w1: Waypoint, method: str):
        p0 = w0.pose[:3, 3]
        p1 = w1.pose[:3, 3]
        R0 = w0.pose[:3, :3]
        R1 = w1.pose[:3, :3]
        trans_dist = np.linalg.norm(p1 - p0)
        rot_dist = Rot.from_matrix(R0.T @ R1).magnitude()
        steps = max(
            int(np.ceil(trans_dist / self.trans_spacing)),
            int(np.ceil(rot_dist / self.rot_spacing)),
            1,
        )
        ts = np.linspace(0.0, 1.0, steps + 1)
        segment = []
        for i, t in enumerate(ts):
            pos = self._interp_pos(p0, p1, t, method)
            rot = self._interp_rot(R0, R1, t)
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = rot.astype(np.float32)
            pose[:3, 3] = pos.astype(np.float32)
            grip = w0.gripper_state if i < len(ts) - 1 else w1.gripper_state
            segment.append(Waypoint(pose=pose, gripper_state=int(grip)))
        return segment

    def interpolate(self, waypoints: List[Waypoint], method: str = "linear") -> List[Waypoint]:
        if len(waypoints) < 2:
            return waypoints
        trajectory: List[Waypoint] = []
        for i in range(len(waypoints) - 1):
            segment = self._segment(waypoints[i], waypoints[i + 1], method)
            if i > 0:
                segment = segment[1:]
            trajectory.extend(segment)
        return trajectory
