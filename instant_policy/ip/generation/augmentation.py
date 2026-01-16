from typing import List

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from ip.generation.waypoint_sampler import Waypoint


class TrajectoryAugmenter:
    def __init__(
        self,
        disturbance_prob: float,
        gripper_noise_prob: float,
        pose_noise_std: float,
        rot_noise_deg: float,
    ):
        self.disturbance_prob = disturbance_prob
        self.gripper_noise_prob = gripper_noise_prob
        self.pose_noise_std = pose_noise_std
        self.rot_noise_deg = rot_noise_deg

    def _copy_traj(self, traj: List[Waypoint]) -> List[Waypoint]:
        return [Waypoint(pose=np.array(w.pose, copy=True), gripper_state=int(w.gripper_state)) for w in traj]

    def _inject_disturbance(self, traj: List[Waypoint], rng: np.random.Generator):
        if len(traj) < 6:
            return traj
        center = int(rng.integers(len(traj) // 4, 3 * len(traj) // 4))
        window = int(rng.integers(3, 7))
        trans_mag = float(rng.uniform(0.02, 0.05))
        rot_mag = float(rng.uniform(5.0, 15.0))
        trans_dir = rng.normal(size=3)
        trans_dir = trans_dir / (np.linalg.norm(trans_dir) + 1e-8)
        rot_axis = rng.normal(size=3)
        rot_axis = rot_axis / (np.linalg.norm(rot_axis) + 1e-8)
        for i in range(len(traj)):
            dist = abs(i - center)
            if dist > window:
                continue
            w = traj[i]
            weight = np.exp(-0.5 * (dist / max(1, window)) ** 2)
            w.pose[:3, 3] += trans_dir * trans_mag * weight
            rot = Rot.from_rotvec(np.deg2rad(rot_mag) * rot_axis * weight)
            R = Rot.from_matrix(w.pose[:3, :3])
            w.pose[:3, :3] = (rot * R).as_matrix()
        return traj

    def _add_gripper_noise(self, traj: List[Waypoint], rng: np.random.Generator):
        if len(traj) == 0:
            return traj
        num_flips = int(rng.integers(1, min(4, len(traj) + 1)))
        indices = rng.choice(len(traj), size=num_flips, replace=False)
        for idx in indices:
            traj[idx].gripper_state = 1 - traj[idx].gripper_state
        return traj

    def _perturb_poses(self, traj: List[Waypoint], rng: np.random.Generator):
        for w in traj:
            w.pose[:3, 3] += rng.normal(scale=self.pose_noise_std, size=3)
            rot_noise = Rot.from_euler("xyz", rng.normal(scale=np.deg2rad(self.rot_noise_deg), size=3))
            R = Rot.from_matrix(w.pose[:3, :3])
            w.pose[:3, :3] = (rot_noise * R).as_matrix()
        return traj

    def augment(self, traj: List[Waypoint], rng: np.random.Generator) -> List[Waypoint]:
        aug = self._copy_traj(traj)
        if rng.uniform(0.0, 1.0) < self.disturbance_prob:
            aug = self._inject_disturbance(aug, rng)
        if rng.uniform(0.0, 1.0) < self.gripper_noise_prob:
            aug = self._add_gripper_noise(aug, rng)
        aug = self._perturb_poses(aug, rng)
        return aug
