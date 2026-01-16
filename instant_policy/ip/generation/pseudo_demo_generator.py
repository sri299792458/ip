import os
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from ip.generation.augmentation import TrajectoryAugmenter
from ip.generation.config import GenerationConfig
from ip.generation.geometry import make_transform, random_rotation, transform_points
from ip.generation.renderer import DepthRenderer
from ip.generation.scene_builder import SceneBuilder, Scene
from ip.generation.trajectory_interpolator import TrajectoryInterpolator
from ip.generation.waypoint_sampler import Waypoint, WaypointSampler
from ip.utils.common_utils import downsample_pcd
from ip.utils.data_proc import sample_to_cond_demo, sample_to_live, save_sample


class PseudoDemoGenerator:
    def __init__(self, config: GenerationConfig, scene_encoder=None):
        self.config = config
        self.scene_encoder = scene_encoder
        self.scene_builder = SceneBuilder(
            shapenet_path=config.shapenet_path,
            workspace_bounds=config.workspace_bounds,
            table_height=config.table_height,
            object_scale_range=config.object_scale_range,
            num_objects_range=config.num_objects_range,
            shapenet_index_path=config.shapenet_index_path,
            max_meshes=config.max_meshes,
            cache_meshes=config.cache_meshes,
        )
        self.waypoint_sampler = WaypointSampler(
            bias_prob=config.bias_prob,
            num_waypoints_range=config.num_waypoints_range,
        )
        self.interpolator = TrajectoryInterpolator(
            trans_spacing=config.trans_spacing,
            rot_spacing_deg=config.rot_spacing_deg,
        )
        self.augmenter = TrajectoryAugmenter(
            disturbance_prob=config.disturbance_prob,
            gripper_noise_prob=config.gripper_noise_prob,
            pose_noise_std=config.pose_noise_std,
            rot_noise_deg=config.rot_noise_deg,
        )
        self.renderer = DepthRenderer(
            cameras=config.cameras,
            downsample_voxel=config.render_downsample_voxel,
            max_points_per_obs=config.max_points_per_obs,
        )

    def _sample_start_pose(self, rng: np.random.Generator):
        bounds = self.config.gripper_bounds
        pos = rng.uniform(bounds[:, 0], bounds[:, 1])
        R = random_rotation(rng)
        return make_transform(R, pos)

    def _vary_scene(self, scene: Scene, rng: np.random.Generator):
        varied = scene.copy()
        pos_std = self.config.object_pose_noise_std
        yaw_std = np.deg2rad(self.config.object_yaw_noise_deg)
        for obj in varied.objects:
            pos_noise = rng.normal(scale=pos_std, size=3)
            pos_noise[2] = 0.0
            yaw = rng.normal(scale=yaw_std)
            Rz = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)
            obj.pose[:3, :3] = Rz @ obj.pose[:3, :3]
            obj.pose[:3, 3] += pos_noise.astype(np.float32)
        return varied

    def _perturb_waypoints(self, waypoints: List[Waypoint], rng: np.random.Generator):
        for w in waypoints:
            w.pose[:3, 3] += rng.normal(scale=0.03, size=3)
            yaw = rng.normal(scale=np.deg2rad(10.0))
            pitch = rng.normal(scale=np.deg2rad(10.0))
            roll = rng.normal(scale=np.deg2rad(10.0))
            Rn = np.array([
                [1.0, 0.0, 0.0],
                [0.0, np.cos(roll), -np.sin(roll)],
                [0.0, np.sin(roll), np.cos(roll)],
            ], dtype=np.float32)
            Rp = np.array([
                [np.cos(pitch), 0.0, np.sin(pitch)],
                [0.0, 1.0, 0.0],
                [-np.sin(pitch), 0.0, np.cos(pitch)],
            ], dtype=np.float32)
            Ry = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)
            w.pose[:3, :3] = (Ry @ Rp @ Rn) @ w.pose[:3, :3]
        return waypoints

    def _closest_object(self, scene: Scene, gripper_pos: np.ndarray):
        best_idx = None
        best_dist = None
        for idx, obj in enumerate(scene.objects):
            pts = transform_points(obj.surface_points, obj.pose)
            d = np.linalg.norm(pts - gripper_pos[None, :], axis=1)
            dist = float(np.min(d))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is None:
            return None
        if best_dist is not None and best_dist > self.config.attach_radius:
            return None
        return best_idx

    def _render_trajectory(self, scene: Scene, traj: List[Waypoint]):
        pcds = []
        attached_idx = None
        attached_offset = None
        last_grip = None
        for w in traj:
            grip = int(w.gripper_state)
            if last_grip is None:
                if grip == 1 and self.config.attach_on_grasp:
                    obj_idx = self._closest_object(scene, w.pose[:3, 3])
                    if obj_idx is not None:
                        obj = scene.objects[obj_idx]
                        attached_idx = obj_idx
                        attached_offset = np.linalg.inv(w.pose) @ obj.pose
                last_grip = grip
            if grip != last_grip:
                if grip == 1 and self.config.attach_on_grasp:
                    obj_idx = self._closest_object(scene, w.pose[:3, 3])
                    if obj_idx is not None:
                        obj = scene.objects[obj_idx]
                        attached_idx = obj_idx
                        attached_offset = np.linalg.inv(w.pose) @ obj.pose
                else:
                    attached_idx = None
                    attached_offset = None
                last_grip = grip
            if attached_idx is not None and attached_offset is not None:
                obj = scene.objects[attached_idx]
                obj.pose = w.pose @ attached_offset
            pcd = self.renderer.render_observation(scene)
            if pcd.size == 0:
                pcd = self._fallback_pcd(scene)
            pcds.append(pcd)
        return pcds

    def _fallback_pcd(self, scene: Scene):
        points = []
        for obj in scene.objects:
            pts = transform_points(obj.surface_points, obj.pose)
            points.append(pts)
        if not points:
            return np.zeros((1, 3), dtype=np.float32)
        points = np.concatenate(points, axis=0)
        if self.config.render_downsample_voxel is not None:
            points = downsample_pcd(points, voxel_size=self.config.render_downsample_voxel)
        return points.astype(np.float32)

    def generate_demo(self, base_scene: Scene, waypoint_specs, rng: np.random.Generator):
        scene = self._vary_scene(base_scene, rng)
        start_pose = self._sample_start_pose(rng)
        waypoints = self.waypoint_sampler.resolve_waypoints(scene, waypoint_specs)
        waypoints = self._perturb_waypoints(waypoints, rng)
        start_grip = waypoints[0].gripper_state if waypoints else int(rng.integers(0, 2))
        waypoints = [Waypoint(pose=start_pose, gripper_state=start_grip)] + waypoints
        method = rng.choice(self.config.interpolation_methods)
        traj = self.interpolator.interpolate(waypoints, method=method)
        traj = self.augmenter.augment(traj, rng)
        traj = self.interpolator.interpolate(traj, method="linear")
        pcds = self._render_trajectory(scene, traj)
        sample = {
            "pcds": pcds,
            "T_w_es": [w.pose for w in traj],
            "grips": [w.gripper_state for w in traj],
        }
        return sample

    def generate_task(self, rng: np.random.Generator):
        base_scene = self.scene_builder.generate_scene(rng)
        waypoint_specs = self.waypoint_sampler.sample_waypoint_specs(base_scene, rng)
        num_demos = int(rng.integers(self.config.num_demos_per_task[0], self.config.num_demos_per_task[1] + 1))
        demos = []
        for _ in range(num_demos):
            demos.append(self.generate_demo(base_scene, waypoint_specs, rng))
        return demos

    def _build_samples(self, demos: List[dict], rng: np.random.Generator):
        samples = []
        for live_idx in range(len(demos)):
            if self.config.randomize_num_demos:
                num_ctx = int(rng.integers(self.config.num_context_range[0], self.config.num_context_range[1] + 1))
            else:
                num_ctx = self.config.num_context_demos
            candidates = [i for i in range(len(demos)) if i != live_idx]
            if not candidates:
                candidates = [live_idx]
            if len(candidates) >= num_ctx:
                ctx_indices = rng.choice(candidates, size=num_ctx, replace=False)
            else:
                ctx_indices = rng.choice(candidates, size=num_ctx, replace=True)
            cond_demos = [
                sample_to_cond_demo(demos[i], self.config.num_waypoints_demo, num_points=self.config.num_points)
                for i in ctx_indices
            ]
            live = sample_to_live(
                demos[live_idx],
                self.config.pred_horizon,
                num_points=self.config.num_points,
                trans_space=self.config.live_spacing_trans,
                rot_space=self.config.live_spacing_rot,
                subsample=self.config.subsample_live,
            )
            samples.append({"demos": cond_demos, "live": live})
        return samples

    def _next_offset(self, save_dir: str):
        if not os.path.exists(save_dir):
            return 0
        files = [f for f in os.listdir(save_dir) if f.startswith("data_") and f.endswith(".pt")]
        if not files:
            return 0
        indices = []
        for fname in files:
            try:
                indices.append(int(fname.split("_")[1].split(".")[0]))
            except ValueError:
                continue
        return max(indices) + 1 if indices else 0

    def generate_dataset(
        self,
        num_tasks: int,
        save_dir: str,
        task_start: int = 0,
        append: bool = False,
    ):
        os.makedirs(save_dir, exist_ok=True)
        offset = self._next_offset(save_dir) if append else 0
        rng = np.random.default_rng(self.config.seed + task_start)
        for task_idx in tqdm(range(task_start, task_start + num_tasks), desc="Generating pseudo-tasks"):
            task_rng = np.random.default_rng(self.config.seed + task_idx)
            demos = self.generate_task(task_rng)
            samples = self._build_samples(demos, task_rng)
            for sample in samples:
                save_sample(sample, save_dir=save_dir, offset=offset, scene_encoder=self.scene_encoder)
                offset += len(sample["live"]["obs"])
