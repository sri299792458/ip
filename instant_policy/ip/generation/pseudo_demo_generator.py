import os
import tempfile
from typing import Dict, List, Optional

import numpy as np
import trimesh
from tqdm import tqdm

import torch

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

from ip.generation.augmentation import TrajectoryAugmenter
from ip.generation.config import GenerationConfig
from ip.generation.geometry import make_transform, random_rotation, transform_points
from ip.generation.scene_builder import SceneBuilder, Scene
from ip.generation.trajectory_interpolator import TrajectoryInterpolator
from ip.generation.waypoint_sampler import Waypoint, WaypointSampler
from ip.utils.data_proc import sample_to_cond_demo, sample_to_live, save_sample


class PseudoDemoGenerator:
    OPEN = 1
    CLOSED = 0
    # Policy frame convention used in deployment/training:
    # UR flange/base -> policy origin offset along +Z is ~88 mm.
    POLICY_ORIGIN_Z_FROM_URDF_BASE_M = 0.088

    def __init__(self, config: GenerationConfig, scene_encoder=None, build_renderer: bool = True):
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
        self.gripper_mesh = self._load_gripper_mesh()
        self.gripper_surface_points = trimesh.sample.sample_surface(
            self.gripper_mesh,
            self.scene_builder.surface_sample_count,
        )[0].astype(np.float32)
        self.grasp_capture_region = self._estimate_grasp_capture_region(self.gripper_mesh)
        self.waypoint_sampler = WaypointSampler(
            bias_prob=config.bias_prob,
            num_waypoints_range=config.num_waypoints_range,
            forced_skill=config.forced_skill,
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
        self.renderer = None
        if build_renderer:
            from ip.generation.renderer import DepthRenderer
            self.renderer = DepthRenderer(
                cameras=config.cameras,
                downsample_voxel=config.render_downsample_voxel,
                max_points_per_obs=config.max_points_per_obs,
                gripper_mesh=self.gripper_mesh,
                visual_width=config.render_visual_width,
                visual_height=config.render_visual_height,
            )

    def _load_gripper_mesh(self) -> trimesh.Trimesh:
        path = self.config.gripper_mesh_path
        if path is None:
            raise RuntimeError(
                "gripper_mesh_path is required for paper-fidelity pseudo-demo generation. "
                "Build one with: python -m ip.scripts.build_robotiq_mesh --out <path>.obj"
            )

        mesh = trimesh.load(path, force="mesh", process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        if not isinstance(mesh, trimesh.Trimesh):
            raise RuntimeError(f"Unsupported gripper mesh type: {type(mesh)}")
        if mesh.vertices.size == 0:
            raise RuntimeError(f"Empty gripper mesh: {path}")
        mesh = mesh.copy()
        mesh.remove_unreferenced_vertices()
        ext = np.asarray(mesh.extents, dtype=np.float64)
        max_extent = float(np.max(ext))
        min_extent = float(np.min(ext))
        if max_extent > 1.0 or min_extent < 0.005:
            raise RuntimeError(
                f"Gripper mesh extents look invalid for meter units: {ext.tolist()} from {path}. "
                "Expected a metric Robotiq 2F-85 mesh."
            )
        mesh = self._canonicalize_gripper_mesh_frame(mesh)
        return mesh

    def _canonicalize_gripper_mesh_frame(self, mesh: trimesh.Trimesh):
        """Map URDF/base-link mesh frame to the policy-origin frame.

        Keep pseudo generation consistent with deployment/model conventions where
        the policy origin is offset from flange/base by ~88 mm in +Z.
        We center XY using the fingertip band to keep symmetry robust across mesh variants.
        """
        out = mesh.copy()
        verts = np.asarray(out.vertices, dtype=np.float64)
        if verts.shape[0] == 0:
            return out
        z = verts[:, 2]
        z_thr = float(np.quantile(z, 0.98))
        tip_band = verts[z >= z_thr]
        if tip_band.shape[0] == 0:
            tip_xy = np.array([0.0, 0.0], dtype=np.float64)
        else:
            tip_xy = np.mean(tip_band[:, :2], axis=0)
        policy_origin = np.array(
            [float(tip_xy[0]), float(tip_xy[1]), self.POLICY_ORIGIN_Z_FROM_URDF_BASE_M],
            dtype=np.float64,
        )
        out.apply_translation(-policy_origin)
        out.remove_unreferenced_vertices()
        return out

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

    def _object_distance(self, scene: Scene, gripper_pose: np.ndarray, obj_index: int):
        if obj_index < 0 or obj_index >= len(scene.objects):
            return None
        grip_pts = transform_points(self.gripper_surface_points, gripper_pose)
        obj = scene.objects[obj_index]
        obj_pts = transform_points(obj.surface_points, obj.pose)
        diff = obj_pts[:, None, :] - grip_pts[None, :, :]
        d_sq = np.einsum("ijk,ijk->ij", diff, diff)
        return float(np.sqrt(np.min(d_sq)))

    def _estimate_grasp_capture_region(self, gripper_mesh: trimesh.Trimesh):
        """Estimate a jaw-capture prism from the canonical gripper mesh.

        Distance-to-open-mesh alone misses thin objects that lie between open fingers.
        This region approximates where an object would be captured when the gripper closes.
        """
        verts = np.asarray(gripper_mesh.vertices, dtype=np.float64)
        if verts.shape[0] == 0:
            raise RuntimeError("Cannot estimate grasp-capture region from empty gripper mesh.")

        z_vals = verts[:, 2]
        tip_z = float(np.quantile(z_vals, 0.98))
        tip_band = verts[z_vals >= float(np.quantile(z_vals, 0.90))]
        if tip_band.shape[0] < 16:
            tip_band = verts

        left = tip_band[tip_band[:, 1] > 0.0]
        right = tip_band[tip_band[:, 1] < 0.0]
        if left.shape[0] > 0 and right.shape[0] > 0:
            # Inner jaw boundaries near the fingertip area.
            y_min = float(np.max(right[:, 1]))
            y_max = float(np.min(left[:, 1]))
        else:
            y_half = float(np.quantile(np.abs(tip_band[:, 1]), 0.75))
            y_min = -y_half
            y_max = y_half

        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
            y_half = float(np.quantile(np.abs(tip_band[:, 1]), 0.75))
            y_min = -y_half
            y_max = y_half

        x_half = float(np.quantile(np.abs(tip_band[:, 0]), 0.95))
        ext = np.asarray(gripper_mesh.extents, dtype=np.float64)
        xy_margin = float(max(0.003, 0.05 * min(ext[0], ext[1])))
        z_depth = float(np.clip(0.6 * ext[2], 0.04, 0.08))

        return {
            "x_min": float(-x_half - xy_margin),
            "x_max": float(x_half + xy_margin),
            "y_min": float(y_min - xy_margin),
            "y_max": float(y_max + xy_margin),
            # Front-only capture: disallow backside grasping behind policy origin.
            "z_min": float(max(0.0, tip_z - z_depth)),
            "z_max": float(tip_z + xy_margin),
        }

    def _object_capture_count(self, scene: Scene, gripper_pose: np.ndarray, obj_index: int) -> int:
        if obj_index < 0 or obj_index >= len(scene.objects):
            return 0
        region = self.grasp_capture_region
        obj = scene.objects[obj_index]
        obj_pts_w = transform_points(obj.surface_points, obj.pose)
        obj_pts_e = transform_points(obj_pts_w, np.linalg.inv(gripper_pose))

        inside = (
            (obj_pts_e[:, 0] >= region["x_min"]) &
            (obj_pts_e[:, 0] <= region["x_max"]) &
            (obj_pts_e[:, 1] >= region["y_min"]) &
            (obj_pts_e[:, 1] <= region["y_max"]) &
            (obj_pts_e[:, 2] >= region["z_min"]) &
            (obj_pts_e[:, 2] <= region["z_max"])
        )
        return int(np.count_nonzero(inside))

    def _object_attach_metrics(self, scene: Scene, gripper_pose: np.ndarray, obj_index: int):
        dist = self._object_distance(scene, gripper_pose, obj_index)
        cap_count = self._object_capture_count(scene, gripper_pose, obj_index)
        return dist, cap_count

    def _should_attach(self, _dist: Optional[float], cap_count: int) -> bool:
        # First-principles policy: attach only when object points are inside
        # the front jaw-capture region at close transition.
        return cap_count >= int(self.config.attach_capture_min_points)

    @staticmethod
    def _new_attach_stats() -> Dict[str, float]:
        return {
            "close_events": 0.0,
            "target_close_events": 0.0,
            "untargeted_close_events": 0.0,
            "attach_success": 0.0,
            "attach_success_target": 0.0,
            "attach_success_untargeted": 0.0,
            "attach_miss_target": 0.0,
            "attach_miss_untargeted": 0.0,
            "detach_events": 0.0,
            "dist_sum_attached": 0.0,
            "dist_count_attached": 0.0,
            "cap_sum_attached": 0.0,
            "dist_sum_failed_target": 0.0,
            "dist_count_failed_target": 0.0,
            "cap_sum_failed_target": 0.0,
            "hard_negative_probes": 0.0,
            "hard_negative_false_attach": 0.0,
            "target_close_with_other_eligible": 0.0,
            "target_close_other_eligible_count_sum": 0.0,
            "hard_negative_other_eligible": 0.0,
            "hard_negative_other_eligible_count_sum": 0.0,
        }

    @staticmethod
    def _accumulate_attach_stats(dst: Dict[str, float], src: Dict[str, float]) -> None:
        for key, val in src.items():
            dst[key] = float(dst.get(key, 0.0)) + float(val)

    def _render_trajectory(
        self,
        scene: Scene,
        traj: List[Waypoint],
        render_dir: Optional[str] = None,
        video_writer=None,
        collect_attach_stats: bool = False,
        skip_render: bool = False,
        hard_negative_local_offsets: Optional[np.ndarray] = None,
        hard_negative_offset_scale: Optional[float] = None,
        hard_negative_offset_min_m: float = 0.04,
        hard_negative_offset_max_m: float = 0.12,
    ):
        pcds = []
        adjusted_traj = []
        attached_idx = None
        attached_offset = None
        last_grip = None
        attach_stats = self._new_attach_stats() if collect_attach_stats else None
        render_stride = max(1, int(self.config.render_stride))
        visual_idx = 0
        if not skip_render:
            if self.renderer is None:
                raise RuntimeError("Renderer is not initialized for pseudo-demo rendering.")
            visual_idx = int(self.config.render_visual_camera)
            if visual_idx < 0 or visual_idx >= len(self.renderer.cameras):
                visual_idx = 0
        for w in traj:
            grip = int(w.gripper_state)
            pose = w.pose
            if not np.all(np.isfinite(pose)):
                raise RuntimeError(
                    "Non-finite pose encountered during pseudo trajectory render. "
                    "Check interpolation settings and waypoint generation."
                )
            if last_grip is None:
                last_grip = grip
            elif grip != last_grip:
                # Paper Appendix D: attach/detach when gripper state changes.
                if grip == self.CLOSED and last_grip == self.OPEN and self.config.attach_on_grasp:
                    if attach_stats is not None:
                        attach_stats["close_events"] += 1.0
                    obj_idx = None
                    attach_dist = None
                    attach_cap_count = 0
                    targeted = w.obj_index is not None
                    if attach_stats is not None:
                        if targeted:
                            attach_stats["target_close_events"] += 1.0
                        else:
                            attach_stats["untargeted_close_events"] += 1.0
                    # First-principles object-centric behavior:
                    # if this waypoint targets an object, attach only that object.
                    if targeted:
                        target_idx = int(w.obj_index)
                        if target_idx < 0 or target_idx >= len(scene.objects):
                            target_idx = -1
                        if attach_stats is not None and target_idx >= 0:
                            other_eligible_count = 0
                            for cand_idx in range(len(scene.objects)):
                                if cand_idx == target_idx:
                                    continue
                                d_cand, cap_cand = self._object_attach_metrics(scene, pose, cand_idx)
                                if self._should_attach(d_cand, cap_cand):
                                    other_eligible_count += 1
                            attach_stats["target_close_other_eligible_count_sum"] += float(other_eligible_count)
                            if other_eligible_count > 0:
                                attach_stats["target_close_with_other_eligible"] += 1.0
                        if (
                            attach_stats is not None
                            and target_idx >= 0
                            and (hard_negative_local_offsets is not None or hard_negative_offset_scale is not None)
                        ):
                            if hard_negative_local_offsets is not None:
                                offsets = np.asarray(hard_negative_local_offsets, dtype=np.float64).reshape(-1, 3)
                            else:
                                obj_extent = float(np.max(scene.objects[target_idx].mesh.extents))
                                d = float(np.clip(
                                    float(hard_negative_offset_scale) * obj_extent,
                                    float(hard_negative_offset_min_m),
                                    float(hard_negative_offset_max_m),
                                ))
                                offsets = np.array(
                                    [
                                        [d, 0.0, 0.0],
                                        [-d, 0.0, 0.0],
                                        [0.0, d, 0.0],
                                        [0.0, -d, 0.0],
                                        [0.0, 0.0, d],
                                        [0.0, 0.0, -d],
                                    ],
                                    dtype=np.float64,
                                )
                            for offs_local in offsets:
                                pose_probe = np.array(pose, copy=True)
                                pose_probe[:3, 3] += pose[:3, :3] @ offs_local
                                d_probe, cap_probe = self._object_attach_metrics(
                                    scene, pose_probe, target_idx
                                )
                                attach_stats["hard_negative_probes"] += 1.0
                                if self._should_attach(d_probe, cap_probe):
                                    attach_stats["hard_negative_false_attach"] += 1.0
                                other_eligible_probe = 0
                                for cand_idx in range(len(scene.objects)):
                                    if cand_idx == target_idx:
                                        continue
                                    d_cand_probe, cap_cand_probe = self._object_attach_metrics(
                                        scene, pose_probe, cand_idx
                                    )
                                    if self._should_attach(d_cand_probe, cap_cand_probe):
                                        other_eligible_probe += 1
                                attach_stats["hard_negative_other_eligible_count_sum"] += float(other_eligible_probe)
                                if other_eligible_probe > 0:
                                    attach_stats["hard_negative_other_eligible"] += 1.0
                        d, cap_count = self._object_attach_metrics(scene, pose, target_idx)
                        attach_dist = d
                        attach_cap_count = int(cap_count)
                        if self._should_attach(d, cap_count):
                            obj_idx = target_idx
                    else:
                        # For non-object-centric waypoints, choose the best attach candidate:
                        # prioritize jaw-capture support, then nearest distance.
                        best_key = None
                        for cand_idx in range(len(scene.objects)):
                            d, cap_count = self._object_attach_metrics(scene, pose, cand_idx)
                            if not self._should_attach(d, cap_count):
                                continue
                            dist_val = float(d) if d is not None else 1e9
                            key = (int(cap_count), -dist_val)
                            if best_key is None or key > best_key:
                                best_key = key
                                obj_idx = cand_idx
                                attach_dist = d
                                attach_cap_count = int(cap_count)

                    if obj_idx is not None:
                        obj = scene.objects[obj_idx]
                        attached_idx = obj_idx
                        attached_offset = np.linalg.inv(pose) @ obj.pose
                        if attach_stats is not None:
                            attach_stats["attach_success"] += 1.0
                            if targeted:
                                attach_stats["attach_success_target"] += 1.0
                            else:
                                attach_stats["attach_success_untargeted"] += 1.0
                            attach_stats["cap_sum_attached"] += float(attach_cap_count)
                            if attach_dist is not None:
                                attach_stats["dist_sum_attached"] += float(attach_dist)
                                attach_stats["dist_count_attached"] += 1.0
                    elif attach_stats is not None:
                        if targeted:
                            attach_stats["attach_miss_target"] += 1.0
                            attach_stats["cap_sum_failed_target"] += float(attach_cap_count)
                            if attach_dist is not None:
                                attach_stats["dist_sum_failed_target"] += float(attach_dist)
                                attach_stats["dist_count_failed_target"] += 1.0
                        else:
                            attach_stats["attach_miss_untargeted"] += 1.0
                elif grip == self.OPEN and last_grip == self.CLOSED:
                    attached_idx = None
                    attached_offset = None
                    if attach_stats is not None:
                        attach_stats["detach_events"] += 1.0
                last_grip = grip
            if attached_idx is not None and attached_offset is not None:
                obj = scene.objects[attached_idx]
                obj.pose = pose @ attached_offset
            if skip_render:
                pcd = np.zeros((0, 3), dtype=np.float32)
            else:
                pcd = self.renderer.render_observation(scene)
            color, depth = None, None
            if not skip_render and pcd.size == 0:
                raise RuntimeError(
                    "Rendered empty point cloud. Check camera setup/workspace bounds; "
                    "paper setup expects segmented observations from 3 depth cameras."
                )
            pcds.append(pcd)
            adjusted_traj.append(Waypoint(pose=pose, gripper_state=grip, obj_index=w.obj_index))
            if (
                not skip_render
                and (render_dir is not None or video_writer is not None)
                and len(pcds) % render_stride == 0
            ):
                color, depth = self.renderer.render_visual(scene, pose, visual_idx)
                frame_idx = len(pcds) - 1
                if render_dir is not None:
                    self._save_render_frame(render_dir, frame_idx, color, depth)
                if video_writer is not None:
                    video_writer.append_data(color)
        if collect_attach_stats:
            return pcds, adjusted_traj, attach_stats
        return pcds, adjusted_traj

    def _save_render_frame(self, render_dir, frame_idx, color, depth):
        os.makedirs(render_dir, exist_ok=True)
        color_path = os.path.join(render_dir, f"frame_{frame_idx:05d}.png")
        if imageio is not None:
            imageio.imwrite(color_path, color)
        else:
            import matplotlib.pyplot as plt
            plt.imsave(color_path, color)
        if self.config.render_save_depth and depth is not None:
            depth_path = os.path.join(render_dir, f"frame_{frame_idx:05d}_depth.png")
            depth_norm = depth.copy()
            depth_norm[depth_norm <= 0] = np.nan
            min_d = np.nanmin(depth_norm)
            max_d = np.nanmax(depth_norm)
            if np.isfinite(min_d) and np.isfinite(max_d) and max_d > min_d:
                depth_vis = (depth_norm - min_d) / (max_d - min_d)
            else:
                depth_vis = np.zeros_like(depth_norm)
            if imageio is not None:
                imageio.imwrite(depth_path, (depth_vis * 255).astype(np.uint8))
            else:
                import matplotlib.pyplot as plt
                plt.imsave(depth_path, depth_vis, cmap="gray")

    def _open_video_writer(self, video_dir: Optional[str]):
        if not self.config.render_make_videos or video_dir is None:
            return None
        if imageio is None:
            print("imageio not available; skipping video generation.")
            return None
        os.makedirs(video_dir, exist_ok=True)
        out_path = os.path.join(video_dir, f"demo.{self.config.render_video_ext}")
        return imageio.get_writer(out_path, fps=self.config.render_video_fps)

    def _build_demo_trajectory(self, base_scene: Scene, waypoint_specs, rng: np.random.Generator):
        scene = self._vary_scene(base_scene, rng)
        start_pose = self._sample_start_pose(rng)
        waypoints = self.waypoint_sampler.resolve_waypoints(scene, waypoint_specs)
        start_grip = waypoints[0].gripper_state if waypoints else int(rng.integers(0, 2))
        waypoints = [Waypoint(pose=start_pose, gripper_state=start_grip, obj_index=None)] + waypoints
        method = rng.choice(self.config.interpolation_methods)
        traj = self.interpolator.interpolate(waypoints, method=method)
        # Keep attachment physics driven by clean gripper transitions.
        # Paper-style 10% gripper corruption is applied to saved labels later.
        traj = self.augmenter.augment_motion(traj, rng)
        traj = self.interpolator.interpolate(traj, method="linear")
        return scene, traj

    def evaluate_demo_attach_stats(
        self,
        base_scene: Scene,
        waypoint_specs,
        rng: np.random.Generator,
        hard_negative_local_offsets: Optional[np.ndarray] = None,
        hard_negative_offset_scale: Optional[float] = None,
        hard_negative_offset_min_m: float = 0.04,
        hard_negative_offset_max_m: float = 0.12,
    ):
        scene, traj = self._build_demo_trajectory(base_scene, waypoint_specs, rng)
        _, _, attach_stats = self._render_trajectory(
            scene,
            traj,
            collect_attach_stats=True,
            skip_render=True,
            hard_negative_local_offsets=hard_negative_local_offsets,
            hard_negative_offset_scale=hard_negative_offset_scale,
            hard_negative_offset_min_m=hard_negative_offset_min_m,
            hard_negative_offset_max_m=hard_negative_offset_max_m,
        )
        return attach_stats

    def evaluate_task_attach_stats(
        self,
        rng: np.random.Generator,
        hard_negative_local_offsets: Optional[np.ndarray] = None,
        hard_negative_offset_scale: Optional[float] = None,
        hard_negative_offset_min_m: float = 0.04,
        hard_negative_offset_max_m: float = 0.12,
    ):
        base_scene = self.scene_builder.generate_scene(rng)
        waypoint_specs = self.waypoint_sampler.sample_waypoint_specs(base_scene, rng)
        num_demos = int(rng.integers(self.config.num_demos_per_task[0], self.config.num_demos_per_task[1] + 1))
        totals = self._new_attach_stats()
        totals["num_demos"] = float(num_demos)
        for _ in range(num_demos):
            stats = self.evaluate_demo_attach_stats(
                base_scene,
                waypoint_specs,
                rng,
                hard_negative_local_offsets=hard_negative_local_offsets,
                hard_negative_offset_scale=hard_negative_offset_scale,
                hard_negative_offset_min_m=hard_negative_offset_min_m,
                hard_negative_offset_max_m=hard_negative_offset_max_m,
            )
            self._accumulate_attach_stats(totals, stats)
        return totals

    def generate_demo(
        self,
        base_scene: Scene,
        waypoint_specs,
        rng: np.random.Generator,
        render_dir: Optional[str] = None,
        video_dir: Optional[str] = None,
    ):
        scene, traj = self._build_demo_trajectory(base_scene, waypoint_specs, rng)
        video_writer = self._open_video_writer(video_dir)
        try:
            pcds, dyn_traj = self._render_trajectory(
                scene,
                traj,
                render_dir=render_dir,
                video_writer=video_writer,
            )
        finally:
            if video_writer is not None:
                video_writer.close()
        label_traj = self.augmenter.augment_gripper_labels(dyn_traj, rng)
        sample = {
            "pcds": pcds,
            "T_w_es": [w.pose for w in dyn_traj],
            "grips": [w.gripper_state for w in label_traj],
        }
        return sample

    def generate_task(self, rng: np.random.Generator, task_idx: Optional[int] = None):
        base_scene = self.scene_builder.generate_scene(rng)
        waypoint_specs = self.waypoint_sampler.sample_waypoint_specs(base_scene, rng)
        num_demos = int(rng.integers(self.config.num_demos_per_task[0], self.config.num_demos_per_task[1] + 1))
        demos = []
        render_root = self.config.render_dir or os.path.join(self.config.save_dir, "_renders")
        if self.config.render_video_dir is not None:
            video_root = self.config.render_video_dir
        elif self.config.save_renders:
            video_root = render_root
        else:
            video_root = os.path.join(self.config.save_dir, "_videos")
        for demo_idx in range(num_demos):
            task_tag = "task" if task_idx is None else f"task_{task_idx:06d}"
            demo_tag = f"demo_{demo_idx:02d}"
            render_dir = None
            if self.config.save_renders:
                render_dir = os.path.join(render_root, task_tag, demo_tag)
            video_dir = None
            if self.config.render_make_videos:
                video_dir = os.path.join(video_root, task_tag, demo_tag)
            demos.append(
                self.generate_demo(
                    base_scene,
                    waypoint_specs,
                    rng,
                    render_dir=render_dir,
                    video_dir=video_dir,
                )
            )
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

    def _next_offset(self, save_dir: str, min_idx: int = 0, max_idx: Optional[int] = None):
        if not os.path.exists(save_dir):
            return min_idx
        files = [f for f in os.listdir(save_dir) if f.startswith("data_") and f.endswith(".pt")]
        if not files:
            return min_idx
        indices = []
        for fname in files:
            try:
                indices.append(int(fname.split("_")[1].split(".")[0]))
            except ValueError:
                continue
        if max_idx is not None:
            indices = [i for i in indices if min_idx <= i < max_idx]
        if not indices:
            return min_idx
        return max(indices) + 1

    def _next_task_offset(self, save_dir: str, min_idx: int = 0, max_idx: Optional[int] = None):
        if not os.path.exists(save_dir):
            return min_idx
        files = [f for f in os.listdir(save_dir) if f.startswith("task_") and f.endswith(".pt")]
        if not files:
            return min_idx
        indices = []
        for fname in files:
            try:
                indices.append(int(fname.split("_")[1].split(".")[0]))
            except ValueError:
                continue
        if max_idx is not None:
            indices = [i for i in indices if min_idx <= i < max_idx]
        if not indices:
            return min_idx
        return max(indices) + 1

    def _shard_range(self, buffer_size: int, shard_id: int, num_shards: int):
        if num_shards < 1:
            raise ValueError("num_shards must be >= 1")
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError("shard_id must be in [0, num_shards)")
        base = buffer_size // num_shards
        extra = buffer_size % num_shards
        if shard_id < extra:
            start = shard_id * (base + 1)
            end = start + base + 1
        else:
            start = shard_id * base + extra
            end = start + base
        if end <= start:
            raise ValueError("Invalid shard range; increase buffer_size.")
        return start, end

    def generate_dataset(
        self,
        num_tasks: int,
        save_dir: str,
        task_start: int = 0,
        append: bool = False,
        buffer_size: Optional[int] = None,
        shard_id: int = 0,
        num_shards: int = 1,
        fill_buffer: bool = False,
    ):
        if self.config.storage_format == "trajectory":
            return self._generate_dataset_trajectory(
                num_tasks=num_tasks,
                save_dir=save_dir,
                task_start=task_start,
                append=append,
                buffer_size=buffer_size,
                shard_id=shard_id,
                num_shards=num_shards,
                fill_buffer=fill_buffer,
            )
        os.makedirs(save_dir, exist_ok=True)
        if buffer_size is not None:
            shard_start, shard_end = self._shard_range(buffer_size, shard_id, num_shards)
        else:
            shard_start, shard_end = 0, None
        offset = self._next_offset(save_dir, shard_start, shard_end) if append else shard_start
        rng = np.random.default_rng(self.config.seed + task_start)
        wrapped = False
        for task_idx in tqdm(range(task_start, task_start + num_tasks), desc="Generating pseudo-tasks"):
            task_rng = np.random.default_rng(self.config.seed + task_idx)
            demos = self.generate_task(task_rng, task_idx=task_idx)
            samples = self._build_samples(demos, task_rng)
            for sample in samples:
                live_len = len(sample["live"]["obs"])
                if shard_end is not None:
                    shard_size = shard_end - shard_start
                    if live_len >= shard_size:
                        raise RuntimeError("Sample length exceeds shard size. Increase buffer_size.")
                    if offset + live_len > shard_end:
                        if fill_buffer:
                            # For bootstrap fill, place the sample flush to shard end
                            # so trailing indices are not left permanently empty.
                            offset = max(shard_start, shard_end - live_len)
                        else:
                            offset = shard_start
                        wrapped = True
                save_sample(sample, save_dir=save_dir, offset=offset, scene_encoder=self.scene_encoder)
                offset += live_len
                if fill_buffer and wrapped:
                    return

    def _package_task(self, demos: List[dict]):
        stored = []
        use_float16 = self.config.pcd_storage_dtype == "float16"
        for demo in demos:
            cond = sample_to_cond_demo(demo, self.config.num_waypoints_demo, num_points=self.config.num_points)
            if use_float16:
                pcds = [pcd.astype(np.float16, copy=False) for pcd in demo["pcds"]]
            else:
                pcds = demo["pcds"]
            stored.append({
                "pcds": pcds,
                "T_w_es": demo["T_w_es"],
                "grips": demo["grips"],
                "cond": cond,
            })
        return stored

    def _save_task(self, task_data: dict, save_dir: str, index: int):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"task_{index}.pt")
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_task_", suffix=".pt", dir=save_dir)
        os.close(fd)
        try:
            torch.save(task_data, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _generate_dataset_trajectory(
        self,
        num_tasks: int,
        save_dir: str,
        task_start: int = 0,
        append: bool = False,
        buffer_size: Optional[int] = None,
        shard_id: int = 0,
        num_shards: int = 1,
        fill_buffer: bool = False,
    ):
        os.makedirs(save_dir, exist_ok=True)
        if buffer_size is not None:
            shard_start, shard_end = self._shard_range(buffer_size, shard_id, num_shards)
        else:
            shard_start, shard_end = 0, None
        offset = self._next_task_offset(save_dir, shard_start, shard_end) if append else shard_start
        rng = np.random.default_rng(self.config.seed + task_start)
        wrapped = False
        for task_idx in tqdm(range(task_start, task_start + num_tasks), desc="Generating pseudo-tasks"):
            task_rng = np.random.default_rng(self.config.seed + task_idx)
            demos = self.generate_task(task_rng, task_idx=task_idx)
            task_data = {
                "task_id": task_idx,
                "demos": self._package_task(demos),
            }
            if shard_end is not None:
                shard_size = shard_end - shard_start
                if shard_size < 1:
                    raise RuntimeError("Invalid shard size for trajectory buffer.")
                if offset >= shard_end:
                    offset = shard_start
                    wrapped = True
            self._save_task(task_data, save_dir, offset)
            offset += 1
            if fill_buffer and wrapped:
                return
