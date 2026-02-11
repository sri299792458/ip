from __future__ import annotations

import os
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh
from tqdm import tqdm

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

from ip.generation.geometry import transform_points
from ip.generation.renderer import DepthRenderer
from ip.generation.scene_builder import Scene, SceneBuilder
from ip.generation_bimanual.config import BimanualGenerationConfig
from ip.generation_bimanual.primitives import PrimitiveTrajectory, sample_trajectory


class BimanualPseudoDemoGenerator:
    OPEN = 1
    CLOSED = 0

    # Policy frame convention used in deployment/training:
    # UR flange/base -> policy origin offset along +Z is ~88 mm.
    POLICY_ORIGIN_Z_FROM_URDF_BASE_M = 0.088

    def __init__(self, config: BimanualGenerationConfig):
        self.config = config
        self.config.validate()

        self.scene_builder = SceneBuilder(
            shapenet_path=config.shapenet_path,
            workspace_bounds=config.workspace_bounds,
            table_height=config.table_height,
            object_scale_range=config.object_scale_range,
            num_objects_range=config.num_objects_range,
            shapenet_index_path=config.shapenet_index_path,
            max_meshes=config.max_meshes,
            cache_meshes=config.cache_meshes,
            surface_sample_count=config.surface_sample_count,
        )

        self.gripper_mesh = self._load_gripper_mesh()
        self.gripper_surface_points = trimesh.sample.sample_surface(
            self.gripper_mesh,
            self.scene_builder.surface_sample_count,
        )[0].astype(np.float32)
        self.grasp_capture_region = self._estimate_grasp_capture_region(self.gripper_mesh)

        self.renderer = DepthRenderer(
            cameras=config.cameras,
            downsample_voxel=config.render_downsample_voxel,
            max_points_per_obs=config.max_points_per_obs,
            gripper_mesh=self.gripper_mesh,
            visual_width=config.render_visual_width,
            visual_height=config.render_visual_height,
        )

    def _sample_task_name(self, rng: np.random.Generator) -> str:
        if self.config.forced_task is not None:
            return self.config.forced_task

        tasks = self.config.task_names
        if not tasks:
            raise RuntimeError("task_names is empty")

        if self.config.task_weights is None:
            idx = int(rng.integers(0, len(tasks)))
            return tasks[idx]

        w = np.asarray(self.config.task_weights, dtype=np.float64)
        w = w / np.sum(w)
        idx = int(rng.choice(np.arange(len(tasks)), p=w))
        return tasks[idx]

    @staticmethod
    def _list_existing_indices(save_dir: str) -> List[int]:
        out = []
        for p in glob(os.path.join(save_dir, "task_*.pt")):
            stem = os.path.splitext(os.path.basename(p))[0]
            try:
                out.append(int(stem.split("_")[-1]))
            except ValueError:
                continue
        return sorted(out)

    def _clear_existing(self, save_dir: str) -> None:
        for p in glob(os.path.join(save_dir, "task_*.pt")):
            os.remove(p)

    def _load_gripper_mesh(self) -> trimesh.Trimesh:
        path = self.config.gripper_mesh_path
        if path is None:
            raise RuntimeError(
                "gripper_mesh_path is required for paper-fidelity bimanual pseudo generation. "
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

        return self._canonicalize_gripper_mesh_frame(mesh)

    def _canonicalize_gripper_mesh_frame(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Map URDF/base-link mesh frame to the policy-origin frame."""
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
            [
                float(tip_xy[0]),
                float(tip_xy[1]),
                self.POLICY_ORIGIN_Z_FROM_URDF_BASE_M,
            ],
            dtype=np.float64,
        )
        out.apply_translation(-policy_origin)
        out.remove_unreferenced_vertices()
        return out

    def _estimate_grasp_capture_region(self, gripper_mesh: trimesh.Trimesh) -> Dict[str, float]:
        """Estimate a jaw-capture prism from canonical gripper mesh."""
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

    def _should_attach(self, cap_count: int) -> bool:
        return cap_count >= int(self.config.attach_capture_min_points)

    def _sample_fixed_points(self, points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = int(self.config.num_points)
        m = int(points.shape[0])
        if m <= 0:
            raise RuntimeError(
                "Rendered empty point cloud. Check camera setup/workspace bounds for bimanual pseudo generation."
            )
        if m == n:
            return points.astype(np.float32, copy=False)
        if m > n:
            idx = rng.choice(m, size=n, replace=False)
        else:
            idx = rng.choice(m, size=n, replace=True)
        return points[idx].astype(np.float32, copy=False)

    @staticmethod
    def _apply_object_poses(scene: Scene, object_poses: np.ndarray) -> None:
        for i, obj in enumerate(scene.objects):
            obj.pose = np.asarray(object_poses[i], dtype=np.float32)

    def _scene_at_step(self, scene: Scene, object_pose_seq: np.ndarray, step: int) -> Scene:
        out = scene.copy()
        self._apply_object_poses(out, object_pose_seq[step])
        return out

    def _simulate_object_poses(self, scene: Scene, traj: PrimitiveTrajectory) -> np.ndarray:
        steps = int(traj.left_seq.shape[0])
        if traj.right_seq.shape[0] != steps:
            raise RuntimeError("Left/right sequence length mismatch")
        if traj.grip_left.shape[0] != steps or traj.grip_right.shape[0] != steps:
            raise RuntimeError("Grip sequence length mismatch")

        sim_scene = scene.copy()
        num_objects = len(sim_scene.objects)
        object_pose_seq = np.zeros((steps, num_objects, 4, 4), dtype=np.float32)

        left_attached_idx: Optional[int] = None
        right_attached_idx: Optional[int] = None
        left_offset: Optional[np.ndarray] = None
        right_offset: Optional[np.ndarray] = None

        # Treat t=0 as transition from open for robustness.
        last_left = self.OPEN
        last_right = self.OPEN

        for t in range(steps):
            left_pose = np.asarray(traj.left_seq[t], dtype=np.float32)
            right_pose = np.asarray(traj.right_seq[t], dtype=np.float32)
            left_grip = int(traj.grip_left[t])
            right_grip = int(traj.grip_right[t])
            left_target = int(traj.left_targets[t])
            right_target = int(traj.right_targets[t])

            if left_grip != last_left:
                if left_grip == self.CLOSED and last_left == self.OPEN:
                    if 0 <= left_target < num_objects:
                        cap = self._object_capture_count(sim_scene, left_pose, left_target)
                        if self._should_attach(cap):
                            if right_attached_idx == left_target:
                                right_attached_idx = None
                                right_offset = None
                            left_attached_idx = left_target
                            left_offset = np.linalg.inv(left_pose) @ sim_scene.objects[left_target].pose
                elif left_grip == self.OPEN and last_left == self.CLOSED:
                    left_attached_idx = None
                    left_offset = None

            if right_grip != last_right:
                if right_grip == self.CLOSED and last_right == self.OPEN:
                    if 0 <= right_target < num_objects:
                        cap = self._object_capture_count(sim_scene, right_pose, right_target)
                        if self._should_attach(cap):
                            if left_attached_idx == right_target:
                                left_attached_idx = None
                                left_offset = None
                            right_attached_idx = right_target
                            right_offset = np.linalg.inv(right_pose) @ sim_scene.objects[right_target].pose
                elif right_grip == self.OPEN and last_right == self.CLOSED:
                    right_attached_idx = None
                    right_offset = None

            last_left = left_grip
            last_right = right_grip

            attached = set()
            if left_attached_idx is not None:
                attached.add(int(left_attached_idx))
            if right_attached_idx is not None:
                attached.add(int(right_attached_idx))

            # Scripted object motion applies only to objects not currently attached.
            for obj_idx, pose_seq in traj.scripted_object_poses.items():
                if obj_idx in attached:
                    continue
                if 0 <= t < pose_seq.shape[0]:
                    sim_scene.objects[obj_idx].pose = np.asarray(pose_seq[t], dtype=np.float32)

            if left_attached_idx is not None and left_offset is not None:
                sim_scene.objects[left_attached_idx].pose = (left_pose @ left_offset).astype(np.float32)
            if right_attached_idx is not None and right_offset is not None:
                sim_scene.objects[right_attached_idx].pose = (right_pose @ right_offset).astype(np.float32)

            for i, obj in enumerate(sim_scene.objects):
                object_pose_seq[t, i] = np.asarray(obj.pose, dtype=np.float32)

        return object_pose_seq

    def _save_render_frame(
        self,
        render_dir: str,
        frame_idx: int,
        color: np.ndarray,
        depth: Optional[np.ndarray],
    ) -> None:
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
        out_path = os.path.join(video_dir, f"trajectory.{self.config.render_video_ext}")
        return imageio.get_writer(out_path, fps=self.config.render_video_fps)

    def _maybe_render_debug_visuals(
        self,
        scene: Scene,
        traj: PrimitiveTrajectory,
        object_pose_seq: np.ndarray,
        sample_tag: str,
    ) -> None:
        if not self.config.save_renders and not self.config.render_make_videos:
            return

        visual_idx = int(self.config.render_visual_camera)
        if visual_idx < 0 or visual_idx >= len(self.renderer.cameras):
            visual_idx = 0

        render_dir = None
        if self.config.save_renders:
            root = self.config.render_dir or os.path.join(self.config.save_dir, "_renders")
            render_dir = os.path.join(root, sample_tag)

        video_dir = None
        if self.config.render_make_videos:
            root = self.config.render_video_dir or os.path.join(self.config.save_dir, "_videos")
            video_dir = os.path.join(root, sample_tag)

        writer = self._open_video_writer(video_dir)
        stride = max(1, int(self.config.render_stride))

        try:
            steps = int(traj.left_seq.shape[0])
            for t in range(0, steps, stride):
                scene_t = self._scene_at_step(scene, object_pose_seq, t)
                color, depth = self.renderer.render_visual(
                    scene_t,
                    traj.left_seq[t],
                    visual_idx,
                    extra_gripper_poses=[traj.right_seq[t]],
                )
                if render_dir is not None:
                    self._save_render_frame(render_dir, t, color, depth)
                if writer is not None:
                    writer.append_data(color)
        finally:
            if writer is not None:
                writer.close()

    def _sample_to_world_batch(
        self,
        task_name: str,
        rng: np.random.Generator,
        sample_tag: str,
    ) -> Dict[str, torch.Tensor]:
        scene = self.scene_builder.generate_scene(rng)
        traj = sample_trajectory(task_name, self.config, scene, rng)
        object_pose_seq = self._simulate_object_poses(scene, traj)

        steps = int(traj.left_seq.shape[0])
        horizon = int(self.config.pred_horizon)
        max_start = steps - (horizon + 1)
        start = int(rng.integers(0, max_start + 1))

        scene_start = self._scene_at_step(scene, object_pose_seq, start)
        points_world = self.renderer.render_observation(scene_start)
        points_world = self._sample_fixed_points(points_world, rng)

        self._maybe_render_debug_visuals(scene, traj, object_pose_seq, sample_tag)

        pcd_dtype = torch.float16 if self.config.pcd_storage_dtype == "float16" else torch.float32
        sample = {
            "points_world": torch.as_tensor(points_world[None, ...], dtype=pcd_dtype),
            "T_w_left_current": torch.as_tensor(traj.left_seq[start][None, ...], dtype=torch.float32),
            "T_w_right_current": torch.as_tensor(traj.right_seq[start][None, ...], dtype=torch.float32),
            "T_w_left_future": torch.as_tensor(
                traj.left_seq[start + 1 : start + 1 + horizon][None, ...],
                dtype=torch.float32,
            ),
            "T_w_right_future": torch.as_tensor(
                traj.right_seq[start + 1 : start + 1 + horizon][None, ...],
                dtype=torch.float32,
            ),
            "grip_left_current": torch.as_tensor([traj.grip_left[start]], dtype=torch.float32),
            "grip_right_current": torch.as_tensor([traj.grip_right[start]], dtype=torch.float32),
            "grip_left_future": torch.as_tensor(
                traj.grip_left[start + 1 : start + 1 + horizon][None, ...],
                dtype=torch.float32,
            ),
            "grip_right_future": torch.as_tensor(
                traj.grip_right[start + 1 : start + 1 + horizon][None, ...],
                dtype=torch.float32,
            ),
        }
        return sample

    def _slot_iter_non_fill(self):
        for k in range(int(self.config.num_samples)):
            global_idx = int(self.config.task_start + k)
            if global_idx % self.config.num_shards != self.config.shard_id:
                continue
            if self.config.buffer_size is None:
                file_idx = global_idx
            else:
                file_idx = global_idx % int(self.config.buffer_size)
            yield global_idx, file_idx

    def _slot_iter_fill(self):
        if self.config.buffer_size is None:
            raise RuntimeError("fill_buffer requires buffer_size")

        target_slots = [
            i
            for i in range(int(self.config.buffer_size))
            if i % self.config.num_shards == self.config.shard_id
        ]
        filled = set()
        global_idx = int(self.config.task_start)

        while len(filled) < len(target_slots):
            if global_idx % self.config.num_shards != self.config.shard_id:
                global_idx += 1
                continue
            file_idx = global_idx % int(self.config.buffer_size)
            if file_idx in filled:
                global_idx += 1
                continue
            filled.add(file_idx)
            yield global_idx, file_idx
            global_idx += 1

    def _count_non_fill_slots(self) -> int:
        total = 0
        for k in range(int(self.config.num_samples)):
            global_idx = int(self.config.task_start + k)
            if global_idx % self.config.num_shards == self.config.shard_id:
                total += 1
        return total

    def generate_dataset(self) -> None:
        save_dir = self.config.save_dir
        os.makedirs(save_dir, exist_ok=True)

        if not self.config.append and self.config.buffer_size is None:
            self._clear_existing(save_dir)

        if self.config.buffer_size is None and self.config.append and self.config.task_start == 0:
            existing = self._list_existing_indices(save_dir)
            if existing:
                self.config.task_start = int(existing[-1] + 1)

        slots = self._slot_iter_fill() if self.config.fill_buffer else self._slot_iter_non_fill()
        if self.config.fill_buffer:
            total_slots = len(
                [
                    i
                    for i in range(int(self.config.buffer_size))
                    if i % self.config.num_shards == self.config.shard_id
                ]
            )
        else:
            total_slots = self._count_non_fill_slots()

        written = 0
        for global_idx, file_idx in tqdm(slots, total=total_slots, desc="Generating bimanual pseudo samples"):
            sample_rng = np.random.default_rng(self.config.seed + global_idx)
            task_name = self._sample_task_name(sample_rng)
            sample_tag = f"sample_{global_idx:07d}"
            sample = self._sample_to_world_batch(task_name, sample_rng, sample_tag)
            out_path = os.path.join(save_dir, f"task_{file_idx:07d}.pt")
            torch.save(sample, out_path)
            written += 1

        if written == 0:
            raise RuntimeError(
                "No samples were written. Check shard_id/num_shards and generation settings."
            )
