from typing import Dict, List, Optional

import numpy as np
import pyrender
import trimesh

from ip.utils.common_utils import downsample_pcd
from ip.generation.config import CameraConfig


class DepthRenderer:
    def __init__(
        self,
        cameras: List[CameraConfig],
        downsample_voxel: Optional[float] = None,
        max_points_per_obs: Optional[int] = None,
        gripper_mesh: Optional[trimesh.Trimesh] = None,
        visual_width: Optional[int] = None,
        visual_height: Optional[int] = None,
    ):
        if not cameras:
            raise ValueError("At least one camera config is required.")
        self.cameras = cameras
        self.downsample_voxel = downsample_voxel
        self.max_points_per_obs = max_points_per_obs
        self.obs_renderer = pyrender.OffscreenRenderer(cameras[0].width, cameras[0].height)
        self.visual_width = int(visual_width) if visual_width is not None else int(cameras[0].width)
        self.visual_height = int(visual_height) if visual_height is not None else int(cameras[0].height)
        self.visual_renderer = pyrender.OffscreenRenderer(self.visual_width, self.visual_height)
        # Pyrender mesh primitives are renderer-context bound. Keep per-context caches.
        self.mesh_cache_obs: Dict[int, pyrender.Mesh] = {}
        self.mesh_cache_visual: Dict[int, pyrender.Mesh] = {}
        if gripper_mesh is None:
            gripper_mesh = trimesh.creation.icosphere(radius=0.015)
        self.gripper_mesh = pyrender.Mesh.from_trimesh(gripper_mesh, smooth=False)
        blue_gripper = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.12, 0.32, 0.95, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.7,
        )
        self.gripper_mesh_visual = pyrender.Mesh.from_trimesh(
            gripper_mesh, material=blue_gripper, smooth=False
        )

    def _mesh_key(self, mesh):
        return id(mesh)

    def _get_mesh(self, mesh, for_visual: bool = False):
        key = self._mesh_key(mesh)
        cache = self.mesh_cache_visual if for_visual else self.mesh_cache_obs
        if key not in cache:
            cache[key] = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        return cache[key]

    def _depth_to_pointcloud(self, depth: np.ndarray, cam: CameraConfig):
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth
        mask = (z > cam.z_near) & (z < cam.z_far)
        if not np.any(mask):
            return np.zeros((0, 3), dtype=np.float32)
        x = (u - cam.cx) * z / cam.fx
        y = (v - cam.cy) * z / cam.fy
        points = np.stack([x, y, z], axis=-1)
        points = points[mask]
        points_world = (cam.pose[:3, :3] @ points.T).T + cam.pose[:3, 3]
        return points_world.astype(np.float32)

    def render_observation(self, scene, visual_idx: Optional[int] = None):
        pyr_scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.5, 0.5, 0.5])
        for obj in scene.objects:
            mesh = self._get_mesh(obj.mesh, for_visual=False)
            pyr_scene.add(mesh, pose=obj.pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        pyr_scene.add(light, pose=np.eye(4))

        pcds = []
        vis_color = None
        vis_depth = None
        for idx, cam in enumerate(self.cameras):
            camera = pyrender.IntrinsicsCamera(
                fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy, znear=cam.z_near, zfar=cam.z_far
            )
            cam_node = pyr_scene.add(camera, pose=cam.pose)
            color, depth = self.obs_renderer.render(pyr_scene)
            pyr_scene.remove_node(cam_node)
            pcd = self._depth_to_pointcloud(depth, cam)
            if pcd.size > 0:
                pcds.append(pcd)
            if visual_idx is not None and idx == visual_idx:
                vis_color = color
                vis_depth = depth

        if not pcds:
            empty = np.zeros((0, 3), dtype=np.float32)
            if visual_idx is not None:
                return empty, vis_color, vis_depth
            return empty
        points = np.concatenate(pcds, axis=0)
        if self.downsample_voxel is not None:
            points = downsample_pcd(points, voxel_size=self.downsample_voxel)
        if self.max_points_per_obs is not None and len(points) > self.max_points_per_obs:
            idx = np.random.choice(len(points), size=self.max_points_per_obs, replace=False)
            points = points[idx]
        points = points.astype(np.float32)
        if visual_idx is not None:
            return points, vis_color, vis_depth
        return points

    def render_visual(self, scene, gripper_pose: np.ndarray, visual_idx: int):
        pyr_scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.5, 0.5, 0.5])
        for obj in scene.objects:
            mesh = self._get_mesh(obj.mesh, for_visual=True)
            pyr_scene.add(mesh, pose=obj.pose)
        pyr_scene.add(self.gripper_mesh_visual, pose=gripper_pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        pyr_scene.add(light, pose=np.eye(4))
        cam = self.cameras[visual_idx]
        sx = float(self.visual_width) / float(cam.width)
        sy = float(self.visual_height) / float(cam.height)
        camera = pyrender.IntrinsicsCamera(
            fx=cam.fx * sx,
            fy=cam.fy * sy,
            cx=cam.cx * sx,
            cy=cam.cy * sy,
            znear=cam.z_near,
            zfar=cam.z_far,
        )
        cam_node = pyr_scene.add(camera, pose=cam.pose)
        color, depth = self.visual_renderer.render(pyr_scene)
        pyr_scene.remove_node(cam_node)
        return color, depth
