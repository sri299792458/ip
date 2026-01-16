import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import trimesh

from ip.generation.geometry import make_transform, transform_points


@dataclass
class ObjectInstance:
    mesh: trimesh.Trimesh
    pose: np.ndarray
    mesh_path: str
    category: str
    name: str
    surface_points: np.ndarray
    surface_normals: np.ndarray

    def copy(self):
        return ObjectInstance(
            mesh=self.mesh,
            pose=np.array(self.pose, copy=True),
            mesh_path=self.mesh_path,
            category=self.category,
            name=self.name,
            surface_points=self.surface_points,
            surface_normals=self.surface_normals,
        )


@dataclass
class Scene:
    objects: List[ObjectInstance]
    workspace_bounds: np.ndarray
    table_height: float

    def copy(self):
        return Scene(
            objects=[obj.copy() for obj in self.objects],
            workspace_bounds=np.array(self.workspace_bounds, copy=True),
            table_height=float(self.table_height),
        )


class SceneBuilder:
    def __init__(
        self,
        shapenet_path: str,
        workspace_bounds: np.ndarray,
        table_height: float,
        object_scale_range: Tuple[float, float],
        num_objects_range: Tuple[int, int],
        shapenet_index_path: Optional[str] = None,
        max_meshes: Optional[int] = None,
        cache_meshes: bool = False,
        surface_sample_count: int = 512,
    ):
        self.shapenet_path = shapenet_path
        self.workspace_bounds = workspace_bounds
        self.table_height = table_height
        self.object_scale_range = object_scale_range
        self.num_objects_range = num_objects_range
        self.shapenet_index_path = shapenet_index_path
        self.max_meshes = max_meshes
        self.cache_meshes = cache_meshes
        self.surface_sample_count = surface_sample_count
        self.mesh_cache = {}

        self.mesh_index = self._load_mesh_index()
        if not self.mesh_index:
            raise RuntimeError(f"No meshes found under {self.shapenet_path}")

    def _load_mesh_index(self):
        if self.shapenet_index_path and os.path.exists(self.shapenet_index_path):
            with open(self.shapenet_index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data

        mesh_paths = []
        exts = (".obj", ".ply", ".stl")
        for root, _, files in os.walk(self.shapenet_path):
            for fname in files:
                if fname.endswith(exts):
                    if fname not in ("model.obj", "model_normalized.obj"):
                        continue
                    mesh_paths.append(os.path.join(root, fname))

        if not mesh_paths:
            for root, _, files in os.walk(self.shapenet_path):
                for fname in files:
                    if fname.endswith(exts):
                        mesh_paths.append(os.path.join(root, fname))

        if self.max_meshes is not None:
            mesh_paths = mesh_paths[: self.max_meshes]

        if self.shapenet_index_path:
            with open(self.shapenet_index_path, "w", encoding="utf-8") as f:
                json.dump(mesh_paths, f)
        return mesh_paths

    def _load_mesh_base(self, mesh_path: str):
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        if not isinstance(mesh, trimesh.Trimesh):
            raise RuntimeError(f"Unsupported mesh type: {type(mesh)} for {mesh_path}")
        if mesh.vertices.size == 0:
            raise RuntimeError(f"Empty mesh: {mesh_path}")
        mesh.remove_unreferenced_vertices()
        center = mesh.bounding_box.centroid
        mesh.apply_translation(-center)
        extents = mesh.extents
        max_extent = float(np.max(extents))
        if max_extent < 1e-6:
            raise RuntimeError(f"Degenerate mesh: {mesh_path}")
        mesh.apply_scale(1.0 / max_extent)
        return mesh

    def _get_mesh(self, mesh_path: str):
        if self.cache_meshes:
            if mesh_path not in self.mesh_cache:
                self.mesh_cache[mesh_path] = self._load_mesh_base(mesh_path)
            return self.mesh_cache[mesh_path].copy()
        return self._load_mesh_base(mesh_path)

    def _sample_object(self, rng: np.random.Generator):
        mesh_path = rng.choice(self.mesh_index)
        mesh = self._get_mesh(mesh_path)
        scale = rng.uniform(self.object_scale_range[0], self.object_scale_range[1])
        mesh.apply_scale(scale)
        points, face_idx = trimesh.sample.sample_surface(mesh, self.surface_sample_count)
        normals = mesh.face_normals[face_idx]
        category = os.path.relpath(mesh_path, self.shapenet_path).split(os.sep)[0]
        name = os.path.basename(mesh_path)
        obj = ObjectInstance(
            mesh=mesh,
            pose=np.eye(4, dtype=np.float32),
            mesh_path=mesh_path,
            category=category,
            name=name,
            surface_points=points.astype(np.float32),
            surface_normals=normals.astype(np.float32),
        )
        return obj

    def _aabb_xy(self, mesh: trimesh.Trimesh, pose: np.ndarray):
        verts = transform_points(mesh.vertices, pose)
        min_xy = np.min(verts[:, :2], axis=0)
        max_xy = np.max(verts[:, :2], axis=0)
        return min_xy, max_xy

    def _overlaps(self, a_min, a_max, b_min, b_max, margin=0.01):
        return not (a_max[0] + margin < b_min[0] or a_min[0] - margin > b_max[0] or
                    a_max[1] + margin < b_min[1] or a_min[1] - margin > b_max[1])

    def _sample_pose(self, obj: ObjectInstance, existing: List[ObjectInstance], rng: np.random.Generator):
        bounds = self.workspace_bounds
        for _ in range(50):
            x = rng.uniform(bounds[0, 0], bounds[0, 1])
            y = rng.uniform(bounds[1, 0], bounds[1, 1])
            yaw = rng.uniform(0.0, 2.0 * np.pi)
            Rz = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)
            pose = make_transform(Rz, np.array([x, y, 0.0], dtype=np.float32))
            min_z = np.min(transform_points(obj.mesh.vertices, pose)[:, 2])
            pose[2, 3] += self.table_height - min_z

            cand_min, cand_max = self._aabb_xy(obj.mesh, pose)
            collide = False
            for other in existing:
                other_min, other_max = self._aabb_xy(other.mesh, other.pose)
                if self._overlaps(cand_min, cand_max, other_min, other_max):
                    collide = True
                    break
            if not collide:
                return pose
        return pose

    def generate_scene(self, rng: np.random.Generator):
        num_objects = rng.integers(self.num_objects_range[0], self.num_objects_range[1] + 1)
        objects: List[ObjectInstance] = []
        for _ in range(num_objects):
            obj = self._sample_object(rng)
            pose = self._sample_pose(obj, objects, rng)
            obj.pose = pose.astype(np.float32)
            objects.append(obj)
        return Scene(objects=objects, workspace_bounds=self.workspace_bounds, table_height=self.table_height)
