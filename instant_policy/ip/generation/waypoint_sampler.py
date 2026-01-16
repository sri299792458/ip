from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ip.generation.geometry import make_transform, pose_from_approach, random_rotation, transform_points
from ip.generation.scene_builder import Scene


@dataclass
class WaypointSpec:
    pose: np.ndarray
    gripper_state: int
    obj_index: Optional[int] = None


@dataclass
class Waypoint:
    pose: np.ndarray
    gripper_state: int


class WaypointSampler:
    def __init__(
        self,
        bias_prob: float,
        num_waypoints_range: Tuple[int, int],
    ):
        self.bias_prob = bias_prob
        self.num_waypoints_range = num_waypoints_range

    def _sample_surface_point(self, obj, rng: np.random.Generator):
        idx = int(rng.integers(0, len(obj.surface_points)))
        point_local = obj.surface_points[idx]
        normal_local = obj.surface_normals[idx]
        point_world = transform_points(point_local[None, :], obj.pose)[0]
        normal_world = obj.pose[:3, :3] @ normal_local
        normal_world = normal_world / (np.linalg.norm(normal_world) + 1e-8)
        return point_world, normal_world

    def _sample_object_pose(self, obj, rng: np.random.Generator):
        point_world, normal_world = self._sample_surface_point(obj, rng)
        if rng.uniform(0.0, 1.0) < 0.7:
            approach = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            approach = -normal_world
        offset = rng.uniform(0.0, 0.05)
        position = point_world - approach * offset
        return pose_from_approach(position, approach, rng)

    def _sample_random_pose(self, bounds: np.ndarray, rng: np.random.Generator):
        pos = rng.uniform(bounds[:, 0], bounds[:, 1])
        R = random_rotation(rng)
        return make_transform(R, pos)

    def _to_local_spec(self, obj_index: Optional[int], world_pose: np.ndarray, scene: Scene, grip: int):
        if obj_index is None:
            return WaypointSpec(pose=world_pose, gripper_state=grip, obj_index=None)
        obj_pose = scene.objects[obj_index].pose
        local_pose = np.linalg.inv(obj_pose) @ world_pose
        return WaypointSpec(pose=local_pose, gripper_state=grip, obj_index=obj_index)

    def _random_waypoints(self, scene: Scene, rng: np.random.Generator):
        num_wp = int(rng.integers(self.num_waypoints_range[0], self.num_waypoints_range[1] + 1))
        waypoints = []
        grip_state = int(rng.integers(0, 2))
        change_indices = rng.choice(np.arange(1, num_wp), size=min(2, num_wp - 1), replace=False)
        change_indices = set(change_indices.tolist())
        for i in range(num_wp):
            obj_index = int(rng.integers(0, len(scene.objects)))
            obj = scene.objects[obj_index]
            pose = self._sample_object_pose(obj, rng)
            if i in change_indices:
                grip_state = 1 - grip_state
            waypoints.append(self._to_local_spec(obj_index, pose, scene, grip_state))
        return waypoints

    def _grasp_waypoints(self, scene: Scene, rng: np.random.Generator):
        obj_index = int(rng.integers(0, len(scene.objects)))
        obj = scene.objects[obj_index]
        grasp_pose = self._sample_object_pose(obj, rng)
        approach = grasp_pose[:3, 2]
        pre_pose = grasp_pose.copy()
        pre_pose[:3, 3] -= approach * rng.uniform(0.05, 0.1)
        lift_pose = grasp_pose.copy()
        lift_pose[:3, 3] += np.array([0.0, 0.0, rng.uniform(0.05, 0.15)], dtype=np.float32)
        waypoints = [
            self._to_local_spec(obj_index, pre_pose, scene, 0),
            self._to_local_spec(obj_index, grasp_pose, scene, 1),
            self._to_local_spec(obj_index, lift_pose, scene, 1),
        ]
        return waypoints

    def _pick_place_waypoints(self, scene: Scene, rng: np.random.Generator):
        obj_index = int(rng.integers(0, len(scene.objects)))
        obj = scene.objects[obj_index]
        grasp_pose = self._sample_object_pose(obj, rng)
        approach = grasp_pose[:3, 2]
        pre_pose = grasp_pose.copy()
        pre_pose[:3, 3] -= approach * rng.uniform(0.05, 0.1)
        lift_pose = grasp_pose.copy()
        lift_pose[:3, 3] += np.array([0.0, 0.0, rng.uniform(0.08, 0.18)], dtype=np.float32)
        if len(scene.objects) > 1:
            place_obj_index = (obj_index + 1) % len(scene.objects)
            place_obj = scene.objects[place_obj_index]
            place_pose = self._sample_object_pose(place_obj, rng)
            place_spec = self._to_local_spec(place_obj_index, place_pose, scene, 1)
        else:
            place_pose = self._sample_random_pose(scene.workspace_bounds, rng)
            place_pose[:3, 3][2] = max(place_pose[:3, 3][2], scene.table_height + 0.08)
            place_spec = self._to_local_spec(None, place_pose, scene, 1)
        release_pose = place_spec.pose.copy()
        waypoints = [
            self._to_local_spec(obj_index, pre_pose, scene, 0),
            self._to_local_spec(obj_index, grasp_pose, scene, 1),
            self._to_local_spec(obj_index, lift_pose, scene, 1),
            place_spec,
            WaypointSpec(pose=release_pose, gripper_state=0, obj_index=place_spec.obj_index),
        ]
        return waypoints

    def _open_waypoints(self, scene: Scene, rng: np.random.Generator):
        obj_index = int(rng.integers(0, len(scene.objects)))
        obj = scene.objects[obj_index]
        grasp_pose = self._sample_object_pose(obj, rng)
        approach = grasp_pose[:3, 2]
        pre_pose = grasp_pose.copy()
        pre_pose[:3, 3] -= approach * rng.uniform(0.05, 0.1)
        pull_dir = grasp_pose[:3, 0]
        pull_pose = grasp_pose.copy()
        pull_pose[:3, 3] += pull_dir * rng.uniform(0.05, 0.2)
        waypoints = [
            self._to_local_spec(obj_index, pre_pose, scene, 0),
            self._to_local_spec(obj_index, grasp_pose, scene, 1),
            self._to_local_spec(obj_index, pull_pose, scene, 1),
        ]
        return waypoints

    def _close_waypoints(self, scene: Scene, rng: np.random.Generator):
        obj_index = int(rng.integers(0, len(scene.objects)))
        obj = scene.objects[obj_index]
        grasp_pose = self._sample_object_pose(obj, rng)
        approach = grasp_pose[:3, 2]
        pre_pose = grasp_pose.copy()
        pre_pose[:3, 3] -= approach * rng.uniform(0.05, 0.1)
        push_dir = -grasp_pose[:3, 0]
        push_pose = grasp_pose.copy()
        push_pose[:3, 3] += push_dir * rng.uniform(0.05, 0.2)
        waypoints = [
            self._to_local_spec(obj_index, pre_pose, scene, 0),
            self._to_local_spec(obj_index, grasp_pose, scene, 1),
            self._to_local_spec(obj_index, push_pose, scene, 1),
        ]
        return waypoints

    def sample_waypoint_specs(self, scene: Scene, rng: np.random.Generator) -> List[WaypointSpec]:
        if rng.uniform(0.0, 1.0) < self.bias_prob:
            skill = rng.choice(["grasp", "pick_place", "open", "close"])
            if skill == "grasp":
                return self._grasp_waypoints(scene, rng)
            if skill == "pick_place":
                return self._pick_place_waypoints(scene, rng)
            if skill == "open":
                return self._open_waypoints(scene, rng)
            return self._close_waypoints(scene, rng)
        return self._random_waypoints(scene, rng)

    def resolve_waypoints(self, scene: Scene, specs: List[WaypointSpec]) -> List[Waypoint]:
        waypoints: List[Waypoint] = []
        for spec in specs:
            if spec.obj_index is None:
                pose = spec.pose
            else:
                obj_pose = scene.objects[spec.obj_index].pose
                pose = obj_pose @ spec.pose
            waypoints.append(Waypoint(pose=pose, gripper_state=int(spec.gripper_state)))
        return waypoints
