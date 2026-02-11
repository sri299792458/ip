from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from ip.generation.geometry import make_transform
from ip.generation.scene_builder import Scene
from ip.generation_bimanual.config import PERACT2_BIMANUAL_VARIATIONS


OPEN = 1
CLOSED = 0


@dataclass
class PrimitiveTrajectory:
    task_name: str
    variation_index: int
    left_seq: np.ndarray  # [T, 4, 4]
    right_seq: np.ndarray  # [T, 4, 4]
    grip_left: np.ndarray  # [T]
    grip_right: np.ndarray  # [T]
    left_targets: np.ndarray  # [T] int, -1 means no target.
    right_targets: np.ndarray  # [T] int, -1 means no target.
    scripted_object_poses: Dict[int, np.ndarray] = field(default_factory=dict)  # idx -> [T,4,4]


# -----------------------------
# Generic helpers
# -----------------------------

def _pose(x: float, y: float, z: float, yaw: float = 0.0) -> np.ndarray:
    # Keep gripper z-axis roughly downward to the tabletop.
    R = Rot.from_euler("xyz", [np.pi, 0.0, yaw]).as_matrix().astype(np.float32)
    return make_transform(R, np.array([x, y, z], dtype=np.float32))


def _interp_pose_track(total_steps: int, keyframes: List[Tuple[int, np.ndarray]]) -> np.ndarray:
    keyframes = sorted(keyframes, key=lambda x: x[0])
    if keyframes[0][0] != 0:
        keyframes = [(0, keyframes[0][1])] + keyframes
    if keyframes[-1][0] != total_steps - 1:
        keyframes = keyframes + [(total_steps - 1, keyframes[-1][1])]

    out = np.zeros((total_steps, 4, 4), dtype=np.float32)
    for i in range(len(keyframes) - 1):
        s0, T0 = keyframes[i]
        s1, T1 = keyframes[i + 1]
        if s1 < s0:
            raise ValueError("Keyframes must be non-decreasing")
        if s1 == s0:
            out[s0] = T0
            continue

        ts = np.arange(s0, s1 + 1)
        alpha = (ts - s0).astype(np.float32) / float(s1 - s0)

        p0 = T0[:3, 3]
        p1 = T1[:3, 3]
        pos = (1.0 - alpha[:, None]) * p0[None, :] + alpha[:, None] * p1[None, :]

        r0 = Rot.from_matrix(T0[:3, :3])
        r1 = Rot.from_matrix(T1[:3, :3])
        slerp = Slerp([float(s0), float(s1)], Rot.concatenate([r0, r1]))
        rots = slerp(ts.astype(np.float64)).as_matrix().astype(np.float32)

        seg = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], len(ts), axis=0)
        seg[:, :3, :3] = rots
        seg[:, :3, 3] = pos
        out[s0 : s1 + 1] = seg

    return out


def _piecewise_grip(total_steps: int, key_states: List[Tuple[int, int]]) -> np.ndarray:
    key_states = sorted(key_states, key=lambda x: x[0])
    if key_states[0][0] != 0:
        key_states = [(0, key_states[0][1])] + key_states
    out = np.zeros((total_steps,), dtype=np.float32)
    for i in range(len(key_states)):
        s0, g = key_states[i]
        s1 = total_steps if i + 1 == len(key_states) else key_states[i + 1][0]
        out[s0:s1] = float(g)
    return out


def _target_seq(total_steps: int, default: int = -1) -> np.ndarray:
    return np.full((total_steps,), int(default), dtype=np.int64)


def _obj_center(scene: Scene, obj_idx: int) -> np.ndarray:
    return np.array(scene.objects[obj_idx].pose[:3, 3], dtype=np.float32)


def _obj_size(scene: Scene, obj_idx: int) -> float:
    return float(np.max(scene.objects[obj_idx].mesh.extents))


def _choose_indices(scene: Scene, count: int, rng: np.random.Generator) -> List[int]:
    n = len(scene.objects)
    if n < 1:
        raise RuntimeError("Scene has no objects")
    if n >= count:
        return [int(x) for x in rng.choice(np.arange(n), size=count, replace=False)]
    return [int(x) for x in rng.choice(np.arange(n), size=count, replace=True)]


def _extrema_x_indices(scene: Scene) -> Tuple[int, int]:
    if len(scene.objects) < 2:
        idx = 0
        return idx, idx
    xs = np.array([obj.pose[0, 3] for obj in scene.objects], dtype=np.float32)
    return int(np.argmin(xs)), int(np.argmax(xs))


def _make_common_steps(min_steps: int, max_steps: int, rng: np.random.Generator) -> Tuple[int, int, int, int]:
    steps = int(rng.integers(min_steps, max_steps + 1))
    s_a = max(2, steps // 4)
    s_b = max(s_a + 1, steps // 2)
    s_c = max(s_b + 1, int(0.75 * steps))
    return steps, s_a, s_b, s_c


def _home_left(cfg):
    return _pose(float(cfg.left_home[0]), float(cfg.left_home[1]), float(cfg.left_home[2]), yaw=np.deg2rad(180.0))


def _home_right(cfg):
    return _pose(float(cfg.right_home[0]), float(cfg.right_home[1]), float(cfg.right_home[2]), yaw=0.0)


def _script_linear_pose(T0: np.ndarray, T1: np.ndarray, steps: int, s0: int, s1: int) -> np.ndarray:
    seq = np.repeat(T0[None, :, :], steps, axis=0).astype(np.float32)
    if s1 <= s0:
        seq[s0:] = T1
        return seq
    for t in range(steps):
        if t <= s0:
            seq[t] = T0
        elif t >= s1:
            seq[t] = T1
        else:
            a = (t - s0) / float(s1 - s0)
            T = np.array(T0, copy=True)
            T[:3, 3] = (1.0 - a) * T0[:3, 3] + a * T1[:3, 3]
            seq[t] = T.astype(np.float32)
    return seq


# -----------------------------
# Primitive families
# -----------------------------

def _primitive_cooperative_lift(task_name: str, variation: int, cfg, scene: Scene, rng: np.random.Generator) -> PrimitiveTrajectory:
    del variation
    obj_idx = _choose_indices(scene, 1, rng)[0]
    c = _obj_center(scene, obj_idx)
    size = _obj_size(scene, obj_idx)

    steps, s_a, s_b, s_c = _make_common_steps(cfg.min_steps, cfg.max_steps, rng)
    side = max(0.06, 0.55 * size)
    z_touch = c[2] + 0.02
    z_pre = z_touch + 0.08
    z_lift = z_touch + max(0.12, 1.2 * size)

    left = _interp_pose_track(
        steps,
        [
            (0, _home_left(cfg)),
            (s_a, _pose(float(c[0] - side), float(c[1]), float(z_pre), yaw=np.deg2rad(180))),
            (s_b, _pose(float(c[0] - side), float(c[1]), float(z_touch), yaw=np.deg2rad(180))),
            (s_c, _pose(float(c[0] - side), float(c[1]), float(z_lift), yaw=np.deg2rad(180))),
            (steps - 1, _pose(float(c[0] - side), float(c[1]), float(z_lift), yaw=np.deg2rad(180))),
        ],
    )
    right = _interp_pose_track(
        steps,
        [
            (0, _home_right(cfg)),
            (s_a, _pose(float(c[0] + side), float(c[1]), float(z_pre), yaw=0.0)),
            (s_b, _pose(float(c[0] + side), float(c[1]), float(z_touch), yaw=0.0)),
            (s_c, _pose(float(c[0] + side), float(c[1]), float(z_lift), yaw=0.0)),
            (steps - 1, _pose(float(c[0] + side), float(c[1]), float(z_lift), yaw=0.0)),
        ],
    )

    grip_left = _piecewise_grip(steps, [(0, OPEN), (s_b, CLOSED)])
    grip_right = _piecewise_grip(steps, [(0, OPEN), (s_b, CLOSED)])

    left_targets = _target_seq(steps)
    right_targets = _target_seq(steps)
    left_targets[s_b:] = obj_idx
    right_targets[s_b:] = obj_idx

    return PrimitiveTrajectory(
        task_name=task_name,
        variation_index=0,
        left_seq=left,
        right_seq=right,
        grip_left=grip_left,
        grip_right=grip_right,
        left_targets=left_targets,
        right_targets=right_targets,
    )


def _primitive_dual_push_sync(task_name: str, variation: int, cfg, scene: Scene, rng: np.random.Generator) -> PrimitiveTrajectory:
    del variation
    idx_l, idx_r = _extrema_x_indices(scene)
    c_l = _obj_center(scene, idx_l)
    c_r = _obj_center(scene, idx_r)

    steps, s_a, s_b, s_c = _make_common_steps(cfg.min_steps, cfg.max_steps, rng)
    left = _interp_pose_track(
        steps,
        [
            (0, _home_left(cfg)),
            (s_a, _pose(float(c_l[0]), float(c_l[1]), float(c_l[2] + 0.09), yaw=np.deg2rad(180))),
            (s_b, _pose(float(c_l[0]), float(c_l[1]), float(c_l[2] + 0.02), yaw=np.deg2rad(180))),
            (s_c, _pose(float(c_l[0]), float(c_l[1]), float(c_l[2] + 0.09), yaw=np.deg2rad(180))),
            (steps - 1, _home_left(cfg)),
        ],
    )
    right = _interp_pose_track(
        steps,
        [
            (0, _home_right(cfg)),
            (s_a, _pose(float(c_r[0]), float(c_r[1]), float(c_r[2] + 0.09), yaw=0.0)),
            (s_b, _pose(float(c_r[0]), float(c_r[1]), float(c_r[2] + 0.02), yaw=0.0)),
            (s_c, _pose(float(c_r[0]), float(c_r[1]), float(c_r[2] + 0.09), yaw=0.0)),
            (steps - 1, _home_right(cfg)),
        ],
    )

    return PrimitiveTrajectory(
        task_name=task_name,
        variation_index=0,
        left_seq=left,
        right_seq=right,
        grip_left=np.full((steps,), float(OPEN), dtype=np.float32),
        grip_right=np.full((steps,), float(OPEN), dtype=np.float32),
        left_targets=_target_seq(steps),
        right_targets=_target_seq(steps),
    )


def _primitive_dual_push_transport(task_name: str, variation: int, cfg, scene: Scene, rng: np.random.Generator) -> PrimitiveTrajectory:
    del variation
    obj_idx = _choose_indices(scene, 1, rng)[0]
    c = _obj_center(scene, obj_idx)
    size = _obj_size(scene, obj_idx)

    steps, s_a, s_b, s_c = _make_common_steps(cfg.min_steps, cfg.max_steps, rng)
    side = max(0.06, 0.45 * size)

    goal = np.array([c[0], min(c[1] + 0.22, cfg.workspace_bounds[1, 1] - 0.05), c[2]], dtype=np.float32)

    left = _interp_pose_track(
        steps,
        [
            (0, _home_left(cfg)),
            (s_a, _pose(float(c[0] - side), float(c[1]), float(c[2] + 0.05), yaw=np.deg2rad(180))),
            (s_b, _pose(float(goal[0] - side), float(goal[1]), float(goal[2] + 0.05), yaw=np.deg2rad(180))),
            (s_c, _pose(float(goal[0] - side), float(goal[1]), float(goal[2] + 0.08), yaw=np.deg2rad(180))),
            (steps - 1, _home_left(cfg)),
        ],
    )
    right = _interp_pose_track(
        steps,
        [
            (0, _home_right(cfg)),
            (s_a, _pose(float(c[0] + side), float(c[1]), float(c[2] + 0.05), yaw=0.0)),
            (s_b, _pose(float(goal[0] + side), float(goal[1]), float(goal[2] + 0.05), yaw=0.0)),
            (s_c, _pose(float(goal[0] + side), float(goal[1]), float(goal[2] + 0.08), yaw=0.0)),
            (steps - 1, _home_right(cfg)),
        ],
    )

    T0 = np.array(scene.objects[obj_idx].pose, copy=True)
    T1 = np.array(T0, copy=True)
    T1[:3, 3] = goal
    scripted = {obj_idx: _script_linear_pose(T0, T1, steps, s_a, s_b)}

    return PrimitiveTrajectory(
        task_name=task_name,
        variation_index=0,
        left_seq=left,
        right_seq=right,
        grip_left=np.full((steps,), float(OPEN), dtype=np.float32),
        grip_right=np.full((steps,), float(OPEN), dtype=np.float32),
        left_targets=_target_seq(steps),
        right_targets=_target_seq(steps),
        scripted_object_poses=scripted,
    )


def _primitive_container(task_name: str, variation: int, cfg, scene: Scene, rng: np.random.Generator) -> PrimitiveTrajectory:
    ids = _choose_indices(scene, 2, rng)
    item_idx, container_idx = ids[0], ids[1]

    item_c = _obj_center(scene, item_idx)
    cont_c = _obj_center(scene, container_idx)

    steps, s_a, s_b, s_c = _make_common_steps(cfg.min_steps, cfg.max_steps, rng)

    if task_name == "bimanual_put_item_in_drawer":
        z_off = [-0.04, 0.0, 0.04][int(variation) % 3]
        target = np.array([cont_c[0], cont_c[1], cont_c[2] + z_off], dtype=np.float32)
        release_step = min(steps - 2, s_c + 1)
    elif task_name == "bimanual_put_bottle_in_fridge":
        target = np.array([cont_c[0], cont_c[1], cont_c[2] + 0.03], dtype=np.float32)
        release_step = min(steps - 2, s_c + 1)
    else:
        # bimanual_take_tray_out_of_oven
        target = np.array([item_c[0], max(item_c[1] + 0.20, cont_c[1]), item_c[2]], dtype=np.float32)
        release_step = steps - 2

    left_handle = np.array([cont_c[0] - 0.06, cont_c[1], cont_c[2] + 0.07], dtype=np.float32)
    left = _interp_pose_track(
        steps,
        [
            (0, _home_left(cfg)),
            (s_a, _pose(float(left_handle[0]), float(left_handle[1]), float(left_handle[2] + 0.06), yaw=np.deg2rad(170))),
            (s_b, _pose(float(left_handle[0]), float(left_handle[1]), float(left_handle[2]), yaw=np.deg2rad(170))),
            (s_c, _pose(float(left_handle[0]), float(left_handle[1]), float(left_handle[2] + 0.03), yaw=np.deg2rad(170))),
            (steps - 1, _home_left(cfg)),
        ],
    )

    right = _interp_pose_track(
        steps,
        [
            (0, _home_right(cfg)),
            (s_a, _pose(float(item_c[0]), float(item_c[1]), float(item_c[2] + 0.09), yaw=0.0)),
            (s_b, _pose(float(item_c[0]), float(item_c[1]), float(item_c[2] + 0.02), yaw=0.0)),
            (s_c, _pose(float(target[0]), float(target[1]), float(target[2] + 0.02), yaw=0.0)),
            (steps - 1, _pose(float(target[0]), float(target[1]), float(target[2] + 0.07), yaw=0.0)),
        ],
    )

    grip_left = _piecewise_grip(steps, [(0, OPEN), (s_b, CLOSED), (s_c + 1, OPEN)])
    if task_name == "bimanual_take_tray_out_of_oven":
        grip_right = _piecewise_grip(steps, [(0, OPEN), (s_b, CLOSED), (release_step, OPEN)])
    else:
        grip_right = _piecewise_grip(steps, [(0, OPEN), (s_b, CLOSED), (release_step, OPEN)])

    left_targets = _target_seq(steps)
    right_targets = _target_seq(steps)
    left_targets[s_b : s_c + 1] = container_idx
    right_targets[s_b : release_step + 1] = item_idx

    return PrimitiveTrajectory(
        task_name=task_name,
        variation_index=int(variation),
        left_seq=left,
        right_seq=right,
        grip_left=grip_left,
        grip_right=grip_right,
        left_targets=left_targets,
        right_targets=right_targets,
    )


def _primitive_handover(task_name: str, variation: int, cfg, scene: Scene, rng: np.random.Generator) -> PrimitiveTrajectory:
    n = len(scene.objects)
    if task_name == "bimanual_handover_item" and n > 1:
        item_idx = int(variation % n)
    else:
        item_idx = _choose_indices(scene, 1, rng)[0]

    c = _obj_center(scene, item_idx)
    steps, s_a, s_b, s_c = _make_common_steps(cfg.min_steps, cfg.max_steps, rng)
    transfer = min(steps - 3, s_c)

    handover = np.array([0.0, c[1] + 0.02, max(c[2] + 0.14, cfg.table_height + 0.14)], dtype=np.float32)
    carry = np.array([0.18, handover[1], handover[2] + 0.03], dtype=np.float32)

    left = _interp_pose_track(
        steps,
        [
            (0, _home_left(cfg)),
            (s_a, _pose(float(c[0]), float(c[1]), float(c[2] + 0.09), yaw=np.deg2rad(180))),
            (s_b, _pose(float(c[0]), float(c[1]), float(c[2] + 0.02), yaw=np.deg2rad(180))),
            (transfer, _pose(float(handover[0] - 0.03), float(handover[1]), float(handover[2]), yaw=np.deg2rad(180))),
            (steps - 1, _home_left(cfg)),
        ],
    )
    right = _interp_pose_track(
        steps,
        [
            (0, _home_right(cfg)),
            (s_a, _pose(float(handover[0] + 0.05), float(handover[1]), float(handover[2] + 0.06), yaw=0.0)),
            (transfer, _pose(float(handover[0] + 0.02), float(handover[1]), float(handover[2]), yaw=0.0)),
            (steps - 2, _pose(float(carry[0]), float(carry[1]), float(carry[2]), yaw=0.0)),
            (steps - 1, _pose(float(carry[0]), float(carry[1]), float(carry[2] + 0.03), yaw=0.0)),
        ],
    )

    grip_left = _piecewise_grip(steps, [(0, OPEN), (s_b, CLOSED), (transfer + 1, OPEN)])
    grip_right = _piecewise_grip(steps, [(0, OPEN), (transfer, CLOSED)])

    left_targets = _target_seq(steps)
    right_targets = _target_seq(steps)
    left_targets[s_b : transfer + 1] = item_idx
    right_targets[transfer:] = item_idx

    return PrimitiveTrajectory(
        task_name=task_name,
        variation_index=int(variation),
        left_seq=left,
        right_seq=right,
        grip_left=grip_left,
        grip_right=grip_right,
        left_targets=left_targets,
        right_targets=right_targets,
    )


def _primitive_two_endpoint_tension(task_name: str, variation: int, cfg, scene: Scene, rng: np.random.Generator) -> PrimitiveTrajectory:
    del task_name, variation
    idx_l, idx_r = _extrema_x_indices(scene)
    c_l = _obj_center(scene, idx_l)
    c_r = _obj_center(scene, idx_r)

    steps, s_a, s_b, s_c = _make_common_steps(cfg.min_steps, cfg.max_steps, rng)

    left = _interp_pose_track(
        steps,
        [
            (0, _home_left(cfg)),
            (s_a, _pose(float(c_l[0]), float(c_l[1]), float(c_l[2] + 0.09), yaw=np.deg2rad(180))),
            (s_b, _pose(float(c_l[0]), float(c_l[1]), float(c_l[2] + 0.02), yaw=np.deg2rad(180))),
            (s_c, _pose(-0.24, 0.04, cfg.table_height + 0.12, yaw=np.deg2rad(180))),
            (steps - 1, _pose(-0.25, 0.05, cfg.table_height + 0.12, yaw=np.deg2rad(180))),
        ],
    )
    right = _interp_pose_track(
        steps,
        [
            (0, _home_right(cfg)),
            (s_a, _pose(float(c_r[0]), float(c_r[1]), float(c_r[2] + 0.09), yaw=0.0)),
            (s_b, _pose(float(c_r[0]), float(c_r[1]), float(c_r[2] + 0.02), yaw=0.0)),
            (s_c, _pose(0.24, -0.04, cfg.table_height + 0.12, yaw=0.0)),
            (steps - 1, _pose(0.25, -0.05, cfg.table_height + 0.12, yaw=0.0)),
        ],
    )

    grip_left = _piecewise_grip(steps, [(0, OPEN), (s_b, CLOSED)])
    grip_right = _piecewise_grip(steps, [(0, OPEN), (s_b, CLOSED)])

    left_targets = _target_seq(steps)
    right_targets = _target_seq(steps)
    left_targets[s_b:] = idx_l
    right_targets[s_b:] = idx_r

    return PrimitiveTrajectory(
        task_name="bimanual_straighten_rope",
        variation_index=0,
        left_seq=left,
        right_seq=right,
        grip_left=grip_left,
        grip_right=grip_right,
        left_targets=left_targets,
        right_targets=right_targets,
    )


def _primitive_tool_plus_receptacle(task_name: str, variation: int, cfg, scene: Scene, rng: np.random.Generator) -> PrimitiveTrajectory:
    del task_name, variation
    idx = _choose_indices(scene, 3, rng)
    dustpan_idx, broom_idx, dirt_idx = idx[0], idx[1], idx[2]

    c_dust = _obj_center(scene, dustpan_idx)
    c_broom = _obj_center(scene, broom_idx)
    c_dirt = _obj_center(scene, dirt_idx)

    steps, s_a, s_b, s_c = _make_common_steps(cfg.min_steps, cfg.max_steps, rng)

    sweep_goal = np.array([c_dust[0] + 0.03, c_dust[1] + 0.02, c_broom[2]], dtype=np.float32)

    left = _interp_pose_track(
        steps,
        [
            (0, _home_left(cfg)),
            (s_a, _pose(float(c_dust[0]), float(c_dust[1]), float(c_dust[2] + 0.09), yaw=np.deg2rad(180))),
            (s_b, _pose(float(c_dust[0]), float(c_dust[1]), float(c_dust[2] + 0.02), yaw=np.deg2rad(180))),
            (steps - 1, _pose(float(c_dust[0]), float(c_dust[1]), float(c_dust[2] + 0.03), yaw=np.deg2rad(180))),
        ],
    )

    right = _interp_pose_track(
        steps,
        [
            (0, _home_right(cfg)),
            (s_a, _pose(float(c_broom[0]), float(c_broom[1]), float(c_broom[2] + 0.09), yaw=0.0)),
            (s_b, _pose(float(c_broom[0]), float(c_broom[1]), float(c_broom[2] + 0.02), yaw=0.0)),
            (s_c, _pose(float(c_dirt[0]), float(c_dirt[1]), float(c_dirt[2] + 0.03), yaw=0.0)),
            (steps - 1, _pose(float(sweep_goal[0]), float(sweep_goal[1]), float(sweep_goal[2] + 0.03), yaw=0.0)),
        ],
    )

    grip_left = _piecewise_grip(steps, [(0, OPEN), (s_b, CLOSED)])
    grip_right = _piecewise_grip(steps, [(0, OPEN), (s_b, CLOSED)])

    left_targets = _target_seq(steps)
    right_targets = _target_seq(steps)
    left_targets[s_b:] = dustpan_idx
    right_targets[s_b:] = broom_idx

    # Move dirt toward dustpan while sweeping, unless dirt is already attached by arm logic.
    T0 = np.array(scene.objects[dirt_idx].pose, copy=True)
    T1 = np.array(T0, copy=True)
    T1[:3, 3] = np.array([sweep_goal[0], sweep_goal[1], max(cfg.table_height + 0.01, T0[2, 3])], dtype=np.float32)
    scripted = {dirt_idx: _script_linear_pose(T0, T1, steps, s_c, steps - 1)}

    return PrimitiveTrajectory(
        task_name="bimanual_sweep_to_dustpan",
        variation_index=0,
        left_seq=left,
        right_seq=right,
        grip_left=grip_left,
        grip_right=grip_right,
        left_targets=left_targets,
        right_targets=right_targets,
        scripted_object_poses=scripted,
    )


TASK_TO_PRIMITIVE: Dict[str, Callable[[str, int, object, Scene, np.random.Generator], PrimitiveTrajectory]] = {
    "bimanual_push_box": _primitive_dual_push_transport,
    "bimanual_lift_ball": _primitive_cooperative_lift,
    "bimanual_dual_push_buttons": _primitive_dual_push_sync,
    "bimanual_pick_plate": _primitive_cooperative_lift,
    "bimanual_put_item_in_drawer": _primitive_container,
    "bimanual_put_bottle_in_fridge": _primitive_container,
    "bimanual_handover_item": _primitive_handover,
    "bimanual_pick_laptop": _primitive_cooperative_lift,
    "bimanual_straighten_rope": _primitive_two_endpoint_tension,
    "bimanual_sweep_to_dustpan": _primitive_tool_plus_receptacle,
    "bimanual_lift_tray": _primitive_cooperative_lift,
    "bimanual_handover_item_easy": _primitive_handover,
    "bimanual_take_tray_out_of_oven": _primitive_container,
}


def variation_count(task_name: str) -> int:
    return int(PERACT2_BIMANUAL_VARIATIONS.get(task_name, 1))


def sample_trajectory(task_name: str, cfg, scene: Scene, rng: np.random.Generator) -> PrimitiveTrajectory:
    if task_name not in TASK_TO_PRIMITIVE:
        raise ValueError(f"Unsupported bimanual task: {task_name}")

    v_count = variation_count(task_name)
    v_idx = int(rng.integers(0, v_count))
    traj = TASK_TO_PRIMITIVE[task_name](task_name, v_idx, cfg, scene, rng)

    if traj.left_seq.shape[0] <= cfg.pred_horizon:
        raise RuntimeError(
            f"Generated trajectory for {task_name} too short: {traj.left_seq.shape[0]} <= pred_horizon={cfg.pred_horizon}"
        )
    if traj.left_seq.shape != traj.right_seq.shape:
        raise RuntimeError("Left/right trajectory length mismatch")
    if traj.left_targets.shape[0] != traj.left_seq.shape[0]:
        raise RuntimeError("left_targets length mismatch")
    if traj.right_targets.shape[0] != traj.right_seq.shape[0]:
        raise RuntimeError("right_targets length mismatch")
    return traj
