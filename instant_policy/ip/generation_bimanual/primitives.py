from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from ip.generation.geometry import make_transform, transform_points
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
    scene_points_seq: List[np.ndarray]  # len T, each [N, 3]


def _pose(x: float, y: float, z: float, yaw: float = 0.0) -> np.ndarray:
    # Gripper z-axis points roughly down to table.
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
            raise ValueError("Keyframes must be non-decreasing in step index")
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


def _sample_box_surface(size_xyz: Tuple[float, float, float], n: int, rng: np.random.Generator) -> np.ndarray:
    sx, sy, sz = size_xyz
    face = rng.integers(0, 6, size=n)
    u = rng.uniform(-0.5, 0.5, size=(n, 2)).astype(np.float32)
    p = np.zeros((n, 3), dtype=np.float32)

    # x faces
    m = face == 0
    p[m, 0] = +0.5 * sx
    p[m, 1] = u[m, 0] * sy
    p[m, 2] = u[m, 1] * sz
    m = face == 1
    p[m, 0] = -0.5 * sx
    p[m, 1] = u[m, 0] * sy
    p[m, 2] = u[m, 1] * sz

    # y faces
    m = face == 2
    p[m, 1] = +0.5 * sy
    p[m, 0] = u[m, 0] * sx
    p[m, 2] = u[m, 1] * sz
    m = face == 3
    p[m, 1] = -0.5 * sy
    p[m, 0] = u[m, 0] * sx
    p[m, 2] = u[m, 1] * sz

    # z faces
    m = face == 4
    p[m, 2] = +0.5 * sz
    p[m, 0] = u[m, 0] * sx
    p[m, 1] = u[m, 1] * sy
    m = face == 5
    p[m, 2] = -0.5 * sz
    p[m, 0] = u[m, 0] * sx
    p[m, 1] = u[m, 1] * sy

    return p


def _sample_cylinder_surface(radius: float, height: float, n: int, rng: np.random.Generator) -> np.ndarray:
    p = np.zeros((n, 3), dtype=np.float32)
    mode = rng.integers(0, 3, size=n)  # side/top/bottom
    ang = rng.uniform(0.0, 2.0 * np.pi, size=n)
    z = rng.uniform(-0.5 * height, 0.5 * height, size=n)

    m = mode == 0
    p[m, 0] = radius * np.cos(ang[m])
    p[m, 1] = radius * np.sin(ang[m])
    p[m, 2] = z[m]

    # caps
    r = np.sqrt(rng.uniform(0.0, 1.0, size=n)) * radius
    x = r * np.cos(ang)
    y = r * np.sin(ang)

    m = mode == 1
    p[m, 0] = x[m]
    p[m, 1] = y[m]
    p[m, 2] = +0.5 * height

    m = mode == 2
    p[m, 0] = x[m]
    p[m, 1] = y[m]
    p[m, 2] = -0.5 * height
    return p


def _sample_sphere_surface(radius: float, n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=(n, 3)).astype(np.float32)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    return radius * v


def _sample_disc(radius: float, thickness: float, n: int, rng: np.random.Generator) -> np.ndarray:
    return _sample_cylinder_surface(radius=radius, height=thickness, n=n, rng=rng)


def _sample_table(bounds: np.ndarray, z: float, n: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.uniform(bounds[0, 0], bounds[0, 1], size=n)
    y = rng.uniform(bounds[1, 0], bounds[1, 1], size=n)
    zz = np.full((n,), z, dtype=np.float32)
    return np.stack([x, y, zz], axis=1).astype(np.float32)


def _attach_points(local_points: np.ndarray, T_seq: np.ndarray) -> List[np.ndarray]:
    return [transform_points(local_points, T_seq[t]).astype(np.float32) for t in range(T_seq.shape[0])]


def _midpoint_pose(T_a: np.ndarray, T_b: np.ndarray, z_offset: float = 0.0) -> np.ndarray:
    p = 0.5 * (T_a[:3, 3] + T_b[:3, 3])
    p[2] += float(z_offset)
    # Keep object upright by default.
    return _pose(float(p[0]), float(p[1]), float(p[2]), yaw=0.0)


def _subsample_points(points: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] <= n:
        return points
    idx = rng.choice(points.shape[0], size=n, replace=False)
    return points[idx]


def _make_common_steps(min_steps: int, max_steps: int, rng: np.random.Generator) -> Tuple[int, int, int, int]:
    steps = int(rng.integers(min_steps, max_steps + 1))
    s_a = max(2, steps // 4)
    s_b = max(s_a + 1, steps // 2)
    s_c = max(s_b + 1, int(0.75 * steps))
    return steps, s_a, s_b, s_c


def _primitive_cooperative_lift(task_name: str, variation: int, cfg, rng: np.random.Generator) -> PrimitiveTrajectory:
    del variation
    steps, s_a, s_b, s_c = _make_common_steps(cfg.min_steps, cfg.max_steps, rng)

    if task_name == "bimanual_lift_ball":
        obj_local = _sample_sphere_surface(radius=0.045, n=450, rng=rng)
        side = 0.055
        z_touch = 0.045
    elif task_name == "bimanual_pick_plate":
        obj_local = _sample_disc(radius=0.090, thickness=0.012, n=650, rng=rng)
        side = 0.100
        z_touch = 0.016
    elif task_name == "bimanual_pick_laptop":
        obj_local = _sample_box_surface((0.28, 0.20, 0.03), n=900, rng=rng)
        side = 0.130
        z_touch = 0.025
    else:
        # bimanual_lift_tray
        obj_local = _sample_box_surface((0.26, 0.18, 0.02), n=900, rng=rng)
        side = 0.120
        z_touch = 0.020

    obj_center = np.array([0.0, 0.02 + rng.uniform(-0.03, 0.03), cfg.table_height + z_touch], dtype=np.float32)

    left_track = _interp_pose_track(
        steps,
        [
            (0, _pose(*cfg.left_home, yaw=np.deg2rad(180))),
            (s_a, _pose(float(obj_center[0] - side), float(obj_center[1]), float(obj_center[2] + 0.08), yaw=np.deg2rad(180))),
            (s_b, _pose(float(obj_center[0] - side), float(obj_center[1]), float(obj_center[2]), yaw=np.deg2rad(180))),
            (s_c, _pose(float(obj_center[0] - side), float(obj_center[1]), float(obj_center[2] + 0.18), yaw=np.deg2rad(180))),
            (steps - 1, _pose(float(obj_center[0] - side), float(obj_center[1]), float(obj_center[2] + 0.18), yaw=np.deg2rad(180))),
        ],
    )
    right_track = _interp_pose_track(
        steps,
        [
            (0, _pose(*cfg.right_home, yaw=0.0)),
            (s_a, _pose(float(obj_center[0] + side), float(obj_center[1]), float(obj_center[2] + 0.08), yaw=0.0)),
            (s_b, _pose(float(obj_center[0] + side), float(obj_center[1]), float(obj_center[2]), yaw=0.0)),
            (s_c, _pose(float(obj_center[0] + side), float(obj_center[1]), float(obj_center[2] + 0.18), yaw=0.0)),
            (steps - 1, _pose(float(obj_center[0] + side), float(obj_center[1]), float(obj_center[2] + 0.18), yaw=0.0)),
        ],
    )

    grip_left = _piecewise_grip(steps, [(0, OPEN), (s_b, CLOSED)])
    grip_right = _piecewise_grip(steps, [(0, OPEN), (s_b, CLOSED)])

    table = _sample_table(cfg.workspace_bounds, cfg.table_height, n=1600, rng=rng)
    obj_seq = []
    T_obj0 = _pose(float(obj_center[0]), float(obj_center[1]), float(obj_center[2]), yaw=0.0)
    for t in range(steps):
        if t < s_b:
            obj_seq.append(T_obj0)
        else:
            obj_seq.append(_midpoint_pose(left_track[t], right_track[t], z_offset=-0.01))
    obj_points_seq = _attach_points(obj_local, np.stack(obj_seq, axis=0))

    scene_points_seq = [
        _subsample_points(np.concatenate([table, obj_points_seq[t]], axis=0), cfg.num_points, rng)
        for t in range(steps)
    ]

    return PrimitiveTrajectory(
        task_name=task_name,
        variation_index=0,
        left_seq=left_track,
        right_seq=right_track,
        grip_left=grip_left,
        grip_right=grip_right,
        scene_points_seq=scene_points_seq,
    )


def _primitive_dual_push_sync(task_name: str, variation: int, cfg, rng: np.random.Generator) -> PrimitiveTrajectory:
    del variation
    steps, s_a, s_b, s_c = _make_common_steps(cfg.min_steps, cfg.max_steps, rng)

    button_z = cfg.table_height + 0.035
    b_left = np.array([-0.14, 0.10, button_z], dtype=np.float32)
    b_right = np.array([0.14, 0.10, button_z], dtype=np.float32)

    left_track = _interp_pose_track(
        steps,
        [
            (0, _pose(*cfg.left_home, yaw=np.deg2rad(180))),
            (s_a, _pose(float(b_left[0]), float(b_left[1]), float(b_left[2] + 0.08), yaw=np.deg2rad(180))),
            (s_b, _pose(float(b_left[0]), float(b_left[1]), float(b_left[2] + 0.01), yaw=np.deg2rad(180))),
            (s_c, _pose(float(b_left[0]), float(b_left[1]), float(b_left[2] + 0.08), yaw=np.deg2rad(180))),
            (steps - 1, _pose(*cfg.left_home, yaw=np.deg2rad(180))),
        ],
    )
    right_track = _interp_pose_track(
        steps,
        [
            (0, _pose(*cfg.right_home, yaw=0.0)),
            (s_a, _pose(float(b_right[0]), float(b_right[1]), float(b_right[2] + 0.08), yaw=0.0)),
            (s_b, _pose(float(b_right[0]), float(b_right[1]), float(b_right[2] + 0.01), yaw=0.0)),
            (s_c, _pose(float(b_right[0]), float(b_right[1]), float(b_right[2] + 0.08), yaw=0.0)),
            (steps - 1, _pose(*cfg.right_home, yaw=0.0)),
        ],
    )

    grip_left = np.full((steps,), float(OPEN), dtype=np.float32)
    grip_right = np.full((steps,), float(OPEN), dtype=np.float32)

    table = _sample_table(cfg.workspace_bounds, cfg.table_height, n=1550, rng=rng)
    button_local = _sample_cylinder_surface(radius=0.022, height=0.03, n=250, rng=rng)
    cap_local = _sample_disc(radius=0.018, thickness=0.008, n=120, rng=rng)

    T_l = _pose(float(b_left[0]), float(b_left[1]), float(b_left[2]), yaw=0.0)
    T_r = _pose(float(b_right[0]), float(b_right[1]), float(b_right[2]), yaw=0.0)
    T_mid = _pose(0.0, 0.10 + rng.uniform(-0.04, 0.04), float(button_z), yaw=0.0)

    static = np.concatenate(
        [
            transform_points(button_local, T_l),
            transform_points(button_local, T_r),
            transform_points(button_local, T_mid),
            transform_points(cap_local, T_l),
            transform_points(cap_local, T_r),
        ],
        axis=0,
    ).astype(np.float32)
    scene_points_seq = [
        _subsample_points(np.concatenate([table, static], axis=0), cfg.num_points, rng)
        for _ in range(steps)
    ]

    return PrimitiveTrajectory(
        task_name=task_name,
        variation_index=0,
        left_seq=left_track,
        right_seq=right_track,
        grip_left=grip_left,
        grip_right=grip_right,
        scene_points_seq=scene_points_seq,
    )


def _primitive_dual_push_transport(task_name: str, variation: int, cfg, rng: np.random.Generator) -> PrimitiveTrajectory:
    del variation
    steps, s_a, s_b, s_c = _make_common_steps(cfg.min_steps, cfg.max_steps, rng)

    box_local = _sample_box_surface((0.10, 0.10, 0.08), n=750, rng=rng)
    box_start = np.array([0.0, -0.02, cfg.table_height + 0.04], dtype=np.float32)
    box_goal = np.array([0.0, 0.20, cfg.table_height + 0.04], dtype=np.float32)

    side = 0.075
    left_track = _interp_pose_track(
        steps,
        [
            (0, _pose(*cfg.left_home, yaw=np.deg2rad(180))),
            (s_a, _pose(float(box_start[0] - side), float(box_start[1]), float(box_start[2] + 0.04), yaw=np.deg2rad(180))),
            (s_b, _pose(float(box_goal[0] - side), float(box_goal[1]), float(box_goal[2] + 0.04), yaw=np.deg2rad(180))),
            (s_c, _pose(float(box_goal[0] - side), float(box_goal[1]), float(box_goal[2] + 0.06), yaw=np.deg2rad(180))),
            (steps - 1, _pose(*cfg.left_home, yaw=np.deg2rad(180))),
        ],
    )
    right_track = _interp_pose_track(
        steps,
        [
            (0, _pose(*cfg.right_home, yaw=0.0)),
            (s_a, _pose(float(box_start[0] + side), float(box_start[1]), float(box_start[2] + 0.04), yaw=0.0)),
            (s_b, _pose(float(box_goal[0] + side), float(box_goal[1]), float(box_goal[2] + 0.04), yaw=0.0)),
            (s_c, _pose(float(box_goal[0] + side), float(box_goal[1]), float(box_goal[2] + 0.06), yaw=0.0)),
            (steps - 1, _pose(*cfg.right_home, yaw=0.0)),
        ],
    )

    grip_left = np.full((steps,), float(OPEN), dtype=np.float32)
    grip_right = np.full((steps,), float(OPEN), dtype=np.float32)

    box_seq = []
    for t in range(steps):
        if t <= s_a:
            alpha = 0.0
        elif t >= s_b:
            alpha = 1.0
        else:
            alpha = (t - s_a) / float(max(1, s_b - s_a))
        pos = (1.0 - alpha) * box_start + alpha * box_goal
        box_seq.append(_pose(float(pos[0]), float(pos[1]), float(pos[2]), yaw=0.0))
    box_points_seq = _attach_points(box_local, np.stack(box_seq, axis=0))

    table = _sample_table(cfg.workspace_bounds, cfg.table_height, n=1500, rng=rng)
    target_area = _sample_box_surface((0.18, 0.16, 0.004), n=300, rng=rng)
    T_target = _pose(float(box_goal[0]), float(box_goal[1]), cfg.table_height + 0.002, yaw=0.0)
    target_points = transform_points(target_area, T_target).astype(np.float32)

    scene_points_seq = [
        _subsample_points(np.concatenate([table, target_points, box_points_seq[t]], axis=0), cfg.num_points, rng)
        for t in range(steps)
    ]

    return PrimitiveTrajectory(
        task_name=task_name,
        variation_index=0,
        left_seq=left_track,
        right_seq=right_track,
        grip_left=grip_left,
        grip_right=grip_right,
        scene_points_seq=scene_points_seq,
    )


def _primitive_container(task_name: str, variation: int, cfg, rng: np.random.Generator) -> PrimitiveTrajectory:
    steps, s_a, s_b, s_c = _make_common_steps(cfg.min_steps, cfg.max_steps, rng)

    if task_name == "bimanual_put_item_in_drawer":
        container_center = np.array([0.05, 0.16, cfg.table_height + 0.08], dtype=np.float32)
        start_obj = np.array([0.02, -0.04, cfg.table_height + 0.035], dtype=np.float32)
        target_obj = np.array([container_center[0], container_center[1], cfg.table_height + 0.06], dtype=np.float32)
        obj_local = _sample_box_surface((0.05, 0.05, 0.05), n=380, rng=rng)
        # Drawer height variation.
        target_obj[2] += 0.02 * float(variation - 1)
        mode = "put"
    elif task_name == "bimanual_put_bottle_in_fridge":
        container_center = np.array([0.16, 0.05, cfg.table_height + 0.18], dtype=np.float32)
        start_obj = np.array([0.00, -0.05, cfg.table_height + 0.055], dtype=np.float32)
        target_obj = np.array([container_center[0], container_center[1], cfg.table_height + 0.12], dtype=np.float32)
        obj_local = _sample_cylinder_surface(radius=0.028, height=0.14, n=520, rng=rng)
        mode = "put"
    else:
        # bimanual_take_tray_out_of_oven
        container_center = np.array([0.12, -0.14, cfg.table_height + 0.10], dtype=np.float32)
        start_obj = np.array([container_center[0], container_center[1], cfg.table_height + 0.09], dtype=np.float32)
        target_obj = np.array([0.0, 0.00, cfg.table_height + 0.09], dtype=np.float32)
        obj_local = _sample_box_surface((0.24, 0.16, 0.02), n=820, rng=rng)
        mode = "take"

    # Left arm stabilizes/opening handle, right arm manipulates object.
    handle_pos = np.array([container_center[0] - 0.05, container_center[1], container_center[2]], dtype=np.float32)

    left_track = _interp_pose_track(
        steps,
        [
            (0, _pose(*cfg.left_home, yaw=np.deg2rad(170))),
            (s_a, _pose(float(handle_pos[0]), float(handle_pos[1]), float(handle_pos[2] + 0.05), yaw=np.deg2rad(170))),
            (s_b, _pose(float(handle_pos[0]), float(handle_pos[1]), float(handle_pos[2] + 0.02), yaw=np.deg2rad(170))),
            (s_c, _pose(float(handle_pos[0]), float(handle_pos[1]), float(handle_pos[2] + 0.03), yaw=np.deg2rad(170))),
            (steps - 1, _pose(*cfg.left_home, yaw=np.deg2rad(170))),
        ],
    )

    pre_obj = start_obj + np.array([0.0, 0.0, 0.08], dtype=np.float32)
    right_track = _interp_pose_track(
        steps,
        [
            (0, _pose(*cfg.right_home, yaw=0.0)),
            (s_a, _pose(float(pre_obj[0]), float(pre_obj[1]), float(pre_obj[2]), yaw=0.0)),
            (s_b, _pose(float(start_obj[0]), float(start_obj[1]), float(start_obj[2] + 0.01), yaw=0.0)),
            (s_c, _pose(float(target_obj[0]), float(target_obj[1]), float(target_obj[2] + 0.02), yaw=0.0)),
            (steps - 1, _pose(float(target_obj[0]), float(target_obj[1]), float(target_obj[2] + 0.08), yaw=0.0)),
        ],
    )

    if mode == "put":
        grip_right = _piecewise_grip(steps, [(0, OPEN), (s_b, CLOSED), (s_c + 1, OPEN)])
    else:
        grip_right = _piecewise_grip(steps, [(0, OPEN), (s_b, CLOSED), (steps - 2, OPEN)])
    grip_left = _piecewise_grip(steps, [(0, OPEN), (s_b - 1, CLOSED), (s_c + 1, OPEN)])

    # Object follows right gripper while grasped.
    obj_seq = []
    T_start = _pose(float(start_obj[0]), float(start_obj[1]), float(start_obj[2]), yaw=0.0)
    T_target = _pose(float(target_obj[0]), float(target_obj[1]), float(target_obj[2]), yaw=0.0)
    was_grasped = False
    for t in range(steps):
        if grip_right[t] <= 0.5:
            was_grasped = True
            p = right_track[t][:3, 3] + np.array([0.0, 0.0, -0.02], dtype=np.float32)
            obj_seq.append(_pose(float(p[0]), float(p[1]), float(p[2]), yaw=0.0))
        else:
            if not was_grasped:
                obj_seq.append(T_start)
            else:
                obj_seq.append(T_target)
    obj_points_seq = _attach_points(obj_local, np.stack(obj_seq, axis=0))

    table = _sample_table(cfg.workspace_bounds, cfg.table_height, n=1450, rng=rng)
    container_shell_local = _sample_box_surface((0.26, 0.20, 0.20), n=900, rng=rng)
    T_container = _pose(float(container_center[0]), float(container_center[1]), float(container_center[2]), yaw=0.0)
    container_points = transform_points(container_shell_local, T_container).astype(np.float32)

    scene_points_seq = [
        _subsample_points(np.concatenate([table, container_points, obj_points_seq[t]], axis=0), cfg.num_points, rng)
        for t in range(steps)
    ]

    return PrimitiveTrajectory(
        task_name=task_name,
        variation_index=variation,
        left_seq=left_track,
        right_seq=right_track,
        grip_left=grip_left,
        grip_right=grip_right,
        scene_points_seq=scene_points_seq,
    )


def _primitive_handover(task_name: str, variation: int, cfg, rng: np.random.Generator) -> PrimitiveTrajectory:
    steps, s_a, s_b, s_c = _make_common_steps(cfg.min_steps, cfg.max_steps, rng)
    transfer_step = min(steps - 3, s_c)

    item_local = _sample_box_surface((0.05, 0.05, 0.06), n=420, rng=rng)
    item_start = np.array([-0.10, 0.00, cfg.table_height + 0.035], dtype=np.float32)
    handover_pos = np.array([0.0, 0.02, cfg.table_height + 0.16], dtype=np.float32)
    carry_pos = np.array([0.16, 0.02, cfg.table_height + 0.20], dtype=np.float32)

    left_track = _interp_pose_track(
        steps,
        [
            (0, _pose(*cfg.left_home, yaw=np.deg2rad(180))),
            (s_a, _pose(float(item_start[0]), float(item_start[1]), float(item_start[2] + 0.07), yaw=np.deg2rad(180))),
            (s_b, _pose(float(item_start[0]), float(item_start[1]), float(item_start[2] + 0.01), yaw=np.deg2rad(180))),
            (transfer_step, _pose(float(handover_pos[0] - 0.03), float(handover_pos[1]), float(handover_pos[2]), yaw=np.deg2rad(180))),
            (steps - 1, _pose(*cfg.left_home, yaw=np.deg2rad(180))),
        ],
    )

    right_track = _interp_pose_track(
        steps,
        [
            (0, _pose(*cfg.right_home, yaw=0.0)),
            (s_a, _pose(float(handover_pos[0] + 0.05), float(handover_pos[1]), float(handover_pos[2] + 0.06), yaw=0.0)),
            (transfer_step, _pose(float(handover_pos[0] + 0.02), float(handover_pos[1]), float(handover_pos[2]), yaw=0.0)),
            (steps - 2, _pose(float(carry_pos[0]), float(carry_pos[1]), float(carry_pos[2]), yaw=0.0)),
            (steps - 1, _pose(float(carry_pos[0]), float(carry_pos[1]), float(carry_pos[2] + 0.05), yaw=0.0)),
        ],
    )

    grip_left = _piecewise_grip(steps, [(0, OPEN), (s_b, CLOSED), (transfer_step + 1, OPEN)])
    grip_right = _piecewise_grip(steps, [(0, OPEN), (transfer_step, CLOSED)])

    obj_seq = []
    for t in range(steps):
        if t <= transfer_step:
            p = left_track[t][:3, 3] + np.array([0.0, 0.0, -0.015], dtype=np.float32)
        else:
            p = right_track[t][:3, 3] + np.array([0.0, 0.0, -0.015], dtype=np.float32)
        obj_seq.append(_pose(float(p[0]), float(p[1]), float(p[2]), yaw=0.0))
    obj_points_seq = _attach_points(item_local, np.stack(obj_seq, axis=0))

    table = _sample_table(cfg.workspace_bounds, cfg.table_height, n=1500, rng=rng)

    # Hard variant includes distractor objects (5 benchmark variations).
    n_distractors = 0
    if task_name == "bimanual_handover_item":
        n_distractors = 4
    distractors = []
    for _ in range(n_distractors):
        d_local = _sample_box_surface((0.05, 0.05, 0.05), n=160, rng=rng)
        dx = rng.uniform(-0.18, 0.18)
        dy = rng.uniform(-0.10, 0.16)
        dz = cfg.table_height + 0.03
        T_d = _pose(float(dx), float(dy), float(dz), yaw=rng.uniform(-np.pi, np.pi))
        distractors.append(transform_points(d_local, T_d))
    distractors = np.concatenate(distractors, axis=0).astype(np.float32) if distractors else np.zeros((0, 3), dtype=np.float32)

    scene_points_seq = [
        _subsample_points(np.concatenate([table, distractors, obj_points_seq[t]], axis=0), cfg.num_points, rng)
        for t in range(steps)
    ]

    return PrimitiveTrajectory(
        task_name=task_name,
        variation_index=variation,
        left_seq=left_track,
        right_seq=right_track,
        grip_left=grip_left,
        grip_right=grip_right,
        scene_points_seq=scene_points_seq,
    )


def _primitive_two_endpoint_tension(task_name: str, variation: int, cfg, rng: np.random.Generator) -> PrimitiveTrajectory:
    del task_name, variation
    steps, s_a, s_b, s_c = _make_common_steps(cfg.min_steps, cfg.max_steps, rng)

    left_end0 = np.array([-0.08, 0.02, cfg.table_height + 0.01], dtype=np.float32)
    right_end0 = np.array([0.08, -0.02, cfg.table_height + 0.01], dtype=np.float32)

    left_track = _interp_pose_track(
        steps,
        [
            (0, _pose(*cfg.left_home, yaw=np.deg2rad(180))),
            (s_a, _pose(float(left_end0[0]), float(left_end0[1]), float(left_end0[2] + 0.07), yaw=np.deg2rad(180))),
            (s_b, _pose(float(left_end0[0]), float(left_end0[1]), float(left_end0[2] + 0.01), yaw=np.deg2rad(180))),
            (s_c, _pose(-0.22, 0.04, cfg.table_height + 0.10, yaw=np.deg2rad(180))),
            (steps - 1, _pose(-0.24, 0.05, cfg.table_height + 0.10, yaw=np.deg2rad(180))),
        ],
    )
    right_track = _interp_pose_track(
        steps,
        [
            (0, _pose(*cfg.right_home, yaw=0.0)),
            (s_a, _pose(float(right_end0[0]), float(right_end0[1]), float(right_end0[2] + 0.07), yaw=0.0)),
            (s_b, _pose(float(right_end0[0]), float(right_end0[1]), float(right_end0[2] + 0.01), yaw=0.0)),
            (s_c, _pose(0.22, -0.04, cfg.table_height + 0.10, yaw=0.0)),
            (steps - 1, _pose(0.24, -0.05, cfg.table_height + 0.10, yaw=0.0)),
        ],
    )

    grip_left = _piecewise_grip(steps, [(0, OPEN), (s_b, CLOSED)])
    grip_right = _piecewise_grip(steps, [(0, OPEN), (s_b, CLOSED)])

    table = _sample_table(cfg.workspace_bounds, cfg.table_height, n=1600, rng=rng)

    rope_n = 520
    u = np.linspace(0.0, 1.0, rope_n, dtype=np.float32)
    scene_points_seq = []
    for t in range(steps):
        if t < s_b:
            left_end = left_end0
            right_end = right_end0
            sag = 0.065
        else:
            left_end = left_track[t][:3, 3] + np.array([0.0, 0.0, -0.015], dtype=np.float32)
            right_end = right_track[t][:3, 3] + np.array([0.0, 0.0, -0.015], dtype=np.float32)
            alpha = (t - s_b) / float(max(1, steps - 1 - s_b))
            sag = (1.0 - alpha) * 0.060 + alpha * 0.006

        line = (1.0 - u[:, None]) * left_end[None, :] + u[:, None] * right_end[None, :]
        line[:, 2] -= (4.0 * u * (1.0 - u) * sag)
        line += rng.normal(scale=0.0015, size=line.shape).astype(np.float32)

        pts = np.concatenate([table, line.astype(np.float32)], axis=0)
        scene_points_seq.append(_subsample_points(pts, cfg.num_points, rng))

    return PrimitiveTrajectory(
        task_name="bimanual_straighten_rope",
        variation_index=0,
        left_seq=left_track,
        right_seq=right_track,
        grip_left=grip_left,
        grip_right=grip_right,
        scene_points_seq=scene_points_seq,
    )


def _primitive_tool_plus_receptacle(task_name: str, variation: int, cfg, rng: np.random.Generator) -> PrimitiveTrajectory:
    del task_name, variation
    steps, s_a, s_b, s_c = _make_common_steps(cfg.min_steps, cfg.max_steps, rng)

    dustpan_start = np.array([-0.10, 0.04, cfg.table_height + 0.02], dtype=np.float32)
    broom_start = np.array([0.14, -0.04, cfg.table_height + 0.02], dtype=np.float32)
    sweep_start = np.array([0.10, -0.02, cfg.table_height + 0.03], dtype=np.float32)
    sweep_end = np.array([-0.02, 0.05, cfg.table_height + 0.03], dtype=np.float32)

    left_track = _interp_pose_track(
        steps,
        [
            (0, _pose(*cfg.left_home, yaw=np.deg2rad(180))),
            (s_a, _pose(float(dustpan_start[0]), float(dustpan_start[1]), float(dustpan_start[2] + 0.08), yaw=np.deg2rad(180))),
            (s_b, _pose(float(dustpan_start[0]), float(dustpan_start[1]), float(dustpan_start[2] + 0.01), yaw=np.deg2rad(180))),
            (s_c, _pose(float(sweep_end[0] - 0.02), float(sweep_end[1] + 0.02), float(dustpan_start[2] + 0.02), yaw=np.deg2rad(180))),
            (steps - 1, _pose(float(sweep_end[0] - 0.02), float(sweep_end[1] + 0.02), float(dustpan_start[2] + 0.03), yaw=np.deg2rad(180))),
        ],
    )

    right_track = _interp_pose_track(
        steps,
        [
            (0, _pose(*cfg.right_home, yaw=0.0)),
            (s_a, _pose(float(broom_start[0]), float(broom_start[1]), float(broom_start[2] + 0.08), yaw=0.0)),
            (s_b, _pose(float(broom_start[0]), float(broom_start[1]), float(broom_start[2] + 0.01), yaw=0.0)),
            (s_c, _pose(float(sweep_start[0]), float(sweep_start[1]), float(sweep_start[2]), yaw=0.0)),
            (steps - 1, _pose(float(sweep_end[0]), float(sweep_end[1]), float(sweep_end[2]), yaw=0.0)),
        ],
    )

    grip_left = _piecewise_grip(steps, [(0, OPEN), (s_b, CLOSED)])
    grip_right = _piecewise_grip(steps, [(0, OPEN), (s_b, CLOSED)])

    table = _sample_table(cfg.workspace_bounds, cfg.table_height, n=1400, rng=rng)

    broom_local = _sample_box_surface((0.28, 0.03, 0.02), n=550, rng=rng)
    dustpan_local = _sample_box_surface((0.12, 0.08, 0.02), n=320, rng=rng)

    broom_seq = []
    dustpan_seq = []
    for t in range(steps):
        if grip_right[t] <= 0.5:
            pb = right_track[t][:3, 3] + np.array([0.0, 0.0, -0.02], dtype=np.float32)
            broom_seq.append(_pose(float(pb[0]), float(pb[1]), float(pb[2]), yaw=0.0))
        else:
            broom_seq.append(_pose(float(broom_start[0]), float(broom_start[1]), float(broom_start[2]), yaw=0.0))

        if grip_left[t] <= 0.5:
            pd = left_track[t][:3, 3] + np.array([0.0, 0.0, -0.02], dtype=np.float32)
            dustpan_seq.append(_pose(float(pd[0]), float(pd[1]), float(pd[2]), yaw=0.0))
        else:
            dustpan_seq.append(_pose(float(dustpan_start[0]), float(dustpan_start[1]), float(dustpan_start[2]), yaw=0.0))

    broom_points_seq = _attach_points(broom_local, np.stack(broom_seq, axis=0))
    dustpan_points_seq = _attach_points(dustpan_local, np.stack(dustpan_seq, axis=0))

    # Dirt cluster translated by sweep progress.
    dirt_seed = rng.normal(size=(260, 3)).astype(np.float32)
    dirt_seed[:, :2] *= 0.018
    dirt_seed[:, 2] = np.abs(dirt_seed[:, 2]) * 0.001
    dirt_center0 = np.array([0.05, 0.0, cfg.table_height + 0.002], dtype=np.float32)
    dirt_center1 = np.array([sweep_end[0] - 0.02, sweep_end[1] + 0.01, cfg.table_height + 0.002], dtype=np.float32)

    scene_points_seq = []
    for t in range(steps):
        if t <= s_c:
            alpha = 0.0
        else:
            alpha = (t - s_c) / float(max(1, steps - 1 - s_c))
        c = (1.0 - alpha) * dirt_center0 + alpha * dirt_center1
        dirt = dirt_seed + c[None, :]

        pts = np.concatenate(
            [table, broom_points_seq[t], dustpan_points_seq[t], dirt.astype(np.float32)],
            axis=0,
        )
        scene_points_seq.append(_subsample_points(pts, cfg.num_points, rng))

    return PrimitiveTrajectory(
        task_name="bimanual_sweep_to_dustpan",
        variation_index=0,
        left_seq=left_track,
        right_seq=right_track,
        grip_left=grip_left,
        grip_right=grip_right,
        scene_points_seq=scene_points_seq,
    )


TASK_TO_PRIMITIVE: Dict[str, Callable[[str, int, object, np.random.Generator], PrimitiveTrajectory]] = {
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


def sample_trajectory(task_name: str, cfg, rng: np.random.Generator) -> PrimitiveTrajectory:
    if task_name not in TASK_TO_PRIMITIVE:
        raise ValueError(f"Unsupported bimanual task: {task_name}")
    v_count = variation_count(task_name)
    v_idx = int(rng.integers(0, v_count))
    traj = TASK_TO_PRIMITIVE[task_name](task_name, v_idx, cfg, rng)
    if traj.left_seq.shape[0] <= cfg.pred_horizon:
        raise RuntimeError(
            f"Generated trajectory for {task_name} is too short: {traj.left_seq.shape[0]} <= pred_horizon={cfg.pred_horizon}"
        )
    return traj
