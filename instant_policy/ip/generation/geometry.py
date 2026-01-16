import numpy as np
from scipy.spatial.transform import Rotation as Rot


def make_transform(R, t):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def look_at(eye, target, up=None):
    if up is None:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    forward = target - eye
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    else:
        forward = forward / forward_norm
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        right = right / right_norm
    up = np.cross(right, forward)
    R = np.stack([right, up, -forward], axis=1)
    return make_transform(R, eye)


def pose_from_approach(position, approach, rng=None):
    position = np.asarray(position, dtype=np.float32)
    z_axis = np.asarray(approach, dtype=np.float32)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-6:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        z_axis = z_axis / z_norm
    if abs(z_axis[2]) < 0.9:
        x_axis = np.cross(z_axis, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    else:
        x_axis = np.cross(z_axis, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
    if rng is None:
        angle = np.random.uniform(0.0, 2.0 * np.pi)
    else:
        angle = rng.uniform(0.0, 2.0 * np.pi)
    Rz = Rot.from_rotvec(angle * z_axis).as_matrix()
    x_axis = Rz @ x_axis
    y_axis = np.cross(z_axis, x_axis)
    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    return make_transform(R, position)


def random_rotation(rng=None):
    if rng is None:
        return Rot.random().as_matrix().astype(np.float32)
    return Rot.random(random_state=rng).as_matrix().astype(np.float32)


def transform_points(points, T):
    return (T[:3, :3] @ points.T).T + T[:3, 3]


def copy_pose(T):
    return np.array(T, dtype=np.float32, copy=True)
