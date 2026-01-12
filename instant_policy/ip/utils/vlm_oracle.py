import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .common_utils import pose_to_transform


def _matrix_from_object(obj):
    if hasattr(obj, 'get_matrix'):
        mat = np.array(obj.get_matrix(), dtype=float).reshape(-1)
        if mat.size == 12:
            mat = mat.reshape(3, 4)
            mat = np.vstack([mat, [0.0, 0.0, 0.0, 1.0]])
        elif mat.size == 16:
            mat = mat.reshape(4, 4)
        else:
            raise ValueError(f"Unexpected matrix size from object: {mat.size}")
        return mat

    if hasattr(obj, 'get_pose'):
        pose = np.array(obj.get_pose(), dtype=float).reshape(-1)
        if pose.size != 7:
            raise ValueError(f"Unexpected pose size from object: {pose.size}")
        return pose_to_transform(pose)

    if hasattr(obj, 'get_position') and hasattr(obj, 'get_orientation'):
        pos = np.array(obj.get_position(), dtype=float).reshape(3)
        euler = np.array(obj.get_orientation(), dtype=float).reshape(3)
        quat = Rot.from_euler('xyz', euler).as_quat()
        return pose_to_transform(np.concatenate([pos, quat], axis=0))

    raise ValueError("Waypoint object does not expose a usable pose.")


def _parse_gripper_state(ext, current_state):
    if not ext:
        return current_state

    for key in ('open_gripper', 'close_gripper'):
        token = f"{key}("
        if token not in ext:
            continue
        start = ext.index(token) + len(token)
        end = ext.find(')', start)
        param = ext[start:end].strip() if end != -1 else ''
        if param:
            try:
                return float(param)
            except ValueError:
                pass
        return 1.0 if key == 'open_gripper' else 0.0

    return current_state


def vlm_outputs_from_waypoints(task_or_env, current_obs=None):
    """
    Build a VLM-style output dict from RLBench task waypoints.

    This uses the task's built-in waypoints (dummies/paths) and parses
    waypoint extension strings for gripper commands.
    """
    task = task_or_env._task if hasattr(task_or_env, '_task') else task_or_env
    waypoints = task.get_waypoints()
    if not waypoints:
        raise ValueError("No waypoints found for task.")

    gripper_state = float(current_obs.gripper_open) if current_obs is not None else 1.0
    poses = []
    grips = []

    for waypoint in waypoints:
        obj = waypoint.get_waypoint_object()
        pose = _matrix_from_object(obj)
        gripper_state = _parse_gripper_state(waypoint.get_ext(), gripper_state)
        poses.append(pose)
        grips.append(1 if gripper_state >= 0.5 else 0)

    return {
        'poses_4x4': poses,
        'gripper_states': grips,
        'trajectory_3d': [pose[:3, 3] for pose in poses],
    }
