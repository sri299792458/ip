"""
VLM Keypoint to Demonstration Conversion

This module converts VLM-generated 2D keypoints (start + goal) into synthetic
demonstrations compatible with Instant Policy's 10-waypoint format.

Phase 1 Implementation: Trajectory Interpolation
- Linear interpolation between start and goal keypoints
- Heuristic gripper states
- Point cloud extraction from current observation
"""

import numpy as np
from .common_utils import transform_pcd
from .data_proc import subsample_pcd


def pixel_to_3d(pixel_coords, depth_image, camera_intrinsics):
    """
    Lift 2D pixel coordinates to 3D using depth information.

    Args:
        pixel_coords: (u, v) pixel coordinates in image
        depth_image: depth map (H x W)
        camera_intrinsics: dict with 'fx', 'fy', 'cx', 'cy'

    Returns:
        point_3d: (x, y, z) in camera frame

    Raises:
        ValueError: If pixel coordinates are out of bounds or depth is invalid
    """
    u, v = pixel_coords
    h, w = depth_image.shape

    # Validate pixel coordinates
    if not (0 <= u < w and 0 <= v < h):
        raise ValueError(f"Pixel coordinates ({u}, {v}) out of bounds for image size ({w}, {h})")

    depth = depth_image[int(v), int(u)]

    # Validate depth value
    if np.isnan(depth) or np.isinf(depth) or depth <= 0.0:
        raise ValueError(f"Invalid depth value {depth} at pixel ({u}, {v})")

    # Reasonable depth bounds (0.1m to 10m)
    if depth < 0.1 or depth > 10.0:
        raise ValueError(f"Depth value {depth} out of reasonable range [0.1, 10.0] at pixel ({u}, {v})")

    # Camera intrinsics
    fx = camera_intrinsics['fx']
    fy = camera_intrinsics['fy']
    cx = camera_intrinsics['cx']
    cy = camera_intrinsics['cy']

    # Back-project to 3D
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    return np.array([x, y, z])


def interpolate_trajectory(start_3d, goal_3d, num_waypoints=10, method='linear'):
    """
    Generate waypoints by interpolating between start and goal.

    Args:
        start_3d: (x, y, z) start position in world frame
        goal_3d: (x, y, z) goal position in world frame
        num_waypoints: number of waypoints to generate (default 10)
        method: 'linear' or 'minimum_jerk'

    Returns:
        waypoints: (num_waypoints, 3) array of 3D positions
    """
    if method == 'linear':
        # Simple linear interpolation
        alphas = np.linspace(0, 1, num_waypoints)
        waypoints = start_3d[None, :] * (1 - alphas[:, None]) + goal_3d[None, :] * alphas[:, None]
        return waypoints

    elif method == 'minimum_jerk':
        # Minimum jerk trajectory (smooth acceleration profile)
        t = np.linspace(0, 1, num_waypoints)
        # 5th order polynomial for minimum jerk
        s = 10 * t**3 - 15 * t**4 + 6 * t**5
        waypoints = start_3d[None, :] * (1 - s[:, None]) + goal_3d[None, :] * s[:, None]
        return waypoints

    else:
        raise ValueError(f"Unknown interpolation method: {method}")


def compute_approach_orientation(current_pos, goal_pos, method='point_down'):
    """
    Compute end-effector orientation for a waypoint.

    Args:
        current_pos: (x, y, z) current position
        goal_pos: (x, y, z) goal position
        method: 'point_down' or 'approach_vector'

    Returns:
        rotation_matrix: (3, 3) rotation matrix for gripper orientation
    """
    if method == 'point_down':
        # Simple: gripper always points down (common for top-down grasps)
        # Z-axis points down, X-axis points forward
        z_axis = np.array([0, 0, -1])
        x_axis = np.array([1, 0, 0])
        y_axis = np.cross(z_axis, x_axis)
        R = np.column_stack([x_axis, y_axis, z_axis])
        return R

    elif method == 'approach_vector':
        # Orient gripper toward goal
        direction = goal_pos - current_pos
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-6:
            # At goal, use point_down
            return compute_approach_orientation(current_pos, goal_pos, method='point_down')

        direction = direction / direction_norm

        # Z-axis along approach direction
        z_axis = direction
        # X-axis perpendicular to Z (choose arbitrary perpendicular)
        if abs(z_axis[2]) < 0.9:
            x_axis = np.cross(z_axis, np.array([0, 0, 1]))
        else:
            x_axis = np.cross(z_axis, np.array([0, 1, 0]))
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        R = np.column_stack([x_axis, y_axis, z_axis])
        return R

    else:
        raise ValueError(f"Unknown orientation method: {method}")


def construct_ee_pose(position, orientation_matrix):
    """
    Construct SE(3) transformation matrix from position and orientation.

    Args:
        position: (3,) translation vector
        orientation_matrix: (3, 3) rotation matrix

    Returns:
        T: (4, 4) transformation matrix
    """
    T = np.eye(4)
    T[:3, 3] = position
    T[:3, :3] = orientation_matrix
    return T


def extract_local_pointcloud(full_pcd, center, radius=0.3, num_points=2048):
    """
    Extract local point cloud around a center point.

    Args:
        full_pcd: (N, 3) full point cloud in world frame
        center: (3,) center position
        radius: radius for local extraction
        num_points: target number of points

    Returns:
        local_pcd: (num_points, 3) subsampled local point cloud

    Raises:
        ValueError: If full_pcd is empty
    """
    # Guard against empty input
    if len(full_pcd) == 0:
        raise ValueError("Cannot extract local point cloud from empty full_pcd")

    # Compute distances from center
    distances = np.linalg.norm(full_pcd - center[None, :], axis=1)

    # Filter points within radius
    mask = distances < radius
    local_pcd = full_pcd[mask]

    # If too few points or empty mask, use full point cloud
    if len(local_pcd) < num_points // 2:
        local_pcd = full_pcd

    # Guard against empty local_pcd before subsampling
    if len(local_pcd) == 0:
        raise ValueError(f"No points found within radius {radius} of center {center}, and full_pcd is empty")

    # Use subsample_pcd from data_proc.py to match training data distribution
    # This includes outlier removal and proper subsampling
    local_pcd_subsampled = subsample_pcd(local_pcd, num_points=num_points)

    return local_pcd_subsampled


def vlm_keypoints_to_demo(
    start_keypoint_2d,
    goal_keypoint_2d,
    depth_image,
    camera_intrinsics,
    current_pcd_world,
    camera_to_world_transform,
    num_waypoints=10,
    interpolation_method='linear',
    orientation_method='point_down'
):
    """
    Convert VLM keypoints to synthetic demonstration.

    Args:
        start_keypoint_2d: (u, v) pixel coordinates for start
        goal_keypoint_2d: (u, v) pixel coordinates for goal
        depth_image: (H, W) depth map
        camera_intrinsics: dict with camera parameters
        current_pcd_world: (N, 3) point cloud in world frame
        camera_to_world_transform: (4, 4) transform from camera to world
        num_waypoints: number of waypoints (default 10)
        interpolation_method: trajectory interpolation method
        orientation_method: gripper orientation method

    Returns:
        synthetic_demo: dict with keys 'obs', 'grips', 'T_w_es'
            - obs: list of (2048, 3) point clouds in EE frame
            - grips: list of gripper states (10,) in {0, 1} (0=closed, 1=open)
            - T_w_es: list of (4, 4) SE(3) matrices
    """
    # Step 1: Lift 2D keypoints to 3D in camera frame
    start_3d_cam = pixel_to_3d(start_keypoint_2d, depth_image, camera_intrinsics)
    goal_3d_cam = pixel_to_3d(goal_keypoint_2d, depth_image, camera_intrinsics)

    # Transform to world frame
    start_3d_world = transform_pcd(start_3d_cam[None, :], camera_to_world_transform)[0]
    goal_3d_world = transform_pcd(goal_3d_cam[None, :], camera_to_world_transform)[0]

    # Step 2: Generate trajectory waypoints
    waypoints_3d_world = interpolate_trajectory(
        start_3d_world,
        goal_3d_world,
        num_waypoints=num_waypoints,
        method=interpolation_method
    )

    # Step 3: Construct synthetic demo
    synthetic_demo = {'obs': [], 'grips': [], 'T_w_es': []}

    for i, waypoint_pos in enumerate(waypoints_3d_world):
        # 3a. Compute gripper orientation
        next_pos = goal_3d_world if i == num_waypoints - 1 else waypoints_3d_world[i + 1]
        orientation = compute_approach_orientation(waypoint_pos, next_pos, method=orientation_method)

        # 3b. Construct SE(3) pose
        T_w_e = construct_ee_pose(waypoint_pos, orientation)
        synthetic_demo['T_w_es'].append(T_w_e)

        # 3c. Extract local point cloud and transform to EE frame
        local_pcd_world = extract_local_pointcloud(
            current_pcd_world,
            center=waypoint_pos,
            radius=0.3,
            num_points=2048
        )
        # Transform to end-effector frame
        T_e_w = np.linalg.inv(T_w_e)
        pcd_ee_frame = transform_pcd(local_pcd_world, T_e_w)
        synthetic_demo['obs'].append(pcd_ee_frame)

        # 3d. Gripper state: heuristic
        # Open for first half, closed for second half
        # Use {0, 1} format (not {-1, 1}) - normalization happens later in pipeline
        if i < num_waypoints // 2:
            grip_state = 1  # Open
        else:
            grip_state = 0  # Closed (or ready to grasp/push)
        synthetic_demo['grips'].append(grip_state)

    return synthetic_demo


def vlm_keypoints_to_demo_simple(
    start_3d_world,
    goal_3d_world,
    current_pcd_world,
    num_waypoints=10
):
    """
    Simplified version that takes 3D positions directly (for testing).

    Args:
        start_3d_world: (3,) start position in world frame
        goal_3d_world: (3,) goal position in world frame
        current_pcd_world: (N, 3) point cloud in world frame
        num_waypoints: number of waypoints

    Returns:
        synthetic_demo: dict with 'obs', 'grips', 'T_w_es'
            - obs: list of (2048, 3) point clouds in EE frame
            - grips: list of gripper states in {0, 1} (0=closed, 1=open)
            - T_w_es: list of (4, 4) SE(3) matrices
    """
    # Generate trajectory
    waypoints_3d_world = interpolate_trajectory(
        start_3d_world,
        goal_3d_world,
        num_waypoints=num_waypoints,
        method='linear'
    )

    synthetic_demo = {'obs': [], 'grips': [], 'T_w_es': []}

    for i, waypoint_pos in enumerate(waypoints_3d_world):
        # Simple point-down orientation
        orientation = compute_approach_orientation(
            waypoint_pos,
            goal_3d_world,
            method='point_down'
        )

        # Construct pose
        T_w_e = construct_ee_pose(waypoint_pos, orientation)
        synthetic_demo['T_w_es'].append(T_w_e)

        # Extract and transform point cloud
        local_pcd_world = extract_local_pointcloud(
            current_pcd_world,
            center=waypoint_pos,
            radius=0.3,
            num_points=2048
        )
        T_e_w = np.linalg.inv(T_w_e)
        pcd_ee_frame = transform_pcd(local_pcd_world, T_e_w)
        synthetic_demo['obs'].append(pcd_ee_frame)

        # Gripper state heuristic
        # Use {0, 1} format (not {-1, 1}) - normalization happens later in pipeline
        grip_state = 1 if i < num_waypoints // 2 else 0
        synthetic_demo['grips'].append(grip_state)

    return synthetic_demo


def _resample_indices(num_src, num_dst):
    if num_src <= 0:
        raise ValueError("num_src must be > 0")
    if num_dst <= 0:
        raise ValueError("num_dst must be > 0")
    if num_src == num_dst:
        return np.arange(num_src, dtype=int)
    return np.round(np.linspace(0, num_src - 1, num_dst)).astype(int)


def _resample_positions(points, num_waypoints):
    points = np.asarray(points, dtype=float)
    if len(points.shape) != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N, 3)")
    if points.shape[0] == num_waypoints:
        return points
    t_src = np.linspace(0.0, 1.0, points.shape[0])
    t_dst = np.linspace(0.0, 1.0, num_waypoints)
    out = np.zeros((num_waypoints, 3), dtype=float)
    for d in range(3):
        out[:, d] = np.interp(t_dst, t_src, points[:, d])
    return out


def _resample_binary(states, num_waypoints):
    states = np.asarray(states, dtype=int).reshape(-1)
    idx = _resample_indices(len(states), num_waypoints)
    return states[idx].tolist()


def synthesize_demo_from_vlm(
    vlm_outputs,
    current_pcd_world=None,
    num_waypoints=10,
    interpolation_method='linear',
    orientation_method='point_down'
):
    """
    Synthesize a demo from rich VLM outputs.

    Expected vlm_outputs fields (all optional except one of poses_4x4/keypoints_3d/trajectory_3d):
        - poses_4x4: list of 4x4 end-effector poses (preferred if available)
        - keypoints_3d: list of 3D points (>=2)
        - trajectory_3d: list of 3D points (arbitrary length)
        - gripper_states: list of {0,1} states
        - orientations: list of 3x3 rotation matrices
        - pcd_sequence_world: list of (N,3) point clouds (same length as trajectory_3d)

    Returns:
        synthetic_demo: dict with 'obs', 'grips', 'T_w_es'
    """
    if not isinstance(vlm_outputs, dict):
        raise ValueError("vlm_outputs must be a dict")

    poses_4x4 = vlm_outputs.get('poses_4x4') or vlm_outputs.get('T_w_es')
    trajectory_3d = vlm_outputs.get('trajectory_3d')
    keypoints_3d = vlm_outputs.get('keypoints_3d')
    idx = None

    if poses_4x4 is not None:
        poses = np.asarray(poses_4x4, dtype=float)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError("poses_4x4 must be a list of 4x4 matrices")
        idx = _resample_indices(len(poses), num_waypoints)
        poses = poses[idx]
        traj_points = poses[:, :3, 3]
        orientations = poses[:, :3, :3]
        src_len = len(poses_4x4)
    elif trajectory_3d is not None and len(trajectory_3d) > 2:
        traj_points_all = np.asarray(trajectory_3d, dtype=float)
        if traj_points_all.ndim != 2 or traj_points_all.shape[1] != 3:
            raise ValueError("trajectory_3d must be (N, 3)")
        idx = _resample_indices(traj_points_all.shape[0], num_waypoints)
        traj_points = traj_points_all[idx]
        orientations = None
        src_len = len(trajectory_3d)
    elif trajectory_3d is not None:
        traj_points = _resample_positions(trajectory_3d, num_waypoints)
        orientations = None
        src_len = len(trajectory_3d)
    elif keypoints_3d is not None and len(keypoints_3d) >= 2:
        kp = np.asarray(keypoints_3d, dtype=float)
        traj_points = interpolate_trajectory(kp[0], kp[-1],
                                             num_waypoints=num_waypoints,
                                             method=interpolation_method)
        orientations = None
        src_len = len(keypoints_3d)
    else:
        raise ValueError("vlm_outputs must include poses_4x4, trajectory_3d, or keypoints_3d with >=2 points")

    # Resample gripper states if provided; otherwise use a simple heuristic.
    if 'gripper_states' in vlm_outputs:
        if idx is not None:
            grips = np.asarray(vlm_outputs['gripper_states'], dtype=int)[idx].tolist()
        else:
            grips = _resample_binary(vlm_outputs['gripper_states'], num_waypoints)
    else:
        grips = [1 if i < num_waypoints // 2 else 0 for i in range(num_waypoints)]

    if 'orientations' in vlm_outputs:
        ori_seq = np.asarray(vlm_outputs['orientations'], dtype=float)
        if ori_seq.shape[0] > 0 and orientations is None:
            if idx is None:
                idx = _resample_indices(ori_seq.shape[0], num_waypoints)
            orientations = ori_seq[idx]

    # Optional point cloud sequence (nearest neighbor).
    pcd_sequence = None
    if 'pcd_sequence_world' in vlm_outputs:
        pcd_seq = vlm_outputs['pcd_sequence_world']
        if idx is None:
            idx = _resample_indices(len(pcd_seq), num_waypoints)
        pcd_sequence = [subsample_pcd(pcd_seq[i], num_points=2048) for i in idx]

    synthetic_demo = {'obs': [], 'grips': grips, 'T_w_es': []}

    for i, waypoint_pos in enumerate(traj_points):
        if orientations is not None:
            R = orientations[i]
        else:
            next_pos = traj_points[-1] if i == num_waypoints - 1 else traj_points[i + 1]
            R = compute_approach_orientation(waypoint_pos, next_pos, method=orientation_method)

        T_w_e = construct_ee_pose(waypoint_pos, R)
        synthetic_demo['T_w_es'].append(T_w_e)

        if pcd_sequence is not None:
            local_pcd_world = pcd_sequence[i]
        else:
            if current_pcd_world is None:
                raise ValueError("current_pcd_world is required when pcd_sequence_world is not provided")
            local_pcd_world = extract_local_pointcloud(
                current_pcd_world,
                center=waypoint_pos,
                radius=0.3,
                num_points=2048
            )

        T_e_w = np.linalg.inv(T_w_e)
        pcd_ee_frame = transform_pcd(local_pcd_world, T_e_w)
        synthetic_demo['obs'].append(pcd_ee_frame)

    return synthetic_demo
