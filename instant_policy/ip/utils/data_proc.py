import os
import tempfile

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot
from ip.utils.common_utils import transform_pcd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch


def _atomic_torch_save(obj, path: str):
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_data_", suffix=".pt", dir=parent)
    os.close(fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def save_sample(sample, save_dir=None, offset=0, scene_encoder=None):
    # First, subsample demo pcds and concatenate them.
    joint_demo_pcd = []
    joint_demo_grasp = []
    batch_indices = []

    num_demos = len(sample['demos'])
    num_traj_waypoints = len(sample['demos'][0]['obs'])
    num_points = len(sample['live']['obs'][0])
    for n in range(len(sample['demos'])):
        for i in range(len(sample['demos'][n]['obs'])):
            joint_demo_pcd.append(sample['demos'][n]['obs'][i])
            joint_demo_grasp.append(sample['demos'][n]['grips'][i])
            batch_indices.append(np.zeros(len(sample['demos'][n]['obs'][i])) + i + n * len(sample['demos'][n]['obs']))

    joint_demo_pcd = np.concatenate(joint_demo_pcd)
    joint_demo_grasp = (np.array(joint_demo_grasp) - 0.5) * 2
    batch_indices = np.concatenate(batch_indices)

    data = Data(
        pos_demos=torch.tensor(joint_demo_pcd, dtype=torch.float32),
        graps_demos=torch.tensor(joint_demo_grasp, dtype=torch.float32).view(num_demos,
                                                                             num_traj_waypoints,
                                                                             1).unsqueeze(0),
        batch_demos=torch.tensor(batch_indices, dtype=torch.int64),
        batch_pos_obs=torch.tensor(np.zeros(num_points), dtype=torch.int64),
        demo_T_w_es=torch.tensor(np.stack([sample['demos'][n]['T_w_es'] for n in range(num_demos)]),
                                 dtype=torch.float32).unsqueeze(0),
    )

    if scene_encoder is not None:
        bs = 1
        demo_scene_node_embds, demo_scene_node_pos, demo_scene_node_batch = \
            scene_encoder(None,
                          data.pos_demos.to(next(scene_encoder.parameters()).device),
                          data.batch_demos.to(next(scene_encoder.parameters()).device))

        demo_scene_node_embds = to_dense_batch(demo_scene_node_embds, demo_scene_node_batch, fill_value=0)[0]
        demo_scene_node_embds = demo_scene_node_embds.view(bs, num_demos, num_traj_waypoints, -1,
                                                           scene_encoder.embd_dim)
        demo_scene_node_pos = to_dense_batch(demo_scene_node_pos, demo_scene_node_batch, fill_value=0)[0]
        demo_scene_node_pos = demo_scene_node_pos.view(bs, num_demos, num_traj_waypoints, -1, 3)
        data.demo_scene_node_embds = demo_scene_node_embds.detach().cpu()
        data.demo_scene_node_pos = demo_scene_node_pos.detach().cpu()

        data.pos_demos = None
        data.batch_demos = None
        # data.batch_pos_obs = None

    k = 0
    for i in range(len(sample['live']['obs'])):
        data.pos_obs = torch.tensor(sample['live']['obs'][i], dtype=torch.float32)
        if scene_encoder is not None:
            live_scene_node_embds, live_scene_node_pos, live_scene_node_batch = \
                scene_encoder(None,
                              data.pos_obs.to(next(scene_encoder.parameters()).device),
                              torch.tensor(np.zeros(num_points),
                                           dtype=torch.int64).to(next(scene_encoder.parameters()).device))
            data.live_scene_node_embds = live_scene_node_embds.detach().cpu().unsqueeze(0)
            data.live_scene_node_pos = live_scene_node_pos.detach().cpu().unsqueeze(0)
            # data.pos_obs = None

        data.current_grip = torch.tensor(sample['live']['grips'][i], dtype=torch.float32).unsqueeze(0)
        data.current_grip = (data.current_grip - 0.5) * 2
        data.actions = torch.tensor(sample['live']['actions'][i], dtype=torch.float32).unsqueeze(0)
        data.actions_grip = torch.tensor(sample['live']['actions_grip'][i], dtype=torch.float32).unsqueeze(0)
        data.actions_grip = (data.actions_grip - 0.5) * 2
        data.T_w_e = torch.tensor(sample['live']['T_w_es'][i], dtype=torch.float32).unsqueeze(0)
        if save_dir is not None:
            _atomic_torch_save(data, f'{save_dir}/data_{k + offset}.pt')
            k += 1
        else:
            return data


def sample_to_cond_demo(sample, num_waypoints, num_points=2048):
    traj_indices = extract_waypoints(
        np.array(sample['T_w_es']),
        np.array(sample['grips']),
        num_waypoints=num_waypoints,
    )

    pcds = [transform_pcd(subsample_pcd(sample['pcds'][idx], num_points),
                          np.linalg.inv(sample['T_w_es'][idx])) for idx in traj_indices]

    demo_sample = {'obs': pcds,
                   'grips': [sample['grips'][idx] for idx in traj_indices],
                   'T_w_es': [sample['T_w_es'][idx] for idx in traj_indices]}

    return demo_sample


def sample_to_live(sample, pred_horizon, num_points=2048, trans_space=0.01, rot_space=5, subsample=True):
    if subsample:
        sample['T_w_es'], sample['grips'], sample['pcds'] = \
            subsample_traj(sample['T_w_es'], sample['grips'], pcds=sample['pcds'], trans_space=trans_space,
                           rot_space=rot_space)

    live_data = {'obs': [], 'actions': [], 'actions_grip': []}
    live_data['T_w_es'] = sample['T_w_es']
    live_data['obs'] = [transform_pcd(subsample_pcd(sample['pcds'][idx], num_points),
                                      np.linalg.inv(sample['T_w_es'][idx])) for idx in range(len(sample['pcds']))]
    live_data['grips'] = sample['grips']
    for i in range(len(sample['pcds'])):
        actions = []
        actions_grip = []
        for j in range(1, pred_horizon + 1):
            if i + j < len(sample['pcds']):
                actions.append(np.linalg.inv(sample['T_w_es'][i]) @ sample['T_w_es'][i + j])
                actions_grip.append(sample['grips'][i + j])
            else:
                actions.append(np.eye(4))
                actions_grip.append(sample['grips'][-1])
        live_data['actions'].append(np.array(actions))
        live_data['actions_grip'].append(actions_grip)
    return live_data


def subsample_traj(traj, grips, pcds=None, trans_space=0.01, rot_space=3):
    subsampled_traj = [traj[0]]
    subsampled_grips = [grips[0]]

    if pcds is not None:
        subsampled_pcds = [pcds[0]]

    i = 1
    while i < len(traj):
        trans_dist = np.linalg.norm(traj[i][:3, 3] - subsampled_traj[-1][:3, 3])
        rot_dist = np.linalg.norm(
            Rot.from_matrix(traj[i][:3, :3] @ subsampled_traj[-1][:3, :3].T).as_rotvec(degrees=True))
        if trans_dist < trans_space and rot_dist < rot_space and grips[i] == subsampled_grips[-1]:
            i += 1  # Skip if the change between timesteps is too small.
        elif (trans_dist > trans_space or rot_dist > rot_space):
            # If the change between timesteps is too big, create an intermediate waypoint and adjust observation accordingly.
            # This can introduce some gitter in the point cloud of the grasped object between timesteps, but it helps a lot.
            # If demonstrations are recorded fast enough, this condition can be skipped entirely.
            new_wp = subsampled_traj[-1].copy()
            err = traj[i][:3, 3] - subsampled_traj[-1][:3, 3]
            rot_vec_delta = Rot.from_matrix(subsampled_traj[-1][:3, :3].T @ traj[i][:3, :3]).as_rotvec()
            if trans_dist > trans_space and rot_dist > rot_space:
                if trans_dist / trans_space > rot_dist / rot_space:
                    new_wp[:3, 3] += trans_space * err / np.linalg.norm(err)
                    rot_vec_delta = rot_vec_delta * trans_space / trans_dist
                else:
                    new_wp[:3, 3] += err * rot_space / rot_dist
                    rot_vec_delta = np.deg2rad(rot_space) * rot_vec_delta / np.linalg.norm(rot_vec_delta)
            elif trans_dist > trans_space:
                new_wp[:3, 3] += trans_space * err / np.linalg.norm(err)
                rot_vec_delta = rot_vec_delta * trans_space / trans_dist
            else:
                new_wp[:3, 3] += err * rot_space / rot_dist
                rot_vec_delta = np.deg2rad(rot_space) * rot_vec_delta / np.linalg.norm(rot_vec_delta)
            new_wp[:3, :3] = new_wp[:3, :3] @ Rot.from_rotvec(rot_vec_delta).as_matrix()

            subsampled_traj.append(new_wp)
            subsampled_grips.append(subsampled_grips[-1])
            if pcds is not None:
                subsampled_pcds.append(pcds[i])
        else:
            subsampled_traj.append(traj[i])
            subsampled_grips.append(grips[i])
            if pcds is not None:
                subsampled_pcds.append(pcds[i])
            i += 1

    # Last one is always added.
    subsampled_grips.append(grips[-1])
    subsampled_traj.append(traj[-1])

    if pcds is not None:
        subsampled_pcds.append(pcds[-1])
        return subsampled_traj, subsampled_grips, subsampled_pcds
    return subsampled_traj, subsampled_grips


def extract_waypoints(traj, traj_states, num_waypoints):
    """Select L waypoints from a demo trajectory.

    First-principles selection:
      - Preserve stage events (gripper open/close transitions).
      - Preserve geometric progress (SE(3) motion coverage).
      - Remove near-static pauses before selecting fixed-count waypoints.
    """

    n = len(traj)
    if num_waypoints <= 0:
        return []
    if n == 0:
        return []

    if n <= num_waypoints:
        return list(range(n)) + [n - 1] * (num_waypoints - n)

    traj = np.asarray(traj, dtype=np.float64)
    traj_states = np.asarray(traj_states, dtype=np.float64)
    if traj_states.shape[0] != n:
        raise RuntimeError(
            f"Waypoint extraction expected {n} gripper states, got {traj_states.shape[0]}."
        )
    if not np.all(np.isfinite(traj_states)):
        raise RuntimeError("Waypoint extraction received non-finite gripper state values.")

    # Gripper labels are expected to be binary (RLBench-style 0/1). Keep this strict
    # so ambiguous continuous values do not silently change waypoint events.
    is_zero = np.isclose(traj_states, 0.0, atol=1e-6)
    is_one = np.isclose(traj_states, 1.0, atol=1e-6)
    if not np.all(is_zero | is_one):
        bad = traj_states[~(is_zero | is_one)]
        sample_bad = np.array2string(bad[:5], precision=4, separator=", ")
        raise RuntimeError(
            "Waypoint extraction expects binary gripper states in {0,1}; "
            f"got non-binary values (sample): {sample_bad}"
        )
    traj_states = np.where(is_one, 1.0, 0.0)

    rot_scale = 0.2  # ~1cm per 3deg
    motion_deadband = 6.0e-3

    def _pose_err(Ta: np.ndarray, Tb: np.ndarray) -> float:
        dist_trans = np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])
        dist_rot = Rot.from_matrix(Ta[:3, :3] @ Tb[:3, :3].T).magnitude()
        return float(dist_trans + rot_scale * dist_rot)

    # Step 1: remove near-static pause frames while preserving grip transitions.
    compressed = [0]
    for i in range(1, n):
        prev = compressed[-1]
        moved_enough = _pose_err(traj[prev], traj[i]) >= motion_deadband
        grip_flipped = traj_states[i] != traj_states[prev]
        if moved_enough or grip_flipped:
            compressed.append(i)
    if compressed[-1] != (n - 1):
        compressed.append(n - 1)
    compressed = sorted(set(compressed))

    if len(compressed) <= num_waypoints:
        return compressed + [compressed[-1]] * (num_waypoints - len(compressed))

    traj_c = traj[compressed]
    states_c = traj_states[compressed]
    m = len(compressed)

    # Step 2: mandatory anchors: start/end and both sides of each grip transition.
    anchors = {0, m - 1}
    change_idxs = []
    for i in range(1, m):
        if states_c[i] != states_c[i - 1]:
            change_idxs.append(i)
            anchors.add(i)
            anchors.add(i - 1)
    anchors = sorted(anchors)

    # If events exceed budget, keep endpoints + transition frames first.
    if len(anchors) > num_waypoints:
        keep = {0, m - 1}
        for i in change_idxs:
            if len(keep) >= num_waypoints:
                break
            keep.add(i)
        for i in change_idxs:
            if len(keep) >= num_waypoints:
                break
            keep.add(i - 1)
        selected = sorted(keep)
        out = [compressed[i] for i in selected]
        if len(out) < num_waypoints:
            out += [out[-1]] * (num_waypoints - len(out))
        return out[:num_waypoints]

    # Build cumulative arc length over compressed path.
    cum_arc = np.zeros(m, dtype=np.float64)
    for i in range(1, m):
        cum_arc[i] = cum_arc[i - 1] + _pose_err(traj_c[i - 1], traj_c[i])

    # Step 3: fill remaining slots by farthest-point sampling over cumulative arc.
    selected = set(anchors)
    while len(selected) < num_waypoints:
        candidates = [i for i in range(m) if i not in selected]
        if not candidates:
            break
        sel_sorted = sorted(selected)
        best = max(
            candidates,
            key=lambda c: min(abs(cum_arc[c] - cum_arc[s]) for s in sel_sorted),
        )
        selected.add(int(best))

    out = [compressed[i] for i in sorted(selected)]
    if len(out) < num_waypoints:
        out += [out[-1]] * (num_waypoints - len(out))
    return out[:num_waypoints]


def pose_error(T1, T2, rot_scale=0.01):
    dist_trans = np.linalg.norm(T1[:3, 3] - T2[:3, 3])
    dist_rot = Rot.from_matrix(T1[:3, :3] @ T2[:3, :3].T).magnitude()
    return dist_trans + rot_scale * dist_rot


def subsample_pcd(sample, num_points=2048):
    sample_filtered, _ = remove_statistical_outliers(sample, nb_neighbors=20, std_ratio=2.0)
    rand_idx = np.random.choice(len(sample_filtered), num_points, replace=True if len(sample_filtered) < num_points else False)
    return sample_filtered[rand_idx]


def remove_statistical_outliers(point_cloud, nb_neighbors=20, std_ratio=2.0):
    # Create a PointCloud object from the NumPy array
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Perform statistical outlier removal
    [filtered_pcd, inlier_indices] = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)

    # Convert the filtered PointCloud back to a NumPy array
    filtered_point_cloud = np.asarray(filtered_pcd.points)

    return filtered_point_cloud, inlier_indices
