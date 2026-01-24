import os
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from ip.utils.common_utils import transform_pcd
from ip.utils.data_proc import subsample_pcd, subsample_traj


def _task_index(path):
    name = os.path.basename(path)
    if not name.startswith("task_") or not name.endswith(".pt"):
        return None
    stem = name.split("_", 1)[1].split(".", 1)[0]
    try:
        return int(stem)
    except ValueError:
        return None


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        data_path,
        num_samples=None,
        num_demos=2,
        traj_horizon=10,
        pred_horizon=8,
        num_points=2048,
        rand_g_prob=0.0,
        subsample_live=False,
        live_spacing_trans=0.01,
        live_spacing_rot=3.0,
    ):
        self.data_path = data_path
        self.num_demos = num_demos
        self.traj_horizon = traj_horizon
        self.pred_horizon = pred_horizon
        self.num_points = num_points
        self.rand_g_prob = rand_g_prob
        self.subsample_live = subsample_live
        self.live_spacing_trans = live_spacing_trans
        self.live_spacing_rot = live_spacing_rot

        files = glob(os.path.join(data_path, "task_*.pt"))
        indexed = []
        for path in files:
            idx = _task_index(path)
            if idx is not None:
                indexed.append((idx, path))
        indexed.sort(key=lambda x: x[0])
        self.task_files = [p for _, p in indexed]
        if not self.task_files:
            raise RuntimeError(f"No task_*.pt files found in {data_path}")

        self.num_tasks = len(self.task_files)
        self.num_samples = num_samples if num_samples is not None else self.num_tasks
        self._rng = np.random.default_rng()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rng = self._rng
        if self.num_tasks == 0:
            raise RuntimeError("Empty task set.")
        task_idx = idx % self.num_tasks
        while True:
            task_path = self.task_files[task_idx]
            try:
                task = torch.load(task_path, map_location="cpu")
                demos = task["demos"]
                if not demos:
                    raise RuntimeError("Task has no demos.")

                live_idx = int(rng.integers(0, len(demos)))
                live_demo = demos[live_idx]
                ctx_indices = self._sample_context_indices(len(demos), live_idx, rng)
                cond_demos = [demos[i]["cond"] for i in ctx_indices]
                data = self._build_cond_data(cond_demos)

                live_step = self._build_live_step(live_demo, rng)
                pos_obs, actions, actions_grip, T_w_e, current_grip = live_step
                data.pos_obs = torch.tensor(pos_obs, dtype=torch.float32)
                data.current_grip = torch.tensor(current_grip, dtype=torch.float32).unsqueeze(0)
                data.current_grip = (data.current_grip - 0.5) * 2
                data.actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(0)
                data.actions_grip = torch.tensor(actions_grip, dtype=torch.float32).unsqueeze(0)
                data.actions_grip = (data.actions_grip - 0.5) * 2
                data.T_w_e = torch.tensor(T_w_e, dtype=torch.float32).unsqueeze(0)

                if rng.uniform() < self.rand_g_prob:
                    data.current_grip *= -1

                return data
            except Exception:
                task_idx = int(rng.integers(0, self.num_tasks))

    def _sample_context_indices(self, total, live_idx, rng):
        candidates = [i for i in range(total) if i != live_idx]
        if not candidates:
            candidates = [live_idx]
        if len(candidates) >= self.num_demos:
            return rng.choice(candidates, size=self.num_demos, replace=False).tolist()
        return rng.choice(candidates, size=self.num_demos, replace=True).tolist()

    def _build_cond_data(self, cond_demos):
        num_demos = len(cond_demos)
        num_traj_waypoints = len(cond_demos[0]["obs"])
        if num_traj_waypoints != self.traj_horizon:
            raise RuntimeError(
                f"Cond demo horizon {num_traj_waypoints} != expected {self.traj_horizon}"
            )

        joint_demo_pcd = []
        joint_demo_grasp = []
        batch_indices = []
        for n, demo in enumerate(cond_demos):
            for i, obs in enumerate(demo["obs"]):
                joint_demo_pcd.append(np.asarray(obs, dtype=np.float32))
                joint_demo_grasp.append(demo["grips"][i])
                batch_indices.append(np.zeros(len(obs)) + i + n * num_traj_waypoints)

        joint_demo_pcd = np.concatenate(joint_demo_pcd, axis=0)
        joint_demo_grasp = (np.array(joint_demo_grasp, dtype=np.float32) - 0.5) * 2
        batch_indices = np.concatenate(batch_indices, axis=0)

        demo_T_w_es = np.stack([demo["T_w_es"] for demo in cond_demos], axis=0)
        data = Data(
            pos_demos=torch.tensor(joint_demo_pcd, dtype=torch.float32),
            graps_demos=torch.tensor(joint_demo_grasp, dtype=torch.float32)
            .view(num_demos, num_traj_waypoints, 1)
            .unsqueeze(0),
            batch_demos=torch.tensor(batch_indices, dtype=torch.int64),
            batch_pos_obs=torch.tensor(np.zeros(self.num_points), dtype=torch.int64),
            demo_T_w_es=torch.tensor(demo_T_w_es, dtype=torch.float32).unsqueeze(0),
        )
        return data

    def _build_live_step(self, demo, rng):
        traj = demo["T_w_es"]
        grips = demo["grips"]
        pcds = [np.asarray(p, dtype=np.float32) for p in demo["pcds"]]

        if self.subsample_live:
            traj, grips, pcds = subsample_traj(
                traj,
                grips,
                pcds=pcds,
                trans_space=self.live_spacing_trans,
                rot_space=self.live_spacing_rot,
            )

        t = int(rng.integers(0, len(traj)))
        T_w_e = np.asarray(traj[t], dtype=np.float32)
        inv_T = np.linalg.inv(T_w_e)

        pcd = subsample_pcd(pcds[t], self.num_points)
        pos_obs = transform_pcd(pcd, inv_T)

        actions = []
        actions_grip = []
        for j in range(1, self.pred_horizon + 1):
            if t + j < len(traj):
                next_T = np.asarray(traj[t + j], dtype=np.float32)
                actions.append(inv_T @ next_T)
                actions_grip.append(grips[t + j])
            else:
                actions.append(np.eye(4, dtype=np.float32))
                actions_grip.append(grips[-1])

        return pos_obs, np.stack(actions, axis=0), np.array(actions_grip, dtype=np.float32), T_w_e, grips[t]
