from __future__ import annotations

import os
from glob import glob
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from ip.generation_bimanual.config import BimanualGenerationConfig
from ip.generation_bimanual.primitives import sample_trajectory


class BimanualPseudoDemoGenerator:
    def __init__(self, config: BimanualGenerationConfig):
        self.config = config
        self.config.validate()
        self.rng = np.random.default_rng(int(config.seed))

    def _sample_task_name(self) -> str:
        if self.config.forced_task is not None:
            return self.config.forced_task

        tasks = self.config.task_names
        if not tasks:
            raise RuntimeError("task_names is empty")

        if self.config.task_weights is None:
            idx = int(self.rng.integers(0, len(tasks)))
            return tasks[idx]

        w = np.asarray(self.config.task_weights, dtype=np.float64)
        w = w / np.sum(w)
        idx = int(self.rng.choice(np.arange(len(tasks)), p=w))
        return tasks[idx]

    @staticmethod
    def _list_existing_indices(save_dir: str) -> List[int]:
        out = []
        for p in glob(os.path.join(save_dir, "task_*.pt")):
            stem = os.path.splitext(os.path.basename(p))[0]
            try:
                out.append(int(stem.split("_")[-1]))
            except ValueError:
                continue
        return sorted(out)

    def _clear_existing(self, save_dir: str) -> None:
        for p in glob(os.path.join(save_dir, "task_*.pt")):
            os.remove(p)

    def _sample_to_world_batch(self, task_name: str) -> Dict[str, torch.Tensor]:
        traj = sample_trajectory(task_name, self.config, self.rng)
        steps = traj.left_seq.shape[0]
        horizon = int(self.config.pred_horizon)

        max_start = steps - (horizon + 1)
        start = int(self.rng.integers(0, max_start + 1))

        pts = traj.scene_points_seq[start]
        pcd_dtype = torch.float16 if self.config.pcd_storage_dtype == "float16" else torch.float32

        sample = {
            "points_world": torch.as_tensor(pts[None, ...], dtype=pcd_dtype),
            "T_w_left_current": torch.as_tensor(traj.left_seq[start][None, ...], dtype=torch.float32),
            "T_w_right_current": torch.as_tensor(traj.right_seq[start][None, ...], dtype=torch.float32),
            "T_w_left_future": torch.as_tensor(
                traj.left_seq[start + 1 : start + 1 + horizon][None, ...], dtype=torch.float32
            ),
            "T_w_right_future": torch.as_tensor(
                traj.right_seq[start + 1 : start + 1 + horizon][None, ...], dtype=torch.float32
            ),
            "grip_left_current": torch.as_tensor([traj.grip_left[start]], dtype=torch.float32),
            "grip_right_current": torch.as_tensor([traj.grip_right[start]], dtype=torch.float32),
            "grip_left_future": torch.as_tensor(
                traj.grip_left[start + 1 : start + 1 + horizon][None, ...], dtype=torch.float32
            ),
            "grip_right_future": torch.as_tensor(
                traj.grip_right[start + 1 : start + 1 + horizon][None, ...], dtype=torch.float32
            ),
        }
        return sample

    def _slot_iter_non_fill(self):
        for k in range(int(self.config.num_samples)):
            global_idx = int(self.config.task_start + k)
            if global_idx % self.config.num_shards != self.config.shard_id:
                continue
            if self.config.buffer_size is None:
                file_idx = global_idx
            else:
                file_idx = global_idx % int(self.config.buffer_size)
            yield global_idx, file_idx

    def _slot_iter_fill(self):
        # Fill each ring-buffer slot owned by this shard exactly once.
        if self.config.buffer_size is None:
            raise RuntimeError("fill_buffer requires buffer_size")

        target_slots = [
            i
            for i in range(int(self.config.buffer_size))
            if i % self.config.num_shards == self.config.shard_id
        ]
        filled = set()
        global_idx = int(self.config.task_start)

        while len(filled) < len(target_slots):
            if global_idx % self.config.num_shards != self.config.shard_id:
                global_idx += 1
                continue
            file_idx = global_idx % int(self.config.buffer_size)
            if file_idx in filled:
                global_idx += 1
                continue
            filled.add(file_idx)
            yield global_idx, file_idx
            global_idx += 1

    def _count_non_fill_slots(self) -> int:
        total = 0
        for k in range(int(self.config.num_samples)):
            global_idx = int(self.config.task_start + k)
            if global_idx % self.config.num_shards == self.config.shard_id:
                total += 1
        return total

    def generate_dataset(self) -> None:
        save_dir = self.config.save_dir
        os.makedirs(save_dir, exist_ok=True)

        if not self.config.append and self.config.buffer_size is None:
            self._clear_existing(save_dir)

        if self.config.buffer_size is None and self.config.append and self.config.task_start == 0:
            existing = self._list_existing_indices(save_dir)
            if existing:
                self.config.task_start = int(existing[-1] + 1)

        slots = self._slot_iter_fill() if self.config.fill_buffer else self._slot_iter_non_fill()
        if self.config.fill_buffer:
            total_slots = len(
                [
                    i
                    for i in range(int(self.config.buffer_size))
                    if i % self.config.num_shards == self.config.shard_id
                ]
            )
        else:
            total_slots = self._count_non_fill_slots()

        written = 0
        for _, file_idx in tqdm(slots, total=total_slots, desc="Generating bimanual pseudo samples"):
            task_name = self._sample_task_name()
            sample = self._sample_to_world_batch(task_name)
            out_path = os.path.join(save_dir, f"task_{file_idx:07d}.pt")
            torch.save(sample, out_path)
            written += 1

        if written == 0:
            raise RuntimeError(
                "No samples were written. Check shard_id/num_shards and generation settings."
            )
