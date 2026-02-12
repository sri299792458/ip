from torch.utils.data import Dataset
import torch
import os
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from collections import OrderedDict


class RunningDataset(Dataset):
    def __init__(self, data_path, num_samples, rec=False, rand_g_prob=0.0, random_rotation=False,
                 require_lang=False, sample_cache_size=0):
        self.data_path = data_path
        self.num_samples = num_samples
        self.rand_g_prob = rand_g_prob
        self.random_rotation = random_rotation
        self.rec = rec
        self.sample_cache_size = int(max(0, sample_cache_size))
        self._sample_cache = OrderedDict()
        if rec:
            self.data_attr = [
                'pos',
                'queries',
                'batch_queries',
                'batch_pos',
                'occupancy',
            ]
        else:
            self.data_attr = [
                # 'pos_demos',
                # 'graps_demos',
                # 'batch_demos',
                # 'pos_obs',
                # 'current_grip',
                # 'batch_pos_obs',
                # 'past_actions',
                # 'past_actions_grip',
                'actions',
                'actions_grip',
            ]
            if require_lang:
                self.data_attr.append('lang_emb')

    def __len__(self):
        return self.num_samples

    def _path(self, idx: int) -> str:
        return os.path.join(self.data_path, f"data_{idx}.pt")

    def _validate_data(self, data):
        for attr in self.data_attr:
            assert hasattr(data, attr)

    def _load_with_optional_cache(self, idx: int):
        path = self._path(idx)
        if self.sample_cache_size <= 0:
            data = torch.load(path)
            self._validate_data(data)
            return data

        mtime = os.path.getmtime(path)
        cached = self._sample_cache.get(idx)
        if cached is not None:
            cached_mtime, cached_data = cached
            if cached_mtime == mtime:
                self._sample_cache.move_to_end(idx)
                return cached_data

        data = torch.load(path)
        self._validate_data(data)
        self._sample_cache[idx] = (mtime, data)
        self._sample_cache.move_to_end(idx)
        while len(self._sample_cache) > self.sample_cache_size:
            self._sample_cache.popitem(last=False)
        return data

    def __getitem__(self, idx):
        while True:
            try:
                data = self._load_with_optional_cache(idx)

                if np.random.uniform() < self.rand_g_prob:
                    # Clone only when mutating to avoid poisoning cached entries.
                    data = data.clone()
                    data.current_grip *= -1

                if self.random_rotation and self.rec:
                    if not hasattr(data, "clone"):
                        raise RuntimeError("Data object does not support clone() for rotation augmentation.")
                    data = data.clone()
                    R = torch.tensor(Rot.random().as_matrix(), dtype=data.pos.dtype, device=data.pos.device)
                    data.pos = data.pos @ R.T
                    data.queries = data.queries @ R.T
                return data
            except Exception as e:
                idx = np.random.randint(0, self.num_samples)
