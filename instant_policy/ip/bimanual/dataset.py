"""Bimanual dataset helpers."""

from __future__ import annotations

import os
from dataclasses import fields
from glob import glob
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from .data_adapter import BimanualWorldBatch


REQUIRED_KEYS = [f.name for f in fields(BimanualWorldBatch)]


class BimanualFileDataset(Dataset):
    """Dataset for bimanual world-batch samples saved as .pt files."""

    def __init__(self, data_path: str, pattern: str | None = None):
        self.data_path = data_path
        patterns = [pattern] if pattern else ["task_*.pt", "data_*.pt", "*.pt"]

        files: List[str] = []
        for p in patterns:
            files.extend(glob(os.path.join(data_path, p)))
            if files:
                break
        self.files = sorted(files)

        if not self.files:
            raise RuntimeError(f"No bimanual data files found in {data_path}")

    def __len__(self):
        return len(self.files)

    def _to_dict(self, item) -> Dict[str, torch.Tensor]:
        if isinstance(item, dict):
            data = item
        else:
            data = {}
            for k in REQUIRED_KEYS:
                if not hasattr(item, k):
                    raise KeyError(
                        f"Missing key '{k}' in sample. Required keys: {REQUIRED_KEYS}"
                    )
                data[k] = getattr(item, k)

        for k in REQUIRED_KEYS:
            if k not in data:
                raise KeyError(
                    f"Missing key '{k}' in sample. Required keys: {REQUIRED_KEYS}"
                )
        return data

    def __getitem__(self, idx: int):
        sample = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        sample = self._to_dict(sample)
        wb = BimanualWorldBatch(**sample)
        wb.validate()
        return wb


def collate_bimanual_world_batch(batch: List[BimanualWorldBatch]) -> BimanualWorldBatch:
    """Stack list of world-batch samples into one batched world-batch."""

    if not batch:
        raise ValueError("Empty batch")

    stacked = {}
    for k in REQUIRED_KEYS:
        vals = [getattr(x, k) for x in batch]
        if vals[0].ndim == 0:
            vals = [v.unsqueeze(0) for v in vals]
        stacked[k] = torch.stack(vals, dim=0)
    wb = BimanualWorldBatch(**stacked)
    wb.validate()
    return wb
