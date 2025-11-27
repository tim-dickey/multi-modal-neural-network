"""Dataset selection and automated split assembly.

Provides `build_dataloaders` to construct train/val/test loaders from
`config['data']['datasets']` entries. Each dataset entry supports:
- name: identifier
- type: one of multimodal, coco_captions, imagenet
- splits: mapping of split name to ratio (must sum to 1.0) (optional, defaults to train:1.0)
- use_in: restrict which splits this dataset contributes to (optional)
- remaining keys are forwarded to the underlying dataset class constructor
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import random

import torch
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset

from .dataset import (
    MultiModalDataset,
    COCOCaptionsDataset,
    ImageNetDataset,
)

_DATASET_TYPES: Dict[str, Any] = {
    "multimodal": MultiModalDataset,
    "coco_captions": COCOCaptionsDataset,
    "imagenet": ImageNetDataset,
}


def _instantiate(cfg: Dict[str, Any]) -> Dataset:
    ds_type = cfg.get("type")
    if ds_type not in _DATASET_TYPES:
        raise ValueError(f"Unknown dataset type: {ds_type}")
    cls = _DATASET_TYPES[ds_type]
    ignore = {"type", "splits", "use_in", "name", "enabled"}
    kwargs = {k: v for k, v in cfg.items() if k not in ignore}
    return cls(**kwargs)


def _split_indices(n: int, splits: Dict[str, float]) -> Dict[str, List[int]]:
    if not splits:
        return {"train": list(range(n))}
    total = sum(splits.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0 (got {total:.4f})")
    indices = list(range(n))
    random.shuffle(indices)
    out: Dict[str, List[int]] = {}
    cursor = 0
    items = list(splits.items())
    for i, (name, ratio) in enumerate(items):
        remain = len(indices) - cursor
        count = int(ratio * len(indices)) if i < len(items) - 1 else remain
        out[name] = indices[cursor : cursor + count]
        cursor += count
    return out


def build_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """Build train/val/test DataLoaders from config.

    Expected structure:
    data:
      batch_size: 32
      num_workers: 4
      pin_memory: true
      shuffle_train: true
      datasets:
        - name: multimodal_core
          type: multimodal
          data_dir: ./data/multimodal
          splits: {train: 0.8, val: 0.1, test: 0.1}
          enabled: true
        - name: captions_aux
          type: coco_captions
          root: ./data/coco/images
          ann_file: ./data/coco/annotations/captions_train2017.json
          splits: {train: 1.0}
          use_in: [train]

    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_cfg = config.get("data", {})
    ds_cfgs: List[Dict[str, Any]] = data_cfg.get("datasets", [])
    if not ds_cfgs:
        raise ValueError("config.data.datasets is empty.")

    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", True))
    shuffle_train = bool(data_cfg.get("shuffle_train", True))

    buckets: Dict[str, List[Dataset]] = {"train": [], "val": [], "test": []}

    for entry in ds_cfgs:
        if entry.get("enabled", True) is False:
            continue
        ds = _instantiate(entry)
        splits = entry.get("splits", {"train": 1.0})
        indices_map = _split_indices(len(ds), splits)
        use_in = entry.get("use_in")
        for split_name, split_indices in indices_map.items():
            if split_name not in buckets:
                continue
            if use_in and split_name not in use_in:
                continue
            if not split_indices:
                continue
            buckets[split_name].append(Subset(ds, split_indices))

    def _merge(parts: List[Dataset]) -> Optional[Dataset]:
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return ConcatDataset(parts)

    train_ds = _merge(buckets["train"])
    val_ds = _merge(buckets["val"])
    test_ds = _merge(buckets["test"])

    if train_ds is None:
        raise ValueError("No training data assembled. Check splits / enabled flags.")

    def _make_loader(ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    train_loader = _make_loader(train_ds, shuffle_train)
    val_loader = _make_loader(val_ds, False) if val_ds else None
    test_loader = _make_loader(test_ds, False) if test_ds else None
    return train_loader, val_loader, test_loader

__all__ = ["build_dataloaders"]
