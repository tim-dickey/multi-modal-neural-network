"""Tests for dataset selector build_dataloaders."""

import random
import json
from pathlib import Path
import pytest

from src.data.selector import build_dataloaders


def _create_multimodal_dir(base: Path, name: str, count: int = 10):
    """Create a dummy multimodal dataset directory with annotations.json and images."""
    ds_dir = base / name
    ds_dir.mkdir()
    images_dir = ds_dir / "images"
    images_dir.mkdir()
    annotations = []
    from PIL import Image
    import numpy as np

    for i in range(count):
        img = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        img_path = images_dir / f"image_{i}.jpg"
        img.save(img_path)
        annotations.append(
            {
                "image_id": i,
                "image_path": str(img_path),
                "caption": f"Sample caption {i}",
                "label": i % 3,
            }
        )

    with open(ds_dir / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(annotations, f)
    return ds_dir


class TestBuildDataLoaders:
    def test_basic_multi_dataset_assembly(self, temp_data_dir):
        random.seed(0)
        ds1 = _create_multimodal_dir(temp_data_dir, "ds1", count=10)
        ds2 = _create_multimodal_dir(temp_data_dir, "ds2", count=5)

        config = {
            "data": {
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
                "shuffle_train": False,
                "datasets": [
                    {
                        "name": "ds1",
                        "type": "multimodal",
                        "data_path": str(ds1),
                        "splits": {"train": 0.6, "val": 0.2, "test": 0.2},
                        "enabled": True,
                    },
                    {
                        "name": "ds2",
                        "type": "multimodal",
                        "data_path": str(ds2),
                        "splits": {"train": 1.0},
                        "use_in": ["train"],  # Only contribute to train
                        "enabled": True,
                    },
                ],
            }
        }

        train_loader, val_loader, test_loader = build_dataloaders(config)

        # ds1 splits (10 items): train=6, val=2, test=2. ds2 contributes 5 to train only.
        assert len(train_loader.dataset) == 11  # 6 + 5
        assert val_loader is not None and len(val_loader.dataset) == 2
        assert test_loader is not None and len(test_loader.dataset) == 2

        # Check one batch from train loader
        batch = next(iter(train_loader))
        assert "image" in batch and "label" in batch
        assert batch["image"].shape[0] == 2

    def test_disabled_dataset_excluded(self, temp_data_dir):
        random.seed(0)
        ds1 = _create_multimodal_dir(temp_data_dir, "ds_enabled", count=8)
        ds2 = _create_multimodal_dir(temp_data_dir, "ds_disabled", count=8)

        config = {
            "data": {
                "batch_size": 4,
                "num_workers": 0,
                "pin_memory": False,
                "shuffle_train": False,
                "datasets": [
                    {
                        "name": "enabled",
                        "type": "multimodal",
                        "data_path": str(ds1),
                        "splits": {"train": 1.0},
                        "enabled": True,
                    },
                    {
                        "name": "disabled",
                        "type": "multimodal",
                        "data_path": str(ds2),
                        "splits": {"train": 1.0},
                        "enabled": False,
                    },
                ],
            }
        }

        train_loader, val_loader, test_loader = build_dataloaders(config)
        assert len(train_loader.dataset) == 8  # Only first dataset
        assert val_loader is None and test_loader is None

    def test_invalid_split_sum_raises(self, temp_data_dir):
        ds1 = _create_multimodal_dir(temp_data_dir, "broken", count=5)
        config = {
            "data": {
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
                "datasets": [
                    {
                        "name": "broken",
                        "type": "multimodal",
                        "data_path": str(ds1),
                        "splits": {"train": 0.5, "val": 0.3},  # Sum = 0.8
                    }
                ],
            }
        }
        with pytest.raises(ValueError):
            build_dataloaders(config)
