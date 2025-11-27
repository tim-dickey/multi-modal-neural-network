"""Data loading and preprocessing.

Exports dataset classes, legacy helper creation functions, and the new
`build_dataloaders` selector utility for assembling multi-source train/val/test
loaders from the configuration.
"""

from .dataset import (
    COCOCaptionsDataset,
    ImageNetDataset,
    MultiModalDataset,
    create_dataloader,
    create_dataset_from_config,
)
from .selector import build_dataloaders

__all__ = [
    "MultiModalDataset",
    "COCOCaptionsDataset",
    "ImageNetDataset",
    "create_dataloader",
    "create_dataset_from_config",
    "build_dataloaders",
]
