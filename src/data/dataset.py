"""Dataset loading and preprocessing for multi-modal training."""

import json
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    cast,
)

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

if TYPE_CHECKING:
    pass


class CocoLike(Protocol):
    def getCatIds(self) -> Iterable[int]: ...

    def getImgIds(self) -> Iterable[int]: ...

    def loadAnns(self, ids: Iterable[int]) -> list[Dict[str, Any]]: ...

    def loadCats(self, ids: Iterable[int]) -> list[Dict[str, Any]]: ...

    def loadImgs(self, ids: Iterable[int]) -> list[Dict[str, Any]]: ...


COCOType = "_COCO | CocoLike"


class MultiModalDataset(Dataset):
    """Base dataset class for multi-modal image-text data."""

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        img_size: int = 224,
        max_text_length: int = 512,
        tokenizer: Optional[Any] = None,
        *,
        augment: bool = True,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.split = split
        self.img_size = img_size
        self.max_text_length = max_text_length
        self.tokenizer = tokenizer
        self.augment = augment and (split == "train")

        # Load annotations
        self.samples = self._load_annotations()

        # Image transformations
        self.transform = self._get_transforms()

    def _load_annotations(self) -> List[Dict[str, Any]]:
        """Load dataset annotations."""
        # Try different annotation file names
        possible_files = [
            self.data_path / "annotations.json",
            self.data_path / f"{self.split}.json",
            self.data_path / f"annotations_{self.split}.json",
        ]

        for annotation_file in possible_files:
            if annotation_file.exists():
                try:
                    with open(annotation_file, "r", encoding="utf-8") as f:
                        data = cast(List[Dict[str, Any]], json.load(f))
                    return data
                except (json.JSONDecodeError, OSError, ValueError) as e:
                    # Malformed JSON or IO issue; try next file or fallback
                    print(f"Failed to read annotations from {annotation_file}: {e}")
                    continue

        # Return dummy data for demonstration
        return [
            {
                "image_path": "image_0.jpg",
                "caption": "A sample image caption",
                "label": 0,
            }
        ]

    def _get_transforms(self) -> transforms.Compose:
        """Get image transformations."""
        if self.augment:
            return transforms.Compose(
                [
                    transforms.Resize(self.img_size + 32),
                    transforms.RandomCrop(self.img_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize(self.img_size),
                    transforms.CenterCrop(self.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def _load_image(self, image_path: str) -> Image.Image:
        """Load and return PIL Image."""
        full_path = self.data_path / image_path
        if full_path.exists():
            try:
                return Image.open(full_path).convert("RGB")
            except (OSError, ValueError) as e:
                print(f"Failed to open image {full_path}: {e}")
        # Return dummy image for demonstration
        return Image.new("RGB", (self.img_size, self.img_size), color="gray")

    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text input."""
        if self.tokenizer is not None:
            # Use provided tokenizer (e.g., from transformers)
            encoding = self.tokenizer(
                text,
                max_length=self.max_text_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
            }
        else:
            # Simple character-based encoding
            input_ids = (
                [1]
                + [min(ord(c) % 30000, 30521) for c in text[: self.max_text_length - 2]]
                + [2]
            )
            attention_mask = [1] * len(input_ids)

            # Pad to max length
            while len(input_ids) < self.max_text_length:
                input_ids.append(0)
                attention_mask.append(0)

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        sample = self.samples[idx]

        # Load and transform image
        image = self._load_image(sample["image_path"])
        image = self.transform(image)

        # Tokenize text
        text_encoding = self._tokenize_text(sample["caption"])

        # Prepare output
        output = {
            "image": image,
            "input_ids": text_encoding["input_ids"],
            "attention_mask": text_encoding["attention_mask"],
            "label": torch.tensor(sample.get("label", 0), dtype=torch.long),
        }

        return output

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples."""
        images = torch.stack([item["image"] for item in batch])
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_masks = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["label"] for item in batch])

        return {
            "image": images,
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "label": labels,
        }


class COCOCaptionsDataset(MultiModalDataset):
    """Dataset for COCO Captions."""

    def _load_annotations(self) -> List[Dict[str, Any]]:
        """Load COCO annotations."""
        from pycocotools.coco import COCO

        try:
            annotation_file = (
                self.data_path / "annotations" / f"captions_{self.split}2017.json"
            )
            coco = COCO(annotation_file)

            samples = []
            img_ids = coco.getImgIds()

            for img_id in img_ids:
                img_info = coco.loadImgs(img_id)[0]
                ann_ids = coco.getAnnIds(imgIds=img_id)
                anns = coco.loadAnns(ann_ids)

                for ann in anns:
                    ann_dict = cast(Dict[str, Any], ann)
                    samples.append(
                        {
                            "image_path": f"{self.split}2017/{img_info['file_name']}",
                            "caption": ann_dict["caption"],
                            "label": 0,  # COCO doesn't have labels for captioning
                        }
                    )

            return samples
        except (FileNotFoundError, OSError, KeyError, ValueError, ImportError):
            # Expected data/IO/import issues: fallback to parent implementation
            return super()._load_annotations()


class ImageNetDataset(Dataset):
    """Simple ImageNet-style dataset."""

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        img_size: int = 224,
        *,
        augment: bool = True,
    ):
        super().__init__()
        self.data_path = Path(data_path) / split
        self.img_size = img_size
        self.augment = augment and (split == "train")

        # Load image paths and labels
        self.samples = self._load_samples()
        self.transform = self._get_transforms()

    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load image paths and labels."""
        samples = []

        # Assume directory structure: data_path/class_name/image.jpg
        if self.data_path.exists():
            class_dirs = sorted([d for d in self.data_path.iterdir() if d.is_dir()])

            for label, class_dir in enumerate(class_dirs):
                for img_path in class_dir.glob("*.jpg"):
                    samples.append((str(img_path), label))
                for img_path in class_dir.glob("*.JPEG"):
                    samples.append((str(img_path), label))

        return samples if samples else [("dummy.jpg", 0)]

    def _get_transforms(self) -> transforms.Compose:
        """Get image transformations."""
        if self.augment:
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(self.img_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.3, 0.3, 0.3),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(self.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except (OSError, FileNotFoundError, ValueError):
            image = Image.new("RGB", (self.img_size, self.img_size))

        image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "input_ids": torch.zeros(512, dtype=torch.long),  # Dummy for vision-only
            "attention_mask": torch.zeros(512, dtype=torch.long),
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    *,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Create a DataLoader with specified parameters."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


def create_dataset_from_config(config: Dict) -> Tuple[Dataset, Dataset]:
    """
    Create train and validation datasets from config.

    Args:
        config: Configuration dictionary

    Returns:
        (train_dataset, val_dataset)
    """
    data_config = config.get("data", {})
    dataset_name = data_config.get("train_dataset", "coco_captions")

    # Get tokenizer if needed
    tokenizer = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    except (ImportError, OSError, ValueError):
        # Missing dependency or local/IO error when loading pretrained tokenizer
        tokenizer = None

    train_dataset: Dataset
    val_dataset: Dataset

    if dataset_name == "coco_captions":
        train_dataset = COCOCaptionsDataset(
            data_path=data_config.get("data_path", "./data/coco"),
            split="train",
            img_size=config.get("model", {})
            .get("vision_encoder", {})
            .get("img_size", 224),
            tokenizer=tokenizer,
            augment=True,
        )
        val_dataset = COCOCaptionsDataset(
            data_path=data_config.get("data_path", "./data/coco"),
            split="val",
            img_size=config.get("model", {})
            .get("vision_encoder", {})
            .get("img_size", 224),
            tokenizer=tokenizer,
            augment=False,
        )
    elif dataset_name == "imagenet":
        train_dataset = ImageNetDataset(
            data_path=data_config.get("data_path", "./data/imagenet"),
            split="train",
            img_size=config.get("model", {})
            .get("vision_encoder", {})
            .get("img_size", 224),
            augment=True,
        )
        val_dataset = ImageNetDataset(
            data_path=data_config.get("data_path", "./data/imagenet"),
            split="val",
            img_size=config.get("model", {})
            .get("vision_encoder", {})
            .get("img_size", 224),
            augment=False,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_dataset, val_dataset


def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    *,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory

    Returns:
        (train_loader, val_loader)
    """
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


def get_transforms(config: Dict, *, is_train: bool = True) -> transforms.Compose:
    """
    Get image transforms based on configuration.

    Args:
        config: Configuration dictionary
        is_train: Whether this is for training (includes augmentation)

    Returns:
        Composed transforms
    """
    img_size = config.get("data", {}).get("image_size", 224)
    aug_config = config.get("data", {}).get("augmentation", {})

    if is_train:
        transform_list = []

        if aug_config.get("random_crop", True):
            transform_list.append(
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0))
            )
        else:
            transform_list.append(transforms.Resize(img_size))
            transform_list.append(transforms.CenterCrop(img_size))

        if aug_config.get("random_flip", True):
            transform_list.append(transforms.RandomHorizontalFlip())

        if aug_config.get("color_jitter", True):
            transform_list.append(
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                )
            )

        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        transform_list = [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

    return transforms.Compose(transform_list)
