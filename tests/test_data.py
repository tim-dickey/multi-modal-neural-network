"""Tests for data pipeline."""

import pytest
import torch
from pathlib import Path
from src.data.dataset import (
    MultiModalDataset,
    COCOCaptionsDataset,
    ImageNetDataset,
    create_data_loaders,
    get_transforms
)


class TestDataTransforms:
    """Tests for data transformations."""
    
    def test_get_transforms_train(self):
        """Test training transforms."""
        config = {
            'data': {
                'image_size': 224,
                'augmentation': {
                    'random_crop': True,
                    'random_flip': True,
                    'color_jitter': True
                }
            }
        }
        
        transforms = get_transforms(config, is_train=True)
        assert transforms is not None
        
        # Test on sample image
        image = torch.randint(0, 255, (3, 256, 256), dtype=torch.uint8)
        transformed = transforms(image)
        
        assert transformed.shape == (3, 224, 224)
        assert transformed.dtype == torch.float32
        
    def test_get_transforms_eval(self):
        """Test evaluation transforms."""
        config = {
            'data': {
                'image_size': 224,
                'augmentation': {}
            }
        }
        
        transforms = get_transforms(config, is_train=False)
        assert transforms is not None
        
        # Test on sample image
        image = torch.randint(0, 255, (3, 256, 256), dtype=torch.uint8)
        transformed = transforms(image)
        
        assert transformed.shape == (3, 224, 224)
        assert transformed.dtype == torch.float32


class TestMultiModalDataset:
    """Tests for MultiModalDataset."""
    
    def test_dataset_creation(self, temp_data_dir, model_config):
        """Test dataset creation."""
        # Create dummy data
        images_dir = temp_data_dir / "images"
        images_dir.mkdir()
        annotations_file = temp_data_dir / "annotations.json"
        
        # Create dummy images
        import json
        from PIL import Image
        import numpy as np
        
        annotations = []
        for i in range(5):
            img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            img_path = images_dir / f"image_{i}.jpg"
            img.save(img_path)
            
            annotations.append({
                "image_id": i,
                "image_path": str(img_path),
                "caption": f"This is test caption {i}",
                "label": i % 3
            })
        
        with open(annotations_file, 'w') as f:
            json.dump(annotations, f)
        
        # Create dataset
        dataset = MultiModalDataset(
            annotations_file=str(annotations_file),
            images_dir=str(images_dir),
            config=model_config,
            is_train=True
        )
        
        assert len(dataset) == 5
        
    def test_dataset_getitem(self, temp_data_dir, model_config):
        """Test getting items from dataset."""
        # Create dummy data
        images_dir = temp_data_dir / "images"
        images_dir.mkdir()
        annotations_file = temp_data_dir / "annotations.json"
        
        import json
        from PIL import Image
        import numpy as np
        
        annotations = []
        for i in range(3):
            img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            img_path = images_dir / f"image_{i}.jpg"
            img.save(img_path)
            
            annotations.append({
                "image_id": i,
                "image_path": str(img_path),
                "caption": f"Test caption {i}",
                "label": i
            })
        
        with open(annotations_file, 'w') as f:
            json.dump(annotations, f)
        
        dataset = MultiModalDataset(
            annotations_file=str(annotations_file),
            images_dir=str(images_dir),
            config=model_config,
            is_train=False
        )
        
        # Get first item
        item = dataset[0]
        
        assert 'image' in item
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'label' in item
        
        assert item['image'].shape == (3, 224, 224)
        assert item['input_ids'].ndim == 1
        assert item['attention_mask'].ndim == 1
        assert isinstance(item['label'], int)
        
    def test_dataset_collate_fn(self, temp_data_dir, model_config):
        """Test batch collation."""
        # Create dummy data
        images_dir = temp_data_dir / "images"
        images_dir.mkdir()
        annotations_file = temp_data_dir / "annotations.json"
        
        import json
        from PIL import Image
        import numpy as np
        
        annotations = []
        for i in range(4):
            img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            img_path = images_dir / f"image_{i}.jpg"
            img.save(img_path)
            
            annotations.append({
                "image_id": i,
                "image_path": str(img_path),
                "caption": f"Caption {i}",
                "label": i
            })
        
        with open(annotations_file, 'w') as f:
            json.dump(annotations, f)
        
        dataset = MultiModalDataset(
            annotations_file=str(annotations_file),
            images_dir=str(images_dir),
            config=model_config,
            is_train=False
        )
        
        # Get batch
        batch = [dataset[i] for i in range(2)]
        collated = dataset.collate_fn(batch)
        
        assert collated['image'].shape[0] == 2
        assert collated['input_ids'].shape[0] == 2
        assert collated['attention_mask'].shape[0] == 2
        assert len(collated['label']) == 2


class TestCOCOCaptionsDataset:
    """Tests for COCO Captions dataset."""
    
    def test_coco_dataset_creation(self, temp_data_dir, model_config):
        """Test COCO dataset creation."""
        # Create dummy COCO annotations
        images_dir = temp_data_dir / "images"
        images_dir.mkdir()
        annotations_file = temp_data_dir / "coco_annotations.json"
        
        import json
        from PIL import Image
        import numpy as np
        
        images = []
        annotations = []
        
        for i in range(3):
            img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            img_path = images_dir / f"{i:012d}.jpg"
            img.save(img_path)
            
            images.append({
                "id": i,
                "file_name": f"{i:012d}.jpg",
                "height": 256,
                "width": 256
            })
            
            annotations.append({
                "id": i,
                "image_id": i,
                "caption": f"A test image number {i}"
            })
        
        coco_data = {
            "images": images,
            "annotations": annotations
        }
        
        with open(annotations_file, 'w') as f:
            json.dump(coco_data, f)
        
        # Create dataset
        dataset = COCOCaptionsDataset(
            annotations_file=str(annotations_file),
            images_dir=str(images_dir),
            config=model_config,
            is_train=False
        )
        
        assert len(dataset) > 0


class TestImageNetDataset:
    """Tests for ImageNet dataset."""
    
    def test_imagenet_dataset_creation(self, temp_data_dir, model_config):
        """Test ImageNet dataset creation."""
        # Create dummy ImageNet structure
        from PIL import Image
        import numpy as np
        
        # Create class directories
        for class_id in range(3):
            class_dir = temp_data_dir / f"n{class_id:08d}"
            class_dir.mkdir()
            
            # Create dummy images
            for img_id in range(2):
                img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
                img_path = class_dir / f"{class_id}_{img_id}.JPEG"
                img.save(img_path)
        
        # Create dataset
        dataset = ImageNetDataset(
            root_dir=str(temp_data_dir),
            config=model_config,
            is_train=True
        )
        
        assert len(dataset) == 6  # 3 classes * 2 images
        
    def test_imagenet_getitem(self, temp_data_dir, model_config):
        """Test getting ImageNet items."""
        from PIL import Image
        import numpy as np
        
        # Create class directories
        for class_id in range(2):
            class_dir = temp_data_dir / f"n{class_id:08d}"
            class_dir.mkdir()
            
            img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            img_path = class_dir / f"{class_id}_0.JPEG"
            img.save(img_path)
        
        dataset = ImageNetDataset(
            root_dir=str(temp_data_dir),
            config=model_config,
            is_train=False
        )
        
        item = dataset[0]
        
        assert 'image' in item
        assert 'label' in item
        assert item['image'].shape == (3, 224, 224)


class TestDataLoaders:
    """Tests for data loader creation."""
    
    def test_create_data_loaders(self, temp_data_dir, model_config):
        """Test data loader creation."""
        # Create dummy dataset files
        images_dir = temp_data_dir / "images"
        images_dir.mkdir()
        
        train_file = temp_data_dir / "train.json"
        val_file = temp_data_dir / "val.json"
        
        import json
        from PIL import Image
        import numpy as np
        
        # Create training data
        train_annotations = []
        for i in range(8):
            img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            img_path = images_dir / f"train_{i}.jpg"
            img.save(img_path)
            
            train_annotations.append({
                "image_id": i,
                "image_path": str(img_path),
                "caption": f"Train caption {i}",
                "label": i % 3
            })
        
        with open(train_file, 'w') as f:
            json.dump(train_annotations, f)
        
        # Create validation data
        val_annotations = []
        for i in range(4):
            img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            img_path = images_dir / f"val_{i}.jpg"
            img.save(img_path)
            
            val_annotations.append({
                "image_id": i,
                "image_path": str(img_path),
                "caption": f"Val caption {i}",
                "label": i % 3
            })
        
        with open(val_file, 'w') as f:
            json.dump(val_annotations, f)
        
        # Update config
        model_config['data']['train_annotations'] = str(train_file)
        model_config['data']['val_annotations'] = str(val_file)
        model_config['data']['images_dir'] = str(images_dir)
        model_config['data']['batch_size'] = 2
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(model_config)
        
        assert train_loader is not None
        assert val_loader is not None
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        
    def test_data_loader_iteration(self, temp_data_dir, model_config):
        """Test iterating through data loader."""
        # Create dummy data
        images_dir = temp_data_dir / "images"
        images_dir.mkdir()
        annotations_file = temp_data_dir / "annotations.json"
        
        import json
        from PIL import Image
        import numpy as np
        
        annotations = []
        for i in range(6):
            img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            img_path = images_dir / f"image_{i}.jpg"
            img.save(img_path)
            
            annotations.append({
                "image_id": i,
                "image_path": str(img_path),
                "caption": f"Caption {i}",
                "label": i % 2
            })
        
        with open(annotations_file, 'w') as f:
            json.dump(annotations, f)
        
        dataset = MultiModalDataset(
            annotations_file=str(annotations_file),
            images_dir=str(images_dir),
            config=model_config,
            is_train=False
        )
        
        from torch.utils.data import DataLoader
        
        data_loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=dataset.collate_fn
        )
        
        # Iterate through one batch
        for batch in data_loader:
            assert batch['image'].shape[0] == 2
            assert batch['input_ids'].shape[0] == 2
            assert batch['attention_mask'].shape[0] == 2
            assert len(batch['label']) == 2
            break


@pytest.mark.slow
class TestDataPipelinePerformance:
    """Tests for data pipeline performance."""
    
    def test_data_loading_speed(self, temp_data_dir, model_config):
        """Test data loading speed."""
        import time
        
        # Create larger dataset
        images_dir = temp_data_dir / "images"
        images_dir.mkdir()
        annotations_file = temp_data_dir / "annotations.json"
        
        import json
        from PIL import Image
        import numpy as np
        
        annotations = []
        for i in range(20):
            img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            img_path = images_dir / f"image_{i}.jpg"
            img.save(img_path)
            
            annotations.append({
                "image_id": i,
                "image_path": str(img_path),
                "caption": f"Caption {i}",
                "label": i % 5
            })
        
        with open(annotations_file, 'w') as f:
            json.dump(annotations, f)
        
        dataset = MultiModalDataset(
            annotations_file=str(annotations_file),
            images_dir=str(images_dir),
            config=model_config,
            is_train=False
        )
        
        from torch.utils.data import DataLoader
        
        data_loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,  # Single worker for testing
            collate_fn=dataset.collate_fn
        )
        
        # Time loading
        start_time = time.time()
        for batch in data_loader:
            pass
        elapsed = time.time() - start_time
        
        # Should be reasonably fast (< 5 seconds for 20 images)
        assert elapsed < 5.0
