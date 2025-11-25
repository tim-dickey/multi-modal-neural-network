"""Pytest configuration and shared fixtures for testing."""

import pytest
import torch
import tempfile
import yaml
from pathlib import Path
from PIL import Image
import numpy as np


@pytest.fixture
def device():
    """Get available device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def batch_size():
    """Default batch size for tests."""
    return 2


@pytest.fixture
def img_size():
    """Default image size for tests."""
    return 224


@pytest.fixture
def seq_length():
    """Default sequence length for tests."""
    return 128


@pytest.fixture
def hidden_dim():
    """Default hidden dimension for tests."""
    return 256


@pytest.fixture
def num_classes():
    """Default number of classes for tests."""
    return 10


@pytest.fixture
def sample_images(batch_size, img_size):
    """Generate sample image tensors."""
    return torch.randn(batch_size, 3, img_size, img_size)


@pytest.fixture
def sample_text_inputs(batch_size, seq_length):
    """Generate sample text input tensors."""
    input_ids = torch.randint(0, 30522, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }


@pytest.fixture
def sample_labels(batch_size, num_classes):
    """Generate sample labels."""
    return torch.randint(0, num_classes, (batch_size,))


@pytest.fixture
def vision_encoder_config(img_size, hidden_dim):
    """Configuration for vision encoder."""
    return {
        'img_size': img_size,
        'patch_size': 16,
        'in_channels': 3,
        'hidden_dim': hidden_dim,
        'num_layers': 4,  # Smaller for testing
        'num_heads': 4,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
        'use_cls_token': True
    }


@pytest.fixture
def text_encoder_config(hidden_dim, seq_length):
    """Configuration for text encoder."""
    return {
        'vocab_size': 30522,
        'hidden_dim': hidden_dim,
        'num_layers': 4,  # Smaller for testing
        'num_heads': 4,
        'max_seq_length': seq_length,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
        'use_cls_token': True
    }


@pytest.fixture
def fusion_config(hidden_dim):
    """Configuration for fusion layer."""
    return {
        'type': 'early',
        'hidden_dim': hidden_dim,
        'num_layers': 2,  # Smaller for testing
        'num_heads': 4,
        'mlp_ratio': 4.0,
        'dropout': 0.1
    }


@pytest.fixture
def double_loop_config(hidden_dim):
    """Configuration for double-loop controller."""
    return {
        'model_hidden_dim': hidden_dim,
        'controller_type': 'lstm',
        'hidden_dim': 128,
        'update_frequency': 10,
        'meta_lr': 1e-5
    }


@pytest.fixture
def head_config(hidden_dim, num_classes):
    """Configuration for task head."""
    return {
        'type': 'classification',
        'hidden_dim': hidden_dim,
        'num_classes': num_classes,
        'dropout': 0.1
    }


@pytest.fixture
def model_config(vision_encoder_config, text_encoder_config, 
                fusion_config, double_loop_config, head_config):
    """Complete model configuration."""
    return {
        'model': {
            'vision_encoder': vision_encoder_config,
            'text_encoder': text_encoder_config,
            'fusion': fusion_config,
            'double_loop': double_loop_config,
            'heads': head_config,
            'use_double_loop': True
        },
        'training': {
            'max_epochs': 2,
            'inner_lr': 1e-3,
            'micro_batch_size': 2,
            'gradient_accumulation': 1,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'weight_decay': 0.01,
            'max_grad_norm': 1.0,
            'mixed_precision': None,  # Disable for testing
            'gradient_checkpointing': False,
            'warmup_steps': 5
        },
        'data': {
            'train_dataset': 'dummy',
            'val_dataset': 'dummy',
            'batch_size': 2,
            'num_workers': 0,
            'pin_memory': False
        },
        'logging': {
            'project': 'test-project',
            'experiment': 'test-experiment',
            'log_every': 1,
            'save_every': 100,
            'use_wandb': False
        },
        'paths': {
            'output_dir': './test_outputs',
            'checkpoint_dir': './test_checkpoints',
            'log_dir': './test_logs'
        },
        'hardware': {
            'device': 'cpu',
            'max_memory': '2GB'
        }
    }


@pytest.fixture
def temp_config_file(model_config):
    """Create a temporary config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(model_config, f)
        config_path = f.name
    
    yield config_path
    
    # Cleanup
    Path(config_path).unlink(missing_ok=True)


@pytest.fixture
def sample_pil_image(img_size):
    """Generate a sample PIL Image."""
    img_array = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def temp_image_file(sample_pil_image):
    """Create a temporary image file."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        sample_pil_image.save(f.name)
        image_path = f.name
    
    yield image_path
    
    # Cleanup
    Path(image_path).unlink(missing_ok=True)


@pytest.fixture
def dummy_dataset_path():
    """Create a temporary directory for dummy dataset."""
    temp_dir = tempfile.mkdtemp()
    
    # Create dummy structure
    data_path = Path(temp_dir)
    (data_path / 'train').mkdir(exist_ok=True)
    (data_path / 'val').mkdir(exist_ok=True)
    
    # Create dummy annotation files
    train_data = [
        {
            'image_path': 'dummy_train.jpg',
            'caption': 'A test image',
            'label': 0
        }
    ]
    val_data = [
        {
            'image_path': 'dummy_val.jpg',
            'caption': 'Another test image',
            'label': 1
        }
    ]
    
    with open(data_path / 'train.json', 'w') as f:
        import json
        json.dump(train_data, f)
    
    with open(data_path / 'val.json', 'w') as f:
        import json
        json.dump(val_data, f)
    
    yield str(data_path)
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def cleanup_test_dirs():
    """Cleanup test directories after each test."""
    yield
    
    # Cleanup
    import shutil
    for dir_name in ['test_outputs', 'test_checkpoints', 'test_logs']:
        shutil.rmtree(dir_name, ignore_errors=True)


def pytest_configure(config):
    """Pytest configuration hook."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
