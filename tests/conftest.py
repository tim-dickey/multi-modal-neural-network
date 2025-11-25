"""Pytest configuration and shared fixtures for testing."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock
import asyncio

import numpy as np
import pytest
import torch
import yaml
from PIL import Image


# Device and hardware fixtures
@pytest.fixture
def device():
    """Get available device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def gpu_available():
    """Check if GPU is available."""
    return torch.cuda.is_available()


# Basic data fixtures
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


# Sample data fixtures
@pytest.fixture
def sample_images(batch_size, img_size):
    """Generate sample image tensors."""
    return torch.randn(batch_size, 3, img_size, img_size)


@pytest.fixture
def sample_text_inputs(batch_size, seq_length):
    """Generate sample text input tensors."""
    input_ids = torch.randint(0, 30522, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


@pytest.fixture
def sample_labels(batch_size, num_classes):
    """Generate sample labels."""
    return torch.randint(0, num_classes, (batch_size,))


@pytest.fixture
def sample_embeddings(batch_size, hidden_dim):
    """Generate sample embedding tensors."""
    return torch.randn(batch_size, hidden_dim)


# Model configuration fixtures
@pytest.fixture
def vision_encoder_config(img_size, hidden_dim):
    """Configuration for vision encoder."""
    return {
        "img_size": img_size,
        "patch_size": 16,
        "in_channels": 3,
        "hidden_dim": hidden_dim,
        "num_layers": 4,  # Smaller for testing
        "num_heads": 4,
        "mlp_ratio": 4.0,
        "dropout": 0.1,
        "use_cls_token": True,
    }


@pytest.fixture
def text_encoder_config(hidden_dim, seq_length):
    """Configuration for text encoder."""
    return {
        "vocab_size": 30522,
        "hidden_dim": hidden_dim,
        "num_layers": 4,  # Smaller for testing
        "num_heads": 4,
        "max_seq_length": seq_length,
        "mlp_ratio": 4.0,
        "dropout": 0.1,
        "use_cls_token": True,
    }


@pytest.fixture
def fusion_config(hidden_dim):
    """Configuration for fusion layer."""
    return {
        "type": "early",
        "hidden_dim": hidden_dim,
        "num_layers": 2,  # Smaller for testing
        "num_heads": 4,
        "mlp_ratio": 4.0,
        "dropout": 0.1,
    }


@pytest.fixture
def double_loop_config(hidden_dim):
    """Configuration for double-loop controller."""
    return {
        "model_hidden_dim": hidden_dim,
        "controller_type": "lstm",
        "hidden_dim": 128,
        "update_frequency": 10,
        "meta_lr": 1e-5,
    }


@pytest.fixture
def head_config(hidden_dim, num_classes):
    """Configuration for task head."""
    return {
        "type": "classification",
        "hidden_dim": hidden_dim,
        "num_classes": num_classes,
        "dropout": 0.1,
    }


@pytest.fixture
def model_config(
    vision_encoder_config,
    text_encoder_config,
    fusion_config,
    double_loop_config,
    head_config,
):
    """Complete model configuration."""
    return {
        "model": {
            "vision_encoder": vision_encoder_config,
            "text_encoder": text_encoder_config,
            "fusion": fusion_config,
            "double_loop": double_loop_config,
            "heads": head_config,
            "use_double_loop": True,
        },
        "training": {
            "max_epochs": 2,
            "inner_lr": 1e-3,
            "micro_batch_size": 2,
            "gradient_accumulation": 1,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "mixed_precision": None,  # Disable for testing
            "gradient_checkpointing": False,
            "warmup_steps": 5,
        },
        "data": {
            "train_dataset": "dummy",
            "val_dataset": "dummy",
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
        },
        "logging": {
            "project": "test-project",
            "experiment": "test-experiment",
            "log_every": 1,
            "save_every": 100,
            "use_wandb": False,
        },
        "paths": {
            "output_dir": "./test_outputs",
            "checkpoint_dir": "./test_checkpoints",
            "log_dir": "./test_logs",
        },
        "hardware": {"device": "cpu", "max_memory": "2GB"},
    }


# API Integration fixtures
@pytest.fixture
def api_config():
    """Default API configuration for testing."""
    return {
        "timeout": 30,
        "max_retries": 3,
        "retry_delay": 0.1,
        "cache_dir": "./test_cache",
    }


@pytest.fixture
def mock_api_response():
    """Mock API response fixture."""
    from src.integrations.base import APIResponse
    return APIResponse(
        success=True,
        data={"result": "test"},
        metadata={"query": "test query"}
    )


@pytest.fixture
def mock_failed_api_response():
    """Mock failed API response fixture."""
    from src.integrations.base import APIResponse
    return APIResponse(
        success=False,
        data=None,
        error="Mock API error"
    )


@pytest.fixture
def mock_wolfram_client():
    """Mock Wolfram Alpha client."""
    client = Mock()
    mock_result = Mock()
    mock_result.success = True
    mock_result.pods = []

    client.query.return_value = mock_result
    return client


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Mock OpenAI response"

    client.chat.completions.create.return_value = mock_response
    return client


@pytest.fixture
def mock_requests_session():
    """Mock requests session for HTTP testing."""
    session = Mock()
    mock_response = Mock()
    mock_response.json.return_value = {"result": "success"}
    mock_response.status_code = 200

    session.get.return_value = mock_response
    session.post.return_value = mock_response
    return session


# File and directory fixtures
@pytest.fixture
def temp_config_file(model_config):
    """Create a temporary config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
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
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
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
    (data_path / "train").mkdir(exist_ok=True)
    (data_path / "val").mkdir(exist_ok=True)

    # Create dummy annotation files
    train_data = [
        {"image_path": "dummy_train.jpg", "caption": "A test image", "label": 0}
    ]
    val_data = [
        {"image_path": "dummy_val.jpg", "caption": "Another test image", "label": 1}
    ]

    with open(data_path / "train.json", "w") as f:
        import json
        json.dump(train_data, f)

    with open(data_path / "val.json", "w") as f:
        import json
        json.dump(val_data, f)

    yield str(data_path)

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory for tests."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory for tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory for tests."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


# Async testing fixtures
@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Mock utilities
@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables."""
    def _mock_env_vars(**kwargs):
        for key, value in kwargs.items():
            monkeypatch.setenv(key, value)
    return _mock_env_vars


@pytest.fixture
def mock_file_operations(monkeypatch):
    """Mock file operations."""
    def _mock_open(content):
        mock_file = Mock()
        mock_file.read.return_value = content
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=None)
        return mock_file

    def _patch_open(content):
        monkeypatch.setattr("builtins.open", lambda *args, **kwargs: _mock_open(content))

    return _patch_open


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_test_dirs():
    """Cleanup test directories after each test."""
    yield

    # Cleanup
    import shutil

    for dir_name in ["test_outputs", "test_checkpoints", "test_logs", "test_cache"]:
        shutil.rmtree(dir_name, ignore_errors=True)


# Pytest configuration
def pytest_configure(config):
    """Pytest configuration hook."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers", "api: marks API integration tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks performance benchmark tests"
    )
    config.addinivalue_line(
        "markers", "async: marks async tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test properties."""
    for item in items:
        # Auto-mark GPU tests
        if "gpu" in item.keywords or "cuda" in str(item.fspath):
            item.add_marker(pytest.mark.gpu)

        # Auto-mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Auto-mark API tests
        if "api" in str(item.fspath) or "integration" in str(item.fspath):
            item.add_marker(pytest.mark.api)


@pytest.fixture(scope="session")
def session_temp_dir(tmp_path_factory):
    """Create a session-wide temporary directory."""
    return tmp_path_factory.mktemp("session_temp")


@pytest.fixture
def device():
    """Get available device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    return {"input_ids": input_ids, "attention_mask": attention_mask}


@pytest.fixture
def sample_labels(batch_size, num_classes):
    """Generate sample labels."""
    return torch.randint(0, num_classes, (batch_size,))


@pytest.fixture
def vision_encoder_config(img_size, hidden_dim):
    """Configuration for vision encoder."""
    return {
        "img_size": img_size,
        "patch_size": 16,
        "in_channels": 3,
        "hidden_dim": hidden_dim,
        "num_layers": 4,  # Smaller for testing
        "num_heads": 4,
        "mlp_ratio": 4.0,
        "dropout": 0.1,
        "use_cls_token": True,
    }


@pytest.fixture
def text_encoder_config(hidden_dim, seq_length):
    """Configuration for text encoder."""
    return {
        "vocab_size": 30522,
        "hidden_dim": hidden_dim,
        "num_layers": 4,  # Smaller for testing
        "num_heads": 4,
        "max_seq_length": seq_length,
        "mlp_ratio": 4.0,
        "dropout": 0.1,
        "use_cls_token": True,
    }


@pytest.fixture
def fusion_config(hidden_dim):
    """Configuration for fusion layer."""
    return {
        "type": "early",
        "hidden_dim": hidden_dim,
        "num_layers": 2,  # Smaller for testing
        "num_heads": 4,
        "mlp_ratio": 4.0,
        "dropout": 0.1,
    }


@pytest.fixture
def double_loop_config(hidden_dim):
    """Configuration for double-loop controller."""
    return {
        "model_hidden_dim": hidden_dim,
        "controller_type": "lstm",
        "hidden_dim": 128,
        "update_frequency": 10,
        "meta_lr": 1e-5,
    }


@pytest.fixture
def head_config(hidden_dim, num_classes):
    """Configuration for task head."""
    return {
        "type": "classification",
        "hidden_dim": hidden_dim,
        "num_classes": num_classes,
        "dropout": 0.1,
    }


@pytest.fixture
def model_config(
    vision_encoder_config,
    text_encoder_config,
    fusion_config,
    double_loop_config,
    head_config,
):
    """Complete model configuration."""
    return {
        "model": {
            "vision_encoder": vision_encoder_config,
            "text_encoder": text_encoder_config,
            "fusion": fusion_config,
            "double_loop": double_loop_config,
            "heads": head_config,
            "use_double_loop": True,
        },
        "training": {
            "max_epochs": 2,
            "inner_lr": 1e-3,
            "micro_batch_size": 2,
            "gradient_accumulation": 1,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "mixed_precision": None,  # Disable for testing
            "gradient_checkpointing": False,
            "warmup_steps": 5,
        },
        "data": {
            "train_dataset": "dummy",
            "val_dataset": "dummy",
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
        },
        "logging": {
            "project": "test-project",
            "experiment": "test-experiment",
            "log_every": 1,
            "save_every": 100,
            "use_wandb": False,
        },
        "paths": {
            "output_dir": "./test_outputs",
            "checkpoint_dir": "./test_checkpoints",
            "log_dir": "./test_logs",
        },
        "hardware": {"device": "cpu", "max_memory": "2GB"},
    }


@pytest.fixture
def temp_config_file(model_config):
    """Create a temporary config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
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
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
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
    (data_path / "train").mkdir(exist_ok=True)
    (data_path / "val").mkdir(exist_ok=True)

    # Create dummy annotation files
    train_data = [
        {"image_path": "dummy_train.jpg", "caption": "A test image", "label": 0}
    ]
    val_data = [
        {"image_path": "dummy_val.jpg", "caption": "Another test image", "label": 1}
    ]

    with open(data_path / "train.json", "w") as f:
        import json

        json.dump(train_data, f)

    with open(data_path / "val.json", "w") as f:
        import json

        json.dump(val_data, f)

    yield str(data_path)

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory for tests."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory for tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture(autouse=True)
def cleanup_test_dirs():
    """Cleanup test directories after each test."""
    yield

    # Cleanup
    import shutil

    for dir_name in ["test_outputs", "test_checkpoints", "test_logs"]:
        shutil.rmtree(dir_name, ignore_errors=True)


def pytest_configure(config):
    """Pytest configuration hook."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")
