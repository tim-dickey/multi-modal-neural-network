"""Comprehensive unit tests for the training module.

Tests cover:
- Trainer class and training loop
- DeviceManager for device detection/configuration
- CheckpointManager for save/load operations
- TrainingState for state management
- LoggingManager for logging setup
- TrainingComponentsFactory for component creation
- Training defaults centralization
- Loss functions
- Optimizer and scheduler creation
- Gradient clipping and adaptive LR
"""

import logging
import types
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
import torch
import torch.nn as nn

from src.training.checkpoint_manager import CheckpointManager
from src.training.device_manager import DeviceManager
from src.training.losses import (
    ContrastiveLoss,
    CrossEntropyLoss,
    FocalLoss,
    MetaLoss,
    create_loss_function,
)
from src.training.optimizer import (
    AdaptiveLRController,
    GradientClipper,
    create_optimizer,
    create_scheduler,
    get_parameter_groups,
)
from src.training.trainer import Trainer
from src.training.training_defaults import (
    CHECKPOINT,
    DATA,
    LOSS,
    TRAINING,
)
from src.training.training_state import (
    LoggingManager,
    TrainingComponentsFactory,
    TrainingState,
)


# =============================================================================
# Fixtures
# =============================================================================


class DummyMetricsLogger:
    """Mock metrics logger for testing."""

    def __init__(self, log_dir: str, experiment_name: str) -> None:
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.logged_epochs: list[tuple[int, Dict[str, float], Dict[str, float] | None]] = []

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float] | None,
    ) -> None:
        self.logged_epochs.append((epoch, train_metrics, val_metrics))


class SimpleModel(nn.Module):
    """Simple test model producing logits for 3 classes."""

    def __init__(self, use_double_loop: bool = False):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(4, 3))
        self.use_double_loop = use_double_loop

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        result = {"logits": self.net(images)}
        if self.use_double_loop:
            result["meta_info"] = {"meta_loss": torch.tensor(0.1)}
        return result


def make_simple_trainer(tmp_path: Path, **overrides) -> Trainer:
    """Create a minimal trainer for testing without heavy initialization."""
    t = object.__new__(Trainer)

    # Minimal config with defaults
    t.config = {
        "training": {
            "max_epochs": 1,
            "inner_lr": 1e-3,
            "log_interval": 1,
            "gradient_clip": 0.0,
        },
        "paths": {"output_dir": str(tmp_path / "outputs")},
    }
    t.config.update(overrides.get("config", {}))

    # Device
    t.device = torch.device("cpu")

    # Logger
    t.logger = logging.getLogger("trainer")

    # Model
    t.model = SimpleModel(use_double_loop=overrides.get("use_double_loop", False))

    # Criterion and optimizer
    t.criterion = nn.CrossEntropyLoss()
    t.meta_criterion = None
    t.optimizer = torch.optim.SGD(t.model.parameters(), lr=1e-3)

    # Scheduler with state_dict and step
    t.scheduler = types.SimpleNamespace(
        state_dict=lambda: {},
        load_state_dict=lambda x: None,
        step=lambda: None,
    )

    # Metrics and wandb
    t.metrics_logger = DummyMetricsLogger(
        log_dir=str(tmp_path / "logs"), experiment_name="test"
    )
    t.wandb_logger = None

    # Training state
    t.training_state = TrainingState()
    t.current_epoch = 0
    t.global_step = 0
    t.best_val_loss = float("inf")

    # Paths
    t.output_dir = tmp_path / "outputs"
    t.checkpoint_dir = tmp_path / "checkpoints"
    t.output_dir.mkdir(parents=True, exist_ok=True)
    t.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint manager
    t.checkpoint_manager = CheckpointManager(
        checkpoint_dir=t.checkpoint_dir,
        logger=t.logger,
    )

    return t


# =============================================================================
# Training Defaults Tests
# =============================================================================


class TestTrainingDefaults:
    """Tests for centralized training defaults."""

    def test_training_defaults_are_frozen(self):
        """Ensure defaults cannot be modified."""
        with pytest.raises(Exception):  # FrozenInstanceError
            TRAINING.max_epochs = 100

    def test_training_defaults_values(self):
        """Verify expected default values."""
        assert TRAINING.max_epochs == 50
        assert TRAINING.warmup_steps == 1000
        assert TRAINING.learning_rate == 3e-4
        assert TRAINING.weight_decay == 0.01
        assert TRAINING.adam_beta1 == 0.9
        assert TRAINING.adam_beta2 == 0.999
        assert TRAINING.max_grad_norm == 1.0

    def test_data_defaults_values(self):
        """Verify data loading defaults."""
        assert DATA.batch_size == 32
        assert DATA.num_workers == 4
        assert DATA.pin_memory is True

    def test_loss_defaults_values(self):
        """Verify loss function defaults."""
        assert LOSS.label_smoothing == 0.0
        assert LOSS.contrastive_temperature == 0.07
        assert LOSS.focal_alpha == 0.25
        assert LOSS.meta_loss_weight == 0.1

    def test_checkpoint_defaults_values(self):
        """Verify checkpoint defaults."""
        assert CHECKPOINT.max_checkpoints is None
        assert CHECKPOINT.save_safetensors is True


# =============================================================================
# TrainingState Tests
# =============================================================================


class TestTrainingState:
    """Tests for TrainingState class."""

    def test_init_defaults(self):
        """Test default initialization."""
        state = TrainingState()
        assert state.current_epoch == 0
        assert state.global_step == 0
        assert state.best_val_loss == float("inf")

    def test_init_custom(self):
        """Test custom initialization."""
        state = TrainingState(current_epoch=5, global_step=1000, best_val_loss=0.5)
        assert state.current_epoch == 5
        assert state.global_step == 1000
        assert state.best_val_loss == 0.5

    def test_update_epoch(self):
        """Test epoch update."""
        state = TrainingState()
        state.update_epoch(10)
        assert state.current_epoch == 10

    def test_increment_step(self):
        """Test step increment."""
        state = TrainingState()
        new_step = state.increment_step()
        assert new_step == 1
        assert state.global_step == 1

        new_step = state.increment_step()
        assert new_step == 2

    def test_update_best_loss_improved(self):
        """Test best loss update when improved."""
        state = TrainingState(best_val_loss=1.0)
        is_best = state.update_best_loss(0.5)
        assert is_best is True
        assert state.best_val_loss == 0.5

    def test_update_best_loss_not_improved(self):
        """Test best loss update when not improved."""
        state = TrainingState(best_val_loss=0.5)
        is_best = state.update_best_loss(1.0)
        assert is_best is False
        assert state.best_val_loss == 0.5

    def test_restore_from_checkpoint(self):
        """Test state restoration from checkpoint."""
        state = TrainingState()
        checkpoint = {"epoch": 10, "global_step": 500, "best_val_loss": 0.25}
        state.restore_from_checkpoint(checkpoint)
        assert state.current_epoch == 11  # epoch + 1
        assert state.global_step == 500
        assert state.best_val_loss == 0.25

    def test_to_dict(self):
        """Test serialization to dict."""
        state = TrainingState(current_epoch=5, global_step=100, best_val_loss=0.3)
        d = state.to_dict()
        assert d == {"epoch": 5, "global_step": 100, "best_val_loss": 0.3}

    def test_repr(self):
        """Test string representation."""
        state = TrainingState(current_epoch=1, global_step=10, best_val_loss=0.5)
        repr_str = repr(state)
        assert "epoch=1" in repr_str
        assert "step=10" in repr_str


# =============================================================================
# DeviceManager Tests
# =============================================================================


class TestDeviceManager:
    """Tests for DeviceManager class."""

    def test_device_manager_cpu(self):
        """Test device manager with CPU."""
        config = {"hardware": {"device": "cpu"}}
        dm = DeviceManager(config)
        assert dm.device.type == "cpu"
        assert dm.is_cpu is True
        assert dm.is_cuda is False

    def test_device_override(self):
        """Test device override takes precedence."""
        config = {"hardware": {"device": "cuda"}}
        dm = DeviceManager(config, device_override="cpu")
        assert dm.device.type == "cpu"

    def test_device_properties(self):
        """Test device type properties."""
        config = {"hardware": {"device": "cpu"}}
        dm = DeviceManager(config)
        assert dm.device_type == "cpu"
        assert dm.is_cpu is True
        assert dm.is_cuda is False
        assert dm.is_mps is False

    def test_move_to_device(self):
        """Test tensor/module movement."""
        config = {"hardware": {"device": "cpu"}}
        dm = DeviceManager(config)
        tensor = torch.randn(2, 3)
        moved = dm.move_to_device(tensor)
        assert moved.device.type == "cpu"

    def test_move_batch_to_device(self):
        """Test batch movement."""
        config = {"hardware": {"device": "cpu"}}
        dm = DeviceManager(config)
        batch = {"images": torch.randn(2, 3), "labels": torch.tensor([0, 1])}
        moved = dm.move_batch_to_device(batch)
        assert all(v.device.type == "cpu" for v in moved.values())

    def test_create_grad_scaler_cpu(self):
        """Test GradScaler creation on CPU returns None."""
        config = {"hardware": {"device": "cpu"}}
        dm = DeviceManager(config)
        scaler = dm.create_grad_scaler()
        assert scaler is None

    def test_get_autocast_context(self):
        """Test autocast context creation."""
        config = {"hardware": {"device": "cpu"}}
        dm = DeviceManager(config)
        ctx = dm.get_autocast_context()
        assert ctx is not None

    def test_repr(self):
        """Test string representation."""
        config = {"hardware": {"device": "cpu"}}
        dm = DeviceManager(config)
        repr_str = repr(dm)
        assert "DeviceManager" in repr_str
        assert "cpu" in repr_str


# =============================================================================
# CheckpointManager Tests
# =============================================================================


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    def test_init_creates_directory(self, tmp_path):
        """Test that init creates checkpoint directory."""
        ckpt_dir = tmp_path / "checkpoints"
        cm = CheckpointManager(checkpoint_dir=ckpt_dir)
        assert ckpt_dir.exists()

    def test_save_to_explicit_path(self, tmp_path):
        """Test saving to explicit path."""
        cm = CheckpointManager(checkpoint_dir=tmp_path)
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = types.SimpleNamespace(state_dict=lambda: {})

        path = tmp_path / "explicit.pt"
        saved_path = cm.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            global_step=100,
            best_val_loss=0.5,
            config={"test": True},
            path=str(path),
        )
        assert saved_path == path
        assert path.exists()

    def test_save_creates_latest_and_best(self, tmp_path):
        """Test saving creates latest.pt and best.pt."""
        cm = CheckpointManager(checkpoint_dir=tmp_path, save_safetensors=False)
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = types.SimpleNamespace(state_dict=lambda: {})

        cm.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            global_step=100,
            best_val_loss=0.5,
            config={},
            is_best=True,
        )

        assert (tmp_path / "latest.pt").exists()
        assert (tmp_path / "best.pt").exists()

    def test_save_epoch(self, tmp_path):
        """Test save_epoch creates epoch-specific checkpoint."""
        cm = CheckpointManager(checkpoint_dir=tmp_path, save_safetensors=False)
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = types.SimpleNamespace(state_dict=lambda: {})

        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        saved_path = cm.save_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=5,
            global_step=500,
            best_val_loss=0.3,
            config={},
            output_dir=output_dir,
        )
        assert saved_path == output_dir / "checkpoint_0005.pt"
        assert saved_path.exists()

    def test_load_checkpoint(self, tmp_path):
        """Test loading checkpoint."""
        cm = CheckpointManager(checkpoint_dir=tmp_path, save_safetensors=False)
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = types.SimpleNamespace(state_dict=lambda: {})

        # Save
        path = tmp_path / "test.pt"
        cm.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=3,
            global_step=300,
            best_val_loss=0.4,
            config={"key": "value"},
            path=str(path),
        )

        # Load
        loaded = cm.load(str(path), device=torch.device("cpu"))
        assert loaded["epoch"] == 3
        assert loaded["global_step"] == 300
        assert loaded["best_val_loss"] == 0.4
        assert loaded["config"]["key"] == "value"

    def test_restore_training_state(self, tmp_path):
        """Test restoring training state from checkpoint."""
        cm = CheckpointManager(checkpoint_dir=tmp_path, save_safetensors=False)
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = types.SimpleNamespace(
            state_dict=lambda: {}, load_state_dict=lambda x: None
        )

        # Save
        path = tmp_path / "restore.pt"
        cm.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=5,
            global_step=500,
            best_val_loss=0.2,
            config={},
            path=str(path),
        )

        # Load and restore
        checkpoint = cm.load(str(path), device=torch.device("cpu"))
        new_model = SimpleModel()
        new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01)

        restored = cm.restore_training_state(
            checkpoint=checkpoint,
            model=new_model,
            optimizer=new_optimizer,
            scheduler=scheduler,
        )

        assert restored["epoch"] == 6  # epoch + 1
        assert restored["global_step"] == 500
        assert restored["best_val_loss"] == 0.2

    def test_get_latest_checkpoint(self, tmp_path):
        """Test getting latest checkpoint."""
        cm = CheckpointManager(checkpoint_dir=tmp_path, save_safetensors=False)

        # No checkpoint
        assert cm.get_latest_checkpoint() is None

        # Create latest.pt
        (tmp_path / "latest.pt").touch()
        assert cm.get_latest_checkpoint() == tmp_path / "latest.pt"

    def test_get_best_checkpoint(self, tmp_path):
        """Test getting best checkpoint."""
        cm = CheckpointManager(checkpoint_dir=tmp_path, save_safetensors=False)

        # No checkpoint
        assert cm.get_best_checkpoint() is None

        # Create best.pt
        (tmp_path / "best.pt").touch()
        assert cm.get_best_checkpoint() == tmp_path / "best.pt"

    def test_list_checkpoints(self, tmp_path):
        """Test listing all checkpoints."""
        cm = CheckpointManager(checkpoint_dir=tmp_path, save_safetensors=False)

        # Create checkpoints
        (tmp_path / "latest.pt").touch()
        (tmp_path / "best.pt").touch()
        (tmp_path / "checkpoint_0001.pt").touch()
        (tmp_path / "checkpoint_0002.pt").touch()

        checkpoints = cm.list_checkpoints()
        assert len(checkpoints) == 4

    def test_cleanup_old_checkpoints(self, tmp_path):
        """Test cleanup of old checkpoints."""
        cm = CheckpointManager(
            checkpoint_dir=tmp_path, max_checkpoints=2, save_safetensors=False
        )

        # Create multiple epoch checkpoints
        for i in range(5):
            (tmp_path / f"checkpoint_{i:04d}.pt").touch()

        cm._cleanup_old_checkpoints()

        # Should keep only the last 2
        remaining = list(tmp_path.glob("checkpoint_*.pt"))
        assert len(remaining) == 2
        assert (tmp_path / "checkpoint_0003.pt").exists()
        assert (tmp_path / "checkpoint_0004.pt").exists()


# =============================================================================
# Loss Function Tests
# =============================================================================


class TestLossFunctions:
    """Tests for loss functions."""

    def test_cross_entropy_loss(self):
        """Test CrossEntropyLoss."""
        loss_fn = CrossEntropyLoss()
        logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        loss = loss_fn(logits, labels)
        assert loss.dim() == 0  # scalar
        assert loss.item() >= 0

    def test_cross_entropy_with_smoothing(self):
        """Test CrossEntropyLoss with label smoothing."""
        loss_fn = CrossEntropyLoss(label_smoothing=0.1)
        logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        loss = loss_fn(logits, labels)
        assert loss.item() >= 0

    def test_contrastive_loss(self):
        """Test ContrastiveLoss."""
        loss_fn = ContrastiveLoss(temperature=0.07)
        image_features = torch.randn(8, 256)
        text_features = torch.randn(8, 256)
        loss = loss_fn(image_features, text_features)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_focal_loss(self):
        """Test FocalLoss."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        loss = loss_fn(logits, labels)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_meta_loss_without_meta_info(self):
        """Test MetaLoss when no meta info provided."""
        loss_fn = MetaLoss()
        task_loss = torch.tensor(1.5)
        loss = loss_fn(task_loss, None)
        assert loss.item() == 1.5

    def test_meta_loss_with_meta_info(self):
        """Test MetaLoss with meta info."""
        loss_fn = MetaLoss()
        task_loss = torch.tensor(1.0)
        meta_info = {"meta_loss": torch.tensor([0.5, 0.5])}
        loss = loss_fn(task_loss, meta_info)
        # 1.0 + 0.1 * 0.5 = 1.05
        assert loss.item() == pytest.approx(1.05, rel=1e-4)

    def test_create_loss_function_cross_entropy(self):
        """Test loss factory for cross entropy."""
        config = {"training": {"loss_type": "cross_entropy", "label_smoothing": 0.1}}
        loss_fn = create_loss_function(config)
        assert isinstance(loss_fn, CrossEntropyLoss)

    def test_create_loss_function_contrastive(self):
        """Test loss factory for contrastive loss."""
        config = {"training": {"loss_type": "contrastive", "temperature": 0.05}}
        loss_fn = create_loss_function(config)
        assert isinstance(loss_fn, ContrastiveLoss)
        assert loss_fn.temperature == 0.05

    def test_create_loss_function_focal(self):
        """Test loss factory for focal loss."""
        config = {
            "training": {"loss_type": "focal", "focal_alpha": 0.5, "focal_gamma": 1.5}
        }
        loss_fn = create_loss_function(config)
        assert isinstance(loss_fn, FocalLoss)
        assert loss_fn.alpha == 0.5
        assert loss_fn.gamma == 1.5

    def test_create_loss_function_unknown(self):
        """Test loss factory raises for unknown type."""
        config = {"training": {"loss_type": "unknown"}}
        with pytest.raises(ValueError, match="Unknown loss type"):
            create_loss_function(config)


# =============================================================================
# Optimizer Tests
# =============================================================================


class TestOptimizer:
    """Tests for optimizer creation."""

    def test_get_parameter_groups(self):
        """Test parameter group separation."""
        model = SimpleModel()
        groups = get_parameter_groups(model, weight_decay=0.01)
        assert len(groups) == 2
        assert groups[0]["weight_decay"] == 0.01
        assert groups[1]["weight_decay"] == 0.0

    def test_create_optimizer_adamw(self):
        """Test AdamW optimizer creation."""
        model = SimpleModel()
        config = {"training": {"optimizer": "adamw", "inner_lr": 1e-4}}
        optimizer = create_optimizer(model, config)
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_create_optimizer_adam(self):
        """Test Adam optimizer creation."""
        model = SimpleModel()
        config = {"training": {"optimizer": "adam", "inner_lr": 1e-4}}
        optimizer = create_optimizer(model, config)
        assert isinstance(optimizer, torch.optim.Adam)

    def test_create_optimizer_sgd(self):
        """Test SGD optimizer creation."""
        model = SimpleModel()
        config = {"training": {"optimizer": "sgd", "inner_lr": 1e-2}}
        optimizer = create_optimizer(model, config)
        assert isinstance(optimizer, torch.optim.SGD)

    def test_create_optimizer_unknown(self):
        """Test unknown optimizer raises error."""
        model = SimpleModel()
        config = {"training": {"optimizer": "unknown"}}
        with pytest.raises(ValueError, match="Unknown optimizer"):
            create_optimizer(model, config)

    def test_create_optimizer_invalid_lr(self):
        """Test invalid learning rate raises error."""
        model = SimpleModel()
        config = {"training": {"inner_lr": "invalid"}}
        with pytest.raises(ValueError, match="Invalid learning rate"):
            create_optimizer(model, config)


# =============================================================================
# Scheduler Tests
# =============================================================================


class TestScheduler:
    """Tests for scheduler creation."""

    def test_create_scheduler_cosine(self):
        """Test cosine scheduler creation."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        config = {"training": {"scheduler": "cosine", "max_epochs": 10}}
        scheduler, freq = create_scheduler(optimizer, config, steps_per_epoch=100)
        assert freq == "step"

    def test_create_scheduler_linear(self):
        """Test linear scheduler creation."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        config = {"training": {"scheduler": "linear", "max_epochs": 10}}
        scheduler, freq = create_scheduler(optimizer, config, steps_per_epoch=100)
        assert freq == "step"

    def test_create_scheduler_plateau(self):
        """Test plateau scheduler creation."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        config = {"training": {"scheduler": "plateau"}}
        scheduler, freq = create_scheduler(optimizer, config, steps_per_epoch=100)
        assert freq == "epoch"

    def test_create_scheduler_constant(self):
        """Test constant scheduler creation."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        config = {"training": {"scheduler": "constant"}}
        scheduler, freq = create_scheduler(optimizer, config, steps_per_epoch=100)
        assert freq == "step"

    def test_create_scheduler_unknown(self):
        """Test unknown scheduler raises error."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        config = {"training": {"scheduler": "unknown"}}
        with pytest.raises(ValueError, match="Unknown scheduler"):
            create_scheduler(optimizer, config, steps_per_epoch=100)


# =============================================================================
# GradientClipper Tests
# =============================================================================


class TestGradientClipper:
    """Tests for GradientClipper."""

    def test_gradient_clipper_default(self):
        """Test default initialization uses TRAINING defaults."""
        clipper = GradientClipper()
        assert clipper.max_norm == TRAINING.max_grad_norm
        assert clipper.norm_type == TRAINING.grad_norm_type

    def test_gradient_clipper_custom(self):
        """Test custom initialization."""
        clipper = GradientClipper(max_norm=0.5, norm_type=1.0)
        assert clipper.max_norm == 0.5
        assert clipper.norm_type == 1.0

    def test_gradient_clipper_call(self):
        """Test gradient clipping."""
        clipper = GradientClipper(max_norm=1.0)
        model = SimpleModel()
        loss = model(images=torch.randn(2, 1, 2, 2))["logits"].sum()
        loss.backward()
        grad_norm = clipper(model.parameters())
        assert grad_norm >= 0

    def test_gradient_clipper_no_grads(self):
        """Test clipper with no gradients."""
        clipper = GradientClipper()
        model = SimpleModel()
        grad_norm = clipper(model.parameters())
        assert grad_norm == 0.0


# =============================================================================
# AdaptiveLRController Tests
# =============================================================================


class TestAdaptiveLRController:
    """Tests for AdaptiveLRController."""

    def test_default_initialization(self):
        """Test default initialization uses TRAINING defaults."""
        controller = AdaptiveLRController()
        assert controller.base_lr == TRAINING.learning_rate
        assert controller.min_scale == TRAINING.adaptive_lr_min_scale
        assert controller.max_scale == TRAINING.adaptive_lr_max_scale

    def test_custom_initialization(self):
        """Test custom initialization."""
        controller = AdaptiveLRController(
            base_lr=1e-3, min_scale=0.5, max_scale=1.5
        )
        assert controller.base_lr == 1e-3
        assert controller.min_scale == 0.5
        assert controller.max_scale == 1.5

    def test_update_lr(self):
        """Test learning rate update."""
        controller = AdaptiveLRController(base_lr=1e-3, min_scale=0.1, max_scale=2.0)
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Scale of 0 -> min_scale * base_lr
        controller.update_lr(optimizer, torch.tensor(0.0))
        assert controller.get_current_lr(optimizer) == pytest.approx(1e-4)

        # Scale of 1 -> max_scale * base_lr
        controller.update_lr(optimizer, torch.tensor(1.0))
        assert controller.get_current_lr(optimizer) == pytest.approx(2e-3)


# =============================================================================
# TrainingComponentsFactory Tests
# =============================================================================


class TestTrainingComponentsFactory:
    """Tests for TrainingComponentsFactory."""

    def test_create_criterion(self):
        """Test criterion creation."""
        model = SimpleModel()
        config = {"training": {"loss_type": "cross_entropy"}}
        factory = TrainingComponentsFactory(
            model=model, config=config, train_loader=[]
        )
        criterion = factory.create_criterion()
        assert isinstance(criterion, CrossEntropyLoss)

    def test_create_meta_criterion_without_double_loop(self):
        """Test meta criterion is None without double loop."""
        model = SimpleModel(use_double_loop=False)
        factory = TrainingComponentsFactory(model=model, config={}, train_loader=[])
        meta_criterion = factory.create_meta_criterion()
        assert meta_criterion is None

    def test_create_meta_criterion_with_double_loop(self):
        """Test meta criterion creation with double loop."""
        model = SimpleModel(use_double_loop=True)
        factory = TrainingComponentsFactory(model=model, config={}, train_loader=[])
        meta_criterion = factory.create_meta_criterion()
        assert isinstance(meta_criterion, MetaLoss)

    def test_create_optimizer(self):
        """Test optimizer creation via factory."""
        model = SimpleModel()
        config = {"training": {"optimizer": "adamw"}}
        factory = TrainingComponentsFactory(
            model=model, config=config, train_loader=[]
        )
        optimizer = factory.create_optimizer()
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_create_scheduler(self):
        """Test scheduler creation via factory."""
        model = SimpleModel()
        config = {"training": {"scheduler": "cosine"}}
        factory = TrainingComponentsFactory(
            model=model, config=config, train_loader=[1, 2, 3]  # 3 batches
        )
        optimizer = factory.create_optimizer()
        scheduler, freq = factory.create_scheduler(optimizer)
        assert freq == "step"

    def test_create_gradient_clipper(self):
        """Test gradient clipper creation."""
        model = SimpleModel()
        config = {"training": {"max_grad_norm": 0.5}}
        factory = TrainingComponentsFactory(
            model=model, config=config, train_loader=[]
        )
        clipper = factory.create_gradient_clipper()
        assert clipper.max_norm == 0.5

    def test_create_all(self):
        """Test creating all components at once."""
        model = SimpleModel(use_double_loop=True)
        config = {
            "training": {
                "optimizer": "adamw",
                "scheduler": "cosine",
                "max_grad_norm": 1.0,
            }
        }
        factory = TrainingComponentsFactory(
            model=model, config=config, train_loader=[1, 2]
        )
        components = factory.create_all()

        assert "criterion" in components
        assert "meta_criterion" in components
        assert "optimizer" in components
        assert "scheduler" in components
        assert "scheduler_update_freq" in components
        assert "grad_clipper" in components
        assert "adaptive_lr" in components


# =============================================================================
# Trainer Tests
# =============================================================================


class TestTrainer:
    """Tests for Trainer class."""

    def test_normalize_batch_images_labels(self, tmp_path):
        """Test batch normalization with images/labels keys."""
        t = make_simple_trainer(tmp_path)
        batch = {"images": torch.randn(2, 1, 2, 2), "labels": torch.tensor([0, 1])}
        normalized = t._normalize_batch(batch)
        assert "images" in normalized
        assert "labels" in normalized

    def test_normalize_batch_image_label(self, tmp_path):
        """Test batch normalization with image/label keys (singular)."""
        t = make_simple_trainer(tmp_path)
        batch = {"image": torch.randn(2, 1, 2, 2), "label": torch.tensor([0, 1])}
        normalized = t._normalize_batch(batch)
        assert "images" in normalized
        assert "labels" in normalized

    def test_train_epoch(self, tmp_path):
        """Test training one epoch."""
        t = make_simple_trainer(tmp_path)
        batch = {"images": torch.randn(2, 1, 2, 2), "labels": torch.tensor([0, 1])}
        t.train_loader = [batch]
        t.val_loader = [batch]

        metrics = t.train_epoch(0)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_train_step(self, tmp_path):
        """Test single training step."""
        t = make_simple_trainer(tmp_path)
        batch = {"images": torch.randn(2, 1, 2, 2), "labels": torch.tensor([0, 1])}
        t.train_loader = [batch]

        loss, acc = t.train_step(batch)
        assert loss is not None
        assert acc is not None

    def test_validate(self, tmp_path):
        """Test validation."""
        t = make_simple_trainer(tmp_path)
        batch = {"images": torch.randn(2, 1, 2, 2), "labels": torch.tensor([0, 1])}
        t.val_loader = [batch]

        metrics = t.validate()
        assert isinstance(metrics["loss"], float)
        assert isinstance(metrics["accuracy"], float)

    def test_validate_no_loader(self, tmp_path):
        """Test validation with no loader returns zeros."""
        t = make_simple_trainer(tmp_path)
        t.val_loader = None

        metrics = t.validate()
        assert metrics["loss"] == 0.0
        assert metrics["accuracy"] == 0.0

    def test_save_checkpoint_to_path(self, tmp_path):
        """Test saving checkpoint to explicit path."""
        t = make_simple_trainer(tmp_path)
        path = str(tmp_path / "ckpt.pt")
        t.save_checkpoint(path=path, epoch=0, step=1, is_best=True)
        assert Path(path).exists()

    def test_train_epoch_empty_loader_raises(self, tmp_path):
        """Test that empty loader raises ValueError."""
        t = make_simple_trainer(tmp_path)
        t.train_loader = []

        with pytest.raises(ValueError, match="empty"):
            t.train_epoch(0)

    def test_train_epoch_with_gradient_clipping(self, tmp_path):
        """Test training with gradient clipping enabled."""
        t = make_simple_trainer(tmp_path)
        t.config["training"]["gradient_clip"] = 1.0
        batch = {"images": torch.randn(2, 1, 2, 2), "labels": torch.tensor([0, 1])}
        t.train_loader = [batch]

        metrics = t.train_epoch(0)
        assert "loss" in metrics


# =============================================================================
# LoggingManager Tests
# =============================================================================


class TestLoggingManager:
    """Tests for LoggingManager."""

    def test_init_creates_log_dir(self, tmp_path):
        """Test that init creates log directory."""
        config = {"paths": {"log_dir": str(tmp_path / "logs")}}
        lm = LoggingManager(config=config, output_dir=tmp_path)
        assert (tmp_path / "logs").exists()

    def test_log_methods(self, tmp_path):
        """Test logging methods don't raise."""
        config = {"logging": {"use_wandb": False}}
        lm = LoggingManager(config=config, output_dir=tmp_path)

        # These should not raise
        lm.log_info("Test message")
        lm.log_warning("Test warning")
        lm.log_epoch(1, {"loss": 0.5}, {"loss": 0.4})

    def test_finish(self, tmp_path):
        """Test finish method."""
        config = {"logging": {"use_wandb": False}}
        lm = LoggingManager(config=config, output_dir=tmp_path)
        lm.finish()  # Should not raise


# =============================================================================
# Integration Tests
# =============================================================================


class TestTrainerIntegration:
    """Integration tests for trainer workflow."""

    def test_full_training_loop(self, tmp_path):
        """Test complete training workflow."""
        t = make_simple_trainer(tmp_path)

        # Setup data
        batch = {"images": torch.randn(4, 1, 2, 2), "labels": torch.tensor([0, 1, 2, 0])}
        t.train_loader = [batch, batch]  # 2 batches
        t.val_loader = [batch]

        # Train for 2 epochs manually
        for epoch in range(2):
            train_metrics = t.train_epoch(epoch)
            val_metrics = t.validate()
            t.metrics_logger.log_epoch(epoch, train_metrics, val_metrics)

            if val_metrics["loss"] < t.best_val_loss:
                t.best_val_loss = val_metrics["loss"]
                t.save_checkpoint(is_best=True)

        assert t.metrics_logger.logged_epochs[-1][0] == 1

    def test_checkpoint_save_load_cycle(self, tmp_path):
        """Test saving and loading checkpoint restores state."""
        t = make_simple_trainer(tmp_path)
        batch = {"images": torch.randn(2, 1, 2, 2), "labels": torch.tensor([0, 1])}
        t.train_loader = [batch]

        # Train one epoch
        t.train_epoch(0)
        t.current_epoch = 5
        t.global_step = 100
        t.best_val_loss = 0.25

        # Save checkpoint
        ckpt_path = tmp_path / "checkpoints" / "test.pt"
        t.save_checkpoint(path=str(ckpt_path))

        # Create new trainer and load
        t2 = make_simple_trainer(tmp_path)
        t2.train_loader = [batch]
        t2.load_checkpoint(str(ckpt_path))

        assert t2.current_epoch == 6  # epoch + 1
        assert t2.global_step == 100
        assert t2.best_val_loss == 0.25

