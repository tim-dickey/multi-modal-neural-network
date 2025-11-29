"""Tests for training utilities."""

import pytest
import torch
import torch.nn as nn

from src.training.losses import (
    ContrastiveLoss,
    CrossEntropyLoss,
    FocalLoss,
    MetaLoss,
    MultiTaskLoss,
    create_loss_function,
)
from src.training.optimizer import (
    AdaptiveLRController,
    GradientClipper,
    create_optimizer,
    create_scheduler,
    get_parameter_groups,
)


class TestLossFunctions:
    """Tests for loss functions."""

    def test_cross_entropy_loss(self, batch_size, num_classes):
        """Test cross-entropy loss."""
        loss_fn = CrossEntropyLoss(label_smoothing=0.1)

        logits = torch.randn(batch_size, num_classes)
        labels = torch.randint(0, num_classes, (batch_size,))

        loss = loss_fn(logits, labels)

        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0

    def test_contrastive_loss(self, batch_size, hidden_dim):
        """Test contrastive loss."""
        loss_fn = ContrastiveLoss(temperature=0.07)

        image_features = torch.randn(batch_size, hidden_dim)
        text_features = torch.randn(batch_size, hidden_dim)

        loss = loss_fn(image_features, text_features)

        assert loss.ndim == 0
        assert loss.item() > 0

    def test_focal_loss(self, batch_size, num_classes):
        """Test focal loss."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

        logits = torch.randn(batch_size, num_classes)
        labels = torch.randint(0, num_classes, (batch_size,))

        loss = loss_fn(logits, labels)

        assert loss.ndim == 0
        assert loss.item() > 0

    def test_multi_task_loss(self, batch_size, num_classes):
        """Test multi-task loss."""
        task_names = ["task1", "task2"]
        loss_fns = {"task1": CrossEntropyLoss(), "task2": CrossEntropyLoss()}

        loss_fn = MultiTaskLoss(
            task_names=task_names, loss_fns=loss_fns, use_uncertainty_weighting=True
        )

        predictions = {
            "task1": torch.randn(batch_size, num_classes),
            "task2": torch.randn(batch_size, num_classes),
        }
        targets = {
            "task1": torch.randint(0, num_classes, (batch_size,)),
            "task2": torch.randint(0, num_classes, (batch_size,)),
        }

        losses = loss_fn(predictions, targets)

        assert "total_loss" in losses
        assert "task1_loss" in losses
        assert "task2_loss" in losses

    def test_meta_loss(self):
        """Test meta loss."""
        loss_fn = MetaLoss()

        task_loss = torch.tensor(0.5)
        meta_info = {"meta_loss": torch.tensor([0.1, 0.2])}

        combined_loss = loss_fn(task_loss, meta_info)

        assert combined_loss.item() > task_loss.item()

    def test_create_loss_function(self, model_config):
        """Test loss function factory."""
        model_config["training"]["loss_type"] = "cross_entropy"
        loss_fn = create_loss_function(model_config)

        assert isinstance(loss_fn, nn.Module)


class TestOptimizer:
    """Tests for optimizer configuration."""

    def test_create_optimizer(self, model_config):
        """Test optimizer creation."""
        from src.models import create_multi_modal_model

        model = create_multi_modal_model(model_config)
        optimizer = create_optimizer(model, model_config)

        assert optimizer is not None
        assert len(optimizer.param_groups) > 0

    def test_optimizer_parameter_groups(self):
        """Test parameter grouping for weight decay."""
        # Create simple model
        model = nn.Sequential(nn.Linear(10, 20), nn.LayerNorm(20), nn.Linear(20, 5))

        param_groups = get_parameter_groups(model, weight_decay=0.01)

        assert len(param_groups) == 2
        assert param_groups[0]["weight_decay"] == 0.01
        assert param_groups[1]["weight_decay"] == 0.0

    def test_create_scheduler(self, model_config):
        """Test scheduler creation."""
        from src.models import create_multi_modal_model

        model = create_multi_modal_model(model_config)
        optimizer = create_optimizer(model, model_config)

        scheduler, update_freq = create_scheduler(
            optimizer, model_config, steps_per_epoch=100
        )

        assert scheduler is not None
        assert update_freq in ["step", "epoch"]

    def test_scheduler_types(self, model_config):
        """Test different scheduler types."""
        from src.models import create_multi_modal_model

        model = create_multi_modal_model(model_config)
        optimizer = create_optimizer(model, model_config)

        for scheduler_type in ["cosine", "linear", "constant"]:
            model_config["training"]["scheduler"] = scheduler_type
            scheduler, _ = create_scheduler(
                optimizer, model_config, steps_per_epoch=100
            )
            assert scheduler is not None

    def test_gradient_clipper(self):
        """Test gradient clipping."""
        clipper = GradientClipper(max_norm=1.0)

        # Create dummy model and gradients
        model = nn.Linear(10, 5)
        x = torch.randn(2, 10)
        y = torch.randn(2, 5)

        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        loss.backward()

        # Clip gradients
        grad_norm = clipper(model.parameters())

        assert grad_norm >= 0

        # Check that gradients are clipped
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.norm().item()
                assert param_norm <= 1.0 + 1e-6  # Allow small numerical error

    def test_create_sgd_optimizer(self, model_config):
        """Test creating SGD optimizer."""
        from src.models import create_multi_modal_model

        model = create_multi_modal_model(model_config)
        model_config["training"]["optimizer"] = "sgd"
        model_config["training"]["momentum"] = 0.9
        
        optimizer = create_optimizer(model, model_config)
        
        assert optimizer is not None
        assert "SGD" in type(optimizer).__name__

    def test_create_scheduler_plateau(self, model_config):
        """Test plateau scheduler creation."""
        from src.models import create_multi_modal_model

        model = create_multi_modal_model(model_config)
        optimizer = create_optimizer(model, model_config)
        
        model_config["training"]["scheduler"] = "plateau"
        scheduler, update_freq = create_scheduler(
            optimizer, model_config, steps_per_epoch=100
        )
        
        assert scheduler is not None
        assert update_freq == "epoch"

    def test_create_scheduler_constant(self, model_config):
        """Test constant scheduler creation."""
        from src.models import create_multi_modal_model

        model = create_multi_modal_model(model_config)
        optimizer = create_optimizer(model, model_config)
        
        model_config["training"]["scheduler"] = "constant"
        scheduler, update_freq = create_scheduler(
            optimizer, model_config, steps_per_epoch=100
        )
        
        assert scheduler is not None
        assert update_freq == "step"

    def test_invalid_optimizer_type(self, model_config):
        """Test that invalid optimizer type raises error."""
        from src.models import create_multi_modal_model

        model = create_multi_modal_model(model_config)
        model_config["training"]["optimizer"] = "invalid_optimizer"
        
        with pytest.raises((ValueError, KeyError)):
            create_optimizer(model, model_config)


class TestAdaptiveLRController:
    """Tests for AdaptiveLRController."""

    def test_adaptive_lr_controller_init(self):
        """Test AdaptiveLRController initialization."""
        controller = AdaptiveLRController(base_lr=0.001, min_scale=0.1, max_scale=2.0)
        
        assert controller.base_lr == 0.001
        assert controller.min_scale == 0.1
        assert controller.max_scale == 2.0

    def test_adaptive_lr_controller_update_lr(self):
        """Test AdaptiveLRController update_lr."""
        params = torch.randn(10, requires_grad=True)
        optimizer = torch.optim.SGD([params], lr=0.1)
        controller = AdaptiveLRController(base_lr=0.1, min_scale=0.1, max_scale=2.0)

        # Update with scale factor of 0.5 (middle of range)
        controller.update_lr(optimizer, torch.tensor(0.5))
        
        # Expected: 0.1 + 0.5 * (2.0 - 0.1) = 1.05
        # new_lr = 0.1 * 1.05 = 0.105
        expected_scale = 0.1 + 0.5 * (2.0 - 0.1)
        expected_lr = 0.1 * expected_scale
        assert abs(optimizer.param_groups[0]["lr"] - expected_lr) < 1e-6

    def test_adaptive_lr_controller_get_current_lr(self):
        """Test AdaptiveLRController get_current_lr."""
        params = torch.randn(10, requires_grad=True)
        optimizer = torch.optim.SGD([params], lr=0.05)
        controller = AdaptiveLRController()
        
        current_lr = controller.get_current_lr(optimizer)
        assert current_lr == 0.05

    def test_adaptive_lr_controller_scale_range(self):
        """Test AdaptiveLRController respects scale range."""
        params = torch.randn(10, requires_grad=True)
        optimizer = torch.optim.SGD([params], lr=0.1)
        controller = AdaptiveLRController(base_lr=0.1, min_scale=0.5, max_scale=1.5)
        
        # Test min scale (lr_scale=0)
        controller.update_lr(optimizer, torch.tensor(0.0))
        assert abs(optimizer.param_groups[0]["lr"] - 0.1 * 0.5) < 1e-6
        
        # Test max scale (lr_scale=1)
        controller.update_lr(optimizer, torch.tensor(1.0))
        assert abs(optimizer.param_groups[0]["lr"] - 0.1 * 1.5) < 1e-6


class TestGradientAccumulation:
    """Tests for gradient accumulation."""

    def test_gradient_accumulation_equivalence(self, model_config):
        """Test that gradient accumulation gives equivalent gradients."""
        from src.models import create_multi_modal_model

        # Create two identical models
        model1 = create_multi_modal_model(model_config)
        model2 = create_multi_modal_model(model_config)

        # Copy weights
        model2.load_state_dict(model1.state_dict())

        # Set both models to eval mode to disable dropout and other stochastic layers
        model1.eval()
        model2.eval()

        # Generate sample data
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)
        input_ids = torch.randint(0, 30522, (batch_size, 128))
        attention_mask = torch.ones(batch_size, 128)
        labels = torch.randint(0, 10, (batch_size,))

        # Model 1: Single large batch
        outputs1 = model1(images, input_ids, attention_mask)
        loss1 = nn.functional.cross_entropy(outputs1["logits"], labels)
        loss1.backward()

        # Store gradients from model 1
        grads1 = [
            p.grad.clone() if p.grad is not None else None for p in model1.parameters()
        ]

        # Model 2: Accumulated small batches
        for i in range(2):
            start_idx = i * 2
            end_idx = start_idx + 2

            outputs2 = model2(
                images[start_idx:end_idx],
                input_ids[start_idx:end_idx],
                attention_mask[start_idx:end_idx],
            )
            loss2 = nn.functional.cross_entropy(
                outputs2["logits"], labels[start_idx:end_idx]
            )
            # Scale loss by number of accumulation steps
            (loss2 / 2).backward()

        # Check that gradients are equivalent
        # Since models are in eval mode (no dropout), gradients should be identical
        for p2, g1 in zip(model2.parameters(), grads1):
            if g1 is not None and p2.grad is not None:
                assert torch.allclose(
                    p2.grad, g1, rtol=1e-3, atol=1e-5
                ), f"Gradient mismatch: max diff = {(p2.grad - g1).abs().max()}"


class TestLearningRateScheduling:
    """Tests for learning rate scheduling."""

    def test_warmup_schedule(self, model_config):
        """Test warmup scheduling."""
        from src.models import create_multi_modal_model

        model = create_multi_modal_model(model_config)
        optimizer = create_optimizer(model, model_config)

        model_config["training"]["warmup_steps"] = 5
        scheduler, update_freq = create_scheduler(
            optimizer, model_config, steps_per_epoch=10
        )

        initial_lr = optimizer.param_groups[0]["lr"]

        # Learning rate should increase during warmup
        lrs = [initial_lr]
        for _ in range(5):
            if update_freq == "step":
                # Call optimizer.step() before scheduler.step() to match
                # PyTorch's recommended ordering and avoid warnings.
                # Some optimizers may require gradients; ignore in tests
                try:
                    optimizer.step()
                except (RuntimeError, ValueError):
                    # Expected: optimizer may require gradients to be set
                    pass
                scheduler.step()

            lrs.append(optimizer.param_groups[0]["lr"])

        # After warmup, LR should have increased
        assert lrs[-1] > lrs[0]

    def test_cosine_annealing(self, model_config):
        """Test cosine annealing schedule."""
        from src.models import create_multi_modal_model

        model = create_multi_modal_model(model_config)
        optimizer = create_optimizer(model, model_config)

        model_config["training"]["scheduler"] = "cosine"
        model_config["training"]["warmup_steps"] = 0
        scheduler, update_freq = create_scheduler(
            optimizer, model_config, steps_per_epoch=100
        )

        initial_lr = optimizer.param_groups[0]["lr"]

        # Step through many iterations
        lrs = [initial_lr]
        for _ in range(100):
            if update_freq == "step":
                try:
                    optimizer.step()
                except Exception:
                    pass
                scheduler.step()

            lrs.append(optimizer.param_groups[0]["lr"])

        # Learning rate should decrease
        assert lrs[-1] < lrs[0]


@pytest.mark.slow
class TestTrainingLoop:
    """Tests for training loop components."""

    def test_single_training_step(self, model_config):
        """Test a single training step."""
        from src.models import create_multi_modal_model
        from src.training.losses import create_loss_function

        model = create_multi_modal_model(model_config)
        optimizer = create_optimizer(model, model_config)
        criterion = create_loss_function(model_config)

        # Generate sample data
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        input_ids = torch.randint(0, 30522, (batch_size, 128))
        attention_mask = torch.ones(batch_size, 128)
        labels = torch.randint(0, 10, (batch_size,))

        # Training step
        model.train()
        optimizer.zero_grad()

        outputs = model(images, input_ids, attention_mask)
        loss = criterion(outputs["logits"], labels)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
