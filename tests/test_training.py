"""Tests for training utilities."""

import pytest
import torch
import torch.nn as nn
from src.training.losses import (
    CrossEntropyLoss,
    ContrastiveLoss,
    FocalLoss,
    MultiTaskLoss,
    MetaLoss,
    create_loss_function
)
from src.training.optimizer import (
    create_optimizer,
    create_scheduler,
    GradientClipper,
    get_parameter_groups
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
        task_names = ['task1', 'task2']
        loss_fns = {
            'task1': CrossEntropyLoss(),
            'task2': CrossEntropyLoss()
        }
        
        loss_fn = MultiTaskLoss(
            task_names=task_names,
            loss_fns=loss_fns,
            use_uncertainty_weighting=True
        )
        
        predictions = {
            'task1': torch.randn(batch_size, num_classes),
            'task2': torch.randn(batch_size, num_classes)
        }
        targets = {
            'task1': torch.randint(0, num_classes, (batch_size,)),
            'task2': torch.randint(0, num_classes, (batch_size,))
        }
        
        losses = loss_fn(predictions, targets)
        
        assert 'total_loss' in losses
        assert 'task1_loss' in losses
        assert 'task2_loss' in losses
        
    def test_meta_loss(self):
        """Test meta loss."""
        loss_fn = MetaLoss()
        
        task_loss = torch.tensor(0.5)
        meta_info = {
            'meta_loss': torch.tensor([0.1, 0.2])
        }
        
        combined_loss = loss_fn(task_loss, meta_info)
        
        assert combined_loss.item() > task_loss.item()
        
    def test_create_loss_function(self, model_config):
        """Test loss function factory."""
        model_config['training']['loss_type'] = 'cross_entropy'
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
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.LayerNorm(20),
            nn.Linear(20, 5)
        )
        
        param_groups = get_parameter_groups(model, weight_decay=0.01)
        
        assert len(param_groups) == 2
        assert param_groups[0]['weight_decay'] == 0.01
        assert param_groups[1]['weight_decay'] == 0.0
        
    def test_create_scheduler(self, model_config):
        """Test scheduler creation."""
        from src.models import create_multi_modal_model
        
        model = create_multi_modal_model(model_config)
        optimizer = create_optimizer(model, model_config)
        
        scheduler, update_freq = create_scheduler(
            optimizer,
            model_config,
            steps_per_epoch=100
        )
        
        assert scheduler is not None
        assert update_freq in ['step', 'epoch']
        
    def test_scheduler_types(self, model_config):
        """Test different scheduler types."""
        from src.models import create_multi_modal_model
        
        model = create_multi_modal_model(model_config)
        optimizer = create_optimizer(model, model_config)
        
        for scheduler_type in ['cosine', 'linear', 'constant']:
            model_config['training']['scheduler'] = scheduler_type
            scheduler, _ = create_scheduler(
                optimizer,
                model_config,
                steps_per_epoch=100
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


class TestGradientAccumulation:
    """Tests for gradient accumulation."""
    
    def test_gradient_accumulation_equivalence(self, model_config):
        """Test that gradient accumulation gives equivalent results."""
        from src.models import create_multi_modal_model
        
        # Create two identical models
        model1 = create_multi_modal_model(model_config)
        model2 = create_multi_modal_model(model_config)
        
        # Copy weights
        model2.load_state_dict(model1.state_dict())
        
        # Create optimizers
        optimizer1 = create_optimizer(model1, model_config)
        optimizer2 = create_optimizer(model2, model_config)
        
        # Generate sample data
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)
        input_ids = torch.randint(0, 30522, (batch_size, 128))
        attention_mask = torch.ones(batch_size, 128)
        labels = torch.randint(0, 10, (batch_size,))
        
        # Model 1: Single large batch
        optimizer1.zero_grad()
        outputs1 = model1(images, input_ids, attention_mask)
        loss1 = nn.functional.cross_entropy(outputs1['logits'], labels)
        loss1.backward()
        optimizer1.step()
        
        # Model 2: Accumulated small batches
        optimizer2.zero_grad()
        for i in range(2):
            start_idx = i * 2
            end_idx = start_idx + 2
            
            outputs2 = model2(
                images[start_idx:end_idx],
                input_ids[start_idx:end_idx],
                attention_mask[start_idx:end_idx]
            )
            loss2 = nn.functional.cross_entropy(
                outputs2['logits'],
                labels[start_idx:end_idx]
            )
            # Scale loss by number of accumulation steps
            (loss2 / 2).backward()
        
        optimizer2.step()
        
        # Check that models have similar parameters (within tolerance)
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            # They won't be exactly equal due to numerical differences
            # but should be close
            assert torch.allclose(p1, p2, rtol=1e-3, atol=1e-5)


class TestLearningRateScheduling:
    """Tests for learning rate scheduling."""
    
    def test_warmup_schedule(self, model_config):
        """Test warmup scheduling."""
        from src.models import create_multi_modal_model
        
        model = create_multi_modal_model(model_config)
        optimizer = create_optimizer(model, model_config)
        
        model_config['training']['warmup_steps'] = 5
        scheduler, update_freq = create_scheduler(
            optimizer,
            model_config,
            steps_per_epoch=10
        )
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Learning rate should increase during warmup
        lrs = [initial_lr]
        for _ in range(5):
            if update_freq == 'step':
                scheduler.step()
            lrs.append(optimizer.param_groups[0]['lr'])
        
        # After warmup, LR should have increased
        assert lrs[-1] > lrs[0]
        
    def test_cosine_annealing(self, model_config):
        """Test cosine annealing schedule."""
        from src.models import create_multi_modal_model
        
        model = create_multi_modal_model(model_config)
        optimizer = create_optimizer(model, model_config)
        
        model_config['training']['scheduler'] = 'cosine'
        model_config['training']['warmup_steps'] = 0
        scheduler, update_freq = create_scheduler(
            optimizer,
            model_config,
            steps_per_epoch=100
        )
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Step through many iterations
        lrs = [initial_lr]
        for _ in range(100):
            if update_freq == 'step':
                scheduler.step()
            lrs.append(optimizer.param_groups[0]['lr'])
        
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
        loss = criterion(outputs['logits'], labels)
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
