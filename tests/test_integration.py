"""Integration tests for the multi-modal neural network."""

import pytest
import torch


@pytest.mark.integration
class TestEndToEndTraining:
    """End-to-end integration tests."""

    def test_complete_training_step(self, model_config, temp_output_dir):
        """Test a complete training step with all components."""
        from src.models import create_multi_modal_model
        from src.training.losses import create_loss_function
        from src.training.optimizer import create_optimizer, create_scheduler

        # Create model
        model = create_multi_modal_model(model_config)

        # Create optimizer and scheduler
        optimizer = create_optimizer(model, model_config)
        scheduler, update_freq = create_scheduler(
            optimizer, model_config, steps_per_epoch=10
        )

        # Create loss function
        criterion = create_loss_function(model_config)

        # Generate sample batch
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

        assert loss.item() > 0

        loss.backward()
        optimizer.step()

        if update_freq == "step":
            scheduler.step()

    def test_training_epoch(self, model_config, temp_output_dir):
        """Test training for one epoch."""
        from src.models import create_multi_modal_model
        from src.training.trainer import Trainer

        # Update config for testing
        model_config["training"]["num_epochs"] = 1
        model_config["training"]["save_steps"] = 5
        model_config["output_dir"] = str(temp_output_dir)

        # Create model
        model = create_multi_modal_model(model_config)

        # Create dummy data loaders
        batch_size = 2
        num_batches = 3

        class DummyDataset:
            def __len__(self):
                return batch_size * num_batches

            def __getitem__(self, idx):
                return {
                    "image": torch.randn(3, 224, 224),
                    "input_ids": torch.randint(0, 30522, (128,)),
                    "attention_mask": torch.ones(128),
                    "label": idx % 10,
                }

        from torch.utils.data import DataLoader

        def collate_fn(batch):
            return {
                "image": torch.stack([b["image"] for b in batch]),
                "input_ids": torch.stack([b["input_ids"] for b in batch]),
                "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
                "label": torch.tensor([b["label"] for b in batch]),
            }

        train_loader = DataLoader(
            DummyDataset(), batch_size=batch_size, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            DummyDataset(), batch_size=batch_size, collate_fn=collate_fn
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=model_config,
        )

        # Train for one epoch
        trainer.train()

        # Check that checkpoints were saved
        checkpoints = list(temp_output_dir.glob("checkpoint_*.pt"))
        assert len(checkpoints) > 0

    def test_checkpoint_save_load(self, model_config, temp_output_dir):
        """Test checkpoint saving and loading."""
        from src.models import create_multi_modal_model
        from src.training.trainer import Trainer

        # Create model
        model = create_multi_modal_model(model_config)

        # Save initial state
        initial_params = {
            name: param.clone() for name, param in model.named_parameters()
        }

        # Create dummy data loader
        class DummyDataset:
            def __len__(self):
                return 4

            def __getitem__(self, idx):
                return {
                    "image": torch.randn(3, 224, 224),
                    "input_ids": torch.randint(0, 30522, (128,)),
                    "attention_mask": torch.ones(128),
                    "label": idx % 10,
                }

        from torch.utils.data import DataLoader

        def collate_fn(batch):
            return {
                "image": torch.stack([b["image"] for b in batch]),
                "input_ids": torch.stack([b["input_ids"] for b in batch]),
                "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
                "label": torch.tensor([b["label"] for b in batch]),
            }

        train_loader = DataLoader(DummyDataset(), batch_size=2, collate_fn=collate_fn)

        model_config["output_dir"] = str(temp_output_dir)

        trainer = Trainer(
            model=model, train_loader=train_loader, val_loader=None, config=model_config
        )

        # Save checkpoint
        checkpoint_path = temp_output_dir / "test_checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path), epoch=0, step=0)

        assert checkpoint_path.exists()

        # Modify model parameters
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)

        # Load checkpoint
        trainer.load_checkpoint(str(checkpoint_path))

        # Check that parameters are restored
        for name, param in model.named_parameters():
            assert torch.allclose(param, initial_params[name], atol=1e-5)

    def test_inference(self, model_config):
        """Test model inference."""
        from src.models import create_multi_modal_model

        model = create_multi_modal_model(model_config)
        model.eval()

        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        input_ids = torch.randint(0, 30522, (batch_size, 128))
        attention_mask = torch.ones(batch_size, 128)

        with torch.no_grad():
            outputs = model(images, input_ids, attention_mask)

        assert "logits" in outputs
        assert outputs["logits"].shape[0] == batch_size


@pytest.mark.integration
@pytest.mark.gpu
class TestGPUTraining:
    """Integration tests for GPU training."""

    def test_training_on_gpu(self, model_config):
        """Test training on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from src.models import create_multi_modal_model
        from src.training.losses import create_loss_function
        from src.training.optimizer import create_optimizer

        device = torch.device("cuda")

        # Create model on GPU
        model = create_multi_modal_model(model_config)
        model = model.to(device)

        optimizer = create_optimizer(model, model_config)
        criterion = create_loss_function(model_config)

        # Generate sample batch on GPU
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        input_ids = torch.randint(0, 30522, (batch_size, 128)).to(device)
        attention_mask = torch.ones(batch_size, 128).to(device)
        labels = torch.randint(0, 10, (batch_size,)).to(device)

        # Training step
        model.train()
        optimizer.zero_grad()

        outputs = model(images, input_ids, attention_mask)
        loss = criterion(outputs["logits"], labels)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0

    def test_mixed_precision_training(self, model_config):
        """Test mixed precision training."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from src.models import create_multi_modal_model
        from src.training.losses import create_loss_function
        from src.training.optimizer import create_optimizer

        device = torch.device("cuda")

        # Enable mixed precision
        model_config["training"]["mixed_precision"] = True

        model = create_multi_modal_model(model_config)
        model = model.to(device)

        optimizer = create_optimizer(model, model_config)
        criterion = create_loss_function(model_config)

        # Create gradient scaler
        scaler = torch.cuda.amp.GradScaler()

        # Generate sample batch
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        input_ids = torch.randint(0, 30522, (batch_size, 128)).to(device)
        attention_mask = torch.ones(batch_size, 128).to(device)
        labels = torch.randint(0, 10, (batch_size,)).to(device)

        # Training step with mixed precision
        model.train()
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs["logits"], labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        assert loss.item() > 0


@pytest.mark.integration
class TestMultiTaskTraining:
    """Tests for multi-task training."""

    def test_multi_task_forward(self, model_config):
        """Test multi-task forward pass."""
        from src.models import create_multi_modal_model

        # Configure for multi-task
        model_config["model"]["head_type"] = "multi_task"
        model_config["model"]["task_configs"] = {
            "classification": {"num_classes": 10},
            "regression": {"output_dim": 1},
        }

        model = create_multi_modal_model(model_config)
        model.eval()

        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        input_ids = torch.randint(0, 30522, (batch_size, 128))
        attention_mask = torch.ones(batch_size, 128)

        with torch.no_grad():
            outputs = model(images, input_ids, attention_mask)

        assert "classification" in outputs
        assert "regression" in outputs

    def test_multi_task_training_step(self, model_config):
        """Test training step for multi-task learning."""
        from src.models import create_multi_modal_model
        from src.training.losses import CrossEntropyLoss, MultiTaskLoss
        from src.training.optimizer import create_optimizer

        # Configure for multi-task
        model_config["model"]["head_type"] = "multi_task"
        model_config["model"]["task_configs"] = {
            "task1": {"num_classes": 10},
            "task2": {"num_classes": 5},
        }

        model = create_multi_modal_model(model_config)
        optimizer = create_optimizer(model, model_config)

        # Create multi-task loss
        loss_fn = MultiTaskLoss(
            task_names=["task1", "task2"],
            loss_fns={"task1": CrossEntropyLoss(), "task2": CrossEntropyLoss()},
        )

        # Generate sample batch
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        input_ids = torch.randint(0, 30522, (batch_size, 128))
        attention_mask = torch.ones(batch_size, 128)
        targets = {
            "task1": torch.randint(0, 10, (batch_size,)),
            "task2": torch.randint(0, 5, (batch_size,)),
        }

        # Training step
        model.train()
        optimizer.zero_grad()

        outputs = model(images, input_ids, attention_mask)
        losses = loss_fn(outputs, targets)

        total_loss = losses["total_loss"]
        total_loss.backward()
        optimizer.step()

        assert total_loss.item() > 0
        assert "task1_loss" in losses
        assert "task2_loss" in losses


@pytest.mark.integration
class TestDoubleLoopLearning:
    """Tests for double-loop learning."""

    def test_double_loop_training_step(self, model_config):
        """Test training step with double-loop learning."""
        from src.models import create_multi_modal_model
        from src.training.losses import MetaLoss, create_loss_function
        from src.training.optimizer import create_optimizer

        # Enable double-loop learning
        model_config["model"]["use_double_loop"] = True

        model = create_multi_modal_model(model_config)
        optimizer = create_optimizer(model, model_config)

        task_criterion = create_loss_function(model_config)
        meta_criterion = MetaLoss()

        # Generate sample batch
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        input_ids = torch.randint(0, 30522, (batch_size, 128))
        attention_mask = torch.ones(batch_size, 128)
        labels = torch.randint(0, 10, (batch_size,))

        # Training step
        model.train()
        optimizer.zero_grad()

        outputs = model(images, input_ids, attention_mask)
        task_loss = task_criterion(outputs["logits"], labels)

        # Combine with meta loss if available
        if "meta_info" in outputs:
            total_loss = meta_criterion(task_loss, outputs["meta_info"])
        else:
            total_loss = task_loss

        total_loss.backward()
        optimizer.step()

        assert total_loss.item() > 0
