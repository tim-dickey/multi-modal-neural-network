"""Integration tests for the multi-modal neural network."""

import json
from unittest.mock import Mock, patch

import pytest
import torch

from src.data.dataset import (
    MultiModalDataset,
    create_dataloader,
    create_data_loaders,
    get_transforms,
)
from src.data.selector import build_dataloaders
from src.integrations.base import APIResponse, KnowledgeInjector
from src.integrations.knowledge_injection import (
    AdditiveInjection,
    AttentionInjection,
    KnowledgeInjectionManager,
    MultiplicativeInjection,
    create_injection_strategy,
)
from src.utils.config import (
    ConfigNamespace,
    load_config,
    merge_configs,
    save_config,
    validate_config,
)


# ============================================================================
# DATA PIPELINE INTEGRATION TESTS
# ============================================================================


@pytest.mark.integration
class TestDataPipelineIntegration:
    """Integration tests for data pipeline with training."""

    def test_multimodal_dataset_with_model(self, model_config, temp_data_dir):
        """Test MultiModalDataset produces batches compatible with model."""
        from src.models import create_multi_modal_model

        # Create annotation file
        annotations = [
            {"image_path": "img1.jpg", "caption": "A test image", "label": 0},
            {"image_path": "img2.jpg", "caption": "Another test", "label": 1},
            {"image_path": "img3.jpg", "caption": "Third image", "label": 2},
            {"image_path": "img4.jpg", "caption": "Fourth one", "label": 3},
        ]
        with open(temp_data_dir / "annotations.json", "w") as f:
            json.dump(annotations, f)

        # Create dataset
        dataset = MultiModalDataset(
            data_path=str(temp_data_dir),
            split="train",
            img_size=224,
            max_text_length=128,
            augment=True,
        )

        # Create dataloader
        loader = create_dataloader(
            dataset, batch_size=2, num_workers=0, shuffle=True, drop_last=True
        )

        # Get a batch
        batch = next(iter(loader))

        # Create model and run forward pass
        model = create_multi_modal_model(model_config)
        model.eval()

        with torch.no_grad():
            outputs = model(
                batch["image"], batch["input_ids"], batch["attention_mask"]
            )

        assert "logits" in outputs
        assert outputs["logits"].shape[0] == 2

    def test_dataset_transforms_consistency(self, model_config, temp_data_dir):
        """Test that train/val transforms produce consistent tensor shapes."""
        # Create annotation file
        annotations = [
            {"image_path": "img.jpg", "caption": "Test", "label": 0}
            for _ in range(10)
        ]
        with open(temp_data_dir / "annotations.json", "w") as f:
            json.dump(annotations, f)

        train_dataset = MultiModalDataset(
            str(temp_data_dir), split="train", img_size=224, augment=True
        )
        val_dataset = MultiModalDataset(
            str(temp_data_dir), split="val", img_size=224, augment=False
        )

        train_sample = train_dataset[0]
        val_sample = val_dataset[0]

        # Both should produce same tensor shapes
        assert train_sample["image"].shape == val_sample["image"].shape
        assert train_sample["input_ids"].shape == val_sample["input_ids"].shape

    def test_get_transforms_with_config(self):
        """Test get_transforms uses config values correctly."""
        config = {
            "data": {
                "image_size": 256,
                "augmentation": {
                    "random_crop": True,
                    "random_flip": True,
                    "color_jitter": False,
                },
            }
        }

        train_transform = get_transforms(config, is_train=True)
        val_transform = get_transforms(config, is_train=False)

        # Apply transforms to dummy image
        from PIL import Image

        dummy_img = Image.new("RGB", (300, 300), color="red")

        train_tensor = train_transform(dummy_img)
        val_tensor = val_transform(dummy_img)

        # Both should be 256x256
        assert train_tensor.shape == (3, 256, 256)
        assert val_tensor.shape == (3, 256, 256)

    def test_create_data_loaders_pair(self, temp_data_dir):
        """Test create_data_loaders returns working train/val loaders."""
        # Create annotation files
        for split in ["train", "val"]:
            annotations = [
                {"image_path": f"{split}_{i}.jpg", "caption": f"{split} {i}", "label": i}
                for i in range(4)
            ]
            with open(temp_data_dir / f"{split}.json", "w") as f:
                json.dump(annotations, f)

        train_ds = MultiModalDataset(str(temp_data_dir), split="train", augment=True)
        val_ds = MultiModalDataset(str(temp_data_dir), split="val", augment=False)

        train_loader, val_loader = create_data_loaders(
            train_ds, val_ds, batch_size=2, num_workers=0
        )

        # Get batches from both
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))

        assert train_batch["image"].shape[0] == 2
        assert val_batch["image"].shape[0] == 2

    def test_build_dataloaders_from_config(self, temp_data_dir):
        """Test build_dataloaders constructs loaders from config."""
        # Create annotation file for multimodal dataset
        annotations = [
            {"image_path": f"img_{i}.jpg", "caption": f"Caption {i}", "label": i % 3}
            for i in range(20)
        ]
        with open(temp_data_dir / "annotations.json", "w") as f:
            json.dump(annotations, f)

        config = {
            "data": {
                "batch_size": 4,
                "num_workers": 0,
                "pin_memory": False,
                "shuffle_train": True,
                "datasets": [
                    {
                        "name": "test_dataset",
                        "type": "multimodal",
                        "data_path": str(temp_data_dir),
                        "splits": {"train": 0.8, "val": 0.2},
                        "enabled": True,
                    }
                ],
            }
        }

        train_loader, val_loader, test_loader = build_dataloaders(config)

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is None  # No test split defined

        train_batch = next(iter(train_loader))
        assert "image" in train_batch

    def test_dataset_with_trainer(self, model_config, temp_data_dir, temp_output_dir):
        """Test dataset integrates with Trainer class."""
        from src.models import create_multi_modal_model
        from src.training.trainer import Trainer

        # Create annotations
        annotations = [
            {"image_path": f"img_{i}.jpg", "caption": f"Caption {i}", "label": i % 10}
            for i in range(8)
        ]
        with open(temp_data_dir / "annotations.json", "w") as f:
            json.dump(annotations, f)

        # Create datasets with max_text_length matching model config
        max_seq = model_config["model"]["text_encoder"]["max_seq_length"]
        train_ds = MultiModalDataset(
            str(temp_data_dir), split="train", augment=True, max_text_length=max_seq
        )
        val_ds = MultiModalDataset(
            str(temp_data_dir), split="val", augment=False, max_text_length=max_seq
        )

        train_loader, val_loader = create_data_loaders(
            train_ds, val_ds, batch_size=2, num_workers=0
        )

        # Create model and trainer
        model = create_multi_modal_model(model_config)
        model_config["output_dir"] = str(temp_output_dir)
        model_config["training"]["num_epochs"] = 1

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=model_config,
        )

        # Should be able to run training
        trainer.train()


# ============================================================================
# CONFIG TO TRAINING INTEGRATION TESTS
# ============================================================================


@pytest.mark.integration
class TestConfigToTrainingIntegration:
    """Integration tests for config loading through training."""

    def test_load_config_and_create_model(self, temp_output_dir):
        """Test loading config file and creating model from it."""
        from src.models import create_multi_modal_model

        # Create a valid config file
        config = {
            "model": {
                "vision_encoder": {
                    "img_size": 224,
                    "patch_size": 16,
                    "in_channels": 3,
                    "hidden_dim": 256,
                    "num_layers": 4,
                    "num_heads": 4,
                    "mlp_ratio": 4.0,
                    "dropout": 0.1,
                },
                "text_encoder": {
                    "vocab_size": 30522,
                    "hidden_dim": 256,
                    "num_layers": 4,
                    "num_heads": 4,
                    "max_seq_length": 128,
                    "mlp_ratio": 4.0,
                    "dropout": 0.1,
                },
                "fusion": {
                    "type": "early",
                    "hidden_dim": 256,
                    "num_layers": 2,
                    "num_heads": 4,
                },
                "heads": {"type": "classification", "hidden_dim": 256, "num_classes": 10},
                "double_loop": {"hidden_dim": 128, "meta_lr": 1e-5},
                "use_double_loop": False,
            },
            "training": {
                "max_epochs": 1,
                "inner_lr": 1e-3,
                "optimizer": "adamw",
                "weight_decay": 0.01,
            },
            "data": {"batch_size": 2, "num_workers": 0},
        }

        config_path = temp_output_dir / "test_config.yaml"
        save_config(config, config_path)

        # Load config
        loaded_config = load_config(config_path)

        # Create model from loaded config
        model = create_multi_modal_model(loaded_config)

        assert model is not None
        assert hasattr(model, "forward")

    def test_validate_config_with_training(self, temp_output_dir):
        """Test config validation catches missing fields before training."""
        # Valid config
        valid_config = {
            "model": {
                "vision_encoder": {"hidden_dim": 256},
                "text_encoder": {"hidden_dim": 256},
                "fusion": {"type": "early"},
                "heads": {"type": "classification"},
            },
            "training": {"max_epochs": 1, "inner_lr": 1e-3},
            "data": {"batch_size": 2},
        }

        assert validate_config(valid_config) is True

        # Invalid config missing section
        invalid_config = {
            "model": {"vision_encoder": {}},
            "training": {"max_epochs": 1},
            # missing 'data'
        }

        with pytest.raises(ValueError, match="Missing required config section: data"):
            validate_config(invalid_config)

    def test_merge_configs_for_experiments(self):
        """Test merging base config with experiment overrides."""
        base_config = {
            "model": {"hidden_dim": 256, "num_layers": 4},
            "training": {"max_epochs": 10, "lr": 1e-3},
        }

        override_config = {
            "model": {"hidden_dim": 512},  # Override
            "training": {"max_epochs": 5},  # Override
        }

        merged = merge_configs(base_config, override_config)

        assert merged["model"]["hidden_dim"] == 512
        assert merged["model"]["num_layers"] == 4  # Preserved
        assert merged["training"]["max_epochs"] == 5
        assert merged["training"]["lr"] == 1e-3  # Preserved

    def test_config_namespace_access(self, model_config):
        """Test ConfigNamespace provides dot-access to config."""
        ns = ConfigNamespace(model_config)

        assert hasattr(ns, "model")
        assert hasattr(ns, "training")
        assert hasattr(ns.model, "vision_encoder")

        # Convert back to dict
        recovered = ns.to_dict()
        assert "model" in recovered
        assert "training" in recovered

    def test_full_config_to_training_pipeline(self, temp_output_dir, temp_data_dir):
        """Test complete flow from config file to training."""
        from src.models import create_multi_modal_model
        from src.training.losses import create_loss_function
        from src.training.optimizer import create_optimizer

        # Create config
        config = {
            "model": {
                "vision_encoder": {
                    "img_size": 224,
                    "patch_size": 16,
                    "in_channels": 3,
                    "hidden_dim": 128,
                    "num_layers": 2,
                    "num_heads": 2,
                    "mlp_ratio": 2.0,
                    "dropout": 0.1,
                },
                "text_encoder": {
                    "vocab_size": 30522,
                    "hidden_dim": 128,
                    "num_layers": 2,
                    "num_heads": 2,
                    "max_seq_length": 64,
                    "mlp_ratio": 2.0,
                    "dropout": 0.1,
                },
                "fusion": {"type": "early", "hidden_dim": 128, "num_layers": 1, "num_heads": 2},
                "heads": {"type": "classification", "hidden_dim": 128, "num_classes": 5},
                "double_loop": {"hidden_dim": 64, "meta_lr": 1e-5},
                "use_double_loop": False,
            },
            "training": {
                "max_epochs": 1,
                "inner_lr": 1e-3,
                "optimizer": "adamw",
                "weight_decay": 0.01,
            },
            "data": {"batch_size": 2, "num_workers": 0},
        }

        # Save and reload config
        config_path = temp_output_dir / "full_config.yaml"
        save_config(config, config_path)
        loaded_config = load_config(config_path)

        # Validate
        assert validate_config(loaded_config)

        # Create components
        model = create_multi_modal_model(loaded_config)
        optimizer = create_optimizer(model, loaded_config)
        criterion = create_loss_function(loaded_config)

        # Run a training step
        images = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 30522, (2, 64))
        attention_mask = torch.ones(2, 64)
        labels = torch.randint(0, 5, (2,))

        model.train()
        optimizer.zero_grad()
        outputs = model(images, input_ids, attention_mask)
        loss = criterion(outputs["logits"], labels)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0


# ============================================================================
# KNOWLEDGE INJECTION INTEGRATION TESTS
# ============================================================================


@pytest.mark.integration
class TestKnowledgeInjectionIntegration:
    """Integration tests for knowledge injection with model inference."""

    def test_additive_injection_with_model_output(self, model_config):
        """Test additive injection modifies model output correctly."""
        from src.models import create_multi_modal_model

        model = create_multi_modal_model(model_config)
        model.eval()

        images = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 30522, (2, 128))
        attention_mask = torch.ones(2, 128)

        with torch.no_grad():
            outputs = model(images, input_ids, attention_mask)
            original_logits = outputs["logits"].clone()

        # Apply additive injection
        strategy = AdditiveInjection(weight=0.5)
        knowledge = torch.ones_like(original_logits) * 2.0
        injected = strategy.inject(original_logits, knowledge)

        expected = original_logits + 0.5 * knowledge
        assert torch.allclose(injected, expected)

    def test_multiplicative_injection_with_model_output(self, model_config):
        """Test multiplicative injection scales model output."""
        from src.models import create_multi_modal_model

        model = create_multi_modal_model(model_config)
        model.eval()

        images = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 30522, (2, 128))
        attention_mask = torch.ones(2, 128)

        with torch.no_grad():
            outputs = model(images, input_ids, attention_mask)
            original_logits = outputs["logits"].clone()

        # Apply multiplicative injection
        strategy = MultiplicativeInjection(weight=0.2)
        injected = strategy.inject(original_logits, 1.0)

        # Should scale by (1 + weight)
        expected = original_logits * 1.2
        assert torch.allclose(injected, expected)

    def test_attention_injection_with_model_features(self):
        """Test attention injection blends features correctly."""
        hidden_dim = 128
        batch_size = 2
        seq_len = 10

        model_output = torch.randn(batch_size, seq_len, hidden_dim)
        knowledge = torch.randn(batch_size, seq_len, hidden_dim)

        strategy = AttentionInjection(hidden_dim=hidden_dim, weight=0.3)
        injected = strategy.inject(model_output, knowledge)

        # Output shape should match input
        assert injected.shape == model_output.shape

    def test_knowledge_manager_with_mock_injector(self, model_config):
        """Test KnowledgeInjectionManager coordinates injection."""
        from src.models import create_multi_modal_model

        model = create_multi_modal_model(model_config)
        model.eval()

        # Create manager
        manager = KnowledgeInjectionManager({"default_injection_weight": 0.1})

        # Create mock injector
        mock_injector = Mock(spec=KnowledgeInjector)
        mock_injector.inject_knowledge.return_value = {
            "injected": True,
            "knowledge": torch.ones(2, 10),
        }

        manager.register_injector("mock", mock_injector)
        manager.register_strategy("additive", AdditiveInjection(weight=0.2))

        # Get model output
        images = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 30522, (2, 128))
        attention_mask = torch.ones(2, 128)

        with torch.no_grad():
            outputs = model(images, input_ids, attention_mask)

        # Inject knowledge
        result = manager.inject_knowledge(
            input_data={"text": "test query"},
            model_output=outputs["logits"],
            injector_name="mock",
            strategy_name="additive",
        )

        assert result["success"] is True
        assert "modified_output" in result

    def test_create_injection_strategy_factory(self):
        """Test factory creates correct strategy types."""
        additive = create_injection_strategy("additive", weight=0.3)
        assert isinstance(additive, AdditiveInjection)

        multiplicative = create_injection_strategy("multiplicative", weight=0.2)
        assert isinstance(multiplicative, MultiplicativeInjection)

        attention = create_injection_strategy("attention", hidden_dim=256, weight=0.1)
        assert isinstance(attention, AttentionInjection)

        # Unknown defaults to additive
        unknown = create_injection_strategy("unknown")
        assert isinstance(unknown, AdditiveInjection)


# ============================================================================
# API INTEGRATION WITH INFERENCE TESTS
# ============================================================================


@pytest.mark.integration
class TestAPIIntegrationWithInference:
    """Integration tests for API integrations with model inference."""

    def test_wolfram_integration_with_model(self, model_config, api_config):
        """Test Wolfram Alpha integration can enhance model predictions."""
        from src.integrations.wolfram_alpha import WolframAlphaIntegration
        from src.models import create_multi_modal_model

        model = create_multi_modal_model(model_config)
        model.eval()

        # Mock Wolfram client
        with patch("wolframalpha.Client") as mock_client_cls:
            mock_result = Mock()
            mock_result.success = True
            mock_pod = Mock()
            mock_pod.title = "Result"
            mock_subpod = Mock()
            mock_subpod.plaintext = "42"
            mock_pod.subpods = [mock_subpod]
            mock_result.pods = [mock_pod]
            mock_client_cls.return_value.query.return_value = mock_result

            integration = WolframAlphaIntegration("test_key", api_config)
            response = integration.query("What is 6 * 7?")

            assert response.success is True
            assert response.data is not None

    def test_api_response_used_for_knowledge_injection(self, model_config, api_config):
        """Test API response can be used for knowledge injection pipeline."""
        from src.models import create_multi_modal_model

        model = create_multi_modal_model(model_config)
        model.eval()

        # Create mock API response
        api_response = APIResponse(
            success=True,
            data={"result": "The answer is 42", "confidence": 0.95},
            metadata={"source": "test_api"},
        )

        # Get model output
        images = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 30522, (2, 128))
        attention_mask = torch.ones(2, 128)

        with torch.no_grad():
            outputs = model(images, input_ids, attention_mask)

        # Simulate using API data to adjust predictions
        if api_response.success and api_response.data.get("confidence", 0) > 0.9:
            # High confidence API result could influence model output
            adjustment = torch.zeros_like(outputs["logits"])
            adjustment[:, 0] = 0.1  # Boost first class
            adjusted_logits = outputs["logits"] + adjustment

            assert adjusted_logits.shape == outputs["logits"].shape

    def test_knowledge_injector_with_api_integration(self, api_config):
        """Test KnowledgeInjector works with API integration."""
        from src.integrations.wolfram_alpha import (
            WolframAlphaIntegration,
            WolframKnowledgeInjector,
        )

        with patch("wolframalpha.Client") as mock_client_cls:
            mock_result = Mock()
            mock_result.success = True
            mock_pod = Mock()
            mock_pod.title = "Result"
            mock_subpod = Mock()
            mock_subpod.plaintext = "Answer text"
            mock_pod.subpods = [mock_subpod]
            mock_result.pods = [mock_pod]
            mock_client_cls.return_value.query.return_value = mock_result

            integration = WolframAlphaIntegration("test_key", api_config)

            injector_config = {"injection_weight": 0.1, "validation_threshold": 0.5}
            injector = WolframKnowledgeInjector(integration, injector_config)

            # Simulate low confidence model output
            model_output = torch.randn(2, 10)
            model_output_dict = {"logits": model_output, "confidence": 0.3}

            result = injector.inject_knowledge(
                input_data="What is the speed of light?", model_output=model_output_dict
            )

            assert "injected" in result or "query_result" in result


# ============================================================================
# END-TO-END TRAINING TESTS (EXISTING + ENHANCED)
# ============================================================================


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

        # Create gradient scaler using the non-deprecated API
        scaler = torch.amp.GradScaler("cuda")

        # Generate sample batch
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        input_ids = torch.randint(0, 30522, (batch_size, 128)).to(device)
        attention_mask = torch.ones(batch_size, 128).to(device)
        labels = torch.randint(0, 10, (batch_size,)).to(device)

        # Training step with mixed precision
        model.train()
        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
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


# ============================================================================
# DATA SELECTOR INTEGRATION TESTS
# ============================================================================


@pytest.mark.integration
class TestDataSelectorIntegration:
    """Integration tests for data selector with multiple datasets."""

    def test_selector_with_multiple_datasets(self, temp_data_dir):
        """Test selector can handle multiple dataset configs."""
        # Create two dataset directories
        ds1_dir = temp_data_dir / "dataset1"
        ds2_dir = temp_data_dir / "dataset2"
        ds1_dir.mkdir()
        ds2_dir.mkdir()

        # Create annotations for each
        for ds_dir, prefix in [(ds1_dir, "ds1"), (ds2_dir, "ds2")]:
            annotations = [
                {"image_path": f"{prefix}_{i}.jpg", "caption": f"{prefix} caption {i}", "label": i % 5}
                for i in range(10)
            ]
            with open(ds_dir / "annotations.json", "w") as f:
                json.dump(annotations, f)

        config = {
            "data": {
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
                "shuffle_train": True,
                "datasets": [
                    {
                        "name": "dataset1",
                        "type": "multimodal",
                        "data_path": str(ds1_dir),
                        "splits": {"train": 0.8, "val": 0.2},
                        "enabled": True,
                    },
                    {
                        "name": "dataset2",
                        "type": "multimodal",
                        "data_path": str(ds2_dir),
                        "splits": {"train": 1.0},
                        "use_in": ["train"],
                        "enabled": True,
                    },
                ],
            }
        }

        train_loader, val_loader, test_loader = build_dataloaders(config)

        assert train_loader is not None
        # Train should have samples from both datasets
        train_samples = sum(1 for _ in train_loader)
        assert train_samples > 0

    def test_selector_disabled_dataset(self, temp_data_dir):
        """Test selector skips disabled datasets."""
        annotations = [
            {"image_path": f"img_{i}.jpg", "caption": f"Caption {i}", "label": i}
            for i in range(10)
        ]
        with open(temp_data_dir / "annotations.json", "w") as f:
            json.dump(annotations, f)

        config = {
            "data": {
                "batch_size": 2,
                "num_workers": 0,
                "datasets": [
                    {
                        "name": "disabled_ds",
                        "type": "multimodal",
                        "data_path": str(temp_data_dir),
                        "splits": {"train": 1.0},
                        "enabled": False,  # Disabled
                    },
                    {
                        "name": "enabled_ds",
                        "type": "multimodal",
                        "data_path": str(temp_data_dir),
                        "splits": {"train": 1.0},
                        "enabled": True,
                    },
                ],
            }
        }

        train_loader, _, _ = build_dataloaders(config)
        # Only enabled dataset contributes
        assert train_loader is not None

    def test_selector_empty_config_raises(self):
        """Test selector raises error on empty datasets config."""
        config = {"data": {"datasets": []}}

        with pytest.raises(ValueError, match="datasets is empty"):
            build_dataloaders(config)


# ============================================================================
# FULL PIPELINE INTEGRATION TESTS
# ============================================================================


@pytest.mark.integration
class TestFullPipelineIntegration:
    """Integration tests for the complete training pipeline."""

    def test_config_to_inference_pipeline(self, temp_output_dir):
        """Test complete pipeline from config to inference."""
        from src.models import create_multi_modal_model

        # Create and save config
        config = {
            "model": {
                "vision_encoder": {
                    "img_size": 224,
                    "patch_size": 16,
                    "in_channels": 3,
                    "hidden_dim": 128,
                    "num_layers": 2,
                    "num_heads": 2,
                    "mlp_ratio": 2.0,
                    "dropout": 0.0,
                },
                "text_encoder": {
                    "vocab_size": 30522,
                    "hidden_dim": 128,
                    "num_layers": 2,
                    "num_heads": 2,
                    "max_seq_length": 64,
                    "mlp_ratio": 2.0,
                    "dropout": 0.0,
                },
                "fusion": {"type": "early", "hidden_dim": 128, "num_layers": 1, "num_heads": 2},
                "heads": {"type": "classification", "hidden_dim": 128, "num_classes": 10},
                "double_loop": {"hidden_dim": 64, "meta_lr": 1e-5},
                "use_double_loop": False,
            },
            "training": {"max_epochs": 1, "inner_lr": 1e-3},
            "data": {"batch_size": 2},
        }

        config_path = temp_output_dir / "pipeline_config.yaml"
        save_config(config, config_path)

        # Load config
        loaded = load_config(config_path)
        assert validate_config(loaded)

        # Create model
        model = create_multi_modal_model(loaded)

        # Save model weights
        model_path = temp_output_dir / "model.pt"
        torch.save(model.state_dict(), model_path)

        # Load model weights (simulating inference time)
        model2 = create_multi_modal_model(loaded)
        model2.load_state_dict(torch.load(model_path, weights_only=True))
        model2.eval()

        # Run inference
        images = torch.randn(1, 3, 224, 224)
        input_ids = torch.randint(0, 30522, (1, 64))
        attention_mask = torch.ones(1, 64)

        with torch.no_grad():
            outputs = model2(images, input_ids, attention_mask)

        assert "logits" in outputs
        assert outputs["logits"].shape == (1, 10)

    def test_training_with_validation_logging(self, model_config, temp_output_dir, temp_data_dir):
        """Test training loop with validation and logging integration."""
        from src.models import create_multi_modal_model
        from src.training.trainer import Trainer

        # Create dataset files
        annotations = [
            {"image_path": f"img_{i}.jpg", "caption": f"Caption {i}", "label": i % 10}
            for i in range(12)
        ]
        with open(temp_data_dir / "annotations.json", "w") as f:
            json.dump(annotations, f)

        # Create datasets with matching sequence length
        max_seq = model_config["model"]["text_encoder"]["max_seq_length"]
        train_ds = MultiModalDataset(
            str(temp_data_dir), split="train", augment=True, max_text_length=max_seq
        )
        val_ds = MultiModalDataset(
            str(temp_data_dir), split="val", augment=False, max_text_length=max_seq
        )

        train_loader, val_loader = create_data_loaders(
            train_ds, val_ds, batch_size=2, num_workers=0
        )

        # Setup training
        model_config["output_dir"] = str(temp_output_dir)
        model_config["training"]["num_epochs"] = 1
        model_config["logging"]["use_wandb"] = False

        model = create_multi_modal_model(model_config)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=model_config,
        )

        # Train
        trainer.train()

        # Verify outputs
        assert (temp_output_dir / "checkpoint_final.pt").exists() or len(
            list(temp_output_dir.glob("checkpoint_*.pt"))
        ) > 0

    def test_model_with_all_fusion_types(self, model_config):
        """Test model works with different fusion configurations."""
        from src.models import create_multi_modal_model

        # Only test supported fusion types
        fusion_types = ["early", "late"]

        for fusion_type in fusion_types:
            config = model_config.copy()
            config["model"] = model_config["model"].copy()
            config["model"]["fusion"] = {
                "type": fusion_type,
                "hidden_dim": 256,
                "num_layers": 2,
                "num_heads": 4,
            }

            model = create_multi_modal_model(config)
            model.eval()

            images = torch.randn(2, 3, 224, 224)
            input_ids = torch.randint(0, 30522, (2, 128))
            attention_mask = torch.ones(2, 128)

            with torch.no_grad():
                outputs = model(images, input_ids, attention_mask)

            assert "logits" in outputs, f"Failed for fusion type: {fusion_type}"


# ============================================================================
# CROSS-MODULE INTEGRATION TESTS
# ============================================================================


@pytest.mark.integration
class TestCrossModuleIntegration:
    """Tests verifying integration between different module groups."""

    def test_data_to_model_to_training(self, model_config, temp_data_dir, temp_output_dir):
        """Test data pipeline feeds correctly into model and training."""
        from src.models import create_multi_modal_model
        from src.training.losses import create_loss_function
        from src.training.optimizer import create_optimizer

        # Setup data
        annotations = [
            {"image_path": f"img_{i}.jpg", "caption": f"Test {i}", "label": i % 5}
            for i in range(6)
        ]
        with open(temp_data_dir / "annotations.json", "w") as f:
            json.dump(annotations, f)

        # Match sequence length to model config
        max_seq = model_config["model"]["text_encoder"]["max_seq_length"]
        dataset = MultiModalDataset(
            str(temp_data_dir), split="train", max_text_length=max_seq
        )
        loader = create_dataloader(dataset, batch_size=2, num_workers=0)

        # Setup model
        model = create_multi_modal_model(model_config)
        optimizer = create_optimizer(model, model_config)
        criterion = create_loss_function(model_config)

        # Training loop
        model.train()
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            outputs = model(batch["image"], batch["input_ids"], batch["attention_mask"])
            loss = criterion(outputs["logits"], batch["label"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        assert total_loss > 0

    def test_config_merging_with_training(self, temp_output_dir):
        """Test config merging integrates with training setup."""
        from src.models import create_multi_modal_model
        from src.training.optimizer import create_optimizer

        base_config = {
            "model": {
                "vision_encoder": {
                    "img_size": 224,
                    "patch_size": 16,
                    "in_channels": 3,
                    "hidden_dim": 256,
                    "num_layers": 4,
                    "num_heads": 4,
                    "mlp_ratio": 4.0,
                    "dropout": 0.1,
                },
                "text_encoder": {
                    "vocab_size": 30522,
                    "hidden_dim": 256,
                    "num_layers": 4,
                    "num_heads": 4,
                    "max_seq_length": 128,
                },
                "fusion": {"type": "early", "hidden_dim": 256},
                "heads": {"type": "classification", "hidden_dim": 256, "num_classes": 10},
                "double_loop": {"hidden_dim": 128},
            },
            "training": {"max_epochs": 10, "inner_lr": 1e-3, "optimizer": "adamw"},
            "data": {"batch_size": 32},
        }

        # Experiment override
        experiment_override = {
            "training": {"inner_lr": 5e-4, "max_epochs": 5},
            "data": {"batch_size": 16},
        }

        merged = merge_configs(base_config, experiment_override)

        # Verify merge
        assert merged["training"]["inner_lr"] == 5e-4
        assert merged["training"]["max_epochs"] == 5
        assert merged["training"]["optimizer"] == "adamw"  # Preserved
        assert merged["data"]["batch_size"] == 16

        # Use merged config for training
        model = create_multi_modal_model(merged)
        optimizer = create_optimizer(model, merged)

        assert optimizer is not None

    def test_knowledge_injection_in_inference_loop(self, model_config):
        """Test knowledge injection integrates with inference loop."""
        from src.models import create_multi_modal_model

        model = create_multi_modal_model(model_config)
        model.eval()

        # Create injection manager
        manager = KnowledgeInjectionManager({"default_injection_weight": 0.1})
        manager.register_strategy("additive", AdditiveInjection(weight=0.15))

        # Mock injector
        mock_injector = Mock(spec=KnowledgeInjector)

        # Run inference with potential injection
        images = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 30522, (2, 128))
        attention_mask = torch.ones(2, 128)

        with torch.no_grad():
            outputs = model(images, input_ids, attention_mask)
            base_logits = outputs["logits"]

            # Simulate injection based on confidence
            confidence = torch.softmax(base_logits, dim=-1).max(dim=-1).values.mean()

            if confidence < 0.5:
                # Low confidence - would trigger injection
                mock_injector.inject_knowledge.return_value = {
                    "injected": True,
                    "knowledge": torch.ones_like(base_logits) * 0.1,
                }
                manager.register_injector("mock", mock_injector)

                result = manager.inject_knowledge(
                    input_data="test",
                    model_output=base_logits,
                    strategy_name="additive",
                )

                if result.get("success"):
                    final_logits = result["modified_output"]
                else:
                    final_logits = base_logits
            else:
                final_logits = base_logits

        assert final_logits.shape == base_logits.shape
