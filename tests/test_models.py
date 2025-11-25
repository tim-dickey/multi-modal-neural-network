"""Tests for model components."""

import pytest
import torch

from src.models import (
    ClassificationHead,
    ContrastiveHead,
    DoubleLoopController,
    FusionLayer,
    MultiModalModel,
    TextEncoder,
    VisionEncoder,
    create_fusion_layer,
    create_multi_modal_model,
    create_text_encoder,
    create_vision_encoder,
)


class TestVisionEncoder:
    """Tests for Vision Transformer encoder."""

    def test_vision_encoder_creation(self, vision_encoder_config):
        """Test that vision encoder can be created."""
        encoder = create_vision_encoder(vision_encoder_config)
        assert isinstance(encoder, VisionEncoder)

    def test_vision_encoder_forward(self, vision_encoder_config, sample_images):
        """Test forward pass through vision encoder."""
        encoder = create_vision_encoder(vision_encoder_config)
        encoder.eval()

        with torch.no_grad():
            cls_token, patch_tokens = encoder(sample_images)

        batch_size = sample_images.shape[0]
        hidden_dim = vision_encoder_config["hidden_dim"]

        # Check output shapes
        assert cls_token.shape == (batch_size, hidden_dim)
        assert patch_tokens.shape[0] == batch_size
        assert patch_tokens.shape[2] == hidden_dim

    def test_vision_encoder_no_cls_token(self, vision_encoder_config, sample_images):
        """Test vision encoder without CLS token."""
        vision_encoder_config["use_cls_token"] = False
        encoder = create_vision_encoder(vision_encoder_config)
        encoder.eval()

        with torch.no_grad():
            cls_token, patch_tokens = encoder(sample_images)

        assert cls_token is None
        assert patch_tokens is not None

    def test_vision_encoder_different_image_sizes(self, vision_encoder_config):
        """Test vision encoder with different image sizes."""
        encoder = create_vision_encoder(vision_encoder_config)
        encoder.eval()

        # Test multiple image sizes
        for size in [224, 256, 384]:
            images = torch.randn(2, 3, size, size)
            # This should raise an error for incompatible sizes
            # or we need to adjust patch_size
            if size == vision_encoder_config["img_size"]:
                with torch.no_grad():
                    cls_token, patch_tokens = encoder(images)
                assert cls_token is not None

    def test_vision_encoder_parameter_count(self, vision_encoder_config):
        """Test that parameter count is reasonable."""
        encoder = create_vision_encoder(vision_encoder_config)
        total_params = sum(p.numel() for p in encoder.parameters())

        # Should be less than 100M parameters for small config
        assert total_params < 100_000_000
        assert total_params > 0


class TestTextEncoder:
    """Tests for BERT-style text encoder."""

    def test_text_encoder_creation(self, text_encoder_config):
        """Test that text encoder can be created."""
        encoder = create_text_encoder(text_encoder_config)
        assert isinstance(encoder, TextEncoder)

    def test_text_encoder_forward(self, text_encoder_config, sample_text_inputs):
        """Test forward pass through text encoder."""
        encoder = create_text_encoder(text_encoder_config)
        encoder.eval()

        with torch.no_grad():
            cls_token, sequence_output = encoder(
                input_ids=sample_text_inputs["input_ids"],
                attention_mask=sample_text_inputs["attention_mask"],
            )

        batch_size = sample_text_inputs["input_ids"].shape[0]
        seq_length = sample_text_inputs["input_ids"].shape[1]
        hidden_dim = text_encoder_config["hidden_dim"]

        # Check output shapes
        assert cls_token.shape == (batch_size, hidden_dim)
        assert sequence_output.shape == (batch_size, seq_length, hidden_dim)

    def test_text_encoder_with_masking(
        self, text_encoder_config, batch_size, seq_length
    ):
        """Test text encoder with partial masking."""
        encoder = create_text_encoder(text_encoder_config)
        encoder.eval()

        input_ids = torch.randint(0, 30522, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        # Mask out last half
        attention_mask[:, seq_length // 2 :] = 0

        with torch.no_grad():
            cls_token, sequence_output = encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )

        assert cls_token is not None
        assert sequence_output is not None


class TestFusionLayer:
    """Tests for multi-modal fusion layer."""

    def test_early_fusion_creation(self, fusion_config):
        """Test early fusion layer creation."""
        fusion_config["type"] = "early"
        fusion = create_fusion_layer(fusion_config)
        assert isinstance(fusion, FusionLayer)

    def test_late_fusion_creation(self, fusion_config):
        """Test late fusion layer creation."""
        fusion_config["type"] = "late"
        fusion = create_fusion_layer(fusion_config)
        assert isinstance(fusion, FusionLayer)

    def test_early_fusion_forward(self, fusion_config, batch_size, hidden_dim):
        """Test early fusion forward pass."""
        fusion_config["type"] = "early"
        fusion = create_fusion_layer(fusion_config)
        fusion.eval()

        # Create dummy features
        vision_features = torch.randn(batch_size, 196, hidden_dim)  # 14x14 patches
        text_features = torch.randn(batch_size, 128, hidden_dim)

        with torch.no_grad():
            fused_output, vision_seq_len = fusion(
                vision_features=vision_features, text_features=text_features
            )

        # Check output
        assert fused_output.shape[0] == batch_size
        assert fused_output.shape[2] == hidden_dim
        assert vision_seq_len == 196

    def test_late_fusion_forward(self, fusion_config, batch_size, hidden_dim):
        """Test late fusion forward pass."""
        fusion_config["type"] = "late"
        fusion = create_fusion_layer(fusion_config)
        fusion.eval()

        # Create dummy pooled features
        vision_pooled = torch.randn(batch_size, hidden_dim)
        text_pooled = torch.randn(batch_size, hidden_dim)

        with torch.no_grad():
            fused_output, _ = fusion(
                vision_features=None,
                text_features=None,
                vision_pooled=vision_pooled,
                text_pooled=text_pooled,
            )

        assert fused_output.shape == (batch_size, hidden_dim)


class TestDoubleLoopController:
    """Tests for double-loop learning controller."""

    def test_controller_creation(self, double_loop_config):
        """Test controller creation."""
        from src.models import create_double_loop_controller

        controller = create_double_loop_controller(double_loop_config)
        assert isinstance(controller, DoubleLoopController)

    def test_controller_forward(self, double_loop_config, batch_size, hidden_dim):
        """Test controller forward pass."""
        from src.models import create_double_loop_controller

        controller = create_double_loop_controller(double_loop_config)
        controller.eval()

        # Create dummy inputs
        model_features = torch.randn(batch_size, hidden_dim)
        loss = torch.randn(batch_size, 1)
        accuracy = torch.randn(batch_size, 1)
        gradient_norm = torch.randn(batch_size, 1)

        with torch.no_grad():
            output = controller(model_features, loss, accuracy, gradient_norm)

        assert "lr_scale" in output
        assert "arch_adaptation" in output
        assert "meta_loss" in output

    def test_controller_reset(self, double_loop_config):
        """Test controller state reset."""
        from src.models import create_double_loop_controller

        controller = create_double_loop_controller(double_loop_config)
        controller.reset()

        assert controller.step_count == 0
        assert len(controller.loss_history) == 0


class TestTaskHeads:
    """Tests for task-specific heads."""

    def test_classification_head(self, head_config, batch_size, hidden_dim):
        """Test classification head."""
        head = ClassificationHead(
            hidden_dim=hidden_dim, num_classes=head_config["num_classes"]
        )
        head.eval()

        features = torch.randn(batch_size, hidden_dim)

        with torch.no_grad():
            logits = head(features)

        assert logits.shape == (batch_size, head_config["num_classes"])

    def test_contrastive_head(self, batch_size, hidden_dim):
        """Test contrastive head."""
        head = ContrastiveHead(hidden_dim=hidden_dim, projection_dim=128)
        head.eval()

        image_features = torch.randn(batch_size, hidden_dim)
        text_features = torch.randn(batch_size, hidden_dim)

        with torch.no_grad():
            similarity = head(image_features, text_features)

        assert similarity.shape == (batch_size, batch_size)


class TestMultiModalModel:
    """Tests for complete multi-modal model."""

    def test_model_creation(self, model_config):
        """Test model creation from config."""
        model = create_multi_modal_model(model_config)
        assert isinstance(model, MultiModalModel)

    def test_model_forward(self, model_config, sample_images, sample_text_inputs):
        """Test full forward pass."""
        model = create_multi_modal_model(model_config)
        model.eval()

        with torch.no_grad():
            outputs = model(
                images=sample_images,
                input_ids=sample_text_inputs["input_ids"],
                attention_mask=sample_text_inputs["attention_mask"],
            )

        assert "logits" in outputs
        batch_size = sample_images.shape[0]
        num_classes = model_config["model"]["heads"]["num_classes"]
        assert outputs["logits"].shape == (batch_size, num_classes)

    def test_model_with_features(self, model_config, sample_images, sample_text_inputs):
        """Test forward pass with feature extraction."""
        model = create_multi_modal_model(model_config)
        model.eval()

        with torch.no_grad():
            outputs = model(
                images=sample_images,
                input_ids=sample_text_inputs["input_ids"],
                attention_mask=sample_text_inputs["attention_mask"],
                return_features=True,
            )

        assert "features" in outputs
        assert "vision_cls" in outputs["features"]
        assert "text_cls" in outputs["features"]

    def test_model_parameter_count(self, model_config):
        """Test model parameter count."""
        model = create_multi_modal_model(model_config)

        total_params = model.get_num_parameters(trainable_only=False)
        trainable_params = model.get_num_parameters(trainable_only=True)

        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params

    def test_model_freeze_encoders(self, model_config):
        """Test freezing encoder parameters."""
        model = create_multi_modal_model(model_config)

        initial_trainable = model.get_num_parameters(trainable_only=True)

        model.freeze_vision_encoder()
        after_freeze_vision = model.get_num_parameters(trainable_only=True)
        assert after_freeze_vision < initial_trainable

        model.freeze_text_encoder()
        after_freeze_both = model.get_num_parameters(trainable_only=True)
        assert after_freeze_both < after_freeze_vision

        model.unfreeze_all()
        after_unfreeze = model.get_num_parameters(trainable_only=True)
        assert after_unfreeze == initial_trainable

    @pytest.mark.slow
    def test_model_backward_pass(
        self, model_config, sample_images, sample_text_inputs, sample_labels
    ):
        """Test backward pass through model."""
        model = create_multi_modal_model(model_config)
        model.train()

        outputs = model(
            images=sample_images,
            input_ids=sample_text_inputs["input_ids"],
            attention_mask=sample_text_inputs["attention_mask"],
        )

        logits = outputs["logits"]
        loss = torch.nn.functional.cross_entropy(logits, sample_labels)
        loss.backward()

        # Check that gradients were computed
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break

        assert has_grad, "No gradients were computed"
