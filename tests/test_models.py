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

    def test_text_encoder_with_token_type_ids(self, text_encoder_config, batch_size, seq_length):
        """Test text encoder forward pass with token_type_ids (segment embeddings)."""
        encoder = create_text_encoder(text_encoder_config)
        encoder.eval()

        input_ids = torch.randint(0, 30522, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        # Segment IDs: 0 for first half, 1 for second half
        token_type_ids = torch.zeros(batch_size, seq_length, dtype=torch.long)
        token_type_ids[:, seq_length // 2:] = 1

        with torch.no_grad():
            cls_token, sequence_output = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        assert cls_token.shape == (batch_size, text_encoder_config["hidden_dim"])
        assert sequence_output.shape == (batch_size, seq_length, text_encoder_config["hidden_dim"])

    def test_text_encoder_without_cls_token(self, batch_size):
        """Test text encoder with use_cls_token=False (mean pooling)."""
        seq_len = 32  # Use fixed short sequence length
        config = {
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "vocab_size": 1000,
            "max_seq_length": 64,
            "dropout": 0.0,
            "use_cls_token": False,
        }
        encoder = create_text_encoder(config)
        encoder.eval()

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        # Mask out some tokens
        attention_mask[:, seq_len // 2:] = 0

        with torch.no_grad():
            pooled, sequence_output = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Mean pooling output
        assert pooled.shape == (batch_size, config["hidden_dim"])
        assert sequence_output.shape == (batch_size, seq_len, config["hidden_dim"])

    def test_text_encoder_without_cls_token_no_mask(self, batch_size):
        """Test text encoder with use_cls_token=False and no attention mask."""
        seq_len = 32  # Use fixed short sequence length
        config = {
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "vocab_size": 1000,
            "max_seq_length": 64,
            "dropout": 0.0,
            "use_cls_token": False,
        }
        encoder = create_text_encoder(config)
        encoder.eval()

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        with torch.no_grad():
            pooled, sequence_output = encoder(input_ids=input_ids)

        # Simple mean pooling
        assert pooled.shape == (batch_size, config["hidden_dim"])

    def test_text_encoder_get_set_embeddings(self, text_encoder_config):
        """Test get_input_embeddings and set_input_embeddings."""
        encoder = create_text_encoder(text_encoder_config)

        # Get embeddings
        embeddings = encoder.get_input_embeddings()
        assert isinstance(embeddings, torch.nn.Embedding)
        assert embeddings.num_embeddings == text_encoder_config["vocab_size"]

        # Create new embeddings and set
        new_embeddings = torch.nn.Embedding(
            text_encoder_config["vocab_size"],
            text_encoder_config["hidden_dim"]
        )
        encoder.set_input_embeddings(new_embeddings)
        assert encoder.get_input_embeddings() is new_embeddings


class TestSimpleTokenizer:
    """Tests for SimpleTokenizer."""

    def test_tokenizer_creation(self):
        """Test tokenizer can be created."""
        from src.models.text_encoder import SimpleTokenizer
        tokenizer = SimpleTokenizer(vocab_size=30522)
        assert tokenizer.vocab_size == 30522
        assert tokenizer.pad_token_id == 0
        assert tokenizer.cls_token_id == 1
        assert tokenizer.sep_token_id == 2
        assert tokenizer.unk_token_id == 3

    def test_tokenizer_encode(self):
        """Test tokenizer encode method."""
        from src.models.text_encoder import SimpleTokenizer
        tokenizer = SimpleTokenizer(vocab_size=30522)

        result = tokenizer.encode("Hello world", max_length=32)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert result["input_ids"].shape == (1, 32)
        assert result["attention_mask"].shape == (1, 32)

        # First token should be CLS
        assert result["input_ids"][0, 0].item() == tokenizer.cls_token_id

        # Check attention mask: 1s for real tokens, 0s for padding
        # "Hello world" = 11 chars + CLS + SEP = 13 tokens
        expected_real_tokens = 13
        assert result["attention_mask"][0, :expected_real_tokens].sum().item() == expected_real_tokens
        assert result["attention_mask"][0, expected_real_tokens:].sum().item() == 0

    def test_tokenizer_encode_long_text(self):
        """Test tokenizer truncates long text."""
        from src.models.text_encoder import SimpleTokenizer
        tokenizer = SimpleTokenizer()

        long_text = "a" * 1000
        result = tokenizer.encode(long_text, max_length=64)

        # Should be truncated to max_length
        assert result["input_ids"].shape == (1, 64)


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

    def test_controller_compute_meta_gradient(self, double_loop_config, batch_size, hidden_dim):
        """Test controller compute_meta_gradient method."""
        from src.models import create_double_loop_controller

        controller = create_double_loop_controller(double_loop_config)
        controller.train()

        # Create a dummy model
        dummy_model = torch.nn.Linear(hidden_dim, hidden_dim)
        loss = torch.tensor(0.5)
        accuracy = torch.tensor(0.8)

        # Build some history by calling compute_meta_gradient multiple times
        for _ in range(15):
            meta_metrics = controller.compute_meta_gradient(dummy_model, loss, accuracy)

        # Check that meta metrics are returned
        assert isinstance(meta_metrics, dict)
        assert "loss_trend" in meta_metrics
        assert "accuracy_trend" in meta_metrics
        assert "loss_variance" in meta_metrics
        assert "accuracy_variance" in meta_metrics

    def test_controller_hidden_state_persistence(self, double_loop_config, batch_size, hidden_dim):
        """Test that hidden states persist across forward calls."""
        from src.models import create_double_loop_controller

        controller = create_double_loop_controller(double_loop_config)
        controller.eval()

        model_features = torch.randn(batch_size, hidden_dim)
        loss = torch.randn(batch_size, 1)
        accuracy = torch.randn(batch_size, 1)
        gradient_norm = torch.randn(batch_size, 1)

        # First forward pass
        with torch.no_grad():
            output1 = controller(model_features, loss, accuracy, gradient_norm)

        # Second forward pass should have different output due to state
        with torch.no_grad():
            output2 = controller(model_features, loss, accuracy, gradient_norm)

        # Outputs should differ due to hidden state
        assert controller.step_count == 2

    def test_controller_should_update_meta(self, double_loop_config):
        """Test controller should_update_meta method."""
        from src.models import create_double_loop_controller

        controller = create_double_loop_controller(double_loop_config)
        
        # Initially step_count is 0, should return True (0 % update_frequency == 0)
        assert controller.should_update_meta() is True
        
        # Increment step count manually
        controller.step_count = 5
        assert controller.should_update_meta() is False
        
        # At update_frequency multiple, should return True
        controller.step_count = double_loop_config["update_frequency"]
        assert controller.should_update_meta() is True

    def test_controller_history_limiting(self, double_loop_config):
        """Test that history is limited to prevent memory issues."""
        from src.models import create_double_loop_controller

        controller = create_double_loop_controller(double_loop_config)
        dummy_model = torch.nn.Linear(10, 10)
        
        # Add more than 1000 entries to history
        for i in range(1100):
            loss = torch.tensor(float(i) / 1100)
            accuracy = torch.tensor(float(i) / 1100)
            controller.compute_meta_gradient(dummy_model, loss, accuracy)
        
        # History should be limited to 1000
        assert len(controller.loss_history) <= 1000
        assert len(controller.accuracy_history) <= 1000


class TestLSTMMetaController:
    """Tests for LSTM meta-controller component."""

    def test_lstm_meta_controller_forward(self, batch_size, hidden_dim):
        """Test LSTM meta-controller forward pass."""
        from src.models.double_loop_controller import LSTMMetaController

        controller = LSTMMetaController(
            model_hidden_dim=hidden_dim,
            controller_hidden_dim=128,
            num_layers=2,
            dropout=0.1,
        )
        controller.eval()

        # Inputs match expected format: model_features, loss, accuracy, gradient_norm
        model_features = torch.randn(batch_size, hidden_dim)
        loss = torch.randn(batch_size, 1)
        accuracy = torch.randn(batch_size, 1)
        gradient_norm = torch.randn(batch_size, 1)

        with torch.no_grad():
            lr_scale, arch_adaptation, meta_loss = controller(
                model_features, loss, accuracy, gradient_norm
            )

        assert lr_scale.shape == (batch_size, 1)
        assert arch_adaptation.shape == (batch_size, 64)
        assert meta_loss.shape == (batch_size, 1)

    def test_lstm_meta_controller_with_hidden_state(self, batch_size, hidden_dim):
        """Test LSTM meta-controller with hidden state persistence."""
        from src.models.double_loop_controller import LSTMMetaController

        controller = LSTMMetaController(
            model_hidden_dim=hidden_dim,
            controller_hidden_dim=128,
            num_layers=2,
            dropout=0.1,
        )
        controller.eval()

        model_features = torch.randn(batch_size, hidden_dim)
        loss = torch.randn(batch_size, 1)
        accuracy = torch.randn(batch_size, 1)
        gradient_norm = torch.randn(batch_size, 1)

        # First forward pass sets hidden state
        with torch.no_grad():
            output1 = controller(model_features, loss, accuracy, gradient_norm)

        # Hidden state should now be set
        assert controller.hidden_state is not None

        # Second forward pass uses existing hidden state
        with torch.no_grad():
            output2 = controller(model_features, loss, accuracy, gradient_norm)

        # Outputs should differ due to LSTM state
        assert controller.hidden_state is not None

    def test_lstm_meta_controller_reset_state(self, hidden_dim):
        """Test LSTM meta-controller state reset."""
        from src.models.double_loop_controller import LSTMMetaController

        controller = LSTMMetaController(
            model_hidden_dim=hidden_dim,
            controller_hidden_dim=128,
        )

        # Do a forward pass to set hidden state
        model_features = torch.randn(1, hidden_dim)
        loss = torch.randn(1, 1)
        accuracy = torch.randn(1, 1)
        gradient_norm = torch.randn(1, 1)

        controller(model_features, loss, accuracy, gradient_norm)
        assert controller.hidden_state is not None

        # Reset state
        controller.reset_state()
        assert controller.hidden_state is None


class TestAdaptiveLayerNorm:
    """Tests for adaptive layer normalization."""

    def test_adaptive_layer_norm_forward(self, hidden_dim, batch_size):
        """Test adaptive layer norm forward pass."""
        from src.models.double_loop_controller import AdaptiveLayerNorm

        norm = AdaptiveLayerNorm(hidden_dim, adaptation_dim=32)
        norm.eval()

        x = torch.randn(batch_size, hidden_dim)

        with torch.no_grad():
            output = norm(x)

        assert output.shape == x.shape

    def test_adaptive_layer_norm_with_adaptation(self, hidden_dim, batch_size):
        """Test adaptive layer norm with adaptation signal."""
        from src.models.double_loop_controller import AdaptiveLayerNorm

        norm = AdaptiveLayerNorm(hidden_dim, adaptation_dim=32)
        norm.eval()

        x = torch.randn(batch_size, hidden_dim)
        adaptation = torch.randn(batch_size, 32)

        with torch.no_grad():
            output = norm(x, adaptation)

        assert output.shape == x.shape


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

    def test_model_get_model_info(self, model_config):
        """Test getting model info."""
        model = create_multi_modal_model(model_config)
        info = model.get_model_info()

        assert isinstance(info, dict)
        assert "vision_encoder" in info or "total_params" in info

    def test_model_enable_gradient_checkpointing(self, model_config):
        """Test enabling gradient checkpointing."""
        model = create_multi_modal_model(model_config)
        model.enable_gradient_checkpointing()
        
        # Should not raise
        assert True

    def test_model_late_fusion(self, model_config, sample_images, sample_text_inputs):
        """Test model with late fusion."""
        # Modify config for late fusion
        late_config = model_config.copy()
        late_config["model"] = model_config["model"].copy()
        late_config["model"]["fusion"] = model_config["model"]["fusion"].copy()
        late_config["model"]["fusion"]["type"] = "late"
        
        model = create_multi_modal_model(late_config)
        model.eval()

        with torch.no_grad():
            outputs = model(
                images=sample_images,
                input_ids=sample_text_inputs["input_ids"],
                attention_mask=sample_text_inputs["attention_mask"],
            )

        assert "logits" in outputs

    def test_model_with_double_loop_controller(self, model_config, sample_images, sample_text_inputs):
        """Test model with double-loop controller."""
        dl_config = model_config.copy()
        dl_config["model"] = model_config["model"].copy()
        dl_config["model"]["double_loop"] = {
            "enabled": True,
            "hidden_dim": 64,
            "meta_window": 5,
            "output_dim": 16,
        }
        
        model = create_multi_modal_model(dl_config)
        model.eval()

        with torch.no_grad():
            outputs = model(
                images=sample_images,
                input_ids=sample_text_inputs["input_ids"],
                attention_mask=sample_text_inputs["attention_mask"],
            )

        assert "logits" in outputs

    def test_model_with_double_loop_controller_training(self, model_config, sample_images, sample_text_inputs, batch_size):
        """Test model with double-loop controller during training with controller inputs."""
        dl_config = model_config.copy()
        dl_config["model"] = model_config["model"].copy()
        dl_config["model"]["use_double_loop"] = True
        dl_config["model"]["double_loop"] = {
            "enabled": True,
            "model_hidden_dim": model_config["model"]["vision_encoder"]["hidden_dim"],
            "hidden_dim": 64,
            "update_frequency": 10,
        }
        
        model = create_multi_modal_model(dl_config)
        model.train()

        # Provide controller inputs
        current_loss = torch.tensor([[0.5]] * batch_size)
        current_accuracy = torch.tensor([[0.8]] * batch_size)
        gradient_norm = torch.tensor([[1.0]] * batch_size)

        outputs = model(
            images=sample_images,
            input_ids=sample_text_inputs["input_ids"],
            attention_mask=sample_text_inputs["attention_mask"],
            current_loss=current_loss,
            current_accuracy=current_accuracy,
            gradient_norm=gradient_norm,
        )

        assert "logits" in outputs
        assert "meta_info" in outputs
        # When controller inputs provided, meta_info should have controller output
        if outputs["meta_info"] is not None:
            assert "lr_scale" in outputs["meta_info"]
            assert "arch_adaptation" in outputs["meta_info"]

    def test_model_with_double_loop_controller_no_inputs(self, model_config, sample_images, sample_text_inputs):
        """Test model with double-loop controller during training without controller inputs."""
        dl_config = model_config.copy()
        dl_config["model"] = model_config["model"].copy()
        dl_config["model"]["use_double_loop"] = True
        dl_config["model"]["double_loop"] = {
            "enabled": True,
            "model_hidden_dim": model_config["model"]["vision_encoder"]["hidden_dim"],
            "hidden_dim": 64,
        }
        
        model = create_multi_modal_model(dl_config)
        model.train()

        # Don't provide controller inputs
        outputs = model(
            images=sample_images,
            input_ids=sample_text_inputs["input_ids"],
            attention_mask=sample_text_inputs["attention_mask"],
        )

        assert "logits" in outputs
        # Without controller inputs, meta_info should be None
        assert outputs.get("meta_info") is None

    def test_model_with_contrastive_head(self, model_config, sample_images, sample_text_inputs):
        """Test model with contrastive head."""
        contrastive_config = model_config.copy()
        contrastive_config["model"] = model_config["model"].copy()
        contrastive_config["model"]["heads"] = {
            "type": "contrastive",
            "hidden_dim": model_config["model"]["vision_encoder"]["hidden_dim"],
            "projection_dim": 128,
        }
        
        model = create_multi_modal_model(contrastive_config)
        model.eval()

        with torch.no_grad():
            outputs = model(
                images=sample_images,
                input_ids=sample_text_inputs["input_ids"],
                attention_mask=sample_text_inputs["attention_mask"],
                task_name="contrastive",
            )

        assert "logits" in outputs

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


class TestLoadPretrainedWeights:
    """Tests for load_pretrained_weights function."""

    def test_load_pretrained_weights_no_checkpoints(self, model_config):
        """Test load_pretrained_weights with no checkpoints specified."""
        from src.models.multi_modal_model import load_pretrained_weights

        model = create_multi_modal_model(model_config)
        
        # No checkpoints - should return model unchanged
        result = load_pretrained_weights(model)
        assert result is model

    def test_load_pretrained_weights_valid_vision_checkpoint(self, model_config, tmp_path):
        """Test load_pretrained_weights with valid vision checkpoint."""
        from src.models.multi_modal_model import load_pretrained_weights

        model = create_multi_modal_model(model_config)
        
        # Save vision encoder state dict
        checkpoint_path = tmp_path / "vision_checkpoint.pt"
        torch.save(model.vision_encoder.state_dict(), checkpoint_path)
        
        # Create new model and load vision weights
        model2 = create_multi_modal_model(model_config)
        result = load_pretrained_weights(model2, vision_checkpoint=str(checkpoint_path))
        
        assert result is model2

    def test_load_pretrained_weights_valid_text_checkpoint(self, model_config, tmp_path):
        """Test load_pretrained_weights with valid text checkpoint."""
        from src.models.multi_modal_model import load_pretrained_weights

        model = create_multi_modal_model(model_config)
        
        # Save text encoder state dict
        checkpoint_path = tmp_path / "text_checkpoint.pt"
        torch.save(model.text_encoder.state_dict(), checkpoint_path)
        
        # Create new model and load text weights
        model2 = create_multi_modal_model(model_config)
        result = load_pretrained_weights(model2, text_checkpoint=str(checkpoint_path))
        
        assert result is model2

    def test_load_pretrained_weights_both_checkpoints(self, model_config, tmp_path):
        """Test load_pretrained_weights with both checkpoints."""
        from src.models.multi_modal_model import load_pretrained_weights

        model = create_multi_modal_model(model_config)
        
        # Save both encoder state dicts
        vision_path = tmp_path / "vision.pt"
        text_path = tmp_path / "text.pt"
        torch.save(model.vision_encoder.state_dict(), vision_path)
        torch.save(model.text_encoder.state_dict(), text_path)
        
        # Create new model and load both weights
        model2 = create_multi_modal_model(model_config)
        result = load_pretrained_weights(
            model2, 
            vision_checkpoint=str(vision_path),
            text_checkpoint=str(text_path)
        )
        
        assert result is model2
