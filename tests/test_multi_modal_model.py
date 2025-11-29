import torch

import pytest

from src.models import multi_modal_model as mmm


class DummyEncoder(torch.nn.Module):
    def __init__(self, hidden_dim=32, seq_len=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

    def forward(self, x=None, input_ids=None, **kwargs):
        # Handle both vision (positional x) and text (keyword input_ids)
        if x is not None:
            batch = x.shape[0]
        elif input_ids is not None:
            batch = input_ids.shape[0]
        else:
            batch = 1
        device = (
            x.device
            if x is not None
            else (input_ids.device if input_ids is not None else "cpu")
        )
        cls = torch.randn(batch, self.hidden_dim, device=device)
        features = torch.randn(batch, self.seq_len, self.hidden_dim, device=device)
        return cls, features


class DummyFusion(torch.nn.Module):
    def __init__(self, hidden_dim=32, fusion_type="early"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fusion_type = fusion_type

    def forward(
        self,
        vision_features=None,
        text_features=None,
        vision_pooled=None,
        text_pooled=None,
        **kwargs,
    ):
        if self.fusion_type == "late":
            # Late fusion returns pooled features directly
            pooled = vision_pooled + text_pooled
            return pooled, None
        # Early fusion - concatenate on sequence dim
        fused = vision_features + text_features
        return fused, None


def make_minimal_model(monkeypatch, fusion_type="early"):
    # Monkeypatch creators to simple dummies
    monkeypatch.setattr(
        mmm, "create_vision_encoder", lambda cfg: DummyEncoder(hidden_dim=32)
    )
    monkeypatch.setattr(
        mmm, "create_text_encoder", lambda cfg: DummyEncoder(hidden_dim=32)
    )
    monkeypatch.setattr(
        mmm,
        "create_fusion_layer",
        lambda cfg: DummyFusion(hidden_dim=32, fusion_type=fusion_type),
    )
    monkeypatch.setattr(
        mmm, "create_double_loop_controller", lambda cfg: (lambda **k: {"ok": True})
    )
    monkeypatch.setattr(
        mmm,
        "create_task_head",
        lambda cfg: mmm.MultiTaskHead(hidden_dim=32, tasks={"default": {}}),
    )

    cfg = {
        "model": {
            "vision_encoder": {},
            "text_encoder": {},
            "fusion": {"hidden_dim": 32, "type": fusion_type},
            "double_loop": {},
            "heads": {},
        },
        "training": {},
    }
    model = mmm.create_multi_modal_model(cfg)
    return model


def test_forward_and_features(monkeypatch):
    model = make_minimal_model(monkeypatch)
    batch = torch.randn(2, 3, 64, 64)
    # Call with no text inputs - code path should fill zeros for text
    out = model(images=batch, input_ids=None, return_features=True)
    assert "logits" in out or "default" in out
    assert "features" in out

    # Test parameter counting and freezing
    tot = model.get_num_parameters(trainable_only=False)
    assert tot > 0
    model.freeze_vision_encoder()
    # Ensure parameters were frozen
    assert all(not p.requires_grad for p in model.vision_encoder.parameters())
    model.unfreeze_all()
    assert all(p.requires_grad for p in model.parameters())


def test_forward_no_text_input(monkeypatch):
    """Test forward pass with no text input - should create zero text features."""
    model = make_minimal_model(monkeypatch)
    batch = torch.randn(2, 3, 64, 64)

    # Call with no text inputs
    out = model(images=batch, input_ids=None)

    # Should still produce output
    assert "logits" in out or "default" in out


def test_forward_late_fusion(monkeypatch):
    """Test forward pass with late fusion type."""
    model = make_minimal_model(monkeypatch, fusion_type="late")

    batch = torch.randn(2, 3, 64, 64)
    input_ids = torch.randint(0, 100, (2, 16))

    out = model(images=batch, input_ids=input_ids)

    assert "logits" in out or "default" in out


def test_apply_task_head_helper(monkeypatch):
    """Test _apply_task_head helper method."""
    model = make_minimal_model(monkeypatch)

    # Test with multi-task head
    pooled = torch.randn(2, 32)
    vision_cls = torch.randn(2, 32)
    text_cls = torch.randn(2, 32)

    outputs = model._apply_task_head(pooled, vision_cls, text_cls, task_name="default")
    assert isinstance(outputs, dict)


def test_apply_task_head_single_task(monkeypatch):
    """Test _apply_task_head with a single-task head."""

    # Create a simple classification head
    class SimpleHead(torch.nn.Module):
        def __init__(self, hidden_dim, num_classes=10):
            super().__init__()
            self.fc = torch.nn.Linear(hidden_dim, num_classes)

        def forward(self, x):
            return self.fc(x)

    monkeypatch.setattr(
        mmm, "create_vision_encoder", lambda cfg: DummyEncoder(hidden_dim=32)
    )
    monkeypatch.setattr(
        mmm, "create_text_encoder", lambda cfg: DummyEncoder(hidden_dim=32)
    )
    monkeypatch.setattr(
        mmm, "create_fusion_layer", lambda cfg: DummyFusion(hidden_dim=32)
    )
    monkeypatch.setattr(
        mmm, "create_double_loop_controller", lambda cfg: (lambda **k: {"ok": True})
    )
    monkeypatch.setattr(mmm, "create_task_head", lambda cfg: SimpleHead(hidden_dim=32))

    cfg = {"model": {"fusion": {"hidden_dim": 32}}, "training": {}}
    model = mmm.create_multi_modal_model(cfg)

    pooled = torch.randn(2, 32)
    vision_cls = torch.randn(2, 32)
    text_cls = torch.randn(2, 32)

    outputs = model._apply_task_head(pooled, vision_cls, text_cls, task_name=None)
    assert "logits" in outputs


def test_apply_task_head_contrastive(monkeypatch):
    """Test _apply_task_head with a contrastive head."""

    class ContrastiveHead(torch.nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.proj = torch.nn.Linear(hidden_dim, hidden_dim)

        def forward(self, vision_features, text_features):
            return torch.matmul(vision_features, text_features.T)

    monkeypatch.setattr(
        mmm, "create_vision_encoder", lambda cfg: DummyEncoder(hidden_dim=32)
    )
    monkeypatch.setattr(
        mmm, "create_text_encoder", lambda cfg: DummyEncoder(hidden_dim=32)
    )
    monkeypatch.setattr(
        mmm, "create_fusion_layer", lambda cfg: DummyFusion(hidden_dim=32)
    )
    monkeypatch.setattr(
        mmm, "create_double_loop_controller", lambda cfg: (lambda **k: {"ok": True})
    )
    monkeypatch.setattr(
        mmm, "create_task_head", lambda cfg: ContrastiveHead(hidden_dim=32)
    )

    cfg = {"model": {"fusion": {"hidden_dim": 32}}, "training": {}}
    model = mmm.create_multi_modal_model(cfg)

    pooled = torch.randn(2, 32)
    vision_cls = torch.randn(2, 32)
    text_cls = torch.randn(2, 32)

    # Test with task_name="contrastive"
    outputs = model._apply_task_head(
        pooled, vision_cls, text_cls, task_name="contrastive"
    )
    assert "logits" in outputs


def test_apply_task_head_no_forward(monkeypatch):
    """Test _apply_task_head when head has no forward method (passthrough)."""

    class PassthroughHead:
        pass

    monkeypatch.setattr(
        mmm, "create_vision_encoder", lambda cfg: DummyEncoder(hidden_dim=32)
    )
    monkeypatch.setattr(
        mmm, "create_text_encoder", lambda cfg: DummyEncoder(hidden_dim=32)
    )
    monkeypatch.setattr(
        mmm, "create_fusion_layer", lambda cfg: DummyFusion(hidden_dim=32)
    )
    monkeypatch.setattr(
        mmm, "create_double_loop_controller", lambda cfg: (lambda **k: {"ok": True})
    )
    monkeypatch.setattr(mmm, "create_task_head", lambda cfg: PassthroughHead())

    cfg = {"model": {"fusion": {"hidden_dim": 32}}, "training": {}}
    model = mmm.create_multi_modal_model(cfg)

    pooled = torch.randn(2, 32)
    vision_cls = torch.randn(2, 32)
    text_cls = torch.randn(2, 32)

    # Head with no forward should return pooled as logits
    outputs = model._apply_task_head(pooled, vision_cls, text_cls, task_name=None)
    assert "logits" in outputs
    assert torch.equal(outputs["logits"], pooled)


def test_load_pretrained_weights(monkeypatch, tmp_path):
    """Test loading pretrained weights for vision and text encoders."""
    model = make_minimal_model(monkeypatch)

    # Create mock checkpoint files
    vision_ckpt_path = tmp_path / "vision.pt"
    text_ckpt_path = tmp_path / "text.pt"

    # Save model state dicts
    torch.save(model.vision_encoder.state_dict(), vision_ckpt_path)
    torch.save(model.text_encoder.state_dict(), text_ckpt_path)

    # Test loading
    result = mmm.load_pretrained_weights(
        model,
        vision_checkpoint=str(vision_ckpt_path),
        text_checkpoint=str(text_ckpt_path),
        strict=False,
        allow_external=True,
    )

    assert result is model


def test_load_pretrained_weights_with_wrapper(monkeypatch, tmp_path):
    """Test loading weights from wrapped checkpoint (model_state_dict key)."""
    model = make_minimal_model(monkeypatch)

    # Create mock checkpoint files with wrapper dict
    vision_ckpt_path = tmp_path / "vision_wrapped.pt"
    text_ckpt_path = tmp_path / "text_wrapped.pt"

    # Save model state dicts in wrapper format
    torch.save(
        {"model_state_dict": model.vision_encoder.state_dict()}, vision_ckpt_path
    )
    torch.save({"model_state_dict": model.text_encoder.state_dict()}, text_ckpt_path)

    # Test loading
    result = mmm.load_pretrained_weights(
        model,
        vision_checkpoint=str(vision_ckpt_path),
        text_checkpoint=str(text_ckpt_path),
        strict=False,
        allow_external=True,
    )

    assert result is model


def test_freeze_text_encoder(monkeypatch):
    """Test freezing text encoder parameters."""
    model = make_minimal_model(monkeypatch)

    # Freeze text encoder
    model.freeze_text_encoder()

    # Verify all text encoder params are frozen
    assert all(not p.requires_grad for p in model.text_encoder.parameters())


def test_enable_gradient_checkpointing(monkeypatch):
    """Test enabling gradient checkpointing."""
    model = make_minimal_model(monkeypatch)

    assert model.gradient_checkpointing is False
    model.enable_gradient_checkpointing()
    assert model.gradient_checkpointing is True


def test_get_model_info(monkeypatch):
    """Test getting model architecture information."""
    model = make_minimal_model(monkeypatch)

    info = model.get_model_info()

    assert "total_parameters" in info
    assert "trainable_parameters" in info
    assert "fusion_type" in info
    assert "use_double_loop" in info
    assert info["fusion_type"] == "early"
