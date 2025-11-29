import torch
import types

import pytest

from src.models import multi_modal_model as mmm


class DummyEncoder(torch.nn.Module):
    def __init__(self, hidden_dim=32, seq_len=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

    def forward(self, x, **kwargs):
        batch = x.shape[0]
        cls = torch.randn(batch, self.hidden_dim, device=x.device)
        features = torch.randn(batch, self.seq_len, self.hidden_dim, device=x.device)
        return cls, features


class DummyFusion(torch.nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, vision_features=None, text_features=None, **kwargs):
        # Concatenate on last dim
        fused = vision_features + text_features
        return fused, None


def make_minimal_model(monkeypatch):
    # Monkeypatch creators to simple dummies
    monkeypatch.setattr(mmm, "create_vision_encoder", lambda cfg: DummyEncoder(hidden_dim=32))
    monkeypatch.setattr(mmm, "create_text_encoder", lambda cfg: DummyEncoder(hidden_dim=32))
    monkeypatch.setattr(mmm, "create_fusion_layer", lambda cfg: DummyFusion(hidden_dim=32))
    monkeypatch.setattr(mmm, "create_double_loop_controller", lambda cfg: (lambda **k: {"ok": True}))
    monkeypatch.setattr(mmm, "create_task_head", lambda cfg: mmm.MultiTaskHead(hidden_dim=32, tasks={"default": {}}))

    cfg = {"model": {"vision_encoder": {}, "text_encoder": {}, "fusion": {"hidden_dim": 32}, "double_loop": {}, "heads": {}}, "training": {}}
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
