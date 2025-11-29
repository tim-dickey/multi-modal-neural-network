import importlib
import types
import torch

from src.utils import gpu_utils, npu_utils


def test_detect_gpu_info_no_cuda(monkeypatch):
    # Simulate no CUDA available
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    info = gpu_utils.detect_gpu_info()
    assert info["available"] is False
    # get_optimal_device with prefer_gpu False
    assert gpu_utils.get_optimal_device(prefer_gpu=False) == "cpu"


def test_handle_device_string_and_override(monkeypatch):
    # Simulate CUDA unavailable
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert gpu_utils._ensure_valid_device_string("cuda:0") == "cpu"

    # Now simulate available with one device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert gpu_utils._handle_gpu_id_override(0) == "cuda:0"
    # request out-of-range -> fallback
    assert gpu_utils._handle_gpu_id_override(5) == "cuda:0"


def test_npu_external_info_and_best_device(monkeypatch):
    # External npu info mapping
    info = npu_utils._external_npu_info("Google Coral Edge TPU")
    assert info["available"] is True
    assert info["recommended_device"] == "edge_tpu"

    # get_best_available_device returns a device string (cpu/mps/openvino/cuda/etc.)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # Simulate no mps
    monkeypatch.setattr(torch.backends, "mps", types.SimpleNamespace(is_available=lambda: False))
    dev = npu_utils.get_best_available_device()
    assert isinstance(dev, str)
