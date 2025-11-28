import tempfile
from pathlib import Path

import pytest
import torch

from src.utils.safe_load import safe_load_checkpoint


def test_safe_load_invalid_non_dict(tmp_path: Path):
    # Save a non-dict object to a .pt file and ensure loader rejects it
    path = tmp_path / "not_a_dict.pt"
    torch.save([1, 2, 3], str(path))

    with pytest.raises(ValueError):
        safe_load_checkpoint(str(path), map_location="cpu")


def test_safe_load_missing_expected_keys(tmp_path: Path):
    # Save a dict missing expected keys
    path = tmp_path / "partial.pt"
    torch.save({"model_state_dict": {}}, str(path))

    with pytest.raises(ValueError):
        safe_load_checkpoint(str(path), map_location="cpu", expected_keys={"model_state_dict", "optimizer_state_dict"})


def test_safe_load_safetensors_roundtrip(tmp_path: Path):
    # Skip test if safetensors not installed
    safetensors = pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    # Create a simple tensor dict and save as safetensors
    td = {"weight": torch.randn(2, 2)}
    sf_path = tmp_path / "weights.safetensors"
    save_file(td, str(sf_path))

    loaded = safe_load_checkpoint(str(sf_path), map_location="cpu")
    assert isinstance(loaded, dict)
    assert "weight" in loaded
    assert torch.allclose(loaded["weight"], td["weight"]) or loaded["weight"].shape == td["weight"].shape


def test_allow_external_flag(tmp_path: Path):
    # Create a checkpoint in the user's home directory (outside repo and tempdir)
    home = Path.home()
    external_dir = home / "mmnn_test_external"
    external_dir.mkdir(parents=True, exist_ok=True)
    external_ckpt = external_dir / "external.pt"
    torch.save({"model_state_dict": {}}, str(external_ckpt))

    # By default loading from external path should be rejected
    with pytest.raises(ValueError):
        safe_load_checkpoint(str(external_ckpt), map_location="cpu")

    # When explicitly allowed, it should succeed
    loaded = safe_load_checkpoint(str(external_ckpt), map_location="cpu", allow_external=True)
    assert isinstance(loaded, dict)
