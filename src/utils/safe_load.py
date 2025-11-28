"""Safe-loading helpers for torch checkpoints.

This module provides a small defensive wrapper around `torch.load` to
centralize checks and optional safetensors support. It does not make
unpickling safe against malicious files, but it improves diagnostics
and encourages using `safetensors` when available.
"""
from typing import Optional, Set, Dict, Any

import torch


def _looks_like_state_dict(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    # Keys should be strings and values typically tensors
    return all(isinstance(k, str) for k in obj.keys())


def safe_load_checkpoint(
    path: str,
    *,
    map_location: Optional[object] = None,
    expected_keys: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Load a checkpoint with basic validation and optional safetensors support.

    Args:
        path: Path to checkpoint file. If file ends with `.safetensors` this
            attempts to use the `safetensors` loader.
        map_location: Forwarded to the underlying loader.
        expected_keys: If provided, the returned dict must contain these keys.

    Returns:
        The loaded checkpoint dict.

    Raises:
        ValueError: if loaded object is not a dict or missing expected keys.
    """
    # Prefer safetensors loader for .safetensors files when available
    if str(path).endswith(".safetensors"):
        try:
            from safetensors.torch import load_file as _st_load

            data = _st_load(path, device=map_location if map_location is not None else "cpu")
            # safetensors loaders return a mapping of tensors; normalize to dict
            if not isinstance(data, dict):
                raise ValueError("safetensors loader returned unexpected type")
            # Convert to CPU-backed tensors if map_location provided
            if map_location is not None:
                device = map_location if isinstance(map_location, torch.device) else map_location
                data = {k: v.to(device) for k, v in data.items()}
            return dict(data)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Failed to load safetensors checkpoint {path}: {exc}") from exc

    # Fallback to torch.load for normal .pt/.pth files
    try:
        loaded = torch.load(path, map_location=map_location)
    except Exception as exc:  # pragma: no cover - surface user-friendly error
        raise ValueError(f"Failed to load checkpoint {path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ValueError(
            f"Checkpoint {path} did not contain a dictionary-like checkpoint. "
            "Expected a dict with model/optimizer state_dicts."
        )

    if expected_keys is not None and not expected_keys.issubset(set(loaded.keys())):
        missing = expected_keys.difference(set(loaded.keys()))
        raise ValueError(f"Checkpoint {path} is missing expected keys: {missing}")

    return loaded
