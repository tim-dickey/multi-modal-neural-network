"""Safe-loading helpers for torch checkpoints.

This module provides a small defensive wrapper around `torch.load` to
centralize checks and optional safetensors support. It does not make
unpickling safe against malicious files, but it improves diagnostics
and encourages using `safetensors` when available.
"""
from typing import Optional, Set, Dict, Any
from pathlib import Path
import tempfile
import os

import torch


def _looks_like_state_dict(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    # Keys should be strings and values typically tensors
    return all(isinstance(k, str) for k in obj.keys())


def _get_trusted_roots() -> Set[Path]:
    """Return the set of trusted root directories for checkpoint loading."""
    try:
        repo_root = Path(__file__).resolve().parents[2]
    except Exception:
        repo_root = Path.cwd()

    return {repo_root.resolve(), Path(tempfile.gettempdir()).resolve()}


def _is_path_in_trusted_roots(resolved_path: Path, trusted_roots: Set[Path]) -> bool:
    """Check if a path is within any of the trusted root directories."""
    try:
        # Preferred (Python 3.9+): use is_relative_to for robust checks
        for tr in trusted_roots:
            if resolved_path.is_relative_to(tr):
                return True
        return False
    except Exception:
        # Fallback: compare commonpath
        try:
            rp = str(resolved_path.resolve())
            roots = [str(tr.resolve()) for tr in trusted_roots]
            return any(os.path.commonpath([rp, r]) == r for r in roots)
        except Exception:
            return False


def _validate_checkpoint_path(path: str, *, allow_external: bool = False) -> Path:
    """Validate checkpoint path for security concerns.

    Args:
        path: Path to the checkpoint file.
        allow_external: Whether to allow loading from untrusted paths.

    Returns:
        Resolved Path object.

    Raises:
        ValueError: If path is a remote URL or untrusted external path.
    """
    lower = str(path).lower()
    if lower.startswith("http://") or lower.startswith("https://"):
        raise ValueError("Refuse to load checkpoint from remote URL")

    p = Path(path)
    try:
        resolved = p.resolve()
    except Exception:
        resolved = p

    if not allow_external:
        trusted_roots = _get_trusted_roots()
        if not _is_path_in_trusted_roots(resolved, trusted_roots):
            raise ValueError(
                "Loading checkpoints from external/untrusted paths is disabled "
                "by default; set allow_external=True to override when necessary."
            )

    return resolved


def _normalize_map_location(map_location: Optional[object]) -> str:
    """Normalize map_location to a device string for safetensors."""
    if map_location is None:
        return "cpu"
    try:
        return map_location if isinstance(map_location, str) else str(map_location)
    except Exception:
        return "cpu"


def _load_safetensors(path: str, map_location: Optional[object]) -> Dict[str, Any]:
    """Load a checkpoint using the safetensors library.

    Args:
        path: Path to the .safetensors file.
        map_location: Device to load tensors to.

    Returns:
        Dictionary of loaded tensors.

    Raises:
        ValueError: If loading fails.
    """
    try:
        from safetensors.torch import load_file as _st_load

        st_device = _normalize_map_location(map_location)
        data = _st_load(path, device=st_device)

        if not isinstance(data, dict):
            raise ValueError("safetensors loader returned unexpected type")

        # Apply map_location if specified
        if map_location is not None:
            device = (
                map_location
                if isinstance(map_location, torch.device)
                else map_location
            )
            data = {k: v.to(device) for k, v in data.items()}

        return dict(data)
    except Exception as exc:
        raise ValueError(
            f"Failed to load safetensors checkpoint {path}: {exc}"
        ) from exc


def _load_torch_checkpoint(path: str, map_location: Optional[object]) -> Dict[str, Any]:
    """Load a checkpoint using torch.load.

    Args:
        path: Path to the .pt/.pth file.
        map_location: Device mapping for loaded tensors.

    Returns:
        Dictionary containing the checkpoint data.

    Raises:
        ValueError: If loading fails or result is not a dict.
    """
    try:
        loaded = torch.load(path, map_location=map_location)
    except Exception as exc:
        raise ValueError(f"Failed to load checkpoint {path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ValueError(
            f"Checkpoint {path} did not contain a dictionary-like checkpoint. "
            "Expected a dict with model/optimizer state_dicts."
        )

    return loaded


def _validate_checkpoint_keys(
    data: Dict[str, Any], path: str, expected_keys: Optional[Set[str]]
) -> None:
    """Validate that checkpoint contains expected keys.

    Args:
        data: Loaded checkpoint dictionary.
        path: Path to checkpoint (for error messages).
        expected_keys: Set of keys that must be present.

    Raises:
        ValueError: If any expected keys are missing.
    """
    if expected_keys is not None and not expected_keys.issubset(set(data.keys())):
        missing = expected_keys.difference(set(data.keys()))
        raise ValueError(f"Checkpoint {path} is missing expected keys: {missing}")


def safe_load_checkpoint(
    path: str,
    *,
    map_location: Optional[object] = None,
    expected_keys: Optional[Set[str]] = None,
    allow_external: bool = False,
) -> Dict[str, Any]:
    """Load a checkpoint with basic validation and optional safetensors support.

    Args:
        path: Path to checkpoint file. If file ends with `.safetensors` this
            attempts to use the `safetensors` loader.
        map_location: Forwarded to the underlying loader.
        expected_keys: If provided, the returned dict must contain these keys.
        allow_external: Allow loading from paths outside trusted directories.

    Returns:
        The loaded checkpoint dict.

    Raises:
        ValueError: if loaded object is not a dict or missing expected keys.
    """
    # Validate path security
    _validate_checkpoint_path(path, allow_external=allow_external)

    # Load using appropriate loader
    if str(path).endswith(".safetensors"):
        data = _load_safetensors(path, map_location)
    else:
        data = _load_torch_checkpoint(path, map_location)

    # Validate keys if required
    _validate_checkpoint_keys(data, path, expected_keys)

    return data
