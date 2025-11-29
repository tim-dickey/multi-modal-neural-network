"""Checkpoint management for model saving and loading."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set

import torch
import torch.nn as nn


class CheckpointManager:
    """Manages saving and loading of model checkpoints.

    Provides a unified interface for checkpoint operations with support for:
    - PyTorch .pt files
    - Safetensors format for secure loading
    - Best model tracking
    - Checkpoint rotation
    """

    # Default expected keys in a checkpoint
    DEFAULT_EXPECTED_KEYS: Set[str] = {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "epoch",
        "global_step",
        "best_val_loss",
        "config",
    }

    def __init__(
        self,
        checkpoint_dir: Path,
        logger: Optional[logging.Logger] = None,
        *,
        save_safetensors: bool = True,
        max_checkpoints: Optional[int] = None,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            logger: Optional logger instance
            save_safetensors: Whether to also save in safetensors format
            max_checkpoints: Maximum number of checkpoints to keep (None = unlimited)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.logger = logger or logging.getLogger(__name__)
        self.save_safetensors = save_safetensors
        self.max_checkpoints = max_checkpoints

        # Ensure directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        global_step: int,
        best_val_loss: float,
        config: Dict[str, Any],
        path: Optional[str] = None,
        *,
        is_best: bool = False,
    ) -> Path:
        """Save a checkpoint.

        Args:
            model: The model to save
            optimizer: The optimizer to save
            scheduler: The scheduler to save
            epoch: Current epoch number
            global_step: Current global step
            best_val_loss: Best validation loss seen
            config: Training configuration
            path: Optional explicit path to save to
            is_best: Whether this is the best model so far

        Returns:
            Path where checkpoint was saved
        """
        checkpoint = self._build_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            best_val_loss=best_val_loss,
            config=config,
        )

        if path is not None:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, save_path)
            self.logger.info("Checkpoint saved to %s", save_path)
            return save_path

        # Default saving strategy
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        # Also save in safetensors format if enabled
        if self.save_safetensors:
            self._save_safetensors(checkpoint["model_state_dict"], latest_path)

        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            self.logger.info("New best model saved to %s", best_path)

        self.logger.info("Checkpoint saved to %s", latest_path)

        # Cleanup old checkpoints if limit set
        if self.max_checkpoints is not None:
            self._cleanup_old_checkpoints()

        return latest_path

    def save_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        global_step: int,
        best_val_loss: float,
        config: Dict[str, Any],
        output_dir: Path,
    ) -> Path:
        """Save an epoch-specific checkpoint.

        Args:
            model: The model to save
            optimizer: The optimizer to save
            scheduler: The scheduler to save
            epoch: Current epoch number
            global_step: Current global step
            best_val_loss: Best validation loss seen
            config: Training configuration
            output_dir: Directory to save the checkpoint

        Returns:
            Path where checkpoint was saved
        """
        epoch_path = output_dir / f"checkpoint_{epoch:04d}.pt"
        return self.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            best_val_loss=best_val_loss,
            config=config,
            path=str(epoch_path),
        )

    def load(
        self,
        checkpoint_path: str,
        device: torch.device,
        *,
        expected_keys: Optional[Set[str]] = None,
        allow_external: bool = False,
    ) -> Dict[str, Any]:
        """Load a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to map tensors to
            expected_keys: Optional set of expected keys in checkpoint
            allow_external: Whether to allow loading from external paths

        Returns:
            Loaded checkpoint dictionary
        """
        from ..utils.safe_load import safe_load_checkpoint

        self.logger.info("Loading checkpoint from %s", checkpoint_path)

        if expected_keys is None:
            expected_keys = self.DEFAULT_EXPECTED_KEYS

        checkpoint = safe_load_checkpoint(
            checkpoint_path,
            map_location=device,
            expected_keys=expected_keys,
            allow_external=allow_external,
        )

        return checkpoint

    def restore_training_state(
        self,
        checkpoint: Dict[str, Any],
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
    ) -> Dict[str, Any]:
        """Restore training state from a checkpoint.

        Args:
            checkpoint: Loaded checkpoint dictionary
            model: Model to restore state to
            optimizer: Optimizer to restore state to
            scheduler: Scheduler to restore state to

        Returns:
            Dictionary with restored state info (epoch, global_step, best_val_loss)
        """
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        restored_state = {
            "epoch": checkpoint["epoch"] + 1,
            "global_step": checkpoint["global_step"],
            "best_val_loss": checkpoint["best_val_loss"],
        }

        self.logger.info("Resumed from epoch %d", restored_state["epoch"])
        return restored_state

    def _build_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        global_step: int,
        best_val_loss: float,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build checkpoint dictionary.

        Args:
            model: The model
            optimizer: The optimizer
            scheduler: The scheduler
            epoch: Current epoch
            global_step: Current global step
            best_val_loss: Best validation loss
            config: Configuration

        Returns:
            Checkpoint dictionary
        """
        return {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "config": config,
        }

    def _save_safetensors(
        self, model_state_dict: Dict[str, torch.Tensor], pt_path: Path
    ) -> None:
        """Save model state dict in safetensors format.

        Args:
            model_state_dict: Model state dictionary
            pt_path: Path of the .pt file (will use same name with .safetensors)
        """
        try:
            from safetensors.torch import save_file

            safetensors_path = pt_path.with_suffix(".safetensors")
            save_file(model_state_dict, str(safetensors_path))
            self.logger.info("Also saved model_state_dict to %s", safetensors_path)
        except ImportError:
            self.logger.debug("safetensors not installed, skipping safetensors save")
        except Exception as e:
            self.logger.debug("safetensors save failed: %s", e)

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints if exceeding max_checkpoints limit."""
        if self.max_checkpoints is None:
            return

        # Find all epoch checkpoints (checkpoint_XXXX.pt pattern)
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))

        # Keep the most recent ones
        if len(checkpoints) > self.max_checkpoints:
            for ckpt in checkpoints[: -self.max_checkpoints]:
                ckpt.unlink()
                self.logger.debug("Removed old checkpoint: %s", ckpt)

                # Also remove corresponding safetensors file if exists
                safetensors_path = ckpt.with_suffix(".safetensors")
                if safetensors_path.exists():
                    safetensors_path.unlink()

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to the latest checkpoint if it exists.

        Returns:
            Path to latest.pt or None if not found
        """
        latest_path = self.checkpoint_dir / "latest.pt"
        return latest_path if latest_path.exists() else None

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to the best checkpoint if it exists.

        Returns:
            Path to best.pt or None if not found
        """
        best_path = self.checkpoint_dir / "best.pt"
        return best_path if best_path.exists() else None

    def list_checkpoints(self) -> list[Path]:
        """List all available checkpoints.

        Returns:
            List of checkpoint paths
        """
        checkpoints = []

        # Add special checkpoints if they exist
        if (self.checkpoint_dir / "latest.pt").exists():
            checkpoints.append(self.checkpoint_dir / "latest.pt")
        if (self.checkpoint_dir / "best.pt").exists():
            checkpoints.append(self.checkpoint_dir / "best.pt")

        # Add epoch checkpoints
        checkpoints.extend(sorted(self.checkpoint_dir.glob("checkpoint_*.pt")))

        return checkpoints
