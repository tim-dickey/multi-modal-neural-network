"""Training state and component initialization."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils.logging import MetricsLogger, WandbLogger

from .training_defaults import TRAINING

# Type hints for IDE support - these are used in method signatures
# Import here to avoid circular imports at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .losses import MetaLoss  # noqa: F401
    from .optimizer import AdaptiveLRController, GradientClipper  # noqa: F401


class TrainingState:
    """Manages training state including epoch, step, and metrics tracking.

    This class encapsulates the mutable state that changes during training,
    providing a clean interface for state management and serialization.
    """

    def __init__(
        self,
        current_epoch: int = 0,
        global_step: int = 0,
        best_val_loss: float = float("inf"),
    ):
        """Initialize training state.

        Args:
            current_epoch: Starting epoch number
            global_step: Starting global step
            best_val_loss: Initial best validation loss
        """
        self.current_epoch = current_epoch
        self.global_step = global_step
        self.best_val_loss = best_val_loss

    def update_epoch(self, epoch: int) -> None:
        """Update current epoch."""
        self.current_epoch = epoch

    def increment_step(self) -> int:
        """Increment global step and return new value."""
        self.global_step += 1
        return self.global_step

    def update_best_loss(self, val_loss: float) -> bool:
        """Update best validation loss if improved.

        Args:
            val_loss: Current validation loss

        Returns:
            True if this is a new best, False otherwise
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            return True
        return False

    def restore_from_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Restore state from a checkpoint dictionary.

        Args:
            checkpoint: Checkpoint dictionary with state info
        """
        self.current_epoch = checkpoint.get("epoch", 0) + 1
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization.

        Returns:
            Dictionary representation of state
        """
        return {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }

    def __repr__(self) -> str:
        return (
            f"TrainingState(epoch={self.current_epoch}, "
            f"step={self.global_step}, best_loss={self.best_val_loss:.4f})"
        )


class LoggingManager:
    """Manages logging setup for training.

    Handles file logging, metrics logging, and wandb integration.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: Path,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize logging manager.

        Args:
            config: Configuration dictionary with 'logging' section
            output_dir: Output directory for logs
            logger: Optional pre-configured logger
        """
        self.config = config
        self.output_dir = output_dir
        self.log_dir = self._get_log_dir()

        # Initialize components
        self.logger = logger or self._setup_file_logger()
        self.metrics_logger = self._setup_metrics_logger()
        self.wandb_logger = self._setup_wandb_logger()

    def _get_log_dir(self) -> Path:
        """Get or create log directory."""
        log_dir = Path(
            self.config.get("paths", {}).get(
                "log_dir", str(self.output_dir / "logs")
            )
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def _setup_file_logger(self) -> logging.Logger:
        """Setup file-based logger."""
        from ..utils.logging import setup_logger

        return setup_logger(
            name="trainer", log_file=str(self.log_dir / "training.log")
        )

    def _setup_metrics_logger(self) -> MetricsLogger:
        """Setup metrics logger."""
        experiment_name = self.config.get("logging", {}).get("experiment", "default")
        return MetricsLogger(
            log_dir=str(self.log_dir), experiment_name=experiment_name
        )

    def _setup_wandb_logger(self) -> Optional[WandbLogger]:
        """Setup wandb logger if enabled."""
        use_wandb = self.config.get("logging", {}).get("use_wandb", True)

        if not use_wandb:
            return None

        project_name = self.config.get("logging", {}).get(
            "project", "multi-modal-net"
        )
        experiment_name = self.config.get("logging", {}).get("experiment", "default")

        return WandbLogger(
            project=project_name,
            experiment=experiment_name,
            config=self.config,
            enabled=True,
        )

    def log_info(self, message: str, *args: Any) -> None:
        """Log an info message."""
        self.logger.info(message, *args)

    def log_warning(self, message: str, *args: Any) -> None:
        """Log a warning message."""
        self.logger.warning(message, *args)

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log epoch summary.

        Args:
            epoch: Epoch number
            train_metrics: Training metrics
            val_metrics: Optional validation metrics
        """
        self.metrics_logger.log_epoch(epoch, train_metrics, val_metrics)

    def log_to_wandb(
        self, metrics: Dict[str, Any], step: Optional[int] = None
    ) -> None:
        """Log metrics to wandb if enabled.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        if self.wandb_logger:
            self.wandb_logger.log(metrics, step=step)

    def finish(self) -> None:
        """Finish logging (close wandb run, etc.)."""
        if self.wandb_logger:
            self.wandb_logger.finish()


class TrainingComponentsFactory:
    """Factory for creating training components (loss, optimizer, scheduler, etc.)."""

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_loader: DataLoader,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize factory.

        Args:
            model: The model being trained
            config: Configuration dictionary
            train_loader: Training data loader (for steps_per_epoch)
            logger: Optional logger
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.logger = logger or logging.getLogger(__name__)

    def create_criterion(self) -> nn.Module:
        """Create the loss function.

        Returns:
            Loss function module
        """
        from .losses import create_loss_function

        return create_loss_function(self.config)

    def create_meta_criterion(self) -> Optional[Any]:
        """Create meta-loss if model uses double-loop learning.

        Returns:
            MetaLoss instance or None
        """
        if getattr(self.model, "use_double_loop", False):
            from .losses import MetaLoss

            return MetaLoss()
        return None

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create the optimizer.

        Returns:
            Optimizer instance
        """
        from .optimizer import create_optimizer

        return create_optimizer(self.model, self.config)

    def create_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> Tuple[Any, str]:
        """Create learning rate scheduler.

        Args:
            optimizer: The optimizer

        Returns:
            Tuple of (scheduler, update_frequency)
        """
        from .optimizer import create_scheduler

        steps_per_epoch = max(1, len(self.train_loader))
        return create_scheduler(optimizer, self.config, steps_per_epoch=steps_per_epoch)

    def create_gradient_clipper(self) -> Any:
        """Create gradient clipper.

        Returns:
            GradientClipper instance
        """
        from .optimizer import GradientClipper

        max_grad_norm = self.config.get("training", {}).get(
            "max_grad_norm", TRAINING.max_grad_norm
        )
        return GradientClipper(max_norm=max_grad_norm)

    def create_adaptive_lr_controller(
        self,
    ) -> Optional[Any]:
        """Create adaptive LR controller if model uses double-loop learning.

        Returns:
            AdaptiveLRController instance or None
        """
        if getattr(self.model, "use_double_loop", False):
            from .optimizer import AdaptiveLRController

            base_lr = self.config.get("training", {}).get(
                "inner_lr", TRAINING.learning_rate
            )
            return AdaptiveLRController(base_lr=base_lr)
        return None

    def create_all(self) -> Dict[str, Any]:
        """Create all training components.

        Returns:
            Dictionary with all components
        """
        criterion = self.create_criterion()
        meta_criterion = self.create_meta_criterion()
        optimizer = self.create_optimizer()
        scheduler, scheduler_update_freq = self.create_scheduler(optimizer)
        grad_clipper = self.create_gradient_clipper()
        adaptive_lr = self.create_adaptive_lr_controller()

        return {
            "criterion": criterion,
            "meta_criterion": meta_criterion,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scheduler_update_freq": scheduler_update_freq,
            "grad_clipper": grad_clipper,
            "adaptive_lr": adaptive_lr,
        }
