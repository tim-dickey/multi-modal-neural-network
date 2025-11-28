"""Logging utilities for training."""

import contextlib
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def setup_logger(
    name: str = "multi_modal",
    log_file: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up logger with console and optional file handlers.

    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level

    Returns:
        Logger instance

    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers (close them first to avoid ResourceWarnings)
    if logger.handlers:
        for h in list(logger.handlers):
            with contextlib.suppress(OSError, ValueError):
                h.flush()
            with contextlib.suppress(OSError, ValueError):
                h.close()
        logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


class MetricsLogger:
    """Logger for training metrics."""

    def __init__(self, log_dir: str, experiment_name: str) -> None:
        """Initialize the metrics logger."""
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.txt"

        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metrics file
        with self.metrics_file.open("w", encoding="utf-8") as f:
            f.write(f"Experiment: {experiment_name}\n")
            # include timezone info
            f.write(f"Started: {datetime.now().astimezone().isoformat()}\n")
            f.write("-" * 80 + "\n")

    def log_metrics(self, step: int, metrics: dict[str, Any], prefix: str = "") -> None:
        """Log metrics to file.

        Args:
            step: Training step
            metrics: Dictionary of metric names and values
            prefix: Optional prefix for metric names

        """
        with self.metrics_file.open("a", encoding="utf-8") as f:
            f.write(f"\nStep {step}:\n")
            for key, value in metrics.items():
                metric_name = f"{prefix}{key}" if prefix else key
                if isinstance(value, float):
                    f.write(f"  {metric_name}: {value:.6f}\n")
                else:
                    f.write(f"  {metric_name}: {value}\n")

    def log_epoch(
        self,
        epoch: int,
        train_metrics: dict[str, Any],
        val_metrics: dict[str, Any] | None = None,
    ) -> None:
        """Log epoch summary."""
        with self.metrics_file.open("a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Epoch {epoch} Summary:\n")
            f.write(f"{'=' * 80}\n")

            f.write("Train metrics:\n")
            for key, value in train_metrics.items():
                f.write(f"  {key}: {value:.6f}\n")

            if val_metrics:
                f.write("\nValidation metrics:\n")
                for key, value in val_metrics.items():
                    f.write(f"  {key}: {value:.6f}\n")

            f.write("-" * 80 + "\n")


class WandbLogger:
    """Wrapper for Weights & Biases logging."""

    def __init__(
        self,
        project: str,
        experiment: str,
        config: dict[str, Any],
        *,
        enabled: bool = True,
    ) -> None:
        """Initialize a lightweight wandb wrapper."""
        self.enabled = enabled

        if self.enabled:
            try:
                import wandb  # noqa: PLC0415

                self.wandb = wandb
                self.run = wandb.init(project=project, name=experiment, config=config)
            except ImportError:
                logging.getLogger(__name__).warning(
                    "wandb not installed, disabling wandb logging",
                )
                self.enabled = False
            except (ValueError, RuntimeError) as e:
                logging.getLogger(__name__).warning(
                    "Failed to initialize wandb: %s",
                    e,
                )
                self.enabled = False

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics to wandb."""
        if self.enabled:
            self.wandb.log(metrics, step=step)

    def log_image(self, key: str, image: Any, step: int | None = None) -> None:
        """Log image to wandb."""
        if self.enabled:
            self.wandb.log({key: self.wandb.Image(image)}, step=step)

    def finish(self) -> None:
        """Finish wandb run."""
        if self.enabled:
            self.wandb.finish()


def log_model_info(logger: logging.Logger, model: Any) -> None:
    """Log model architecture information."""
    logger.info("%s", "=" * 80)
    logger.info("Model Information:")
    logger.info("%s", "=" * 80)

    if hasattr(model, "get_model_info"):
        info = model.get_model_info()
        for key, value in info.items():
            logger.info("  %s: %s", key, value)

    if hasattr(model, "get_num_parameters"):
        total_params = model.get_num_parameters(trainable_only=False)
        trainable_params = model.get_num_parameters(trainable_only=True)
        logger.info("  Total parameters: %d", total_params)
        logger.info("  Trainable parameters: %d", trainable_params)
        logger.info("  Non-trainable parameters: %d", total_params - trainable_params)

    logger.info("%s", "=" * 80)
