"""Main training loop for multi-modal neural network."""

import logging
from collections.abc import Sized
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import create_dataloader, create_dataset_from_config
from ..data.selector import build_dataloaders
from ..models.multi_modal_model import MultiModalModel, create_multi_modal_model
from ..utils.config import load_config, save_config, validate_config
from ..utils.logging import log_model_info
from .checkpoint_manager import CheckpointManager
from .device_manager import DeviceManager
from .training_state import LoggingManager, TrainingComponentsFactory, TrainingState


class Trainer:
    """Main trainer class for multi-modal neural network.

    Supports two construction modes:
    1) Config-driven: provide `config_path` to build model and data loaders
    2) Injected objects: provide `model`, `train_loader`, `val_loader`, and `config`

    Uses decomposed components for better maintainability:
    - DeviceManager: Handles device detection and configuration
    - CheckpointManager: Manages checkpoint saving/loading
    - LoggingManager: Handles logging setup
    - TrainingState: Tracks training progress
    - TrainingComponentsFactory: Creates optimizer, scheduler, etc.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        *,
        model: Optional[MultiModalModel] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        resume_from: Optional[str] = None,
        device: Optional[str] = None,
    ):
        # Resolve configuration
        if config is None:
            if config_path is None:
                raise ValueError("Either `config_path` or `config` must be provided")
            self.config = load_config(config_path)
        else:
            self.config = config

        validate_config(self.config)

        # Predeclare a logger so it can be used during early device detection
        self._temp_logger: logging.Logger = logging.getLogger("trainer")

        # Predeclare data loader and state attributes for type checkers
        self.train_loader: Any = None
        self.val_loader: Optional[Any] = None
        self.test_loader: Optional[Any] = None
        self.current_epoch: int = 0
        self.global_step: int = 0
        self.best_val_loss: float = float("inf")
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None

        # Setup device with GPU/NPU detection using DeviceManager
        self.device_manager = DeviceManager(
            config=self.config,
            device_override=device,
            logger=self._temp_logger,
        )
        self.device = self.device_manager.device

        # Paths, logging, and experiment setup using LoggingManager
        self._init_paths_and_logging()

        # Resolve model and move to device
        self._init_model(model)

        # Resolve data loaders (selector-aware)
        self._init_data_loaders(train_loader, val_loader)

        # Training components: losses, optimizer, scheduler, clipper, adapters
        self._init_training_components()

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            logger=self.logger,
        )

        # Mixed precision and resume
        self._init_amp_and_resume(resume_from)

        # Persist effective config
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_config(self.config, str(self.output_dir / "config.yaml"))

    def _init_paths_and_logging(self) -> None:
        """Create output, checkpoint, and log directories and initialize loggers."""
        explicit_output = self.config.get("output_dir")
        self.output_dir = Path(
            explicit_output
            if explicit_output is not None
            else self.config.get("paths", {}).get("output_dir", "./outputs")
        )
        self.checkpoint_dir = Path(
            self.config.get("paths", {}).get(
                "checkpoint_dir", str(self.output_dir / "checkpoints")
            )
        )

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Use LoggingManager for all logging setup
        self.logging_manager = LoggingManager(
            config=self.config,
            output_dir=self.output_dir,
        )
        self.logger = self.logging_manager.logger
        self.metrics_logger = self.logging_manager.metrics_logger
        self.wandb_logger = self.logging_manager.wandb_logger

        self.logger.info("Using device: %s", self.device)

    def _init_model(self, model: Optional[MultiModalModel]) -> None:
        """Create or accept provided model and move it to device."""
        if model is None:
            self.logger.info("Creating model...")
            self.model = create_multi_modal_model(self.config)
        else:
            self.model = model
        self.model.to(self.device)
        log_model_info(self.logger, self.model)

    def _init_data_loaders(
        self, train_loader: Optional[DataLoader], val_loader: Optional[DataLoader]
    ) -> None:
        """Initialize training/validation/test loaders from selector or legacy API.

        This method dispatches to focused helper methods to keep branching
        logic small and testable.
        """
        if train_loader is None and val_loader is None:
            data_section = self.config.get("data", {})
            if "datasets" in data_section:
                self._build_selector_loaders()
            else:
                self._build_legacy_loaders()
        else:
            self._assign_injected_loaders(train_loader, val_loader)

    def _build_selector_loaders(self) -> None:
        """Build train/val/test loaders using the selector API."""
        self.logger.info("Building dataloaders via selector...")
        self.train_loader, self.val_loader, self.test_loader = build_dataloaders(
            self.config
        )
        self.logger.info(
            "Selector data built: train=%d val=%d test=%d",
            len(self.train_loader),
            len(self.val_loader) if self.val_loader else 0,
            len(self.test_loader) if self.test_loader else 0,
        )

    def _build_legacy_loaders(self) -> None:
        """Build train/val loaders using the legacy dataset API."""
        self.logger.info("Loading datasets (legacy path)...")
        train_dataset, val_dataset = create_dataset_from_config(self.config)
        data_config = self.config.get("data", {})
        self.train_loader = create_dataloader(
            train_dataset,
            batch_size=data_config.get("batch_size", 32),
            num_workers=data_config.get("num_workers", 4),
            shuffle=True,
            pin_memory=data_config.get("pin_memory", True),
        )
        self.val_loader = create_dataloader(
            val_dataset,
            batch_size=data_config.get("batch_size", 32),
            num_workers=data_config.get("num_workers", 4),
            shuffle=False,
            pin_memory=data_config.get("pin_memory", True),
        )
        self.test_loader = None
        n_train = (
            len(cast(Sized, train_dataset))
            if hasattr(train_dataset, "__len__")
            else "unknown"
        )
        n_val = (
            len(cast(Sized, val_dataset))
            if hasattr(val_dataset, "__len__")
            else "unknown"
        )
        self.logger.info("Train samples: %s", n_train)
        self.logger.info("Val samples: %s", n_val)

    def _assign_injected_loaders(
        self, train_loader: Optional[DataLoader], val_loader: Optional[DataLoader]
    ) -> None:
        """Accept externally provided loaders (legacy/injected API)."""
        self.train_loader = train_loader if train_loader is not None else []
        self.val_loader = val_loader
        self.test_loader = None

    def _init_training_components(self) -> None:
        """Create loss, optimizer, scheduler, gradient clipping and adaptive LR.

        Uses TrainingComponentsFactory for consistent component creation.
        """
        factory = TrainingComponentsFactory(
            model=self.model,
            config=self.config,
            train_loader=self.train_loader,
            logger=self.logger,
        )

        components = factory.create_all()

        self.criterion = components["criterion"]
        self.meta_criterion = components["meta_criterion"]
        self.optimizer = components["optimizer"]
        self.scheduler = components["scheduler"]
        self.scheduler_update_freq = components["scheduler_update_freq"]
        self.grad_clipper = components["grad_clipper"]
        self.adaptive_lr = components["adaptive_lr"]

        # Initialize training state using TrainingState
        self.training_state = TrainingState()
        self.current_epoch = self.training_state.current_epoch
        self.global_step = self.training_state.global_step
        self.best_val_loss = self.training_state.best_val_loss

    def _init_amp_and_resume(self, resume_from: Optional[str]) -> None:
        """Initialize mixed-precision utilities and optionally resume from checkpoint."""
        # Mixed precision training
        self.use_amp = (
            self.config.get("training", {}).get("mixed_precision", "bf16") is not None
        )
        if self.use_amp:
            # Use the modern torch.amp API. Only enable GradScaler for CUDA devices.
            if getattr(self.device, "type", "cpu") == "cuda":
                # Use the device-aware GradScaler API
                # Use the cuda-specific GradScaler in torch.cuda.amp for compatibility
                self.scaler = torch.cuda.amp.GradScaler()
            else:
                # No CUDA device: avoid creating a CUDA GradScaler
                self.scaler = None

        # Resume from checkpoint if provided
        if resume_from is not None:
            self.load_checkpoint(resume_from)

    def train(self) -> None:
        """Main training loop."""
        # Support both `max_epochs` and `num_epochs`
        max_epochs = self.config.get("training", {}).get(
            "max_epochs",
            self.config.get("training", {}).get("num_epochs", 50),
        )

        self.logger.info("Starting training...")
        self.logger.info(f"Training for {max_epochs} epochs")

        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch

            # Train one epoch
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Log epoch summary
            self.logger.info(f"\nEpoch {epoch} Summary:")
            self.logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
            self.logger.info(f"  Train Acc:  {train_metrics['accuracy']:.4f}")
            self.logger.info(f"  Val Loss:   {val_metrics['loss']:.4f}")
            self.logger.info(f"  Val Acc:    {val_metrics['accuracy']:.4f}")

            self.metrics_logger.log_epoch(epoch, train_metrics, val_metrics)

            # Save checkpoint
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.save_checkpoint(is_best=True)
                self.logger.info(
                    f"  New best model! Val loss: {self.best_val_loss:.4f}"
                )

            # Always save epoch checkpoint for test expectations (checkpoint_*.pt)
            epoch_ckpt = self.output_dir / f"checkpoint_{epoch:04d}.pt"
            self.save_checkpoint(
                path=str(epoch_ckpt), epoch=epoch, step=self.global_step
            )

            # Optional periodic additional checkpointing
            save_every = self.config.get("training", {}).get("save_steps")
            if (
                save_every is not None
                and save_every > 0
                and (epoch + 1) % save_every == 0
            ):
                self.save_checkpoint(is_best=False)

        self.logger.info("Training completed!")
        if self.wandb_logger:
            self.wandb_logger.finish()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Guard: empty data loader
        if not hasattr(self, "train_loader") or self.train_loader is None:
            raise ValueError("Trainer.train_loader is not initialized.")
        if len(self.train_loader) == 0:
            raise ValueError(
                "Training data loader is empty. Check your config:\n"
                "  - data.train_path exists and has files\n"
                "  - data.batch_size > 0\n"
                "  - dataset split is not empty\n"
            )

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {
                k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            batch = self._normalize_batch(batch)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(
                images=batch.get("images"),
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
            )
            logits = outputs["logits"]
            loss = self.criterion(logits, batch["labels"])

            loss.backward()

            # Optional gradient clipping
            clip_val = self.config.get("training", {}).get("gradient_clip", 0.0)
            if clip_val and clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)

            self.optimizer.step()

            # Metrics
            total_loss += float(loss.item())
            preds = logits.argmax(dim=-1)
            total_correct += int((preds == batch["labels"]).sum().item())
            total_samples += int(batch["labels"].size(0))

            # Ensure a sane log interval (avoid zero or non-int config values)
            log_interval = max(1, int(self.config.get("training", {}).get("log_interval", 10)))
            if (batch_idx % log_interval) == 0:
                # Use logger formatting to keep line lengths short
                self.logger.info(
                    "Epoch %d [%d/%d] Loss: %.4f",
                    epoch,
                    batch_idx,
                    len(self.train_loader),
                    loss.item(),
                )

        # Safe averaging
        num_batches = max(1, len(self.train_loader))
        metrics = {
            "loss": total_loss / num_batches,
            "accuracy": (total_correct / total_samples) if total_samples > 0 else 0.0,
        }
        return metrics

    def _normalize_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize batch keys to a consistent schema expected by the model/trainer.

        Accepts either image/label or images/labels keys and returns a unified dict.
        """
        normalized = dict(batch)
        if "images" not in normalized and "image" in normalized:
            normalized["images"] = normalized.pop("image")
        if "labels" not in normalized and "label" in normalized:
            normalized["labels"] = normalized.pop("label")
        return normalized

    def train_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single training step."""
        batch = self._normalize_batch(batch)
        # Prepare inputs
        images = batch.get("images")
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        labels = batch.get("labels")

        # Forward pass
        outputs = self.model(
            images=images, input_ids=input_ids, attention_mask=attention_mask
        )

        logits = outputs["logits"]

        # Compute loss
        loss = self.criterion(logits, labels)

        # Add meta-loss if using double-loop learning
        if (
            self.model.use_double_loop
            and "meta_info" in outputs
            and outputs["meta_info"] is not None
        ):
            loss = self.meta_criterion(loss, outputs["meta_info"])

        # Compute accuracy
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == labels).float().mean()

        return loss, accuracy

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation loop."""
        # If no validation loader, return neutral metrics
        if self.val_loader is None:
            return {"loss": 0.0, "accuracy": 0.0}

        self.model.eval()

        total_loss = 0.0
        total_correct: float = 0.0
        total_samples = 0

        for batch in tqdm(self.val_loader, desc="Validating"):
            # Move batch to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            batch = self._normalize_batch(batch)

            # Forward pass
            outputs = self.model(
                images=batch.get("images"),
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
            )

            logits = outputs["logits"]
            labels = batch.get("labels")

            if labels is None:
                continue

            # Compute loss
            loss = self.criterion(logits, labels)

            # Compute accuracy
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == labels).float().sum()

            total_loss += loss.item()
            total_correct += accuracy.item()
            total_samples += labels.size(0)

        metrics = {
            "loss": total_loss / len(self.val_loader),
            "accuracy": total_correct / total_samples,
        }

        # Log to wandb
        if self.wandb_logger:
            self.wandb_logger.log(
                {"val/loss": metrics["loss"], "val/accuracy": metrics["accuracy"]},
                step=self.global_step,
            )

        return metrics

    def save_checkpoint(
        self,
        path: Optional[str] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        *,
        is_best: bool = False,
    ) -> None:
        """Save model checkpoint using CheckpointManager.

        - If `path` is provided, save directly to that path.
        - Otherwise, save to default locations under `checkpoint_dir`.
        """
        self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch if epoch is None else epoch,
            global_step=self.global_step if step is None else step,
            best_val_loss=self.best_val_loss,
            config=self.config,
            path=path,
            is_best=is_best,
        )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint using CheckpointManager."""
        # Allow callers to opt-in to loading from external paths via config
        allow_external = self.config.get("security", {}).get(
            "allow_external_checkpoints", False
        )

        checkpoint = self.checkpoint_manager.load(
            checkpoint_path=checkpoint_path,
            device=self.device,
            allow_external=allow_external,
        )

        restored_state = self.checkpoint_manager.restore_training_state(
            checkpoint=checkpoint,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

        # Update trainer state
        self.current_epoch = restored_state["epoch"]
        self.global_step = restored_state["global_step"]
        self.best_val_loss = restored_state["best_val_loss"]

        # Also update TrainingState
        self.training_state.current_epoch = self.current_epoch
        self.training_state.global_step = self.global_step
        self.training_state.best_val_loss = self.best_val_loss
