"""Main training loop for multi-modal neural network."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sized, Tuple, cast

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import create_dataloader, create_dataset_from_config
from ..data.selector import build_dataloaders
from ..models.multi_modal_model import MultiModalModel, create_multi_modal_model
from ..training.losses import MetaLoss, create_loss_function
from ..training.optimizer import (
    AdaptiveLRController,
    GradientClipper,
    create_optimizer,
    create_scheduler,
)
from ..utils.config import load_config, save_config, validate_config
from ..utils.logging import MetricsLogger, WandbLogger, log_model_info, setup_logger


class Trainer:
    """Main trainer class for multi-modal neural network.

    Supports two construction modes:
    1) Config-driven: provide `config_path` to build model and data loaders
    2) Injected objects: provide `model`, `train_loader`, `val_loader`, and `config`
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
        # (we'll replace it with the configured logger after paths are resolved)
        self.logger: logging.Logger = logging.getLogger("trainer")

        # Setup device with GPU/NPU detection (helpers will use imports)
        self._init_device_and_detection(device)

        # Paths, logging, and experiment setup
        self._init_paths_and_logging()

        # Resolve model and move to device
        self._init_model(model)

        # Resolve data loaders (selector-aware)
        self._init_data_loaders(train_loader, val_loader)

        # Training components: losses, optimizer, scheduler, clipper, adapters
        self._init_training_components()

        # Mixed precision and resume
        self._init_amp_and_resume(resume_from)

        # Persist effective config
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_config(self.config, str(self.output_dir / "config.yaml"))

    def _init_device_and_detection(self, device: Optional[str]) -> None:
        """Initialize `self.device` using config and hardware detection helpers."""
        from ..utils.gpu_utils import configure_device_for_training, detect_gpu_info
        from ..utils.npu_utils import detect_npu_info, get_best_available_device

        if device is None:
            device_config = self.config.get("hardware", {}).get("device", "auto")
            gpu_id = self.config.get("hardware", {}).get("gpu_id")
            prefer_npu = self.config.get("hardware", {}).get("prefer_npu", False)

            # Handle "auto" device selection
            if device_config == "auto":
                device_config = get_best_available_device(prefer_npu=prefer_npu)

                # Log what was detected
                if device_config == "cuda":
                    gpu_info = detect_gpu_info()
                    if gpu_info["available"]:
                        self.logger.info(
                            "Using CUDA GPU: %s", gpu_info["devices"][0]["name"]
                        )
                elif device_config in ["openvino", "ryzenai", "mps", "privateuseone"]:
                    npu_info = detect_npu_info()
                    self.logger.info("Using NPU: %s", npu_info.get("device_name"))
                else:
                    self.logger.warning(
                        "No GPU or NPU detected. Training will use CPU. "
                        "For GPU support, install PyTorch with CUDA. "
                        "For NPU support, install an appropriate SDK "
                        "(OpenVINO, DirectML, etc.)"
                    )

            # Handle NPU device strings
            if device_config in ["npu", "openvino", "ryzenai", "privateuseone"]:
                self.logger.info(
                    "NPU detected (%s), using CPU for training. "
                    "NPU can be used for optimized inference via ONNX export.",
                    device_config,
                )
                self.device = torch.device("cpu")
            elif device_config == "mps":
                self.device = torch.device(
                    "mps" if torch.backends.mps.is_available() else "cpu"
                )
            else:
                self.device = configure_device_for_training(
                    device=device_config, gpu_id=gpu_id, verbose=False
                )
        else:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")

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
        log_dir_path = Path(
            self.config.get("paths", {}).get("log_dir", str(self.output_dir / "logs"))
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir_path.mkdir(parents=True, exist_ok=True)

        # Setup logging (replace the temporary logger declared above)
        self.logger = setup_logger(
            name="trainer", log_file=str(log_dir_path / "training.log")
        )
        self.logger.info("Using device: %s", self.device)

        # Setup metrics logger
        experiment_name = self.config.get("logging", {}).get("experiment", "default")
        self.metrics_logger = MetricsLogger(
            log_dir=str(log_dir_path), experiment_name=experiment_name
        )

        # Setup wandb if enabled
        use_wandb = self.config.get("logging", {}).get("use_wandb", True)
        if use_wandb:
            project_name = self.config.get("logging", {}).get(
                "project", "multi-modal-net"
            )
            self.wandb_logger: Optional[WandbLogger] = WandbLogger(
                project=project_name,
                experiment=experiment_name,
                config=self.config,
                enabled=True,
            )
        else:
            self.wandb_logger = None

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
        self.train_loader: Any
        self.val_loader: Optional[Any]
        self.test_loader: Optional[Any]

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

        Split into smaller helpers to reduce cyclomatic complexity and make
        unit-testing individual initialization blocks easier.
        """
        self._create_criterion_and_meta()
        self._create_optimizer_and_scheduler()
        self._create_clipping_adaptive_and_state()

    def _create_criterion_and_meta(self) -> None:
        """Create loss functions used during training."""
        self.criterion = create_loss_function(self.config)
        if getattr(self.model, "use_double_loop", False):
            self.meta_criterion = MetaLoss()

    def _create_optimizer_and_scheduler(self) -> None:
        """Create optimizer and learning-rate scheduler."""
        self.optimizer = create_optimizer(self.model, self.config)
        steps_per_epoch = max(1, len(self.train_loader))
        self.scheduler, self.scheduler_update_freq = create_scheduler(
            self.optimizer, self.config, steps_per_epoch=steps_per_epoch
        )

    def _create_clipping_adaptive_and_state(self) -> None:
        """Setup gradient clipping, adaptive LR controller and training state."""
        max_grad_norm = self.config.get("training", {}).get("max_grad_norm", 1.0)
        self.grad_clipper = GradientClipper(max_norm=max_grad_norm)

        if getattr(self.model, "use_double_loop", False):
            self.adaptive_lr = AdaptiveLRController(
                base_lr=self.config.get("training", {}).get("inner_lr", 3e-4)
            )

        # Initial training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

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

            if (
                batch_idx % int(self.config.get("training", {}).get("log_interval", 10))
                == 0
            ):
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

    @torch.no_grad()  # type: ignore[misc]
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
        """Save model checkpoint.

        - If `path` is provided, save directly to that path.
        - Otherwise, save to default locations under `checkpoint_dir`.
        """
        checkpoint = {
            "epoch": self.current_epoch if epoch is None else epoch,
            "global_step": self.global_step if step is None else step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        if path is not None:
            ckpt_path = Path(path)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, ckpt_path)
            self.logger.info(f"Checkpoint saved to {ckpt_path}")
            return

        # Default saving strategy
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
        # Also save model_state_dict as a safetensors file for safer loading
        try:
            from safetensors.torch import save_file as _st_save

            safetensors_path = latest_path.with_suffix(".safetensors")
            # save_file expects a mapping of name -> tensor
            _st_save(checkpoint["model_state_dict"], str(safetensors_path))
            self.logger.info(f"Also saved model_state_dict to {safetensors_path}")
        except Exception:
            # If safetensors is not available or save fails, continue silently
            self.logger.debug("safetensors save skipped or failed; leaving .pt only")
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
        self.logger.info(f"Checkpoint saved to {latest_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")

        from ..utils.safe_load import safe_load_checkpoint
        # Allow callers to opt-in to loading from external paths via config
        allow_external = self.config.get("security", {}).get(
            "allow_external_checkpoints", False
        )

        checkpoint = safe_load_checkpoint(
            checkpoint_path,
            map_location=self.device,
            expected_keys={
                "model_state_dict",
                "optimizer_state_dict",
                "scheduler_state_dict",
                "epoch",
                "global_step",
                "best_val_loss",
                "config",
            },
            allow_external=allow_external,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        self.logger.info(f"Resumed from epoch {self.current_epoch}")
