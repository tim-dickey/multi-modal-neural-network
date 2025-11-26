"""Main training loop for multi-modal neural network."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import create_dataloader, create_dataset_from_config
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

        # Setup device with GPU/NPU detection
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
                if device_config == 'cuda':
                    gpu_info = detect_gpu_info()
                    if gpu_info['available']:
                        self.logger.info(f"Using CUDA GPU: {gpu_info['devices'][0]['name']}")
                elif device_config in ['openvino', 'ryzenai', 'mps', 'privateuseone']:
                    npu_info = detect_npu_info()
                    self.logger.info(f"Using NPU: {npu_info['device_name']}")
                else:
                    self.logger.warning("No GPU or NPU detected. Training will use CPU. "
                                      "For GPU support, install PyTorch with CUDA. "
                                      "For NPU support, install appropriate SDK (OpenVINO, DirectML, etc.)")
            
            # Handle NPU device strings
            if device_config in ['npu', 'openvino', 'ryzenai', 'privateuseone']:
                # NPU detected - use CPU for now as PyTorch NPU support is limited
                # Users can implement custom NPU inference in evaluate mode
                self.logger.info(f"NPU detected ({device_config}), using CPU for training. "
                               "NPU can be used for optimized inference via ONNX export.")
                self.device = torch.device('cpu')
            elif device_config == 'mps':
                # Apple Silicon
                self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            else:
                self.device = configure_device_for_training(
                    device=device_config, 
                    gpu_id=gpu_id,
                    verbose=False  # We'll log it ourselves below
                )
        else:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Resolve important paths with sensible fallbacks
        # Prefer explicit top-level `output_dir` override if present
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

        # Setup logging
        self.logger = setup_logger(name="trainer", log_file=str(log_dir_path / "training.log"))
        self.logger.info(f"Using device: {self.device}")

        # Setup metrics logger
        experiment_name = self.config.get("logging", {}).get("experiment", "default")
        self.metrics_logger = MetricsLogger(log_dir=str(log_dir_path), experiment_name=experiment_name)

        # Setup wandb if enabled
        use_wandb = self.config.get("logging", {}).get("use_wandb", True)
        if use_wandb:
            project_name = self.config.get("logging", {}).get("project", "multi-modal-net")
            self.wandb_logger: Optional[WandbLogger] = WandbLogger(
                project=project_name, experiment=experiment_name, config=self.config, enabled=True
            )
        else:
            self.wandb_logger = None

        # Resolve model
        if model is None:
            self.logger.info("Creating model...")
            self.model = create_multi_modal_model(self.config)
        else:
            self.model = model
        self.model.to(self.device)
        log_model_info(self.logger, self.model)

        # Resolve data loaders
        # Declare loader attributes once for consistent typing
        self.train_loader: Any
        self.val_loader: Optional[Any]
        if train_loader is None and val_loader is None:
            self.logger.info("Loading datasets...")
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
            try:
                self.logger.info(f"Train samples: {len(train_dataset)}")  # type: ignore
                self.logger.info(f"Val samples: {len(val_dataset)}")  # type: ignore
            except Exception:
                pass
        else:
            # Use provided loaders as-is; do not auto-create datasets
            self.train_loader = train_loader if train_loader is not None else []
            self.val_loader = val_loader

        # Create loss function
        self.criterion = create_loss_function(self.config)
        if getattr(self.model, "use_double_loop", False):
            self.meta_criterion = MetaLoss()

        # Create optimizer and scheduler
        self.optimizer = create_optimizer(self.model, self.config)
        steps_per_epoch = max(1, len(self.train_loader))
        self.scheduler, self.scheduler_update_freq = create_scheduler(
            self.optimizer, self.config, steps_per_epoch=steps_per_epoch
        )

        # Gradient clipping
        max_grad_norm = self.config.get("training", {}).get("max_grad_norm", 1.0)
        self.grad_clipper = GradientClipper(max_norm=max_grad_norm)

        # Adaptive LR controller for double-loop learning
        if getattr(self.model, "use_double_loop", False):
            self.adaptive_lr = AdaptiveLRController(
                base_lr=self.config.get("training", {}).get("inner_lr", 3e-4)
            )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Mixed precision training
        self.use_amp = self.config.get("training", {}).get("mixed_precision", "bf16") is not None
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        # Resume from checkpoint if provided
        if resume_from is not None:
            self.load_checkpoint(resume_from)

        # Persist effective config
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_config(self.config, str(self.output_dir / "config.yaml"))

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
            train_metrics = self.train_epoch()

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
            self.save_checkpoint(path=str(epoch_ckpt), epoch=epoch, step=self.global_step)

            # Optional periodic additional checkpointing
            save_every = self.config.get("training", {}).get("save_steps")
            if save_every is not None and save_every > 0 and (epoch + 1) % save_every == 0:
                self.save_checkpoint(is_best=False)

        self.logger.info("Training completed!")
        if self.wandb_logger:
            self.wandb_logger.finish()

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_correct: float = 0.0
        total_samples = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    loss, accuracy = self.train_step(batch)
            else:
                loss, accuracy = self.train_step(batch)

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = self.grad_clipper(self.model.parameters())
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                grad_norm = self.grad_clipper(self.model.parameters())
                self.optimizer.step()

            # Update scheduler
            if self.scheduler_update_freq == "step":
                self.scheduler.step()

            # Update metrics
            total_loss += loss.item()
            label_tensor = batch.get("labels") if "labels" in batch else batch.get("label")
            if isinstance(label_tensor, torch.Tensor):
                bs = label_tensor.size(0)
                total_correct += accuracy.item() * bs
                total_samples += bs

            # Log progress
            avg_loss = total_loss / (batch_idx + 1)
            avg_acc = total_correct / total_samples
            progress_bar.set_postfix(
                {
                    "loss": f"{avg_loss:.4f}",
                    "acc": f"{avg_acc:.4f}",
                    "lr": f'{self.optimizer.param_groups[0]["lr"]:.6f}',
                }
            )

            # Log to wandb
            if (
                self.wandb_logger
                and batch_idx % self.config.get("logging", {}).get("log_every", 50) == 0
            ):
                self.wandb_logger.log(
                    {
                        "train/loss": loss.item(),
                        "train/accuracy": accuracy.item(),
                        "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                        "train/gradient_norm": grad_norm,
                    },
                    step=self.global_step,
                )

            self.global_step += 1

        # Update scheduler if per-epoch
        if self.scheduler_update_freq == "epoch":
            self.scheduler.step()

        return {
            "loss": total_loss / len(self.train_loader),
            "accuracy": total_correct / total_samples,
        }

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
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
        self.logger.info(f"Checkpoint saved to {latest_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        self.logger.info(f"Resumed from epoch {self.current_epoch}")
