"""Default values for training configuration.

Centralizes all magic numbers and default values used throughout the training
module to improve configurability and maintainability.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingDefaults:
    """Default training hyperparameters."""

    # Epochs and steps
    max_epochs: int = 50
    warmup_steps: int = 1000
    log_interval: int = 10
    save_steps: int = 5000
    eval_steps: int = 10000

    # Learning rate
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01

    # Optimizer
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    warmup_start_factor: float = 0.01
    warmup_end_factor: float = 1.0

    # Gradient clipping
    max_grad_norm: float = 1.0
    grad_norm_type: float = 2.0
    gradient_clip: float = 0.0  # 0 = disabled

    # Adaptive LR controller
    adaptive_lr_min_scale: float = 0.1
    adaptive_lr_max_scale: float = 2.0


@dataclass(frozen=True)
class DataDefaults:
    """Default data loading parameters."""

    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    shuffle_train: bool = True


@dataclass(frozen=True)
class LossDefaults:
    """Default loss function parameters."""

    label_smoothing: float = 0.0
    contrastive_temperature: float = 0.07
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    meta_loss_weight: float = 0.1


@dataclass(frozen=True)
class CheckpointDefaults:
    """Default checkpoint settings."""

    max_checkpoints: int | None = None  # None = unlimited
    save_safetensors: bool = True


# Global default instances for convenient access
TRAINING = TrainingDefaults()
DATA = DataDefaults()
LOSS = LossDefaults()
CHECKPOINT = CheckpointDefaults()
