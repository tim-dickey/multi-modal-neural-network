"""Training utilities."""

from .checkpoint_manager import CheckpointManager
from .device_manager import DeviceManager
from .losses import (
    ContrastiveLoss,
    CrossEntropyLoss,
    FocalLoss,
    MetaLoss,
    MultiTaskLoss,
    create_loss_function,
)
from .optimizer import (
    AdaptiveLRController,
    GradientClipper,
    create_optimizer,
    create_scheduler,
)
from .trainer import Trainer
from .training_defaults import (
    CHECKPOINT,
    DATA,
    LOSS,
    TRAINING,
    CheckpointDefaults,
    DataDefaults,
    LossDefaults,
    TrainingDefaults,
)
from .training_state import LoggingManager, TrainingComponentsFactory, TrainingState

__all__ = [
    # Main trainer
    "Trainer",
    # Decomposed components
    "DeviceManager",
    "CheckpointManager",
    "TrainingState",
    "LoggingManager",
    "TrainingComponentsFactory",
    # Defaults
    "TRAINING",
    "DATA",
    "LOSS",
    "CHECKPOINT",
    "TrainingDefaults",
    "DataDefaults",
    "LossDefaults",
    "CheckpointDefaults",
    # Losses
    "CrossEntropyLoss",
    "ContrastiveLoss",
    "FocalLoss",
    "MultiTaskLoss",
    "MetaLoss",
    "create_loss_function",
    # Optimizer/Scheduler
    "create_optimizer",
    "create_scheduler",
    "GradientClipper",
    "AdaptiveLRController",
]
