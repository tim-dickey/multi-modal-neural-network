"""Optimizer and learning rate scheduler configuration."""

from typing import Any, Dict, List, Optional, Tuple, cast

import torch

from .training_defaults import TRAINING
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
)


def get_parameter_groups(
    model: torch.nn.Module,
    weight_decay: Optional[float] = None,
    no_decay_keywords: Optional[List[str]] = None,
) -> List[Dict]:
    if weight_decay is None:
        weight_decay = TRAINING.weight_decay
    """
    Separate parameters into groups with and without weight decay.

    Args:
        model: PyTorch model
        weight_decay: weight decay value
        no_decay_keywords: list of keywords for params without decay

    Returns:
        List of parameter groups
    """
    if no_decay_keywords is None:
        no_decay_keywords = ["bias", "LayerNorm", "layer_norm", "ln", "bn"]

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if parameter should have no decay
        if any(keyword in name for keyword in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    parameter_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return parameter_groups


def create_optimizer(model: torch.nn.Module, config: Dict) -> torch.optim.Optimizer:
    """
    Create optimizer from config.

    Args:
        model: PyTorch model
        config: Configuration dictionary

    Returns:
        Optimizer instance
    """
    training_config = config.get("training", {})
    optimizer_name = training_config.get("optimizer", "adamw").lower()
    lr = training_config.get("inner_lr", TRAINING.learning_rate)
    weight_decay = training_config.get("weight_decay", TRAINING.weight_decay)

    # Normalize types from config (strings -> floats)
    try:
        lr = float(lr)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid learning rate: {lr}") from exc

    try:
        weight_decay = float(weight_decay)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid weight_decay: {weight_decay}") from exc

    # Get parameter groups with proper weight decay
    param_groups = get_parameter_groups(model, weight_decay)

    optimizer: torch.optim.Optimizer

    if optimizer_name == "adamw":
        optimizer = AdamW(
            param_groups,
            lr=lr,
            betas=(TRAINING.adam_beta1, TRAINING.adam_beta2),
            eps=TRAINING.adam_epsilon,
        )
    elif optimizer_name == "adam":
        optimizer = Adam(
            param_groups,
            lr=lr,
            betas=(TRAINING.adam_beta1, TRAINING.adam_beta2),
            eps=TRAINING.adam_epsilon,
        )
    elif optimizer_name == "sgd":
        optimizer = SGD(param_groups, lr=lr, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer, config: Dict, steps_per_epoch: int
) -> Tuple[torch.optim.lr_scheduler.LRScheduler, str]:
    """
    Create learning rate scheduler from config.

    Args:
        optimizer: PyTorch optimizer
        config: Configuration dictionary
        steps_per_epoch: number of training steps per epoch

    Returns:
        (scheduler, update_frequency) where update_frequency is 'epoch' or 'step'
    """
    training_config = config.get("training", {})
    scheduler_name = training_config.get("scheduler", "cosine").lower()
    warmup_steps = training_config.get("warmup_steps", TRAINING.warmup_steps)
    max_epochs = training_config.get("max_epochs", TRAINING.max_epochs)
    total_steps = steps_per_epoch * max_epochs

    scheduler: torch.optim.lr_scheduler.LRScheduler

    if scheduler_name == "cosine":
        # Cosine annealing with linear warmup
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=TRAINING.warmup_start_factor,
            end_factor=TRAINING.warmup_end_factor,
            total_iters=warmup_steps,
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=training_config.get("min_lr", TRAINING.min_learning_rate),
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        # Mark optimizer as having taken an initial step to avoid
        # PyTorch warning when users call `scheduler.step()` before
        # `optimizer.step()` in simple unit tests or scripts.
        if getattr(optimizer, "_step_count", 0) == 0:
            try:
                setattr(optimizer, "_step_count", 1)
            except (AttributeError, TypeError):
                # Best-effort; avoid crashing if attribute is unavailable
                pass

        return scheduler, "step"

    elif scheduler_name == "linear":
        # Linear warmup then linear decay
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step / warmup_steps)
            else:
                return max(
                    0.0, float((total_steps - step) / (total_steps - warmup_steps))
                )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        if getattr(optimizer, "_step_count", 0) == 0:
            try:
                setattr(optimizer, "_step_count", 1)
            except (AttributeError, TypeError):
                pass
        return scheduler, "step"

    elif scheduler_name == "plateau":
        # Reduce on plateau
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=training_config.get("min_lr", TRAINING.min_learning_rate),
        )
        return scheduler, "epoch"

    elif scheduler_name == "constant":
        # Constant learning rate with warmup
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step / warmup_steps)
            else:
                return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        if getattr(optimizer, "_step_count", 0) == 0:
            try:
                setattr(optimizer, "_step_count", 1)
            except (AttributeError, TypeError):
                pass
        return scheduler, "step"

    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


class GradientClipper:
    """Utility for gradient clipping."""

    def __init__(
        self,
        max_norm: Optional[float] = None,
        norm_type: Optional[float] = None,
    ):
        if max_norm is None:
            max_norm = TRAINING.max_grad_norm
        if norm_type is None:
            norm_type = TRAINING.grad_norm_type
        self.max_norm = max_norm
        self.norm_type = norm_type

    def __call__(self, parameters: Any) -> float:
        """
        Clip gradients and return the gradient norm.

        Args:
            parameters: model parameters

        Returns:
            gradient norm before clipping
        """
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        parameters = [p for p in parameters if p.grad is not None]

        if len(parameters) == 0:
            return 0.0

        # Compute gradient norm
        total_norm = torch.norm(
            torch.stack(
                [
                    torch.norm(cast(torch.Tensor, p.grad).detach(), self.norm_type)
                    for p in parameters
                ]
            ),
            self.norm_type,
        )

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(parameters, self.max_norm, self.norm_type)

        return float(total_norm.item())


class AdaptiveLRController:
    """Adaptive learning rate controller for double-loop learning."""

    def __init__(
        self,
        base_lr: Optional[float] = None,
        min_scale: Optional[float] = None,
        max_scale: Optional[float] = None,
    ):
        if base_lr is None:
            base_lr = TRAINING.learning_rate
        if min_scale is None:
            min_scale = TRAINING.adaptive_lr_min_scale
        if max_scale is None:
            max_scale = TRAINING.adaptive_lr_max_scale
        self.base_lr = base_lr
        self.min_scale = min_scale
        self.max_scale = max_scale

    def update_lr(
        self, optimizer: torch.optim.Optimizer, lr_scale: torch.Tensor
    ) -> None:
        """
        Update optimizer learning rate based on meta-controller signal.

        Args:
            optimizer: PyTorch optimizer
            lr_scale: scaling factor from meta-controller (0 to 1)
        """
        # Map lr_scale to [min_scale, max_scale]
        scale = self.min_scale + lr_scale.item() * (self.max_scale - self.min_scale)
        new_lr = self.base_lr * scale

        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

    def get_current_lr(self, optimizer: torch.optim.Optimizer) -> float:
        """Get current learning rate."""
        return float(optimizer.param_groups[0]["lr"])
