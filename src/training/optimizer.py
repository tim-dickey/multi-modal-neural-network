"""Optimizer and learning rate scheduler configuration."""

import torch
from torch.optim import AdamW, SGD, Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    ReduceLROnPlateau
)
from typing import Dict, Tuple, List, Optional
import math


def get_parameter_groups(
    model: torch.nn.Module,
    weight_decay: float = 0.01,
    no_decay_keywords: List[str] = None
) -> List[Dict]:
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
        no_decay_keywords = ['bias', 'LayerNorm', 'layer_norm', 'ln', 'bn']
        
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
        {
            'params': decay_params,
            'weight_decay': weight_decay
        },
        {
            'params': no_decay_params,
            'weight_decay': 0.0
        }
    ]
    
    return parameter_groups


def create_optimizer(
    model: torch.nn.Module,
    config: Dict
) -> torch.optim.Optimizer:
    """
    Create optimizer from config.
    
    Args:
        model: PyTorch model
        config: Configuration dictionary
        
    Returns:
        Optimizer instance
    """
    training_config = config.get('training', {})
    optimizer_name = training_config.get('optimizer', 'adamw').lower()
    lr = training_config.get('inner_lr', 3e-4)
    weight_decay = training_config.get('weight_decay', 0.01)
    
    # Get parameter groups with proper weight decay
    param_groups = get_parameter_groups(model, weight_decay)
    
    if optimizer_name == 'adamw':
        optimizer = AdamW(
            param_groups,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif optimizer_name == 'adam':
        optimizer = Adam(
            param_groups,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif optimizer_name == 'sgd':
        optimizer = SGD(
            param_groups,
            lr=lr,
            momentum=0.9,
            nesterov=True
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict,
    steps_per_epoch: int
) -> Tuple[torch.optim.lr_scheduler._LRScheduler, str]:
    """
    Create learning rate scheduler from config.
    
    Args:
        optimizer: PyTorch optimizer
        config: Configuration dictionary
        steps_per_epoch: number of training steps per epoch
        
    Returns:
        (scheduler, update_frequency) where update_frequency is 'epoch' or 'step'
    """
    training_config = config.get('training', {})
    scheduler_name = training_config.get('scheduler', 'cosine').lower()
    warmup_steps = training_config.get('warmup_steps', 1000)
    max_epochs = training_config.get('max_epochs', 50)
    total_steps = steps_per_epoch * max_epochs
    
    if scheduler_name == 'cosine':
        # Cosine annealing with linear warmup
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=training_config.get('min_lr', 1e-6)
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
        return scheduler, 'step'
        
    elif scheduler_name == 'linear':
        # Linear warmup then linear decay
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return max(0.0, (total_steps - step) / (total_steps - warmup_steps))
                
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return scheduler, 'step'
        
    elif scheduler_name == 'plateau':
        # Reduce on plateau
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=training_config.get('min_lr', 1e-6)
        )
        return scheduler, 'epoch'
        
    elif scheduler_name == 'constant':
        # Constant learning rate with warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return 1.0
                
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return scheduler, 'step'
        
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


class GradientClipper:
    """Utility for gradient clipping."""
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
        
    def __call__(self, parameters) -> float:
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
            torch.stack([torch.norm(p.grad.detach(), self.norm_type) for p in parameters]),
            self.norm_type
        )
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(parameters, self.max_norm, self.norm_type)
        
        return total_norm.item()


class AdaptiveLRController:
    """Adaptive learning rate controller for double-loop learning."""
    
    def __init__(
        self,
        base_lr: float = 3e-4,
        min_scale: float = 0.1,
        max_scale: float = 2.0
    ):
        self.base_lr = base_lr
        self.min_scale = min_scale
        self.max_scale = max_scale
        
    def update_lr(
        self,
        optimizer: torch.optim.Optimizer,
        lr_scale: torch.Tensor
    ):
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
            param_group['lr'] = new_lr
            
    def get_current_lr(self, optimizer: torch.optim.Optimizer) -> float:
        """Get current learning rate."""
        return optimizer.param_groups[0]['lr']
