"""Loss functions for multi-modal training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class CrossEntropyLoss(nn.Module):
    """Standard cross-entropy loss for classification."""
    
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, num_classes)
            labels: (batch_size,)
        Returns:
            loss: scalar
        """
        return self.loss_fn(logits, labels)


class ContrastiveLoss(nn.Module):
    """Contrastive loss for image-text matching (CLIP-style)."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: (batch_size, hidden_dim)
            text_features: (batch_size, hidden_dim)
        Returns:
            loss: scalar
        """
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity matrix
        logits = (image_features @ text_features.t()) / self.temperature
        
        # Labels are the diagonal (positive pairs)
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=logits.device)
        
        # Symmetric loss: image-to-text and text-to-image
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        
        loss = (loss_i2t + loss_t2i) / 2
        return loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, num_classes)
            labels: (batch_size,)
        Returns:
            loss: scalar
        """
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class MultiTaskLoss(nn.Module):
    """Multi-task loss with learnable task weights."""
    
    def __init__(
        self,
        task_names: list,
        loss_fns: Dict[str, nn.Module],
        use_uncertainty_weighting: bool = True
    ):
        super().__init__()
        self.task_names = task_names
        self.loss_fns = nn.ModuleDict(loss_fns)
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        if use_uncertainty_weighting:
            # Learnable log variance for each task
            self.log_vars = nn.Parameter(torch.zeros(len(task_names)))
            
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: dict of task predictions
            targets: dict of task targets
        Returns:
            dict with 'total_loss' and individual task losses
        """
        losses = {}
        total_loss = 0
        
        for i, task_name in enumerate(self.task_names):
            if task_name in predictions and task_name in targets:
                task_loss = self.loss_fns[task_name](
                    predictions[task_name],
                    targets[task_name]
                )
                losses[f'{task_name}_loss'] = task_loss
                
                if self.use_uncertainty_weighting:
                    # Uncertainty weighting: loss / (2 * var) + log(var)
                    precision = torch.exp(-self.log_vars[i])
                    weighted_loss = precision * task_loss + self.log_vars[i]
                    total_loss = total_loss + weighted_loss
                else:
                    total_loss = total_loss + task_loss
                    
        losses['total_loss'] = total_loss
        return losses


class MetaLoss(nn.Module):
    """Meta-loss for double-loop learning."""
    
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        task_loss: torch.Tensor,
        meta_info: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Compute meta-loss that guides the meta-controller.
        
        Args:
            task_loss: main task loss
            meta_info: information from double-loop controller
        Returns:
            combined loss
        """
        if meta_info is None or 'meta_loss' not in meta_info:
            return task_loss
            
        # Combine task loss with meta-loss from controller
        meta_loss = meta_info['meta_loss'].mean()
        
        # Meta-loss encourages the controller to predict future loss trends
        combined_loss = task_loss + 0.1 * meta_loss
        
        return combined_loss


def create_loss_function(config: Dict) -> nn.Module:
    """
    Factory function to create loss function from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Loss function module
    """
    loss_type = config.get('training', {}).get('loss_type', 'cross_entropy')
    
    if loss_type == 'cross_entropy':
        return CrossEntropyLoss(
            label_smoothing=config.get('training', {}).get('label_smoothing', 0.0)
        )
    elif loss_type == 'contrastive':
        return ContrastiveLoss(
            temperature=config.get('training', {}).get('temperature', 0.07)
        )
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=config.get('training', {}).get('focal_alpha', 0.25),
            gamma=config.get('training', {}).get('focal_gamma', 2.0)
        )
    elif loss_type == 'multitask':
        # Define task-specific losses
        task_losses = {}
        for task_name, task_config in config.get('tasks', {}).items():
            task_loss_type = task_config.get('loss_type', 'cross_entropy')
            if task_loss_type == 'cross_entropy':
                task_losses[task_name] = CrossEntropyLoss()
            elif task_loss_type == 'contrastive':
                task_losses[task_name] = ContrastiveLoss()
            # Add more task-specific losses as needed
                
        return MultiTaskLoss(
            task_names=list(config.get('tasks', {}).keys()),
            loss_fns=task_losses,
            use_uncertainty_weighting=config.get('training', {}).get('use_uncertainty_weighting', True)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
