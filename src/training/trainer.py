"""Main training loop for multi-modal neural network."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Any
from tqdm import tqdm
import time

from ..models.multi_modal_model import create_multi_modal_model
from ..data.dataset import create_dataset_from_config, create_dataloader
from ..training.losses import create_loss_function, MetaLoss
from ..training.optimizer import create_optimizer, create_scheduler, GradientClipper, AdaptiveLRController
from ..utils.config import load_config, save_config, validate_config
from ..utils.logging import setup_logger, MetricsLogger, WandbLogger, log_model_info


class Trainer:
    """Main trainer class for multi-modal neural network."""
    
    def __init__(
        self,
        config_path: str,
        resume_from: Optional[str] = None,
        device: Optional[str] = None
    ):
        # Load and validate configuration
        self.config = load_config(config_path)
        validate_config(self.config)
        
        # Setup device
        if device is None:
            device = self.config.get('hardware', {}).get('device', 'cuda')
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        log_dir = Path(self.config.get('paths', {}).get('log_dir', './logs'))
        self.logger = setup_logger(
            name='trainer',
            log_file=str(log_dir / 'training.log')
        )
        self.logger.info(f"Using device: {self.device}")
        
        # Setup metrics loggers
        experiment_name = self.config.get('logging', {}).get('experiment', 'default')
        self.metrics_logger = MetricsLogger(
            log_dir=str(log_dir),
            experiment_name=experiment_name
        )
        
        # Setup wandb if enabled
        use_wandb = self.config.get('logging', {}).get('use_wandb', True)
        if use_wandb:
            project_name = self.config.get('logging', {}).get('project', 'multi-modal-net')
            self.wandb_logger = WandbLogger(
                project=project_name,
                experiment=experiment_name,
                config=self.config,
                enabled=True
            )
        else:
            self.wandb_logger = None
            
        # Create model
        self.logger.info("Creating model...")
        self.model = create_multi_modal_model(self.config)
        self.model.to(self.device)
        log_model_info(self.logger, self.model)
        
        # Create datasets and dataloaders
        self.logger.info("Loading datasets...")
        train_dataset, val_dataset = create_dataset_from_config(self.config)
        
        data_config = self.config.get('data', {})
        self.train_loader = create_dataloader(
            train_dataset,
            batch_size=data_config.get('batch_size', 32),
            num_workers=data_config.get('num_workers', 4),
            shuffle=True,
            pin_memory=data_config.get('pin_memory', True)
        )
        
        self.val_loader = create_dataloader(
            val_dataset,
            batch_size=data_config.get('batch_size', 32),
            num_workers=data_config.get('num_workers', 4),
            shuffle=False,
            pin_memory=data_config.get('pin_memory', True)
        )
        
        self.logger.info(f"Train samples: {len(train_dataset)}")
        self.logger.info(f"Val samples: {len(val_dataset)}")
        
        # Create loss function
        self.criterion = create_loss_function(self.config)
        if self.model.use_double_loop:
            self.meta_criterion = MetaLoss()
        
        # Create optimizer and scheduler
        self.optimizer = create_optimizer(self.model, self.config)
        self.scheduler, self.scheduler_update_freq = create_scheduler(
            self.optimizer,
            self.config,
            steps_per_epoch=len(self.train_loader)
        )
        
        # Gradient clipping
        max_grad_norm = self.config.get('training', {}).get('max_grad_norm', 1.0)
        self.grad_clipper = GradientClipper(max_norm=max_grad_norm)
        
        # Adaptive LR controller for double-loop learning
        if self.model.use_double_loop:
            self.adaptive_lr = AdaptiveLRController(
                base_lr=self.config.get('training', {}).get('inner_lr', 3e-4)
            )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Mixed precision training
        self.use_amp = self.config.get('training', {}).get('mixed_precision', 'bf16') is not None
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Resume from checkpoint if provided
        if resume_from is not None:
            self.load_checkpoint(resume_from)
            
        # Save config
        output_dir = Path(self.config.get('paths', {}).get('output_dir', './outputs'))
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(self.config, str(output_dir / 'config.yaml'))
        
    def train(self):
        """Main training loop."""
        max_epochs = self.config.get('training', {}).get('max_epochs', 50)
        
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
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(is_best=True)
                self.logger.info(f"  New best model! Val loss: {self.best_val_loss:.4f}")
            
            # Regular checkpoint
            save_every = self.config.get('logging', {}).get('save_every', 5000)
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(is_best=False)
                
        self.logger.info("Training completed!")
        if self.wandb_logger:
            self.wandb_logger.finish()
            
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
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
            if self.scheduler_update_freq == 'step':
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            total_correct += accuracy.item() * batch['labels'].size(0)
            total_samples += batch['labels'].size(0)
            
            # Log progress
            avg_loss = total_loss / (batch_idx + 1)
            avg_acc = total_correct / total_samples
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{avg_acc:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Log to wandb
            if self.wandb_logger and batch_idx % self.config.get('logging', {}).get('log_every', 50) == 0:
                self.wandb_logger.log({
                    'train/loss': loss.item(),
                    'train/accuracy': accuracy.item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/gradient_norm': grad_norm
                }, step=self.global_step)
            
            self.global_step += 1
        
        # Update scheduler if per-epoch
        if self.scheduler_update_freq == 'epoch':
            self.scheduler.step()
        
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': total_correct / total_samples
        }
    
    def train_step(self, batch: Dict) -> tuple:
        """Single training step."""
        # Prepare inputs
        images = batch.get('images')
        input_ids = batch.get('input_ids')
        attention_mask = batch.get('attention_mask')
        labels = batch.get('labels')
        
        # Forward pass
        outputs = self.model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = outputs['logits']
        
        # Compute loss
        loss = self.criterion(logits, labels)
        
        # Add meta-loss if using double-loop learning
        if self.model.use_double_loop and 'meta_info' in outputs and outputs['meta_info'] is not None:
            loss = self.meta_criterion(loss, outputs['meta_info'])
        
        # Compute accuracy
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == labels).float().mean()
        
        return loss, accuracy
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation loop."""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                images=batch.get('images'),
                input_ids=batch.get('input_ids'),
                attention_mask=batch.get('attention_mask')
            )
            
            logits = outputs['logits']
            labels = batch.get('labels')
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Compute accuracy
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == labels).float().sum()
            
            total_loss += loss.item()
            total_correct += accuracy.item()
            total_samples += labels.size(0)
        
        metrics = {
            'loss': total_loss / len(self.val_loader),
            'accuracy': total_correct / total_samples
        }
        
        # Log to wandb
        if self.wandb_logger:
            self.wandb_logger.log({
                'val/loss': metrics['loss'],
                'val/accuracy': metrics['accuracy']
            }, step=self.global_step)
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.get('paths', {}).get('checkpoint_dir', './checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
        
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f"Resumed from epoch {self.current_epoch}")
