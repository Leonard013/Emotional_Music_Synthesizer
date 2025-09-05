"""
Training pipeline for the music transformer.
Handles training loop, validation, and model checkpointing.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import wandb
from pathlib import Path

from ..models import MusicTransformer, TransformerConfig
from .optimizer import get_optimizer, get_scheduler
from .loss import MusicLoss
from .metrics import MusicMetrics


class TrainingConfig:
    """Configuration for training."""
    
    def __init__(self,
                 learning_rate: float = 1e-4,
                 batch_size: int = 8,
                 num_epochs: int = 100,
                 warmup_steps: int = 1000,
                 max_grad_norm: float = 1.0,
                 weight_decay: float = 0.01,
                 save_every: int = 10,
                 eval_every: int = 5,
                 log_every: int = 100,
                 use_wandb: bool = False,
                 wandb_project: str = "whistle-music-synthesizer",
                 device: str = "auto",
                 mixed_precision: bool = True,
                 gradient_accumulation_steps: int = 1,
                 early_stopping_patience: int = 10):
        """
        Initialize training configuration.
        
        Args:
            learning_rate: Learning rate
            batch_size: Batch size
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps
            max_grad_norm: Maximum gradient norm for clipping
            weight_decay: Weight decay for regularization
            save_every: Save model every N epochs
            eval_every: Evaluate model every N epochs
            log_every: Log metrics every N steps
            use_wandb: Whether to use Weights & Biases logging
            wandb_project: W&B project name
            device: Device to use ('auto', 'cpu', 'cuda')
            mixed_precision: Whether to use mixed precision training
            gradient_accumulation_steps: Gradient accumulation steps
            early_stopping_patience: Early stopping patience
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
        self.save_every = save_every
        self.eval_every = eval_every
        self.log_every = log_every
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.device = device
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.early_stopping_patience = early_stopping_patience


class MusicTrainer:
    """Trainer for the music transformer model."""
    
    def __init__(self,
                 model: MusicTransformer,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 config: Optional[TrainingConfig] = None,
                 output_dir: str = "./outputs"):
        """
        Initialize trainer.
        
        Args:
            model: Music transformer model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            output_dir: Output directory for checkpoints and logs
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = self._setup_device()
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = get_optimizer(
            model=self.model,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = get_scheduler(
            optimizer=self.optimizer,
            warmup_steps=self.config.warmup_steps,
            total_steps=len(self.train_loader) * self.config.num_epochs
        )
        
        # Setup loss function
        self.criterion = MusicLoss()
        
        # Setup metrics
        self.metrics = MusicMetrics()
        
        # Setup mixed precision
        if self.config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Setup logging
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                config=self.config.__dict__
            )
    
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.config.device)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = {}
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass with mixed precision
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        conditioning=batch.get('conditioning'),
                        attention_mask=batch.get('attention_mask'),
                        labels=batch['target_ids']
                    )
                    loss = outputs['loss']
            else:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    conditioning=batch.get('conditioning'),
                    attention_mask=batch.get('attention_mask'),
                    labels=batch['target_ids']
                )
                loss = outputs['loss']
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.config.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Update metrics
            epoch_loss += loss.item() * self.config.gradient_accumulation_steps
            
            # Calculate metrics
            with torch.no_grad():
                metrics = self.metrics.calculate_batch_metrics(
                    outputs['logits'],
                    batch['target_ids']
                )
                for key, value in metrics.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = 0.0
                    epoch_metrics[key] += value
            
            # Log progress
            if self.global_step % self.config.log_every == 0:
                self._log_metrics({
                    'train/loss': loss.item() * self.config.gradient_accumulation_steps,
                    'train/learning_rate': self.scheduler.get_last_lr()[0],
                    **{f'train/{k}': v for k, v in metrics.items()}
                })
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item() * self.config.gradient_accumulation_steps:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        # Average metrics
        num_batches = len(self.train_loader)
        epoch_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        epoch_metrics['loss'] = epoch_loss / num_batches
        
        return epoch_metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        val_loss = 0.0
        val_metrics = {}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                if self.config.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            conditioning=batch.get('conditioning'),
                            attention_mask=batch.get('attention_mask'),
                            labels=batch['target_ids']
                        )
                        loss = outputs['loss']
                else:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        conditioning=batch.get('conditioning'),
                        attention_mask=batch.get('attention_mask'),
                        labels=batch['target_ids']
                    )
                    loss = outputs['loss']
                
                val_loss += loss.item()
                
                # Calculate metrics
                metrics = self.metrics.calculate_batch_metrics(
                    outputs['logits'],
                    batch['target_ids']
                )
                for key, value in metrics.items():
                    if key not in val_metrics:
                        val_metrics[key] = 0.0
                    val_metrics[key] += value
        
        # Average metrics
        num_batches = len(self.val_loader)
        val_metrics = {k: v / num_batches for k, v in val_metrics.items()}
        val_metrics['loss'] = val_loss / num_batches
        
        return val_metrics
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        device_batch = {}
        for key, value in batch.items():
            if key == 'metadata':
                device_batch[key] = value
            elif isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics."""
        if self.config.use_wandb:
            wandb.log(metrics, step=self.global_step)
        else:
            print(f"Step {self.global_step}: {metrics}")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__,
            'model_config': self.model.config.__dict__
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.get_model_size():,}")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Log epoch metrics
            epoch_metrics = {f'train/{k}': v for k, v in train_metrics.items()}
            self._log_metrics(epoch_metrics)
            
            # Validation
            if self.val_loader is not None and epoch % self.config.eval_every == 0:
                val_metrics = self.validate()
                val_epoch_metrics = {f'val/{k}': v for k, v in val_metrics.items()}
                self._log_metrics(val_epoch_metrics)
                
                # Check for best model
                val_loss = val_metrics['loss']
                is_best = val_loss < self.best_val_loss
                
                if is_best:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break
            
            # Save checkpoint
            if epoch % self.config.save_every == 0:
                self.save_checkpoint(epoch, is_best=(self.val_loader is None))
        
        # Save final model
        self.save_checkpoint(self.current_epoch, is_best=True)
        
        if self.config.use_wandb:
            wandb.finish()
        
        print("Training completed!")
    
    def generate_sample(self, 
                       input_ids: torch.Tensor,
                       conditioning: Optional[torch.Tensor] = None,
                       max_length: int = 256) -> torch.Tensor:
        """Generate a sample sequence."""
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids.to(self.device),
                conditioning=conditioning.to(self.device) if conditioning is not None else None,
                max_length=max_length,
                temperature=0.8,
                do_sample=True
            )
        return generated
