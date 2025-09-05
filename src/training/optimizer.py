"""
Optimizer and scheduler utilities for training.
"""

import torch
import torch.optim as optim
from typing import Dict, Any, Optional
import math


def get_optimizer(model: torch.nn.Module,
                  learning_rate: float = 1e-4,
                  weight_decay: float = 0.01,
                  optimizer_type: str = "adamw",
                  **kwargs) -> torch.optim.Optimizer:
    """
    Get optimizer for the model.
    
    Args:
        model: PyTorch model
        learning_rate: Learning rate
        weight_decay: Weight decay
        optimizer_type: Type of optimizer ('adamw', 'adam', 'sgd')
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer instance
    """
    if optimizer_type.lower() == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
            **kwargs
        )
    elif optimizer_type.lower() == "adam":
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
            **kwargs
        )
    elif optimizer_type.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def get_scheduler(optimizer: torch.optim.Optimizer,
                  warmup_steps: int = 1000,
                  total_steps: int = 100000,
                  scheduler_type: str = "cosine",
                  **kwargs) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Get learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        scheduler_type: Type of scheduler ('cosine', 'linear', 'constant')
        **kwargs: Additional scheduler arguments
        
    Returns:
        Scheduler instance
    """
    if scheduler_type.lower() == "cosine":
        return CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            **kwargs
        )
    elif scheduler_type.lower() == "linear":
        return LinearWarmupScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            **kwargs
        )
    elif scheduler_type.lower() == "constant":
        return ConstantWarmupScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing scheduler with warmup."""
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 warmup_steps: int,
                 total_steps: int,
                 min_lr_ratio: float = 0.1):
        """
        Initialize cosine warmup scheduler.
        
        Args:
            optimizer: Optimizer instance
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            min_lr_ratio: Minimum learning rate ratio
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer)
    
    def get_lr(self):
        """Get current learning rate."""
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            return [
                base_lr * self.last_epoch / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            return [
                base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))
                for base_lr in self.base_lrs
            ]


class LinearWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup scheduler."""
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 warmup_steps: int,
                 total_steps: int):
        """
        Initialize linear warmup scheduler.
        
        Args:
            optimizer: Optimizer instance
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer)
    
    def get_lr(self):
        """Get current learning rate."""
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            return [
                base_lr * self.last_epoch / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Linear decay phase
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            return [
                base_lr * (1 - progress)
                for base_lr in self.base_lrs
            ]


class ConstantWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Constant learning rate with warmup."""
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 warmup_steps: int):
        """
        Initialize constant warmup scheduler.
        
        Args:
            optimizer: Optimizer instance
            warmup_steps: Number of warmup steps
        """
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)
    
    def get_lr(self):
        """Get current learning rate."""
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            return [
                base_lr * self.last_epoch / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Constant phase
            return self.base_lrs


class WarmupScheduler:
    """Wrapper for warmup scheduling."""
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 warmup_steps: int,
                 scheduler: torch.optim.lr_scheduler._LRScheduler):
        """
        Initialize warmup scheduler wrapper.
        
        Args:
            optimizer: Optimizer instance
            warmup_steps: Number of warmup steps
            scheduler: Base scheduler
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.scheduler = scheduler
        self.step_count = 0
    
    def step(self):
        """Step the scheduler."""
        if self.step_count < self.warmup_steps:
            # Warmup phase
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['initial_lr'] * self.step_count / self.warmup_steps
        else:
            # Use base scheduler
            self.scheduler.step()
        
        self.step_count += 1
    
    def get_last_lr(self):
        """Get last learning rate."""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
    
    def state_dict(self):
        """Get state dict."""
        return {
            'step_count': self.step_count,
            'scheduler_state_dict': self.scheduler.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.step_count = state_dict['step_count']
        self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
