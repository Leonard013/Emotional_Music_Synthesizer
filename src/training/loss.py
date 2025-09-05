"""
Loss functions for music generation training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any


class MusicLoss(nn.Module):
    """Main loss function for music generation."""
    
    def __init__(self, 
                 ignore_index: int = 0,
                 label_smoothing: float = 0.0,
                 use_focal_loss: bool = False,
                 focal_alpha: float = 1.0,
                 focal_gamma: float = 2.0):
        """
        Initialize music loss.
        
        Args:
            ignore_index: Index to ignore in loss calculation
            label_smoothing: Label smoothing factor
            use_focal_loss: Whether to use focal loss
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
        """
        super().__init__()
        
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        if use_focal_loss:
            self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    def forward(self, 
                logits: torch.Tensor, 
                targets: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate loss.
        
        Args:
            logits: Model predictions [batch_size, seq_len, vocab_size]
            targets: Target tokens [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Loss value
        """
        # Flatten for loss calculation
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        if self.use_focal_loss:
            loss = self.focal_loss(logits_flat, targets_flat)
        else:
            loss = F.cross_entropy(
                logits_flat,
                targets_flat,
                ignore_index=self.ignore_index,
                label_smoothing=self.label_smoothing
            )
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask_flat = attention_mask.view(-1)
            loss = loss * mask_flat
            loss = loss.sum() / mask_flat.sum()
        
        return loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize focal loss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss.
        
        Args:
            inputs: Model predictions [batch_size, num_classes]
            targets: Target classes [batch_size]
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for representation learning."""
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.1):
        """
        Initialize contrastive loss.
        
        Args:
            margin: Margin for contrastive loss
            temperature: Temperature for similarity calculation
        """
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(self, 
                anchor: torch.Tensor, 
                positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        """
        Calculate contrastive loss.
        
        Args:
            anchor: Anchor embeddings [batch_size, embedding_dim]
            positive: Positive embeddings [batch_size, embedding_dim]
            negative: Negative embeddings [batch_size, embedding_dim]
            
        Returns:
            Contrastive loss value
        """
        # Calculate similarities
        pos_sim = F.cosine_similarity(anchor, positive, dim=-1)
        neg_sim = F.cosine_similarity(anchor, negative, dim=-1)
        
        # Contrastive loss
        pos_loss = -torch.log(torch.sigmoid(pos_sim / self.temperature))
        neg_loss = -torch.log(torch.sigmoid(-neg_sim / self.temperature))
        
        loss = pos_loss + neg_loss
        return loss.mean()


class PerplexityLoss(nn.Module):
    """Perplexity-based loss for language modeling."""
    
    def __init__(self, ignore_index: int = 0):
        """
        Initialize perplexity loss.
        
        Args:
            ignore_index: Index to ignore in loss calculation
        """
        super().__init__()
        self.ignore_index = ignore_index
    
    def forward(self, 
                logits: torch.Tensor, 
                targets: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate perplexity loss.
        
        Args:
            logits: Model predictions [batch_size, seq_len, vocab_size]
            targets: Target tokens [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Perplexity loss value
        """
        # Calculate cross-entropy loss
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=self.ignore_index,
            reduction='none'
        )
        
        # Reshape to original dimensions
        ce_loss = ce_loss.view(targets.size())
        
        # Apply attention mask if provided
        if attention_mask is not None:
            ce_loss = ce_loss * attention_mask
            valid_tokens = attention_mask.sum()
        else:
            valid_tokens = (targets != self.ignore_index).sum()
        
        # Calculate perplexity
        perplexity = torch.exp(ce_loss.sum() / valid_tokens)
        
        return perplexity


class MultiTaskLoss(nn.Module):
    """Multi-task loss combining different objectives."""
    
    def __init__(self, 
                 task_weights: Optional[Dict[str, float]] = None,
                 ignore_index: int = 0):
        """
        Initialize multi-task loss.
        
        Args:
            task_weights: Weights for different tasks
            ignore_index: Index to ignore in loss calculation
        """
        super().__init__()
        
        self.task_weights = task_weights or {
            'generation': 1.0,
            'perplexity': 0.1,
            'contrastive': 0.05
        }
        
        self.generation_loss = MusicLoss(ignore_index=ignore_index)
        self.perplexity_loss = PerplexityLoss(ignore_index=ignore_index)
        self.contrastive_loss = ContrastiveLoss()
    
    def forward(self, 
                outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate multi-task loss.
        
        Args:
            outputs: Model outputs for different tasks
            targets: Target values for different tasks
            
        Returns:
            Dictionary of losses for each task
        """
        losses = {}
        total_loss = 0.0
        
        # Generation loss
        if 'logits' in outputs and 'target_ids' in targets:
            gen_loss = self.generation_loss(
                outputs['logits'],
                targets['target_ids'],
                targets.get('attention_mask')
            )
            losses['generation'] = gen_loss
            total_loss += self.task_weights['generation'] * gen_loss
        
        # Perplexity loss
        if 'logits' in outputs and 'target_ids' in targets:
            perp_loss = self.perplexity_loss(
                outputs['logits'],
                targets['target_ids'],
                targets.get('attention_mask')
            )
            losses['perplexity'] = perp_loss
            total_loss += self.task_weights['perplexity'] * perp_loss
        
        # Contrastive loss
        if all(key in outputs for key in ['anchor', 'positive', 'negative']):
            cont_loss = self.contrastive_loss(
                outputs['anchor'],
                outputs['positive'],
                outputs['negative']
            )
            losses['contrastive'] = cont_loss
            total_loss += self.task_weights['contrastive'] * cont_loss
        
        losses['total'] = total_loss
        return losses


class AdaptiveLoss(nn.Module):
    """Adaptive loss that adjusts weights based on training progress."""
    
    def __init__(self, 
                 base_loss: nn.Module,
                 adaptation_rate: float = 0.01,
                 min_weight: float = 0.1,
                 max_weight: float = 10.0):
        """
        Initialize adaptive loss.
        
        Args:
            base_loss: Base loss function
            adaptation_rate: Rate of weight adaptation
            min_weight: Minimum weight value
            max_weight: Maximum weight value
        """
        super().__init__()
        
        self.base_loss = base_loss
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Initialize adaptive weight
        self.register_buffer('adaptive_weight', torch.tensor(1.0))
        self.register_buffer('loss_history', torch.zeros(100))
        self.register_buffer('history_index', torch.tensor(0))
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Calculate adaptive loss.
        
        Args:
            *args: Arguments for base loss
            **kwargs: Keyword arguments for base loss
            
        Returns:
            Adaptive loss value
        """
        # Calculate base loss
        loss = self.base_loss(*args, **kwargs)
        
        # Update loss history
        self.loss_history[self.history_index] = loss.detach()
        self.history_index = (self.history_index + 1) % 100
        
        # Adapt weight based on loss variance
        if self.history_index == 0:  # History is full
            loss_std = self.loss_history.std()
            if loss_std > 0:
                # Increase weight if loss is unstable
                weight_change = self.adaptation_rate * (loss_std - 1.0)
                self.adaptive_weight = torch.clamp(
                    self.adaptive_weight + weight_change,
                    self.min_weight,
                    self.max_weight
                )
        
        return self.adaptive_weight * loss
