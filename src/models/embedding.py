"""
Embedding layers for the music transformer.
Handles token embeddings and conditioning embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class TokenEmbedding(nn.Module):
    """Token embedding layer for musical tokens."""
    
    def __init__(self, vocab_size: int, d_model: int, dropout: float = 0.1):
        """
        Initialize token embedding.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.embedding.weight, mean=0, std=math.sqrt(2.0 / self.d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of token embedding.
        
        Args:
            x: Input token indices [batch_size, seq_len]
            
        Returns:
            Embedded tokens [batch_size, seq_len, d_model]
        """
        return self.dropout(self.embedding(x) * math.sqrt(self.d_model))


class ConditioningEmbedding(nn.Module):
    """Embedding layer for conditioning features."""
    
    def __init__(self, 
                 feature_dim: int, 
                 d_model: int, 
                 dropout: float = 0.1,
                 use_mlp: bool = True):
        """
        Initialize conditioning embedding.
        
        Args:
            feature_dim: Input feature dimension
            d_model: Model dimension
            dropout: Dropout rate
            use_mlp: Whether to use MLP for feature transformation
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.use_mlp = use_mlp
        
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, d_model * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.Dropout(dropout)
            )
        else:
            self.linear = nn.Linear(feature_dim, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of conditioning embedding.
        
        Args:
            x: Input features [batch_size, feature_dim]
            
        Returns:
            Embedded conditioning [batch_size, d_model]
        """
        if self.use_mlp:
            return self.mlp(x)
        else:
            return self.dropout(self.linear(x))


class MusicalEmbedding(nn.Module):
    """Specialized embedding for musical tokens with type information."""
    
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int, 
                 token_types: dict,
                 dropout: float = 0.1):
        """
        Initialize musical embedding.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            token_types: Dictionary mapping token ranges to types
            dropout: Dropout rate
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_types = token_types
        
        # Main token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model - 8)  # Reserve space for type embedding
        
        # Type embedding (8 dimensions for different token types)
        self.type_embedding = nn.Embedding(len(token_types), 8)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.token_embedding.weight, mean=0, std=math.sqrt(2.0 / (self.d_model - 8)))
        nn.init.normal_(self.type_embedding.weight, mean=0, std=math.sqrt(2.0 / 8))
    
    def _get_token_type(self, token: int) -> int:
        """Get token type index for a given token."""
        for type_idx, (type_name, (start, end)) in enumerate(self.token_types.items()):
            if start <= token < end:
                return type_idx
        return 0  # Default type
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of musical embedding.
        
        Args:
            x: Input token indices [batch_size, seq_len]
            
        Returns:
            Embedded tokens with type information [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = x.size()
        
        # Get token embeddings
        token_emb = self.token_embedding(x)  # [batch_size, seq_len, d_model-8]
        
        # Get type embeddings
        type_indices = torch.zeros_like(x)
        for i in range(batch_size):
            for j in range(seq_len):
                type_indices[i, j] = self._get_token_type(x[i, j].item())
        
        type_emb = self.type_embedding(type_indices)  # [batch_size, seq_len, 8]
        
        # Concatenate token and type embeddings
        combined_emb = torch.cat([token_emb, type_emb], dim=-1)  # [batch_size, seq_len, d_model]
        
        # Apply layer normalization and dropout
        output = self.dropout(self.layer_norm(combined_emb))
        
        return output * math.sqrt(self.d_model)


class AdaptiveEmbedding(nn.Module):
    """Adaptive embedding that adjusts based on conditioning."""
    
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int, 
                 conditioning_dim: int,
                 dropout: float = 0.1):
        """
        Initialize adaptive embedding.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            conditioning_dim: Conditioning feature dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.conditioning_dim = conditioning_dim
        
        # Base embedding
        self.base_embedding = nn.Embedding(vocab_size, d_model)
        
        # Conditioning transformation
        self.conditioning_transform = nn.Sequential(
            nn.Linear(conditioning_dim, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model)
        )
        
        # Adaptive scaling
        self.adaptive_scale = nn.Sequential(
            nn.Linear(conditioning_dim, d_model),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.base_embedding.weight, mean=0, std=math.sqrt(2.0 / self.d_model))
    
    def forward(self, 
                x: torch.Tensor, 
                conditioning: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of adaptive embedding.
        
        Args:
            x: Input token indices [batch_size, seq_len]
            conditioning: Conditioning features [batch_size, conditioning_dim]
            
        Returns:
            Adaptive embedded tokens [batch_size, seq_len, d_model]
        """
        # Get base embeddings
        base_emb = self.base_embedding(x)  # [batch_size, seq_len, d_model]
        
        # Transform conditioning
        cond_transform = self.conditioning_transform(conditioning)  # [batch_size, d_model]
        cond_transform = cond_transform.unsqueeze(1).expand(-1, x.size(1), -1)  # [batch_size, seq_len, d_model]
        
        # Get adaptive scaling
        adaptive_scale = self.adaptive_scale(conditioning)  # [batch_size, d_model]
        adaptive_scale = adaptive_scale.unsqueeze(1).expand(-1, x.size(1), -1)  # [batch_size, seq_len, d_model]
        
        # Combine base embedding with conditioning
        adaptive_emb = base_emb + cond_transform
        adaptive_emb = adaptive_emb * adaptive_scale
        
        return self.dropout(adaptive_emb * math.sqrt(self.d_model))


class EmbeddingProjection(nn.Module):
    """Projection layer for embedding dimension changes."""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        """
        Initialize embedding projection.
        
        Args:
            input_dim: Input embedding dimension
            output_dim: Output embedding dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.projection = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of embedding projection.
        
        Args:
            x: Input embeddings [batch_size, seq_len, input_dim]
            
        Returns:
            Projected embeddings [batch_size, seq_len, output_dim]
        """
        return self.layer_norm(self.dropout(self.projection(x)))
