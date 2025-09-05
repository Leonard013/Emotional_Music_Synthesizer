"""
Main transformer model for music generation.
Combines all components into a complete model architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple, List
from .attention import TransformerBlock, ConditionalTransformerBlock, PositionalEncoding
from .embedding import TokenEmbedding, ConditioningEmbedding, MusicalEmbedding


class TransformerConfig:
    """Configuration class for the music transformer."""
    
    def __init__(self,
                 vocab_size: int = 2000,
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 2048,
                 max_seq_len: int = 1024,
                 conditioning_dim: int = 16,
                 dropout: float = 0.1,
                 use_conditioning: bool = True,
                 use_musical_embedding: bool = True,
                 token_types: Optional[Dict] = None):
        """
        Initialize transformer configuration.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            conditioning_dim: Conditioning feature dimension
            dropout: Dropout rate
            use_conditioning: Whether to use conditioning
            use_musical_embedding: Whether to use musical embedding
            token_types: Token type definitions for musical embedding
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.conditioning_dim = conditioning_dim
        self.dropout = dropout
        self.use_conditioning = use_conditioning
        self.use_musical_embedding = use_musical_embedding
        self.token_types = token_types or {
            'special': (0, 3),
            'velocity': (3, 35),
            'time': (35, 1035),
            'note': (1035, 1123)
        }


class MusicTransformer(nn.Module):
    """Main transformer model for music generation."""
    
    def __init__(self, config: TransformerConfig):
        """
        Initialize music transformer.
        
        Args:
            config: Transformer configuration
        """
        super().__init__()
        
        self.config = config
        
        # Embedding layers
        if config.use_musical_embedding:
            self.token_embedding = MusicalEmbedding(
                vocab_size=config.vocab_size,
                d_model=config.d_model,
                token_types=config.token_types,
                dropout=config.dropout
            )
        else:
            self.token_embedding = TokenEmbedding(
                vocab_size=config.vocab_size,
                d_model=config.d_model,
                dropout=config.dropout
            )
        
        # Conditioning embedding
        if config.use_conditioning:
            self.conditioning_embedding = ConditioningEmbedding(
                feature_dim=config.conditioning_dim,
                d_model=config.d_model,
                dropout=config.dropout
            )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=config.d_model,
            max_len=config.max_seq_len
        )
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList()
        for _ in range(config.n_layers):
            if config.use_conditioning:
                layer = ConditionalTransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout
                )
            else:
                layer = TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout
                )
            self.transformer_layers.append(layer)
        
        # Output projection
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask for autoregressive generation.
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
            
        Returns:
            Causal mask tensor
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0)  # Add batch dimension
    
    def forward(self, 
                input_ids: torch.Tensor,
                conditioning: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the transformer.
        
        Args:
            input_ids: Input token sequences [batch_size, seq_len]
            conditioning: Conditioning features [batch_size, conditioning_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target labels for training [batch_size, seq_len]
            
        Returns:
            Dictionary containing logits and loss (if labels provided)
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Token embeddings
        x = self.token_embedding(input_ids)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Conditioning embedding
        if self.config.use_conditioning and conditioning is not None:
            cond_emb = self.conditioning_embedding(conditioning)  # [batch_size, d_model]
            cond_emb = cond_emb.unsqueeze(1)  # [batch_size, 1, d_model]
        else:
            cond_emb = None
        
        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len, device)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: [batch_size, seq_len] -> [batch_size, 1, seq_len, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            causal_mask = causal_mask * attention_mask
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            if self.config.use_conditioning and cond_emb is not None:
                x = layer(x, cond_emb, causal_mask)
            else:
                x = layer(x, causal_mask)
        
        # Final layer normalization
        x = self.layer_norm(x)
        
        # Output projection
        logits = self.output_projection(x)  # [batch_size, seq_len, vocab_size]
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for loss calculation
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=0  # Ignore padding tokens
            )
        
        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': x
        }
    
    def generate(self, 
                 input_ids: torch.Tensor,
                 conditioning: Optional[torch.Tensor] = None,
                 max_length: int = 512,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 do_sample: bool = True,
                 pad_token_id: int = 0,
                 eos_token_id: int = 2) -> torch.Tensor:
        """
        Generate music sequences autoregressively.
        
        Args:
            input_ids: Initial input sequence [batch_size, seq_len]
            conditioning: Conditioning features [batch_size, conditioning_dim]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            
        Returns:
            Generated sequence [batch_size, generated_length]
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        # Initialize generation
        generated = input_ids.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # Forward pass
                outputs = self.forward(
                    input_ids=generated,
                    conditioning=conditioning
                )
                
                # Get next token logits
                next_token_logits = outputs['logits'][:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    top_k = min(top_k, next_token_logits.size(-1))
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                    next_token_logits.scatter_(-1, top_k_indices, top_k_logits)
                
                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Update generated sequence
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Check for end-of-sequence
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
                
                # Stop if all sequences are finished
                if finished.all():
                    break
        
        return generated
    
    def get_model_size(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_model(self, path: str):
        """Save model to file."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
    
    @classmethod
    def load_model(cls, path: str, device: torch.device = None):
        """Load model from file."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
