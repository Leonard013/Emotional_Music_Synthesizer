"""
Metrics for evaluating music generation models.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter
import math


class MusicMetrics:
    """Metrics calculator for music generation."""
    
    def __init__(self, vocab_size: int = 2000):
        """
        Initialize metrics calculator.
        
        Args:
            vocab_size: Vocabulary size
        """
        self.vocab_size = vocab_size
    
    def calculate_batch_metrics(self, 
                               logits: torch.Tensor, 
                               targets: torch.Tensor,
                               attention_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Calculate metrics for a batch.
        
        Args:
            logits: Model predictions [batch_size, seq_len, vocab_size]
            targets: Target tokens [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Get predictions
        predictions = torch.argmax(logits, dim=-1)
        
        # Calculate accuracy
        if attention_mask is not None:
            correct = (predictions == targets) & attention_mask
            total = attention_mask.sum()
        else:
            correct = (predictions == targets)
            total = targets.numel()
        
        accuracy = correct.sum().float() / total if total > 0 else 0.0
        metrics['accuracy'] = accuracy.item()
        
        # Calculate perplexity
        perplexity = self.calculate_perplexity(logits, targets, attention_mask)
        metrics['perplexity'] = perplexity
        
        # Calculate BLEU score (simplified)
        bleu = self.calculate_bleu(predictions, targets, attention_mask)
        metrics['bleu'] = bleu
        
        # Calculate diversity metrics
        diversity = self.calculate_diversity(predictions, attention_mask)
        metrics.update(diversity)
        
        return metrics
    
    def calculate_perplexity(self, 
                           logits: torch.Tensor, 
                           targets: torch.Tensor,
                           attention_mask: Optional[torch.Tensor] = None) -> float:
        """
        Calculate perplexity.
        
        Args:
            logits: Model predictions [batch_size, seq_len, vocab_size]
            targets: Target tokens [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Perplexity value
        """
        # Calculate cross-entropy loss
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=0,
            reduction='none'
        )
        
        # Reshape to original dimensions
        ce_loss = ce_loss.view(targets.size())
        
        # Apply attention mask if provided
        if attention_mask is not None:
            ce_loss = ce_loss * attention_mask
            valid_tokens = attention_mask.sum()
        else:
            valid_tokens = (targets != 0).sum()
        
        # Calculate perplexity
        if valid_tokens > 0:
            perplexity = torch.exp(ce_loss.sum() / valid_tokens)
            return perplexity.item()
        else:
            return float('inf')
    
    def calculate_bleu(self, 
                      predictions: torch.Tensor, 
                      targets: torch.Tensor,
                      attention_mask: Optional[torch.Tensor] = None) -> float:
        """
        Calculate simplified BLEU score.
        
        Args:
            predictions: Predicted tokens [batch_size, seq_len]
            targets: Target tokens [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            BLEU score
        """
        batch_size = predictions.size(0)
        bleu_scores = []
        
        for i in range(batch_size):
            pred_seq = predictions[i].cpu().numpy()
            target_seq = targets[i].cpu().numpy()
            
            # Apply attention mask if provided
            if attention_mask is not None:
                mask = attention_mask[i].cpu().numpy()
                pred_seq = pred_seq[mask.astype(bool)]
                target_seq = target_seq[mask.astype(bool)]
            
            # Remove padding tokens
            pred_seq = pred_seq[pred_seq != 0]
            target_seq = target_seq[target_seq != 0]
            
            if len(pred_seq) == 0 or len(target_seq) == 0:
                bleu_scores.append(0.0)
                continue
            
            # Calculate n-gram precision
            precisions = []
            for n in range(1, 5):  # 1-gram to 4-gram
                pred_ngrams = self._get_ngrams(pred_seq, n)
                target_ngrams = self._get_ngrams(target_seq, n)
                
                if len(pred_ngrams) == 0:
                    precisions.append(0.0)
                    continue
                
                # Count matches
                matches = 0
                for ngram in pred_ngrams:
                    if ngram in target_ngrams:
                        matches += 1
                
                precision = matches / len(pred_ngrams)
                precisions.append(precision)
            
            # Calculate BLEU score
            if all(p > 0 for p in precisions):
                bleu = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
            else:
                bleu = 0.0
            
            # Apply brevity penalty
            if len(pred_seq) < len(target_seq):
                bp = math.exp(1 - len(target_seq) / len(pred_seq))
                bleu *= bp
            
            bleu_scores.append(bleu)
        
        return np.mean(bleu_scores)
    
    def _get_ngrams(self, sequence: np.ndarray, n: int) -> List[tuple]:
        """Get n-grams from sequence."""
        if len(sequence) < n:
            return []
        return [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]
    
    def calculate_diversity(self, 
                          predictions: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Calculate diversity metrics.
        
        Args:
            predictions: Predicted tokens [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary of diversity metrics
        """
        batch_size = predictions.size(0)
        
        # Collect all tokens
        all_tokens = []
        for i in range(batch_size):
            seq = predictions[i].cpu().numpy()
            
            # Apply attention mask if provided
            if attention_mask is not None:
                mask = attention_mask[i].cpu().numpy()
                seq = seq[mask.astype(bool)]
            
            # Remove padding tokens
            seq = seq[seq != 0]
            all_tokens.extend(seq)
        
        if len(all_tokens) == 0:
            return {'distinct_1': 0.0, 'distinct_2': 0.0, 'distinct_3': 0.0, 'distinct_4': 0.0}
        
        # Calculate distinct-n metrics
        distinct_metrics = {}
        for n in range(1, 5):
            ngrams = self._get_ngrams(np.array(all_tokens), n)
            unique_ngrams = len(set(ngrams))
            total_ngrams = len(ngrams)
            
            distinct_metrics[f'distinct_{n}'] = unique_ngrams / total_ngrams if total_ngrams > 0 else 0.0
        
        return distinct_metrics
    
    def calculate_sequence_metrics(self, 
                                 generated_sequences: List[List[int]],
                                 reference_sequences: List[List[int]]) -> Dict[str, float]:
        """
        Calculate sequence-level metrics.
        
        Args:
            generated_sequences: List of generated sequences
            reference_sequences: List of reference sequences
            
        Returns:
            Dictionary of sequence metrics
        """
        metrics = {}
        
        # Calculate average sequence length
        gen_lengths = [len(seq) for seq in generated_sequences]
        ref_lengths = [len(seq) for seq in reference_sequences]
        
        metrics['avg_gen_length'] = np.mean(gen_lengths)
        metrics['avg_ref_length'] = np.mean(ref_lengths)
        metrics['length_ratio'] = metrics['avg_gen_length'] / metrics['avg_ref_length'] if metrics['avg_ref_length'] > 0 else 0.0
        
        # Calculate sequence diversity
        gen_diversity = self._calculate_sequence_diversity(generated_sequences)
        ref_diversity = self._calculate_sequence_diversity(reference_sequences)
        
        metrics['gen_diversity'] = gen_diversity
        metrics['ref_diversity'] = ref_diversity
        metrics['diversity_ratio'] = gen_diversity / ref_diversity if ref_diversity > 0 else 0.0
        
        return metrics
    
    def _calculate_sequence_diversity(self, sequences: List[List[int]]) -> float:
        """Calculate diversity of a set of sequences."""
        if len(sequences) == 0:
            return 0.0
        
        # Collect all unique tokens
        all_tokens = set()
        for seq in sequences:
            all_tokens.update(seq)
        
        # Calculate average unique token ratio
        diversity_scores = []
        for seq in sequences:
            if len(seq) == 0:
                diversity_scores.append(0.0)
                continue
            
            unique_tokens = len(set(seq))
            diversity_scores.append(unique_tokens / len(seq))
        
        return np.mean(diversity_scores)
    
    def calculate_musical_metrics(self, 
                                generated_sequences: List[List[int]],
                                reference_sequences: List[List[int]]) -> Dict[str, float]:
        """
        Calculate music-specific metrics.
        
        Args:
            generated_sequences: List of generated sequences
            reference_sequences: List of reference sequences
            
        Returns:
            Dictionary of musical metrics
        """
        metrics = {}
        
        # Calculate note density
        gen_density = self._calculate_note_density(generated_sequences)
        ref_density = self._calculate_note_density(reference_sequences)
        
        metrics['gen_note_density'] = gen_density
        metrics['ref_note_density'] = ref_density
        metrics['density_ratio'] = gen_density / ref_density if ref_density > 0 else 0.0
        
        # Calculate pitch range
        gen_range = self._calculate_pitch_range(generated_sequences)
        ref_range = self._calculate_pitch_range(reference_sequences)
        
        metrics['gen_pitch_range'] = gen_range
        metrics['ref_pitch_range'] = ref_range
        metrics['range_ratio'] = gen_range / ref_range if ref_range > 0 else 0.0
        
        return metrics
    
    def _calculate_note_density(self, sequences: List[List[int]]) -> float:
        """Calculate average note density."""
        if len(sequences) == 0:
            return 0.0
        
        densities = []
        for seq in sequences:
            if len(seq) == 0:
                densities.append(0.0)
                continue
            
            # Count note tokens (assuming note tokens are in a specific range)
            note_tokens = sum(1 for token in seq if 1035 <= token < 1123)  # Note range from MIDI processor
            density = note_tokens / len(seq)
            densities.append(density)
        
        return np.mean(densities)
    
    def _calculate_pitch_range(self, sequences: List[List[int]]) -> float:
        """Calculate average pitch range."""
        if len(sequences) == 0:
            return 0.0
        
        ranges = []
        for seq in sequences:
            # Extract note tokens
            note_tokens = [token for token in seq if 1035 <= token < 1123]
            
            if len(note_tokens) == 0:
                ranges.append(0.0)
                continue
            
            # Convert to MIDI note numbers
            midi_notes = [token - 1035 + 21 for token in note_tokens]  # Convert to MIDI note range
            
            if len(midi_notes) == 0:
                ranges.append(0.0)
                continue
            
            pitch_range = max(midi_notes) - min(midi_notes)
            ranges.append(pitch_range)
        
        return np.mean(ranges)


def calculate_metrics(logits: torch.Tensor, 
                     targets: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None,
                     vocab_size: int = 2000) -> Dict[str, float]:
    """
    Calculate all metrics for a batch.
    
    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        targets: Target tokens [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        vocab_size: Vocabulary size
        
    Returns:
        Dictionary of all metrics
    """
    metrics_calculator = MusicMetrics(vocab_size)
    return metrics_calculator.calculate_batch_metrics(logits, targets, attention_mask)
