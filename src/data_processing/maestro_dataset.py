"""
MAESTRO dataset loader and preprocessing.
Handles loading, preprocessing, and batching of MAESTRO dataset.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader
from .midi_processor import MidiProcessor
import random
from pathlib import Path


class MaestroDataset(Dataset):
    """PyTorch Dataset for MAESTRO dataset."""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 max_files: Optional[int] = None,
                 midi_processor: Optional[MidiProcessor] = None,
                 cache_dir: Optional[str] = None):
        """
        Initialize MAESTRO dataset.
        
        Args:
            data_dir: Directory containing MAESTRO dataset
            split: Dataset split ('train', 'validation', 'test')
            max_files: Maximum number of files to load (None for all)
            midi_processor: MIDI processor instance
            cache_dir: Directory to cache processed data
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_files = max_files
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Initialize MIDI processor
        if midi_processor is None:
            self.midi_processor = MidiProcessor()
        else:
            self.midi_processor = midi_processor
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Filter by split
        self.split_files = self.metadata[self.metadata['split'] == split]
        
        if max_files is not None:
            self.split_files = self.split_files.head(max_files)
        
        # Cache processed sequences
        self.cached_sequences = {}
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache()
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load MAESTRO metadata."""
        metadata_path = self.data_dir / 'maestro-v3.0.0.json'
        
        if not metadata_path.exists():
            # Try CSV format
            metadata_path = self.data_dir / 'maestro-v3.0.0.csv'
            if metadata_path.exists():
                return pd.read_csv(metadata_path)
            else:
                raise FileNotFoundError(f"Metadata file not found in {self.data_dir}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return pd.DataFrame(metadata)
    
    def _load_cache(self):
        """Load cached processed sequences."""
        cache_file = self.cache_dir / f'{self.split}_cache.json'
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                self.cached_sequences = json.load(f)
    
    def _save_cache(self):
        """Save processed sequences to cache."""
        if self.cache_dir:
            cache_file = self.cache_dir / f'{self.split}_cache.json'
            with open(cache_file, 'w') as f:
                json.dump(self.cached_sequences, f)
    
    def _process_midi_file(self, midi_path: str) -> List[List[int]]:
        """
        Process a single MIDI file into token sequences.
        
        Args:
            midi_path: Path to MIDI file
            
        Returns:
            List of token sequences
        """
        try:
            # Load MIDI
            midi = self.midi_processor.load_midi(midi_path)
            
            # Extract events
            events = self.midi_processor.extract_events(midi)
            
            # Convert to tokens
            tokens = self.midi_processor.events_to_tokens(events)
            
            # Create sequence pairs
            pairs = self.midi_processor.create_sequence_pairs(tokens)
            
            return [pair[0] for pair in pairs]  # Return input sequences
            
        except Exception as e:
            print(f"Error processing {midi_path}: {e}")
            return []
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.split_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing input and target sequences
        """
        row = self.split_files.iloc[idx]
        midi_filename = row['midi_filename']
        midi_path = self.data_dir / midi_filename
        
        # Check cache first
        cache_key = str(midi_path)
        if cache_key in self.cached_sequences:
            sequences = self.cached_sequences[cache_key]
        else:
            # Process MIDI file
            sequences = self._process_midi_file(str(midi_path))
            
            # Cache the result
            self.cached_sequences[cache_key] = sequences
            self._save_cache()
        
        if not sequences:
            # Return empty sequence if processing failed
            empty_seq = [self.midi_processor.PAD_TOKEN] * self.midi_processor.max_sequence_length
            return {
                'input_ids': torch.tensor(empty_seq, dtype=torch.long),
                'target_ids': torch.tensor(empty_seq, dtype=torch.long),
                'attention_mask': torch.zeros(self.midi_processor.max_sequence_length, dtype=torch.bool)
            }
        
        # Select a random sequence
        sequence = random.choice(sequences)
        
        # Create target sequence (shifted by 1)
        target_sequence = sequence[1:] + [self.midi_processor.END_TOKEN]
        
        # Pad sequences
        input_ids = self.midi_processor.pad_sequence(sequence)
        target_ids = self.midi_processor.pad_sequence(target_sequence)
        
        # Create attention mask
        attention_mask = torch.ones(len(sequence), dtype=torch.bool)
        if len(sequence) < self.midi_processor.max_sequence_length:
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(self.midi_processor.max_sequence_length - len(sequence), dtype=torch.bool)
            ])
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'attention_mask': attention_mask,
            'metadata': {
                'composer': row.get('canonical_composer', ''),
                'title': row.get('canonical_title', ''),
                'year': row.get('year', ''),
                'duration': row.get('duration', 0)
            }
        }


class MaestroDataLoader:
    """Data loader for MAESTRO dataset with custom collation."""
    
    def __init__(self, 
                 dataset: MaestroDataset,
                 batch_size: int = 8,
                 shuffle: bool = True,
                 num_workers: int = 4,
                 pin_memory: bool = True):
        """
        Initialize data loader.
        
        Args:
            dataset: MAESTRO dataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn
        )
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Custom collation function for batching.
        
        Args:
            batch: List of dataset items
            
        Returns:
            Batched tensors
        """
        # Stack tensors
        input_ids = torch.stack([item['input_ids'] for item in batch])
        target_ids = torch.stack([item['target_ids'] for item in batch])
        attention_masks = torch.stack([item['attention_mask'] for item in batch])
        
        # Collect metadata
        metadata = [item['metadata'] for item in batch]
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'attention_mask': attention_masks,
            'metadata': metadata
        }
    
    def __iter__(self):
        """Iterate over batches."""
        return iter(self.dataloader)
    
    def __len__(self) -> int:
        """Get number of batches."""
        return len(self.dataloader)


def create_maestro_dataloaders(data_dir: str,
                              batch_size: int = 8,
                              max_files_per_split: Optional[Dict[str, int]] = None,
                              cache_dir: Optional[str] = None,
                              num_workers: int = 4) -> Dict[str, MaestroDataLoader]:
    """
    Create data loaders for all MAESTRO splits.
    
    Args:
        data_dir: Directory containing MAESTRO dataset
        batch_size: Batch size for all loaders
        max_files_per_split: Maximum files per split
        cache_dir: Directory to cache processed data
        num_workers: Number of worker processes
        
    Returns:
        Dictionary of data loaders for each split
    """
    if max_files_per_split is None:
        max_files_per_split = {'train': None, 'validation': None, 'test': None}
    
    dataloaders = {}
    
    for split in ['train', 'validation', 'test']:
        dataset = MaestroDataset(
            data_dir=data_dir,
            split=split,
            max_files=max_files_per_split.get(split),
            cache_dir=cache_dir
        )
        
        dataloader = MaestroDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers
        )
        
        dataloaders[split] = dataloader
    
    return dataloaders


def get_dataset_statistics(data_dir: str) -> Dict:
    """
    Get statistics about the MAESTRO dataset.
    
    Args:
        data_dir: Directory containing MAESTRO dataset
        
    Returns:
        Dictionary of dataset statistics
    """
    dataset = MaestroDataset(data_dir, split='train')
    metadata = dataset.metadata
    
    stats = {
        'total_files': len(metadata),
        'splits': metadata['split'].value_counts().to_dict(),
        'composers': metadata['canonical_composer'].value_counts().head(10).to_dict(),
        'total_duration_hours': metadata['duration'].sum() / 3600,
        'avg_duration_minutes': metadata['duration'].mean() / 60,
        'year_range': {
            'min': metadata['year'].min(),
            'max': metadata['year'].max()
        }
    }
    
    return stats
