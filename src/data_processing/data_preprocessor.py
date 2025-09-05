"""
Data preprocessing utilities for whistle and MIDI data.
Handles feature alignment and data preparation for training.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from .midi_processor import MidiProcessor
from ..whistle_analysis import FeatureExtractor, PitchDetector, AudioProcessor


class DataPreprocessor:
    """Preprocesses whistle and MIDI data for model training."""
    
    def __init__(self, 
                 midi_processor: Optional[MidiProcessor] = None,
                 feature_extractor: Optional[FeatureExtractor] = None,
                 pitch_detector: Optional[PitchDetector] = None,
                 audio_processor: Optional[AudioProcessor] = None):
        """
        Initialize data preprocessor.
        
        Args:
            midi_processor: MIDI processor instance
            feature_extractor: Feature extractor instance
            pitch_detector: Pitch detector instance
            audio_processor: Audio processor instance
        """
        self.midi_processor = midi_processor or MidiProcessor()
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.pitch_detector = pitch_detector or PitchDetector()
        self.audio_processor = audio_processor or AudioProcessor()
    
    def extract_whistle_features(self, audio: np.ndarray) -> Dict:
        """
        Extract features from whistle audio.
        
        Args:
            audio: Whistle audio signal
            
        Returns:
            Dictionary of extracted features
        """
        # Preprocess audio
        processed_audio = self.audio_processor.preprocess_whistle(audio)
        
        # Extract pitch features
        pitch_features = self.pitch_detector.extract_pitch_features(processed_audio)
        
        # Extract all features
        all_features = self.feature_extractor.extract_all_features(processed_audio, pitch_features)
        
        return all_features
    
    def features_to_conditioning_vector(self, features: Dict) -> torch.Tensor:
        """
        Convert extracted features to conditioning vector for the model.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Conditioning vector tensor
        """
        # Extract key features for conditioning
        conditioning_features = []
        
        # Pitch features
        if 'pitch' in features:
            pitch = features['pitch']
            conditioning_features.extend([
                pitch.get('pitch_mean', 0),
                pitch.get('pitch_std', 0),
                pitch.get('pitch_range', 0),
                pitch.get('voiced_ratio', 0)
            ])
        
        # Rhythm features
        if 'rhythm' in features:
            rhythm = features['rhythm']
            conditioning_features.extend([
                rhythm.get('tempo', 120) / 200.0,  # Normalize tempo
                rhythm.get('rhythm_regularity', 0),
                rhythm.get('ioi_cv', 0)
            ])
        
        # Dynamics features
        if 'dynamics' in features:
            dynamics = features['dynamics']
            conditioning_features.extend([
                dynamics.get('energy_mean', 0),
                dynamics.get('dynamic_range', 0) / 60.0  # Normalize dynamic range
            ])
        
        # Timbre features
        if 'timbre' in features:
            timbre = features['timbre']
            conditioning_features.extend([
                timbre.get('harmonic_ratio', 0),
                timbre.get('spectral_flatness_mean', 0)
            ])
        
        # Pad or truncate to fixed size
        target_size = 16  # Fixed conditioning vector size
        if len(conditioning_features) < target_size:
            conditioning_features.extend([0.0] * (target_size - len(conditioning_features)))
        else:
            conditioning_features = conditioning_features[:target_size]
        
        return torch.tensor(conditioning_features, dtype=torch.float32)
    
    def create_whistle_midi_pairs(self, 
                                 whistle_audio: np.ndarray,
                                 midi_file: str,
                                 num_pairs: int = 10) -> List[Dict]:
        """
        Create whistle-MIDI pairs for training.
        
        Args:
            whistle_audio: Whistle audio signal
            midi_file: Path to MIDI file
            num_pairs: Number of pairs to create
            
        Returns:
            List of whistle-MIDI pairs
        """
        # Extract whistle features
        whistle_features = self.extract_whistle_features(whistle_audio)
        conditioning_vector = self.features_to_conditioning_vector(whistle_features)
        
        # Load and process MIDI
        midi = self.midi_processor.load_midi(midi_file)
        events = self.midi_processor.extract_events(midi)
        tokens = self.midi_processor.events_to_tokens(events)
        
        # Create sequence pairs
        pairs = self.midi_processor.create_sequence_pairs(tokens)
        
        # Create training pairs
        training_pairs = []
        for i, (input_seq, target_seq) in enumerate(pairs[:num_pairs]):
            training_pairs.append({
                'input_ids': torch.tensor(input_seq, dtype=torch.long),
                'target_ids': torch.tensor(target_seq, dtype=torch.long),
                'conditioning': conditioning_vector,
                'whistle_features': whistle_features,
                'metadata': {
                    'midi_file': midi_file,
                    'pair_index': i
                }
            })
        
        return training_pairs
    
    def align_whistle_to_midi(self, 
                             whistle_audio: np.ndarray,
                             midi_events: List[Dict],
                             alignment_strategy: str = 'tempo_based') -> Dict:
        """
        Align whistle audio to MIDI events.
        
        Args:
            whistle_audio: Whistle audio signal
            midi_events: List of MIDI events
            alignment_strategy: Strategy for alignment
            
        Returns:
            Alignment information
        """
        # Extract whistle tempo
        whistle_features = self.extract_whistle_features(whistle_audio)
        whistle_tempo = whistle_features.get('rhythm', {}).get('tempo', 120)
        
        # Calculate MIDI tempo (simplified)
        if len(midi_events) > 1:
            total_duration = midi_events[-1]['time'] - midi_events[0]['time']
            note_count = len([e for e in midi_events if e['type'] == 'note_on'])
            midi_tempo = (note_count / total_duration) * 60 if total_duration > 0 else 120
        else:
            midi_tempo = 120
        
        # Calculate tempo ratio
        tempo_ratio = whistle_tempo / midi_tempo if midi_tempo > 0 else 1.0
        
        # Align events
        aligned_events = []
        for event in midi_events:
            aligned_event = event.copy()
            aligned_event['time'] = event['time'] * tempo_ratio
            aligned_events.append(aligned_event)
        
        return {
            'whistle_tempo': whistle_tempo,
            'midi_tempo': midi_tempo,
            'tempo_ratio': tempo_ratio,
            'aligned_events': aligned_events,
            'alignment_strategy': alignment_strategy
        }
    
    def create_training_batch(self, 
                            whistle_audio: np.ndarray,
                            midi_files: List[str],
                            batch_size: int = 8) -> Dict[str, torch.Tensor]:
        """
        Create a training batch from whistle audio and MIDI files.
        
        Args:
            whistle_audio: Whistle audio signal
            midi_files: List of MIDI file paths
            batch_size: Batch size
            
        Returns:
            Training batch dictionary
        """
        all_pairs = []
        
        # Create pairs for each MIDI file
        for midi_file in midi_files:
            pairs = self.create_whistle_midi_pairs(whistle_audio, midi_file, num_pairs=2)
            all_pairs.extend(pairs)
        
        # Sample batch_size pairs
        if len(all_pairs) > batch_size:
            import random
            all_pairs = random.sample(all_pairs, batch_size)
        
        # Pad to batch_size if needed
        while len(all_pairs) < batch_size:
            all_pairs.append(all_pairs[0])  # Duplicate first pair
        
        # Stack tensors
        input_ids = torch.stack([pair['input_ids'] for pair in all_pairs])
        target_ids = torch.stack([pair['target_ids'] for pair in all_pairs])
        conditioning = torch.stack([pair['conditioning'] for pair in all_pairs])
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'conditioning': conditioning,
            'batch_size': len(all_pairs)
        }
    
    def preprocess_for_inference(self, whistle_audio: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Preprocess whistle audio for inference.
        
        Args:
            whistle_audio: Whistle audio signal
            
        Returns:
            Preprocessed data for inference
        """
        # Extract features
        features = self.extract_whistle_features(whistle_audio)
        conditioning = self.features_to_conditioning_vector(features)
        
        # Create initial sequence (just start token)
        initial_sequence = [self.midi_processor.START_TOKEN]
        input_ids = self.midi_processor.pad_sequence(initial_sequence)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),  # Add batch dimension
            'conditioning': conditioning.unsqueeze(0),  # Add batch dimension
            'features': features
        }
    
    def postprocess_generated_sequence(self, 
                                     generated_tokens: List[int],
                                     features: Dict) -> Dict:
        """
        Postprocess generated token sequence.
        
        Args:
            generated_tokens: Generated token sequence
            features: Original whistle features
            
        Returns:
            Postprocessed generation result
        """
        # Convert tokens to events
        events = self.midi_processor.tokens_to_events(generated_tokens)
        
        # Convert events to MIDI
        midi = self.midi_processor.events_to_midi(events)
        
        # Apply tempo adjustment based on whistle features
        whistle_tempo = features.get('rhythm', {}).get('tempo', 120)
        if whistle_tempo != 120:
            # Adjust tempo
            tempo_ratio = whistle_tempo / 120.0
            for instrument in midi.instruments:
                for note in instrument.notes:
                    note.start *= tempo_ratio
                    note.end *= tempo_ratio
        
        return {
            'midi': midi,
            'events': events,
            'tokens': generated_tokens,
            'tempo_adjustment': whistle_tempo / 120.0
        }
