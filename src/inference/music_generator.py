"""
Advanced music generation utilities.
Provides various generation strategies and post-processing options.
"""

import torch
import numpy as np
import pretty_midi
from typing import Dict, List, Optional, Tuple, Union, Any
import random
from pathlib import Path

from ..models import MusicTransformer
from ..data_processing import MidiProcessor
from .whistle_to_music import WhistleToMusicConverter


class MusicGenerator:
    """Advanced music generator with various strategies."""
    
    def __init__(self, 
                 model_path: str,
                 device: str = "auto",
                 seed: Optional[int] = None):
        """
        Initialize music generator.
        
        Args:
            model_path: Path to trained model
            device: Device for inference
            seed: Random seed for reproducibility
        """
        self.converter = WhistleToMusicConverter(model_path, device)
        self.midi_processor = MidiProcessor()
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
    
    def generate_with_style(self, 
                           whistle_audio: Union[str, np.ndarray],
                           style: str = "classical",
                           intensity: float = 0.5,
                           complexity: float = 0.5,
                           **kwargs) -> Dict:
        """
        Generate music with specific style characteristics.
        
        Args:
            whistle_audio: Whistle audio input
            style: Music style ("classical", "romantic", "baroque", "impressionist")
            intensity: Style intensity (0.0 to 1.0)
            complexity: Musical complexity (0.0 to 1.0)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated music result
        """
        # Adjust generation parameters based on style
        style_params = self._get_style_parameters(style, intensity, complexity)
        
        # Merge with user parameters
        generation_params = {**style_params, **kwargs}
        
        # Generate music
        result = self.converter.convert_whistle_to_music(
            whistle_audio=whistle_audio,
            style=style,
            **generation_params
        )
        
        # Post-process based on style
        result = self._post_process_style(result, style, intensity)
        
        return result
    
    def _get_style_parameters(self, 
                             style: str, 
                             intensity: float, 
                             complexity: float) -> Dict:
        """Get generation parameters for specific style."""
        base_params = {
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.9,
            'max_length': 512
        }
        
        style_configs = {
            'classical': {
                'temperature': 0.7 + intensity * 0.2,
                'top_k': 40 + int(complexity * 20),
                'max_length': 400 + int(complexity * 200)
            },
            'romantic': {
                'temperature': 0.8 + intensity * 0.3,
                'top_k': 30 + int(complexity * 30),
                'max_length': 600 + int(complexity * 300)
            },
            'baroque': {
                'temperature': 0.6 + intensity * 0.2,
                'top_k': 60 + int(complexity * 10),
                'max_length': 300 + int(complexity * 150)
            },
            'impressionist': {
                'temperature': 0.9 + intensity * 0.2,
                'top_k': 20 + int(complexity * 40),
                'max_length': 500 + int(complexity * 250)
            }
        }
        
        if style in style_configs:
            base_params.update(style_configs[style])
        
        return base_params
    
    def _post_process_style(self, 
                           result: Dict, 
                           style: str, 
                           intensity: float) -> Dict:
        """Post-process generated music based on style."""
        midi = result['midi']
        
        if style == 'romantic' and intensity > 0.5:
            # Add more expressive dynamics
            self._enhance_dynamics(midi, intensity)
        
        elif style == 'baroque':
            # Add ornamentation
            self._add_ornamentation(midi, intensity)
        
        elif style == 'impressionist':
            # Add harmonic complexity
            self._add_harmonic_complexity(midi, intensity)
        
        return result
    
    def _enhance_dynamics(self, midi: pretty_midi.PrettyMIDI, intensity: float):
        """Enhance dynamics for romantic style."""
        for instrument in midi.instruments:
            for note in instrument.notes:
                # Increase dynamic range
                note.velocity = min(127, int(note.velocity * (1 + intensity * 0.5)))
    
    def _add_ornamentation(self, midi: pretty_midi.PrettyMIDI, intensity: float):
        """Add baroque ornamentation."""
        for instrument in midi.instruments:
            new_notes = []
            for note in instrument.notes:
                new_notes.append(note)
                
                # Add trill with probability based on intensity
                if random.random() < intensity * 0.3:
                    trill_note = pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch + 2,  # Major second
                        start=note.start + 0.1,
                        end=note.start + 0.2
                    )
                    new_notes.append(trill_note)
            
            instrument.notes = new_notes
    
    def _add_harmonic_complexity(self, midi: pretty_midi.PrettyMIDI, intensity: float):
        """Add harmonic complexity for impressionist style."""
        for instrument in midi.instruments:
            new_notes = []
            for note in instrument.notes:
                new_notes.append(note)
                
                # Add harmonic extensions
                if random.random() < intensity * 0.4:
                    # Add major 7th
                    harmonic_note = pretty_midi.Note(
                        velocity=int(note.velocity * 0.7),
                        pitch=note.pitch + 11,  # Major 7th
                        start=note.start,
                        end=note.end
                    )
                    new_notes.append(harmonic_note)
            
            instrument.notes = new_notes
    
    def generate_variations(self, 
                           whistle_audio: Union[str, np.ndarray],
                           num_variations: int = 5,
                           variation_type: str = "random",
                           **kwargs) -> List[Dict]:
        """
        Generate multiple variations of the same whistle.
        
        Args:
            whistle_audio: Whistle audio input
            num_variations: Number of variations to generate
            variation_type: Type of variation ("random", "style", "complexity")
            **kwargs: Base generation parameters
            
        Returns:
            List of variation results
        """
        variations = []
        
        for i in range(num_variations):
            if variation_type == "random":
                # Random parameter variation
                var_params = self._get_random_variation_params(**kwargs)
            elif variation_type == "style":
                # Style-based variation
                styles = ["classical", "romantic", "baroque", "impressionist"]
                style = random.choice(styles)
                var_params = self._get_style_parameters(style, 0.5, 0.5)
                var_params.update(kwargs)
            elif variation_type == "complexity":
                # Complexity-based variation
                complexity = i / (num_variations - 1) if num_variations > 1 else 0.5
                var_params = self._get_complexity_variation_params(complexity, **kwargs)
            else:
                var_params = kwargs.copy()
            
            # Generate variation
            result = self.converter.convert_whistle_to_music(
                whistle_audio=whistle_audio,
                **var_params
            )
            
            result['variation_info'] = {
                'variation_id': i,
                'variation_type': variation_type,
                'parameters': var_params
            }
            
            variations.append(result)
        
        return variations
    
    def _get_random_variation_params(self, **base_params) -> Dict:
        """Get random variation parameters."""
        params = base_params.copy()
        
        # Vary temperature
        if 'temperature' in params:
            params['temperature'] = np.clip(
                params['temperature'] + np.random.uniform(-0.3, 0.3),
                0.1, 2.0
            )
        
        # Vary top_k
        if 'top_k' in params:
            params['top_k'] = max(10, int(params['top_k'] + np.random.randint(-20, 20)))
        
        # Vary top_p
        if 'top_p' in params:
            params['top_p'] = np.clip(
                params['top_p'] + np.random.uniform(-0.2, 0.2),
                0.1, 1.0
            )
        
        return params
    
    def _get_complexity_variation_params(self, complexity: float, **base_params) -> Dict:
        """Get complexity-based variation parameters."""
        params = base_params.copy()
        
        # Adjust parameters based on complexity
        params['temperature'] = 0.5 + complexity * 0.5
        params['top_k'] = int(20 + complexity * 60)
        params['max_length'] = int(200 + complexity * 400)
        
        return params
    
    def generate_with_constraints(self, 
                                 whistle_audio: Union[str, np.ndarray],
                                 constraints: Dict[str, Any],
                                 **kwargs) -> Dict:
        """
        Generate music with specific constraints.
        
        Args:
            whistle_audio: Whistle audio input
            constraints: Dictionary of constraints
            **kwargs: Generation parameters
            
        Returns:
            Generated music result
        """
        # Generate initial music
        result = self.converter.convert_whistle_to_music(
            whistle_audio=whistle_audio,
            **kwargs
        )
        
        # Apply constraints
        result = self._apply_constraints(result, constraints)
        
        return result
    
    def _apply_constraints(self, result: Dict, constraints: Dict) -> Dict:
        """Apply constraints to generated music."""
        midi = result['midi']
        
        # Apply tempo constraint
        if 'tempo' in constraints:
            self._adjust_tempo(midi, constraints['tempo'])
        
        # Apply key constraint
        if 'key' in constraints:
            self._transpose_to_key(midi, constraints['key'])
        
        # Apply time_signature constraint
        if 'time_signature' in constraints:
            self._adjust_time_signature(midi, constraints['time_signature'])
        
        # Apply duration constraint
        if 'max_duration' in constraints:
            self._limit_duration(midi, constraints['max_duration'])
        
        return result
    
    def _adjust_tempo(self, midi: pretty_midi.PrettyMIDI, target_tempo: float):
        """Adjust tempo of MIDI."""
        # Calculate current tempo (simplified)
        if len(midi.instruments) > 0 and len(midi.instruments[0].notes) > 0:
            first_note = midi.instruments[0].notes[0]
            last_note = midi.instruments[0].notes[-1]
            current_duration = last_note.end - first_note.start
            
            # Estimate current tempo (simplified)
            current_tempo = 120  # Default assumption
            
            # Calculate tempo ratio
            tempo_ratio = target_tempo / current_tempo
            
            # Adjust all note times
            for instrument in midi.instruments:
                for note in instrument.notes:
                    note.start *= tempo_ratio
                    note.end *= tempo_ratio
    
    def _transpose_to_key(self, midi: pretty_midi.PrettyMIDI, target_key: str):
        """Transpose MIDI to target key."""
        # Key mapping (simplified)
        key_offsets = {
            'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
            'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
        }
        
        if target_key in key_offsets:
            offset = key_offsets[target_key]
            
            for instrument in midi.instruments:
                for note in instrument.notes:
                    note.pitch = (note.pitch + offset) % 12 + (note.pitch // 12) * 12
    
    def _adjust_time_signature(self, midi: pretty_midi.PrettyMIDI, time_signature: Tuple[int, int]):
        """Adjust time signature (simplified implementation)."""
        # This is a placeholder - actual implementation would be more complex
        pass
    
    def _limit_duration(self, midi: pretty_midi.PrettyMIDI, max_duration: float):
        """Limit duration of MIDI."""
        for instrument in midi.instruments:
            # Remove notes that exceed max duration
            instrument.notes = [
                note for note in instrument.notes 
                if note.end <= max_duration
            ]
            
            # Truncate notes that start before but end after max duration
            for note in instrument.notes:
                if note.start < max_duration < note.end:
                    note.end = max_duration
    
    def generate_ensemble(self, 
                         whistle_audio: Union[str, np.ndarray],
                         instruments: List[str] = None,
                         **kwargs) -> Dict:
        """
        Generate music for multiple instruments.
        
        Args:
            whistle_audio: Whistle audio input
            instruments: List of instrument names
            **kwargs: Generation parameters
            
        Returns:
            Generated ensemble music
        """
        if instruments is None:
            instruments = ['piano', 'violin', 'cello']
        
        # Generate base melody
        base_result = self.converter.convert_whistle_to_music(
            whistle_audio=whistle_audio,
            **kwargs
        )
        
        # Create ensemble
        ensemble_midi = pretty_midi.PrettyMIDI()
        
        # Add instruments
        instrument_programs = {
            'piano': 0,
            'violin': 40,
            'cello': 42,
            'flute': 73,
            'oboe': 68,
            'clarinet': 71,
            'trumpet': 56,
            'horn': 60
        }
        
        for i, instrument_name in enumerate(instruments):
            if instrument_name in instrument_programs:
                instrument = pretty_midi.Instrument(
                    program=instrument_programs[instrument_name]
                )
                
                # Copy and modify notes for this instrument
                for note in base_result['midi'].instruments[0].notes:
                    # Transpose and adjust for different instruments
                    new_note = pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=self._transpose_for_instrument(note.pitch, instrument_name),
                        start=note.start + i * 0.1,  # Slight delay
                        end=note.end + i * 0.1
                    )
                    instrument.notes.append(new_note)
                
                ensemble_midi.instruments.append(instrument)
        
        # Update result
        base_result['midi'] = ensemble_midi
        base_result['ensemble_info'] = {
            'instruments': instruments,
            'num_instruments': len(instruments)
        }
        
        return base_result
    
    def _transpose_for_instrument(self, pitch: int, instrument: str) -> int:
        """Transpose pitch for specific instrument."""
        transpositions = {
            'piano': 0,
            'violin': 0,
            'cello': -12,  # One octave lower
            'flute': 0,
            'oboe': 0,
            'clarinet': 0,
            'trumpet': 0,
            'horn': -7  # Perfect fifth lower
        }
        
        return pitch + transpositions.get(instrument, 0)
    
    def save_generation_report(self, 
                              results: List[Dict], 
                              output_path: str):
        """Save a detailed generation report."""
        report = {
            'generation_info': {
                'num_generations': len(results),
                'timestamp': str(np.datetime64('now')),
                'model_info': self.converter.get_model_info()
            },
            'results': []
        }
        
        for i, result in enumerate(results):
            result_info = {
                'generation_id': i,
                'metadata': result.get('metadata', {}),
                'variation_info': result.get('variation_info', {}),
                'ensemble_info': result.get('ensemble_info', {}),
                'output_path': result.get('output_path', '')
            }
            report['results'].append(result_info)
        
        # Save as JSON
        import json
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Generation report saved to {output_path}")
