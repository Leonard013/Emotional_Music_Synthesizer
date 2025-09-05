"""
MIDI processing utilities for the MAESTRO dataset.
Handles MIDI file loading, preprocessing, and conversion to model inputs.
"""

import numpy as np
import pretty_midi
import mido
from typing import List, Dict, Tuple, Optional, Union
import torch
from collections import defaultdict
import librosa


class MidiProcessor:
    """Processes MIDI files for training and inference."""
    
    def __init__(self, 
                 max_sequence_length: int = 1024,
                 min_note: int = 21,  # A0
                 max_note: int = 108,  # C8
                 velocity_bins: int = 32,
                 time_resolution: float = 0.125):  # 1/8 note
        """
        Initialize MIDI processor.
        
        Args:
            max_sequence_length: Maximum sequence length for model input
            min_note: Minimum MIDI note number
            max_note: Maximum MIDI note number
            velocity_bins: Number of velocity quantization bins
            time_resolution: Time resolution in seconds
        """
        self.max_sequence_length = max_sequence_length
        self.min_note = min_note
        self.max_note = max_note
        self.velocity_bins = velocity_bins
        self.time_resolution = time_resolution
        self.note_range = max_note - min_note + 1
        
        # Special tokens
        self.PAD_TOKEN = 0
        self.START_TOKEN = 1
        self.END_TOKEN = 2
        self.VELOCITY_OFFSET = 3
        self.TIME_OFFSET = self.VELOCITY_OFFSET + velocity_bins
        self.NOTE_OFFSET = self.TIME_OFFSET + 1000  # Support up to 1000 time steps
        
        self.vocab_size = self.NOTE_OFFSET + self.note_range
        
    def load_midi(self, file_path: str) -> pretty_midi.PrettyMIDI:
        """
        Load MIDI file.
        
        Args:
            file_path: Path to MIDI file
            
        Returns:
            PrettyMIDI object
        """
        try:
            return pretty_midi.PrettyMIDI(file_path)
        except Exception as e:
            raise ValueError(f"Error loading MIDI file {file_path}: {e}")
    
    def quantize_time(self, time: float) -> int:
        """
        Quantize time to discrete steps.
        
        Args:
            time: Time in seconds
            
        Returns:
            Quantized time step
        """
        return int(round(time / self.time_resolution))
    
    def quantize_velocity(self, velocity: int) -> int:
        """
        Quantize velocity to bins.
        
        Args:
            velocity: MIDI velocity (0-127)
            
        Returns:
            Quantized velocity bin
        """
        return min(int(velocity * self.velocity_bins / 128), self.velocity_bins - 1)
    
    def extract_notes(self, midi: pretty_midi.PrettyMIDI) -> List[Dict]:
        """
        Extract note events from MIDI.
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            List of note dictionaries
        """
        notes = []
        
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue
                
            for note in instrument.notes:
                # Filter notes within range
                if self.min_note <= note.pitch <= self.max_note:
                    note_dict = {
                        'start_time': note.start,
                        'end_time': note.end,
                        'pitch': note.pitch,
                        'velocity': note.velocity,
                        'duration': note.end - note.start
                    }
                    notes.append(note_dict)
        
        # Sort by start time
        notes.sort(key=lambda x: x['start_time'])
        
        return notes
    
    def extract_events(self, midi: pretty_midi.PrettyMIDI) -> List[Dict]:
        """
        Extract all musical events (notes, pedals) from MIDI.
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            List of event dictionaries
        """
        events = []
        
        # Extract note events
        notes = self.extract_notes(midi)
        for note in notes:
            # Note on event
            events.append({
                'time': note['start_time'],
                'type': 'note_on',
                'pitch': note['pitch'],
                'velocity': note['velocity']
            })
            
            # Note off event
            events.append({
                'time': note['end_time'],
                'type': 'note_off',
                'pitch': note['pitch'],
                'velocity': 0
            })
        
        # Extract pedal events
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue
                
            # Sustain pedal (CC 64)
            for cc in instrument.control_changes:
                if cc.number == 64:  # Sustain pedal
                    events.append({
                        'time': cc.time,
                        'type': 'sustain_pedal',
                        'value': cc.value
                    })
        
        # Sort by time
        events.sort(key=lambda x: x['time'])
        
        return events
    
    def events_to_tokens(self, events: List[Dict]) -> List[int]:
        """
        Convert events to token sequence.
        
        Args:
            events: List of musical events
            
        Returns:
            List of tokens
        """
        tokens = [self.START_TOKEN]
        
        current_time = 0
        
        for event in events:
            # Add time tokens if needed
            event_time = self.quantize_time(event['time'])
            time_diff = event_time - current_time
            
            if time_diff > 0:
                # Add time step tokens
                for _ in range(min(time_diff, 1000)):  # Cap at 1000 time steps
                    tokens.append(self.TIME_OFFSET + 1)  # +1 for time step
                current_time = event_time
            
            # Add event token
            if event['type'] == 'note_on':
                velocity_bin = self.quantize_velocity(event['velocity'])
                note_token = self.NOTE_OFFSET + (event['pitch'] - self.min_note)
                tokens.extend([
                    self.VELOCITY_OFFSET + velocity_bin,
                    note_token
                ])
            elif event['type'] == 'note_off':
                note_token = self.NOTE_OFFSET + (event['pitch'] - self.min_note)
                tokens.append(note_token)  # Note off uses same token as note on
            elif event['type'] == 'sustain_pedal':
                # Add pedal token (simplified)
                tokens.append(self.NOTE_OFFSET + self.note_range)  # Special pedal token
        
        tokens.append(self.END_TOKEN)
        
        return tokens
    
    def tokens_to_events(self, tokens: List[int]) -> List[Dict]:
        """
        Convert token sequence back to events.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of musical events
        """
        events = []
        current_time = 0
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            if token == self.START_TOKEN:
                i += 1
                continue
            elif token == self.END_TOKEN:
                break
            elif self.TIME_OFFSET <= token < self.NOTE_OFFSET:
                # Time step token
                time_steps = token - self.TIME_OFFSET
                current_time += time_steps * self.time_resolution
                i += 1
            elif self.VELOCITY_OFFSET <= token < self.TIME_OFFSET:
                # Velocity token
                velocity_bin = token - self.VELOCITY_OFFSET
                velocity = int(velocity_bin * 128 / self.velocity_bins)
                
                # Next token should be note
                if i + 1 < len(tokens):
                    note_token = tokens[i + 1]
                    if self.NOTE_OFFSET <= note_token < self.NOTE_OFFSET + self.note_range:
                        pitch = self.min_note + (note_token - self.NOTE_OFFSET)
                        events.append({
                            'time': current_time,
                            'type': 'note_on',
                            'pitch': pitch,
                            'velocity': velocity
                        })
                        i += 2
                    else:
                        i += 1
                else:
                    i += 1
            elif self.NOTE_OFFSET <= token < self.NOTE_OFFSET + self.note_range:
                # Note token (note off)
                pitch = self.min_note + (token - self.NOTE_OFFSET)
                events.append({
                    'time': current_time,
                    'type': 'note_off',
                    'pitch': pitch,
                    'velocity': 0
                })
                i += 1
            else:
                i += 1
        
        return events
    
    def events_to_midi(self, events: List[Dict], tempo: float = 120.0) -> pretty_midi.PrettyMIDI:
        """
        Convert events back to MIDI.
        
        Args:
            events: List of musical events
            tempo: Tempo in BPM
            
        Returns:
            PrettyMIDI object
        """
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)  # Piano
        
        # Track active notes
        active_notes = {}
        
        for event in events:
            if event['type'] == 'note_on':
                active_notes[event['pitch']] = event['time']
            elif event['type'] == 'note_off' and event['pitch'] in active_notes:
                start_time = active_notes[event['pitch']]
                note = pretty_midi.Note(
                    velocity=event['velocity'],
                    pitch=event['pitch'],
                    start=start_time,
                    end=event['time']
                )
                instrument.notes.append(note)
                del active_notes[event['pitch']]
        
        # Close any remaining notes
        for pitch, start_time in active_notes.items():
            note = pretty_midi.Note(
                velocity=64,
                pitch=pitch,
                start=start_time,
                end=start_time + 1.0  # Default duration
            )
            instrument.notes.append(note)
        
        midi.instruments.append(instrument)
        return midi
    
    def create_sequence_pairs(self, tokens: List[int]) -> List[Tuple[List[int], List[int]]]:
        """
        Create input-target pairs for training.
        
        Args:
            tokens: Full token sequence
            
        Returns:
            List of (input, target) pairs
        """
        pairs = []
        
        for i in range(1, len(tokens) - self.max_sequence_length):
            input_seq = tokens[i:i + self.max_sequence_length]
            target_seq = tokens[i + 1:i + self.max_sequence_length + 1]
            pairs.append((input_seq, target_seq))
        
        return pairs
    
    def pad_sequence(self, tokens: List[int], pad_length: int = None) -> List[int]:
        """
        Pad token sequence to specified length.
        
        Args:
            tokens: Input token sequence
            pad_length: Target length (default: max_sequence_length)
            
        Returns:
            Padded token sequence
        """
        if pad_length is None:
            pad_length = self.max_sequence_length
        
        if len(tokens) >= pad_length:
            return tokens[:pad_length]
        else:
            return tokens + [self.PAD_TOKEN] * (pad_length - len(tokens))
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token mappings."""
        return {
            'PAD': self.PAD_TOKEN,
            'START': self.START_TOKEN,
            'END': self.END_TOKEN,
            'VELOCITY_OFFSET': self.VELOCITY_OFFSET,
            'TIME_OFFSET': self.TIME_OFFSET,
            'NOTE_OFFSET': self.NOTE_OFFSET
        }
