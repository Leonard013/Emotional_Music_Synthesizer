"""
MIDI utilities for the whistle-to-music synthesizer.
"""

import numpy as np
import pretty_midi
import librosa
import soundfile as sf
from typing import Union, Optional, Dict, List, Tuple, Any
from pathlib import Path
import warnings


def midi_to_audio(midi_path: Union[str, Path],
                  output_path: Optional[Union[str, Path]] = None,
                  sample_rate: int = 22050,
                  duration: Optional[float] = None) -> np.ndarray:
    """
    Convert MIDI file to audio.
    
    Args:
        midi_path: Path to MIDI file
        output_path: Path to save audio file (optional)
        sample_rate: Sample rate for audio
        duration: Maximum duration in seconds (optional)
        
    Returns:
        Audio signal as numpy array
    """
    try:
        # Load MIDI file
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        
        # Synthesize audio
        if duration is None:
            duration = midi.get_end_time()
        
        audio = midi.synthesize(fs=sample_rate, duration=duration)
        
        # Save audio if output path provided
        if output_path:
            sf.write(output_path, audio, sample_rate)
        
        return audio
    
    except Exception as e:
        print(f"Error converting MIDI to audio: {e}")
        return np.array([])


def audio_to_midi(audio_path: Union[str, Path],
                  output_path: Optional[Union[str, Path]] = None,
                  sample_rate: int = 22050) -> pretty_midi.PrettyMIDI:
    """
    Convert audio file to MIDI (simplified implementation).
    
    Args:
        audio_path: Path to audio file
        output_path: Path to save MIDI file (optional)
        sample_rate: Sample rate for audio processing
        
    Returns:
        PrettyMIDI object
    """
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=sample_rate)
        
        # Simple pitch detection and MIDI conversion
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)  # Piano
        
        # Detect onsets
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # Detect pitch for each onset
        for onset_time in onset_times:
            # Extract audio segment around onset
            start_sample = int(onset_time * sr)
            end_sample = min(start_sample + int(0.5 * sr), len(audio))
            segment = audio[start_sample:end_sample]
            
            if len(segment) > 0:
                # Detect pitch
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    segment, fmin=80, fmax=2000, sr=sr
                )
                
                # Get most likely pitch
                if np.any(voiced_flag):
                    pitch_freq = np.median(f0[voiced_flag])
                    if not np.isnan(pitch_freq):
                        # Convert frequency to MIDI note
                        midi_note = int(12 * np.log2(pitch_freq / 440.0) + 69)
                        midi_note = np.clip(midi_note, 21, 108)  # A0 to C8
                        
                        # Create note
                        note = pretty_midi.Note(
                            velocity=64,
                            pitch=midi_note,
                            start=onset_time,
                            end=onset_time + 0.5
                        )
                        instrument.notes.append(note)
        
        midi.instruments.append(instrument)
        
        # Save MIDI if output path provided
        if output_path:
            midi.write(str(output_path))
        
        return midi
    
    except Exception as e:
        print(f"Error converting audio to MIDI: {e}")
        return pretty_midi.PrettyMIDI()


def get_midi_info(midi_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about MIDI file.
    
    Args:
        midi_path: Path to MIDI file
        
    Returns:
        Dictionary with MIDI information
    """
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        
        # Basic information
        duration = midi.get_end_time()
        tempo = midi.estimate_tempo()
        
        # Count notes and instruments
        total_notes = sum(len(instrument.notes) for instrument in midi.instruments)
        num_instruments = len(midi.instruments)
        
        # Get note statistics
        all_notes = []
        for instrument in midi.instruments:
            all_notes.extend(instrument.notes)
        
        if all_notes:
            pitches = [note.pitch for note in all_notes]
            velocities = [note.velocity for note in all_notes]
            durations = [note.end - note.start for note in all_notes]
            
            pitch_range = (min(pitches), max(pitches))
            velocity_range = (min(velocities), max(velocities))
            duration_range = (min(durations), max(durations))
        else:
            pitch_range = (0, 0)
            velocity_range = (0, 0)
            duration_range = (0, 0)
        
        # Get time signature
        time_signatures = midi.time_signature_changes
        if time_signatures:
            time_sig = time_signatures[0]
            time_signature = (time_sig.numerator, time_sig.denominator)
        else:
            time_signature = (4, 4)  # Default
        
        # Get key signature
        key_signatures = midi.key_signature_changes
        if key_signatures:
            key_sig = key_signatures[0]
            key_signature = key_sig.key_number
        else:
            key_signature = 0  # C major
        
        return {
            'duration_seconds': duration,
            'tempo_bpm': tempo,
            'total_notes': total_notes,
            'num_instruments': num_instruments,
            'pitch_range': pitch_range,
            'velocity_range': velocity_range,
            'duration_range': duration_range,
            'time_signature': time_signature,
            'key_signature': key_signature,
            'instruments': [
                {
                    'program': instrument.program,
                    'is_drum': instrument.is_drum,
                    'num_notes': len(instrument.notes)
                }
                for instrument in midi.instruments
            ]
        }
    
    except Exception as e:
        print(f"Error getting MIDI info: {e}")
        return {}


def transpose_midi(midi_path: Union[str, Path],
                  output_path: Union[str, Path],
                  semitones: int) -> bool:
    """
    Transpose MIDI file by specified number of semitones.
    
    Args:
        midi_path: Path to input MIDI file
        output_path: Path to output MIDI file
        semitones: Number of semitones to transpose (positive = up, negative = down)
        
    Returns:
        True if successful
    """
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        
        # Transpose all notes
        for instrument in midi.instruments:
            for note in instrument.notes:
                note.pitch = np.clip(note.pitch + semitones, 0, 127)
        
        # Save transposed MIDI
        midi.write(str(output_path))
        return True
    
    except Exception as e:
        print(f"Error transposing MIDI: {e}")
        return False


def change_tempo(midi_path: Union[str, Path],
                output_path: Union[str, Path],
                tempo_ratio: float) -> bool:
    """
    Change tempo of MIDI file.
    
    Args:
        midi_path: Path to input MIDI file
        output_path: Path to output MIDI file
        tempo_ratio: Tempo ratio (1.0 = original, 2.0 = double speed, 0.5 = half speed)
        
    Returns:
        True if successful
    """
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        
        # Adjust all note times
        for instrument in midi.instruments:
            for note in instrument.notes:
                note.start *= tempo_ratio
                note.end *= tempo_ratio
        
        # Adjust control changes
        for instrument in midi.instruments:
            for cc in instrument.control_changes:
                cc.time *= tempo_ratio
        
        # Save tempo-changed MIDI
        midi.write(str(output_path))
        return True
    
    except Exception as e:
        print(f"Error changing tempo: {e}")
        return False


def extract_melody(midi_path: Union[str, Path],
                  output_path: Union[str, Path],
                  instrument_program: int = 0) -> bool:
    """
    Extract melody from MIDI file (highest pitched instrument).
    
    Args:
        midi_path: Path to input MIDI file
        output_path: Path to output MIDI file
        instrument_program: Program number for melody instrument
        
    Returns:
        True if successful
    """
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        
        # Find instrument with highest average pitch
        best_instrument = None
        best_avg_pitch = 0
        
        for instrument in midi.instruments:
            if instrument.notes:
                avg_pitch = np.mean([note.pitch for note in instrument.notes])
                if avg_pitch > best_avg_pitch:
                    best_avg_pitch = avg_pitch
                    best_instrument = instrument
        
        if best_instrument is None:
            return False
        
        # Create new MIDI with only melody
        melody_midi = pretty_midi.PrettyMIDI()
        melody_instrument = pretty_midi.Instrument(program=instrument_program)
        melody_instrument.notes = best_instrument.notes.copy()
        melody_midi.instruments.append(melody_instrument)
        
        # Save melody MIDI
        melody_midi.write(str(output_path))
        return True
    
    except Exception as e:
        print(f"Error extracting melody: {e}")
        return False


def merge_midi_files(midi_paths: List[Union[str, Path]],
                    output_path: Union[str, Path]) -> bool:
    """
    Merge multiple MIDI files into one.
    
    Args:
        midi_paths: List of MIDI file paths
        output_path: Path to output merged MIDI file
        
    Returns:
        True if successful
    """
    try:
        merged_midi = pretty_midi.PrettyMIDI()
        
        for midi_path in midi_paths:
            midi = pretty_midi.PrettyMIDI(str(midi_path))
            
            # Add all instruments from this MIDI
            for instrument in midi.instruments:
                merged_midi.instruments.append(instrument)
        
        # Save merged MIDI
        merged_midi.write(str(output_path))
        return True
    
    except Exception as e:
        print(f"Error merging MIDI files: {e}")
        return False


def quantize_midi(midi_path: Union[str, Path],
                 output_path: Union[str, Path],
                 quantization_level: float = 0.25) -> bool:
    """
    Quantize MIDI timing to specified level.
    
    Args:
        midi_path: Path to input MIDI file
        output_path: Path to output MIDI file
        quantization_level: Quantization level in beats (0.25 = 16th notes)
        
    Returns:
        True if successful
    """
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        
        # Quantize all note times
        for instrument in midi.instruments:
            for note in instrument.notes:
                # Quantize start time
                note.start = round(note.start / quantization_level) * quantization_level
                # Quantize end time
                note.end = round(note.end / quantization_level) * quantization_level
        
        # Save quantized MIDI
        midi.write(str(output_path))
        return True
    
    except Exception as e:
        print(f"Error quantizing MIDI: {e}")
        return False


def add_reverb_to_midi(midi_path: Union[str, Path],
                      output_path: Union[str, Path],
                      reverb_amount: float = 0.3) -> bool:
    """
    Add reverb effect to MIDI by extending note durations.
    
    Args:
        midi_path: Path to input MIDI file
        output_path: Path to output MIDI file
        reverb_amount: Reverb amount (0.0 to 1.0)
        
    Returns:
        True if successful
    """
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        
        # Extend note durations for reverb effect
        for instrument in midi.instruments:
            for note in instrument.notes:
                # Extend note end time
                note.end += (note.end - note.start) * reverb_amount
                
                # Reduce velocity for reverb tail
                note.velocity = int(note.velocity * (1 - reverb_amount * 0.5))
        
        # Save MIDI with reverb
        midi.write(str(output_path))
        return True
    
    except Exception as e:
        print(f"Error adding reverb to MIDI: {e}")
        return False


def analyze_midi_complexity(midi_path: Union[str, Path]) -> Dict[str, float]:
    """
    Analyze complexity of MIDI file.
    
    Args:
        midi_path: Path to MIDI file
        
    Returns:
        Dictionary of complexity metrics
    """
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        
        # Get all notes
        all_notes = []
        for instrument in midi.instruments:
            all_notes.extend(instrument.notes)
        
        if not all_notes:
            return {'complexity': 0.0}
        
        # Calculate complexity metrics
        pitches = [note.pitch for note in all_notes]
        velocities = [note.velocity for note in all_notes]
        durations = [note.end - note.start for note in all_notes]
        
        # Pitch complexity (range and variance)
        pitch_range = max(pitches) - min(pitches)
        pitch_variance = np.var(pitches)
        
        # Velocity complexity
        velocity_variance = np.var(velocities)
        
        # Duration complexity
        duration_variance = np.var(durations)
        
        # Note density
        duration = midi.get_end_time()
        note_density = len(all_notes) / duration if duration > 0 else 0
        
        # Overall complexity score
        complexity = (
            pitch_range / 88.0 +  # Normalize to piano range
            pitch_variance / 1000.0 +
            velocity_variance / 1000.0 +
            duration_variance / 10.0 +
            note_density / 10.0
        ) / 5.0
        
        return {
            'complexity': complexity,
            'pitch_range': pitch_range,
            'pitch_variance': pitch_variance,
            'velocity_variance': velocity_variance,
            'duration_variance': duration_variance,
            'note_density': note_density,
            'total_notes': len(all_notes),
            'duration': duration
        }
    
    except Exception as e:
        print(f"Error analyzing MIDI complexity: {e}")
        return {'complexity': 0.0}
