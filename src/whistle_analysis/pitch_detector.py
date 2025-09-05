"""
Pitch detection and analysis for whistle audio.
Extracts fundamental frequency and related features.
"""

import numpy as np
import librosa
from typing import Tuple, Optional, List
from scipy.signal import find_peaks
import torch


class PitchDetector:
    """Detects and analyzes pitch in whistle audio."""
    
    def __init__(self, sample_rate: int = 22050, hop_length: int = 512):
        """
        Initialize pitch detector.
        
        Args:
            sample_rate: Audio sample rate
            hop_length: Hop length for analysis
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frame_length = hop_length * 4
        
        # Pitch detection parameters
        self.fmin = 80.0  # Minimum frequency (Hz)
        self.fmax = 2000.0  # Maximum frequency (Hz)
        
    def detect_pitch_pyin(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect pitch using PYIN algorithm.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Tuple of (frequencies, voiced_flags)
        """
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            threshold=0.1
        )
        return f0, voiced_flag
    
    def detect_pitch_crepe(self, audio: np.ndarray) -> np.ndarray:
        """
        Detect pitch using CREPE model (if available).
        
        Args:
            audio: Input audio signal
            
        Returns:
            Array of pitch frequencies
        """
        try:
            import crepe
            time, frequency, confidence, activation = crepe.predict(
                audio, 
                sr=self.sample_rate, 
                model_capacity='medium',
                viterbi=True
            )
            return frequency
        except ImportError:
            print("CREPE not available, falling back to PYIN")
            f0, _ = self.detect_pitch_pyin(audio)
            return f0
    
    def smooth_pitch(self, f0: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Smooth pitch contour using median filtering.
        
        Args:
            f0: Raw pitch frequencies
            window_size: Size of smoothing window
            
        Returns:
            Smoothed pitch contour
        """
        from scipy.ndimage import median_filter
        
        # Replace NaN values with 0 for filtering
        f0_clean = np.nan_to_num(f0, nan=0.0)
        
        # Apply median filter
        smoothed = median_filter(f0_clean, size=window_size)
        
        # Restore NaN values where original was NaN
        smoothed[np.isnan(f0)] = np.nan
        
        return smoothed
    
    def extract_pitch_features(self, audio: np.ndarray) -> dict:
        """
        Extract comprehensive pitch features from audio.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary of pitch features
        """
        # Detect pitch
        f0, voiced_flag = self.detect_pitch_pyin(audio)
        
        # Smooth pitch contour
        f0_smooth = self.smooth_pitch(f0)
        
        # Convert to MIDI note numbers
        midi_notes = self.freq_to_midi(f0_smooth)
        
        # Calculate pitch statistics
        valid_f0 = f0_smooth[~np.isnan(f0_smooth)]
        
        features = {
            'f0_raw': f0,
            'f0_smooth': f0_smooth,
            'voiced_flag': voiced_flag,
            'midi_notes': midi_notes,
            'pitch_mean': np.mean(valid_f0) if len(valid_f0) > 0 else 0,
            'pitch_std': np.std(valid_f0) if len(valid_f0) > 0 else 0,
            'pitch_range': np.ptp(valid_f0) if len(valid_f0) > 0 else 0,
            'voiced_ratio': np.mean(voiced_flag) if voiced_flag is not None else 0
        }
        
        return features
    
    def freq_to_midi(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Convert frequencies to MIDI note numbers.
        
        Args:
            frequencies: Array of frequencies in Hz
            
        Returns:
            Array of MIDI note numbers
        """
        # Handle NaN values
        valid_mask = ~np.isnan(frequencies)
        midi_notes = np.full_like(frequencies, np.nan)
        
        if np.any(valid_mask):
            valid_freqs = frequencies[valid_mask]
            midi_notes[valid_mask] = 12 * np.log2(valid_freqs / 440.0) + 69
        
        return midi_notes
    
    def midi_to_freq(self, midi_notes: np.ndarray) -> np.ndarray:
        """
        Convert MIDI note numbers to frequencies.
        
        Args:
            midi_notes: Array of MIDI note numbers
            
        Returns:
            Array of frequencies in Hz
        """
        # Handle NaN values
        valid_mask = ~np.isnan(midi_notes)
        frequencies = np.full_like(midi_notes, np.nan)
        
        if np.any(valid_mask):
            valid_midi = midi_notes[valid_mask]
            frequencies[valid_mask] = 440.0 * (2 ** ((valid_midi - 69) / 12))
        
        return frequencies
    
    def quantize_pitch(self, f0: np.ndarray, semitone_tolerance: float = 0.5) -> np.ndarray:
        """
        Quantize pitch to nearest semitone.
        
        Args:
            f0: Input pitch frequencies
            semitone_tolerance: Tolerance for quantization in semitones
            
        Returns:
            Quantized pitch frequencies
        """
        # Convert to MIDI notes
        midi_notes = self.freq_to_midi(f0)
        
        # Quantize to nearest integer MIDI note
        quantized_midi = np.round(midi_notes)
        
        # Convert back to frequencies
        quantized_f0 = self.midi_to_freq(quantized_midi)
        
        return quantized_f0
    
    def extract_pitch_contour(self, audio: np.ndarray, segment_length: float = 0.1) -> List[dict]:
        """
        Extract pitch contour segments for analysis.
        
        Args:
            audio: Input audio signal
            segment_length: Length of each segment in seconds
            
        Returns:
            List of pitch contour segments
        """
        # Detect pitch
        f0, voiced_flag = self.detect_pitch_pyin(audio)
        
        # Calculate segment size in frames
        segment_frames = int(segment_length * self.sample_rate / self.hop_length)
        
        segments = []
        for i in range(0, len(f0), segment_frames):
            end_idx = min(i + segment_frames, len(f0))
            segment_f0 = f0[i:end_idx]
            segment_voiced = voiced_flag[i:end_idx] if voiced_flag is not None else None
            
            # Calculate segment features
            valid_f0 = segment_f0[~np.isnan(segment_f0)]
            
            segment_info = {
                'start_frame': i,
                'end_frame': end_idx,
                'start_time': i * self.hop_length / self.sample_rate,
                'end_time': end_idx * self.hop_length / self.sample_rate,
                'f0_mean': np.mean(valid_f0) if len(valid_f0) > 0 else np.nan,
                'f0_std': np.std(valid_f0) if len(valid_f0) > 0 else np.nan,
                'voiced_ratio': np.mean(segment_voiced) if segment_voiced is not None else 0,
                'f0_contour': segment_f0
            }
            
            segments.append(segment_info)
        
        return segments
