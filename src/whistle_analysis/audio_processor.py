"""
Audio processing utilities for whistle analysis.
Handles loading, preprocessing, and basic audio operations.
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional, Union
import torch
import torchaudio


class AudioProcessor:
    """Handles audio loading, preprocessing, and basic operations."""
    
    def __init__(self, sample_rate: int = 22050, hop_length: int = 512):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Target sample rate for audio processing
            hop_length: Hop length for STFT and other operations
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frame_length = hop_length * 4  # 2048 for hop_length=512
        
    def load_audio(self, file_path: str, mono: bool = True) -> np.ndarray:
        """
        Load audio file and convert to target sample rate.
        
        Args:
            file_path: Path to audio file
            mono: Whether to convert to mono
            
        Returns:
            Audio signal as numpy array
        """
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=mono)
            return audio
        except Exception as e:
            raise ValueError(f"Error loading audio file {file_path}: {e}")
    
    def save_audio(self, audio: np.ndarray, file_path: str) -> None:
        """
        Save audio array to file.
        
        Args:
            audio: Audio signal
            file_path: Output file path
        """
        sf.write(file_path, audio, self.sample_rate)
    
    def normalize_audio(self, audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """
        Normalize audio to target dB level.
        
        Args:
            audio: Input audio signal
            target_db: Target dB level
            
        Returns:
            Normalized audio signal
        """
        # Calculate RMS and convert to dB
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            current_db = 20 * np.log10(rms)
            gain_db = target_db - current_db
            gain_linear = 10**(gain_db / 20)
            return audio * gain_linear
        return audio
    
    def apply_preemphasis(self, audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
        """
        Apply preemphasis filter to audio.
        
        Args:
            audio: Input audio signal
            coeff: Preemphasis coefficient
            
        Returns:
            Preemphasized audio signal
        """
        return np.append(audio[0], audio[1:] - coeff * audio[:-1])
    
    def remove_silence(self, audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """
        Remove silence from beginning and end of audio.
        
        Args:
            audio: Input audio signal
            threshold: Silence threshold
            
        Returns:
            Audio with silence removed
        """
        # Find non-silent regions
        non_silent = np.abs(audio) > threshold
        
        # Find first and last non-silent samples
        first_non_silent = np.argmax(non_silent)
        last_non_silent = len(audio) - np.argmax(non_silent[::-1])
        
        return audio[first_non_silent:last_non_silent]
    
    def resample_audio(self, audio: np.ndarray, target_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate.
        
        Args:
            audio: Input audio signal
            target_sr: Target sample rate
            
        Returns:
            Resampled audio signal
        """
        if target_sr != self.sample_rate:
            return librosa.resample(audio, orig_sr=self.sample_rate, target_sr=target_sr)
        return audio
    
    def compute_stft(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute Short-Time Fourier Transform.
        
        Args:
            audio: Input audio signal
            
        Returns:
            STFT magnitude spectrogram
        """
        stft = librosa.stft(
            audio, 
            n_fft=self.frame_length, 
            hop_length=self.hop_length,
            window='hann'
        )
        return np.abs(stft)
    
    def compute_mel_spectrogram(self, audio: np.ndarray, n_mels: int = 128) -> np.ndarray:
        """
        Compute mel spectrogram.
        
        Args:
            audio: Input audio signal
            n_mels: Number of mel bins
            
        Returns:
            Mel spectrogram
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
            n_mels=n_mels,
            fmax=self.sample_rate // 2
        )
        return librosa.power_to_db(mel_spec, ref=np.max)
    
    def preprocess_whistle(self, audio: np.ndarray) -> np.ndarray:
        """
        Complete preprocessing pipeline for whistle audio.
        
        Args:
            audio: Raw whistle audio
            
        Returns:
            Preprocessed audio ready for feature extraction
        """
        # Remove silence
        audio = self.remove_silence(audio)
        
        # Normalize
        audio = self.normalize_audio(audio)
        
        # Apply preemphasis
        audio = self.apply_preemphasis(audio)
        
        return audio
