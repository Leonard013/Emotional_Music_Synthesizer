"""
Audio utilities for the whistle-to-music synthesizer.
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Union, Optional, Tuple, List
from pathlib import Path
import warnings


def convert_audio_format(input_path: Union[str, Path],
                        output_path: Union[str, Path],
                        target_sr: int = 22050,
                        target_format: str = 'wav',
                        mono: bool = True) -> bool:
    """
    Convert audio file to different format.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file
        target_sr: Target sample rate
        target_format: Target format ('wav', 'mp3', 'flac', etc.)
        mono: Whether to convert to mono
        
    Returns:
        True if successful
    """
    try:
        # Load audio
        audio, sr = librosa.load(input_path, sr=target_sr, mono=mono)
        
        # Save audio
        sf.write(output_path, audio, target_sr, format=target_format)
        
        return True
    
    except Exception as e:
        print(f"Error converting audio: {e}")
        return False


def normalize_audio(audio: np.ndarray, 
                   target_db: float = -20.0) -> np.ndarray:
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


def trim_silence(audio: np.ndarray, 
                sr: int = 22050,
                threshold: float = 0.01,
                frame_length: int = 2048,
                hop_length: int = 512) -> np.ndarray:
    """
    Trim silence from beginning and end of audio.
    
    Args:
        audio: Input audio signal
        sr: Sample rate
        threshold: Silence threshold
        frame_length: Frame length for analysis
        hop_length: Hop length for analysis
        
    Returns:
        Audio with silence trimmed
    """
    # Use librosa's trim function
    trimmed_audio, _ = librosa.effects.trim(
        audio,
        top_db=20 * np.log10(threshold),
        frame_length=frame_length,
        hop_length=hop_length
    )
    
    return trimmed_audio


def apply_preemphasis(audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """
    Apply preemphasis filter to audio.
    
    Args:
        audio: Input audio signal
        coeff: Preemphasis coefficient
        
    Returns:
        Preemphasized audio signal
    """
    return np.append(audio[0], audio[1:] - coeff * audio[:-1])


def apply_high_pass_filter(audio: np.ndarray, 
                          sr: int = 22050,
                          cutoff: float = 80.0) -> np.ndarray:
    """
    Apply high-pass filter to audio.
    
    Args:
        audio: Input audio signal
        sr: Sample rate
        cutoff: Cutoff frequency in Hz
        
    Returns:
        Filtered audio signal
    """
    from scipy import signal
    
    # Design high-pass filter
    nyquist = sr / 2
    normalized_cutoff = cutoff / nyquist
    b, a = signal.butter(4, normalized_cutoff, btype='high')
    
    # Apply filter
    filtered_audio = signal.filtfilt(b, a, audio)
    
    return filtered_audio


def apply_low_pass_filter(audio: np.ndarray, 
                         sr: int = 22050,
                         cutoff: float = 8000.0) -> np.ndarray:
    """
    Apply low-pass filter to audio.
    
    Args:
        audio: Input audio signal
        sr: Sample rate
        cutoff: Cutoff frequency in Hz
        
    Returns:
        Filtered audio signal
    """
    from scipy import signal
    
    # Design low-pass filter
    nyquist = sr / 2
    normalized_cutoff = cutoff / nyquist
    b, a = signal.butter(4, normalized_cutoff, btype='low')
    
    # Apply filter
    filtered_audio = signal.filtfilt(b, a, audio)
    
    return filtered_audio


def apply_band_pass_filter(audio: np.ndarray, 
                          sr: int = 22050,
                          low_cutoff: float = 80.0,
                          high_cutoff: float = 8000.0) -> np.ndarray:
    """
    Apply band-pass filter to audio.
    
    Args:
        audio: Input audio signal
        sr: Sample rate
        low_cutoff: Low cutoff frequency in Hz
        high_cutoff: High cutoff frequency in Hz
        
    Returns:
        Filtered audio signal
    """
    from scipy import signal
    
    # Design band-pass filter
    nyquist = sr / 2
    low_norm = low_cutoff / nyquist
    high_norm = high_cutoff / nyquist
    b, a = signal.butter(4, [low_norm, high_norm], btype='band')
    
    # Apply filter
    filtered_audio = signal.filtfilt(b, a, audio)
    
    return filtered_audio


def add_noise(audio: np.ndarray, 
              noise_level: float = 0.01,
              noise_type: str = 'white') -> np.ndarray:
    """
    Add noise to audio signal.
    
    Args:
        audio: Input audio signal
        noise_level: Noise level (0.0 to 1.0)
        noise_type: Type of noise ('white', 'pink', 'brown')
        
    Returns:
        Audio with added noise
    """
    if noise_type == 'white':
        noise = np.random.normal(0, noise_level, len(audio))
    elif noise_type == 'pink':
        # Simplified pink noise
        noise = np.random.normal(0, noise_level, len(audio))
        # Apply 1/f filter approximation
        from scipy import signal
        b = [1, -0.5]
        a = [1]
        noise = signal.lfilter(b, a, noise)
    elif noise_type == 'brown':
        # Brown noise (1/f^2)
        noise = np.random.normal(0, noise_level, len(audio))
        from scipy import signal
        b = [1, -0.8]
        a = [1]
        noise = signal.lfilter(b, a, noise)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return audio + noise


def apply_compression(audio: np.ndarray, 
                     threshold: float = 0.5,
                     ratio: float = 4.0,
                     attack: float = 0.01,
                     release: float = 0.1,
                     sr: int = 22050) -> np.ndarray:
    """
    Apply dynamic range compression to audio.
    
    Args:
        audio: Input audio signal
        threshold: Compression threshold (0.0 to 1.0)
        ratio: Compression ratio
        attack: Attack time in seconds
        release: Release time in seconds
        sr: Sample rate
        
    Returns:
        Compressed audio signal
    """
    # Convert times to samples
    attack_samples = int(attack * sr)
    release_samples = int(release * sr)
    
    # Initialize envelope
    envelope = np.zeros_like(audio)
    compressed = np.zeros_like(audio)
    
    # Simple compression algorithm
    for i in range(len(audio)):
        # Calculate envelope
        if i == 0:
            envelope[i] = abs(audio[i])
        else:
            if abs(audio[i]) > envelope[i-1]:
                # Attack
                alpha = 1.0 - np.exp(-1.0 / attack_samples)
            else:
                # Release
                alpha = 1.0 - np.exp(-1.0 / release_samples)
            
            envelope[i] = alpha * abs(audio[i]) + (1 - alpha) * envelope[i-1]
        
        # Apply compression
        if envelope[i] > threshold:
            # Above threshold, apply compression
            gain = threshold + (envelope[i] - threshold) / ratio
            gain_db = 20 * np.log10(gain / envelope[i])
            gain_linear = 10**(gain_db / 20)
        else:
            # Below threshold, no compression
            gain_linear = 1.0
        
        compressed[i] = audio[i] * gain_linear
    
    return compressed


def apply_reverb(audio: np.ndarray, 
                sr: int = 22050,
                room_size: float = 0.5,
                damping: float = 0.5,
                wet_level: float = 0.3) -> np.ndarray:
    """
    Apply reverb effect to audio.
    
    Args:
        audio: Input audio signal
        sr: Sample rate
        room_size: Room size (0.0 to 1.0)
        damping: Damping factor (0.0 to 1.0)
        wet_level: Wet signal level (0.0 to 1.0)
        
    Returns:
        Audio with reverb applied
    """
    # Simple reverb using multiple delays
    reverb_delays = [0.03, 0.05, 0.07, 0.11, 0.13, 0.17, 0.19, 0.23]
    reverb_gains = [0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.05]
    
    reverb_signal = np.zeros_like(audio)
    
    for delay, gain in zip(reverb_delays, reverb_gains):
        delay_samples = int(delay * sr)
        if delay_samples < len(audio):
            # Add delayed and attenuated signal
            delayed = np.pad(audio[:-delay_samples], (delay_samples, 0), 'constant')
            reverb_signal += delayed * gain * room_size
    
    # Apply damping (simple low-pass filter)
    from scipy import signal
    b, a = signal.butter(2, damping, btype='low')
    reverb_signal = signal.filtfilt(b, a, reverb_signal)
    
    # Mix dry and wet signals
    dry_level = 1.0 - wet_level
    output = audio * dry_level + reverb_signal * wet_level
    
    return output


def detect_clipping(audio: np.ndarray, 
                   threshold: float = 0.99) -> Tuple[bool, List[int]]:
    """
    Detect clipping in audio signal.
    
    Args:
        audio: Input audio signal
        threshold: Clipping threshold
        
    Returns:
        Tuple of (is_clipped, clipping_indices)
    """
    clipping_indices = np.where(np.abs(audio) >= threshold)[0].tolist()
    is_clipped = len(clipping_indices) > 0
    
    return is_clipped, clipping_indices


def fix_clipping(audio: np.ndarray, 
                threshold: float = 0.99) -> np.ndarray:
    """
    Fix clipping in audio signal by soft limiting.
    
    Args:
        audio: Input audio signal
        threshold: Clipping threshold
        
    Returns:
        Audio with clipping fixed
    """
    # Apply soft limiting
    limited_audio = np.tanh(audio / threshold) * threshold
    
    return limited_audio


def calculate_rms(audio: np.ndarray) -> float:
    """
    Calculate RMS (Root Mean Square) of audio signal.
    
    Args:
        audio: Input audio signal
        
    Returns:
        RMS value
    """
    return np.sqrt(np.mean(audio**2))


def calculate_peak(audio: np.ndarray) -> float:
    """
    Calculate peak value of audio signal.
    
    Args:
        audio: Input audio signal
        
    Returns:
        Peak value
    """
    return np.max(np.abs(audio))


def calculate_dynamic_range(audio: np.ndarray) -> float:
    """
    Calculate dynamic range of audio signal.
    
    Args:
        audio: Input audio signal
        
    Returns:
        Dynamic range in dB
    """
    rms = calculate_rms(audio)
    peak = calculate_peak(audio)
    
    if rms > 0:
        return 20 * np.log10(peak / rms)
    else:
        return 0.0


def analyze_audio_quality(audio: np.ndarray, 
                         sr: int = 22050) -> dict:
    """
    Analyze audio quality metrics.
    
    Args:
        audio: Input audio signal
        sr: Sample rate
        
    Returns:
        Dictionary of quality metrics
    """
    # Basic metrics
    rms = calculate_rms(audio)
    peak = calculate_peak(audio)
    dynamic_range = calculate_dynamic_range(audio)
    
    # Clipping detection
    is_clipped, clipping_indices = detect_clipping(audio)
    clipping_percentage = len(clipping_indices) / len(audio) * 100
    
    # Spectral analysis
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))
    
    return {
        'rms': rms,
        'peak': peak,
        'dynamic_range_db': dynamic_range,
        'is_clipped': is_clipped,
        'clipping_percentage': clipping_percentage,
        'spectral_centroid': spectral_centroid,
        'spectral_rolloff': spectral_rolloff,
        'zero_crossing_rate': zero_crossing_rate,
        'duration_seconds': len(audio) / sr,
        'sample_rate': sr
    }
