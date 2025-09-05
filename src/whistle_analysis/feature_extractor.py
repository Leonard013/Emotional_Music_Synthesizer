"""
Feature extraction for whistle audio analysis.
Extracts musical features including rhythm, dynamics, and spectral characteristics.
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from scipy.stats import skew, kurtosis
import torch


class FeatureExtractor:
    """Extracts comprehensive musical features from whistle audio."""
    
    def __init__(self, sample_rate: int = 22050, hop_length: int = 512):
        """
        Initialize feature extractor.
        
        Args:
            sample_rate: Audio sample rate
            hop_length: Hop length for analysis
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frame_length = hop_length * 4
        
    def extract_rhythm_features(self, audio: np.ndarray) -> Dict:
        """
        Extract rhythm-related features.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary of rhythm features
        """
        # Onset detection
        onset_frames = librosa.onset.onset_detect(
            y=audio, 
            sr=self.sample_rate, 
            hop_length=self.hop_length,
            units='frames'
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sample_rate, hop_length=self.hop_length)
        
        # Tempo estimation
        tempo, beats = librosa.beat.beat_track(
            y=audio, 
            sr=self.sample_rate, 
            hop_length=self.hop_length
        )
        
        # Calculate inter-onset intervals
        if len(onset_times) > 1:
            iois = np.diff(onset_times)
            ioi_mean = np.mean(iois)
            ioi_std = np.std(iois)
            ioi_cv = ioi_std / ioi_mean if ioi_mean > 0 else 0
        else:
            ioi_mean = ioi_std = ioi_cv = 0
        
        # Rhythm regularity
        rhythm_regularity = 1.0 / (1.0 + ioi_cv) if ioi_cv > 0 else 0
        
        return {
            'tempo': tempo,
            'onset_count': len(onset_frames),
            'onset_times': onset_times,
            'ioi_mean': ioi_mean,
            'ioi_std': ioi_std,
            'ioi_cv': ioi_cv,
            'rhythm_regularity': rhythm_regularity,
            'beat_frames': beats
        }
    
    def extract_spectral_features(self, audio: np.ndarray) -> Dict:
        """
        Extract spectral characteristics.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary of spectral features
        """
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            audio, hop_length=self.hop_length
        )[0]
        
        # MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate, n_mfcc=13, hop_length=self.hop_length
        )
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        return {
            'spectral_centroid_mean': np.mean(spectral_centroids),
            'spectral_centroid_std': np.std(spectral_centroids),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'spectral_rolloff_std': np.std(spectral_rolloff),
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'spectral_bandwidth_std': np.std(spectral_bandwidth),
            'zcr_mean': np.mean(zcr),
            'zcr_std': np.std(zcr),
            'mfccs': mfccs,
            'spectral_contrast': spectral_contrast
        }
    
    def extract_dynamics_features(self, audio: np.ndarray) -> Dict:
        """
        Extract dynamics and energy features.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary of dynamics features
        """
        # RMS energy
        rms = librosa.feature.rms(
            y=audio, hop_length=self.hop_length
        )[0]
        
        # Spectral energy
        stft = librosa.stft(audio, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        spectral_energy = np.sum(magnitude, axis=0)
        
        # Energy statistics
        energy_mean = np.mean(rms)
        energy_std = np.std(rms)
        energy_max = np.max(rms)
        energy_min = np.min(rms)
        energy_range = energy_max - energy_min
        
        # Dynamic range
        dynamic_range = 20 * np.log10(energy_max / energy_min) if energy_min > 0 else 0
        
        # Energy envelope
        energy_envelope = rms
        
        return {
            'energy_mean': energy_mean,
            'energy_std': energy_std,
            'energy_max': energy_max,
            'energy_min': energy_min,
            'energy_range': energy_range,
            'dynamic_range': dynamic_range,
            'energy_envelope': energy_envelope,
            'spectral_energy': spectral_energy
        }
    
    def extract_timbre_features(self, audio: np.ndarray) -> Dict:
        """
        Extract timbral characteristics.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary of timbre features
        """
        # Chroma features
        chroma = librosa.feature.chroma_stft(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # Tonnetz features
        tonnetz = librosa.feature.tonnetz(
            y=audio, sr=self.sample_rate
        )
        
        # Spectral flatness
        spectral_flatness = librosa.feature.spectral_flatness(
            y=audio, hop_length=self.hop_length
        )[0]
        
        # Harmonic-percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(audio)
        
        # Harmonic ratio
        harmonic_energy = np.sum(y_harmonic**2)
        percussive_energy = np.sum(y_percussive**2)
        total_energy = harmonic_energy + percussive_energy
        harmonic_ratio = harmonic_energy / total_energy if total_energy > 0 else 0
        
        return {
            'chroma_mean': np.mean(chroma, axis=1),
            'chroma_std': np.std(chroma, axis=1),
            'tonnetz_mean': np.mean(tonnetz, axis=1),
            'tonnetz_std': np.std(tonnetz, axis=1),
            'spectral_flatness_mean': np.mean(spectral_flatness),
            'spectral_flatness_std': np.std(spectral_flatness),
            'harmonic_ratio': harmonic_ratio,
            'chroma': chroma,
            'tonnetz': tonnetz
        }
    
    def extract_statistical_features(self, features: np.ndarray) -> Dict:
        """
        Extract statistical features from any feature array.
        
        Args:
            features: Input feature array
            
        Returns:
            Dictionary of statistical features
        """
        if len(features) == 0:
            return {}
        
        # Remove NaN values
        clean_features = features[~np.isnan(features)]
        
        if len(clean_features) == 0:
            return {}
        
        return {
            'mean': np.mean(clean_features),
            'std': np.std(clean_features),
            'min': np.min(clean_features),
            'max': np.max(clean_features),
            'median': np.median(clean_features),
            'skewness': skew(clean_features),
            'kurtosis': kurtosis(clean_features),
            'range': np.ptp(clean_features),
            'q25': np.percentile(clean_features, 25),
            'q75': np.percentile(clean_features, 75)
        }
    
    def extract_all_features(self, audio: np.ndarray, pitch_features: Dict = None) -> Dict:
        """
        Extract all musical features from whistle audio.
        
        Args:
            audio: Input audio signal
            pitch_features: Optional pre-computed pitch features
            
        Returns:
            Comprehensive feature dictionary
        """
        features = {}
        
        # Rhythm features
        features['rhythm'] = self.extract_rhythm_features(audio)
        
        # Spectral features
        features['spectral'] = self.extract_spectral_features(audio)
        
        # Dynamics features
        features['dynamics'] = self.extract_dynamics_features(audio)
        
        # Timbre features
        features['timbre'] = self.extract_timbre_features(audio)
        
        # Add pitch features if provided
        if pitch_features is not None:
            features['pitch'] = pitch_features
        
        # Extract statistical features for key metrics
        if 'pitch' in features and 'f0_smooth' in features['pitch']:
            f0_stats = self.extract_statistical_features(features['pitch']['f0_smooth'])
            features['pitch_stats'] = f0_stats
        
        # Overall feature vector for ML models
        feature_vector = self._create_feature_vector(features)
        features['feature_vector'] = feature_vector
        
        return features
    
    def _create_feature_vector(self, features: Dict) -> np.ndarray:
        """
        Create a flat feature vector from all extracted features.
        
        Args:
            features: Dictionary of all features
            
        Returns:
            Flattened feature vector
        """
        vector_parts = []
        
        # Add scalar features
        scalar_features = [
            'tempo', 'onset_count', 'ioi_mean', 'ioi_std', 'ioi_cv', 'rhythm_regularity',
            'spectral_centroid_mean', 'spectral_centroid_std', 'spectral_rolloff_mean',
            'spectral_rolloff_std', 'spectral_bandwidth_mean', 'spectral_bandwidth_std',
            'zcr_mean', 'zcr_std', 'energy_mean', 'energy_std', 'energy_max',
            'energy_min', 'energy_range', 'dynamic_range', 'spectral_flatness_mean',
            'spectral_flatness_std', 'harmonic_ratio'
        ]
        
        for feature_name in scalar_features:
            if 'rhythm' in features and feature_name in features['rhythm']:
                vector_parts.append(features['rhythm'][feature_name])
            elif 'spectral' in features and feature_name in features['spectral']:
                vector_parts.append(features['spectral'][feature_name])
            elif 'dynamics' in features and feature_name in features['dynamics']:
                vector_parts.append(features['dynamics'][feature_name])
            elif 'timbre' in features and feature_name in features['timbre']:
                vector_parts.append(features['timbre'][feature_name])
            else:
                vector_parts.append(0.0)  # Default value
        
        # Add pitch statistics
        if 'pitch_stats' in features:
            pitch_stats = features['pitch_stats']
            for stat_name in ['mean', 'std', 'min', 'max', 'median', 'skewness', 'kurtosis']:
                vector_parts.append(pitch_stats.get(stat_name, 0.0))
        
        # Add chroma features (12 values)
        if 'timbre' in features and 'chroma_mean' in features['timbre']:
            vector_parts.extend(features['timbre']['chroma_mean'])
        
        return np.array(vector_parts)
