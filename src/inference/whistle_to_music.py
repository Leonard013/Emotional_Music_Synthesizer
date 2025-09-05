"""
Main inference pipeline for converting whistles to classical music.
"""

import torch
import numpy as np
import librosa
import soundfile as sf
import pretty_midi
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import time

from ..models import MusicTransformer, TransformerConfig
from ..whistle_analysis import AudioProcessor, FeatureExtractor, PitchDetector
from ..data_processing import MidiProcessor, DataPreprocessor


class WhistleToMusicConverter:
    """Main converter for whistle-to-music transformation."""
    
    def __init__(self, 
                 model_path: str,
                 device: str = "auto",
                 sample_rate: int = 22050):
        """
        Initialize whistle-to-music converter.
        
        Args:
            model_path: Path to trained model
            device: Device to use for inference
            sample_rate: Audio sample rate
        """
        self.device = self._setup_device(device)
        self.sample_rate = sample_rate
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Initialize processors
        self.audio_processor = AudioProcessor(sample_rate=sample_rate)
        self.feature_extractor = FeatureExtractor(sample_rate=sample_rate)
        self.pitch_detector = PitchDetector(sample_rate=sample_rate)
        self.midi_processor = MidiProcessor()
        self.data_preprocessor = DataPreprocessor(
            midi_processor=self.midi_processor,
            feature_extractor=self.feature_extractor,
            pitch_detector=self.pitch_detector,
            audio_processor=self.audio_processor
        )
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup inference device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _load_model(self, model_path: str) -> MusicTransformer:
        """Load trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            # Load from training checkpoint
            config_dict = checkpoint.get('model_config', checkpoint.get('config', {}))
            config = TransformerConfig(**config_dict)
            model = MusicTransformer(config)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Load from model checkpoint
            model = MusicTransformer.load_model(model_path, self.device)
        
        model.to(self.device)
        return model
    
    def convert_whistle_to_music(self, 
                                whistle_audio: Union[str, np.ndarray],
                                output_path: Optional[str] = None,
                                max_length: int = 512,
                                temperature: float = 0.8,
                                top_k: int = 50,
                                top_p: float = 0.9,
                                style: str = "classical") -> Dict:
        """
        Convert whistle audio to classical music.
        
        Args:
            whistle_audio: Path to whistle audio file or audio array
            output_path: Path to save generated MIDI file
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            style: Music style ("classical", "romantic", "baroque")
            
        Returns:
            Dictionary containing generated music and metadata
        """
        # Load and preprocess whistle audio
        if isinstance(whistle_audio, str):
            audio = self.audio_processor.load_audio(whistle_audio)
        else:
            audio = whistle_audio
        
        # Extract features
        features = self.data_preprocessor.extract_whistle_features(audio)
        conditioning = self.data_preprocessor.features_to_conditioning_vector(features)
        
        # Apply style conditioning
        conditioning = self._apply_style_conditioning(conditioning, style)
        
        # Generate music
        generated_tokens = self._generate_music(
            conditioning=conditioning,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        # Convert tokens to MIDI
        result = self.data_preprocessor.postprocess_generated_sequence(
            generated_tokens, features
        )
        
        # Save MIDI file if output path provided
        if output_path:
            result['midi'].write(output_path)
            result['output_path'] = output_path
        
        # Add metadata
        result['metadata'] = {
            'whistle_features': features,
            'generation_params': {
                'max_length': max_length,
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p,
                'style': style
            },
            'model_info': {
                'device': str(self.device),
                'vocab_size': self.model.config.vocab_size
            }
        }
        
        return result
    
    def _apply_style_conditioning(self, 
                                 conditioning: torch.Tensor, 
                                 style: str) -> torch.Tensor:
        """Apply style-specific conditioning."""
        style_vectors = {
            'classical': torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'romantic': torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'baroque': torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'impressionist': torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        }
        
        if style in style_vectors:
            style_vector = style_vectors[style]
            # Concatenate style vector to conditioning
            conditioning = torch.cat([conditioning, style_vector])
        
        return conditioning
    
    def _generate_music(self, 
                       conditioning: torch.Tensor,
                       max_length: int = 512,
                       temperature: float = 0.8,
                       top_k: int = 50,
                       top_p: float = 0.9) -> List[int]:
        """Generate music tokens from conditioning."""
        # Create initial sequence with start token
        initial_tokens = [self.midi_processor.START_TOKEN]
        input_ids = torch.tensor(initial_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        conditioning = conditioning.unsqueeze(0).to(self.device)
        
        # Generate sequence
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                conditioning=conditioning,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True
            )
        
        # Convert to list and remove batch dimension
        generated_tokens = generated.squeeze(0).cpu().tolist()
        
        return generated_tokens
    
    def batch_convert(self, 
                     whistle_files: List[str],
                     output_dir: str,
                     **kwargs) -> List[Dict]:
        """
        Convert multiple whistle files to music.
        
        Args:
            whistle_files: List of whistle audio file paths
            output_dir: Directory to save generated MIDI files
            **kwargs: Additional arguments for convert_whistle_to_music
            
        Returns:
            List of conversion results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for i, whistle_file in enumerate(whistle_files):
            print(f"Processing {i+1}/{len(whistle_files)}: {whistle_file}")
            
            # Generate output filename
            input_name = Path(whistle_file).stem
            output_path = output_dir / f"{input_name}_generated.mid"
            
            try:
                result = self.convert_whistle_to_music(
                    whistle_audio=whistle_file,
                    output_path=str(output_path),
                    **kwargs
                )
                results.append(result)
                print(f"Generated: {output_path}")
                
            except Exception as e:
                print(f"Error processing {whistle_file}: {e}")
                results.append({
                    'error': str(e),
                    'input_file': whistle_file
                })
        
        return results
    
    def convert_with_variations(self, 
                               whistle_audio: Union[str, np.ndarray],
                               num_variations: int = 3,
                               output_dir: str = "./variations",
                               **kwargs) -> List[Dict]:
        """
        Generate multiple variations of the same whistle.
        
        Args:
            whistle_audio: Whistle audio file or array
            num_variations: Number of variations to generate
            output_dir: Directory to save variations
            **kwargs: Additional arguments for conversion
            
        Returns:
            List of variation results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        variations = []
        
        for i in range(num_variations):
            print(f"Generating variation {i+1}/{num_variations}")
            
            # Add variation to output path
            if isinstance(whistle_audio, str):
                input_name = Path(whistle_audio).stem
            else:
                input_name = "whistle"
            
            output_path = output_dir / f"{input_name}_variation_{i+1}.mid"
            
            # Vary generation parameters
            variation_kwargs = kwargs.copy()
            variation_kwargs['temperature'] = kwargs.get('temperature', 0.8) + np.random.uniform(-0.2, 0.2)
            variation_kwargs['top_k'] = kwargs.get('top_k', 50) + np.random.randint(-10, 10)
            
            try:
                result = self.convert_whistle_to_music(
                    whistle_audio=whistle_audio,
                    output_path=str(output_path),
                    **variation_kwargs
                )
                variations.append(result)
                
            except Exception as e:
                print(f"Error generating variation {i+1}: {e}")
                variations.append({
                    'error': str(e),
                    'variation': i+1
                })
        
        return variations
    
    def analyze_whistle(self, 
                       whistle_audio: Union[str, np.ndarray]) -> Dict:
        """
        Analyze whistle audio and return detailed features.
        
        Args:
            whistle_audio: Whistle audio file or array
            
        Returns:
            Dictionary of analysis results
        """
        # Load audio
        if isinstance(whistle_audio, str):
            audio = self.audio_processor.load_audio(whistle_audio)
        else:
            audio = whistle_audio
        
        # Extract comprehensive features
        features = self.data_preprocessor.extract_whistle_features(audio)
        
        # Create analysis report
        analysis = {
            'audio_info': {
                'duration': len(audio) / self.sample_rate,
                'sample_rate': self.sample_rate,
                'rms_energy': np.sqrt(np.mean(audio**2))
            },
            'pitch_analysis': {
                'mean_pitch': features.get('pitch', {}).get('pitch_mean', 0),
                'pitch_std': features.get('pitch', {}).get('pitch_std', 0),
                'pitch_range': features.get('pitch', {}).get('pitch_range', 0),
                'voiced_ratio': features.get('pitch', {}).get('voiced_ratio', 0)
            },
            'rhythm_analysis': {
                'tempo': features.get('rhythm', {}).get('tempo', 120),
                'rhythm_regularity': features.get('rhythm', {}).get('rhythm_regularity', 0),
                'onset_count': features.get('rhythm', {}).get('onset_count', 0)
            },
            'dynamics_analysis': {
                'energy_mean': features.get('dynamics', {}).get('energy_mean', 0),
                'dynamic_range': features.get('dynamics', {}).get('dynamic_range', 0)
            },
            'timbre_analysis': {
                'harmonic_ratio': features.get('timbre', {}).get('harmonic_ratio', 0),
                'spectral_centroid': features.get('spectral', {}).get('spectral_centroid_mean', 0)
            },
            'conditioning_vector': self.data_preprocessor.features_to_conditioning_vector(features).tolist()
        }
        
        return analysis
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            'model_parameters': self.model.get_model_size(),
            'trainable_parameters': self.model.get_trainable_parameters(),
            'config': self.model.config.__dict__,
            'device': str(self.device),
            'vocab_size': self.model.config.vocab_size
        }
