#!/usr/bin/env python3
"""
Script to analyze whistle audio and extract features.
"""

import argparse
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.inference import WhistleToMusicConverter
from src.utils import setup_logging, save_json


def main():
    parser = argparse.ArgumentParser(description="Analyze whistle audio")
    parser.add_argument("--whistle_path", type=str, required=True,
                       help="Path to whistle audio file")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Path to save analysis results (JSON)")
    parser.add_argument("--model_path", type=str, default="outputs/best_model.pt",
                       help="Path to trained model (for feature extraction)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda, mps)")
    parser.add_argument("--log_level", type=str, default="INFO",
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(log_level=args.log_level)
    
    logger.info(f"Analyzing whistle: {args.whistle_path}")
    
    try:
        # Create converter (needed for feature extraction)
        converter = WhistleToMusicConverter(
            model_path=args.model_path,
            device=args.device
        )
        
        # Analyze whistle
        logger.info("Extracting features...")
        analysis = converter.analyze_whistle(args.whistle_path)
        
        # Print analysis results
        logger.info("Whistle Analysis Results:")
        logger.info("=" * 50)
        
        # Audio info
        audio_info = analysis['audio_info']
        logger.info("Audio Information:")
        logger.info(f"  Duration: {audio_info['duration']:.2f} seconds")
        logger.info(f"  Sample Rate: {audio_info['sample_rate']} Hz")
        logger.info(f"  RMS Energy: {audio_info['rms_energy']:.4f}")
        
        # Pitch analysis
        pitch_analysis = analysis['pitch_analysis']
        logger.info("\nPitch Analysis:")
        logger.info(f"  Mean Pitch: {pitch_analysis['mean_pitch']:.1f} Hz")
        logger.info(f"  Pitch Std: {pitch_analysis['pitch_std']:.1f} Hz")
        logger.info(f"  Pitch Range: {pitch_analysis['pitch_range']:.1f} Hz")
        logger.info(f"  Voiced Ratio: {pitch_analysis['voiced_ratio']:.2f}")
        
        # Rhythm analysis
        rhythm_analysis = analysis['rhythm_analysis']
        logger.info("\nRhythm Analysis:")
        logger.info(f"  Tempo: {rhythm_analysis['tempo']:.1f} BPM")
        logger.info(f"  Rhythm Regularity: {rhythm_analysis['rhythm_regularity']:.2f}")
        logger.info(f"  Onset Count: {rhythm_analysis['onset_count']}")
        
        # Dynamics analysis
        dynamics_analysis = analysis['dynamics_analysis']
        logger.info("\nDynamics Analysis:")
        logger.info(f"  Energy Mean: {dynamics_analysis['energy_mean']:.4f}")
        logger.info(f"  Dynamic Range: {dynamics_analysis['dynamic_range']:.1f} dB")
        
        # Timbre analysis
        timbre_analysis = analysis['timbre_analysis']
        logger.info("\nTimbre Analysis:")
        logger.info(f"  Harmonic Ratio: {timbre_analysis['harmonic_ratio']:.2f}")
        logger.info(f"  Spectral Centroid: {timbre_analysis['spectral_centroid']:.1f} Hz")
        
        # Conditioning vector
        conditioning_vector = analysis['conditioning_vector']
        logger.info(f"\nConditioning Vector ({len(conditioning_vector)} dimensions):")
        logger.info(f"  {conditioning_vector}")
        
        # Save analysis if output path specified
        if args.output_path:
            save_json(analysis, args.output_path)
            logger.info(f"\nAnalysis saved to: {args.output_path}")
        
        # Generate music sample
        logger.info("\nGenerating music sample...")
        result = converter.convert_whistle_to_music(
            whistle_audio=args.whistle_path,
            max_length=256,
            temperature=0.8
        )
        
        if 'output_path' in result:
            logger.info(f"Sample music generated: {result['output_path']}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
