#!/usr/bin/env python3
"""
Inference script for the whistle-to-music synthesizer.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.inference import WhistleToMusicConverter, MusicGenerator
from src.utils import load_config, setup_logging, ensure_dir
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Convert whistle to classical music")
    parser.add_argument("--whistle_path", type=str, required=True,
                       help="Path to whistle audio file")
    parser.add_argument("--model_path", type=str, default="outputs/best_model.pt",
                       help="Path to trained model")
    parser.add_argument("--output_path", type=str, default="data/generated/output.mid",
                       help="Path to save generated MIDI file")
    parser.add_argument("--config", type=str, default="configs/inference_config.yaml",
                       help="Path to inference configuration file")
    parser.add_argument("--style", type=str, default="classical",
                       choices=["classical", "romantic", "baroque", "impressionist"],
                       help="Music style")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum generation length")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda, mps)")
    parser.add_argument("--log_level", type=str, default="INFO",
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(log_level=args.log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model_path:
        config['model']['model_path'] = args.model_path
    if args.device:
        config['model']['device'] = args.device
    if args.temperature:
        config['generation']['temperature'] = args.temperature
    if args.max_length:
        config['generation']['max_length'] = args.max_length
    
    logger.info("Starting inference with configuration:")
    logger.info(f"  Whistle: {args.whistle_path}")
    logger.info(f"  Model: {config['model']['model_path']}")
    logger.info(f"  Output: {args.output_path}")
    logger.info(f"  Style: {args.style}")
    logger.info(f"  Temperature: {config['generation']['temperature']}")
    logger.info(f"  Max Length: {config['generation']['max_length']}")
    
    # Ensure output directory exists
    output_path = Path(args.output_path)
    ensure_dir(output_path.parent)
    
    try:
        # Create converter
        logger.info("Loading model...")
        converter = WhistleToMusicConverter(
            model_path=config['model']['model_path'],
            device=config['model']['device']
        )
        
        # Convert whistle to music
        logger.info("Converting whistle to music...")
        result = converter.convert_whistle_to_music(
            whistle_audio=args.whistle_path,
            output_path=args.output_path,
            style=args.style,
            temperature=config['generation']['temperature'],
            max_length=config['generation']['max_length'],
            top_k=config['generation']['top_k'],
            top_p=config['generation']['top_p']
        )
        
        logger.info("Conversion completed!")
        logger.info(f"Generated MIDI saved to: {args.output_path}")
        
        # Log generation info
        if 'metadata' in result:
            metadata = result['metadata']
            logger.info("Generation metadata:")
            logger.info(f"  Model parameters: {metadata.get('model_info', {}).get('model_parameters', 'N/A')}")
            logger.info(f"  Device: {metadata.get('model_info', {}).get('device', 'N/A')}")
            
            # Log whistle analysis
            whistle_features = metadata.get('whistle_features', {})
            if 'pitch' in whistle_features:
                pitch_info = whistle_features['pitch']
                logger.info("Whistle analysis:")
                logger.info(f"  Mean pitch: {pitch_info.get('pitch_mean', 0):.1f} Hz")
                logger.info(f"  Pitch range: {pitch_info.get('pitch_range', 0):.1f} Hz")
                logger.info(f"  Voiced ratio: {pitch_info.get('voiced_ratio', 0):.2f}")
            
            if 'rhythm' in whistle_features:
                rhythm_info = whistle_features['rhythm']
                logger.info(f"  Tempo: {rhythm_info.get('tempo', 0):.1f} BPM")
                logger.info(f"  Rhythm regularity: {rhythm_info.get('rhythm_regularity', 0):.2f}")
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
