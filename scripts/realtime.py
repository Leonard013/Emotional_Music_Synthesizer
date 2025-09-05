#!/usr/bin/env python3
"""
Real-time whistle-to-music conversion script.
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.inference import RealTimeProcessor, StreamingMusicGenerator
from src.utils import load_config, setup_logging
import sounddevice as sd


def main():
    parser = argparse.ArgumentParser(description="Real-time whistle-to-music conversion")
    parser.add_argument("--model_path", type=str, default="outputs/best_model.pt",
                       help="Path to trained model")
    parser.add_argument("--config", type=str, default="configs/inference_config.yaml",
                       help="Path to inference configuration file")
    parser.add_argument("--input_device", type=int, default=None,
                       help="Audio input device ID")
    parser.add_argument("--output_device", type=int, default=None,
                       help="Audio output device ID")
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
    if args.input_device is not None:
        config['realtime']['input_device'] = args.input_device
    if args.output_device is not None:
        config['realtime']['output_device'] = args.output_device
    
    logger.info("Starting real-time processing...")
    logger.info(f"Model: {config['model']['model_path']}")
    logger.info(f"Input device: {config['realtime']['input_device']}")
    logger.info(f"Output device: {config['realtime']['output_device']}")
    
    # List audio devices
    logger.info("Available audio devices:")
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            logger.info(f"  {i}: {device['name']} ({device['max_input_channels']} in, {device['max_output_channels']} out)")
    except Exception as e:
        logger.warning(f"Could not list audio devices: {e}")
    
    try:
        # Create real-time processor
        processor = RealTimeProcessor(
            model_path=config['model']['model_path'],
            input_device=config['realtime']['input_device'],
            output_device=config['realtime']['output_device'],
            device=config['model']['device']
        )
        
        # Create music generator
        generator = StreamingMusicGenerator(
            output_device=config['realtime']['output_device']
        )
        
        # Set callbacks
        def on_music_generated(result):
            logger.info("Music generated! Playing...")
            generator.play_midi(result)
        
        def on_error(error):
            logger.error(f"Processing error: {error}")
        
        processor.set_callbacks(
            on_music_generated=on_music_generated,
            on_error=on_error
        )
        
        # Set processing parameters
        processor.set_processing_parameters(
            temperature=config['generation']['temperature'],
            max_length=config['generation']['max_length'],
            min_whistle_duration=config['audio']['min_whistle_duration'],
            silence_threshold=config['audio']['silence_threshold']
        )
        
        # Start processing
        logger.info("Starting real-time processing...")
        logger.info("Whistle into your microphone to generate music!")
        logger.info("Press Ctrl+C to stop")
        
        processor.start_processing()
        generator.start_playback()
        
        # Keep running until interrupted
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Stopping real-time processing...")
        
        # Stop processing
        processor.stop_processing()
        generator.stop_playback()
        generator.cleanup()
        
        logger.info("Real-time processing stopped")
        
    except Exception as e:
        logger.error(f"Error during real-time processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
