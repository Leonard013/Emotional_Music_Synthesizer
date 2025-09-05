#!/usr/bin/env python3
"""
Setup script for the whistle-to-music synthesizer.
"""

import argparse
import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils import setup_logging, ensure_dir, list_audio_files


def main():
    parser = argparse.ArgumentParser(description="Setup whistle-to-music synthesizer")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Data directory")
    parser.add_argument("--maestro_url", type=str, 
                       default="https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip",
                       help="MAESTRO dataset download URL")
    parser.add_argument("--skip_download", action="store_true",
                       help="Skip MAESTRO dataset download")
    parser.add_argument("--log_level", type=str, default="INFO",
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(log_level=args.log_level)
    
    logger.info("Setting up whistle-to-music synthesizer...")
    
    # Create directory structure
    logger.info("Creating directory structure...")
    data_dir = Path(args.data_dir)
    directories = [
        data_dir,
        data_dir / "maestro",
        data_dir / "whistles",
        data_dir / "generated",
        data_dir / "cache",
        "outputs",
        "logs"
    ]
    
    for directory in directories:
        ensure_dir(directory)
        logger.info(f"  Created: {directory}")
    
    # Download MAESTRO dataset if not skipped
    if not args.skip_download:
        maestro_dir = data_dir / "maestro"
        maestro_zip = maestro_dir / "maestro-v3.0.0.zip"
        
        if not maestro_zip.exists():
            logger.info("Downloading MAESTRO dataset...")
            logger.info(f"URL: {args.maestro_url}")
            logger.info("This may take a while (101GB download)...")
            
            try:
                subprocess.run([
                    "wget", "-O", str(maestro_zip), args.maestro_url
                ], check=True)
                logger.info("Download completed!")
            except subprocess.CalledProcessError as e:
                logger.error(f"Download failed: {e}")
                logger.info("You can download manually and extract to data/maestro/")
        else:
            logger.info("MAESTRO dataset already downloaded")
        
        # Extract if zip exists
        if maestro_zip.exists():
            logger.info("Extracting MAESTRO dataset...")
            try:
                subprocess.run([
                    "unzip", "-o", str(maestro_zip), "-d", str(maestro_dir)
                ], check=True)
                logger.info("Extraction completed!")
            except subprocess.CalledProcessError as e:
                logger.error(f"Extraction failed: {e}")
                logger.info("You can extract manually to data/maestro/")
    
    # Check for whistle files
    whistle_dir = data_dir / "whistles"
    whistle_files = list_audio_files(whistle_dir)
    
    if whistle_files:
        logger.info(f"Found {len(whistle_files)} whistle files in {whistle_dir}")
    else:
        logger.info(f"No whistle files found in {whistle_dir}")
        logger.info("Add your whistle audio files to this directory")
    
    # Create example configuration files
    logger.info("Creating example configuration files...")
    
    # Training config
    training_config = {
        'model': {
            'vocab_size': 2000,
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 6,
            'd_ff': 2048,
            'max_seq_len': 1024,
            'conditioning_dim': 16,
            'dropout': 0.1,
            'use_conditioning': True,
            'use_musical_embedding': True
        },
        'training': {
            'learning_rate': 1e-4,
            'batch_size': 8,
            'num_epochs': 100,
            'warmup_steps': 1000,
            'max_grad_norm': 1.0,
            'weight_decay': 0.01,
            'gradient_accumulation_steps': 1,
            'mixed_precision': True,
            'early_stopping_patience': 10
        },
        'data': {
            'maestro_dir': str(data_dir / "maestro"),
            'cache_dir': str(data_dir / "cache"),
            'max_files_per_split': {
                'train': None,
                'validation': 50,
                'test': 50
            },
            'num_workers': 4,
            'pin_memory': True
        },
        'logging': {
            'use_wandb': False,
            'wandb_project': 'whistle-music-synthesizer',
            'log_every': 100,
            'save_every': 10,
            'eval_every': 5
        },
        'output': {
            'output_dir': './outputs',
            'save_best_model': True,
            'save_checkpoints': True
        },
        'device': 'auto'
    }
    
    # Save training config
    import yaml
    with open("configs/training_config.yaml", 'w') as f:
        yaml.dump(training_config, f, default_flow_style=False, indent=2)
    
    logger.info("Setup completed!")
    logger.info("\nNext steps:")
    logger.info("1. Add your whistle audio files to data/whistles/")
    logger.info("2. Train the model: python scripts/train.py")
    logger.info("3. Convert whistles: python scripts/inference.py --whistle_path data/whistles/your_whistle.wav")
    logger.info("4. Real-time conversion: python scripts/realtime.py")


if __name__ == "__main__":
    main()
