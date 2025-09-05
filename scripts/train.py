#!/usr/bin/env python3
"""
Training script for the whistle-to-music synthesizer.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models import MusicTransformer, TransformerConfig
from src.training import MusicTrainer, TrainingConfig
from src.data_processing import create_maestro_dataloaders
from src.utils import load_config, setup_logging, log_system_info
import torch


def main():
    parser = argparse.ArgumentParser(description="Train whistle-to-music synthesizer")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                       help="Path to training configuration file")
    parser.add_argument("--data_dir", type=str, default="data/maestro",
                       help="Path to MAESTRO dataset directory")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Output directory for checkpoints and logs")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda, mps)")
    parser.add_argument("--log_level", type=str, default="INFO",
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(log_level=args.log_level)
    log_system_info(logger)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data_dir:
        config['data']['maestro_dir'] = args.data_dir
    if args.output_dir:
        config['output']['output_dir'] = args.output_dir
    if args.device:
        config['device'] = args.device
    
    logger.info("Starting training with configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Create model configuration
    model_config = TransformerConfig(**config['model'])
    
    # Create training configuration
    training_config = TrainingConfig(**config['training'])
    
    # Create data loaders
    logger.info("Loading MAESTRO dataset...")
    dataloaders = create_maestro_dataloaders(
        data_dir=config['data']['maestro_dir'],
        batch_size=training_config.batch_size,
        max_files_per_split=config['data']['max_files_per_split'],
        cache_dir=config['data']['cache_dir'],
        num_workers=config['data']['num_workers']
    )
    
    logger.info(f"Dataset loaded:")
    for split, dataloader in dataloaders.items():
        logger.info(f"  {split}: {len(dataloader)} batches")
    
    # Create model
    logger.info("Creating model...")
    model = MusicTransformer(model_config)
    logger.info(f"Model created with {model.get_model_size():,} parameters")
    
    # Create trainer
    trainer = MusicTrainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['validation'],
        config=training_config,
        output_dir=config['output']['output_dir']
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
