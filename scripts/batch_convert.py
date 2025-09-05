#!/usr/bin/env python3
"""
Batch conversion script for multiple whistle files.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.inference import WhistleToMusicConverter, MusicGenerator
from src.utils import setup_logging, list_audio_files, ensure_dir
import json


def main():
    parser = argparse.ArgumentParser(description="Batch convert whistles to music")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing whistle audio files")
    parser.add_argument("--output_dir", type=str, default="data/generated/batch",
                       help="Directory to save generated MIDI files")
    parser.add_argument("--model_path", type=str, default="outputs/best_model.pt",
                       help="Path to trained model")
    parser.add_argument("--config", type=str, default="configs/inference_config.yaml",
                       help="Path to inference configuration file")
    parser.add_argument("--style", type=str, default="classical",
                       choices=["classical", "romantic", "baroque", "impressionist"],
                       help="Music style")
    parser.add_argument("--num_variations", type=int, default=1,
                       help="Number of variations per whistle")
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
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    
    # Find whistle files
    input_dir = Path(args.input_dir)
    whistle_files = list_audio_files(input_dir)
    
    if not whistle_files:
        logger.error(f"No audio files found in {input_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(whistle_files)} whistle files")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Style: {args.style}")
    logger.info(f"Variations per file: {args.num_variations}")
    
    try:
        # Create converter
        logger.info("Loading model...")
        converter = WhistleToMusicConverter(
            model_path=config['model']['model_path'],
            device=config['model']['device']
        )
        
        # Create music generator for variations
        generator = MusicGenerator(
            model_path=config['model']['model_path'],
            device=config['model']['device']
        )
        
        # Process each whistle file
        results = []
        for i, whistle_file in enumerate(whistle_files):
            logger.info(f"Processing {i+1}/{len(whistle_files)}: {whistle_file.name}")
            
            try:
                if args.num_variations > 1:
                    # Generate multiple variations
                    variations = generator.generate_variations(
                        whistle_audio=str(whistle_file),
                        num_variations=args.num_variations,
                        variation_type="random",
                        temperature=config['generation']['temperature'],
                        max_length=config['generation']['max_length'],
                        style=args.style
                    )
                    
                    # Save variations
                    for j, variation in enumerate(variations):
                        output_name = f"{whistle_file.stem}_variation_{j+1}.mid"
                        output_path = output_dir / output_name
                        
                        if 'midi' in variation:
                            variation['midi'].write(str(output_path))
                            logger.info(f"  Generated: {output_path}")
                        
                        results.append({
                            'input_file': str(whistle_file),
                            'output_file': str(output_path),
                            'variation': j+1,
                            'style': args.style
                        })
                
                else:
                    # Generate single version
                    output_name = f"{whistle_file.stem}_generated.mid"
                    output_path = output_dir / output_name
                    
                    result = converter.convert_whistle_to_music(
                        whistle_audio=str(whistle_file),
                        output_path=str(output_path),
                        style=args.style,
                        temperature=config['generation']['temperature'],
                        max_length=config['generation']['max_length'],
                        top_k=config['generation']['top_k'],
                        top_p=config['generation']['top_p']
                    )
                    
                    logger.info(f"  Generated: {output_path}")
                    
                    results.append({
                        'input_file': str(whistle_file),
                        'output_file': str(output_path),
                        'style': args.style
                    })
                
            except Exception as e:
                logger.error(f"  Error processing {whistle_file.name}: {e}")
                results.append({
                    'input_file': str(whistle_file),
                    'error': str(e)
                })
        
        # Save batch results
        results_file = output_dir / "batch_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Batch conversion completed!")
        logger.info(f"Results saved to: {results_file}")
        
        # Summary
        successful = len([r for r in results if 'error' not in r])
        failed = len(results) - successful
        
        logger.info(f"Summary:")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Total files: {len(results)}")
        
    except Exception as e:
        logger.error(f"Error during batch conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
