#!/usr/bin/env python3
"""
Simple CLI for training style-aware translation models using GRPO.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --output ./my_output
"""

import argparse
import logging
import sys
from pathlib import Path

from trainer import GRPOStyleTrainer
from config_manager import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_sample_dataset():
    """Load a sample dataset for demonstration."""
    # Simple WMT24 sample data
    data = [
        {
            'src_text': "Siso's depictions of land, water center new gallery exhibition\n",
            'tgt_text': '西索画作成为新画廊展览的焦点\n',
            'lang_pair': 'en-zh',
        },
        {
            'src_text': '"People Swimming in the Swimming Pool" from 2022 is one Vicente Siso artwork.\n',
            'tgt_text': '2022年的《泳池戏水》是维森特·西索的又一作品。\n',
            'lang_pair': 'en-zh',
        },
    ]

    # Format as prompts
    formatted_data = []
    for item in data:
        prompt = f"""A conversation between User and Assistant. The User asks for a translation from English to Chinese, and the Assistant translates it. The final translation are enclosed within <translate> </translate> tags.

User:{item['src_text']}
Assistant:"""

        formatted_data.append({
            'prompt': prompt,
            'src_text': item['src_text'],
            'tgt_text': item['tgt_text'],
            'lang_pair': item['lang_pair'],
        })

    return formatted_data


def main():
    parser = argparse.ArgumentParser(description='Train style-aware translation model')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory (overrides config)'
    )
    parser.add_argument(
        '--sample-data',
        action='store_true',
        help='Use built-in sample dataset'
    )
    parser.add_argument(
        '--data-file',
        type=str,
        help='Path to training data file (parquet or jsonl)'
    )

    args = parser.parse_args()

    # Load configuration
    try:
        logger.info(f"Loading config from: {args.config}")
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Create trainer
    try:
        trainer = GRPOStyleTrainer(config)
    except Exception as e:
        logger.error(f"Failed to create trainer: {e}")
        sys.exit(1)

    # Load dataset
    if args.sample_data:
        logger.info("Using sample dataset")
        dataset = load_sample_dataset()
    elif args.data_file:
        logger.info(f"Loading data from: {args.data_file}")
        # TODO: Implement data loading from file
        logger.error("Loading from file not yet implemented")
        sys.exit(1)
    else:
        logger.info("Using sample dataset (use --data-file to load custom data)")
        dataset = load_sample_dataset()

    # Run training
    try:
        trainer.train(dataset, output_dir=args.output)
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
