#!/usr/bin/env python3
"""
RL training entry point using Hydra for configuration management.

Usage:
    python scripts/train_rl.py
    python scripts/train_rl.py env=server reward=style_weighted
    python scripts/train_rl.py --config-name=config
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trainer.r1_trainer import GRPOStyleTrainer
from src.rewards import RewardFactory

logger = logging.getLogger(__name__)


def load_sample_dataset():
    """Load a sample dataset for demonstration."""
    data = [
        {
            'src_text': "Siso's depictions of land, water center new gallery exhibition",
            'tgt_text': '西索画作成为新画廊展览的焦点',
            'lang_pair': 'en-zh',
        },
        {
            'src_text': '"People Swimming in the Swimming Pool" from 2022 is one Vicente Siso artwork.',
            'tgt_text': '2022年的《泳池戏水》是维森特·西索的又一作品。',
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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training function.

    Args:
        cfg: Hydra configuration
    """
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, cfg.get('log_level', 'INFO')),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger.info("=" * 80)
    logger.info("RL Training with Hydra Configuration")
    logger.info("=" * 80)
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    try:
        # Create reward manager
        logger.info("Creating reward manager...")
        reward_manager = RewardFactory.create_from_config(OmegaConf.to_container(cfg, resolve=True))

        # Create trainer
        logger.info("Initializing GRPO trainer...")
        trainer = GRPOStyleTrainer(cfg, reward_manager=reward_manager)

        # Load dataset
        # TODO: Load from cfg.data.train_file
        logger.info("Loading training dataset...")
        dataset = load_sample_dataset()
        logger.info(f"Loaded {len(dataset)} training samples")

        # Train
        logger.info("Starting training...")
        trainer.train(dataset)

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
