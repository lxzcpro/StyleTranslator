#!/usr/bin/env python3
"""
RL training entry point.
Loads real data and handles environment configuration automatically.
"""

import logging
import sys
import os
import pandas as pd
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict  # <--- Added open_dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trainer.r1_trainer import GRPOStyleTrainer
from src.rewards import RewardFactory

logger = logging.getLogger(__name__)

def load_real_dataset(file_path: str, max_samples: int = None):
    """
    Load real training data from Parquet or JSONL.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        # Fallback for relative paths from project root
        file_path = Path(__file__).parent.parent.parent / file_path
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    if file_path.suffix == '.parquet':
        df = pd.read_parquet(file_path)
    elif file_path.suffix == '.jsonl':
        df = pd.read_json(file_path, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    # Column mapping logic to handle various dataset formats
    col_map = {}
    if 'src_text' in df.columns: col_map['src'] = 'src_text'
    elif 'source' in df.columns: col_map['src'] = 'source'
    elif 'src' in df.columns: col_map['src'] = 'src'
    
    if 'tgt_text' in df.columns: col_map['tgt'] = 'tgt_text'
    elif 'target' in df.columns: col_map['tgt'] = 'target'
    elif 'tgt' in df.columns: col_map['tgt'] = 'tgt'

    if 'src' not in col_map:
        raise ValueError(f"Could not find source text column. Available: {df.columns.tolist()}")

    # Sampling
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    formatted_data = []
    for _, row in df.iterrows():
        src = row[col_map['src']]
        tgt = row.get(col_map.get('tgt'), "") 
        
        # Construct Prompt
        prompt = (
            "A conversation between User and Assistant. The User asks for a translation from English to Chinese, "
            "and the Assistant translates it. The final translation is enclosed within <translate> </translate> tags.\n\n"
            f"User: {src}\n"
            "Assistant:"
        )

        formatted_data.append({
            'prompt': prompt,
            'src_text': src,
            'tgt_text': tgt,
            'lang_pair': row.get('lang_pair', 'en-zh'),
        })

    return formatted_data

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, cfg.get('log_level', 'INFO')),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Suppress verbose logs
    logging.getLogger("src.rewards.semantic").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    logger.info("Starting RL Training")

    with open_dict(cfg):
        if 'reward' not in cfg:
            cfg.reward = {}

        if os.environ.get('CHINESE_BERT_PATH'):
            cfg.reward.chinese_bert_path = os.environ['CHINESE_BERT_PATH']

        if os.environ.get('ENGLISH_BERT_PATH'):
            cfg.reward.english_bert_path = os.environ['ENGLISH_BERT_PATH']

    try:
        # Create reward manager
        reward_manager = RewardFactory.create_from_config(OmegaConf.to_container(cfg, resolve=True))

        # Create trainer
        trainer = GRPOStyleTrainer(cfg, reward_manager=reward_manager)

        # Load REAL dataset
        train_file = cfg.data.get('train_file', 'data/train/train_style.parquet')
        max_samples = cfg.data.get('max_train_samples', 1000)
        
        dataset = load_real_dataset(train_file, max_samples)
        logger.info(f"Loaded {len(dataset)} training samples from {train_file}")

        # Train
        trainer.train(dataset)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()