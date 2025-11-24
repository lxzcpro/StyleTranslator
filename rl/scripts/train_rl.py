#!/usr/bin/env python3
"""
RL training entry point.
Optimized for Qwen-1.5B: Direct Translation with Forced Prefix.
"""

import logging
import sys
import os
import warnings

# === LOGGING CONFIG ===
os.environ["LIT_NO_VERSION_CHECK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="pytorch_lightning")
warnings.filterwarnings("ignore", module="lightning_fabric")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("src.rewards.semantic").setLevel(logging.ERROR)

import pandas as pd
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from transformers import AutoTokenizer

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trainer.r1_trainer import GRPOStyleTrainer
from src.rewards import RewardFactory

logger = logging.getLogger(__name__)

def load_real_dataset(file_path: str, tokenizer, max_samples: int = None):
    """
    Load dataset and format prompts with Forced Prefix strategy.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        file_path = Path(__file__).parent.parent.parent / file_path
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    if file_path.suffix == '.parquet':
        df = pd.read_parquet(file_path)
    elif file_path.suffix == '.jsonl':
        df = pd.read_json(file_path, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    col_map = {}
    if 'src_text' in df.columns: col_map['src'] = 'src_text'
    elif 'source' in df.columns: col_map['src'] = 'source'
    elif 'src' in df.columns: col_map['src'] = 'src'
    
    if 'tgt_text' in df.columns: col_map['tgt'] = 'tgt_text'
    elif 'target' in df.columns: col_map['tgt'] = 'target'
    elif 'tgt' in df.columns: col_map['tgt'] = 'tgt'

    if 'src' not in col_map:
        raise ValueError(f"Could not find source text column. Available: {df.columns.tolist()}")

    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    formatted_data = []
    for _, row in df.iterrows():
        src = row[col_map['src']]
        tgt = row.get(col_map.get('tgt'), "") 
        
        # === SIMPLE SYSTEM PROMPT (Direct Translation) ===
        messages = [
            {
                "role": "system", 
                "content": "You are a professional translator. Translate the text into Chinese. Output the translation enclosed in <translate> and </translate> tags."
            },
            # === ONE-SHOT EXAMPLE ===
            {
                "role": "user",
                "content": "The weather is nice today."
            },
            {
                "role": "assistant",
                "content": "<translate>今天天气很好。</translate>"
            },
            # Actual Input
            {
                "role": "user", 
                "content": src
            }
        ]
        
        # === FORCED PREFIX ===
        # Even for 1.5B, starting with <translate> ensures valid format and faster convergence.
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_str += "<translate>"

        formatted_data.append({
            'prompt': prompt_str,
            'src_text': src,
            'tgt_text': tgt,
            'lang_pair': row.get('lang_pair', 'en-zh'),
        })

    return formatted_data

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.get('log_level', 'INFO')),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)

    logger.info("Starting RL Training (Qwen-1.5B Direct Mode)")

    with open_dict(cfg):
        if 'reward' not in cfg: cfg.reward = {}
        if 'reward' not in cfg:
            cfg.reward = {}

        if os.environ.get('CHINESE_BERT_PATH'):
            cfg.reward.chinese_bert_path = os.environ['CHINESE_BERT_PATH']
        if os.environ.get('ENGLISH_BERT_PATH'):
            cfg.reward.english_bert_path = os.environ['ENGLISH_BERT_PATH']

    try:
        # 1. Load Tokenizer (Updated default to 1.5B)
        model_path = cfg.model.get('path', 'Qwen/Qwen2.5-1.5B-Instruct')
        logger.info(f"Loading tokenizer from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 2. Load Data
        train_file = cfg.data.get('train_file', 'data/train/train_style.parquet')
        max_samples = cfg.data.get('max_train_samples', 1000)
        
        dataset = load_real_dataset(train_file, tokenizer, max_samples)
        logger.info(f"Loaded {len(dataset)} formatted training samples.")

        # 3. Start Trainer
        reward_manager = RewardFactory.create_from_config(OmegaConf.to_container(cfg, resolve=True))
        trainer = GRPOStyleTrainer(cfg, reward_manager=reward_manager)

        trainer.train(dataset)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()