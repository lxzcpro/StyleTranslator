"""
GRPO Trainer module - Fixed for Numerical Stability (BF16).
"""

import logging
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from omegaconf import DictConfig, OmegaConf

from ..rewards import RewardFactory, RewardManager

logger = logging.getLogger(__name__)


class GRPOStyleTrainer:
    """
    GRPO trainer for style-aware translation.
    """

    def __init__(
        self,
        config: DictConfig,
        reward_manager: Optional[RewardManager] = None
    ):
        self.config = config
        self.device = self._setup_device()

        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.reward_manager = reward_manager
        self.trainer: Optional[GRPOTrainer] = None
        self.current_dataset: Optional[List[Dict[str, Any]]] = None

        logger.info(f"GRPOStyleTrainer initialized on device: {self.device}")

    def _setup_device(self) -> str:
        device_config = self.config.get('device', 'auto')
        if device_config == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_config

    def setup_model_and_tokenizer(self) -> None:
        model_config = self.config.get('model', {})
        model_path = model_config.get('path', 'Qwen/Qwen2.5-0.5B-Instruct')
        
        logger.info(f"Loading model from: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Ensure pad token is set to avoid generation warnings/errors
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # === FIX: Use bfloat16 to prevent NaN/Inf overflow on Ampere GPUs ===
        if self.device == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            logger.info("Using bfloat16 for numerical stability")
        else:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            logger.info(f"Using {dtype} (bf16 not available)")
            
        device_map = "auto" if self.device == "cuda" else None

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            # Optional: enable flash attention for speed if using bf16
            attn_implementation="flash_attention_2" if dtype == torch.bfloat16 else None
        )
        logger.info("Model and tokenizer loaded successfully")

    def setup_reward_manager(self) -> None:
        if self.reward_manager is None:
            logger.info("Creating reward manager from config")
            self.reward_manager = RewardFactory.create_from_config(
                OmegaConf.to_container(self.config, resolve=True)
            )

    def create_reward_function(self) -> Callable:
        # Map prompts to metadata for O(1) lookup during training
        prompt_map = {}
        if self.current_dataset:
            for item in self.current_dataset:
                if 'prompt' in item:
                    prompt_map[item['prompt']] = {
                        'src': item.get('src_text', ''),
                        'ref': item.get('tgt_text', ''),
                        'lang': item.get('lang_pair', 'en-zh')
                    }

        def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
            src_texts = []
            reference_texts = []
            lang_pairs = []

            for prompt in prompts:
                data = prompt_map.get(prompt, {})
                src_texts.append(data.get('src', ''))
                reference_texts.append(data.get('ref', ''))
                lang_pairs.append(data.get('lang', 'en-zh'))

            try:
                rewards = self.reward_manager.calculate_batch_rewards(
                    generated_texts=completions,
                    source_texts=src_texts,
                    prompts=prompts,
                    language_pairs=lang_pairs,
                    reference_texts=reference_texts
                )
                
                self._log_reward_details(prompts, completions, rewards)
                return [float(r.total_score) for r in rewards]

            except Exception as e:
                logger.error(f"Error calculating rewards: {e}", exc_info=True)
                return [0.0] * len(completions)

        return reward_fn

    def _log_reward_details(self, prompts: List[str], completions: List[str], rewards: List[Any]) -> None:
        if not rewards: return
        r = rewards[0]
        if hasattr(r, 'total_score'):
            msg = f"Sample Score: {r.total_score:.3f}"
            if hasattr(r, 'components') and r.components:
                c = r.components
                msg += f" (Fmt: {c.format_score:.2f}, Sem: {c.semantic_score:.2f}, Sty: {c.style_score:.2f})"
            logger.info(msg)

    def train(self, train_dataset: List[Dict[str, Any]], output_dir: Optional[str] = None) -> None:
        self.setup_model_and_tokenizer()
        self.setup_reward_manager()
        self.current_dataset = train_dataset

        hf_dataset = Dataset.from_list(train_dataset)

        if output_dir is None:
            output_dir = self.config.get('output', {}).get('output_dir', './outputs')
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        training_cfg = self.config.get('training', {})
        output_cfg = self.config.get('output', {})
        
        num_generations = training_cfg.get('num_generations', 4)
        batch_size = training_cfg.get('batch_size', 1)
        max_length = training_cfg.get('max_length', 512)
        
        generation_batch_size = batch_size * num_generations
        
        # Check if we should enable bf16 training
        use_bf16 = (self.device == "cuda" and torch.cuda.is_bf16_supported())

        grpo_config = GRPOConfig(
            output_dir=output_dir,
            num_train_epochs=training_cfg.get('num_epochs', 1),
            per_device_train_batch_size=batch_size,
            learning_rate=training_cfg.get('learning_rate', 1e-5),
            logging_steps=output_cfg.get('logging_steps', 10),
            save_steps=output_cfg.get('save_steps', 100),
            beta=training_cfg.get('beta', 0.04),
            # GRPO Specifics
            num_generations=num_generations,
            max_completion_length=max_length,
            max_prompt_length=max_length,
            generation_batch_size=generation_batch_size,
            # === FIX: Enable BF16 training ===
            bf16=use_bf16,
            fp16=not use_bf16, # Fallback to fp16 only if bf16 not supported
        )

        reward_fn = self.create_reward_function()

        logger.info(f"Creating GRPO trainer (BF16: {use_bf16})")
        self.trainer = GRPOTrainer(
            model=self.model,
            args=grpo_config,
            processing_class=self.tokenizer,
            reward_funcs=[reward_fn],
            train_dataset=hf_dataset,
        )

        logger.info("Starting training...")
        self.trainer.train()
        logger.info("Training completed")