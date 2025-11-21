"""
GRPO Trainer module - Qwen-1.5B Mode.
Fixed: Indentation of 'train' method and generation_batch_size calculation.
"""

import logging
import warnings
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import numpy as np

import torch
import wandb
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model, TaskType

from ..rewards import RewardFactory, RewardManager

logger = logging.getLogger(__name__)

class GRPOStyleTrainer:
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
        # Default to Qwen 1.5B
        model_path = model_config.get('path', 'Qwen/Qwen2.5-1.5B-Instruct')
        
        logger.info(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.device == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            
        device_map = "auto" if self.device == "cuda" else None

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation="flash_attention_2" if dtype == torch.bfloat16 else None
        )

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )

        self.model = get_peft_model(self.model, peft_config)
        
        self.model.print_trainable_parameters()
        
        logger.info(f"Model loaded with LoRA: {model_path} (dtype: {dtype})")

    def setup_reward_manager(self) -> None:
        if self.reward_manager is None:
            self.reward_manager = RewardFactory.create_from_config(
                OmegaConf.to_container(self.config, resolve=True)
            )

    def create_reward_function(self) -> Callable:
        prompt_map = {}
        if self.current_dataset:
            for item in self.current_dataset:
                if 'prompt' in item:
                    prompt_map[item['prompt']] = item
                    prompt_map[item['prompt'].strip()] = item

        def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
            src_texts = []
            reference_texts = []
            lang_pairs = []
            
            # === RECONSTRUCT FULL XML ===
            # Prepend <translate> to match the Forced Prefix strategy
            full_generations = ["<translate>" + c for c in completions]
            
            failed_indices = []

            for idx, prompt in enumerate(prompts):
                data = prompt_map.get(prompt)
                if not data:
                    data = prompt_map.get(prompt.strip())
                
                if not data:
                    if idx == 0:
                         logger.warning(f"Prompt lookup failed for prefix: {prompt[:30]}...")
                    failed_indices.append(idx)
                    src_texts.append("")
                    reference_texts.append("")
                    lang_pairs.append("en-zh")
                else:
                    src_texts.append(data.get('src_text', ''))
                    reference_texts.append(data.get('tgt_text', ''))
                    lang_pairs.append(data.get('lang_pair', 'en-zh'))

            try:
                rewards = self.reward_manager.calculate_batch_rewards(
                    generated_texts=full_generations,
                    source_texts=src_texts,
                    prompts=prompts,
                    language_pairs=lang_pairs,
                    reference_texts=reference_texts
                )
                
                if wandb.run:
                    valid_rewards = [r for i, r in enumerate(rewards) if i not in failed_indices]
                    if valid_rewards:
                        stats = {
                            "train/reward_total": np.mean([r.total_score for r in valid_rewards]),
                            "train/reward_format": np.mean([r.components.format_score for r in valid_rewards]),
                            "train/reward_semantic": np.mean([r.components.semantic_score for r in valid_rewards]),
                            "train/reward_style": np.mean([r.components.style_score for r in valid_rewards]),
                        }
                        wandb.log(stats)

                self._log_reward_details(prompts, full_generations, rewards)
                
                final_scores = []
                for i, r in enumerate(rewards):
                    if i in failed_indices:
                        final_scores.append(0.0)
                    else:
                        final_scores.append(float(r.total_score))
                        
                return final_scores

            except Exception as e:
                logger.error(f"Error calculating rewards: {e}")
                return [0.0] * len(completions)

        return reward_fn

    def _log_reward_details(self, prompts: List[str], completions: List[str], rewards: List[Any]) -> None:
        if not rewards: return
        
        preview = completions[0].replace('\n', ' ')[:100]
        logger.info(f"Eval Text: {preview}")
        
        r = rewards[0]
        if hasattr(r, 'total_score'):
            msg = f"Score: {r.total_score:.3f}"
            if hasattr(r, 'components') and r.components:
                c = r.components
                msg += f" | Fmt: {c.format_score:.2f} | Sem: {c.semantic_score:.2f} | Sty: {c.style_score:.2f}"
            # logger.info(msg)

    # === CRITICAL: Ensure this method is indented at the class level, NOT inside _log_reward_details ===
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
        max_len = training_cfg.get('max_length', 512)
        
        # Explicitly calculate generation_batch_size
        generation_batch_size = batch_size * num_generations

        grpo_config = GRPOConfig(
            output_dir=output_dir,
            num_train_epochs=training_cfg.get('num_epochs', 1),
            per_device_train_batch_size=batch_size,
            learning_rate=training_cfg.get('learning_rate', 1e-6),
            logging_steps=output_cfg.get('logging_steps', 10),
            save_steps=output_cfg.get('save_steps', 100),
            beta=training_cfg.get('beta', 0.04),
            num_generations=num_generations,
            max_completion_length=max_len,
            max_prompt_length=max_len,
            generation_batch_size=generation_batch_size,
            bf16=(self.device == "cuda" and torch.cuda.is_bf16_supported()),
            report_to=["wandb"], 
            gradient_accumulation_steps=training_cfg.get('gradient_accumulation_steps', 1),
            # use_vllm=False,
            # vllm_gpu_memory_utilization=0.4, 
            # vllm_dtype="bfloat16",
        )

        reward_fn = self.create_reward_function()

        self.trainer = GRPOTrainer(
            model=self.model,
            args=grpo_config,
            processing_class=self.tokenizer,
            reward_funcs=[reward_fn],
            train_dataset=hf_dataset,
        )

        logger.info("Training started")
        self.trainer.train()
        logger.info("Training completed")