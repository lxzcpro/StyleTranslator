"""
GRPO Trainer module - Clean, modular training orchestration with Hydra config support.
"""

import logging
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from omegaconF import DictConfig, OmegaConf

from ..rewards import RewardFactory, RewardManager

logger = logging.getLogger(__name__)


class GRPOStyleTrainer:
    """
    GRPO trainer for style-aware translation.

    This class orchestrates the training process using GRPO (Group Relative Policy Optimization)
    with custom reward functions for format, semantic, and style consistency.
    """

    def __init__(
        self,
        config: DictConfig,
        reward_manager: Optional[RewardManager] = None
    ):
        """
        Initialize trainer.

        Args:
            config: Hydra configuration object
            reward_manager: Optional pre-configured reward manager
        """
        self.config = config
        self.device = self._setup_device()

        # Initialize components
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.reward_manager = reward_manager
        self.trainer: Optional[GRPOTrainer] = None

        # State
        self.current_dataset: Optional[List[Dict[str, Any]]] = None

        logger.info(f"GRPOStyleTrainer initialized on device: {self.device}")

    def _setup_device(self) -> str:
        """Setup computation device."""
        device_config = self.config.get('device', 'auto')
        if device_config == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device_config

        if device == "cuda" and torch.cuda.is_available():
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Using CPU device")

        return device

    def setup_model_and_tokenizer(self) -> None:
        """Load model and tokenizer."""
        model_config = self.config.get('model', {})
        model_path = model_config.get('path', 'Qwen/Qwen2.5-0.5B-Instruct')

        logger.info(f"Loading model from: {model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        device_map = "auto" if self.device == "cuda" else None

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
        )

        logger.info("Model and tokenizer loaded successfully")

    def setup_reward_manager(self) -> None:
        """Setup reward manager if not provided."""
        if self.reward_manager is None:
            logger.info("Creating reward manager from config")
            self.reward_manager = RewardFactory.create_from_config(
                OmegaConf.to_container(self.config, resolve=True)
            )

    def create_reward_function(self) -> Callable:
        """
        Create reward function for GRPO trainer.

        Returns:
            Reward function that takes prompts and completions
        """
        def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
            """
            Compute rewards for generated completions.

            Args:
                prompts: Input prompts
                completions: Generated completions
                **kwargs: Additional arguments

            Returns:
                List of reward scores
            """
            # Extract source texts and language pairs from prompts
            src_texts = []
            lang_pairs = []

            for prompt in prompts:
                # Extract source text from prompt
                if "User:" in prompt:
                    user_input = prompt.split("User:")[1].split("\n")[0].strip()
                    src_texts.append(user_input)
                    lang_pairs.append('en-zh')  # TODO: Extract from prompt or config
                else:
                    src_texts.append("")
                    lang_pairs.append('en-zh')

            # Get reference translations if available
            reference_texts = self._get_reference_texts(len(completions))
            if not reference_texts:
                reference_texts = src_texts
                logger.debug("No references available, using source texts")

            # Calculate rewards
            try:
                rewards = self.reward_manager.calculate_batch_rewards(
                    generated_texts=completions,
                    source_texts=src_texts,
                    prompts=prompts,
                    language_pairs=lang_pairs,
                    reference_texts=reference_texts
                )

                # Log reward details
                self._log_reward_details(prompts, completions, rewards)

                # Extract total scores
                return [r.total_score for r in rewards]

            except Exception as e:
                logger.error(f"Error calculating rewards: {e}", exc_info=True)
                # Return neutral rewards on error
                return [0.0] * len(completions)

        return reward_fn

    def _get_reference_texts(self, count: int) -> Optional[List[str]]:
        """
        Get reference texts from current dataset.

        Args:
            count: Number of references needed

        Returns:
            List of reference texts or None
        """
        if self.current_dataset is None:
            return None

        references = []
        for i in range(count):
            if i < len(self.current_dataset):
                references.append(self.current_dataset[i].get('tgt_text', ''))
            else:
                references.append('')

        return references if any(references) else None

    def _log_reward_details(
        self,
        prompts: List[str],
        completions: List[str],
        rewards: List[Any]
    ) -> None:
        """Log detailed reward information."""
        logger.info("=== Reward Details ===")
        for i, reward in enumerate(rewards):
            logger.info(f"Sample {i+1}:")
            logger.info(f"  Total: {reward.total_score:.3f}")
            logger.info(f"  Format: {reward.components.format_score:.3f}")
            logger.info(f"  Semantic: {reward.components.semantic_score:.3f}")
            logger.info(f"  Style: {reward.components.style_score:.3f}")
            if not reward.is_valid:
                logger.warning(f"  Invalid: {reward.error_message}")

    def train(
        self,
        train_dataset: List[Dict[str, Any]],
        output_dir: Optional[str] = None
    ) -> None:
        """
        Run training.

        Args:
            train_dataset: Training dataset
            output_dir: Optional output directory (uses config if not provided)
        """
        # Setup components
        self.setup_model_and_tokenizer()
        self.setup_reward_manager()

        # Store dataset reference
        self.current_dataset = train_dataset

        # Prepare output directory
        if output_dir is None:
            output_dir = self.config.get('output', {}).get('output_dir', './outputs')
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Create GRPO config
        training = self.config.get('training', {})
        output_cfg = self.config.get('output', {})
        grpo_config = GRPOConfig(
            output_dir=output_dir,
            num_train_epochs=training.get('num_epochs', 1),
            per_device_train_batch_size=training.get('batch_size', 1),
            learning_rate=training.get('learning_rate', 1e-5),
            logging_steps=output_cfg.get('logging_steps', 10),
            save_steps=output_cfg.get('save_steps', 100),
            beta=training.get('beta', 0.04),
        )

        # Extract prompts from dataset
        prompts = [item['prompt'] for item in train_dataset]

        # Create reward function
        reward_fn = self.create_reward_function()

        # Create trainer
        logger.info("Creating GRPO trainer")
        self.trainer = GRPOTrainer(
            model=self.model,
            config=grpo_config,
            tokenizer=self.tokenizer,
            reward_function=reward_fn,
        )

        # Run training
        logger.info("Starting training...")
        self.trainer.train()
        logger.info("Training completed")
