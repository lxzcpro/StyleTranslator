"""
Factory for creating reward calculators and manager.
Implements the Factory pattern for better testability and maintainability.
"""

import logging
import os
from typing import Optional

from .base import FormatRewardBase, SemanticRewardBase, StyleRewardBase
from .format import FormatReward
from .semantic import CometSemanticReward
from .style import StyleRewardModel, MockStyleRewardModel
from .manager import RewardManager, RewardWeights, LanguageMapper

logger = logging.getLogger(__name__)


class RewardFactory:
    """Factory for creating reward components."""

    @staticmethod
    def create_format_reward() -> FormatRewardBase:
        """Create format reward calculator."""
        return FormatReward()

    @staticmethod
    def create_semantic_reward(
        model_name: str = "wmt22-cometkiwi-da",
        model_path: Optional[str] = None,
        device: str = "cpu"
    ) -> SemanticRewardBase:
        """
        Create semantic reward calculator.

        Args:
            model_name: COMET model name
            model_path: Optional path to model checkpoint
            device: Device to run on ('cpu' or 'cuda')

        Returns:
            CometSemanticReward instance
        """
        return CometSemanticReward(
            model_name=model_name,
            model_path=model_path,
            device=device
        )

    @staticmethod
    def create_style_reward(
        chinese_bert_path: str,
        english_bert_path: str,
        style_types: list,
        device: str = "auto",
        test_mode: bool = False
    ) -> StyleRewardBase:
        """
        Create style reward calculator.

        Args:
            chinese_bert_path: Path to Chinese BERT checkpoint
            english_bert_path: Path to English BERT checkpoint
            style_types: List of style type names
            device: Device to run on ('auto', 'cpu', or 'cuda')
            test_mode: If True, use mock model for testing

        Returns:
            StyleRewardModel or MockStyleRewardModel instance
        """
        if test_mode:
            logger.info("Creating MockStyleRewardModel (test mode)")
            return MockStyleRewardModel(style_types=style_types)

        logger.info("Creating StyleRewardModel (production mode)")
        return StyleRewardModel(
            chinese_bert_path=chinese_bert_path,
            english_bert_path=english_bert_path,
            style_types=style_types,
            device=device
        )

    @staticmethod
    def create_reward_manager(
        format_reward: FormatRewardBase,
        semantic_reward: SemanticRewardBase,
        style_reward: StyleRewardBase,
        format_weight: float = 1.0,
        semantic_weight: float = 6.0,
        style_weight: float = 4.0,
        language_mapper: Optional[LanguageMapper] = None
    ) -> RewardManager:
        """
        Create reward manager with injected dependencies.

        Args:
            format_reward: Format reward calculator
            semantic_reward: Semantic reward calculator
            style_reward: Style reward calculator
            format_weight: Weight for format reward
            semantic_weight: Weight for semantic reward
            style_weight: Weight for style reward
            language_mapper: Optional custom language mapper

        Returns:
            RewardManager instance
        """
        weights = RewardWeights(
            format_weight=format_weight,
            semantic_weight=semantic_weight,
            style_weight=style_weight
        )

        return RewardManager(
            format_reward=format_reward,
            semantic_reward=semantic_reward,
            style_reward=style_reward,
            weights=weights,
            language_mapper=language_mapper
        )

    @classmethod
    def create_from_config(cls, config: dict) -> RewardManager:
        """
        Create complete reward manager from configuration.

        Args:
            config: Configuration dictionary (e.g., from config.yaml)

        Returns:
            Fully configured RewardManager instance
        """
        reward_config = config.get('reward', {})
        model_config = config.get('model', {})

        # Create individual reward calculators
        format_reward = cls.create_format_reward()

        semantic_reward = cls.create_semantic_reward(
            model_name=reward_config.get('comet_model', 'wmt22-cometkiwi-da'),
            model_path=reward_config.get('comet_path'),
            device=reward_config.get('comet_device', 'cpu')
        )

        # Validate BERT model paths if not in test mode
        test_mode = reward_config.get('test_mode', False)
        chinese_bert_path = reward_config.get('chinese_bert_path', '')
        english_bert_path = reward_config.get('english_bert_path', '')

        if not test_mode:
            if not chinese_bert_path:
                logger.warning("chinese_bert_path not provided in config. Using mock model for style reward.")
                test_mode = True
            elif not os.path.exists(chinese_bert_path):
                logger.warning(f"chinese_bert_path does not exist: {chinese_bert_path}. Using mock model for style reward.")
                test_mode = True

            if not english_bert_path:
                logger.warning("english_bert_path not provided in config. Using mock model for style reward.")
                test_mode = True
            elif not os.path.exists(english_bert_path):
                logger.warning(f"english_bert_path does not exist: {english_bert_path}. Using mock model for style reward.")
                test_mode = True

        style_reward = cls.create_style_reward(
            chinese_bert_path=chinese_bert_path,
            english_bert_path=english_bert_path,
            style_types=reward_config.get('style_types', ['law', 'literature', 'news', 'science']),
            device=model_config.get('device', 'auto'),
            test_mode=test_mode
        )

        # Create reward manager
        return cls.create_reward_manager(
            format_reward=format_reward,
            semantic_reward=semantic_reward,
            style_reward=style_reward,
            format_weight=reward_config.get('format_reward_weight', 1.0),
            semantic_weight=reward_config.get('semantic_reward_weight', 6.0),
            style_weight=reward_config.get('style_reward_weight', 4.0)
        )
