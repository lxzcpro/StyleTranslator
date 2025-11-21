"""
Factory for creating reward components.
"""

import logging
import os
import torch
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
    def _resolve_device(device: str) -> str:
        """Resolve 'auto' to actual device string."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @staticmethod
    def create_format_reward() -> FormatRewardBase:
        return FormatReward()

    @classmethod
    def create_semantic_reward(
        cls,
        model_name: str = "wmt22-cometkiwi-da",
        model_path: Optional[str] = None,
        device: str = "cpu"
    ) -> SemanticRewardBase:
        # FIX: Resolve device to avoid COMET crashing on 'auto'
        resolved_device = cls._resolve_device(device)
        return CometSemanticReward(
            model_name=model_name,
            model_path=model_path,
            device=resolved_device
        )

    @classmethod
    def create_style_reward(
        cls,
        chinese_bert_path: str,
        english_bert_path: str,
        style_types: list,
        device: str = "auto",
        test_mode: bool = False
    ) -> StyleRewardBase:
        if test_mode:
            logger.info("Creating MockStyleRewardModel (test mode)")
            return MockStyleRewardModel(style_types=style_types)

        logger.info("Creating StyleRewardModel (production mode)")
        # StyleRewardModel handles 'auto' internally, but we can resolve it too
        resolved_device = cls._resolve_device(device)
        return StyleRewardModel(
            chinese_bert_path=chinese_bert_path,
            english_bert_path=english_bert_path,
            style_types=style_types,
            device=resolved_device
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
        reward_config = config.get('reward', {})
        model_config = config.get('model', {})

        format_reward = cls.create_format_reward()

        # FIX: Ensure comet gets a clean device string
        semantic_reward = cls.create_semantic_reward(
            model_name=reward_config.get('comet_model', 'wmt22-cometkiwi-da'),
            model_path=reward_config.get('comet_path'),
            device=reward_config.get('comet_device', 'cpu')
        )

        test_mode = reward_config.get('test_mode', False)
        chinese_bert_path = reward_config.get('chinese_bert_path', '')
        english_bert_path = reward_config.get('english_bert_path', '')

        # Auto-fallback logic
        if not test_mode:
            if not chinese_bert_path or not os.path.exists(chinese_bert_path):
                logger.warning(f"Invalid chinese_bert_path: {chinese_bert_path}. Using mock style reward.")
                test_mode = True
            elif not english_bert_path or not os.path.exists(english_bert_path):
                logger.warning(f"Invalid english_bert_path: {english_bert_path}. Using mock style reward.")
                test_mode = True

        style_reward = cls.create_style_reward(
            chinese_bert_path=chinese_bert_path,
            english_bert_path=english_bert_path,
            style_types=reward_config.get('style_types', ['law', 'literature', 'news', 'science']),
            device=model_config.get('device', 'auto'),
            test_mode=test_mode
        )

        return cls.create_reward_manager(
            format_reward=format_reward,
            semantic_reward=semantic_reward,
            style_reward=style_reward,
            format_weight=reward_config.get('format_weight', 1.0),
            semantic_weight=reward_config.get('semantic_weight', 6.0),
            style_weight=reward_config.get('style_weight', 4.0)
        )