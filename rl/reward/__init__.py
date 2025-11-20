"""
Reward module for RL training.

This module provides a modular reward system with:
- Format validation (XML tags)
- Semantic similarity (COMET)
- Style consistency (BERT-based)

Usage:
    from rl.reward import RewardFactory, load_config

    config = load_config('config.yaml')
    reward_manager = RewardFactory.create_from_config(config.to_dict())

    rewards = reward_manager.calculate_batch_rewards(
        generated_texts=generated_texts,
        source_texts=source_texts,
        prompts=prompts,
        language_pairs=language_pairs,
        reference_texts=reference_texts
    )
"""

from .base_reward import (
    BaseReward,
    FormatRewardBase,
    SemanticRewardBase,
    StyleRewardBase,
    RewardResult
)
from .format_score import FormatReward
from .comet_score import CometSemanticReward
from .style_score import StyleRewardModel, MockStyleRewardModel
from .reward_manager_v2 import (
    RewardManager,
    RewardWeights,
    RewardComponents,
    RewardOutput,
    LanguageMapper
)
from .reward_factory import RewardFactory

# Keep backward compatibility with old reward_manager
from .reward_manager import RewardManager as LegacyRewardManager

__all__ = [
    # Base classes
    'BaseReward',
    'FormatRewardBase',
    'SemanticRewardBase',
    'StyleRewardBase',
    'RewardResult',

    # Reward calculators
    'FormatReward',
    'CometSemanticReward',
    'StyleRewardModel',
    'MockStyleRewardModel',

    # Reward manager v2
    'RewardManager',
    'RewardWeights',
    'RewardComponents',
    'RewardOutput',
    'LanguageMapper',

    # Factory
    'RewardFactory',

    # Legacy (for backward compatibility)
    'LegacyRewardManager',
]
