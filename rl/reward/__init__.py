"""
Reward module for RL training.

This module provides a modular reward system with:
- Format validation (XML tags)
- Semantic similarity (COMET)
- Style consistency (BERT-based)

Usage:
    from rl.reward import RewardFactory

    config = {'reward': {...}, 'model': {...}}
    reward_manager = RewardFactory.create_from_config(config)

    rewards = reward_manager.calculate_batch_rewards(
        generated_texts, source_texts, prompts,
        language_pairs, reference_texts
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
from .reward_manager import (
    RewardManager,
    RewardWeights,
    RewardComponents,
    RewardOutput,
    LanguageMapper
)
from .reward_factory import RewardFactory

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

    # Reward manager
    'RewardManager',
    'RewardWeights',
    'RewardComponents',
    'RewardOutput',
    'LanguageMapper',

    # Factory
    'RewardFactory',
]
