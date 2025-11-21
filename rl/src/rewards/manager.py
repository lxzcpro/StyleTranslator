"""
Improved Reward Manager with dependency injection and better separation of concerns.
"""
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np

from .base import FormatRewardBase, SemanticRewardBase, StyleRewardBase

logger = logging.getLogger(__name__)


@dataclass
class RewardWeights:
    """Encapsulates reward weight configuration."""
    format_weight: float = 1.0
    semantic_weight: float = 6.0
    style_weight: float = 4.0

    def __post_init__(self):
        """Validate weights."""
        if self.format_weight < 0 or self.semantic_weight < 0 or self.style_weight < 0:
            raise ValueError("All weights must be non-negative")

    def normalize(self) -> 'RewardWeights':
        """Return normalized weights that sum to 1."""
        total = self.format_weight + self.semantic_weight + self.style_weight
        if total == 0:
            raise ValueError("At least one weight must be positive")
        return RewardWeights(
            format_weight=self.format_weight / total,
            semantic_weight=self.semantic_weight / total,
            style_weight=self.style_weight / total
        )


@dataclass
class RewardComponents:
    """Individual reward component scores."""
    format_score: float
    semantic_score: float
    style_score: float


@dataclass
class RewardOutput:
    """Complete reward output with scores and metadata."""
    total_score: float
    components: RewardComponents
    details: Dict[str, Any]
    is_valid: bool = True
    error_message: str = ""


class LanguageMapper:
    """Handles language code mapping for model selection."""

    DEFAULT_MAPPING = {
        'en': 'english',
        'zh': 'chinese',
        'de': 'english',  # German uses English model
        'ja': 'chinese',  # Japanese uses Chinese model
    }

    def __init__(self, custom_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize language mapper.

        Args:
            custom_mapping: Optional custom language mapping
        """
        self.mapping = custom_mapping if custom_mapping else self.DEFAULT_MAPPING.copy()

    def get_model_language(self, lang_code: str) -> str:
        """
        Get model language for a given language code.

        Args:
            lang_code: ISO 639-1 language code (e.g., 'en', 'zh')

        Returns:
            Model language identifier

        Raises:
            ValueError: If language code is not supported
        """
        if lang_code not in self.mapping:
            raise ValueError(
                f"Unsupported language: {lang_code}. "
                f"Supported languages: {list(self.mapping.keys())}"
            )
        return self.mapping[lang_code]


class RewardManager:
    """
    Improved reward manager using dependency injection.

    This class orchestrates multiple reward calculators and combines their
    scores according to configured weights.
    """

    # Minimum format score threshold - below this, other rewards are zeroed
    FORMAT_THRESHOLD = 0.3

    def __init__(
        self,
        format_reward: FormatRewardBase,
        semantic_reward: SemanticRewardBase,
        style_reward: StyleRewardBase,
        weights: RewardWeights,
        language_mapper: Optional[LanguageMapper] = None
    ):
        """
        Initialize reward manager with injected dependencies.

        Args:
            format_reward: Format validation reward calculator
            semantic_reward: Semantic similarity reward calculator
            style_reward: Style consistency reward calculator
            weights: Reward weight configuration
            language_mapper: Optional custom language mapper
        """
        self.format_reward = format_reward
        self.semantic_reward = semantic_reward
        self.style_reward = style_reward
        self.weights = weights
        self.language_mapper = language_mapper or LanguageMapper()

    def calculate_single_reward(
        self,
        generated_text: str,
        source_text: str,
        prompt: str,
        language_pair: str,
        reference_text: Optional[str] = None
    ) -> RewardOutput:
        """
        Calculate reward for a single generation.

        Args:
            generated_text: Generated text from model
            source_text: Source text
            prompt: Input prompt
            language_pair: Language pair (e.g., 'en-zh')
            reference_text: Optional reference translation

        Returns:
            RewardOutput with total score and component breakdown
        """
        try:
            # 1. Calculate format reward
            format_result = self.format_reward.calculate(
                generated_text=generated_text,
                prompt=prompt
            )
            format_score = format_result.score

            # Extract translation content
            translation = self.format_reward.extract_translation(generated_text)

            # 2. Calculate semantic reward
            if reference_text is None:
                reference_text = source_text
                logger.debug("No reference provided, using source as reference")

            semantic_result = self.semantic_reward.calculate(
                source=source_text,
                reference=reference_text,
                hypothesis=translation if translation else generated_text
            )
            semantic_score = semantic_result.score

            # 3. Calculate style reward
            # Validate and parse language pair
            parts = language_pair.split('-')
            if len(parts) != 2:
                logger.error(f"Invalid language pair format: {language_pair}. Expected format: 'source-target'")
                raise ValueError(f"Invalid language pair format: {language_pair}")
            source_lang, target_lang = parts
            source_model_lang = self.language_mapper.get_model_language(source_lang)
            target_model_lang = self.language_mapper.get_model_language(target_lang)

            if translation:
                style_result = self.style_reward.calculate(
                    source_text=source_text,
                    target_text=translation,
                    source_lang=source_model_lang,
                    target_lang=target_model_lang
                )
                style_score = style_result.score
            else:
                style_score = 0.0
                style_result = None

            # 4. Combine rewards with penalty for poor format
            if format_score <= self.FORMAT_THRESHOLD:
                total_score = -1.0
                semantic_score = 0.0
                style_score = 0.0
                is_valid = False
                error_msg = f"Format score {format_score:.2f} below threshold {self.FORMAT_THRESHOLD}"
            else:
                total_score = (
                    self.weights.format_weight * format_score +
                    self.weights.semantic_weight * semantic_score +
                    self.weights.style_weight * style_score
                )
                is_valid = True
                error_msg = ""

            return RewardOutput(
                total_score=total_score,
                components=RewardComponents(
                    format_score=format_score,
                    semantic_score=semantic_score,
                    style_score=style_score
                ),
                details={
                    'format_details': format_result.details,
                    'semantic_details': semantic_result.details,
                    'style_details': style_result.details if style_result else {},
                    'translation': translation
                },
                is_valid=is_valid,
                error_message=error_msg
            )

        except Exception as e:
            logger.error(f"Error calculating reward: {e}", exc_info=True)
            return RewardOutput(
                total_score=-1.0,
                components=RewardComponents(0.0, 0.0, 0.0),
                details={'error': str(e)},
                is_valid=False,
                error_message=str(e)
            )

    def calculate_batch_rewards(
        self,
        generated_texts: List[str],
        source_texts: List[str],
        prompts: List[str],
        language_pairs: List[str],
        reference_texts: Optional[List[str]] = None
    ) -> List[RewardOutput]:
        """
        Calculate rewards for a batch of generations.

        Args:
            generated_texts: List of generated texts
            source_texts: List of source texts
            prompts: List of prompts
            language_pairs: List of language pairs
            reference_texts: Optional list of reference translations

        Returns:
            List of RewardOutput objects
        """
        batch_size = len(generated_texts)

        # Ensure all lists have the same length
        if not (len(source_texts) == len(prompts) == len(language_pairs) == batch_size):
            raise ValueError("All input lists must have the same length")

        if reference_texts and len(reference_texts) != batch_size:
            raise ValueError("Reference texts list must match batch size")

        results = []
        for i in range(batch_size):
            ref_text = reference_texts[i] if reference_texts else None
            result = self.calculate_single_reward(
                generated_text=generated_texts[i],
                source_text=source_texts[i],
                prompt=prompts[i],
                language_pair=language_pairs[i],
                reference_text=ref_text
            )
            results.append(result)

        return results

    def get_statistics(self, rewards: List[RewardOutput]) -> Dict[str, Any]:
        """
        Calculate statistics from a list of rewards.

        Args:
            rewards: List of RewardOutput objects

        Returns:
            Dictionary with statistics
        """
        if not rewards:
            return {}

        total_scores = [r.total_score for r in rewards]
        format_scores = [r.components.format_score for r in rewards]
        semantic_scores = [r.components.semantic_score for r in rewards]
        style_scores = [r.components.style_score for r in rewards]

        valid_count = sum(1 for r in rewards if r.is_valid)

        return {
            'total': {
                'mean': float(np.mean(total_scores)),
                'std': float(np.std(total_scores)),
                'min': float(np.min(total_scores)),
                'max': float(np.max(total_scores)),
            },
            'format': {
                'mean': float(np.mean(format_scores)),
                'std': float(np.std(format_scores)),
            },
            'semantic': {
                'mean': float(np.mean(semantic_scores)),
                'std': float(np.std(semantic_scores)),
            },
            'style': {
                'mean': float(np.mean(style_scores)),
                'std': float(np.std(style_scores)),
            },
            'validity': {
                'valid_count': valid_count,
                'total_count': len(rewards),
                'valid_ratio': valid_count / len(rewards)
            }
        }
