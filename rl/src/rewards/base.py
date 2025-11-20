"""
Base classes for reward calculators following SOLID principles.
All reward calculators should inherit from these base classes.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class RewardResult:
    """Standard reward result structure."""
    score: float
    details: Dict[str, Any]

    def __post_init__(self):
        """Validate score is within expected range."""
        if not isinstance(self.score, (int, float)):
            raise ValueError(f"Score must be numeric, got {type(self.score)}")


class BaseReward(ABC):
    """Abstract base class for all reward calculators."""

    @abstractmethod
    def calculate(self, **kwargs) -> RewardResult:
        """
        Calculate reward score.

        Returns:
            RewardResult: Contains score and detailed information
        """
        pass

    @abstractmethod
    def batch_calculate(self, batch_data: List[Dict[str, Any]]) -> List[RewardResult]:
        """
        Calculate rewards for a batch of inputs.

        Args:
            batch_data: List of input dictionaries

        Returns:
            List of RewardResult objects
        """
        pass

    def get_name(self) -> str:
        """Get the name of this reward calculator."""
        return self.__class__.__name__


class FormatRewardBase(BaseReward):
    """Base class for format validation rewards."""

    @abstractmethod
    def extract_translation(self, generated_text: str) -> str:
        """Extract translation content from generated text."""
        pass


class SemanticRewardBase(BaseReward):
    """Base class for semantic similarity rewards."""

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the underlying model."""
        pass


class StyleRewardBase(BaseReward):
    """Base class for style consistency rewards."""

    @abstractmethod
    def predict_style_probabilities(self, text: str, language: str) -> Any:
        """Predict style probability distribution for text."""
        pass

    @abstractmethod
    def calculate_similarity(self, source_text: str, target_text: str,
                           source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Calculate style similarity between source and target."""
        pass
