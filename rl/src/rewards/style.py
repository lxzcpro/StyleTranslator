"""
Style reward module for calculating translation style consistency.
Supports Chinese and English BERT models for style prediction.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import Dict, List, Any
import logging

from style_detector.model.model import StyleDetector
from .base import StyleRewardBase, RewardResult

logger = logging.getLogger(__name__)


class StyleRewardModel(StyleRewardBase):
    """Style reward model using BERT-based style detection."""

    def __init__(
        self,
        chinese_bert_path: str,
        english_bert_path: str,
        style_types: List[str],
        device: str = "auto"
    ):
        """
        Initialize style reward model.

        Args:
            chinese_bert_path: Path to Chinese BERT checkpoint
            english_bert_path: Path to English BERT checkpoint
            style_types: List of style type labels
            device: Device to use ('auto', 'cpu', or 'cuda')
        """
        self.device = self._get_device(device)
        self.style_types = style_types
        self.num_styles = len(style_types)

        # Load Chinese BERT model and tokenizer
        self.chinese_model = self._load_model(chinese_bert_path)
        self.chinese_tokenizer = AutoTokenizer.from_pretrained(
            self.chinese_model.hparams.model_name
        )

        # Load English BERT model and tokenizer
        self.english_model = self._load_model(english_bert_path)
        self.english_tokenizer = AutoTokenizer.from_pretrained(
            self.english_model.hparams.model_name
        )

        logger.info(f"StyleRewardModel initialized on {self.device}")

    def _get_device(self, device: str) -> str:
        """Determine device to use."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self, checkpoint_path: str) -> StyleDetector:
        """
        Load style detector model from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint

        Returns:
            Loaded StyleDetector model

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            RuntimeError: If model loading fails
        """
        try:
            model = StyleDetector.load_from_checkpoint(checkpoint_path)
            model.eval()
            model.to(self.device)
            return model
        except FileNotFoundError:
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {checkpoint_path}: {e}")

    def predict_style_probabilities(self, text: str, language: str) -> torch.Tensor:
        """
        Predict style probability distribution for text.

        Args:
            text: Input text
            language: Language identifier ('chinese' or 'english')

        Returns:
            Style probability distribution tensor

        Raises:
            ValueError: If language is not supported
        """
        if language == 'chinese':
            tokenizer = self.chinese_tokenizer
            model = self.chinese_model
        elif language == 'english':
            tokenizer = self.english_tokenizer
            model = self.english_model
        else:
            raise ValueError(
                f"Unsupported language: {language}. "
                f"Supported: 'chinese', 'english'"
            )

        # Encode text
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Model inference
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=-1)
            # Only squeeze the batch dimension (dim=0) to maintain consistent shape
            if probabilities.dim() > 1:
                probabilities = probabilities.squeeze(0)

        return probabilities

    def calculate(self, **kwargs) -> RewardResult:
        """
        Calculate style reward (implements BaseReward interface).

        Args:
            **kwargs: Must contain 'source_text', 'target_text', 'source_lang', 'target_lang'

        Returns:
            RewardResult with similarity score and details

        Raises:
            KeyError: If required parameters are missing
        """
        try:
            source_text = kwargs['source_text']
            target_text = kwargs['target_text']
            source_lang = kwargs['source_lang']
            target_lang = kwargs['target_lang']
        except KeyError as e:
            raise KeyError(f"Missing required parameter for StyleRewardModel.calculate: {e}")

        result = self.calculate_similarity(
            source_text, target_text, source_lang, target_lang
        )
        return RewardResult(score=result['similarity_score'], details=result)

    def batch_calculate(self, batch_data: List[Dict[str, Any]]) -> List[RewardResult]:
        """
        Calculate rewards for a batch (implements BaseReward interface).

        Args:
            batch_data: List of dicts with 'source_text', 'target_text',
                       'source_lang', 'target_lang' keys

        Returns:
            List of RewardResult objects
        """
        results = []
        for item in batch_data:
            try:
                result = self.calculate(
                    source_text=item['source_text'],
                    target_text=item['target_text'],
                    source_lang=item['source_lang'],
                    target_lang=item['target_lang']
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error calculating style reward: {e}")
                results.append(RewardResult(score=0.0, details={'error': str(e)}))
        return results

    def calculate_similarity(
        self,
        source_text: str,
        target_text: str,
        source_lang: str,
        target_lang: str
    ) -> Dict[str, Any]:
        """
        Alias for calculate_style_similarity (to match base class interface).
        """
        return self.calculate_style_similarity(
            source_text, target_text, source_lang, target_lang
        )

    def calculate_style_similarity(
        self,
        source_text: str,
        target_text: str,
        source_lang: str,
        target_lang: str
    ) -> Dict[str, Any]:
        """
        Calculate style similarity between source and target text.

        Args:
            source_text: Source text
            target_text: Target text
            source_lang: Source language
            target_lang: Target language

        Returns:
            Dictionary with similarity score and style details
        """
        # Get style probability distributions
        source_style_probs = self.predict_style_probabilities(
            source_text, source_lang
        )
        target_style_probs = self.predict_style_probabilities(
            target_text, target_lang
        )

        # Calculate cosine similarity
        cosine_similarity = F.cosine_similarity(
            source_style_probs, target_style_probs, dim=-1
        ).item()

        # Get dominant style types
        source_main_style_idx = torch.argmax(source_style_probs, dim=-1).item()
        target_main_style_idx = torch.argmax(target_style_probs, dim=-1).item()

        result = {
            'similarity_score': cosine_similarity,
            'source_style_probs': source_style_probs.cpu().numpy().tolist(),
            'target_style_probs': target_style_probs.cpu().numpy().tolist(),
            'source_main_style': self.style_types[source_main_style_idx],
            'target_main_style': self.style_types[target_main_style_idx],
            'style_match': source_main_style_idx == target_main_style_idx
        }

        return result

    def batch_calculate_similarity(
        self,
        source_texts: List[str],
        target_texts: List[str],
        source_lang: str,
        target_lang: str
    ) -> List[Dict[str, Any]]:
        """
        Calculate style similarity for batches.

        Args:
            source_texts: List of source texts
            target_texts: List of target texts
            source_lang: Source language
            target_lang: Target language

        Returns:
            List of similarity result dictionaries
        """
        results = []
        for source_text, target_text in zip(source_texts, target_texts):
            result = self.calculate_style_similarity(
                source_text, target_text, source_lang, target_lang
            )
            results.append(result)
        return results


class MockStyleRewardModel(StyleRewardBase):
    """Mock style reward model for testing without actual BERT models."""

    def __init__(self, style_types: List[str]):
        """
        Initialize mock style reward model.

        Args:
            style_types: List of style type labels
        """
        self.style_types = style_types
        self.num_styles = len(style_types)
        logger.info("MockStyleRewardModel initialized (test mode)")

    def predict_style_probabilities(self, text: str, language: str) -> torch.Tensor:
        """
        Generate random style probabilities for testing.

        Args:
            text: Input text
            language: Language identifier

        Returns:
            Random normalized probability distribution
        """
        random_probs = torch.rand(1, self.num_styles)
        style_probs = F.softmax(random_probs, dim=-1)
        # Squeeze batch dimension for consistency with real model
        return style_probs.squeeze(0)

    def calculate(self, **kwargs) -> RewardResult:
        """
        Calculate mock style reward.

        Args:
            **kwargs: Must contain 'source_text', 'target_text', 'source_lang', 'target_lang'

        Returns:
            RewardResult with similarity score and details

        Raises:
            KeyError: If required parameters are missing
        """
        try:
            source_text = kwargs['source_text']
            target_text = kwargs['target_text']
            source_lang = kwargs['source_lang']
            target_lang = kwargs['target_lang']
        except KeyError as e:
            raise KeyError(f"Missing required parameter for MockStyleRewardModel.calculate: {e}")

        result = self.calculate_similarity(
            source_text, target_text, source_lang, target_lang
        )
        return RewardResult(score=result['similarity_score'], details=result)

    def batch_calculate(self, batch_data: List[Dict[str, Any]]) -> List[RewardResult]:
        """Calculate mock rewards for batch."""
        results = []
        for item in batch_data:
            result = self.calculate(
                source_text=item['source_text'],
                target_text=item['target_text'],
                source_lang=item['source_lang'],
                target_lang=item['target_lang']
            )
            results.append(result)
        return results

    def calculate_similarity(
        self,
        source_text: str,
        target_text: str,
        source_lang: str,
        target_lang: str
    ) -> Dict[str, Any]:
        """Alias for calculate_style_similarity."""
        return self.calculate_style_similarity(
            source_text, target_text, source_lang, target_lang
        )

    def calculate_style_similarity(
        self,
        source_text: str,
        target_text: str,
        source_lang: str,
        target_lang: str
    ) -> Dict[str, Any]:
        """
        Generate mock style similarity.

        Args:
            source_text: Source text
            target_text: Target text
            source_lang: Source language
            target_lang: Target language

        Returns:
            Mock similarity result
        """
        # Generate random style probabilities
        source_style_probs = self.predict_style_probabilities(source_text, source_lang)
        target_style_probs = self.predict_style_probabilities(target_text, target_lang)

        # Random cosine similarity (0.6-1.0 range)
        cosine_similarity = torch.rand(1).item() * 0.4 + 0.6

        # Get dominant styles
        source_main_style_idx = torch.argmax(source_style_probs, dim=-1).item()
        target_main_style_idx = torch.argmax(target_style_probs, dim=-1).item()

        result = {
            'similarity_score': cosine_similarity,
            'source_style_probs': source_style_probs.cpu().numpy().tolist() if source_style_probs.dim() == 1 else source_style_probs.squeeze(0).cpu().numpy().tolist(),
            'target_style_probs': target_style_probs.cpu().numpy().tolist() if target_style_probs.dim() == 1 else target_style_probs.squeeze(0).cpu().numpy().tolist(),
            'source_main_style': self.style_types[source_main_style_idx],
            'target_main_style': self.style_types[target_main_style_idx],
            'style_match': source_main_style_idx == target_main_style_idx
        }

        return result

    def batch_calculate_similarity(
        self,
        source_texts: List[str],
        target_texts: List[str],
        source_lang: str,
        target_lang: str
    ) -> List[Dict[str, Any]]:
        """Calculate mock style similarity for batches."""
        results = []
        for source_text, target_text in zip(source_texts, target_texts):
            result = self.calculate_style_similarity(
                source_text, target_text, source_lang, target_lang
            )
            results.append(result)
        return results
