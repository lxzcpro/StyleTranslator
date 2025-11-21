import torch
import logging
from typing import List, Dict, Optional, Any
from comet import download_model, load_from_checkpoint
from .base import SemanticRewardBase, RewardResult

logger = logging.getLogger(__name__)


class CometSemanticReward(SemanticRewardBase):
    """
    Semantic reward calculator based on the COMET model.
    Uses the wmt22-cometkiwi-da model to evaluate translation quality.
    """

    def __init__(self, model_name: str = "wmt22-cometkiwi-da",
                 model_path: str = None,
                 device: str = None):
        """
        Initialize the COMET semantic reward model.

        Args:
            model_name: COMET model name, default is wmt22-cometkiwi-da.
            device: Device to run on, None for automatic selection.
        """
        # Automatic device selection
        if device is None:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"COMET model will use GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = "cpu"
                logger.info("COMET model will use CPU")
        else:
            self.device = device

        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the COMET model."""
        try:
            logger.info(f"Loading COMET model: {self.model_name}")
            if not self.model_path or str(self.model_path).lower() == "none":
                logger.info("No COMET model path provided, downloading model")
                self.model_path = download_model(self.model_name)
            else:
                logger.info(f"Using specified COMET model path: {self.model_path}")
            
            self.model = load_from_checkpoint(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"COMET model loaded successfully: {self.model_name}")
        except FileNotFoundError as e:
            logger.error(f"COMET model file not found: {e}")
            logger.warning("Will use simulated semantic reward scores")
            self.model = None
        except (RuntimeError, OSError) as e:
            # Catch model loading errors but not critical errors
            logger.error(f"Failed to load COMET model: {e}")
            logger.warning("Will use simulated semantic reward scores")
            self.model = None
        except KeyboardInterrupt:
            # Re-raise user interrupts
            raise
        except Exception as e:
            # Log unexpected errors with full traceback but don't crash
            logger.error(f"Unexpected error while loading COMET model: {e}", exc_info=True)
            logger.warning("Will use simulated semantic reward scores")
            self.model = None

    def calculate(self, **kwargs) -> RewardResult:
        """
        Calculate semantic reward for single input (implements BaseReward interface).

        Args:
            **kwargs: Must contain 'source', 'reference', 'hypothesis'

        Returns:
            RewardResult with score and details

        Raises:
            KeyError: If required parameters are missing
        """
        try:
            source = kwargs['source']
            reference = kwargs['reference']
            hypothesis = kwargs['hypothesis']
        except KeyError as e:
            raise KeyError(f"Missing required parameter for CometSemanticReward.calculate: {e}")

        scores = self.calculate_semantic_reward([source], [reference], [hypothesis])
        return RewardResult(
            score=scores[0] if scores else 0.0,
            details={'source': source, 'reference': reference, 'hypothesis': hypothesis}
        )

    def batch_calculate(self, batch_data: List[Dict[str, Any]]) -> List[RewardResult]:
        """
        Calculate rewards for a batch (implements BaseReward interface).

        Args:
            batch_data: List of dicts with 'source', 'reference', 'hypothesis' keys

        Returns:
            List of RewardResult objects
        """
        sources = [item['source'] for item in batch_data]
        references = [item['reference'] for item in batch_data]
        hypotheses = [item['hypothesis'] for item in batch_data]

        scores = self.calculate_semantic_reward(sources, references, hypotheses)

        return [
            RewardResult(score=score, details=batch_data[i])
            for i, score in enumerate(scores)
        ]

    def calculate_semantic_reward(self, source_texts: List[str], reference_texts: List[str],
                                  hypothesis_texts: List[str]) -> List[float]:
        """
        Calculate semantic reward scores.

        Args:
            source_texts: List of source texts
            reference_texts: List of reference translations
            hypothesis_texts: List of actual translations (model generated)

        Returns:
            List of semantic reward scores, usually between 0 and 1.
        """
        if not source_texts or not reference_texts or not hypothesis_texts:
            logger.warning("Empty input lists provided to calculate_semantic_reward")
            return []

        if not self.model:
            # If model failed to load, return simulated scores
            logger.warning("COMET model not loaded, using simulated semantic reward scores")
            return [0.5 + 0.3 * (hash(h) % 1000) / 1000.0 for h in hypothesis_texts]

        try:
            # Prepare COMET input data
            data = []
            for src, ref, hyp in zip(source_texts, reference_texts, hypothesis_texts):
                data.append({
                    "src": src,
                    "ref": ref,
                    "mt": hyp
                })

            # Log debug info
            logger.info(
                f"COMET calculation: source_count={len(source_texts)}, ref_count={len(reference_texts)}, hyp_count={len(hypothesis_texts)}")
            if source_texts and reference_texts and hypothesis_texts:
                logger.info(f"Example - Source: '{source_texts[0][:50]}...'")
                logger.info(f"Example - Reference: '{reference_texts[0][:50]}...'")
                logger.info(f"Example - Hypothesis: '{hypothesis_texts[0][:50]}...'")

            # Use COMET model to predict quality scores
            with torch.no_grad():
                gpus = 0 if self.device == "cpu" else 1
                scores = self.model.predict(data, batch_size=8, gpus=gpus)

            # COMET scores are usually between 0-1, can be used directly as reward
            semantic_rewards = scores["scores"]

            if semantic_rewards:
                avg_score = sum(semantic_rewards) / len(semantic_rewards)
                logger.info(f"Semantic reward calculation complete, average score: {avg_score:.3f}")
                logger.info(f"Semantic reward score details: {semantic_rewards}")
            else:
                logger.warning("Semantic reward calculation complete but result is empty")
            return semantic_rewards

        except Exception as e:
            logger.error(f"Error calculating semantic reward: {e}")
            logger.error(f"Error details: {str(e)}")
            # Return medium quality simulated scores
            return [0.5] * len(source_texts)

    def calculate_batch_reward(self, batch_data: List[Dict]) -> List[float]:
        """
        Batch calculate semantic rewards.

        Args:
            batch_data: List of dictionaries containing source, reference, hypothesis

        Returns:
            List of semantic reward scores
        """
        source_texts = [item['source'] for item in batch_data]
        reference_texts = [item['reference'] for item in batch_data]
        hypothesis_texts = [item['hypothesis'] for item in batch_data]

        return self.calculate_semantic_reward(source_texts, reference_texts, hypothesis_texts)

    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "loaded": self.model is not None
        }