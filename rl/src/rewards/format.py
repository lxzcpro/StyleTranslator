"""
Format reward module for validating LLM-generated translation output.
Checks for required XML tags (<think>, <translate>) and content validity.
"""

import re
from typing import Dict, Any, List
from .base import FormatRewardBase, RewardResult


class FormatReward(FormatRewardBase):
    """Format validation reward calculator."""

    # Reward weights for different components
    THINK_TAG_WEIGHT = 0.3
    TRANSLATE_TAG_WEIGHT = 0.4
    CONTENT_WEIGHT = 0.3
    MIN_CONTENT_LENGTH = 2

    def __init__(self):
        """Initialize format reward with regex patterns."""
        self.think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
        self.translate_pattern = re.compile(r'<translate>(.*?)</translate>', re.DOTALL)

    def calculate(self, **kwargs) -> RewardResult:
        """
        Calculate format reward (implements BaseReward interface).

        Args:
            **kwargs: Must contain 'generated_text', optionally 'prompt'

        Returns:
            RewardResult with score and details
        """
        generated_text = kwargs.get('generated_text', '')
        prompt = kwargs.get('prompt', '')
        reward_info = self.calculate_reward(generated_text, prompt)
        return RewardResult(
            score=reward_info['total_reward'],
            details=reward_info
        )

    def batch_calculate(self, batch_data: List[Dict[str, Any]]) -> List[RewardResult]:
        """
        Calculate rewards for a batch (implements BaseReward interface).

        Args:
            batch_data: List of dicts with 'generated_text' and 'prompt' keys

        Returns:
            List of RewardResult objects
        """
        results = []
        for item in batch_data:
            result = self.calculate(
                generated_text=item.get('generated_text', ''),
                prompt=item.get('prompt', '')
            )
            results.append(result)
        return results

    def calculate_reward(self, generated_text: str, prompt: str) -> Dict[str, Any]:
        """
        Calculate format reward with detailed breakdown.

        Args:
            generated_text: Generated text from model
            prompt: Input prompt

        Returns:
            Dict with total_reward and detailed validation results
        """
        reward_info = {
            'total_reward': 0.0,
            'format_valid': False,
            'has_think_tag': False,
            'has_translate_tag': False,
            'think_content': '',
            'translate_content': '',
            'error_message': ''
        }

        try:
            # Check for think tag
            think_match = self.think_pattern.search(generated_text)
            if think_match:
                reward_info['has_think_tag'] = True
                reward_info['think_content'] = think_match.group(1).strip()

            # Check for translate tag
            translate_match = self.translate_pattern.search(generated_text)
            if translate_match:
                reward_info['has_translate_tag'] = True
                reward_info['translate_content'] = translate_match.group(1).strip()

            # Calculate base reward
            base_reward = 0.0
            if reward_info['has_think_tag']:
                base_reward += self.THINK_TAG_WEIGHT
            if reward_info['has_translate_tag']:
                base_reward += self.TRANSLATE_TAG_WEIGHT

            # Check content validity
            if reward_info['has_translate_tag'] and reward_info['translate_content']:
                if len(reward_info['translate_content']) >= self.MIN_CONTENT_LENGTH:
                    base_reward += self.CONTENT_WEIGHT
                    reward_info['format_valid'] = True
                else:
                    reward_info['error_message'] = 'Translation content too short'
            else:
                if not reward_info['has_translate_tag']:
                    reward_info['error_message'] = 'Missing translate tag'
                elif not reward_info['translate_content']:
                    reward_info['error_message'] = 'Empty translation content'

            reward_info['total_reward'] = base_reward

        except Exception as e:
            reward_info['error_message'] = f'Error in format checking: {str(e)}'
            reward_info['total_reward'] = 0.0

        return reward_info

    def extract_translation(self, generated_text: str) -> str:
        """
        Extract translation content from generated text.

        Args:
            generated_text: Generated text from model

        Returns:
            Extracted translation content or empty string
        """
        translate_match = self.translate_pattern.search(generated_text)
        if translate_match:
            return translate_match.group(1).strip()
        return ""

    def batch_calculate_reward(self, generated_texts: list, prompts: list) -> list:
        """
        Calculate format rewards for a batch (legacy interface).

        Args:
            generated_texts: List of generated texts
            prompts: List of input prompts

        Returns:
            List of reward info dictionaries
        """
        rewards = []
        for text, prompt in zip(generated_texts, prompts):
            reward = self.calculate_reward(text, prompt)
            rewards.append(reward)
        return rewards
