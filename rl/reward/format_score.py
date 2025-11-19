"""
格式奖励模块：检查LLM生成的结果是否符合prompt要求
主要检查是否包含<think>和<translate>标签，以及内容格式
"""

import re
from typing import Dict, Any, Tuple


class FormatReward:
    """格式奖励计算器"""

    def __init__(self):
        # 定义正则表达式模式
        self.think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
        self.translate_pattern = re.compile(r'<translate>(.*?)</translate>', re.DOTALL)

    def calculate_reward(self, generated_text: str, prompt: str) -> Dict[str, Any]:
        """
        计算格式奖励
        
        Args:
            generated_text: 模型生成的文本
            prompt: 输入的prompt
            
        Returns:
            Dict包含奖励分数和详细信息
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
            # 检查是否包含think标签
            think_match = self.think_pattern.search(generated_text)
            if think_match:
                reward_info['has_think_tag'] = True
                reward_info['think_content'] = think_match.group(1).strip()

            # 检查是否包含translate标签
            translate_match = self.translate_pattern.search(generated_text)
            if translate_match:
                reward_info['has_translate_tag'] = True
                reward_info['translate_content'] = translate_match.group(1).strip()

            # 计算基础奖励
            base_reward = 0.0
            if reward_info['has_think_tag']:
                base_reward += 0.3
            if reward_info['has_translate_tag']:
                base_reward += 0.4

            # 检查内容有效性
            if reward_info['has_translate_tag'] and reward_info['translate_content']:
                # 翻译内容不能太短（至少2个字符）
                if len(reward_info['translate_content']) >= 2:
                    base_reward += 0.3
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
        从生成的文本中提取翻译内容
        
        Args:
            generated_text: 模型生成的文本
            
        Returns:
            提取的翻译内容
        """
        translate_match = self.translate_pattern.search(generated_text)
        if translate_match:
            return translate_match.group(1).strip()
        return ""

    def batch_calculate_reward(self, generated_texts: list, prompts: list) -> list:
        """
        批量计算格式奖励
        
        Args:
            generated_texts: 模型生成的文本列表
            prompts: 输入的prompt列表
            
        Returns:
            奖励信息列表
        """
        rewards = []
        for text, prompt in zip(generated_texts, prompts):
            reward = self.calculate_reward(text, prompt)
            rewards.append(reward)
        return rewards
