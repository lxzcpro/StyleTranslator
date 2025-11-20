"""
奖励管理器：整合格式奖励和风格奖励
支持测试模式和正式模式的切换
"""
import logging
from typing import Dict, List, Any, Tuple, Optional
import torch
import numpy as np
from .format_score import FormatReward
from .style_score import StyleRewardModel, MockStyleRewardModel
from .comet_score import CometSemanticReward

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


class RewardManager:
    """奖励管理器，整合多种奖励机制"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化奖励管理器
        
        Args:
            config: 配置字典（从config.yaml加载）
        """
        self.config = config
        self.format_reward = FormatReward()

        # 初始化语义奖励模型
        self.semantic_reward = CometSemanticReward(
            model_name=config['reward'].get('comet_model', 'wmt22-cometkiwi-da'),
            model_path=config['reward'].get('comet_path', None),
            device=config['reward'].get('comet_device', 'cpu')
        )

        # 根据测试模式选择风格奖励模型
        if config['reward']['test_mode']:
            self.style_reward = MockStyleRewardModel(
                style_types=config['reward']['style_types']
            )
            logger.info("使用MockStyleRewardModel（测试模式）")
        else:
            # 使用从checkpoint加载的StyleRewardModel
            chinese_bert_path = config['reward']['chinese_bert_path']
            english_bert_path = config['reward']['english_bert_path']
            style_types = config['reward']['style_types']
            device = config['model']['device']

            logger.info(f"从checkpoint加载StyleRewardModel，中文模型路径: {chinese_bert_path}")
            logger.info(f"从checkpoint加载StyleRewardModel，英文模型路径: {english_bert_path}")

            self.style_reward = StyleRewardModel(
                chinese_bert_path=chinese_bert_path,
                english_bert_path=english_bert_path,
                style_types=style_types,
                device=device
            )

        # 奖励权重（从配置文件加载）
        self.format_weight = config['reward']['format_reward_weight']
        self.semantic_weight = config['reward'].get('semantic_reward_weight', 0.7)
        self.style_weight = config['reward']['style_reward_weight']

        logger.info(
            f"奖励权重配置 - 格式: {self.format_weight}, 语义: {self.semantic_weight}, 风格: {self.style_weight}")

        # 缓存机制（用于缓存源文本的向量表示）
        self.source_embedding_cache = {}

    def calculate_total_reward(self, generated_texts: List[str],
                               source_texts: List[str],
                               prompts: List[str],
                               language_pairs: List[str],
                               reference_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        计算总奖励（包含格式、语义、风格三种奖励）
        
        Args:
            generated_texts: 生成的文本列表
            source_texts: 源文本列表
            prompts: 提示文本列表
            language_pairs: 语言对列表（如 ['en-zh', 'de-en']）
            
        Returns:
            奖励信息字典
        """
        batch_size = len(generated_texts)

        # 1. 计算格式奖励
        format_rewards = self.format_reward.batch_calculate_reward(
            generated_texts, prompts)

        # 提取翻译内容用于语义奖励计算
        translations = []
        for text in generated_texts:
            translation = self.format_reward.extract_translation(text)
            translations.append(translation)

        # 2. 计算语义奖励（使用COMET模型）
        if reference_texts:
            # 如果有参考译文，使用参考译文进行COMET计算
            semantic_rewards = self.semantic_reward.calculate_semantic_reward(
                source_texts, reference_texts, translations)
            logger.info(f"使用参考译文进行COMET语义奖励计算，参考文本数量: {len(reference_texts)}")
        else:
            # 如果没有参考译文，使用源文本作为参考（降级方案）
            semantic_rewards = self.semantic_reward.calculate_semantic_reward(
                source_texts, source_texts, translations)
            logger.warning("没有提供参考译文，使用源文本作为参考进行COMET计算")

        # 3. 计算风格奖励
        style_rewards = self._calculate_style_rewards(
            generated_texts, source_texts, language_pairs)

        # 4. 计算总奖励
        total_rewards = []
        reward_details = []

        for i in range(batch_size):
            format_score = format_rewards[i]['total_reward']
            semantic_score = semantic_rewards[i]
            style_score = style_rewards[i]['style_score']

            # 加权求和
            if format_score <= 0.3:
                total_score = -1
                semantic_score = 0
                style_score = 0
            else:
                total_score = (self.format_weight * format_score +
                               self.semantic_weight * semantic_score +
                               self.style_weight * style_score)

            total_rewards.append(total_score)

            # 详细奖励信息
            reward_detail = {
                'format_reward': format_score,
                'semantic_reward': semantic_score,
                'style_reward': style_score,
                'total_reward': total_score,
                'format_details': format_rewards[i],
                'style_details': style_rewards[i]
            }
            reward_details.append(reward_detail)

        return {
            'total_rewards': total_rewards,
            'reward_details': reward_details,
            'format_rewards': format_rewards,
            'semantic_rewards': semantic_rewards,
            'style_rewards': style_rewards
        }

    def _calculate_style_rewards(self, generated_texts: List[str],
                                 source_texts: List[str],
                                 language_pairs: List[str]) -> List[Dict[str, Any]]:
        """
        计算风格奖励
        
        Args:
            generated_texts: 生成的文本列表
            source_texts: 源文本列表
            language_pairs: 语言对列表
            
        Returns:
            风格奖励信息列表
        """
        style_rewards = []

        for i, (generated_text, source_text, lang_pair) in enumerate(
                zip(generated_texts, source_texts, language_pairs)):

            # 解析语言对
            source_lang, target_lang = lang_pair.split('-')

            # 映射语言到模型语言类型
            lang_map = {
                'en': 'english',
                'zh': 'chinese',
                'de': 'english',  # 德语用英文模型处理
                'ja': 'chinese'  # 日语用中文模型处理
            }

            source_model_lang = lang_map.get(source_lang, 'english')
            target_model_lang = lang_map.get(target_lang, 'chinese')

            # 提取翻译内容

            translation = self.format_reward.extract_translation(generated_text)
            if not translation:
                # 如果没有提取到翻译内容，给予惩罚
                style_rewards.append({
                    'style_score': 0.0,
                    'similarity_score': 0.0,
                    'error': 'No translation extracted',
                    'source_main_style': 'unknown',
                    'target_main_style': 'unknown',
                    'style_match': False
                })
                continue

            try:
                # 计算风格相似度
                style_result = self.style_reward.calculate_style_similarity(
                    source_text, translation, source_model_lang, target_model_lang)

                # 风格奖励基于相似度得分
                style_score = style_result['similarity_score']

                style_rewards.append({
                    'style_score': style_score,
                    'similarity_score': style_result['similarity_score'],
                    'source_main_style': style_result['source_main_style'],
                    'target_main_style': style_result['target_main_style'],
                    'style_match': style_result['style_match'],
                    'style_details': style_result
                })

            except Exception as e:
                # 处理错误情况
                style_rewards.append({
                    'style_score': 0.0,
                    'similarity_score': 0.0,
                    'error': str(e),
                    'source_main_style': 'unknown',
                    'target_main_style': 'unknown',
                    'style_match': False
                })

        return style_rewards

    def get_reward_statistics(self, reward_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算奖励统计信息
        
        Args:
            reward_history: 历史奖励列表
            
        Returns:
            统计信息
        """
        if not reward_history:
            return {}

        all_rewards = []
        format_rewards = []
        style_rewards = []

        for reward_info in reward_history:
            for detail in reward_info['reward_details']:
                all_rewards.append(detail['total_reward'])
                format_rewards.append(detail['format_reward'])
                style_rewards.append(detail['style_reward'])

        stats = {
            'total_reward': {
                'mean': np.mean(all_rewards),
                'std': np.std(all_rewards),
                'min': np.min(all_rewards),
                'max': np.max(all_rewards)
            },
            'format_reward': {
                'mean': np.mean(format_rewards),
                'std': np.std(format_rewards)
            },
            'style_reward': {
                'mean': np.mean(style_rewards),
                'std': np.std(style_rewards)
            }
        }

        return stats

    def clear_cache(self):
        """清除缓存"""
        self.source_embedding_cache.clear()
