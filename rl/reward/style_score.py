"""
风格奖励模块：计算翻译文本的风格一致性
包括中文BERT和英文BERT模型的风格预测
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import Dict, List, Any, Tuple
import numpy as np
from style_detector.model.model import StyleDetector


class StyleRewardModel:
    """风格奖励模型"""

    def __init__(self, chinese_bert_path: str, english_bert_path: str,
                 style_types: List[str], device: str = "auto"):
        """
        初始化风格奖励模型
        
        Args:
            chinese_bert_path: 中文BERT模型路径
            english_bert_path: 英文BERT模型路径
            style_types: 风格类型列表
            device: 计算设备
        """
        self.device = self._get_device(device)
        self.style_types = style_types
        self.num_styles = len(style_types)

        # 加载中文BERT模型和分词器
        self.chinese_model = self._load_model_from_checkpoint(chinese_bert_path)
        self.chinese_tokenizer = AutoTokenizer.from_pretrained(self.chinese_model.hparams.model_name)

        # 加载英文BERT模型和分词器
        self.english_model = self._load_model_from_checkpoint(english_bert_path)
        self.english_tokenizer = AutoTokenizer.from_pretrained(self.english_model.hparams.model_name)

    def _get_device(self, device: str) -> str:
        """获取计算设备"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
        
    def _load_model_from_checkpoint(self, checkpoint_path: str) -> StyleDetector:
        """
        从checkpoint加载风格检测器模型
        
        Args:
            checkpoint_path: 模型checkpoint路径
            
        Returns:
            加载好的StyleDetector模型
        """
        # 使用StyleDetector的load_from_checkpoint方法加载模型 - 与bert_test2.py保持一致
        model = StyleDetector.load_from_checkpoint(checkpoint_path)
        model.eval()
        model.to(self.device)
        return model

    def predict_style_probabilities(self, text: str, language: str) -> torch.Tensor:
        """
        预测文本在各种风格上的概率分布
        
        Args:
            text: 输入文本
            language: 语言类型
            
        Returns:
            风格概率分布
        """
        if language == 'chinese':
            tokenizer = self.chinese_tokenizer
            model = self.chinese_model
        elif language == 'english':
            tokenizer = self.english_tokenizer
            model = self.english_model
        else:
            raise ValueError(f"Unsupported language: {language}")

        # 编码文本
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # 模型推理 - 与bert_test2.py保持一致
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            # probabilities = torch.softmax(logits, dim=-1).squeeze()

        return logits

    def calculate_style_similarity(self, source_text: str, target_text: str,
                                   source_lang: str, target_lang: str) -> Dict[str, Any]:
        """
        计算源文本和目标文本的风格相似度
        
        Args:
            source_text: 源文本
            target_text: 目标文本
            source_lang: 源语言
            target_lang: 目标语言
            
        Returns:
            风格相似度信息
        """
        # 获取风格概率分布
        source_style_probs = self.predict_style_probabilities(
            source_text, source_lang)
        target_style_probs = self.predict_style_probabilities(
            target_text, target_lang)

        # 计算余弦相似度
        cosine_similarity = F.cosine_similarity(
            source_style_probs, target_style_probs, dim=-1).item()

        # 获取主要风格类型
        source_main_style_idx = torch.argmax(source_style_probs, dim=-1).item()
        target_main_style_idx = torch.argmax(target_style_probs, dim=-1).item()

        # # 计算主要风格概率之差的绝对值
        # source_main_prob = source_style_probs[source_main_style_idx].item()
        # target_main_prob = target_style_probs[target_main_style_idx].item()
        # prob_diff_abs = max(source_main_prob - target_main_prob,0)
        #
        # # 添加调节系数：1-（主要风格概率差的绝对值）
        # # 当主要风格概率差异较大时，系数较小，增强区分度
        # adjustment_coefficient = 1 - prob_diff_abs
        #
        # # 应用调节系数
        # adjusted_similarity = cosine_similarity * adjustment_coefficient

        result = {
            'similarity_score': cosine_similarity,
            'source_style_probs': source_style_probs.squeeze().cpu().numpy().tolist(),
            'target_style_probs': target_style_probs.squeeze().cpu().numpy().tolist(),
            'source_main_style': self.style_types[source_main_style_idx],
            'target_main_style': self.style_types[target_main_style_idx],
            'style_match': source_main_style_idx == target_main_style_idx
        }

        return result

    def batch_calculate_similarity(self, source_texts: List[str],
                                   target_texts: List[str],
                                   source_lang: str, target_lang: str) -> List[Dict[str, Any]]:
        """
        批量计算风格相似度
        
        Args:
            source_texts: 源文本列表
            target_texts: 目标文本列表
            source_lang: 源语言
            target_lang: 目标语言
            
        Returns:
            相似度信息列表
        """
        results = []
        for source_text, target_text in zip(source_texts, target_texts):
            result = self.calculate_style_similarity(
                source_text, target_text, source_lang, target_lang)
            results.append(result)
        return results


class MockStyleRewardModel:
    """模拟风格奖励模型（用于测试）"""

    def __init__(self, style_types: List[str]):
        """
        初始化模拟风格奖励模型
        
        Args:
            style_types: 风格类型列表
        """
        self.style_types = style_types
        self.num_styles = len(style_types)

    def predict_style_probabilities(self, text: str, language: str) -> torch.Tensor:
        """
        模拟风格概率预测（随机生成）
        
        Args:
            text: 输入文本
            language: 语言类型
            
        Returns:
            随机生成的风格概率分布
        """
        # 随机生成风格概率分布
        random_probs = torch.rand(1, self.num_styles)
        # 归一化使其成为有效的概率分布
        style_probs = F.softmax(random_probs, dim=-1)
        return style_probs

    def calculate_style_similarity(self, source_text: str, target_text: str,
                                   source_lang: str, target_lang: str) -> Dict[str, Any]:
        """
        模拟风格相似度计算
        
        Args:
            source_text: 源文本
            target_text: 目标文本
            source_lang: 源语言
            target_lang: 目标语言
            
        Returns:
            模拟的相似度信息
        """
        # 随机生成风格概率分布
        source_style_probs = self.predict_style_probabilities(source_text, source_lang)
        target_style_probs = self.predict_style_probabilities(target_text, target_lang)

        # 计算余弦相似度（随机值）
        cosine_similarity = (torch.rand(1).item() * 0.4 + 0.6)  # 0.6-1.0之间的随机值

        # 获取主要风格类型
        source_main_style_idx = torch.argmax(source_style_probs, dim=-1).item()
        target_main_style_idx = torch.argmax(target_style_probs, dim=-1).item()

        result = {
            'similarity_score': cosine_similarity,
            'source_style_probs': source_style_probs.squeeze().cpu().numpy().tolist(),
            'target_style_probs': target_style_probs.squeeze().cpu().numpy().tolist(),
            'source_main_style': self.style_types[source_main_style_idx],
            'target_main_style': self.style_types[target_main_style_idx],
            'style_match': source_main_style_idx == target_main_style_idx
        }

        return result

    def batch_calculate_similarity(self, source_texts: List[str],
                                   target_texts: List[str],
                                   source_lang: str, target_lang: str) -> List[Dict[str, Any]]:
        """
        批量计算风格相似度（模拟）
        
        Args:
            source_texts: 源文本列表
            target_texts: 目标文本列表
            source_lang: 源语言
            target_lang: 目标语言
            
        Returns:
            相似度信息列表
        """
        results = []
        for source_text, target_text in zip(source_texts, target_texts):
            result = self.calculate_style_similarity(
                source_text, target_text, source_lang, target_lang)
            results.append(result)
        return results
