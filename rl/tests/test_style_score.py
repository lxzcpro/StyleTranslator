#!/usr/bin/env python3
"""
风格评分测试脚本
测试StyleRewardModel的风格分类和相似度计算功能
"""

import json
import logging
import numpy as np
import torch
from typing import Dict, List, Any
import sys
import os
import yaml

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reward.style_score import StyleRewardModel, MockStyleRewardModel

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_test_data(file_path: str) -> List[Dict[str, Any]]:
    """加载测试数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['data']
    except Exception as e:
        logger.error(f"加载测试数据失败: {e}")
        return []

def load_config(config_path: str) -> Dict[str, Any]:
    """从YAML文件加载配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        # 返回默认配置
        return {
            'model': {'device': 'cpu'},
            'reward': {
                'test_mode': True,
                'style_types': ['law', 'science', 'news', 'literature'],
                'chinese_bert_path': 'mock_path',
                'english_bert_path': 'mock_path'
            }
        }

def initialize_style_model(config: Dict[str, Any]) -> Any:
    """初始化风格模型"""
    if config['reward']['test_mode']:
        logger.info("使用模拟风格奖励模型")
        return MockStyleRewardModel(config['reward']['style_types'])
    else:
        logger.info("使用真实风格奖励模型")
        logger.info(f"中文BERT模型路径: {config['reward']['chinese_bert_path']}")
        logger.info(f"英文BERT模型路径: {config['reward']['english_bert_path']}")
        return StyleRewardModel(
            chinese_bert_path=config['reward']['chinese_bert_path'],
            english_bert_path=config['reward']['english_bert_path'],
            style_types=config['reward']['style_types'],
            device=config['model']['device']
        )

def calculate_style_matrix(style_model: Any, test_data: List[Dict[str, Any]], 
                          style_types: List[str]) -> List[Dict[str, Any]]:
    """计算风格矩阵并返回结果"""
    results = []
    
    for i, test_item in enumerate(test_data):
        original_en = test_item['en']
        reference_zh = test_item['zh']
        translations = test_item['translations']
        
        logger.info(f"\n处理测试项 {i+1}: {original_en[:50]}...")
        
        item_results = {
            'original_en': original_en,
            'reference_zh': reference_zh,
            'style_analysis': {}
        }
        
        # 计算原文风格
        en_style_probs = style_model.predict_style_probabilities(original_en, 'english')
        en_style_probs_np = en_style_probs.squeeze().cpu().numpy().tolist()
        en_main_style_idx = torch.argmax(en_style_probs, dim=-1).item()
        
        item_results['style_analysis']['source'] = {
            'text': original_en,
            'language': 'english',
            'style_probs': {style_types[j]: prob for j, prob in enumerate(en_style_probs_np)},
            'main_style': style_types[en_main_style_idx]
        }
        
        # 计算参考译文风格
        zh_style_probs = style_model.predict_style_probabilities(reference_zh, 'chinese')
        zh_style_probs_np = zh_style_probs.squeeze().cpu().numpy().tolist()
        zh_main_style_idx = torch.argmax(zh_style_probs, dim=-1).item()
        
        item_results['style_analysis']['reference'] = {
            'text': reference_zh,
            'language': 'chinese',
            'style_probs': {style_types[j]: prob for j, prob in enumerate(zh_style_probs_np)},
            'main_style': style_types[zh_main_style_idx]
        }
        
        # 计算每个翻译结果的风格相似度
        item_results['style_analysis']['translations'] = {}
        
        for trans_type, trans_text in translations.items():
            # 计算风格相似度
            similarity_result = style_model.calculate_style_similarity(
                original_en, trans_text, 'english', 'chinese')
            
            item_results['style_analysis']['translations'][trans_type] = {
                'text': trans_text,
                'similarity_score': similarity_result['similarity_score'],
                'style_match': similarity_result['style_match'],
                'source_main_style': similarity_result['source_main_style'],
                'target_main_style': similarity_result['target_main_style'],
                'source_style_probs': {style_types[j]: prob for j, prob in enumerate(similarity_result['source_style_probs'])},
                'target_style_probs': {style_types[j]: prob for j, prob in enumerate(similarity_result['target_style_probs'])}
            }
            
            logger.info(f"  - {trans_type}: 相似度 = {similarity_result['similarity_score']:.4f}, "
                      f"源风格: {similarity_result['source_main_style']}, "
                      f"目标风格: {similarity_result['target_main_style']}")
        
        results.append(item_results)
    
    return results

def print_style_matrix(results: List[Dict[str, Any]], style_types: List[str]):
    """打印风格矩阵"""
    print("\n" + "="*80)
    print("风格分析矩阵")
    print("="*80)
    
    for i, result in enumerate(results):
        print(f"\n测试项 {i+1}:")
        print(f"原文 (英文): {result['original_en']}")
        print(f"参考译文 (中文): {result['reference_zh']}")
        
        # 打印源文本风格概率分布
        print(f"\n原文风格分布:")
        en_probs = result['style_analysis']['source']['style_probs']
        for style in style_types:
            print(f"  {style}: {en_probs[style]:.4f}")
        print(f"  主要风格: {result['style_analysis']['source']['main_style']}")
        
        # 打印翻译结果风格分析
        print(f"\n翻译结果风格分析:")
        for trans_type, trans_info in result['style_analysis']['translations'].items():
            print(f"\n  {trans_type}:")
            print(f"  文本: {trans_info['text']}")
            print(f"  相似度: {trans_info['similarity_score']:.4f}")
            print(f"  风格匹配: {'✓' if trans_info['style_match'] else '✗'}")
            print(f"  源风格: {trans_info['source_main_style']}, 目标风格: {trans_info['target_main_style']}")
            
            # 打印目标文本风格概率分布
            print("  目标风格分布:")
            zh_probs = trans_info['target_style_probs']
            for style in style_types:
                print(f"    {style}: {zh_probs[style]:.4f}")

def main():
    """主函数"""
    logger.info("开始风格评分测试")
    
    # 从config.yaml加载配置
    config = load_config("config.yaml")
    style_types = config['reward']['style_types']
    
    logger.info(f"风格类型: {', '.join(style_types)}")
    
    # 初始化风格模型
    style_model = initialize_style_model(config)
    
    # 加载测试数据
    test_data = load_test_data("enzh_fake_trans.json")
    
    if not test_data:
        logger.error("无法加载测试数据")
        return
    
    logger.info(f"成功加载 {len(test_data)} 条测试数据")
    
    # 计算风格矩阵
    results = calculate_style_matrix(style_model, test_data, style_types)
    
    # 打印风格矩阵
    print_style_matrix(results, style_types)
    
    logger.info("风格评分测试完成")

if __name__ == "__main__":
    main()