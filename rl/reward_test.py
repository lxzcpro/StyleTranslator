#!/usr/bin/env python3
"""
奖励体系测试脚本
测试完整的奖励函数：格式奖励、COMET语义奖励、风格奖励
"""

import json
import logging
import numpy as np
from typing import Dict, List, Any
import sys
import os
import yaml

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reward.reward_manager import RewardManager
from reward.format_score import FormatReward

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


def create_mock_responses(original_en: str, translations: Dict[str, str]) -> List[str]:
    """创建符合格式要求的模拟翻译结果"""
    mock_responses = []
    
    # 四种翻译结果，每种都添加格式标签
    for translation_type, translation_text in translations.items():
        # 添加思考过程（简单模拟）和翻译结果
        mock_response = f"<think>思考过程：将英文句子'{original_en[:30]}...'翻译成中文</think>\n"
        mock_response += f"<translate>{translation_text}"
        mock_responses.append(mock_response)
    
    return mock_responses


def print_reward_details(test_round: int, original_en: str, reference_zh: str, 
                        mock_responses: List[str], reward_results: Dict[str, Any]):
    """打印详细的奖励信息"""
    print(f"\n{'='*80}")
    print(f"第 {test_round + 1} 轮测试")
    print(f"{'='*80}")
    
    print(f"\n📄 原文 (英文):")
    print(f"   {original_en}")
    
    print(f"\n📄 参考译文 (中文):")
    print(f"   {reference_zh}")
    
    print(f"\n🎯 四条模拟翻译结果及奖励分数:")
    
    translation_types = ["correct_style_correct_meaning", 
                        "correct_style_wrong_meaning",
                        "wrong_style_correct_meaning", 
                        "wrong_style_wrong_meaning"]
    
    for i, (response, trans_type) in enumerate(zip(mock_responses, translation_types)):
        format_reward = reward_results['format_rewards'][i]
        semantic_reward = reward_results['semantic_rewards'][i]
        style_reward = reward_results['style_rewards'][i]
        total_reward = reward_results['total_rewards'][i]
        
        # 提取翻译内容
        format_reward_obj = FormatReward()
        translation_content = format_reward_obj.extract_translation(response)
        
        print(f"\n   翻译 {i+1} ({trans_type}):")
        print(f"   内容: {translation_content}")
        print(f"   格式奖励: {format_reward['total_reward']:.3f} (有效: {format_reward['format_valid']})")
        print(f"   语义奖励: {semantic_reward:.3f}")
        print(f"   风格奖励: {style_reward['style_score']:.3f}")
        print(f"   总奖励:   {total_reward:.3f}")
        
        if format_reward['error_message']:
            print(f"   格式错误: {format_reward['error_message']}")


def print_style_vectors(test_round: int, original_en: str, reference_zh: str, 
                       style_rewards: List[Dict[str, Any]], style_types: List[str]):
    """打印风格向量信息"""
    print(f"\n🎨 风格分析 (第 {test_round + 1} 轮):")
    print(f"原文: {original_en}")
    print(f"参考: {reference_zh}")
    
    # 使用配置文件中的风格类型列表
    style_types_str = ", ".join(style_types)
    print(f"\n风格向量 (1×4): [{style_types_str}]")
    
    translation_types = ["correct_style_correct_meaning", 
                        "correct_style_wrong_meaning",
                        "wrong_style_correct_meaning", 
                        "wrong_style_wrong_meaning"]
    
    for i, (style_reward, trans_type) in enumerate(zip(style_rewards, translation_types)):
        # 从风格奖励结果中提取信息
        similarity_score = style_reward['similarity_score']
        source_style = style_reward.get('source_main_style', 'unknown')
        target_style = style_reward.get('target_main_style', 'unknown')
        
        # 获取详细的风格概率分布（如果存在）
        source_probs = style_reward.get('style_details', {}).get('source_style_probs', [])
        target_probs = style_reward.get('style_details', {}).get('target_style_probs', [])
        
        print(f"翻译 {i+1} ({trans_type}):")
        print(f"  相似度: {similarity_score:.4f}")
        print(f"  源风格: {source_style}")
        print(f"  目标风格: {target_style}")
        print(f"  风格匹配: {'✓' if style_reward.get('style_match', False) else '✗'}")
        
        # 如果有风格概率分布，打印详细信息
        if source_probs and len(source_probs) == len(style_types):
            print("  源文本风格概率:")
            for style, prob in zip(style_types, source_probs):
                print(f"    - {style}: {prob:.4f}")
        
        if target_probs and len(target_probs) == len(style_types):
            print("  目标文本风格概率:")
            for style, prob in zip(style_types, target_probs):
                print(f"    - {style}: {prob:.4f}")


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
                'format_reward_weight': 0.2,
                'semantic_reward_weight': 0.7,
                'style_reward_weight': 0.1,
                'chinese_bert_path': 'mock_path',
                'english_bert_path': 'mock_path',
                'comet_model': 'wmt22-cometkiwi-da',
                'comet_device': 'cpu',
                'comet_path': None
            }
        }


def main():
    """主函数"""
    logger.info("开始奖励体系测试")
    
    # 从config.yaml加载配置
    config = load_config("config.yaml")
    
    # 检查是否使用测试模式
    if config['reward']['test_mode']:
        logger.info("当前为测试模式，将使用模拟风格奖励模型")
    else:
        logger.info("当前为正式模式，将使用真实BERT风格奖励模型")
        logger.info(f"中文BERT模型路径: {config['reward']['chinese_bert_path']}")
        logger.info(f"英文BERT模型路径: {config['reward']['english_bert_path']}")
    
    # 打印配置信息
    logger.info(f"奖励权重 - 格式: {config['reward']['format_reward_weight']}, "
               f"语义: {config['reward']['semantic_reward_weight']}, "
               f"风格: {config['reward']['style_reward_weight']}")
    logger.info(f"COMET模型: {config['reward']['comet_model']}, "
               f"设备: {config['reward']['comet_device']}")
    logger.info(f"风格类型: {', '.join(config['reward']['style_types'])}")
    
    # 初始化奖励管理器
    reward_manager = RewardManager(config)
    
    # 加载测试数据
    test_data = load_test_data("enzh_fake_trans.json")
    
    if not test_data:
        logger.error("无法加载测试数据")
        return
    
    logger.info(f"成功加载 {len(test_data)} 条测试数据")
    
    # 进行四轮测试
    for round_idx, test_item in enumerate(test_data):
        if round_idx >= 4:  # 只测试前4轮
            break
            
        original_en = test_item['en']
        reference_zh = test_item['zh']
        translations = test_item['translations']
        
        # 创建模拟响应
        mock_responses = create_mock_responses(original_en, translations)
        
        # 创建提示（模拟翻译任务）
        prompts = [f"请将以下英文翻译成中文：{original_en}"] * 4
        
        # 设置语言对
        language_pairs = ['en-zh'] * 4
        
        logger.info(f"\n开始第 {round_idx + 1} 轮测试...")
        
        try:
            # 计算奖励
            reward_results = reward_manager.calculate_total_reward(
                generated_texts=mock_responses,
                source_texts=[original_en] * 4,
                prompts=prompts,
                language_pairs=language_pairs,
                reference_texts=[reference_zh] * 4  # 使用参考译文进行COMET计算
            )
            
            # 打印详细奖励信息
            print_reward_details(round_idx, original_en, reference_zh, 
                               mock_responses, reward_results)
            
            # 打印风格向量信息，传入配置中的风格类型
            print_style_vectors(round_idx, original_en, reference_zh, 
                               reward_results['style_rewards'],
                               config['reward']['style_types'])
            
            logger.info(f"第 {round_idx + 1} 轮测试完成")
            
        except Exception as e:
            logger.error(f"第 {round_idx + 1} 轮测试失败: {e}")
            continue
    
    logger.info("奖励体系测试完成")


if __name__ == "__main__":
    main()