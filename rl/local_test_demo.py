"""
本地测试运行器：使用TRL GRPO训练Qwen2.5-0.5B模型
支持GPU/CPU自动切换，用于测试整个流程的连通性
"""

import os
import torch
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from tqdm import tqdm

# TRL相关导入
try:
    from trl import GRPOConfig, GRPOTrainer
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM,
        TrainingArguments
    )
except ImportError as e:
    print(f"请安装trl库: pip install trl")
    raise e

# 自定义模块
from reward.reward_manager import RewardManager

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalTestRunner:
    """本地测试运行器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化运行器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self.load_config(config_path)
        self.device = self.setup_device()
        self.tokenizer = None
        self.model = None
        self.reward_manager = None
        self.current_dataset = None  # 保存当前数据集引用
        
        logger.info(f"使用设备: {self.device}")
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info("配置文件加载成功")
            return config
        except FileNotFoundError:
            logger.error(f"配置文件 {config_path} 不存在")
            raise
        except yaml.YAMLError as e:
            logger.error(f"配置文件格式错误: {e}")
            raise
    
    def setup_device(self) -> str:
        """设置计算设备"""
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"CUDA可用，使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.info("CUDA不可用，使用CPU")
        
        # 更新配置中的设备设置
        self.config['model']['device'] = device
        return device
    
    def setup_model_and_tokenizer(self):
        """设置模型和分词器"""
        model_path = self.config['model']['path']
        
        logger.info(f"加载模型: {model_path}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        
        logger.info("模型和分词器加载成功")
    
    def setup_reward_manager(self):
        """设置奖励管理器"""
        logger.info("设置奖励管理器")
        self.reward_manager = RewardManager(self.config)
    
    def create_sample_dataset(self) -> List[Dict[str, Any]]:
        """
        创建训练数据集（使用用户指定的WMT24数据）
        
        Returns:
            训练数据集
        """
        # 使用用户指定的WMT24数据
        test_data = [
            {
                'src_text': "Siso's depictions of land, water center new gallery exhibition\n",
                'tgt_text': '西索画作成为新画廊展览的焦点\n',
                'lang_pair': 'en-zh',
                'data_source': 'test_wmt24'
            },
            {
                'src_text': '"People Swimming in the Swimming Pool" from 2022 is one Vicente Siso artwork that will display at Tierra del Sol Gallery beginning Jan. 13. (photo courtesy of Vicente Siso)\n',
                'tgt_text': '2022年的《泳池戏水》是维森特·西索的又一作品，将于1月13日开始在Tierra del Sol画廊展出。（照片由维森特·西索提供）\n',
                'lang_pair': 'en-zh',
                'data_source': 'test_wmt24'
            },
            {
                'src_text': 'Tierra del Sol is pleased to present "Vicente Siso: Memories of the Land and Water" at the new gallery location in West Hollywood. Siso has been an artist in the Studio Arts Program since 2012, this marks his debut solo exhibition. Siso was born 1962 in Madrid and raised between Venezuela, Trinidad and Miami; he moved with his family to Southern California in his early 20s.\n',
                'tgt_text': 'Tierra del Sol高兴在西好莱坞新画廊展出"维森特·西索：水与陆的记忆"。西索自2012年起成为工作室艺术项目的艺术家，这是他首次举办个展。西索于1962年出生在马德里，在委内瑞拉、特立尼达和迈阿密长大，20岁左右随家人移居南加州。\n',
                'lang_pair': 'en-zh',
                'data_source': 'test_wmt24'
            },
            {
                'src_text': 'Masterfully working across subject matter, Siso has generated a prolific series of landscapes, portraits, and still-life works rendered in either acrylic, pastel, pencil or watercolor. Drawing from family portraits, his own reference photographs, and recollection, his colorful compositions demonstrate his range of interests and skill across media. Siso\'s tropical landscapes and seascapes reflect the geographies of his past, employing rich patterns and incorporating people to make meaningful connections between culture, memory and the environment. Siso titles his artworks in a mix of Spanish and English, signifying the celebrated and integral complexities of his life in Los Angeles County. "Vicente Siso: Memories of the Land and Water" opens on Saturday, Jan. 13, with a reception from 6-8 p.m. The exhibition is on view through Sunday, March 3.\n',
                'tgt_text': '西索熟练掌握各种画风，用丙烯颜料、蜡笔、铅笔或水彩创作了一系列的风景、肖像和静物作品。他从家庭照、参考照片和回忆中汲取灵感，通过丰富多彩的构图展现自己的兴趣范围和跨媒介技能。西索的热带风景画和海景画复原过去的地理面貌，运用了丰富的构图，让人物融入其中，在文化、记忆和环境之间建立了有意义的联系。西索画作名称混杂着西班牙语和英语，象征着他在洛杉矶县的生活精彩又跌宕《维森特·西索：水与陆的记忆》将于1月13日（周六）开展，展览时间为晚上6-8点。展览将持续到3月3日（周日）。\n',
                'lang_pair': 'en-zh',
                'data_source': 'test_wmt24'
            }
        ]
        
        # 构建符合TRL格式的数据集
        formatted_data = []
        for item in test_data:
            # 构建提示 - 使用RL模板格式
            prompt = f"""A conversation between User and Assistant. The User asks for a translation from English to Chinese, and the Assistant translates it. The final translation are enclosed within <translate> </translate> tags, i.e., <translate> final translation here </translate>. 

User:{item['src_text']}
Assistant:"""
            
            formatted_item = {
                'prompt': prompt,
                'src_text': item['src_text'],
                'tgt_text': item['tgt_text'],
                'lang_pair': item['lang_pair'],
                'data_source': item['data_source']
            }
            formatted_data.append(formatted_item)
        
        logger.info(f"创建训练数据集，包含 {len(formatted_data)} 条样本")
        logger.info("数据集包含平行语料库（原文+参考译文），可用于COMET语义奖励计算")
        # 打印前几个样本的信息
        for i, item in enumerate(formatted_data[:2]):
            logger.info(f"样本 {i+1}: src='{item['src_text'][:50]}...', tgt='{item['tgt_text'][:50]}...'")
        return formatted_data
    
    def custom_reward_function(self, prompts: List[str], completions: List[str], 
                             **kwargs) -> List[float]:
        """
        自定义奖励函数（供GRPOTrainer使用）
        
        Args:
            prompts: 提示文本列表
            completions: 生成的完成文本列表
            **kwargs: 其他参数
            
        Returns:
            奖励分数列表
        """
        # 提取源文本和语言对
        src_texts = []
        lang_pairs = []
        
        for prompt in prompts:
            # 提取用户输入的源文本
            if "User:" in prompt:
                user_part = prompt.split("User:")[1].split("\n")[0].strip()
                src_texts.append(user_part)
                lang_pairs.append('en-zh')  # 英译中
            else:
                src_texts.append("")
                lang_pairs.append('en-zh')
        
        # 打印每一步的生成结果（用于调试和分析）
        logger.info("=== GRPO组内生成结果 ===")
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            logger.info(f"样本 {i+1}:")
            logger.info(f"源文本: {src_texts[i]}")
            logger.info(f"生成结果: {completion}")
            
            # 提取翻译内容
            translation = self.reward_manager.format_reward.extract_translation(completion)
            logger.info(f"提取的翻译: {translation}")
            logger.info("-" * 50)
        
        # 计算奖励（注意：语义奖励需要参考译文，当前用源文本替代，TODO：提供真实参考译文）
# 获取参考译文（从数据集中）
        reference_texts = []
        if hasattr(self, 'current_dataset') and self.current_dataset is not None:
            # 从当前数据集中获取参考译文
            for i in range(len(completions)):
                if i < len(self.current_dataset):
                    reference_texts.append(self.current_dataset[i]['tgt_text'])
                else:
                    reference_texts.append("")  # 如果没有参考译文，使用空字符串
        else:
            # 如果没有数据集，使用源文本作为参考（降级方案）
            reference_texts = src_texts
        
        # 计算奖励
        reward_results = self.reward_manager.calculate_total_reward(
            completions, src_texts, prompts, lang_pairs, reference_texts)
        
        # 打印奖励详情（更新后包含语义奖励）
        logger.info("=== 奖励计算详情 ===")
        for i, (reward, detail) in enumerate(zip(reward_results['total_rewards'], reward_results['reward_details'])):
            logger.info(f"样本 {i+1}: 总奖励 = {reward:.3f}")
            logger.info(f"  格式奖励 = {detail['format_reward']:.3f} (权重: {self.reward_manager.format_weight:.1f})")
            logger.info(f"  语义奖励 = {detail['semantic_reward']:.3f} (权重: {self.reward_manager.semantic_weight:.1f})")
            logger.info(f"  风格奖励 = {detail['style_reward']:.3f} (权重: {self.reward_manager.style_weight:.1f})")
            logger.info(f"  主要风格: {detail['style_details'].get('source_main_style', 'unknown')} -> {detail['style_details'].get('target_main_style', 'unknown')}")
            logger.info(f"  风格匹配: {detail['style_details'].get('style_match', False)}")
        
        return reward_results['total_rewards']
    
    def run_training(self):
        """运行训练"""
        logger.info("开始运行GRPO训练")
        
        # 设置模型和奖励管理器
        self.setup_model_and_tokenizer()
        self.setup_reward_manager()
        
        # 创建测试数据
        train_data = self.create_sample_dataset()
        
        # 保存数据集引用，用于奖励计算时获取参考译文
        self.current_dataset = train_data
        
        # 设置训练参数 - 使用用户指定的2个epoch
        num_epochs = 2
        generation_batch_size = 4  # 必须与num_generations相同或为其倍数
        logger.info(f"训练参数: {num_epochs} 个epoch, 每样本生成 {self.config['training']['num_generations']} 个结果, batch_size = {generation_batch_size}")
        
        training_config = GRPOConfig(
            model_init_kwargs={
                "torch_dtype": torch.float32,
            },
            num_generations=self.config['training']['num_generations'],
            max_prompt_length=self.config['training']['max_length'],
            max_completion_length=self.config['training']['max_length'],
            num_train_epochs=num_epochs,
            per_device_train_batch_size=4,  # 设置为4以匹配num_generations
            # learning_rate=self.config['training']['learning_rate'],
            # beta=self.config['training']['beta'],
            # gamma=self.config['training']['gamma'],
            logging_steps=self.config['output']['logging_steps'],
            output_dir=self.config['output']['output_dir'],
            save_steps=50,  # 更频繁的保存
            report_to="none",  # 禁用wandb等报告
            remove_unused_columns=False,  # 保留所有列
            gradient_accumulation_steps=1,
            warmup_steps=0,
            lr_scheduler_type="constant",
            # 禁用bf16和fp16，使用fp32
            bf16=False,
            fp16=False,
            # 禁用GPU相关优化
            dataloader_pin_memory=False,
            dataloader_num_workers=0
        )
        
        # 创建训练器
        trainer = GRPOTrainer(
            model=self.model,
            reward_funcs=self.custom_reward_function,
            args=training_config,
            train_dataset=train_data,
            # tokenizer=self.tokenizer,
            # device=self.device,
        )
        
        logger.info("开始训练...")
        
        # 运行训练
        try:
            trainer.train()
            logger.info("训练完成！")
            
            # 保存模型
            output_dir = Path(self.config['output']['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            trainer.save_model(str(output_dir / "final_model"))
            self.tokenizer.save_pretrained(str(output_dir / "final_model"))
            
            logger.info(f"模型已保存到: {output_dir / 'final_model'}")
            
        except Exception as e:
            logger.error(f"训练过程中出现错误: {e}")
            raise
    
    def run_inference_test(self):
        """运行推理测试"""
        logger.info("运行推理测试")
        
        if self.model is None or self.tokenizer is None:
            self.setup_model_and_tokenizer()
        
        if self.reward_manager is None:
            self.setup_reward_manager()
        
        # 测试用例
        test_cases = [
            {
                'src_text': 'Hello, world!',
                'lang_pair': 'en-zh'
            }
        ]
        
        for case in test_cases:
            # 构建提示
            prompt = f"""A conversation between User and Assistant. The User asks for a translation from English to Chinese, and the Assistant translates it. The final translation are enclosed within <translate> </translate> tags, i.e., <translate> final translation here </translate>. 

User:{case['src_text']}
Assistant:"""
            
            # 生成多个结果
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            generated_texts = []
            for i in range(self.config['training']['num_generations']):
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=inputs['input_ids'].shape[1] + 100,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_texts.append(generated_text)
            
            # 计算奖励
            prompts_list = [prompt] * len(generated_texts)
            src_texts_list = [case['src_text']] * len(generated_texts)
            lang_pairs_list = [case['lang_pair']] * len(generated_texts)
            
            reward_results = self.reward_manager.calculate_total_reward(
                generated_texts, src_texts_list, prompts_list, lang_pairs_list)
            
            # 打印结果
            logger.info(f"测试案例: {case['src_text']}")
            for i, (text, reward) in enumerate(zip(generated_texts, reward_results['total_rewards'])):
                logger.info(f"生成结果 {i+1} (奖励: {reward:.3f}):")
                logger.info(f"文本: {text}")
                logger.info("-" * 50)
    
    def run(self, mode: str = "train"):
        """运行完整流程 - 直接进行GRPO训练，跳过推理测试
        
        Args:
            mode: 运行模式 - "train" (训练模式) 或 "test" (测试模式)
        """
        try:
            if mode == "train":
                logger.info("开始真正的GRPO训练...")
                self.run_training()
            elif mode == "test":
                # 只有在测试模式下才运行推理测试
                self.run_inference_test()
                logger.info("开始简化训练流程（测试模式）...")
                self.run_simplified_training()
            else:
                raise ValueError(f"不支持的运行模式: {mode}")
            
        except Exception as e:
            logger.error(f"运行过程中出现错误: {e}")
            raise
    
    def run_simplified_training(self):
        """运行简化版训练（用于测试流程）"""
        logger.info("运行简化版训练流程")

        # 创建测试数据
        train_data = self.create_sample_dataset()

        # 模拟训练循环
        reward_history = []

        for epoch in range(2):  # 使用用户指定的2个epoch
            logger.info(f"Epoch {epoch + 1}/2")

            epoch_rewards = []

            for batch_idx, item in enumerate(train_data):
                # 模拟生成过程
                generated_texts = []
                for i in range(self.config['training']['num_generations']):
                    # 这里应该是模型生成，现在用模拟数据
                    mock_generation = f"<think>Let me translate this...</think><translate>模拟翻译结果{i+1}</translate>"
                    generated_texts.append(mock_generation)

                # 计算奖励
                prompts = [item['prompt']] * len(generated_texts)
                src_texts = [item['src_text']] * len(generated_texts)
                lang_pairs = [item['lang_pair']] * len(generated_texts)

                reward_results = self.reward_manager.calculate_total_reward(
                    generated_texts, src_texts, prompts, lang_pairs)

                epoch_rewards.append(reward_results)

                # 打印批次信息
                avg_reward = np.mean(reward_results['total_rewards'])
                logger.info(f"  Batch {batch_idx + 1}: 平均奖励 = {avg_reward:.3f}")

            reward_history.extend(epoch_rewards)

        # 计算统计信息
        stats = self.reward_manager.get_reward_statistics(reward_history)
        logger.info("训练统计信息:")
        logger.info(f"总奖励 - 均值: {stats['total_reward']['mean']:.3f}, 标准差: {stats['total_reward']['std']:.3f}")
        logger.info(f"格式奖励 - 均值: {stats['format_reward']['mean']:.3f}")
        logger.info(f"风格奖励 - 均值: {stats['style_reward']['mean']:.3f}")

        logger.info("简化训练流程完成！")


def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='本地GRPO训练运行器')
    parser.add_argument('--mode', type=str, choices=['test', 'train'], default='train',
                      help='运行模式: train(训练模式) 或 test(测试模式)')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='配置文件路径')
    
    args = parser.parse_args()
    
    # 创建运行器
    runner = LocalTestRunner(args.config)
    
    # 运行完整流程
    runner.run(mode=args.mode)


if __name__ == "__main__":
    main()