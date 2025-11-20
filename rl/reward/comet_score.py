import torch
import logging
from typing import List, Dict, Optional, Any
from comet import download_model, load_from_checkpoint
from .base_reward import SemanticRewardBase, RewardResult

logger = logging.getLogger(__name__)


class CometSemanticReward(SemanticRewardBase):
    """
    基于COMET模型的语义奖励计算器
    使用wmt22-cometkiwi-da模型评估翻译质量
    """

    def __init__(self, model_name: str = "wmt22-cometkiwi-da",
                 model_path: str = None,
                 device: str = None):
        """
        初始化COMET语义奖励模型

        Args:
            model_name: COMET模型名称，默认使用wmt22-cometkiwi-da
            device: 运行设备，None表示自动选择
        """
        # 自动选择设备
        if device is None:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"COMET模型将使用GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = "cpu"
                logger.info("COMET模型将使用CPU")
        else:
            self.device = device

        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载COMET模型"""
        try:
            logger.info(f"正在加载COMET模型: {self.model_name}")
            if self.model_path is None or self.model_path == "None":
                logger.info("未提供COMET模型路径，将下载模型")
                self.model_path = download_model(self.model_name)
            else:
                logger.info(f"使用指定的COMET模型路径: {self.model_path}")
            self.model = load_from_checkpoint(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"COMET模型加载成功: {self.model_name}")
        except Exception as e:
            logger.error(f"加载COMET模型失败: {e}")
            logger.warning("将使用模拟的语义奖励分数")
            self.model = None

    def calculate(self, source: str, reference: str, hypothesis: str) -> RewardResult:
        """
        Calculate semantic reward for single input (implements BaseReward interface).

        Args:
            source: Source text
            reference: Reference translation
            hypothesis: Generated translation

        Returns:
            RewardResult with score and details
        """
        scores = self.calculate_semantic_reward([source], [reference], [hypothesis])
        return RewardResult(
            score=scores[0],
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
        计算语义奖励分数

        Args:
            source_texts: 源文本列表
            reference_texts: 参考译文列表
            hypothesis_texts: 实际译文列表（模型生成的翻译）

        Returns:
            语义奖励分数列表，范围通常在0-1之间
        """
        if not self.model:
            # 如果模型加载失败，返回模拟分数
            logger.warning("COMET模型未加载，使用模拟语义奖励分数")
            return [0.5 + 0.3 * (hash(h) % 1000) / 1000.0 for h in hypothesis_texts]

        try:
            # 准备COMET输入数据
            data = []
            for src, ref, hyp in zip(source_texts, reference_texts, hypothesis_texts):
                data.append({
                    "src": src,
                    "ref": ref,
                    "mt": hyp
                })

            # 打印调试信息
            logger.info(
                f"COMET计算: 源文本数量={len(source_texts)}, 参考译文数量={len(reference_texts)}, 假设译文数量={len(hypothesis_texts)}")
            if len(source_texts) > 0:
                logger.info(f"示例 - 源文本: '{source_texts[0][:50]}...'")
                logger.info(f"示例 - 参考译文: '{reference_texts[0][:50]}...'")
                logger.info(f"示例 - 假设译文: '{hypothesis_texts[0][:50]}...'")

            # 使用COMET模型预测质量分数
            with torch.no_grad():
                gpus = 0 if self.device == "cpu" else 1
                scores = self.model.predict(data, batch_size=8, gpus=gpus)

            # COMET分数通常在0-1之间，可以直接用作奖励
            semantic_rewards = scores["scores"]

            logger.info(f"语义奖励计算完成，平均分数: {sum(semantic_rewards) / len(semantic_rewards):.3f}")
            logger.info(f"语义奖励分数详情: {semantic_rewards}")
            return semantic_rewards

        except Exception as e:
            logger.error(f"计算语义奖励时出错: {e}")
            logger.error(f"错误详情: {str(e)}")
            # 返回中等质量的模拟分数
            return [0.5] * len(source_texts)

    def calculate_batch_reward(self, batch_data: List[Dict]) -> List[float]:
        """
        批量计算语义奖励

        Args:
            batch_data: 包含source、reference、hypothesis的字典列表

        Returns:
            语义奖励分数列表
        """
        source_texts = [item['source'] for item in batch_data]
        reference_texts = [item['reference'] for item in batch_data]
        hypothesis_texts = [item['hypothesis'] for item in batch_data]

        return self.calculate_semantic_reward(source_texts, reference_texts, hypothesis_texts)

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "loaded": self.model is not None
        }