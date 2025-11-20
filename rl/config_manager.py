"""
Configuration management module using Pydantic for validation.
Supports loading from YAML files and environment variables.
"""

import os
from pathlib import Path
from typing import List, Optional, Literal
from dataclasses import dataclass, field
import yaml


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: Literal["auto", "cpu", "cuda"] = "auto"


@dataclass
class TrainingConfig:
    """Training configuration."""
    num_generations: int = 4
    batch_size: int = 1
    num_epochs: int = 1
    max_length: int = 512
    learning_rate: float = 1e-5
    beta: float = 0.04  # GRPO KL regularization
    gamma: float = 1.0  # Reward discount factor


@dataclass
class RewardConfig:
    """Reward model configuration."""
    chinese_bert_path: str = ""
    english_bert_path: str = ""
    style_types: List[str] = field(default_factory=lambda: ["law", "literature", "news", "science"])

    # Reward weights
    format_reward_weight: float = 1.0
    semantic_reward_weight: float = 6.0
    style_reward_weight: float = 4.0

    # COMET configuration
    comet_model: str = "wmt22-cometkiwi-da"
    comet_device: Literal["cpu", "cuda"] = "cpu"
    comet_path: Optional[str] = None

    # Mode toggle
    test_mode: bool = False

    def __post_init__(self):
        """Validate and resolve paths after initialization."""
        # Resolve paths from environment variables if they're not set
        if not self.chinese_bert_path:
            self.chinese_bert_path = os.getenv(
                'CHINESE_BERT_PATH',
                'models/berts/chinese_style_detector_final.ckpt'
            )
        if not self.english_bert_path:
            self.english_bert_path = os.getenv(
                'ENGLISH_BERT_PATH',
                'models/berts/english_style_detector_final.ckpt'
            )
        if not self.comet_path:
            self.comet_path = os.getenv('COMET_MODEL_PATH', None)


@dataclass
class DataConfig:
    """Data configuration."""
    train_file: str = "data/train/train_style.parquet"
    test_file: str = "data/test/test_style.parquet"
    max_train_samples: int = 1000
    max_test_samples: int = 100


@dataclass
class OutputConfig:
    """Output configuration."""
    output_dir: str = "./outputs"
    logging_steps: int = 10
    save_steps: int = 100


@dataclass
class RLConfig:
    """Main RL configuration containing all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'RLConfig':
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            RLConfig instance
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            reward=RewardConfig(**config_dict.get('reward', {})),
            data=DataConfig(**config_dict.get('data', {})),
            output=OutputConfig(**config_dict.get('output', {}))
        )

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'RLConfig':
        """
        Load configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            RLConfig instance
        """
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            reward=RewardConfig(**config_dict.get('reward', {})),
            data=DataConfig(**config_dict.get('data', {})),
            output=OutputConfig(**config_dict.get('output', {}))
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'reward': self.reward.__dict__,
            'data': self.data.__dict__,
            'output': self.output.__dict__
        }

    def validate(self) -> None:
        """Validate configuration values."""
        # Validate weights sum
        total_weight = (
            self.reward.format_reward_weight +
            self.reward.semantic_reward_weight +
            self.reward.style_reward_weight
        )
        if total_weight <= 0:
            raise ValueError(
                f"Sum of reward weights must be positive, got {total_weight}"
            )

        # Validate paths exist (only in production mode)
        if not self.reward.test_mode:
            paths_to_check = [
                ('Chinese BERT', self.reward.chinese_bert_path),
                ('English BERT', self.reward.english_bert_path),
            ]

            for name, path in paths_to_check:
                if path and not os.path.exists(path):
                    raise FileNotFoundError(
                        f"{name} model not found at: {path}. "
                        f"Set the appropriate environment variable or update config.yaml"
                    )

        # Validate training params
        if self.training.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.training.num_generations <= 0:
            raise ValueError("Number of generations must be positive")


def load_config(config_path: Optional[str] = None) -> RLConfig:
    """
    Load configuration from file or use defaults.

    Args:
        config_path: Optional path to config file. If None, looks for 'config.yaml'
                    in current directory or uses defaults.

    Returns:
        RLConfig instance
    """
    if config_path is None:
        # Try to find config.yaml in current directory or rl/ directory
        for path in ['config.yaml', 'rl/config.yaml', '../config.yaml']:
            if os.path.exists(path):
                config_path = path
                break

    if config_path and os.path.exists(config_path):
        config = RLConfig.from_yaml(config_path)
    else:
        print(f"Warning: Config file not found, using defaults")
        config = RLConfig()

    # Validate configuration
    # Note: We skip validation in test mode to allow testing without model files
    if not config.reward.test_mode:
        try:
            config.validate()
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print("Continuing anyway. Set test_mode=true in config to skip validation.")

    return config
