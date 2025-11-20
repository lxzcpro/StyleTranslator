# RL Module - Style-Aware Translation Training

Reinforcement Learning module for training translation models with GRPO (Group Relative Policy Optimization) and multi-component reward functions.

## Features

- **Multi-Component Rewards**: Format validation, semantic quality (COMET), and style consistency (BERT)
- **Modular Architecture**: Clean separation with dependency injection
- **Type-Safe**: Full type hints and dataclasses
- **Configurable**: YAML-based configuration with environment variable support
- **Test Mode**: Mock models for development without actual model files

## Quick Start

```bash
# Install dependencies
pip install torch transformers trl comet-ml pyyaml

# Train with default config
python train.py --config config.yaml

# Train with test mode (no model files needed)
python train.py --sample-data
```

## Architecture

```
rl/
├── config.yaml           # Configuration
├── config_manager.py     # Config loading & validation
├── trainer.py            # GRPO training orchestration
├── train.py              # CLI entry point
├── reward/               # Modular reward system
│   ├── base_reward.py    # Abstract base classes
│   ├── format_score.py   # XML tag validation
│   ├── comet_score.py    # Semantic similarity (COMET)
│   ├── style_score.py    # Style consistency (BERT)
│   ├── reward_manager.py # Reward orchestration
│   └── reward_factory.py # Component factory
└── tests/                # Test suite
```

## Configuration

Edit `config.yaml` or use environment variables:

```yaml
reward:
  chinese_bert_path: 'models/berts/chinese_style_detector_final.ckpt'
  english_bert_path: 'models/berts/english_style_detector_final.ckpt'
  format_reward_weight: 1
  semantic_reward_weight: 6
  style_reward_weight: 4
  test_mode: false  # true for mock models
```

Environment variables:
```bash
export CHINESE_BERT_PATH=/path/to/chinese_bert.ckpt
export ENGLISH_BERT_PATH=/path/to/english_bert.ckpt
export COMET_MODEL_PATH=/path/to/comet.ckpt
```

## Usage

### Training

```python
from rl.trainer import GRPOStyleTrainer
from rl.config_manager import load_config

config = load_config('config.yaml')
trainer = GRPOStyleTrainer(config)

dataset = [
    {
        'prompt': 'Translate: Hello',
        'src_text': 'Hello',
        'tgt_text': '你好',
        'lang_pair': 'en-zh'
    }
]

trainer.train(dataset)
```

### Reward System Only

```python
from rl.reward import RewardFactory

config = load_config('config.yaml')
reward_manager = RewardFactory.create_from_config(config.to_dict())

rewards = reward_manager.calculate_batch_rewards(
    generated_texts=['<think>...</think><translate>你好</translate>'],
    source_texts=['Hello'],
    prompts=['Translate: Hello'],
    language_pairs=['en-zh'],
    reference_texts=['你好']
)

print(rewards[0].total_score)  # Combined score
print(rewards[0].components)   # Format, semantic, style breakdown
```

## Reward Components

1. **Format Reward** (weight: 1.0)
   - Validates `<think>` and `<translate>` tags
   - Checks content validity

2. **Semantic Reward** (weight: 6.0)
   - Uses COMET model for translation quality
   - Compares against reference translation

3. **Style Reward** (weight: 4.0)
   - BERT-based style classification
   - Cosine similarity between source/target styles
   - Supports 4 styles: law, literature, news, science

## Development

```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=rl tests/

# Format code
black rl/
```

## Design Principles

- **SOLID**: Single responsibility, dependency injection, interface segregation
- **DRY**: No code duplication, shared base classes
- **Type Safety**: Full type hints, dataclasses for structured data
- **Robustness**: Comprehensive error handling, validation, logging

## API Reference

### RewardManager

```python
class RewardManager:
    def calculate_single_reward(
        generated_text: str,
        source_text: str,
        prompt: str,
        language_pair: str,
        reference_text: Optional[str] = None
    ) -> RewardOutput
```

### GRPOStyleTrainer

```python
class GRPOStyleTrainer:
    def train(
        train_dataset: List[Dict[str, Any]],
        output_dir: Optional[str] = None
    ) -> None
```

### RewardFactory

```python
class RewardFactory:
    @classmethod
    def create_from_config(config: dict) -> RewardManager
```

## License

See project root for license information.
