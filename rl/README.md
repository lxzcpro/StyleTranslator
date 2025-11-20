# RL Module - Refactored Architecture

This directory contains the refactored Reinforcement Learning (RL) module for style-aware translation using GRPO (Group Relative Policy Optimization).

## 🎯 Key Improvements

### Software Engineering Principles Applied

1. **SOLID Principles**
   - **Single Responsibility**: Each module has one clear purpose
   - **Open/Closed**: Extensible through inheritance (base reward classes)
   - **Liskov Substitution**: All reward calculators implement common interface
   - **Interface Segregation**: Separate interfaces for different reward types
   - **Dependency Injection**: Components receive dependencies rather than creating them

2. **Design Patterns**
   - **Factory Pattern**: `RewardFactory` for creating reward components
   - **Strategy Pattern**: Pluggable reward calculators
   - **Facade Pattern**: `RewardManager` provides simple interface to complex subsystem

3. **Code Quality**
   - Type hints throughout
   - Comprehensive error handling
   - Structured logging
   - Modular architecture
   - Eliminated code duplication

## 📁 Architecture Overview

```
rl/
├── config_manager.py          # Centralized configuration management
├── trainer.py                 # Clean GRPO training orchestration
├── train.py                   # CLI entry point
├── local_test_demo.py         # Legacy trainer (backward compatibility)
├── config.yaml                # Configuration file
│
├── reward/                    # Reward system (modular)
│   ├── __init__.py           # Public API exports
│   ├── base_reward.py        # Abstract base classes
│   ├── format_score.py       # Format validation reward
│   ├── comet_score.py        # Semantic similarity reward (COMET)
│   ├── style_score.py        # Style consistency reward (BERT)
│   ├── reward_manager.py     # Legacy manager (backward compat)
│   ├── reward_manager_v2.py  # Improved manager with DI
│   └── reward_factory.py     # Factory for creating rewards
│
├── data/                      # Data processing
│   └── process_data.py       # Dataset preparation
│
└── tests/                     # Organized tests
    ├── README.md
    ├── test_bert_loading.py
    ├── test_bert_model.py
    ├── test_reward_system.py
    └── test_style_score.py
```

## 🚀 Quick Start

### Using the New Modular Trainer

```bash
# Train with config file
python train.py --config config.yaml

# Train with custom output directory
python train.py --config config.yaml --output ./my_output

# Train with sample data
python train.py --sample-data
```

### Programmatic Usage

```python
from rl.trainer import GRPOStyleTrainer
from rl.config_manager import load_config

# Load configuration
config = load_config('config.yaml')

# Create trainer
trainer = GRPOStyleTrainer(config)

# Prepare your dataset
dataset = [
    {
        'prompt': '...',
        'src_text': '...',
        'tgt_text': '...',
        'lang_pair': 'en-zh'
    },
    # more samples...
]

# Train
trainer.train(dataset)
```

### Using the Reward System Independently

```python
from rl.reward import RewardFactory
from rl.config_manager import load_config

# Create reward manager
config = load_config('config.yaml')
reward_manager = RewardFactory.create_from_config(config.to_dict())

# Calculate rewards
rewards = reward_manager.calculate_batch_rewards(
    generated_texts=['<think>...</think><translate>你好</translate>'],
    source_texts=['Hello'],
    prompts=['Translate: Hello'],
    language_pairs=['en-zh'],
    reference_texts=['你好']
)

# Access results
for reward in rewards:
    print(f"Total: {reward.total_score}")
    print(f"Format: {reward.components.format_score}")
    print(f"Semantic: {reward.components.semantic_score}")
    print(f"Style: {reward.components.style_score}")
```

## 🔧 Configuration

Configuration is managed through `config.yaml` and environment variables:

```yaml
model:
  name: "Qwen/Qwen2.5-0.5B-Instruct"
  device: "auto"  # auto, cpu, cuda

training:
  num_epochs: 2
  batch_size: 1
  learning_rate: 1e-5

reward:
  # Model paths (can use environment variables)
  chinese_bert_path: 'models/berts/chinese_style_detector_final.ckpt'
  english_bert_path: 'models/berts/english_style_detector_final.ckpt'

  # Weights
  format_reward_weight: 1
  semantic_reward_weight: 6
  style_reward_weight: 4

  # Mode
  test_mode: false  # true for mock models
```

### Environment Variables

```bash
export CHINESE_BERT_PATH=/path/to/chinese_bert.ckpt
export ENGLISH_BERT_PATH=/path/to/english_bert.ckpt
export COMET_MODEL_PATH=/path/to/comet.ckpt
```

## 🏗️ Reward System Architecture

### Base Classes

All reward calculators inherit from abstract base classes:

```python
from rl.reward import BaseReward, RewardResult

class MyCustomReward(BaseReward):
    def calculate(self, **kwargs) -> RewardResult:
        # Your implementation
        return RewardResult(score=0.9, details={...})

    def batch_calculate(self, batch_data) -> List[RewardResult]:
        # Batch implementation
        pass
```

### Reward Components

1. **Format Reward** (`format_score.py`)
   - Validates XML tags (`<think>`, `<translate>`)
   - Checks content validity
   - Weight: 1.0 (default)

2. **Semantic Reward** (`comet_score.py`)
   - Uses COMET model for translation quality
   - Compares against reference translation
   - Weight: 6.0 (default)

3. **Style Reward** (`style_score.py`)
   - BERT-based style classification
   - Cosine similarity between source and target styles
   - Weight: 4.0 (default)

### Dependency Injection

The new architecture uses dependency injection for better testability:

```python
from rl.reward import (
    FormatReward,
    CometSemanticReward,
    StyleRewardModel,
    RewardManager,
    RewardWeights
)

# Create individual components
format_reward = FormatReward()
semantic_reward = CometSemanticReward(model_name="wmt22-cometkiwi-da")
style_reward = StyleRewardModel(
    chinese_bert_path="...",
    english_bert_path="...",
    style_types=["law", "literature", "news", "science"]
)

# Inject into manager
weights = RewardWeights(format_weight=1, semantic_weight=6, style_weight=4)
manager = RewardManager(
    format_reward=format_reward,
    semantic_reward=semantic_reward,
    style_reward=style_reward,
    weights=weights
)
```

## 🧪 Testing

```bash
# Run all tests
pytest rl/tests/

# Run specific test
pytest rl/tests/test_reward_system.py

# With coverage
pytest --cov=rl rl/tests/
```

## 🐛 Bug Fixes

### Critical Fixes Applied

1. **Logits vs Probabilities Bug** (`style_score.py:97`)
   - **Before**: Returned raw logits
   - **After**: Properly applies softmax to convert to probabilities
   - **Impact**: Fixes mathematically incorrect cosine similarity calculation

2. **Code Duplication**
   - **Before**: `model.py` existed in both `rl/` and `style_detector/model/`
   - **After**: Single source of truth in `style_detector/model/`
   - **Impact**: Eliminates maintenance burden and version drift

3. **Hardcoded Paths**
   - **Before**: Absolute Windows paths in config
   - **After**: Relative paths with environment variable support
   - **Impact**: Portable across systems

## 📊 Backward Compatibility

Old code continues to work:

```python
# Legacy usage (still works)
from reward.reward_manager import RewardManager

config = load_config('config.yaml')
manager = RewardManager(config)
result = manager.calculate_total_reward(...)
```

## 🔄 Migration Guide

### From Old to New API

**Before:**
```python
from reward.reward_manager import RewardManager
manager = RewardManager(config)
result = manager.calculate_total_reward(
    generated_texts, source_texts, prompts, language_pairs, reference_texts
)
scores = result['total_rewards']
```

**After:**
```python
from rl.reward import RewardFactory
manager = RewardFactory.create_from_config(config)
rewards = manager.calculate_batch_rewards(
    generated_texts, source_texts, prompts, language_pairs, reference_texts
)
scores = [r.total_score for r in rewards]
```

### Benefits

- ✅ Type-safe `RewardOutput` objects
- ✅ Better error handling
- ✅ Clearer component separation
- ✅ Easier to test
- ✅ More maintainable

## 📝 Best Practices

1. **Use the Factory**: Always use `RewardFactory` to create components
2. **Inject Dependencies**: Pass dependencies rather than creating them
3. **Use Type Hints**: Leverage type hints for better IDE support
4. **Handle Errors**: Check `reward.is_valid` and `reward.error_message`
5. **Configure via YAML**: Keep configuration in `config.yaml`
6. **Test Mode**: Use `test_mode: true` for development without model files

## 🤝 Contributing

When adding new reward calculators:

1. Inherit from appropriate base class (`BaseReward`, `FormatRewardBase`, etc.)
2. Implement required methods (`calculate`, `batch_calculate`)
3. Return `RewardResult` objects
4. Add factory method to `RewardFactory`
5. Write tests in `tests/`

## 📚 Further Reading

- [GRPO Algorithm](https://github.com/huggingface/trl)
- [COMET Metric](https://github.com/Unbabel/COMET)
- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)
- [Design Patterns](https://refactoring.guru/design-patterns)

---

**Note**: The old `local_test_demo.py` is kept for backward compatibility but new development should use the modular `trainer.py` + `train.py` approach.
