# RL Module - Style-Aware Translation with GRPO

Reinforcement Learning module for training translation models with multi-component rewards using Hydra configuration management.

## Features

- **Multi-Component Rewards**: Format + Semantic (COMET) + Style (BERT)
- **Hydra Configuration**: Compose configs from modular YAML files
- **Modular Architecture**: Clean separation with dependency injection
- **Environment-Specific Configs**: Local development vs production server
- **Type-Safe**: Full type hints and OmegaConf integration

## Quick Start

```bash
# Install dependencies
pip install torch transformers trl comet-ml hydra-core omegaconf

# Train with default config
python scripts/train_rl.py

# Train with server environment and style-weighted rewards
python scripts/train_rl.py env=server reward=style_weighted

# Override specific parameters
python scripts/train_rl.py training.num_epochs=3 training.learning_rate=1e-4
```

## Project Structure

```
rl/
├── configs/                    # Hydra configuration
│   ├── config.yaml             # Main config (defaults)
│   ├── env/                    # Environment settings
│   │   ├── local.yaml
│   │   └── server.yaml
│   ├── reward/                 # Reward configurations
│   │   ├── default.yaml
│   │   └── style_weighted.yaml
│   └── model/                  # Model configurations
│       └── qwen_0.5b.yaml
│
├── src/                        # Core source code
│   ├── rewards/                # Reward system
│   │   ├── base.py             # Base classes
│   │   ├── format.py           # Format validation
│   │   ├── semantic.py         # COMET semantic
│   │   ├── style.py            # BERT style
│   │   ├── manager.py          # Reward orchestration
│   │   └── factory.py          # Component factory
│   ├── trainer/                # Training logic
│   │   └── r1_trainer.py       # GRPO trainer
│   └── utils/                  # Utilities
│       └── io.py               # I/O helpers
│
└── scripts/                    # Entry points
    └── train_rl.py             # Training script
```

## Configuration

### Compose Configurations

```bash
# Use different environments
python scripts/train_rl.py env=local        # Local development
python scripts/train_rl.py env=server       # Production server

# Use different reward configurations
python scripts/train_rl.py reward=default          # Balanced rewards
python scripts/train_rl.py reward=style_weighted   # Emphasize style

# Combine multiple configs
python scripts/train_rl.py env=server reward=style_weighted
```

### Override Parameters

```bash
# Command-line overrides
python scripts/train_rl.py training.num_epochs=5
python scripts/train_rl.py reward.format_weight=2.0
python scripts/train_rl.py device=cuda
```

### Environment Variables

For production, use environment variables:

```bash
export CHINESE_BERT_PATH=/path/to/chinese_bert.ckpt
export ENGLISH_BERT_PATH=/path/to/english_bert.ckpt
export COMET_MODEL_PATH=/path/to/comet.ckpt

python scripts/train_rl.py env=server
```

## Usage

### Training

```python
import hydra
from omegaconf import DictConfig
from src.trainer.r1_trainer import GRPOStyleTrainer

@hydra.main(config_path="configs", config_name="config")
def train(cfg: DictConfig):
    trainer = GRPOStyleTrainer(cfg)
    dataset = load_your_dataset()
    trainer.train(dataset)
```

### Programmatic API

```python
from omegaconf import OmegaConf
from src.trainer.r1_trainer import GRPOStyleTrainer
from src.rewards import RewardFactory

# Load config
cfg = OmegaConf.load('configs/config.yaml')

# Create components
reward_manager = RewardFactory.create_from_config(OmegaConf.to_container(cfg))
trainer = GRPOStyleTrainer(cfg, reward_manager=reward_manager)

# Train
trainer.train(dataset)
```

## Reward Components

| Component | Default Weight | Description |
|-----------|----------------|-------------|
| Format    | 1.0            | XML tag validation (`<think>`, `<translate>`) |
| Semantic  | 6.0            | COMET translation quality |
| Style     | 4.0            | BERT style consistency |

Customize weights in `configs/reward/*.yaml`.

## Development

```bash
# Run with test mode (mock models)
python scripts/train_rl.py test_mode=true

# Check configuration
python scripts/train_rl.py --cfg job

# Print full resolved config
python scripts/train_rl.py --cfg all
```

## Design Principles

- **Separation of Concerns**: Configs, rewards, training logic are isolated
- **Composability**: Mix and match environment, reward, model configs
- **Type Safety**: OmegaConf + type hints throughout
- **Dependency Injection**: Components receive dependencies, don't create them
- **Robustness**: Comprehensive error handling and validation

## Hydra Features

- **Config Composition**: Combine multiple YAML files
- **Command-Line Overrides**: Change any parameter from CLI
- **Automatic Logging**: Hydra manages output directories
- **Multi-Run**: Easy hyperparameter sweeps
- **Interpolation**: Reference other config values

## License

See project root for license information.
