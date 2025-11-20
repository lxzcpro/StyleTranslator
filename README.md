# StyleTranslator

Neural machine translation with reinforcement learning for style-aware translation. Maintains source text style (law, literature, news, science) in target translation.

## Features

- **Style-Aware Translation**: Preserves document style across languages
- **Multi-Component Rewards**: Format validation + Semantic quality (COMET) + Style consistency (BERT)
- **GRPO Training**: Group Relative Policy Optimization for RL fine-tuning
- **Hydra Configuration**: Flexible, composable configuration management
- **Modular Architecture**: Clean separation with dependency injection

## Quick Start

```bash
# Install dependencies
conda env create -f environment.yml
conda activate style_translator

# Train style detector (BERT-based classifier)
cd style_detector
python train.py

# Train translation model with RL
cd ../rl
python scripts/train_rl.py
```

## Project Structure

```
StyleTranslator/
├── style_detector/         # Style classification (BERT)
│   ├── corpus/             # Corpus generation
│   ├── dataset/            # Dataset loaders
│   ├── model/              # StyleDetector model
│   ├── config.yaml         # Training config
│   └── train.py            # Training script
│
├── rl/                     # RL training module
│   ├── configs/            # Hydra configurations
│   │   ├── env/            # Environment settings
│   │   ├── reward/         # Reward presets
│   │   └── model/          # Model configs
│   ├── src/
│   │   ├── rewards/        # Reward system
│   │   ├── trainer/        # GRPO trainer
│   │   └── utils/          # Utilities
│   ├── scripts/            # Entry points
│   └── data/               # Training data
│
└── utils/                  # Shared utilities
    └── convert_ckpt_to_hf.py
```

## Components

### 1. Style Detector

BERT-based classifier for detecting text style (law, literature, news, science).

```bash
cd style_detector
python train.py  # Train on your corpus
```

**Config**: `style_detector/config.yaml`

### 2. RL Training

GRPO-based reinforcement learning for style-aware translation.

```bash
cd rl

# Default training
python scripts/train_rl.py

# Server environment + style-weighted rewards
python scripts/train_rl.py env=server reward=style_weighted

# Override parameters
python scripts/train_rl.py training.num_epochs=5 device=cuda
```

**Configs**: `rl/configs/`

## Reward System

| Component | Weight | Description |
|-----------|--------|-------------|
| Format    | 1.0    | XML tag validation (`<think>`, `<translate>`) |
| Semantic  | 6.0    | COMET translation quality |
| Style     | 4.0    | BERT style consistency (source ↔ target) |

Customize in `rl/configs/reward/*.yaml`.

## Configuration

### Environment-Specific

```bash
# Local development
python scripts/train_rl.py env=local

# Production server (uses environment variables)
export CHINESE_BERT_PATH=/path/to/model.ckpt
export ENGLISH_BERT_PATH=/path/to/model.ckpt
python scripts/train_rl.py env=server
```

### Reward Presets

- `default.yaml`: Balanced (format=1, semantic=6, style=4)
- `style_weighted.yaml`: Style-focused (format=1, semantic=4, style=8)

## Development

```bash
# Run tests
cd rl
pytest tests/

# Check configuration
python scripts/train_rl.py --cfg job

# Print resolved config
python scripts/train_rl.py --cfg all
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers
- TRL (Transformers Reinforcement Learning)
- COMET
- Hydra
- PyTorch Lightning

See `environment.yml` for full dependencies.

## Architecture

### Style Detection

```
Text → BERT → Style Classifier → [law, literature, news, science]
```

### RL Training

```
Source Text → LLM → Translation
           ↓
    Format + Semantic + Style Rewards
           ↓
      GRPO Optimization
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- TRL library for GRPO implementation
- COMET for translation quality metrics
- Transformers library for model infrastructure
