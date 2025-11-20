# RL Module Tests

This directory contains tests for the RL (Reinforcement Learning) module.

## Test Files

- **test_bert_loading.py**: Tests for loading BERT models from checkpoints
- **test_bert_model.py**: Tests for BERT model inference and predictions
- **test_reward_system.py**: Integration tests for the complete reward system
- **test_style_score.py**: Unit tests for style scoring functionality

## Running Tests

```bash
# Run all tests
pytest rl/tests/

# Run specific test file
pytest rl/tests/test_reward_system.py

# Run with verbose output
pytest -v rl/tests/

# Run with coverage
pytest --cov=rl rl/tests/
```

## Test Configuration

Tests use the configuration from `../config.yaml`. For tests that require model files,
ensure the paths in the config are correct or set `test_mode: true` to use mock models.
