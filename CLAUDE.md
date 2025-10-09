# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements the Dynamic Learning Rate Scheduler (DLRS) for PyTorch, based on the paper "Improving Neural Network Training using Dynamic Learning Rate Schedule for PINNs and Image Classification" (arXiv:2507.21749v1).

DLRS is an epoch-level learning rate scheduler that automatically adjusts learning rates based on loss dynamics during training.

## Key Architecture

### Core Algorithm (dlrs/scheduler.py)

The DLRSScheduler inherits from torch.optim.lr_scheduler.LRScheduler and implements:

1. **Loss Analysis**: Collects batch losses during an epoch
2. **Slope Calculation**: Computes normalized loss slope ΔL = (L_last - L_first) / L_mean
3. **Granularity Adjustment**: Determines adjustment scale n = floor(log10(α))
4. **LR Update**: Applies α_new = α - (10^n × δ × ΔL)

Key insight: When ΔL < 0 (loss decreasing), the subtraction becomes addition, so LR increases.

### Three Cases

- **Convergence** (ΔL < 0): Uses delta_i, increases LR
- **Divergence** (ΔL > 1): Uses delta_d, decreases LR
- **Stagnation** (0 ≤ ΔL ≤ 1): Uses delta_o, makes minimal adjustments

### Critical Implementation Details

1. Scheduler operates at EPOCH level, not batch level
2. Must accumulate batch losses during epoch, then call step() once per epoch
3. step() takes a list of batch losses, not a single loss value
4. Includes safety checks for zero division and minimum LR bounds

## Common Development Commands

### Run Tests

```bash
pytest tests/ -v
```

With coverage:
```bash
pytest tests/ --cov=dlrs --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_scheduler.py -v
```

### Run Examples

MNIST example:
```bash
python examples/mnist_example.py --epochs 10 --device cpu
```

With verbose output:
```bash
python examples/mnist_example.py --epochs 5 --verbose
```

### Install Package

Development mode (editable):
```bash
pip install -e .
```

## Project Structure

```
dlrs/
  __init__.py       # Package exports
  scheduler.py      # DLRSScheduler implementation
  utils.py          # Helper functions and LossRecorder

tests/
  test_scheduler.py # Scheduler unit tests
  test_utils.py     # Utility function tests

examples/
  mnist_example.py  # MNIST demonstration

docs/
  API_REFERENCE.md  # Complete API documentation
  INSTALLATION.md   # Installation instructions
```

## Testing Strategy

### Unit Tests (tests/test_scheduler.py)

- **Initialization Tests**: Validate hyperparameters
- **Update Logic Tests**: Verify LR increases/decreases correctly
- **Edge Cases**: Zero loss, identical losses, min_lr bounds
- **Multi-step Tests**: Convergence/divergence trajectories
- **Integration Tests**: With real optimizers and networks

### Test Fixtures

- `simple_optimizer`: Basic SGD optimizer
- `convergent_losses`: Decreasing loss sequence
- `divergent_losses`: Increasing loss sequence
- `stagnant_losses`: Minimal change sequence

## Common Patterns

### Basic Usage

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = DLRSScheduler(optimizer)

for epoch in range(num_epochs):
    batch_losses = []
    for batch in dataloader:
        loss = train_step(batch)
        batch_losses.append(loss.item())
    scheduler.step(batch_losses)
```

### With LossRecorder

```python
from dlrs import DLRSScheduler, LossRecorder

recorder = LossRecorder()
scheduler = DLRSScheduler(optimizer)

for epoch in range(num_epochs):
    for batch in dataloader:
        loss = train_step(batch)
        recorder.add(loss.item())
    scheduler.step(recorder.get_losses())
    recorder.reset()
```

## Default Hyperparameters

From the paper:
- delta_d = 0.5 (divergence)
- delta_o = 1.0 (stagnation)
- delta_i = 0.1 (convergence)
- min_lr = 1e-8

## Git Configuration

Author: Thabhelo
Email: thabhelo@deepubuntu.com
GitHub: https://github.com/Thabhelo

## Code Style Guidelines

- No emojis in code or comments
- No em-dashes
- Clear, descriptive variable names
- Comprehensive docstrings with parameter types
- Type hints where appropriate
- PEP 8 compliant

## Important Notes

1. The scheduler MUST receive batch losses after each epoch, not after each batch
2. Minimum 2 batch losses required to compute slope
3. Learning rate adjustments are relative to current LR magnitude (via granularity n)
4. state_dict() and load_state_dict() supported for checkpointing
5. Compatible with all PyTorch optimizers (SGD, Adam, AdamW, etc.)
