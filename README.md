# DLRS: Dynamic Learning Rate Scheduler for PyTorch

A PyTorch implementation of the Dynamic Learning Rate Scheduler (DLRS) algorithm from the paper ["Improving Neural Network Training using Dynamic Learning Rate Schedule for PINNs and Image Classification"](https://arxiv.org/abs/2507.21749) (arXiv:2507.21749v1).

DLRS automatically adjusts learning rates based on loss dynamics during training, eliminating the need for manual learning rate tuning and schedules.

## Key Features

- Adaptive learning rate adjustment based on loss slope analysis
- Works with any PyTorch optimizer (SGD, Adam, AdamW, etc.)
- Drop-in replacement for standard PyTorch schedulers
- Minimal configuration required
- Suitable for both image classification and PINNs

## Installation

### From Source

```bash
git clone https://github.com/Thabhelo/pytorch-dlrs.git
cd pytorch-dlrs
pip install -e .
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy 1.21+

## Quick Start

```python
import torch
from dlrs import DLRSScheduler

model = YourModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = DLRSScheduler(optimizer)

for epoch in range(num_epochs):
    batch_losses = []
    for batch in dataloader:
        loss = train_step(batch)
        batch_losses.append(loss.item())

    scheduler.step(batch_losses)
```

## How It Works

DLRS analyzes the trend of batch losses within each epoch to determine whether the model is:

- **Converging** (ΔL < 0): Increases learning rate to accelerate training
- **Diverging** (ΔL > 1): Decreases learning rate to stabilize training
- **Stagnating** (0 ≤ ΔL ≤ 1): Makes minimal adjustments

The algorithm computes:

1. Normalized loss slope: ΔL = (L_last - L_first) / L_mean
2. Adjustment granularity: n = floor(log10(α))
3. Learning rate update: α_new = α - (10^n × δ × ΔL)

## Parameters

- `delta_d` (float, default=0.5): Decremental factor for divergence
- `delta_o` (float, default=1.0): Adjustment factor for stagnation
- `delta_i` (float, default=0.1): Incremental factor for convergence
- `min_lr` (float, default=1e-8): Minimum learning rate bound

## Examples

### MNIST Classification

```bash
python examples/mnist_example.py --epochs 10 --device cpu
```

See the [examples](examples/) directory for more detailed usage.

## Results

Based on the original paper, DLRS achieves:

- Faster convergence compared to fixed learning rates
- Better final accuracy on image classification tasks
- Improved training stability for PINNs

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ --cov=dlrs --cov-report=html
```

## Citation

If you use DLRS in your research, please cite the original paper:

```bibtex
@article{dlrs2024,
  title={Improving Neural Network Training using Dynamic Learning Rate Schedule for PINNs and Image Classification},
  author={[Authors]},
  journal={arXiv preprint arXiv:2507.21749},
  year={2024}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
