# DLRS: Dynamic Learning Rate Scheduler for PyTorch

DLRS is an independent implementation of the loss-driven learning rate rule in [Improving neural network training using dynamic learning rate schedule for PINNs and image classification](https://arxiv.org/abs/2507.21749) by Veerababu Dharanalakota, Ashwin Arvind Raikar, and Prasanta Kumar Ghosh. It changes an optimizer's learning rates using the first, last, and mean batch loss of an epoch.

**Status: experimental / Alpha.** This package is not part of PyTorch. Correct implementation of the update rule does not guarantee convergence, better accuracy, or unchanged training quality. Large-scale production readiness, broad accelerator coverage, and upstream acceptance have not been established. See the [reliability audit](docs/RELIABILITY_AUDIT.md) for measured evidence and remaining gaps.

The [single-GPU CUDA validation](docs/CUDA_VALIDATION.md) passed all 18 required installed-wheel checks on a free Colab T4; two native-BF16 cases were unsupported and skipped. A [pinned notebook](notebooks/free_colab_cuda_validation.ipynb) reproduces the bounded check. The [readiness ledger](docs/UPSTREAM_READINESS.md) separates completed checks from remaining deployment and upstream requirements.

The local revision completed 96 planned trials across MNIST, CIFAR-10, a manufactured PINN, and a small Transformer. MNIST and Transformer met their provisional comparison margins; CIFAR-10 and the PINN failed against cosine. CIFAR-10 also failed to establish the required tolerance against fixed LR. These results block a general replacement recommendation; see the [workload results](docs/WORKLOAD_VALIDATION.md) and [Transformer results](docs/TRANSFORMER_VALIDATION.md).

## Installation

Install an appropriate PyTorch build for your platform first using the [official instructions](https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install pytorch-dlrs
```

The package declares Python 3.8+ and PyTorch 2.0+; available combinations depend on PyTorch's wheels. These lower bounds are compatibility targets, not a claim that every newer release or platform has been tested. The [compatibility report](docs/COMPATIBILITY_VALIDATION.md) records executed local checks; the [CI matrix](.github/workflows/ci.yml) and [contribution guide](CONTRIBUTING.md) define the ongoing checks. NumPy, torchvision, plotting tools, and test tools are optional, not runtime dependencies of `dlrs`.

For the unreleased `0.3.0.dev0` source revision:

```bash
git clone https://github.com/Thabhelo/pytorch-dlrs.git
cd pytorch-dlrs
python -m pip install -e '.[dev]'
```

The published PyPI package and this development revision differ in behavior; read the [migration notes](docs/FAQ.md#migrating-from-021) before replacing 0.2.1 in an existing run. For release history, consult [CHANGELOG.md](CHANGELOG.md).

## Quick start

Run `scheduler.step(batch_losses)` once at the end of an epoch, after optimizer updates. Losses must be finite, nonnegative scalar values in batch order. The following example is self-contained:

```python
import torch
from dlrs import DLRSScheduler

torch.manual_seed(0)
model = torch.nn.Linear(4, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
scheduler = DLRSScheduler(optimizer, min_lr=1e-6, max_lr=0.1)
inputs = torch.randn(32, 4)
targets = torch.randn(32, 1)

for epoch in range(3):
    batch_losses = []
    for data, target in zip(inputs.split(8), targets.split(8)):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(data), target)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.detach().item())
    scheduler.step(batch_losses)
    print(epoch, scheduler.get_last_lr())
```

`max_lr=0.1` is illustrative; determine useful bounds with a controlled comparison on your workload. A bound limits a rate, not training risk. With mixed precision, use the unscaled loss and account for optimizer steps skipped by the scaler. With gradient accumulation, explicitly define whether each observation represents a microbatch or a completed optimizer update.

## Update rule

For at least two losses, DLRS computes `slope = (last - first) / mean` and, independently for each positive learning rate `lr`:

```text
scale = 10 ** floor(log10(lr))
delta = delta_i if slope < 0 else delta_o if slope <= 1 else delta_d
new_lr = clamp(lr - scale * delta * slope, min_lr, max_lr)
```

A negative slope increases the rate; a positive slope decreases it. These names describe a heuristic observation, not a proof that the model is converging or diverging. The rule uses the endpoints and mean, so batch ordering, uneven batch sizes, and noisy losses matter. [Design notes](docs/DESIGN.md) describe numerical behavior and limitations.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `delta_d` | `0.5` | Factor for slope greater than 1 |
| `delta_o` | `1.0` | Factor for slope from 0 through 1 |
| `delta_i` | `0.1` | Factor for negative slope |
| `min_lr` | `1e-8` | Floor applied during positive-rate updates |
| `max_lr` | `None` | Optional cap; `None` permits unbounded growth |
| `last_epoch` | `-1` | Initial epoch index, following PyTorch scheduler initialization |
| `verbose` | `False` | Print update diagnostics |

Bounds apply during loss-driven updates; construction and skipped/no-loss steps do not rewrite initial or externally changed rates. Zero-rate groups remain zero. Invalid values are rejected before applying a partial update. Empty/single-observation and all-zero loss sequences warn and leave rates unchanged. `step()` without losses advances the scheduler counter but does not perform a DLRS adjustment.

## Integration contract

- Use a normal eager PyTorch optimizer with scalar real learning rates. Tensor rates must be writable scalar floating-point values without gradients; their identity is preserved. Graph capture, compiled or differentiable optimizers, sharded optimizers, and arbitrary custom optimizers are not certified.
- Serialize optimizer updates, scheduler steps, and checkpoint operations for each optimizer. The scheduler provides no thread synchronization.
- Keep optimizer parameter groups fixed after constructing the scheduler. Adding, removing, or reordering groups requires a new scheduler and an explicit state migration.
- For distributed training, every rank needs identical scheduler observations, state, and resulting learning rates. DLRS performs no collectives. Calling `step()` only on rank 0 leaves other optimizers out of sync; see [the DDP example](docs/FAQ.md#distributed-training).
- `ChainedScheduler` and `SequentialLR` do not forward batch losses. They must not be used to orchestrate DLRS. Multiple schedulers also change each other's behavior; use an explicitly tested training-loop policy if combining them.
- `DLRSOnPlateau` uses a rolling window of epoch metrics, not within-epoch losses. Its historical `patience` argument sets the window length (`patience + 2`), not bad-epoch patience. `mode='max'` reverses the slope direction. It is **not** a drop-in replacement for `ReduceLROnPlateau`.
- The Lightning example supports a deliberately narrow single-process training loop; the Transformers callback is disabled pending a correct integration. Examples are not universal framework compatibility guarantees.

Save and restore both optimizer and scheduler state. Construct the scheduler before loading either state, load the optimizer, then load the matching scheduler state. See [checkpointing and migration](docs/FAQ.md#checkpointing).

## Validation and examples

The local validation reports separate [training modes](docs/TRAINING_MODES_VALIDATION.md),
[distributed recovery](docs/DISTRIBUTED_VALIDATION.md),
[version compatibility](docs/COMPATIBILITY_VALIDATION.md),
[vision/PINN workload quality](docs/WORKLOAD_VALIDATION.md),
[Transformer workload quality](docs/TRANSFORMER_VALIDATION.md),
[ordinary-use behavior against the previous release](docs/DIFFERENTIAL_COMPATIBILITY.md), and
[scheduler scaling](docs/SCALABILITY_VALIDATION.md). A successful functional test
is not evidence that an alternative learning-rate policy preserves your model
quality. Use the workload acceptance results and their stated limits.

```bash
python -m pytest tests --cov=dlrs --cov-report=term-missing
ruff check dlrs tests examples benchmarks setup.py .github/scripts
mypy dlrs
python -m build
python -m twine check dist/*
```

To run dataset/plot examples from a checkout, install `python -m pip install -e '.[examples]'` with compatible torch/torchvision versions. The [MNIST example](examples/mnist_example.py), [PINN example](examples/pinns_helmholtz.py), and [benchmark runner](benchmarks/run_benchmarks.py) illustrate usage. They do not establish performance superiority or a safe default for an unrelated workload. Benchmark reports must record seeds, software/hardware, tuning budgets, failures, and held-out evaluation. Retain the existing scheduler as a baseline and compare quality as well as runtime before adopting DLRS.

## Citation and attribution

```bibtex
@article{DHARANALAKOTA2025100697,
  title = {Improving neural network training using dynamic learning rate schedule for PINNs and image classification},
  author = {Dharanalakota, Veerababu and Raikar, Ashwin Arvind and Ghosh, Prasanta Kumar},
  journal = {Machine Learning with Applications},
  volume = {21},
  pages = {100697},
  year = {2025},
  doi = {10.1016/j.mlwa.2025.100697}
}
```

Implementation by Thabhelo. [MIT License](LICENSE). [Contribution guide](CONTRIBUTING.md), [FAQ](docs/FAQ.md), and [issues](https://github.com/Thabhelo/pytorch-dlrs/issues).
