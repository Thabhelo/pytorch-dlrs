# DLRS design and numerical contract

DLRS is a loss-driven heuristic implemented independently from the [original paper](https://arxiv.org/abs/2507.21749). The supported behavior below describes this package; passing unit tests establishes those tested behaviors, not convergence or comparative model quality.

## Mathematical rule

Given batch losses `L[0], ..., L[B-1]`, with `B >= 2` and a positive mean:

```text
slope = (L[B-1] - L[0]) / mean(L)
scale = 10 ** floor(log10(lr))
new_lr = lr - scale * delta * slope
```

The factor is `delta_i` for negative slopes, `delta_o` for `0 <= slope <= 1`, and `delta_d` for slopes greater than 1. The equality at 1 follows the inclusive piecewise definition in section 3.2; Algorithm 1's strict middle condition instead falls through to its final branch at equality. Each positive-rate group is clamped to `min_lr` and, if supplied, `max_lr`. Zero-rate groups remain frozen at zero. Bounds are applied when a loss-driven update is computed, not at construction or on skipped/no-loss steps. An initial or externally assigned rate outside the bounds remains outside until such an update.

The [correctness contract](CORRECTNESS_CONTRACT.md) specifies exact input domains, floating-point behavior, state transitions and restoration. The mathematical reference is separate from the executable floating-point algorithm; the implementation does not promise correctly rounded evaluation of the entire real-valued equation.

All groups use the same slope, with an independently computed scale. Their original LR ratios are not generally preserved: crossing a power-of-ten boundary or reaching a bound changes the relative updates. The diagnostic `adjustment` is the last group's proposed adjustment, not a vector of realized, post-clamp changes. Inspect `get_last_lr()` or optimizer groups for actual rates.

## Input and failure behavior

Losses must be finite nonnegative real scalars. Scalar tensor observations are detached and copied to values; they do not retain an autograd graph. Booleans, complex values, nonscalar tensors, NaN, infinity, and negative losses are rejected. Negative-valued objectives need a justified transformation or a different scheduler; an arbitrary additive shift changes the rule.

Delta factors and `min_lr` must be finite and strictly positive. `max_lr` must be finite and strictly greater than `min_lr` when provided. Group rates must be finite and nonnegative. Writable scalar floating tensor rates without gradients are updated in place, with representability checks; this does not imply support for graph capture or differentiable optimizers.

Validation and rate computation precede mutation, so an invalid input or a failing later group must not leave earlier groups partially updated. The optimizer's parameter-group count and order are fixed at scheduler construction. Create a new scheduler when the optimizer's topology changes.

- `step(None)`: no loss-driven update; advances the epoch counter.
- Fewer than two observations: warning, unchanged rates, counter advances.
- All-zero losses: warning, unchanged rates, counter advances.
- Identical positive losses: zero slope; ordinary bound handling still applies to positive rates.
- Invalid observation, hyperparameter, checkpoint, or rate: exception before applying the update.

The loss calculation uses rescaling to avoid overflowing the sum or endpoint difference. There is no arbitrary near-zero-loss cutoff: a positive rescaling should preserve the mathematical slope, within floating-point limits. Underflow, finite precision, and quantization still apply. A missing upper bound explicitly permits growth; a finite bound is not a claim of stable training.

## What the slope does and does not measure

The statistic uses two endpoints and an unweighted mean of batch losses. It is not a fitted regression, a validation trend, a gradient estimate, or a statistical convergence test. Reordering batches can reverse the sign even when the set of losses is unchanged. Unequal batch sizes produce a mean of batch means rather than a mean over examples. Changes to loss reduction, masking, curriculum, task weighting, accumulation, or distributed aggregation can therefore change behavior.

The rule is invariant to multiplication of all losses by a positive constant in exact arithmetic. It is not invariant to an additive offset. Calling the regimes “convergence”, “stagnation”, and “divergence” is paper terminology, not a verified diagnosis of optimization.

For `B` observations, `G` parameter groups, and `T` parameter tensor objects (not tensor elements), ordinary validation and calculation cost O(B + G + T). If `H` groups use tensor learning rates, sorting their storage ranges and scanning for overlap adds O(H log H) worst-case work. Losses, group identity snapshots, and prepared updates use O(B + G + T) storage. The full epoch loss list alone consumes O(B) storage. Converting accelerator losses to Python values can synchronize the device, and a distributed recipe adds communication. Measure the complete training loop before stating overhead.

## Scheduler lifecycle and checkpoints

The scheduler follows PyTorch's initialization convention: construction performs an initial counter step without a DLRS loss update. User calls belong after optimizer updates. `step(None)` retains its historical no-op behavior; it is not evidence that composition wrappers can provide losses.

A checkpoint contains hyperparameters, epoch/counter state, diagnostics, and one learning rate per optimizer group. Loading validates the state and group count before applying it. Restore model, optimizer, scheduler, data-order, and RNG state consistently for an exact training continuation. Restoring scheduler state alone cannot restore optimizer momentum or model weights. Tensor LR storage identity is retained when supported.

## Epoch-metric wrapper

`DLRSOnPlateau` is a separate heuristic over a rolling window of `patience + 2` nonnegative epoch metrics. Once the window fills it applies the DLRS rule on every call. `mode='min'` uses the normal slope; `mode='max'` reverses its direction so improving larger-is-better metrics favor increases. History is bounded to the window length.

The name and `patience` argument predate this clarified contract. This wrapper has no best-metric tracking, threshold, bad-epoch patience, or cooldown. It is not equivalent to PyTorch's `ReduceLROnPlateau`; neither API similarity nor inheriting a scheduler base class establishes framework interchangeability.

## Compatibility boundaries

The [CI configuration](../.github/workflows/ci.yml) exercises selected CPU Python/PyTorch combinations. It does not cover every optimizer, accelerator, compiler, sharding system, framework, or multi-node configuration. DDP requires synchronized inputs and scheduler state on every rank; this package does not synchronize them automatically. `ChainedScheduler` and `SequentialLR` do not supply the required loss observations.

Before changing a real training system, compare against its existing scheduler under matched model initialization, data order, tuning budget, and stopping policy. Report task quality, failures, runtime, and variation over seeds. Results on a small classification or PINN example do not demonstrate universal or large-scale reliability.
