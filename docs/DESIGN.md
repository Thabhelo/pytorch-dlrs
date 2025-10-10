# DLRS Algorithm Design Notes

This document explains the mathematical foundation and implementation details of the Dynamic Learning Rate Scheduler (DLRS).

## Algorithm Overview

DLRS automatically adjusts the learning rate based on the slope of batch losses within each training epoch. Unlike schedulers that use fixed decay schedules or monitor validation metrics, DLRS reacts to **within-epoch loss dynamics**.

## Mathematical Foundation

### Normalized Loss Slope

For epoch $j$, given batch losses $L_1^{(j)}, L_2^{(j)}, \ldots, L_B^{(j)}$:

1. **First loss**: $L_{\text{first}}^{(j)} = L_1^{(j)}$
2. **Last loss**: $L_{\text{last}}^{(j)} = L_B^{(j)}$
3. **Mean loss**: $\overline{L}^{(j)} = \frac{1}{B} \sum_{b=1}^{B} L_b^{(j)}$

The **normalized loss slope** is:

$$\Delta L_j = \frac{L_{\text{last}}^{(j)} - L_{\text{first}}^{(j)}}{\overline{L}^{(j)}}$$

This normalization makes $\Delta L_j$ scale-invariant: whether your loss is 0.1 or 100.0, the slope is comparable.

### Three Cases

DLRS categorizes loss dynamics into three cases:

| Case | Condition | Interpretation | Action |
|------|-----------|----------------|--------|
| **Convergence** | $\Delta L_j < 0$ | Loss is decreasing within the epoch | Increase LR (accelerate) |
| **Stagnation** | $0 \le \Delta L_j \le 1$ | Loss is flat or slightly increasing | Make modest adjustments |
| **Divergence** | $\Delta L_j > 1$ | Loss is increasing rapidly | Decrease LR (stabilize) |

### Adjustment Granularity

The adjustment scale adapts to the current learning rate magnitude:

$$n_j = \lfloor \log_{10}(\alpha_j) \rfloor$$

$$\text{granularity} = 10^{n_j}$$

**Example**: If $\alpha_j = 0.01$, then $n_j = -2$ and granularity = $10^{-2} = 0.01$.

This ensures that:
- Large LRs (e.g., 1.0) get larger adjustments (granularity = 1.0)
- Small LRs (e.g., 0.001) get smaller adjustments (granularity = 0.001)

### Learning Rate Update

The **adjustment magnitude** is:

$$\alpha_{\delta,j} = 10^{n_j} \times \delta_{\text{case}} \times \Delta L_j$$

where $\delta_{\text{case}}$ is:
- $\delta_i$ (incremental factor) if converging
- $\delta_o$ (stagnation factor) if stagnating
- $\delta_d$ (decremental factor) if diverging

The **new learning rate** is:

$$\alpha_{j+1} = \alpha_j - \alpha_{\delta,j}$$

**Key insight**: When $\Delta L_j < 0$ (convergence), the subtraction becomes addition, so the LR increases.

Finally, clamp to bounds:

$$\alpha_{j+1} = \text{clamp}(\alpha_{j+1}, \alpha_{\text{min}}, \alpha_{\text{max}})$$

## Implementation Details

### Numerical Stability

1. **Zero mean loss**: If $|\overline{L}^{(j)}| < 10^{-10}$, skip the update to avoid division by zero.

2. **Zero slope**: If $\Delta L_j = 0$, the adjustment is zero and LR remains unchanged.

3. **Very small LR**: If $\alpha_j \le 0$, default to $n_j = -8$ to prevent log errors.

### Multi-Param-Group Behavior

DLRS computes **one slope per epoch** and applies it to all parameter groups. Each group's LR is updated independently:

```python
for group in optimizer.param_groups:
    current_lr = group['lr']
    n = floor(log10(current_lr))
    adjustment = (10 ** n) * delta * slope
    new_lr = clamp(current_lr - adjustment, min_lr, max_lr)
    group['lr'] = new_lr
```

This means:
- All groups react to the same loss dynamics
- Each group's adjustment scales with its own current LR

If you need different slopes per group, you must track per-group losses separately and subclass `DLRSScheduler`.

### Edge Cases

**Insufficient data**: If fewer than 2 batch losses are provided, the scheduler cannot compute a slope and skips the update.

**Identical losses**: If all batch losses are identical (e.g., all 1.0), then $\Delta L_j = 0$ and the LR does not change.

**Explosive growth**: If LR grows unbounded, set `max_lr` to cap it.

**Negative LR**: The clamping to `min_lr` prevents negative LRs. If `min_lr` itself is set improperly (<= 0), initialization will raise a `ValueError`.

## Design Rationale

### Why within-epoch slopes?

Standard schedulers (StepLR, ExponentialLR) follow fixed decay schedules. ReduceLROnPlateau reacts to validation metrics between epochs. Neither captures **within-epoch training dynamics**.

DLRS fills this gap by analyzing the trajectory of batch losses as training progresses through an epoch:
- If the model is learning quickly (losses drop), accelerate by increasing LR
- If training is unstable (losses spike), stabilize by decreasing LR
- If progress is slow but steady, make modest adjustments

This is especially valuable for:
- **PINNs**: Loss landscapes are highly non-convex with many local minima
- **Small datasets**: Validation metrics are noisy; within-epoch trends are more informative
- **Non-standard tasks**: When you don't know a good decay schedule in advance

### Why log-scaled granularity?

Consider two scenarios:
1. $\alpha = 1.0$, slope = -0.5 → want to increase LR by ~0.1
2. $\alpha = 0.001$, slope = -0.5 → want to increase LR by ~0.0001

Using a fixed adjustment would either:
- Be too small for case 1
- Be too large for case 2

Log-scaling ensures adjustments are proportional to the current LR magnitude.

### Why three delta parameters?

Separate control over convergence, stagnation, and divergence allows fine-tuning:
- Set `delta_d` high (e.g., 0.7) for aggressive stabilization in unstable training
- Set `delta_i` low (e.g., 0.05) for cautious LR increases
- Set `delta_o` medium (e.g., 1.0) for standard behavior in flat regions

This three-way split matches the paper's empirical findings that different phases of training require different responsiveness.

## Comparison to Related Schedulers

| Scheduler | Monitors | Frequency | Can Increase LR? | Requires Validation? |
|-----------|----------|-----------|------------------|----------------------|
| **StepLR** | Fixed epochs | Fixed intervals | No | No |
| **ExponentialLR** | Fixed epochs | Every epoch | No | No |
| **CosineAnnealingLR** | Fixed epochs | Every epoch | Yes (cyclical) | No |
| **ReduceLROnPlateau** | Scalar metric | After epochs | No | Yes |
| **DLRS** | Batch losses | Every epoch | Yes | No |

## Hyperparameter Sensitivity

Based on the original paper and empirical testing:

**delta_d** (decremental factor):
- Range: [0.1, 1.0]
- Default: 0.5
- Higher values → more aggressive LR reduction during divergence
- Lower values → gentler response to instability

**delta_o** (stagnation factor):
- Range: [0.5, 2.0]
- Default: 1.0
- Controls adjustment size when loss is flat
- Typically left at 1.0 unless you have specific needs

**delta_i** (incremental factor):
- Range: [0.01, 0.5]
- Default: 0.1
- Higher values → faster LR growth during convergence
- Lower values → more conservative LR increases

**min_lr**:
- Default: 1e-8
- Acts as a safety floor to prevent LR collapse
- Set higher (e.g., 1e-6) if you want to stop decreasing earlier

**max_lr**:
- Default: None (unbounded)
- Set to a safe upper bound (e.g., 1.0 for SGD, 0.01 for Adam)
- Prevents runaway LR growth

## Extensions and Future Work

Possible enhancements (not yet implemented):

1. **Per-group slopes**: Track separate loss slopes for different param groups (e.g., encoder vs decoder)
2. **Momentum-based smoothing**: Use exponential moving average of slopes to reduce noise
3. **Gradient norm integration**: Combine loss slope with gradient magnitude for richer signal
4. **Adaptive thresholds**: Learn the boundaries between convergence/stagnation/divergence dynamically

## References

Original paper:
- Veerababu Dharanalakota, Ashwin Arvind Raikar, and Prasanta Kumar Ghosh (2025). "Improving neural network training using dynamic learning rate schedule for PINNs and image classification." *Machine Learning with Applications*, 21, 100697. DOI: https://doi.org/10.1016/j.mlwa.2025.100697

ArXiv preprint:
- https://arxiv.org/abs/2507.21749
