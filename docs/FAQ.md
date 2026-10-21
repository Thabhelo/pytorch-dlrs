# FAQ and migration notes

## Is DLRS ready for critical or large-scale training?

There is no such assurance from this repository. The package remains experimental. The [audit](RELIABILITY_AUDIT.md) records corrected implementation behavior, the measured distributed/accelerator/scale/quality scope, and remaining deployment gaps. Use the current scheduler as a control, validate on representative runs, and retain a tested rollback/checkpoint path. The original paper's reported results are not measurements of this implementation on your workload.

## Which class should I use?

`DLRSScheduler.step(batch_losses)` implements the within-epoch loss rule. `DLRSOnPlateau.step(metric)` instead observes a rolling sequence of epoch metrics. Both may increase the learning rate.

For `DLRSOnPlateau`, `patience=0` means a two-metric window; `patience=2` means four metrics are required, after which each call uses the most recent four. It does not wait for a number of bad epochs. In `mode='max'`, increasing metrics produce the improvement direction. Metrics must still be finite and nonnegative. Use PyTorch's `ReduceLROnPlateau` when its patience, threshold, cooldown, and decrease-only behavior are intended.

## Distributed training

DDP processes have separate optimizers. PyTorch synchronizes gradients and expects identical optimizer updates across ranks; it does not automatically synchronize this scheduler's state or learning rates. See [PyTorch's DDP documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).

Every rank must supply the same observations and call the scheduler in the same order. The [distributed validation report](DISTRIBUTED_VALIDATION.md) documents the tested two-process CPU/Gloo policy, unequal local batch sizes, complete epoch-checkpoint restart, coordinated input rejection, and bounded rank-failure probes.

The repository's [illustrative CPU/Gloo helper](../examples/distributed_policy.py) validates observations collectively before backward or optimizer updates. For a per-example mean, it computes a sample-weighted global observation and scales the local backward pass by `world_size * local_count / global_count`. This correction matters when local batch sizes differ: ordinary DDP averages gradients across ranks, not across individual examples.

```python
# EVERY rank: initialized CPU/Gloo group with an explicit operation timeout,
# ordinary DDP, identical optimizer/scheduler configuration, no accumulation.
# This helper is an example in the checkout, not an installed dlrs export.
from examples.distributed_policy import require_equal_iterations, reduce_sample_mean

require_equal_iterations(len(train_loader))
batch_losses = []
for inputs, targets in train_loader:
    optimizer.zero_grad(set_to_none=True)
    local_mean = criterion(model(inputs), targets)  # Mean over local examples.
    observation = reduce_sample_mean(local_mean, inputs.shape[0])
    (local_mean * observation.backward_scale).backward()
    optimizer.step()
    batch_losses.append(observation.mean)

scheduler.step(batch_losses)  # EVERY rank
```

Use `DistributedSampler.set_epoch(epoch)` when using that sampler. Token losses, masks, weights, gradient accumulation, empty local batches, uneven iteration counts, `join()`, custom communication hooks, and sharded optimizers need their own denominator and synchronization contracts. The helper deliberately rejects unsupported CPU/Gloo input contracts and does not establish CUDA/NCCL or multi-node support. DLRS's mean across the sequence of batch observations remains the existing algorithm; it is not automatically a sample-weighted epoch mean.

A coordinated validation error prevents the subsequent backward/optimizer/scheduler updates in this calling order, but the preceding forward pass may already have consumed RNG or changed buffers. A rank that exits cannot participate in validation. Stop the attempt and restore a complete agreed checkpoint after failures; use process-group timeouts and a launcher that cleans up failed peers. Exact epoch-boundary restart and failures immediately before observation validation have been tested separately; automatic elastic recovery and arbitrary failure locations have not.

Another policy can broadcast one rank's observations to everyone, but that represents that rank's data rather than a global mean. Updating only rank 0 without synchronizing all rates and scheduler state is incorrect. Restore matching complete checkpoints on every rank before continuing.

## Checkpointing

Save the model, optimizer, and scheduler together. Restore after constructing the model, optimizer, and scheduler; constructing a scheduler after restoring the optimizer can change the scheduler's view of initial rates. PyTorch documents this ordering in [Optimizer.load_state_dict](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.load_state_dict.html).

```python
# Saving an epoch-boundary checkpoint:
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
}
torch.save(checkpoint, 'checkpoint.pt')

# Recreate model, optimizer, and scheduler with matching parameter groups first.
checkpoint = torch.load('checkpoint.pt', map_location='cpu', weights_only=True)
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
scheduler.load_state_dict(checkpoint['scheduler'])
```

Serialize scheduler, optimizer, checkpoint, and external learning-rate mutations at an agreed training boundary; the scheduler does not provide thread synchronization. If storing snapshots in memory while training continues, deep-copy the whole checkpoint; model and optimizer state can contain live tensor references. For complete reproducibility, also save epoch/data position, random states, scaler state, and framework state. A mid-epoch checkpoint must preserve the pending loss observations. Scheduler loading checks group count and saved values; it cannot identify a semantically different model that happens to have the same number of groups.

## Can I combine DLRS with another scheduler?

`ChainedScheduler` invokes each child's `step()` without losses, so DLRS would only perform a no-op counter step. `SequentialLR` also does not forward batch losses and expects a different step contract. Neither is supported for DLRS. See [ChainedScheduler's API](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ChainedScheduler.html).

An explicit training-loop policy could call DLRS with losses and then a second scheduler, but order changes the result and may violate DLRS's intended bounds. Checkpoint and test the combined policy; it has no guarantee of preserving either algorithm's standalone behavior.

## Optimizers, AMP, and frameworks

Normal eager optimizers with finite scalar rates are the intended scope. Multiple parameter groups share a slope but need not maintain their LR ratios. Zero-rate groups stay frozen. Writable scalar floating-point tensor rates without gradients are supported with in-place updates; this does not certify compiled, capturable, differentiable, fused, sharded, or custom optimizer modes.

With AMP, collect the unscaled loss. `GradScaler.step()` may skip an optimizer update, so define what an epoch observation means when that occurs and avoid treating a skipped optimizer step as completed. Gradient accumulation changes observation frequency and, sometimes, returned loss normalization. The core scheduler cannot infer these choices.

The [Lightning callback example](../examples/lightning_callback.py) is restricted to one process, one optimizer, automatic optimization, full precision (`32-true` or `64-true`), no gradient accumulation, and no other configured scheduler. It rejects unsupported settings. Callback contract tests are not a substitute for running the exact installed Lightning version through train/save/resume.

The [Transformers callback](../examples/huggingface_callback.py) is disabled because standard callback logs do not expose every raw batch loss and Trainer owns its own scheduler. A correct Trainer integration must obtain the intended losses, manage scheduler ownership, and restore state before it can be advertised.

## Troubleshooting

**The rate does not change.** Check for a zero initial rate, missing losses, fewer than two observations, all-zero losses, a zero endpoint slope, or a reached bound. `step()` with no argument does not infer previously recorded losses.

**The rate grows too much.** A decreasing within-epoch loss causes increases by design. `max_lr=None` has no upper cap. Choose a measured cap and reconsider `delta_i`, data ordering, and whether the statistic is useful for this workload. A smaller cap can still perform worse than your baseline.

**A loss/rate/checkpoint is rejected.** Inspect for NaN, infinity, negatives, booleans, nonscalar values, mismatched groups, or corrupted state. Do not silently drop observations or retry only some ranks: doing so can change the training policy or desynchronize distributed state.

**How much overhead is there?** No universal percentage has been established. The calculation scans the observations, stores a list, and updates parameter groups; `.item()` can synchronize accelerators and distributed aggregation costs communication. Measure end-to-end throughput and memory at your actual scale.

## Migrating from 0.2.1

These changes are unreleased until a new version is published:

- Invalid/nonfinite/negative losses, metrics, hyperparameters, and rates now fail before a partial update. Previously some corrupted values could propagate. Fix the source instead of relying on silent clamping or skipped errors.
- Positive losses below the previous `1e-10` mean threshold now produce a scale-invariant update. All-zero loss sequences still skip. Very small positive losses can therefore change the LR trajectory on resume.
- Zero-rate groups remain zero instead of being raised to `min_lr`. Set a positive rate explicitly if a group should resume learning.
- `DLRSOnPlateau(mode='max')` now reverses the slope direction; the old implementation ignored `mode`. Runs using that mode will change. History is bounded; `patience` retains its rolling-window meaning.
- Checkpoint validation rejects malformed states and mismatched group counts instead of silently accepting partial restoration. Restore optimizer and scheduler together with matching groups.
- Adding, removing or reordering live parameter groups or their parameters is rejected. Create a new scheduler and migrate state explicitly when topology changes.
- Tensor learning-rate storage is preserved, while initial/base metadata is copied independently. Unsupported representations fail explicitly. Previously overwritten initial metadata cannot be reconstructed from an old checkpoint.
- Corrected floating-point boundary handling can change adjustment choices at thresholds, decades and extreme magnitudes. See the [complete breaking-change notes](RELIABILITY_AUDIT.md#breaking-changes-requiring-review).
- The correctness-contract revision chooses the decade of the exact stored float. For example, `lr=1e-6`, losses `[1.0, 0.0]` and `min_lr=1e-12` now produce `1.02e-6` instead of `1.2e-6` with default factors. This affects ordinary decimal-looking rates, not only extreme inputs. Existing checkpoint formats still load, but continuation across this correction need not preserve the old trajectory.
- A mathematically nonzero scheduler slope too small to represent as a Python float now raises before updating rates or state. Negative exact fractions that round to `-0.0` are rejected before conversion. Rescale observations under an explicit application policy rather than silently dropping a rejected observation.
- The framework examples have a narrower, enforced scope. Transformers construction now fails rather than silently appearing to schedule correctly.
- Runtime installation no longer pulls in torchvision, NumPy, or test tools. Install `.[examples]` or `.[dev]` when needed, using compatible PyTorch/torchvision versions.

Review these intentional corrections before resuming a long-running experiment. An old experiment's exact behavior requires its original package/environment; a corrected implementation cannot promise an identical trajectory.
