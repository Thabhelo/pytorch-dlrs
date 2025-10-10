# FAQ and Troubleshooting

## Frequently Asked Questions

### General

**Q: When should I use DLRS instead of standard schedulers?**

A: Use DLRS when:
- You want adaptive learning rate adjustment without manual tuning
- Your loss dynamics are non-monotonic (common in PINNs)
- You need a scheduler that reacts to within-epoch loss trends
- Standard fixed schedules (StepLR, ExponentialLR) don't work well for your problem

**Q: How does DLRS differ from ReduceLROnPlateau?**

A: Key differences:
- **DLRS** analyzes loss slope within each epoch using all batch losses
- **ReduceLROnPlateau** only looks at scalar metrics between epochs
- **DLRS** can increase or decrease LR based on convergence/divergence
- **ReduceLROnPlateau** only decreases LR when metric plateaus

**Q: Can I use DLRS with any optimizer?**

A: Yes. DLRS works with all PyTorch optimizers (SGD, Adam, AdamW, etc.) because it only adjusts the learning rate parameter.

**Q: Should I use DLRSScheduler or DLRSOnPlateau?**

A:
- **DLRSScheduler**: Use when you can collect per-batch losses during training (more accurate)
- **DLRSOnPlateau**: Use for drop-in compatibility with training loops that call `scheduler.step(metric)`

### Multi-GPU and Distributed Training

**Q: Does DLRS work with DataParallel or DistributedDataParallel?**

A: Yes. DLRS operates on the optimizer, which is shared across replicas. Ensure:
- Batch losses are collected on the main process
- `scheduler.step()` is called only on rank 0 in distributed settings

Example for DDP:
```python
if rank == 0:
    scheduler.step(batch_losses)
```

**Q: How do I handle multi-param-group optimizers?**

A: DLRS computes one loss slope per epoch and applies the same adjustment factor to all parameter groups. If you need per-group behavior, subclass DLRSScheduler and override `_update_learning_rate()`.

### Hyperparameter Tuning

**Q: What are good starting values for delta_d, delta_o, delta_i?**

A: The defaults work well for most tasks:
- `delta_d=0.5`: Decrease LR by 50% of the granularity when diverging
- `delta_o=1.0`: Make standard adjustments when stagnating
- `delta_i=0.1`: Increase LR by 10% of the granularity when converging

Start with these and adjust if needed:
- Increase `delta_d` (→ 0.7) if training is too unstable
- Decrease `delta_i` (→ 0.05) if LR grows too fast
- Adjust `delta_o` (→ 0.5 or 1.5) if stagnation handling is too aggressive/conservative

**Q: How do I choose min_lr and max_lr?**

A:
- `min_lr`: Set to 1e-8 or 1e-7. Prevents LR from becoming zero.
- `max_lr`: Optional upper bound. Use if you know a safe maximum (e.g., 1.0 for SGD). Leave as `None` for unbounded growth.

**Q: My learning rate keeps growing. What should I do?**

A:
1. Set a `max_lr` bound: `DLRSScheduler(optimizer, max_lr=0.1)`
2. Reduce `delta_i` to slow down LR increases
3. Check if your loss is actually converging (decreasing within epochs)

### Troubleshooting

**Q: I get a warning "Mean loss is near zero"**

A: This happens when all batch losses are ~0.0. The scheduler skips the update because normalized slope is undefined. Solutions:
- Check if your model has collapsed (outputting zeros)
- Verify loss function is computing non-zero values
- Ensure you're logging loss correctly (use `.item()` on tensors)

**Q: My learning rate became negative or NaN**

A: This shouldn't happen with proper guards. Possible causes:
- Very large loss values causing overflow
- Incorrect `min_lr` setting (must be > 0)
- Bug in custom subclass overriding `_update_learning_rate()`

If this occurs, please file an issue with a reproducible example.

**Q: Learning rate explodes even with max_lr set**

A: Check:
- `max_lr` is properly passed to the scheduler constructor
- You're not manually modifying `optimizer.param_groups[0]['lr']` elsewhere
- Your loss isn't decreasing extremely fast (causing very negative slopes)

**Q: Scheduler doesn't adjust the learning rate at all**

A: Common causes:
- Passing `None` to `step()` → performs no adjustment
- Passing a list with < 2 elements → cannot compute slope
- All batch losses are identical → slope is zero, no adjustment needed

**Q: Training is unstable with DLRS**

A: Try:
1. Set a `max_lr` bound to prevent excessive growth
2. Increase `delta_d` to react more strongly to divergence
3. Use a smaller initial learning rate
4. Combine DLRS with gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`

### State Management and Checkpointing

**Q: How do I save and resume training with DLRS?**

A: Use `state_dict()` and `load_state_dict()`:

```python
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'epoch': epoch
}
torch.save(checkpoint, 'checkpoint.pth')

checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
scheduler.load_state_dict(checkpoint['scheduler'])
```

**Q: What information is stored in state_dict()?**

A: DLRS stores:
- Hyperparameters: `delta_d`, `delta_o`, `delta_i`, `min_lr`, `max_lr`
- Epoch tracking: `last_epoch`
- Diagnostic info: `loss_slope`, `adjustment`, `_last_lr`
- **New in v0.1.1**: Per-param-group learning rates (`param_group_lrs`)

### Integration with Training Frameworks

**Q: Can I use DLRS with PyTorch Lightning?**

A: Yes. See `examples/lightning_callback.py` for a ready-to-use callback:

```python
from examples.lightning_callback import DLRSLightningCallback

trainer = pl.Trainer(callbacks=[DLRSLightningCallback(delta_d=0.5)])
```

**Q: Can I use DLRS with Hugging Face Transformers?**

A: Yes. See `examples/huggingface_callback.py` for a TrainerCallback:

```python
from examples.huggingface_callback import DLRSTrainerCallback

trainer = Trainer(
    model=model,
    callbacks=[DLRSTrainerCallback(delta_d=0.5)]
)
```

**Q: Does DLRS work with mixed precision training (AMP)?**

A: Yes. DLRS is agnostic to gradient scaling. Example:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for epoch in range(epochs):
    batch_losses = []
    for data, target in dataloader:
        optimizer.zero_grad()
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_losses.append(loss.item())
    scheduler.step(batch_losses)
```

### Performance

**Q: Does DLRS slow down training?**

A: No. The overhead is negligible:
- Computes one slope per epoch (3 operations on a short list)
- No expensive operations like validation during training
- Benchmark overhead: < 0.1% of epoch time

**Q: Can I use DLRS with very large batch sizes?**

A: Yes. DLRS only needs 2+ batch losses per epoch. With very large batches:
- You have fewer batches per epoch → less fine-grained slope estimation
- Consider using smaller batches if training dynamics allow

### Advanced Usage

**Q: Can I apply different delta values to different parameter groups?**

A: Not directly. The current implementation uses a single slope for all groups. To customize:

```python
class MultiGroupDLRS(DLRSScheduler):
    def _update_learning_rate(self):
        # Custom logic per param group
        for i, group in enumerate(self.optimizer.param_groups):
            # Compute group-specific adjustment
            pass
```

**Q: Can I log the loss slope for analysis?**

A: Yes. Access `scheduler.loss_slope` after each `step()`:

```python
scheduler.step(batch_losses)
print(f"Loss slope: {scheduler.loss_slope:.6f}")
```

**Q: How do I combine DLRS with another scheduler?**

A: Chain them using `torch.optim.lr_scheduler.ChainedScheduler` (PyTorch 1.4+):

```python
scheduler1 = DLRSScheduler(optimizer)
scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])
```

Note: Chaining may lead to unexpected behavior if both schedulers aggressively modify LR.

## Still having issues?

If your question isn't answered here:
1. Check the [examples/](../examples) directory for code patterns
2. Read the [API documentation](API.md)
3. Open an issue: https://github.com/Thabhelo/pytorch-dlrs/issues
