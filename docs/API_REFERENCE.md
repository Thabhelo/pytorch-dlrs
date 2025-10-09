# API Reference

## DLRSScheduler

```python
dlrs.DLRSScheduler(
    optimizer,
    delta_d=0.5,
    delta_o=1.0,
    delta_i=0.1,
    min_lr=1e-8,
    last_epoch=-1,
    verbose=False
)
```

Dynamic Learning Rate Scheduler that adjusts learning rate based on loss dynamics.

### Parameters

- **optimizer** (torch.optim.Optimizer): Wrapped optimizer.
- **delta_d** (float, optional): Decremental factor for divergence case (ΔL > 1). Default: 0.5
  - Typical range: [0.1, 1.0]
  - Higher values cause more aggressive learning rate reduction when loss diverges
- **delta_o** (float, optional): Adjustment factor for stagnation case (0 ≤ ΔL ≤ 1). Default: 1.0
  - Typical range: [0.5, 2.0]
  - Controls learning rate sensitivity in flat regions
- **delta_i** (float, optional): Incremental factor for convergence case (ΔL < 0). Default: 0.1
  - Typical range: [0.01, 0.5]
  - Higher values cause more aggressive learning rate increases when loss decreases
- **min_lr** (float, optional): Minimum learning rate bound. Default: 1e-8
  - Prevents learning rate from becoming too small
- **last_epoch** (int, optional): The index of last epoch. Default: -1
- **verbose** (bool, optional): If True, prints learning rate updates. Default: False

### Methods

#### step(batch_losses)

Update learning rate based on batch losses from the current epoch.

**Parameters:**
- **batch_losses** (List[float]): List of loss values from each batch in the epoch. Must contain at least 2 values.

**Example:**
```python
scheduler = DLRSScheduler(optimizer)
batch_losses = []

for batch in dataloader:
    loss = train_step(batch)
    batch_losses.append(loss.item())

scheduler.step(batch_losses)
```

#### get_last_lr()

Returns the last computed learning rate.

**Returns:**
- List[float]: List of learning rates for each parameter group

#### state_dict()

Returns the state of the scheduler as a dictionary.

**Returns:**
- dict: Scheduler state

#### load_state_dict(state_dict)

Loads the scheduler state.

**Parameters:**
- **state_dict** (dict): Scheduler state

## LossRecorder

```python
dlrs.LossRecorder()
```

Helper class to record and analyze batch losses during training.

### Methods

#### add(loss)

Add a batch loss value.

**Parameters:**
- **loss** (float): Loss value to record

#### reset()

Clear all recorded losses.

#### get_losses()

Return list of all recorded losses.

**Returns:**
- List[float]: All recorded losses

#### get_mean()

Return mean of recorded losses.

**Returns:**
- float or None: Mean loss, or None if no losses recorded

#### get_min()

Return minimum recorded loss.

**Returns:**
- float or None: Minimum loss, or None if no losses recorded

#### get_max()

Return maximum recorded loss.

**Returns:**
- float or None: Maximum loss, or None if no losses recorded

#### get_std()

Return standard deviation of recorded losses.

**Returns:**
- float or None: Standard deviation, or None if insufficient data

#### get_loss_slope()

Calculate normalized loss slope from recorded losses.

**Returns:**
- float or None: Normalized slope (last - first) / mean, or None if insufficient data

**Example:**
```python
recorder = LossRecorder()

for batch in dataloader:
    loss = train_step(batch)
    recorder.add(loss.item())

slope = recorder.get_loss_slope()
mean_loss = recorder.get_mean()
recorder.reset()
```

## Utility Functions

### validate_hyperparameters(delta_d, delta_o, delta_i)

Validate DLRS hyperparameter values.

**Parameters:**
- **delta_d** (float): Decremental factor
- **delta_o** (float): Stagnation factor
- **delta_i** (float): Incremental factor

**Raises:**
- InvalidHyperparameter: If any parameter is invalid

### calculate_normalized_loss_slope(first_loss, last_loss, mean_loss)

Calculate normalized loss slope with safe division.

**Parameters:**
- **first_loss** (float): Loss from first batch
- **last_loss** (float): Loss from last batch
- **mean_loss** (float): Mean loss across batches

**Returns:**
- float or None: Normalized slope, or None if mean_loss is near zero

### collect_batch_losses(dataloader, model, loss_fn, device='cpu')

Collect loss values for each batch in a dataloader.

**Parameters:**
- **dataloader** (DataLoader): DataLoader to iterate over
- **model** (nn.Module): Model to evaluate
- **loss_fn** (Callable): Loss function
- **device** (str, optional): Device to run on. Default: 'cpu'

**Returns:**
- List[float]: Loss value for each batch

## Common Usage Patterns

### Basic Usage

```python
import torch
from dlrs import DLRSScheduler

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = DLRSScheduler(optimizer)

for epoch in range(100):
    batch_losses = []
    for batch in train_loader:
        loss = train_step(batch)
        batch_losses.append(loss.item())
    scheduler.step(batch_losses)
```

### With LossRecorder

```python
from dlrs import DLRSScheduler, LossRecorder

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
scheduler = DLRSScheduler(optimizer)
recorder = LossRecorder()

for epoch in range(100):
    for batch in train_loader:
        loss = train_step(batch)
        recorder.add(loss.item())

    scheduler.step(recorder.get_losses())
    print(f"Epoch {epoch}: Mean Loss = {recorder.get_mean():.4f}")
    recorder.reset()
```

### Saving and Loading

```python
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'epoch': epoch
}
torch.save(checkpoint, 'checkpoint.pt')

checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
scheduler.load_state_dict(checkpoint['scheduler'])
```

## Hyperparameter Tuning Guide

### For Image Classification

**Fast convergence (aggressive):**
- delta_d = 0.7
- delta_o = 1.5
- delta_i = 0.2

**Stable training (conservative):**
- delta_d = 0.3
- delta_o = 0.8
- delta_i = 0.05

**Balanced (default):**
- delta_d = 0.5
- delta_o = 1.0
- delta_i = 0.1

### For PINNs

**Recommended starting point:**
- delta_d = 0.5
- delta_o = 1.0
- delta_i = 0.1
- min_lr = 1e-6

### For Large Batch Sizes

With larger batch sizes (>256), consider:
- Reducing delta_i (e.g., 0.05) for more gradual LR increases
- Increasing initial learning rate

## Troubleshooting

### Learning rate becomes very small

**Solution:** Increase `min_lr` parameter or adjust `delta_d` to be less aggressive.

### Learning rate oscillates too much

**Solution:** Reduce `delta_o` to make adjustments less sensitive in stagnation regions.

### Training diverges early

**Solution:** Start with a lower initial learning rate or increase `delta_d` for more aggressive reduction.

### No improvement over fixed learning rate

**Solution:** Ensure you're collecting batch losses correctly and passing them to `scheduler.step()` after each epoch (not after each batch).
