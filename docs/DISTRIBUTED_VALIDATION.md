# Distributed loss policy and recovery validation

Date: 2026-10-20. This expands the original equal-batch, healthy-rank Gloo test.
It validates an **explicit caller policy on two CPU processes on one machine**.
DLRS still performs no communication, rank coordination, checkpoint persistence,
or automatic recovery. This is not multi-node, NCCL, or production acceptance.

The implementation is the illustrative
[`examples/distributed_policy.py`](../examples/distributed_policy.py), exercised by
[`tests/test_distributed_recovery.py`](../tests/test_distributed_recovery.py).
The example is not part of the installed `dlrs` package API.

## Observation and gradient contract

Every rank executes the same number of iterations, in the same order, with
identical model, optimizer, and scheduler configuration. Each local loss is a
finite, nonnegative **mean over examples**, and every local example count is
positive. For local count `n_r` and local mean `l_r`, the common observation is

```text
N = sum_r n_r
L = sum_r (n_r * l_r) / N
```

An unweighted average of the rank means would give a small rank's examples too
much weight. The test uses local batch counts `[1, 3, 2, 4]` and `[5, 2, 6, 1]`:
the counts differ both across ranks and across iterations. The tested global
means differ measurably from the unweighted rank means.

Ordinary DDP averages gradients over ranks. To obtain the gradient of the global
example mean with unequal local counts, backward uses the additional factor
`world_size * n_r / N` on each local mean. This affects training gradients; it is
separate from producing common DLRS observations. PyTorch documents that DDP
synchronizes gradients and expects equivalent optimizer updates on every rank.
See the [DDP API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html).

A deterministic reference test compares every distributed optimizer update with
a single-process update on the concatenated global batch. It compares model
parameters and loss/rate trajectories with `rtol=2e-13`, `atol=1e-15`, allowing
ordinary differences in floating-point reduction order.

The DLRS batch API then averages its **sequence of global batch observations**
according to the existing algorithm. It does not replace that sequence with a
sample-weighted epoch mean. These are different policies when global batch sizes
vary. Token means, masks, example weights, and gradient accumulation require a
separately specified and tested denominator/observation policy.

## Illustrative CPU/Gloo usage

Initialize an ordinary Gloo process group with an explicit timeout on every rank,
construct identical model/optimizer/scheduler state, and use the helper before
backward and optimizer updates:

```python
from examples.distributed_policy import require_equal_iterations, reduce_sample_mean

# The helper is an example in this repository, not an installed dlrs export.
require_equal_iterations(len(train_loader))
observations = []
for inputs, targets in train_loader:
    optimizer.zero_grad(set_to_none=True)
    local_mean = criterion(model(inputs), targets)
    observation = reduce_sample_mean(local_mean, inputs.shape[0])
    (local_mean * observation.backward_scale).backward()
    optimizer.step()
    observations.append(observation.mean)

scheduler.step(observations)  # On EVERY rank.
```

The helper deliberately accepts CPU scalar floating tensors and Gloo only. It
checks equal positive iteration counts before training. It rejects empty local
batches, nonscalar losses, invalid counts, nonfinite or negative means, and
nonfinite weighted sums. Counts are bounded so the reduced float64 denominator
is exactly representable. A failed global weighted-sum reduction also raises
on all participating ranks.

This contract assumes ordinary DDP gradient averaging without a custom
communication hook, `join()`, `no_sync()` accumulation, or changed world size.
It does not make a dynamically exhausting data loader safe. Checking the initial
iteration counts cannot prevent a later data-loading or process failure.

## Coordinated invalid-input rejection

Local validation records a failure flag and all-reduces the flags before raising.
Consequently, one healthy rank does not raise early while its peer waits forever
in the subsequent loss reduction. Tests inject NaN, a negative mean, a zero count,
and a nonscalar loss on only rank 1. Another test causes a global sum overflow
although each local observation is finite. Both ranks report the same rejection.

These tests first execute real training, so model tensors and populated optimizer
and scheduler state are present. Rejection preserves those snapshots exactly,
and a following barrier succeeds. Unequal iteration counts are separately
rejected collectively before training.

The protection is conditional: every healthy participating rank must reach the
same collective. A rank that exits or stalls cannot report its validation flag.
Also, a forward pass may already have consumed random numbers or updated model
buffers before its loss is checked. The helper does not roll back those effects.
Abort the training attempt and restore a complete agreed checkpoint; do not drop
one observation or retry only one rank.

## Epoch checkpoint restart

Two independent optimizer configurations are tested:

- SGD with momentum and weight decay, two float-LR parameter groups.
- AdamW with weight decay, two scalar float64 tensor-LR groups, `foreach=False`.

Each experiment uses three fresh two-rank process groups: uninterrupted training
for four epochs; training for two epochs and saving; and resumption in fresh
processes for epochs three and four. There are four synchronized iterations per
epoch. The model uses dropout, so each rank's RNG state matters.

Each rank's checkpoint contains model, optimizer and full scheduler state, its
CPU RNG state, the next epoch, and the completed observation/rate history. Data
is generated from explicit epoch/iteration/rank seeds, making the position
reconstructible. On restore, model, optimizer and scheduler are constructed
first, then their checkpoint states and the rank's RNG state are restored.

The resumed result must equal the uninterrupted result **exactly** for every
model tensor, optimizer slot, scheduler field, observation/rate history, next
epoch, and rank-local RNG state. Model, optimizer and scheduler state must also
agree exactly across ranks at each epoch. Tensor learning-rate objects retain
their identity while scheduling within a process.

This is epoch-boundary recovery from a complete local checkpoint. The tests do
not validate atomic multi-file publication, durable storage, a torn checkpoint,
remote storage, data-loader worker queues, AMP scalers, arbitrary mid-epoch
stochastic recovery, or automatic elastic restart after a rank failure.

## Rank failure, timeout, and launcher cleanup

Fault probes inject two failures after a successful startup barrier, immediately
before observation validation:

- Rank 1 exits without cleanup using exit code 17. Rank 0 must receive a
  communication error before any optimizer or scheduler mutation.
- Rank 1 stalls for eight seconds. The process group has a four-second operation
  timeout; rank 0 must report a communication error within fifteen seconds and
  retain its pre-call state.

Every launch has a separate 45-second parent deadline, including startup and
shutdown. These are process-monotonic-clock deadlines under the tested execution
conditions, not unconditional wall-clock response guarantees. System suspend can
pause measured timers; the shared host entered sleep during this validation
session. The parent terminates any remaining children, joins them, and escalates
to process kill if necessary. Each worker limits intra-op and inter-op Torch
threads to one; there are never more than two child ranks running in this suite.
No collective is attempted to continue training on a failed process group.
PyTorch exposes process-group operation timeouts and documents explicit cleanup;
see [distributed initialization and shutdown](https://docs.pytorch.org/docs/2.14/distributed.html).

These probes establish bounded behavior for the tested failure locations on
local Gloo. They do not establish fault tolerance during backward, optimizer
mutation, checkpoint writes, network partitions, hardware faults, or NCCL's
asynchronous execution. The fresh-process checkpoint test and the failure tests
are separate experiments; no automatic recovery orchestrator is being claimed.

## Results and reproduction

The initial focused run on macOS arm64, Python 3.11.12 and PyTorch 2.14.1 passed
all **11 distributed tests in 65.06 seconds**. Its retained raw JSON records
scheduler SHA-256 `7dbb75c149b20d479410f7abba19103123ea1969bd43023cdbbaf04fc92b4fd7`,
before the later tensor initial/base metadata alias correction. The two optimizer
restart experiments matched exactly in that run.
The rank-exit and stalled-rank probes reported errors within their asserted
fifteen-second monotonic-clock bounds; their measured communication times are in
the raw JSON. Those timings are not uninterrupted wall-clock guarantees across
host suspension.
Ruff passed for the example and test, and the package passed mypy after the fix.

The corrected scheduler SHA-256
`ecd98f7d379af9cdf68492ee56d5d6248d526df2605ffa80d3b19d18ef8ddae7` was subsequently
covered by all 11 distributed cases in the 315-case primary suite and the
six-cell [compatibility matrix](COMPATIBILITY_VALIDATION.md). The original
focused-run evidence is preserved rather than relabeled as a measurement of
the final source. See the [reliability audit](RELIABILITY_AUDIT.md#follow-up-validation)
for the later tensor metadata correction and primary-suite evidence.

The deeper tensor-LR restart test exposed one additional scheduler defect:
`load_state_dict()` converted tensor `base_lrs` metadata into Python floats.
Training numerics were unchanged in the reproduction, but complete checkpoint
state did not round-trip. The loader now clones and preserves the original
metadata types. A dedicated unit regression checks tensor dtype and ownership,
and the fresh-process AdamW experiment now passes the complete-state comparison.

Run in a clean environment containing the package's test dependencies:

```bash
python -m pytest tests/test_distributed_recovery.py -q \
  --junitxml=.audit/distributed-validation-junit.xml
ruff check examples/distributed_policy.py tests/test_distributed_recovery.py
```

The accompanying [`distributed-validation-local.json`](distributed-validation-local.json)
records the actual local environment and test results. This report's coverage is
limited to CPU/Gloo on macOS arm64, two ranks, full-precision eager execution,
the two optimizers above, and the stated model/data/checkpoint policy. Larger
world sizes, multiple machines, Linux/Windows distributed behavior, CUDA/NCCL,
MPS distributed execution, FSDP/ZeRO, mixed precision, and different framework
recovery paths remain unverified by this suite.
