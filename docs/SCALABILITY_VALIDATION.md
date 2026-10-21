# Local scheduler scaling measurements

On 2026-10-20, 24 small eager-execution cases exercised DLRS with different optimizer structures, observation counts, and learning-rate storage. These measurements expose scaling costs; they do **not** certify large-model reliability, establish a latency SLO, or show constant overhead.

The [measurement harness](../benchmarks/scheduler_scale_validation.py) and [raw JSON](scheduler-scale-local.json) contain all samples, source hashes, environment details, and Python allocation measurements. This extends the earlier [ordinary/threshold overhead probe](scheduler-overhead-local.json): the new focus is parameter-tensor traversal, tensor learning rates, accelerator scalar extraction, and retained/checkpoint memory.

The reported measurements were refreshed after the compatibility correction that clones tensor LR metadata during construction and checkpoint restoration. The final run completed 24 cases in 40.06 seconds and records scheduler source hash `ecd98f7d379af9cdf68492ee56d5d6248d526df2605ffa80d3b19d18ef8ddae7`. The [earlier raw measurement](scheduler-scale-pre-metadata-fix.json) remains unchanged with its original source fingerprints. These unpaired, contended runs cannot attribute timing differences to that correction.

## Method and limits

The machine used Python 3.11.12, Torch 2.14.1, macOS 26.6.2 arm64, and the Apple M3 MPS accelerator. CPU work used two Torch threads. CUDA was unavailable; MPS CPU fallback was unset. A separate workload study was active concurrently; the primary test suite had finished and compatibility runs paused during this refresh. The cases were collected sequentially, with five timed samples after one warmup per operation. They were **not** randomized, paired, or isolated from system contention; differences between rows cannot establish relative speed.

CPU cases use 1, 1,000, or 10,000 one-element float32 parameter tensors; one or 32 parameter groups where possible; Python-float or independent scalar-tensor LRs; and 1,000 or 100,000 Python-float loss observations. The four MPS cases use only one or 32 scalar parameters/groups and 1,000 Python-float observations. An actual SGD update precedes measurement. Construction, input allocation, and this optimizer update are excluded from the operation timings. The maximum model-value payload is only 40,000 bytes: parameter-tensor count here tests scheduler bookkeeping, not realistic model capacity or optimizer throughput.

Every case verifies that scheduler operations leave model parameter values unchanged and preserve the latest observation sequence in the checkpoint. Timings synchronize the measuring process before and after each MPS operation. The tiny MPS cases share accelerator resources with other work, so their latencies include contention and should be treated especially cautiously.

## Observed costs

These are medians from individual cases, in milliseconds. Ranges below span case medians, not confidence intervals.

| Operation and configuration | Observed median |
|---|---:|
| CPU full step, 1,000 observations, all tested structures | 0.62–4.17 ms |
| CPU full step, 100,000 observations, all tested structures | 87.86–185.03 ms |
| CPU group validation, one parameter tensor | 0.003–0.006 ms |
| CPU group validation, 10,000 parameter tensors | 0.787–3.104 ms |
| CPU checkpoint copy, 100,000 observations | 31.67–59.11 ms |
| MPS full step, 32 groups, float LRs | 1.16 ms |
| MPS full step, 32 groups, tensor LRs | 17.80 ms |
| MPS extraction of 32 LR tensor scalars | 5.33 ms |
| MPS preparation of 32 tensor LRs | 6.70 ms |
| MPS storage validation and application, 32 tensor LRs | 0.39 ms |
| MPS checkpoint copy, 32 tensor LRs | 17.08 ms |

The diagnostic operations are timed separately and overlap in work. Group validation includes LR scalar extraction; LR preparation allocates and reads candidate tensors; checkpoint copying also validates groups and clones rates. **Do not add these rows or subtract them to attribute total runtime.** Their inputs and the background load also differ over time. For example, the one-group MPS tensor-LR full-step median was 2.39 ms in this refresh versus 27.64 ms in the preserved earlier run. This variation prevents a stable accelerator cost or latency guarantee from these measurements.

Source inspection explains the relevant scaling dimensions. Ordinary scheduler steps traverse every parameter tensor to verify group identity, inspect every supplied loss, and sort tensor-LR storage ranges. Work therefore depends on parameter-tensor count, observation count, and group count; the ordinary path is approximately `O(P + N + G log G)`. Checkpoint generation also walks parameter groups and copies the latest loss history. The exact slope-threshold fallback has separate arithmetic costs covered by the earlier overhead probe. This run uses an ordinary slope, not that fallback.

Scalar tensor learning rates on MPS force host-visible reads during validation and representability checks. The diagnostic extraction measurements demonstrate real accelerator interaction, but cannot isolate a hardware-only `.item()` latency under contention. Loss observations here are already Python floats. Supplying thousands of individual accelerator loss tensors would introduce additional scalar reads; that path was not timed, and no throughput claim is made for it.

## Memory and checkpoint behavior

`tracemalloc` measures Python allocations only. It excludes native tensor storage, allocator caches, driver memory, and other processes. Model/input construction is outside the per-operation allocation measurements. Tensor payload counts are logical values, not native allocation footprints.

At 100,000 observations, step peak Python allocations were about 4.00–4.02 MB when the caller's existing float list was outside tracing. A separate retention probe allocated **new float objects** for each of eight 100,000-observation epochs and released each caller input. Retained traced memory after each epoch stayed between 3,200,336 and 3,201,768 bytes; the latest checkpoint retained exactly 100,000 losses. Peak memory during replacement reached 10,406,592 bytes because old and new histories and temporary normalized values coexist during a step. The measurement is short-run retention evidence, not a long-running native-memory leak test.

For one float-LR group and 100,000 observations, the serialized scheduler checkpoint was 901,805 bytes at **all three** tested parameter-tensor counts. With 32 tensor-LR groups it was 924,803 bytes. The checkpoint includes LR metadata and loss history, not model parameter values or optimizer momentum. The scheduler itself still keeps parameter identities proportional to parameter-tensor count. Saving a full training checkpoint adds model, optimizer, RNG, and other state beyond these scheduler-only measurements.

The plateau wrapper retained exactly 10 metrics after 1,000 updates with `patience=8`. Core DLRS retains one complete supplied epoch and **does not enforce a fixed observation cap**. Applications with long or streaming epochs must bound the supplied sequence themselves; `LossRecorder(max_losses=...)` can reject excess observations explicitly. The latest-epoch policy prevents history from accumulating across calls, but does not make memory usage independent of epoch length.

## Reproduction

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m benchmarks.scheduler_scale_validation \
  --repeats 5 --mps --concurrent-workloads --output docs/scheduler-scale-local.json
```

Omit `--mps` when unavailable and omit `--concurrent-workloads` for an otherwise idle run. Idle measurements, realistic model/optimizer states, CUDA, longer retention runs, and deployment-specific latency budgets remain separate validation work. Neither these measurements nor the passing functional checks guarantee non-regression for an unseen training workload.
