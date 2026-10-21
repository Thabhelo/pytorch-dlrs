# Critical-deployment and upstream readiness ledger

Updated 2026-10-21. This is a local decision record, not an upstream proposal or
certification. The audited package is **experimental and opt-in**. Its tests
support a bounded implementation contract; its workload results do not support
replacing arbitrary production schedulers. No critical deployment target,
production quality tolerance or operational budget has been specified.

## Completed platform and packaging evidence

[PR #2](https://github.com/Thabhelo/pytorch-dlrs/pull/2) merged into `2fa7373`.
[CI run 39562002520](https://github.com/Thabhelo/pytorch-dlrs/actions/runs/39562002520)
completed successfully on PR head
`99dfde35182a5ce2c0dc74c55f46c0b9932c0db4`. These are observed executions,
not merely configured jobs. The suite then collected 409 cases per platform;
skips are excluded from passing counts.

| Executed job | Passed / skipped |
|---|---:|
| Linux, Python 3.8 / Torch 2.0.0 | 393 / 16 |
| Linux, Python 3.9 / Torch 2.4.1 | 399 / 10 |
| Linux, Python 3.10 / Torch 2.5.1 | 399 / 10 |
| Linux, Python 3.11 / Torch 2.6.0 | 399 / 10 |
| Linux, Python 3.12 / Torch 2.14.1 | 399 / 10 |
| Linux, Python 3.12 / workflow's `latest` selection | 399 / 10 |
| Linux, Python 3.14 / Torch 2.14.1 | 399 / 10 |
| macOS, Python 3.12 / Torch 2.14.1 | 405 / 4 |
| Windows, Python 3.11 / Torch 2.6.0 | 399 / 10 |
| Separate Lightning 2.6.6 CPU integration job | 4 / 0 |
| Installed wheel, Linux Python 3.11.17 / Torch 2.6.0+cpu | 399 / 10 |
| Installed sdist, same environment | 399 / 10 |

The four common platform skips require the separately tested Lightning
dependency. Six additional Linux/Windows skips require MPS hardware. The floor
job also skips six CPU GradScaler cases because that API is unavailable.
All 11 distributed recovery cases passed in each of the nine platform jobs.
macOS CI selects loopback explicitly for its single-host Gloo tests; it is not
a multi-host network test. See the [distributed report](DISTRIBUTED_VALIDATION.md).

The [distribution job](https://github.com/Thabhelo/pytorch-dlrs/actions/runs/39562002520/job/119096468536)
tested noneditable installations outside the checkout, removed the extracted
source package before tests, and retained environments, dependency metadata and
JUnit outcomes. Downloaded archive hashes matched the recorded build artifacts:

| Artifact | SHA-256 |
|---|---|
| `pytorch_dlrs-0.3.0.dev0-py3-none-any.whl` | `8ff7188d2db35b9021de049c7957ddfe2aa7bcbfa50c9bd0672c8c8f057f5251` |
| `pytorch_dlrs-0.3.0.dev0.tar.gz` | `abc157c7b3999d83aade6ed1b80c1866fa4315b56ef001094eaa678f847e5ed9` |

Lint/type checks also passed. PyPI publication was skipped; tested artifacts are
not evidence of a published release. This closes the measured Linux/Windows and
package gap in the earlier audit. The [2026-10-20 compatibility report](COMPATIBILITY_VALIDATION.md)
keeps its original 315-case local snapshot and supplemental checks; neither that
snapshot nor the CI's 409-case count should be relabeled as a later test run.

## Completed bounded CUDA evidence

The separate [CUDA validation report](CUDA_VALIDATION.md) records an actual
installed-wheel execution at `4700e58` on a free Colab NVIDIA T4, compute
capability 7.5. Python 3.13.15 / Torch 2.11.0+cu130 / CUDA 13.0 / driver
580.82.07 executed **18 required passes and two native-BF16 unsupported skips**
in 5.07 seconds of pytest time. This is a separate 20-case targeted run, not a
rerun of the 409-case CPU/platform suite or a training-throughput benchmark.

The required cases cover eager SGD and AdamW (`foreach=False`), float and CUDA
tensor LRs, actual float32 parameter updates against an independent rate
reference, bounds/identity, FP16 overflow-skipped updates, and exact stochastic
checkpoint continuation including optimizer/scheduler/scaler and RNG state.
Four cases also compare float32/FP16 gradient accumulation with full effective
batches, including a final five-sample window split into three and two samples.
The installed package ran outside the checkout; imported runtime files matched
the supplied wheel and reviewed source, and fingerprints were unchanged after
execution. Reproduction and the explicit passing-case contract are documented
in [the free Colab instructions](FREE_COLAB_VALIDATION.md).

| Retained CUDA artifact | SHA-256 |
|---|---|
| Tested `pytorch_dlrs-0.3.0.dev0-py3-none-any.whl` | `6fc56b91067b0a2f7603ebe77c567feae969cd07155766d03a42d53df3547408` |
| Raw evidence archive | `8e26f00ab97b60969625e5e8e2d3a30194bb92aa630730d69f69283bc9852eb4` |

This wheel belongs to the subsequent CUDA run and is distinct from the earlier
PR #2 build. Native BF16 was not validated: emulated BF16 does not satisfy that
capability gate. Other GPUs, fused/capturable optimizers,
CUDA graphs, compiled optimizer/scheduler execution and GPU framework adapters
remain unverified. The test does not assess workload quality or critical-use
reliability.

## Six acceptance gates

“Complete” applies only to a stated scope. “Partial” identifies useful evidence
with open requirements. “Blocked” identifies a missing deployment decision or
failed criterion, rather than a claim that further engineering is impossible.

| Gate | Status | Requirement to close the remaining gap |
|---|---|---|
| 1. Platform, device and artifact compatibility | **Partial**; measured CPU/package and T4 checks **complete** | The declared T4 float32/FP16, float/tensor LR, accumulation and resume checks passed. Validate native BF16 on capable hardware and any additional intended GPU, driver, Torch, precision or optimizer configuration before extending support. Retain artifacts and capability skips. |
| 2. Distributed deployment and recovery | **Blocked: deployment unspecified** | Name intended world sizes, nodes, network, storage, sharding, data stream and orchestrator. Test matching observations/updates across ranks, worker and communication failures, backward and checkpoint-write failures, stale/partial checkpoint rejection and restoration from atomically published durable state. Establish application recovery invariants and time bounds. |
| 3. Model quality and training stability | **Blocked for tested failed configurations** | Define acceptable quality and failure-rate margins against the actual production scheduler before new trials. Use realistic models, data, horizon and equal tuning budgets, held-out selection boundaries and paired replications; retain every failure. Obtain workload-owner acceptance. |
| 4. Resource cost and endurance | **Blocked: target and budgets unspecified** | Predeclare steady/tail latency, host/device memory, synchronization, throughput, recovery and run-duration budgets. Measure the full training application on target hardware through representative long runs, restart cycles and realistic group/observation counts. |
| 5. Exact framework/training-mode contract | **Partial** | Scoped CPU/MPS integrations and T4 eager FP16 skip/recovery checks passed. Freeze exact deployment versions, optimizer options, precision, compilation boundaries and checkpoint timing. GPU framework adapters and different accumulation policies need new end-to-end evidence; application recovery must include model, optimizer, scheduler, scaler, RNG and data cursor/order. |
| 6. Upstream design and maintainer acceptance | **Partial preparation; acceptance outstanding** | Agree on usefulness, API, maintenance and compatibility with PyTorch maintainers; port applicable cases into the native test suite, run selected upstream CI and complete human review under the current contribution process. No upstream contact or submission has been made by this work. |

The [72-trial vision/PINN study](WORKLOAD_VALIDATION.md) failed its overall quality
criterion. CIFAR-10 was 3.762 percentage points worse than cosine on average,
and its interval did not establish the predefined margin against fixed LR.
The PINN's paired geometric-mean error ratio versus cosine was 3.114, above the
1.10 limit. MNIST and the separate [24-trial Transformer study](TRANSFORMER_VALIDATION.md)
passed their provisional mean criteria; individual degradations still occurred.
The 96 completed trials are not a universal or large-model non-regression result.
New tuning or a changed policy needs a new declared study and cannot erase these
negative outcomes.

The [scalability measurements](SCALABILITY_VALIDATION.md) and
[training-mode checks](TRAINING_MODES_VALIDATION.md) provide starting evidence for
gates 4–5. They do not supply production service limits. The free Colab execution
adds the bounded CUDA evidence above at no hardware charge. It does not
establish multi-node/NCCL behavior, FSDP/ZeRO, production storage
durability, elastic recovery, uninterrupted endurance or quality acceptance for
the failed configurations. Its passing implementation checks do not close
those deployment and quality gates.

## Supported opt-in API and compatibility decisions

The public implementation is in [scheduler.py](../dlrs/scheduler.py); usage and
integration boundaries are in the [FAQ](FAQ.md). It does not modify PyTorch
optimizers, native schedulers or their defaults when unused.

- `DLRSScheduler(optimizer, ...)` accepts ordered finite nonnegative scalar loss
  observations through `step(batch_losses)` after completed optimizer updates.
  It needs at least two observations for an adjustment. Empty/short/all-zero
  inputs do not adjust the LR; calls still advance its step bookkeeping.
- `DLRSOnPlateau.step(metric)` uses a bounded rolling window of `patience + 2`
  nonnegative metrics. `mode='max'` reverses the direction. Its `patience` is not
  a bad-epoch count, and this wrapper is not native `ReduceLROnPlateau`.
- Parameter-group membership/order is fixed. Scalar floating tensor LRs must
  be writable, detached and representable; live LR tensor identity is retained.
  Zero-rate groups remain frozen. `max_lr=None` provides no upper bound;
  selecting a cap does not prove stability.
- Restore matching model, optimizer and scheduler state, with the scheduler
  constructed before optimizer loading and scheduler loading afterward. Restore
  scaler/RNG/data state as required. DLRS does not own the training loop,
  distributed observation reduction or skipped-update policy.
- The verified Lightning adapter is restricted to one process, one optimizer,
  automatic optimization, full precision and no accumulation or competing
  scheduler. HF Trainer is explicitly disabled. SequentialLR/ChainedScheduler,
  compiled/differentiable scheduler stepping and CUDA graph capture are outside
  the supported API. Model-only compilation evidence leaves scheduler and
  optimizer eager.

Relative to 0.2.1, intentional changes include rejecting invalid/signed inputs
and malformed state, updating tiny positive observations, freezing zero LRs,
correcting `mode='max'`, bounding wrapper history, rejecting changed group
topology, and preserving tensor LR/metadata ownership. Extreme-value arithmetic
and dtype bounds are corrected. The framework restrictions and benchmark/PINN
corrections also change prior behavior. Review the [audit migration list](RELIABILITY_AUDIT.md#breaking-changes-requiring-review)
before upgrading; accepted legacy checkpoints still need valid values and
matching groups. This is not a drop-in promise for every earlier use.

## Mapping to native PyTorch scheduler tests

Read-only upstream review used PyTorch main commit
`f54c229e83e1eed063cca5f4abad92b749c8649d`, retrieved 2026-10-21. The native
[scheduler implementation](https://github.com/pytorch/pytorch/blob/f54c229e83e1eed063cca5f4abad92b749c8649d/torch/optim/lr_scheduler.py)
and [scheduler tests](https://github.com/pytorch/pytorch/blob/f54c229e83e1eed063cca5f4abad92b749c8649d/test/optim/test_lrscheduler.py)
are reference contracts, not executed upstream acceptance of DLRS. The current
package suite runs against installed PyTorch; DLRS has not been inserted into
that source tree or its native test parametrizations.

| Native test area | Existing DLRS evidence | Upstream integration work still required |
|---|---|---|
| Initial rates, `last_epoch`, `get_last_lr` | Initialization/get-last-rate tests in [test_scheduler.py](../tests/test_scheduler.py); independent tensor metadata in [test_scheduler_safety.py](../tests/test_scheduler_safety.py) | Add DLRS cases to the upstream harness and agree on initialization and resume semantics. |
| `test_constant_initial_lr`; tensor type preservation | Tensor identity, independent `base_lrs`/`initial_lr`, alias/dtype/rounding regressions | Port these cases with native assertion utilities and the chosen device/dtype matrix. |
| Scheduler state save/load and checkpoint tests | Atomic invalid-state rejection and legacy compatibility; real `weights_only=True` continuation in [training modes](../tests/test_training_modes.py) and [distributed recovery](../tests/test_distributed_recovery.py) | Include both public APIs in native serialization parametrization; specify the versioned state schema and migration policy. |
| Optimizer/scheduler ordering and overridden-step warnings | Training loops step optimizers before DLRS; skip/accumulation policies are exercised | Decide warning behavior for this metric-driven override; current DLRS does not reproduce all base `step()` warnings. |
| Sequential/chained composition and dynamic groups | Fixed topology is enforced; composition is documented unsupported | Native automatic composition cannot supply batch observations. Agree on explicit exclusion or a reviewed interface; do not silently register it as a general chainable scheduler. |
| Native expected learning-rate trajectories | Independent high-precision oracle, boundaries and randomized trajectories in [training contracts](../tests/test_training_contract.py) | Add the DLRS-specific mathematical contract and agreed equality/zero-loss cases; native schedulers' formulas are not DLRS oracles. |

PyTorch's [optimizer documentation](https://docs.pytorch.org/docs/2.14/optim.html#how-to-adjust-learning-rate)
places scheduler calls after optimizer updates. That is the intended DLRS use.
However, its loss argument replaces the ordinary optional epoch argument;
`get_lr()` returns current-rate snapshots; checkpoint loading also restores
optimizer rates; and the scalar wrapper is not an `LRScheduler` subclass.
These are explicit design differences for maintainer review, not proof of full
native interchangeability. No existing PyTorch scheduler behavior or default
should be changed to make DLRS fit.

## Upstream decision and human review

Before any upstream work, a human owner must decide whether the evidence and
maintenance burden justify a new scheduler, explain the unmet user need, and
review the code and negative workload results. PyTorch's
[contribution guide](https://docs.pytorch.org/docs/2.14/community/contribution_guide.html)
encourages discussion of new features and extra scrutiny for core changes.
Passing this repository's CI is not maintainer agreement or acceptance.

The reviewed [CONTRIBUTING workflow](https://github.com/pytorch/pytorch/blob/f54c229e83e1eed063cca5f4abad92b749c8649d/CONTRIBUTING.md#issue-and-pr-workflow)
allows PR triage with an `actionable` issue, repository write access, or named
maintainer preapproval. For an ordinary external contribution, obtain the
applicable maintainer decision first. Its new-feature guidance excludes
AI-generated solution explanations from the initial problem issue. This ledger
is internal preparation and should not be pasted into an upstream issue.

The reviewed [AI policy](https://github.com/pytorch/pytorch/blob/f54c229e83e1eed063cca5f4abad92b749c8649d/AI_POLICY.md)
requires meaningful human responsibility, review and disclosure of AI-generated
material; it rejects fully autonomous contributions. A human must understand
and stand behind any proposed change, confirm the current contribution/CLA
requirements, and review the minimal implementation, tests and documentation.
Recheck upstream rules at submission time. No issue, message, PR, push or PyPI
release is authorized by this readiness assessment.
