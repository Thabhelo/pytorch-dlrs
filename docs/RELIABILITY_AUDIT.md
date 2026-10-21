# Reliability audit and revision

Date: 2026-10-20. Repository: `Thabhelo/pytorch-dlrs`.
Reviewed baseline: `c0ea1d159e4caa4950358282506e9358ae9a67ea` (0.2.1).
Revision: local branch `codex/dlrs-reliability-audit`, planned 0.3 development version.

## Decision

**The baseline was not suitable for a critical-use or PyTorch-upstream readiness claim.**
This revision fixes reproducible implementation defects and substantially expands
verification. It remains an experimental, opt-in training policy. Passing the
checks below does not establish universal accuracy, convergence, non-regression,
distributed fault tolerance, or large-scale production suitability.

**The completed workload study does not support adopting this revision as a
general replacement for an existing production scheduler.** All 96 final
scientific trials completed, but CIFAR-10 and the manufactured PINN failed their
predefined quality gates against cosine. CIFAR-10 averaged 68.586% accuracy for
DLRS versus 72.348% for cosine; its interval also failed to establish the required
tolerance against fixed LR. The PINN's paired geometric-mean error ratio versus
cosine was 3.114, above the 1.10 limit. MNIST and the small Transformer met their
provisional margins; that is not a claim of zero degradation in every run.

No changes to PyTorch itself, GitHub publication, or PyPI release were performed
as part of the 2026-10-20 local validation. The configured CI matrix is not evidence of
successful remote runs.

## Findings and changes

| Priority | Baseline defect and consequence | Revision and evidence |
|---|---|---|
| High | Nonfinite losses and parameters passed validation; `[1.0, NaN]` set an optimizer LR to NaN. A later failing group could leave earlier groups changed. | Validate finite inputs, configuration, group structure and every proposed rate before applying any rate. Regressions exercise invalid input, malformed checkpoints, tensor representability, and atomic rejection. |
| High | FAQ instructed only rank 0 to step DLRS in DDP. Each rank has its own optimizer; differing LRs can make replicas diverge. | Removed this instruction. Document explicit shared loss observations and stepping on every rank. A two-process Gloo test checks identical model parameters and LRs across five epochs. DLRS itself performs no distributed collectives. |
| High | `DLRSOnPlateau(mode='max')` ignored `mode`; improving metrics reduced the rate as though they were worsening losses. | Reverse the normalized slope before choosing the factor in max mode. Preserve and clarify the historical rolling-window interpretation of `patience`, rather than claiming PyTorch plateau semantics. |
| High | Checkpoint loading accepted invalid configuration and mismatched group counts using truncating `zip`; returned mutable state could alias the scheduler. | Versioned, copied state; complete validation before mutation; group-count checking and actual parameter-identity checks for live topology. Resume tests compare subsequent losses and final model tensors exactly for SGD, Adam and AdamW, with both scheduler APIs. |
| High | Lightning restoration could discard state loaded before scheduler creation, and examples allowed competing or unsynchronized scheduler updates. HF used stale aggregate log records as batch observations and an ineffective restore hook. | Lightning defers restore, checks ownership and its supported training envelope, bounds its retained losses, and has a real CPU epoch-checkpoint resume test. The unsafe HF callback now fails explicitly with an explanation. |
| Medium | A zero LR could be raised to `min_lr`, unexpectedly unfreezing a parameter group. Tensor LRs were replaced by Python values. | Zero LRs stay frozen; floating scalar tensor LRs keep their object identity. Unsupported tensor forms and unrepresentable results fail before mutation. |
| Medium | A fixed `1e-10` mean-loss cutoff contradicted scale invariance, and summing very large losses could overflow. Rounded `log10` at a decade boundary could choose the wrong adjustment scale. | Scale-aware arithmetic, all-zero handling, and checked decade selection. Independent high-precision formula comparisons, threshold and floating-boundary tests, and loss rescaling from `1e-300` through `1e300`. |
| Medium | The scalar wrapper retained every metric forever and aliased its history in checkpoints. | Retain at most `patience + 2` metrics and copy state. A 10,000-step test checks bounds and bounded history. The batch API and LossRecorder still use O(B) memory per retained epoch. |
| Medium | `collect_batch_losses` permanently switched the model to evaluation mode, including when evaluation raised. | Restore each submodule's original training flag in a `finally` block, preserving mixed training/evaluation states. |
| Medium | The Helmholtz example claimed `sin(pi*x)` solves `u'' + u = 0`; it does not. | Use the manufactured forcing `(1-pi**2)*sin(pi*x)`, verify analytic residual/boundary values, preserve parameter gradients, and test a short training run. |
| Medium | Benchmark used held-out test loss to drive ReduceLROnPlateau, evaluated the fixed-rate baseline twice, did not pair random seeds, and weighted unequal batches equally. Failures could still end with a successful completion message. | Separate training/validation/test sets; one validation per epoch and one final test; paired seeded trials; sample-weighted metrics; accelerator synchronization for wall time; explicit failure records and nonzero exit on incomplete runs. Synthetic tests cover accounting and deterministic repeated training. |
| Medium | Duplicate package metadata/dependency lists and unconstrained overlapping CI matrices obscured the installation/support contract. Documentation claimed universal compatibility and production readiness without evidence. | One package metadata source, torch-only runtime dependencies, separate development/example dependencies, boundary-version CI, installed wheel/sdist smoke checks, and narrowed claims with migration notes. |

## Baseline reproduction

The original suite passed **59 tests** on Python 3.11.12 / PyTorch 2.14.1 while
these independent probes still reproduced defects:

```text
DLRSScheduler.step([1.0, NaN])            -> optimizer LR becomes NaN
mode='max', metrics 0.5 then 0.8          -> LR 0.1 becomes 0.05384615384615385
initial LR 0, losses [1.0, 0.5]          -> LR becomes 1e-8
losses [1e-20, 5e-21], initial LR 0.1    -> no adjustment due to cutoff
checkpoint param_group_lrs=[]            -> silently accepted for one group
collect_batch_losses(training_model, ...) -> model.training becomes False
```

## Algorithm fidelity and its limits

The update matches the paper's normalized first-to-last batch-loss change and
decade-scaled adjustment. The paper is internally inconsistent at slope exactly
one: its Section 3.2 piecewise expression includes equality in the stagnation
case, while Algorithm 1 uses a strict inequality. This implementation keeps
`0 <= slope <= 1`, consistent with its previous behavior and Section 3.2.
See the [original paper](https://arxiv.org/html/2507.21749v1).

The numerical guardrails are implementation choices: finite nonnegative loss
observations, frozen zero LRs, explicit bounds, rejection of invalid state, and
all-zero no-op behavior. The epoch-metric wrapper is a repository extension,
not evidence that the paper evaluated this wrapper. Signed objectives/metrics
are deliberately unsupported; changing the sign of the mean otherwise reverses
the intended slope interpretation.

There cannot be a guarantee that substituting DLRS improves every training run.
The endpoint losses come from different minibatches, so their difference can
reflect batch difficulty or order rather than progress at fixed data. An LR
increase can worsen the next update even on a convex quadratic: for
`L(theta)=theta**2/2`, `theta=1`, a fixed LR of `1.9` produces loss `0.405`.
The DLRS observation `[2, 1]` increases that LR to about `1.9667`, producing
loss about `0.4672` on the same next gradient. This is a counterexample to a
universal non-regression claim, not an evaluation of typical DLRS performance.
Bounds constrain LR values; they do not certify training stability.

## Initial revision verification

| Check | Environment | Result |
|---|---|---|
| Full suite, including actual two-process Gloo test | macOS arm64, Python 3.11.12, PyTorch 2.14.1 | **263 passed, 1 skipped**; core line coverage **93%** |
| Full suite at declared PyTorch minimum | macOS arm64, Python 3.10.20, PyTorch 2.0.0 | **263 passed, 1 skipped**; only upstream TypedStorage deprecation warnings |
| Optional real Lightning test, run separately | Python 3.11.12, PyTorch 2.14.1, Lightning 2.6.6, CPU | **1 passed**; exact continuation across fresh processes |
| Ruff across package, tests, examples, benchmarks and setup shim | Ruff 0.17.1 | Passed |
| Mypy package check | Mypy 1.19.1, Python 3.11 checking target | Passed, 3 source files |
| Distribution build and metadata | build / twine; development version 0.3.0.dev0 | Wheel and source distribution passed |
| Installed-artifact smoke | Clean Python 3.11 environment with torch 2.14.1; outside checkout | Both artifacts import, contain `py.typed`, expose matching versions and torch-only runtime requirements, and pass a numerical/checkpoint smoke |
| CI definition | Local YAML and embedded shell syntax checks; official CPU wheel availability checks | Passed; remote jobs not executed |

The one skip in each full suite is the real Lightning test, because the base
test environments do not install that optional framework. It was explicitly
run and passed in its own environment; a dedicated pinned CI job now runs it.
The initial pass did not execute Python 3.8 itself or accelerator training.
The follow-up validation below adds actual floor-version, MPS, distributed
recovery, and training-mode evidence. Other OS coverage must not be inferred
from macOS runs.

Commands used, with the corresponding environment's Python on PATH:

```bash
python -m pytest tests --cov=dlrs --cov-report=term-missing -q
python -m pytest tests -q
python -m pytest tests/test_integrations.py -k real_lightning -q
ruff check dlrs tests examples benchmarks setup.py
mypy dlrs
python -m build
python -m twine check dist/*
python -m benchmarks.scheduler_overhead --output docs/scheduler-overhead-local.json
```

The suite includes independent equation checks, deterministic model training,
checkpoint serialization/restoration, numerical extremes, malformed input/state,
multi-group and tensor behavior, utility side effects, example contracts,
benchmark accounting, and actual two-process Gloo execution.

The real Lightning check runs three fresh processes: uninterrupted three-epoch
training, one-epoch training with a checkpoint, and resumption to epoch three.
It compares model tensors, LRs, and complete scheduler state exactly on a fixed
deterministic CPU dataset. Test doubles separately exercise rejected settings;
those doubles do not establish framework integration compatibility.

### Local scale measurements

The reproducible [scheduler overhead script](../benchmarks/scheduler_overhead.py)
was run on macOS arm64 / Python 3.11.12 / PyTorch 2.14.1 with five timed steps
per configuration. [Raw results](scheduler-overhead-local.json) record 10, 1,000
and 100,000 observations and 1 or 100 float-LR groups. At 100,000 observations
and 100 groups, the observed median was approximately **55 ms per ordinary step**
and **194 ms at the slope-one boundary**, with about **3.82 MiB of newly traced
Python allocations** during a separate step. Boundary handling uses an exact
comparison to prevent selection of the wrong factor.

These are illustrative local scheduler measurements, not GPU training overhead
percentages or latency guarantees. Input creation, optimizer construction,
pre-existing scheduler state, tensor storage, training, accelerator transfers,
and networking are outside the timing/allocation figures. Test repeated runs
on the intended deployment hardware. Validation scans observations and parameter
tensor identities; tensor-LR alias checking additionally sorts address ranges.
The memory and runtime do not remain constant as an epoch or optimizer grows.

## Follow-up validation

The follow-up on the same date expands the initial revision rather than treating
configured CI jobs or synthetic smoke tests as executed deployment evidence:

The corrected primary-environment suite passed **315 tests with zero skips** in
146.80 seconds (macOS arm64, Python 3.11.12, PyTorch 2.14.1, Lightning 2.6.6).
Core line coverage was **93%**: scheduler 94%, utilities 91%. Nine warnings
came from upstream Lightning deprecation/performance notices. This full run
includes the tensor-metadata corrections, strict checkpoint type comparison,
and deterministic and stochastic MPS restart checks described below. Its JUnit
record is `.audit/final-fixed-validation.xml`; the scheduler source SHA-256 is
`ecd98f7d379af9cdf68492ee56d5d6248d526df2605ffa80d3b19d18ef8ddae7`.
Ruff 0.17.1 and mypy 2.4.0 also passed the package checks. Coverage percentages
are line coverage, not a probability of correctness or a certification.

- [Distributed validation](DISTRIBUTED_VALIDATION.md): 11 actual two-process
  CPU/Gloo tests, including sample-weighted observations and gradients for
  unequal local batch sizes, complete fresh-process SGD/AdamW restart, coordinated
  invalid-input rejection, and bounded rank-exit/stall errors. These cover the
  documented caller policy on one machine. They do not provide an elastic or
  durable-checkpoint recovery system.
- [Training-mode validation](TRAINING_MODES_VALIDATION.md): 23 actual integration cases plus a comparator regression for
  weighted accumulation, CPU bf16 and skipped GradScaler updates, stochastic RNG
  checkpoint continuation, MPS float/tensor learning rates, compiled models with
  eager scheduling, and real Lightning rejection of unsupported modes.
- [Compatibility validation](COMPATIBILITY_VALIDATION.md): exact interpreter and
  framework versions, results and skips, plus tests against installed artifacts.
- [Workload comparison](WORKLOAD_VALIDATION.md): a frozen equal-budget tuning and
  five-seed confirmation protocol, all 72 trials completed, with every successful
  and failed trial retained. MNIST passes its provisional margin; CIFAR-10 fails
  both baseline gates, and the PINN fails against cosine. Independent recomputation
  of all raw records confirms selection, pairing, summary arithmetic and the
  negative acceptance results. All 16 dataset files and five frozen source files
  passed a final integrity check.
- [Transformer comparison](TRANSFORMER_VALIDATION.md): 24 fixed-budget runs,
  with all five paired confirmation seeds and both provisional NLL criteria
  passing. One seed was slightly worse than fixed-rate training; mean acceptance
  is not every-run non-regression. A separate raw-record recomputation confirms
  the means, selected rates, pairing, and interval arithmetic.
- [Ordinary-use differential check](DIFFERENTIAL_COMPATIBILITY.md): 20,000
  float learning-rate comparisons against the original revision matched within
  `rtol=1e-12`, `atol=1e-15`, across 1,000 eight-epoch trajectories. This excludes
  the intentional fixes and does not establish training-quality equivalence.
- [Scalability validation](SCALABILITY_VALIDATION.md): bounded synthetic optimizer
  topology and tensor-rate measurements. Local measurements do not establish a
  production latency or memory service-level objective.

The deeper distributed checkpoint comparison found and fixed a further defect:
loading a checkpoint converted tensor `base_lrs` metadata to Python floats.
The loader now retains independent copies with the original types. Training
numerics were unchanged in the reproduction, but the complete checkpoint now
round-trips. Testing the actual Python 3.8 floor also replaced newer-Python-only
floating-point helpers in the safety tests with exact hexadecimal constants.

The expanded PyTorch 2.0 run then exposed a second tensor-metadata issue: the
older superclass aliases `initial_lr` and `base_lrs` to the live LR tensor.
In-place rate updates consequently changed the purported initial values and
broke complete-state restart comparisons. Construction now detaches those
metadata tensors; checkpoint loading also detaches aliases reintroduced by an
older optimizer checkpoint before applying restored rates. The saved metadata
value is preserved; an already overwritten historical initial value cannot be
reconstructed. The previously failing two-rank tensor-AdamW restart and both old
and current Torch scheduler regressions pass after the correction. The workload
study was restarted under this corrected source fingerprint, retaining the
interrupted development records separately.

Native probes without any DLRS import reproduced optimizer post-hooks ceasing
to fire after model compilation on PyTorch 2.0.0, and MPS random/dropout state
failing exact replay on the tested PyTorch 2.0.0 and 2.4.1 builds. Full-precision tests now count their
explicit successful optimizer calls; post-hooks are reserved for actual scaler
skip detection. Stochastic MPS restart is now conditional on an explicit native replay check,
with the capability recorded in JUnit. A failed precondition skips only that
unverified continuation coverage; it is not a successful restart. Current Torch
passes the corresponding native probes and continuation tests.
These are framework limits, not reasons to claim broad passing coverage.

The final suite contains **384 cases**: the 315-case compatibility snapshot,
20 Transformer protocol-validity cases and 49 independent-summary verifier
regressions. The verifier rejects omitted/duplicate baseline comparisons,
truncated intervals, mismatched budgets or identities, tuning/test leakage,
and inconsistent counts or arithmetic; its checks remain active under
`python -O`. It verifies both completed scientific studies without changing
their outcomes. The additional tests do not retroactively increase the executed
315-case compatibility rows. Exact final wheel/source-archive hashes and
installed-artifact outcomes are retained separately in
`.audit/distribution-validation/`, as described in the compatibility report;
embedding an archive's own final hash in its contents would change that hash.

## Breaking changes requiring review

- Invalid/nonfinite/negative observations and malformed state now raise errors.
  Tiny positive losses now produce updates instead of silently skipping them.
- Zero LR groups stay frozen, and live parameter-group topology changes are
  rejected. Resume must retain parameter/group ordering. Same-sized groups
  reordered across different program runs cannot be identified automatically.
- `mode='max'` now changes behavior correctly. Scalar history is bounded.
  `patience` remains a rolling observation-window parameter, not a count of bad
  epochs. This wrapper is not a drop-in ReduceLROnPlateau implementation.
- Tensor LR storage is preserved; unsupported tensor representations fail
  explicitly. Tensor checkpoint values remain subject to the target dtype's
  representability and precision.
- The HF callback is disabled. Lightning rejects distributed, accumulated,
  mixed-precision, manual-optimization and competing-scheduler configurations.
- Benchmark JSON now contains `results` and `failures`, validation trajectories,
  and one final test measurement. The PINN example now solves the stated
  manufactured problem. Neither change is directly comparable to old outputs.

## Remaining critical-deployment and upstream gates

**Status update, 2026-10-21:** the Linux, Windows and installed-package portion
of gate 1 is now **complete for the measured matrix**. After the dated local
validation above, [PR #2](https://github.com/Thabhelo/pytorch-dlrs/pull/2) merged
into `2fa7373`. Its [final CI run](https://github.com/Thabhelo/pytorch-dlrs/actions/runs/39562002520)
tested head `99dfde35182a5ce2c0dc74c55f46c0b9932c0db4`: all nine platform
cells passed, including all 11 distributed recovery cases in every cell.
Each wheel/sdist installation passed 399 tests with 10 explicit skips, with
the checkout's importable package removed; the separate Lightning job passed
four tests. Publishing to PyPI was skipped. These results supersede the earlier
unexecuted CI gap without rewriting the historical local reports.

A subsequent [installed-wheel CUDA run](CUDA_VALIDATION.md) at `8358f1b` passed
all 18 required checks on a free Colab NVIDIA T4, with two native-BF16 cases
skipped because the device does not support that capability. The measured
environment was Python 3.13.15, Torch 2.11.0+cu130, CUDA 13.0 and driver
580.82.07. Float32 training, float/tensor LR storage, FP16 skipped-update handling
stochastic checkpoint continuation and sample-weighted accumulation with a
partial final batch passed.
See the [reproduction instructions](FREE_COLAB_VALIDATION.md) and
[readiness ledger](UPSTREAM_READINESS.md) for counts,
artifact identities, native PyTorch contract mapping and acceptance requirements.

| Gate | Current status | Evidence and remaining acceptance |
|---|---|---|
| 1. Platforms, devices and package artifacts | **Partial**; measured CPU/package and T4 checks complete | Linux, Windows, macOS, wheel/sdist and one T4 installed-wheel configuration passed their declared checks. Native BF16 was unsupported, and other GPUs/precision/optimizer combinations and deployment hardware remain unverified. |
| 2. Deployment distribution and recovery | **Blocked: target unspecified** | Two-rank, single-host Gloo tests cover the explicit example policy. Specify and validate the actual world sizes, network, storage, data stream, sharding and recovery orchestration, including backward/write failures and durable checkpoint publication. Multi-node, NCCL, FSDP/ZeRO and elastic operation remain unverified. |
| 3. Workload quality | **Blocked for the failed configurations** | CIFAR-10 and the manufactured PINN failed their predefined comparison with cosine; CIFAR-10 also did not establish its margin against fixed LR. MNIST and the small Transformer passed provisional margins. Require a separately predeclared comparison against the intended production scheduler at realistic scale and horizon; preserve the negative results. |
| 4. Deployment performance and endurance | **Blocked: budgets and target unspecified** | Local scalability measurements are exploratory. Set and measure latency, total host/device memory, synchronization, recovery and endurance limits on target hardware. O(B) epoch observations and parameter/group validation costs remain relevant; Python allocation measurements are not total memory or an overhead guarantee. |
| 5. Framework and checkpoint integration | **Partial** | Explicit eager training, supported Lightning configurations and bounded T4 FP16 skip/recovery cases have evidence. Validate exact deployment versions and application checkpoint boundaries; the recorded accumulation policy passed. Compiled-model tests keep optimizer/scheduler eager. CUDA graphs, compiled/differentiable stepping, SequentialLR/ChainedScheduler, HF Trainer and arbitrary mid-epoch stochastic Lightning recovery are outside the supported scope. |
| 6. Upstream compatibility and acceptance | **Partial preparation; review outstanding** | Documented opt-in API and migration boundaries are ready for review. A native PyTorch implementation/test integration, maintainer agreement on usefulness and API, and meaningful human review remain necessary. No PyTorch push, issue or submission is authorized by this work. |

Here “complete” applies only to the stated evidence scope; “partial” means some
requirements are verified; “blocked” identifies an unmet decision or failed
acceptance criterion. No actual critical deployment target or production
tolerances have been supplied. A free single-GPU notebook cannot close the
network, storage, sharding, endurance or failed workload-quality gates.

Packaging maintenance: the modern local setuptools backend emits a deprecation
warning for the existing TOML license-table form. Builds and metadata checks
pass. The table is retained for source builds with older Python-compatible
backends; revisit the backend/Python-floor strategy before that form is removed.

PyTorch documents its [optimizer/scheduler ordering and state behavior](https://docs.pytorch.org/docs/stable/optim.html),
[DDP execution model](https://docs.pytorch.org/docs/stable/notes/ddp.html),
[reproducibility limitations](https://docs.pytorch.org/docs/stable/notes/randomness.html),
and [contribution process](https://docs.pytorch.org/docs/stable/community/contribution_guide.html).
Those contracts informed this review; they do not certify this third-party package.
