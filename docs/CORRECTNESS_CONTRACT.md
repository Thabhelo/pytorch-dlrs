# DLRS correctness contract

Contract version 1, 2026-10-21. This specifies the supported behavior of
`DLRSScheduler` and `DLRSOnPlateau` in this repository. It is a reviewable target,
not a claim that the entire implementation has been formally verified. The
[audit report](CORRECTNESS_VERIFICATION.md) distinguishes mathematical proofs,
executed tests, deliberate-defect detection, and remaining assumptions.

## Scope and assumptions

The caller owns the model, optimizer updates, loss observations and complete
training checkpoint. DLRS changes learning rates only when explicitly called.
It does not synchronize workers, detect a skipped optimizer update, validate
model quality, or select an acceptable training policy.

Calls are sequential. Ordinary PyTorch optimizer groups and accessible scalar
values are assumed; concurrent mutation, malicious Python scalar/tensor
subclasses, process termination, allocation failure and device/driver failure
during writes are outside the failure-atomicity guarantee below. Supported
tensor-rate behavior is qualified by device and dtype tests, not by the mere
existence of a floating dtype in PyTorch. The CPU numeric suite exercises
float16, bfloat16, float32 and float64; recorded CUDA evidence has its own scope.

These assumptions are obligations for an application review. They do not become
true merely because a test suite passes. In particular, validation before writes
is not a durable transaction or a recovery guarantee after a hardware failure.

## C1. Scalar and configuration domain

- Observations are finite, nonnegative real scalars. Booleans, complex values,
  strings and nonscalar tensors are rejected. A tensor observation is detached
  and read as a scalar; no computation graph is retained.
- Negative real inputs are rejected **before** conversion to binary64, including
  exact negative fractions that would round to `-0.0`. Signed floating zero is
  accepted as zero. Accepted wider/exact real values are subsequently converted
  to Python floats; positive values can round, including underflow to zero.
  Subsequent numerical statements refer to those converted values.
- Learning rates are finite and nonnegative. Tensor rates must be writable
  floating scalar tensors, detached from gradients and outside inference-tensor
  restrictions. Zero-rate groups are frozen, exempt from the positive lower bound.
- `delta_d`, `delta_o`, `delta_i` and `min_lr` are finite and strictly positive.
  `max_lr` is absent or finite and strictly greater than `min_lr`.
- `last_epoch` is an integer at least -1; booleans are not integers for this API.
  `verbose` is a boolean. Wrapper `mode` is `min` or `max`; `patience` is a
  nonnegative integer, excluding booleans.
- A finite ordered iterable is required for an epoch. A caller iterator may raise
  its own exception; partial consumption must not commit a partial scheduler step.
  An infinite iterator is outside the bounded-execution contract.

## C2. Exact mathematical reference

For converted losses `x[0] ... x[n-1]`, `n >= 2`, total `T > 0`, and direction
`q = +1` for the ordinary scheduler / minimum-mode wrapper or `q = -1` for the
maximum-mode wrapper, define the reference over exact real values:

```text
s = q * n * (x[n-1] - x[0]) / T
d = delta_d if s > 1; delta_o if 0 <= s <= 1; delta_i if s < 0
g(a) = 10^e, where e is the unique integer satisfying 10^e <= a < 10^(e+1)
r(a) = 0 if a = 0; otherwise clamp(a - g(a)*d*s, min_lr, max_lr)
```

The upper clamp is omitted when `max_lr` is absent. Each group uses its own
current rate and scale. Relative rates between groups need not remain constant.

The source is the piecewise equation in
[section 3.2 of arXiv:2507.21749v1](https://arxiv.org/html/2507.21749v1#S3.SS2).
This package deliberately includes equality at one in the middle regime.
Algorithm 1's strict middle condition instead falls through to its final branch
at equality; the package follows the explicit piecewise equation. Bounds,
frozen groups, validation, state semantics and the metric wrapper are package
contracts in addition to that equation.

For nonnegative observations, `abs(s) <= n`. Positive multiplication of all
losses preserves the exact slope; adding a constant generally does not. Before
clamping, a positive slope decreases a positive rate and a negative slope
increases it. After clamping, nonincrease/nondecrease requires that the starting
rate already satisfy the bounds. Clamping an initially out-of-bounds rate can
move it in the other direction. These statements make no claim about loss,
gradients, convergence or eventual model quality.

## C3. Floating-point implementation contract

The mathematical reference and the executable floating-point algorithm are
separate objects. DLRS does **not** promise a correctly rounded evaluation of the
entire exact-real formula, bitwise invariance under arbitrary loss rescaling,
or a universal relative error bound near cancellation.

1. Losses are normalized by a binary power chosen from their maximum, then
   summed with `math.fsum`. This avoids the ordinary overflowing raw sum.
2. A rational calculation resolves slopes near the discontinuity at one
   (`abs(computed_slope - 1) <= 1e-15`), subnormal endpoint normalization,
   subnormal computed slopes, and a computed zero with unequal endpoints.
   The rational branch comparison is made before converting its slope to float.
   A nonzero rational slope that rounds to zero is rejected before committing
   any update. This is an explicit representability limitation.
3. The decade is selected from the exact value of the stored binary64 rate,
   using `Decimal.from_float(rate).adjusted()`. No decimal spelling is substituted
   for that value. For example, the float written `1e-6` is slightly below the
   exact decimal boundary and therefore belongs to the preceding decade.
4. On the ordinary path, the decimal power and subsequent products/subtraction
   are floating-point operations in the implementation's stated order. An
   exceptional intermediate scale/product uses a 50-digit Decimal calculation.
   That fallback uses its own round-to-nearest-even context and exponent limits;
   caller Decimal precision, traps, rounding and exponent limits must not affect
   the result or be modified. It uses the exact current float and the decimal
   round-trip spellings of the floating delta and slope.
5. Nonfinite proposed rates or adjustments are rejected before clamping; a bound
   must not silently turn an arithmetic failure into a passing update.
6. Tensor conversion must retain a finite positive value for a positive result.
   Bound rounding moves inward by a representable neighbor if needed. If no
   positive representable value satisfies the bounds, the whole update fails.
   Accepted tensor updates preserve the live tensor's identity, dtype and device.

An example of legitimate cancellation: binary64 `0.1` minus the rounded
binary64 power `0.1` is zero, while the exact-real calculation using decimal
`10^-1` can leave a tiny positive residual. Tests compare continuous arithmetic
using a declared error budget tied to operation magnitudes, and check discrete
branches separately. The sampled budget is not a proved global floating-point
error theorem. Compiler, hardware and library rounding assumptions remain part
of the trusted execution environment.

## C4. Step and failure transitions

| Input event | Rates | Diagnostics / observations | Counters |
|---|---|---|---|
| `step(None)` | Unchanged; no clamping | Empty losses, zero slope and adjustment | Advance once |
| Fewer than two losses | Unchanged; warning | Retain converted losses, zero diagnostics | Advance once |
| All-zero losses | Unchanged; warning | Retain zeros, zero diagnostics | Advance once |
| Valid active losses | Apply prepared group updates, including bounds | Retain converted losses and calculated diagnostics | Advance once |
| Rejected input, topology, arithmetic, representation or checkpoint | No additional mutation | Previous scheduler state retained | Unchanged |

Construction follows the PyTorch initial no-op step convention. Initial or
externally assigned out-of-bounds rates are not clamped until an active update.
Identical positive losses are active: zero slope still permits bound handling.
`adjustment` describes the last group's proposed pre-clamp change, not every
group's realized change. `get_lr()` returns current optimizer-rate snapshots;
`get_last_lr()` returns snapshots from the last successful scheduler step.

Validate every group and prepare every update before any rate write. The group
signature is the ordered sequence of ordered parameter identities, captured at
construction; changing it is rejected. Optimizer checkpoint loading may replace
group dictionaries while preserving that signature. Groups with indistinguishable
signatures (for example empty groups) cannot have their identity distinguished
by this mechanism; callers must retain their logical order.

Tensor storage shared across groups is permitted only when overlapping writes
have identical location, dtype and value. Conflicting or partially overlapping
writes are rejected before changing a rate. Invalid caller changes already made
to an optimizer are not rolled back; rejection guarantees no additional DLRS
mutation. Exceptions from device writes or allocation after validation are
outside the ordinary validation-atomicity guarantee.

## C5. Snapshots and restoration

Returned rates, tensor metadata and checkpoint containers must not alias live
mutable state. Mutating a returned snapshot before the next step must change
neither the optimizer's rate nor the scheduler's cached rate or base metadata.

Version-1 checkpoints require every declared field and exactly one value per
group for rate vectors. Values and representability are validated before any
live update. Accepted legacy version-0 checkpoints use documented defaults;
unknown versions are rejected. Checkpoints describe state, not an authenticated
history: the loader does not prove that the supplied diagnostics resulted from
the supplied losses. Arbitrary semantically inconsistent but individually valid
diagnostics are not a claim of training-history authenticity.

Loading restores the checkpoint's hyperparameters, counters, diagnostics,
base-rate metadata, cached rates and optimizer rates. It need not clamp restored
rates; an out-of-bounds saved rate is allowed until an active update. Tensor live
identity and independent initial/base metadata are preserved where supported.

Construct the scheduler before optimizer restoration, then restore scheduler
state. Exact training continuation also requires the model, optimizer internal
state, scaler, random generators and data order/cursor. DLRS alone cannot provide
that application-level recovery guarantee.

## C6. Metric wrapper

The history contains the most recent `patience + 2` converted metrics. Before
the window fills, a valid call records a metric without stepping the inner
scheduler. Once full, each valid call applies the selected direction to the
whole latest window. Failure of an inner update must not append the new metric.
Changing `patience` changes the next window; it is not a count of bad epochs.

Wrapper restoration validates wrapper fields before loading inner state;
restored history is truncated to capacity. The legacy wait counter is preserved
as metadata. The wrapper provides no best-metric tracker, cooldown or equivalence
to native `ReduceLROnPlateau`.

## Evidence and review requirements

| Contract area | Verification method | Limit |
|---|---|---|
| C2 slope, delta, clamping, locality; C6 abstract length | Solver-checked exact-real / integer model | No refinement proof to Python, tensors or floating arithmetic |
| C1/C3 discontinuities and extreme inputs | Independently derived rational oracle and adversarial tests | Finite coverage; operation-scale error budget |
| C4/C5/C6 observable transitions and ownership | Regression, failure and real checkpoint-continuation tests | Tested stacks and ordinary failure assumptions |
| Ability of tests to detect defects | Deliberate source mutations in isolated copies | Only the listed mutations; survivors/infrastructure failures remain visible |
| CUDA execution and package provenance | Pinned installed-wheel runs with retained raw artifacts | Exact recorded source and environment; older runs cannot qualify changed code |
| Independent human judgment | Review specification, proof assumptions, implementation and negative results | **Outstanding; AI reviews do not satisfy this gate** |

The solver and its arithmetic engine are trusted. Replaying solver queries with
the same solver is reproducibility, not independent proof-certificate checking.
The Python implementation, CPython/libm, Decimal/Fraction, PyTorch, OS and device
stack are not formally verified by this work. The production, quality and
upstream gates remain in the [readiness ledger](UPSTREAM_READINESS.md).
