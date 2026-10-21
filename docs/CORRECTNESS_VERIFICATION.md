# Correctness verification, 2026-10-21

This milestone defines a [versioned behavior contract](CORRECTNESS_CONTRACT.md),
fixes five implementation defects, and checks the contract through separate
mathematical models, independent numerical references, state-transition tests,
and deliberately defective source copies. It does **not** establish that DLRS is
100% correct, that it cannot degrade a workload, or that it is ready for critical
production use or inclusion in PyTorch.

The local core was published as
[`2ff033d66d49b0adc994d30b00dfd23dfb6664d0`](https://github.com/Thabhelo/pytorch-dlrs/commit/2ff033d66d49b0adc994d30b00dfd23dfb6664d0).
The local evidence applies to the source hashes in
[source identity](correctness-validation-2026-10-21/source-identity.json).
The [machine-readable summary](correctness-validation-2026-10-21/summary.json)
records the result counts and archive hashes. The archived
[hosted CI run](https://github.com/Thabhelo/pytorch-dlrs/actions/runs/39666786762)
passed at `23c84133440b8071ea541fbdf8b6a8fca6773c66`; it predates the later
audit source-binding correction described below. Exact-revision outcomes for
subsequent changes are available in the [PR checks](https://github.com/Thabhelo/pytorch-dlrs/pull/4/checks).
The changed core has a separate verified
[installed-wheel T4 rerun](CUDA_VALIDATION.md#correctness-revision-rerun): 18
required cases passed and two unsupported native-BF16 cases skipped. Earlier
T4 results retain their original source identities.

## Defects found and corrected

| Defect | Corrected behavior | Evidence and limits |
|---|---|---|
| Rounded `log10` selected the wrong decade near decimal boundaries, including ordinary rates | Select the decade of the exact stored binary64 value using `Decimal.from_float(rate).adjusted()` | An independent rational exponent oracle checks all finite decimal-power neighbors and additional extreme values. The float written `1e-6` is below exact decimal `10^-6`; a recorded case changes the next rate from `1.2e-6` to `1.02e-6`. Ordinary trajectories can change. |
| Binary normalization could lose subnormal endpoint information or turn a nonzero slope into zero | Use a rational fallback for the specified delicate cases; reject a nonzero scheduler slope that cannot be represented as a float before any state change | Exact-fraction comparisons, branch-boundary tests and scheduler/wrapper rejection checks. This is not arbitrary rescaling invariance or a global floating-point error theorem. |
| A negative exact real could convert to `-0.0` and pass the nonnegative check | Check the original real's sign before binary64 conversion | Tiny negative `Fraction` inputs are rejected by observations, external rates and checkpoint fields. Positive exact inputs can still round to zero, as the contract states. |
| Exceptional Decimal calculations inherited application traps, exponent limits or rounding; the old code already set precision to 50 | Use a private, explicit Decimal context and leave the caller's context unchanged | Tests vary those settings and precision for large and subnormal updates. The fallback remains a finite-precision implementation, not exact-real evaluation of the entire formula. |
| CPU Half inward rounding relied on `torch.nextafter`, absent on the declared PyTorch 2.0 floor | Select adjacent positive finite CPU float16/bfloat16 values using their ordered 16-bit representations | Independent IEEE-grid bounds and no-representable-value rejection checks pass on the floor and current local stacks. Other device/dtype combinations require their own execution evidence. |

These changes preserve the exact mathematical intent stated in the contract,
but they intentionally change previously incorrect behavior. They are not a
promise of unchanged training curves. See the [migration notes](FAQ.md#migrating-from-021).

The CPU 16-bit neighbor helper also received an exhaustive check over its stated
positive finite input domain: 31,743 float16 values and 32,639 bfloat16 values in
both directions, **128,764 direction checks per environment**. Both the floor
and current stacks matched independent IEEE decoding; the current stack also
matched native `torch.nextafter`. The probe checks smallest-value movement to
positive zero, largest-value movement to infinity, unchanged inputs and separate
output storage. Its [floor](correctness-validation-2026-10-21/cpu16-floor.json)
and [current](correctness-validation-2026-10-21/cpu16-modern.json) records retain
source hashes. This exhaustive finite-domain helper test does not prove the
whole implementation.

## Implementation checks

The new numerical and state suites contain **86 cases: 62 numerical and 24
state-transition cases**. The numerical reference uses rational arithmetic to
select decimal decades, compute reference slopes and select branches; it does
not reuse the production decade helper. Tensor-bound references use independent
IEEE bit layouts. Coverage includes decimal-power neighbors, subnormal values,
both sides of slope one, power-of-two loss scaling, 400 seeded wide-exponent
inputs, finite updates with overflowing intermediates, tensor bounds, impossible
representation intervals, snapshot ownership and failure atomicity.

Continuous arithmetic comparisons use a declared operation-scale error budget
of eight binary64 ULPs of the relevant magnitudes, with discrete decisions
checked separately. Cancellation is explicitly tested as a case where the
implementation need not be the correctly rounded exact-real answer. Passing
these finite samples does not prove that error budget for every input.

| Local execution | Observed result | Scope |
|---|---|---|
| Python 3.8.20 / PyTorch 2.0.0, numerical and state suites | 86 passed, 2.90 seconds | Declared version floor on macOS arm64 |
| Python 3.11.12 / PyTorch 2.14.1, full suite before the final comparator regressions | 575 passed, 25 skipped, 66.19 seconds; 600 collected | Includes all 86 numerical/state cases, real local MPS and Lightning checks; 20 CUDA cases and five solver-dependent tests explicitly skipped |
| Python 3.11.12 / PyTorch 2.14.1, post-comparator safety and mutation-harness tests | 147 passed, 3.40 seconds | 117 safety cases plus 30 mutation-harness cases |
| Python 3.8.20 / PyTorch 2.0.0, same post-comparator safety and mutation-harness tests | 147 passed, 2.96 seconds | Same inventory on the floor |
| Python 3.11.12 / PyTorch 2.14.1, mutation-harness tests after the portability correction | 31 passed, 2.21 seconds | Includes the actual report hook exercised with Windows-style paths |
| Python 3.11.12 / PyTorch 2.14.1, mutation-harness tests after the source-binding correction | 37 passed, 7.97 seconds | Six additional cases reject evidence-input or scratch-file changes |
| Python 3.8.20 / PyTorch 2.0.0, same source-binding harness tests | 37 passed, 6.07 seconds | Same guard inventory on the floor |

Four type-comparison regressions were added after the 600-case full run. The
resulting suite inventory was **604**, then **605** after the Windows report-key
regression. The archived hosted run below executed the 605-case inventory.
Six later source-binding regressions bring the inventory to **611**, with **59**
dedicated proof/audit guard tests (22 plus 37). Those inventory counts do not
assert a hosted result for a later revision. The separate 147-case runs cover
the comparator additions; overlapping runs must not be summed as disjoint tests.

The retained [local test archive](correctness-validation-2026-10-21/local-tests.tar.gz)
contains the full-run log/JUnit, both post-comparator helper JUnit files, the floor focused
log and relevant source snapshots. JUnit skip records remain visible. Elapsed
times describe this local execution, not a service-level guarantee.

## Archived hosted execution

[Run 39666786762](https://github.com/Thabhelo/pytorch-dlrs/actions/runs/39666786762)
completed successfully on head `23c84133440b8071ea541fbdf8b6a8fca6773c66`:
13 jobs passed and the release-only PyPI publication job was skipped. All nine
matrix cells collected 605 cases, with no failures or errors. The
[complete receipt](correctness-validation-2026-10-21/ci-39666786762-summary.json)
retains each skipped test and its reason, observed Python/PyTorch versions,
artifact identities and package-install evidence.

| Hosted cell at the recorded head | Passed | Explicitly skipped |
|---|---:|---:|
| macOS, Python 3.12 / PyTorch 2.14.1 | 576 | 29 |
| Ubuntu 22.04, Python 3.8 / PyTorch 2.0.0 | 564 | 41 |
| Ubuntu 22.04, Python 3.9 / PyTorch 2.4.1 | 570 | 35 |
| Ubuntu, Python 3.10 / PyTorch 2.5.1 | 570 | 35 |
| Ubuntu, Python 3.11 / PyTorch 2.6.0 | 570 | 35 |
| Ubuntu, Python 3.12 / PyTorch 2.14.1 | 570 | 35 |
| Ubuntu, Python 3.12 / latest PyTorch, resolved to 2.14.1+cpu | 570 | 35 |
| Ubuntu, Python 3.14 / PyTorch 2.14.1 | 570 | 35 |
| Windows, Python 3.11 / PyTorch 2.6.0 | 570 | 35 |

All cells skipped 20 CUDA cases, five optional Z3 tests and four optional
Lightning tests. Linux and Windows additionally skipped six MPS cases; the
PyTorch 2.0 floor skipped six CPU GradScaler cases because the required API was
absent. The dedicated Lightning job passed all four cases. The dedicated
correctness job passed **53 guard tests**, all **25 mathematical obligations**
and the **14-mutation campaign**. Those separate checks close their stated
coverage only; skips elsewhere remain skips.

Both the wheel and source distribution separately passed **570 cases with 35
explicit skips** on Python 3.11.17 / PyTorch 2.6.0+cpu. Tests ran from an extracted
source tree after its `dlrs` package directory was removed; the isolated import
probe resolved the installed package under `site-packages`. Both packaged core
copies match commit `2ff033d66d49b0adc994d30b00dfd23dfb6664d0` byte-for-byte.

| Hosted distribution | SHA256 |
|---|---|
| Wheel | `a1358e5bbd84f7ec59e76ab78bf73ed773a5367362a818cc1086e7bb5e1981b6` |
| Source distribution | `bae2df9c96b1fec3775df19f803ba1aa8651d94a8e7f418a3271e208c9591330` |

All 13 downloaded artifact ZIP digests match GitHub's API digests. The
[CI archive](correctness-validation-2026-10-21/ci-39666786762.tar.gz) preserves
those raw ZIPs, official run logs, API metadata and the receipt. Its layout record
describes extraction into the paths named by the receipt. This is an archived
result for the stated head, not a claim that the later 611-case inventory or
59-guard job has already passed. Later exact-revision outcomes are available in
the [PR checks](https://github.com/Thabhelo/pytorch-dlrs/pull/4/checks).

## Solver-checked mathematical obligations

The optional proof runner checked **25 obligations** with Z3 package
`5.3.1.0` (solver `5.3.1`). For every obligation, its assumptions were
satisfiable and the assumptions together with the negated conclusion were
unsatisfiable. All 50 saved SMT-LIB query byte hashes were verified and all 50
queries were replayed successfully. The campaign elapsed time was 0.3152 seconds.
The [full proof report](correctness-validation-2026-10-21/proof-report.json)
records each assertion, result, model for satisfiable assumptions, solver
statistics, timeout and exact source identity. The
[proof archive](correctness-validation-2026-10-21/proofs.tar.gz) includes every query.

The obligations cover:

- Slope bounds `-n <= s <= n`, endpoint sign, positive loss scaling and maximum-mode reversal.
- Positive factor selection, including the middle branch at slope exactly one.
- Unclamped update direction, frozen zero rates, bounds and clamp idempotence.
- Monotonicity for rates that start within bounds, and a conditional relative raw-adjustment bound.
- Independence of separate mathematical group state cells with shared configuration held fixed.
- Base case, inductive step and saturation for an abstract `patience + 2` history-length recurrence.

The slope summary assumes nonnegative endpoints, positive total,
`total >= first + last`, and `n >= 2`. Allowing real `n` relaxes the integer
sequence-length domain; this proves a summary implication, not a theorem about
the code that validates or sums a Python list. Division is expressed through a
relation with a positive total. Granularity is a positive mathematical parameter;
the model does not prove its Python computation. Group locality does not prove
memory separation, and the window recurrence does not prove which values are
retained.

Z3's real arithmetic is mathematical arithmetic, separate from its floating-point
theory. Solver results can be `unknown`; those results, timeouts, contradictory
premises, missing queries and satisfiable negated claims do not pass this runner.
See the official [arithmetic semantics](https://microsoft.github.io/z3guide/docs/theories/Arithmetic/)
and [solver API](https://z3prover.github.io/api/html/classz3py_1_1_solver.html).
The solver and the model translation remain trusted. Replaying the same solver
is reproducibility, not independent proof-certificate checking.

There is **no refinement proof from this model to Python/PyTorch**, no IEEE-754
rounding proof, no global error bound, and no proof of allocation/device-write
atomicity, checkpoint authenticity, optimizer behavior, convergence or workload
quality. The runner has no runtime-package dependency; Z3 is optional development
tooling. A missing solver produces an unavailable result and a nonzero exit,
rather than a successful proof gate.

All **22 proof-harness tests passed with Z3 installed**, including replay of the
queries, actual counterexamples for deliberately false claims, contradictory
assumptions, timeout/unknown handling and inventory checks. The log and JUnit
are retained with the local test evidence. In environments without Z3, five of
those tests explicitly skip; those skips are not mathematical proof evidence.

## Deliberate-defect audit, including its initial survivor

The audit applies **14 explicitly selected mutations** to isolated source copies.
It first requires a clean 36-case baseline, then runs the relevant baseline test
identities for each mutation. A mutation counts as detected only when normal
test execution reaches an assertion failure: import errors, collection failures,
timeouts, unexpected skips, setup/teardown failures, changed input hashes or
different environments are inconclusive. Every case retains its command,
mutated source, selected test identities, raw log and structured test report.

| Campaign | Baseline | Detected / survived / inconclusive | Elapsed |
|---|---|---|---|
| [Initial summary](correctness-validation-2026-10-21/mutation-initial-summary.json) | 36 passed | 13 / 1 / 0 | 35.39 seconds |
| [After the comparator correction](correctness-validation-2026-10-21/mutation-final-summary.json) | 36 passed | 14 / 0 / 0 | 36.97 seconds |
| [After the portability correction](correctness-validation-2026-10-21/mutation-portable-final-summary.json) | 36 passed | 14 / 0 / 0 | 37.80 seconds |
| [After canonical source binding](correctness-validation-2026-10-21/mutation-bound-final-summary.json) | 36 passed | 14 / 0 / 0 | 44.36 seconds |

The initial survivor was `checkpoint_applied_before_validation`: the deliberately
broken loader committed epoch metadata before validating a checkpoint. Its
boolean epoch compared equal to integer one because the test helper used value
equality without checking types (`True == 1`). The survivor was retained, not
discarded or renamed. A recursive type-aware comparison and four focused
regressions corrected that test blind spot. The next campaign used the same
14 mutation definitions and the same production source; only the safety test
comparison and its regressions changed. All 14 selected defects then produced
real assertion failures.

A subsequent static portability review found that the report hook represented
relative test paths with Windows backslashes while selectors used forward
slashes. The harness now normalizes those report keys with `as_posix()`. A
regression executes the actual report hook using Windows-style paths. Its
31-case harness suite passed, and a third campaign again detected all 14
mutations. Production source and mutation definitions were unchanged. The
archived 605-case Windows run subsequently passed with that regression included.

A further audit found a provenance weakness: each fresh scratch copy supplied
its own expected hashes. A helper or test changed in retained evidence inputs
after the passing baseline could therefore become a new accepted expectation,
even though it differed from the original protocol. This could affect future
runs; it is not evidence that a saved campaign was corrupted. Independent
[reconstruction of the historical campaigns](correctness-validation-2026-10-21/canonical-history-verification.json)
checked their retained input and scratch bytes, recorded imported-source/test
identities and exactly declared mutations against each frozen protocol. All
three original local campaigns and the archived hosted campaign matched.

The corrected harness derives expected hashes from the original in-memory
snapshot plus only the declared scheduler mutation. It checks complete retained
input and scratch-file inventories before and after every case. Changes,
missing files or unexpected plugins make the result inconclusive; a resulting
assertion failure cannot count as detection. Six new regressions cover these
paths. The fourth campaign passed its 36-case baseline and detected all 14
mutations, with all **60 per-case before/after integrity checks** successful,
all **236 original manifest entries** verified and live/snapshot inputs unchanged.
The production core and mutation definitions remain unchanged.

The mutations exercise factor selection, loss direction, group-specific rates,
decimal decades, frozen zero, both clamps, validation timing, checkpoint rate
restoration, snapshot aliasing, wrapper history and maximum-mode direction.
This is a targeted sensitivity check, not an exhaustive mutation score or an
independent proof of correctness. Passing the selected mutants cannot rule out
an untested defect or a mistaken specification. All four complete campaigns remain
available as [initial](correctness-validation-2026-10-21/mutation-initial.tar.gz),
[post-comparator](correctness-validation-2026-10-21/mutation-final.tar.gz),
[portable](correctness-validation-2026-10-21/mutation-portable-final.tar.gz) and
[source-bound](correctness-validation-2026-10-21/mutation-bound-final.tar.gz)
archives. The latter includes the current harness/test source and both 37-case
guard JUnit results; earlier snapshots remain unchanged.

## Cost, reproduction and remaining gates

The changed core at `2ff033d66d49b0adc994d30b00dfd23dfb6664d0` was installed as
a wheel and tested on a free Colab Tesla T4 on 2026-10-21, 19:57:14–19:57:28 UTC,
using the existing PyTorch `2.11.0+cu130` build. The retained result is **18 passed,
two unsupported native-BF16 skips, zero failures/errors and eight warnings**, in
8.93 seconds of pytest time. Verification checked all 27 outer and 12 inner
manifest hashes, both wheel copies and imported source identities against the
changed core. See the [CUDA rerun report](CUDA_VALIDATION.md#correctness-revision-rerun)
and its [raw result](correctness-validation-2026-10-21/cuda-summary.json). This is
single-process, single-GPU evidence; it does not establish native BF16, other
GPU stacks, distributed CUDA or a production recovery guarantee.

The [timing report](correctness-validation-2026-10-21/timing-report.json)
compares the frozen current source with baseline `d1d850b006cf0d1048a7cec8050c2c79dec91e12`
on an Apple M3, Python 3.11.12 and PyTorch 2.14.1, with one intra-op and one
inter-op thread. Twelve cases cover 10, 1,000 and 100,000 observations; one or
100 parameter groups; and ordinary or exact-slope-one paths. Each case uses
three warmups per revision and 15 pairs with randomized revision order. The
timed region is one scheduler call on a fresh, preconstructed scheduler.

Median paired current/baseline ratios range from **1.0074 to 1.1885**, with
median paired increases of **1.416 microseconds to 2.359 milliseconds**. The
largest relative increase is the ordinary path with 10 observations and 100
groups: revision medians are about 0.234 and 0.278 milliseconds. These measurements
show an overhead increase in the sampled cases, not a speedup or a negligible-cost
guarantee. They do not measure model training throughput. Team validation jobs
were coordinated to avoid intentional overlap, but the shared desktop and OS
background activity were not isolated. All 180 raw pairs, source snapshots,
protocol and probe are retained in the local test archive; no acceptance threshold
or large-scale performance conclusion was declared.

The dedicated CI job installs the pinned optional solver, runs the proof CLI and
the harness regression tests, then runs the mutation CLI. Failure propagates to
the build dependency; raw queries and mutation reports are retained even on
failure. The runtime compatibility matrix remains separate. This describes the
configured checks. The archived hosted result above is pinned to its exact head;
subsequent exact-revision execution is tracked in the PR checks, without rewriting
historical receipts or implying that test inventories are executed passes.

To reproduce in a development environment containing the package test
dependencies and `z3-solver==5.3.1.0`, choose new output directories:

```bash
python benchmarks/verify_contract.py --output /tmp/dlrs-proofs-new
python -m pytest tests/test_contract_proofs.py tests/test_mutation_audit.py -q
python benchmarks/audit_mutations.py --output /tmp/dlrs-mutations-new
```

Each archive has an adjacent `*.members.json` manifest hashing **every regular
archive member**, including the campaigns' original manifests. Archive SHA256s
are recorded in `summary.json`, outside the archives to avoid a circular hash.
The original raw records are retained byte-for-byte; historical source paths in
them describe where those runs occurred, not a requirement to use that path.

The earlier [workload study](WORKLOAD_VALIDATION.md) contains negative CIFAR and
PINN quality gates. Those results and their source hashes are unchanged. Quality
of the newly changed core is **unmeasured** by those earlier trials. This work
cannot support a universal nondegradation claim.

Independent human review of the specification, assumptions, implementation and
negative evidence remains open; AI reviews do not close that gate. Broader CUDA
coverage, multi-node distributed behavior, application-specific recovery and
critical production qualification remain separate open gates. No PyTorch
upstream acceptance or safety certification is implied.
