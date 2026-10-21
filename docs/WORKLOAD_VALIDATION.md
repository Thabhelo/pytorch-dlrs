# Workload comparison protocol and results

Date: 2026-10-20. **All 72 trials completed; overall quality acceptance failed.**
MNIST meets the provisional margin; CIFAR-10 and the PINN fail against cosine.
DLRS remains experimental and opt-in. These results do not support a critical-use
or large-scale replacement claim.

## Frozen design

The [protocol](../benchmarks/validation_protocol.json) was written before training.
The [runner](../benchmarks/validate_workloads.py) fingerprints production and
benchmark source, the protocol, Torch, torchvision, NumPy, Python, and platform.
Resuming with changed evidence inputs fails. Raw completed and failed trials are
saved atomically under [workload-validation-2026-10-20](workload-validation-2026-10-20/).
A failed trial is retained and is not silently retried or dropped. The
[dataset integrity record](workload-validation-2026-10-20/dataset-integrity.json)
adds SHA-256 hashes of the actual local MNIST and CIFAR-10 files, alongside
torchvision's own download-integrity checks.

A preliminary execution was stopped after seven completed MNIST tuning trials
when concurrent floor-version testing exposed a tensor initial-rate metadata
alias defect. Its completed records and interruption reason remain in
[the development archive](workload-development-2026-10-20/). They are excluded
from final acceptance. The final execution restarts all planned trials under
the corrected code fingerprint; datasets, seeds, models, budgets, caps,
selection rules and acceptance margins are unchanged. The interrupted eighth
trial was stopped by the audit, not by a numerical training failure.

Each workload compares DLRS, a fixed learning rate, and cosine annealing. All
three policies receive three candidate initial learning rates at tuning seed 0.
Only the final validation loss selects a rate. Each selected configuration then
runs at fresh confirmation seeds 11, 12, 13, 14 and 15. The held-out test set is
never evaluated during tuning and is evaluated once per confirmation run.
Paired runs share initialization, data split, and seeded sampling; the runner
checks initialization and split hashes before calculating comparisons. Policy
execution order rotates to reduce systematic order effects, but the laptop is
shared and timings are not controlled deployment measurements. The local
execution runs the CIFAR-10 subset in a separate process/output directory while
the MNIST/PINN subset progresses. Completed records are combined only after
checking identical protocol/source/environment manifests and preserving every
trial byte-for-byte; the canonical runner then recomputes the full summary from
those cached records. This concurrency changes resource contention, not the
models, seeds, budgets, tuning rule or acceptance margins.

| Workload | Model / optimizer | Budget per trial | Data |
|---|---|---|---|
| MNIST | Repository MNISTCNN, SGD momentum 0.9 and weight decay 0.0005 | 6 epochs, batch 128 | 54,000 training / 6,000 validation; 10,000 test |
| CIFAR-10 | Repository SimpleCNN, same SGD | 6 epochs, batch 128 | 45,000 training / 5,000 validation; 10,000 test |
| Manufactured Helmholtz PINN | Three 32-unit hidden layers, Adam | 400 epochs, four 64-point batches per epoch | Fresh collocation points, 257 validation points, 1,024 held-out midpoint-grid points |

The vision initial-rate grid is 0.01, 0.03, 0.1 and DLRS's cap is 0.5. The PINN
grid is 0.0001, 0.001, 0.01 and DLRS's cap is 0.01. Other DLRS parameters retain
the implementation defaults. Budgets and caps were fixed before results. This
is equal-budget initial-rate tuning; it does not exhaustively tune each policy's
other hyperparameters. Six vision epochs are a bounded short-horizon comparison,
not convergence evidence for a normal production training schedule.

The PINN solves `u'' + u = (1 - pi^2) sin(pi*x)` with zero boundary values and
analytic solution `sin(pi*x)`. Tuning uses residual/boundary validation loss;
confirmation reports relative RMSE against the analytic solution on a separate
midpoint grid. The manufactured problem is useful for a known answer, not a
proxy for all scientific models.

## Predefined acceptance and interpretation

The numerical margins below are provisional study thresholds, not user-approved
production tolerances. A pass can permit a small measured quality reduction; it
is not a guarantee of unchanged quality.

Every confirmation trial must finish with finite losses, gradients and measured
outputs. Nonfinite loss/gradient checks occur before optimizer mutation in the
benchmark loop; they do not roll back earlier valid updates or forward-pass
buffer/RNG effects. Every failure is part of the evidence.

For each vision dataset, the nominal 95% bootstrap interval's lower endpoint
for paired DLRS minus baseline test accuracy must be at least -1 percentage
point, against **both** tuned baselines. For the PINN, the upper endpoint for
the geometric mean paired DLRS/baseline relative-RMSE ratio must be at most 1.10,
again against both baselines. Five paired seed outcomes produce 5^5 = 3,125
ordered bootstrap resamples. The runner enumerates those resamples exactly.
That is exact enumeration of the empirical bootstrap distribution, **not** exact
95% population coverage; five seeds provide limited uncertainty resolution.
Intervals are conditional on this dataset, split, budget and chosen tuning seed.
No multiple-comparison-adjusted or universal non-inferiority claim is made.

Timing includes training, validation, and the final test where applicable;
accelerator work is synchronized. Paired median time ratios are exploratory
because compatibility and other local work share CPU/GPU resources. No timing
SLO, native/device peak-memory acceptance, large-model throughput, or deployment
recovery claim is established. Those remain deployment-specific gates. The Mac also entered sleep during
execution and the existing processes resumed afterward; see the
[execution notes](workload-validation-2026-10-20/execution-environment-notes.json).
Recorded timer intervals do not establish uninterrupted elapsed wall time.

The CLI records execution success separately from quality acceptance. Partial
runs cannot mark the full protocol accepted; all three datasets are required.
A quality failure produces a nonzero exit even when every training run completed.

## Completed results

All **72 planned trials completed with finite results**: 27 tuning trials and 45
confirmation trials. Every policy has all five confirmation seeds for every
workload. Execution passed; **overall quality acceptance failed**. The final
runner exits nonzero for this quality failure, not for a crashed training job.
The separate [Transformer study](TRANSFORMER_VALIDATION.md) adds 24 completed
trials, making 96 final scientific trials across the four workloads.

| Workload | Selected DLRS LR | Selected fixed LR | Selected cosine LR | Provisional quality gate |
|---|---:|---:|---:|---|
| MNIST | 0.03 | 0.01 | 0.1 | Pass |
| CIFAR-10 | 0.01 | 0.03 | 0.03 | Fail |
| Helmholtz PINN | 0.01 | 0.001 | 0.01 | Fail |

Final held-out means and sample standard deviations across five seeds:

| Workload / metric | DLRS | Fixed LR | Cosine |
|---|---:|---:|---:|
| MNIST / accuracy %, higher is better | 98.966 ± 0.080 | 98.896 ± 0.083 | 99.140 ± 0.045 |
| CIFAR-10 / accuracy %, higher is better | 68.586 ± 0.896 | 68.634 ± 1.186 | 72.348 ± 0.606 |
| Helmholtz PINN / relative RMSE, lower is better | 0.000962005 ± 0.00100028 | 0.00623954 ± 0.00388306 | 0.000193628 ± 8.76183e-05 |

Paired effects use the predefined five-seed bootstrap rule. Vision values are
DLRS minus baseline accuracy in percentage points; PINN values are geometric
mean ratios of DLRS/baseline relative RMSE, not ratios of arithmetic means.

| Workload | Baseline | Paired effect | Nominal 95% interval | Gate |
|---|---|---:|---:|---|
| MNIST | Fixed LR | 0.07 | [0.006, 0.134] | Pass |
| MNIST | Cosine | -0.174 | [-0.248, -0.068] | Pass |
| CIFAR-10 | Fixed LR | -0.048 | [-1.366, 1.06] | Fail |
| CIFAR-10 | Cosine | -3.762 | [-4.196, -3.238] | Fail |
| Helmholtz PINN | Fixed LR | 0.0995376 | [0.0326087, 0.35749] | Pass |
| Helmholtz PINN | Cosine | 3.11377 | [1.34988, 6.96238] | Fail |

The MNIST pass allows the predefined one-point margin; its mean accuracy is
still lower than cosine. CIFAR-10 and the PINN fail against cosine. These are
**adoption blockers for the tested configurations**, despite passing numerical
and implementation checks. They do not show that every DLRS configuration fails,
nor that the policy preserves every run when an average criterion passes.

CIFAR-10 also fails to establish non-inferiority against fixed LR: its mean
difference is only -0.048 points, but the interval reaches -1.366 points, beyond
the permitted -1-point margin. That is insufficient evidence of non-inferiority,
not proof of a large mean disadvantage against the fixed-rate baseline.

### Per-seed held-out outcomes

| Vision dataset | Seed | DLRS accuracy % | Fixed LR accuracy % | Cosine accuracy % |
|---|---:|---:|---:|---:|
| MNIST | 11 | 98.94 | 98.82 | 99.16 |
| MNIST | 12 | 99.09 | 99.02 | 99.06 |
| MNIST | 13 | 98.90 | 98.91 | 99.15 |
| MNIST | 14 | 98.90 | 98.91 | 99.16 |
| MNIST | 15 | 99.00 | 98.82 | 99.17 |
| CIFAR-10 | 11 | 69.97 | 68.18 | 73.34 |
| CIFAR-10 | 12 | 67.74 | 67.34 | 71.89 |
| CIFAR-10 | 13 | 68.18 | 68.10 | 72.46 |
| CIFAR-10 | 14 | 68.97 | 69.12 | 71.86 |
| CIFAR-10 | 15 | 68.07 | 70.43 | 72.19 |

| PINN seed | DLRS relative RMSE | Fixed LR relative RMSE | Cosine relative RMSE |
|---|---:|---:|---:|
| 11 | 0.0021113001 | 0.0026962049 | 0.00022841949 |
| 12 | 0.0001767342 | 0.0047604786 | 0.00029440195 |
| 13 | 0.0019982483 | 0.0061008851 | 0.00023689659 |
| 14 | 0.00033457659 | 0.012834558 | 0.00013140138 |
| 15 | 0.00018916576 | 0.0048055598 | 7.7018653e-05 |

### Interpretation and evidence integrity

A bounded review of CIFAR training and saved trajectories found no implementation
or harness defect in the inspected path: policies share the model, optimizer,
split, initialization and update budget; constant and cosine rates follow their
specified schedules, and selection uses final validation loss. For DLRS,
the tuning losses at initial rates 0.01 and 0.03 were close:
`0.876827747` versus `0.879857909`, respectively. The selected 0.01 rate was therefore
correct under the frozen rule, even though 0.03 had higher tuning accuracy.
Changing the selection metric after seeing held-out outcomes would change the
study. No such retuning, trial replacement, or acceptance-margin change was made.
The different rate trajectories do not by themselves establish the cause of
the observed quality gap. Individual minibatch slopes were not saved.

The [independent checker](../benchmarks/check_workload_summary.py) recomputes
selection, complete seed pairing, means, standard deviations, intervals and
acceptance directly from all 72 raw records. Its bootstrap implementation uses
multinomial-weighted count vectors rather than the training runner's ordered
resamples. It also checks stored budgets, finite metrics, tuning/test separation,
rate continuity, counts and timing arithmetic. Its [verification output](workload-validation-2026-10-20/independent-summary-check.json)
binds the summary, manifest and every trial by SHA-256. Passing this check means
the saved evidence is internally consistent; it does not turn failed quality
criteria into passes or establish execution provenance independently.

See the [full summary](workload-validation-2026-10-20/summary.json),
[final status](workload-validation-2026-10-20/status.json),
[byte-preserving merge record](workload-validation-2026-10-20/execution-provenance.json),
the [final source and dataset integrity check](workload-validation-2026-10-20/final-integrity-check.json),
and [all raw trials](workload-validation-2026-10-20/trials/). Large-scale use still
requires a new, prespecified comparison on the intended model, data, training
horizon and hardware, with deployment-approved tolerances. This study does not
support replacing a working production scheduler by default.

## Reproduction

With the exact recorded compatible Torch/torchvision/NumPy versions, from the
checkout:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m benchmarks.validate_workloads \
  --device mps --data-root /tmp/dlrs-validation-data \
  --output docs/workload-validation-2026-10-20
```

Vision executes on MPS; the PINN executes on CPU. A new device, dependency,
protocol, or fingerprinted source requires a new output directory. The
`--datasets` option can continue a subset while preserving other completed
results. Reproducing on CUDA or CPU provides new environment-specific evidence;
it does not promise bitwise equality across hardware.

To independently check the complete saved evidence without retraining:

```bash
python -m benchmarks.check_workload_summary docs/workload-validation-2026-10-20
```
