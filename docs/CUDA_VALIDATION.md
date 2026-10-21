# Single-GPU CUDA validation — 2026-10-21

**Result: all 18 required cases passed; two native-BF16 cases were unsupported
and skipped.** This is installed-wheel execution evidence for one recorded
configuration, not critical-deployment certification or a workload-quality result.

The run used an ordinary free Google Colab notebook. The resource panel showed
“You are not subscribed” and zero compute units. No paid plan, credits, premium
runtime, Drive mount, remote shell or distributed workers were used. Availability
of another free GPU is not guaranteed; see the [Colab workflow](FREE_COLAB_VALIDATION.md).

## Executed configuration

| Item | Recorded value |
|---|---|
| Source/test/runner revision | `4700e584711861bc54e8b62b752270b6c173b414` |
| Package | `pytorch-dlrs 0.3.0.dev0`, noneditable wheel installation |
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu130; Colab's existing build retained |
| CUDA build / driver | 13.0 / 580.82.07 |
| Device | Tesla T4, compute capability 7.5, one visible GPU |
| Optimizers | SGD with momentum and AdamW; eager, `foreach=False` |
| Precision | Float32 and FP16 with a real CUDA GradScaler |
| Native BF16 | Unsupported; two optional cases skipped |
| Test result | 18 passed, 2 skipped, 0 failed/errors; pytest exit 0 |
| Test duration | 5.07 seconds reported by pytest; not a throughput benchmark |
| Execution window | 2026-10-21 08:01:27–08:01:37 UTC |

The [summary](cuda-validation-2026-10-21/expanded-summary.json) lists every case and outcome.
The [environment](cuda-validation-2026-10-21/expanded-environment.json) records GPU memory,
cuDNN, package versions, import paths and capabilities. The source checkout was
clean at the recorded revision. Later notebook/report additions do not alter
the executed source, test or runner fingerprints.

## What passed

- Real float32 training with two parameter groups, SGD and AdamW, float and
  CUDA-tensor learning rates, compared with an independent high-precision
  update equation; bounds and tensor identity checks passed.
- Exact stochastic checkpoint continuation with float and tensor rates,
  restoring model, optimizer, scheduler and CPU/CUDA/shuffle random state.
- FP16 overflow injection causing one or every optimizer update to be skipped.
  Skipped updates did not change model/optimizer state or contribute scheduler
  observations; results matched an independently omitted-update reference.
- Exact FP16 stochastic continuation including GradScaler and random state.
- Float32 and FP16 accumulation for both optimizers compared with independent
  full-effective-batch forward/backward passes. The final five-sample window
  uses microbatches of three and two; gradients and observations are weighted by
  sample count. Eight actual updates per path, gradients, update vectors,
  observation counts, rates and scaler state were checked.

Accumulation comparisons allow predeclared rounding tolerances for different
matrix/reduction shapes. FP16 bounds global gradient-vector error at 0.5% plus
2e-6, update-vector error at 0.8% plus 2e-6, loss error at 0.1% plus 2e-6 and
rate error at 0.2%. Float32 uses tighter bounds in the test. The fixture checks
that a naive equally weighted final loss would exceed its tolerance. This is
numerical agreement within those bounds, separate from exact checkpoint replay.

The runner copied only the CUDA test into a temporary directory outside the
checkout and ran isolated Python against the installed package. Imported core
files and version matched the retained wheel and source. Their fingerprints,
the copied test and wheel remained unchanged after execution. An exact case
inventory prevents missing or skipped required checks from yielding a pass.

Eight warnings concern the deprecated `torch.cuda.amp.GradScaler` spelling used
by these tests to retain compatibility with the declared Torch 2.0 floor; the
scaler was enabled and executed on CUDA. The warnings are retained in the raw log.

## Retained evidence

The [expanded evidence bundle](cuda-validation-2026-10-21/expanded-evidence.tar.gz)
contains the exact tested wheel, JUnit report, logs, command records, environment
and source identities, plus setup logs and the notebook process outcome. The
reusable notebook's exact setup, run and export cells were executed through
Colab's normal interface. The bundle was recovered through its visible output
after browser download delivery was unavailable. Its SHA-256 matched the value
printed in Colab. All 27 outer-manifest entries and all 12 entries in the runner's
internal file manifest were verified locally. The JSON files beside it are
byte-identical convenience copies.

| Artifact | SHA-256 |
|---|---|
| Expanded evidence bundle | `8e26f00ab97b60969625e5e8e2d3a30194bb92aa630730d69f69283bc9852eb4` |
| Expanded-run tested wheel | `6fc56b91067b0a2f7603ebe77c567feae969cd07155766d03a42d53df3547408` |
| Initial evidence bundle | `64c5575400428fd7f3878dbdbc6229410d8a6daab1caaeef6863bf3396c85db7` |
| Initial-run tested wheel | `554a3c57276e6bfa22995a0db93d066bcc706989e7eaa1c85ad45f7cbb247d45` |

See the expanded [source identity](cuda-validation-2026-10-21/expanded-source.json),
[wheel metadata](cuda-validation-2026-10-21/expanded-wheel.json), and
[internal file manifest](cuda-validation-2026-10-21/expanded-evidence-sha256.json).
That internal manifest names files in the bundle's `evidence/` directory.

The [initial bundle](cuda-validation-2026-10-21/evidence.tar.gz) and
[initial summary](cuda-validation-2026-10-21/summary.json) remain unchanged.
That run at `97a3d62bbabc296770ea5e583f4c5c1d8f9105f8` passed 14 required cases
with two BF16 skips at 07:48:45–07:49:01 UTC, in 8.72 seconds of pytest time.
The four accumulation cases were added afterward; the expanded run reran all
original cases too. The wheel hashes differ because these are separate builds;
the runtime source fingerprints remain identical.

Before the accumulation extension, the local Mac suite passed **430 tests**, with **16 explicit CUDA skips**.
The 21 new runner regressions also passed on Python 3.8/Torch 2.0. These checks
validate the runner and preserve existing CPU/MPS behavior; they are distinct
from the actual T4 passes. The expanded CUDA inventory collects 20 cases, all
explicitly skipped without CUDA. Eight additional CPU cases exercise the actual
notebook run-cell source to prevent a passing display after nonzero exits,
interruptions, timeouts, or invalid summaries.

## Remaining limits

This closes the measured T4 installed-wheel execution gap. It does not establish
native BF16, other GPU/driver/Torch combinations, different accumulation policies, fused or
capturable optimizers, compiled scheduler stepping, CUDA graphs, distributed
CUDA/NCCL, multi-node/sharded/elastic recovery, durable production checkpoints,
resource budgets, endurance or large-model quality.

The 96 prior workload trials and their negative CIFAR/PINN results are unchanged.
No scheduler implementation or default changed in this validation update.
The [readiness ledger](UPSTREAM_READINESS.md) states the acceptance criteria still
requiring a concrete deployment, new evidence or maintainer review. No PyTorch
upstream submission or PyPI release was made.
