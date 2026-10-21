# Single-GPU CUDA validation — 2026-10-21

**Result: all 18 required cases passed; two native-BF16 cases were unsupported
and skipped.** This is installed-wheel execution evidence for one recorded
configuration, not critical-deployment certification or a workload-quality result.

The run used an ordinary free Google Colab notebook. The resource panel showed
“You are not subscribed” and zero compute units. No paid plan, credits, premium
runtime, Drive mount, remote shell or distributed workers were used. Availability
of another free GPU is not guaranteed; see the [Colab workflow](FREE_COLAB_VALIDATION.md).

## Correctness-revision rerun

The corrected core at [`2ff033d66d49b0adc994d30b00dfd23dfb6664d0`](https://github.com/Thabhelo/pytorch-dlrs/commit/2ff033d66d49b0adc994d30b00dfd23dfb6664d0)
was independently installed and tested again on a free T4 at **19:57:14–19:57:28 UTC
on 2026-10-21**. All **18 required cases passed**, with **two native-BF16 skips**,
zero failures/errors, eight GradScaler API deprecation warnings and exit zero.
Pytest reported **8.93 seconds**; the runner measured 10.29 seconds including
process overhead. This is execution/recovery evidence, not a throughput result.
Python 3.13.15, Torch 2.11.0+cu130, CUDA 13.0 and driver 580.82.07 were unchanged.
The resource panel again showed no subscription and zero compute units.

The installed package ran outside the checkout. All three imported runtime
files matched the retained wheel and current source; the runner and test hashes
were unchanged. All **27 outer-bundle entries and 12 inner evidence entries**
were independently hash-verified after export. Both retained wheel copies matched.

| Correctness-revision artifact | SHA-256 |
|---|---|
| [Raw evidence bundle](correctness-validation-2026-10-21/cuda-evidence.tar.gz) | `c2a3c0c1b8cb67f2c2ab5e06ff3cc36b939460cf7a7843b3c59e367ff513aaaa` |
| Installed wheel, retained inside the bundle | `b6df730df7733c0cf74a9c82e1c3673808c285c3f7c27931a98add33955799f6` |
| `dlrs/scheduler.py` | `8cf8897aad9bc5e1616a5347aa1bc7212ccf412fab0970d0bc00c27065a6e484` |
| `dlrs/utils.py` | `1cd09b46d885d569c5ecb651bed5bb039b9d0b2ca6b4ffb2049afa162a48638b` |

The [new summary](correctness-validation-2026-10-21/cuda-summary.json),
[source fingerprint](correctness-validation-2026-10-21/cuda-source.json),
[wheel metadata](correctness-validation-2026-10-21/cuda-wheel.json) and
[bundle manifest](correctness-validation-2026-10-21/cuda-bundle-sha256.json)
are available separately. The notebook now pins this corrected source revision.
Later audit-tool portability fixes and documentation/evidence additions do not
change these executed runtime, runner or CUDA-test bytes. The same limits below
apply; native BF16 and deployment-specific qualification remain open.

## Earlier executed configuration


| Item | Recorded value |
|---|---|
| Source/test/runner revision | `8358f1b4c065a5098de708bbeee1891dd5a69cc2` |
| Package | `pytorch-dlrs 0.3.0.dev0`, noneditable wheel installation |
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu130; Colab's existing build retained |
| CUDA build / driver | 13.0 / 580.82.07 |
| Device | Tesla T4, compute capability 7.5, one visible GPU |
| Optimizers | SGD with momentum and AdamW; eager, `foreach=False` |
| Precision | Float32 and FP16 with a real CUDA GradScaler |
| Native BF16 | Unsupported; two optional cases skipped |
| Test result | 18 passed, 2 skipped, 0 failed/errors; pytest exit 0 |
| Test duration | 10.25 seconds reported by pytest; not a throughput benchmark |
| Execution window | 2026-10-21 08:54:22–08:54:38 UTC |

The [summary](cuda-validation-2026-10-21/final-summary.json) lists every case and outcome.
The final bundle's `evidence/environment.json` records GPU memory, cuDNN, package
versions, import paths and capabilities. The source checkout was clean at the
recorded revision. Later notebook-pin and report additions do not alter the
executed source, test or runner fingerprints.

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

The [final evidence bundle](cuda-validation-2026-10-21/final-evidence.tar.gz)
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
| Final evidence bundle | `4e2a3f663f00dec3316505b49759f1db21634ed8dcb7f4f78533abab190ec14a` |
| Final-run tested wheel | `29da11de500cf3ac00a49e7a4ff7c48e8fd7e4f35113552d1e3e88ba05d9347a` |
| Expanded evidence bundle | `8e26f00ab97b60969625e5e8e2d3a30194bb92aa630730d69f69283bc9852eb4` |
| Expanded-run tested wheel | `6fc56b91067b0a2f7603ebe77c567feae969cd07155766d03a42d53df3547408` |
| Initial evidence bundle | `64c5575400428fd7f3878dbdbc6229410d8a6daab1caaeef6863bf3396c85db7` |
| Initial-run tested wheel | `554a3c57276e6bfa22995a0db93d066bcc706989e7eaa1c85ad45f7cbb247d45` |

See the final [source identity](cuda-validation-2026-10-21/final-source.json),
[wheel metadata](cuda-validation-2026-10-21/final-wheel.json), and
[internal file manifest](cuda-validation-2026-10-21/final-evidence-sha256.json).
That internal manifest names files in the bundle's `evidence/` directory.

The [expanded bundle](cuda-validation-2026-10-21/expanded-evidence.tar.gz) and
[expanded summary](cuda-validation-2026-10-21/expanded-summary.json) preserve the
first 18-case run at `4700e584711861bc54e8b62b752270b6c173b414`, executed
08:01:27–08:01:37 UTC in 5.07 seconds of pytest time. Review then found two runner
issues: checkout-source mode required installed metadata, and an existing sibling
archive could be overwritten. Both were fixed with four regression cases. The
final run repeated all 18 required checks with that corrected runner. Its notebook
also requires a completed process and exit zero before displaying a pass.

The [initial bundle](cuda-validation-2026-10-21/evidence.tar.gz) and
[initial summary](cuda-validation-2026-10-21/summary.json) remain unchanged.
That run at `97a3d62bbabc296770ea5e583f4c5c1d8f9105f8` passed 14 required cases
with two BF16 skips at 07:48:45–07:49:01 UTC, in 8.72 seconds of pytest time.
The four accumulation cases were added afterward; the expanded run reran all
original cases too. The wheel hashes differ because these are separate builds;
the runtime source fingerprints remain identical.

Before the accumulation extension, the local Mac suite passed **430 tests**, with **16 explicit CUDA skips**.
The final 25 runner regressions passed on Python 3.11 and Python 3.8/Torch 2.0. These checks
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
