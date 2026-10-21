# Free Colab CUDA validation

The first [recorded T4 run](CUDA_VALIDATION.md) passed all 14 required installed-wheel
cases on 2026-10-21, with two native-BF16 capability skips. That evidence is
preserved separately. The expanded run and subsequent correctness-revision rerun passed:
18 required checks, including four accumulation cases, and two native-BF16 skips.

This is a bounded, opt-in check of an installed DLRS wheel on **one CUDA GPU in
one process**. It uses the runtime's existing PyTorch GPU build. It does not
start distributed workers, mount Google Drive, or acquire paid compute.

The [notebook](../notebooks/free_colab_cuda_validation.ipynb) pins the executable
source and tests to commit
`2ff033d66d49b0adc994d30b00dfd23dfb6664d0`. The notebook itself is a user interface
for that revision; changing its setup prose does not change the tested source.
The runner also supports a clearly labeled checkout-source mode.

## Run through the ordinary notebook interface

1. Open the notebook in [Google Colab](https://colab.research.google.com/), using
   **File → Open notebook → GitHub** or uploading the `.ipynb` file.
2. Under **Runtime → Change runtime type**, select a free GPU if offered, with
   standard memory. Remain on the free plan. Do not purchase compute units,
   select a paid/premium runtime, use Enterprise, or connect through SSH.
3. Run the cells in order while present. Setup fetches the pinned source,
   installs test/build tools, builds a wheel with `--no-deps
   --no-build-isolation`, then installs that wheel with `--no-deps`. It verifies
   that the PyTorch version, CUDA build, and import path remain unchanged.
4. Review the result, then run the download cell even if setup or validation
   failed. It also collects partial evidence after an interrupted run.
5. After the download completes, choose **Runtime → Disconnect and delete
   runtime** to release the GPU.

Free GPU availability and GPU types vary. If a free GPU is unavailable, retain
the unavailable result and stop; this workflow has no paid fallback. Ordinary
notebook use matters: free managed runtimes restrict remote-control shells and
distributed workers. These constraints come from the
[official Colab FAQ](https://research.google.com/colaboratory/faq.html).

## Exactly what is tested

The explicit inventory contains 20 small synthetic cases:

| Cases | Required behavior |
| --- | --- |
| 4 | SGD/AdamW real float32 updates agree with an independent learning-rate equation, using float or CUDA-tensor rates; group bounds and tensor identity hold. |
| 4 | Float32 stochastic checkpoint continuation restores model, optimizer, scheduler, and CPU/CUDA/shuffle RNG state, using both rate representations. |
| 4 | FP16 GradScaler overflow skips one or all updates without adding loss observations or changing skipped optimizer/model state. |
| 2 | FP16 stochastic continuation restores scaler and RNG state as well as model/optimizer/scheduler state. |
| 4 | SGD/AdamW float32 and FP16 accumulation agrees with full effective batches, including a final window of five samples split into microbatches of three and two. |
| 2 optional | BF16 eager training agrees with the independent reference when native BF16 is supported. |

Only the two named BF16 cases may skip, and only when the independently recorded
native capability check is false. Emulated BF16 support is recorded separately;
it does not establish native support on a T4. The 18 mandatory cases must all
pass. Missing, duplicate, unexpected, failed, errored, or skipped mandatory cases
prevent a passing CUDA gate. A nonzero pytest exit also prevents a pass.

The selected GPU is exposed as `cuda:0`; the runner limits visibility to the
first device already visible to the caller. It sets deterministic cuBLAS
configuration before importing PyTorch, and limits numerical CPU threads to two.
Only `tests/test_cuda_validation.py` runs, with no distributed suite or workers.

## Installed-package proof and retained evidence

The notebook invokes:

```sh
python benchmarks/validate_cuda.py \
  --installed-package \
  --wheel /path/to/pytorch_dlrs-0.3.0.dev0-py3-none-any.whl \
  --output /path/to/new-evidence-directory \
  --timeout 300
```

The output directory and its adjacent `<directory>.tar.gz` archive must both be
new. Archive creation is exclusive, so a later collision cannot overwrite an
existing archive either.
The runner does not install packages. `--installed-package` requires the exact
wheel that was installed. It retains that wheel, its hash and dependency
metadata, and verifies that its core files match the fingerprinted source.

Both the environment probe and pytest use Python isolated mode, a temporary
working directory outside the checkout, and a sanitized import environment.
Only the self-contained CUDA test file is copied there. The probe must import
DLRS from site-packages, outside the checkout, with the expected version and
source hashes. Imported core, reviewed source, copied test, and wheel hashes are
checked again after testing. Source-mode execution omits both package arguments
and is explicitly recorded as `checkout_source`; it is not installed-wheel proof.

The evidence bundle includes:

- `summary.json`: outcome, exit code, required gate, optional BF16 status, and
  every JUnit case outcome, including skip reasons.
- `environment.json`: Python/platform, installed package versions, PyTorch
  build, CUDA/cuDNN versions, selected GPU model/memory/compute capability,
  capability checks, and the imported DLRS file identities.
- `source.json`, `wheel.json`, and the exact wheel: source hashes, Git revision
  and tracked-change status, package identity, and archive hash.
- `nvidia-smi.log`: GPU model, driver version, and memory; a missing driver tool
  is explicitly recorded and does not substitute for PyTorch's allocation and
  synchronization probe.
- `pytest.xml`, when produced, plus command records and continuously written
  logs. The notebook adds setup logs, before/after PyTorch identities, and its
  execution status.
- SHA-256 manifests for retained files and a printed hash for the downloaded
  bundle. No credential store, full environment-variable dump, or Git checkout
  is included.

The default pytest deadline is 300 seconds (maximum configurable 600), with a
separate 60-second environment probe and 10-second driver query. Normal timeout,
SIGTERM, or keyboard interruption terminates the active child and archives
partial evidence. If the process is killed abruptly, the download cell can
bundle files already written while the runtime remains available. A deleted or
terminated Colab runtime can lose its entire filesystem; no local-only workflow
can guarantee recovery after that event.

| Runner outcome | Exit code | CUDA gate |
| --- | --- | --- |
| Required cases pass; optional BF16 outcomes match capability | 0 | Passed within this scope |
| PyTorch cannot use CUDA | 2 | Unavailable; never passed |
| Failure, timeout, malformed/incomplete evidence, or source mismatch | 1 | Not passed |
| Keyboard interruption or SIGTERM | 130 | Not passed |

## Scope of the result

This checks execution, rate handling, overflow policy, checkpoint continuation,
and wheel imports for the **recorded** GPU/PyTorch/Python combination. It does not
establish compatibility for other GPUs or CUDA builds, CUDA graphs, fused or
capturable optimizers, compiled scheduler execution, distributed CUDA/NCCL,
multi-node recovery, sustained performance, or long-running training.

It does not change the existing workload-quality results. Passing correctness
checks cannot establish that DLRS will improve or preserve every user's model
quality; see [workload validation](WORKLOAD_VALIDATION.md) and the
[reliability audit](RELIABILITY_AUDIT.md).

Initial local preparation tested 21 CPU-safe runner regressions and confirmed that both
source and installed-wheel execution on the CUDA-free Mac produce exit 2,
`unavailable`, a false CUDA gate, and retained archives. Those are validation of
the runner's refusal behavior, not GPU results. Actual CUDA acceptance requires
the downloaded report from a completed GPU run.
