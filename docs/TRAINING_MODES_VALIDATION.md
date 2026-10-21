# Local training-mode validation

**Scope:** The final Torch 2.14.1 module run passed all **24 tests: 23 integration cases and one checkpoint-comparator regression** on 2026-10-20, using actual CPU execution and the available Apple MPS accelerator. A separate Torch 2.0.0 run passed 13 cases and explicitly skipped 11 unsupported or unavailable cases. These checks extend compatibility evidence; they do not establish accuracy improvements, production-scale reliability, or universal non-regression.

The implementation tests are in [test_training_modes.py](../tests/test_training_modes.py). [Machine-readable evidence](training-modes-local.json) records the environment, source hashes, commands, results, and compiler probe. Tests ran in scoped groups to avoid competing with separate vision workloads on the GPU.

## Environment and results

Python 3.11.12, PyTorch 2.14.1, Lightning 2.6.6, Apple M3, macOS 26.6.2 arm64. CPU tests used two threads. MPS was built and available; CUDA was unavailable. `PYTORCH_ENABLE_MPS_FALLBACK` and `TORCH_COMPILE_DISABLE` were unset.

| Actual execution | Cases | Verified behavior |
|---|---:|---|
| CPU float64 accumulation, SGD and AdamW | 2 | Accumulated microbatches match concatenated effective batches within explicit numerical tolerances, including an unequal final microbatch. |
| CPU bf16 autocast + CPU GradScaler, SGD and AdamW | 2 | Predictions actually have bf16 dtype; injected nonfinite gradients cause a real skipped optimizer update, no parameter change, and no loss observation for that update. |
| CPU bf16, every update skipped, SGD and AdamW | 2 | Optimizer and scheduler checkpoint states remain unchanged while the scaler backs off. |
| CPU dropout + shuffled inputs, float32 and bf16, SGD and AdamW | 4 | Epoch-boundary restart exactly reproduces subsequent losses, parameters, optimizer state, scheduler state, and scaler state. BF16 checkpoints follow an injected overflow/backoff event. |
| CPU tensor learning rates, SGD and AdamW | 2 | Stochastic model and optimizer continuation exactly match after checkpoint restoration, including scalar tensor LR state. |
| MPS float32, SGD and AdamW | 2 | Real accelerator training and epoch-boundary restart exactly reproduce subsequent losses and all saved model/optimizer/scheduler state. These MPS models do not use dropout. |
| MPS dropout + shuffled inputs, SGD and AdamW | 2 | Restoring CPU/MPS RNG and shuffle-generator state exactly reproduces subsequent stochastic training losses, model parameters, optimizer state, and scheduler state. |
| MPS scalar tensor learning rates, SGD and AdamW | 2 | Training retains LR tensor identity and device, respects configured bounds, and restores scheduler LR state in place. |
| CPU compiled model, SGD and AdamW | 2 | Dynamo invokes the AOT-eager backend; the model trains identically to its eager counterpart while the optimizer and scheduler remain eager. |
| Real Lightning CPU rejection paths | 3 | Accumulation of two batches, bf16 mixed precision, and another LR scheduler are rejected before the first training step. Model parameters remain unchanged. |

The latest complete module run, including the native MPS replay precondition,
passed **24 tests with no skips in 11.92 seconds** against the corrected scheduler.
It includes the strict checkpoint comparator, both stochastic MPS restart cases,
and assertions that the compiler backend actually receives graphs. An earlier
complete module run passed the same 24 cases in 13.09 seconds before the native
capability precondition replaced the version exclusion. The primary full-suite
run at that earlier test snapshot passed **315 tests with no skips** in 146.80
seconds. These historical test snapshots remain distinct in the evidence.

The comparator regression rejects Python floats substituted for scalar tensors
and mismatched container types, tensor dtypes, or devices. Earlier incremental
runs and source fingerprints remain in the JSON evidence. They are historical
checks, not extra independent cases or substitutes for the final complete run.

## Observation policy tested

Each scheduler observation is the **sample-weighted, unscaled loss of one completed optimizer update**. An effective batch has six samples, split into two microbatches of three; the final effective batch has five samples, split into three and two. Each microbatch loss is weighted by its sample count before backward, so the last update is not incorrectly divided by a constant accumulation factor.

A real optimizer step hook determines whether `GradScaler.step` executed the optimizer. Ordinary eager updates are counted after the direct `optimizer.step` call returns successfully; they do not depend on a hook. Only completed updates contribute observations. The test injects `Inf` into a gradient to exercise the actual scaler rejection path; the scaler and optimizer are not mocked. If every attempted update is skipped, the caller does not call DLRS. This policy is implemented in the test training loop, not inferred or enforced by the scheduler itself.

For mixed precision, scaling and update decisions belong to the optimizer-update boundary, after accumulated gradients are ready. See the [PyTorch AMP examples](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html). The CPU checks do not demonstrate CUDA scaler behavior, MPS mixed precision, or bf16 acceleration speed.

## Checkpoint requirements

The tested continuation saves the model, optimizer, scheduler, CPU RNG state, explicit shuffle-generator state, and scaler state where used. MPS RNG state is saved in the accelerator trials. Random state is restored **after** constructing and loading the replacement model and optimizer, because construction can consume randomness.

All continuation comparisons occur within the same installed framework and device configuration. Exact equality here does not promise equality across releases or hardware. PyTorch documents that reproducibility is conditional on platform and implementation details in its [reproducibility guidance](https://docs.pytorch.org/docs/2.14/notes/randomness.html). Mid-epoch continuation, worker-process RNG, external augmentation libraries, and data-stream positions need additional state and separate validation.

## Verified older-framework limitations and native capability gate

The actual Python 3.8.20 / Torch 2.0.0 floor run exposed two framework behaviors. A later Python 3.9.6 / Torch 2.4.1 run exposed the same MPS RNG limitation. The [native reproduction](../benchmarks/native_training_mode_probe.py) imports no DLRS. Its [Torch 2.0.0 output](native-training-mode-torch200.json), [Torch 2.4.1 output](native-training-mode-torch241.json), and [Torch 2.14.1 comparison](native-training-mode-torch2141.json) preserve the same probe's results and source hash.

* **Optimizer hooks after model compilation:** Torch 2.0.0 calls SGD/AdamW post-step hooks before `torch.compile(model)`, then stops calling them for both existing and newly constructed optimizers even though parameter values change. This made the original test helper misclassify real updates as skipped. The helper now counts successful direct eager optimizer calls explicitly; GradScaler cases still use actual hook-based skip detection. The floor's compiled-model and CPU tensor-LR restart cases now pass. DLRS itself does not use these hooks. The same native probe retains working hooks on Torch 2.14.1.
* **Stochastic MPS restart:** On the observed Torch 2.0.0 and 2.4.1 builds, saving and restoring the MPS generator state reports matching state bytes but does not replay subsequent native `torch.rand` or dropout outputs. Synchronizing before checkpoint capture does not repair the mismatch. Torch 2.14.1 replays all native probe outputs exactly and passes both stochastic training restart cases. These observations do not identify the first fixed release or establish behavior in untested patch releases.

After the helper correction and core alias fix, the floor training-mode module reports **13 passed, 11 skipped in 2.19 seconds**: six cases lack the CPU GradScaler API, two exclude the verified native MPS replay failure, and three lack optional Lightning. These skips are missing or unsupported coverage, not successful validation. The final Torch 2.14.1 run executes all 24 cases without skips.

The initial precise version exclusion was replaced after the 2.4.1 finding by an **actual native capability precondition**, independently reviewed before matrix continuation. Before creating any DLRS model or scheduler, each stochastic MPS test checks state advancement, reported state restoration, and exact replay of three native random outputs and three dropout outputs. It records the full per-operation outcome in JUnit suite properties. Only an explicitly detected native replay failure skips stochastic DLRS coverage; unexpected exceptions still fail, and subsequent DLRS continuation mismatches remain failures. Non-dropout MPS tests and all other applicable modes continue to execute. The preflight restores saved CPU/MPS state through the native APIs, but restored bytes cannot promise replay on an affected backend.

The new gate correctly records native `rand` and dropout failures and skips the two stochastic cases on Torch 2.4.1. On Torch 2.14.1, both capabilities succeed and the complete 24-test module passes again in **11.92 seconds**. The JSON preserves the earlier failures, exact-version exclusion run, and all subsequent capability-gated results with separate source fingerprints. A native-precondition skip is explicitly missing stochastic continuation coverage, never a scheduler compatibility pass.

## Compiler boundary

The supported pattern exercised here compiles only the model with AOT-eager and leaves `optimizer.step` and `scheduler.step` outside compilation. AOT-eager provides Dynamo/AOTAutograd coverage; it is not Inductor, Triton, GPU compiler, or speed evidence.

A separate read-only probe wrapped `DLRSScheduler.step` in `torch.compile(fullgraph=True)` with a counting backend. Python-float losses returned successfully but produced **zero captured graphs**. Tensor-valued losses raised a data-dependent-guard `UserError`, with optimizer and scheduler state unchanged. A successful float-only call must therefore not be described as compiled-scheduler support. Scheduler capture and compiled optimizer behavior remain unsupported.

## Framework boundary and reproduction

The Lightning callback retains its existing narrow envelope: one process, one optimizer, automatic optimization, full precision, no gradient accumulation, and no other configured scheduler. The new CPU tests validate rejection through actual Lightning `Trainer.fit`; they do not broaden callback acceptance. The separate existing Lightning checkpoint test covers its supported full-precision path.

With an environment containing the intended Torch version, pytest, and optional Lightning:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m pytest tests/test_training_modes.py -q
```

MPS cases skip when the accelerator is unavailable; stochastic MPS restart also requires the recorded native random/dropout replay capability. CPU GradScaler cases skip when the installed Torch version lacks that API; optional Lightning cases skip without Lightning. A skip records missing coverage, not a passing compatibility result. Version-specific results belong in the compatibility matrix rather than being inferred from this local run.
