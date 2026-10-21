# Predeclared bounded Transformer comparison

**Status: complete; the predeclared provisional mean-NLL criterion passed against both baselines.** All 24 trials completed with finite results. This is a separate workload study and does not alter the existing MNIST, CIFAR-10, or Helmholtz PINN protocol or results. It adds evidence for one small language-model configuration; it does not certify all Transformer workloads, modern LLM training, or production non-regression.

The frozen design is specified in [transformer_validation_protocol.json](../benchmarks/transformer_validation_protocol.json), with the separate [runner](../benchmarks/validate_transformer_workload.py) and [scientific-validity tests](../tests/test_transformer_validation.py). Results are retained under `docs/transformer-validation-2026-10-20/`, independently of the earlier study.

## Observed results

All nine tuning trials and all 15 confirmation trials completed: 9,216 optimizer updates and 9,437,184 training target bytes in total. Every policy selected initial LR **0.003**, the highest candidate in the declared grid. These are equal-search-budget results, not evidence that any policy was optimally tuned; extending the grid could change the comparison.

| Policy | Selected initial LR | Confirmation seeds completed | Mean final test NLL, nats/byte |
|---|---:|---:|---:|
| NoScheduler | 0.003 | 5/5 | 2.319081 |
| CosineAnnealingLR | 0.003 | 5/5 | 2.344672 |
| DLRS | 0.003 | 5/5 | 2.316681 |

Lower NLL is better. The predeclared criterion considers the paired mean difference and its exploratory percentile interval:

| Baseline | Mean DLRS minus baseline NLL | Pointwise 95% bootstrap interval | Upper endpoint ≤ 0.05? |
|---|---:|---:|---|
| NoScheduler | −0.002401 | [−0.004836, −0.000435] | Yes |
| CosineAnnealingLR | −0.027991 | [−0.040241, −0.015742] | Yes |

This is a pass of the provisional **mean** criterion, not every-run non-regression. For seed 11, DLRS was slightly worse than NoScheduler by 0.000615 nats/byte. The individual final test outcomes are retained here to make that distinction visible:

| Confirmation seed | NoScheduler | CosineAnnealingLR | DLRS |
|---|---:|---:|---:|
| 11 | 2.300125 | 2.344184 | 2.300740 |
| 12 | 2.329913 | 2.342793 | 2.328308 |
| 13 | 2.312359 | 2.329007 | 2.309308 |
| 14 | 2.311966 | 2.326015 | 2.310994 |
| 15 | 2.341045 | 2.381363 | 2.334054 |

DLRS's adaptation path was exercised: confirmation training-window rates ranged from 0.003 to approximately 0.003104, with both increases and decreases in the saved trajectories. This is evidence of modest adaptation over six windows, not long-horizon or large-adjustment stability.

The sum of recorded trial times was **1,027.67 seconds (17.13 minutes)**. The raw summary also retains paired median timing ratios of 0.756 against NoScheduler and 0.801 against cosine. Runs shared the laptop with changing concurrent work, and these ratios **must not be interpreted as DLRS speedups**. The study did not control contention sufficiently for a causal runtime comparison.

The [raw summary](transformer-validation-2026-10-20/summary.json), [selected rates](transformer-validation-2026-10-20/selected_lrs.json), [execution log](transformer-validation-2026-10-20/execution.log), and [all trial records](transformer-validation-2026-10-20/trials/) preserve the complete outcome, including per-window losses, learning rates, sample/initialization hashes, target counts, and timings. No trial was retried or shortened, and no tuning record contains a test result.

An [independent summary check](transformer-validation-2026-10-20/independent-summary-check.json) verified all 24 raw records, the tuning grid and selections, seed coverage, policy means, pairing hashes, canonical manifest identity, and final acceptance. Its [separate pure-Python helper](../benchmarks/check_workload_summary.py) recomputed the percentile endpoints using 126 multinomial-weighted bootstrap count vectors, rather than the runner's 3,125 ordered resamples. Both intervals matched exactly. This verifies the recorded calculation; it does not enlarge the scientific coverage or remove the five-seed uncertainty limitations.

## Data and held-out boundary

The corpus is the Tiny Shakespeare text distributed by [karpathy/char-rnn at commit 6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e](https://github.com/karpathy/char-rnn/blob/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt). Shakespeare's original text is public domain. The downloaded artifact is pinned independently by SHA-256 `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed` and length 1,115,394 bytes. It is stored under `/tmp/dlrs-validation-data/`; the corpus itself is not added to the repository.

The contiguous split is fixed before outcomes:

| Split | Half-open byte range | Bytes |
|---|---|---:|
| Training | `[0, 892315)` | 892,315 |
| Validation | `[892315, 1003854)` | 111,539 |
| Test | `[1003854, 1115394)` | 111,540 |

Inputs are raw bytes with a fixed 256-symbol vocabulary. There is no learned tokenizer or discovery of held-out vocabulary. No context crosses a split boundary. Whole-file checksum verification reads all bytes for integrity, but tuning constructs only training and validation token tensors and never computes test outputs or metrics. All nine tuning trials finish before rates are selected and any confirmation test evaluation begins.

Validation and test NLL score every target byte except the first byte of their respective split exactly once. Consecutive 64-target blocks reset context at block boundaries; a final partial block is included with its actual target count. This is a fixed short-context next-byte evaluation, not a sliding full-context LLM perplexity benchmark.

## Fixed model and training budget

The model has two causal pre-LayerNorm Transformer blocks, width 64, four attention heads, feedforward width 128, learned positions, dropout 0.1, a final LayerNorm, and an untied byte-output projection. Layers are constructed and initialized independently. The model contains 103,936 parameters. Causal masking is tested by changing future input bytes and requiring earlier logits to remain unchanged.

Each trial uses AdamW with betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.01`, and gradient norm clipping at `1.0`. Execution is CPU float32, at most two Torch threads, with deterministic algorithms requested. There is no mixed precision, compilation, gradient accumulation, data-worker process, or accelerator training in this study.

Every trial receives **384 optimizer updates**, arranged as six scheduler windows of 64 updates. Each update samples 16 independent contiguous sequences of 64 training target bytes with replacement. This is 393,216 training target bytes per trial and 9,437,184 across all 24 planned trials. It is a bounded sampled training run, not a complete training-corpus epoch. A separate sampling generator and the same model/dropout seed produce matched initialization and sample-index streams across policies for each paired seed; hashes verify both.

The compared policies are `NoScheduler`, `CosineAnnealingLR`, and `DLRS`. All use the same training horizon, optimizer, data, and model. Cosine steps once per window with `T_max=6`; DLRS receives the 64 ordered, unscaled minibatch losses at that same boundary, using its default factors and bounds `[1e-6, 0.01]`. The learning rate used for each window is recorded separately from the final post-training rate.

There are three initial LR candidates per policy: `0.0003`, `0.001`, and `0.003`, all on tuning seed 0. The lowest **final validation NLL** selects one rate per policy, with declared grid order breaking ties. There is no early stopping, best-epoch selection, retry of failed candidates, or result-driven change to the grid/horizon. Confirmation uses five new seeds, 11–15, for each selected policy. Policy order rotates across candidates/seeds to avoid always timing the same policy first; this does not make shared-machine timing controlled evidence.

The fixed update and token counts bound computation. There is **no default wall-clock cutoff**: the initial 10–20 minute estimate is a planning target, not a user-imposed deadline. A separate synthetic-token timing pilot informs the final reviewed design without reading real text or inspecting language-model quality. Contention may make the complete 24-trial run slower. An explicitly configured resource interruption would be retained as incomplete; the horizon is never shortened to manufacture a completed comparison.

The [synthetic timing pilot](transformer-validation-2026-10-20/synthetic_timing.json) used the exact model and batch shapes with uniformly random bytes, four warm-up updates, and 16 measured updates. With two CPU threads it measured a median 0.1638 seconds per update and projected approximately 1,668 seconds (27.8 minutes) for all 24 trials plus held-out evaluation. This is an estimate under possible concurrent work, excludes startup and scheduler/I/O costs, and is not a runtime guarantee. No corpus tokens or language-model quality outcomes were accessed by this pilot. The reviewed budget remained 384 updates per trial.

## Provisional comparison criterion

The primary metric is final held-out next-byte NLL in natural-log units, lower being better. The criterion is declared before Transformer outcomes and is explicitly **PROVISIONAL**:

* All 15 confirmation trials must finish with finite results.
* For each tuned baseline, form five paired `DLRS NLL - baseline NLL` differences.
* Enumerate all `5^5 = 3,125` ordered bootstrap resamples and take a pointwise 95% percentile interval for the mean difference.
* The interval's upper endpoint must be at most **0.05 nats per byte** against both baselines.

The margin corresponds to approximately `exp(0.05) = 1.0513` times perplexity. It is a provisional research bound, not a downstream production risk budget or a guarantee that a 5% change is acceptable to users. Five seeds provide limited uncertainty evidence. Exhaustively enumerating bootstrap resamples does not make the nominal interval coverage exact, and the two pointwise intervals are not simultaneous confidence guarantees. Failures, missing pairs, mismatched initialization/data streams, or budget exhaustion prevent acceptance. A pass applies only to this corpus, model, horizon, and equal LR-search budget.

## Reproducibility and execution gate

Preparation verifies the corpus and writes an output manifest containing the complete protocol, source hashes, data split, and environment. Existing output directories reject changed protocol, code, corpus, or environment. Every trial is saved atomically, including failed or incomplete trials; completed matching trials are reused without silently retrying failures. Test metrics remain out of tuning records. The selected rates and final summary are separate artifacts.

Prepare without training:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m benchmarks.validate_transformer_workload --prepare-only
```

After protocol review, execute the frozen study:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m benchmarks.validate_transformer_workload
```

The synthetic unit tests check causal masking in training and evaluation paths, complete target accounting and the next-byte shift, test-set isolation during tuning, paired initialization and sample streams, numerical failure handling before optimizer mutation, selection rules, acceptance completeness, corpus/manifest integrity, optional resource-limit handling, and retained/corrupted trial records. All 20 cases passed on the primary Python 3.11.12 / Torch 2.14.1 environment in 10.84 seconds before execution; Ruff also passed. These fixtures are correctness checks and contribute no scientific workload outcomes. Cached records must also match the declared update/token budgets and finite-metric schema. The manifest hashes the actually imported DLRS source, including installed-package runs, and records NumPy and actual thread settings. Incomplete execution returns a nonzero status; a completed comparison that fails the provisional quality criterion is a valid negative result.

The [manifest](transformer-validation-2026-10-20/manifest.json) was created before scientific training at `2026-10-20T21:10:54.447810+00:00`. The protocol SHA-256 is `bb8eb48c26ed42d4b263fab58173987f0d27b6f79ab310d2563417546fe0cb4e`; the runner SHA-256 is `b6997e000189df06b563623e2d20e9d5a304039ab2bcbba067c4c5015183e35c`; the tested scheduler SHA-256 is `ecd98f7d379af9cdf68492ee56d5d6248d526df2605ffa80d3b19d18ef8ddae7`. NumPy is 1.26.4, the platform is macOS 26.6.2 arm64, actual Torch thread count is two, and deterministic algorithms are enabled. Independent review and the synthetic checks completed before the 24-trial run was authorized and started. The process exited successfully after saving all 24 records and the final summary; the frozen protocol and runner were not changed during execution.

After the study, PR review identified a command-line exit-status defect: an
ordinary trial exception was retained as `failed`, but the command could still
exit zero. The corrected command records `execution_passed` and returns status
2 if any planned trial fails, remains incomplete, or is missing. A complete
finite comparison still returns zero when it produces a negative quality result;
consumers must inspect `accepted` separately. Failed tuning candidates remain
excluded from rate selection, but still make the overall execution unsuccessful.
Five focused synthetic CLI cases cover successful execution, negative quality,
tuning/confirmation failures, resource interruption, and reuse of cached failures.

This post-study correction changes only execution reporting. It does not change
training, budgets, selection, the scheduler, or any of the 24 saved trials. Their
historical manifest and runner hash above remain intact. The exact
[runner used for the study is retained at commit `11def238`](https://github.com/Thabhelo/pytorch-dlrs/blob/11def2388473e488cf8111d919aa1073524a2aa9/benchmarks/validate_transformer_workload.py).
Use that pinned revision to reproduce the recorded source fingerprint; the
corrected runner requires a new output directory because its source hash differs.

Remaining external gaps include CUDA, multi-node/sharded training, long contexts, large vocabularies/models, instruction tuning, pretrained-model fine-tuning, realistic production text streams, and long-run endurance. A small byte Transformer on public-domain text does not resolve those gaps.

The host also slept during the study and resumed the existing process. The
[execution notes](transformer-validation-2026-10-20/execution-environment-notes.json)
record observed suspension examples. The 17.13 minutes is the sum of recorded
trial timer intervals, not uninterrupted elapsed wall time; it cannot establish
training speed or a speedup over either baseline.
