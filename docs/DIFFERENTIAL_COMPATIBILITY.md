# Ordinary-use differential behavior check

On 2026-10-20, a bounded scheduler-only probe compared the revised scheduler with
the original commit `c0ea1d159e4caa4950358282506e9358ae9a67ea`. It found **no
unexpected numerical drift in the sampled ordinary-use domain**. This is not a
training-quality comparison or a universal backward-compatibility guarantee.

The probe used 1,000 independent seeded trajectories, each containing eight
epochs and one to four optimizer parameter groups: 8,000 loss sequences and
20,000 learning-rate comparisons. Half the trajectories used constructor
defaults; half used positive factors within the documented typical ranges and
explicit bounds. Losses were positive, finite, and well scaled. The three
slope regimes were sampled separately, at least 0.05 away from the zero/one
branch thresholds. Live rates stayed away from zero, configured bounds,
and floating-point decade boundaries. No optimizer parameter update was run.

Every rate, slope, and reported adjustment matched under `rtol=1e-12` and
`atol=1e-15`. Epoch counts, initial float base rates, and stored loss values also
matched. Of the 20,000 float-rate comparisons, 15,176 were exactly equal.
Maximum observed differences were:

| Quantity | Maximum difference |
|---|---:|
| Learning rate, absolute | `2.220446049250313e-16` |
| Learning rate, relative | `2.2039815680629813e-15` |
| Normalized slope, absolute | `1.1102230246251565e-15` |
| Reported adjustment, absolute | `4.163336342344337e-17` |
| Epoch counter | `0` |

The run used Python 3.11.12 and PyTorch 2.14.1 on macOS arm64, with one intra-op
and one inter-op CPU thread, and took 5.68 seconds including source retrieval.
The exact original scheduler and utilities were fetched at the pinned commit,
checked against recorded SHA-256 digests, and loaded from an isolated temporary
directory. The local Git read had failed because the cloud-backed `packed-refs`
file timed out; no Git metadata, branch, or checkout was changed.

The revised scheduler hash was
`ecd98f7d379af9cdf68492ee56d5d6248d526df2605ffa80d3b19d18ef8ddae7`.
The [machine-readable result](differential-compatibility-local.json) records
all source hashes, sampling seed, counts, tolerances, environment, and exclusions.
The [reproducible probe](../benchmarks/compare_scheduler_revisions.py) runs with:

```bash
python -m benchmarks.compare_scheduler_revisions \
  --output /tmp/dlrs-differential-comparison.json
```

Intentional changes remain outside this comparison and require migration review:

- Zero learning rates stay frozen; tiny positive losses are no longer skipped
  solely because their mean is below the old absolute threshold.
- Invalid, negative, boolean, and nonfinite observations/configuration, malformed
  checkpoints, and changed parameter-group topology are rejected.
- Numerical behavior at exact regime/decade boundaries, extreme magnitudes, and
  subnormal values is corrected rather than preserved from the baseline.
- Tensor learning-rate identity, initial/base metadata ownership, copied loss
  history, and checkpoint validation/ownership have stronger guarantees.
- The scalar wrapper honors `mode='max'` and bounds retained history. Its
  historical rolling-window `patience` behavior is documented explicitly.
- No-observation, short, and all-zero calls clear stale diagnostics; internal
  step counters, verbose output, and checkpoint fields intentionally differ.

Those changes are described in the [reliability audit](RELIABILITY_AUDIT.md).
This probe excludes tensor rates, wrapper behavior, checkpoint schema migration,
framework integration, other Torch versions, and optimizer/training outcomes.
Matching ordinary LR updates cannot establish that replacing another scheduler
preserves accuracy, convergence, or production reliability.
