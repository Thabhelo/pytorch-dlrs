# Contributing

DLRS is an experimental independent package. Changes should preserve a stated, testable contract and distinguish numerical correctness from training-quality evidence. Report a reproducer with Python/PyTorch versions, optimizer, learning rates, losses, expected result, and the actual failure.

## Development checks

Use an isolated environment and install a PyTorch build appropriate for the machine before installing the development extras:

```bash
python -m pip install -e '.[dev]'
python -m pytest tests --cov=dlrs --cov-report=term-missing
ruff check dlrs tests examples benchmarks setup.py .github/scripts
mypy dlrs
python -m build
python -m twine check dist/*
```

`pyproject.toml` is the only metadata/dependency source. `setup.py` is a compatibility entry point. Test or dataset dependencies do not belong in runtime requirements. Test/development extras use NumPy 1.x below Python 3.13 for old PyTorch compatibility, and NumPy 2.x on Python 3.13+ where NumPy 1.x wheels are unavailable. The wheel must contain `dlrs/py.typed`; the source distribution must include tests, examples, and documentation.

For a correctness fix, add a regression test that fails on the old implementation and checks observable behavior. Useful coverage includes independent formula calculations, invalid-input atomicity, extreme finite values, frozen/tensor rates, fixed parameter groups, complete checkpoint continuation, and metric-window semantics. Tests that merely copy the implementation are not an independent check.

## CI scope

The consolidated workflow selects combinations with upstream wheels rather than taking an invalid full cross product:

| Platform | Python | PyTorch |
| --- | --- | --- |
| Linux | 3.8 | 2.0.0 (declared minimum) |
| Linux | 3.9 | 2.4.1 |
| Linux | 3.10 | 2.5.1 |
| Linux | 3.11 | 2.6.0 |
| Linux | 3.12 | 2.14.1 and latest available CPU wheel |
| Linux | 3.14 | 2.14.1 (NumPy 2.x test environment) |
| macOS | 3.12 | 2.14.1 |
| Windows | 3.11 | 2.6.0 |

The version boundaries remain explicit. The unpinned latest job detects future incompatibilities and prints the resolved version. Tooling is checked separately on Python 3.11 with PyTorch 2.14.1, Ruff 0.17.1, and mypy 2.4.0. This current typing reference describes scalar and tensor learning rates; older PyTorch annotations describe only float rates. The runtime matrix continues to check those older releases. A required integration job uses Python 3.11, PyTorch 2.14.1, and Lightning 2.6.6 on CPU to compare uninterrupted training against checkpoint/resume in fresh processes and verify that accumulation, mixed precision, and competing schedulers are rejected before optimizer updates; the core matrix does not require Lightning. Each job installs the package, runs dependency checks, and retains JUnit results, including skipped-capability evidence. The build job installs both the wheel and source archive into a clean environment and runs the full tests extracted from the source archive. It removes the extracted `dlrs` directory, verifies that imports resolve to the installed package, checks dependencies and metadata, and retains logs and artifact hashes. `.github/scripts/validate_distributions.py` performs the same validation locally against an isolated interpreter supplied with `--python`. Published release tags must match package metadata, and publication uses the verified artifacts.

This table describes configured CI, not evidence that the workflow has run for an unpushed local revision. Local results and unavailable environments are recorded in the [compatibility report](docs/COMPATIBILITY_VALIDATION.md) and [audit](docs/RELIABILITY_AUDIT.md). CPU CI does not establish CUDA/ROCm/MPS/NCCL, multi-node, AMP, graph-capture, framework-version, or large-scale coverage. Adding one of those claims requires an executable acceptance test on the actual stack and retained results.

## Training evidence

Compare against the existing workload scheduler, not only an untuned or disabled baseline. Match initial weights, data partitions/order, optimizer, training budget, tuning budget, and evaluation policy. Use validation data for choices; evaluate the held-out test set only after choices are fixed. Record multiple seeds, uncertainty, failures, resource use, device synchronization for timing, and the complete environment. A faster scheduler step is not evidence of better time-to-quality.

Any change to the equation, epsilon policy, bounds, loss reduction, wrapper semantics, state format, or framework ownership can change training trajectories. Document it in the unreleased changelog and migration notes. Do not describe a behavioral correction as guaranteed to leave every user's training unchanged.

## Correctness audits

Review the [contract](docs/CORRECTNESS_CONTRACT.md) and [verification evidence](docs/CORRECTNESS_VERIFICATION.md) when changing numerical or state behavior. The dedicated CI job installs `z3-solver==5.3.1.0`, requires all 25 mathematical obligations, exercises proof/audit failure guards and requires all 14 declared mutations to be detected. Z3 is development tooling, not a package runtime dependency. An absent solver in ordinary platform tests is an explicit skip; it cannot satisfy the separate proof job.

```bash
python benchmarks/verify_contract.py --output /tmp/dlrs-new-proof-evidence
python benchmarks/audit_mutations.py --output /tmp/dlrs-new-mutation-evidence
```

Both output paths must be new. Preserve failed attempts and review every surviving mutation; do not relabel infrastructure failures as detected defects. Extending the selected mutation set is useful, but its pass rate is not a probability that DLRS is correct.

## A possible PyTorch contribution

Passing this repository's tests is only one prerequisite. Before preparing a new core scheduler proposal, discuss whether the feature belongs in PyTorch, provide independent evidence of utility and adoption, specify the API and maintenance burden, and agree on the required regression/integration coverage. PyTorch's [contribution guidance](https://github.com/pytorch/pytorch/wiki/The-Ultimate-Guide-to-PyTorch-Contributions) and [technical contribution instructions](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md) govern an upstream contribution. The pinned upstream review in the [readiness ledger](docs/UPSTREAM_READINESS.md#upstream-decision-and-human-review) records triage routes through an `actionable` issue, repository write access or named maintainer preapproval, together with human-responsibility and AI-disclosure requirements. Recheck contribution and CLA requirements when preparing a proposal. A local audit cannot certify acceptance or replace upstream review and CI.

Keep a proposal additive: do not replace an existing default scheduler or alter users' optimizer behavior without explicit opt-in. Retain comparison tests showing existing optimizer and scheduler paths remain unchanged. Establish exact checkpoint, composition, device, and distributed contracts before claiming compatibility.
