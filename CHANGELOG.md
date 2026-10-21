# Changelog

All notable changes to pytorch-dlrs will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0.dev0] - 2026-10-21

This reliability revision remains experimental. It corrects implementation and
integration defects; it does not certify large-scale readiness, convergence,
performance superiority, or PyTorch acceptance. Intentional behavior changes are
listed in [migration notes](docs/FAQ.md#migrating-from-021).

### Fixed
- Select decimal decades from the exact binary64 rate, including normal values
  immediately below a decimal boundary; this can change ordinary LR trajectories.
- Preserve tiny loss endpoints during normalization using a rational fallback;
  reject a nonzero scheduler slope that cannot be represented as a float.
- Reject negative exact real inputs before they can round to signed zero.
- Isolate the exceptional Decimal arithmetic from application context settings.
- Round CPU float16/bfloat16 tensor bounds inward on PyTorch 2.0, whose CPU Half
  implementation lacks `nextafter`.
- Validate finite nonnegative observations and learning rates, finite positive
  hyperparameters, and checkpoint shape/content before applying partial updates.
- Preserve frozen zero-rate groups, scalar tensor LR identity, and tensor-valued
  base-rate metadata when restoring checkpoints; keep tensor initial/base rates
  independent from live LR storage on older PyTorch versions.
- Compute slopes without the old near-zero cutoff or avoidable overflow.
- Apply `DLRSOnPlateau(mode='max')` in the correct direction and bound its history.
- Restore utility model modes and reject invalid recorded losses.
- Correct the PINN example's forcing term and restrict callback examples to
  explicitly supported lifecycles; disable the incomplete Transformers callback.
- Remove incorrect rank-0-only DDP and scheduler-chaining instructions.
- Centralize metadata in `pyproject.toml`; separate runtime, tests, and example
  dependencies; include type markers and source documentation in distributions.

### Changed
- Add a versioned correctness contract, independent numerical/state tests,
  solver-checked mathematical model and a required deliberate-defect audit.
  These do not constitute a proof of the complete implementation or model quality.
- Clarify the metric wrapper's historical rolling-window meaning of `patience`;
  it is not a `ReduceLROnPlateau` replacement.
- Consolidate CI with explicit Python/PyTorch boundaries, isolated tooling,
  interpreter-specific NumPy test dependencies, full test runs against
  independently installed wheel/source archives, required
  real Lightning integration checks, and release tag/version validation.
- Replace unsupported universal compatibility, negligible-overhead, and
  production-readiness claims with measured scope and validation requirements.

## [0.2.1] - 2025-10-10

### Fixed
- PEP 621 license format compliance (changed to `{text = "MIT"}`)
- Mypy configuration for cross-version compatibility (disabled `warn_unused_ignores`)
- GitHub Actions CI upgraded artifact actions from v3 to v4

## [0.2.0] - 2025-10-09

Added testing, type annotations, and ecosystem examples. The original release described itself as production-ready; that claim was not supported by large-scale or broad integration evidence and is withdrawn. The package remains experimental.

### Added
- **API Improvements**:
  - `max_lr` parameter to cap learning rate growth
  - `DLRSOnPlateau` class for ReduceLROnPlateau-style API compatibility
  - Full type hints across all modules (PEP 484 compliant)
  - `py.typed` marker for type checking support

- **Documentation**:
  - Comprehensive FAQ (docs/FAQ.md) covering troubleshooting, multi-GPU, and advanced usage
  - Design notes (docs/DESIGN.md) explaining mathematical foundations and algorithm details
  - Updated README with citation matching arXiv journal reference

- **Examples**:
  - PINNs demo (examples/pinns_helmholtz.py) solving 1D Helmholtz equation
  - CUDA-enabled examples defaulting to GPU when available
  - PyTorch Lightning callback (examples/lightning_callback.py)
  - Hugging Face Transformers callback (examples/huggingface_callback.py)

- **Benchmarking**:
  - Comprehensive benchmark script (benchmarks/run_benchmarks.py)
  - Compares DLRS vs NoScheduler, StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau, AdamW+Cosine
  - Generates markdown reports with accuracy and timing metrics

- **Packaging & CI**:
  - PEP 621 compliant pyproject.toml
  - Automated GitHub Actions CI workflow with:
    - Multi-Python version testing (3.8-3.12)
    - Linting (ruff), type checking (mypy), and test coverage
    - Wheel building and PyPI publishing on release
  - Ruff and mypy configuration for code quality

### Changed
- **State Management**:
  - `state_dict()` now saves per-param-group learning rates
  - `load_state_dict()` restores LR state for each param group
  - Improved checkpoint compatibility for resuming training

- **Safety & Robustness**:
  - Added upper bound clamping when `max_lr` is set
  - Enhanced numerical stability checks for zero/NaN handling
  - Improved warnings for edge cases (zero mean loss, insufficient batch losses)

- **Documentation**:
  - Multi-param-group behavior clearly documented
  - API parity notes explaining differences from standard schedulers
  - Examples updated to use CUDA by default

### Fixed
- State dict now properly saves and restores optimizer learning rates
- License format in pyproject.toml updated to SPDX-compliant string

## [0.1.1] - 2025-10-09

### Added
- Initial PyPI release
- Core DLRS algorithm implementation
- MNIST example
- Basic tests
- README with usage instructions

### Changed
- Renamed Author section to Code Author in README

### Fixed
- Citation updated with journal reference (MLWA 2025)

## [0.1.0] - 2025-07-29

- Initial development release
