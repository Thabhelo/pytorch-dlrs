# Changelog

All notable changes to pytorch-dlrs will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-10-09

Production-ready release with comprehensive testing, type safety, and ecosystem integrations.

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
