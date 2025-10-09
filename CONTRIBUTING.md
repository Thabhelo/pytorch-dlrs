# Contributing to pytorch-dlrs

Thank you for considering contributing to pytorch-dlrs. This document outlines the process for contributing to this project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/Thabhelo/pytorch-dlrs.git
   cd pytorch-dlrs
   ```
3. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

Install the package in development mode with test dependencies:

```bash
pip install -e .
pip install pytest pytest-cov
```

## Making Changes

1. Make your changes in your feature branch
2. Add or update tests as needed
3. Ensure all tests pass:
   ```bash
   pytest tests/ -v
   ```
4. Update documentation if you are adding new features

## Code Style

- Follow PEP 8 guidelines for Python code
- Write clear, descriptive commit messages
- Keep functions focused and well-documented
- Add docstrings to new functions and classes

## Testing

Before submitting your changes:

```bash
# Run all tests
pytest tests/ -v

# Check code coverage
pytest tests/ --cov=dlrs --cov-report=html
```

All tests must pass before your contribution can be merged.

## Submitting Changes

1. Commit your changes:
   ```bash
   git add .
   git commit -m "Brief description of your changes"
   ```

2. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

3. Create a Pull Request on GitHub with:
   - Clear description of the changes
   - Reference to any related issues
   - Results of test runs

## Reporting Issues

When reporting issues, please include:

- Python version
- PyTorch version
- Operating system
- Steps to reproduce the issue
- Expected vs actual behavior
- Error messages or logs

## Questions

If you have questions about contributing, please open an issue on GitHub or contact the maintainer at thabhelo@deepubuntu.com.
