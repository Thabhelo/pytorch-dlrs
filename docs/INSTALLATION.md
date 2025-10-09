# Installation Guide

## Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher
- NumPy 1.21 or higher

## Installation from Source

### Step 1: Clone the Repository

```bash
git clone https://github.com/Thabhelo/pytorch-dlrs.git
cd pytorch-dlrs
```

### Step 2: Create Virtual Environment (Recommended)

Using venv:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Using conda:
```bash
conda create -n dlrs python=3.10
conda activate dlrs
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Package

For development (editable install):
```bash
pip install -e .
```

For regular installation:
```bash
pip install .
```

## Verify Installation

Test that DLRS is properly installed:

```python
import torch
from dlrs import DLRSScheduler

model = torch.nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
scheduler = DLRSScheduler(optimizer)

print("DLRS successfully installed!")
```

## Running Tests

Verify everything works by running the test suite:

```bash
pytest tests/ -v
```

## Running Examples

Try the MNIST example:

```bash
python examples/mnist_example.py --epochs 5
```

## Tested Configurations

DLRS has been tested with:

- Python: 3.8, 3.9, 3.10, 3.11
- PyTorch: 2.0, 2.1, 2.2
- Operating Systems: Linux, macOS, Windows

## GPU Support

DLRS works with both CPU and GPU. To use GPU:

```bash
python examples/mnist_example.py --device cuda
```

For Apple Silicon (M1/M2):

```bash
python examples/mnist_example.py --device mps
```

## Troubleshooting

### Import Error

If you get an import error:
```
ModuleNotFoundError: No module named 'dlrs'
```

Make sure you installed the package:
```bash
pip install -e .
```

### PyTorch Version

If you encounter compatibility issues, ensure PyTorch 2.0+ is installed:
```bash
pip install --upgrade torch
```

### Test Failures

If tests fail, ensure all dependencies are installed:
```bash
pip install -r requirements.txt
pytest tests/ -v
```
