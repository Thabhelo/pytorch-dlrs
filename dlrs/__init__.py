"""
DLRS: Dynamic Learning Rate Scheduler for PyTorch

A PyTorch implementation of the Dynamic Learning Rate Scheduler from the paper:
"Improving Neural Network Training using Dynamic Learning Rate Schedule for
PINNs and Image Classification" (arXiv:2507.21749v1)

Author: Thabhelo (thabhelo@deepubuntu.com)
"""

from dlrs.scheduler import DLRSScheduler
from dlrs.utils import LossRecorder

__version__ = "0.1.0"
__author__ = "Thabhelo"
__email__ = "thabhelo@deepubuntu.com"

__all__ = ["DLRSScheduler", "LossRecorder"]
