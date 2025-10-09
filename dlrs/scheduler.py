"""
Dynamic Learning Rate Scheduler (DLRS) for PyTorch

Implementation of the DLRS algorithm from:
"Improving Neural Network Training using Dynamic Learning Rate Schedule for
PINNs and Image Classification" (arXiv:2507.21749v1)
"""

import math
import warnings
from typing import List, Union
import torch
from torch.optim.lr_scheduler import LRScheduler


class DLRSScheduler(LRScheduler):
    """
    Dynamic Learning Rate Scheduler that adjusts learning rate based on loss dynamics.

    The scheduler analyzes the trend of batch losses within an epoch to determine
    whether the model is converging, diverging, or stagnating, and adjusts the
    learning rate accordingly.

    Algorithm:
        1. Collect batch losses during an epoch
        2. Compute normalized loss slope: ΔL_j = (L_last - L_first) / L_mean
        3. Compute adjustment granularity: n = floor(log10(α_j))
        4. Calculate adjustment: α_δ_j = 10^n × δ_case × ΔL_j
        5. Update learning rate: α_{j+1} = α_j - α_δ_j

    When ΔL_j < 0 (loss decreasing), the subtraction becomes addition (LR increases).

    Parameters:
        optimizer (torch.optim.Optimizer): Optimizer to adjust learning rate for
        delta_d (float): Decremental factor for divergence (ΔL_j > 1). Default: 0.5
        delta_o (float): Stagnation factor for flat regions (0 <= ΔL_j < 1). Default: 1.0
        delta_i (float): Incremental factor for convergence (ΔL_j < 0). Default: 0.1
        min_lr (float): Minimum learning rate bound. Default: 1e-8
        last_epoch (int): The index of last epoch. Default: -1
        verbose (bool): If True, prints a message to stdout for each update. Default: False

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        >>> scheduler = DLRSScheduler(optimizer, delta_d=0.5, delta_o=1.0, delta_i=0.1)
        >>>
        >>> for epoch in range(100):
        >>>     batch_losses = []
        >>>     for batch in dataloader:
        >>>         loss = train_step(batch)
        >>>         batch_losses.append(loss.item())
        >>>     scheduler.step(batch_losses)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        delta_d: float = 0.5,
        delta_o: float = 1.0,
        delta_i: float = 0.1,
        min_lr: float = 1e-8,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.delta_d = delta_d
        self.delta_o = delta_o
        self.delta_i = delta_i
        self.min_lr = min_lr
        self.verbose = verbose

        self._validate_hyperparameters()

        self.batch_losses = []
        self.loss_slope = 0.0
        self.adjustment = 0.0

        super().__init__(optimizer, last_epoch)

    def _validate_hyperparameters(self):
        """Validate hyperparameter values."""
        if not isinstance(self.delta_d, (int, float)) or self.delta_d <= 0:
            raise ValueError(f"delta_d must be positive, got {self.delta_d}")
        if not isinstance(self.delta_o, (int, float)) or self.delta_o <= 0:
            raise ValueError(f"delta_o must be positive, got {self.delta_o}")
        if not isinstance(self.delta_i, (int, float)) or self.delta_i <= 0:
            raise ValueError(f"delta_i must be positive, got {self.delta_i}")
        if not isinstance(self.min_lr, (int, float)) or self.min_lr <= 0:
            raise ValueError(f"min_lr must be positive, got {self.min_lr}")

        if self.delta_d < 0.1 or self.delta_d > 1.0:
            warnings.warn(f"delta_d={self.delta_d} is outside typical range [0.1, 1.0]")
        if self.delta_o < 0.5 or self.delta_o > 2.0:
            warnings.warn(f"delta_o={self.delta_o} is outside typical range [0.5, 2.0]")
        if self.delta_i < 0.01 or self.delta_i > 0.5:
            warnings.warn(f"delta_i={self.delta_i} is outside typical range [0.01, 0.5]")

    def step(self, batch_losses: Union[List[float], None] = None):
        """
        Update learning rate based on batch losses from the current epoch.

        Parameters:
            batch_losses (List[float]): List of loss values from each batch in the epoch.
                                       If None, performs standard step without adjustment.
        """
        if batch_losses is None:
            super().step()
            return

        if len(batch_losses) < 2:
            warnings.warn("Need at least 2 batch losses to compute slope, skipping update")
            super().step()
            return

        self.batch_losses = batch_losses
        self._update_learning_rate()

        self.last_epoch += 1
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _update_learning_rate(self):
        """Compute and apply learning rate adjustment based on loss dynamics."""
        first_loss = self.batch_losses[0]
        last_loss = self.batch_losses[-1]
        mean_loss = sum(self.batch_losses) / len(self.batch_losses)

        if abs(mean_loss) < 1e-10:
            warnings.warn("Mean loss is near zero, skipping learning rate update")
            return

        self.loss_slope = (last_loss - first_loss) / mean_loss

        for group in self.optimizer.param_groups:
            current_lr = group['lr']

            n = math.floor(math.log10(current_lr)) if current_lr > 0 else -8
            granularity = 10 ** n

            if self.loss_slope > 1.0:
                delta = self.delta_d
            elif 0 <= self.loss_slope <= 1.0:
                delta = self.delta_o
            else:
                delta = self.delta_i

            self.adjustment = granularity * delta * self.loss_slope
            new_lr = current_lr - self.adjustment

            new_lr = max(new_lr, self.min_lr)

            group['lr'] = new_lr

            if self.verbose:
                print(f"Epoch {self.last_epoch + 1}: "
                      f"loss_slope={self.loss_slope:.6f}, "
                      f"adjustment={self.adjustment:.6e}, "
                      f"lr: {current_lr:.6e} -> {new_lr:.6e}")

    def get_lr(self):
        """
        Compute learning rates for each parameter group.
        Required by LRScheduler base class.
        """
        return [group['lr'] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        """Return last computed learning rate."""
        return self._last_lr if hasattr(self, '_last_lr') else [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        """Return scheduler state for checkpointing."""
        state = {
            'delta_d': self.delta_d,
            'delta_o': self.delta_o,
            'delta_i': self.delta_i,
            'min_lr': self.min_lr,
            'last_epoch': self.last_epoch,
            'loss_slope': self.loss_slope,
            'adjustment': self.adjustment,
            '_last_lr': self._last_lr if hasattr(self, '_last_lr') else None
        }
        return state

    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint."""
        self.delta_d = state_dict['delta_d']
        self.delta_o = state_dict['delta_o']
        self.delta_i = state_dict['delta_i']
        self.min_lr = state_dict['min_lr']
        self.last_epoch = state_dict['last_epoch']
        self.loss_slope = state_dict.get('loss_slope', 0.0)
        self.adjustment = state_dict.get('adjustment', 0.0)
        if state_dict.get('_last_lr') is not None:
            self._last_lr = state_dict['_last_lr']
