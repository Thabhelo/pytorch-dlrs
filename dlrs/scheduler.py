"""
Dynamic Learning Rate Scheduler (DLRS) for PyTorch

Implementation of the DLRS algorithm from:
"Improving Neural Network Training using Dynamic Learning Rate Schedule for
PINNs and Image Classification" (arXiv:2507.21749v1)
"""

import math
import warnings
from typing import List, Optional

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
        max_lr: Optional[float] = None,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.delta_d = delta_d
        self.delta_o = delta_o
        self.delta_i = delta_i
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.verbose = verbose

        self._validate_hyperparameters()

        self.batch_losses: List[float] = []
        self.loss_slope: float = 0.0
        self.adjustment: float = 0.0

        super().__init__(optimizer, last_epoch)

    def _validate_hyperparameters(self) -> None:
        """Validate hyperparameter values."""
        if not isinstance(self.delta_d, (int, float)) or self.delta_d <= 0:
            raise ValueError(f"delta_d must be positive, got {self.delta_d}")
        if not isinstance(self.delta_o, (int, float)) or self.delta_o <= 0:
            raise ValueError(f"delta_o must be positive, got {self.delta_o}")
        if not isinstance(self.delta_i, (int, float)) or self.delta_i <= 0:
            raise ValueError(f"delta_i must be positive, got {self.delta_i}")
        if not isinstance(self.min_lr, (int, float)) or self.min_lr <= 0:
            raise ValueError(f"min_lr must be positive, got {self.min_lr}")

        if self.max_lr is not None:
            if not isinstance(self.max_lr, (int, float)) or self.max_lr <= 0:
                raise ValueError(f"max_lr must be positive or None, got {self.max_lr}")
            if self.max_lr <= self.min_lr:
                raise ValueError(f"max_lr ({self.max_lr}) must be greater than min_lr ({self.min_lr})")

        if self.delta_d < 0.1 or self.delta_d > 1.0:
            warnings.warn(f"delta_d={self.delta_d} is outside typical range [0.1, 1.0]", stacklevel=2)
        if self.delta_o < 0.5 or self.delta_o > 2.0:
            warnings.warn(f"delta_o={self.delta_o} is outside typical range [0.5, 2.0]", stacklevel=2)
        if self.delta_i < 0.01 or self.delta_i > 0.5:
            warnings.warn(f"delta_i={self.delta_i} is outside typical range [0.01, 0.5]", stacklevel=2)

    def step(self, batch_losses: Optional[List[float]] = None) -> None:  # type: ignore[override]
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
            warnings.warn("Need at least 2 batch losses to compute slope, skipping update", stacklevel=2)
            super().step()
            return

        self.batch_losses = batch_losses
        self._update_learning_rate()

        self.last_epoch += 1
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _update_learning_rate(self) -> None:
        """Compute and apply learning rate adjustment based on loss dynamics."""
        first_loss = self.batch_losses[0]
        last_loss = self.batch_losses[-1]
        mean_loss = sum(self.batch_losses) / len(self.batch_losses)

        if abs(mean_loss) < 1e-10:
            warnings.warn("Mean loss is near zero, skipping learning rate update", stacklevel=2)
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
            if self.max_lr is not None:
                new_lr = min(new_lr, self.max_lr)

            group['lr'] = new_lr

            if self.verbose:
                print(f"Epoch {self.last_epoch + 1}: "
                      f"loss_slope={self.loss_slope:.6f}, "
                      f"adjustment={self.adjustment:.6e}, "
                      f"lr: {current_lr:.6e} -> {new_lr:.6e}")

    def get_lr(self) -> List[float]:  # type: ignore[override]
        """
        Compute learning rates for each parameter group.
        Required by LRScheduler base class.
        """
        return [group['lr'] for group in self.optimizer.param_groups]

    def get_last_lr(self) -> List[float]:
        """Return last computed learning rate."""
        return self._last_lr if hasattr(self, '_last_lr') else [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing."""
        state = {
            'delta_d': self.delta_d,
            'delta_o': self.delta_o,
            'delta_i': self.delta_i,
            'min_lr': self.min_lr,
            'max_lr': self.max_lr,
            'last_epoch': self.last_epoch,
            'loss_slope': self.loss_slope,
            'adjustment': self.adjustment,
            '_last_lr': self._last_lr if hasattr(self, '_last_lr') else None,
            'param_group_lrs': [group['lr'] for group in self.optimizer.param_groups]
        }
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state from checkpoint."""
        self.delta_d = state_dict['delta_d']
        self.delta_o = state_dict['delta_o']
        self.delta_i = state_dict['delta_i']
        self.min_lr = state_dict['min_lr']
        self.max_lr = state_dict.get('max_lr', None)
        self.last_epoch = state_dict['last_epoch']
        self.loss_slope = state_dict.get('loss_slope', 0.0)
        self.adjustment = state_dict.get('adjustment', 0.0)
        if state_dict.get('_last_lr') is not None:
            self._last_lr = state_dict['_last_lr']

        if 'param_group_lrs' in state_dict:
            for group, lr in zip(self.optimizer.param_groups, state_dict['param_group_lrs']):
                group['lr'] = lr


class DLRSOnPlateau:
    """
    ReduceLROnPlateau-style wrapper around DLRSScheduler for drop-in compatibility.

    This variant provides a metric-based API similar to ReduceLROnPlateau, where you
    call step(metric) once per epoch with a single scalar metric. Under the hood, it
    uses a LossRecorder to collect per-batch losses and applies DLRS updates.

    Use this when you want standard scheduler API compatibility. For fine-grained
    control over batch-level metrics, use DLRSScheduler directly.

    Parameters:
        optimizer (torch.optim.Optimizer): Optimizer to adjust learning rate for
        mode (str): One of 'min' or 'max'. In 'min' mode, lr will be reduced when
                   the metric has stopped decreasing. Default: 'min'
        delta_d (float): Decremental factor for divergence (ΔL_j > 1). Default: 0.5
        delta_o (float): Stagnation factor for flat regions (0 <= ΔL_j < 1). Default: 1.0
        delta_i (float): Incremental factor for convergence (ΔL_j < 0). Default: 0.1
        min_lr (float): Minimum learning rate bound. Default: 1e-8
        max_lr (float): Maximum learning rate bound. Default: None
        patience (int): Number of epochs to wait before applying DLRS after metric stops
                       improving. Default: 0 (immediate)
        verbose (bool): If True, prints a message to stdout for each update. Default: False

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> scheduler = DLRSOnPlateau(optimizer, mode='min', patience=2)
        >>>
        >>> for epoch in range(100):
        >>>     train_loss = train_one_epoch(model, dataloader, optimizer)
        >>>     val_loss = validate(model, val_loader)
        >>>     scheduler.step(val_loss)

    Note:
        Unlike ReduceLROnPlateau which only reacts to metric plateaus, DLRSOnPlateau
        applies the DLRS algorithm's loss-slope-based updates. The 'patience' parameter
        controls how many epochs to wait before considering the metric history.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        mode: str = 'min',
        delta_d: float = 0.5,
        delta_o: float = 1.0,
        delta_i: float = 0.1,
        min_lr: float = 1e-8,
        max_lr: Optional[float] = None,
        patience: int = 0,
        verbose: bool = False
    ):
        if mode not in ('min', 'max'):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        self.optimizer = optimizer
        self.mode = mode
        self.patience = patience
        self.verbose = verbose

        self._scheduler = DLRSScheduler(
            optimizer=optimizer,
            delta_d=delta_d,
            delta_o=delta_o,
            delta_i=delta_i,
            min_lr=min_lr,
            max_lr=max_lr,
            verbose=verbose
        )

        self._metric_history: List[float] = []
        self._wait_count: int = 0

    def step(self, metric: float) -> None:
        """
        Update learning rate based on a single metric value.

        Parameters:
            metric (float): Metric to monitor (e.g., validation loss or accuracy)
        """
        if not isinstance(metric, (int, float)):
            raise TypeError(f"metric must be numeric, got {type(metric)}")

        self._metric_history.append(float(metric))

        if len(self._metric_history) < 2:
            return

        if len(self._metric_history) > self.patience + 1:
            metrics_to_use = self._metric_history[-(self.patience + 2):]
            self._scheduler.step(metrics_to_use)

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing."""
        return {
            'scheduler_state': self._scheduler.state_dict(),
            'mode': self.mode,
            'patience': self.patience,
            'metric_history': self._metric_history,
            'wait_count': self._wait_count
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state from checkpoint."""
        self._scheduler.load_state_dict(state_dict['scheduler_state'])
        self.mode = state_dict['mode']
        self.patience = state_dict['patience']
        self._metric_history = state_dict.get('metric_history', [])
        self._wait_count = state_dict.get('wait_count', 0)

    def get_last_lr(self) -> List[float]:
        """Return last computed learning rate."""
        return self._scheduler.get_last_lr()
