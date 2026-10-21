"""
Utility functions and helper classes for DLRS scheduler
"""

import math
import numbers
import warnings
from typing import Callable, List, Optional

import torch
import torch.nn as nn


class DLRSError(Exception):
    """Base exception for DLRS-related errors."""
    pass


class InvalidHyperparameter(DLRSError):
    """Raised when invalid hyperparameter values are provided."""
    pass


def _validate_scalar(value, name: str, nonnegative: bool = False) -> float:
    """Return a finite real scalar without retaining a tensor's autograd graph."""
    if isinstance(value, torch.Tensor):
        if value.ndim != 0 or value.dtype == torch.bool or value.is_complex():
            raise ValueError(f"{name} must be a finite real scalar")
        try:
            value = value.detach().item()
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(f"{name} must have an accessible real scalar value") from exc
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real scalar")
    # Preserve the input's sign before binary64 conversion: a negative Fraction
    # or wider real can underflow to -0.0 and otherwise pass the check below.
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _stable_mean(values: List[float]) -> float:
    """Mean of validated nonnegative floats, avoiding an overflowing sum."""
    scale = max(values)
    if scale == 0:
        return 0.0
    return scale * (math.fsum(value / scale for value in values) / len(values))


def validate_hyperparameters(delta_d: float, delta_o: float, delta_i: float) -> None:
    """
    Validate DLRS hyperparameter values.

    Parameters:
        delta_d (float): Decremental factor for divergence
        delta_o (float): Stagnation factor
        delta_i (float): Incremental factor for convergence

    Raises:
        InvalidHyperparameter: If any hyperparameter is invalid

    Example:
        >>> validate_hyperparameters(0.5, 1.0, 0.1)  # Valid
        >>> validate_hyperparameters(-0.5, 1.0, 0.1)  # Raises InvalidHyperparameter
    """
    try:
        delta_d = _validate_scalar(delta_d, "delta_d")
        delta_o = _validate_scalar(delta_o, "delta_o")
        delta_i = _validate_scalar(delta_i, "delta_i")
    except ValueError as exc:
        raise InvalidHyperparameter(str(exc)) from exc

    if delta_d <= 0:
        raise InvalidHyperparameter(f"delta_d must be positive, got {delta_d}")
    if delta_o <= 0:
        raise InvalidHyperparameter(f"delta_o must be positive, got {delta_o}")
    if delta_i <= 0:
        raise InvalidHyperparameter(f"delta_i must be positive, got {delta_i}")

    if delta_d < 0.1 or delta_d > 1.0:
        warnings.warn(f"delta_d={delta_d} is outside typical range [0.1, 1.0]", stacklevel=2)
    if delta_o < 0.5 or delta_o > 2.0:
        warnings.warn(f"delta_o={delta_o} is outside typical range [0.5, 2.0]", stacklevel=2)
    if delta_i < 0.01 or delta_i > 0.5:
        warnings.warn(f"delta_i={delta_i} is outside typical range [0.01, 0.5]", stacklevel=2)


def calculate_normalized_loss_slope(
    first_loss: float,
    last_loss: float,
    mean_loss: float
) -> Optional[float]:
    """
    Calculate normalized loss slope with safe division.

    Computes: ΔL = (last_loss - first_loss) / mean_loss

    Parameters:
        first_loss (float): Loss from first batch in epoch
        last_loss (float): Loss from last batch in epoch
        mean_loss (float): Mean loss across all batches

    Returns:
        float: Normalized loss slope, or None if mean_loss is zero.
            Inputs must be finite, nonnegative real scalars. Positive losses
            are not discarded based on their absolute scale.

    Example:
        >>> calculate_normalized_loss_slope(1.0, 0.5, 0.75)
        -0.6666666666666666
        >>> calculate_normalized_loss_slope(0.5, 1.0, 0.75)
        0.6666666666666666
        >>> calculate_normalized_loss_slope(1.0, 0.5, 0.0)  # Returns None
    """
    first_loss = _validate_scalar(first_loss, "first_loss", nonnegative=True)
    last_loss = _validate_scalar(last_loss, "last_loss", nonnegative=True)
    mean_loss = _validate_scalar(mean_loss, "mean_loss", nonnegative=True)
    if mean_loss == 0:
        warnings.warn("Mean loss is zero, cannot compute normalized slope", stacklevel=2)
        return None
    slope = (last_loss - first_loss) / mean_loss
    if not math.isfinite(slope):
        raise ValueError("Normalized loss slope is not representable as a finite float")
    return slope


def collect_batch_losses(
    dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    loss_fn: Callable,
    device: str = 'cpu'
) -> List[float]:
    """
    Collect loss values for each batch in a dataloader.

    Useful for validation and testing. Does not update model parameters.
    Restores every module's original training flag, even if evaluation fails.
    The loss function must return a finite, nonnegative scalar.

    Parameters:
        dataloader (DataLoader): DataLoader to iterate over
        model (nn.Module): Model to evaluate
        loss_fn (Callable): Loss function
        device (str): Device to run on ('cpu' or 'cuda')

    Returns:
        List[float]: Loss value for each batch

    Example:
        >>> losses = collect_batch_losses(val_loader, model, nn.CrossEntropyLoss())
        >>> print(f"Average validation loss: {sum(losses) / len(losses):.4f}")
    """
    training_modes = [(module, module.training) for module in model.modules()]
    batch_losses = []
    try:
        model.eval()
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                batch_losses.append(_validate_scalar(loss, "loss", nonnegative=True))
    finally:
        # Calling model.train(...) here would erase intentionally mixed modes,
        # for example a frozen BatchNorm inside an otherwise training model.
        for module, training in training_modes:
            module.training = training

    return batch_losses


class LossRecorder:
    """
    Helper class to record and analyze batch losses during training.

    Example:
        >>> recorder = LossRecorder()
        >>> for batch in dataloader:
        >>>     loss = train_step(batch)
        >>>     recorder.add(loss.item())
        >>> slope = recorder.get_loss_slope()
        >>> recorder.reset()
    """

    def __init__(self, max_losses: Optional[int] = None):
        """Optionally cap retained losses; exceeding the cap raises before append.

        The default retains all losses until reset. A cap never silently drops
        the first batch, which would change the scheduler's endpoint statistic.
        """
        if max_losses is not None and (
            isinstance(max_losses, bool) or not isinstance(max_losses, int) or max_losses <= 0
        ):
            raise ValueError("max_losses must be a positive integer or None")
        self.max_losses = max_losses
        self.losses: List[float] = []

    def add(self, loss: float) -> None:
        """
        Add a batch loss value.

        Parameters:
            loss (float): Loss value to record
        """
        value = _validate_scalar(loss, "loss", nonnegative=True)
        if self.max_losses is not None and len(self.losses) >= self.max_losses:
            raise ValueError("max_losses reached; reset the recorder before recording another epoch")
        self.losses.append(value)

    def reset(self) -> None:
        """Clear all recorded losses."""
        self.losses = []

    def get_losses(self) -> List[float]:
        """Return list of all recorded losses."""
        return self.losses.copy()

    def get_mean(self) -> Optional[float]:
        """Return mean of recorded losses."""
        if not self.losses:
            return None
        return _stable_mean(self.losses)

    def get_min(self) -> Optional[float]:
        """Return minimum recorded loss."""
        if not self.losses:
            return None
        return min(self.losses)

    def get_max(self) -> Optional[float]:
        """Return maximum recorded loss."""
        if not self.losses:
            return None
        return max(self.losses)

    def get_std(self) -> Optional[float]:
        """Return standard deviation of recorded losses."""
        if len(self.losses) < 2:
            return None
        scale = max(self.losses)
        if scale == 0:
            return 0.0
        mean = math.fsum(x / scale for x in self.losses) / len(self.losses)
        variance = math.fsum((x / scale - mean) ** 2 for x in self.losses) / len(self.losses)
        return scale * math.sqrt(variance)

    def get_loss_slope(self) -> Optional[float]:
        """
        Calculate normalized loss slope from recorded losses.

        Returns:
            float: Normalized slope (last - first) / mean, or None if insufficient data
        """
        if len(self.losses) < 2:
            warnings.warn("Need at least 2 losses to compute slope", stacklevel=2)
            return None

        # Normalize before averaging so even subnormal positive losses keep
        # their relative scale instead of rounding their mean to zero.
        scale = max(self.losses)
        if scale == 0:
            return calculate_normalized_loss_slope(0.0, 0.0, 0.0)
        mean = math.fsum(value / scale for value in self.losses) / len(self.losses)
        return calculate_normalized_loss_slope(
            self.losses[0] / scale, self.losses[-1] / scale, mean
        )

    def __len__(self) -> int:
        """Return number of recorded losses."""
        return len(self.losses)

    def __repr__(self) -> str:
        """String representation of recorder state."""
        if not self.losses:
            return "LossRecorder(empty)"
        return f"LossRecorder(n={len(self.losses)}, mean={self.get_mean():.4f})"
