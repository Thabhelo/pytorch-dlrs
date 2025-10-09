"""
Utility functions and helper classes for DLRS scheduler
"""

import warnings
from typing import List, Optional, Callable
import torch
import torch.nn as nn


class DLRSError(Exception):
    """Base exception for DLRS-related errors."""
    pass


class InvalidHyperparameter(DLRSError):
    """Raised when invalid hyperparameter values are provided."""
    pass


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
    if not isinstance(delta_d, (int, float)):
        raise InvalidHyperparameter(f"delta_d must be numeric, got {type(delta_d)}")
    if not isinstance(delta_o, (int, float)):
        raise InvalidHyperparameter(f"delta_o must be numeric, got {type(delta_o)}")
    if not isinstance(delta_i, (int, float)):
        raise InvalidHyperparameter(f"delta_i must be numeric, got {type(delta_i)}")

    if delta_d <= 0:
        raise InvalidHyperparameter(f"delta_d must be positive, got {delta_d}")
    if delta_o <= 0:
        raise InvalidHyperparameter(f"delta_o must be positive, got {delta_o}")
    if delta_i <= 0:
        raise InvalidHyperparameter(f"delta_i must be positive, got {delta_i}")

    if delta_d < 0.1 or delta_d > 1.0:
        warnings.warn(f"delta_d={delta_d} is outside typical range [0.1, 1.0]")
    if delta_o < 0.5 or delta_o > 2.0:
        warnings.warn(f"delta_o={delta_o} is outside typical range [0.5, 2.0]")
    if delta_i < 0.01 or delta_i > 0.5:
        warnings.warn(f"delta_i={delta_i} is outside typical range [0.01, 0.5]")


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
        float: Normalized loss slope, or None if mean_loss is near zero

    Example:
        >>> calculate_normalized_loss_slope(1.0, 0.5, 0.75)
        -0.6666666666666666
        >>> calculate_normalized_loss_slope(0.5, 1.0, 0.75)
        0.6666666666666666
        >>> calculate_normalized_loss_slope(1.0, 0.5, 0.0)  # Returns None
    """
    if abs(mean_loss) < 1e-10:
        warnings.warn("Mean loss is near zero, cannot compute normalized slope")
        return None

    return (last_loss - first_loss) / mean_loss


def collect_batch_losses(
    dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    loss_fn: Callable,
    device: str = 'cpu'
) -> List[float]:
    """
    Collect loss values for each batch in a dataloader.

    Useful for validation and testing. Does not update model parameters.

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
    model.eval()
    batch_losses = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            batch_losses.append(loss.item())

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

    def __init__(self):
        self.losses: List[float] = []

    def add(self, loss: float) -> None:
        """
        Add a batch loss value.

        Parameters:
            loss (float): Loss value to record
        """
        if not isinstance(loss, (int, float)):
            raise ValueError(f"Loss must be numeric, got {type(loss)}")
        self.losses.append(float(loss))

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
        return sum(self.losses) / len(self.losses)

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
        mean = self.get_mean()
        variance = sum((x - mean) ** 2 for x in self.losses) / len(self.losses)
        return variance ** 0.5

    def get_loss_slope(self) -> Optional[float]:
        """
        Calculate normalized loss slope from recorded losses.

        Returns:
            float: Normalized slope (last - first) / mean, or None if insufficient data
        """
        if len(self.losses) < 2:
            warnings.warn("Need at least 2 losses to compute slope")
            return None

        first_loss = self.losses[0]
        last_loss = self.losses[-1]
        mean_loss = self.get_mean()

        return calculate_normalized_loss_slope(first_loss, last_loss, mean_loss)

    def __len__(self) -> int:
        """Return number of recorded losses."""
        return len(self.losses)

    def __repr__(self) -> str:
        """String representation of recorder state."""
        if not self.losses:
            return "LossRecorder(empty)"
        return f"LossRecorder(n={len(self.losses)}, mean={self.get_mean():.4f})"
