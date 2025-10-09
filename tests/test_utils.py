"""
Unit tests for utility functions and LossRecorder
"""

import pytest
import warnings
import torch
import torch.nn as nn
from dlrs.utils import (
    validate_hyperparameters,
    calculate_normalized_loss_slope,
    collect_batch_losses,
    LossRecorder,
    InvalidHyperparameter
)


class TestValidateHyperparameters:
    """Test hyperparameter validation."""

    def test_valid_hyperparameters(self):
        """Test with valid hyperparameter values."""
        validate_hyperparameters(0.5, 1.0, 0.1)

    def test_invalid_type_delta_d(self):
        """Test that non-numeric delta_d raises error."""
        with pytest.raises(InvalidHyperparameter):
            validate_hyperparameters("0.5", 1.0, 0.1)

    def test_negative_delta_d(self):
        """Test that negative delta_d raises error."""
        with pytest.raises(InvalidHyperparameter):
            validate_hyperparameters(-0.5, 1.0, 0.1)

    def test_zero_delta_o(self):
        """Test that zero delta_o raises error."""
        with pytest.raises(InvalidHyperparameter):
            validate_hyperparameters(0.5, 0.0, 0.1)

    def test_warning_out_of_range(self):
        """Test warnings for out-of-range values."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_hyperparameters(5.0, 1.0, 0.1)
            assert len(w) > 0


class TestCalculateNormalizedLossSlope:
    """Test normalized loss slope calculation."""

    def test_decreasing_loss(self):
        """Test with decreasing loss (convergence)."""
        slope = calculate_normalized_loss_slope(1.0, 0.5, 0.75)
        assert slope < 0
        assert abs(slope - (-0.666666)) < 0.01

    def test_increasing_loss(self):
        """Test with increasing loss (divergence)."""
        slope = calculate_normalized_loss_slope(0.5, 1.0, 0.75)
        assert slope > 0
        assert abs(slope - 0.666666) < 0.01

    def test_no_change(self):
        """Test with no loss change."""
        slope = calculate_normalized_loss_slope(1.0, 1.0, 1.0)
        assert slope == 0.0

    def test_zero_mean_loss(self):
        """Test with zero mean loss."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = calculate_normalized_loss_slope(1.0, 0.5, 0.0)
            assert result is None
            assert len(w) > 0


class TestCollectBatchLosses:
    """Test batch loss collection."""

    def test_collect_losses_cpu(self):
        """Test collecting losses on CPU."""
        model = nn.Linear(10, 2)
        loss_fn = nn.CrossEntropyLoss()

        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 10),
            torch.randint(0, 2, (100,))
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)

        losses = collect_batch_losses(dataloader, model, loss_fn, device='cpu')

        assert len(losses) == 10
        assert all(isinstance(loss, float) for loss in losses)

    def test_model_in_eval_mode(self):
        """Test that model is set to eval mode."""
        model = nn.Sequential(nn.Linear(10, 2), nn.Dropout(0.5))
        loss_fn = nn.CrossEntropyLoss()

        dataset = torch.utils.data.TensorDataset(
            torch.randn(50, 10),
            torch.randint(0, 2, (50,))
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)

        model.train()
        losses = collect_batch_losses(dataloader, model, loss_fn)

        assert model.training is False


class TestLossRecorder:
    """Test LossRecorder class."""

    def test_initialization(self):
        """Test recorder initialization."""
        recorder = LossRecorder()
        assert len(recorder) == 0
        assert recorder.get_losses() == []

    def test_add_loss(self):
        """Test adding losses."""
        recorder = LossRecorder()
        recorder.add(1.0)
        recorder.add(0.5)
        assert len(recorder) == 2
        assert recorder.get_losses() == [1.0, 0.5]

    def test_add_invalid_loss(self):
        """Test adding non-numeric loss."""
        recorder = LossRecorder()
        with pytest.raises(ValueError):
            recorder.add("1.0")

    def test_reset(self):
        """Test resetting recorder."""
        recorder = LossRecorder()
        recorder.add(1.0)
        recorder.add(0.5)
        recorder.reset()
        assert len(recorder) == 0

    def test_get_mean(self):
        """Test mean calculation."""
        recorder = LossRecorder()
        recorder.add(1.0)
        recorder.add(2.0)
        recorder.add(3.0)
        assert recorder.get_mean() == 2.0

    def test_get_mean_empty(self):
        """Test mean with no losses."""
        recorder = LossRecorder()
        assert recorder.get_mean() is None

    def test_get_min(self):
        """Test minimum calculation."""
        recorder = LossRecorder()
        recorder.add(1.0)
        recorder.add(0.5)
        recorder.add(2.0)
        assert recorder.get_min() == 0.5

    def test_get_max(self):
        """Test maximum calculation."""
        recorder = LossRecorder()
        recorder.add(1.0)
        recorder.add(0.5)
        recorder.add(2.0)
        assert recorder.get_max() == 2.0

    def test_get_std(self):
        """Test standard deviation calculation."""
        recorder = LossRecorder()
        recorder.add(1.0)
        recorder.add(2.0)
        recorder.add(3.0)
        std = recorder.get_std()
        assert std is not None
        assert std > 0

    def test_get_std_insufficient_data(self):
        """Test std with insufficient data."""
        recorder = LossRecorder()
        recorder.add(1.0)
        assert recorder.get_std() is None

    def test_get_loss_slope(self):
        """Test loss slope calculation."""
        recorder = LossRecorder()
        recorder.add(1.0)
        recorder.add(0.5)
        slope = recorder.get_loss_slope()
        assert slope < 0

    def test_get_loss_slope_insufficient_data(self):
        """Test slope with insufficient data."""
        recorder = LossRecorder()
        recorder.add(1.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            slope = recorder.get_loss_slope()
            assert slope is None
            assert len(w) > 0

    def test_repr(self):
        """Test string representation."""
        recorder = LossRecorder()
        assert "empty" in repr(recorder)

        recorder.add(1.0)
        recorder.add(2.0)
        repr_str = repr(recorder)
        assert "n=2" in repr_str
        assert "mean=" in repr_str
