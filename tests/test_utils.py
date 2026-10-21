"""
Unit tests for utility functions and LossRecorder
"""

import warnings

import pytest
import torch
import torch.nn as nn

from dlrs.utils import (
    InvalidHyperparameter,
    LossRecorder,
    calculate_normalized_loss_slope,
    collect_batch_losses,
    validate_hyperparameters,
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
        """Evaluation must not leave a training model in eval mode."""
        model = nn.Sequential(nn.Linear(10, 2), nn.Dropout(0.5))
        loss_fn = nn.CrossEntropyLoss()

        dataset = torch.utils.data.TensorDataset(
            torch.randn(50, 10),
            torch.randint(0, 2, (50,))
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)

        model.train()
        collect_batch_losses(dataloader, model, loss_fn)

        assert model.training is True


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


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), True, 1j])
@pytest.mark.parametrize("position", range(3))
def test_hyperparameters_reject_nonfinite_and_nonreal_values(bad, position):
    values = [0.5, 1.0, 0.1]
    values[position] = bad
    with pytest.raises(InvalidHyperparameter):
        validate_hyperparameters(*values)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, True, 1j, torch.ones(1)])
def test_recorder_rejects_invalid_loss_without_changing_history(bad):
    recorder = LossRecorder()
    recorder.add(1.0)
    with pytest.raises(ValueError):
        recorder.add(bad)
    assert recorder.get_losses() == [1.0]


@pytest.mark.parametrize("scale", [1.0, 1e-300, 1e300])
def test_normalized_slope_is_invariant_to_positive_scale(scale):
    assert calculate_normalized_loss_slope(scale, 0.5 * scale, 0.75 * scale) == pytest.approx(-2 / 3)


@pytest.mark.parametrize("position", range(3))
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, True])
def test_slope_rejects_invalid_inputs(position, bad):
    values = [1.0, 0.5, 0.75]
    values[position] = bad
    with pytest.raises(ValueError):
        calculate_normalized_loss_slope(*values)


def test_recorder_large_losses_have_finite_mean_and_std():
    recorder = LossRecorder()
    recorder.add(1e308)
    recorder.add(0)
    recorder.add(1e308)
    assert recorder.get_mean() == pytest.approx(1e308 * (2 / 3))
    assert recorder.get_std() == pytest.approx(1e308 * (2 / 9) ** 0.5)


def test_recorder_subnormal_losses_keep_slope_and_tensor_is_detached():
    recorder = LossRecorder()
    recorder.add(torch.tensor(5e-324, dtype=torch.float64, requires_grad=True))
    recorder.add(0.0)
    assert recorder.get_losses() == [5e-324, 0.0]
    assert recorder.get_loss_slope() == -2.0


def test_recorder_explicit_limit_never_drops_first_loss():
    recorder = LossRecorder(max_losses=2)
    recorder.add(2.0)
    recorder.add(1.0)
    with pytest.raises(ValueError, match="max_losses"):
        recorder.add(0.0)
    assert recorder.get_losses() == [2.0, 1.0]
    recorder.reset()
    recorder.add(4.0)
    assert recorder.get_losses() == [4.0]


@pytest.mark.parametrize("limit", [0, -1, True, 2.5])
def test_recorder_rejects_invalid_limit(limit):
    with pytest.raises(ValueError, match="max_losses"):
        LossRecorder(max_losses=limit)


@pytest.mark.parametrize("failure", [False, True])
def test_collect_restores_mixed_module_modes_and_preserves_gradients(failure):
    model = nn.Sequential(nn.Linear(2, 2), nn.Dropout(), nn.BatchNorm1d(2))
    model.train()
    model[2].eval()
    original_modes = [module.training for module in model.modules()]
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    original_grads = [parameter.grad.clone() for parameter in model.parameters()]
    original_buffers = {name: buffer.clone() for name, buffer in model.named_buffers()}
    loader = [(torch.ones(3, 2), torch.zeros(3, dtype=torch.long))]

    def loss_fn(outputs, targets):
        assert not any(module.training for module in model.modules())
        assert not torch.is_grad_enabled()
        if failure:
            raise RuntimeError("evaluation failed")
        return nn.functional.cross_entropy(outputs, targets)

    if failure:
        with pytest.raises(RuntimeError, match="evaluation failed"):
            collect_batch_losses(loader, model, loss_fn)
    else:
        assert len(collect_batch_losses(loader, model, loss_fn)) == 1
    assert [module.training for module in model.modules()] == original_modes
    assert all(torch.equal(p.grad, grad) for p, grad in zip(model.parameters(), original_grads))
    assert all(torch.equal(buffer, original_buffers[name]) for name, buffer in model.named_buffers())


def test_collect_invalid_loss_restores_mode():
    model = nn.Linear(2, 1)
    with pytest.raises(ValueError, match="finite"):
        collect_batch_losses([(torch.ones(1, 2), torch.zeros(1))], model, lambda *_: float("nan"))
    assert model.training


def test_recorder_unavailable_tensor_value_raises_clear_error():
    recorder = LossRecorder()
    with pytest.raises(ValueError, match="accessible"):
        recorder.add(torch.empty((), device="meta"))
    assert len(recorder) == 0


def test_recorder_accepts_numpy_real_scalar():
    import numpy as np

    recorder = LossRecorder()
    recorder.add(np.float32(1.0))
    assert recorder.get_losses() == [1.0]
