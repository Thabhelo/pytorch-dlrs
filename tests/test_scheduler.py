"""
Unit tests for DLRSScheduler
"""

import pytest
import torch
import torch.nn as nn
import warnings
from dlrs.scheduler import DLRSScheduler


@pytest.fixture
def simple_optimizer():
    """Create a simple optimizer for testing."""
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return optimizer


@pytest.fixture
def convergent_losses():
    """Loss sequence showing convergence (steadily decreasing)."""
    return [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


@pytest.fixture
def divergent_losses():
    """Loss sequence showing divergence (steadily increasing)."""
    return [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


@pytest.fixture
def stagnant_losses():
    """Loss sequence showing stagnation (small changes)."""
    return [1.0, 1.01, 0.99, 1.02, 0.98, 1.0]


@pytest.fixture
def identical_losses():
    """Loss sequence with no change."""
    return [1.0, 1.0, 1.0, 1.0, 1.0]


class TestSchedulerInitialization:
    """Test DLRSScheduler initialization."""

    def test_default_hyperparameters(self, simple_optimizer):
        """Test scheduler creation with default hyperparameters."""
        scheduler = DLRSScheduler(simple_optimizer)
        assert scheduler.delta_d == 0.5
        assert scheduler.delta_o == 1.0
        assert scheduler.delta_i == 0.1
        assert scheduler.min_lr == 1e-8

    def test_custom_hyperparameters(self, simple_optimizer):
        """Test scheduler creation with custom hyperparameters."""
        scheduler = DLRSScheduler(
            simple_optimizer,
            delta_d=0.7,
            delta_o=1.5,
            delta_i=0.2,
            min_lr=1e-6
        )
        assert scheduler.delta_d == 0.7
        assert scheduler.delta_o == 1.5
        assert scheduler.delta_i == 0.2
        assert scheduler.min_lr == 1e-6

    def test_invalid_delta_d(self, simple_optimizer):
        """Test that invalid delta_d raises ValueError."""
        with pytest.raises(ValueError):
            DLRSScheduler(simple_optimizer, delta_d=-0.5)
        with pytest.raises(ValueError):
            DLRSScheduler(simple_optimizer, delta_d=0)

    def test_invalid_delta_o(self, simple_optimizer):
        """Test that invalid delta_o raises ValueError."""
        with pytest.raises(ValueError):
            DLRSScheduler(simple_optimizer, delta_o=-1.0)

    def test_invalid_delta_i(self, simple_optimizer):
        """Test that invalid delta_i raises ValueError."""
        with pytest.raises(ValueError):
            DLRSScheduler(simple_optimizer, delta_i=0)

    def test_optimizer_compatibility_sgd(self):
        """Test compatibility with SGD optimizer."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = DLRSScheduler(optimizer)
        assert scheduler.optimizer is optimizer

    def test_optimizer_compatibility_adam(self):
        """Test compatibility with Adam optimizer."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = DLRSScheduler(optimizer)
        assert scheduler.optimizer is optimizer


class TestLearningRateUpdateLogic:
    """Test learning rate update logic based on loss dynamics."""

    def test_lr_increases_on_convergence(self, simple_optimizer, convergent_losses):
        """Test that LR increases when loss decreases (convergence)."""
        scheduler = DLRSScheduler(simple_optimizer, delta_i=0.1)
        initial_lr = simple_optimizer.param_groups[0]['lr']

        scheduler.step(convergent_losses)
        new_lr = simple_optimizer.param_groups[0]['lr']

        assert new_lr > initial_lr, "LR should increase on convergence"

    def test_lr_decreases_on_divergence(self, simple_optimizer, divergent_losses):
        """Test that LR decreases when loss increases significantly (divergence)."""
        scheduler = DLRSScheduler(simple_optimizer, delta_d=0.5)
        initial_lr = simple_optimizer.param_groups[0]['lr']

        scheduler.step(divergent_losses)
        new_lr = simple_optimizer.param_groups[0]['lr']

        assert new_lr < initial_lr, "LR should decrease on divergence"

    def test_lr_adjusts_on_stagnation(self, simple_optimizer, stagnant_losses):
        """Test that LR adjusts minimally on stagnation."""
        scheduler = DLRSScheduler(simple_optimizer, delta_o=1.0)
        initial_lr = simple_optimizer.param_groups[0]['lr']

        scheduler.step(stagnant_losses)
        new_lr = simple_optimizer.param_groups[0]['lr']

        first_loss = stagnant_losses[0]
        last_loss = stagnant_losses[-1]
        if first_loss == last_loss:
            assert new_lr == initial_lr, "LR should not change when first and last loss are identical"
        else:
            assert new_lr != initial_lr, "LR should adjust on stagnation"

    def test_lr_formula_convergence(self, simple_optimizer):
        """Test the mathematical formula for convergence case."""
        losses = [1.0, 0.5]
        scheduler = DLRSScheduler(simple_optimizer, delta_i=0.1, min_lr=1e-10)
        initial_lr = 0.1

        scheduler.step(losses)

        first_loss, last_loss = losses[0], losses[-1]
        mean_loss = (first_loss + last_loss) / 2
        slope = (last_loss - first_loss) / mean_loss

        assert slope < 0, "Slope should be negative for convergence"

        expected_n = -1
        expected_adjustment = (10 ** expected_n) * 0.1 * slope
        expected_lr = initial_lr - expected_adjustment

        actual_lr = simple_optimizer.param_groups[0]['lr']
        assert abs(actual_lr - expected_lr) < 1e-9

    def test_lr_formula_divergence(self, simple_optimizer):
        """Test the mathematical formula for divergence case."""
        losses = [0.5, 1.6]
        scheduler = DLRSScheduler(simple_optimizer, delta_d=0.5, min_lr=1e-10)
        initial_lr = 0.1

        scheduler.step(losses)

        first_loss, last_loss = losses[0], losses[-1]
        mean_loss = (first_loss + last_loss) / 2
        slope = (last_loss - first_loss) / mean_loss

        assert slope > 1.0, "Slope should be > 1 for divergence"

        expected_n = -1
        expected_adjustment = (10 ** expected_n) * 0.5 * slope
        expected_lr = initial_lr - expected_adjustment

        actual_lr = simple_optimizer.param_groups[0]['lr']
        assert abs(actual_lr - expected_lr) < 1e-9


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_mean_loss(self, simple_optimizer):
        """Test with mean loss near zero."""
        losses = [0.0, 0.0, 0.0]
        scheduler = DLRSScheduler(simple_optimizer)
        initial_lr = simple_optimizer.param_groups[0]['lr']

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scheduler.step(losses)
            assert len(w) > 0

        new_lr = simple_optimizer.param_groups[0]['lr']
        assert new_lr == initial_lr, "LR should not change with zero mean loss"

    def test_very_small_lr(self, simple_optimizer):
        """Test with very small learning rate."""
        simple_optimizer.param_groups[0]['lr'] = 1e-7
        losses = [1.0, 0.5]
        scheduler = DLRSScheduler(simple_optimizer, min_lr=1e-10)

        scheduler.step(losses)
        new_lr = simple_optimizer.param_groups[0]['lr']

        assert new_lr > 0, "LR should remain positive"

    def test_very_large_lr(self, simple_optimizer):
        """Test with very large learning rate."""
        simple_optimizer.param_groups[0]['lr'] = 100.0
        losses = [0.5, 1.5]
        scheduler = DLRSScheduler(simple_optimizer)

        scheduler.step(losses)
        new_lr = simple_optimizer.param_groups[0]['lr']

        assert new_lr > 0, "LR should remain positive"
        assert new_lr < 100.0, "LR should decrease on divergence"

    def test_identical_losses(self, simple_optimizer, identical_losses):
        """Test with identical batch losses (zero slope)."""
        scheduler = DLRSScheduler(simple_optimizer)
        initial_lr = simple_optimizer.param_groups[0]['lr']

        scheduler.step(identical_losses)
        new_lr = simple_optimizer.param_groups[0]['lr']

        assert new_lr == initial_lr, "LR should not change with zero slope"

    def test_single_loss(self, simple_optimizer):
        """Test with only one loss value."""
        losses = [1.0]
        scheduler = DLRSScheduler(simple_optimizer)
        initial_lr = simple_optimizer.param_groups[0]['lr']

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scheduler.step(losses)
            assert len(w) > 0

        new_lr = simple_optimizer.param_groups[0]['lr']
        assert new_lr == initial_lr

    def test_min_lr_bound(self, simple_optimizer):
        """Test that LR respects minimum bound."""
        losses = [1.0, 0.1]
        scheduler = DLRSScheduler(simple_optimizer, min_lr=0.05)

        for _ in range(10):
            scheduler.step(losses)

        final_lr = simple_optimizer.param_groups[0]['lr']
        assert final_lr >= 0.05, "LR should not go below min_lr"


class TestMultiStepScenarios:
    """Test multi-epoch scenarios."""

    def test_convergence_trajectory(self, simple_optimizer):
        """Test LR trajectory over multiple convergent epochs."""
        scheduler = DLRSScheduler(simple_optimizer, delta_i=0.1)
        lr_history = [simple_optimizer.param_groups[0]['lr']]

        for _ in range(5):
            losses = [1.0, 0.8, 0.6, 0.4]
            scheduler.step(losses)
            lr_history.append(simple_optimizer.param_groups[0]['lr'])

        for i in range(len(lr_history) - 1):
            assert lr_history[i + 1] >= lr_history[i], "LR should increase or stay same during convergence"

    def test_divergence_trajectory(self, simple_optimizer):
        """Test LR trajectory over multiple divergent epochs."""
        scheduler = DLRSScheduler(simple_optimizer, delta_d=0.5)
        lr_history = [simple_optimizer.param_groups[0]['lr']]

        for _ in range(5):
            losses = [0.4, 0.6, 0.8, 1.0]
            scheduler.step(losses)
            lr_history.append(simple_optimizer.param_groups[0]['lr'])

        for i in range(len(lr_history) - 1):
            assert lr_history[i + 1] <= lr_history[i], "LR should decrease during divergence"

    def test_mixed_scenario(self, simple_optimizer):
        """Test alternating convergence and divergence."""
        scheduler = DLRSScheduler(simple_optimizer)

        convergent = [1.0, 0.5]
        divergent = [0.5, 1.0]

        scheduler.step(convergent)
        lr_after_convergence = simple_optimizer.param_groups[0]['lr']

        scheduler.step(divergent)
        lr_after_divergence = simple_optimizer.param_groups[0]['lr']

        assert lr_after_convergence > 0.1
        assert lr_after_divergence < lr_after_convergence


class TestIntegration:
    """Integration tests with actual neural networks."""

    def test_with_simple_network(self):
        """Test with a simple neural network."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = DLRSScheduler(optimizer)

        losses = [0.5, 0.4, 0.3]
        scheduler.step(losses)

        assert optimizer.param_groups[0]['lr'] > 0

    def test_get_last_lr(self, simple_optimizer):
        """Test get_last_lr() method."""
        scheduler = DLRSScheduler(simple_optimizer)
        losses = [1.0, 0.5]

        scheduler.step(losses)
        last_lr = scheduler.get_last_lr()

        assert isinstance(last_lr, list)
        assert len(last_lr) == 1
        assert last_lr[0] == simple_optimizer.param_groups[0]['lr']

    def test_state_dict(self, simple_optimizer):
        """Test saving and loading scheduler state."""
        scheduler = DLRSScheduler(simple_optimizer, delta_d=0.7)
        losses = [1.0, 0.5]
        scheduler.step(losses)

        state = scheduler.state_dict()

        assert 'delta_d' in state
        assert state['delta_d'] == 0.7
        assert 'last_epoch' in state

    def test_load_state_dict(self, simple_optimizer):
        """Test loading scheduler state."""
        scheduler1 = DLRSScheduler(simple_optimizer, delta_d=0.7)
        losses = [1.0, 0.5]
        scheduler1.step(losses)

        state = scheduler1.state_dict()

        model2 = nn.Linear(10, 1)
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1)
        scheduler2 = DLRSScheduler(optimizer2)
        scheduler2.load_state_dict(state)

        assert scheduler2.delta_d == 0.7
        assert scheduler2.last_epoch == scheduler1.last_epoch

    def test_no_losses_step(self, simple_optimizer):
        """Test step without providing losses."""
        scheduler = DLRSScheduler(simple_optimizer)
        initial_lr = simple_optimizer.param_groups[0]['lr']

        scheduler.step(None)

        new_lr = simple_optimizer.param_groups[0]['lr']
        assert new_lr == initial_lr
