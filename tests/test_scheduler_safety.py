"""Regression tests for failure atomicity, numerical edges, and state ownership."""

import copy
import math
from decimal import Decimal, localcontext

import pytest
import torch

from dlrs import DLRSOnPlateau, DLRSScheduler


def optimizer(*rates):
    return torch.optim.SGD([{'params': [torch.nn.Parameter(torch.tensor(1.0))], 'lr': rate}
                            for rate in rates], lr=0.1)


def scalar_rates(opt):
    return [float(group['lr']) for group in opt.param_groups]


def assert_same_state(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        assert actual[key] == expected[key], key


@pytest.mark.parametrize('field', ['delta_d', 'delta_o', 'delta_i', 'min_lr', 'max_lr'])
@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf, True, 0.0, -1.0])
def test_invalid_hyperparameters_rejected_before_optimizer_mutation(field, value):
    opt = optimizer(0.1)
    before = copy.deepcopy(opt.state_dict())
    with pytest.raises(ValueError):
        DLRSScheduler(opt, **{field: value})
    assert opt.state_dict() == before


@pytest.mark.parametrize('bad', [math.nan, math.inf, -math.inf, -1.0, True, 1j,
                                 '1', torch.tensor([1.0]), torch.tensor(True)])
def test_invalid_loss_does_not_change_any_state(bad):
    opt = optimizer(0.1, 0.01)
    scheduler = DLRSScheduler(opt)
    scheduler.step([1.0, 0.9])
    before = scheduler.state_dict()
    with pytest.raises(ValueError):
        scheduler.step([1.0, bad])
    assert_same_state(scheduler.state_dict(), before)


@pytest.mark.parametrize('losses', [[1e-300, 5e-301], [1e308, 5e307],
                                    [float.fromhex("0x0.0000000000001p-1022") * 2, float.fromhex("0x0.0000000000001p-1022")]])
def test_loss_scale_does_not_change_normalized_update(losses):
    opt = optimizer(0.1)
    scheduler = DLRSScheduler(opt)
    scheduler.step(losses)
    assert scheduler.loss_slope == pytest.approx(-2 / 3)
    assert scalar_rates(opt) == pytest.approx([0.1 + 0.01 * 2 / 3])


def test_all_zero_losses_clear_stale_diagnostics_without_changing_rate():
    opt = optimizer(0.1)
    scheduler = DLRSScheduler(opt)
    scheduler.step([1.0, 0.5])
    rates = scalar_rates(opt)
    epoch = scheduler.last_epoch
    with pytest.warns(UserWarning, match='Mean loss is zero'):
        scheduler.step([0.0, 0.0])
    assert scalar_rates(opt) == rates
    assert scheduler.loss_slope == scheduler.adjustment == 0.0
    assert scheduler.last_epoch == epoch + 1


def test_slope_one_uses_stagnation_factor():
    opt = optimizer(0.1)
    scheduler = DLRSScheduler(opt)
    scheduler.step([1.0, 3.0])
    assert scheduler.loss_slope == 1.0
    assert opt.param_groups[0]['lr'] == scheduler.min_lr


def test_below_decade_uses_lower_granularity():
    initial = float.fromhex("0x1.9999999999999p-4")
    opt = optimizer(initial)
    scheduler = DLRSScheduler(opt)
    scheduler.step([1.0, 0.5])
    assert opt.param_groups[0]['lr'] == pytest.approx(initial + 0.001 * 2 / 3, abs=1e-17)


def test_subnormal_granularity_does_not_disappear():
    tiny = float.fromhex("0x0.0000000000001p-1022")
    opt = optimizer(tiny)
    with pytest.warns(UserWarning):
        scheduler = DLRSScheduler(opt, delta_i=10.0, min_lr=tiny)
    scheduler.step([1.0, 0.0])
    assert opt.param_groups[0]['lr'] > tiny
    assert math.isfinite(opt.param_groups[0]['lr'])


def test_zero_lr_groups_remain_frozen():
    opt = optimizer(0.0, 0.1)
    scheduler = DLRSScheduler(opt)
    scheduler.step([1.0, 0.0])
    assert scalar_rates(opt)[0] == 0.0
    assert scalar_rates(opt)[1] > 0.1


def test_invalid_later_group_is_detected_before_earlier_group_changes():
    opt = optimizer(0.1, 0.01)
    scheduler = DLRSScheduler(opt)
    opt.param_groups[1]['lr'] = math.nan
    before = scheduler.get_last_lr()
    epoch = scheduler.last_epoch
    with pytest.raises(ValueError):
        scheduler.step([1.0, 0.5])
    assert opt.param_groups[0]['lr'] == 0.1
    assert scheduler.get_last_lr() == before
    assert scheduler.last_epoch == epoch


def test_overflow_in_later_group_is_atomic():
    opt = optimizer(0.1, 1e308)
    with pytest.warns(UserWarning):
        scheduler = DLRSScheduler(opt, delta_i=10.0)
    before = scheduler.state_dict()
    with pytest.raises(ValueError, match='overflow'):
        scheduler.step([1.0, 0.0])
    assert_same_state(scheduler.state_dict(), before)


@pytest.mark.parametrize('mutation', ['add', 'remove', 'reorder', 'replace_parameter'])
def test_changed_parameter_groups_fail_without_mutation(mutation):
    opt = optimizer(0.1, 0.01)
    scheduler = DLRSScheduler(opt)
    if mutation == 'add':
        opt.add_param_group({'params': [torch.nn.Parameter(torch.tensor(1.0))], 'lr': 0.001})
    elif mutation == 'remove':
        opt.param_groups.pop()
    elif mutation == 'reorder':
        opt.param_groups.reverse()
    else:
        opt.param_groups[0]['params'] = [torch.nn.Parameter(torch.tensor(1.0))]
    before = scalar_rates(opt)
    with pytest.raises(ValueError, match='parameter groups changed'):
        scheduler.step([1.0, 0.5])
    assert scalar_rates(opt) == before


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.float64, torch.bfloat16])
def test_tensor_lr_identity_dtype_and_snapshot_ownership(dtype):
    rate = torch.tensor(0.1, dtype=dtype)
    opt = optimizer(rate)
    scheduler = DLRSScheduler(opt)
    saved = scheduler.state_dict()
    last = scheduler.get_last_lr()
    scheduler.step([1.0, 0.5])
    assert opt.param_groups[0]['lr'] is rate
    assert rate.dtype == dtype
    assert float(rate) > float(saved['param_group_lrs'][0])
    assert last[0] == saved['_last_lr'][0]
    last[0].fill_(7)
    assert float(scheduler.get_last_lr()[0]) < 1
    scheduler.load_state_dict(saved)
    assert opt.param_groups[0]['lr'] is rate
    assert rate == saved['param_group_lrs'][0]
    saved['param_group_lrs'][0].fill_(9)
    assert float(rate) < 1


@pytest.mark.parametrize('explicit_initial_alias', [False, True])
def test_tensor_base_rate_metadata_stays_independent_across_steps_and_resume(explicit_initial_alias):
    rate = torch.tensor(0.1, dtype=torch.float64)
    opt = optimizer(rate)
    if explicit_initial_alias:
        opt.param_groups[0]['initial_lr'] = rate
    scheduler = DLRSScheduler(opt)

    def assert_metadata_independent(current_optimizer, current_scheduler):
        group = current_optimizer.param_groups[0]
        initial, base = group['initial_lr'], current_scheduler.base_lrs[0]
        assert isinstance(initial, torch.Tensor) and isinstance(base, torch.Tensor)
        assert initial.dtype == base.dtype == torch.float64
        assert len({group['lr'].data_ptr(), initial.data_ptr(), base.data_ptr()}) == 3
        assert float(initial) == float(base) == 0.1

    assert_metadata_independent(opt, scheduler)
    scheduler.step([1.0, 0.5])
    assert float(rate) > 0.1
    assert_metadata_independent(opt, scheduler)
    optimizer_state = copy.deepcopy(opt.state_dict())
    scheduler_state = scheduler.state_dict()

    resumed_optimizer = optimizer(torch.tensor(0.25, dtype=torch.float64))
    resumed = DLRSScheduler(resumed_optimizer)
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed.load_state_dict(scheduler_state)
    assert_metadata_independent(resumed_optimizer, resumed)
    for active in (scheduler, resumed):
        active.step([0.5, 0.25])
    assert_metadata_independent(opt, scheduler)
    assert_metadata_independent(resumed_optimizer, resumed)
    torch.testing.assert_close(resumed_optimizer.param_groups[0]['lr'], rate, rtol=0, atol=0)


def test_loading_aliased_optimizer_metadata_detaches_before_restoring_live_rate():
    rate = torch.tensor(0.1, dtype=torch.float64)
    opt = optimizer(rate)
    scheduler = DLRSScheduler(opt)
    scheduler.step([1.0, 0.5])
    checkpoint = scheduler.state_dict()
    # Reproduce the alias layout from older PyTorch optimizer checkpoints.
    saved_optimizer = copy.deepcopy(opt.state_dict())
    saved_optimizer['param_groups'][0]['initial_lr'] = saved_optimizer['param_groups'][0]['lr']
    opt.load_state_dict(saved_optimizer)
    group = opt.param_groups[0]
    assert group['initial_lr'] is group['lr']
    stored_initial = group['initial_lr'].clone()
    checkpoint['param_group_lrs'][0].fill_(0.05)
    scheduler.load_state_dict(checkpoint)
    assert group['initial_lr'].data_ptr() != group['lr'].data_ptr()
    torch.testing.assert_close(group['initial_lr'], stored_initial, rtol=0, atol=0)
    scheduler.step([1.0, 0.5])
    torch.testing.assert_close(group['initial_lr'], stored_initial, rtol=0, atol=0)
    assert float(scheduler.base_lrs[0]) == 0.1


def test_tensor_min_bound_rounds_inward():
    rate = torch.tensor(0.1, dtype=torch.float32)
    opt = optimizer(rate)
    scheduler = DLRSScheduler(opt, min_lr=1e-8)
    scheduler.step([0.0, 1.0])
    assert float(rate) >= 1e-8
    assert float(rate) < 1.000001e-8


def test_tensor_max_bound_rounds_inward():
    rate = torch.tensor(0.1, dtype=torch.float32)
    opt = optimizer(rate)
    scheduler = DLRSScheduler(opt, max_lr=0.11)
    scheduler.step([1.0, 0.0])
    assert float(rate) <= 0.11
    assert float(rate) > 0.109999


def test_tensor_underflow_failure_is_atomic():
    rate = torch.tensor(0.1, dtype=torch.float16)
    opt = optimizer(0.1, rate)
    scheduler = DLRSScheduler(opt, min_lr=1e-12)
    before = scheduler.state_dict()
    with pytest.raises(ValueError, match='representable'):
        scheduler.step([0.0] * 30 + [1.0])
    assert_same_state(scheduler.state_dict(), before)


def test_scalar_tensor_losses_are_copied_without_graphs():
    first = torch.tensor(1.0, requires_grad=True)
    last = torch.tensor(0.5, requires_grad=True)
    losses = [first, last]
    scheduler = DLRSScheduler(optimizer(0.1))
    scheduler.step(losses)
    with torch.no_grad():
        first.fill_(8)
    losses.append(9)
    assert scheduler.batch_losses == [1.0, 0.5]
    assert all(isinstance(value, float) for value in scheduler.batch_losses)


@pytest.mark.parametrize('field,value', [
    ('delta_d', math.nan), ('min_lr', -1), ('max_lr', 1e-10),
    ('last_epoch', True), ('_step_count', -1), ('loss_slope', math.inf),
    ('adjustment', math.nan), ('param_group_lrs', []),
    ('param_group_lrs', [0.1, math.nan]), ('_last_lr', [0.1]),
    ('base_lrs', [0.1]), ('batch_losses', [1.0, -1.0]),
    ('state_version', 99), ('verbose', 1),
])
def test_malformed_checkpoint_is_rejected_atomically(field, value):
    scheduler = DLRSScheduler(optimizer(0.1, 0.01))
    scheduler.step([1.0, 0.5])
    before = scheduler.state_dict()
    bad = copy.deepcopy(before)
    bad[field] = value
    with pytest.raises(ValueError):
        scheduler.load_state_dict(bad)
    assert_same_state(scheduler.state_dict(), before)


def test_checkpoint_and_get_last_lr_are_independent_snapshots():
    scheduler = DLRSScheduler(optimizer(0.1))
    scheduler.step([1.0, 0.5])
    saved = scheduler.state_dict()
    saved['batch_losses'].append(7)
    saved['_last_lr'][0] = 9
    saved['base_lrs'][0] = 9
    saved['param_group_lrs'][0] = 9
    scheduler.get_last_lr()[0] = 9
    assert scheduler.batch_losses == [1.0, 0.5]
    assert scheduler.get_last_lr()[0] < 1
    assert scheduler.base_lrs == [0.1]


def test_optimizer_load_before_scheduler_load_preserves_topology():
    source_opt = optimizer(0.1, 0.01)
    source = DLRSScheduler(source_opt)
    source.step([1.0, 0.5])
    target_opt = optimizer(0.3, 0.03)
    target = DLRSScheduler(target_opt)
    target_opt.load_state_dict(source_opt.state_dict())
    target.load_state_dict(source.state_dict())
    source.step([0.5, 0.3])
    target.step([0.5, 0.3])
    assert_same_state(source.state_dict(), target.state_dict())


def test_legacy_checkpoint_without_new_fields_resumes():
    source = DLRSScheduler(optimizer(0.1))
    source.step([1.0, 0.5])
    saved = source.state_dict()
    for key in ['state_version', 'base_lrs', '_step_count', 'batch_losses', 'verbose']:
        del saved[key]
    target = DLRSScheduler(optimizer(0.01))
    target.load_state_dict(saved)
    source.step([0.5, 0.3])
    target.step([0.5, 0.3])
    assert source.get_last_lr() == target.get_last_lr()
    assert source.last_epoch == target.last_epoch


@pytest.mark.parametrize('mode,metrics', [('min', [1.0, 0.5]), ('max', [0.5, 1.0])])
def test_wrapper_improvement_increases_lr_in_both_modes(mode, metrics):
    opt = optimizer(0.1)
    scheduler = DLRSOnPlateau(opt, mode=mode)
    for metric in metrics:
        scheduler.step(metric)
    assert scalar_rates(opt)[0] == pytest.approx(0.1 + 0.01 * 2 / 3)


@pytest.mark.parametrize('mode,metrics', [('min', [0.5, 1.0]), ('max', [1.0, 0.5])])
def test_wrapper_deterioration_decreases_lr_in_both_modes(mode, metrics):
    opt = optimizer(0.1)
    scheduler = DLRSOnPlateau(opt, mode=mode)
    for metric in metrics:
        scheduler.step(metric)
    assert scalar_rates(opt)[0] == pytest.approx(0.1 - 0.1 * 2 / 3)


@pytest.mark.parametrize('patience', [-1, 1.5, True, '2'])
def test_wrapper_rejects_invalid_patience(patience):
    opt = optimizer(0.1)
    before = opt.state_dict()
    with pytest.raises(ValueError):
        DLRSOnPlateau(opt, patience=patience)
    assert opt.state_dict() == before


def test_wrapper_patience_is_bounded_rolling_window_and_resumes():
    opt = optimizer(0.1)
    wrapper = DLRSOnPlateau(opt, patience=2)
    direct_opt = optimizer(0.1)
    direct = DLRSScheduler(direct_opt)
    metrics = [10.0 / (index + 1) for index in range(100)]
    for index, metric in enumerate(metrics):
        wrapper.step(metric)
        if index >= 3:
            direct.step(metrics[index - 3:index + 1])
        assert wrapper.get_last_lr() == direct.get_last_lr()
        assert len(wrapper._metric_history) <= 4
    saved = wrapper.state_dict()
    resumed = DLRSOnPlateau(optimizer(0.5))
    resumed.load_state_dict(saved)
    saved['metric_history'].append(8)
    wrapper.step(0.08)
    resumed.step(0.08)
    assert resumed.state_dict() == wrapper.state_dict()


@pytest.mark.parametrize('bad', [math.nan, math.inf, -1.0, True, torch.tensor([1.0])])
def test_wrapper_invalid_metric_does_not_poison_history(bad):
    wrapper = DLRSOnPlateau(optimizer(0.1))
    wrapper.step(1.0)
    before = wrapper.state_dict()
    with pytest.raises(ValueError):
        wrapper.step(bad)
    assert wrapper.state_dict() == before


def test_wrapper_invalid_metadata_does_not_load_underlying_rates():
    wrapper = DLRSOnPlateau(optimizer(0.1))
    before = wrapper.state_dict()
    invalid = copy.deepcopy(before)
    invalid['scheduler_state']['param_group_lrs'] = [9.0]
    invalid['mode'] = 'invalid'
    with pytest.raises(ValueError):
        wrapper.load_state_dict(invalid)
    assert wrapper.state_dict() == before


@pytest.mark.parametrize('last,expected_adjustment', [
    (float.fromhex("0x1.7ffffffffffffp+1"), 0.1), (3.0, 0.1),
    (float.fromhex("0x1.8000000000001p+1"), 0.05),
])
def test_exact_slope_branch_at_neighbors_of_one(last, expected_adjustment):
    scheduler = DLRSScheduler(optimizer(0.1))
    scheduler.step([1.0, last])
    assert scheduler.adjustment == pytest.approx(expected_adjustment)


@pytest.mark.parametrize('mode,metrics', [
    ('min', [1.0, float.fromhex("0x1.fffffffffffffp+0"), 3.0]),
    ('max', [3.0, float.fromhex("0x1.fffffffffffffp+0"), 1.0]),
])
def test_rounded_mean_does_not_hide_true_slope_above_one(mode, metrics):
    wrapper = DLRSOnPlateau(optimizer(0.1), mode=mode, patience=1)
    for metric in metrics:
        wrapper.step(metric)
    assert wrapper.get_last_lr() == pytest.approx([0.05])


def test_inference_tensor_rate_rejected_before_other_group_mutation():
    with torch.inference_mode():
        rate = torch.tensor(0.1)
    opt = optimizer(0.1, rate)
    with pytest.raises(ValueError, match='writable'):
        DLRSScheduler(opt)
    assert scalar_rates(opt) == pytest.approx([0.1, 0.1])


def test_shared_tensor_lr_groups_update_together():
    rate = torch.tensor(0.1)
    opt = optimizer(rate, rate)
    scheduler = DLRSScheduler(opt)
    scheduler.step([1.0, 0.5])
    assert opt.param_groups[0]['lr'] is opt.param_groups[1]['lr'] is rate
    assert float(rate) > 0.1


def test_shared_tensor_lr_conflicting_checkpoint_is_atomic():
    rate = torch.tensor(0.1)
    scheduler = DLRSScheduler(optimizer(0.1, rate, rate))
    before = scheduler.state_dict()
    invalid = copy.deepcopy(before)
    invalid['param_group_lrs'] = [0.2, 0.3, 0.4]
    with pytest.raises(ValueError, match='share tensor storage'):
        scheduler.load_state_dict(invalid)
    assert_same_state(scheduler.state_dict(), before)


def test_tensor_overflow_failure_is_atomic():
    rate = torch.tensor(1e38)
    with pytest.warns(UserWarning):
        scheduler = DLRSScheduler(optimizer(0.1, rate), delta_i=20.0)
    before = scheduler.state_dict()
    with pytest.raises(ValueError, match='representable'):
        scheduler.step([1.0, 0.0])
    assert_same_state(scheduler.state_dict(), before)


def test_subnormal_intermediate_product_is_not_rounded_away():
    tiny = float.fromhex("0x0.0000000000001p-1022")
    opt = optimizer(3 * tiny)
    scheduler = DLRSScheduler(opt, delta_d=0.1, min_lr=tiny)
    scheduler.step([0.0, 0.0, 1.0])
    assert opt.param_groups[0]['lr'] == 2 * tiny


def test_versioned_checkpoint_missing_rates_is_rejected():
    scheduler = DLRSScheduler(optimizer(0.1))
    before = scheduler.state_dict()
    incomplete = copy.deepcopy(before)
    del incomplete['param_group_lrs']
    with pytest.raises(ValueError, match='missing'):
        scheduler.load_state_dict(incomplete)
    assert_same_state(scheduler.state_dict(), before)


def test_partially_overlapping_tensor_views_are_rejected_atomically():
    storage = torch.tensor([1.0], dtype=torch.float64)
    wide = storage[0]
    narrow = storage.view(torch.float32)[0]
    scheduler = DLRSScheduler(optimizer(0.1, wide, narrow))
    before = scheduler.state_dict()
    with pytest.raises(ValueError, match='share tensor storage'):
        scheduler.step([1.0, 0.5])
    assert_same_state(scheduler.state_dict(), before)


def test_zero_slope_does_not_overflow_large_valid_factor():
    opt = optimizer(1e308)
    with pytest.warns(UserWarning):
        scheduler = DLRSScheduler(opt, delta_o=1e308)
    scheduler.step([1.0, 1.0])
    assert opt.param_groups[0]['lr'] == 1e308


@pytest.mark.parametrize('factor', [0.01, 0.00075])
def test_subnormal_intermediate_rounding_matches_decimal_oracle(factor):
    initial = 1e-320
    tiny = float.fromhex("0x0.0000000000001p-1022")
    opt = optimizer(initial)
    with pytest.warns(UserWarning):
        scheduler = DLRSScheduler(opt, delta_d=factor, min_lr=tiny)
    scheduler.step([0.0] * 99 + [1.0])
    with localcontext() as context:
        context.prec = 64
        decimal_rate = Decimal.from_float(initial)
        exact = decimal_rate - Decimal(10) ** decimal_rate.adjusted() * Decimal(str(factor)) * 100
        expected = max(float(exact), tiny)
    assert opt.param_groups[0]['lr'] == expected


def test_rounded_subnormal_power_does_not_promote_decade():
    tiny = float.fromhex("0x0.0000000000001p-1022")
    initial = 2 * tiny
    opt = optimizer(initial)
    scheduler = DLRSScheduler(opt, min_lr=tiny)
    scheduler.step([1.0] + [0.0] * 99)
    with localcontext() as context:
        context.prec = 64
        current = Decimal.from_float(initial)
        expected = float(current + Decimal(10) ** current.adjusted() * Decimal('0.1') * 100)
    assert opt.param_groups[0]['lr'] == expected


def test_tensor_checkpoint_preserves_base_rate_metadata_and_ownership():
    source = DLRSScheduler(optimizer(torch.tensor(0.1, dtype=torch.float64), 0.01))
    source.step([1.0, 0.5])
    checkpoint = source.state_dict()
    target = DLRSScheduler(optimizer(torch.tensor(0.2, dtype=torch.float64), 0.02))
    target.load_state_dict(checkpoint)
    restored = target.state_dict()
    assert isinstance(restored['base_lrs'][0], torch.Tensor)
    assert restored['base_lrs'][0].dtype == checkpoint['base_lrs'][0].dtype
    torch.testing.assert_close(restored['base_lrs'][0], checkpoint['base_lrs'][0], rtol=0, atol=0)
    assert isinstance(restored['base_lrs'][1], float)
    checkpoint['base_lrs'][0].fill_(9)
    assert float(target.base_lrs[0]) == 0.1
