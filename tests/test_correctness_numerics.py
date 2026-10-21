"""Independent exact-input numerical checks, not a global error proof.

Regime and decade decisions use rational values of the represented inputs.
The final rate is allowed ordinary binary64 operation rounding: these bounded
experiments enforce an eight-ULP error budget at the operation's magnitude,
not correctly rounded exact-real subtraction after severe cancellation.
"""

import math
import random
import struct
import sys
import warnings
from fractions import Fraction

import pytest
import torch

from dlrs import DLRSScheduler

TINY = float.fromhex("0x0.0000000000001p-1022")


def _power_ten(exponent):
    return Fraction(10 ** exponent) if exponent >= 0 else Fraction(1, 10 ** -exponent)


def _decade(rate):
    """Integer binary search over exact powers; no log10 or Decimal.adjusted."""
    value = Fraction(rate)
    low, high = -324, 309
    while high - low > 1:
        middle = (low + high) // 2
        if _power_ten(middle) <= value:
            low = middle
        else:
            high = middle
    assert _power_ten(low) <= value < _power_ten(low + 1)
    return low


def _oracle(rate, losses, *, delta_d=0.5, delta_o=1.0, delta_i=0.1,
            minimum=TINY, maximum=None):
    values = [Fraction(value) for value in losses]
    slope = (values[-1] - values[0]) * len(values) / sum(values)
    factor = delta_d if slope > 1 else delta_o if slope >= 0 else delta_i
    adjustment = _power_ten(_decade(rate)) * Fraction(factor) * slope if rate else Fraction()
    proposed = Fraction(rate) - adjustment
    result = max(proposed, Fraction(minimum)) if rate else Fraction()
    if maximum is not None and rate:
        result = min(result, Fraction(maximum))
    scale = max(abs(Fraction(rate)), abs(adjustment), abs(proposed), abs(result))
    return {"rate": result, "slope": slope, "adjustment": adjustment, "scale": scale}


def _ulp(value):
    """Binary64 spacing, including subnormals; compatible with Python 3.8."""
    value = abs(value)
    if value < sys.float_info.min:
        return TINY
    return math.ldexp(1.0, math.frexp(value)[1] - 53)


def _neighbor(value, direction):
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    return struct.unpack(">d", struct.pack(">Q", bits + direction))[0]


def _scheduler(rates, **kwargs):
    groups = [{"params": [torch.nn.Parameter(torch.zeros(()))], "lr": rate} for rate in rates]
    optimizer = torch.optim.SGD(groups)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scheduler = DLRSScheduler(optimizer, min_lr=kwargs.pop("min_lr", TINY), **kwargs)
    return optimizer, scheduler


def _assert_forward_error(actual, oracle, context):
    scale = float(oracle["scale"])
    error = abs(Fraction(actual) - oracle["rate"])
    allowance = 8 * Fraction(_ulp(scale))
    assert math.isfinite(actual) and error <= allowance, (context, actual, float(oracle["rate"]), error, allowance)


def test_distinct_parameter_group_decades_match_fraction_oracle():
    rates = [0.3, 0.03, 0.003, 1e-6, 1e-30, 1e30, 0.0]
    losses = [5.0, 2.0, 1.0]
    optimizer, scheduler = _scheduler(rates)
    scheduler.step(losses)
    for initial, group in zip(rates, optimizer.param_groups):
        _assert_forward_error(group["lr"], _oracle(initial, losses), initial)


@pytest.mark.parametrize("losses", [[2.0, 1.0], [2.0, 3.0]])
def test_all_finite_decimal_boundary_neighbors_use_exact_input_decade(losses):
    # Every rounded finite 10**k and its adjacent positive binary64 values.
    # The rational exponent can differ from k even when float prints 1e-k.
    rates = sorted({neighbor for exponent in range(-323, 309)
                    for neighbor in (_neighbor(float(_power_ten(exponent)), -1),
                                     float(_power_ten(exponent)),
                                     _neighbor(float(_power_ten(exponent)), 1))
                    if neighbor > 0 and math.isfinite(neighbor)})
    optimizer, scheduler = _scheduler(rates)
    scheduler.step(losses)
    for initial, group in zip(rates, optimizer.param_groups):
        _assert_forward_error(group["lr"], _oracle(initial, losses), initial.hex())


def test_smallest_decades_have_an_observable_independent_update():
    # A larger negative slope makes a wrong decade observable even at one ULP
    # of the subnormal grid; the losses still have an exact finite mean.
    losses = [1.0] + [0.0] * 31
    rates = [count * TINY for count in range(1, 65)]
    optimizer, scheduler = _scheduler(rates)
    scheduler.step(losses)
    for initial, group in zip(rates, optimizer.param_groups):
        assert group["lr"] == float(_oracle(initial, losses)["rate"]), initial.hex()


def test_normal_subnormal_transition_and_largest_rates_match_fraction_oracle():
    rates = [_neighbor(sys.float_info.min, -1), sys.float_info.min,
             _neighbor(sys.float_info.min, 1), _neighbor(sys.float_info.max, -1),
             sys.float_info.max]
    losses = [1.0, 2.0]
    optimizer, scheduler = _scheduler(rates)
    scheduler.step(losses)
    for initial, group in zip(rates, optimizer.param_groups):
        _assert_forward_error(group["lr"], _oracle(initial, losses), initial.hex())


@pytest.mark.parametrize("last", [_neighbor(3.0, -1), 3.0, _neighbor(3.0, 1)])
def test_exact_fraction_selects_regime_on_both_sides_of_one(last):
    losses = [1.0, last]
    optimizer, scheduler = _scheduler([0.75], delta_d=0.25, delta_o=1.5)
    scheduler.step(losses)
    expected = _oracle(0.75, losses, delta_d=0.25, delta_o=1.5)
    _assert_forward_error(optimizer.param_groups[0]["lr"], expected, last.hex())
    assert scheduler.loss_slope == float(expected["slope"])


@pytest.mark.parametrize("losses", [[TINY, 1.0, 2 * TINY], [2 * TINY, 1.0, TINY],
                                   [0.0, 1.0, TINY], [TINY, 1.0, 0.0]])
def test_subnormal_endpoint_normalization_preserves_exact_slope(losses):
    optimizer, scheduler = _scheduler([1.0], delta_o=1e308, delta_i=1e308)
    scheduler.step(losses)
    expected = _oracle(1.0, losses, delta_o=1e308, delta_i=1e308)
    assert scheduler.loss_slope == float(expected["slope"])
    _assert_forward_error(optimizer.param_groups[0]["lr"], expected, losses)


@pytest.mark.parametrize("exponent", [-1074, -1070, -1022, -1000, -500, 0, 500, 1000, 1020])
@pytest.mark.parametrize("pattern", [[1.0, 3.0], [3.0, 1.0], [1.0, 5.0, 2.0], [2.0, 1.0, 2.0]])
def test_power_of_two_loss_scales_match_exact_fraction_observations(exponent, pattern):
    losses = [math.ldexp(value, exponent) for value in pattern]
    optimizer, scheduler = _scheduler([0.75, 1e-100, 1e100])
    scheduler.step(losses)
    expected_slope = _oracle(0.75, losses)["slope"]
    assert scheduler.loss_slope == pytest.approx(float(expected_slope), rel=4e-16, abs=0)
    for initial, group in zip([0.75, 1e-100, 1e100], optimizer.param_groups):
        _assert_forward_error(group["lr"], _oracle(initial, losses), (exponent, pattern))


def test_seeded_wide_exponent_inputs_stay_within_operation_scale_error_budget():
    generator = random.Random(94021)
    for _ in range(400):
        rate = math.ldexp(generator.uniform(0.5, 1.0), generator.randint(-1073, 1000))
        losses = [math.ldexp(generator.uniform(0.5, 1.0), generator.randint(-1073, 1023))
                  for _ in range(generator.randint(2, 12))]
        optimizer, scheduler = _scheduler([rate])
        expected = _oracle(rate, losses)
        if expected["slope"] and float(expected["slope"]) == 0.0:
            # A genuine nonzero slope must not silently select the zero case.
            # Rejection ownership/atomicity is checked in the state suite.
            with pytest.raises(ValueError, match="not representable"):
                scheduler.step(losses)
            continue
        scheduler.step(losses)
        _assert_forward_error(optimizer.param_groups[0]["lr"], expected, (rate, losses))
        assert abs(Fraction(scheduler.loss_slope) - expected["slope"]) <= 4 * Fraction(_ulp(float(expected["slope"])))


def test_binary64_cancellation_is_bounded_but_not_correctly_rounded_exact_real():
    optimizer, scheduler = _scheduler([0.1])
    scheduler.step([1.0, 3.0])
    expected = _oracle(0.1, [1.0, 3.0])
    # 0.1's actual value is above exact 1/10. The implemented binary64
    # granularity rounds to 0.1, so subtraction cancels before min clamping.
    assert expected["rate"] == Fraction(0.1) - Fraction(1, 10) > 0
    assert optimizer.param_groups[0]["lr"] == TINY
    _assert_forward_error(optimizer.param_groups[0]["lr"], expected, "cancellation")


def test_intermediate_product_overflow_can_have_a_finite_update():
    losses = [1.0, 1e300, 2.0]
    optimizer, scheduler = _scheduler([1e308], delta_o=100.0)
    scheduler.step(losses)
    expected = _oracle(1e308, losses, delta_o=100.0)
    assert math.isfinite(scheduler.adjustment)
    assert scheduler.adjustment == pytest.approx(float(expected["adjustment"]), rel=3e-16)
    _assert_forward_error(optimizer.param_groups[0]["lr"], expected, losses)


def _dtype_value(bits, dtype):
    # Independent IEEE layouts, without torch casts/nextafter as the oracle.
    if dtype == torch.float16:
        return struct.unpack(">e", struct.pack(">H", bits))[0]
    if dtype == torch.bfloat16:
        return struct.unpack(">f", struct.pack(">I", bits << 16))[0]
    if dtype == torch.float32:
        return struct.unpack(">f", struct.pack(">I", bits))[0]
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def _inward_tensor_bound(bound, dtype, *, upward):
    greatest = {torch.float16: 0x7bff, torch.bfloat16: 0x7f7f,
                torch.float32: 0x7f7fffff, torch.float64: 0x7fefffffffffffff}[dtype]
    low, high = 0, greatest + 1
    while high - low > 1:
        middle = (low + high) // 2
        if Fraction(_dtype_value(middle, dtype)) <= Fraction(bound):
            low = middle
        else:
            high = middle
    exact = _dtype_value(low, dtype)
    return _dtype_value(low + 1, dtype) if upward and exact < bound else exact


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("endpoint", ["minimum", "maximum"])
def test_tensor_clamped_bounds_match_independent_ieee_grid(dtype, endpoint):
    bound = 0.25001
    value = torch.tensor(0.125 if endpoint == "minimum" else 0.5, dtype=dtype)
    kwargs = {"min_lr": bound} if endpoint == "minimum" else {"max_lr": bound}
    optimizer, scheduler = _scheduler([value], **kwargs)
    scheduler.step([1.0, 1.0])
    assert optimizer.param_groups[0]["lr"] is value
    assert float(value) == _inward_tensor_bound(bound, dtype, upward=endpoint == "minimum")


@pytest.mark.parametrize("dtype,gap", [(torch.float16, 2.0 ** -12),
                                      (torch.bfloat16, 2.0 ** -9),
                                      (torch.float32, 2.0 ** -25)])
def test_no_representable_tensor_value_rejects_all_group_updates(dtype, gap):
    minimum, maximum = 0.25 + gap / 4, 0.25 + 3 * gap / 4
    assert _inward_tensor_bound(minimum, dtype, upward=True) > maximum
    value = torch.tensor(0.5, dtype=dtype)
    optimizer, scheduler = _scheduler([0.5, value], min_lr=minimum, max_lr=maximum)
    epoch = scheduler.last_epoch
    before = scheduler.get_last_lr()
    with pytest.raises(ValueError, match="representable"):
        scheduler.step([1.0, 1.0])
    assert optimizer.param_groups[0]["lr"] == 0.5 and float(value) == 0.5
    assert scheduler.last_epoch == epoch
    assert scheduler.get_last_lr()[0] == before[0]
    torch.testing.assert_close(scheduler.get_last_lr()[1], before[1], rtol=0, atol=0)
