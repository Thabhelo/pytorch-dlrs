"""Public state-transition and input-domain checks for the correctness contract."""

import copy
import warnings
from decimal import ROUND_UP, Inexact, localcontext
from fractions import Fraction

import pytest
import torch

from dlrs import DLRSOnPlateau, DLRSScheduler
from dlrs.utils import LossRecorder, calculate_normalized_loss_slope

NEGATIVE_UNDERFLOW = Fraction(-1, 10**1000)


def make_scheduler(tensor=False):
    rate = torch.tensor(0.1, dtype=torch.float64) if tensor else 0.1
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=rate)
    return DLRSScheduler(opt), opt


@pytest.mark.parametrize("consumer", ["scheduler", "wrapper", "recorder", "first", "last", "mean"])
def test_negative_exact_input_cannot_hide_behind_float_underflow(consumer):
    assert NEGATIVE_UNDERFLOW < 0 and float(NEGATIVE_UNDERFLOW) == 0.0
    scheduler, opt = make_scheduler()
    wrapper = DLRSOnPlateau(opt)
    recorder = LossRecorder()
    scheduler_before, wrapper_before = scheduler.state_dict(), wrapper.state_dict()
    actions = {
        "scheduler": lambda: scheduler.step([1.0, NEGATIVE_UNDERFLOW]),
        "wrapper": lambda: wrapper.step(NEGATIVE_UNDERFLOW),
        "recorder": lambda: recorder.add(NEGATIVE_UNDERFLOW),
        "first": lambda: calculate_normalized_loss_slope(NEGATIVE_UNDERFLOW, 1.0, 1.0),
        "last": lambda: calculate_normalized_loss_slope(1.0, NEGATIVE_UNDERFLOW, 1.0),
        "mean": lambda: calculate_normalized_loss_slope(1.0, 1.0, NEGATIVE_UNDERFLOW),
    }
    with pytest.raises(ValueError, match="nonnegative"):
        actions[consumer]()
    assert scheduler.state_dict() == scheduler_before
    assert wrapper.state_dict() == wrapper_before
    assert recorder.losses == []


@pytest.mark.parametrize("field", ["param_group_lrs", "_last_lr", "base_lrs", "batch_losses"])
def test_negative_exact_checkpoint_value_is_rejected_atomically(field):
    scheduler, _ = make_scheduler()
    scheduler.step([1.0, 0.5])
    before = scheduler.state_dict()
    invalid = copy.deepcopy(before)
    invalid[field] = [NEGATIVE_UNDERFLOW]
    with pytest.raises(ValueError, match="nonnegative"):
        scheduler.load_state_dict(invalid)
    assert scheduler.state_dict() == before


def test_negative_exact_external_rate_is_rejected_before_state_update():
    scheduler, opt = make_scheduler()
    before = scheduler.state_dict()
    opt.param_groups[0]["lr"] = NEGATIVE_UNDERFLOW
    with pytest.raises(ValueError, match="nonnegative"):
        scheduler.step([1.0, 0.5])
    assert opt.param_groups[0]["lr"] is NEGATIVE_UNDERFLOW
    opt.param_groups[0]["lr"] = before["param_group_lrs"][0]
    assert scheduler.state_dict() == before


@pytest.mark.parametrize("api", ["get_lr", "get_last_lr", "state_dict"])
def test_rate_snapshots_do_not_alias_live_or_cached_state(api):
    scheduler, opt = make_scheduler(tensor=True)
    live = opt.param_groups[0]["lr"]
    expected = float(live)
    if api == "state_dict":
        snapshot = scheduler.state_dict()
        for field in ("param_group_lrs", "_last_lr", "base_lrs"):
            snapshot[field][0].fill_(9.0)
    else:
        getattr(scheduler, api)()[0].fill_(9.0)
    # Check before any step could replace cached state and hide an alias.
    assert opt.param_groups[0]["lr"] is live
    assert float(live) == expected
    assert float(scheduler.get_last_lr()[0]) == expected
    assert float(scheduler.base_lrs[0]) == expected
    assert float(opt.param_groups[0]["initial_lr"]) == expected


@pytest.mark.parametrize("wrapper", [False, True])
def test_unrepresentable_nonzero_slope_is_rejected_without_committing(wrapper):
    tiny = float.fromhex("0x0.0000000000001p-1022")
    scheduler, opt = make_scheduler()
    if wrapper:
        scheduler = DLRSOnPlateau(opt, patience=1)
        scheduler.step(tiny)
        scheduler.step(1e308)
    before = scheduler.state_dict()
    with pytest.raises(ValueError, match="nonzero loss slope.*representable"):
        scheduler.step(0.0 if wrapper else [tiny, 1e308, 0.0])
    assert scheduler.state_dict() == before


def test_observation_generator_failure_does_not_commit_partial_epoch():
    scheduler, _ = make_scheduler()
    scheduler.step([1.0, 0.5])
    before = scheduler.state_dict()

    def observations():
        yield 1.0
        yield 0.5
        raise RuntimeError("input stream failed")

    with pytest.raises(RuntimeError, match="input stream failed"):
        scheduler.step(observations())
    assert scheduler.state_dict() == before


def test_signed_zero_is_still_an_allowed_zero_observation():
    scheduler, opt = make_scheduler()
    scheduler.step([1.0, -0.0])
    assert opt.param_groups[0]["lr"] > 0.1


@pytest.mark.parametrize("profile", ["trap_inexact", "narrow_exponents", "different_rounding"])
@pytest.mark.parametrize("case", ["large_zero_adjustment", "subnormal_update"])
def test_decimal_fallback_is_independent_of_caller_context(profile, case):
    tiny = float.fromhex("0x0.0000000000001p-1022")

    def run():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if case == "large_zero_adjustment":
                rate, config, losses = 1e308, {"delta_o": 1e308}, [1.0, 1.0]
            else:
                rate, config, losses = 3 * tiny, {"delta_d": 0.1, "min_lr": tiny}, [0.0, 0.0, 1.0]
            opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=rate)
            scheduler = DLRSScheduler(opt, **config)
            scheduler.step(losses)
            return scheduler.state_dict()

    expected = run()
    with localcontext() as context:
        if profile == "trap_inexact":
            context.traps[Inexact] = True
        elif profile == "narrow_exponents":
            context.Emin, context.Emax = -2, 2
        else:
            context.prec, context.rounding = 2, ROUND_UP
        before = (context.prec, context.rounding, context.Emin, context.Emax,
                  dict(context.traps), dict(context.flags))
        assert run() == expected
        assert (context.prec, context.rounding, context.Emin, context.Emax,
                dict(context.traps), dict(context.flags)) == before
