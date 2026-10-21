"""Independent formula, training-resume and explicit DDP caller contracts.

These tests establish bounded implementation behavior, not an accuracy guarantee.
"""

import copy
import datetime
import io
import math
import random
from decimal import Decimal, localcontext

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dlrs import DLRSOnPlateau, DLRSScheduler


def _paper_update(rate, losses, minimum=1e-300, maximum=1e10):
    # Independent high-precision equation, without the implementation's helpers.
    with localcontext() as context:
        context.prec = 80
        values = [Decimal(str(x)) for x in losses]
        slope = (values[-1] - values[0]) / (sum(values) / len(values))
        delta = Decimal("0.5") if slope > 1 else Decimal("1") if slope >= 0 else Decimal("0.1")
        rate = Decimal(str(rate))
        updated = rate - Decimal(10) ** rate.adjusted() * delta * slope
        return float(min(max(updated, Decimal(str(minimum))), Decimal(str(maximum))))


def test_randomized_updates_match_independent_decimal_reference():
    rng = random.Random(9153)
    parameter = torch.nn.Parameter(torch.zeros(1))
    for _ in range(500):
        rate = rng.uniform(1.05, 9.8) * 10 ** rng.randint(-25, 8)
        losses = [float(rng.randint(1, 1000)) for _ in range(rng.randint(2, 25))]
        optimizer = torch.optim.SGD([parameter], lr=rate)
        scheduler = DLRSScheduler(optimizer, min_lr=1e-300, max_lr=1e10)
        optimizer.step()
        scheduler.step(losses)
        assert scheduler.get_last_lr()[0] == pytest.approx(
            _paper_update(rate, losses), rel=2e-13, abs=0.0
        )


@pytest.mark.parametrize("scale", [1e-300, 1e-15, 1.0, 1e300])
def test_positive_loss_scaling_preserves_formula(scale):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    scheduler = DLRSScheduler(optimizer)
    optimizer.step()
    scheduler.step([scale * x for x in [2.0, 3.0, 1.0]])
    assert scheduler.get_last_lr()[0] == pytest.approx(_paper_update(0.1, [2.0, 3.0, 1.0]))


@pytest.mark.parametrize("optimizer_type", [torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW])
@pytest.mark.parametrize("plateau", [False, True])
def test_checkpoint_resume_matches_uninterrupted_training(optimizer_type, plateau):
    torch.manual_seed(38)
    inputs = torch.randn(24, 3, dtype=torch.float64)
    targets = torch.randn(24, 1, dtype=torch.float64)

    def setup():
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1)
        ).double()
        optimizer = optimizer_type(
            [
                {"params": model[0].parameters(), "lr": 0.01},
                {"params": model[2].parameters(), "lr": 0.002},
            ]
        )
        scheduler = (
            DLRSOnPlateau(optimizer, patience=2, max_lr=0.1)
            if plateau
            else DLRSScheduler(optimizer, max_lr=0.1)
        )
        return model, optimizer, scheduler

    def epoch(model, optimizer, scheduler):
        losses = []
        for x, y in zip(inputs.split(4), targets.split(4)):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step(sum(losses) / len(losses) if plateau else losses)
        return losses

    model, optimizer, scheduler = setup()
    for _ in range(4):
        epoch(model, optimizer, scheduler)
    checkpoint = io.BytesIO()
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        checkpoint,
    )
    expected_losses = [epoch(model, optimizer, scheduler) for _ in range(6)]
    checkpoint.seek(0)
    state = torch.load(checkpoint, weights_only=True)
    resumed_model, resumed_optimizer, resumed_scheduler = setup()
    resumed_model.load_state_dict(state["model"])
    # This replaces parameter-group dictionaries; parameter identity stays intact.
    resumed_optimizer.load_state_dict(state["optimizer"])
    resumed_scheduler.load_state_dict(state["scheduler"])
    actual_losses = [epoch(resumed_model, resumed_optimizer, resumed_scheduler) for _ in range(6)]
    assert actual_losses == expected_losses
    assert resumed_scheduler.get_last_lr() == scheduler.get_last_lr()
    for actual, expected in zip(resumed_model.parameters(), model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_long_run_bounds_and_bounded_wrapper_checkpoint():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    scheduler = DLRSOnPlateau(optimizer, min_lr=1e-6, max_lr=0.1, patience=3)
    for epoch in range(10_000):
        optimizer.step()
        scheduler.step([0.01, 2.0, 1.0, 0.001][epoch % 4])
        assert 1e-6 <= optimizer.param_groups[0]["lr"] <= 0.1
        assert math.isfinite(optimizer.param_groups[0]["lr"])
    assert len(scheduler.state_dict()["metric_history"]) <= 5
    snapshot = copy.deepcopy(scheduler.state_dict())
    scheduler.step(1.0)
    assert snapshot != scheduler.state_dict()


def _ddp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        torch.manual_seed(21)
        model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(2, 1).double())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
        scheduler = DLRSScheduler(optimizer, max_lr=0.1)
        for epoch in range(5):
            losses = []
            for batch in range(3):
                inputs = torch.full((2, 2), (rank + batch + 1) / 10, dtype=torch.float64)
                targets = torch.full((2, 1), (epoch + 1) / 10, dtype=torch.float64)
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(model(inputs), targets)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            # Equal batch sizes/counts: the caller creates one common ordered loss
            # sequence and EVERY rank steps its own scheduler and optimizer.
            reduced = torch.tensor(losses, dtype=torch.float64)
            dist.all_reduce(reduced)
            reduced /= dist.get_world_size()
            scheduler.step(reduced.tolist())
            local = torch.cat(
                [
                    torch.tensor(scheduler.get_last_lr(), dtype=torch.float64),
                    *[parameter.detach().flatten() for parameter in model.parameters()],
                ]
            )
            gathered = [torch.empty_like(local) for _ in range(2)]
            dist.all_gather(gathered, local)
            torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable"
)
def test_two_process_ddp_with_explicit_global_losses(tmp_path):
    mp.spawn(_ddp_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)
