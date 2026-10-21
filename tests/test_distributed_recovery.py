"""Bounded two-process Gloo integration, rejection, and restart experiments.

The explicit synchronization policy lives in examples/distributed_policy.py;
DLRS itself does not coordinate ranks or recover failed processes.
"""

import copy
import datetime
import faulthandler
import json
import math
import multiprocessing
import os
import time
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from dlrs import DLRSScheduler
from examples.distributed_policy import (
    DistributedObservationError,
    reduce_sample_mean,
    require_equal_iterations,
)

pytestmark = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable"
)

WORLD_SIZE = 2
BATCHES = 4
TOTAL_EPOCHS = 4
SPLIT_EPOCH = 2


def _assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


def _batch(rank, epoch, step):
    counts = ((1, 3, 2, 4), (5, 2, 6, 1))
    generator = torch.Generator().manual_seed(813 + epoch * 101 + step * 7 + rank)
    inputs = torch.randn(counts[rank][step], 3, generator=generator, dtype=torch.float64)
    targets = (0.6 * inputs[:, :1] - 0.3 * inputs[:, 1:2]
               + 0.15 * inputs[:, 2:3].square() + 0.1 * rank)
    return inputs, targets


def _make_optimizer(model, kind):
    rates = (0.012, 0.004)
    tensor_rates = kind == "adamw_tensor"
    groups = [
        {"params": list(model[0].parameters()),
         "lr": torch.tensor(rates[0], dtype=torch.float64) if tensor_rates else rates[0]},
        {"params": list(model[3].parameters()),
         "lr": torch.tensor(rates[1], dtype=torch.float64) if tensor_rates else rates[1]},
    ]
    if kind == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, weight_decay=0.01, foreach=False)
    if kind == "adamw_tensor":
        return torch.optim.AdamW(groups, weight_decay=0.01, foreach=False)
    raise AssertionError(f"unexpected optimizer {kind}")


def _setup(rank, kind, dropout):
    torch.manual_seed(271)
    module = torch.nn.Sequential(
        torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Dropout(dropout), torch.nn.Linear(5, 1)
    ).double()
    model = torch.nn.parallel.DistributedDataParallel(module)
    optimizer = _make_optimizer(module, kind)
    scheduler = DLRSScheduler(optimizer, min_lr=1e-6, max_lr=0.03)
    # Each rank has its own stochastic stream; checkpoint it separately.
    torch.manual_seed(991 + rank)
    return model, optimizer, scheduler


def _snapshot(model, optimizer, scheduler, next_epoch, history):
    return copy.deepcopy({
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": torch.get_rng_state(),
        "next_epoch": next_epoch,
        "history": history,
    })


def _assert_rank_agreement(model, optimizer, scheduler):
    local = {"model": model.module.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict()}
    gathered = [None for _ in range(WORLD_SIZE)]
    dist.all_gather_object(gathered, local)
    _assert_nested_equal(gathered[0], gathered[1])


def _train_epoch(rank, epoch, model, optimizer, scheduler, reference=None):
    require_equal_iterations(BATCHES)
    observations = []
    details = []
    rate_objects = [group["lr"] for group in optimizer.param_groups]
    model.train()
    for step in range(BATCHES):
        inputs, targets = _batch(rank, epoch, step)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        observation = reduce_sample_mean(loss, inputs.shape[0])

        # Independent arithmetic check, including a deliberately unequal-rank mean.
        local = torch.tensor([loss.detach().item(), inputs.shape[0]], dtype=torch.float64)
        gathered = [torch.empty_like(local) for _ in range(WORLD_SIZE)]
        dist.all_gather(gathered, local)
        expected_mean = math.fsum(float(item[0]) * int(item[1]) for item in gathered)
        expected_mean /= sum(int(item[1]) for item in gathered)
        assert observation.mean == expected_mean
        assert observation.sample_count == sum(int(item[1]) for item in gathered)
        assert observation.backward_scale == WORLD_SIZE * inputs.shape[0] / observation.sample_count
        (loss * observation.backward_scale).backward()
        optimizer.step()
        observations.append(observation.mean)
        details.append({"mean": observation.mean, "count": observation.sample_count,
                        "unweighted_rank_mean": sum(float(item[0]) for item in gathered) / WORLD_SIZE})

        if reference is not None:
            reference_model, reference_optimizer, reference_losses = reference
            reference_optimizer.zero_grad(set_to_none=True)
            batches = [_batch(other_rank, epoch, step) for other_rank in range(WORLD_SIZE)]
            combined_inputs = torch.cat([batch[0] for batch in batches])
            combined_targets = torch.cat([batch[1] for batch in batches])
            reference_loss = torch.nn.functional.mse_loss(
                reference_model(combined_inputs), combined_targets
            )
            reference_loss.backward()
            reference_optimizer.step()
            reference_losses.append(reference_loss.item())
            assert observation.mean == pytest.approx(reference_loss.item(), rel=2e-13, abs=1e-15)
            for actual, expected in zip(model.module.parameters(), reference_model.parameters()):
                torch.testing.assert_close(actual, expected, rtol=2e-13, atol=1e-15)
    scheduler.step(observations)
    for group, rate in zip(optimizer.param_groups, rate_objects):
        if isinstance(rate, torch.Tensor):
            assert group["lr"] is rate
    _assert_rank_agreement(model, optimizer, scheduler)
    return {"epoch": epoch, "observations": observations, "details": details,
            "rates": [float(rate) for rate in scheduler.get_last_lr()]}


def _training_worker(rank, config):
    model, optimizer, scheduler = _setup(rank, config["optimizer"], dropout=0.2)
    start_epoch = 0
    history = []
    if config.get("resume"):
        saved = torch.load(Path(config["resume"]) / f"rank-{rank}.pt", weights_only=True)
        model.module.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch = saved["next_epoch"]
        history = saved["history"]
        torch.set_rng_state(saved["rng"])
    for epoch in range(start_epoch, config["epochs"]):
        history.append(_train_epoch(rank, epoch, model, optimizer, scheduler))
    result = _snapshot(model, optimizer, scheduler, config["epochs"], history)
    assert any(abs(detail["mean"] - detail["unweighted_rank_mean"]) > 1e-6
               for epoch in history for detail in epoch["details"])
    return result


def _reference_worker(rank, config):
    model, optimizer, scheduler = _setup(rank, "sgd", dropout=0.0)
    reference_model = copy.deepcopy(model.module)
    reference_optimizer = _make_optimizer(reference_model, "sgd")
    reference_scheduler = DLRSScheduler(reference_optimizer, min_lr=1e-6, max_lr=0.03)
    histories = []
    for epoch in range(2):
        reference_losses = []
        histories.append(_train_epoch(
            rank, epoch, model, optimizer, scheduler,
            (reference_model, reference_optimizer, reference_losses),
        ))
        reference_scheduler.step(reference_losses)
        assert scheduler.get_last_lr() == pytest.approx(reference_scheduler.get_last_lr(), rel=2e-13)
    return {"history": histories, "parameter_match_rtol": 2e-13,
            "parameter_match_atol": 1e-15}


def _invalid_worker(rank, config):
    model, optimizer, scheduler = _setup(rank, "sgd", dropout=0.0)
    _train_epoch(rank, 0, model, optimizer, scheduler)
    before = _snapshot(model, optimizer, scheduler, 1, [])
    loss = torch.tensor(1.0, dtype=torch.float64)
    count = 3
    kind = config["invalid"]
    if rank == 1:
        if kind == "nan":
            loss.fill_(math.nan)
        elif kind == "negative":
            loss.fill_(-1)
        elif kind == "zero_count":
            count = 0
        elif kind == "nonscalar":
            loss = torch.tensor([1.0, 2.0])
    if kind == "global_overflow":
        loss.fill_(1e308)
        count = 1
    try:
        reduce_sample_mean(loss, count)
    except DistributedObservationError as exc:
        error = str(exc)
    else:
        raise AssertionError("invalid observations were accepted")
    _assert_nested_equal(_snapshot(model, optimizer, scheduler, 1, []), before)
    # These healthy ranks can still execute a collective after a validation error;
    # this is different from a failed process-group operation.
    dist.barrier()
    return {"rejected": True, "error": error, "state_unchanged": True}


def _iteration_worker(rank, config):
    try:
        require_equal_iterations(4 + rank)
    except DistributedObservationError as exc:
        return {"rejected": True, "error": str(exc)}
    raise AssertionError("unequal iteration counts were accepted")


def _failure_worker(rank, config):
    model, optimizer, scheduler = _setup(rank, "sgd", dropout=0.0)
    before = _snapshot(model, optimizer, scheduler, 0, [])
    dist.barrier()
    if rank == 1:
        if config["failure"] == "exit":
            os._exit(17)
        time.sleep(8.0)  # Deliberately exceed the four-second collective timeout.
        return {"injected_stall_seconds": 8.0}
    started = time.monotonic()
    try:
        reduce_sample_mean(torch.tensor(1.0), 2)
    except (RuntimeError, OSError) as exc:
        elapsed = time.monotonic() - started
        assert elapsed < 15.0, f"collective failure was not bounded: {elapsed} seconds"
        _assert_nested_equal(_snapshot(model, optimizer, scheduler, 0, []), before)
        return {"communication_failed": True, "error_type": type(exc).__name__,
                "error": str(exc), "elapsed_seconds": elapsed, "state_unchanged": True}
    raise AssertionError("a missing rank unexpectedly completed the collective")


def _record_progress(directory, rank, phase):
    path = directory / f"rank-{rank}-progress.json"
    progress = json.loads(path.read_text()) if path.exists() else {
        "pid": os.getpid(), "rank": rank, "events": [],
    }
    progress["events"].append({"phase": phase, "monotonic_seconds": time.monotonic()})
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(progress, indent=2))
    temporary.replace(path)


def _worker_diagnostics(directory, processes):
    reports = [f"Worker pid={process.pid} exitcode={process.exitcode} alive={process.is_alive()}"
               for process in processes]
    for path in sorted(directory.glob("rank-*")):
        if path.name.endswith(("-progress.json", "-stacks.txt", "-error.txt")):
            # Keep parent failure output bounded; complete files remain artifacts.
            reports.append(f"{path.name}:\n{path.read_text(errors='replace')[-20000:]}")
    if not any(directory.glob("rank-*-progress.json")):
        reports.append("No worker entry marker: inspect spawn/bootstrap/module import startup.")
    return "\n".join(reports)


def _worker_entry(rank, directory, case, config):
    directory = Path(directory)
    stacks = (directory / f"rank-{rank}-stacks.txt").open("w")
    faulthandler.enable(file=stacks, all_threads=True)
    # Diagnostics only: neither this timer nor the phase markers alter deadlines.
    faulthandler.dump_traceback_later(20.0, repeat=True, file=stacks)
    try:
        _record_progress(directory, rank, "worker_entry")
        _record_progress(directory, rank, "setting_intraop_threads")
        torch.set_num_threads(1)
        _record_progress(directory, rank, "setting_interop_threads")
        torch.set_num_interop_threads(1)
        timeout = 4 if case == "failure" else 15
        _record_progress(directory, rank, "initializing_process_group")
        dist.init_process_group(
            "gloo", init_method=(directory / "rendezvous").as_uri(), rank=rank,
            world_size=WORLD_SIZE, timeout=datetime.timedelta(seconds=timeout),
        )
        workers = {"training": _training_worker, "reference": _reference_worker,
                   "invalid": _invalid_worker, "iterations": _iteration_worker,
                   "failure": _failure_worker}
        _record_progress(directory, rank, "running_" + case)
        result = workers[case](rank, config)
        _record_progress(directory, rank, "saving_result")
        torch.save(result, directory / f"rank-{rank}.pt")
        (directory / f"rank-{rank}.json").write_text(json.dumps(
            {"pid": os.getpid(), "rank": rank, "case": case, "completed": True}
        ))
    except BaseException:
        (directory / f"rank-{rank}-error.txt").write_text(traceback.format_exc())
        raise
    finally:
        try:
            _record_progress(directory, rank, "destroying_process_group")
            if dist.is_initialized():
                dist.destroy_process_group()
            _record_progress(directory, rank, "worker_finished")
        finally:
            faulthandler.cancel_dump_traceback_later()
            faulthandler.disable()
            stacks.close()


def _run_pair(directory, case, config, expected_exitcodes=(0, 0)):
    directory.mkdir()
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_worker_entry, args=(rank, str(directory), case, config))
                 for rank in range(WORLD_SIZE)]
    started = []
    deadline = time.monotonic() + 45.0
    try:
        for process in processes:
            process.start()
            started.append(process)
        for process in started:
            process.join(max(0.0, deadline - time.monotonic()))
        if any(process.is_alive() for process in started):
            pytest.fail("distributed child processes exceeded the 45-second launcher deadline\n"
                        + _worker_diagnostics(directory, started))
        errors = _worker_diagnostics(directory, started)
        assert tuple(process.exitcode for process in started) == expected_exitcodes, errors
        outputs = []
        for rank, expected in enumerate(expected_exitcodes):
            if expected == 0:
                outputs.append(torch.load(directory / f"rank-{rank}.pt", weights_only=True))
            else:
                outputs.append(None)
        return outputs
    finally:
        for process in started:
            if process.is_alive():
                process.terminate()
        for process in started:
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
            assert not process.is_alive(), "failed to clean up a distributed child process"
            process.close()


def test_weighted_observations_and_gradients_match_global_batch(tmp_path):
    result = _run_pair(tmp_path / "reference", "reference", {})
    assert result[0] == result[1]
    assert any(abs(detail["mean"] - detail["unweighted_rank_mean"]) > 1e-6
               for epoch in result[0]["history"] for detail in epoch["details"])


@pytest.mark.parametrize("optimizer_kind", ["sgd", "adamw_tensor"])
def test_epoch_restart_in_fresh_processes_matches_uninterrupted(tmp_path, optimizer_kind):
    full_dir, prefix_dir, resumed_dir = (tmp_path / name for name in ("full", "prefix", "resumed"))
    uninterrupted = _run_pair(full_dir, "training", {"optimizer": optimizer_kind, "epochs": TOTAL_EPOCHS})
    _run_pair(prefix_dir, "training", {"optimizer": optimizer_kind, "epochs": SPLIT_EPOCH})
    resumed = _run_pair(resumed_dir, "training", {
        "optimizer": optimizer_kind, "epochs": TOTAL_EPOCHS, "resume": str(prefix_dir),
    })
    for rank in range(WORLD_SIZE):
        _assert_nested_equal(resumed[rank], uninterrupted[rank])
    pids = [json.loads(path.read_text())["pid"] for directory in (full_dir, prefix_dir, resumed_dir)
            for path in directory.glob("rank-[0-9].json")]
    assert len(set(pids)) == 6, "the restart experiment must use six fresh worker processes"


@pytest.mark.parametrize("invalid", ["nan", "negative", "zero_count", "nonscalar", "global_overflow"])
def test_invalid_observation_is_rejected_on_every_rank(tmp_path, invalid):
    result = _run_pair(tmp_path / invalid, "invalid", {"invalid": invalid})
    assert result[0] == result[1]
    assert result[0]["rejected"] and result[0]["state_unchanged"]


def test_unequal_iteration_counts_fail_collectively_before_training(tmp_path):
    result = _run_pair(tmp_path / "iterations", "iterations", {})
    assert result[0] == result[1]
    assert result[0]["rejected"]


@pytest.mark.parametrize("failure", ["exit", "stall"])
def test_rank_failure_or_stall_is_bounded_without_scheduler_mutation(tmp_path, failure):
    result = _run_pair(tmp_path / failure, "failure", {"failure": failure},
                       expected_exitcodes=(0, 17) if failure == "exit" else (0, 0))
    assert result[0]["communication_failed"]
    assert result[0]["state_unchanged"]
    assert result[0]["elapsed_seconds"] < 15.0
