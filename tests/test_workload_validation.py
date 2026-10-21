"""Protect held-out evaluation, paired acceptance, and finite-update guards."""

import copy
import json
import sys
import types

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from benchmarks import run_benchmarks as benchmark
from benchmarks import validate_workloads as validation


def protocol():
    return json.loads((validation.ROOT / "benchmarks/validation_protocol.json").read_text())


def test_tuning_never_constructs_test_dataset_and_fixed_split_survives_seed_change(monkeypatch):
    class FakeMNIST(TensorDataset):
        def __init__(self, root, train, transform=None, download=False):
            assert train, "Tuning must never open held-out test data"
            inputs = torch.arange(20, dtype=torch.float32).view(-1, 1) / 20
            super().__init__(inputs, (inputs[:, 0] > 0.5).long())

    monkeypatch.setitem(
        sys.modules,
        "torchvision",
        types.SimpleNamespace(
            datasets=types.SimpleNamespace(MNIST=FakeMNIST),
            transforms=types.SimpleNamespace(
                Compose=lambda items: None, ToTensor=lambda: None, Normalize=lambda *args: None
            ),
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "MNISTCNN",
        lambda: torch.nn.Sequential(torch.nn.Linear(1, 2), torch.nn.LogSoftmax(dim=1)),
    )
    results = [
        benchmark.run_experiment(
            "MNIST",
            "DLRS",
            torch.device("cpu"),
            1,
            4,
            0.01,
            seed=seed,
            split_seed=42,
            evaluate_test=False,
        )
        for seed in [11, 12]
    ]
    assert results[0]["split_sha256"] == results[1]["split_sha256"]
    assert results[0]["initial_model_sha256"] != results[1]["initial_model_sha256"]
    for result in results:
        assert "test_loss" not in result and "final_accuracy" not in result


@pytest.mark.parametrize("failure", ["loss", "gradient"])
def test_benchmark_rejects_nonfinite_before_parameter_or_optimizer_mutation(failure):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = copy.deepcopy(model.state_dict())
    loader = DataLoader(TensorDataset(torch.ones(2, 1), torch.ones(2, 1)), batch_size=2)
    if failure == "gradient":
        model.weight.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))

    def criterion(output, target):
        return (output - target).square().mean() * (float("inf") if failure == "loss" else 1)

    with pytest.raises((ValueError, RuntimeError), match="[Nn]on.?finite"):
        benchmark.train_epoch(model, torch.device("cpu"), loader, optimizer, criterion)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert not optimizer.state


@pytest.mark.parametrize("policy", ["DLRS", "NoScheduler", "CosineAnnealingLR"])
def test_pinn_tuning_is_repeatable_and_has_no_test_measurement(policy):
    config = dict(
        protocol()["pinn"],
        epochs=2,
        batches_per_epoch=2,
        interior_points_per_batch=8,
        validation_points=9,
        test_points=16,
    )
    first = validation.run_pinn(policy, 0.001, 5, config, False)
    second = validation.run_pinn(policy, 0.001, 5, config, False)
    assert first["train_loss"] == second["train_loss"]
    assert first["validation_loss"] == second["validation_loss"]
    assert "test_loss" not in first and "relative_rmse" not in first
    confirmation = validation.run_pinn(policy, 0.001, 5, config, True)
    assert confirmation["train_loss"] == first["train_loss"]
    assert confirmation["relative_rmse"] >= 0


def test_exact_bootstrap_constant_and_input_bounds():
    assert validation.paired_interval([2.0] * 5) == [2.0, 2.0]
    assert validation.paired_interval([-1.0, 1.0]) == [-1.0, 1.0]
    with pytest.raises(ValueError):
        validation.paired_interval([])


def paired_records(dataset="MNIST"):
    config = protocol()
    metric = "relative_rmse" if dataset == "HelmholtzPINN" else "final_accuracy"
    records = [
        {
            "dataset": dataset,
            "phase": "confirmation",
            "seed": seed,
            "policy": policy,
            "status": "passed",
            "result": {
                metric: 0.1 if dataset == "HelmholtzPINN" else 90.0,
                "initial_model_sha256": str(seed),
                "split_sha256": "fixed",
                "total_time": 10.0,
            },
        }
        for seed in config["confirmation_seeds"]
        for policy in config["policies"]
    ]
    return records, {dataset: dict.fromkeys(config["policies"], 0.01)}, config


@pytest.mark.parametrize("dataset", ["MNIST", "HelmholtzPINN"])
def test_acceptance_requires_complete_matched_noninferior_trials(dataset):
    records, selections, config = paired_records(dataset)
    assert validation.summarize(records, selections, config)[0]["accepted"]
    for row in records:
        if row["policy"] == "DLRS":
            row["result"]["relative_rmse" if dataset == "HelmholtzPINN" else "final_accuracy"] = (
                0.12 if dataset == "HelmholtzPINN" else 88.0
            )
    assert not validation.summarize(records, selections, config)[0]["accepted"]
    records[0]["status"] = "failed"
    summary = validation.summarize(records, selections, config)[0]
    assert not summary["accepted"]
    assert summary["policies"]["NoScheduler"]["failed"] == 1


@pytest.mark.parametrize("key", ["initial_model_sha256", "split_sha256"])
def test_pair_mismatch_fails_instead_of_reporting_comparison(key):
    records, selections, config = paired_records()
    records[0]["result"][key] = "different"
    with pytest.raises(ValueError, match="did not share"):
        validation.summarize(records, selections, config)


def test_failed_trial_is_recorded_and_resume_does_not_hide_it(monkeypatch, tmp_path):
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise ValueError("intentional failure")

    monkeypatch.setattr(validation, "run_pinn", fail)
    arguments = (tmp_path, "HelmholtzPINN", "tuning", 0, "DLRS", 0.01, protocol(), "cpu", tmp_path)
    first = validation.trial(*arguments)
    assert first["status"] == "failed" and "intentional failure" in first["error"]
    assert validation.trial(*arguments) == first
    assert len(calls) == 1
