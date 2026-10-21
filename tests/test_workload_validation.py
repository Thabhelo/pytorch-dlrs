"""Protect held-out evaluation, paired acceptance, and finite-update guards."""

import copy
import hashlib
import json
import shutil
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
    arguments = (tmp_path, "HelmholtzPINN", "tuning", 0, "DLRS", 0.01, protocol(), "cpu", tmp_path, {})
    first = validation.trial(*arguments)
    assert first["status"] == "failed" and "intentional failure" in first["error"]
    assert validation.trial(*arguments) == first
    assert len(calls) == 1


@pytest.fixture
def vision_inputs(monkeypatch, tmp_path):
    """Tiny loader-shaped files; no download, deserialization, or training."""
    calls = []

    class FakeMNIST:
        resources = [(name + ".gz", "unused") for name in (
            "train-images-idx3-ubyte", "train-labels-idx1-ubyte",
            "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte",
        )]
        training_file = "training.pt"
        test_file = "test.pt"

        def __init__(self, root, train, download):
            assert train, "Fingerprinting must never construct the held-out dataset"
            calls.append(("MNIST", download))

    class FakeCIFAR10:
        base_folder = "cifar-10-batches-py"
        filename = "cifar-10-python.tar.gz"
        train_list = [(f"data_batch_{index}", "unused") for index in range(1, 6)]
        test_list = [("test_batch", "unused")]
        meta = {"filename": "batches.meta"}

        def __init__(self, root, train, download):
            assert train, "Fingerprinting must never construct the held-out dataset"
            calls.append(("CIFAR10", download))

    monkeypatch.setitem(sys.modules, "torchvision", types.SimpleNamespace(
        datasets=types.SimpleNamespace(MNIST=FakeMNIST, CIFAR10=FakeCIFAR10),
    ))
    monkeypatch.setattr(validation, "code_fingerprint", lambda: {"runner": "fixed"})
    monkeypatch.setattr(validation, "version", lambda name: "test-version")
    root = tmp_path / "data"
    names = ["MNIST/raw/" + resource for resource, _ in FakeMNIST.resources]
    names += [name[:-3] for name in names]
    names += ["cifar-10-batches-py/" + name for name, _ in
              [*FakeCIFAR10.train_list, *FakeCIFAR10.test_list]]
    names += ["cifar-10-batches-py/batches.meta", FakeCIFAR10.filename]
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    return root, calls, names


def test_manifest_binds_loader_inputs_and_unchanged_resume_never_downloads(vision_inputs, tmp_path):
    root, calls, names = vision_inputs
    output = tmp_path / "results"
    first = validation.prepare_manifest(output, protocol(), "cpu", root)
    files = {name: value for dataset in first["dataset_fingerprints"].values()
             for name, value in dataset["files"].items()}
    assert set(files) == set(names)
    for name, value in files.items():
        assert value == {"bytes": len(name.encode()),
                         "sha256": hashlib.sha256(name.encode()).hexdigest()}
    original = (output / "manifest.json").read_bytes()
    moved = tmp_path / "same-data-new-location"
    shutil.copytree(root, moved)
    assert validation.prepare_manifest(output, protocol(), "cpu", moved) == first
    assert calls == [("MNIST", True), ("CIFAR10", True)]
    assert (output / "manifest.json").read_bytes() == original


@pytest.mark.parametrize("name", [
    "MNIST/raw/train-images-idx3-ubyte", "MNIST/raw/train-labels-idx1-ubyte",
    "MNIST/raw/t10k-images-idx3-ubyte", "MNIST/raw/t10k-labels-idx1-ubyte",
    "MNIST/processed/training.pt", "MNIST/processed/test.pt",
    "cifar-10-batches-py/data_batch_1", "cifar-10-batches-py/test_batch",
    "cifar-10-batches-py/batches.meta", "cifar-10-python.tar.gz",
])
def test_changed_data_aborts_subset_resume_before_cache_or_new_trial(
    vision_inputs, tmp_path, monkeypatch, name,
):
    root, calls, _ = vision_inputs
    if "/processed/" in name:
        for legacy in ("training.pt", "test.pt"):
            path = root / "MNIST/processed" / legacy
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(legacy.encode())
    output = tmp_path / "results"
    validation.prepare_manifest(output, protocol(), "cpu", root)
    cached = output / "trials/cached.json"
    validation.atomic_json(cached, {"old": "evidence"})
    before = {path: path.read_bytes() for path in output.rglob("*.json")}
    changed = root / name
    # Preserve file length: the guard must hash contents, not just size or paths.
    changed.write_bytes(b"!" + changed.read_bytes()[1:])
    config_path = tmp_path / "protocol.json"
    validation.atomic_json(config_path, protocol())
    monkeypatch.setattr(sys, "argv", [
        "validate_workloads", "--protocol", str(config_path), "--output", str(output),
        "--data-root", str(root), "--datasets", "HelmholtzPINN",
    ])
    monkeypatch.setattr(validation, "trial", lambda *a, **k: pytest.fail("Reached a trial"))
    with pytest.raises(ValueError, match="Dataset contents changed"):
        validation.main()
    assert {path: path.read_bytes() for path in output.rglob("*.json")} == before
    assert calls == [("MNIST", True), ("CIFAR10", True)]


@pytest.mark.parametrize("change", ["add_legacy", "remove_legacy", "remove_raw"])
def test_loader_source_changes_or_missing_inputs_fail_closed(vision_inputs, tmp_path, change):
    root, _, _ = vision_inputs
    legacy = root / "MNIST/processed"
    if change == "remove_legacy":
        legacy.mkdir()
        for name in ("training.pt", "test.pt"):
            (legacy / name).write_bytes(name.encode())
    output = tmp_path / "results"
    validation.prepare_manifest(output, protocol(), "cpu", root)
    if change == "add_legacy":
        legacy.mkdir()
        for name in ("training.pt", "test.pt"):
            (legacy / name).write_bytes(name.encode())
    elif change == "remove_legacy":
        (legacy / "training.pt").unlink()
    else:
        (root / "MNIST/raw/train-images-idx3-ubyte").unlink()
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="Dataset contents changed|Cannot fingerprint"):
        validation.prepare_manifest(output, protocol(), "cpu", root)
    assert (output / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("cached", [False, True])
def test_trial_rechecks_dataset_before_new_or_cached_work(vision_inputs, tmp_path, monkeypatch, cached):
    root, _, _ = vision_inputs
    output = tmp_path / "results"
    saved = validation.prepare_manifest(output, protocol(), "cpu", root)
    path = output / "trials/MNIST-tuning-s0-DLRS-lr0.01.json"
    if cached:
        validation.atomic_json(path, {"dataset": "MNIST", "phase": "tuning", "seed": 0,
                                      "policy": "DLRS", "lr": 0.01, "status": "passed"})
    before = path.read_bytes() if cached else None
    (root / "MNIST/raw/train-labels-idx1-ubyte").write_bytes(b"changed between trials")
    monkeypatch.setattr(validation, "run_experiment", lambda *a, **k: pytest.fail("Started training"))
    with pytest.raises(ValueError, match="Dataset contents changed"):
        validation.trial(output, "MNIST", "tuning", 0, "DLRS", 0.01, protocol(), "cpu", root,
                         saved["dataset_fingerprints"])
    assert (path.read_bytes() if path.exists() else None) == before


@pytest.mark.parametrize("mutate", [False, True])
def test_new_trial_disables_download_and_rechecks_inputs_after_execution(
    vision_inputs, tmp_path, monkeypatch, mutate,
):
    root, _, _ = vision_inputs
    output = tmp_path / "results"
    saved = validation.prepare_manifest(output, protocol(), "cpu", root)

    def synthetic_trial(*args, **kwargs):
        assert kwargs["download"] is False
        assert kwargs["evaluate_test"] is False
        if mutate:
            (root / "MNIST/raw/train-images-idx3-ubyte").write_bytes(b"changed during trial")
        return {}

    monkeypatch.setattr(validation, "run_experiment", synthetic_trial)
    arguments = (output, "MNIST", "tuning", 0, "DLRS", 0.01, protocol(), "cpu", root,
                 saved["dataset_fingerprints"])
    if mutate:
        with pytest.raises(ValueError, match="Dataset contents changed"):
            validation.trial(*arguments)
        assert not list((output / "trials").glob("*.json"))
    else:
        assert validation.trial(*arguments)["status"] == "passed"


@pytest.mark.parametrize("legacy", ["manifest_without_hashes", "trials_without_manifest"])
def test_unbound_existing_evidence_is_not_blessed_or_overwritten(vision_inputs, tmp_path, legacy):
    root, calls, _ = vision_inputs
    output = tmp_path / "results"
    saved = validation.prepare_manifest(output, protocol(), "cpu", root)
    if legacy == "manifest_without_hashes":
        saved.pop("dataset_fingerprints")
        validation.atomic_json(output / "manifest.json", saved)
        message = "no dataset fingerprints"
    else:
        (output / "manifest.json").unlink()
        validation.atomic_json(output / "trials/cached.json", {"old": "evidence"})
        message = "no manifest"
    before = {path: path.read_bytes() for path in output.rglob("*.json")}
    with pytest.raises(ValueError, match=message):
        validation.prepare_manifest(output, protocol(), "cpu", root)
    assert {path: path.read_bytes() for path in output.rglob("*.json")} == before
    assert calls == [("MNIST", True), ("CIFAR10", True)]
