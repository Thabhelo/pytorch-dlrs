"""Checks for fair accounting and failures without dataset downloads."""

import math
import sys
import types

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from benchmarks import run_benchmarks as benchmark


def test_evaluation_weights_partial_batches_by_sample_count():
    inputs = torch.tensor([[4.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
    targets = torch.tensor([0, 0, 0])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    loss, accuracy = benchmark.evaluate(
        torch.nn.Identity(), torch.device("cpu"), loader, torch.nn.CrossEntropyLoss()
    )
    assert loss == pytest.approx(torch.nn.functional.cross_entropy(inputs, targets).item())
    assert accuracy == pytest.approx(200 / 3)


def test_empty_loaders_fail():
    loader = DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)))
    with pytest.raises(ValueError, match="empty"):
        benchmark.evaluate(
            torch.nn.Identity(), torch.device("cpu"), loader, torch.nn.CrossEntropyLoss()
        )


def test_finite_extreme_batch_loss_does_not_overflow_weighted_mean():
    model = torch.nn.Linear(2, 2).double()
    inputs = torch.ones(3, 2, dtype=torch.float64)
    targets = torch.zeros(3, dtype=torch.long)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def criterion(output, targets):
        return output.sum() * 0 + output.new_tensor(1e308)

    train_loss, _ = benchmark.train_epoch(model, torch.device("cpu"), loader, optimizer, criterion)
    test_loss, _ = benchmark.evaluate(model, torch.device("cpu"), loader, criterion)
    assert math.isfinite(train_loss) and train_loss == pytest.approx(1e308)
    assert math.isfinite(test_loss) and test_loss == pytest.approx(1e308)


@pytest.mark.parametrize("scheduler_name", ["NoScheduler", "DLRS", "ReduceLROnPlateau"])
def test_benchmark_has_separate_validation_and_single_final_test(monkeypatch, scheduler_name):
    class FakeMNIST(TensorDataset):
        def __init__(self, root, train, transform=None, download=False):
            count = 20 if train else 7
            super().__init__(
                torch.arange(count, dtype=torch.float32).view(-1, 1),
                torch.zeros(count, dtype=torch.long),
            )

    transform_stub = types.SimpleNamespace(
        Compose=lambda items: None,
        ToTensor=lambda: None,
        Normalize=lambda *args: None,
    )
    monkeypatch.setitem(
        sys.modules,
        "torchvision",
        types.SimpleNamespace(
            datasets=types.SimpleNamespace(MNIST=FakeMNIST),
            transforms=transform_stub,
        ),
    )
    monkeypatch.setattr(benchmark, "MNISTCNN", lambda: torch.nn.Linear(1, 2))
    evaluations = []
    training = []

    def train(model, device, loader, optimizer, criterion, collect_losses):
        training.append(list(loader.dataset.indices))
        optimizer.step()
        return 1.0, [2.0, 1.0]

    def evaluate(model, device, loader, criterion):
        evaluations.append(loader.dataset)
        return (3.0, 50.0) if len(loader.dataset) == 7 else (1.0, 60.0)

    monkeypatch.setattr(benchmark, "train_epoch", train)
    monkeypatch.setattr(benchmark, "evaluate", evaluate)
    result = benchmark.run_experiment(
        "MNIST", scheduler_name, torch.device("cpu"), epochs=3, batch_size=4, lr=0.01, seed=19
    )
    assert len(evaluations) == 4  # validation once/epoch, test once/run
    assert [len(dataset) for dataset in evaluations] == [2, 2, 2, 7]
    assert set(training[0]).isdisjoint(evaluations[0].indices)
    assert result["learning_rates"][0] == 0.01
    assert result["test_loss"] == 3.0
    assert result["final_accuracy"] == 50.0
    assert result["seed"] == 19


def test_failed_benchmark_exits_nonzero_and_records_failures(monkeypatch, tmp_path):
    def fail(**kwargs):
        raise RuntimeError("intentional failed trial")

    monkeypatch.setattr(benchmark, "run_experiment", fail)
    output = tmp_path / "results.md"
    monkeypatch.setattr(
        sys, "argv", ["benchmark", "--quick", "--seeds", "0", "--output", str(output)]
    )
    with pytest.raises(SystemExit, match="incomplete"):
        benchmark.main()
    assert "INCOMPLETE" in output.read_text()
    assert "intentional failed trial" in output.with_suffix(".json").read_text()


def test_paired_seed_repeats_real_training_without_dataset_downloads(monkeypatch):
    class FakeMNIST(TensorDataset):
        def __init__(self, root, train, transform=None, download=False):
            count = 20 if train else 8
            inputs = torch.arange(count, dtype=torch.float32).view(-1, 1) / count
            super().__init__(inputs, (inputs[:, 0] > 0.5).long())

    monkeypatch.setitem(
        sys.modules,
        "torchvision",
        types.SimpleNamespace(
            datasets=types.SimpleNamespace(MNIST=FakeMNIST),
            transforms=types.SimpleNamespace(
                Compose=lambda items: None,
                ToTensor=lambda: None,
                Normalize=lambda *args: None,
            ),
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "MNISTCNN",
        lambda: torch.nn.Sequential(torch.nn.Linear(1, 2), torch.nn.LogSoftmax(dim=1)),
    )
    original_train = benchmark.train_epoch
    initial_models = []

    def track(model, *args, **kwargs):
        initial_models.append(torch.cat([p.detach().flatten().clone() for p in model.parameters()]))
        return original_train(model, *args, **kwargs)

    monkeypatch.setattr(benchmark, "train_epoch", track)
    results = [
        benchmark.run_experiment("MNIST", name, torch.device("cpu"), 2, 4, 0.01, seed=7)
        for name in ["DLRS", "DLRS", "NoScheduler"]
    ]
    for key in [
        "train_loss",
        "validation_loss",
        "validation_accuracy",
        "learning_rates",
        "test_loss",
        "final_accuracy",
    ]:
        assert results[0][key] == results[1][key]
    torch.testing.assert_close(initial_models[0], initial_models[2], rtol=0, atol=0)
    torch.testing.assert_close(initial_models[0], initial_models[4], rtol=0, atol=0)
