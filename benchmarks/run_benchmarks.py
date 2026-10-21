"""
Benchmark script comparing DLRS against standard PyTorch schedulers.

Runs paired-seed exploratory experiments on MNIST and CIFAR-10, measuring
validation trajectories, final test accuracy and wall time vs:
- No scheduler (fixed LR)
- StepLR
- ExponentialLR
- CosineAnnealingLR
- ReduceLROnPlateau
- AdamW with CosineAnnealingLR (optimizer baseline)
"""

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

from dlrs import DLRSScheduler


class SimpleCNN(nn.Module):
    """Simple CNN for image classification."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MNISTCNN(nn.Module):
    """CNN for MNIST."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def _weighted_mean_update(mean: float, count: int, value: float, batch_size: int) -> float:
    """Combine finite batch means without forming an overflowing weighted sum."""
    scale = max(abs(mean), abs(value))
    if scale == 0:
        return 0.0
    total = count + batch_size
    normalized = math.fsum(
        [
            (mean / scale) * (count / total),
            (value / scale) * (batch_size / total),
        ]
    )
    # A convex combination lies in [-1, 1]; roundoff must not overflow scale.
    return scale * max(-1.0, min(1.0, normalized))


def train_epoch(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    collect_losses: bool = False,
) -> Tuple[float, List[float]]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    batch_losses = []

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        batch_loss = loss.item()
        if not math.isfinite(batch_loss):
            raise ValueError("Non-finite training loss; benchmark failed before optimizer update")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), math.inf, error_if_nonfinite=True)
        optimizer.step()

        total_loss = _weighted_mean_update(total_loss, total_samples, batch_loss, target.size(0))
        total_samples += target.size(0)
        if collect_losses:
            batch_losses.append(batch_loss)

    if total_samples == 0:
        raise ValueError("Training loader is empty")
    avg_loss = total_loss
    return avg_loss, batch_losses


def evaluate(
    model: nn.Module, device: torch.device, test_loader: DataLoader, criterion: nn.Module
) -> Tuple[float, float]:
    """Evaluate model on test set."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            batch_loss = criterion(output, target).item()
            if not math.isfinite(batch_loss):
                raise ValueError("Non-finite evaluation loss; benchmark failed")
            test_loss = _weighted_mean_update(test_loss, total_samples, batch_loss, target.size(0))
            total_samples += target.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    if total_samples == 0:
        raise ValueError("Evaluation loader is empty")
    accuracy = 100.0 * correct / total_samples

    return test_loss, accuracy


def run_experiment(
    dataset_name: str,
    scheduler_name: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int = 0,
    num_workers: int = 0,
    validation_fraction: float = 0.1,
    dataset_root: str = "./data",
    split_seed: int = None,
    evaluate_test: bool = True,
    max_lr: float = None,
    download: bool = True,
) -> Dict:
    """Run one paired-seed trial; the test set never controls the scheduler."""
    from torchvision import datasets, transforms

    if epochs <= 0 or batch_size <= 0 or num_workers < 0:
        raise ValueError("epochs/batch_size must be positive and num_workers nonnegative")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"\nRunning {scheduler_name} on {dataset_name}...")

    if dataset_name == "MNIST":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_dataset = datasets.MNIST(dataset_root, train=True, download=download, transform=transform)
        validation_dataset = datasets.MNIST(dataset_root, train=True, transform=transform)
        test_dataset = (
            datasets.MNIST(dataset_root, train=False, transform=transform)
            if evaluate_test
            else None
        )
        model = MNISTCNN().to(device)
        criterion = nn.NLLLoss()
    elif dataset_name == "CIFAR10":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        train_dataset = datasets.CIFAR10(
            dataset_root, train=True, download=download, transform=transform_train
        )
        validation_dataset = datasets.CIFAR10(dataset_root, train=True, transform=transform_test)
        test_dataset = (
            datasets.CIFAR10(dataset_root, train=False, transform=transform_test)
            if evaluate_test
            else None
        )
        model = SimpleCNN(num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    indices = torch.randperm(
        len(train_dataset),
        generator=torch.Generator().manual_seed(seed if split_seed is None else split_seed),
    ).tolist()
    validation_size = max(1, int(len(indices) * validation_fraction))
    if validation_size >= len(indices):
        raise ValueError("Need at least one training and one validation example")
    train_dataset = Subset(train_dataset, indices[validation_size:])
    validation_dataset = Subset(validation_dataset, indices[:validation_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size * 2,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=batch_size * 2,
            num_workers=num_workers,
            generator=torch.Generator().manual_seed(seed + 2),
        )
        if test_dataset is not None
        else None
    )

    if scheduler_name == "AdamW+Cosine":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

        if scheduler_name == "NoScheduler":
            scheduler = None
        elif scheduler_name == "DLRS":
            scheduler = DLRSScheduler(
                optimizer, delta_d=0.5, delta_o=1.0, delta_i=0.1, max_lr=max_lr
            )
        elif scheduler_name == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=max(epochs // 3, 1), gamma=0.1
            )
        elif scheduler_name == "ExponentialLR":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif scheduler_name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_name == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=5
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

    results = {
        "scheduler": scheduler_name,
        "dataset": dataset_name,
        "seed": seed,
        "environment": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "device": str(device),
            "platform": platform.platform(),
        },
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "initial_lr": lr,
            "validation_fraction": validation_fraction,
            "num_workers": num_workers,
            "split_seed": seed if split_seed is None else split_seed,
            "max_lr": max_lr,
            "train_examples": len(train_dataset),
            "validation_examples": len(validation_dataset),
            "test_evaluated": evaluate_test,
        },
        "initial_model_sha256": hashlib.sha256(
            b"".join(parameter.detach().cpu().numpy().tobytes() for parameter in model.parameters())
        ).hexdigest(),
        "split_sha256": hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
        "train_loss": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "learning_rates": [],
        "next_learning_rates": [],
        "epoch_times": [],
    }

    synchronize(device)
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        synchronize(device)
        epoch_start = time.perf_counter()
        current_lr = optimizer.param_groups[0]["lr"]

        collect_losses = scheduler_name == "DLRS"
        train_loss, batch_losses = train_epoch(
            model, device, train_loader, optimizer, criterion, collect_losses
        )
        validation_loss, validation_accuracy = evaluate(model, device, validation_loader, criterion)

        if scheduler is not None:
            if scheduler_name == "DLRS":
                scheduler.step(batch_losses)
            elif scheduler_name == "ReduceLROnPlateau":
                scheduler.step(validation_loss)
            else:
                scheduler.step()
        synchronize(device)
        epoch_time = time.perf_counter() - epoch_start
        results["train_loss"].append(train_loss)
        results["validation_loss"].append(validation_loss)
        results["validation_accuracy"].append(validation_accuracy)
        results["learning_rates"].append(current_lr)
        results["next_learning_rates"].append(optimizer.param_groups[0]["lr"])
        results["epoch_times"].append(epoch_time)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:2d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {validation_loss:.4f} | "
                f"Val Acc: {validation_accuracy:.2f}% | "
                f"LR: {current_lr:.6f} | "
                f"Time: {epoch_time:.2f}s"
            )

    if test_loader is not None:
        results["test_loss"], results["final_accuracy"] = evaluate(
            model, device, test_loader, criterion
        )
    synchronize(device)
    total_time = time.perf_counter() - start_time
    results["total_time"] = total_time
    results["best_validation_accuracy"] = max(results["validation_accuracy"])

    print(
        f"Finished {scheduler_name}: "
        f"Final Acc: {results.get('final_accuracy', 'held out')}, "
        f"Best Val Acc: {results['best_validation_accuracy']:.2f}%, "
        f"Total Time: {total_time:.1f}s"
    )

    return results


def synchronize(device: torch.device) -> None:
    """Include outstanding accelerator work in measured wall time."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def generate_report(all_results: List[Dict], output_path: Path) -> None:
    """Generate markdown report from benchmark results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# DLRS Benchmark Results\n\n")
        f.write("Comparison of DLRS against standard PyTorch schedulers on MNIST and CIFAR-10.\n\n")
        f.write(
            "Exploratory paired-seed trials, not a non-regression or large-scale reliability certification. "
            "Schedulers share seeds, splits, initial model weights and batch order. "
            "Only validation data controls ReduceLROnPlateau; test accuracy is measured once at the end. "
            "AdamW+Cosine also changes the optimizer, and is not a scheduler-only comparison. "
            "Timing includes training, validation, and final test evaluation; no warmup adjustment.\n\n"
        )

        for dataset_name in ["MNIST", "CIFAR10"]:
            dataset_results = [r for r in all_results if r["dataset"] == dataset_name]
            if not dataset_results:
                continue

            f.write(f"## {dataset_name}\n\n")
            f.write(
                "| Scheduler | Seed | Final Test Acc (%) | Best Val Acc (%) | Total Time (s) |\n"
            )
            f.write(
                "|-----------|------|--------------------|------------------|----------------|\n"
            )

            for result in dataset_results:
                f.write(
                    f"| {result['scheduler']:<20} | "
                    f"{result['seed']} | "
                    f"{result['final_accuracy']:>6.2f} | "
                    f"{result['best_validation_accuracy']:>6.2f} | "
                    f"{result['total_time']:>8.1f} |\n"
                )

            f.write("\n")
            f.write(
                "| Scheduler | Trials | Mean Test Acc (%) | Sample SD |\n|---|---:|---:|---:|\n"
            )
            for name in dict.fromkeys(r["scheduler"] for r in dataset_results):
                values = [r["final_accuracy"] for r in dataset_results if r["scheduler"] == name]
                spread = f"{statistics.stdev(values):.3f}" if len(values) > 1 else "not estimated"
                f.write(f"| {name} | {len(values)} | {statistics.mean(values):.3f} | {spread} |\n")
            f.write("\n")

    print(f"\nBenchmark report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run DLRS benchmarks")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to use (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs per experiment (default: 10)"
    )
    parser.add_argument("--batch-size", type=int, default=128, help="batch size (default: 128)")
    parser.add_argument(
        "--lr", type=float, default=0.1, help="initial learning rate (default: 0.1)"
    )
    parser.add_argument(
        "--quick", action="store_true", help="run quick benchmark (MNIST only, 5 epochs)"
    )
    parser.add_argument(
        "--output", type=str, default="docs/BENCHMARKS.md", help="output path for markdown report"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="paired trial seeds (default: 0 1 2)",
    )
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    if args.quick:
        datasets_to_test = ["MNIST"]
        epochs = 5
        print("Running quick benchmark (MNIST, 5 epochs)")
    else:
        datasets_to_test = ["MNIST", "CIFAR10"]
        epochs = args.epochs
        print(f"Running full benchmark ({epochs} epochs per experiment)")

    schedulers_to_test = [
        "NoScheduler",
        "DLRS",
        "StepLR",
        "ExponentialLR",
        "CosineAnnealingLR",
        "ReduceLROnPlateau",
        "AdamW+Cosine",
    ]

    all_results = []
    failures = []

    for dataset_name in datasets_to_test:
        for seed in args.seeds:
            for scheduler_name in schedulers_to_test:
                try:
                    result = run_experiment(
                        dataset_name=dataset_name,
                        scheduler_name=scheduler_name,
                        device=device,
                        epochs=epochs,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        seed=seed,
                        num_workers=args.num_workers,
                    )
                    all_results.append(result)
                except Exception as e:
                    failures.append(
                        {
                            "scheduler": scheduler_name,
                            "dataset": dataset_name,
                            "seed": seed,
                            "error": str(e),
                        }
                    )
                    print(f"ERROR: Failed to run {scheduler_name} on {dataset_name}: {e}")

    output_path = Path(args.output)
    generate_report(all_results, output_path)

    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump({"results": all_results, "failures": failures}, f, indent=2, allow_nan=False)
    print(f"Raw results saved to {json_path}")

    print("\n" + "=" * 60)
    if failures:
        with output_path.open("a") as f:
            f.write(f"\nINCOMPLETE: {len(failures)} trials failed. See JSON failure records.\n")
        raise SystemExit(f"Benchmark incomplete: {len(failures)} trials failed")
    print("Benchmark Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
