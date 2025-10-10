"""
Benchmark script comparing DLRS against standard PyTorch schedulers.

Runs quick experiments on MNIST and CIFAR-10 to measure time-to-accuracy
and final accuracy vs:
- No scheduler (fixed LR)
- StepLR
- ExponentialLR
- CosineAnnealingLR
- ReduceLROnPlateau
- AdamW with CosineAnnealingLR (optimizer baseline)
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
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


def train_epoch(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    collect_losses: bool = False
) -> Tuple[float, List[float]]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    batch_losses = []

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        if collect_losses:
            batch_losses.append(batch_loss)

    avg_loss = total_loss / len(train_loader)
    return avg_loss, batch_losses


def evaluate(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
    criterion: nn.Module
) -> Tuple[float, float]:
    """Evaluate model on test set."""
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    return test_loss, accuracy


def run_experiment(
    dataset_name: str,
    scheduler_name: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float
) -> Dict:
    """Run single experiment with specified scheduler."""
    print(f"\nRunning {scheduler_name} on {dataset_name}...")

    if dataset_name == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        model = MNISTCNN().to(device)
        criterion = nn.NLLLoss()
    elif dataset_name == "CIFAR10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)
        model = SimpleCNN(num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=2)

    if scheduler_name == "AdamW+Cosine":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

        if scheduler_name == "NoScheduler":
            scheduler = None
        elif scheduler_name == "DLRS":
            scheduler = DLRSScheduler(optimizer, delta_d=0.5, delta_o=1.0, delta_i=0.1)
        elif scheduler_name == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=0.1)
        elif scheduler_name == "ExponentialLR":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif scheduler_name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_name == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

    results = {
        'scheduler': scheduler_name,
        'dataset': dataset_name,
        'train_loss': [],
        'test_loss': [],
        'test_accuracy': [],
        'learning_rates': [],
        'epoch_times': []
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        collect_losses = (scheduler_name == "DLRS")
        train_loss, batch_losses = train_epoch(model, device, train_loader, optimizer, criterion, collect_losses)

        if scheduler is not None:
            if scheduler_name == "DLRS":
                scheduler.step(batch_losses)
            elif scheduler_name == "ReduceLROnPlateau":
                test_loss, test_acc = evaluate(model, device, test_loader, criterion)
                scheduler.step(test_loss)
            else:
                scheduler.step()
        else:
            test_loss, test_acc = evaluate(model, device, test_loader, criterion)

        if scheduler_name != "ReduceLROnPlateau":
            test_loss, test_acc = evaluate(model, device, test_loader, criterion)

        epoch_time = time.time() - epoch_start

        current_lr = optimizer.param_groups[0]['lr']
        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)
        results['test_accuracy'].append(test_acc)
        results['learning_rates'].append(current_lr)
        results['epoch_times'].append(epoch_time)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | "
                  f"Test Acc: {test_acc:.2f}% | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {epoch_time:.2f}s")

    total_time = time.time() - start_time
    results['total_time'] = total_time
    results['final_accuracy'] = results['test_accuracy'][-1]
    results['best_accuracy'] = max(results['test_accuracy'])

    print(f"Finished {scheduler_name}: "
          f"Final Acc: {results['final_accuracy']:.2f}%, "
          f"Best Acc: {results['best_accuracy']:.2f}%, "
          f"Total Time: {total_time:.1f}s")

    return results


def generate_report(all_results: List[Dict], output_path: Path) -> None:
    """Generate markdown report from benchmark results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("# DLRS Benchmark Results\n\n")
        f.write("Comparison of DLRS against standard PyTorch schedulers on MNIST and CIFAR-10.\n\n")

        for dataset_name in ["MNIST", "CIFAR10"]:
            dataset_results = [r for r in all_results if r['dataset'] == dataset_name]
            if not dataset_results:
                continue

            f.write(f"## {dataset_name}\n\n")
            f.write("| Scheduler | Final Acc (%) | Best Acc (%) | Total Time (s) |\n")
            f.write("|-----------|---------------|--------------|----------------|\n")

            for result in dataset_results:
                f.write(f"| {result['scheduler']:<20} | "
                       f"{result['final_accuracy']:>6.2f} | "
                       f"{result['best_accuracy']:>6.2f} | "
                       f"{result['total_time']:>8.1f} |\n")

            f.write("\n")

    print(f"\nBenchmark report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Run DLRS benchmarks')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='device to use (default: cuda if available, else cpu)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='number of epochs per experiment (default: 10)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='initial learning rate (default: 0.1)')
    parser.add_argument('--quick', action='store_true',
                       help='run quick benchmark (MNIST only, 5 epochs)')
    parser.add_argument('--output', type=str, default='docs/BENCHMARKS.md',
                       help='output path for markdown report')

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
        "AdamW+Cosine"
    ]

    all_results = []

    for dataset_name in datasets_to_test:
        for scheduler_name in schedulers_to_test:
            try:
                result = run_experiment(
                    dataset_name=dataset_name,
                    scheduler_name=scheduler_name,
                    device=device,
                    epochs=epochs,
                    batch_size=args.batch_size,
                    lr=args.lr
                )
                all_results.append(result)
            except Exception as e:
                print(f"ERROR: Failed to run {scheduler_name} on {dataset_name}: {e}")
                continue

    output_path = Path(args.output)
    generate_report(all_results, output_path)

    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Raw results saved to {json_path}")

    print("\n" + "=" * 60)
    print("Benchmark Complete")
    print("=" * 60)


if __name__ == '__main__':
    main()
