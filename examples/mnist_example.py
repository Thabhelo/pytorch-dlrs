"""
MNIST Example with DLRS Scheduler

Demonstrates using the DLRS scheduler on MNIST digit classification.
Compares training with DLRS vs fixed learning rate.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dlrs import DLRSScheduler
from dlrs.utils import _validate_scalar


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
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


def train_epoch(model, device, train_loader, optimizer, loss_fn, use_dlrs=False):
    """
    Train for one epoch.

    Returns:
        tuple: (average_loss, batch_losses) if use_dlrs, else just average_loss
    """
    model.train()
    mean_loss = 0.0
    total_samples = 0
    batch_losses = [] if use_dlrs else None

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        batch_loss = _validate_scalar(loss, "training loss", nonnegative=True)
        loss.backward()
        optimizer.step()

        if len(target) == 0:
            continue
        total_samples += len(target)
        mean_loss += (batch_loss - mean_loss) * (len(target) / total_samples)

        if use_dlrs:
            batch_losses.append(batch_loss)

    if total_samples == 0:
        raise ValueError("Training loader must contain at least one sample")
    avg_loss = mean_loss

    if use_dlrs:
        return avg_loss, batch_losses
    return avg_loss


def test(model, device, test_loader, loss_fn):
    """Evaluate observed samples using a mean-reduced loss and restore modes."""
    modes = [(module, module.training) for module in model.modules()]
    test_loss = 0.0
    correct = 0
    total_samples = 0
    try:
        model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                batch_loss = _validate_scalar(loss_fn(output, target), "test loss", nonnegative=True)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                if len(target) == 0:
                    continue
                total_samples += len(target)
                test_loss += (batch_loss - test_loss) * (len(target) / total_samples)
    finally:
        for module, training in modes:
            module.training = training
    if total_samples == 0:
        raise ValueError("Test loader must contain at least one sample")
    accuracy = 100.0 * correct / total_samples

    return test_loss, accuracy


def train_with_dlrs(args):
    """Train model using DLRS scheduler."""
    from torchvision import datasets, transforms

    torch.manual_seed(args.seed)
    print("\n" + "=" * 60)
    print("Training with DLRS Scheduler")
    print("=" * 60)

    device = torch.device(args.device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = DLRSScheduler(
        optimizer,
        delta_d=args.delta_d,
        delta_o=args.delta_o,
        delta_i=args.delta_i,
        max_lr=args.max_lr,
        verbose=args.verbose
    )
    loss_fn = nn.NLLLoss()

    results = {
        'train_loss': [],
        'test_loss': [],
        'test_accuracy': [],
        'learning_rates': []
    }

    for epoch in range(1, args.epochs + 1):
        train_loss, batch_losses = train_epoch(
            model, device, train_loader, optimizer, loss_fn, use_dlrs=True
        )

        scheduler.step(batch_losses)

        test_loss, test_accuracy = test(model, device, test_loader, loss_fn)

        current_lr = optimizer.param_groups[0]['lr']
        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)
        results['test_accuracy'].append(test_accuracy)
        results['learning_rates'].append(current_lr)

        print(f"Epoch {epoch:2d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Test Acc: {test_accuracy:.2f}% | "
              f"LR: {current_lr:.6f}")

    return results


def train_without_dlrs(args):
    """Train model with fixed learning rate (baseline)."""
    from torchvision import datasets, transforms

    torch.manual_seed(args.seed)
    print("\n" + "=" * 60)
    print("Training with Fixed Learning Rate (Baseline)")
    print("=" * 60)

    device = torch.device(args.device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.NLLLoss()

    results = {
        'train_loss': [],
        'test_loss': [],
        'test_accuracy': [],
        'learning_rates': []
    }

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, device, train_loader, optimizer, loss_fn, use_dlrs=False
        )

        test_loss, test_accuracy = test(model, device, test_loader, loss_fn)

        current_lr = optimizer.param_groups[0]['lr']
        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)
        results['test_accuracy'].append(test_accuracy)
        results['learning_rates'].append(current_lr)

        print(f"Epoch {epoch:2d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Test Acc: {test_accuracy:.2f}% | "
              f"LR: {current_lr:.6f}")

    return results


def compare_results(dlrs_results, baseline_results):
    """Print comparison between DLRS and baseline."""
    print("\n" + "=" * 60)
    print("Comparison: DLRS vs Baseline")
    print("=" * 60)

    dlrs_final_acc = dlrs_results['test_accuracy'][-1]
    baseline_final_acc = baseline_results['test_accuracy'][-1]

    dlrs_best_acc = max(dlrs_results['test_accuracy'])
    baseline_best_acc = max(baseline_results['test_accuracy'])

    dlrs_final_loss = dlrs_results['test_loss'][-1]
    baseline_final_loss = baseline_results['test_loss'][-1]

    print("\nFinal Test Accuracy:")
    print(f"  DLRS:     {dlrs_final_acc:.2f}%")
    print(f"  Baseline: {baseline_final_acc:.2f}%")
    print(f"  Difference: {dlrs_final_acc - baseline_final_acc:+.2f}%")

    print("\nBest Test Accuracy:")
    print(f"  DLRS:     {dlrs_best_acc:.2f}%")
    print(f"  Baseline: {baseline_best_acc:.2f}%")

    print("\nFinal Test Loss:")
    print(f"  DLRS:     {dlrs_final_loss:.4f}")
    print(f"  Baseline: {baseline_final_loss:.4f}")
    print(f"  Difference: {dlrs_final_loss - baseline_final_loss:+.4f}")

    print("\nLearning Rate (Final):")
    print(f"  DLRS:     {dlrs_results['learning_rates'][-1]:.6f}")
    print(f"  Baseline: {baseline_results['learning_rates'][-1]:.6f}")


def main():
    parser = argparse.ArgumentParser(description='MNIST Example with DLRS')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='training batch size (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        help='test batch size (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--max-lr', type=float, default=0.01,
                        help='explicit DLRS upper bound (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='shared initialization/shuffle seed for both runs (default: 0)')
    parser.add_argument('--delta-d', type=float, default=0.5,
                        help='DLRS delta_d parameter (default: 0.5)')
    parser.add_argument('--delta-o', type=float, default=1.0,
                        help='DLRS delta_o parameter (default: 1.0)')
    parser.add_argument('--delta-i', type=float, default=0.1,
                        help='DLRS delta_i parameter (default: 0.1)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='device to use (default: cuda if available, else cpu)')
    parser.add_argument('--no-baseline', action='store_true',
                        help='skip baseline training')
    parser.add_argument('--verbose', action='store_true',
                        help='verbose scheduler output')

    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.test_batch_size < 1:
        parser.error('epochs and batch sizes must be positive')

    print(f"Using device: {args.device}")
    print("Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Initial LR: {args.lr}")
    print(f"  DLRS parameters: delta_d={args.delta_d}, delta_o={args.delta_o}, delta_i={args.delta_i}")

    dlrs_results = train_with_dlrs(args)

    if not args.no_baseline:
        baseline_results = train_without_dlrs(args)
        compare_results(dlrs_results, baseline_results)
    else:
        print("\nSkipped baseline training (--no-baseline flag set)")

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)


if __name__ == '__main__':
    main()
