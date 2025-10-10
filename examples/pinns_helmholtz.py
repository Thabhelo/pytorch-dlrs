"""
PINNs Example: 1D Helmholtz Equation with DLRS

Solves the 1D Helmholtz equation using Physics-Informed Neural Networks:
    d²u/dx² + k²u = f(x)  for x in [0, 1]
    u(0) = 0, u(1) = 0  (Dirichlet boundary conditions)

With k=1 and f(x)=0, the exact solution is:
    u(x) = sin(πx)

This example demonstrates DLRS on a PINN task, where loss dynamics
can be highly non-monotonic during training.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from dlrs import DLRSScheduler


class HelmholtzPINN(nn.Module):
    """Physics-Informed Neural Network for 1D Helmholtz equation."""

    def __init__(self, hidden_layers: int = 3, hidden_dim: int = 32):
        super().__init__()
        layers = []
        layers.append(nn.Linear(1, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def exact_solution(x: np.ndarray) -> np.ndarray:
    """Exact solution for k=1, f=0: u(x) = sin(πx)."""
    return np.sin(np.pi * x)


def compute_pde_residual(
    model: nn.Module,
    x: torch.Tensor,
    k: float = 1.0,
    f: float = 0.0
) -> torch.Tensor:
    """
    Compute PDE residual: d²u/dx² + k²u - f.

    Uses automatic differentiation to compute derivatives.
    """
    x.requires_grad_(True)
    u = model(x)

    u_x = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x, x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True
    )[0]

    residual = u_xx + (k ** 2) * u - f
    return residual


def pinn_loss(
    model: nn.Module,
    x_interior: torch.Tensor,
    x_boundary: torch.Tensor,
    k: float = 1.0,
    f: float = 0.0,
    lambda_pde: float = 1.0,
    lambda_bc: float = 10.0
) -> torch.Tensor:
    """
    Compute total PINN loss: PDE residual + boundary condition loss.

    Parameters:
        model: Neural network
        x_interior: Interior collocation points
        x_boundary: Boundary points (x=0, x=1)
        k: Wave number in Helmholtz equation
        f: Forcing term
        lambda_pde: Weight for PDE residual loss
        lambda_bc: Weight for boundary condition loss

    Returns:
        Total loss
    """
    residual = compute_pde_residual(model, x_interior, k, f)
    loss_pde = torch.mean(residual ** 2)

    u_boundary = model(x_boundary)
    target_boundary = torch.zeros_like(u_boundary)
    loss_bc = torch.mean((u_boundary - target_boundary) ** 2)

    total_loss = lambda_pde * loss_pde + lambda_bc * loss_bc
    return total_loss


def train_pinn(
    model: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: DLRSScheduler,
    epochs: int,
    n_interior: int,
    n_boundary: int,
    batches_per_epoch: int,
    k: float = 1.0,
    verbose: bool = False
) -> dict:
    """Train PINN with DLRS scheduler."""
    results = {
        'loss': [],
        'learning_rates': [],
        'l2_error': []
    }

    x_boundary = torch.tensor([[0.0], [1.0]], device=device, dtype=torch.float32)

    for epoch in range(1, epochs + 1):
        batch_losses = []

        for _ in range(batches_per_epoch):
            x_interior = torch.rand(n_interior, 1, device=device, dtype=torch.float32)

            optimizer.zero_grad()
            loss = pinn_loss(model, x_interior, x_boundary, k=k)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        scheduler.step(batch_losses)

        mean_loss = np.mean(batch_losses)
        current_lr = optimizer.param_groups[0]['lr']

        x_test = torch.linspace(0, 1, 100, device=device).view(-1, 1)
        with torch.no_grad():
            u_pred = model(x_test).cpu().numpy().flatten()
        u_exact = exact_solution(x_test.cpu().numpy().flatten())
        l2_error = np.sqrt(np.mean((u_pred - u_exact) ** 2))

        results['loss'].append(mean_loss)
        results['learning_rates'].append(current_lr)
        results['l2_error'].append(l2_error)

        if verbose or epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | "
                  f"Loss: {mean_loss:.6f} | "
                  f"L2 Error: {l2_error:.6f} | "
                  f"LR: {current_lr:.6e}")

    return results


def plot_results(model: nn.Module, device: torch.device, results: dict, output_path: str = None):
    """Plot PINN solution and training curves."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed, skipping plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    x_test = torch.linspace(0, 1, 100, device=device).view(-1, 1)
    with torch.no_grad():
        u_pred = model(x_test).cpu().numpy().flatten()
    x_test_np = x_test.cpu().numpy().flatten()
    u_exact = exact_solution(x_test_np)

    axes[0, 0].plot(x_test_np, u_exact, 'b-', label='Exact', linewidth=2)
    axes[0, 0].plot(x_test_np, u_pred, 'r--', label='PINN', linewidth=2)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('u(x)')
    axes[0, 0].set_title('Solution: u(x) = sin(πx)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(x_test_np, np.abs(u_pred - u_exact), 'k-', linewidth=2)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('|u_pred - u_exact|')
    axes[0, 1].set_title('Pointwise Error')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)

    epochs = range(1, len(results['loss']) + 1)
    axes[1, 0].plot(epochs, results['loss'], 'b-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    ax_lr = axes[1, 1]
    ax_lr.plot(epochs, results['learning_rates'], 'g-', linewidth=2)
    ax_lr.set_xlabel('Epoch')
    ax_lr.set_ylabel('Learning Rate', color='g')
    ax_lr.tick_params(axis='y', labelcolor='g')
    ax_lr.set_yscale('log')
    ax_lr.grid(True, alpha=0.3)

    ax_l2 = ax_lr.twinx()
    ax_l2.plot(epochs, results['l2_error'], 'r-', linewidth=2, alpha=0.7)
    ax_l2.set_ylabel('L2 Error', color='r')
    ax_l2.tick_params(axis='y', labelcolor='r')
    ax_l2.set_yscale('log')

    axes[1, 1].set_title('Learning Rate & L2 Error')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='PINN for 1D Helmholtz Equation with DLRS')
    parser.add_argument('--epochs', type=int, default=500,
                       help='number of training epochs (default: 500)')
    parser.add_argument('--batches-per-epoch', type=int, default=10,
                       help='number of batches per epoch (default: 10)')
    parser.add_argument('--n-interior', type=int, default=100,
                       help='number of interior collocation points per batch (default: 100)')
    parser.add_argument('--n-boundary', type=int, default=2,
                       help='number of boundary points (default: 2)')
    parser.add_argument('--hidden-layers', type=int, default=3,
                       help='number of hidden layers (default: 3)')
    parser.add_argument('--hidden-dim', type=int, default=32,
                       help='hidden layer dimension (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='initial learning rate (default: 1e-3)')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       choices=['cpu', 'cuda', 'mps'],
                       help='device to use (default: cuda if available, else cpu)')
    parser.add_argument('--delta-d', type=float, default=0.5,
                       help='DLRS delta_d parameter (default: 0.5)')
    parser.add_argument('--delta-o', type=float, default=1.0,
                       help='DLRS delta_o parameter (default: 1.0)')
    parser.add_argument('--delta-i', type=float, default=0.1,
                       help='DLRS delta_i parameter (default: 0.1)')
    parser.add_argument('--verbose', action='store_true',
                       help='print loss every epoch')
    parser.add_argument('--plot', action='store_true',
                       help='plot results after training')
    parser.add_argument('--output', type=str, default=None,
                       help='output path for plot (default: show plot)')

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"\nConfiguration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batches per epoch: {args.batches_per_epoch}")
    print(f"  Interior points per batch: {args.n_interior}")
    print(f"  Network: {args.hidden_layers} layers x {args.hidden_dim} units")
    print(f"  Initial LR: {args.lr}")
    print(f"  DLRS: delta_d={args.delta_d}, delta_o={args.delta_o}, delta_i={args.delta_i}")

    model = HelmholtzPINN(hidden_layers=args.hidden_layers, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = DLRSScheduler(
        optimizer,
        delta_d=args.delta_d,
        delta_o=args.delta_o,
        delta_i=args.delta_i,
        min_lr=1e-8,
        verbose=False
    )

    print("\n" + "=" * 60)
    print("Training PINN for 1D Helmholtz Equation")
    print("=" * 60)

    results = train_pinn(
        model=model,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        n_interior=args.n_interior,
        n_boundary=args.n_boundary,
        batches_per_epoch=args.batches_per_epoch,
        verbose=args.verbose
    )

    final_l2_error = results['l2_error'][-1]
    print("\n" + "=" * 60)
    print(f"Training Complete")
    print(f"Final L2 Error: {final_l2_error:.6f}")
    print("=" * 60)

    if args.plot:
        plot_results(model, device, results, args.output)


if __name__ == '__main__':
    main()
