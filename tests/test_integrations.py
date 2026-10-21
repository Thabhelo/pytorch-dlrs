"""Contract tests for examples; Lightning doubles are NOT framework validation."""

import copy
import importlib.util
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dlrs import DLRSScheduler
from examples.huggingface_callback import DLRSTrainerCallback
from examples.mnist_example import test as evaluate_mnist
from examples.mnist_example import train_epoch
from examples.pinns_helmholtz import (
    HelmholtzPINN,
    compute_pde_residual,
    pinn_loss,
    train_pinn,
)


@pytest.fixture
def lightning_callback(monkeypatch):
    """Isolate callback logic from optional framework/dependency availability."""
    lightning = types.ModuleType("pytorch_lightning")
    lightning.Trainer = type("Trainer", (), {})
    lightning.LightningModule = type("LightningModule", (), {})
    callbacks = types.ModuleType("pytorch_lightning.callbacks")
    callbacks.Callback = type("Callback", (), {})
    monkeypatch.setitem(sys.modules, "pytorch_lightning", lightning)
    monkeypatch.setitem(sys.modules, "pytorch_lightning.callbacks", callbacks)
    spec = importlib.util.spec_from_file_location(
        "_dlrs_lightning_example_test", Path(__file__).parents[1] / "examples/lightning_callback.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DLRSLightningCallback


def trainer_for(callback, optimizer=None, **overrides):
    if optimizer is None:
        optimizer = torch.optim.SGD([nn.Parameter(torch.ones(1))], lr=0.01)
    fields = dict(
        optimizers=[optimizer], world_size=1, lr_scheduler_configs=[],
        accumulate_grad_batches=1, precision="32-true", callbacks=[callback],
        logger=None, global_step=0, current_epoch=0,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize("fields,model_fields", [
    ({"world_size": 2}, {}),
    ({"optimizers": []}, {}),
    ({"optimizers": [object(), object()]}, {}),
    ({"lr_scheduler_configs": [object()]}, {}),
    ({"accumulate_grad_batches": 2}, {}),
    ({"precision": "16-mixed"}, {}),
    ({}, {"automatic_optimization": False}),
])
def test_lightning_rejects_unsupported_training_before_scheduler_creation(
    lightning_callback, fields, model_fields,
):
    callback = lightning_callback()
    model = SimpleNamespace(automatic_optimization=True)
    model.__dict__.update(model_fields)
    trainer = trainer_for(callback, **fields)
    with pytest.raises(RuntimeError):
        callback.on_train_start(trainer, model)
    assert callback.scheduler is None


def test_lightning_rejects_duplicate_lr_owners(lightning_callback):
    callback = lightning_callback()
    trainer = trainer_for(callback, callbacks=[callback, lightning_callback()])
    with pytest.raises(RuntimeError, match="Only one"):
        callback.on_train_start(trainer, SimpleNamespace(automatic_optimization=True))


def test_lightning_collects_actual_losses_and_updates_once(lightning_callback):
    callback = lightning_callback()
    trainer = trainer_for(callback)
    model = SimpleNamespace(automatic_optimization=True)
    callback.on_train_start(trainer, model)
    callback.on_train_batch_end(trainer, model, torch.tensor(2.0), None, 0)
    callback.on_train_batch_end(trainer, model, {"loss": torch.tensor(1.0)}, None, 1)
    callback.on_train_epoch_end(trainer, model)
    assert callback.scheduler.loss_slope == pytest.approx(-2 / 3)
    lr = trainer.optimizers[0].param_groups[0]["lr"]
    assert lr > 0.01
    assert callback.batch_losses == []
    with pytest.warns(UserWarning, match="Fewer than 2"):
        callback.on_train_epoch_end(trainer, model)
    assert trainer.optimizers[0].param_groups[0]["lr"] == lr


def test_lightning_checkpoint_before_start_matches_continuation(lightning_callback):
    callback = lightning_callback()
    trainer = trainer_for(callback)
    model = SimpleNamespace(automatic_optimization=True)
    callback.on_train_start(trainer, model)
    for i, value in enumerate([2.0, 1.0]):
        callback.on_train_batch_end(trainer, model, {"loss": value}, None, i)
    callback.on_train_epoch_end(trainer, model)
    callback.on_train_batch_end(trainer, model, {"loss": 4.0}, None, 0)
    state = callback.state_dict()
    optimizer_state = copy.deepcopy(trainer.optimizers[0].state_dict())
    restored = lightning_callback()
    restored.load_state_dict(state)
    state["batch_losses"].append(999)  # caller-owned mutable state must be isolated
    restored_trainer = trainer_for(restored)
    restored_trainer.optimizers[0].load_state_dict(optimizer_state)
    restored.on_train_start(restored_trainer, model)
    assert restored.batch_losses == [4.0]
    for item, item_trainer in [(callback, trainer), (restored, restored_trainer)]:
        item.on_train_batch_end(item_trainer, model, {"loss": 2.0}, None, 1)
        item.on_train_epoch_end(item_trainer, model)
    assert restored.scheduler.state_dict() == callback.scheduler.state_dict()
    assert restored_trainer.optimizers[0].param_groups[0]["lr"] == trainer.optimizers[0].param_groups[0]["lr"]


@pytest.mark.parametrize("output", [{"other": 1.0}, {"loss": float("nan")}, {"loss": -1.0}, {"loss": torch.ones(2)}])
def test_lightning_rejects_invalid_loss_without_partial_append(lightning_callback, output):
    callback = lightning_callback()
    trainer = trainer_for(callback)
    model = SimpleNamespace(automatic_optimization=True)
    callback.on_train_start(trainer, model)
    with pytest.raises(ValueError):
        callback.on_train_batch_end(trainer, model, output, None, 0)
    assert callback.batch_losses == []


def test_lightning_loss_limit_never_truncates_and_checkpoint_is_independent(lightning_callback):
    callback = lightning_callback(max_batches=2)
    trainer = trainer_for(callback)
    model = SimpleNamespace(automatic_optimization=True)
    callback.on_train_start(trainer, model)
    for i in range(2):
        callback.on_train_batch_end(trainer, model, {"loss": float(i)}, None, i)
    state = callback.state_dict()
    state["batch_losses"].clear()
    with pytest.raises(RuntimeError, match="max_batches"):
        callback.on_train_batch_end(trainer, model, {"loss": 2.0}, None, 2)
    assert callback.batch_losses == [0.0, 1.0]


def test_huggingface_unsafe_callback_fails_closed():
    with pytest.raises(NotImplementedError, match="raw per-batch losses"):
        DLRSTrainerCallback()


def test_pinn_analytic_solution_satisfies_pde_and_boundary():
    class Analytic(nn.Module):
        def forward(self, x):
            return torch.sin(math.pi * x)

    x = torch.linspace(0, 1, 23, dtype=torch.float64).reshape(-1, 1)
    boundary = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    for k in [1.0, 2.0]:
        residual = compute_pde_residual(Analytic(), x, k=k)
        assert residual.abs().max().item() < 1e-12
        assert pinn_loss(Analytic(), x, boundary, k=k).item() < 1e-24
    assert not x.requires_grad
    assert x.grad is None


def test_pinn_derivatives_preserve_parameter_gradients_without_mutating_coordinates():
    model = HelmholtzPINN(hidden_layers=2, hidden_dim=4).double()
    x = torch.rand(8, 1, dtype=torch.float64)
    loss = compute_pde_residual(model, x).square().mean()
    loss.backward()
    assert not x.requires_grad and x.grad is None
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert any(parameter.grad.abs().sum().item() > 0 for parameter in model.parameters())


def test_pinn_linear_model_has_zero_second_derivative():
    model = nn.Linear(1, 1, bias=True).double()
    x = torch.tensor([[0.2], [0.4]], dtype=torch.float64)
    residual = compute_pde_residual(model, x, k=2.0, f=0.0)
    assert torch.allclose(residual, 4.0 * model(x))
    residual.sum().backward()
    assert model.weight.grad is not None


def test_pinn_small_training_smoke():
    torch.manual_seed(1)
    model = HelmholtzPINN(hidden_layers=1, hidden_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = DLRSScheduler(optimizer, max_lr=0.01)
    results = train_pinn(model, torch.device("cpu"), optimizer, scheduler, 2, 8, 2, 2)
    assert len(results["loss"]) == 2
    assert all(math.isfinite(value) for values in results.values() for value in values)
    with pytest.raises(ValueError, match="exactly two"):
        train_pinn(model, torch.device("cpu"), optimizer, scheduler, 2, 8, 3, 2)


def test_mnist_evaluation_weights_samples_and_restores_mode():
    model = nn.Sequential(nn.Identity(), nn.Dropout())
    model.train()
    model[1].eval()
    data = torch.tensor([[0.0, -1.0], [0.0, -1.0], [-3.0, 0.0]])
    targets = torch.zeros(3, dtype=torch.long)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data, targets), batch_size=2)
    loss, accuracy = evaluate_mnist(model, "cpu", loader, nn.NLLLoss())
    assert loss == 1.0  # unequal final batch must not receive equal batch weight
    assert accuracy == pytest.approx(200 / 3)
    assert model.training and not model[1].training


def test_mnist_train_loss_weights_samples():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    loader = [(torch.tensor([[1.0], [1.0]]), torch.zeros(2, 1)),
              (torch.tensor([[3.0]]), torch.zeros(1, 1))]
    loss, batch_losses = train_epoch(model, "cpu", loader, optimizer, nn.MSELoss(), use_dlrs=True)
    assert loss == pytest.approx(11 / 3)
    assert batch_losses == [1.0, 9.0]


def test_lightning_dynamic_accumulation_change_is_rejected(lightning_callback):
    callback = lightning_callback()
    trainer = trainer_for(callback)
    model = SimpleNamespace(automatic_optimization=True)
    callback.on_train_start(trainer, model)
    trainer.accumulate_grad_batches = 2
    with pytest.raises(RuntimeError, match="gradient accumulation"):
        callback.on_train_batch_end(trainer, model, {"loss": 1.0}, None, 0)
    assert callback.batch_losses == []
    assert trainer.optimizers[0].param_groups[0]["lr"] == 0.01


def test_lightning_checkpoint_rejects_changed_metric(lightning_callback):
    callback = lightning_callback(monitor_key="loss")
    checkpoint = callback.state_dict()
    restored = lightning_callback(monitor_key="other_loss")
    with pytest.raises(ValueError, match="monitor_key"):
        restored.load_state_dict(checkpoint)


def test_real_lightning_epoch_checkpoint_resume_in_fresh_processes(tmp_path):
    """Optional real-framework CPU integration; no stochastic/data-parallel claim."""
    import subprocess

    pytest.importorskip("pytorch_lightning")
    script = r'''
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from examples.lightning_callback import DLRSLightningCallback

stage, root = sys.argv[1], Path(sys.argv[2])
torch.manual_seed(1234)
torch.set_num_threads(1)

class Regression(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        return nn.functional.mse_loss(self.layer(inputs), targets)
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

model = Regression()
callback = DLRSLightningCallback(max_lr=0.1)
trainer = pl.Trainer(
    accelerator="cpu", devices=1, max_epochs=1 if stage == "first" else 3,
    callbacks=[callback], logger=False, enable_progress_bar=False,
    enable_model_summary=False, default_root_dir=str(root / stage),
    deterministic=True,
)
x = torch.linspace(-1, 1, 16).reshape(-1, 1)
loader = DataLoader(TensorDataset(x, 2*x+0.5), batch_size=4, shuffle=False)
trainer.fit(model, loader, ckpt_path=str(root / "resume.ckpt") if stage == "resume" else None)
if stage == "first":
    trainer.save_checkpoint(root / "resume.ckpt")
torch.save({
    "weights": model.state_dict(),
    "lr": trainer.optimizers[0].param_groups[0]["lr"],
    "scheduler": callback.scheduler.state_dict(),
}, root / (stage + ".pt"))
print(json.dumps({"torch": torch.__version__, "lightning": pl.__version__, "stage": stage}))
'''
    for stage in ["full", "first", "resume"]:
        result = subprocess.run(
            [sys.executable, "-c", script, stage, str(tmp_path)],
            cwd=Path(__file__).parents[1], capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    full = torch.load(tmp_path / "full.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resume.pt", weights_only=True)
    assert full["lr"] == resumed["lr"]
    assert full["scheduler"] == resumed["scheduler"]
    assert full["weights"].keys() == resumed["weights"].keys()
    assert all(torch.equal(value, resumed["weights"][key]) for key, value in full["weights"].items())


def test_lightning_missing_scheduler_state_is_rejected_atomically(lightning_callback):
    callback = lightning_callback()
    trainer = trainer_for(callback)
    model = SimpleNamespace(automatic_optimization=True)
    callback.on_train_start(trainer, model)
    callback.on_train_batch_end(trainer, model, {"loss": 1.0}, None, 0)
    before = callback.state_dict()
    with pytest.raises(ValueError, match="missing scheduler"):
        callback.load_state_dict({"scheduler": None, "batch_losses": []})
    assert callback.state_dict() == before


def test_mnist_finite_large_sample_mean_does_not_overflow():
    loader = [(torch.tensor([[1.0, 0.0], [1.0, 0.0]]), torch.zeros(2, dtype=torch.long))]
    loss, accuracy = evaluate_mnist(
        nn.Identity(), "cpu", loader, lambda *_: torch.tensor(1e308, dtype=torch.float64)
    )
    assert loss == 1e308
    assert accuracy == 100.0
