"""Experimental single-process PyTorch Lightning integration.

Supported envelope: one optimizer, automatic optimization, full precision,
no gradient accumulation and no other learning-rate scheduler. Distributed and
mixed-precision training are deliberately rejected until validated against real
framework runs. Return a nonnegative scalar loss from ``training_step`` (or a
mapping containing ``monitor_key``). This callback must be the only owner of
learning-rate updates, including updates made by other callbacks or user code.

Callback checkpoints retain scheduler state and collected losses. Exact
mid-epoch continuation additionally requires the framework/data pipeline to
restore the batch position and RNG state; this callback cannot guarantee that.
"""

import copy
import warnings
from typing import Optional

import torch

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback
except ImportError as exc:
    raise ImportError(
        "PyTorch Lightning not installed. Install with: pip install pytorch-lightning"
    ) from exc

from dlrs import DLRSScheduler
from dlrs.utils import _validate_scalar, validate_hyperparameters


class DLRSLightningCallback(Callback):
    """Apply DLRS once per epoch within the supported envelope above.

    ``max_batches`` caps retained epoch losses and raises before truncating any
    data. It also bounds checkpoint size. This is an experimental example, not
    a distributed or production-validated integration.
    """

    def __init__(
        self,
        delta_d: float = 0.5,
        delta_o: float = 1.0,
        delta_i: float = 0.1,
        min_lr: float = 1e-8,
        max_lr: Optional[float] = None,
        verbose: bool = False,
        monitor_key: str = "loss",
        max_batches: int = 1_000_000,
    ):
        super().__init__()
        validate_hyperparameters(delta_d, delta_o, delta_i)
        if not isinstance(monitor_key, str) or not monitor_key:
            raise ValueError("monitor_key must be a nonempty string")
        if isinstance(max_batches, bool) or not isinstance(max_batches, int) or max_batches < 2:
            raise ValueError("max_batches must be an integer >= 2")
        self.delta_d = delta_d
        self.delta_o = delta_o
        self.delta_i = delta_i
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.verbose = verbose
        self.monitor_key = monitor_key
        self.max_batches = max_batches
        self.scheduler = None
        self.batch_losses = []
        self._pending_scheduler_state = None

    def _validate_training(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Check dynamic settings too, e.g. a scheduled accumulation change."""
        if trainer.world_size != 1:
            raise RuntimeError("DLRS Lightning example supports single-process training only")
        if len(trainer.optimizers) != 1:
            raise RuntimeError("DLRS Lightning example requires exactly one optimizer")
        if trainer.lr_scheduler_configs:
            raise RuntimeError("DLRS must be the only learning-rate scheduler")
        if trainer.accumulate_grad_batches != 1:
            raise RuntimeError("DLRS Lightning example does not support gradient accumulation")
        if not pl_module.automatic_optimization:
            raise RuntimeError("DLRS Lightning example requires automatic optimization")
        if trainer.precision not in (32, 64, "32-true", "64-true"):
            raise RuntimeError("DLRS Lightning example supports full precision only")
        if sum(isinstance(callback, DLRSLightningCallback) for callback in trainer.callbacks) > 1:
            raise RuntimeError("Only one DLRS callback may own the optimizer learning rate")

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Validate ownership before creating or restoring a scheduler."""
        self._validate_training(trainer, pl_module)
        optimizer = trainer.optimizers[0]
        if self.scheduler is not None:
            if self.scheduler.optimizer is not optimizer:
                raise RuntimeError("Create a new DLRS callback for a different optimizer")
            return
        scheduler = DLRSScheduler(
            optimizer,
            delta_d=self.delta_d,
            delta_o=self.delta_o,
            delta_i=self.delta_i,
            min_lr=self.min_lr,
            max_lr=self.max_lr,
            verbose=self.verbose,
        )
        if self._pending_scheduler_state is not None:
            scheduler.load_state_dict(self._pending_scheduler_state)
        self.scheduler = scheduler
        self._pending_scheduler_state = None

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule,
        outputs, batch, batch_idx: int,
    ) -> None:
        """Record each observed batch once; never substitute aggregated logs."""
        if self.scheduler is None:
            raise RuntimeError("DLRS callback has not been initialized")
        self._validate_training(trainer, pl_module)
        if outputs is None:  # Lightning permits explicitly skipped training steps.
            return
        if isinstance(outputs, torch.Tensor):
            if self.monitor_key != "loss":
                raise ValueError("A custom monitor_key requires a mapping training_step output")
            loss = outputs
        elif isinstance(outputs, dict) and self.monitor_key in outputs:
            loss = outputs[self.monitor_key]
        else:
            raise ValueError(f"training_step output must contain {self.monitor_key!r}")
        value = _validate_scalar(loss, self.monitor_key, nonnegative=True)
        if len(self.batch_losses) >= self.max_batches:
            raise RuntimeError("max_batches exceeded; increase the explicit epoch-loss limit")
        self.batch_losses.append(value)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Update once, then clear the consumed losses before optional logging."""
        if self.scheduler is None:
            raise RuntimeError("DLRS callback has not been initialized")
        self._validate_training(trainer, pl_module)
        if len(self.batch_losses) < 2:
            warnings.warn("Fewer than 2 batch losses; skipping DLRS epoch update", stacklevel=2)
            self.batch_losses = []
            return
        self.scheduler.step(self.batch_losses)
        self.batch_losses = []
        if trainer.logger is not None:
            trainer.logger.log_metrics(
                {"learning_rate": trainer.optimizers[0].param_groups[0]["lr"],
                 "dlrs_loss_slope": self.scheduler.loss_slope},
                step=trainer.global_step,
            )

    def state_dict(self):
        """Return independent state, including a restore pending initialization."""
        state = (
            self.scheduler.state_dict() if self.scheduler is not None
            else self._pending_scheduler_state
        )
        return copy.deepcopy({
            "scheduler": state, "batch_losses": self.batch_losses, "monitor_key": self.monitor_key,
        })

    def load_state_dict(self, state_dict):
        """Defer scheduler restoration until Lightning has restored its optimizer."""
        if not isinstance(state_dict, dict) or set(state_dict) not in (
            {"scheduler", "batch_losses"}, {"scheduler", "batch_losses", "monitor_key"},
        ):
            raise ValueError("Invalid DLRS callback checkpoint")
        if state_dict.get("monitor_key", self.monitor_key) != self.monitor_key:
            raise ValueError("Checkpoint monitor_key differs from this callback")
        losses = state_dict["batch_losses"]
        if not isinstance(losses, list) or len(losses) > self.max_batches:
            raise ValueError("Invalid or oversized callback batch_losses checkpoint")
        losses = [_validate_scalar(value, "batch_loss", nonnegative=True) for value in losses]
        state = copy.deepcopy(state_dict["scheduler"])
        if state is not None and not isinstance(state, dict):
            raise ValueError("Invalid callback scheduler checkpoint")
        if state is None and (losses or self.scheduler is not None):
            raise ValueError("Cannot restore missing scheduler state into an active callback")
        if self.scheduler is not None and state is not None:
            self.scheduler.load_state_dict(state)
        else:
            self._pending_scheduler_state = state
        self.batch_losses = losses
