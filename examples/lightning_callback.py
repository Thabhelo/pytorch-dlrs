"""
PyTorch Lightning Callback for DLRS

This module provides a Lightning callback that automatically integrates DLRS
into your Lightning training loop.

Example:
    import pytorch_lightning as pl
    from examples.lightning_callback import DLRSLightningCallback

    class MyModel(pl.LightningModule):
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
            return optimizer

    trainer = pl.Trainer(
        callbacks=[DLRSLightningCallback(delta_d=0.5, delta_i=0.1)]
    )
    trainer.fit(model, datamodule)
"""

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback
except ImportError:
    raise ImportError(
        "PyTorch Lightning not installed. "
        "Install with: pip install pytorch-lightning"
    )

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from dlrs import DLRSScheduler


class DLRSLightningCallback(Callback):
    """
    PyTorch Lightning callback for DLRS scheduler.

    This callback automatically:
    - Creates a DLRS scheduler for your optimizer
    - Collects batch losses during training
    - Calls scheduler.step() at the end of each epoch

    Parameters:
        delta_d (float): Decremental factor for divergence. Default: 0.5
        delta_o (float): Stagnation factor. Default: 1.0
        delta_i (float): Incremental factor for convergence. Default: 0.1
        min_lr (float): Minimum learning rate bound. Default: 1e-8
        max_lr (float): Maximum learning rate bound. Default: None
        verbose (bool): Print LR updates to console. Default: False
        monitor_key (str): Key in batch outputs to use as loss. Default: "loss"

    Example:
        >>> callback = DLRSLightningCallback(delta_d=0.5, delta_i=0.1, verbose=True)
        >>> trainer = pl.Trainer(callbacks=[callback])
        >>> trainer.fit(model, datamodule)
    """

    def __init__(
        self,
        delta_d: float = 0.5,
        delta_o: float = 1.0,
        delta_i: float = 0.1,
        min_lr: float = 1e-8,
        max_lr: Optional[float] = None,
        verbose: bool = False,
        monitor_key: str = "loss"
    ):
        super().__init__()
        self.delta_d = delta_d
        self.delta_o = delta_o
        self.delta_i = delta_i
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.verbose = verbose
        self.monitor_key = monitor_key

        self.scheduler = None
        self.batch_losses = []

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Initialize DLRS scheduler when training starts."""
        optimizers = trainer.optimizers

        if len(optimizers) == 0:
            raise RuntimeError("No optimizer found. Cannot create DLRS scheduler.")

        if len(optimizers) > 1:
            print(
                f"Warning: Multiple optimizers detected ({len(optimizers)}). "
                f"DLRS will only apply to the first optimizer."
            )

        optimizer = optimizers[0]

        self.scheduler = DLRSScheduler(
            optimizer,
            delta_d=self.delta_d,
            delta_o=self.delta_o,
            delta_i=self.delta_i,
            min_lr=self.min_lr,
            max_lr=self.max_lr,
            verbose=self.verbose
        )

        if self.verbose:
            print(f"DLRS scheduler initialized with: "
                  f"delta_d={self.delta_d}, delta_o={self.delta_o}, delta_i={self.delta_i}")

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int
    ) -> None:
        """Collect batch loss after each training step."""
        if outputs is None:
            return

        if isinstance(outputs, dict) and self.monitor_key in outputs:
            loss_value = outputs[self.monitor_key]
        elif hasattr(outputs, self.monitor_key):
            loss_value = getattr(outputs, self.monitor_key)
        else:
            return

        if hasattr(loss_value, 'item'):
            self.batch_losses.append(loss_value.item())
        else:
            self.batch_losses.append(float(loss_value))

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Update learning rate at the end of each epoch."""
        if self.scheduler is None:
            return

        if len(self.batch_losses) < 2:
            print(
                f"Warning: Only {len(self.batch_losses)} batch losses collected. "
                f"Skipping DLRS update for epoch {trainer.current_epoch}."
            )
            self.batch_losses = []
            return

        self.scheduler.step(self.batch_losses)

        current_lr = trainer.optimizers[0].param_groups[0]['lr']
        if trainer.logger is not None:
            trainer.logger.log_metrics(
                {'learning_rate': current_lr, 'dlrs_loss_slope': self.scheduler.loss_slope},
                step=trainer.global_step
            )

        self.batch_losses = []

    def state_dict(self):
        """Return callback state for checkpointing."""
        if self.scheduler is None:
            return {}
        return {
            'scheduler': self.scheduler.state_dict(),
            'batch_losses': self.batch_losses
        }

    def load_state_dict(self, state_dict):
        """Load callback state from checkpoint."""
        if 'scheduler' in state_dict and self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict['scheduler'])
        if 'batch_losses' in state_dict:
            self.batch_losses = state_dict['batch_losses']


if __name__ == '__main__':
    print("Example usage:")
    print("""
    import pytorch_lightning as pl
    from examples.lightning_callback import DLRSLightningCallback

    class MyModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(10, 1)

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.layer(x)
            loss = F.mse_loss(y_hat, y)
            return {'loss': loss}

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.001)

    model = MyModel()
    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[DLRSLightningCallback(delta_d=0.5, verbose=True)]
    )
    trainer.fit(model, train_dataloader)
    """)
