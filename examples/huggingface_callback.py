"""
Hugging Face Transformers Callback for DLRS

This module provides a Trainer callback that integrates DLRS into
Hugging Face Transformers training pipelines.

Example:
    from transformers import Trainer, TrainingArguments
    from examples.huggingface_callback import DLRSTrainerCallback

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[DLRSTrainerCallback(delta_d=0.5, delta_i=0.1)]
    )

    trainer.train()
"""

try:
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
except ImportError:
    raise ImportError(
        "Hugging Face Transformers not installed. "
        "Install with: pip install transformers"
    )

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from dlrs import DLRSScheduler


class DLRSTrainerCallback(TrainerCallback):
    """
    Hugging Face Trainer callback for DLRS scheduler.

    This callback automatically:
    - Creates a DLRS scheduler for the Trainer's optimizer
    - Collects batch losses during training
    - Calls scheduler.step() at the end of each epoch

    Parameters:
        delta_d (float): Decremental factor for divergence. Default: 0.5
        delta_o (float): Stagnation factor. Default: 1.0
        delta_i (float): Incremental factor for convergence. Default: 0.1
        min_lr (float): Minimum learning rate bound. Default: 1e-8
        max_lr (float): Maximum learning rate bound. Default: None
        verbose (bool): Print LR updates to console. Default: False

    Example:
        >>> from transformers import Trainer
        >>> from examples.huggingface_callback import DLRSTrainerCallback
        >>>
        >>> trainer = Trainer(
        ...     model=model,
        ...     args=training_args,
        ...     train_dataset=train_dataset,
        ...     callbacks=[DLRSTrainerCallback(delta_d=0.5, verbose=True)]
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        delta_d: float = 0.5,
        delta_o: float = 1.0,
        delta_i: float = 0.1,
        min_lr: float = 1e-8,
        max_lr: Optional[float] = None,
        verbose: bool = False
    ):
        super().__init__()
        self.delta_d = delta_d
        self.delta_o = delta_o
        self.delta_i = delta_i
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.verbose = verbose

        self.scheduler = None
        self.batch_losses = []
        self.current_epoch = -1

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Initialize DLRS scheduler when training starts."""
        model = kwargs.get('model')
        optimizer = kwargs.get('optimizer')

        if optimizer is None:
            raise RuntimeError("Optimizer not found in kwargs. Cannot create DLRS scheduler.")

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

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Collect batch loss after each training step."""
        if hasattr(state, 'log_history') and len(state.log_history) > 0:
            last_log = state.log_history[-1]
            if 'loss' in last_log:
                self.batch_losses.append(last_log['loss'])

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Update learning rate at the end of each epoch."""
        if self.scheduler is None:
            return

        current_epoch = int(state.epoch)

        if current_epoch == self.current_epoch:
            return

        self.current_epoch = current_epoch

        if len(self.batch_losses) < 2:
            if self.verbose:
                print(
                    f"Warning: Only {len(self.batch_losses)} batch losses collected. "
                    f"Skipping DLRS update for epoch {current_epoch}."
                )
            self.batch_losses = []
            return

        self.scheduler.step(self.batch_losses)

        optimizer = kwargs.get('optimizer')
        if optimizer is not None:
            current_lr = optimizer.param_groups[0]['lr']
            if self.verbose:
                print(f"Epoch {current_epoch}: LR updated to {current_lr:.6e}, "
                      f"Loss slope: {self.scheduler.loss_slope:.6f}")

        self.batch_losses = []

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Save DLRS scheduler state with the checkpoint."""
        if self.scheduler is not None:
            import os
            import json

            output_dir = args.output_dir
            os.makedirs(output_dir, exist_ok=True)

            scheduler_path = os.path.join(output_dir, 'dlrs_scheduler_state.json')
            with open(scheduler_path, 'w') as f:
                json.dump(self.scheduler.state_dict(), f, indent=2)

            if self.verbose:
                print(f"DLRS scheduler state saved to {scheduler_path}")

    def on_load(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Load DLRS scheduler state from checkpoint."""
        if self.scheduler is None:
            return

        import os
        import json

        scheduler_path = os.path.join(args.output_dir, 'dlrs_scheduler_state.json')

        if os.path.exists(scheduler_path):
            with open(scheduler_path, 'r') as f:
                state_dict = json.load(f)
            self.scheduler.load_state_dict(state_dict)

            if self.verbose:
                print(f"DLRS scheduler state loaded from {scheduler_path}")


if __name__ == '__main__':
    print("Example usage:")
    print("""
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments
    )
    from examples.huggingface_callback import DLRSTrainerCallback

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[DLRSTrainerCallback(delta_d=0.5, delta_i=0.1, verbose=True)]
    )

    trainer.train()
    """)
