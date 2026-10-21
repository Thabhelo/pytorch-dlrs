"""The former Hugging Face callback example is intentionally unsupported.

``TrainerCallback.on_step_end`` does not receive the raw training loss.
``TrainerState.log_history`` holds intermittent aggregate logs, so reading its
last entry once per step repeats stale data and does not implement DLRS. Trainer
also owns another scheduler by default; it can overwrite DLRS learning rates.
The former callback's ``on_load`` was not a TrainerCallback restore event, and
its shared output-directory checkpoint was not a per-checkpoint/rank-safe save.

A safe integration needs a version-tested Trainer subclass or custom loop that
captures each raw loss, gives DLRS sole ownership of LR changes, and explicitly
implements checkpoint and distributed semantics. Until then, use a direct
PyTorch training loop and call DLRSScheduler.step with that epoch's ordered
losses. This module deliberately fails before training can be changed.
"""


class DLRSTrainerCallback:
    """Compatibility name retained to fail clearly instead of silently training."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "The DLRS Hugging Face callback is unsupported: Trainer callback hooks "
            "do not supply raw per-batch losses, and Trainer owns LR scheduling. "
            "Use a direct PyTorch loop with DLRSScheduler, or implement and validate "
            "a Trainer subclass with exclusive scheduler ownership and checkpoint support."
        )
