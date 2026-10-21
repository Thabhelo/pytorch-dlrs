"""Loss-driven learning-rate updates from arXiv:2507.21749v1.

DLRS is an opt-in training policy, not a convergence or non-regression guarantee.
"""

import math
import sys
import warnings
from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch.optim.lr_scheduler import LRScheduler

from dlrs.utils import _validate_scalar


def _nonnegative_int(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _losses(values) -> List[float]:
    if isinstance(values, (str, bytes, dict)):
        raise ValueError("losses must be an iterable of finite nonnegative real scalars")
    try:
        return [_validate_scalar(value, "loss", nonnegative=True) for value in values]
    except TypeError as exc:
        raise ValueError("losses must be an iterable of finite nonnegative real scalars") from exc


def _hyperparameters(delta_d, delta_o, delta_i, min_lr, max_lr) -> dict:
    values: Dict[str, Any] = {}
    for name, value in (("delta_d", delta_d), ("delta_o", delta_o),
                        ("delta_i", delta_i), ("min_lr", min_lr), ("max_lr", max_lr)):
        if name == "max_lr" and value is None:
            values[name] = None
            continue
        number = _validate_scalar(value, name)
        if number <= 0:
            raise ValueError(f"{name} must be positive, got {number}")
        values[name] = number
    if max_lr is not None and values['max_lr'] <= values['min_lr']:
        raise ValueError("max_lr must be greater than min_lr")
    return values


def _lr_value(value, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if (value.ndim != 0 or not value.is_floating_point() or value.requires_grad
                or value.is_inference()):
            raise ValueError(f"{name} must be a writable floating scalar tensor without gradients")
    return _validate_scalar(value, name, nonnegative=True)


def _clone_lr(value) -> Union[float, torch.Tensor]:
    return value.detach().clone() if isinstance(value, torch.Tensor) else float(value)


class DLRSScheduler(LRScheduler):
    """Adjust learning rates using an epoch's ordered, nonnegative batch losses.

    For slope ``s = (last - first) / mean`` and current rate ``a > 0``, use
    ``a - 10**floor(log10(a)) * delta * s``, clamped to ``min_lr`` and optional
    ``max_lr``. ``delta_d`` applies to ``s > 1``, ``delta_o`` to ``0 <= s <= 1``,
    and ``delta_i`` to ``s < 0``. A decreasing loss therefore increases the rate.
    The paper's piecewise equation includes ``s == 1`` in the middle case.

    Inputs must be finite real scalars; losses and optimizer rates must be
    nonnegative. A zero optimizer rate stays frozen, exempt from ``min_lr``.
    All-zero loss sequences and fewer than two losses warn and skip adjustment.
    ``step(None)`` also leaves rates unchanged. Every successful call advances
    ``last_epoch``; construction performs PyTorch's initial no-op step.

    Floating scalar tensor rates are updated in place. Parameter groups must
    retain their count and order after construction. This scheduler consumes
    observations explicitly and is not automatically driven by ChainedScheduler
    or SequentialLR. It is not intended for differentiable or captured stepping.
    """

    delta_d: float
    delta_o: float
    delta_i: float
    min_lr: float
    max_lr: Optional[float]

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        delta_d: float = 0.5,
        delta_o: float = 1.0,
        delta_i: float = 0.1,
        min_lr: float = 1e-8,
        max_lr: Optional[float] = None,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch.optim.Optimizer")
        self.__dict__.update(_hyperparameters(delta_d, delta_o, delta_i, min_lr, max_lr))
        _nonnegative_int(last_epoch, "last_epoch", -1)
        if not isinstance(verbose, bool):
            raise ValueError("verbose must be a bool")
        self.verbose = verbose
        for name, low, high in (("delta_d", 0.1, 1.0), ("delta_o", 0.5, 2.0),
                                ("delta_i", 0.01, 0.5)):
            value = getattr(self, name)
            if not low <= value <= high:
                warnings.warn(f"{name}={value} is outside typical range [{low}, {high}]",
                              stacklevel=2)
        for index, group in enumerate(optimizer.param_groups):
            _lr_value(group['lr'], f"param group {index} lr")
            if 'initial_lr' in group:
                _lr_value(group['initial_lr'], f"param group {index} initial_lr")
        self._param_group_ids = tuple(tuple(id(parameter) for parameter in group['params'])
                                      for group in optimizer.param_groups)
        self.batch_losses: List[float] = []
        self.loss_slope = 0.0
        self.adjustment = 0.0
        super().__init__(optimizer, last_epoch)
        # Older PyTorch versions alias tensor initial_lr/base_lrs to the live LR.
        # Metadata must stay fixed while DLRS updates that live tensor in place.
        for group in optimizer.param_groups:
            if isinstance(group['initial_lr'], torch.Tensor):
                group['initial_lr'] = _clone_lr(group['initial_lr'])
        self.base_lrs = [_clone_lr(value) if isinstance(value, torch.Tensor) else value
                         for value in self.base_lrs]

    def _validate_groups(self) -> List[float]:
        if tuple(tuple(id(parameter) for parameter in group['params'])
                 for group in self.optimizer.param_groups) != self._param_group_ids:
            raise ValueError("optimizer parameter groups changed; create a new DLRS scheduler")
        return [_lr_value(group['lr'], f"param group {index} lr")
                for index, group in enumerate(self.optimizer.param_groups)]

    @staticmethod
    def _candidate(current: float, slope: float, delta: float, min_lr: float,
                   max_lr: Optional[float]):
        if current == 0:
            return 0.0, 0.0
        exponent = math.floor(math.log10(current))
        granularity = 10.0 ** exponent
        # log10 may round a value immediately below a decade onto that decade.
        if current < sys.float_info.min:
            # A rounded subnormal power of ten can equal a value mathematically
            # below that decade; compare the exact binary64 value instead.
            exponent = Decimal.from_float(current).adjusted()
            granularity = 10.0 ** exponent
        elif granularity > current:
            exponent -= 1
            granularity = 10.0 ** exponent
        elif exponent < 308 and 10.0 ** (exponent + 1) <= current:
            exponent += 1
            granularity = 10.0 ** exponent
        if (granularity < sys.float_info.min
                or granularity * delta < sys.float_info.min
                or not math.isfinite(granularity * delta)):
            # 10**-324 is smaller than a binary64 float, but its product need not be.
            with localcontext() as context:
                context.prec = 50
                exact_adjustment = Decimal(10) ** exponent * Decimal(str(delta)) * Decimal(str(slope))
                adjustment = float(exact_adjustment)
                proposed = float(Decimal.from_float(current) - exact_adjustment)
        else:
            adjustment = granularity * delta * slope
            proposed = current - adjustment
        if not math.isfinite(adjustment) or not math.isfinite(proposed):
            raise ValueError("learning-rate update overflowed; choose smaller factors or rates")
        result = max(proposed, min_lr)
        if max_lr is not None:
            result = min(result, max_lr)
        return result, adjustment

    @staticmethod
    def _prepare_lr(value: float, target, min_lr: Optional[float] = None,
                    max_lr: Optional[float] = None):
        if not isinstance(target, torch.Tensor):
            return value
        try:
            prepared = torch.full_like(target, value, requires_grad=False)
        except (RuntimeError, OverflowError) as exc:
            raise ValueError("learning rate is not representable in optimizer tensor dtype") from exc
        actual = float(prepared.item())
        if not math.isfinite(actual) or (value > 0 and actual <= 0):
            raise ValueError("learning rate is not representable in optimizer tensor dtype")
        # Round bounds inward so the stored tensor value also satisfies them.
        if value > 0 and min_lr is not None and actual < min_lr:
            prepared = torch.nextafter(prepared, torch.full_like(prepared, math.inf))
            actual = float(prepared.item())
        if value > 0 and max_lr is not None and actual > max_lr:
            prepared = torch.nextafter(prepared, torch.full_like(prepared, -math.inf))
            actual = float(prepared.item())
        if (not math.isfinite(actual) or (value > 0 and actual <= 0)
                or (value > 0 and min_lr is not None and actual < min_lr)
                or (value > 0 and max_lr is not None and actual > max_lr)):
            raise ValueError("learning-rate bounds have no representable tensor value")
        return prepared

    def _apply_lrs(self, rates) -> None:
        # Optimizers can share one tensor LR across groups. Conflicting restored
        # values (or overlapping views of different dtypes) cannot be applied.
        targets: List[Tuple[str, int, int, torch.dtype, Union[float, torch.Tensor]]] = []
        for group, value in zip(self.optimizer.param_groups, rates):
            target = group['lr']
            if isinstance(target, torch.Tensor):
                start = target.data_ptr()
                targets.append((str(target.device), start, start + target.element_size(), target.dtype, value))
        targets.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
        for previous, current in zip(targets, targets[1:]):
            device, start, end, dtype, value = current
            old_device, old_start, old_end, old_dtype, old_value = previous
            if device == old_device and start < old_end:
                if (start != old_start or end != old_end or dtype != old_dtype
                        or float(value) != float(old_value)):
                    raise ValueError("conflicting learning rates share tensor storage")
        with torch.no_grad():
            for group, value in zip(self.optimizer.param_groups, rates):
                if isinstance(group['lr'], torch.Tensor):
                    group['lr'].copy_(value)
                else:
                    group['lr'] = value

    def step(self, batch_losses: Optional[Iterable[float]] = None) -> None:  # type: ignore[override]
        """Validate an epoch completely, then update every parameter group together."""
        self._step_with_direction(batch_losses, 1.0)

    def _step_with_direction(self, batch_losses, direction: float) -> None:
        current_rates = self._validate_groups()
        config = _hyperparameters(self.delta_d, self.delta_o, self.delta_i,
                                  self.min_lr, self.max_lr)
        losses = [] if batch_losses is None else _losses(batch_losses)
        slope = 0.0
        adjustment = 0.0
        prepared = None
        if batch_losses is not None and len(losses) < 2:
            warnings.warn("Need at least 2 batch losses to compute slope, skipping update", stacklevel=3)
        elif losses:
            scale = max(losses)
            if scale == 0:
                warnings.warn("Mean loss is zero, skipping learning rate update", stacklevel=3)
            else:
                # Scale before subtracting and summing: avoids overflow and underflow.
                exponent = math.frexp(scale)[1]
                normalized = [math.ldexp(value, -exponent) for value in losses]
                slope = direction * (normalized[-1] - normalized[0]) / (math.fsum(normalized) / len(losses))
                diverging = slope > 1
                if abs(slope - 1.0) <= 1e-15:
                    # The discontinuous delta switch requires an exact comparison:
                    # a rounded mean may hide a true slope immediately above 1.
                    exact_sum = sum((Fraction(value) for value in losses), Fraction())
                    exact_slope = (Fraction(direction) * (Fraction(losses[-1]) - Fraction(losses[0]))
                                   * len(losses) / exact_sum)
                    slope = float(exact_slope)
                    diverging = exact_slope > 1
                delta = config['delta_d'] if diverging else config['delta_o'] if slope >= 0 else config['delta_i']
                candidates = [self._candidate(rate, slope, delta, config['min_lr'], config['max_lr'])
                              for rate in current_rates]
                prepared = [self._prepare_lr(candidate[0], group['lr'], config['min_lr'], config['max_lr'])
                            for candidate, group in zip(candidates, self.optimizer.param_groups)]
                adjustment = candidates[-1][1]
        if prepared is not None:
            self._apply_lrs(prepared)
        self.batch_losses = losses
        self.loss_slope = slope
        self.adjustment = adjustment
        self.last_epoch += 1
        self._step_count += 1
        self._last_lr = [_clone_lr(group['lr']) for group in self.optimizer.param_groups]
        if self.verbose and prepared is not None:
            print(f"Epoch {self.last_epoch}: loss_slope={slope:.6f}, lr={self.get_last_lr()}")

    def get_lr(self) -> List[Union[float, torch.Tensor]]:  # type: ignore[override]
        """Return independent snapshots of the current optimizer rates."""
        self._validate_groups()
        return [_clone_lr(group['lr']) for group in self.optimizer.param_groups]

    def get_last_lr(self) -> List[Union[float, torch.Tensor]]:
        """Return independent snapshots of the rates from the last successful step."""
        return [_clone_lr(value) for value in self._last_lr]

    def state_dict(self) -> dict:
        """Return a detached checkpoint, including optimizer rates for legacy compatibility."""
        self._validate_groups()
        return deepcopy({
            'state_version': 1,
            'delta_d': self.delta_d, 'delta_o': self.delta_o, 'delta_i': self.delta_i,
            'min_lr': self.min_lr, 'max_lr': self.max_lr, 'verbose': self.verbose,
            'last_epoch': self.last_epoch, '_step_count': self._step_count,
            'base_lrs': self.base_lrs, 'batch_losses': self.batch_losses,
            'loss_slope': self.loss_slope, 'adjustment': self.adjustment,
            '_last_lr': self.get_last_lr(), 'param_group_lrs': self.get_lr(),
        })

    def load_state_dict(self, state_dict: dict) -> None:
        """Validate the whole checkpoint before changing scheduler or optimizer state.

        Checkpoints from earlier DLRS versions without ``state_version`` remain
        accepted when their values and parameter-group counts are valid. Construct
        this scheduler before loading optimizer state; then load scheduler state.
        """
        self._validate_groups()
        if not isinstance(state_dict, dict):
            raise ValueError("scheduler state must be a dictionary")
        version = state_dict.get('state_version', 0)
        if isinstance(version, bool) or not isinstance(version, int) or version not in (0, 1):
            raise ValueError("unsupported scheduler state_version")
        if version == 1:
            required = {'delta_d', 'delta_o', 'delta_i', 'min_lr', 'max_lr', 'verbose',
                        'last_epoch', '_step_count', 'base_lrs', 'batch_losses',
                        'loss_slope', 'adjustment', '_last_lr', 'param_group_lrs'}
            missing = required - state_dict.keys()
            if missing:
                raise ValueError(f"scheduler checkpoint is missing {sorted(missing)}")
        try:
            values = _hyperparameters(state_dict['delta_d'], state_dict['delta_o'],
                                     state_dict['delta_i'], state_dict['min_lr'],
                                     state_dict.get('max_lr'))
            values['last_epoch'] = _nonnegative_int(state_dict['last_epoch'], 'last_epoch', -1)
        except KeyError as exc:
            raise ValueError(f"scheduler checkpoint is missing {exc.args[0]}") from exc
        values['_step_count'] = _nonnegative_int(state_dict.get('_step_count', values['last_epoch'] + 1), '_step_count')
        values['loss_slope'] = _validate_scalar(state_dict.get('loss_slope', 0.0), 'loss_slope')
        values['adjustment'] = _validate_scalar(state_dict.get('adjustment', 0.0), 'adjustment')
        values['batch_losses'] = _losses(state_dict.get('batch_losses', []))
        values['verbose'] = state_dict.get('verbose', self.verbose)
        if not isinstance(values['verbose'], bool):
            raise ValueError("verbose must be a bool")
        count = len(self.optimizer.param_groups)

        def rates(name, fallback):
            items = state_dict.get(name, fallback)
            if not isinstance(items, (list, tuple)) or len(items) != count:
                raise ValueError(f"{name} must contain exactly {count} parameter-group rates")
            return [_lr_value(item, name) for item in items]

        current = rates('param_group_lrs', self.get_lr())
        # Legacy checkpoints may use None for _last_lr.
        if version == 1 and state_dict['_last_lr'] is None:
            raise ValueError("_last_lr must contain parameter-group rates")
        last_fallback = current if state_dict.get('_last_lr') is None else state_dict['_last_lr']
        if not isinstance(last_fallback, (list, tuple)) or len(last_fallback) != count:
            raise ValueError(f"_last_lr must contain exactly {count} parameter-group rates")
        last = [_lr_value(item, '_last_lr') for item in last_fallback]
        rates('base_lrs', self.base_lrs)
        # Preserve checkpoint metadata types as well as values. Tensor base rates
        # remain independent snapshots on every supported PyTorch version.
        values['base_lrs'] = [_clone_lr(item) for item in state_dict.get('base_lrs', self.base_lrs)]
        prepared = [self._prepare_lr(value, group['lr'])
                    for value, group in zip(current, self.optimizer.param_groups)]
        values['_last_lr'] = [self._prepare_lr(value, group['lr'])
                              for value, group in zip(last, self.optimizer.param_groups)]
        # Loading an older optimizer checkpoint can reintroduce an initial_lr
        # alias after construction. Stage copies before any live LR mutation.
        initial_lrs = [(group, _clone_lr(group['initial_lr']))
                       for group in self.optimizer.param_groups
                       if isinstance(group.get('initial_lr'), torch.Tensor)]
        self._apply_lrs(prepared)
        for group, initial_lr in initial_lrs:
            group['initial_lr'] = initial_lr
        self.__dict__.update(values)


class DLRSOnPlateau:
    """Apply DLRS to a rolling window of nonnegative epoch metrics.

    This is a scalar-metric interface, not a ReduceLROnPlateau replacement:
    improving metrics can increase the learning rate. ``patience`` preserves the
    historical DLRS meaning: collect ``patience + 2`` metrics, then update each
    epoch using the latest window. It does not count non-improving epochs.
    ``mode='max'`` reverses the normalized slope before selecting the DLRS factor.
    Signed metrics are unsupported in either mode. History is bounded to the
    window size, and rejected input leaves history and optimizer unchanged.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        mode: str = 'min',
        delta_d: float = 0.5,
        delta_o: float = 1.0,
        delta_i: float = 0.1,
        min_lr: float = 1e-8,
        max_lr: Optional[float] = None,
        patience: int = 0,
        verbose: bool = False,
    ):
        self._validate_options(mode, patience)
        self.optimizer = optimizer
        self.mode = mode
        self.patience = patience
        self.verbose = verbose
        self._scheduler = DLRSScheduler(optimizer, delta_d, delta_o, delta_i,
                                        min_lr, max_lr, verbose=verbose)
        self._metric_history: List[float] = []
        self._wait_count = 0  # Kept only for legacy checkpoint compatibility.

    @staticmethod
    def _validate_options(mode, patience) -> None:
        if not isinstance(mode, str) or mode not in ('min', 'max'):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        _nonnegative_int(patience, 'patience')

    def step(self, metric: float) -> None:
        """Add a finite nonnegative metric and update after the window is full."""
        self._validate_options(self.mode, self.patience)
        self._scheduler._validate_groups()
        value = _validate_scalar(metric, 'metric', nonnegative=True)
        history = (self._metric_history + [value])[-(self.patience + 2):]
        if len(history) == self.patience + 2:
            self._scheduler._step_with_direction(history, 1.0 if self.mode == 'min' else -1.0)
        self._metric_history = history

    def state_dict(self) -> dict:
        """Return an independent checkpoint of the wrapper and underlying scheduler."""
        return {'scheduler_state': self._scheduler.state_dict(), 'mode': self.mode,
                'patience': self.patience, 'metric_history': self._metric_history.copy(),
                'wait_count': self._wait_count}

    def load_state_dict(self, state_dict: dict) -> None:
        """Validate wrapper state before loading either scheduler or optimizer."""
        if not isinstance(state_dict, dict):
            raise ValueError("wrapper state must be a dictionary")
        try:
            mode, patience = state_dict['mode'], state_dict['patience']
            scheduler_state = state_dict['scheduler_state']
        except KeyError as exc:
            raise ValueError(f"wrapper checkpoint is missing {exc.args[0]}") from exc
        self._validate_options(mode, patience)
        history = _losses(state_dict.get('metric_history', []))[-(patience + 2):]
        wait_count = _nonnegative_int(state_dict.get('wait_count', 0), 'wait_count')
        self._scheduler.load_state_dict(scheduler_state)
        self.mode, self.patience = mode, patience
        self._metric_history, self._wait_count = history, wait_count
        self.verbose = self._scheduler.verbose

    def get_last_lr(self) -> List[Union[float, torch.Tensor]]:
        """Return the most recently computed learning rates."""
        return self._scheduler.get_last_lr()
