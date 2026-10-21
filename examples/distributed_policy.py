"""Illustrative, explicitly synchronized CPU/Gloo loss policy.

This is a tested example, not part of the installed ``dlrs`` API. Every rank must
call these functions in the same order on the same process group. Backend timeouts
and a launcher that terminates failed peers are still necessary. See
``docs/DISTRIBUTED_VALIDATION.md`` for the tested scope and recovery limits.
"""

import math
import numbers
from typing import NamedTuple, Optional

import torch
import torch.distributed as dist


class DistributedObservationError(ValueError):
    """All participating ranks rejected an invalid observation or batch contract."""


class GlobalObservation(NamedTuple):
    mean: float
    sample_count: int
    backward_scale: float


def _world_size(group=None) -> int:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("an initialized CPU/Gloo process group is required")
    if dist.get_backend(group) != "gloo":
        raise RuntimeError("this illustrative policy is validated only for CPU/Gloo")
    return dist.get_world_size(group)


def _reject_collectively(invalid: bool, message: str, group=None) -> None:
    # Invalid values never enter a sum. Crucially, local validation does not raise
    # until every healthy participating rank has entered this collective.
    invalid_count = torch.tensor(int(invalid), dtype=torch.int64)
    dist.all_reduce(invalid_count, op=dist.ReduceOp.SUM, group=group)
    if invalid_count.item():
        raise DistributedObservationError(
            f"{message}; invalid on {int(invalid_count.item())} rank(s)"
        )


def require_equal_iterations(local_iterations: int, group=None) -> None:
    """Collectively reject missing/unequal positive iteration counts before DDP.

    This policy does not implement DDP.join(), empty ranks, dynamic data-loader
    exhaustion, or uneven numbers of collective calls.
    """
    world_size = _world_size(group)
    valid = (isinstance(local_iterations, numbers.Integral)
             and not isinstance(local_iterations, bool)
             and 0 < local_iterations <= torch.iinfo(torch.int64).max)
    _reject_collectively(not valid, "invalid iteration count", group)
    local = torch.tensor(int(local_iterations), dtype=torch.int64)
    counts = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(counts, local, group=group)
    if any(int(count.item()) != int(local_iterations) for count in counts):
        raise DistributedObservationError("all ranks must have equal iteration counts")


def reduce_sample_mean(
    local_loss: torch.Tensor,
    local_sample_count: int,
    group: Optional[dist.ProcessGroup] = None,
) -> GlobalObservation:
    """Validate and reduce a per-example mean before backward/optimizer.step().

    Requires a finite, nonnegative CPU scalar mean loss and a positive count on
    every rank. The criterion must average exactly over the stated examples;
    token/masked/weighted criteria need a different denominator contract.

    Use ``(local_loss * result.backward_scale).backward()`` with ordinary DDP:
    it averages rank gradients, so unequal local batch sizes need the factor
    ``world_size * local_sample_count / global_sample_count``. Record
    ``result.mean`` for DLRS, then call DLRS on every rank after the epoch.

    A collective validation failure precedes backward and optimizer/scheduler
    mutation in this calling order. Forward passes may already have consumed RNG
    or updated buffers. Abort and restore a complete checkpoint; do not silently
    skip an observation or retry only one rank.
    """
    world_size = _world_size(group)
    valid_count = (isinstance(local_sample_count, numbers.Integral)
                   and not isinstance(local_sample_count, bool)
                   and 0 < local_sample_count <= (2 ** 53) // world_size)
    valid_loss = (isinstance(local_loss, torch.Tensor)
                  and local_loss.device.type == "cpu"
                  and local_loss.ndim == 0
                  and local_loss.is_floating_point())
    mean = 0.0
    weighted_sum = 0.0
    try:
        if valid_loss:
            mean = float(local_loss.detach().item())
            valid_loss = math.isfinite(mean) and mean >= 0
        if valid_loss and valid_count:
            weighted_sum = mean * int(local_sample_count)
            valid_loss = math.isfinite(weighted_sum)
    except (RuntimeError, TypeError, ValueError, OverflowError):
        valid_loss = False
    _reject_collectively(not (valid_loss and valid_count), "invalid sample-mean observation", group)

    stats = torch.tensor([weighted_sum, float(local_sample_count)], dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=group)
    total_sum, total_count = stats.tolist()
    # Every rank receives the same reduction, so this branch is coordinated too.
    if not math.isfinite(total_sum) or not math.isfinite(total_count) or total_count <= 0:
        raise DistributedObservationError("global sample-mean reduction overflowed")
    return GlobalObservation(
        mean=total_sum / total_count,
        sample_count=int(total_count),
        backward_scale=world_size * int(local_sample_count) / total_count,
    )
