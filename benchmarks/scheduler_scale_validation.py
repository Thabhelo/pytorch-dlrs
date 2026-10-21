"""Exploratory scheduler scaling across optimizer structure and LR storage.

This measures actual eager DLRS calls, not model throughput or deployment SLOs.
Run from an installed checkout with ``python -m benchmarks.scheduler_scale_validation``.
The diagnostic operations overlap and must not be added to explain total time.
"""

import argparse
import gc
import hashlib
import io
import json
import os
import platform
import statistics
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import torch

from dlrs import DLRSOnPlateau, DLRSScheduler


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()


def timed(operation, device, repeats):
    operation()
    synchronize(device)
    samples = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        operation()
        synchronize(device)
        samples.append(time.perf_counter() - start)
    return {
        "median_seconds": statistics.median(samples),
        "minimum_seconds": min(samples),
        "maximum_seconds": max(samples),
        "samples_seconds": samples,
    }


def traced(operation, device):
    gc.collect()
    tracemalloc.start()
    try:
        result = operation()
        synchronize(device)
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Keep the result alive until after reading retained allocation.
    del result
    return {"retained_python_bytes": current, "peak_python_bytes": peak}


def make_optimizer(parameter_count, group_count, lr_kind, device):
    if group_count > parameter_count:
        raise ValueError("each parameter group needs at least one parameter")
    parameters = [torch.nn.Parameter(torch.zeros(1, device=device))
                  for _ in range(parameter_count)]
    groups = []
    for group in range(group_count):
        rate = torch.tensor(0.01, device=device) if lr_kind == "tensor" else 0.01
        groups.append({"params": parameters[group::group_count], "lr": rate})
    optimizer = torch.optim.SGD(groups, foreach=False)
    scheduler = DLRSScheduler(optimizer, min_lr=1e-6, max_lr=0.2)
    # Exercise the optimizer, rather than only constructing inert groups.
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    return parameters, optimizer, scheduler


def serialized_bytes(state):
    stream = io.BytesIO()
    torch.save(state, stream)
    return stream.tell()


def tensor_payload(state):
    """Count logical tensor payload; aliases/native allocator overhead excluded."""
    if isinstance(state, torch.Tensor):
        return {"tensors": 1, "elements": state.numel(),
                "bytes": state.numel() * state.element_size()}
    children = state.values() if isinstance(state, dict) else state if isinstance(state, (list, tuple)) else ()
    result = {"tensors": 0, "elements": 0, "bytes": 0}
    for child in children:
        for key, value in tensor_payload(child).items():
            result[key] += value
    return result


def measure_case(parameter_count, group_count, observations, lr_kind, device, repeats):
    parameters, optimizer, scheduler = make_optimizer(parameter_count, group_count, lr_kind, device)
    parameter_snapshot = [parameter.detach().clone() for parameter in parameters]
    losses = [2.0] * observations
    losses[0], losses[-1] = 4.0, 1.0  # Ordinary slope; boundary fallback measured elsewhere.
    scalar_targets = [group["lr"] for group in optimizer.param_groups]
    prepared = [scheduler._prepare_lr(0.01, target, 1e-6, 0.2) for target in scalar_targets]
    measurements = {
        "full_step": timed(lambda: scheduler.step(losses), device, repeats),
        "group_validation": timed(scheduler._validate_groups, device, repeats),
        "lr_preparation": timed(lambda: [scheduler._prepare_lr(0.01, target, 1e-6, 0.2)
                                         for target in scalar_targets], device, repeats),
        "storage_validation_and_apply": timed(lambda: scheduler._apply_lrs(prepared), device, repeats),
        "scalar_extraction": timed(lambda: [float(target.item()) if isinstance(target, torch.Tensor)
                                             else float(target) for target in scalar_targets], device, repeats),
    }
    scheduler.step(losses)  # Restore internally consistent public state after diagnostic LR writes.
    measurements["state_dict"] = timed(scheduler.state_dict, device, repeats)
    allocations = {
        "step": traced(lambda: scheduler.step(losses), device),
        "state_dict": traced(scheduler.state_dict, device),
    }
    state = scheduler.state_dict()
    unchanged = all(torch.equal(before, parameter.detach())
                    for before, parameter in zip(parameter_snapshot, parameters))
    if not unchanged:
        raise AssertionError("scheduler diagnostics changed model parameters")
    if len(state["batch_losses"]) != observations:
        raise AssertionError("checkpoint did not preserve the latest observation sequence")
    return {
        "device": device, "lr_kind": lr_kind, "parameter_tensors": parameter_count,
        "parameter_elements": parameter_count, "parameter_payload_bytes": parameter_count * 4,
        "groups": group_count, "observations": observations,
        "measurements": measurements, "allocations": allocations,
        "checkpoint_serialized_bytes": serialized_bytes(state),
        "checkpoint_tensor_payload": tensor_payload(state),
        "parameter_values_unchanged": unchanged,
    }


def history_retention(observations, epochs):
    parameters, _, scheduler = make_optimizer(1, 1, "float", "cpu")
    before = parameters[0].detach().clone()
    retained = []
    peaks = []
    tracemalloc.start()
    try:
        for epoch in range(epochs):
            tracemalloc.reset_peak()
            # New float objects each epoch, then release the caller's reference.
            # This exposes accidental retention of earlier epochs' observations.
            losses = [1.0 + ((index + epoch) % 97) / 100.0 for index in range(observations)]
            scheduler.step(losses)
            del losses
            gc.collect()
            current, peak = tracemalloc.get_traced_memory()
            retained.append(current)
            peaks.append(peak)
    finally:
        tracemalloc.stop()
    state = scheduler.state_dict()
    parameter = torch.nn.Parameter(torch.zeros(1))
    plateau = DLRSOnPlateau(torch.optim.SGD([parameter], lr=0.01), patience=8, max_lr=0.2)
    for index in range(1000):
        plateau.step(1.0 + (index % 13) / 100.0)
    window = len(plateau.state_dict()["metric_history"])
    if len(state["batch_losses"]) != observations or window != 10:
        raise AssertionError("history retention differs from configured/latest window")
    if not torch.equal(before, parameters[0].detach()):
        raise AssertionError("scheduler changed model values")
    return {
        "epochs": epochs, "observations_per_epoch": observations,
        "retained_python_bytes_after_each_epoch": retained,
        "peak_python_bytes_during_each_epoch": peaks,
        "final_retained_observations": len(state["batch_losses"]),
        "checkpoint_serialized_bytes": serialized_bytes(state),
        "plateau_updates": 1000, "plateau_patience": 8, "plateau_retained_metrics": window,
        "qualification": "DLRS retains one full supplied epoch; no fixed cap is enforced by the core scheduler.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="scheduler-scale-validation.json")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--mps", action="store_true", help="Add small actual MPS LR-storage probes.")
    parser.add_argument("--concurrent-workloads", action="store_true",
                        help="Record that unrelated workloads were running during collection.")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.mps and not (torch.backends.mps.is_built() and torch.backends.mps.is_available()):
        parser.error("--mps requested but MPS is unavailable")
    torch.set_num_threads(2)
    started = datetime.now(timezone.utc).isoformat()
    cases = []
    for parameters in (1, 1000, 10000):
        for groups in (1, 32):
            if groups > parameters:
                continue
            for observations in (1000, 100000):
                for kind in ("float", "tensor"):
                    result = measure_case(parameters, groups, observations, kind, "cpu", args.repeats)
                    cases.append(result)
                    print(f"cpu parameters={parameters} groups={groups} observations={observations} lr={kind}", flush=True)
    if args.mps:
        for groups in (1, 32):
            for kind in ("float", "tensor"):
                cases.append(measure_case(groups, groups, 1000, kind, "mps", args.repeats))
                print(f"mps parameters={groups} groups={groups} observations=1000 lr={kind}", flush=True)
    root = Path(__file__).resolve().parents[1]
    sources = ("dlrs/scheduler.py", "dlrs/utils.py", "benchmarks/scheduler_scale_validation.py")
    output = {
        "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "platform": platform.platform(), "machine": platform.machine(),
                        "torch_threads": torch.get_num_threads(), "cuda_available": torch.cuda.is_available(),
                        "mps_available": torch.backends.mps.is_available(),
                        "mps_fallback_env": os.getenv("PYTORCH_ENABLE_MPS_FALLBACK")},
        "arguments": vars(args), "command_argv": sys.argv,
        "source_sha256": {source: hashlib.sha256((root / source).read_bytes()).hexdigest() for source in sources},
        "scope": "Exploratory eager scheduler microbenchmarks; no model-throughput, SLO, or large-model acceptance claim.",
        "timing_notes": ["Input/optimizer construction and initial optimizer update excluded.",
                         "MPS timings synchronize this process before and after each operation.",
                         "Stage diagnostics overlap, alter LR state, and are not additive attribution.",
                         "Python tracing excludes native tensor storage, allocator caches, and external-process memory.",
                         "Model tensors contain one float32 value each; tensor counts are not model capacity."],
        "cases": cases,
        "history_retention": history_retention(100000, 8),
    }
    # Capture completion after the history-retention probe as well.
    output["completed_utc"] = datetime.now(timezone.utc).isoformat()
    Path(args.output).write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(cases)} cases to {args.output}", flush=True)


if __name__ == "__main__":
    main()
