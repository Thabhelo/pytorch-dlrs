"""Measure local CPU scheduler overhead; this is not a training benchmark.

Run from an installed source checkout:
    python -m benchmarks.scheduler_overhead --output scheduler-overhead.json

Timing excludes input/optimizer construction. Allocation measures Python-traced
peak bytes during a separate step, excluding the input list and tensor storage.
"""

import argparse
import json
import platform
import statistics
import time
import tracemalloc

import torch

from dlrs import DLRSScheduler


def measure(batch_count, group_count, profile, repeats):
    losses = [2.0] * batch_count
    losses[0], losses[-1] = (1.0, 3.0) if profile == "threshold" else (4.0, 1.0)
    groups = [
        {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 0.01} for _ in range(group_count)
    ]
    optimizer = torch.optim.SGD(groups)
    scheduler = DLRSScheduler(optimizer, max_lr=0.1)
    optimizer.step()
    scheduler.step(losses)  # Warm up imports/one-time initialization outside timing.
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        scheduler.step(losses)
        durations.append(time.perf_counter() - start)
    tracemalloc.start()
    try:
        scheduler.step(losses)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "batches": batch_count,
        "groups": group_count,
        "profile": profile,
        "median_seconds": statistics.median(durations),
        "peak_python_bytes": peak,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", type=int, default=[10, 1000, 100000])
    parser.add_argument("--groups", nargs="+", type=int, default=[1, 100])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default="scheduler-overhead.json")
    args = parser.parse_args()
    if min(args.batches) < 2 or min(args.groups) < 1 or args.repeats < 1:
        parser.error("batches >= 2, groups >= 1, and repeats >= 1 are required")
    results = [
        measure(batches, groups, profile, args.repeats)
        for batches in args.batches
        for groups in args.groups
        for profile in ("ordinary", "threshold")
    ]
    output = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "repeats": args.repeats,
        "scope": "CPU Python scheduler only; float LRs; no training, devices or networking",
        "results": results,
    }
    with open(args.output, "w") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
