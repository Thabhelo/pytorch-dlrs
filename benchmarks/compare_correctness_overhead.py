"""Descriptive paired scheduler-only timings for the correctness revision.

This probe compares a local Git revision with an immutable snapshot of the
working package. It measures one public scheduler.step call, excluding package
imports, scheduler/optimizer construction, and optimizer updates. Results are
not training-quality evidence, deployment latency estimates, or a pass gate.
"""

import argparse
import datetime
import gc
import hashlib
import importlib.util
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "d1d850b006cf0d1048a7cec8050c2c79dec91e12"
FILES = ("dlrs/__init__.py", "dlrs/scheduler.py", "dlrs/utils.py")
RATE = 0.0125


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(*arguments):
    return subprocess.run(["git"] + list(arguments), cwd=ROOT, check=True,
                          capture_output=True, timeout=30).stdout


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def load_package(directory):
    """Load unchanged sources with temporary aliases for their absolute imports.

    Each returned package retains its own scheduler, utils, and bound helper
    objects. The caller's dlrs module registry is restored after each load.
    Neither revision is imported from an installed distribution or live source.
    """
    previous = {name: module for name, module in sys.modules.items()
                if name == "dlrs" or name.startswith("dlrs.")}
    for name in previous:
        del sys.modules[name]
    try:
        spec = importlib.util.spec_from_file_location(
            "dlrs", directory / "dlrs/__init__.py",
            submodule_search_locations=[str(directory / "dlrs")])
        package = importlib.util.module_from_spec(spec)
        sys.modules["dlrs"] = package
        spec.loader.exec_module(package)
        for relative in FILES:
            module = package if relative.endswith("__init__.py") else getattr(package, Path(relative).stem)
            if Path(module.__file__).resolve() != (directory / relative).resolve():
                raise RuntimeError("Imported package source does not match its retained snapshot")
        if package.scheduler._validate_scalar is not package.utils._validate_scalar:
            raise RuntimeError("Scheduler and utility revision bindings differ")
        return package
    finally:
        for name in list(sys.modules):
            if name == "dlrs" or name.startswith("dlrs."):
                del sys.modules[name]
        sys.modules.update(previous)


def make_scheduler(package, group_count, torch):
    groups = [{"params": [torch.nn.Parameter(torch.zeros(()))], "lr": RATE}
              for _ in range(group_count)]
    return package.DLRSScheduler(torch.optim.SGD(groups))


def observation_sequence(count, path):
    # Both recipes have exactly mean 2. Ordinary has slope -1; the second
    # deliberately enters the pre-existing exact rational slope-one path.
    losses = [2.0] * count
    losses[0], losses[-1] = (3.0, 1.0) if path == "ordinary" else (1.0, 3.0)
    return losses


def time_step(scheduler, losses, expected_slope):
    # A fresh preconstructed instance is used exactly once. These state checks
    # and outcome inspection are outside the timed interval.
    if scheduler.last_epoch != 0 or scheduler.batch_losses:
        raise RuntimeError("Timing requires a fresh scheduler")
    if any(group["lr"] != RATE for group in scheduler.optimizer.param_groups):
        raise RuntimeError("Timing rates differ from the predeclared rate")
    start = time.perf_counter_ns()
    scheduler.step(losses)
    elapsed = time.perf_counter_ns() - start
    rates = [group["lr"] for group in scheduler.optimizer.param_groups]
    if scheduler.loss_slope != expected_slope or scheduler.last_epoch != 1:
        raise RuntimeError("A measured call did not follow the intended slope path")
    if scheduler.batch_losses != losses or not all(0 < rate < 1 for rate in rates):
        raise RuntimeError("A measured call did not produce the expected valid state")
    return elapsed, rates


def cpu_description():
    if sys.platform == "darwin":
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                capture_output=True, check=False, timeout=5, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    return platform.processor() or platform.machine()


def run(output, baseline, pairs, warmups, seed, execution_context):
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    baseline_commit = git("rev-parse", baseline + "^{commit}").decode().strip()
    snapshots = {
        "baseline": {name: git("show", baseline_commit + ":" + name) for name in FILES},
        "current": {name: (ROOT / name).read_bytes() for name in FILES},
    }
    packages = {}
    for label, files in snapshots.items():
        directory = output / label
        for name, data in files.items():
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        packages[label] = load_package(directory)
    if (packages["baseline"].scheduler is packages["current"].scheduler
            or packages["baseline"].utils is packages["current"].utils):
        raise RuntimeError("Revision modules are not independent")
    runner_bytes = Path(__file__).read_bytes()
    (output / "probe.py").write_bytes(runner_bytes)
    metadata = {
        "schema_version": 1,
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Descriptive scheduler-only timing; no training-quality, deployment-latency, or acceptance claim.",
        "baseline_commit": baseline_commit,
        "current_head": git("rev-parse", "HEAD").decode().strip(),
        "current_source_is_working_tree_snapshot": True,
        "source_sha256": {label: {name: digest(data) for name, data in files.items()}
                          for label, files in snapshots.items()},
        "imported_source_paths": {label: {name: str((output / label / name).resolve()) for name in FILES}
                                  for label in packages},
        "runner_sha256": digest(runner_bytes),
        "environment": {"python": platform.python_version(), "executable": sys.executable,
                        "torch": str(torch.__version__), "platform": platform.platform(),
                        "cpu": cpu_description(), "logical_cpus": os.cpu_count(),
                        "torch_threads": torch.get_num_threads(),
                        "torch_interop_threads": torch.get_num_interop_threads(),
                        "gc_enabled": gc.isenabled(), "execution_context": execution_context},
        "protocol": {"seed": seed, "pairs_per_case": pairs, "warmups_per_revision_per_case": warmups,
                     "observation_counts": [10, 1000, 100000], "parameter_group_counts": [1, 100],
                     "paths": ["ordinary", "slope_one"], "initial_rate": RATE,
                     "loss_recipe": "All interior values are 2; endpoints 3,1 for ordinary or 1,3 for slope_one.",
                     "timed_region": "One scheduler.step(losses) call using a fresh preconstructed scheduler.",
                     "order": "Seeded randomized case order and per-pair AB/BA order; A=baseline, B=current.",
                     "statistics": "Raw nanoseconds, revision medians, median paired ratio and difference; no threshold."},
    }
    write_json(output / "protocol.json", metadata)
    generator = random.Random(seed)
    cases = [(count, groups, path) for count in (10, 1000, 100000)
             for groups in (1, 100) for path in ("ordinary", "slope_one")]
    generator.shuffle(cases)
    results = []
    for count, group_count, path in cases:
        losses = observation_sequence(count, path)
        expected_slope = -1.0 if path == "ordinary" else 1.0
        # Build all instances before any timing; allocation or constructor work
        # is never charged to either measured revision.
        pool = [{label: make_scheduler(package, group_count, torch)
                 for label, package in packages.items()} for _ in range(warmups + pairs)]
        rows = []
        for index, schedulers in enumerate(pool):
            order = ["baseline", "current"]
            generator.shuffle(order)
            row = {"pair": index - warmups, "order": order, "duration_ns": {}}
            outcomes = {}
            for label in order:
                elapsed, rates = time_step(schedulers[label], losses, expected_slope)
                row["duration_ns"][label] = elapsed
                outcomes[label] = rates
            if outcomes["baseline"] != outcomes["current"]:
                raise RuntimeError("Measured revisions produced different rates for the same controlled inputs")
            if index >= warmups:
                rows.append(row)
        before = [row["duration_ns"]["baseline"] for row in rows]
        after = [row["duration_ns"]["current"] for row in rows]
        result = {"observation_count": count, "parameter_groups": group_count, "path": path,
                  "raw_pairs": rows, "baseline_median_ns": statistics.median(before),
                  "current_median_ns": statistics.median(after),
                  "median_paired_ratio": statistics.median(new / old for old, new in zip(before, after)),
                  "median_paired_difference_ns": statistics.median(new - old for old, new in zip(before, after))}
        results.append(result)
        write_json(output / f"case_{count}_{group_count}_{path}.json", result)
        print("{} observations / {} groups / {}: ratio {:.3f}, delta {:.1f} us".format(
            count, group_count, path, result["median_paired_ratio"],
            result["median_paired_difference_ns"] / 1000), flush=True)
        del pool
    unchanged = all((ROOT / name).read_bytes() == data for name, data in snapshots["current"].items())
    retained = all((output / label / name).read_bytes() == data
                   for label, files in snapshots.items() for name, data in files.items())
    summary = {**metadata, "results": results, "live_sources_unchanged": unchanged,
               "retained_sources_unchanged": retained,
               "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_json(output / "report.json", summary)
    write_json(output / "file-sha256.json", {str(path.relative_to(output)): digest(path.read_bytes())
                                           for path in sorted(output.rglob("*")) if path.is_file()})
    if not unchanged or not retained:
        raise RuntimeError("Source bytes changed during measurement; do not interpret results")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New retained evidence directory")
    parser.add_argument("--baseline", default=BASELINE, help="Existing local Git commit/ref; no network access")
    parser.add_argument("--pairs", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--execution-context", default="Shared desktop; concurrent workload isolation not measured.",
                        help="Operator-provided execution conditions, not independently verified by this probe")
    args = parser.parse_args()
    if not 1 <= args.pairs <= 100 or not 1 <= args.warmups <= 10:
        parser.error("pairs must be in [1,100] and warmups in [1,10]")
    run(args.output.resolve(), args.baseline, args.pairs, args.warmups, args.seed, args.execution_context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
