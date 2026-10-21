"""Bounded float-LR differential probe, not training-quality evidence.

Fetches the exact original scheduler/utilities into an isolated temporary
directory, verifies their hashes, and compares seeded ordinary-use trajectories.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import random
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "c0ea1d159e4caa4950358282506e9358ae9a67ea"
SOURCE_HASHES = {
    "scheduler.py": "6636ac1c19c0ec52ed5521542870ce03212566efb912df093442af209a003bdd",
    "utils.py": "4c1d0d7c08736b0beb72b85f1284b97949e47a3336bde795dda4a9851ec46c36",
}
SEED = 20261020
CASES = 1000
EPOCHS = 8
RTOL, ATOL = 1e-12, 1e-15


def load_baseline(directory):
    modules = {}
    for filename, expected in SOURCE_HASHES.items():
        url = f"https://raw.githubusercontent.com/Thabhelo/pytorch-dlrs/{COMMIT}/dlrs/{filename}"
        with urllib.request.urlopen(url, timeout=15) as response:
            source = response.read()
        if hashlib.sha256(source).hexdigest() != expected:
            raise ValueError("Pinned original source hash mismatch: " + filename)
        path = directory / filename
        path.write_bytes(source)
        name = "original_dlrs_" + path.stem
        specification = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        modules[path.stem] = module
    return modules["scheduler"].DLRSScheduler


def make_optimizer(rates):
    groups = [{"params": [torch.nn.Parameter(torch.tensor(0.0))], "lr": rate}
              for rate in rates]
    return torch.optim.SGD(groups, lr=rates[0])


def branch(slope):
    return "convergence" if slope < 0 else "stagnation" if slope <= 1 else "divergence"


def ordinary_observations(generator, requested):
    for _ in range(10000):
        scale = 10 ** generator.uniform(-2, 2)
        losses = [scale * generator.uniform(0.25, 3.0)
                  for _ in range(generator.randint(4, 64))]
        slope = (losses[-1] - losses[0]) / (sum(losses) / len(losses))
        if ((requested == "convergence" and -1.35 < slope < -0.05)
                or (requested == "stagnation" and 0.05 < slope < 0.6)
                or (requested == "divergence" and 1.05 < slope < 1.35)):
            return losses
    raise AssertionError("Failed to sample an ordinary observation sequence")


def run_probe(original):
    sys.path.insert(0, str(ROOT))
    from dlrs import DLRSScheduler

    generator = random.Random(SEED)
    counts = dict.fromkeys(("convergence", "stagnation", "divergence"), 0)
    maximum = dict.fromkeys(("lr_absolute", "lr_relative", "slope_absolute",
                             "adjustment_absolute"), 0.0)
    exact_rates = compared_rates = 0
    minimum_decade_distance = math.inf
    maximum_epoch_error = 0
    default_cases = 0

    def close(actual, expected):
        assert math.isclose(actual, expected, rel_tol=RTOL, abs_tol=ATOL), (actual, expected)

    for case in range(CASES):
        rates = [generator.uniform(1.3, 8.3) * 10 ** generator.randint(-3, -1)
                 for _ in range(1 + case % 4)]
        config = {} if case % 2 == 0 else {
            "delta_d": generator.uniform(0.2, 0.6),
            "delta_o": generator.uniform(0.5, 1.0),
            "delta_i": generator.uniform(0.03, 0.15),
            "min_lr": 1e-8, "max_lr": 2.0,
        }
        default_cases += not config
        old_optimizer, new_optimizer = make_optimizer(rates), make_optimizer(rates)
        old, new = original(old_optimizer, **config), DLRSScheduler(new_optimizer, **config)
        assert old.last_epoch == new.last_epoch == 0
        for epoch in range(EPOCHS):
            requested = tuple(counts)[(case + epoch) % 3]
            losses = ordinary_observations(generator, requested)
            for group in old_optimizer.param_groups:
                value = group["lr"]
                log_rate = math.log10(value)
                distance = abs(log_rate - round(log_rate))
                minimum_decade_distance = min(minimum_decade_distance, distance)
                assert distance > 1e-9, "Unexpected decade-boundary input"
                assert 1e-7 < value < 1.9, "Unexpected configured-bound proximity"
            old.step(losses)
            new.step(losses)
            assert branch(old.loss_slope) == branch(new.loss_slope) == requested
            counts[requested] += 1
            close(new.loss_slope, old.loss_slope)
            close(new.adjustment, old.adjustment)
            maximum["slope_absolute"] = max(maximum["slope_absolute"], abs(new.loss_slope - old.loss_slope))
            maximum["adjustment_absolute"] = max(maximum["adjustment_absolute"], abs(new.adjustment - old.adjustment))
            maximum_epoch_error = max(maximum_epoch_error, abs(new.last_epoch - old.last_epoch))
            assert new.last_epoch == old.last_epoch
            assert new.batch_losses == old.batch_losses == losses
            assert new.base_lrs == old.base_lrs == rates
            for actual, expected in zip(new.get_last_lr(), old.get_last_lr()):
                close(actual, expected)
                difference = abs(actual - expected)
                maximum["lr_absolute"] = max(maximum["lr_absolute"], difference)
                maximum["lr_relative"] = max(maximum["lr_relative"], difference / abs(expected))
                exact_rates += actual == expected
                compared_rates += 1
    return {
        "trajectories": CASES, "default_configuration_trajectories": default_cases,
        "custom_typical_factor_trajectories": CASES - default_cases,
        "epochs_per_trajectory": EPOCHS, "parameter_groups_per_trajectory": [1, 2, 3, 4],
        "loss_sequences": CASES * EPOCHS, "rate_comparisons": compared_rates,
        "bitwise_equal_float_rate_comparisons": exact_rates,
        "branch_counts": counts, "maximum_differences": maximum,
        "maximum_last_epoch_difference": maximum_epoch_error,
        "minimum_log10_distance_from_integer_decade": minimum_decade_distance,
        "all_comparisons_within_tolerance": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="dlrs-original-differential-") as temporary:
        results = run_probe(load_baseline(Path(temporary)))
    report = {
        "scope": "Ordinary positive finite float learning rates and well-scaled positive losses; scheduler-only behavior, not training quality or universal compatibility.",
        "baseline_commit": COMMIT, "baseline_source_sha256": SOURCE_HASHES,
        "current_scheduler_sha256": hashlib.sha256((ROOT / "dlrs/scheduler.py").read_bytes()).hexdigest(),
        "current_utils_sha256": hashlib.sha256((ROOT / "dlrs/utils.py").read_bytes()).hexdigest(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": SEED, "tolerance": {"relative": RTOL, "absolute": ATOL},
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "platform": platform.platform(), "cpu_threads": 1, "interop_threads": 1},
        "results": results, "elapsed_seconds_including_source_fetch": time.monotonic() - started,
        "excluded_intentional_changes": [
            "zero rates remain frozen; tiny positive losses no longer trigger the old near-zero skip",
            "nonfinite, negative, boolean, malformed inputs/checkpoints are rejected atomically",
            "exact slope/decade boundaries and overflow/subnormal arithmetic have corrected handling",
            "tensor LR identity, initial/base metadata ownership, and checkpoint copying/schema are hardened",
            "parameter-group topology changes are rejected",
            "mode=max reverses scalar-wrapper direction and wrapper history is bounded",
            "step(None)/short/zero observations clear stale diagnostics; internal step counters are maintained",
            "verbose formatting and checkpoint fields intentionally differ",
        ],
        "not_tested": ["optimizer training quality", "actual parameter updates", "tensor LR behavior",
                       "invalid or extreme numeric inputs", "checkpoint schema compatibility",
                       "wrapper behavior", "compiler/framework integrations", "other Torch versions"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
