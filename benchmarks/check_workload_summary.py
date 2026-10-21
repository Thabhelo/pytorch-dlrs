"""Independently verify complete paired summaries and their saved evidence.

Uses multinomial weights over bootstrap count vectors, rather than the runners'
enumeration of ordered resamples. This checks saved arithmetic and consistency,
not execution provenance, population coverage, or unseen deployment safety.
"""

import argparse
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

POLICIES = ("NoScheduler", "CosineAnnealingLR", "DLRS")
BASELINES = POLICIES[:2]


def require(condition, message):
    # Explicit errors keep verification active under python -O as well.
    if not condition:
        raise ValueError(message)


def finite(value, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def read_json(path):
    def reject_constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(path.read_text(), parse_constant=reject_constant)


def interval_from_counts(values):
    count = len(values)
    require(1 <= count <= 6 and all(math.isfinite(value) for value in values),
            "Expected one to six finite paired observations")
    distribution = []
    for frequencies in itertools.product(range(count + 1), repeat=count):
        if sum(frequencies) != count:
            continue
        multiplicity = math.factorial(count)
        for frequency in frequencies:
            multiplicity //= math.factorial(frequency)
        mean = math.fsum(value * frequency for value, frequency in zip(values, frequencies)) / count
        distribution.append((mean, multiplicity))
    distribution.sort()
    require(sum(weight for _, weight in distribution) == count ** count, "Bootstrap weight mismatch")
    endpoints = []
    for probability in (0.025, 0.975):
        rank = math.ceil(probability * count ** count)
        accumulated = 0
        for mean, weight in distribution:
            accumulated += weight
            if accumulated >= rank:
                endpoints.append(mean)
                break
    return endpoints


def close(actual, expected):
    require(type(actual) in (int, float) and type(expected) in (int, float)
            and math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
            f"Numeric mismatch: {actual!r} != {expected!r}")


def integer(actual, expected):
    require(type(actual) is int and actual == expected, "Integer count mismatch")


def series(result, key, count, positive=False, maximum=None):
    values = result[key]
    require(isinstance(values, list) and len(values) == count
            and all(finite(value, positive) for value in values), "Invalid series: " + key)
    if maximum is not None:
        require(all(value <= maximum for value in values), "Out-of-range series: " + key)


def validate_record(record, protocol, manifest, transformer):
    require(record["status"] == "passed", "Verifier requires completed finite trials")
    require(record["policy"] in POLICIES and record["phase"] in ("tuning", "confirmation"),
            "Unknown policy or phase")
    seed = record["seed"]
    seeds = [protocol["tuning_seed"]] if record["phase"] == "tuning" else protocol["confirmation_seeds"]
    require(type(seed) is int and seed in seeds, "Unexpected trial seed")
    measured = record["result"]
    require(digest(measured["initial_model_sha256"]), "Invalid initialization fingerprint")
    confirmation = record["phase"] == "confirmation"
    rate = record["rate" if transformer else "lr"]
    require(finite(rate, positive=True), "Invalid initial learning rate")
    if transformer:
        cfg = protocol["training"]
        integer(measured["optimizer_updates"], cfg["total_optimizer_updates"])
        integer(measured["training_target_bytes"], cfg["training_target_bytes_per_trial"])
        require(type(measured["parameter_count"]) is int and measured["parameter_count"] > 0,
                "Invalid parameter count")
        count = cfg["scheduler_windows"]
        series(measured, "window_train_nll", count)
        series(measured, "window_learning_rates", count)
        close(measured["window_learning_rates"][0], rate)
        require(finite(measured["final_learning_rate"]), "Invalid final learning rate")
        require(finite(record["elapsed_seconds"], True) and finite(measured["elapsed_seconds"], True),
                "Invalid trial duration")
        length = protocol["corpus"]["bytes"]
        sizes = {"validation": length * 9 // 10 - length * 8 // 10 - 1,
                 "test": length - length * 9 // 10 - 1}
        for split in (["validation", "test"] if confirmation else ["validation"]):
            require(finite(measured[split]["nll"]), "Invalid held-out NLL")
            integer(measured[split]["target_bytes"], sizes[split])
        hashes = ["training_index_stream_sha256", "training_split_sha256", "validation_split_sha256"]
        if confirmation:
            hashes.append("test_split_sha256")
            close(measured["test"]["perplexity"], math.exp(measured["test"]["nll"]))
        else:
            require("test" not in measured and "test_split_sha256" not in measured,
                    "Tuning record contains test observations")
        require(all(digest(measured[key]) for key in hashes), "Invalid pairing fingerprint")
    else:
        dataset = record["dataset"]
        require(measured["dataset"] == dataset and measured["scheduler"] == record["policy"]
                and measured["seed"] == seed, "Nested trial identity mismatch")
        pinn = dataset == "HelmholtzPINN"
        cfg = protocol["pinn" if pinn else "vision"]
        count = cfg["epochs"]
        for key in ("train_loss", "validation_loss", "learning_rates"):
            series(measured, key, count)
        close(measured["learning_rates"][0], rate)
        require(finite(measured["total_time"], True), "Invalid trial duration")
        if pinn:
            require(measured["config"] == cfg, "PINN training budget/configuration differs")
            close(measured["initial_lr"], rate)
            require(measured["environment"] == {"device": "cpu", "torch": manifest["torch"]},
                    "PINN environment differs")
            if confirmation:
                require(finite(measured["relative_rmse"], True), "Invalid relative RMSE")
        else:
            require(dataset in ("MNIST", "CIFAR10"), "Unsupported vision dataset")
            total = 60000 if dataset == "MNIST" else 50000
            validation_size = max(1, int(total * protocol["validation_fraction"]))
            expected = {"epochs": count, "batch_size": cfg["batch_size"], "initial_lr": rate,
                        "validation_fraction": protocol["validation_fraction"], "num_workers": 0,
                        "split_seed": protocol["split_seed"], "max_lr": cfg["dlrs_max_lr"],
                        "train_examples": total - validation_size, "validation_examples": validation_size,
                        "test_evaluated": confirmation}
            require(measured["config"] == expected, "Vision training budget/configuration differs")
            require(measured["config"]["test_evaluated"] is confirmation, "Invalid test-evaluation flag")
            require(all(measured["environment"][key] == manifest[key]
                        for key in ("python", "torch", "device", "platform")), "Environment differs")
            require(digest(measured["split_sha256"]), "Invalid split fingerprint")
            series(measured, "validation_accuracy", count, maximum=100)
            series(measured, "next_learning_rates", count)
            series(measured, "epoch_times", count, positive=True)
            require(measured["learning_rates"][1:] == measured["next_learning_rates"][:-1],
                    "Discontinuous learning-rate history")
            close(measured["best_validation_accuracy"], max(measured["validation_accuracy"]))
            if confirmation:
                require(finite(measured["final_accuracy"]) and measured["final_accuracy"] <= 100,
                        "Invalid test accuracy")
        if confirmation:
            require(finite(measured["test_loss"]), "Invalid test loss")
        else:
            require(not any(key in measured for key in ("test_loss", "final_accuracy", "relative_rmse")),
                    "Tuning record contains test observations")


def verify(directory):
    manifest_path = directory / "manifest.json"
    summary_path = directory / "summary.json"
    manifest = read_json(manifest_path)
    protocol = manifest["protocol"]
    paths = sorted((directory / "trials").glob("*.json"))
    records = [read_json(path) for path in paths]
    summary = read_json(summary_path)
    transformer = "corpus" in protocol
    require(len(protocol["policies"]) == 3 and set(protocol["policies"]) == set(POLICIES),
            "Expected all three declared policies")
    seeds = protocol["confirmation_seeds"]
    require(len(seeds) == 5 and len(set(seeds)) == 5 and all(type(seed) is int for seed in seeds)
            and type(protocol["tuning_seed"]) is int and protocol["tuning_seed"] not in seeds,
            "Expected five distinct confirmation seeds separate from tuning")
    if transformer:
        require(isinstance(summary, dict) and not summary.get("incomplete"), "Incomplete Transformer summary")
        canonical_manifest = {key: value for key, value in manifest.items() if key != "created_utc"}
        identity = hashlib.sha256(json.dumps(canonical_manifest, sort_keys=True, allow_nan=False).encode()).hexdigest()
        require(summary["manifest_sha256"] == identity
                and all(record["manifest_sha256"] == identity for record in records), "Manifest identity mismatch")
        require(summary["criterion"] == protocol["acceptance"], "Acceptance criterion differs")
        integer(summary["recorded_trials"], len(records))
        close(summary["total_recorded_trial_seconds"], math.fsum(record["elapsed_seconds"] for record in records))
    summaries = [summary] if transformer else summary
    require(isinstance(summaries, list), "Expected a list of dataset summaries")
    expected_datasets = [None] if transformer else [*protocol["vision"]["datasets"], "HelmholtzPINN"]
    require(len(expected_datasets) == len(set(expected_datasets)), "Duplicate protocol dataset")
    require(len(records) == 24 * len(expected_datasets), "Study must be complete")
    require(len(summaries) == len(expected_datasets)
            and {item.get("dataset") for item in summaries} == set(expected_datasets),
            "Expected exactly one summary per dataset")
    for record in records:
        if not transformer:
            require(record["dataset"] in expected_datasets, "Unexpected trial dataset")
        validate_record(record, protocol, manifest, transformer)
    checked = []
    for item in summaries:
        dataset = item.get("dataset")
        subset = records if transformer else [record for record in records if record["dataset"] == dataset]
        require(set(item["policies"]) == set(POLICIES) and set(item["selected_lrs"]) == set(POLICIES),
                "Expected exactly the declared policies")
        comparisons = item["comparisons"]
        require(isinstance(comparisons, list) and len(comparisons) == len(BASELINES)
                and {comparison["baseline"] for comparison in comparisons} == set(BASELINES),
                "Expected exactly one comparison against each baseline")
        if not transformer:
            require(item["metric"] == ("relative_rmse" if dataset == "HelmholtzPINN" else "final_accuracy"),
                    "Summary metric differs")
        for seed in [protocol["tuning_seed"], *seeds]:
            same_seed = [record["result"] for record in subset if record["seed"] == seed]
            require(len({result["initial_model_sha256"] for result in same_seed}) == 1,
                    "Trials with the same seed differ in initialization")
            if transformer:
                require(len({result["training_index_stream_sha256"] for result in same_seed}) == 1,
                        "Trials with the same seed differ in training observations")
        split_keys = (["training_split_sha256", "validation_split_sha256"] if transformer else
                      ([] if dataset == "HelmholtzPINN" else ["split_sha256"]))
        for key in split_keys:
            require(len({record["result"][key] for record in subset}) == 1, "Data split changed across trials")
        if transformer:
            require(len({record["result"]["parameter_count"] for record in subset}) == 1, "Model size changed")
            require(len({record["result"]["test_split_sha256"] for record in subset
                         if record["phase"] == "confirmation"}) == 1, "Test split changed")
        for policy in POLICIES:
            tuning = [record for record in subset if record["phase"] == "tuning" and record["policy"] == policy]
            require(len(tuning) == 3, "Incomplete or duplicate tuning trials")
            metric = (lambda record: record["result"]["validation"]["nll"]) if transformer else (
                lambda record: record["result"]["validation_loss"][-1])
            rate_key = "rate" if transformer else "lr"
            config = protocol if transformer else protocol["pinn" if dataset == "HelmholtzPINN" else "vision"]
            grid = config["initial_lr_grid"]
            require(len(grid) == 3 and len(set(grid)) == 3 and all(finite(rate, True) for rate in grid)
                    and {record[rate_key] for record in tuning} == set(grid), "Tuning grid differs")
            chosen = min(tuning, key=lambda record: (metric(record), grid.index(record[rate_key])))[rate_key]
            require(chosen == item["selected_lrs"][policy], "Selected rate differs")
            confirmation = [record for record in subset if record["phase"] == "confirmation"
                            and record["policy"] == policy]
            require({record["seed"] for record in confirmation} == set(seeds) and len(confirmation) == len(seeds),
                    "Incomplete or duplicate confirmation trials")
            require(all(record[rate_key] == chosen for record in confirmation), "Confirmation rate differs")
            values = [(record["result"]["test"]["nll"] if transformer else
                       record["result"]["relative_rmse" if dataset == "HelmholtzPINN" else "final_accuracy"])
                      for record in confirmation]
            mean = math.fsum(values) / len(values)
            reported = item["policies"][policy]
            close(mean, reported["mean_test_nll" if transformer else "mean"])
            integer(reported["completed"], len(values))
            integer(reported["failed_or_incomplete" if transformer else "failed"], 0)
            if not transformer:
                sd = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))
                close(sd, reported["sample_sd"])
        for comparison in comparisons:
            baseline = comparison["baseline"]
            differences, ratios = [], []
            for seed in seeds:
                pair = {record["policy"]: record["result"] for record in subset
                        if record["phase"] == "confirmation" and record["seed"] == seed}
                if transformer:
                    difference = pair["DLRS"]["test"]["nll"] - pair[baseline]["test"]["nll"]
                elif dataset == "HelmholtzPINN":
                    difference = math.log(pair["DLRS"]["relative_rmse"] / pair[baseline]["relative_rmse"])
                else:
                    difference = pair["DLRS"]["final_accuracy"] - pair[baseline]["final_accuracy"]
                require(math.isfinite(difference), "Nonfinite paired effect")
                differences.append(difference)
                time_key = "elapsed_seconds" if transformer else "total_time"
                ratios.append(pair["DLRS"][time_key] / pair[baseline][time_key])
            interval = interval_from_counts(differences)
            effect = math.fsum(differences) / len(differences)
            if transformer:
                accepted = interval[1] <= protocol["acceptance"]["nll_noninferiority_margin_nats_per_byte"]
                effect_key, interval_key = "mean_nll_difference", "percentile_bootstrap_95_interval"
                integer(comparison["paired_seeds"], len(seeds))
                time_key = "paired_median_wall_time_ratio"
            else:
                effect_key, interval_key, time_key = "paired_effect", "bootstrap_95_interval", "median_time_ratio"
                if dataset == "HelmholtzPINN":
                    interval = [math.exp(value) for value in interval]
                    effect = math.exp(effect)
                    accepted = interval[1] <= protocol["acceptance"]["pinn_ratio_limit"]
                else:
                    accepted = interval[0] >= -protocol["acceptance"]["vision_noninferiority_pp"]
            close(effect, comparison[effect_key])
            reported_interval = comparison[interval_key]
            require(isinstance(reported_interval, list) and len(reported_interval) == 2,
                    "Expected exactly two interval endpoints")
            for actual, expected in zip(interval, reported_interval):
                close(actual, expected)
            close(statistics.median(ratios), comparison[time_key])
            require(type(comparison["accepted"]) is bool and accepted == comparison["accepted"],
                    "Comparison acceptance differs")
            checked.append({"dataset": dataset or "TinyShakespeare", "baseline": baseline,
                            "effect": effect, "interval": interval, "accepted": accepted})
        require(type(item["accepted"]) is bool
                and item["accepted"] == all(comparison["accepted"] for comparison in comparisons),
                "Overall acceptance differs")
    return {"verified": True, "trial_count": len(records),
            "method": "Multinomial-weighted bootstrap count vectors; paired values and tuning selected independently from raw trials",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "manifest_file_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "summary_file_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "trial_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
            "comparisons": checked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    evidence = verify(args.directory)
    output = args.directory / "independent-summary-check.json"
    output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in evidence.items() if key != "trial_sha256"}, indent=2))


if __name__ == "__main__":
    main()
