"""Malformed-evidence regressions; synthetic records are not workload results."""

import copy
import hashlib
import itertools
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks import check_workload_summary as check


def sha(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def fixture_evidence(transformer):
    policies = list(check.POLICIES)
    protocol = {"policies": policies, "tuning_seed": 0, "confirmation_seeds": [11, 12, 13, 14, 15]}
    if transformer:
        protocol.update(initial_lr_grid=[0.0003, 0.001, 0.003], corpus={"bytes": 100},
                        training={"scheduler_windows": 2, "total_optimizer_updates": 4,
                                  "training_target_bytes_per_trial": 32},
                        acceptance={"nll_noninferiority_margin_nats_per_byte": 0.05})
        manifest = {"protocol": protocol}
        manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        datasets = [None]
    else:
        protocol.update(vision={"datasets": ["MNIST", "CIFAR10"], "epochs": 2, "batch_size": 128,
                                "initial_lr_grid": [0.01, 0.03, 0.1], "dlrs_max_lr": 0.5},
                        pinn={"epochs": 2, "initial_lr_grid": [0.0001, 0.001, 0.01]},
                        validation_fraction=0.1, split_seed=42,
                        acceptance={"pinn_ratio_limit": 1.1, "vision_noninferiority_pp": 1.0})
        manifest = {"protocol": protocol, "python": "3.11", "torch": "2.14", "device": "mps",
                    "platform": "synthetic"}
        datasets = ["MNIST", "CIFAR10", "HelmholtzPINN"]
    records, summaries = [], []
    for dataset in datasets:
        pinn = dataset == "HelmholtzPINN"
        grid = protocol["initial_lr_grid"] if transformer else protocol["pinn" if pinn else "vision"]["initial_lr_grid"]
        chosen = grid[-1]
        means = {"NoScheduler": 3.0, "CosineAnnealingLR": 2.5, "DLRS": 2.0} if transformer else (
            {"NoScheduler": 1.2, "CosineAnnealingLR": 1.0, "DLRS": 0.8} if pinn else
            {"NoScheduler": 91.0, "CosineAnnealingLR": 92.0, "DLRS": 90.0})
        for phase, seeds in (("tuning", [0]), ("confirmation", protocol["confirmation_seeds"])):
            for seed in seeds:
                for policy_index, policy in enumerate(policies):
                    for rate in (grid if phase == "tuning" else [chosen]):
                        confirmation = phase == "confirmation"
                        duration = float(policy_index + 2)
                        result = {"initial_model_sha256": sha(seed)}
                        record = {"phase": phase, "seed": seed, "policy": policy, "status": "passed",
                                  "rate" if transformer else "lr": rate, "result": result}
                        validation = 4.0 - grid.index(rate)
                        if transformer:
                            record.update(manifest_sha256=manifest_hash, elapsed_seconds=duration + 0.1)
                            result.update(parameter_count=100, optimizer_updates=4, training_target_bytes=32,
                                          window_train_nll=[3.0, 2.0], window_learning_rates=[rate, rate],
                                          final_learning_rate=rate, elapsed_seconds=duration,
                                          validation={"nll": validation, "target_bytes": 9},
                                          training_index_stream_sha256=sha((seed, "indices")),
                                          training_split_sha256=sha("train"), validation_split_sha256=sha("val"))
                            if confirmation:
                                result.update(test={"nll": means[policy], "target_bytes": 9,
                                                    "perplexity": math.exp(means[policy])},
                                              test_split_sha256=sha("test"))
                        else:
                            record["dataset"] = dataset
                            result.update(dataset=dataset, scheduler=policy, seed=seed,
                                          train_loss=[3.0, 2.0], validation_loss=[3.0, validation],
                                          learning_rates=[rate, rate], total_time=duration)
                            if pinn:
                                result.update(initial_lr=rate, config=copy.deepcopy(protocol["pinn"]),
                                              environment={"device": "cpu", "torch": manifest["torch"]})
                                if confirmation:
                                    result["relative_rmse"] = means[policy]
                            else:
                                total = 60000 if dataset == "MNIST" else 50000
                                result.update(config={"epochs": 2, "batch_size": 128, "initial_lr": rate,
                                                      "validation_fraction": 0.1, "num_workers": 0,
                                                      "split_seed": 42, "max_lr": 0.5,
                                                      "train_examples": total * 9 // 10,
                                                      "validation_examples": total // 10,
                                                      "test_evaluated": confirmation},
                                              environment={key: manifest[key] for key in ("python", "torch", "device", "platform")},
                                              split_sha256=sha(dataset), validation_accuracy=[80.0, 90.0],
                                              best_validation_accuracy=90.0, next_learning_rates=[rate, rate],
                                              epoch_times=[0.1, 0.2])
                                if confirmation:
                                    result["final_accuracy"] = means[policy]
                            if confirmation:
                                result["test_loss"] = 1.0
                        records.append(record)
        item = {"selected_lrs": dict.fromkeys(policies, chosen), "policies": {}, "comparisons": []}
        if transformer:
            item.update(criterion=copy.deepcopy(protocol["acceptance"]), manifest_sha256=manifest_hash)
        else:
            item.update(dataset=dataset, metric="relative_rmse" if pinn else "final_accuracy")
        for policy in policies:
            item["policies"][policy] = ({"completed": 5, "failed_or_incomplete": 0, "mean_test_nll": means[policy]}
                                        if transformer else {"completed": 5, "failed": 0, "mean": means[policy], "sample_sd": 0.0})
        for index, baseline in enumerate(check.BASELINES):
            effect = means["DLRS"] / means[baseline] if pinn else means["DLRS"] - means[baseline]
            accepted = effect <= (0.05 if transformer else 1.1) if (transformer or pinn) else effect >= -1.0
            comparison = {"baseline": baseline, "accepted": accepted}
            if transformer:
                comparison.update(paired_seeds=5, mean_nll_difference=effect,
                                  percentile_bootstrap_95_interval=[effect, effect],
                                  paired_median_wall_time_ratio=4.0 / (index + 2))
            else:
                comparison.update(paired_effect=effect, bootstrap_95_interval=[effect, effect],
                                  median_time_ratio=4.0 / (index + 2))
            item["comparisons"].append(comparison)
        item["accepted"] = all(comparison["accepted"] for comparison in item["comparisons"])
        summaries.append(item)
    summary = summaries[0] if transformer else summaries
    if transformer:
        summary.update(recorded_trials=24, total_recorded_trial_seconds=sum(record["elapsed_seconds"] for record in records))
    return manifest, records, summary


def save_fixture(directory, evidence):
    manifest, records, summary = evidence
    (directory / "trials").mkdir()
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "summary.json").write_text(json.dumps(summary))
    for index, record in enumerate(records):
        (directory / "trials" / (str(index) + ".json")).write_text(json.dumps(record))


@pytest.mark.parametrize("transformer", [False, True])
def test_complete_summary_verifies_acceptance_and_rejection(tmp_path, transformer):
    evidence = fixture_evidence(transformer)
    save_fixture(tmp_path, evidence)
    result = check.verify(tmp_path)
    assert result["verified"] is True
    assert result["trial_count"] == (24 if transformer else 72)
    assert result["summary_file_sha256"] == hashlib.sha256((tmp_path / "summary.json").read_bytes()).hexdigest()
    assert len(result["comparisons"]) == (2 if transformer else 6)
    assert any(not comparison["accepted"] for comparison in result["comparisons"]) is (not transformer)


def test_multinomial_interval_matches_independent_ordered_resamples():
    values = [-0.7, 0.2, 0.3, 0.7, 1.2]
    ordered = sorted(statistics.mean(sample) for sample in itertools.product(values, repeat=5))
    expected = [ordered[math.ceil(probability * len(ordered)) - 1] for probability in (0.025, 0.975)]
    assert check.interval_from_counts(values) == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize("corruption", [
    "missing_comparisons", "missing_baseline", "duplicate_baseline", "empty_interval", "short_interval",
    "long_interval", "wrong_interval", "wrong_mean", "wrong_timing", "wrong_pair_count", "wrong_trial_count",
    "wrong_total_time", "wrong_criterion", "wrong_acceptance", "boolean_count", "wrong_selected_rate",
    "wrong_confirmation_rate", "duplicate_trial", "wrong_initialization", "wrong_indices", "changed_split",
    "wrong_target_coverage", "short_budget", "short_history", "nan_training", "nan_tuning", "test_in_tuning",
    "negative_nll", "wrong_perplexity", "failed_trial", "incomplete_summary", "wrong_manifest",
])
def test_malformed_transformer_evidence_is_rejected(tmp_path, corruption):
    evidence = fixture_evidence(True)
    _manifest, records, summary = evidence
    trial = next(record for record in records if record["phase"] == "confirmation")
    measured = trial["result"]
    comparison = summary["comparisons"][0]
    if corruption == "missing_comparisons":
        summary["comparisons"] = []
    elif corruption == "missing_baseline":
        summary["comparisons"].pop()
    elif corruption == "duplicate_baseline":
        summary["comparisons"][1] = copy.deepcopy(comparison)
    elif corruption in ("empty_interval", "short_interval", "long_interval"):
        count = {"empty_interval": 0, "short_interval": 1, "long_interval": 3}[corruption]
        comparison["percentile_bootstrap_95_interval"] = [comparison["mean_nll_difference"]] * count
    elif corruption == "wrong_interval":
        comparison["percentile_bootstrap_95_interval"][1] += 0.01
    elif corruption == "wrong_mean":
        summary["policies"]["DLRS"]["mean_test_nll"] += 0.1
    elif corruption == "wrong_timing":
        comparison["paired_median_wall_time_ratio"] += 0.1
    elif corruption == "wrong_pair_count":
        comparison["paired_seeds"] = 4
    elif corruption == "wrong_trial_count":
        summary["recorded_trials"] = 23
    elif corruption == "wrong_total_time":
        summary["total_recorded_trial_seconds"] += 1
    elif corruption == "wrong_criterion":
        summary["criterion"]["nll_noninferiority_margin_nats_per_byte"] = 0.5
    elif corruption == "wrong_acceptance":
        summary["accepted"] = False
    elif corruption == "boolean_count":
        summary["policies"]["DLRS"]["failed_or_incomplete"] = False
    elif corruption == "wrong_selected_rate":
        summary["selected_lrs"]["DLRS"] = 0.001
    elif corruption == "wrong_confirmation_rate":
        trial["rate"] = 0.001
        measured["window_learning_rates"][0] = 0.001
    elif corruption == "duplicate_trial":
        records[-1] = copy.deepcopy(records[-2])
    elif corruption == "wrong_initialization":
        measured["initial_model_sha256"] = sha("changed")
    elif corruption == "wrong_indices":
        measured["training_index_stream_sha256"] = sha("changed")
    elif corruption == "changed_split":
        measured["validation_split_sha256"] = sha("changed")
    elif corruption == "wrong_target_coverage":
        measured["test"]["target_bytes"] -= 1
    elif corruption == "short_budget":
        measured["optimizer_updates"] -= 1
    elif corruption == "short_history":
        measured["window_train_nll"].pop()
    elif corruption == "nan_training":
        measured["window_train_nll"][0] = float("nan")
    elif corruption == "nan_tuning":
        records[0]["result"]["validation"]["nll"] = float("nan")
    elif corruption == "test_in_tuning":
        records[0]["result"]["test"] = copy.deepcopy(measured["test"])
    elif corruption == "negative_nll":
        measured["test"]["nll"] = -1
    elif corruption == "wrong_perplexity":
        measured["test"]["perplexity"] += 1
    elif corruption == "failed_trial":
        trial["status"] = "failed"
    elif corruption == "incomplete_summary":
        summary["incomplete"] = True
    elif corruption == "wrong_manifest":
        trial["manifest_sha256"] = sha("wrong")
    else:
        raise AssertionError(corruption)
    save_fixture(tmp_path, evidence)
    with pytest.raises(ValueError):
        check.verify(tmp_path)


@pytest.mark.parametrize("corruption", [
    "duplicate_dataset", "missing_dataset", "wrong_metric", "wrong_sd", "wrong_nested_identity",
    "wrong_vision_budget", "wrong_pinn_budget", "test_in_tuning", "wrong_split", "discontinuous_rates",
    "bad_accuracy", "nonfinite_validation", "wrong_time_ratio",
])
def test_malformed_main_study_evidence_is_rejected(tmp_path, corruption):
    evidence = fixture_evidence(False)
    _manifest, records, summary = evidence
    measured = records[0]["result"]
    if corruption == "duplicate_dataset":
        summary.append(copy.deepcopy(summary[0]))
    elif corruption == "missing_dataset":
        summary.pop()
    elif corruption == "wrong_metric":
        summary[0]["metric"] = "validation_accuracy"
    elif corruption == "wrong_sd":
        summary[0]["policies"]["DLRS"]["sample_sd"] = 1
    elif corruption == "wrong_nested_identity":
        measured["seed"] = 999
    elif corruption == "wrong_vision_budget":
        measured["config"]["train_examples"] = 100
    elif corruption == "wrong_pinn_budget":
        next(record for record in records if record["dataset"] == "HelmholtzPINN")["result"]["config"]["epochs"] = 1
    elif corruption == "test_in_tuning":
        measured["test_loss"] = 1
    elif corruption == "wrong_split":
        measured["split_sha256"] = sha("changed")
    elif corruption == "discontinuous_rates":
        measured["next_learning_rates"][0] += 0.01
    elif corruption == "bad_accuracy":
        next(record for record in records if record["phase"] == "confirmation")["result"]["final_accuracy"] = 101
    elif corruption == "nonfinite_validation":
        measured["validation_loss"][0] = float("inf")
    elif corruption == "wrong_time_ratio":
        summary[2]["comparisons"][0]["median_time_ratio"] += 1
    else:
        raise AssertionError(corruption)
    save_fixture(tmp_path, evidence)
    with pytest.raises(ValueError):
        check.verify(tmp_path)


def test_optimized_python_cannot_disable_verification(tmp_path):
    evidence = fixture_evidence(True)
    evidence[2]["comparisons"] = []
    save_fixture(tmp_path, evidence)
    script = Path(check.__file__).resolve()
    result = subprocess.run([sys.executable, "-O", str(script), str(tmp_path)], capture_output=True,
                            text=True, timeout=10)
    assert result.returncode != 0
    assert "Expected exactly one comparison against each baseline" in result.stderr
    assert not (tmp_path / "independent-summary-check.json").exists()
