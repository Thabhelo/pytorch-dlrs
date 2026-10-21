"""Execute a frozen tuning/confirmation protocol, retaining every trial.

Each trial is saved atomically. Restarting resumes completed matching trials;
configuration or code changes are rejected rather than mixing evidence.
"""

import argparse
import hashlib
import itertools
import json
import math
import platform
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import torch

from benchmarks.run_benchmarks import run_experiment
from dlrs import DLRSScheduler
from examples.pinns_helmholtz import HelmholtzPINN, pinn_loss

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def code_fingerprint():
    paths = [
        "dlrs/scheduler.py",
        "dlrs/utils.py",
        "benchmarks/run_benchmarks.py",
        "benchmarks/validate_workloads.py",
        "examples/pinns_helmholtz.py",
    ]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}


def paired_interval(values):
    """Percentile interval from every n^n ordered bootstrap resample (n=5)."""
    if not values or len(values) > 6:
        raise ValueError("Expected between one and six paired observations")
    means = sorted(
        statistics.mean(sample) for sample in itertools.product(values, repeat=len(values))
    )
    return [
        means[max(0, math.ceil(0.025 * len(means)) - 1)],
        means[math.ceil(0.975 * len(means)) - 1],
    ]


def run_pinn(policy, lr, seed, config, evaluate_test):
    if policy not in {"DLRS", "CosineAnnealingLR", "NoScheduler"}:
        raise ValueError(f"Unknown policy: {policy}")
    torch.manual_seed(seed)
    model = HelmholtzPINN(hidden_layers=3, hidden_dim=32)
    initial_hash = hashlib.sha256(
        b"".join(p.detach().numpy().tobytes() for p in model.parameters())
    ).hexdigest()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if policy == "DLRS":
        scheduler = DLRSScheduler(optimizer, max_lr=config["dlrs_max_lr"])
    elif policy == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"])
    generator = torch.Generator().manual_seed(seed + 2000)
    boundary = torch.tensor([[0.0], [1.0]])
    validation_x = torch.linspace(0, 1, config["validation_points"]).view(-1, 1)
    rates, losses, validations = [], [], []
    start = time.perf_counter()
    for _epoch in range(config["epochs"]):
        rates.append(optimizer.param_groups[0]["lr"])
        observations = []
        model.train()
        for _ in range(config["batches_per_epoch"]):
            x = torch.rand(config["interior_points_per_batch"], 1, generator=generator)
            optimizer.zero_grad()
            loss = pinn_loss(model, x, boundary)
            value = loss.item()
            if not math.isfinite(value):
                raise ValueError("Nonfinite PINN loss before optimizer update")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), math.inf, error_if_nonfinite=True)
            optimizer.step()
            observations.append(value)
        if policy == "DLRS":
            scheduler.step(observations)
        elif scheduler is not None:
            scheduler.step()
        model.eval()
        validation = pinn_loss(model, validation_x, boundary).item()
        if not math.isfinite(validation):
            raise ValueError("Nonfinite PINN validation loss")
        losses.append(statistics.mean(observations))
        validations.append(validation)
    result = {
        "dataset": "HelmholtzPINN",
        "scheduler": policy,
        "seed": seed,
        "initial_model_sha256": initial_hash,
        "initial_lr": lr,
        "train_loss": losses,
        "validation_loss": validations,
        "learning_rates": rates,
        "config": config,
        "environment": {"device": "cpu", "torch": str(torch.__version__)},
    }
    if evaluate_test:
        x = ((torch.arange(config["test_points"]) + 0.5) / config["test_points"]).view(-1, 1)
        with torch.no_grad():
            truth = torch.sin(math.pi * x)
            error = model(x) - truth
            relative_rmse = (error.square().mean() / truth.square().mean()).sqrt().item()
        result["relative_rmse"] = relative_rmse
        result["test_loss"] = pinn_loss(model, x, boundary).item()
        if not all(math.isfinite(result[key]) for key in ("relative_rmse", "test_loss")):
            raise ValueError("Nonfinite PINN test metric")
    result["total_time"] = time.perf_counter() - start
    return result


def trial(output, dataset, phase, seed, policy, lr, protocol, device, data_root):
    path = output / "trials" / f"{dataset}-{phase}-s{seed}-{policy}-lr{lr!r}.json"
    if path.exists():
        cached = json.loads(path.read_text())
        expected = {"dataset": dataset, "phase": phase, "seed": seed, "policy": policy, "lr": lr}
        if any(cached.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Cached trial metadata mismatch: {path}")
        return cached
    print(f"START {dataset} {phase} seed={seed} policy={policy} lr={lr}", flush=True)
    start = time.perf_counter()
    try:
        if dataset == "HelmholtzPINN":
            result = run_pinn(policy, lr, seed, protocol["pinn"], phase == "confirmation")
        else:
            cfg = protocol["vision"]
            result = run_experiment(
                dataset,
                policy,
                torch.device(device),
                cfg["epochs"],
                cfg["batch_size"],
                lr,
                seed=seed,
                num_workers=0,
                validation_fraction=protocol["validation_fraction"],
                dataset_root=str(data_root),
                split_seed=protocol["split_seed"],
                evaluate_test=phase == "confirmation",
                max_lr=cfg["dlrs_max_lr"],
            )
        record = {
            "status": "passed",
            "dataset": dataset,
            "phase": phase,
            "seed": seed,
            "policy": policy,
            "lr": lr,
            "result": result,
        }
    except Exception as exc:
        record = {
            "status": "failed",
            "dataset": dataset,
            "phase": phase,
            "seed": seed,
            "policy": policy,
            "lr": lr,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.perf_counter() - start,
        }
    atomic_json(path, record)
    print(
        f"END {dataset} {phase} seed={seed} policy={policy} status={record['status']} "
        f"elapsed={time.perf_counter() - start:.1f}s",
        flush=True,
    )
    if device == "mps":
        torch.mps.empty_cache()
    return record


def summarize(records, selections, protocol):
    summaries = []
    for dataset in [*protocol["vision"]["datasets"], "HelmholtzPINN"]:
        dataset_records = [
            r for r in records if r["dataset"] == dataset and r["phase"] == "confirmation"
        ]
        if not dataset_records:
            continue
        metric = "relative_rmse" if dataset == "HelmholtzPINN" else "final_accuracy"
        item = {
            "dataset": dataset,
            "selected_lrs": selections[dataset],
            "metric": metric,
            "policies": {},
            "comparisons": [],
            "accepted": True,
        }
        for policy in protocol["policies"]:
            runs = [r for r in dataset_records if r["policy"] == policy]
            successful = [r for r in runs if r["status"] == "passed"]
            values = [r["result"][metric] for r in successful]
            item["policies"][policy] = {
                "completed": len(values),
                "failed": len(runs) - len(values),
                "mean": statistics.mean(values) if values else None,
                "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            }
            if len(successful) != len(protocol["confirmation_seeds"]):
                item["accepted"] = False
        for baseline in ("NoScheduler", "CosineAnnealingLR"):
            differences, ratios = [], []
            for seed in protocol["confirmation_seeds"]:
                matched = {
                    r["policy"]: r
                    for r in dataset_records
                    if r["seed"] == seed and r["status"] == "passed"
                }
                if "DLRS" not in matched or baseline not in matched:
                    continue
                dlrs, control = matched["DLRS"]["result"], matched[baseline]["result"]
                if dlrs["initial_model_sha256"] != control["initial_model_sha256"]:
                    raise ValueError("Paired trials did not share initialization")
                if dataset != "HelmholtzPINN" and dlrs["split_sha256"] != control["split_sha256"]:
                    raise ValueError("Paired trials did not share data splits")
                differences.append(
                    math.log(dlrs[metric] / control[metric])
                    if dataset == "HelmholtzPINN"
                    else dlrs[metric] - control[metric]
                )
                ratios.append(dlrs["total_time"] / control["total_time"])
            complete = len(differences) == len(protocol["confirmation_seeds"])
            interval = paired_interval(differences) if differences else None
            if interval is not None and dataset == "HelmholtzPINN":
                interval = [math.exp(value) for value in interval]
                accepted = complete and interval[1] <= protocol["acceptance"]["pinn_ratio_limit"]
                effect = math.exp(statistics.mean(differences))
            else:
                accepted = (
                    complete
                    and interval is not None
                    and interval[0] >= -protocol["acceptance"]["vision_noninferiority_pp"]
                )
                effect = statistics.mean(differences) if differences else None
            item["comparisons"].append(
                {
                    "baseline": baseline,
                    "paired_effect": effect,
                    "bootstrap_95_interval": interval,
                    "accepted": accepted,
                    "median_time_ratio": statistics.median(ratios) if ratios else None,
                }
            )
            item["accepted"] &= accepted
        summaries.append(item)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "benchmarks/validation_protocol.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/tmp/dlrs-validation-data"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["MNIST", "CIFAR10", "HelmholtzPINN"],
        default=["MNIST", "CIFAR10", "HelmholtzPINN"],
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    protocol = json.loads(args.protocol.read_text())
    manifest = {
        "protocol": protocol,
        "code_sha256": code_fingerprint(),
        "device": args.device,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": version("torchvision"),
        "numpy": version("numpy"),
        "platform": platform.platform(),
    }
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise SystemExit("Protocol/code/environment changed; use a new output directory")
    atomic_json(manifest_path, manifest)
    records = [
        json.loads(path.read_text()) for path in sorted((args.output / "trials").glob("*.json"))
    ]
    selections = {
        path.name[: -len("-selection.json")]: json.loads(path.read_text())
        for path in args.output.glob("*-selection.json")
    }
    for dataset in args.datasets:
        config = protocol["pinn"] if dataset == "HelmholtzPINN" else protocol["vision"]
        tuning = []
        for index, lr in enumerate(config["initial_lr_grid"]):
            policies = protocol["policies"][index:] + protocol["policies"][:index]
            for policy in policies:
                tuning.append(
                    trial(
                        args.output,
                        dataset,
                        "tuning",
                        protocol["tuning_seed"],
                        policy,
                        lr,
                        protocol,
                        args.device,
                        args.data_root,
                    )
                )
        records = [r for r in records if r["dataset"] != dataset]
        records.extend(tuning)
        selected = {}
        for policy in protocol["policies"]:
            candidates = [r for r in tuning if r["policy"] == policy and r["status"] == "passed"]
            if not candidates:
                atomic_json(args.output / "incomplete.json", records)
                raise SystemExit(f"No successful tuning candidate for {dataset}/{policy}")
            selected[policy] = min(candidates, key=lambda r: r["result"]["validation_loss"][-1])[
                "lr"
            ]
        selections[dataset] = selected
        atomic_json(args.output / f"{dataset}-selection.json", selected)
        print(f"SELECTED {dataset} {selected}", flush=True)
        for index, seed in enumerate(protocol["confirmation_seeds"]):
            offset = index % len(protocol["policies"])
            policies = protocol["policies"][offset:] + protocol["policies"][:offset]
            for policy in policies:
                records.append(
                    trial(
                        args.output,
                        dataset,
                        "confirmation",
                        seed,
                        policy,
                        selected[policy],
                        protocol,
                        args.device,
                        args.data_root,
                    )
                )
        summary = summarize(records, selections, protocol)
        atomic_json(args.output / "summary.json", summary)
        current_summary = next(item for item in summary if item["dataset"] == dataset)
        print(f"SUMMARY {json.dumps(current_summary)}", flush=True)
    summary = summarize(records, selections, protocol)
    atomic_json(args.output / "summary.json", summary)
    protocol_complete = {item["dataset"] for item in summary} == {
        *protocol["vision"]["datasets"],
        "HelmholtzPINN",
    }
    status = {
        "protocol_complete": protocol_complete,
        "execution_passed": all(r["status"] == "passed" for r in records),
        "quality_accepted": protocol_complete and all(item["accepted"] for item in summary),
        "completed_dataset_quality_accepted": bool(summary)
        and all(item["accepted"] for item in summary),
        "datasets_completed": [item["dataset"] for item in summary],
        "trial_count": len(records),
    }
    atomic_json(args.output / "status.json", status)
    print(f"STATUS {json.dumps(status)}", flush=True)
    if not status["execution_passed"] or not status["quality_accepted"]:
        raise SystemExit(
            "Validation gates did not all pass; see retained trial records and summary."
        )


if __name__ == "__main__":
    main()
