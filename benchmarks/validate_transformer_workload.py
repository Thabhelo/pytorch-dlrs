"""Separate predeclared, bounded CPU language-model comparison.

Use --prepare-only to pin the manifest without training. Scientific execution
uses a fixed protocol; changing source, protocol, data, or environment rejects
an existing output directory instead of mixing trial evidence.
"""

import argparse
import hashlib
import inspect
import itertools
import json
import math
import platform
import statistics
import time
import urllib.request
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
from torch import nn

from dlrs import DLRSScheduler
from dlrs.utils import _validate_scalar

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "benchmarks/transformer_validation_protocol.json"
POLICIES = ("NoScheduler", "CosineAnnealingLR", "DLRS")


class StudyBudgetReached(RuntimeError):
    """The fixed wall-clock budget expired; this is incomplete evidence."""


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_fingerprints():
    # Installed-artifact tests intentionally remove ROOT/dlrs. Fingerprint the
    # source actually imported, not a potentially different checkout copy.
    sources = {"dlrs/scheduler.py": Path(inspect.getsourcefile(DLRSScheduler)),
               "dlrs/utils.py": Path(inspect.getsourcefile(_validate_scalar)),
               "benchmarks/validate_transformer_workload.py": Path(__file__)}
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()}


def check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise StudyBudgetReached("Predeclared total trial wall-clock budget exhausted")


def validate_protocol(protocol):
    if tuple(protocol["policies"]) != POLICIES:
        raise ValueError("The protocol must declare all three policies in their fixed order")
    model, training = protocol["model"], protocol["training"]
    for name in ("layers", "width", "heads", "feedforward_width", "context", "vocabulary"):
        if isinstance(model[name], bool) or not isinstance(model[name], int) or model[name] <= 0:
            raise ValueError("Model dimensions must be positive integers")
    if model["vocabulary"] != 256 or model["width"] % model["heads"]:
        raise ValueError("Use the fixed byte vocabulary and divisible attention dimensions")
    if not 0 <= model["dropout"] < 1:
        raise ValueError("Dropout must be in [0,1)")
    for key in ("batch_size", "scheduler_windows", "updates_per_window", "threads"):
        if isinstance(training[key], bool) or not isinstance(training[key], int) or training[key] <= 0:
            raise ValueError("Training counts must be positive integers")
    if training["device"] != "cpu" or training["dtype"] != "float32" or training["threads"] > 2:
        raise ValueError("This protocol is limited to CPU float32 and at most two threads")
    if training["updates_per_window"] < 2:
        raise ValueError("DLRS requires at least two observations per scheduler window")
    updates = training["scheduler_windows"] * training["updates_per_window"]
    if training["total_optimizer_updates"] != updates:
        raise ValueError("Declared update budget disagrees with scheduler windows")
    if training["training_target_bytes_per_trial"] != updates * training["batch_size"] * model["context"]:
        raise ValueError("Declared token budget disagrees with training dimensions")
    rates = protocol["initial_lr_grid"]
    if len(rates) != 3 or len(set(rates)) != 3 or not all(math.isfinite(v) and v > 0 for v in rates):
        raise ValueError("Exactly three distinct finite positive candidate LRs are required")
    if not training["dlrs_min_lr"] <= min(rates) <= max(rates) <= training["dlrs_max_lr"]:
        raise ValueError("Candidate rates must fit the declared DLRS bounds")
    if len(protocol["confirmation_seeds"]) != 5 or len(set(protocol["confirmation_seeds"])) != 5:
        raise ValueError("Exactly five distinct paired confirmation seeds are required")
    if protocol["tuning_seed"] in protocol["confirmation_seeds"]:
        raise ValueError("Tuning and confirmation seeds must be separate")
    margin = protocol["acceptance"]["nll_noninferiority_margin_nats_per_byte"]
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("The NLL margin must be finite and nonnegative")
    budget = training["wall_clock_budget_seconds"]
    if budget is not None and (isinstance(budget, bool) or not math.isfinite(budget) or budget <= 0):
        raise ValueError("Optional wall-clock limit must be positive and finite")


class ByteCorpus:
    """Read only the requested split as tokens; full-file hashing is integrity checking."""

    def __init__(self, path, specification):
        self.path = Path(path)
        digest = hashlib.sha256()
        length = 0
        with self.path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                length += len(chunk)
        if length != specification["bytes"] or digest.hexdigest() != specification["sha256"]:
            raise ValueError("Corpus checksum/length differs from the predeclared source")
        self.sha256 = digest.hexdigest()
        self.bounds = {"train": (0, length * 8 // 10),
                       "validation": (length * 8 // 10, length * 9 // 10),
                       "test": (length * 9 // 10, length)}
        self.opened_splits = []

    def read_split(self, split):
        start, end = self.bounds[split]
        with self.path.open("rb") as stream:
            stream.seek(start)
            raw = stream.read(end - start)
        if len(raw) != end - start:
            raise ValueError("Corpus changed after verification")
        self.opened_splits.append(split)
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8).long(), hashlib.sha256(raw).hexdigest()


class ByteTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, context = config["width"], config["context"]
        self.token = nn.Embedding(config["vocabulary"], width)
        self.position = nn.Embedding(context, width)
        # Construct each layer independently: Encoder's clone constructor would
        # otherwise start every layer with identical parameter values.
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(width, config["heads"], config["feedforward_width"],
                                       config["dropout"], "gelu", batch_first=True, norm_first=True)
            for _ in range(config["layers"])
        ])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, config["vocabulary"], bias=False)
        self.register_buffer("mask", torch.triu(torch.ones(context, context, dtype=torch.bool), diagonal=1))
        self.apply(self._initialize)
        for block in self.blocks:
            nn.init.normal_(block.self_attn.in_proj_weight, std=0.02)
            nn.init.zeros_(block.self_attn.in_proj_bias)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, tokens):
        length = tokens.shape[1]
        if not 0 < length <= self.mask.shape[0]:
            raise ValueError("Input context length is outside the declared model window")
        hidden = self.token(tokens) + self.position(torch.arange(length, device=tokens.device))
        for block in self.blocks:
            hidden = block(hidden, src_mask=self.mask[:length, :length])
        return self.output(self.norm(hidden))


def evaluate_nll(model, tokens, context, batch_size, deadline=None):
    """Score every target once, including the final partial context block."""
    if len(tokens) < 2:
        raise ValueError("Evaluation split needs at least two bytes")
    modes = [(module, module.training) for module in model.modules()]
    total_loss, targets = 0.0, 0
    try:
        model.eval()
        with torch.no_grad():
            full_count = (len(tokens) - 1) // context
            windows = tokens.unfold(0, context + 1, context) if full_count else None
            batches = (windows[index:index + batch_size] for index in range(0, full_count, batch_size))
            for windows_batch in batches:
                check_deadline(deadline)
                logits = model(windows_batch[:, :-1])
                loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                   windows_batch[:, 1:].reshape(-1), reduction="sum")
                value = float(loss)
                if not math.isfinite(value) or value < 0:
                    raise ValueError("Nonfinite/negative evaluation NLL")
                total_loss += value
                targets += windows_batch[:, 1:].numel()
            tail = tokens[full_count * context:]
            if len(tail) > 1:
                check_deadline(deadline)
                logits = model(tail[:-1].unsqueeze(0))
                value = float(nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                           tail[1:], reduction="sum"))
                if not math.isfinite(value) or value < 0:
                    raise ValueError("Nonfinite/negative evaluation NLL")
                total_loss += value
                targets += len(tail) - 1
    finally:
        for module, training in modes:
            module.training = training
    if targets != len(tokens) - 1 or not math.isfinite(total_loss):
        raise ValueError("Evaluation coverage or accumulated NLL is invalid")
    return {"nll": total_loss / targets, "target_bytes": targets}


def training_update(model, optimizer, inputs, targets, clip_norm):
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    value = float(loss.detach())
    if not math.isfinite(value) or value < 0:
        raise ValueError("Nonfinite/negative training loss before optimizer update")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
    optimizer.step()
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise ValueError("Nonfinite parameter after optimizer update")
    return value


def run_trial(policy, rate, seed, protocol, corpus, evaluate_test, deadline=None):
    if policy not in POLICIES:
        raise ValueError("Unknown scheduler policy")
    check_deadline(deadline)
    started = time.monotonic()
    torch.manual_seed(seed)
    cfg, training = protocol["model"], protocol["training"]
    model = ByteTransformer(cfg)
    initial_hash = hashlib.sha256(b"".join(parameter.detach().numpy().tobytes()
                                           for parameter in model.parameters())).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, betas=tuple(training["betas"]),
                                  weight_decay=training["weight_decay"], eps=training["epsilon"], foreach=False)
    scheduler = None
    if policy == "DLRS":
        scheduler = DLRSScheduler(optimizer, min_lr=training["dlrs_min_lr"],
                                  max_lr=training["dlrs_max_lr"], **training["dlrs_factors"])
    elif policy == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, training["scheduler_windows"])
    tokens, train_hash = corpus.read_split("train")
    if len(tokens) <= cfg["context"]:
        raise ValueError("Training split is shorter than one context/target window")
    generator = torch.Generator().manual_seed(seed + 1000)
    sample_hash = hashlib.sha256()
    offsets = torch.arange(cfg["context"] + 1)
    means, rates = [], []
    model.train()
    for _ in range(training["scheduler_windows"]):
        rates.append(float(optimizer.param_groups[0]["lr"]))
        observations = []
        for _ in range(training["updates_per_window"]):
            check_deadline(deadline)
            starts = torch.randint(len(tokens) - cfg["context"], (training["batch_size"],), generator=generator)
            sample_hash.update(starts.numpy().tobytes())
            batch = tokens[starts[:, None] + offsets]
            observations.append(training_update(model, optimizer, batch[:, :-1], batch[:, 1:],
                                                 training["gradient_clip_norm"]))
        means.append(statistics.mean(observations))
        if policy == "DLRS":
            scheduler.step(observations)
        elif scheduler is not None:
            scheduler.step()
        if not math.isfinite(float(optimizer.param_groups[0]["lr"])) or optimizer.param_groups[0]["lr"] < 0:
            raise ValueError("Nonfinite/negative learning rate")
    validation, validation_hash = corpus.read_split("validation")
    validation_result = evaluate_nll(model, validation, cfg["context"], protocol["evaluation"]["batch_size"], deadline)
    result = {"initial_model_sha256": initial_hash, "training_index_stream_sha256": sample_hash.hexdigest(),
              "training_split_sha256": train_hash, "validation_split_sha256": validation_hash,
              "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
              "optimizer_updates": training["total_optimizer_updates"],
              "training_target_bytes": training["training_target_bytes_per_trial"],
              "window_train_nll": means, "window_learning_rates": rates,
              "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
              "validation": validation_result}
    if evaluate_test:
        test, test_hash = corpus.read_split("test")
        result["test"] = evaluate_nll(model, test, cfg["context"], protocol["evaluation"]["batch_size"], deadline)
        result["test"]["perplexity"] = math.exp(result["test"]["nll"])
        result["test_split_sha256"] = test_hash
    result["elapsed_seconds"] = time.monotonic() - started
    check_deadline(deadline)
    return result


def choose_rates(records, protocol):
    selected = {}
    for policy in POLICIES:
        candidates = [record for record in records if record["phase"] == "tuning" and record["policy"] == policy]
        if len(candidates) != len(protocol["initial_lr_grid"]):
            raise ValueError("Every policy must receive the complete declared tuning budget")
        if any("test" in record.get("result", {}) for record in candidates):
            raise ValueError("Tuning records must not contain test measurements")
        if {record["rate"] for record in candidates} != set(protocol["initial_lr_grid"]):
            raise ValueError("Tuning rates differ from the predeclared grid")
        successful = [record for record in candidates if record["status"] == "passed"]
        if not successful:
            raise ValueError("All tuning trials failed for " + policy)
        best = min(successful, key=lambda record: (record["result"]["validation"]["nll"],
                                                   protocol["initial_lr_grid"].index(record["rate"])))
        selected[policy] = best["rate"]
    return selected


def paired_interval(values):
    if len(values) != 5 or not all(math.isfinite(value) for value in values):
        raise ValueError("Exactly five finite paired effects are required")
    means = sorted(statistics.mean(sample) for sample in itertools.product(values, repeat=5))
    return [means[math.ceil(0.025 * len(means)) - 1], means[math.ceil(0.975 * len(means)) - 1]]


def summarize(records, selected, protocol):
    confirmation = [record for record in records if record["phase"] == "confirmation"]
    seeds = protocol["confirmation_seeds"]
    result = {"criterion": protocol["acceptance"], "selected_lrs": selected,
              "accepted": True, "comparisons": [], "policies": {}}
    for policy in POLICIES:
        matches = [record for record in confirmation if record["policy"] == policy]
        passed = [record for record in matches if record["status"] == "passed"]
        complete = len(passed) == len(seeds) and {record["seed"] for record in passed} == set(seeds)
        result["policies"][policy] = {"completed": len(passed), "failed_or_incomplete": len(matches) - len(passed),
                                      "mean_test_nll": statistics.mean(record["result"]["test"]["nll"] for record in passed) if passed else None}
        result["accepted"] &= complete
    for baseline in POLICIES[:2]:
        differences, ratios = [], []
        for seed in seeds:
            matched = {record["policy"]: record["result"] for record in confirmation
                       if record["seed"] == seed and record["status"] == "passed"}
            if "DLRS" not in matched or baseline not in matched:
                continue
            candidate, control = matched["DLRS"], matched[baseline]
            for key in ("initial_model_sha256", "training_index_stream_sha256", "training_split_sha256",
                        "validation_split_sha256", "test_split_sha256"):
                if candidate[key] != control[key]:
                    raise ValueError("Paired trials differ in " + key)
            differences.append(candidate["test"]["nll"] - control["test"]["nll"])
            ratios.append(candidate["elapsed_seconds"] / control["elapsed_seconds"])
        interval = paired_interval(differences) if len(differences) == len(seeds) else None
        passed = interval is not None and interval[1] <= protocol["acceptance"]["nll_noninferiority_margin_nats_per_byte"]
        result["comparisons"].append({"baseline": baseline, "paired_seeds": len(differences),
                                      "mean_nll_difference": statistics.mean(differences) if differences else None,
                                      "percentile_bootstrap_95_interval": interval,
                                      "paired_median_wall_time_ratio": statistics.median(ratios) if ratios else None,
                                      "accepted": passed})
        result["accepted"] &= passed
    return result


def prepare(protocol, data_root, output):
    validate_protocol(protocol)
    data_root.mkdir(parents=True, exist_ok=True)
    corpus_path = data_root / protocol["corpus"]["filename"]
    if not corpus_path.exists():
        with urllib.request.urlopen(protocol["corpus"]["url"], timeout=60) as response:
            raw = response.read(protocol["corpus"]["bytes"] + 1)
        if len(raw) != protocol["corpus"]["bytes"] or hashlib.sha256(raw).hexdigest() != protocol["corpus"]["sha256"]:
            raise ValueError("Downloaded corpus differs from declared checksum/length")
        corpus_path.write_bytes(raw)
    corpus = ByteCorpus(corpus_path, protocol["corpus"])
    manifest = {"protocol": protocol, "protocol_sha256": fingerprint(protocol),
                "source_sha256": source_fingerprints(),
                "corpus_sha256": corpus.sha256, "split_byte_ranges": corpus.bounds,
                "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                                "numpy": version("numpy"),
                                "platform": platform.platform(), "device": "cpu", "dtype": "float32",
                                "threads": torch.get_num_threads(),
                                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}}
    # Normalize tuples to their JSON representation before a restart comparison.
    manifest = json.loads(json.dumps(manifest))
    path = output / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text())
        existing.pop("created_utc", None)
        if existing != manifest:
            raise ValueError("Frozen study manifest differs; use a separate output directory")
    else:
        atomic_json(path, dict(manifest, created_utc=datetime.now(timezone.utc).isoformat()))
    return corpus, fingerprint(manifest)


def validate_record(record, identity, protocol):
    if not isinstance(record, dict) or any(record.get(key) != value for key, value in identity.items()):
        raise ValueError("Cached trial identity differs from the frozen study")

    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0

    if record.get("status") not in ("passed", "failed", "incomplete") or not finite(record.get("elapsed_seconds")):
        raise ValueError("Cached trial status/timing is invalid")
    if record["status"] != "passed":
        if not isinstance(record.get("error"), str) or not record["error"]:
            raise ValueError("Cached unsuccessful trial lacks an error record")
        return
    measured = record.get("result")
    if not isinstance(measured, dict):
        raise ValueError("Cached successful trial lacks measurements")
    cfg = protocol["training"]
    for key, expected in (("optimizer_updates", cfg["total_optimizer_updates"]),
                          ("training_target_bytes", cfg["training_target_bytes_per_trial"])):
        if type(measured.get(key)) is not int or measured[key] != expected:
            raise ValueError("Cached trial does not match the declared " + key)
    if not finite(measured.get("elapsed_seconds")) or measured["elapsed_seconds"] <= 0:
        raise ValueError("Cached trial duration must be positive")
    if type(measured.get("parameter_count")) is not int or measured["parameter_count"] <= 0:
        raise ValueError("Cached model size is invalid")
    for key in ("window_train_nll", "window_learning_rates"):
        values = measured.get(key)
        if not isinstance(values, list) or len(values) != cfg["scheduler_windows"] or not all(map(finite, values)):
            raise ValueError("Cached training windows are invalid")
    if not finite(measured.get("final_learning_rate")):
        raise ValueError("Cached final learning rate is invalid")
    length = protocol["corpus"]["bytes"]
    targets = {"validation": length * 9 // 10 - length * 8 // 10 - 1,
               "test": length - length * 9 // 10 - 1}
    splits = ["validation", "test"] if identity["phase"] == "confirmation" else ["validation"]
    for split in splits:
        metric = measured.get(split)
        if (not isinstance(metric, dict) or not finite(metric.get("nll"))
                or metric.get("target_bytes") != targets[split]):
            raise ValueError("Cached evaluation metric/target coverage is invalid")
    if identity["phase"] == "tuning" and ("test" in measured or "test_split_sha256" in measured):
        raise ValueError("Cached tuning trial contains test measurements")
    if identity["phase"] == "confirmation" and not finite(measured["test"].get("perplexity")):
        raise ValueError("Cached test perplexity is invalid")
    hashes = ["initial_model_sha256", "training_index_stream_sha256", "training_split_sha256", "validation_split_sha256"]
    if identity["phase"] == "confirmation":
        hashes.append("test_split_sha256")
    for key in hashes:
        digest = measured.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("Cached pairing fingerprint is invalid")


def stored_trial(output, phase, seed, policy, rate, protocol, corpus, manifest_hash, remaining):
    identity = {"phase": phase, "seed": seed, "policy": policy, "rate": rate, "manifest_sha256": manifest_hash}
    path = output / "trials" / f"{phase}-s{seed}-{policy}-lr{rate!r}.json"
    if path.exists():
        record = json.loads(path.read_text())
        validate_record(record, identity, protocol)
        return record
    started = time.monotonic()
    print(f"START {phase} seed={seed} policy={policy} lr={rate}", flush=True)
    try:
        deadline = None if remaining is None else started + remaining
        measured = run_trial(policy, rate, seed, protocol, corpus, phase == "confirmation", deadline)
        record = dict(identity, status="passed", result=measured)
    except StudyBudgetReached as error:
        record = dict(identity, status="incomplete", error=str(error))
    except Exception as error:
        record = dict(identity, status="failed", error=f"{type(error).__name__}: {error}")
    record["elapsed_seconds"] = time.monotonic() - started
    validate_record(record, identity, protocol)
    atomic_json(path, record)
    print(f"END status={record['status']} elapsed={record['elapsed_seconds']:.2f}s", flush=True)
    return record


def synthetic_timing(protocol):
    """Time representative shapes using random bytes only; no corpus access or quality output."""
    torch.manual_seed(713)
    cfg, training = protocol["model"], protocol["training"]
    model = ByteTransformer(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False)
    generator = torch.Generator().manual_seed(719)
    batch = torch.randint(256, (training["batch_size"], cfg["context"] + 1), generator=generator)
    times = []
    for index in range(20):
        started = time.monotonic()
        training_update(model, optimizer, batch[:, :-1], batch[:, 1:], training["gradient_clip_norm"])
        if index >= 4:
            times.append(time.monotonic() - started)
    eval_times = {}
    length = protocol["corpus"]["bytes"]
    for split, size in (("validation", length * 9 // 10 - length * 8 // 10),
                        ("test", length - length * 9 // 10)):
        tokens = torch.randint(256, (size,), generator=generator)
        started = time.monotonic()
        evaluate_nll(model, tokens, cfg["context"], protocol["evaluation"]["batch_size"])
        eval_times[split] = time.monotonic() - started
    projected = (24 * statistics.median(times) * training["total_optimizer_updates"]
                 + 24 * eval_times["validation"] + 15 * eval_times["test"])
    return {"scope": "Synthetic uniformly random bytes, no real corpus access or quality metrics",
            "protocol_sha256": fingerprint(protocol), "source_sha256": source_fingerprints(),
            "torch": str(torch.__version__), "python": platform.python_version(),
            "platform": platform.platform(), "threads": torch.get_num_threads(),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "warmup_updates": 4, "timed_update_seconds": times,
            "median_update_seconds": statistics.median(times), "evaluation_seconds": eval_times,
            "projected_24_trial_seconds": projected,
            "qualification": "Rough planning estimate under possible concurrent work; excludes startup and scheduler/IO costs, not a runtime guarantee."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--data-root", type=Path, default=Path("/tmp/dlrs-validation-data"))
    parser.add_argument("--output", type=Path, default=ROOT / "docs/transformer-validation-2026-10-20")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--synthetic-timing", action="store_true", help="Time synthetic shapes without preparing/accessing corpus")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    validate_protocol(protocol)
    torch.set_num_threads(protocol["training"]["threads"])
    torch.use_deterministic_algorithms(True)
    if args.synthetic_timing:
        result = synthetic_timing(protocol)
        atomic_json(args.output / "synthetic_timing.json", result)
        print(json.dumps(result, indent=2))
        return 0
    corpus, manifest_hash = prepare(protocol, args.data_root, args.output)
    if args.prepare_only:
        print("Manifest prepared; no training or held-out evaluation performed.")
        return 0
    records = []
    budget = protocol["training"]["wall_clock_budget_seconds"]

    def execute(phase, seed, policy, rate):
        remaining = None if budget is None else budget - sum(record["elapsed_seconds"] for record in records)
        if remaining is not None and remaining <= 0:
            raise StudyBudgetReached("Total recorded trial budget exhausted")
        record = stored_trial(args.output, phase, seed, policy, rate, protocol, corpus, manifest_hash, remaining)
        records.append(record)
        if record["status"] == "incomplete":
            raise StudyBudgetReached(record["error"])

    selected = {}
    try:
        for index, rate in enumerate(protocol["initial_lr_grid"]):
            order = POLICIES[index % 3:] + POLICIES[:index % 3]
            for policy in order:
                execute("tuning", protocol["tuning_seed"], policy, rate)
        selected = choose_rates(records, protocol)
        atomic_json(args.output / "selected_lrs.json", {"manifest_sha256": manifest_hash, "rates": selected,
                                                       "selection_metric": "final validation NLL"})
        for index, seed in enumerate(protocol["confirmation_seeds"]):
            order = POLICIES[index % 3:] + POLICIES[:index % 3]
            for policy in order:
                execute("confirmation", seed, policy, selected[policy])
        summary = summarize(records, selected, protocol)
    except (StudyBudgetReached, ValueError) as error:
        summary = {"accepted": False, "incomplete": True, "reason": str(error), "selected_lrs": selected}
    summary.update(manifest_sha256=manifest_hash, recorded_trials=len(records),
                   total_recorded_trial_seconds=sum(record["elapsed_seconds"] for record in records))
    expected_trials = len(POLICIES) * (len(protocol["initial_lr_grid"]) + len(protocol["confirmation_seeds"]))
    summary["execution_passed"] = (not summary.get("incomplete", False)
                                   and len(records) == expected_trials
                                   and all(record["status"] == "passed" for record in records))
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["execution_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
