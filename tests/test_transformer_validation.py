"""Scientific validity guards; synthetic fixtures are not workload evidence."""

import copy
import hashlib
import json
import math
import sys
import time

import pytest
import torch

from benchmarks import validate_transformer_workload as study


@pytest.fixture(scope="module", autouse=True)
def limit_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(original)


def tiny_protocol():
    protocol = json.loads(study.DEFAULT_PROTOCOL.read_text())
    protocol["model"].update(layers=1, width=8, heads=2, feedforward_width=16, context=4)
    protocol["training"].update(batch_size=2, scheduler_windows=2, updates_per_window=2,
                                total_optimizer_updates=4, training_target_bytes_per_trial=32)
    protocol["evaluation"]["batch_size"] = 2
    return protocol


def corpus_fixture(tmp_path, protocol):
    raw = bytes(range(64)) * 4
    protocol["corpus"].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    path = tmp_path / protocol["corpus"]["filename"]
    path.write_bytes(raw)
    return study.ByteCorpus(path, protocol["corpus"])


@pytest.mark.parametrize("training", [False, True])
def test_causal_attention_cannot_read_future_tokens(training):
    torch.manual_seed(7)
    config = tiny_protocol()["model"]
    config["context"] = 8
    config["dropout"] = 0.0
    model = study.ByteTransformer(config).train(training)
    original = torch.arange(8).unsqueeze(0)
    altered = original.clone()
    altered[:, 4:] += 40
    with torch.no_grad():
        actual, changed = model(original), model(altered)
    torch.testing.assert_close(actual[:, :4], changed[:, :4], rtol=0, atol=1e-7)
    assert not torch.equal(actual[:, 4:], changed[:, 4:])


@pytest.mark.parametrize("length", [2, 8, 9, 10, 17, 18])
def test_evaluation_covers_each_target_once_including_partial_blocks(length):
    class UniformModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward(self, tokens):
            self.inputs.extend(tokens.flatten().tolist())
            return torch.zeros(*tokens.shape, 256)

    model = UniformModel().train()
    measured = study.evaluate_nll(model, torch.arange(length), context=8, batch_size=2)
    assert measured["target_bytes"] == length - 1
    assert measured["nll"] == pytest.approx(math.log(256), rel=1e-6)
    assert model.inputs == list(range(length - 1))
    assert model.training


def test_evaluation_uses_next_byte_targets_not_inputs_or_another_shift():
    class NextByteModel(torch.nn.Module):
        def forward(self, tokens):
            logits = torch.zeros(*tokens.shape, 256, dtype=torch.float64)
            return logits.scatter(-1, ((tokens + 1) % 256).unsqueeze(-1), 7.0)

    measured = study.evaluate_nll(NextByteModel(), torch.arange(18), context=8, batch_size=2)
    expected = math.log(math.exp(7.0) + 255) - 7.0
    assert measured["nll"] == pytest.approx(expected, rel=1e-6)


def test_all_policies_share_initialization_and_samples_without_tuning_test_access(tmp_path, monkeypatch):
    protocol = tiny_protocol()
    corpus = corpus_fixture(tmp_path, protocol)
    original = corpus.read_split

    def guarded_read(split):
        assert split != "test", "Tuning must not construct the held-out token tensor"
        return original(split)

    monkeypatch.setattr(corpus, "read_split", guarded_read)
    records = [study.run_trial(policy, 0.001, 0, protocol, corpus, False) for policy in study.POLICIES]
    for key in ("initial_model_sha256", "training_index_stream_sha256", "training_split_sha256", "validation_split_sha256"):
        assert len({record[key] for record in records}) == 1
    assert all("test" not in record and "test_split_sha256" not in record for record in records)
    repeat = study.run_trial("DLRS", 0.001, 0, protocol, corpus, False)
    assert repeat["window_train_nll"] == records[-1]["window_train_nll"]
    assert repeat["validation"] == records[-1]["validation"]
    assert corpus.opened_splits == ["train", "validation"] * 4


def test_confirmation_scores_test_only_after_training_and_validation(tmp_path):
    protocol = tiny_protocol()
    corpus = corpus_fixture(tmp_path, protocol)
    record = study.run_trial("NoScheduler", 0.001, 11, protocol, corpus, True)
    assert corpus.opened_splits == ["train", "validation", "test"]
    assert record["test"]["target_bytes"] == corpus.bounds["test"][1] - corpus.bounds["test"][0] - 1
    assert math.isfinite(record["test"]["perplexity"])


@pytest.mark.parametrize("failure", ["loss", "gradient"])
def test_nonfinite_training_values_reject_before_optimizer_mutation(failure, monkeypatch):
    model = study.ByteTransformer(tiny_protocol()["model"])
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    if failure == "gradient":
        next(model.parameters()).register_hook(lambda gradient: torch.full_like(gradient, math.inf))
    else:
        original = model.forward
        monkeypatch.setattr(model, "forward", lambda tokens: original(tokens) * math.inf)
    with pytest.raises((ValueError, RuntimeError), match="[Nn]on.?finite"):
        study.training_update(model, optimizer, torch.zeros(2, 4, dtype=torch.long),
                              torch.ones(2, 4, dtype=torch.long), 1.0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert not optimizer.state


def tuning_records(protocol):
    return [{"phase": "tuning", "policy": policy, "rate": rate, "seed": 0, "status": "passed",
             "result": {"validation": {"nll": 2.0}}}
            for policy in study.POLICIES for rate in protocol["initial_lr_grid"]]


def test_selection_uses_only_final_validation_and_complete_equal_budget():
    protocol = tiny_protocol()
    records = tuning_records(protocol)
    assert study.choose_rates(records, protocol) == dict.fromkeys(study.POLICIES, protocol["initial_lr_grid"][0])
    records[1]["result"]["validation"]["nll"] = 1.0
    assert study.choose_rates(records, protocol)["NoScheduler"] == records[1]["rate"]
    with pytest.raises(ValueError, match="complete declared tuning budget"):
        study.choose_rates(records[1:], protocol)
    records[0]["result"]["test"] = {"nll": 0.1}
    with pytest.raises(ValueError, match="must not contain test"):
        study.choose_rates(records, protocol)


def confirmation_records(protocol):
    return [{"phase": "confirmation", "policy": policy, "seed": seed, "status": "passed",
             "result": {"test": {"nll": 2.0}, "elapsed_seconds": 1.0,
                        "initial_model_sha256": str(seed), "training_index_stream_sha256": str(seed),
                        "training_split_sha256": "train", "validation_split_sha256": "val", "test_split_sha256": "test"}}
            for policy in study.POLICIES for seed in protocol["confirmation_seeds"]]


def test_acceptance_requires_all_pairs_and_predeclared_noninferiority_margin():
    protocol = tiny_protocol()
    records = confirmation_records(protocol)
    assert study.summarize(records, {}, protocol)["accepted"]
    for record in records:
        if record["policy"] == "DLRS":
            record["result"]["test"]["nll"] = 2.1
    assert not study.summarize(records, {}, protocol)["accepted"]
    records = confirmation_records(protocol)
    records[0]["status"] = "failed"
    assert not study.summarize(records, {}, protocol)["accepted"]
    assert not study.summarize(confirmation_records(protocol)[1:], {}, protocol)["accepted"]
    assert study.paired_interval([0.0] * 5) == [0.0, 0.0]


def test_acceptance_rejects_mismatched_initialization_data_and_sample_streams():
    protocol = tiny_protocol()
    for key in ("initial_model_sha256", "training_index_stream_sha256", "training_split_sha256",
                "validation_split_sha256", "test_split_sha256"):
        records = confirmation_records(protocol)
        records[0]["result"][key] = "wrong"
        with pytest.raises(ValueError, match="Paired trials differ"):
            study.summarize(records, {}, protocol)


def test_corpus_integrity_and_manifest_freeze_without_evaluation(tmp_path, monkeypatch):
    protocol = tiny_protocol()
    corpus_fixture(tmp_path, protocol)
    monkeypatch.setattr(study.ByteCorpus, "read_split", lambda *_: pytest.fail("Preparation must not tokenize splits"))
    output = tmp_path / "output"
    _, first = study.prepare(protocol, tmp_path, output)
    _, repeated = study.prepare(protocol, tmp_path, output)
    assert first == repeated
    changed = copy.deepcopy(protocol)
    changed["model"]["dropout"] = 0.2
    with pytest.raises(ValueError, match="Frozen study manifest differs"):
        study.prepare(changed, tmp_path, output)
    (tmp_path / protocol["corpus"]["filename"]).write_bytes(b"wrong")
    with pytest.raises(ValueError, match="checksum/length"):
        study.prepare(protocol, tmp_path, output)


def test_wall_clock_budget_prevents_starting_an_extra_trial(tmp_path):
    protocol = tiny_protocol()
    corpus = corpus_fixture(tmp_path, protocol)
    with pytest.raises(study.StudyBudgetReached):
        study.run_trial("DLRS", 0.001, 0, protocol, corpus, False, deadline=time.monotonic() - 1)
    assert not corpus.opened_splits


def test_failed_trial_is_preserved_and_not_silently_retried(tmp_path, monkeypatch):
    protocol = tiny_protocol()
    corpus = corpus_fixture(tmp_path, protocol)
    calls = []

    def fail(*_):
        calls.append(True)
        raise ValueError("injected trial failure")

    monkeypatch.setattr(study, "run_trial", fail)
    args = (tmp_path / "output", "tuning", 0, "DLRS", 0.001, protocol, corpus, "frozen", 10)
    first = study.stored_trial(*args)
    assert first["status"] == "failed" and "injected trial failure" in first["error"]
    assert study.stored_trial(*args) == first
    assert len(calls) == 1


def test_cached_success_rejects_corrupted_metrics_and_compute_budget(tmp_path):
    protocol = tiny_protocol()
    corpus = corpus_fixture(tmp_path, protocol)
    output = tmp_path / "output"
    args = (output, "tuning", 0, "DLRS", 0.001, protocol, corpus, "frozen", None)
    record = study.stored_trial(*args)
    assert record["status"] == "passed"
    path = next((output / "trials").glob("*.json"))
    mutations = [("optimizer_updates", 1), ("training_target_bytes", 1), ("final_learning_rate", math.nan)]
    for key, value in mutations:
        corrupted = copy.deepcopy(record)
        corrupted["result"][key] = value
        path.write_text(json.dumps(corrupted))
        with pytest.raises(ValueError, match="Cached"):
            study.stored_trial(*args)
    corrupted = copy.deepcopy(record)
    corrupted["result"]["validation"]["nll"] = math.nan
    path.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="Cached evaluation"):
        study.stored_trial(*args)


@pytest.mark.parametrize("outcome,expected_exit,quality_accepted", [
    ("success", 0, True),
    ("quality_failure", 0, False),
    ("failed_tuning", 2, True),
    ("failed_confirmation", 2, False),
    ("incomplete_confirmation", 2, False),
])
def test_cli_exit_distinguishes_execution_failure_from_negative_quality(
    tmp_path, monkeypatch, outcome, expected_exit, quality_accepted
):
    protocol = tiny_protocol()
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["transformer-study", "--protocol", str(protocol_path),
                                    "--output", str(output)])
    monkeypatch.setattr(study, "prepare", lambda *_: (None, "frozen"))
    attempts = []

    def simulated_trial(policy, rate, seed, _protocol, _corpus, evaluate_test, _deadline):
        attempts.append((policy, rate, seed))
        first_tuning = seed == 0 and rate == protocol["initial_lr_grid"][0]
        first_confirmation = seed == protocol["confirmation_seeds"][0]
        if policy == "NoScheduler":
            if ((outcome == "failed_tuning" and first_tuning)
                    or (outcome == "failed_confirmation" and first_confirmation)):
                raise RuntimeError("injected ordinary trial failure")
            if outcome == "incomplete_confirmation" and first_confirmation:
                raise study.StudyBudgetReached("injected resource interruption")
        length = protocol["corpus"]["bytes"]
        cfg = protocol["training"]
        measured = {
            "parameter_count": 100,
            "optimizer_updates": cfg["total_optimizer_updates"],
            "training_target_bytes": cfg["training_target_bytes_per_trial"],
            "window_train_nll": [2.0] * cfg["scheduler_windows"],
            "window_learning_rates": [rate] * cfg["scheduler_windows"],
            "final_learning_rate": rate,
            "elapsed_seconds": 1.0,
            "initial_model_sha256": hashlib.sha256(str(seed).encode()).hexdigest(),
            "training_index_stream_sha256": "a" * 64,
            "training_split_sha256": "b" * 64,
            "validation_split_sha256": "c" * 64,
            "validation": {"nll": 2.0, "target_bytes": length * 9 // 10 - length * 8 // 10 - 1},
        }
        if evaluate_test:
            nll = 2.1 if outcome == "quality_failure" and policy == "DLRS" else 2.0
            measured.update(test={"nll": nll, "perplexity": math.exp(nll),
                                  "target_bytes": length - length * 9 // 10 - 1},
                            test_split_sha256="d" * 64)
        return measured

    monkeypatch.setattr(study, "run_trial", simulated_trial)
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        assert study.main() == expected_exit
        summary = json.loads((output / "summary.json").read_text())
        assert summary["execution_passed"] is (expected_exit == 0)
        assert summary["accepted"] is quality_accepted
        recorded = {path.name: path.read_bytes() for path in (output / "trials").glob("*.json")}
        assert len(recorded) == (10 if outcome == "incomplete_confirmation" else 24)
        statuses = [json.loads(raw)["status"] for raw in recorded.values()]
        assert statuses.count("failed") == int(outcome.startswith("failed_"))
        assert statuses.count("incomplete") == int(outcome == "incomplete_confirmation")
        previous_attempts = attempts.copy()
        # Cached failed/incomplete trials remain failures; resume must not retry them.
        assert study.main() == expected_exit
        assert attempts == previous_attempts
        assert recorded == {path.name: path.read_bytes() for path in (output / "trials").glob("*.json")}
    finally:
        torch.use_deterministic_algorithms(deterministic)
