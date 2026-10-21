"""Run a bounded scheduler mutation audit in retained, isolated source copies.

This is a targeted sensitivity check, not a correctness proof or a training-quality
study. Normal pytest runs exercise only the small harness regression tests. Run
the campaign explicitly: python benchmarks/audit_mutations.py --output NEW_DIR
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAFETY = "tests/test_scheduler_safety.py::"
CONTRACT = "tests/test_training_contract.py::"
NUMERICS = "tests/test_correctness_numerics.py::"


def mutation(name, description, before, after, *tests):
    return {"name": name, "description": description, "before": before,
            "after": after, "tests": list(tests)}


# One deliberate semantic defect per entry. Anchors are exact and fail closed
# after source changes; a campaign must never silently mutate a different site.
MUTATIONS = [
    mutation("slope_one_wrong_factor", "Use divergence factor at slope exactly one.",
             "diverging = exact_slope > 1", "diverging = exact_slope >= 1",
             SAFETY + "test_slope_one_uses_stagnation_factor"),
    mutation("reverse_loss_direction", "Reverse the sign of loss-driven updates.",
             "self._step_with_direction(batch_losses, 1.0)",
             "self._step_with_direction(batch_losses, -1.0)",
             CONTRACT + "test_randomized_updates_match_independent_decimal_reference"),
    mutation("reuse_first_group_rate", "Reuse the first group's rate/decade for every group.",
             "self._candidate(rate, slope, delta, config['min_lr'], config['max_lr'])",
             "self._candidate(current_rates[0], slope, delta, config['min_lr'], config['max_lr'])",
             NUMERICS + "test_distinct_parameter_group_decades_match_fraction_oracle"),
    mutation("rounded_decade", "Restore a rounded log10 decade instead of the exact binary64 decade.",
             "exponent = Decimal.from_float(current).adjusted()",
             "exponent = math.floor(math.log10(current))",
             NUMERICS + "test_distinct_parameter_group_decades_match_fraction_oracle"),
    mutation("unfreeze_zero_rate", "Raise a frozen zero group to the minimum rate.",
             "if current == 0:\n            return 0.0, 0.0",
             "if current == 0:\n            return min_lr, 0.0",
             SAFETY + "test_zero_lr_groups_remain_frozen"),
    mutation("remove_minimum_bound", "Omit the lower learning-rate clamp.",
             "result = max(proposed, min_lr)", "result = proposed",
             SAFETY + "test_slope_one_uses_stagnation_factor"),
    mutation("remove_maximum_bound", "Omit the upper learning-rate clamp.",
             "result = min(result, max_lr)", "result = result",
             "tests/test_scheduler.py::TestIntegration::test_max_lr_bound"),
    mutation("invalid_input_changes_epoch", "Advance bookkeeping before validating losses.",
             "losses = [] if batch_losses is None else _losses(batch_losses)",
             "self.last_epoch += 1\n        losses = [] if batch_losses is None else _losses(batch_losses)",
             SAFETY + "test_invalid_loss_does_not_change_any_state"),
    mutation("checkpoint_applied_before_validation", "Restore epoch metadata before checkpoint validation.",
             "version = state_dict.get('state_version', 0)",
             "self.last_epoch = state_dict.get('last_epoch', self.last_epoch)\n        version = state_dict.get('state_version', 0)",
             SAFETY + "test_malformed_checkpoint_is_rejected_atomically"),
    mutation("checkpoint_does_not_restore_rates", "Drop the optimizer-rate restoration at load.",
             "self._apply_lrs(prepared)\n        for group, initial_lr in initial_lrs:",
             "pass  # Deliberately omit restoring rates.\n        for group, initial_lr in initial_lrs:",
             SAFETY + "test_legacy_checkpoint_without_new_fields_resumes"),
    mutation("shallow_checkpoint", "Return checkpoint lists that alias scheduler state.",
             "return deepcopy({", "return dict({",
             SAFETY + "test_checkpoint_and_get_last_lr_are_independent_snapshots"),
    mutation("aliased_tensor_rate_snapshot", "Return live tensor objects from get_last_lr.",
             "return [_clone_lr(value) for value in self._last_lr]",
             "return list(self._last_lr)",
             "tests/test_correctness_state.py::test_rate_snapshots_do_not_alias_live_or_cached_state"),
    mutation("unbounded_metric_history", "Retain every wrapper metric instead of a bounded window.",
             "history = (self._metric_history + [value])[-(self.patience + 2):]",
             "history = self._metric_history + [value]",
             SAFETY + "test_wrapper_patience_is_bounded_rolling_window_and_resumes"),
    mutation("ignore_max_mode", "Treat a maximizing metric as a minimizing loss.",
             "1.0 if self.mode == 'min' else -1.0", "1.0",
             SAFETY + "test_wrapper_improvement_increases_lr_in_both_modes"),
]


REPORT_PLUGIN = r'''
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import pytest

DATA = {"schema_version": 1, "collected": [], "collection_errors": [],
        "internal_errors": [], "reports": [], "test_sources": {}}

def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

def source_identity():
    import dlrs
    import dlrs.scheduler
    import dlrs.utils
    return {"dlrs/__init__.py": identity(dlrs.__file__),
            "dlrs/scheduler.py": identity(dlrs.scheduler.__file__),
            "dlrs/utils.py": identity(dlrs.utils.__file__)}

def pytest_sessionstart(session):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    DATA["environment"] = {"python": platform.python_version(),
        "executable": sys.executable, "platform": platform.platform(),
        "torch": str(torch.__version__), "pytest": pytest.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads()}
    DATA["source_before"] = source_identity()

def pytest_collection_finish(session):
    DATA["collected"] = [item.nodeid for item in session.items]
    for item in session.items:
        DATA["test_sources"][str(item.path.relative_to(session.config.rootpath))] = identity(item.path)

def pytest_collectreport(report):
    if report.failed:
        DATA["collection_errors"].append(str(report.longrepr))

def pytest_internalerror(excrepr, excinfo):
    DATA["internal_errors"].append(str(excrepr))

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    exc = call.excinfo
    DATA["reports"].append({"nodeid": report.nodeid, "when": report.when,
        "outcome": report.outcome,
        "assertion_failure": bool(exc is not None and issubclass(exc.type, AssertionError)),
        "exception_type": None if exc is None else exc.type.__module__ + "." + exc.type.__name__})

def pytest_sessionfinish(session, exitstatus):
    DATA["exitstatus"] = int(exitstatus)
    DATA["source_after"] = source_identity()
    with open(os.environ["DLRS_MUTATION_REPORT"], "x", encoding="utf-8") as stream:
        json.dump(DATA, stream, indent=2, allow_nan=False)
        stream.write("\n")
'''


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def exact_mutation(source, entry):
    count = source.count(entry["before"])
    if count != 1:
        raise ValueError("{}: expected exactly one mutation anchor, found {}".format(entry["name"], count))
    if entry["before"] == entry["after"]:
        raise ValueError("Mutation must change the source")
    changed = source.replace(entry["before"], entry["after"], 1)
    compile(changed, "dlrs/scheduler.py", "exec")
    return changed


def nodes_for_selectors(collected, selectors):
    selected = set()
    for selector in selectors:
        matches = {node for node in collected if node == selector or node.startswith(selector + "[")}
        if not matches:
            raise ValueError("No collected tests for " + selector)
        selected.update(matches)
    return selected


def classify_run(returncode, timed_out, report, expected_nodes):
    """Only completed assertion failures count; other failures are inconclusive."""
    def result(status, reason, failures=None):
        return {"status": status, "reason": reason, "assertion_failures": failures or []}

    if timed_out:
        return result("inconclusive", "subprocess timeout")
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        return result("inconclusive", "missing or invalid structured pytest report")
    required = ("exitstatus", "collection_errors", "internal_errors", "collected", "reports")
    if any(key not in report for key in required):
        return result("inconclusive", "incomplete structured pytest report")
    if (any(not isinstance(report[key], list) for key in required[1:])
            or any(not isinstance(node, str) for node in report["collected"])):
        return result("inconclusive", "malformed structured pytest report")
    if report["exitstatus"] != returncode or returncode not in (0, 1):
        return result("inconclusive", "pytest process did not finish normally")
    if report["collection_errors"] or report["internal_errors"]:
        return result("inconclusive", "collection or internal error")
    collected = report["collected"]
    if not expected_nodes or len(collected) != len(set(collected)) or set(collected) != set(expected_nodes):
        return result("inconclusive", "collected test identities differ from passing baseline")
    events = {}
    for event in report["reports"]:
        if not isinstance(event, dict):
            return result("inconclusive", "malformed test report")
        key = (event.get("nodeid"), event.get("when"))
        if key in events or key[0] not in expected_nodes or key[1] not in ("setup", "call", "teardown"):
            return result("inconclusive", "unexpected or duplicate test phase")
        events[key] = event
    failures = []
    for node in sorted(expected_nodes):
        for phase in ("setup", "call", "teardown"):
            event = events.get((node, phase))
            if event is None:
                return result("inconclusive", "incomplete test execution")
            if phase != "call" and event.get("outcome") != "passed":
                return result("inconclusive", "setup or teardown did not pass")
            if phase == "call":
                if event.get("outcome") == "failed" and event.get("assertion_failure") is True:
                    failures.append(node)
                elif event.get("outcome") != "passed":
                    return result("inconclusive", "non-assertion failure, skip, or unexpected outcome")
    if failures and returncode == 1:
        return result("killed", "regression assertion failed with no infrastructure errors", failures)
    if not failures and returncode == 0:
        return result("survived", "all selected tests passed")
    return result("inconclusive", "exit code does not agree with test outcomes")


def write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def file_identity(path):
    return {"path": str(path.resolve()), "sha256": sha256(path.read_bytes())}


def run_case(directory, inputs, selectors, expected_nodes, timeout, replacement=None,
             expected_environment=None):
    directory.mkdir()
    scratch = directory / "scratch"
    shutil.copytree(inputs, scratch)
    if replacement is not None:
        (scratch / "dlrs/scheduler.py").write_text(replacement, encoding="utf-8")
    (scratch / "audit_report.py").write_text(REPORT_PLUGIN, encoding="utf-8")
    (scratch / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    expected_sources = {name: file_identity(scratch / name) for name in
                        ("dlrs/__init__.py", "dlrs/scheduler.py", "dlrs/utils.py")}
    expected_tests = {name: file_identity(scratch / name) for name in
                      sorted({selector.split("::")[0] for selector in selectors})}
    command = [sys.executable, "-m", "pytest", "-q", "-c", "pytest.ini", "--rootdir", str(scratch),
               "-p", "audit_report", "-p", "no:cacheprovider", "--tb=short"] + selectors
    overrides = {"PYTHONPATH": str(scratch), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                 "PYTEST_ADDOPTS": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
                 "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                 "NUMEXPR_NUM_THREADS": "1", "DLRS_MUTATION_REPORT": str(directory / "pytest-report.json")}
    write_json(directory / "command.json", {"argv": command, "cwd": str(scratch),
                                           "environment_overrides": overrides, "timeout_seconds": timeout})
    environment = os.environ.copy()
    environment.update(overrides)
    # Neither inherited pytest plug-ins nor Python startup customization should
    # redirect a source-isolated run into an installed/live scheduler package.
    environment.pop("PYTEST_PLUGINS", None)
    started = time.monotonic()
    timed_out = False
    returncode = None
    with (directory / "pytest.log").open("xb") as stream:
        try:
            process = subprocess.run(command, cwd=scratch, env=environment, stdout=stream,
                                     stderr=subprocess.STDOUT, timeout=timeout, check=False)
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
    report_path = directory / "pytest-report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        report = None
    reason = None
    if not isinstance(report, dict):
        reason = "structured report unavailable"
    elif (report.get("source_before") != expected_sources or report.get("source_after") != expected_sources
          or report.get("test_sources") != expected_tests):
        reason = "imported source or collected test path/hash mismatch"
    elif expected_environment is not None and report.get("environment") != expected_environment:
        reason = "environment differs from baseline"
    if expected_nodes is None and isinstance(report, dict):
        try:
            expected_nodes = nodes_for_selectors(report.get("collected", []), selectors)
        except ValueError as exc:
            reason = str(exc)
    classification = classify_run(returncode, timed_out, report, expected_nodes or set())
    if reason:
        classification = {"status": "inconclusive", "reason": reason, "assertion_failures": []}
    result = {**classification, "returncode": returncode, "timed_out": timed_out,
              "elapsed_seconds": time.monotonic() - started, "command": command,
              "source_identity": expected_sources, "test_identity": expected_tests,
              "selected_nodes": sorted(expected_nodes or []),
              "environment": report.get("environment") if isinstance(report, dict) else None}
    write_json(directory / "result.json", result)
    return result


def run_campaign(output, timeout=45.0, source_root=ROOT):
    output = output.resolve()
    # No resume or overwrite: preserve both successful and interrupted evidence.
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    selectors = sorted({test for entry in MUTATIONS for test in entry["tests"]})
    paths = sorted(set(["dlrs/__init__.py", "dlrs/scheduler.py", "dlrs/utils.py"]
                       + [selector.split("::")[0] for selector in selectors]))
    inputs = output / "inputs"
    inputs.mkdir()
    snapshot = {name: (source_root / name).read_bytes() for name in paths}
    if any((source_root / name).read_bytes() != data for name, data in snapshot.items()):
        raise ValueError("Source changed during snapshot preparation")
    source = snapshot["dlrs/scheduler.py"].decode("utf-8")
    changed_sources = {entry["name"]: exact_mutation(source, entry) for entry in MUTATIONS}
    (output / "harness.py").write_bytes(Path(__file__).read_bytes())
    for name, data in snapshot.items():
        path = inputs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    write_json(output / "protocol.json", {"schema_version": 1, "mutations": MUTATIONS,
        "selected_tests": selectors, "timeout_seconds_per_run": timeout,
        "source_sha256": {name: sha256(data) for name, data in snapshot.items()},
        "harness_sha256": sha256(Path(__file__).read_bytes()),
        "report_plugin_sha256": sha256(REPORT_PLUGIN.encode()),
        "python": platform.python_version(), "executable": sys.executable,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Fourteen deliberately selected scheduler defects, not an exhaustive mutation score or correctness proof."})
    cases = output / "cases"
    cases.mkdir()
    baseline = run_case(cases / "baseline", inputs, selectors, None, timeout)
    results = []
    print("baseline: " + baseline["status"], flush=True)
    if baseline["status"] == "survived":
        for entry in MUTATIONS:
            nodes = nodes_for_selectors(baseline["selected_nodes"], entry["tests"])
            result = run_case(cases / entry["name"], inputs, entry["tests"], nodes, timeout,
                              changed_sources[entry["name"]], baseline["environment"])
            results.append({"name": entry["name"], **result})
            print(entry["name"] + ": " + result["status"], flush=True)
    unchanged = all((source_root / name).read_bytes() == data for name, data in snapshot.items())
    counts = {status: sum(item["status"] == status for item in results)
              for status in ("killed", "survived", "inconclusive")}
    complete = baseline["status"] == "survived" and len(results) == len(MUTATIONS)
    summary = {"schema_version": 1, "baseline_passed": baseline["status"] == "survived",
        "status": "complete" if complete and not counts["inconclusive"] else "inconclusive",
        "counts": counts, "mutations_planned": len(MUTATIONS), "baseline": baseline,
        "results": results, "live_inputs_unchanged": unchanged,
        "elapsed_seconds": time.monotonic() - started,
        "scope": "Detecting these selected defects does not establish universal correctness, optimizer quality, or deployment safety."}
    if not unchanged:
        summary["status"] = "inconclusive"
    write_json(output / "summary.json", summary)
    manifest = {str(path.relative_to(output)): sha256(path.read_bytes())
                for path in sorted(output.rglob("*")) if path.is_file()}
    write_json(output / "file-sha256.json", manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory; existing paths are rejected")
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    args = parser.parse_args()
    if not 0 < args.timeout_seconds <= 300:
        parser.error("--timeout-seconds must be finite and in (0, 300]")
    try:
        summary = run_campaign(args.output, args.timeout_seconds)
    except (OSError, ValueError, SyntaxError) as exc:
        print("Mutation audit could not complete: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps({key: summary[key] for key in ("status", "counts", "elapsed_seconds")}, indent=2))
    if summary["status"] != "complete":
        return 2
    return 1 if summary["counts"]["survived"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
