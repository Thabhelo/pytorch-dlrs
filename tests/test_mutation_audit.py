"""False-green guards for the explicit mutation audit, not a full campaign."""

import copy
import json
from pathlib import PureWindowsPath
from types import SimpleNamespace

import pytest

from benchmarks.audit_mutations import (
    REPORT_PLUGIN,
    classify_run,
    exact_mutation,
    nodes_for_selectors,
    run_campaign,
    run_case,
)

NODE = "tests/test_example.py::test_example"


def passing_report():
    return {"schema_version": 1, "exitstatus": 0, "collection_errors": [],
            "internal_errors": [], "collected": [NODE],
            "reports": [{"nodeid": NODE, "when": phase, "outcome": "passed",
                         "assertion_failure": False, "exception_type": None}
                        for phase in ("setup", "call", "teardown")]}


def assertion_report():
    report = passing_report()
    report["exitstatus"] = 1
    report["reports"][1].update(outcome="failed", assertion_failure=True,
                                exception_type="builtins.AssertionError")
    return report


def test_only_completed_assertion_failures_are_kills():
    assert classify_run(1, False, assertion_report(), {NODE})["status"] == "killed"
    assert classify_run(0, False, passing_report(), {NODE})["status"] == "survived"


@pytest.mark.parametrize("defect", ["collection", "setup", "teardown", "internal",
                                   "unexpected_exception", "skip", "missing_call",
                                   "missing_teardown", "duplicate_call", "extra_test",
                                   "missing_test", "duplicate_test", "malformed", "bad_schema"])
def test_infrastructure_or_incomplete_execution_cannot_kill_mutant(defect):
    report = assertion_report()
    if defect == "collection":
        report["collection_errors"].append("ImportError")
    elif defect in ("setup", "teardown"):
        report["reports"][0 if defect == "setup" else 2]["outcome"] = "failed"
    elif defect == "internal":
        report["internal_errors"].append("hook failure")
    elif defect == "unexpected_exception":
        report["reports"][1].update(assertion_failure=False, exception_type="builtins.ValueError")
    elif defect == "skip":
        report["reports"][1]["outcome"] = "skipped"
    elif defect == "missing_call":
        del report["reports"][1]
    elif defect == "missing_teardown":
        del report["reports"][2]
    elif defect == "duplicate_call":
        report["reports"].append(copy.deepcopy(report["reports"][1]))
    elif defect == "extra_test":
        report["collected"].append(NODE + "_extra")
    elif defect == "missing_test":
        report["collected"] = []
    elif defect == "duplicate_test":
        report["collected"].append(NODE)
    elif defect == "malformed":
        report["reports"] = None
    else:
        del report["schema_version"]
    assert classify_run(1, False, report, {NODE})["status"] == "inconclusive"


@pytest.mark.parametrize("returncode,timed_out", [(0, True), (1, True), (2, False),
                                                (3, False), (4, False), (5, False),
                                                (-9, False), (0, False)])
def test_process_failure_or_inconsistent_exit_cannot_kill_mutant(returncode, timed_out):
    assert classify_run(returncode, timed_out, assertion_report(), {NODE})["status"] == "inconclusive"


def test_missing_report_and_zero_tests_are_inconclusive():
    assert classify_run(1, False, None, {NODE})["status"] == "inconclusive"
    assert classify_run(0, False, passing_report(), set())["status"] == "inconclusive"


@pytest.mark.parametrize("source", ["x = 2\n", "x = 1\nx = 1\n"])
def test_changed_or_ambiguous_source_anchor_is_rejected(source):
    with pytest.raises(ValueError, match="exactly one"):
        exact_mutation(source, {"name": "example", "before": "x = 1", "after": "x = 2"})


def test_mutation_replaces_one_site_and_must_compile():
    entry = {"name": "example", "before": "x = 1", "after": "x = 2"}
    assert exact_mutation("x = 1\ny = 3\n", entry) == "x = 2\ny = 3\n"
    with pytest.raises(SyntaxError):
        exact_mutation("x = 1\n", {**entry, "after": "x = ("})
    with pytest.raises(ValueError, match="must change"):
        exact_mutation("x = 1\n", {**entry, "after": "x = 1"})


def test_selectors_match_only_exact_tests_and_their_parameters():
    collected = [NODE, NODE + "[float32]", NODE + "_unrelated"]
    assert nodes_for_selectors(collected, [NODE]) == {NODE, NODE + "[float32]"}
    with pytest.raises(ValueError, match="No collected"):
        nodes_for_selectors(collected, [NODE + "_missing"])


def test_campaign_refuses_existing_evidence_before_reading_sources(tmp_path):
    marker = tmp_path / "existing-evidence.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_campaign(tmp_path, source_root=tmp_path / "missing-source")
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_actual_report_hook_normalizes_windows_test_source_keys():
    namespace = {}
    exec(compile(REPORT_PLUGIN, "audit_report.py", "exec"), namespace)
    root = PureWindowsPath(r"C:\audit\scratch")
    path = root / "tests" / "test_example.py"
    recorded = {"path": str(path), "sha256": "fixture-digest"}
    # Only file reading is replaced: execute the actual collection hook with
    # Windows path semantics, even when this regression runs on another OS.
    namespace["identity"] = lambda actual: recorded if actual == path else None
    session = SimpleNamespace(config=SimpleNamespace(rootpath=root),
                              items=[SimpleNamespace(nodeid=NODE, path=path)])
    namespace["pytest_collection_finish"](session)
    assert namespace["DATA"]["collected"] == [NODE]
    assert namespace["DATA"]["test_sources"] == {"tests/test_example.py": recorded}


def test_real_pytest_reports_distinguish_assertion_runtime_and_setup_errors(tmp_path):
    inputs = tmp_path / "inputs"
    (inputs / "dlrs").mkdir(parents=True)
    (inputs / "tests").mkdir()
    for name in ("__init__.py", "scheduler.py", "utils.py"):
        (inputs / "dlrs" / name).write_text("# Isolated report-hook fixture.\n", encoding="utf-8")
    (inputs / "tests/test_probe.py").write_text(
        "import pytest\n"
        "@pytest.fixture\ndef bad_setup():\n    raise RuntimeError('setup')\n"
        "def test_assert():\n    assert 2 + 2 == 5\n"
        "def test_runtime():\n    raise ValueError('runtime')\n"
        "def test_setup(bad_setup):\n    pass\n"
        "def test_ok():\n    assert 2 + 2 == 4\n", encoding="utf-8")
    selectors = ["tests/test_probe.py::test_" + name for name in ("assert", "runtime", "setup", "ok")]
    result = run_case(tmp_path / "case", inputs, selectors, set(selectors), 45)
    assert result["status"] == "inconclusive"
    assert result["reason"] in ("non-assertion failure, skip, or unexpected outcome",
                                "setup or teardown did not pass")
    report = json.loads((tmp_path / "case/pytest-report.json").read_text(encoding="utf-8"))
    for name, expected in (("assert", "killed"), ("runtime", "inconclusive"),
                           ("setup", "inconclusive"), ("ok", "survived")):
        node = "tests/test_probe.py::test_" + name
        isolated = {**report, "collected": [node], "exitstatus": 0 if name == "ok" else 1,
                    "reports": [event for event in report["reports"] if event["nodeid"] == node]}
        assert classify_run(isolated["exitstatus"], False, isolated, {node})["status"] == expected
    assert result["source_identity"]["dlrs/scheduler.py"]["path"].startswith(str(tmp_path))
