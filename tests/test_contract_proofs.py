"""Proof-gate regressions; a skipped optional solver test is not a proof result."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from benchmarks import verify_contract as contract


@pytest.mark.parametrize("premises,negation,expected", [
    ("sat", "unsat", "proved"),
    ("sat", "sat", "counterexample"),
    ("sat", "unknown", "inconclusive"),
    ("sat", None, "inconclusive"),
    ("unsat", "unsat", "vacuous"),
    ("unknown", "unsat", "inconclusive"),
    (None, None, "inconclusive"),
])
def test_only_satisfiable_premises_and_unsatisfiable_negation_prove_claim(premises, negation, expected):
    assert contract.classify_results(premises, negation) == expected


def successful_records():
    return [{"name": name, "status": "proved", "nonvacuity": {"result": "sat"},
             "negation": {"result": "unsat"}} for name in contract.EXPECTED_OBLIGATIONS]


@pytest.mark.parametrize("defect", ["empty", "missing", "duplicate", "renamed", "unknown", "vacuous", "counterexample"])
def test_incomplete_or_nonproof_results_never_pass_suite(defect):
    records = successful_records()
    assert contract.suite_passed(records)
    if defect == "empty":
        records = []
    elif defect == "missing":
        records.pop()
    elif defect == "duplicate":
        records[-1] = records[0]
    elif defect == "renamed":
        records[-1]["name"] = "replacement_not_reviewed"
    elif defect == "unknown":
        records[-1]["negation"]["result"] = "unknown"
    elif defect == "vacuous":
        records[-1]["nonvacuity"]["result"] = "unsat"
    elif defect == "counterexample":
        records[-1]["negation"]["result"] = "sat"
    assert not contract.suite_passed(records)


def test_missing_solver_records_unavailable_and_nonzero_exit(tmp_path, monkeypatch):
    def missing(_):
        raise ModuleNotFoundError("No module named 'z3'")

    monkeypatch.setattr(contract.importlib, "import_module", missing)
    output = tmp_path / "no-solver"
    assert contract.main(["--output", str(output)]) == 2
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "unavailable" and report["proofs_passed"] is False
    assert report["obligations"] == []


def test_solver_timeout_retains_unknown_reason_and_query(tmp_path):
    class TimedOutSolver:
        def set(self, **options):
            assert options == {"timeout": 1}

        def add(self, *assertions):
            assert assertions == ("assertion",)

        def to_smt2(self):
            return "(assert true)\n(check-sat)\n"

        def check(self):
            return "unknown"

        def statistics(self):
            return []

        def reason_unknown(self):
            return "timeout"

    z3 = SimpleNamespace(SolverFor=lambda _: TimedOutSolver())
    query = tmp_path / "timeout.smt2"
    record = contract.check_query(z3, "QF_NRA", ("assertion",), 1, query)
    assert record["result"] == "unknown" and record["reason_unknown"] == "timeout"
    assert contract.classify_results("sat", record["result"]) == "inconclusive"
    assert query.is_file() and record["timeout_ms"] == 1


def test_previous_proof_evidence_cannot_be_overwritten(tmp_path):
    sentinel = tmp_path / "retained.txt"
    sentinel.write_text("prior evidence")
    with pytest.raises(FileExistsError):
        contract.main(["--output", str(tmp_path)])
    assert sentinel.read_text() == "prior evidence"


def z3_available():
    return pytest.importorskip("z3", reason="Optional Z3 absent; run verify_contract.py separately to obtain proof evidence")


def test_all_exact_model_obligations_with_actual_solver_and_replay(tmp_path):
    z3 = z3_available()
    output = tmp_path / "actual-proofs"
    assert contract.main(["--output", str(output)]) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["proofs_passed"] is True
    assert report["solver"]["version"] == z3.get_version_string()
    assert len(report["obligations"]) == len(contract.EXPECTED_OBLIGATIONS) == 25
    # Replaying the emitted SMT-LIB must agree with the in-process checks.
    for obligation in report["obligations"]:
        for stage, expected in (("nonvacuity", "sat"), ("negation", "unsat")):
            solver = z3.SolverFor(obligation["logic"])
            solver.set(timeout=5000)
            query_path = output / obligation[stage]["query_file"]
            assert hashlib.sha256(query_path.read_bytes()).hexdigest() == obligation[stage]["query_sha256"]
            solver.from_file(str(query_path))
            assert str(solver.check()) == expected


def test_contradictory_assumptions_do_not_prove_a_claim(tmp_path):
    z3 = z3_available()
    value = z3.Real("value")
    obligation = contract.Obligation("contradiction", "Negative control", (value > 0, value < 0), value == value)
    result = contract.verify_obligation(z3, obligation, tmp_path, 5000)
    assert result["status"] == "vacuous"
    assert result["nonvacuity"]["result"] == "unsat"
    assert "negation" not in result


def test_clamp_can_increase_an_initially_out_of_bounds_rate(tmp_path):
    z3 = z3_available()
    rate, lower, upper, slope, granularity, delta = z3.Reals("rate lower upper slope granularity delta")
    result = contract.update(z3, rate, granularity, delta, slope, lower, upper)
    obligation = contract.Obligation(
        "out_of_bounds_direction", "Deliberately false claim without the in-bounds precondition",
        (rate > 0, rate < lower, lower > 0, upper > lower, slope > 0, granularity > 0, delta > 0),
        result <= rate,
    )
    record = contract.verify_obligation(z3, obligation, tmp_path, 5000)
    assert record["status"] == "counterexample"
    assert record["negation"]["result"] == "sat" and record["negation"]["model"]


def test_wrong_delta_boundary_is_detected_by_solver(tmp_path, monkeypatch):
    z3 = z3_available()
    monkeypatch.setattr(contract, "select_delta", lambda z3, s, dd, do, di:
                        z3.If(s >= 1, dd, z3.If(s >= 0, do, di)))
    obligation = next(item for item in contract.build_obligations(z3) if item.name == "delta_equality_at_one")
    result = contract.verify_obligation(z3, obligation, tmp_path, 5000)
    assert result["status"] == "counterexample"


def test_wrong_update_sign_is_detected_by_solver(tmp_path, monkeypatch):
    z3 = z3_available()
    monkeypatch.setattr(contract, "update", lambda z3, rate, granularity, delta, slope, lower, upper=None:
                        z3.If(rate == 0, 0, contract.clamp(z3, rate + granularity * delta * slope, lower, upper)))
    obligation = next(item for item in contract.build_obligations(z3) if item.name == "in_bounds_nonincrease")
    result = contract.verify_obligation(z3, obligation, tmp_path, 5000)
    assert result["status"] == "counterexample"
