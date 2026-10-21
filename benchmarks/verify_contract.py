"""Check exact-arithmetic DLRS abstraction obligations with optional Z3.

This is not a proof of the Python/PyTorch implementation. No runtime package is
imported. Install z3-solver in a separate development environment to run it.

Official semantics/API references:
https://microsoft.github.io/z3guide/docs/theories/Arithmetic/
https://microsoft.github.io/z3guide/programming/Z3%20Python/Introduction/
https://z3prover.github.io/api/html/classz3py_1_1_solver.html
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_OBLIGATIONS = (
    "slope_bounds", "slope_endpoint_sign", "positive_loss_scaling",
    "max_mode_reverses_slope", "delta_diverging", "delta_middle",
    "delta_equality_at_one", "delta_improving", "delta_always_positive",
    "unclamped_positive_slope_decreases", "unclamped_negative_slope_increases",
    "zero_rate_frozen", "two_sided_clamp_bounds", "lower_only_clamp_bound",
    "clamp_idempotent", "in_bounds_nonincrease", "in_bounds_nondecrease",
    "lower_only_in_bounds_nonincrease", "lower_only_in_bounds_nondecrease",
    "in_bounds_zero_slope_identity", "relative_unclamped_adjustment_bound",
    "group_locality", "window_length_base", "window_length_inductive_step",
    "window_length_saturates",
)
LIMITATIONS = [
    "Real variables denote mathematical reals, not IEEE floating-point numbers; no NaN or infinity exists in this model.",
    "The loss summary assumes nonnegative endpoints and total >= first + last, total > 0, n >= 2. Real n relaxes integer sequence length; there is no proof of Python list validation or summation.",
    "Granularity is a positive real parameter. Only the relative-adjustment obligation also assumes g <= rate < 10*g. Correct computation of floor(log10(rate)) is not proved.",
    "No floating-point rounding, underflow/overflow, Decimal/Fraction fallback, tensor conversion, storage alias, mutation atomicity, serialization, PyTorch lifecycle, or hardware behavior is modeled.",
    "Group locality is a property of independent mathematical state cells; it does not prove memory independence or atomic updates in Python.",
    "Window obligations prove a base case and inductive step for an abstract length recurrence, not retention of metric values or the wrapper implementation.",
    "Solver UNSAT is trusted; no independently checked proof certificate or end-to-end refinement proof is produced.",
    "No convergence, workload quality, stability, nondegradation, performance, or production-safety theorem is claimed.",
]


@dataclass(frozen=True)
class Obligation:
    name: str
    description: str
    assumptions: tuple
    conclusion: object
    logic: str = "QF_NRA"


def select_delta(z3, slope, decreasing, ordinary, increasing):
    return z3.If(slope > 1, decreasing, z3.If(slope >= 0, ordinary, increasing))


def clamp(z3, value, lower, upper=None):
    bounded = z3.If(value < lower, lower, value)
    return bounded if upper is None else z3.If(bounded > upper, upper, bounded)


def update(z3, rate, granularity, delta, slope, lower, upper=None):
    return z3.If(rate == 0, 0, clamp(z3, rate - granularity * delta * slope, lower, upper))


def build_obligations(z3):
    """Build polynomial real formulas; every denominator is replaced by a relation."""
    obligations = []

    def add(name, description, assumptions, conclusion, logic="QF_NRA"):
        obligations.append(Obligation(name, description, tuple(assumptions), conclusion, logic))

    first, last, total, n, slope, scaled_slope, factor, reverse = z3.Reals(
        "first last total n slope scaled_slope positive_factor reverse_slope")
    summary = [first >= 0, last >= 0, total > 0, total >= first + last, n >= 2,
               slope * total == n * (last - first)]
    add("slope_bounds", "For every admissible nonnegative loss summary, -n <= slope <= n.",
        summary, z3.And(slope >= -n, slope <= n))
    add("slope_endpoint_sign", "Slope sign agrees exactly with last - first.", summary,
        z3.And((slope > 0) == (last > first), (slope == 0) == (last == first),
               (slope < 0) == (last < first)))
    add("positive_loss_scaling", "Scaling every loss and their sum by a positive factor preserves slope.",
        summary + [factor > 0, scaled_slope * (factor * total) == n * (factor * last - factor * first)],
        scaled_slope == slope)
    add("max_mode_reverses_slope", "The larger-is-better direction negates the mathematical slope.",
        summary + [reverse * total == n * (first - last)], reverse == -slope)

    dd, do, di = z3.Reals("delta_d delta_o delta_i")
    positive_deltas = [dd > 0, do > 0, di > 0]
    delta = select_delta(z3, slope, dd, do, di)
    for name, region, selected in (
        ("delta_diverging", [slope > 1], dd),
        ("delta_middle", [slope >= 0, slope <= 1], do),
        ("delta_equality_at_one", [slope == 1], do),
        ("delta_improving", [slope < 0], di),
    ):
        add(name, "The exact piecewise rule selects the stated positive factor in this nonempty regime.",
            positive_deltas + region, z3.And(delta == selected, delta > 0))
    add("delta_always_positive", "Positive configured factors yield a positive selected factor for any real slope.",
        positive_deltas, delta > 0)

    rate, granularity, lower, upper, arbitrary = z3.Reals("rate granularity lower upper arbitrary")
    positive = positive_deltas + [rate > 0, granularity > 0, lower > 0]
    bounded = positive + [upper > lower]
    in_bounds = bounded + [rate >= lower, rate <= upper]
    raw = rate - granularity * delta * slope
    result = update(z3, rate, granularity, delta, slope, lower, upper)
    lower_only = update(z3, rate, granularity, delta, slope, lower)
    add("unclamped_positive_slope_decreases", "Before clamping, a positive slope strictly reduces a positive rate.",
        positive + [slope > 0], raw < rate)
    add("unclamped_negative_slope_increases", "Before clamping, a negative slope strictly increases a positive rate.",
        positive + [slope < 0], raw > rate)
    add("zero_rate_frozen", "A zero rate remains zero even when the positive lower bound excludes it.",
        positive_deltas + [rate == 0, granularity > 0, lower > 0, upper > lower], result == 0)
    add("two_sided_clamp_bounds", "An active positive-rate update lies between both valid bounds.",
        bounded, z3.And(result >= lower, result <= upper))
    add("lower_only_clamp_bound", "An active positive-rate update respects the lower bound without an upper bound.",
        positive, lower_only >= lower)
    add("clamp_idempotent", "Clamping an already clamped real value has no effect.",
        [lower > 0, upper > lower],
        clamp(z3, clamp(z3, arbitrary, lower, upper), lower, upper) == clamp(z3, arbitrary, lower, upper))
    add("in_bounds_nonincrease", "A nonnegative slope cannot increase a rate initially within both bounds.",
        in_bounds + [slope >= 0], result <= rate)
    add("in_bounds_nondecrease", "A nonpositive slope cannot decrease a rate initially within both bounds.",
        in_bounds + [slope <= 0], result >= rate)
    add("lower_only_in_bounds_nonincrease", "With only a lower bound and rate >= lower, a nonnegative slope cannot increase rate.",
        positive + [rate >= lower, slope >= 0], lower_only <= rate)
    add("lower_only_in_bounds_nondecrease", "With only a lower bound and rate >= lower, a nonpositive slope cannot decrease rate.",
        positive + [rate >= lower, slope <= 0], lower_only >= rate)
    add("in_bounds_zero_slope_identity", "Zero slope preserves a rate initially within both bounds.",
        in_bounds + [slope == 0], result == rate)
    add("relative_unclamped_adjustment_bound", "If the scale satisfies decade bounds, absolute raw adjustment <= rate * selected_delta * n.",
        positive + [n >= 2, slope >= -n, slope <= n, granularity <= rate, rate < 10 * granularity],
        z3.And(raw - rate >= -rate * delta * n, raw - rate <= rate * delta * n))

    other, other_changed, other_g, other_g_changed, next_a, next_b, next_a_again, next_b_again = z3.Reals(
        "other_rate other_rate_changed other_granularity other_granularity_changed next_a next_b next_a_again next_b_again")
    add("group_locality", "Changing another independent group's rate/scale leaves the first group's result unchanged.",
        bounded + [other >= 0, other_changed >= 0, other_g > 0, other_g_changed > 0,
                   next_a == result,
                   next_b == update(z3, other, other_g, delta, slope, lower, upper),
                   next_a_again == update(z3, rate, granularity, delta, slope, lower, upper),
                   next_b_again == update(z3, other_changed, other_g_changed, delta, slope, lower, upper)],
        next_a == next_a_again)

    patience, length, next_length = z3.Ints("patience length next_length")
    capacity = patience + 2
    add("window_length_base", "The empty history satisfies the abstract patience+2 capacity invariant.",
        [patience >= 0, length == 0], z3.And(length >= 0, length <= capacity), "QF_LIA")
    recurrence = next_length == z3.If(length + 1 > capacity, capacity, length + 1)
    add("window_length_inductive_step", "Appending then truncating preserves bounded nonnegative history length.",
        [patience >= 0, length >= 0, length <= capacity, recurrence],
        z3.And(next_length >= 0, next_length <= capacity, next_length >= length), "QF_LIA")
    add("window_length_saturates", "Once full, the abstract window stays at capacity on the next append.",
        [patience >= 0, length == capacity, recurrence], next_length == capacity, "QF_LIA")
    if tuple(item.name for item in obligations) != EXPECTED_OBLIGATIONS:
        raise ValueError("Proof inventory differs from the explicit contract")
    return obligations


def classify_results(nonvacuity, negation):
    """Unknown, contradictory premises, counterexamples, and missing results never prove a claim."""
    if nonvacuity != "sat":
        return "vacuous" if nonvacuity == "unsat" else "inconclusive"
    return {"unsat": "proved", "sat": "counterexample"}.get(negation, "inconclusive")


def check_query(z3, logic, assertions, timeout_ms, query_path):
    solver = z3.SolverFor(logic)
    solver.set(timeout=timeout_ms)
    solver.add(*assertions)
    query = f"(set-option :timeout {timeout_ms})\n(set-logic {logic})\n" + solver.to_smt2()
    query_path.write_text(query, encoding="utf-8")
    started = time.monotonic()
    result = str(solver.check())
    record = {"result": result, "elapsed_seconds": time.monotonic() - started,
              "timeout_ms": timeout_ms, "query_file": query_path.name,
              "query_sha256": hashlib.sha256(query_path.read_bytes()).hexdigest(),
              "statistics": {key: value for key, value in solver.statistics()}}
    if result == "sat":
        record["model"] = solver.model().sexpr()
    elif result == "unknown":
        record["reason_unknown"] = solver.reason_unknown()
    return record


def verify_obligation(z3, obligation, output, timeout_ms):
    record = {"name": obligation.name, "description": obligation.description,
              "logic": obligation.logic, "assumptions": [a.sexpr() for a in obligation.assumptions],
              "conclusion": obligation.conclusion.sexpr(), "status": "inconclusive"}
    record["nonvacuity"] = check_query(z3, obligation.logic, obligation.assumptions, timeout_ms,
                                       output / (obligation.name + "-assumptions.smt2"))
    if record["nonvacuity"]["result"] == "sat":
        record["negation"] = check_query(z3, obligation.logic,
                                         [*obligation.assumptions, z3.Not(obligation.conclusion)],
                                         timeout_ms, output / (obligation.name + "-negation.smt2"))
    record["status"] = classify_results(record["nonvacuity"]["result"],
                                        record.get("negation", {}).get("result"))
    return record


def suite_passed(records):
    return (tuple(record["name"] for record in records) == EXPECTED_OBLIGATIONS
            and all(record["status"] == "proved"
                    and record["nonvacuity"]["result"] == "sat"
                    and record["negation"]["result"] == "unsat" for record in records))


def write_report(path, report):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new directory for report and replayable SMT-LIB queries")
    parser.add_argument("--timeout-ms", type=int, default=5000, help="deadline per solver query, 1..30000 ms")
    args = parser.parse_args(argv)
    if not 1 <= args.timeout_ms <= 30000:
        parser.error("--timeout-ms must be between 1 and 30000")
    output = (args.output or ROOT / ".audit" / ("contract-proofs-" + uuid.uuid4().hex[:12])).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "report.json"
    started = time.monotonic()
    report = {"schema_version": 1, "status": "running", "proofs_passed": False,
              "scope": "exact-real DLRS abstraction plus integer window-length recurrence",
              "quantification": "UNSAT of assumptions and negated conclusion establishes the implication for all free SMT constants.",
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "python": sys.version, "platform": platform.platform(),
              "timeout_ms_per_query": args.timeout_ms, "expected_obligations": list(EXPECTED_OBLIGATIONS),
              "limitations": LIMITATIONS, "obligations": [],
              "source_sha256": {name: (hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                       if (ROOT / name).is_file() else None)
                                for name in ("benchmarks/verify_contract.py", "dlrs/scheduler.py", "dlrs/utils.py")}}
    write_report(report_path, report)
    code = 1
    try:
        z3 = importlib.import_module("z3")
        report["solver"] = {"name": "Z3", "version": z3.get_version_string(),
                            "full_version": z3.get_full_version(),
                            "package_version": importlib.metadata.version("z3-solver")}
        for obligation in build_obligations(z3):
            record = verify_obligation(z3, obligation, output, args.timeout_ms)
            report["obligations"].append(record)
            write_report(report_path, report)
            print(f"{obligation.name}: {record['status']}", flush=True)
        report["proofs_passed"] = suite_passed(report["obligations"])
        report["status"] = "passed" if report["proofs_passed"] else "failed"
        code = 0 if report["proofs_passed"] else 1
    except ModuleNotFoundError as error:
        report.update(status="unavailable", reason=str(error))
        code = 2
    except KeyboardInterrupt:
        report.update(status="interrupted", reason="Interrupted; completed obligations retained")
        code = 130
    except Exception as error:
        report.update(status="failed", reason=str(error), traceback=traceback.format_exc())
    finally:
        report.update(finished_at_utc=datetime.now(timezone.utc).isoformat(), exit_code=code,
                      elapsed_seconds=time.monotonic() - started)
        write_report(report_path, report)
    print(json.dumps({"status": report["status"], "proofs_passed": report["proofs_passed"],
                      "completed_obligations": len(report["obligations"]), "report": str(report_path)}), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
