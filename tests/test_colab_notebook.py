"""Execute the real notebook run cell without allocating compute or subprocesses."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/free_colab_cuda_validation.ipynb"


def _run_cell():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = ["".join(cell["source"]) for cell in notebook["cells"]
             if cell["cell_type"] == "code"]
    matches = [source for source in cells
               if "subprocess.Popen(" in source and "run_state" in source]
    assert len(matches) == 1, "Expected the notebook's single CUDA runner cell"
    return compile(matches[0], str(NOTEBOOK) + ":run-cell", "exec")


@pytest.mark.parametrize(
    "completion,summary,passes",
    [
        (0, {"status": "passed", "cuda_gate_passed": True}, True),
        (9, {"status": "passed", "cuda_gate_passed": True}, False),
        ("interrupt", {"status": "passed", "cuda_gate_passed": True}, False),
        ("timeout", {"status": "passed", "cuda_gate_passed": True}, False),
        (0, None, False),
        (0, {"status": "failed", "cuda_gate_passed": True}, False),
        (0, {"status": "passed", "cuda_gate_passed": False}, False),
        (0, {"status": "passed", "cuda_gate_passed": "true"}, False),
    ],
    ids=["success", "nonzero-exit", "interrupted", "timed-out", "missing-summary",
         "failed-summary", "failed-gate", "nonboolean-gate"],
)
def test_notebook_display_requires_completed_successful_execution(
    tmp_path, capsys, completion, summary, passes
):
    setup, evidence = tmp_path / "setup", tmp_path / "evidence"
    setup.mkdir()
    evidence.mkdir()
    summary_path = evidence / "summary.json"
    original_summary = json.dumps(summary) + "\n" if summary is not None else None
    if original_summary is not None:
        summary_path.write_text(original_summary, encoding="utf-8")

    class FakeProcess:
        terminated = False
        waits = 0

        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                if completion == "interrupt":
                    raise KeyboardInterrupt
                if completion == "timeout":
                    raise subprocess.TimeoutExpired("synthetic CUDA runner", timeout)
                return completion
            assert self.terminated, "Interrupted runner must be terminated before waiting"
            return -15

        def terminate(self):
            self.terminated = True

        def kill(self):
            pytest.fail("This responsive fake process should not need forced termination")

    process = FakeProcess()
    launches = []

    def popen(command, *, stdout, stderr, cwd):
        launches.append(command)
        assert cwd == tmp_path
        stdout.write("Retained synthetic runner output\n")
        return process

    namespace = {
        "setup_state": {"status": "ready"}, "sys": sys, "json": json,
        "SOURCE": tmp_path / "source", "WHEEL": setup / "reviewed.whl",
        "EVIDENCE": evidence, "SETUP": setup, "BASE": tmp_path,
        "subprocess": SimpleNamespace(Popen=popen, STDOUT=subprocess.STDOUT,
                                      TimeoutExpired=subprocess.TimeoutExpired),
    }
    exec(_run_cell(), namespace)

    output = capsys.readouterr().out
    if passes:
        assert "Required CUDA gate: PASSED\n" in output
    else:
        assert "NOT PASSED" in output
        assert "Required CUDA gate: PASSED\n" not in output
    assert len(launches) == 1
    assert (setup / "runner-console.log").read_text() == "Retained synthetic runner output\n"
    recorded = json.loads((setup / "notebook-run.json").read_text())
    if completion in ("interrupt", "timeout"):
        assert recorded["status"] == "interrupted_or_timed_out"
        assert process.terminated
    else:
        assert recorded["status"] == "completed"
        assert recorded["returncode"] == completion
    if original_summary is None:
        assert not summary_path.exists()
    else:
        assert summary_path.read_text() == original_summary, "Raw runner evidence must be preserved"
