"""CPU-safe acceptance checks for the CUDA evidence gate, not CUDA validation."""

import json
import os
import sys
import tarfile
import xml.etree.ElementTree as ET
import zipfile

import pytest

from benchmarks import validate_cuda as runner


def report(tmp_path, *, bf16=True, changed=None, removed=None, added=None):
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    for name in sorted(runner.REQUIRED_CASES | runner.BF16_CASES):
        if name == removed:
            continue
        case = ET.SubElement(suite, "testcase", name=name)
        state = (changed or {}).get(name)
        if state is None and not bf16 and name in runner.BF16_CASES:
            state = "skipped"
        if state:
            ET.SubElement(case, state, message="Synthetic gate-regression fixture")
    if added:
        ET.SubElement(suite, "testcase", name=added)
    path = tmp_path / "pytest.xml"
    ET.ElementTree(root).write(path)
    return path


@pytest.mark.parametrize("bf16", [False, True])
def test_complete_cuda_contract_requires_actual_required_cases(tmp_path, bf16):
    result = runner.evaluate_junit(report(tmp_path, bf16=bf16), bf16)
    assert result["cuda_gate_passed"]
    assert result["mandatory_cases_passed"]
    assert result["counts"]["passed"] == (20 if bf16 else 18)
    assert result["bf16_status"] == ("passed" if bf16 else "unsupported")


@pytest.mark.parametrize("state", ["skipped", "failure", "error"])
def test_required_cuda_nonpass_blocks_gate(tmp_path, state):
    name = sorted(runner.REQUIRED_CASES)[0]
    result = runner.evaluate_junit(report(tmp_path, changed={name: state}), True)
    assert not result["cuda_gate_passed"]
    assert not result["mandatory_cases_passed"]


@pytest.mark.parametrize("defect", ["missing_required", "missing_optional", "duplicate", "unexpected"])
def test_incomplete_or_changed_cuda_inventory_cannot_pass(tmp_path, defect):
    required, optional = sorted(runner.REQUIRED_CASES)[0], sorted(runner.BF16_CASES)[0]
    path = report(tmp_path, removed=required if defect == "missing_required" else
                  optional if defect == "missing_optional" else None,
                  added=required if defect == "duplicate" else
                  "unreviewed_cuda_case" if defect == "unexpected" else None)
    assert not runner.evaluate_junit(path, True)["cuda_gate_passed"]


@pytest.mark.parametrize("bf16", [False, True])
def test_bf16_results_must_agree_with_probe(tmp_path, bf16):
    result = runner.evaluate_junit(report(tmp_path, bf16=not bf16), bf16)
    assert not result["cuda_gate_passed"]
    assert result["bf16_status"] == "failed"


def test_collection_error_outside_testcases_blocks_gate(tmp_path):
    path = report(tmp_path)
    tree = ET.parse(path)
    ET.SubElement(tree.getroot(), "error", message="collection failed")
    tree.write(path)
    assert not runner.evaluate_junit(path, True)["cuda_gate_passed"]


def test_timeout_retains_partial_output_and_failure_state(tmp_path):
    result = runner.run_command([sys.executable, "-u", "-c",
                                 "import time; print('before timeout'); time.sleep(10)"],
                                tmp_path, "timeout", os.environ.copy(), 2)
    assert result["status"] == "timed_out"
    assert "before timeout" in (tmp_path / "timeout.log").read_text()
    assert json.loads((tmp_path / "timeout-command.json").read_text())["status"] == "timed_out"


def test_failing_command_retains_stdout_and_stderr(tmp_path):
    result = runner.run_command([sys.executable, "-c",
                                 "import sys; print('out'); print('err', file=sys.stderr); sys.exit(9)"],
                                tmp_path, "failure", os.environ.copy(), 10)
    assert result["status"] == "failed" and result["returncode"] == 9
    assert set((tmp_path / "failure.log").read_text().splitlines()) == {"out", "err"}


def test_unavailable_cuda_never_runs_pytest_or_passes(tmp_path, monkeypatch, capsys):
    output = tmp_path / "unavailable"
    labels = []

    def fake_run(command, output, label, environment, timeout, **kwargs):
        labels.append(label)
        if label == "environment-probe":
            runner.write_json(output / "environment.json", {"cuda_available": False})
        return {"status": "completed"}

    monkeypatch.setattr(runner, "run_command", fake_run)
    monkeypatch.setattr(runner, "source_identity", lambda: {"files": {}})
    assert runner.main(["--output", str(output)]) == 2
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "unavailable" and not summary["cuda_gate_passed"]
    assert labels == ["nvidia-smi", "environment-probe"]
    with tarfile.open(output.with_name(output.name + ".tar.gz")) as archive:
        assert "unavailable/summary.json" in archive.getnames()
        assert "unavailable/evidence-sha256.json" in archive.getnames()


def test_interruption_retains_nonpassing_archive(tmp_path, monkeypatch, capsys):
    output = tmp_path / "interrupted"

    def interrupt(*_):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "source_identity", interrupt)
    assert runner.main(["--output", str(output)]) == 130
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "interrupted" and not summary["cuda_gate_passed"]
    assert output.with_name(output.name + ".tar.gz").is_file()


def test_previous_evidence_is_never_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        runner.main(["--output", str(tmp_path)])


def test_existing_archive_blocks_run_before_output_directory_creation(tmp_path):
    output = tmp_path / "prior-run"
    archive = tmp_path / "prior-run.tar.gz"
    prior_bytes = b"Prior evidence must survive a reused output name"
    archive.write_bytes(prior_bytes)
    with pytest.raises(FileExistsError, match="existing evidence archive"):
        runner.main(["--output", str(output)])
    assert archive.read_bytes() == prior_bytes
    assert not output.exists()


def test_late_archive_collision_preserves_original_bytes(tmp_path):
    output = tmp_path / "late-collision"
    output.mkdir()
    (output / "summary.json").write_text('{"status": "unavailable"}')
    # Simulate another writer creating the archive after main's preflight.
    archive = tmp_path / "late-collision.tar.gz"
    prior_bytes = b"Another writer completed first"
    archive.write_bytes(prior_bytes)
    with pytest.raises(FileExistsError):
        runner.make_archive(output)
    assert archive.read_bytes() == prior_bytes
    assert (output / "summary.json").read_text() == '{"status": "unavailable"}'


@pytest.mark.parametrize("installed_package", [False, True], ids=["source_only", "installed"])
def test_missing_distribution_metadata_is_allowed_only_for_source_probe(tmp_path, monkeypatch,
                                                                       installed_package):
    import torch

    version = runner.importlib.metadata.version

    def no_dlrs_metadata(name):
        if name == "pytorch-dlrs":
            raise runner.importlib.metadata.PackageNotFoundError(name)
        return version(name)

    monkeypatch.setattr(runner.importlib.metadata, "version", no_dlrs_metadata)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    destination = tmp_path / "environment.json"
    if installed_package:
        with pytest.raises(runner.importlib.metadata.PackageNotFoundError):
            runner.probe_environment(destination, installed_package=True)
        assert not destination.exists()
    else:
        runner.probe_environment(destination)
        environment = json.loads(destination.read_text())
        assert environment["distribution_version"] is None
        assert environment["dlrs_version"]
        assert set(environment["imported_source"]) == {"dlrs/__init__.py", "dlrs/scheduler.py", "dlrs/utils.py"}
        assert environment["cuda_available"] is False


def test_source_mutation_cannot_receive_a_stale_passing_fingerprint(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_text("before")
    identity = {"files": {"source.py": runner.sha256(source)}}
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    source.write_text("after")
    with pytest.raises(ValueError, match="Source changed during validation"):
        runner.verify_unchanged(identity, {"imported_source": {}}, source)


@pytest.mark.parametrize("import_location", ["installed", "checkout", "editable"])
def test_installed_mode_isolates_pytest_and_rejects_source_imports(tmp_path, monkeypatch, import_location):
    checkout = tmp_path / "checkout"
    installed = tmp_path / "site-packages"
    checkout.mkdir()
    installed.mkdir()
    sources = ("dlrs/__init__.py", "dlrs/scheduler.py", "dlrs/utils.py", runner.TEST_FILE)
    for name in sources:
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Synthetic source for runner isolation regression\n")
        installed_path = installed / name
        installed_path.parent.mkdir(parents=True, exist_ok=True)
        installed_path.write_bytes(path.read_bytes())
    identity = {"files": {name: runner.sha256(checkout / name) for name in sources}}
    wheel = tmp_path / "pytorch_dlrs-0.3.0.dev0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in sources[:3]:
            archive.write(checkout / name, name)
        archive.writestr("pytorch_dlrs-0.3.0.dev0.dist-info/METADATA",
                         "Name: pytorch-dlrs\nVersion: 0.3.0.dev0\n")
    invoked_pytest = []

    def fake_run(command, output, label, environment, timeout, cwd=None):
        if label in ("environment-probe", "pytest"):
            assert "-I" in command and cwd != checkout and checkout not in cwd.parents
            assert "PYTHONPATH" not in environment
        if label == "environment-probe":
            imported_root = checkout if import_location == "checkout" else installed
            runner.write_json(output / "environment.json", {
                "cuda_available": True, "visible_device_count": 1, "bf16_supported": False,
                "distribution_version": "0.3.0.dev0", "dlrs_version": "0.3.0.dev0",
                "installed_in_site_packages": import_location == "installed",
                "imported_source": {name: {"path": str(imported_root / name),
                                           "sha256": identity["files"][name]} for name in sources[:3]},
            })
        elif label == "pytest":
            invoked_pytest.append(True)
            test_path = cwd / "test_cuda_validation.py"
            assert str(test_path) in command
            assert test_path.read_bytes() == (checkout / runner.TEST_FILE).read_bytes()
            report(output, bf16=False)
        return {"status": "completed"}

    monkeypatch.setattr(runner, "ROOT", checkout)
    monkeypatch.setattr(runner, "source_identity", lambda: identity)
    monkeypatch.setattr(runner, "run_command", fake_run)
    output = tmp_path / "result"
    code = runner.main(["--output", str(output), "--installed-package", "--wheel", str(wheel)])
    summary = json.loads((output / "summary.json").read_text())
    assert (code == 0) == (import_location == "installed")
    assert summary["cuda_gate_passed"] == (import_location == "installed")
    assert bool(invoked_pytest) == (import_location == "installed")
