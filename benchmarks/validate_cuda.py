"""Bounded, single-process CUDA validation with attributable retained evidence.

Uses the current interpreter and its existing PyTorch installation. This runner
never installs packages, acquires compute, or launches distributed workers.
"""

import argparse
import email
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import time
import traceback
import uuid
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = "tests/test_cuda_validation.py"
# This explicit contract is intentionally independent of pytest collection:
# deleting or skipping a required check must not silently produce a green gate.
REQUIRED_CASES = {
    f"{name}[{rate}-{optimizer}]"
    for name in ("test_cuda_eager_training_matches_independent_rate_reference",
                 "test_cuda_float32_stochastic_checkpoint_restores_all_state")
    for rate in ("float_lr", "cuda_tensor_lr") for optimizer in ("SGD", "AdamW")
} | {
    f"test_cuda_fp16_scaler_overflow_matches_explicitly_omitted_update[{kind}-{optimizer}]"
    for kind in ("one_skipped", "all_skipped") for optimizer in ("SGD", "AdamW")
} | {
    f"test_cuda_fp16_stochastic_checkpoint_restores_scaler_and_rng[{optimizer}]"
    for optimizer in ("SGD", "AdamW")
} | {
    f"test_cuda_gradient_accumulation_matches_full_effective_batches[{precision}-{optimizer}]"
    for precision in ("float32", "fp16") for optimizer in ("SGD", "AdamW")
}
BF16_CASES = {f"test_cuda_bfloat16_eager_reference[{optimizer}]" for optimizer in ("SGD", "AdamW")}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity():
    files = ["dlrs/__init__.py", "dlrs/scheduler.py", "dlrs/utils.py",
             "pyproject.toml", "benchmarks/validate_cuda.py", TEST_FILE]
    identity = {"files": {name: sha256(ROOT / name) for name in files},
                "scope": "checkout source; not an installed-wheel acceptance test"}
    for key, arguments in [("git_head", ["rev-parse", "HEAD"]),
                           ("git_tracked_changes", ["status", "--porcelain", "--untracked-files=no"])]:
        try:
            result = subprocess.run(["git", *arguments], cwd=ROOT, text=True,
                                    capture_output=True, timeout=10)
            identity[key] = result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            identity[key] = None
    return identity


def probe_environment(destination, installed_package=False):
    """Run only in the bounded child, so a CUDA/driver import cannot hang the runner."""
    if not installed_package:
        sys.path.insert(0, str(ROOT))
    import torch

    import dlrs
    import dlrs.scheduler
    import dlrs.utils

    try:
        distribution_version = importlib.metadata.version("pytorch-dlrs")
    except importlib.metadata.PackageNotFoundError:
        if installed_package:
            raise
        distribution_version = None  # A source checkout need not be installed.
    available = torch.cuda.is_available()
    result = {
        "recorded_at_utc": timestamp(), "python": sys.version,
        "executable": sys.executable, "platform": platform.platform(),
        "machine": platform.machine(), "torch": str(torch.__version__),
        "torch_file": torch.__file__, "torch_cuda_build": torch.version.cuda,
        "torch_git_version": torch.version.git_version,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": available, "visible_device_count": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "bf16_supported": False, "selected_device": None,
        "dlrs_version": dlrs.__version__,
        "distribution_version": distribution_version,
        "installed_in_site_packages": Path(dlrs.__file__).resolve().parent.parent in {
            Path(sysconfig.get_path(key)).resolve() for key in ("purelib", "platlib")},
        "imported_source": {name: {"path": str(Path(module.__file__).resolve()),
                                   "sha256": sha256(Path(module.__file__))}
                            for name, module in [("dlrs/__init__.py", dlrs),
                                                 ("dlrs/scheduler.py", dlrs.scheduler),
                                                 ("dlrs/utils.py", dlrs.utils)]},
        "packages": sorted([{"name": dist.metadata["Name"], "version": dist.version}
                            for dist in importlib.metadata.distributions()],
                           key=lambda item: (item["name"] or "").lower()),
    }
    if available:
        torch.cuda.set_device(0)
        properties = torch.cuda.get_device_properties(0)
        result["selected_device"] = {
            "index": 0, "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
            "torch_arch_list": torch.cuda.get_arch_list(),
        }
        try:
            result["bf16_supported"] = bool(torch.cuda.is_bf16_supported(including_emulation=False))
        except TypeError:
            result["bf16_supported"] = (bool(torch.cuda.is_bf16_supported())
                                        and properties.major >= 8)
        result["bf16_including_emulation"] = bool(torch.cuda.is_bf16_supported())
        # Exercise allocation and synchronization, not merely device enumeration.
        torch.ones(1, device="cuda:0").add_(1)
        torch.cuda.synchronize(0)
    write_json(destination, result)


def run_command(command, output, label, environment, timeout, cwd=None):
    """Stream output to disk and terminate the child on timeout or interruption."""
    record = {"command": command, "started_at_utc": timestamp(),
              "timeout_seconds": timeout, "status": "running", "cwd": str(cwd or ROOT)}
    metadata = output / (label + "-command.json")
    write_json(metadata, record)
    started = time.monotonic()
    process = None
    try:
        with (output / (label + ".log")).open("w", encoding="utf-8") as logfile:
            process = subprocess.Popen(command, cwd=cwd or ROOT, env=environment,
                                       stdout=logfile, stderr=subprocess.STDOUT,
                                       start_new_session=(os.name == "posix"))
            record["returncode"] = process.wait(timeout=timeout)
            record["status"] = "completed" if record["returncode"] == 0 else "failed"
    except subprocess.TimeoutExpired:
        record["status"] = "timed_out"
    except KeyboardInterrupt:
        record["status"] = "interrupted"
        raise
    except OSError as error:
        record["status"] = "unavailable"
        record["error"] = str(error)
    finally:
        if process is not None and process.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
            except ProcessLookupError:
                pass  # The child may exit between poll and termination.
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                except ProcessLookupError:
                    pass
                process.wait(timeout=3)
        record["elapsed_seconds"] = time.monotonic() - started
        record["finished_at_utc"] = timestamp()
        write_json(metadata, record)
    return record


def evaluate_junit(path, bf16_supported):
    """Reject missing, duplicate, unexpected, skipped-required, or failing checks."""
    if not REQUIRED_CASES or not BF16_CASES:
        raise ValueError("CUDA test contract has not been defined")
    root = ET.parse(path).getroot()
    cases = []
    for case in root.iter("testcase"):
        outcome = "passed"
        details = ""
        for tag, state in [("skipped", "skipped"), ("failure", "failed"), ("error", "error")]:
            node = case.find(tag)
            if node is not None:
                outcome, details = state, node.attrib.get("message", "")
        cases.append({"name": case.attrib.get("name", ""), "outcome": outcome,
                      "details": details, "seconds": case.attrib.get("time"),
                      "properties": {node.attrib.get("name"): node.attrib.get("value")
                                     for node in case.findall("properties/property")}})
    names = [case["name"] for case in cases]
    expected = REQUIRED_CASES | BF16_CASES
    reasons = []
    if len(names) != len(set(names)) or set(names) != expected:
        reasons.append("Collected cases differ from the explicit CUDA test contract")
    if len(list(root.iter("error"))) or len(list(root.iter("failure"))):
        reasons.append("JUnit contains failures or errors")
    required = [case for case in cases if case["name"] in REQUIRED_CASES]
    optional = [case for case in cases if case["name"] in BF16_CASES]
    mandatory_passed = (len(required) == len(REQUIRED_CASES)
                        and all(case["outcome"] == "passed" for case in required))
    if not mandatory_passed:
        reasons.append("Every required CUDA case must pass without skips")
    expected_optional = "passed" if bf16_supported else "skipped"
    bf16_valid = (len(optional) == len(BF16_CASES)
                 and all(case["outcome"] == expected_optional for case in optional))
    if not bf16_valid:
        reasons.append("BF16 outcomes do not agree with the detected capability")
    return {"cuda_gate_passed": not reasons, "mandatory_cases_passed": mandatory_passed,
            "bf16_status": ("passed" if bf16_supported else "unsupported") if bf16_valid else "failed",
            "reasons": reasons, "cases": cases,
            "counts": {state: sum(case["outcome"] == state for case in cases)
                       for state in ("passed", "skipped", "failed", "error")}}


def make_archive(output):
    files = sorted(path for path in output.iterdir()
                   if path.is_file() and path.name != "evidence-sha256.json")
    write_json(output / "evidence-sha256.json", {path.name: sha256(path) for path in files})
    archive_path = output.with_name(output.name + ".tar.gz")
    # Exclusive creation also protects against a file appearing after preflight.
    with archive_path.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for path in sorted(output.iterdir()):
            if path.is_file():
                archive.add(path, arcname=output.name + "/" + path.name, recursive=False)
    return archive_path


def wheel_identity(wheel, identity, output):
    """Bind an installed-package run to the exact wheel and reviewed source."""
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise ValueError("Expected exactly one wheel metadata record")
        metadata = email.message_from_bytes(archive.read(names[0]))
        if metadata["Name"] != "pytorch-dlrs":
            raise ValueError("The supplied wheel is not pytorch-dlrs")
        for name in ("dlrs/__init__.py", "dlrs/scheduler.py", "dlrs/utils.py"):
            if hashlib.sha256(archive.read(name)).hexdigest() != identity["files"][name]:
                raise ValueError("Wheel source differs from the fingerprinted checkout")
    result = {"filename": wheel.name, "sha256": sha256(wheel), "name": metadata["Name"],
              "version": metadata["Version"], "requires_dist": metadata.get_all("Requires-Dist", [])}
    shutil.copyfile(wheel, output / wheel.name)
    write_json(output / "wheel.json", result)
    return result


def verify_unchanged(identity, detected, test_path, wheel_path=None, wheel=None):
    for name, expected in identity["files"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError("Source changed during validation: " + name)
    for name, imported in detected["imported_source"].items():
        if sha256(Path(imported["path"])) != identity["files"][name]:
            raise ValueError("Imported source changed during validation: " + name)
    if sha256(test_path) != identity["files"][TEST_FILE]:
        raise ValueError("Executed test source changed during validation")
    if wheel_path is not None and sha256(wheel_path) != wheel["sha256"]:
        raise ValueError("Wheel changed during validation")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new evidence directory (must not exist)")
    parser.add_argument("--timeout", type=int, default=300, help="pytest deadline, 1–600 seconds")
    parser.add_argument("--installed-package", action="store_true",
                        help="test the installed wheel outside the source checkout")
    parser.add_argument("--wheel", type=Path, help="exact installed wheel; required with --installed-package")
    parser.add_argument("--probe", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.probe:
        probe_environment(args.probe, args.installed_package)
        return 0
    if not 1 <= args.timeout <= 600:
        parser.error("--timeout must be between 1 and 600 seconds")
    if args.installed_package != (args.wheel is not None):
        parser.error("--installed-package and --wheel must be supplied together")
    output = (args.output or ROOT / ".audit" / ("cuda-" + uuid.uuid4().hex[:12])).resolve()
    archive_path = output.with_name(output.name + ".tar.gz")
    if archive_path.exists() or archive_path.is_symlink():
        raise FileExistsError("Refusing to overwrite existing evidence archive: " + str(archive_path))
    output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS"):
        environment.pop(key, None)
    environment.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="2",
                       MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                       PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONUNBUFFERED="1")
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    environment["CUDA_VISIBLE_DEVICES"] = "0" if visible is None else visible.split(",")[0]
    summary = {"schema_version": 1, "started_at_utc": timestamp(), "status": "running",
               "cuda_gate_passed": False, "scope": "single process, one visible CUDA GPU",
               "package_mode": "installed_wheel" if args.installed_package else "checkout_source",
               "original_cuda_visible_devices": visible,
               "pytest_timeout_seconds": args.timeout, "environment_probe_timeout_seconds": 60}
    summary_path = output / "summary.json"
    write_json(summary_path, summary)
    exit_code = 1
    previous_handler = signal.getsignal(signal.SIGTERM)

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    workspace = None
    try:
        identity = source_identity()
        identity["scope"] = summary["package_mode"]
        write_json(output / "source.json", identity)
        wheel = wheel_identity(args.wheel.resolve(), identity, output) if args.installed_package else None
        workspace = tempfile.TemporaryDirectory(prefix="dlrs-cuda-pytest-")
        working_directory = Path(workspace.name) if args.installed_package else ROOT
        run_command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                     "--format=csv,noheader,nounits"], output, "nvidia-smi", environment, 10)
        probe_command = [sys.executable]
        if args.installed_package:
            probe_command.append("-I")
        probe_command.extend([str(Path(__file__).resolve()), "--probe", str(output / "environment.json")])
        if args.installed_package:
            probe_command.append("--installed-package")
        probe = run_command(probe_command, output, "environment-probe", environment, 60,
                            cwd=working_directory)
        if probe["status"] != "completed":
            summary.update(status=probe["status"], reason="Environment probe did not complete")
            return exit_code
        detected = json.loads((output / "environment.json").read_text(encoding="utf-8"))
        if not detected["cuda_available"]:
            summary.update(status="unavailable", reason="PyTorch cannot use a CUDA GPU")
            exit_code = 2
            return exit_code
        if detected["visible_device_count"] != 1:
            raise ValueError("Expected exactly one visible CUDA GPU")
        for name, imported in detected["imported_source"].items():
            if imported["sha256"] != identity["files"][name]:
                raise ValueError("Imported DLRS source does not match the fingerprinted checkout")
            if args.installed_package and ROOT in Path(imported["path"]).parents:
                raise ValueError("Installed-package validation imported the checkout source")
        if wheel and (detected["distribution_version"] != wheel["version"]
                      or detected["dlrs_version"] != wheel["version"]):
            raise ValueError("Imported/installed version does not match the supplied wheel")
        if args.installed_package and not detected["installed_in_site_packages"]:
            raise ValueError("Expected an installed site-packages distribution, not an editable source")
        write_json(summary_path, {**summary, "stage": "pytest"})
        test_path = ROOT / TEST_FILE
        if args.installed_package:
            test_path = working_directory / Path(TEST_FILE).name
            shutil.copyfile(ROOT / TEST_FILE, test_path)
        test_command = [sys.executable] + (["-I"] if args.installed_package else [])
        test_command.extend(["-m", "pytest", str(test_path), "-q", "-o", "addopts=", "--maxfail=1", "-rA",
                             "--basetemp=" + str(Path(workspace.name) / "test-temp"),
                             "--junitxml=" + str(output / "pytest.xml")])
        test_run = run_command(test_command, output, "pytest", environment, args.timeout,
                               cwd=working_directory)
        summary["pytest"] = test_run
        verify_unchanged(identity, detected, test_path, args.wheel, wheel)
        summary["fingerprints_unchanged_after_tests"] = True
        if (output / "pytest.xml").exists():
            summary["test_results"] = evaluate_junit(output / "pytest.xml", detected["bf16_supported"])
        passed = (test_run["status"] == "completed"
                  and summary.get("test_results", {}).get("cuda_gate_passed", False))
        summary.update(status="passed" if passed else test_run["status"], cuda_gate_passed=passed)
        if not passed and summary["status"] == "completed":
            summary["status"] = "failed"
        exit_code = 0 if passed else 1
    except KeyboardInterrupt:
        summary.update(status="interrupted", reason="Runner interrupted; retained partial evidence")
        exit_code = 130
    except Exception as error:
        summary.update(status="failed", reason=str(error))
        (output / "runner-error.log").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        if workspace is not None:
            workspace.cleanup()
        signal.signal(signal.SIGTERM, previous_handler)
        summary.update(finished_at_utc=timestamp(), exit_code=exit_code)
        write_json(summary_path, summary)
        archive = make_archive(output)
        print(json.dumps({"status": summary["status"], "cuda_gate_passed": summary["cuda_gate_passed"],
                          "summary": str(summary_path), "archive": str(archive)}, indent=2), flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
