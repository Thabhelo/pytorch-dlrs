"""Test installed wheel/sdist packages using tests extracted outside the checkout.

The target interpreter must have pip, a suitable PyTorch build, and test extras.
Neither an editable installation nor a checkout-local dlrs package is accepted.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def extract_source(archive_path, destination):
    """Extract regular files and directories without permitting path/link escapes."""
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("source archive contains an unsafe path")
            target = destination / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("source archive member cannot be read")
                with stream, target.open("wb") as output:
                    shutil.copyfileobj(stream, output)
            else:
                raise ValueError("source archive contains an unsupported link or member")
    roots = list(destination.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise ValueError("expected one source archive root directory")
    return roots[0]


def run(command, cwd, environment, logfile, timeout=600):
    try:
        result = subprocess.run(
            command, cwd=cwd, env=environment, text=True, capture_output=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        # TimeoutExpired can carry bytes even when text=True was requested.
        def decoded(value):
            return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""

        logfile.write_text(decoded(error.stdout) + decoded(error.stderr)
                           + f"\nCommand timed out after {timeout} seconds.\n")
        logfile.with_suffix(".timeout.json").write_text(json.dumps(
            dict(timed_out=True, timeout_seconds=timeout, command=command), indent=2) + "\n")
        print(f"Command timed out; partial output retained in {logfile}", flush=True)
        raise
    logfile.write_text(result.stdout + result.stderr)
    print("\n".join((result.stdout + result.stderr).splitlines()[-20:]), flush=True)
    if result.returncode:
        raise RuntimeError(f"command failed; see {logfile}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, help="non-editable target environment Python")
    parser.add_argument("--dist", default="dist", help="directory containing one wheel and sdist")
    parser.add_argument("--output", required=True, help="directory for logs and JSON/XML evidence")
    args = parser.parse_args()
    python = str(Path(args.python).absolute())  # Do not resolve a venv's interpreter symlink.
    distributions = Path(args.dist).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    wheels = list(distributions.glob("*.whl"))
    sources = list(distributions.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("dist must contain exactly one wheel and one source archive")
    # Record identities before running commands so even an early install or
    # test failure remains attributable to the exact distributions.
    (output / "distributions.json").write_text(json.dumps([
        dict(artifact=artifact.name, sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
        for artifact in (wheels[0], sources[0])
    ], indent=2) + "\n")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = "1"
    evidence = []
    with tempfile.TemporaryDirectory(prefix="dlrs-distribution-tests-") as temporary:
        workspace = Path(temporary)
        extracted = workspace / "extracted"
        extracted.mkdir()
        source = extract_source(sources[0], extracted)
        # Removing the source package is intentional: tests/examples/benchmarks
        # remain available, while dlrs must resolve from the installed artifact.
        shutil.rmtree(source / "dlrs")
        for artifact in (wheels[0], sources[0]):
            kind = "wheel" if artifact.suffix == ".whl" else "sdist"
            run([python, "-m", "pip", "install", "--no-deps", "--force-reinstall", str(artifact)],
                workspace, environment, output / (kind + "-install.log"))
            run([python, "-m", "pip", "check"], workspace, environment,
                output / (kind + "-dependencies.log"))
            run([python, "-m", "pip", "list", "--format=json"], workspace, environment,
                output / (kind + "-installed-packages.log"))
            probe = run([python, "-I", "-c", '''
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sysconfig
import torch
import dlrs
package = Path(dlrs.__file__).resolve().parent
allowed = {Path(sysconfig.get_path(key)).resolve() for key in ('purelib', 'platlib')}
assert package.parent in allowed, ('Not an installed distribution', str(package))
assert (package / 'py.typed').is_file()
assert dlrs.__version__ == importlib.metadata.version('pytorch-dlrs')
runtime = [dep for dep in importlib.metadata.requires('pytorch-dlrs') or [] if 'extra ==' not in dep]
assert len(runtime) == 1 and runtime[0].startswith('torch>='), runtime
parameter = torch.nn.Parameter(torch.tensor(1.0))
optimizer = torch.optim.SGD([parameter], lr=0.1)
scheduler = dlrs.DLRSScheduler(optimizer)
optimizer.step()
scheduler.step([2.0, 1.0])
assert math.isclose(optimizer.param_groups[0]['lr'], 0.1 + 0.01 * (2 / 3))
scheduler.load_state_dict(scheduler.state_dict())
print(json.dumps(dict(package=str(package), version=dlrs.__version__,
                     python=platform.python_version(), torch=torch.__version__,
                     platform=platform.platform(), runtime_dependencies=runtime)))
'''], workspace, environment, output / (kind + "-probe.log"))
            metadata = json.loads(probe.stdout.strip().splitlines()[-1])
            xml = output / (kind + "-tests.xml")
            run([python, "-m", "pytest", "tests", "-q", "--junitxml=" + str(xml)],
                source, environment, output / (kind + "-tests.log"))
            suites = ET.parse(xml).getroot().findall("testsuite")
            counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
                      for key in ("tests", "failures", "errors", "skipped")}
            evidence.append(dict(artifact=artifact.name, kind=kind,
                                 sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                                 environment=metadata, counts=counts,
                                 source_package_removed=True))
            (output / "results.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print("Both installed distributions passed the extracted test suite.", flush=True)


if __name__ == "__main__":
    main()
