"""Diagnose upstream training-mode behavior without importing DLRS.

This probe records evidence; it does not turn framework failures into passes.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn


def hook_probe():
    records = []

    def update(optimizer_name, optimizer=None, parameter=None):
        if optimizer is None:
            parameter = nn.Parameter(torch.ones(2))
            optimizer = getattr(torch.optim, optimizer_name)([parameter], lr=0.01, foreach=False)
        calls = []
        handle = optimizer.register_step_post_hook(lambda *_: calls.append("completed"))
        before = parameter.detach().clone()
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        handle.remove()
        return optimizer, parameter, {"optimizer": optimizer_name, "hook_calls": len(calls),
                                      "parameter_changed": not torch.equal(before, parameter)}

    existing = {}
    for name in ("SGD", "AdamW"):
        optimizer, parameter, result = update(name)
        existing[name] = optimizer, parameter
        records.append(dict(result, phase="before_model_compile"))
    model = torch.compile(nn.Linear(2, 1), backend="aot_eager", fullgraph=True)
    model(torch.ones(2, 2)).sum().backward()
    for name in ("SGD", "AdamW"):
        _, _, result = update(name, *existing[name])
        records.append(dict(result, phase="existing_optimizer_after_model_compile"))
        _, _, result = update(name)
        records.append(dict(result, phase="new_optimizer_after_model_compile"))
    return records


def mps_probe():
    import torch.mps

    if not torch.backends.mps.is_available():
        return {"available": False}
    records = []
    for operation in ("rand", "dropout"):
        for synchronize in (False, True):
            torch.mps.manual_seed(631)
            inputs = torch.ones(256, device="mps")

            def sample(operation=operation, inputs=inputs):
                if operation == "rand":
                    return torch.rand(256, device="mps")
                return nn.functional.dropout(inputs, p=0.25, training=True)

            sample()  # Exercise an already-used generator, as a training checkpoint does.
            if synchronize:
                torch.mps.synchronize()
            state = torch.mps.get_rng_state().clone()
            expected = [sample().cpu() for _ in range(3)]
            advanced = torch.mps.get_rng_state().clone()
            torch.mps.set_rng_state(state)
            restored = torch.mps.get_rng_state().clone()
            actual = [sample().cpu() for _ in range(3)]
            records.append({
                "operation": operation, "synchronize_before_checkpoint": synchronize,
                "state_bytes": state.numel(), "generator_state_advanced": not torch.equal(state, advanced),
                "reported_state_restored": torch.equal(state, restored),
                "outputs_equal": [torch.equal(left, right) for left, right in zip(actual, expected)],
                "maximum_absolute_difference": max(float((left - right).abs().max())
                                                    for left, right in zip(actual, expected)),
            })
    return {"available": True, "results": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    output = {"python": platform.python_version(), "torch": str(torch.__version__),
              "platform": platform.platform(), "hooks": hook_probe(), "mps_rng": mps_probe(),
              "dlrs_imported": "dlrs" in sys.modules,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "completed_utc": datetime.now(timezone.utc).isoformat(),
              "mps_fallback_environment": os.getenv("PYTORCH_ENABLE_MPS_FALLBACK")}
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
