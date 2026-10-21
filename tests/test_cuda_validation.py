"""Bounded, single-process CUDA checks for an opt-in Colab validation run.

These synthetic checks exercise execution and recovery, not workload quality,
CUDA graphs, fused/capturable optimizers, throughput, or distributed training.
The caller records unscaled losses only after completed optimizer updates.
"""

import copy
import io
import math
import os
from contextlib import nullcontext
from decimal import Decimal, localcontext

import pytest
import torch
from torch import nn

from dlrs import DLRSScheduler

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")

DEVICE = torch.device("cuda:0")
MIN_LR = 2.0 ** -14
MAX_LR = 2.0 ** -5


@pytest.fixture(autouse=True)
def deterministic_single_gpu():
    # The standalone runner also sets this before importing torch. Setting it
    # here covers direct pytest runs before their first cuBLAS operation.
    original_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    original_threads = torch.get_num_threads()
    original_deterministic = torch.are_deterministic_algorithms_enabled()
    original_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    original_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    original_benchmark = torch.backends.cudnn.benchmark
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        with torch.cuda.device(DEVICE), torch.random.fork_rng(devices=[0]):
            torch.manual_seed(814)
            yield
            torch.cuda.synchronize(DEVICE)
    finally:
        torch.set_num_threads(original_threads)
        torch.use_deterministic_algorithms(original_deterministic, warn_only=original_warn_only)
        torch.backends.cuda.matmul.allow_tf32 = original_matmul_tf32
        torch.backends.cudnn.allow_tf32 = original_cudnn_tf32
        torch.backends.cudnn.benchmark = original_benchmark
        if original_workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = original_workspace


def _setup(optimizer_name, *, tensor_lr=False, dropout=False):
    model = nn.Sequential(
        nn.Linear(4, 8), nn.Tanh(), nn.Dropout(0.25 if dropout else 0.0), nn.Linear(8, 1)
    ).to(DEVICE)
    rates = [0.012, 0.004]
    if tensor_lr:
        rates = [torch.tensor(rate, device=DEVICE, dtype=torch.float32) for rate in rates]
    groups = [{"params": model[index].parameters(), "lr": rate}
              for index, rate in zip((0, 3), rates)]
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(groups, momentum=0.9, weight_decay=0.001, foreach=False)
    else:
        optimizer = torch.optim.AdamW(groups, weight_decay=0.001, foreach=False)
    scheduler = DLRSScheduler(optimizer, min_lr=MIN_LR, max_lr=MAX_LR)
    return model, optimizer, scheduler


def _data():
    generator = torch.Generator().manual_seed(713)
    inputs = torch.randn(24, 4, generator=generator) * 0.3
    targets = inputs @ torch.tensor([[0.4], [-0.2], [0.6], [-0.1]]) + 0.1
    return inputs.to(DEVICE), targets.to(DEVICE)


def _assert_equal(actual, expected):
    assert type(actual) is type(expected), (type(actual), type(expected))
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_equal(left, right)
    else:
        assert actual == expected


def _paper_rate(rate, losses):
    """Independent high-precision equation; no scheduler/helper calls."""
    with localcontext() as context:
        context.prec = 60
        values = [Decimal.from_float(value) for value in losses]
        slope = (values[-1] - values[0]) / (sum(values) / len(values))
        delta = Decimal("0.5") if slope > 1 else Decimal(1) if slope >= 0 else Decimal("0.1")
        current = Decimal.from_float(rate)
        result = current - Decimal(10) ** current.adjusted() * delta * slope
        return float(min(max(result, Decimal.from_float(MIN_LR)), Decimal.from_float(MAX_LR)))


def _scaler():
    # This CUDA API exists at the project's torch 2.0 floor too.
    scaler = torch.cuda.amp.GradScaler(init_scale=8.0, growth_interval=3)
    assert scaler.is_enabled(), "CUDA GradScaler must not silently disable itself"
    return scaler


def _epoch(model, optimizer, scheduler, *, dtype=None, scaler=None,
           overflow=(), omit=(), shuffle=None):
    inputs, targets = _data()
    if shuffle is not None:
        indices = torch.randperm(len(inputs), generator=shuffle).to(DEVICE)
        inputs, targets = inputs[indices], targets[indices]
    model.train()
    observations, completed, skipped, dtypes = [], [], [], []
    count = [0]

    def completed_step(*_):
        count[0] += 1

    handle = optimizer.register_step_post_hook(completed_step)
    try:
        for window, (x, y) in enumerate(zip(inputs.split(6), targets.split(6))):
            optimizer.zero_grad(set_to_none=True)
            if window in omit:
                # Independent reference: omit the update entirely, without
                # calling optimizer.step or introducing nonfinite gradients.
                assert scaler is not None
                state = scaler.state_dict()
                state["scale"] *= scaler.get_backoff_factor()
                state["_growth_tracker"] = 0
                scaler.load_state_dict(state)
                skipped.append(window)
                continue
            context = torch.autocast("cuda", dtype=dtype) if dtype is not None else nullcontext()
            with context:
                prediction = model(x)
                loss = nn.functional.mse_loss(prediction, y)
            dtypes.append(prediction.dtype)
            raw_loss = loss.detach().item()
            assert math.isfinite(raw_loss) and raw_loss >= 0
            before_model = copy.deepcopy(model.state_dict())
            before_optimizer = copy.deepcopy(optimizer.state_dict())
            if scaler is None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), math.inf, error_if_nonfinite=True)
            else:
                scaler.scale(loss).backward()
            if window in overflow:
                assert scaler is not None
                next(model.parameters()).grad.fill_(math.inf)
            old_count = count[0]
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            if count[0] == old_count:
                skipped.append(window)
                _assert_equal(model.state_dict(), before_model)
                _assert_equal(optimizer.state_dict(), before_optimizer)
            else:
                assert count[0] == old_count + 1
                completed.append(window)
                observations.append(raw_loss)
        if scheduler is not None and len(observations) >= 2:
            scheduler.step(observations)
    finally:
        handle.remove()
    assert all(torch.isfinite(parameter).all().item() for parameter in model.parameters())
    assert all(MIN_LR <= float(group["lr"]) <= MAX_LR for group in optimizer.param_groups)
    return {"observations": observations, "completed": completed, "skipped": skipped,
            "dtypes": dtypes}


def _assert_rate_identity(optimizer, rates):
    for group, original in zip(optimizer.param_groups, rates):
        if isinstance(original, torch.Tensor):
            assert group["lr"] is original
            assert original.device == DEVICE and original.ndim == 0
            assert original.dtype == torch.float32
        else:
            assert isinstance(group["lr"], float)


def _reference_training(optimizer_name, tensor_lr, dtype=None):
    actual = _setup(optimizer_name, tensor_lr=tensor_lr)
    reference = _setup(optimizer_name)
    reference[0].load_state_dict(actual[0].state_dict())
    rates = [group["lr"] for group in actual[1].param_groups]
    for group, rate in zip(reference[1].param_groups, rates):
        group["lr"] = float(rate)
    initial = copy.deepcopy(actual[0].state_dict())
    for _ in range(3):
        observed = _epoch(*actual, dtype=dtype)
        expected = _epoch(reference[0], reference[1], None, dtype=dtype)
        assert observed["completed"] == expected["completed"] == [0, 1, 2, 3]
        assert observed["skipped"] == expected["skipped"] == []
        assert set(observed["dtypes"]) == {dtype or torch.float32}
        assert observed["observations"] == pytest.approx(expected["observations"], rel=3e-5, abs=1e-7)
        assert actual[2].batch_losses == observed["observations"]
        for group in reference[1].param_groups:
            updated = _paper_rate(group["lr"], expected["observations"])
            group["lr"] = float(torch.tensor(updated, dtype=torch.float32)) if tensor_lr else updated
        for left, right in zip(actual[1].param_groups, reference[1].param_groups):
            assert float(left["lr"]) == pytest.approx(right["lr"], rel=5e-6, abs=1e-9)
        for left, right in zip(actual[0].parameters(), reference[0].parameters()):
            torch.testing.assert_close(left, right, rtol=3e-5, atol=2e-7)
        _assert_rate_identity(actual[1], rates)
    assert any(not torch.equal(initial[key], value) for key, value in actual[0].state_dict().items())
    return actual, rates


@pytest.mark.parametrize("optimizer_name", ["SGD", "AdamW"])
@pytest.mark.parametrize("tensor_lr", [False, True], ids=["float_lr", "cuda_tensor_lr"])
def test_cuda_eager_training_matches_independent_rate_reference(optimizer_name, tensor_lr):
    (model, optimizer, scheduler), rates = _reference_training(optimizer_name, tensor_lr)
    # Exercise both clamp endpoints on the real rate storage. Zero-gradient
    # optimizer calls here are separate from the real training comparison above.
    for losses, limit in (([1.0, 0.0], MAX_LR), ([0.0, 1.0], MIN_LR)):
        for _ in range(64):
            optimizer.zero_grad(set_to_none=True)
            optimizer.step()
            scheduler.step(losses)
            _assert_rate_identity(optimizer, rates)
            assert all(MIN_LR <= float(group["lr"]) <= MAX_LR for group in optimizer.param_groups)
            if all(float(group["lr"]) == limit for group in optimizer.param_groups):
                break
        assert all(float(group["lr"]) == limit for group in optimizer.param_groups)
    assert all(torch.isfinite(parameter).all().item() for parameter in model.parameters())


def _resume(optimizer_name, tensor_lr, *, mixed=False):
    model, optimizer, scheduler = _setup(optimizer_name, tensor_lr=tensor_lr, dropout=True)
    scaler = _scaler() if mixed else None
    dtype = torch.float16 if mixed else None
    shuffle = torch.Generator().manual_seed(901)
    for epoch in range(2):
        _epoch(model, optimizer, scheduler, dtype=dtype, scaler=scaler, shuffle=shuffle,
               overflow=(1,) if mixed and epoch == 0 else ())
    checkpoint = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict() if scaler else None,
                "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(DEVICE),
                "shuffle_rng": shuffle.get_state()}, checkpoint)
    saved_cuda_rng = torch.cuda.get_rng_state(DEVICE).clone()
    expected = [_epoch(model, optimizer, scheduler, dtype=dtype, scaler=scaler, shuffle=shuffle)
                for _ in range(2)]
    assert not torch.equal(torch.cuda.get_rng_state(DEVICE), saved_cuda_rng), "Dropout must consume CUDA RNG"
    expected_cpu_rng = torch.get_rng_state().clone()
    expected_cuda_rng = torch.cuda.get_rng_state(DEVICE).clone()
    checkpoint.seek(0)
    saved = torch.load(checkpoint, weights_only=True)
    resumed_model, resumed_optimizer, resumed_scheduler = _setup(
        optimizer_name, tensor_lr=tensor_lr, dropout=True
    )
    resumed_scaler = _scaler() if mixed else None
    resumed_model.load_state_dict(saved["model"])
    resumed_optimizer.load_state_dict(saved["optimizer"])
    rates = [group["lr"] for group in resumed_optimizer.param_groups]
    resumed_scheduler.load_state_dict(saved["scheduler"])
    _assert_rate_identity(resumed_optimizer, rates)
    if resumed_scaler is not None:
        resumed_scaler.load_state_dict(saved["scaler"])
    resumed_shuffle = torch.Generator()
    resumed_shuffle.set_state(saved["shuffle_rng"])
    # Construction consumes RNG; restore all streams only after loading state.
    torch.set_rng_state(saved["cpu_rng"])
    torch.cuda.set_rng_state(saved["cuda_rng"], DEVICE)
    actual = [_epoch(resumed_model, resumed_optimizer, resumed_scheduler, dtype=dtype,
                     scaler=resumed_scaler, shuffle=resumed_shuffle) for _ in range(2)]
    _assert_equal(actual, expected)
    _assert_equal(resumed_model.state_dict(), model.state_dict())
    _assert_equal(resumed_optimizer.state_dict(), optimizer.state_dict())
    _assert_equal(resumed_scheduler.state_dict(), scheduler.state_dict())
    _assert_equal(resumed_shuffle.get_state(), shuffle.get_state())
    _assert_equal(torch.get_rng_state(), expected_cpu_rng)
    _assert_equal(torch.cuda.get_rng_state(DEVICE), expected_cuda_rng)
    _assert_rate_identity(resumed_optimizer, rates)
    if scaler is not None:
        _assert_equal(resumed_scaler.state_dict(), scaler.state_dict())


@pytest.mark.parametrize("optimizer_name", ["SGD", "AdamW"])
@pytest.mark.parametrize("tensor_lr", [False, True], ids=["float_lr", "cuda_tensor_lr"])
def test_cuda_float32_stochastic_checkpoint_restores_all_state(optimizer_name, tensor_lr):
    _resume(optimizer_name, tensor_lr)


@pytest.mark.parametrize("optimizer_name", ["SGD", "AdamW"])
@pytest.mark.parametrize("overflow", [(1,), (0, 1, 2, 3)], ids=["one_skipped", "all_skipped"])
def test_cuda_fp16_scaler_overflow_matches_explicitly_omitted_update(optimizer_name, overflow):
    actual = _setup(optimizer_name)
    reference = _setup(optimizer_name)
    reference[0].load_state_dict(actual[0].state_dict())
    scaler, reference_scaler = _scaler(), _scaler()
    initial_scheduler = copy.deepcopy(actual[2].state_dict())
    observed = _epoch(*actual, dtype=torch.float16, scaler=scaler, overflow=overflow)
    expected = _epoch(*reference, dtype=torch.float16, scaler=reference_scaler, omit=overflow)
    assert observed["completed"] == expected["completed"] == [i for i in range(4) if i not in overflow]
    assert observed["skipped"] == expected["skipped"] == list(overflow)
    assert set(observed["dtypes"]) == {torch.float16}
    assert observed["observations"] == expected["observations"]
    if observed["observations"]:
        assert actual[2].batch_losses == observed["observations"]
    else:
        _assert_equal(actual[2].state_dict(), initial_scheduler)
    assert scaler.get_scale() == reference_scaler.get_scale() == (4.0 if len(overflow) == 1 else 0.5)
    _assert_equal(actual[0].state_dict(), reference[0].state_dict())
    _assert_equal(actual[1].state_dict(), reference[1].state_dict())
    _assert_equal(actual[2].state_dict(), reference[2].state_dict())


@pytest.mark.parametrize("optimizer_name", ["SGD", "AdamW"])
def test_cuda_fp16_stochastic_checkpoint_restores_scaler_and_rng(optimizer_name):
    _resume(optimizer_name, tensor_lr=True, mixed=True)


@pytest.mark.parametrize("optimizer_name", ["SGD", "AdamW"])
def test_cuda_bfloat16_eager_reference(optimizer_name):
    try:
        native_bfloat16 = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:  # Older torch releases have no emulation keyword.
        native_bfloat16 = torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability(DEVICE)[0] >= 8
    if not native_bfloat16:
        pytest.skip("CUDA device lacks native bfloat16; mandatory CUDA cases remain required")
    _reference_training(optimizer_name, tensor_lr=False, dtype=torch.bfloat16)
