"""Small hardware-backed training-mode checks, not scale/performance certification.

The explicit caller policy records one sample-weighted, unscaled loss for each
completed optimizer update. Skipped GradScaler updates add no observation.
Schedulers and optimizers remain eager, including the compiled-model check.
"""

import copy
import io
import json
import math
from contextlib import nullcontext

import pytest
import torch
from torch import nn

from dlrs import DLRSScheduler


@pytest.fixture(scope="module", autouse=True)
def limit_cpu_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(original)


def _cpu_scaler():
    if not hasattr(torch, "amp") or not hasattr(torch.amp, "GradScaler"):
        pytest.skip("This torch version does not expose torch.amp.GradScaler")
    try:
        scaler = torch.amp.GradScaler("cpu", init_scale=8.0, growth_interval=100)
    except TypeError:
        pytest.skip("This torch version does not support CPU GradScaler")
    assert scaler.is_enabled(), "CPU GradScaler must execute, not silently disable"
    return scaler


def _setup(optimizer_name, device="cpu", dtype=torch.float32, dropout=False, tensor_lr=False):
    model = nn.Sequential(
        nn.Linear(3, 8), nn.Tanh(), nn.Dropout(0.25 if dropout else 0.0), nn.Linear(8, 1)
    ).to(device=device, dtype=dtype)
    rate = torch.tensor(0.01, device=device, dtype=dtype) if tensor_lr else 0.01
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=rate, momentum=0.9, foreach=False)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=rate, foreach=False)
    scheduler = DLRSScheduler(optimizer, min_lr=1e-6, max_lr=0.05)
    return model, optimizer, scheduler


def _data(device="cpu", dtype=torch.float32):
    generator = torch.Generator().manual_seed(411)
    inputs = torch.randn(23, 3, generator=generator, dtype=dtype)
    targets = inputs @ torch.tensor([[0.4], [-0.8], [0.3]], dtype=dtype) + 0.2
    return inputs.to(device), targets.to(device)


def _epoch(model, optimizer, scheduler, *, device="cpu", dtype=torch.float32,
           scaler=None, autocast=False, overflow_windows=(), shuffle_generator=None,
           accumulated=True):
    inputs, targets = _data(device, dtype)
    if shuffle_generator is not None:
        indices = torch.randperm(len(inputs), generator=shuffle_generator).to(device)
        inputs, targets = inputs[indices], targets[indices]
    model.train()
    observations, skipped, dtypes = [], [], []
    actual_steps = [0]

    def completed_step(*_):
        actual_steps[0] += 1

    # A direct eager step has no scaler skip decision. Count its completed call
    # explicitly: Torch 2.0.0 model compilation disables native optimizer hooks.
    # Scaled updates still need the real hook to distinguish a skipped update.
    handle = optimizer.register_step_post_hook(completed_step) if scaler is not None else None
    try:
        # Four optimizer updates: 6, 6, 6, and 5 samples. The last update has
        # unequal microbatch sizes (3 and 2); constant loss/2 would be wrong.
        for window, (x, y) in enumerate(zip(inputs.split(6), targets.split(6))):
            optimizer.zero_grad(set_to_none=True)
            before = [parameter.detach().clone() for parameter in model.parameters()]
            unscaled = []
            parts = zip(x.split(3), y.split(3)) if accumulated else [(x, y)]
            for micro_x, micro_y in parts:
                context = torch.autocast("cpu", dtype=torch.bfloat16) if autocast else nullcontext()
                with context:
                    prediction = model(micro_x)
                    loss = nn.functional.mse_loss(prediction, micro_y)
                dtypes.append(prediction.dtype)
                raw_loss = loss.detach().item()
                assert math.isfinite(raw_loss) and raw_loss >= 0
                weight = len(micro_x) / len(x)
                unscaled.append(raw_loss * weight)
                weighted_loss = loss * weight
                if scaler is None:
                    weighted_loss.backward()
                else:
                    scaler.scale(weighted_loss).backward()
            if window in overflow_windows:
                assert scaler is not None
                next(model.parameters()).grad.fill_(float("inf"))
            old_steps = actual_steps[0]
            if scaler is None:
                optimizer.step()
                completed_step()
            else:
                scaler.step(optimizer)
                scaler.update()
            if actual_steps[0] == old_steps:
                skipped.append(window)
                for parameter, original in zip(model.parameters(), before):
                    torch.testing.assert_close(parameter, original, rtol=0, atol=0)
            else:
                observations.append(math.fsum(unscaled))
        # No scheduler call is made unless at least two updates completed.
        if len(observations) >= 2:
            scheduler.step(observations)
    finally:
        if handle is not None:
            handle.remove()
    rate = float(optimizer.param_groups[0]["lr"])
    assert 1e-6 <= rate <= 0.05
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    return {"observations": observations, "skipped": skipped, "dtypes": dtypes,
            "completed_updates": actual_steps[0], "lr": rate}


@pytest.mark.parametrize("optimizer_name", ["SGD", "AdamW"])
def test_accumulation_matches_full_update_batches_including_partial_window(optimizer_name):
    torch.manual_seed(812)
    accumulated = _setup(optimizer_name, dtype=torch.float64)
    reference = _setup(optimizer_name, dtype=torch.float64)
    reference[0].load_state_dict(accumulated[0].state_dict())
    for _ in range(3):
        actual = _epoch(*accumulated, dtype=torch.float64)
        expected = _epoch(*reference, dtype=torch.float64, accumulated=False)
        assert actual["completed_updates"] == len(actual["observations"]) == 4
        assert len(actual["dtypes"]) == 8  # two microbatches per optimizer update
        assert actual["observations"] == pytest.approx(expected["observations"], rel=1e-12)
        assert accumulated[2].batch_losses == actual["observations"]
        assert actual["lr"] == pytest.approx(expected["lr"], rel=1e-12)
        for actual_parameter, expected_parameter in zip(accumulated[0].parameters(), reference[0].parameters()):
            torch.testing.assert_close(actual_parameter, expected_parameter, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("optimizer_name", ["SGD", "AdamW"])
def test_real_cpu_bfloat16_scaler_skips_overflow_without_a_loss_observation(optimizer_name):
    torch.manual_seed(17)
    model, optimizer, scheduler = _setup(optimizer_name)
    scaler = _cpu_scaler()
    initial_scale = scaler.get_scale()
    result = _epoch(model, optimizer, scheduler, scaler=scaler, autocast=True, overflow_windows=(1,))
    assert set(result["dtypes"]) == {torch.bfloat16}
    assert result["skipped"] == [1]
    assert result["completed_updates"] == len(result["observations"]) == 3
    assert scaler.get_scale() < initial_scale
    assert scheduler.batch_losses == result["observations"]
    assert scheduler.loss_slope == pytest.approx(
        (result["observations"][-1] - result["observations"][0]) /
        (sum(result["observations"]) / 3)
    )
    assert len(optimizer.state) == len(list(model.parameters()))


def _rng_state(device):
    state = {"cpu": torch.get_rng_state()}
    if device == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def _restore_rng(state):
    torch.set_rng_state(state["cpu"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def _assert_nested_equal(actual, expected):
    # Scalar tensor comparisons otherwise coerce to a truth value, allowing
    # checkpoint metadata to silently change from a tensor to a Python number.
    assert type(actual) is type(expected), (type(actual), type(expected))
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_nested_equal(left, right)
    else:
        assert actual == expected


def test_checkpoint_comparison_rejects_tensor_metadata_and_container_type_loss():
    tensor = torch.tensor(0.1, dtype=torch.float64)
    correctly_typed = {"base_lrs": [tensor], "last_epoch": 2}
    _assert_nested_equal(copy.deepcopy(correctly_typed), correctly_typed)
    mismatches = [
        ({"base_lrs": [0.1]}, {"base_lrs": [tensor]}),
        ({"base_lrs": [tensor]}, {"base_lrs": [0.1]}),
        ({"base_lrs": [tensor.float()]}, {"base_lrs": [tensor]}),
        ({"base_lrs": (tensor,)}, {"base_lrs": [tensor]}),
        ({"base_lrs": [tensor]}, {"base_lrs": (tensor,)}),
        ({"last_epoch": 2.0}, {"last_epoch": 2}),
    ]
    for actual, expected in mismatches:
        with pytest.raises(AssertionError):
            _assert_nested_equal(actual, expected)


def _resume_trial(optimizer_name, *, device="cpu", autocast=False, dropout=True, tensor_lr=False):
    torch.manual_seed(631)
    if device == "mps":
        torch.mps.manual_seed(631)
    model, optimizer, scheduler = _setup(optimizer_name, device, dropout=dropout, tensor_lr=tensor_lr)
    scaler = _cpu_scaler() if autocast else None
    shuffle = torch.Generator().manual_seed(901)
    for epoch in range(2):
        _epoch(model, optimizer, scheduler, device=device, scaler=scaler,
               autocast=autocast, shuffle_generator=shuffle,
               overflow_windows=(1,) if autocast and epoch == 0 else ())
    checkpoint = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict() if scaler else None,
                "rng": _rng_state(device), "shuffle": shuffle.get_state()}, checkpoint)
    expected = [_epoch(model, optimizer, scheduler, device=device, scaler=scaler,
                       autocast=autocast, shuffle_generator=shuffle) for _ in range(2)]
    checkpoint.seek(0)
    state = torch.load(checkpoint, weights_only=True)
    resumed_model, resumed_optimizer, resumed_scheduler = _setup(
        optimizer_name, device, dropout=dropout, tensor_lr=tensor_lr
    )
    resumed_scaler = _cpu_scaler() if autocast else None
    resumed_model.load_state_dict(state["model"])
    resumed_optimizer.load_state_dict(state["optimizer"])
    resumed_scheduler.load_state_dict(state["scheduler"])
    if resumed_scaler:
        resumed_scaler.load_state_dict(state["scaler"])
    resumed_shuffle = torch.Generator()
    resumed_shuffle.set_state(state["shuffle"])
    # Restore AFTER model/optimizer construction, which can consume random state.
    _restore_rng(state["rng"])
    actual = [_epoch(resumed_model, resumed_optimizer, resumed_scheduler, device=device,
                     scaler=resumed_scaler, autocast=autocast,
                     shuffle_generator=resumed_shuffle) for _ in range(2)]
    assert actual == expected
    _assert_nested_equal(resumed_model.state_dict(), model.state_dict())
    _assert_nested_equal(resumed_optimizer.state_dict(), optimizer.state_dict())
    _assert_nested_equal(resumed_scheduler.state_dict(), scheduler.state_dict())
    if scaler:
        assert resumed_scaler.state_dict() == scaler.state_dict()


@pytest.mark.parametrize("optimizer_name", ["SGD", "AdamW"])
@pytest.mark.parametrize("autocast", [False, True])
def test_stochastic_cpu_resume_restores_rng_shuffle_optimizer_and_scaler(optimizer_name, autocast):
    _resume_trial(optimizer_name, autocast=autocast)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS hardware unavailable")
@pytest.mark.parametrize("optimizer_name", ["SGD", "AdamW"])
def test_real_mps_float32_checkpoint_resume(optimizer_name):
    _resume_trial(optimizer_name, device="mps", dropout=False)
    torch.mps.synchronize()


def _native_mps_replay_capability():
    """Check upstream RNG replay before constructing any DLRS model or scheduler."""
    import torch.mps

    original_cpu = torch.get_rng_state()
    original_mps = torch.mps.get_rng_state()
    results = {}
    try:
        for operation in ("rand", "dropout"):
            torch.mps.manual_seed(631)
            inputs = torch.ones(256, device="mps")

            def sample(operation=operation, inputs=inputs):
                if operation == "rand":
                    return torch.rand(256, device="mps")
                return nn.functional.dropout(inputs, p=0.25, training=True)

            sample()
            torch.mps.synchronize()
            saved = torch.mps.get_rng_state().clone()
            expected = [sample().cpu() for _ in range(3)]
            advanced = torch.mps.get_rng_state().clone()
            torch.mps.set_rng_state(saved)
            restored = torch.mps.get_rng_state().clone()
            actual = [sample().cpu() for _ in range(3)]
            results[operation] = {
                "state_advanced": not torch.equal(saved, advanced),
                "reported_state_restored": torch.equal(saved, restored),
                "outputs_equal": [torch.equal(left, right) for left, right in zip(actual, expected)],
                "maximum_absolute_difference": max(float((left - right).abs().max())
                                                    for left, right in zip(actual, expected)),
            }
    finally:
        torch.set_rng_state(original_cpu)
        torch.mps.set_rng_state(original_mps)
        torch.mps.synchronize()
    return results


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS hardware unavailable")
@pytest.mark.parametrize("optimizer_name", ["SGD", "AdamW"])
def test_real_mps_stochastic_resume_restores_rng_and_shuffle(optimizer_name, record_testsuite_property):
    capability = _native_mps_replay_capability()
    record_testsuite_property("native_mps_replay_" + optimizer_name,
                             json.dumps({"torch": str(torch.__version__), "operations": capability}))
    failed = [name for name, outcome in capability.items()
              if not outcome["state_advanced"] or not outcome["reported_state_restored"]
              or not all(outcome["outputs_equal"])]
    if failed:
        pytest.skip("Native MPS RNG state replay failed for " + ", ".join(failed)
                    + "; DLRS stochastic continuation not exercised (capability recorded in JUnit)")
    _resume_trial(optimizer_name, device="mps", dropout=True)
    torch.mps.synchronize()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS hardware unavailable")
@pytest.mark.parametrize("optimizer_name", ["SGD", "AdamW"])
def test_real_mps_tensor_lr_preserves_identity_and_bounds(optimizer_name):
    torch.manual_seed(918)
    model, optimizer, scheduler = _setup(optimizer_name, "mps", tensor_lr=True)
    rate = optimizer.param_groups[0]["lr"]
    for _ in range(2):
        _epoch(model, optimizer, scheduler, device="mps")
        assert optimizer.param_groups[0]["lr"] is rate
        assert rate.device.type == "mps"
    saved = copy.deepcopy(scheduler.state_dict())
    _epoch(model, optimizer, scheduler, device="mps")
    scheduler.load_state_dict(saved)
    assert optimizer.param_groups[0]["lr"] is rate
    torch.testing.assert_close(rate, saved["param_group_lrs"][0], rtol=0, atol=0)
    torch.mps.synchronize()


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
@pytest.mark.parametrize("optimizer_name", ["SGD", "AdamW"])
def test_compiled_model_aot_eager_keeps_optimizer_and_scheduler_eager(optimizer_name):
    from torch._dynamo.backends.registry import lookup_backend

    captured_graphs = []
    aot_eager = lookup_backend("aot_eager")

    def backend(graph, example_inputs):
        captured_graphs.append(graph)
        return aot_eager(graph, example_inputs)

    torch.manual_seed(617)
    model, optimizer, scheduler = _setup(optimizer_name)
    initial = copy.deepcopy(model.state_dict())
    compiled = torch.compile(model, backend=backend, fullgraph=True)
    actual = [_epoch(compiled, optimizer, scheduler) for _ in range(2)]
    assert captured_graphs, "The model must actually be captured, not silently run eager"
    reference, reference_optimizer, reference_scheduler = _setup(optimizer_name)
    reference.load_state_dict(initial)
    expected = [_epoch(reference, reference_optimizer, reference_scheduler) for _ in range(2)]
    assert actual == expected
    _assert_nested_equal(model.state_dict(), reference.state_dict())
    _assert_nested_equal(scheduler.state_dict(), reference_scheduler.state_dict())


@pytest.mark.parametrize("optimizer_name", ["SGD", "AdamW"])
def test_cpu_tensor_lr_stochastic_optimizer_checkpoint_resume(optimizer_name):
    _resume_trial(optimizer_name, tensor_lr=True)


@pytest.mark.parametrize("configuration,match", [
    ({"accumulate_grad_batches": 2}, "gradient accumulation"),
    ({"precision": "bf16-mixed"}, "full precision"),
    ({"other_scheduler": True}, "only learning-rate scheduler"),
])
def test_real_lightning_rejects_unsupported_modes_before_optimizer_updates(tmp_path, configuration, match):
    pl = pytest.importorskip("pytorch_lightning")
    from examples.lightning_callback import DLRSLightningCallback

    class Model(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(3, 1)
            self.training_calls = 0

        def training_step(self, batch, batch_idx):
            self.training_calls += 1
            return nn.functional.mse_loss(self.layer(batch[0]), batch[1])

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
            if configuration.get("other_scheduler"):
                return {"optimizer": optimizer,
                        "lr_scheduler": torch.optim.lr_scheduler.StepLR(optimizer, 1)}
            return optimizer

    torch.manual_seed(118)
    model = Model()
    before = copy.deepcopy(model.state_dict())
    callback = DLRSLightningCallback()
    arguments = {key: value for key, value in configuration.items() if key != "other_scheduler"}
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, max_epochs=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        default_root_dir=tmp_path, callbacks=[callback], **arguments,
    )
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*_data()), batch_size=6)
    with pytest.raises(RuntimeError, match=match):
        trainer.fit(model, loader)
    assert model.training_calls == 0
    assert callback.scheduler is None
    _assert_nested_equal(model.state_dict(), before)


@pytest.mark.parametrize("optimizer_name", ["SGD", "AdamW"])
def test_all_cpu_scaler_updates_skipped_leave_scheduler_and_optimizer_unchanged(optimizer_name):
    torch.manual_seed(145)
    model, optimizer, scheduler = _setup(optimizer_name)
    scaler = _cpu_scaler()
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    before_scheduler = copy.deepcopy(scheduler.state_dict())
    result = _epoch(model, optimizer, scheduler, scaler=scaler, autocast=True,
                    overflow_windows=(0, 1, 2, 3))
    assert result["observations"] == [] and result["completed_updates"] == 0
    assert result["skipped"] == [0, 1, 2, 3]
    _assert_nested_equal(optimizer.state_dict(), before_optimizer)
    _assert_nested_equal(scheduler.state_dict(), before_scheduler)
    assert scaler.get_scale() == 0.5
