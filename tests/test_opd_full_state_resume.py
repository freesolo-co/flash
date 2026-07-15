"""Pure-python coverage for opd full-state checkpoint save/restore control flow.

These tests deliberately import NO gpu/training packages (torch, peft, numpy, cuda). The save/restore
helpers take torch as an injected argument and import peft lazily, so a fake torch object and a fake
``peft`` module in sys.modules exercise the exact control flow (ordering, weights_only, in-place adapter
restore, fail-closed validation) offline. The real torch/peft/cuda round trip is proved on gpu.
"""

from __future__ import annotations

import json
import os
import random
import sys
import types
from pathlib import Path

import pytest

from flash.engine.worker import opd
from flash.opd_retry_contract import opd_resume_checkpoint_complete

_COMPLETE = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "rng_state.pth",
    "opd_state.json",
)


@pytest.fixture(autouse=True)
def _fake_numpy(monkeypatch):
    state = {"value": ("numpy-rng",)}
    module = types.ModuleType("numpy")
    module.random = types.SimpleNamespace(
        get_state=lambda: state["value"],
        set_state=lambda value: state.__setitem__("value", value),
    )
    monkeypatch.setitem(sys.modules, "numpy", module)


# --- fakes ---------------------------------------------------------------------------------------


class _FakeParam:
    def __init__(self, requires_grad, device):
        self.requires_grad = requires_grad
        self.device = device


class _FakeModel:
    """Records save_pretrained; parameters() yields a frozen param then a trainable one on ``device``."""

    def __init__(self, device="cuda:0"):
        self.device = device
        self.saved_to: list[str] = []

    def parameters(self):
        # a non-trainable base param first, so restore's "first requires_grad param" selection is tested.
        return iter([_FakeParam(False, "cpu"), _FakeParam(True, self.device)])

    def save_pretrained(self, out_dir):
        self.saved_to.append(out_dir)
        Path(out_dir, "adapter_config.json").write_text("{}")
        Path(out_dir, "adapter_model.safetensors").write_bytes(b"\x00")


class _FakeTok:
    def save_pretrained(self, out_dir):
        Path(out_dir, "tokenizer.json").write_text("{}")


class _FakeOptimizer:
    def __init__(self, order):
        self._order = order
        self._state = {"state": {"0": {"exp_avg": "m"}}, "param_groups": [{"lr": 1e-4}]}
        self.loaded = None

    def state_dict(self):
        return self._state

    def load_state_dict(self, state):
        self.loaded = state
        self._order.append("optimizer")


class _FakeCuda:
    def __init__(self, available=False):
        self._available = available

    def is_available(self):
        return self._available

    def get_rng_state_all(self):
        return ["cuda-rng"]

    def set_rng_state_all(self, state):
        pass


class _FakeTorch:
    """Fake torch: save() persists a file AND memoizes the object; load() returns the memoized object."""

    def __init__(self, *, cuda_available=False):
        self.cuda = _FakeCuda(cuda_available)
        self._payload: dict[str, object] = {}
        self.save_calls: list[str] = []
        self.load_calls: list[tuple[str, object]] = []
        self._rng_state = "torch-rng"

    def save(self, obj, path):
        base = os.path.basename(path)
        self.save_calls.append(base)
        self._payload[base] = obj
        with open(path, "wb") as f:
            f.write(b"\x00")

    def load(self, path, map_location=None, weights_only=None):
        base = os.path.basename(path)
        self.load_calls.append((base, weights_only))
        return self._payload[base]

    def get_rng_state(self):
        return self._rng_state

    def set_rng_state(self, state):
        self._rng_state = state


def _install_fake_peft(
    monkeypatch, order, *, set_raises=None, missing_keys=None, unexpected_keys=None
):
    """Install a fake ``peft`` module; record adapter-restore ordering and target model identity."""
    seen = {"set_model_id": None, "load_dir": None, "load_device": None}
    mod = types.ModuleType("peft")

    def _load_peft_weights(ckpt_dir, device=None):
        seen["load_dir"] = ckpt_dir
        seen["load_device"] = device
        return {"__peft_state__": ckpt_dir}

    def _set_peft_model_state_dict(model, state_dict):
        if set_raises is not None:
            raise set_raises
        seen["set_model_id"] = id(model)
        order.append("adapter")
        return types.SimpleNamespace(
            missing_keys=list(missing_keys or []),
            unexpected_keys=list(unexpected_keys or []),
        )

    mod.load_peft_weights = _load_peft_weights
    mod.set_peft_model_state_dict = _set_peft_model_state_dict
    monkeypatch.setitem(sys.modules, "peft", mod)
    return seen


def _accounting(opt_steps):
    """A representative accounting blob whose curve lengths satisfy the honest-metrics invariant."""
    return {
        "loss_curve": [0.5 / (i + 1) for i in range(opt_steps)],
        "coverage_curve": [1.0] * opt_steps,
        "generated_tokens": 123,
        "teacher_input_tokens": 45,
        "truncated_rollouts": 1,
        "granularity_sum": 2.5,
        "granularity_n": 4,
        "samples_seen": 9,
        "teacher_ok": 8,
        "teacher_transient": 1,
        "teacher_error": 0,
        "skip_counts": {"length_capped": 2},
        "no_signal_resamples": 3,
        "no_signal_skipped_steps": 1,
        "episodes_seen": 6,
        "mt_turn_records": 0,
        "opd_phase_seconds": {"rollout": 1.25},
        "opd_phase_counts": {"rollout": 5},
    }


# --- completeness predicate ----------------------------------------------------------------------


def test_complete_checkpoint_is_recognized():
    assert opd_resume_checkpoint_complete(_COMPLETE) is True
    # extra files (tokenizer, etc.) do not break completeness.
    assert opd_resume_checkpoint_complete((*_COMPLETE, "tokenizer.json", "special_tokens_map.json"))
    # either adapter weight format satisfies the adapter requirement.
    assert opd_resume_checkpoint_complete(
        ("adapter_config.json", "adapter_model.bin", "optimizer.pt", "rng_state.pth", "opd_state.json")
    )


@pytest.mark.parametrize(
    "missing", ["adapter_config.json", "optimizer.pt", "rng_state.pth", "opd_state.json"]
)
def test_missing_required_state_file_is_incomplete(missing):
    assert opd_resume_checkpoint_complete([f for f in _COMPLETE if f != missing]) is False


def test_missing_adapter_weight_is_incomplete():
    assert (
        opd_resume_checkpoint_complete(
            ("adapter_config.json", "optimizer.pt", "rng_state.pth", "opd_state.json")
        )
        is False
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [(5, 5), (0, 0), (-1, None), (True, None), (False, None), ("3", None), (3.0, None), (None, None)],
)
def test_nonneg_int_or_none(value, expected):
    assert opd._nonneg_int_or_none(value) == expected


# --- save --------------------------------------------------------------------------------------


def test_save_writes_complete_checkpoint_and_uploads(monkeypatch, tmp_path):
    uploads: list[tuple[int, str]] = []
    monkeypatch.setattr(
        opd,
        "_w",
        types.SimpleNamespace(
            SEED=42,
            upload_resume_checkpoint=lambda step, d: uploads.append((step, d)) or True,
        ),
    )
    torch = _FakeTorch()
    acct = _accounting(3)
    opd._save_opd_resume_checkpoint(
        model=_FakeModel(),
        tok=_FakeTok(),
        optimizer=_FakeOptimizer([]),
        torch=torch,
        out_dir=str(tmp_path),
        opt_steps=3,
        step=7,
        accounting=acct,
    )

    local_dir = os.path.join(str(tmp_path), "resume_ckpt")
    assert opd_resume_checkpoint_complete(os.listdir(local_dir)) is True
    assert set(torch.save_calls) == {"optimizer.pt", "rng_state.pth"}

    with open(os.path.join(local_dir, "opd_state.json")) as f:
        state = json.load(f)
    assert state["contract_version"] == opd.OPD_RESUME_STATE_VERSION
    assert state["seed"] == 42
    assert state["opt_steps"] == 3
    assert state["step"] == 7
    # every accounting key round-trips verbatim alongside the version/seed/step envelope.
    for key, value in acct.items():
        assert state[key] == value

    assert uploads == [(3, local_dir)]


def test_save_is_best_effort_on_upload_failure(monkeypatch, tmp_path):
    def _boom(step, d):
        raise TimeoutError("hf upload outage")

    monkeypatch.setattr(
        opd, "_w", types.SimpleNamespace(SEED=42, upload_resume_checkpoint=_boom)
    )
    # a transient periodic upload failure must not propagate; the next save boundary retries.
    opd._save_opd_resume_checkpoint(
        model=_FakeModel(),
        tok=_FakeTok(),
        optimizer=_FakeOptimizer([]),
        torch=_FakeTorch(),
        out_dir=str(tmp_path),
        opt_steps=2,
        step=2,
        accounting=_accounting(2),
    )


def test_periodic_save_is_best_effort_when_upload_returns_false(monkeypatch, tmp_path):
    monkeypatch.setattr(
        opd,
        "_w",
        types.SimpleNamespace(SEED=42, upload_resume_checkpoint=lambda *_args: False),
    )

    opd._save_opd_resume_checkpoint(
        model=_FakeModel(),
        tok=_FakeTok(),
        optimizer=_FakeOptimizer([]),
        torch=_FakeTorch(),
        out_dir=str(tmp_path),
        opt_steps=2,
        step=2,
        accounting=_accounting(2),
    )


def test_required_save_raises_when_upload_returns_false(monkeypatch, tmp_path):
    monkeypatch.setattr(
        opd,
        "_w",
        types.SimpleNamespace(SEED=42, upload_resume_checkpoint=lambda *_args: False),
    )

    with pytest.raises(opd.RetriableInfraError, match="required opd full-state checkpoint upload failed"):
        opd._save_opd_resume_checkpoint(
            model=_FakeModel(),
            tok=_FakeTok(),
            optimizer=_FakeOptimizer([]),
            torch=_FakeTorch(),
            out_dir=str(tmp_path),
            opt_steps=2,
            step=2,
            accounting=_accounting(2),
            required=True,
        )


def test_required_save_raises_when_local_construction_fails(monkeypatch, tmp_path):
    uploads = []
    monkeypatch.setattr(
        opd,
        "_w",
        types.SimpleNamespace(
            SEED=42,
            upload_resume_checkpoint=lambda *_args: uploads.append(True) or True,
        ),
    )

    class FailingModel(_FakeModel):
        def save_pretrained(self, out_dir):
            raise OSError("local disk unavailable")

    with pytest.raises(
        opd.RetriableInfraError,
        match="required opd full-state checkpoint construction failed",
    ):
        opd._save_opd_resume_checkpoint(
            model=FailingModel(),
            tok=_FakeTok(),
            optimizer=_FakeOptimizer([]),
            torch=_FakeTorch(),
            out_dir=str(tmp_path),
            opt_steps=2,
            step=2,
            accounting=_accounting(2),
            required=True,
        )

    assert uploads == []


def test_save_reuses_local_dir_across_calls(monkeypatch, tmp_path):
    # the local staging dir is reused (rewritten) each save so on-disk state stays bounded to one copy;
    # a stale file from a prior save must not survive into the next checkpoint.
    monkeypatch.setattr(
        opd, "_w", types.SimpleNamespace(SEED=42, upload_resume_checkpoint=lambda *_a: True)
    )
    local_dir = os.path.join(str(tmp_path), "resume_ckpt")
    os.makedirs(local_dir, exist_ok=True)
    Path(local_dir, "stale-shard.bin").write_bytes(b"stale")
    opd._save_opd_resume_checkpoint(
        model=_FakeModel(),
        tok=_FakeTok(),
        optimizer=_FakeOptimizer([]),
        torch=_FakeTorch(),
        out_dir=str(tmp_path),
        opt_steps=1,
        step=1,
        accounting=_accounting(1),
    )
    assert "stale-shard.bin" not in os.listdir(local_dir)


# --- restore round trip ------------------------------------------------------------------------


def test_save_then_restore_round_trips_state_in_place(monkeypatch, tmp_path):
    monkeypatch.setattr(
        opd, "_w", types.SimpleNamespace(SEED=42, upload_resume_checkpoint=lambda *_a: True)
    )
    torch = _FakeTorch()
    save_optimizer = _FakeOptimizer([])
    acct = _accounting(3)
    # pin the global python rng before save; nothing between here and save's random.getstate() consumes
    # it, so the persisted blob equals exactly this state.
    random.seed(20260715)
    saved_rng = random.getstate()
    opd._save_opd_resume_checkpoint(
        model=_FakeModel(),
        tok=_FakeTok(),
        optimizer=save_optimizer,
        torch=torch,
        out_dir=str(tmp_path),
        opt_steps=3,
        step=7,
        accounting=acct,
    )
    saved_dir = tmp_path / "resume_ckpt"
    ckpt_dir = tmp_path / "checkpoint-3"
    saved_dir.rename(ckpt_dir)
    ckpt_dir = str(ckpt_dir)

    order: list[str] = []
    peft_seen = _install_fake_peft(monkeypatch, order)
    model = _FakeModel(device="cuda:1")
    optimizer = _FakeOptimizer(order)
    # perturb the process rng so restore has something distinct to overwrite.
    random.seed(1)
    assert random.getstate() != saved_rng

    restored = opd._restore_opd_full_state(
        ckpt_dir, model=model, optimizer=optimizer, torch=torch
    )

    assert restored is not None
    assert restored["opt_steps"] == 3
    assert restored["step"] == 7
    for key, value in acct.items():
        assert restored[key] == value

    # adapter weights are restored in place into the passed model, before the optimizer state loads,
    # so the optimizer's parameter references stay valid.
    assert order == ["adapter", "optimizer"]
    assert peft_seen["set_model_id"] == id(model)
    assert peft_seen["load_dir"] == ckpt_dir
    assert peft_seen["load_device"] == "cuda:1"  # the first requires_grad param's device
    assert optimizer.loaded == save_optimizer.state_dict()

    # both trusted-artifact loads pass weights_only=false (torch 2.6+ default would reject the rng blob).
    assert torch.load_calls == [("optimizer.pt", False), ("rng_state.pth", False)]
    # the persisted global python rng state was reapplied verbatim (best-effort restore succeeded).
    assert random.getstate() == saved_rng


# --- restore fail-closed matrix ----------------------------------------------------------------


def _write_state_dir(root, *, state, filenames=_COMPLETE):
    os.makedirs(root, exist_ok=True)
    for name in filenames:
        if name == "opd_state.json":
            Path(root, name).write_text(state if isinstance(state, str) else json.dumps(state))
        else:
            Path(root, name).write_bytes(b"\x00")
    return str(root)


def _valid_state(**overrides):
    state = {
        "contract_version": opd.OPD_RESUME_STATE_VERSION,
        "seed": 42,
        "opt_steps": 3,
        "step": 7,
        **_accounting(3),
    }
    state.update(overrides)
    return state


def _restore_expecting_none(monkeypatch, ckpt_dir, *, torch=None, max_opt_steps=None):
    """Run restore and assert it returns None without touching model or optimizer state."""
    monkeypatch.setattr(opd, "_w", types.SimpleNamespace(SEED=42))
    order: list[str] = []
    _install_fake_peft(monkeypatch, order)
    torch = torch or _FakeTorch()
    optimizer = _FakeOptimizer(order)
    result = opd._restore_opd_full_state(
        ckpt_dir,
        model=_FakeModel(),
        optimizer=optimizer,
        torch=torch,
        max_opt_steps=max_opt_steps,
    )
    return result, order, optimizer, torch


def _assert_restore_untouched(order, optimizer, torch):
    assert order == []
    assert optimizer.loaded is None
    assert torch.load_calls == []


def test_restore_rejects_incomplete_dir(monkeypatch, tmp_path):
    ckpt = _write_state_dir(
        tmp_path / "checkpoint-3", state=_valid_state(), filenames=[f for f in _COMPLETE if f != "optimizer.pt"]
    )
    result, order, optimizer, torch = _restore_expecting_none(monkeypatch, ckpt)
    assert result is None
    # fail-closed before any model/optimizer mutation.
    _assert_restore_untouched(order, optimizer, torch)


@pytest.mark.parametrize(
    "overrides",
    [
        {"contract_version": 2},
        {"seed": 99},
        {"opt_steps": -1},
        {"opt_steps": True},
        {"loss_curve": [0.1, 0.2]},  # length != opt_steps (3)
        {"coverage_curve": [1.0]},  # length != opt_steps (3)
        {"loss_curve": "not-a-list"},
        {"step": -5},
    ],
)
def test_restore_rejects_bad_state(monkeypatch, tmp_path, overrides):
    ckpt = _write_state_dir(tmp_path / "checkpoint-3", state=_valid_state(**overrides))
    result, order, optimizer, torch = _restore_expecting_none(monkeypatch, ckpt)
    assert result is None
    _assert_restore_untouched(order, optimizer, torch)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("loss_curve", [0.5, float("nan"), 0.1]),
        ("coverage_curve", [1.0, float("inf"), 1.0]),
    ],
)
def test_restore_rejects_nonfinite_curve_values(monkeypatch, tmp_path, field, value):
    ckpt = _write_state_dir(tmp_path / "checkpoint-3", state=_valid_state(**{field: value}))
    result, order, optimizer, torch = _restore_expecting_none(monkeypatch, ckpt)
    assert result is None
    _assert_restore_untouched(order, optimizer, torch)


def test_restore_rejects_checkpoint_step_mismatch(monkeypatch, tmp_path):
    ckpt = _write_state_dir(tmp_path / "checkpoint-4", state=_valid_state())
    result, order, optimizer, torch = _restore_expecting_none(monkeypatch, ckpt)
    assert result is None
    _assert_restore_untouched(order, optimizer, torch)


def test_restore_rejects_checkpoint_zero(monkeypatch, tmp_path):
    ckpt = _write_state_dir(
        tmp_path / "checkpoint-0",
        state=_valid_state(opt_steps=0, step=0, loss_curve=[], coverage_curve=[]),
    )
    result, order, optimizer, torch = _restore_expecting_none(monkeypatch, ckpt)
    assert result is None
    _assert_restore_untouched(order, optimizer, torch)


def test_restore_rejects_noncanonical_checkpoint_suffix(monkeypatch, tmp_path):
    ckpt = _write_state_dir(tmp_path / "checkpoint-03", state=_valid_state())
    result, order, optimizer, torch = _restore_expecting_none(monkeypatch, ckpt)
    assert result is None
    _assert_restore_untouched(order, optimizer, torch)


def test_restore_rejects_opt_steps_above_max(monkeypatch, tmp_path):
    ckpt = _write_state_dir(tmp_path / "checkpoint-3", state=_valid_state())
    result, order, optimizer, torch = _restore_expecting_none(
        monkeypatch, ckpt, max_opt_steps=2
    )
    assert result is None
    _assert_restore_untouched(order, optimizer, torch)


def test_restore_rejects_step_before_opt_steps(monkeypatch, tmp_path):
    ckpt = _write_state_dir(tmp_path / "checkpoint-3", state=_valid_state(step=2))
    result, order, optimizer, torch = _restore_expecting_none(monkeypatch, ckpt)
    assert result is None
    _assert_restore_untouched(order, optimizer, torch)


@pytest.mark.parametrize("missing_key", list(_accounting(3)))
def test_restore_rejects_missing_accounting_key(monkeypatch, tmp_path, missing_key):
    state = _valid_state()
    state.pop(missing_key)
    ckpt = _write_state_dir(tmp_path / "checkpoint-3", state=state)
    result, order, optimizer, torch = _restore_expecting_none(monkeypatch, ckpt)
    assert result is None
    _assert_restore_untouched(order, optimizer, torch)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("teacher_ok", "8"),
        ("skip_counts", []),
        ("granularity_sum", True),
        ("granularity_sum", float("inf")),
        ("granularity_sum", -0.1),
    ],
)
def test_restore_rejects_wrong_accounting_type(monkeypatch, tmp_path, field, value):
    ckpt = _write_state_dir(tmp_path / "checkpoint-3", state=_valid_state(**{field: value}))
    result, order, optimizer, torch = _restore_expecting_none(monkeypatch, ckpt)
    assert result is None
    _assert_restore_untouched(order, optimizer, torch)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("skip_counts", {"length": -1}),
        ("skip_counts", {"length": True}),
        ("opd_phase_seconds", {"teacher": float("nan")}),
        ("opd_phase_seconds", {"teacher": -0.1}),
        ("opd_phase_counts", {"teacher": False}),
        ("opd_phase_counts", {"teacher": -1}),
    ],
)
def test_restore_rejects_malformed_nested_accounting(monkeypatch, tmp_path, field, value):
    ckpt = _write_state_dir(tmp_path / "checkpoint-3", state=_valid_state(**{field: value}))
    result, order, optimizer, torch = _restore_expecting_none(monkeypatch, ckpt)
    assert result is None
    _assert_restore_untouched(order, optimizer, torch)


@pytest.mark.parametrize("field", ["skip_counts", "opd_phase_seconds", "opd_phase_counts"])
def test_restore_rejects_non_string_nested_keys(monkeypatch, tmp_path, field):
    state = _valid_state()
    state[field] = {1: 1}
    ckpt = _write_state_dir(tmp_path / "checkpoint-3", state=_valid_state())
    monkeypatch.setattr(opd.json, "load", lambda _file: state)
    result, order, optimizer, torch = _restore_expecting_none(monkeypatch, ckpt)
    assert result is None
    _assert_restore_untouched(order, optimizer, torch)


def test_restore_rejects_negative_accounting_counter(monkeypatch, tmp_path):
    ckpt = _write_state_dir(
        tmp_path / "checkpoint-3", state=_valid_state(generated_tokens=-1)
    )
    result, order, optimizer, torch = _restore_expecting_none(monkeypatch, ckpt)
    assert result is None
    _assert_restore_untouched(order, optimizer, torch)


def test_restore_rejects_non_object_state(monkeypatch, tmp_path):
    ckpt = _write_state_dir(tmp_path / "checkpoint-3", state=json.dumps([1, 2, 3]))
    result, *_ = _restore_expecting_none(monkeypatch, ckpt)
    assert result is None


def test_restore_rejects_corrupt_json(monkeypatch, tmp_path):
    ckpt = _write_state_dir(tmp_path / "checkpoint-3", state="{not valid json")
    result, *_ = _restore_expecting_none(monkeypatch, ckpt)
    assert result is None


def test_restore_returns_none_when_torch_load_fails(monkeypatch, tmp_path):
    ckpt = _write_state_dir(tmp_path / "checkpoint-3", state=_valid_state())

    class _RaisingTorch(_FakeTorch):
        def load(self, path, map_location=None, weights_only=None):
            raise RuntimeError("truncated optimizer shard")

    result, _order, optimizer, _torch = _restore_expecting_none(
        monkeypatch, ckpt, torch=_RaisingTorch()
    )
    assert result is None
    assert optimizer.loaded is None


def test_restore_returns_none_when_adapter_load_fails(monkeypatch, tmp_path):
    ckpt = _write_state_dir(tmp_path / "checkpoint-3", state=_valid_state())
    monkeypatch.setattr(opd, "_w", types.SimpleNamespace(SEED=42))
    order: list[str] = []
    _install_fake_peft(monkeypatch, order, set_raises=RuntimeError("adapter shape mismatch"))
    optimizer = _FakeOptimizer(order)
    result = opd._restore_opd_full_state(
        ckpt, model=_FakeModel(), optimizer=optimizer, torch=_FakeTorch()
    )
    assert result is None
    # adapter load raised before the optimizer state was touched.
    assert optimizer.loaded is None


@pytest.mark.parametrize(
    ("missing_keys", "unexpected_keys"),
    [(["base_model.model.lora_A.weight"], []), ([], ["unexpected.lora_B.weight"])],
)
def test_restore_rejects_incompatible_adapter_keys(
    monkeypatch, tmp_path, missing_keys, unexpected_keys
):
    ckpt = _write_state_dir(tmp_path / "checkpoint-3", state=_valid_state())
    monkeypatch.setattr(opd, "_w", types.SimpleNamespace(SEED=42))
    order: list[str] = []
    _install_fake_peft(
        monkeypatch,
        order,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
    )
    torch = _FakeTorch()
    optimizer = _FakeOptimizer(order)

    result = opd._restore_opd_full_state(
        ckpt, model=_FakeModel(), optimizer=optimizer, torch=torch
    )

    assert result is None
    assert order == ["adapter"]
    assert optimizer.loaded is None
    assert torch.load_calls == []


def test_restore_allows_missing_base_model_keys(monkeypatch, tmp_path):
    ckpt = _write_state_dir(tmp_path / "checkpoint-3", state=_valid_state())
    monkeypatch.setattr(opd, "_w", types.SimpleNamespace(SEED=42))
    order: list[str] = []
    _install_fake_peft(
        monkeypatch,
        order,
        missing_keys=[
            "base_model.model.layers.0.self_attn.q_proj.weight",
            "base_model.model.layers.0.mlp.down_proj.weight",
        ],
    )
    torch = _FakeTorch()
    torch._payload["optimizer.pt"] = {"state": {}, "param_groups": []}
    torch._payload["rng_state.pth"] = {
        "python": random.getstate(),
        "numpy": ("numpy-rng",),
        "torch": "torch-rng",
        "cuda": None,
    }
    optimizer = _FakeOptimizer(order)

    result = opd._restore_opd_full_state(
        ckpt, model=_FakeModel(), optimizer=optimizer, torch=torch
    )

    assert result is not None
    assert order == ["adapter", "optimizer"]
    assert optimizer.loaded == {"state": {}, "param_groups": []}


@pytest.mark.parametrize(
    "rng_state",
    [
        ["not", "a", "dict"],
        {},
        {"python": None, "numpy": None, "torch": None, "cuda": None, "extra": None},
    ],
)
def test_restore_rejects_malformed_rng_blob(monkeypatch, tmp_path, rng_state):
    ckpt = _write_state_dir(tmp_path / "checkpoint-3", state=_valid_state())
    monkeypatch.setattr(opd, "_w", types.SimpleNamespace(SEED=42))
    order: list[str] = []
    _install_fake_peft(monkeypatch, order)

    torch = _FakeTorch()
    torch._payload["optimizer.pt"] = {"state": {}, "param_groups": []}
    torch._payload["rng_state.pth"] = rng_state
    result = opd._restore_opd_full_state(
        ckpt, model=_FakeModel(), optimizer=_FakeOptimizer(order), torch=torch
    )
    assert result is None
