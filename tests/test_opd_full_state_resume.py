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
from flash.opd_retry_contract import (
    opd_resume_checkpoint_complete,
    validate_opd_resume_state_metadata,
)

_PROMPT_POOL_FINGERPRINT = "a" * 64
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


class _FakeCudaOutOfMemoryError(RuntimeError):
    pass


class _FakeCuda:
    OutOfMemoryError = _FakeCudaOutOfMemoryError

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
        "train_wall_seconds": 12.5,
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


@pytest.mark.parametrize("value", [-1, True, False, "3", 3.0, None])
def test_shared_metadata_validator_rejects_noninteger_opt_steps(value):
    with pytest.raises(ValueError, match="opt_steps"):
        validate_opd_resume_state_metadata(
            _valid_state(opt_steps=value),
            expected_seed=42,
            checkpoint_step=3,
        )


def test_rollout_request_seeds_continue_identically_after_resume():
    run_seed = 20260715
    generation_count = 19
    resume_boundary = 7
    uninterrupted = [
        opd.rollout_request_seed(run_seed, ordinal) for ordinal in range(generation_count)
    ]
    resumed = [
        opd.rollout_request_seed(run_seed, ordinal)
        for ordinal in range(resume_boundary, generation_count)
    ]

    assert resumed == uninterrupted[resume_boundary:]
    assert all(0 <= seed < 2**63 for seed in uninterrupted)
    assert len(set(uninterrupted)) == generation_count


def test_cumulative_train_wall_and_throughput_match_uninterrupted_run():
    generated_tokens = 600
    uninterrupted_wall = opd._cumulative_train_wall_seconds(0.0, 10.0, 25.0)
    resumed_wall = opd._cumulative_train_wall_seconds(9.0, 100.0, 106.0)

    assert resumed_wall == uninterrupted_wall == 15.0
    assert generated_tokens / resumed_wall == generated_tokens / uninterrupted_wall == 40.0


def test_prompt_pool_fingerprint_is_stable_and_order_sensitive():
    examples = [
        ({"id": 1}, [{"role": "user", "content": "private-a"}], [1, 2]),
        ({"id": 2}, [{"role": "user", "content": "private-b"}], [3, 4]),
    ]

    fingerprint = opd._opd_prompt_pool_fingerprint(examples)

    assert fingerprint == opd._opd_prompt_pool_fingerprint(examples)
    assert fingerprint != opd._opd_prompt_pool_fingerprint(list(reversed(examples)))
    assert len(fingerprint) == 64
    assert "private" not in fingerprint


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
        rollout_seed_ordinal=17,
        prompt_pool_fingerprint=_PROMPT_POOL_FINGERPRINT,
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
    assert state["rollout_seed_ordinal"] == 17
    assert state["prompt_pool_fingerprint"] == _PROMPT_POOL_FINGERPRINT
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
        rollout_seed_ordinal=17,
        prompt_pool_fingerprint=_PROMPT_POOL_FINGERPRINT,
    )


@pytest.mark.parametrize(
    "failure",
    [
        opd.RetriableInfraError("checkpoint upload retry"),
        opd.RequiredSaveError("checkpoint upload rejected"),
    ],
)
def test_nonrequired_save_suppresses_upload_contract_errors(
    monkeypatch, tmp_path, failure
):
    def _fail_upload(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(
        opd,
        "_w",
        types.SimpleNamespace(SEED=42, upload_resume_checkpoint=_fail_upload),
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
        rollout_seed_ordinal=17,
        prompt_pool_fingerprint=_PROMPT_POOL_FINGERPRINT,
        required=False,
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
        rollout_seed_ordinal=17,
        prompt_pool_fingerprint=_PROMPT_POOL_FINGERPRINT,
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
            rollout_seed_ordinal=17,
            prompt_pool_fingerprint=_PROMPT_POOL_FINGERPRINT,
            required=True,
        )


def test_required_save_propagates_retriable_upload_error(monkeypatch, tmp_path):
    failure = opd.RetriableInfraError("checkpoint upload retry")

    def _fail_upload(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(
        opd,
        "_w",
        types.SimpleNamespace(SEED=42, upload_resume_checkpoint=_fail_upload),
    )

    with pytest.raises(opd.RetriableInfraError) as caught:
        opd._save_opd_resume_checkpoint(
            model=_FakeModel(),
            tok=_FakeTok(),
            optimizer=_FakeOptimizer([]),
            torch=_FakeTorch(),
            out_dir=str(tmp_path),
            opt_steps=2,
            step=2,
            accounting=_accounting(2),
            rollout_seed_ordinal=17,
            prompt_pool_fingerprint=_PROMPT_POOL_FINGERPRINT,
            required=True,
        )

    assert caught.value is failure


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
            rollout_seed_ordinal=17,
            prompt_pool_fingerprint=_PROMPT_POOL_FINGERPRINT,
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
        rollout_seed_ordinal=17,
        prompt_pool_fingerprint=_PROMPT_POOL_FINGERPRINT,
    )
    assert "stale-shard.bin" not in os.listdir(local_dir)


def test_required_opd_save_uploads_full_state_before_deployable(monkeypatch, tmp_path):
    events = []

    def upload(step, checkpoint_dir, *, after_upload=None):
        assert step == 2
        assert opd_resume_checkpoint_complete(os.listdir(checkpoint_dir))
        events.append("resume")
        after_upload()
        return True

    monkeypatch.setattr(opd, "_w", types.SimpleNamespace(SEED=42, upload_resume_checkpoint=upload))

    opd._save_opd_resume_checkpoint(
        model=_FakeModel(),
        tok=_FakeTok(),
        optimizer=_FakeOptimizer([]),
        torch=_FakeTorch(),
        out_dir=str(tmp_path),
        opt_steps=2,
        step=2,
        accounting=_accounting(2),
        rollout_seed_ordinal=5,
        prompt_pool_fingerprint=_PROMPT_POOL_FINGERPRINT,
        required=True,
        after_upload=lambda: events.append("deployable"),
    )

    assert events == ["resume", "deployable"]


def test_required_opd_save_propagates_permanent_deployable_failure_after_resume(
    monkeypatch, tmp_path
):
    events = []

    def upload(_step, _checkpoint_dir, *, after_upload=None):
        events.append("resume")
        after_upload()
        return True

    def fail_deployable():
        events.append("deployable")
        raise opd.RequiredSaveError("required adapter is invalid")

    monkeypatch.setattr(opd, "_w", types.SimpleNamespace(SEED=42, upload_resume_checkpoint=upload))

    with pytest.raises(opd.RequiredSaveError, match="required adapter is invalid"):
        opd._save_opd_resume_checkpoint(
            model=_FakeModel(),
            tok=_FakeTok(),
            optimizer=_FakeOptimizer([]),
            torch=_FakeTorch(),
            out_dir=str(tmp_path),
            opt_steps=2,
            step=2,
            accounting=_accounting(2),
            rollout_seed_ordinal=5,
            prompt_pool_fingerprint=_PROMPT_POOL_FINGERPRINT,
            required=True,
            after_upload=fail_deployable,
        )

    assert events == ["resume", "deployable"]


# --- required deployable reconciliation --------------------------------------------------------


def test_resume_republishes_missing_required_deployable_from_checkpoint(monkeypatch, tmp_path):
    checkpoint_dir = str(tmp_path / "checkpoint-3")
    events = []
    monkeypatch.setattr(
        opd,
        "_deployable_adapter_on_hf",
        lambda step: events.append(("lookup", step)) or False,
    )
    monkeypatch.setattr(
        opd,
        "_publish_opd_deployable",
        lambda path, step, **kwargs: events.append(("publish", path, step, kwargs)),
    )

    opd._reconcile_required_opd_deployable(checkpoint_dir, 3, (3, 8))

    assert events == [
        ("lookup", 3),
        (
            "publish",
            checkpoint_dir,
            3,
            {"as_default": False, "save_required": True},
        ),
    ]


def test_resume_skips_present_required_deployable(monkeypatch):
    published = []
    monkeypatch.setattr(opd, "_deployable_adapter_on_hf", lambda _step: True)
    monkeypatch.setattr(
        opd,
        "_publish_opd_deployable",
        lambda *args, **kwargs: published.append((args, kwargs)),
    )

    opd._reconcile_required_opd_deployable("/tmp/checkpoint-3", 3, (3,))

    assert published == []


def test_resume_required_deployable_lookup_failure_is_retriable(monkeypatch):
    failure = opd.RetriableInfraError("hf lookup failed")
    monkeypatch.setattr(
        opd,
        "_deployable_adapter_on_hf",
        lambda _step: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(opd.RetriableInfraError) as caught:
        opd._reconcile_required_opd_deployable("/tmp/checkpoint-3", 3, (3,))

    assert caught.value is failure


def test_resume_required_deployable_upload_failure_is_retriable(monkeypatch):
    failure = opd.RetriableInfraError("hf upload failed")
    monkeypatch.setattr(opd, "_deployable_adapter_on_hf", lambda _step: False)
    monkeypatch.setattr(
        opd,
        "_publish_opd_deployable",
        lambda *args, **kwargs: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(opd.RetriableInfraError) as caught:
        opd._reconcile_required_opd_deployable("/tmp/checkpoint-3", 3, (3,))

    assert caught.value is failure


def test_resume_nonrequired_step_does_not_publish_or_lookup(monkeypatch):
    monkeypatch.setattr(
        opd,
        "_deployable_adapter_on_hf",
        lambda _step: (_ for _ in ()).throw(AssertionError("must not query hf")),
    )
    monkeypatch.setattr(
        opd,
        "_publish_opd_deployable",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not publish")),
    )

    opd._reconcile_required_opd_deployable("/tmp/checkpoint-2", 2, (3, 8))


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
        rollout_seed_ordinal=17,
        prompt_pool_fingerprint=_PROMPT_POOL_FINGERPRINT,
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
        ckpt_dir,
        model=model,
        optimizer=optimizer,
        torch=torch,
        prompt_pool_fingerprint=_PROMPT_POOL_FINGERPRINT,
    )

    assert restored is not None
    assert restored["opt_steps"] == 3
    assert restored["step"] == 7
    assert restored["rollout_seed_ordinal"] == 17
    assert restored["prompt_pool_fingerprint"] == _PROMPT_POOL_FINGERPRINT
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
        "rollout_seed_ordinal": 17,
        "prompt_pool_fingerprint": _PROMPT_POOL_FINGERPRINT,
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
        prompt_pool_fingerprint=_PROMPT_POOL_FINGERPRINT,
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
        {"contract_version": 1},
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


@pytest.mark.parametrize("value", [None, -1, True, 1.5, "7"])
def test_restore_rejects_missing_or_invalid_rollout_seed_ordinal(monkeypatch, tmp_path, value):
    state = _valid_state()
    if value is None:
        state.pop("rollout_seed_ordinal")
    else:
        state["rollout_seed_ordinal"] = value
    ckpt = _write_state_dir(tmp_path / "checkpoint-3", state=state)

    result, order, optimizer, torch = _restore_expecting_none(monkeypatch, ckpt)

    assert result is None
    _assert_restore_untouched(order, optimizer, torch)


@pytest.mark.parametrize("value", [None, -0.1, float("nan"), float("inf"), True])
def test_restore_rejects_missing_or_invalid_train_wall(monkeypatch, tmp_path, value):
    state = _valid_state()
    if value is None:
        state.pop("train_wall_seconds")
    else:
        state["train_wall_seconds"] = value
    ckpt = _write_state_dir(tmp_path / "checkpoint-3", state=state)

    result, order, optimizer, torch = _restore_expecting_none(monkeypatch, ckpt)

    assert result is None
    _assert_restore_untouched(order, optimizer, torch)


def test_restore_rejects_missing_or_mismatched_prompt_pool_fingerprint(monkeypatch, tmp_path):
    for state in (
        {key: value for key, value in _valid_state().items() if key != "prompt_pool_fingerprint"},
        _valid_state(prompt_pool_fingerprint="b" * 64),
    ):
        ckpt = _write_state_dir(tmp_path / "checkpoint-3", state=state)
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
        ckpt,
        model=_FakeModel(),
        optimizer=optimizer,
        torch=_FakeTorch(),
        prompt_pool_fingerprint=_PROMPT_POOL_FINGERPRINT,
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
        ckpt,
        model=_FakeModel(),
        optimizer=optimizer,
        torch=torch,
        prompt_pool_fingerprint=_PROMPT_POOL_FINGERPRINT,
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
        ckpt,
        model=_FakeModel(),
        optimizer=optimizer,
        torch=torch,
        prompt_pool_fingerprint=_PROMPT_POOL_FINGERPRINT,
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
        ckpt,
        model=_FakeModel(),
        optimizer=_FakeOptimizer(order),
        torch=torch,
        prompt_pool_fingerprint=_PROMPT_POOL_FINGERPRINT,
    )
    assert result is None


def test_downloaded_invalid_checkpoint_is_terminal_and_never_starts_fresh():
    started = []

    def restore_then_start():
        state = opd._require_restored_opd_state(None)
        started.append(state)

    with pytest.raises(RuntimeError, match="could not be restored") as exc_info:
        restore_then_start()

    assert not isinstance(exc_info.value, opd.RetriableInfraError)
    assert started == []


@pytest.mark.parametrize(
    ("error", "expected_oom"),
    [(_FakeCudaOutOfMemoryError("cuda oom"), True), (MemoryError("host oom"), False)],
)
def test_restore_memory_failures_propagate_without_retriable_reclassification(
    monkeypatch, tmp_path, error, expected_oom
):
    import flash.engine.worker as worker

    ckpt = _write_state_dir(tmp_path / "checkpoint-3", state=_valid_state())
    monkeypatch.setattr(opd, "_w", types.SimpleNamespace(SEED=42))
    order = []
    _install_fake_peft(monkeypatch, order)

    class _RaisingTorch(_FakeTorch):
        def load(self, path, map_location=None, weights_only=None):
            raise error

    with pytest.raises(type(error)) as exc_info:
        opd._restore_opd_full_state(
            ckpt,
            model=_FakeModel(),
            optimizer=_FakeOptimizer(order),
            torch=_RaisingTorch(),
            prompt_pool_fingerprint=_PROMPT_POOL_FINGERPRINT,
        )

    monkeypatch.setattr(
        worker,
        "is_cuda_oom",
        lambda exc: isinstance(exc, _FakeCudaOutOfMemoryError),
    )
    assert worker._worker_failure_flags(exc_info.value) == {
        "retriable": False,
        "oom": expected_oom,
    }
