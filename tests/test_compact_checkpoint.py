"""Compact recovery omits only frozen base tensors and reconstructs them strictly."""

import sys
from types import SimpleNamespace

import pytest

from flash.engine.worker.train.core.child.compact_checkpoint import compact_state, restore_state

REVISION = "a" * 40


class Model:
    def named_parameters(self, **kwargs):
        return [
            ("base.weight", SimpleNamespace(requires_grad=False)),
            ("lora_A.weight", SimpleNamespace(requires_grad=True)),
        ]

    def named_buffers(self, **kwargs):
        return [("counter", object())]

    def state_dict(self):
        return {"base.weight": "pinned base", "lora_A.weight": "initial adapter", "counter": 0}


def checkpoint():
    state = {"base.weight": "pinned base", "lora_A.weight": "trained adapter", "counter": 7}
    omitted = compact_state(Model(), state)
    return state, {
        "version": 1,
        "model_id": "test/model",
        "revision": REVISION,
        "omitted": omitted,
        "retained": sorted(state),
    }


def test_compact_roundtrip_preserves_trainables_and_mutable_buffers():
    state, manifest = checkpoint()
    assert state == {"lora_A.weight": "trained adapter", "counter": 7}
    restore_state(Model(), state, manifest, model_id="test/model", revision=REVISION)
    assert state == {"base.weight": "pinned base", "lora_A.weight": "trained adapter", "counter": 7}


@pytest.mark.parametrize(
    "mutation", ["base", "trainable", "buffer", "missing", "extra", "duplicate"]
)
def test_corrupt_compact_state_is_rejected(mutation):
    state, manifest = checkpoint()
    if mutation == "base":
        manifest["revision"] = "b" * 40
    elif mutation in ("trainable", "buffer"):
        key = "lora_A.weight" if mutation == "trainable" else "counter"
        del state[key]
        manifest["retained"].remove(key)
        manifest["omitted"] = sorted([*manifest["omitted"], key])
    elif mutation == "missing":
        del state["lora_A.weight"]
    elif mutation == "extra":
        state["unknown"] = 1
    else:
        manifest["omitted"] *= 2
    with pytest.raises(RuntimeError):
        restore_state(Model(), state, manifest, model_id="test/model", revision=REVISION)


@pytest.mark.parametrize(("option", "expected"), [(None, "1"), ("1", "1"), ("0", "0")])
def test_modal_controller_forwards_compact_option_before_rental(monkeypatch, option, expected):
    import time

    from flash.providers.modal import api, jobs
    from tests._helpers.source_snapshot import valid_source_snapshot
    from tests.test_modal_training_lifecycle import spec

    monkeypatch.delenv("FLASH_COMPACT_CHECKPOINTS", raising=False)
    if option is not None:
        monkeypatch.setenv("FLASH_COMPACT_CHECKPOINTS", option)
    monkeypatch.setattr("flash.runtime_capsule.build_capsule", lambda _: (b"capsule", {}))

    def capture(**kwargs):
        assert kwargs["payload"]["env"]["FLASH_COMPACT_CHECKPOINTS"] == expected
        raise RuntimeError("captured before rental")

    monkeypatch.setattr(api, "create", capture)
    with pytest.raises(RuntimeError, match="captured before rental"):
        jobs.submit_attempt(
            spec(), source_snapshot=valid_source_snapshot(), deadline_at=time.time() + 60
        )
    monkeypatch.setenv("FLASH_COMPACT_CHECKPOINTS", "invalid")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        jobs.submit_attempt(
            spec(), source_snapshot=valid_source_snapshot(), deadline_at=time.time() + 60
        )


@pytest.mark.parametrize(("option", "expected"), [(None, "1"), ("1", "1"), ("0", "0")])
def test_training_child_inherits_compact_default_and_pinned_base(monkeypatch, option, expected):
    from flash.engine.worker.train.entry import sft_train

    monkeypatch.delenv("FLASH_COMPACT_CHECKPOINTS", raising=False)
    if option is not None:
        monkeypatch.setenv("FLASH_COMPACT_CHECKPOINTS", option)
    monkeypatch.setattr(
        sft_train._worker_state,
        "JOB_SPEC",
        SimpleNamespace(model="test/model", model_revision=REVISION),
    )
    child = sft_train._build_verl_child_env(shim_dir="/shim", wandb_enabled=False)
    assert child["FLASH_COMPACT_CHECKPOINTS"] == expected
    assert child["FLASH_CHECKPOINT_MODEL_ID"] == "test/model"
    assert child["FLASH_CHECKPOINT_MODEL_REVISION"] == REVISION


@pytest.mark.parametrize("option", [None, "1", "0", "invalid"])
def test_runtime_installs_compact_by_default(monkeypatch, option):
    from flash.engine.worker.train.core.child import compact_checkpoint, runtime

    monkeypatch.delenv("FLASH_COMPACT_CHECKPOINTS", raising=False)
    if option is not None:
        monkeypatch.setenv("FLASH_COMPACT_CHECKPOINTS", option)
    monkeypatch.setenv("FLASH_CHECKPOINT_MODEL_ID", "test/model")
    monkeypatch.setenv("FLASH_CHECKPOINT_MODEL_REVISION", REVISION)
    calls = []
    target = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "verl.utils.checkpoint.fsdp_checkpoint_manager", target)
    monkeypatch.setattr(
        compact_checkpoint, "install", lambda module, **kwargs: calls.append((module, kwargs))
    )
    monkeypatch.setattr(
        runtime, "install_deferred_required", lambda name, marker, module, apply: apply()
    )
    if option == "invalid":
        with pytest.raises(RuntimeError, match="must be 0 or 1"):
            runtime.install_compact_checkpoints("markers.txt")
    else:
        runtime.install_compact_checkpoints("markers.txt")
    assert calls == (
        [(target, {"model_id": "test/model", "revision": REVISION})]
        if option in (None, "1")
        else []
    )


@pytest.mark.parametrize("algorithm", ["sft", "grpo", "opd"])
def test_submission_pins_base_before_other_preparation(monkeypatch, algorithm):
    from flash.core.spec import JobSpec
    from flash.runner.lifecycle import preparation, submit

    spec = JobSpec(model="Qwen/Qwen3.5-9B", algorithm=algorithm)
    resolve = preparation._resolve_model_revision
    monkeypatch.setattr(
        "huggingface_hub.HfApi.model_info", lambda *args, **kwargs: SimpleNamespace(sha=REVISION)
    )

    def check_pin(spec, **kwargs):
        pinned = resolve(spec, **kwargs)
        assert pinned.model_revision == REVISION
        assert pinned.model_revision_auto
        raise RuntimeError("base pinned before further preparation")

    monkeypatch.setattr(preparation, "_resolve_model_revision", check_pin)
    with pytest.raises(RuntimeError, match="base pinned before further preparation"):
        submit.prepare_job(spec)
