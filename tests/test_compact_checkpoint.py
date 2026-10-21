"""Compact recovery omits only frozen base tensors and reconstructs them strictly."""

import sys
from dataclasses import replace
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


def test_training_child_inherits_pinned_base(worker_context, monkeypatch):
    from flash.engine.worker.train.entry import sft_train

    monkeypatch.setattr(
        worker_context,
        "config",
        replace(
            worker_context.config, spec=SimpleNamespace(model="test/model", model_revision=REVISION)
        ),
    )
    child = sft_train._build_verl_child_env(
        shim_dir="/shim", wandb_enabled=False, context=worker_context
    )
    assert child["FLASH_CHECKPOINT_MODEL_ID"] == "test/model"
    assert child["FLASH_CHECKPOINT_MODEL_REVISION"] == REVISION


def test_runtime_always_installs_compact_checkpoints(monkeypatch):
    from flash.engine.worker.train.core.child import compact_checkpoint, runtime

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
    runtime.install_compact_checkpoints("markers.txt")
    assert calls == [(target, {"model_id": "test/model", "revision": REVISION})]


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
