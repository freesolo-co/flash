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
    child = sft_train._build_verl_child_env(wandb_enabled=False, context=worker_context)
    assert child["FLASH_CHECKPOINT_MODEL_ID"] == "test/model"
    assert child["FLASH_CHECKPOINT_MODEL_REVISION"] == REVISION


def test_checkpoint_hooks_are_scoped_to_one_model_and_removed_after_failure(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from flash.engine.worker.train.core.child import compact_checkpoint

    monkeypatch.setenv("FLASH_CHECKPOINT_MODEL_ID", "test/model")
    monkeypatch.setenv("FLASH_CHECKPOINT_MODEL_REVISION", REVISION)
    monkeypatch.setitem(
        sys.modules, "verl.utils.fsdp_utils", SimpleNamespace(fsdp_version=lambda _: 2)
    )

    def model():
        result = torch.nn.Module()
        result.register_parameter(
            "base", torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        )
        result.register_parameter("lora_A", torch.nn.Parameter(torch.tensor(2.0)))
        result.register_buffer("counter", torch.tensor(3.0))
        return result

    selected, untouched = model(), model()
    manager = SimpleNamespace(model=selected, rank=0, should_save_model=True)
    state = {}

    def failing_save():
        with compact_checkpoint.saving(manager, tmp_path):
            state.update(selected.state_dict())
            assert set(state) == {"lora_A", "counter"}
            assert set(untouched.state_dict()) == {"base", "lora_A", "counter"}
            raise RuntimeError("save failed")

    with pytest.raises(RuntimeError, match="save failed"):
        failing_save()
    assert set(selected.state_dict()) == {"base", "lora_A", "counter"}
    state["lora_A"] = torch.tensor(8.0)
    with compact_checkpoint.loading(manager, tmp_path):
        selected.load_state_dict(state, strict=True)
    assert selected.lora_A.item() == 8.0
    assert selected.base.item() == 1.0
    with pytest.raises(RuntimeError, match="Missing key"):
        selected.load_state_dict(state, strict=True)
    with (
        pytest.raises(RuntimeError, match="load failed"),
        compact_checkpoint.loading(manager, tmp_path),
    ):
        raise RuntimeError("load failed")
    assert not selected._load_state_dict_pre_hooks


@pytest.mark.parametrize("algorithm", ["sft", "grpo"])
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
