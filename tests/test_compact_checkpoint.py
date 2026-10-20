"""Compact recovery omits only frozen base tensors and reconstructs them strictly."""

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


def test_modal_controller_forwards_compact_option_before_rental(monkeypatch):
    import time

    from flash.providers.modal import api, jobs
    from tests._helpers.source_snapshot import valid_source_snapshot
    from tests.test_modal_training_lifecycle import spec

    monkeypatch.setenv("FLASH_COMPACT_CHECKPOINTS", "1")
    monkeypatch.setattr("flash.runtime_capsule.build_capsule", lambda _: (b"capsule", {}))

    def capture(**kwargs):
        assert kwargs["payload"]["env"]["FLASH_COMPACT_CHECKPOINTS"] == "1"
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
