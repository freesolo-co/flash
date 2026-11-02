"""Managed controller idle allowance starts after model and checkpoint startup work."""

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


def test_slow_restore_does_not_consume_controller_idle_allowance(tmp_path, monkeypatch):
    from flash.custom import worker
    from flash.schema import spec_from_dict

    spec = spec_from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "custom",
            "custom": {"file": "algorithm.py", "idle_seconds": 30},
        }
    )
    spec = replace(spec, custom={**spec.custom, "resolved_resume": {"path": "saved"}})
    context = SimpleNamespace(
        config=SimpleNamespace(spec=spec, attempt=0), remaining_wall_seconds=lambda: 3600
    )
    clock = [100.0]
    monkeypatch.setattr(worker.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(worker, "load_backend", Mock(return_value=Mock()))
    monkeypatch.setattr(worker, "heartbeat", Mock())
    monkeypatch.setattr(worker, "liveness_heartbeat", lambda *a, **k: nullcontext())
    monkeypatch.setattr(worker, "session_token", lambda *a: "session-token")
    monkeypatch.setattr(worker, "Path", lambda path: tmp_path / Path(path).name)
    monkeypatch.setattr(worker.hf, "hf_upload_file", Mock())

    def restore(*args):
        clock[0] += 45
        return {"checkpoint": {"step": 2}, "algorithm_state": {"cursor": 2}}

    store = Mock()
    store.restore.side_effect = restore
    monkeypatch.setattr(worker, "RunCheckpoints", lambda *a: store)

    def serve(session, *args, **kwargs):
        assert session.step == 2
        assert session.restored_state == {"cursor": 2}
        assert not session.expired()
        clock[0] += 29
        assert not session.expired()
        clock[0] += 2
        assert session.expired()
        session.closed = True

    server = Mock(side_effect=serve)
    monkeypatch.setattr(worker, "serve_session", server)
    worker.run(context=context)
    server.assert_called_once()


def test_warm_start_downloads_verified_source_before_loading_backend(monkeypatch):
    from flash.custom import worker
    from flash.engine.worker.train.entry import sft_train
    from flash.schema import spec_from_dict

    spec = spec_from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "custom",
            "custom": {"file": "algorithm.py"},
            "train": {"init_from_adapter": "source-run/step-874"},
        }
    )
    spec = replace(spec, train=replace(spec.train, lora_rank=32, lora_alpha=64))
    context = SimpleNamespace(
        config=SimpleNamespace(spec=spec, attempt=0), remaining_wall_seconds=lambda: 3600
    )
    calls = []

    def warmstart(model, revision, rank, alpha, targeting, *, context):
        calls.append((model, rank, alpha, targeting is not None))
        return "/tmp/evdl/source/adapter"

    monkeypatch.setattr(sft_train, "_warmstart_adapter_path", warmstart)
    load = Mock(side_effect=RuntimeError("stop after model load"))
    monkeypatch.setattr(worker, "load_backend", load)
    monkeypatch.setattr(worker, "heartbeat", Mock())
    monkeypatch.setattr(worker, "liveness_heartbeat", lambda *a, **k: nullcontext())
    import pytest

    with pytest.raises(RuntimeError, match="stop after model load"):
        worker.run(context=context)
    from flash.core.catalog import supports_image_training

    # image-training models check the source against custom's text-only surface.
    multimodal = supports_image_training("Qwen/Qwen3.5-0.8B")
    assert calls == [("Qwen/Qwen3.5-0.8B", 32, 64, multimodal)]
    assert load.call_args.kwargs["init_adapter"] == "/tmp/evdl/source/adapter"
    assert load.call_args.kwargs["lora_rank"] == 32


def test_warm_start_detects_the_source_adapters_module_layout(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file

    from flash.custom.backend import _uses_language_model_layout

    for name, key, expected in (
        (
            "multimodal",
            "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_A.weight",
            True,
        ),
        ("text", "base_model.model.model.layers.0.mlp.up_proj.lora_A.weight", False),
    ):
        adapter = tmp_path / name
        adapter.mkdir()
        save_file(
            {key: np.zeros((2, 2), dtype=np.float32)}, str(adapter / "adapter_model.safetensors")
        )
        assert _uses_language_model_layout(adapter) is expected
