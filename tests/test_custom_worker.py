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
    monkeypatch.setenv("FLASH_CUSTOM_SESSION_TOKEN", "session-token")
    monkeypatch.setattr(worker, "Path", lambda path: tmp_path / Path(path).name)
    monkeypatch.setattr(worker.hf, "upload_artifact_file", Mock())

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
    # no modality is imposed: an image-trained source (SFT on screenshots) is a valid start.
    assert calls == [("Qwen/Qwen3.5-0.8B", 32, 64, False)]
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


def test_resume_builds_the_lora_from_the_checkpoint_adapter(monkeypatch):
    from flash.custom import worker
    from flash.schema import spec_from_dict

    spec = spec_from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "custom",
            "custom": {"file": "algorithm.py"},
            "train": {"lora_rank": 32, "lora_alpha": 64},
        }
    )
    resolved = {
        "repo": "org/artifacts",
        "revision": "abc",
        "path": "custom/run/checkpoints/step-50",
    }
    spec = replace(spec, custom={**spec.custom, "resolved_resume": resolved})
    context = SimpleNamespace(
        config=SimpleNamespace(spec=spec, attempt=0), remaining_wall_seconds=lambda: 3600
    )
    store = Mock()
    store.download_adapter.return_value = "/tmp/resume/adapter"
    monkeypatch.setattr(worker, "RunCheckpoints", lambda *a: store)
    load = Mock(side_effect=RuntimeError("stop after model load"))
    monkeypatch.setattr(worker, "load_backend", load)
    monkeypatch.setattr(worker, "heartbeat", Mock())
    monkeypatch.setattr(worker, "liveness_heartbeat", lambda *a, **k: nullcontext())
    import pytest

    with pytest.raises(RuntimeError, match="stop after model load"):
        worker.run(context=context)
    assert store.download_adapter.call_args.args[0] == resolved
    # the checkpoint's own adapter config fixes the topology that restore then overwrites.
    assert load.call_args.kwargs["init_adapter"] == "/tmp/resume/adapter"


def _managed_worker(tmp_path, monkeypatch, *, wall=3600, remaining=50_000.0):
    from flash.custom import worker
    from flash.schema import spec_from_dict

    spec = spec_from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "custom",
            "custom": {"file": "algorithm.py", "idle_seconds": 60, "max_wall_seconds": wall},
        }
    )
    context = SimpleNamespace(
        config=SimpleNamespace(spec=spec, attempt=0), remaining_wall_seconds=lambda: remaining
    )
    backend = Mock()
    backend.algorithm_state = {"cursor": 7}
    monkeypatch.setattr(worker, "load_backend", Mock(return_value=backend))
    beats = Mock()
    monkeypatch.setattr(worker, "heartbeat", beats)
    monkeypatch.setattr(worker, "liveness_heartbeat", lambda *a, **k: nullcontext())
    monkeypatch.setenv("FLASH_CUSTOM_SESSION_TOKEN", "session-token")
    monkeypatch.setattr(worker, "Path", lambda path: tmp_path / Path(path).name)
    monkeypatch.setattr(worker.hf, "upload_artifact_file", Mock())
    store = Mock()
    store.published = {}
    monkeypatch.setattr(worker, "RunCheckpoints", lambda *a: store)
    return worker, context, backend, beats


def test_wall_cap_counts_session_time_and_saves_the_reached_step(tmp_path, monkeypatch):
    import pytest

    worker, context, backend, _ = _managed_worker(tmp_path, monkeypatch, wall=1800)
    served = {}

    def serve(session, token, max_wall_seconds, **kwargs):
        served["cap"] = max_wall_seconds
        session.attach()
        session.step = 5  # the controller made progress, then the cap arrived
        return "wall"

    monkeypatch.setattr(worker, "serve_session", serve)
    with pytest.raises(RuntimeError, match=r"wall-clock cap; saved checkpoint .*/step-5"):
        worker.run(context=context)
    # the cap is the session allowance (queue time is not in it), minus the save reserve
    assert served["cap"] == 1800 - worker.cap_save_reserve(1800)
    backend.save_checkpoint.assert_called_once()
    assert backend.save_checkpoint.call_args.args[0] == "step-5"
    assert backend.save_checkpoint.call_args.kwargs["algorithm_state"] == {"cursor": 7}


def test_session_no_controller_reached_is_not_billed(tmp_path, monkeypatch):
    import pytest

    worker, context, backend, beats = _managed_worker(tmp_path, monkeypatch)
    monkeypatch.setattr(worker, "serve_session", lambda session, *a, **k: "ended")
    with pytest.raises(RuntimeError, match="no custom controller attached"):
        worker.run(context=context)
    backend.save_checkpoint.assert_not_called()
    waiting = [c for c in beats.call_args_list if c.args[0] == "custom_waiting"]
    assert waiting
    for call in waiting:
        assert call.kwargs["session_seconds"] == 0
        assert "session_started_at" not in call.kwargs
        assert call.kwargs["controller_attached"] is False


def test_billing_clock_starts_at_controller_attach(tmp_path, monkeypatch):
    worker, context, _, beats = _managed_worker(tmp_path, monkeypatch)
    clock = [100.0]
    monkeypatch.setattr(worker.time, "monotonic", lambda: clock[0])

    def serve(session, *args, tick, **kwargs):
        clock[0] += 500  # GPU idles with no controller: not billable
        session.attach()
        clock[0] += 61
        tick()
        session.closed = True
        return "ended"

    monkeypatch.setattr(worker, "serve_session", serve)
    worker.run(context=context)
    waiting = [c for c in beats.call_args_list if c.args[0] == "custom_waiting"]
    assert waiting[-1].kwargs["session_seconds"] == 61
    assert waiting[-1].kwargs["controller_attached"] is True
    done = next(c for c in beats.call_args_list if c.args[0] == "done")
    assert done.kwargs["session_seconds"] == 61


def test_disconnected_controller_saves_progress_but_failed_session_does_not(tmp_path, monkeypatch):
    import pytest

    worker, context, backend, _ = _managed_worker(tmp_path, monkeypatch)

    def serve(session, *args, **kwargs):
        session.attach()
        session.step = 3
        return "ended"

    monkeypatch.setattr(worker, "serve_session", serve)
    with pytest.raises(RuntimeError, match="disconnected or timed out; saved checkpoint"):
        worker.run(context=context)
    backend.save_checkpoint.assert_called_once()

    worker, context, backend, _ = _managed_worker(tmp_path, monkeypatch)

    def failed(session, *args, **kwargs):
        session.attach()
        session.step = 3
        session.failed = True  # a partially applied update must never be published
        return "ended"

    monkeypatch.setattr(worker, "serve_session", failed)
    with pytest.raises(RuntimeError, match="disconnected or timed out; resume"):
        worker.run(context=context)
    backend.save_checkpoint.assert_not_called()


def test_attach_is_reported_promptly_and_a_lost_session_persists_its_time(tmp_path, monkeypatch):
    import pytest

    worker, context, _, beats = _managed_worker(tmp_path, monkeypatch)
    clock = [100.0]
    monkeypatch.setattr(worker.time, "monotonic", lambda: clock[0])

    def serve(session, *args, tick, **kwargs):
        session.attach()
        clock[0] += 5
        tick()  # the first watchdog tick after attach, well before the 60 s cadence
        attached = beats.call_args_list[-1].kwargs
        assert attached["controller_attached"] is True
        assert "session_started_at" in attached
        tick()
        assert beats.call_args_list[-1].kwargs is attached  # no duplicate beat
        clock[0] += 20
        return "ended"  # preempted or disconnected before the first optimizer step

    monkeypatch.setattr(worker, "serve_session", serve)
    with pytest.raises(RuntimeError, match="disconnected or timed out"):
        worker.run(context=context)
    final = beats.call_args_list[-1]
    assert final.args[0] == "custom_waiting"
    assert final.kwargs["session_seconds"] == 25
