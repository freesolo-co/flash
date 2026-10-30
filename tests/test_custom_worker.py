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
