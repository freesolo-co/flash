"""Watchdog startup failures must not leave the training process running."""

from types import SimpleNamespace

import pytest

from flash.engine.worker.train.entry import backend_common
from flash.engine.worker.verl import diagnostics


def _failing_watchdog(monkeypatch):
    def fail_start(thread):
        raise RuntimeError("cannot start watchdog thread")

    monkeypatch.setattr(diagnostics.threading.Thread, "start", fail_start)
    return diagnostics.VerlChildSilenceWatchdog(diagnostics.ChildOutputTail(), baseline_step=0)


def test_watchdog_can_stop_after_thread_start_fails(monkeypatch):
    watchdog = _failing_watchdog(monkeypatch)
    watchdog.bind(child_alive=lambda: True, teardown=lambda: None)
    with pytest.raises(RuntimeError, match="cannot start watchdog thread"):
        watchdog.start()
    watchdog.stop()


def test_subprocess_is_torn_down_when_watchdog_cannot_start(monkeypatch):
    watchdog = _failing_watchdog(monkeypatch)
    proc = SimpleNamespace(pid=123, returncode=None)
    proc.poll = lambda: proc.returncode
    killed = []

    def kill(child, *, process_group_id):
        killed.append(process_group_id)
        child.returncode = -15

    monkeypatch.setattr(backend_common.subprocess, "Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(backend_common, "adopt_orphaned_descendants", lambda: True)
    monkeypatch.setattr(backend_common, "reap_stragglers", lambda: None)
    monkeypatch.setattr(backend_common, "kill_process_group", kill)
    with pytest.raises(RuntimeError, match="cannot start watchdog thread"):
        backend_common._run_streaming_verl_subprocess(
            ["training-child"], env={}, on_line=lambda line: None, silence_watchdog=watchdog
        )
    assert killed == [proc.pid]
    assert proc.poll() is not None
