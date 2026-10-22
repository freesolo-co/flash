"""Bootstrap setup and interruption failures must release the paid worker."""

from types import SimpleNamespace

import pytest

from flash.providers._lifecycle.bootstrapping import bootstrap


@pytest.mark.parametrize("failure_stage", ["pump_start", "uploader_start", "wait"])
def test_bootstrap_failure_terminates_worker_group(monkeypatch, tmp_path, failure_stage):
    failure = RuntimeError(f"{failure_stage} failed")
    killed = []

    def fail(stage):
        if failure_stage == stage:
            raise failure

    class PumpThread:
        def __init__(self, *, target, **kwargs):
            self.target = target

        def start(self):
            fail("pump_start")
            self.target()

    def wait(**kwargs):
        fail("wait")

    proc = SimpleNamespace(pid=123, stdout=iter(()), wait=wait, returncode=None)

    def start_uploader(*args):
        fail("uploader_start")
        return object(), object()

    monkeypatch.setattr(bootstrap, "_code_dir", lambda payload: str(tmp_path))
    monkeypatch.setattr(bootstrap.threading, "Thread", PumpThread)
    monkeypatch.setattr(
        bootstrap._bootstrap_processes, "start_process_group", lambda *a, **k: (proc, proc.pid)
    )
    monkeypatch.setattr(
        bootstrap._bootstrap_processes,
        "terminate_process_group",
        lambda child, **kwargs: killed.append((child, kwargs)),
    )
    monkeypatch.setattr(bootstrap, "_start_console_uploader", start_uploader)
    monkeypatch.setattr(bootstrap, "_stop_upload_process", lambda *args: True)
    deadline = bootstrap.time.time() + 100
    with pytest.raises(RuntimeError) as raised:
        bootstrap.run_mode({}, {}, "sft", deadline)
    assert raised.value is failure
    assert len(killed) == 1
    assert killed[0][0] is proc
    assert killed[0][1]["process_group_id"] == proc.pid
    assert killed[0][1]["deadline_at"] < deadline


def test_bootstrap_pipe_stall_terminates_remaining_worker_group(monkeypatch, tmp_path):
    killed = []
    proc = SimpleNamespace(pid=123, wait=lambda **kwargs: 0, returncode=0)
    monkeypatch.setattr(bootstrap, "_code_dir", lambda payload: str(tmp_path))
    monkeypatch.setattr(
        bootstrap.threading, "Thread", lambda **kwargs: SimpleNamespace(start=lambda: None)
    )
    monkeypatch.setattr(
        bootstrap.threading,
        "Event",
        lambda: SimpleNamespace(wait=lambda timeout: False),
    )
    monkeypatch.setattr(
        bootstrap._bootstrap_processes, "start_process_group", lambda *a, **k: (proc, proc.pid)
    )
    monkeypatch.setattr(
        bootstrap._bootstrap_processes,
        "terminate_process_group",
        lambda child, **kwargs: killed.append((child, kwargs)),
    )
    monkeypatch.setattr(bootstrap, "_start_console_uploader", lambda *args: (object(), object()))
    monkeypatch.setattr(bootstrap, "_stop_upload_process", lambda *args: True)
    monkeypatch.setattr(bootstrap, "_upload_console_tail_bounded", lambda *args: True)
    with pytest.raises(TimeoutError, match="wall-clock cap"):
        bootstrap.run_mode({}, {}, "sft", bootstrap.time.time() + 100)
    assert len(killed) == 1
    assert killed[0][1]["process_group_id"] == proc.pid
