import threading

import pytest

from flash.engine.worker.train.opd.bridging.bridge import _TeacherAlignmentBridge


def _bridge(**kwargs):
    return _TeacherAlignmentBridge(
        prompts=[],
        teacher=object(),
        thinking_prefill="",
        eos_token_ids=frozenset(),
        stop_sequences=(),
        mutation_callback=lambda: None,
        **kwargs,
    )


@pytest.mark.parametrize("failed_start", [2, 3])
def test_bridge_start_failure_releases_socket_and_started_threads(monkeypatch, failed_start):
    bridge = _bridge(multi_turn=True, max_turns=1, active_env=object())
    real_start = threading.Thread.start
    starts = 0

    def start(thread):
        nonlocal starts
        starts += 1
        if starts == failed_start:
            raise RuntimeError("thread startup failed")
        real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    try:
        with pytest.raises(RuntimeError, match="thread startup failed"):
            bridge.start()
        assert bridge._server.socket.fileno() == -1
        assert not bridge._text_teacher_batcher._thread.is_alive()
        assert bridge._thread is None or not bridge._thread.is_alive()
    finally:
        if bridge._thread is not None and bridge._thread.ident is not None:
            bridge._server.shutdown()
            bridge._thread.join(timeout=2)
        if bridge._server is not None:
            bridge._server.server_close()
        if bridge._text_teacher_batcher is not None:
            bridge._text_teacher_batcher.close()


def test_bridge_close_releases_socket_when_batcher_close_raises(monkeypatch):
    bridge = _bridge()
    bridge.start()
    batcher = bridge._text_teacher_batcher
    close = batcher.close

    def fail_close():
        close()
        raise RuntimeError("batcher close failed")

    monkeypatch.setattr(batcher, "close", fail_close)
    try:
        with pytest.raises(RuntimeError, match="batcher close failed"):
            bridge.close()
        assert bridge._server.socket.fileno() == -1
        assert not bridge._thread.is_alive()
    finally:
        bridge._server.server_close()
        bridge._thread.join(timeout=2)
