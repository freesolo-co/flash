"""Off-thread field sampling must stop if the heartbeat thread cannot start."""

import threading

import pytest

from flash.engine.worker.io import heartbeat


def test_heartbeat_start_failure_stops_the_field_sampler(monkeypatch):
    real_start = threading.Thread.start
    sampler_threads = []
    sampled = threading.Event()

    def start(thread):
        if sampler_threads:
            raise RuntimeError("heartbeat startup failed")
        sampler_threads.append(thread)
        real_start(thread)
        assert sampled.wait(timeout=2)

    def fields():
        sampled.set()
        return {}

    monkeypatch.setattr(threading.Thread, "start", start)
    try:
        with (
            pytest.raises(RuntimeError, match="heartbeat startup failed"),
            heartbeat.liveness_heartbeat("opd_step", fields=fields, sample_off_thread=True),
        ):
            pytest.fail("startup should not succeed")
        sampler_threads[0].join(timeout=1)
        assert not sampler_threads[0].is_alive()
    finally:
        # Stop the original implementation's leaked daemon after the failing assertion.
        thread = sampler_threads[0]
        if thread.is_alive():
            thread._target.__self__._done.set()
            thread.join(timeout=2)
