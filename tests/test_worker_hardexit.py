"""Regression test: the GPU worker must hard-exit on success so a hanging vLLM/torch
teardown can't stall the run.

Bug: after the RL train phase uploaded its adapter + train_meta, the worker process hung at
interpreter shutdown (colocated-vLLM NCCL/CUDA teardown deadlock). Because the Flash handler
runs the train phase via a *blocking* ``subprocess.run`` and only starts the eval phase once
it returns, the eval phase never started — the run froze at the ``rl_train_done`` heartbeat
until the wall-clock cap. ``check=False`` tolerated a segfault-at-exit but not a hang.

Fix: ``worker.main`` calls ``os._exit(0)`` after the handler completes (all artifacts are
already persisted to HF inside the handler), bypassing the hanging teardown.
"""

from __future__ import annotations

import flash.engine.worker as worker


class _HardExit(BaseException):
    """Stand-in for os._exit so the test process survives; BaseException so it propagates
    past the worker's ``except Exception`` block (mirroring a real os._exit)."""

    def __init__(self, code):
        self.code = code


def _patch_common(monkeypatch, fake_exit):
    monkeypatch.setattr(worker.os, "_exit", fake_exit)
    monkeypatch.setattr(worker, "HF_REPO", "")  # skip the idempotency DONE check
    monkeypatch.setattr(worker, "RUN_MODE", "sft")
    monkeypatch.setattr(worker, "heartbeat", lambda *a, **k: None)
    monkeypatch.setattr(worker.time, "sleep", lambda *a, **k: None)


def test_worker_hard_exits_zero_on_success(monkeypatch):
    ran = {"v": False}

    def fake_exit(code=0):
        raise _HardExit(code)

    _patch_common(monkeypatch, fake_exit)
    monkeypatch.setattr(worker, "run_sft", lambda: ran.__setitem__("v", True))

    raised = None
    try:
        worker.main()
    except _HardExit as e:
        raised = e
    assert ran["v"] is True, "handler must run"
    assert raised is not None, "worker.main must hard-exit on success"
    assert raised.code == 0, "must exit 0 on success"


def test_worker_does_not_hard_exit_on_failure(monkeypatch):
    called = {"exit": False}

    def fake_exit(code=0):
        called["exit"] = True
        raise _HardExit(code)

    _patch_common(monkeypatch, fake_exit)

    def boom():
        raise ValueError("train blew up")

    monkeypatch.setattr(worker, "run_sft", boom)

    raised_value_error = False
    try:
        worker.main()
    except ValueError:
        raised_value_error = True
    except _HardExit:
        # A hard-exit here would mean the bug under test; the assertions below catch it.
        pass
    assert raised_value_error, "the real error must propagate (not be swallowed by a hard-exit)"
    assert called["exit"] is False, "must NOT hard-exit when the handler failed"
