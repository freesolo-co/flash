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

import huggingface_hub
import pytest

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
    # main() runs the real boot steps before the handler; this test exercises the hard-exit flow,
    # so stub out the Hopper fla fast-path setup and the alloc-conf finalize.
    monkeypatch.setattr(worker, "_ensure_fla_fastpath_on_hopper", lambda: None)
    monkeypatch.setattr(worker, "finalize_alloc_conf_for_sleep", lambda: None)


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


def test_worker_dispatches_opd_run_mode(monkeypatch):
    """RUN_MODE=='opd' must dispatch to run_opd (the third algorithm's worker handler)."""
    ran = {"v": False}

    def fake_exit(code=0):
        raise _HardExit(code)

    _patch_common(monkeypatch, fake_exit)
    monkeypatch.setattr(worker, "RUN_MODE", "opd")
    monkeypatch.setattr(worker, "run_opd", lambda: ran.__setitem__("v", True))

    raised = None
    try:
        worker.main()
    except _HardExit as e:
        raised = e
    assert ran["v"] is True, "RUN_MODE=opd must invoke run_opd"
    assert raised is not None
    assert raised.code == 0


def test_idempotency_replay_metrics_read_failure_is_retriable(monkeypatch, tmp_path):
    """DONE present but a transient HF read of the persisted metrics.json must surface as a RETRIABLE
    RetriableInfraError (so the run reschedules and a fresh worker re-enters the idempotency path) —
    NOT a SystemExit, which is a BaseException that bypasses the retriable-stamping handler and would
    report a genuinely-succeeded run as a fatal failure."""
    hb = []
    exited = {"v": False}

    def fake_exit(code=0):
        exited["v"] = True
        raise _HardExit(code)

    monkeypatch.setattr(worker.os, "_exit", fake_exit)
    monkeypatch.setattr(worker, "HF_REPO", "owner/run-dataset")
    monkeypatch.setattr(worker, "hf_prefix", lambda: "seed0")
    monkeypatch.setattr(worker, "gpu_diagnostics", lambda **k: {})
    monkeypatch.setattr(worker, "error_artifact_name", lambda *a, **k: "error.txt")
    monkeypatch.setattr(worker, "hf_upload_file", lambda *a, **k: None)
    monkeypatch.setattr(worker, "wandb_finish", lambda *a, **k: None)
    monkeypatch.setattr(worker.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(worker, "heartbeat", lambda *a, **k: hb.append((a, k)))

    done_marker = tmp_path / "DONE"
    done_marker.write_text("")

    def fake_download(*, repo_id, repo_type, filename, token=None):
        if filename.endswith("/DONE"):
            return str(done_marker)
        raise OSError("503 transient HF read")  # metrics.json read keeps failing

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    with pytest.raises(worker.RetriableInfraError):
        worker.main()
    assert exited["v"] is False, "must not hard-exit when the replay read failed"
    err_hbs = [k for _a, k in hb if k.get("retriable") is True]
    assert err_hbs, "the error heartbeat must be stamped retriable=True so the run reschedules"


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
