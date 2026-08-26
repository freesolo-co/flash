"""regression coverage for hard-exiting after successful worker completion.

After the rl train phase persisted its adapter and metadata, interpreter shutdown could hang in
colocated vllm nccl or cuda teardown. The blocking parent process then could not advance to the
next phase before the fixed work deadline. ``worker.main`` bypasses that teardown by calling
``os._exit(0)`` only after the handler completes and its artifacts are durable.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import flash.engine.worker.entry.opd as opd_entry
import flash.engine.worker.entry.sft as sft_entry
import flash.engine.worker.entry.worker as worker
import flash.engine.worker.io.progress as progress_io
import flash.engine.worker.perf as worker_perf
import flash.engine.worker.runtime.state as worker_state
from flash.engine.support.worker_entrypoint import WORKER_FAILURE_LINE


class _HardExit(BaseException):
    """Stand-in for os._exit so the test process survives; BaseException so it propagates
    past the worker's ``except Exception`` block (mirroring a real os._exit)."""

    def __init__(self, code):
        self.code = code


def _patch_common(monkeypatch, fake_exit):
    monkeypatch.setattr(worker.os, "_exit", fake_exit)
    monkeypatch.setattr(worker_state, "HF_REPO", "")  # keep artifact transport out of this test
    monkeypatch.setattr(worker_state, "RUN_MODE", "sft")
    monkeypatch.setattr(progress_io, "publish_progress", lambda *a, **k: None)
    monkeypatch.setattr(worker.result_io, "publish_result", lambda **_kwargs: None)
    monkeypatch.setattr(worker.time, "sleep", lambda *a, **k: None)
    # main() runs the real boot steps before the handler; this test exercises the hard-exit flow,
    # so stub out the Hopper fla fast-path setup.
    monkeypatch.setattr(worker_perf, "_ensure_fla_fastpath_on_hopper", lambda: None)


def _run_safe_entrypoint(tmp_path, sitecustomize):
    (tmp_path / "sitecustomize.py").write_text(sitecustomize)
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(repo_root), env.get("PYTHONPATH", "")])
    return subprocess.run(
        [sys.executable, "-m", "flash.engine.support.worker_entrypoint"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_safe_entrypoint_failure(result, secret):
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert combined.strip() == WORKER_FAILURE_LINE
    assert secret not in combined
    assert "Traceback" not in combined


def test_managed_entrypoint_emits_one_safe_failure_line(tmp_path):
    secret = "managed-private-provider-response"
    result = _run_safe_entrypoint(
        tmp_path,
        "import flash.engine.worker.entry.worker as worker\n"
        "def fail():\n"
        f"    raise RuntimeError({secret!r})\n"
        "worker.main = fail\n",
    )
    _assert_safe_entrypoint_failure(result, secret)


def test_managed_entrypoint_sanitizes_worker_import_failure(tmp_path):
    secret = "managed-private-import-response"
    result = _run_safe_entrypoint(
        tmp_path,
        "import importlib.abc\n"
        "import sys\n"
        "class FailWorkerImport(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path, target=None):\n"
        "        if fullname == 'flash.engine.worker':\n"
        f"            raise RuntimeError({secret!r})\n"
        "        return None\n"
        "sys.meta_path.insert(0, FailWorkerImport())\n",
    )
    _assert_safe_entrypoint_failure(result, secret)


def test_direct_worker_module_emits_one_normal_traceback(tmp_path):
    secret = "direct-worker-diagnostic"
    (tmp_path / "sitecustomize.py").write_text(
        "import flash.engine.worker.entry.worker as worker\n"
        "def fail():\n"
        f"    raise RuntimeError({secret!r})\n"
        "worker.main = fail\n"
    )
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(repo_root), env.get("PYTHONPATH", "")])
    result = subprocess.run(
        [sys.executable, "-m", "flash.engine.worker"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert combined.count("Traceback") == 1
    assert secret in combined


def test_existing_current_fence_result_skips_gpu_setup_and_handler(monkeypatch):
    def fake_exit(code=0):
        raise _HardExit(code)

    _patch_common(monkeypatch, fake_exit)
    monkeypatch.setattr(worker.result_io, "preflight_existing_terminal_result", object)
    monkeypatch.setattr(
        worker,
        "_preflight_gpu_occupancy_for_spec",
        lambda: (_ for _ in ()).throw(AssertionError("gpu setup must be skipped")),
    )
    monkeypatch.setattr(
        sft_entry,
        "run_sft",
        lambda: (_ for _ in ()).throw(AssertionError("handler must be skipped")),
    )

    with pytest.raises(_HardExit) as exc_info:
        worker.main()

    assert exc_info.value.code == 0


def test_transient_result_preflight_prevents_training_as_artifact_transport(monkeypatch):
    _patch_common(monkeypatch, lambda _code=0: None)
    error = worker_perf.RetriableInfraError("result listing unavailable")
    monkeypatch.setattr(
        worker.result_io,
        "preflight_existing_terminal_result",
        lambda: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(
        worker,
        "_preflight_gpu_occupancy_for_spec",
        lambda: (_ for _ in ()).throw(AssertionError("gpu setup must be skipped")),
    )
    monkeypatch.setattr(
        sft_entry,
        "run_sft",
        lambda: (_ for _ in ()).throw(AssertionError("handler must be skipped")),
    )
    published = []
    monkeypatch.setattr(
        worker.result_io,
        "publish_result",
        lambda **kwargs: published.append(kwargs),
    )
    monkeypatch.setattr(worker.hf_io, "hf_upload_file", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(worker.backend_common, "collect_ray_failure_logs", lambda **_kwargs: "")
    monkeypatch.setattr(worker.worker_perf, "gpu_diagnostics", dict)

    with pytest.raises(worker_perf.RetriableInfraError, match="result listing unavailable"):
        worker.main()

    assert published[0]["failure_class"] == "artifact_transport"
    assert published[0]["training_entered"] is False


def test_worker_hard_exits_zero_on_success(monkeypatch):
    ran = {"v": False}

    def fake_exit(code=0):
        raise _HardExit(code)

    _patch_common(monkeypatch, fake_exit)
    monkeypatch.setattr(sft_entry, "run_sft", lambda: ran.__setitem__("v", True))

    raised = None
    try:
        worker.main()
    except _HardExit as e:
        raised = e
    assert ran["v"] is True, "handler must run"
    assert raised is not None, "worker.main must hard-exit on success"
    assert raised.code == 0, "must exit 0 on success"


def test_worker_dispatches_sft_adapter_continuation_to_the_handler(monkeypatch):
    """A warm-started SFT run must reach run_sft, not be refused at dispatch.

    ``sft_train._warmstart_adapter_path`` downloads and validates the source adapter and
    ``sft_train_runner`` hands it to verl as ``model.lora_adapter_path``; the handler has to be
    entered for any of that to happen.
    """
    from flash.core.spec import JobSpec, TrainSpec

    ran = {"v": False}

    def fake_exit(code=0):
        raise _HardExit(code)

    _patch_common(monkeypatch, fake_exit)
    monkeypatch.setattr(
        worker_state,
        "JOB_SPEC",
        JobSpec(
            algorithm="sft",
            train=TrainSpec(init_from_adapter="owner/runs:sft/source-run"),
        ),
    )
    monkeypatch.setattr(sft_entry, "run_sft", lambda: ran.__setitem__("v", True))

    raised = None
    try:
        worker.main()
    except _HardExit as e:
        raised = e
    assert ran["v"] is True, "a warm-started sft run must invoke run_sft"
    assert raised is not None
    assert raised.code == 0


def test_worker_dispatches_opd_run_mode(monkeypatch):
    """RUN_MODE=='opd' must dispatch to run_opd (the third algorithm's worker handler)."""
    ran = {"v": False}

    def fake_exit(code=0):
        raise _HardExit(code)

    _patch_common(monkeypatch, fake_exit)
    monkeypatch.setattr(worker_state, "RUN_MODE", "opd")
    monkeypatch.setattr(opd_entry, "run_opd", lambda: ran.__setitem__("v", True))

    raised = None
    try:
        worker.main()
    except _HardExit as e:
        raised = e
    assert ran["v"] is True, "RUN_MODE=opd must invoke run_opd"
    assert raised is not None
    assert raised.code == 0


def test_worker_does_not_hard_exit_on_failure(monkeypatch):
    called = {"exit": False}

    def fake_exit(code=0):
        called["exit"] = True
        raise _HardExit(code)

    _patch_common(monkeypatch, fake_exit)

    def boom():
        raise ValueError("train blew up")

    monkeypatch.setattr(sft_entry, "run_sft", boom)

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
