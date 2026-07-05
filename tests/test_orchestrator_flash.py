"""Orchestrator end-to-end path with the Flash submission mocked (no RunPod calls)."""

from __future__ import annotations

import importlib
import json
import os
import tempfile


def test_run_job_persists_flash_metrics(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        import flash.providers.runpod.train as flash_train
        import flash.runner as runner

        importlib.reload(flash_train)
        importlib.reload(runner)
        # Storage roots are fixed constants now; redirect via monkeypatch (auto-restored).
        monkeypatch.setattr(runner, "RUNS_DIR", os.path.join(tmp, "runs"))
        monkeypatch.setattr(runner, "RESULTS_DIR", os.path.join(tmp, "results"))
        # _run_job_inner uploads the run code before training; stub it (no HF).
        monkeypatch.setattr("flash.providers._worker.upload_code", lambda repo=None, **_: "mock/repo")
        from flash.spec import GpuSpec, JobSpec, TrainSpec

        captured = {}

        def fake_submit(spec, seed, log=None, **kwargs):
            captured["gpu"] = spec.gpu.type
            captured["seed"] = seed
            return {
                "arm": "runpod",
                "phase": spec.phase,
                "seed": seed,
                "wall_seconds": 3600.0,
                "trained_eval_acc": 0.7,
                "base_eval_acc": 0.3,
                "cost_usd": 0.0,
                "notes": {},
            }

        # Stub the submit/poll path (the seam that used to be the in-process
        # offline shortcut) so the run completes without provisioning a GPU.
        # _run_training resolves it via `from flash.runner import _submit_seed_supervised`.
        monkeypatch.setattr(runner, "_submit_seed_supervised", fake_submit)

        spec = JobSpec(
            run_id="flash-run",
            model="Qwen/Qwen3.5-4B",
            algorithm="grpo",
            train=TrainSpec(steps=2),
            gpu=GpuSpec(type="RTX 4090"),
        )
        status = runner.submit_job(spec, dry_run=False, background=False)

        assert status.state == "done", status.error
        from flash.providers.runpod.pricing import hourly_rate

        # We charge the QUOTE (flash.cost estimate); the measured 1h-on-a-4090 cost lands in metrics.json.
        assert status.cost_usd == runner.charge_usd_for_spec(spec), status.cost_usd
        assert captured["gpu"] == "RTX 4090"

        # Metrics are namespaced by run id so same-phase runs cannot collide.
        metrics_path = os.path.join(tmp, "results", "runpod", "rl", status.run_id, "metrics.json")
        assert os.path.exists(metrics_path)
        with open(metrics_path) as f:
            m = json.load(f)
        assert abs(m["cost_usd"] - hourly_rate("RTX 4090")) < 1e-6  # measured: 1h on the 4090
        assert m["trained_eval_acc"] == 0.7
        assert m["notes"]["runpod_gpu"] == "RTX 4090"


def test_upload_code_forces_private_on_reused_repo(monkeypatch):
    """Run artifact repos are ALWAYS private. Existing repos must be flipped private without calling
    create_repo, so reused environment repos do not consume the HF repository-creation budget."""
    import sys
    import types

    calls = {"info": [], "create": [], "settings": [], "upload": [], "marker": []}

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def repo_info(self, **kw):
            calls["info"].append(kw)
            return types.SimpleNamespace(private=False)

        def create_repo(self, repo, **kw):
            calls["create"].append((repo, kw))

        def update_repo_settings(self, **kw):
            calls["settings"].append(kw)

        def file_exists(self, **kw):
            return False

        def upload_folder(self, **kw):
            calls["upload"].append(kw)

        def upload_file(self, **kw):
            calls["marker"].append(kw)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = _FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    import flash.providers.runpod.train as flash_train

    assert flash_train.upload_code("owner/run-artifacts") == "owner/run-artifacts"

    assert calls["info"], "repo_info should verify whether the repo already exists"
    assert calls["create"] == [], "existing repos must not hit the repository-creation endpoint"
    assert calls["settings"], "update_repo_settings was not called — reused public repo can leak"
    assert calls["settings"][0].get("private") is True
    assert calls["settings"][0].get("repo_id") == "owner/run-artifacts"


def test_upload_code_creates_repo_only_when_missing(monkeypatch):
    import sys
    import types

    calls = {"info": [], "create": [], "settings": [], "upload": [], "marker": []}

    class _NotFound(Exception):
        pass

    _NotFound.__name__ = "RepositoryNotFoundError"

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def repo_info(self, **kw):
            calls["info"].append(kw)
            if len(calls["info"]) == 1:
                raise _NotFound("missing")
            return types.SimpleNamespace(private=True)

        def create_repo(self, repo, **kw):
            calls["create"].append((repo, kw))

        def update_repo_settings(self, **kw):
            calls["settings"].append(kw)

        def file_exists(self, **kw):
            return False

        def upload_folder(self, **kw):
            calls["upload"].append(kw)

        def upload_file(self, **kw):
            calls["marker"].append(kw)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = _FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    import flash.providers.runpod.train as flash_train

    flash_train.upload_code("owner/new-env-artifacts")
    flash_train.upload_code("owner/new-env-artifacts")

    assert len(calls["info"]) == 2
    assert len(calls["create"]) == 1
    assert calls["create"][0][1]["private"] is True
    assert len(calls["settings"]) == 2
    assert len(calls["upload"]) == 2
    assert len(calls["marker"]) == 2


def test_upload_code_rechecks_privacy_on_each_submit(monkeypatch):
    import sys
    import types

    calls = {"info": [], "settings": [], "upload": [], "marker": []}

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def repo_info(self, **kw):
            calls["info"].append(kw)
            return types.SimpleNamespace(private=len(calls["info"]) == 1)

        def create_repo(self, repo, **kw):
            raise AssertionError("existing repo should not be created")

        def update_repo_settings(self, **kw):
            calls["settings"].append(kw)

        def file_exists(self, **kw):
            return False

        def upload_folder(self, **kw):
            calls["upload"].append(kw)

        def upload_file(self, **kw):
            calls["marker"].append(kw)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = _FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    import flash.providers.runpod.train as flash_train

    flash_train.upload_code("owner/rechecked-env-artifacts")
    flash_train.upload_code("owner/rechecked-env-artifacts")

    assert len(calls["info"]) == 2
    assert len(calls["settings"]) == 2
    assert len(calls["upload"]) == 2
    assert len(calls["marker"]) == 2


def test_upload_code_retries_transient_repo_settings(monkeypatch):
    import sys
    import types

    calls = {"settings": 0, "upload": [], "marker": []}

    class _Response:
        status_code = 504

    class _Transient(Exception):
        response = _Response()

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def repo_info(self, **kw):
            return types.SimpleNamespace(private=True)

        def create_repo(self, repo, **kw):
            raise AssertionError("existing repo should not be created")

        def update_repo_settings(self, **kw):
            calls["settings"] += 1
            if calls["settings"] == 1:
                raise _Transient("gateway timeout")

        def file_exists(self, **kw):
            return False

        def upload_folder(self, **kw):
            calls["upload"].append(kw)

        def upload_file(self, **kw):
            calls["marker"].append(kw)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = _FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    import flash.providers.runpod.train as flash_train

    monkeypatch.setattr("flash.providers._worker.time.sleep", lambda _delay: None)

    flash_train.upload_code("owner/transient-settings")

    assert calls["settings"] == 2
    assert calls["upload"]
    assert calls["marker"]


def test_hf_call_honors_retry_after(monkeypatch):
    import flash.providers.runpod.train as flash_train

    sleeps: list[float] = []
    logs: list[tuple[str, tuple]] = []

    class _Response:
        status_code = 429

        def __init__(self) -> None:
            self.headers = {"Retry-After": "17"}

    class _RateLimited(Exception):
        response = _Response()

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _RateLimited("slow down")
        return "ok"

    monkeypatch.setattr("flash.providers._worker.time.sleep", sleeps.append)
    monkeypatch.setattr(flash_train.logger, "warning", lambda msg, *args: logs.append((msg, args)))

    assert flash_train._hf_call(flaky, "upload") == "ok"
    assert sleeps == [17.0]
    assert logs


def test_hf_call_caps_http_date_retry_after(monkeypatch):
    import flash.providers.runpod.train as flash_train

    sleeps: list[float] = []

    class _Response:
        status_code = 429

        def __init__(self) -> None:
            self.headers = {"Retry-After": "Wed, 21 Oct 2099 07:28:00 GMT"}

    class _RateLimited(Exception):
        response = _Response()

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _RateLimited("slow down")
        return "ok"

    monkeypatch.setattr("flash.providers._worker.time.sleep", sleeps.append)
    monkeypatch.setattr(flash_train.logger, "warning", lambda *_args: None)

    assert flash_train._hf_call(flaky, "upload") == "ok"
    assert sleeps == [60.0]


def test_upload_code_uses_content_addressed_prefix(monkeypatch):
    """Code uploads are additive under immutable content-addressed prefixes, so different package
    snapshots cannot overwrite or delete one another inside the shared environment repo."""
    import os
    import re
    import sys
    import types

    import flash

    calls = {"upload": [], "marker": []}

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def repo_info(self, **kw):
            pass

        def create_repo(self, repo, **kw):
            pass

        def update_repo_settings(self, **kw):
            pass

        def file_exists(self, **kw):
            return False

        def upload_folder(self, **kw):
            calls["upload"].append(kw)

        def upload_file(self, **kw):
            calls["marker"].append(kw)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = _FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    import flash.providers.runpod.train as flash_train

    flash_train.upload_code("owner/run-artifacts")
    assert calls["upload"], "upload_folder was not called"
    up = calls["upload"][0]
    assert re.fullmatch(r"code/[0-9a-f]{32}/flash", up["path_in_repo"])
    assert "delete_patterns" not in up
    # still uploads from the real (symlink-collapsed) package dir, and skips bytecode
    assert up["folder_path"] == os.path.realpath(os.path.dirname(os.path.abspath(flash.__file__)))
    assert "*.pyc" in up.get("ignore_patterns", [])
    assert "*.pyo" in up.get("ignore_patterns", [])
    assert (
        calls["marker"][0]["path_in_repo"] == f"{up['path_in_repo']}/.flash-code-snapshot-complete"
    )


def test_upload_code_skips_existing_content_prefix(monkeypatch):
    import sys
    import types

    calls = {"file_exists": [], "upload": []}

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def repo_info(self, **kw):
            return types.SimpleNamespace(private=True)

        def create_repo(self, repo, **kw):
            raise AssertionError("existing repo should not be created")

        def update_repo_settings(self, **kw):
            pass

        def file_exists(self, **kw):
            calls["file_exists"].append(kw)
            return True

        def upload_folder(self, **kw):
            calls["upload"].append(kw)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = _FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    import flash.providers.runpod.train as flash_train

    prefix = "code/0123456789abcdef0123456789abcdef/flash"
    flash_train.upload_code("owner/run-artifacts", code_prefix=prefix)

    assert calls["file_exists"] == [
        {
            "repo_id": "owner/run-artifacts",
            "filename": f"{prefix}/.flash-code-snapshot-complete",
            "repo_type": "dataset",
        }
    ]
    assert calls["upload"] == []


def test_upload_code_reuploads_when_completion_marker_missing(monkeypatch):
    import sys
    import types

    calls = {"file_exists": [], "upload": [], "marker": []}

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def repo_info(self, **kw):
            return types.SimpleNamespace(private=True)

        def create_repo(self, repo, **kw):
            raise AssertionError("existing repo should not be created")

        def update_repo_settings(self, **kw):
            pass

        def file_exists(self, **kw):
            calls["file_exists"].append(kw)
            return False

        def upload_folder(self, **kw):
            calls["upload"].append(kw)

        def upload_file(self, **kw):
            calls["marker"].append(kw)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = _FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    import flash.providers.runpod.train as flash_train

    prefix = "code/0123456789abcdef0123456789abcdef/flash"
    flash_train.upload_code("owner/run-artifacts", code_prefix=prefix)

    assert calls["file_exists"] == [
        {
            "repo_id": "owner/run-artifacts",
            "filename": f"{prefix}/.flash-code-snapshot-complete",
            "repo_type": "dataset",
        }
    ]
    assert calls["upload"], "missing completion marker must force a fresh folder upload"
    assert calls["marker"][0]["path_in_repo"] == f"{prefix}/.flash-code-snapshot-complete"


def test_run_job_background_swallows_exception(monkeypatch):
    """The daemon-thread entrypoint must NOT let _run_job's exception escape (it would surface as
    an alarming 'Exception in thread' traceback for every failed run); the synchronous _run_job
    keeps raising for its callers. Terminal state is already persisted before the raise, so the
    wrapper just logs and returns."""
    import flash.runner as runner

    calls = {"n": 0}

    def boom(spec):
        calls["n"] += 1
        raise RuntimeError("seed 0 failed after retries: job_failed")

    # The wrapper dispatches through the package-level _run_job that tests patch.
    monkeypatch.setattr(runner, "_resolve_init_from_adapter", lambda spec: spec)
    monkeypatch.setattr(runner, "_run_job", boom)
    spec = type("S", (), {"run_id": "bg-run"})()
    # Must not raise — the wrapper swallows it (state already persisted by _run_job_inner).
    runner._run_job_background(spec)
    assert calls["n"] == 1


def test_run_job_background_persists_failed_when_not_yet_terminal(monkeypatch):
    """If _run_job crashes BEFORE _run_job_inner persisted a terminal state (e.g. an import/resolve
    error), the daemon wrapper must still record a terminal `failed` (via terminal-sticky _update) so
    the run doesn't hang non-terminal forever."""
    import os
    import tempfile

    import flash.runner as runner
    from flash.runner import RunStatus

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner, "RUNS_DIR", os.path.join(tmp, "runs"))
        monkeypatch.setattr(runner, "RESULTS_DIR", os.path.join(tmp, "results"))
        os.makedirs(runner.RUNS_DIR, exist_ok=True)
        # a non-terminal (queued) run, as submit_job persists before dispatching the daemon thread
        runner._save_status(RunStatus(run_id="bg-fail", state="queued", spec={}))

        def boom(spec):
            raise RuntimeError("crashed before persisting terminal state")

        monkeypatch.setattr(runner, "_resolve_init_from_adapter", lambda spec: spec)
        monkeypatch.setattr(runner, "_run_job", boom)
        spec = type("S", (), {"run_id": "bg-fail"})()
        runner._run_job_background(spec)  # must not raise

        status = runner.get_status("bg-fail")
        assert status.state == "failed"
        assert "crashed before persisting" in (status.error or "")


def test_run_job_background_does_not_clobber_persisted_failure(monkeypatch):
    """If the run is ALREADY terminal (e.g. _run_job_inner persisted the real, detailed failure
    before the re-raise), the wrapper must NOT overwrite its error with the caught exception."""
    import os
    import tempfile

    import flash.runner as runner
    from flash.runner import RunStatus

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner, "RUNS_DIR", os.path.join(tmp, "runs"))
        monkeypatch.setattr(runner, "RESULTS_DIR", os.path.join(tmp, "results"))
        os.makedirs(runner.RUNS_DIR, exist_ok=True)
        # already terminal, carrying the REAL failure detail _run_job_inner persisted
        runner._save_status(
            RunStatus(run_id="bg-done", state="failed", spec={}, error="real seed failure detail")
        )

        def boom(spec):
            raise RuntimeError("generic wrapper-level error")

        monkeypatch.setattr(runner, "_resolve_init_from_adapter", lambda spec: spec)
        monkeypatch.setattr(runner, "_run_job", boom)
        spec = type("S", (), {"run_id": "bg-done"})()
        runner._run_job_background(spec)  # must not raise

        status = runner.get_status("bg-done")
        assert status.state == "failed"
        assert status.error == "real seed failure detail"  # NOT clobbered by the wrapper
