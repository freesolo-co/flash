"""Orchestrator RunPod routing: submit/cancel, retry, handle persistence, and cost flow."""

from __future__ import annotations

import io
import threading

import pytest

from flash.spec import JobSpec


def _spec(run_id="flash-1700000001-rt01", **gpu_kw) -> JobSpec:
    gpu = {"type": "RTX 4090", "max_retries": 2}
    gpu.update(gpu_kw)
    return JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "sft",
            "run_id": run_id,
            "train": {"epochs": 1, "max_examples": 8, "hf_repo": "owner/runs"},
            "gpu": gpu,
        }
    )


def _alloc(gpu="RTX 4090", rate=0.69, candidates=None):
    from flash.providers.base import Allocation, Candidate

    if candidates is None:
        candidates = (Candidate("runpod", gpu, rate, 24),)
    return Allocation(
        provider=candidates[0].provider,
        gpu=candidates[0].gpu,
        hourly_usd=candidates[0].hourly_usd,
        min_vram_gb=12,
        candidates=tuple(candidates),
    )


def _lambda_handle(instance_id="i1"):
    return {"provider": "lambda", "instance_id": instance_id, "region": "us-east-1", "name": "n"}


def _runpod_handle(endpoint_id="ep", job_id="j"):
    return {
        "provider": "runpod",
        "endpoint_id": endpoint_id,
        "endpoint_name": f"{endpoint_id}-name",
        "job_id": job_id,
    }


@pytest.fixture
def orch(monkeypatch, tmp_path):
    from flash import runner

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    return runner


def _seed_status(orch, spec):
    st = orch.RunStatus(run_id=spec.run_id, state="queued", spec=spec.to_dict())
    orch._save_status(st)
    return st


def test_runpod_allocation_routes_to_runpod_submit(orch, monkeypatch):
    from flash.providers import allocator
    from flash.providers.base import PollResult
    from flash.providers.runpod import jobs as rp_jobs

    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc())
    captured = {}

    def fake_runpod_submit(
        run_spec,
        seed,
        log=None,
        on_handle=None,
        attempt=0,
        runtime_secrets=None,
        **_,
    ):
        captured["gpu_type"] = run_spec.gpu.type
        captured["runtime_secrets"] = dict(runtime_secrets or {})
        if on_handle:
            on_handle(_runpod_handle())
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)
    spec = _spec()
    _seed_status(orch, spec)
    metrics = orch._submit_seed_supervised(
        spec,
        0,
        io.StringIO(),
        runtime_secrets={"WANDB_API_KEY": "user-wb"},
    )
    assert metrics["train_tokens"] == 4096
    assert captured["gpu_type"] == "RTX 4090"
    assert captured["runtime_secrets"] == {"WANDB_API_KEY": "user-wb"}
    remote = orch.get_status(spec.run_id).remote
    assert remote["provider"] == "runpod"
    assert remote["allocated_gpu"] == "RTX 4090"


def test_terminal_race_before_effective_spec_persistence_skips_provider(orch, monkeypatch):
    from flash.providers import allocator
    from flash.providers.base import PollResult
    from flash.providers.runpod import jobs as rp_jobs

    spec = _spec(max_retries=0)
    orch._save_status(orch.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict()))
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc())

    original_spec_with_gpu = orch._spec_with_gpu

    def cancel_after_allocation(run_spec, gpu_type):
        selected = original_spec_with_gpu(run_spec, gpu_type)
        assert orch._update(run_spec.run_id, "cancelled")
        return selected

    provider_calls = []

    def fake_runpod_submit(*args, **kwargs):
        provider_calls.append((args, kwargs))
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(orch, "_spec_with_gpu", cancel_after_allocation)
    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)

    with pytest.raises(orch._RunCancelled):
        orch._submit_seed_supervised(spec, 0, io.StringIO())

    status = orch.get_status(spec.run_id)
    assert status.state == "cancelled"
    assert status.effective_preparation is None
    assert not orch._persist_effective_worker_spec(spec)
    assert provider_calls == []


def test_cancel_waits_for_durable_provider_handle_then_tears_down(orch, monkeypatch):
    from flash.providers import allocator
    from flash.providers.base import PollResult
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs

    spec = _spec(run_id="flash-provider-handshake-cancel", max_retries=0)
    orch._save_status(orch.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict()))
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc())
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda *a, **k: None)

    resource_created = threading.Event()
    allow_handle = threading.Event()
    handle_persisted = threading.Event()
    polling = threading.Event()
    allow_poll = threading.Event()
    cancellation_started = threading.Event()
    cancellation_finished = threading.Event()
    resource_live = {"value": False}
    cancelled_handles = []
    destroyed_handles = []

    def fake_runpod_submit(run_spec, seed, *, on_handle, **kwargs):
        resource_live["value"] = True
        resource_created.set()
        assert allow_handle.wait(timeout=5)
        on_handle(_runpod_handle("ep-handshake", "job-handshake"))
        persisted_remote = orch.get_status(spec.run_id).remote
        assert persisted_remote["endpoint_id"] == "ep-handshake"
        assert persisted_remote["job_id"] == "job-handshake"
        handle_persisted.set()
        polling.set()
        assert allow_poll.wait(timeout=5)
        return PollResult(True, metrics={"train_tokens": 4096})

    def cancel_job(endpoint_id, job_id):
        cancelled_handles.append((endpoint_id, job_id))

    def delete_endpoint(endpoint_id):
        destroyed_handles.append(endpoint_id)
        resource_live["value"] = False
        return True

    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)
    monkeypatch.setattr(runpod_api, "cancel_job", cancel_job)
    monkeypatch.setattr(runpod_api, "delete_endpoint", delete_endpoint)

    submit_errors = []

    def submit():
        try:
            orch._submit_seed_supervised(spec, 0, io.StringIO())
        except Exception as exc:
            submit_errors.append(exc)

    cancel_results = []

    def cancel():
        cancellation_started.set()
        cancel_results.append(orch.cancel_run(spec.run_id))
        cancellation_finished.set()

    submit_thread = threading.Thread(target=submit)
    submit_thread.start()
    assert resource_created.wait(timeout=5)

    cancel_thread = threading.Thread(target=cancel)
    cancel_thread.start()
    assert cancellation_started.wait(timeout=5)
    cancel_thread.join(timeout=0.1)
    assert cancel_thread.is_alive()
    assert not cancellation_finished.is_set()
    assert orch.get_status(spec.run_id).remote is None
    assert resource_live["value"]

    allow_handle.set()
    assert handle_persisted.wait(timeout=5)
    assert polling.wait(timeout=5)
    assert cancellation_finished.wait(timeout=5)
    cancel_thread.join(timeout=5)

    assert not cancel_thread.is_alive()
    assert submit_thread.is_alive()
    assert cancel_results[0].state == "cancelled"
    status = orch.get_status(spec.run_id)
    assert status.state == "cancelled"
    assert status.remote["endpoint_id"] == "ep-handshake"
    assert status.remote["job_id"] == "job-handshake"
    assert cancelled_handles == [("ep-handshake", "job-handshake")]
    assert "ep-handshake" in destroyed_handles
    assert not resource_live["value"]

    allow_poll.set()
    submit_thread.join(timeout=5)
    assert not submit_thread.is_alive()
    assert len(submit_errors) == 1
    assert isinstance(submit_errors[0], orch._RunCancelled)
    assert not resource_live["value"]


def test_concurrent_supervisors_preserve_first_effective_spec_and_provider(orch, monkeypatch):
    from flash.providers import allocator
    from flash.providers.base import PollResult
    from flash.providers.runpod import jobs as rp_jobs

    spec = _spec(run_id="flash-concurrent-supervisors", max_retries=0)
    orch._save_status(orch.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict()))

    def allocate_for_supervisor(*args, **kwargs):
        gpu = "RTX 4090" if threading.current_thread().name == "supervisor-a" else "RTX 5090"
        return _alloc(gpu=gpu)

    monkeypatch.setattr(allocator, "allocate", allocate_for_supervisor)

    resource_created = threading.Event()
    allow_handle = threading.Event()
    polling = threading.Event()
    allow_poll = threading.Event()
    provider_gpus = []

    def fake_runpod_submit(run_spec, seed, *, on_handle, **kwargs):
        provider_gpus.append(run_spec.gpu.type)
        resource_created.set()
        assert allow_handle.wait(timeout=5)
        on_handle(_runpod_handle("ep-first", "job-first"))
        polling.set()
        assert allow_poll.wait(timeout=5)
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)

    results = {}

    def submit(name):
        try:
            results[name] = orch._submit_seed_supervised(spec, 0, io.StringIO())
        except Exception as exc:
            results[name] = exc

    first = threading.Thread(target=submit, args=("first",), name="supervisor-a")
    second = threading.Thread(target=submit, args=("second",), name="supervisor-b")
    first.start()
    assert resource_created.wait(timeout=5)
    second.start()
    second.join(timeout=0.1)
    assert second.is_alive()

    allow_handle.set()
    assert polling.wait(timeout=5)
    second.join(timeout=5)
    assert not second.is_alive()
    assert isinstance(results["second"], orch._RunCancelled)
    assert provider_gpus == ["RTX 4090"]

    status = orch.get_status(spec.run_id)
    worker_spec = status.effective_preparation["worker_spec"]
    assert worker_spec["gpu"]["type"] == "RTX 4090"
    assert status.remote["endpoint_id"] == "ep-first"

    allow_poll.set()
    first.join(timeout=5)
    assert not first.is_alive()
    assert results["first"]["train_tokens"] == 4096
    assert provider_gpus == ["RTX 4090"]


@pytest.mark.parametrize(
    "failure_mode",
    ["provider_exception", "provider_without_callback", "callback_persistence_exception"],
)
def test_provider_submission_paths_release_run_lock(orch, monkeypatch, failure_mode):
    from flash.providers import allocator
    from flash.providers.base import PollResult
    from flash.providers.runpod import jobs as rp_jobs
    from flash.server._locks import _deploy_lock

    spec = _spec(run_id=f"flash-lock-release-{failure_mode}", max_retries=0)
    orch._save_status(orch.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict()))
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc())

    original_update = orch._update

    def fail_remote_persistence(run_id, state, **updates):
        if failure_mode == "callback_persistence_exception" and updates.get("remote") is not None:
            raise RuntimeError("remote persistence failed")
        return original_update(run_id, state, **updates)

    def fake_runpod_submit(run_spec, seed, *, on_handle, **kwargs):
        if failure_mode == "provider_exception":
            raise RuntimeError("provider create failed")
        if failure_mode == "callback_persistence_exception":
            on_handle({"provider": "runpod", "job_id": "job-no-persist"})
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(orch, "_update", fail_remote_persistence)
    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)

    if failure_mode == "provider_without_callback":
        metrics = orch._submit_seed_supervised(spec, 0, io.StringIO())
        assert metrics["train_tokens"] == 4096
    else:
        with pytest.raises(RuntimeError, match="failed after retries"):
            orch._submit_seed_supervised(spec, 0, io.StringIO())

    lock = _deploy_lock(spec.run_id)
    assert lock.acquire(blocking=False)
    lock.release()


def test_sync_submit_persists_resolved_env_sha_before_provider_submission(orch, monkeypatch):
    import flash.envs.loader as env_loader
    from flash.providers import allocator
    from flash.providers.base import PollResult
    from flash.providers.runpod import jobs as rp_jobs
    from flash.runner import lifecycle

    resolved_sha = "a" * 40
    resolved_refs = []
    submitted = []

    def fake_resolve(parsed, *args, **kwargs):
        resolved_refs.append(parsed.canonical())
        return resolved_sha

    def fake_runpod_submit(run_spec, seed, **kwargs):
        persisted = orch.get_status(run_spec.run_id).effective_preparation["worker_spec"]
        assert persisted["environment"]["resolved_sha"] == resolved_sha
        assert persisted["gpu"]["type"] == "RTX 5090"
        assert persisted["gpu"]["network_volume"] == "flash-weights"
        submitted.append(
            {
                "resolved_sha": run_spec.environment.resolved_sha,
                "gpu_type": run_spec.gpu.type,
                "network_volume": run_spec.gpu.network_volume,
            }
        )
        return PollResult(True, metrics={"train_tokens": 4096, "wall_seconds": 1})

    monkeypatch.setattr(env_loader, "_resolve_ref_sha", fake_resolve)
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(gpu="RTX 5090"))
    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)
    monkeypatch.setattr("flash.providers._worker.upload_code", lambda *a, **k: None)
    monkeypatch.setattr(orch, "flash_code_prefix", lambda: "code/test/flash")
    monkeypatch.setattr(orch, "_persist_metrics", lambda *a, **k: 0.0)
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda spec: None)
    monkeypatch.setattr(lifecycle, "_register_checkpoints_best_effort", lambda *a, **k: None)

    public = JobSpec.from_dict(
        {
            **_spec().to_internal_dict(),
            "environment": {"id": "github:owner/repo@main:env/environment.py"},
        }
    )

    status = orch.submit_job(public)

    assert status.state == "done"
    assert resolved_refs == ["github:owner/repo@main:env/environment.py"]
    assert submitted == [
        {
            "resolved_sha": resolved_sha,
            "gpu_type": "RTX 5090",
            "network_volume": "flash-weights",
        }
    ]
    stored = orch.get_status(public.run_id)
    assert stored.spec["environment"]["resolved_sha"] == ""
    worker = stored.effective_preparation["worker_spec"]
    assert worker["environment"]["resolved_sha"] == resolved_sha
    assert worker["gpu"]["type"] == "RTX 5090"
    assert worker["gpu"]["network_volume"] == "flash-weights"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("id", "other/environment@main", id="id"),
        pytest.param("params", {"difficulty": "hard"}, id="kwargs"),
        pytest.param("id", "owner/environment@dev", id="revision"),
        pytest.param(
            "params",
            {"difficulty": "easy", "arbitrary": "value"},
            id="arbitrary-field",
        ),
        pytest.param("pip", ["dep==2"], id="pip"),
        pytest.param("secrets", ["OTHER_TOKEN"], id="secrets"),
    ],
)
def test_effective_spec_rejects_other_environment_mutations(field, value):
    from flash import runner

    public = JobSpec.from_dict(
        {
            **_spec().to_internal_dict(),
            "environment": {
                "id": "owner/environment@main",
                "params": {"difficulty": "easy"},
                "pip": ["dep==1"],
                "secrets": ["TOKEN"],
            },
        }
    )
    worker_dict = public.to_internal_dict()
    worker_dict["environment"][field] = value
    worker = JobSpec.from_dict(worker_dict)

    with pytest.raises(ValueError, match="effective preparation"):
        runner._validate_effective_spec(public, worker)


@pytest.mark.parametrize(
    "resolved_sha",
    [
        pytest.param("main", id="symbolic-ref"),
        pytest.param("not-a-commit", id="arbitrary-string"),
        pytest.param("a" * 39, id="short-hex"),
        pytest.param("g" * 40, id="non-hex"),
        pytest.param(123, id="non-string"),
    ],
)
def test_effective_spec_rejects_malformed_worker_resolved_sha(resolved_sha):
    from dataclasses import replace

    from flash import runner

    public = JobSpec.from_dict(
        {
            **_spec().to_internal_dict(),
            "environment": {"id": "github:owner/repo@main:env/environment.py"},
        }
    )
    worker = replace(
        public,
        environment=replace(public.environment, resolved_sha=resolved_sha),
    )

    with pytest.raises(ValueError, match="effective preparation"):
        runner._validate_effective_spec(public, worker)


@pytest.mark.parametrize(
    "resolved_sha",
    [
        pytest.param("", id="removal"),
        pytest.param("b" * 40, id="replacement"),
    ],
)
def test_effective_spec_rejects_changing_existing_resolved_env_sha(resolved_sha):
    from dataclasses import replace

    from flash import runner

    public = JobSpec.from_dict(
        {
            **_spec().to_internal_dict(),
            "environment": {
                "id": "github:owner/repo@main:env/environment.py",
                "resolved_sha": "a" * 40,
            },
        }
    )
    worker = replace(
        public,
        environment=replace(public.environment, resolved_sha=resolved_sha),
    )

    with pytest.raises(ValueError, match="effective preparation"):
        runner._validate_effective_spec(public, worker)


def test_runpod_cost_projection_flows_into_run_status(orch, monkeypatch):
    spec = _spec()
    _seed_status(orch, spec)
    cost = orch._persist_metrics(
        spec,
        {"train_tokens": 4096, "wall_seconds": 1800, "allocated_gpu": "RTX 4090"},
    )
    assert cost == pytest.approx(0.345)  # 0.5 hr x $0.69/hr (RTX 4090)


def test_infra_retry_walks_to_next_runpod_class_and_deletes_endpoint(orch, monkeypatch):
    from flash.providers import allocator
    from flash.providers.base import Candidate, PollResult
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs

    candidates = (
        Candidate("runpod", "L4", 0.39, 24),
        Candidate("runpod", "H100", 0.49, 48),
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    cancelled, deleted, submitted_gpus = [], [], []
    monkeypatch.setattr(runpod_api, "cancel_job", lambda e, j: cancelled.append((e, j)))
    monkeypatch.setattr(runpod_api, "delete_endpoint", lambda e: deleted.append(e) or True)

    def fake_runpod_submit(run_spec, seed, log=None, on_handle=None, attempt=0, **kwargs):
        submitted_gpus.append(run_spec.gpu.type)
        if attempt == 0:
            on_handle(_runpod_handle("ep1", "j1"))
            return PollResult(False, failure="stalled", detail="no worker progress")
        on_handle(_runpod_handle("ep2", "j2"))
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)
    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()
    metrics = orch._submit_seed_supervised(spec, 0, log)
    assert metrics["train_tokens"] == 4096
    assert submitted_gpus == ["L4", "H100"]
    assert cancelled == [("ep1", "j1")]
    assert "ep1" in deleted
    assert "walking past the cheapest class" in log.getvalue()


def test_unconfirmed_runpod_teardown_retains_handle_and_blocks_retry(orch, monkeypatch):
    from flash.providers import allocator
    from flash.providers.base import Candidate, PollResult
    from flash.providers.runpod import RunpodProvider
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs

    candidates = (
        Candidate("runpod", "L4", 0.39, 24),
        Candidate("runpod", "H100", 0.49, 48),
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))

    submitted_attempts = []
    deleted_endpoints = []
    gc_calls = []
    monkeypatch.setattr(runpod_api, "cancel_job", lambda *a, **k: None)
    monkeypatch.setattr(
        runpod_api,
        "delete_endpoint",
        lambda endpoint_id: deleted_endpoints.append(endpoint_id) or False,
    )
    monkeypatch.setattr(
        RunpodProvider,
        "gc",
        lambda self, cleanup_spec: gc_calls.append(cleanup_spec.run_id),
    )

    def fake_runpod_submit(run_spec, seed, log=None, on_handle=None, attempt=0, **kwargs):
        submitted_attempts.append(attempt)
        on_handle(_runpod_handle("ep-unconfirmed", "job-unconfirmed"))
        return PollResult(False, failure="stalled", detail="no worker progress")

    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)
    spec = _spec(run_id="flash-unconfirmed-runpod-teardown")
    _seed_status(orch, spec)
    log = io.StringIO()

    with pytest.raises(RuntimeError, match="teardown could not be confirmed"):
        orch._submit_seed_supervised(spec, 0, log)

    assert submitted_attempts == [0]
    assert deleted_endpoints
    assert set(deleted_endpoints) == {"ep-unconfirmed"}
    assert gc_calls == [spec.run_id]
    status = orch.get_status(spec.run_id)
    assert status.remote["endpoint_id"] == "ep-unconfirmed"
    assert status.remote["job_id"] == "job-unconfirmed"
    assert "teardown UNCONFIRMED" in log.getvalue()


def _oom_candidates():
    from flash.providers.base import Candidate

    return (
        Candidate("runpod", "A100", 1.0, 80),
        Candidate("runpod", "Pro6000", 2.0, 96),
        Candidate("runpod", "B200", 3.0, 180),
    )


def _run_failed_oom_sequence(orch, monkeypatch, failures, *, max_retries):
    from flash.providers import allocator
    from flash.providers.base import PollResult
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs

    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=_oom_candidates()))
    monkeypatch.setattr(runpod_api, "cancel_job", lambda e, j: None)
    monkeypatch.setattr(runpod_api, "delete_endpoint", lambda e: True)

    submitted = []
    on_last_gpu = []
    failure_iter = iter(failures)

    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, **kwargs):
        submitted.append(run_spec.gpu.type)
        on_last_gpu.append(kwargs.get("on_last_gpu", False))
        on_handle(_runpod_handle(f"ep{attempt}", f"j{attempt}"))
        failure = next(failure_iter)
        detail = "CUDA out of memory" if failure == "oom" else "x"
        return PollResult(False, failure=failure, detail=detail)

    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec(max_retries=max_retries)
    _seed_status(orch, spec)
    with pytest.raises(RuntimeError):
        orch._submit_seed_supervised(spec, 0, io.StringIO())
    return submitted, on_last_gpu


def test_oom_escalates_exactly_once_at_max_retries_1(orch, monkeypatch):
    submitted, _on_last_gpu = _run_failed_oom_sequence(
        orch,
        monkeypatch,
        ["oom", "oom"],
        max_retries=1,
    )
    assert submitted == ["A100", "Pro6000"]


def test_oom_after_infra_failure_still_escalates(orch, monkeypatch):
    submitted, _on_last_gpu = _run_failed_oom_sequence(
        orch,
        monkeypatch,
        ["no_capacity", "oom", "oom"],
        max_retries=1,
    )
    assert submitted == ["A100", "Pro6000", "B200"]


def test_infra_failure_after_oom_uses_infra_budget(orch, monkeypatch):
    submitted, on_last_gpu = _run_failed_oom_sequence(
        orch,
        monkeypatch,
        ["oom", "no_capacity", "oom"],
        max_retries=1,
    )
    assert submitted == ["A100", "Pro6000", "B200"]
    assert on_last_gpu == [False, False, True]


def test_oom_never_escalates_at_max_retries_0(orch, monkeypatch):
    submitted, _on_last_gpu = _run_failed_oom_sequence(
        orch,
        monkeypatch,
        ["oom"],
        max_retries=0,
    )
    assert submitted == ["A100"]


def test_select_candidate_escapes_failed_provider_then_walks_classes():
    """The retry picker prefers cheapest first, escapes a failed provider cross-provider on retry,
    and only walks classes within a provider once every provider has been burned."""
    from flash.providers.base import Candidate
    from flash.runner.lifecycle import _select_candidate

    cands = (
        Candidate("runpod", "H100", 0.49, 48),
        Candidate("runpod", "RTX 6000 Ada", 0.50, 48),
        Candidate("lambda", "H100", 0.50, 48),
    )
    # Attempt 0 (nothing failed): cheapest overall.
    assert _select_candidate(cands, set(), set()) is cands[0]
    # RunPod burned an infra attempt -> escape to the OTHER provider, not the next RunPod class.
    chosen = _select_candidate(cands, {"runpod"}, {("runpod", "H100")})
    assert (chosen.provider, chosen.gpu) == ("lambda", "H100")
    # Both providers burned -> fall back to the cheapest class NOT yet tried (within-provider walk).
    chosen = _select_candidate(
        cands,
        {"runpod", "lambda"},
        {("runpod", "H100"), ("lambda", "H100")},
    )
    assert (chosen.provider, chosen.gpu) == ("runpod", "RTX 6000 Ada")


def test_select_candidate_single_provider_walks_classes():
    """With only one provider configured, the picker degrades to the cheapest untried class."""
    from flash.providers.base import Candidate
    from flash.runner.lifecycle import _select_candidate

    cands = (Candidate("runpod", "L4", 0.39, 24), Candidate("runpod", "H100", 0.49, 48))
    assert _select_candidate(cands, {"runpod"}, {("runpod", "L4")}).gpu == "H100"


def test_select_candidate_single_fitting_gpu_never_breaks():
    """A large model with exactly ONE fitting class (e.g. only H200 fits a 35B run) must keep
    re-picking THAT class on every infra retry — the candidate list only ever holds *fitting*
    classes, so the walk can never escape to a card too small to hold the model, and the picker
    must still return it (not None / not raise) even after it's been tried and its provider burned."""
    from flash.providers.base import Candidate
    from flash.runner.lifecycle import _select_candidate

    only = Candidate("runpod", "H200", 4.0, 141)
    cands = (only,)
    # Attempt 0: the single class.
    assert _select_candidate(cands, set(), set()) is only
    # After it failed infra-shaped (provider burned, class tried), the next retry re-picks the SAME
    # class — there is nowhere else to walk, and the picker must not break.
    assert _select_candidate(cands, {"runpod"}, {("runpod", "H200")}) is only


def test_runpod_no_capacity_retry_escapes_to_other_provider(orch, monkeypatch):
    """Issue 7: a RunPod queue / no-capacity failure must retry on a DIFFERENT provider
    (Lambda) rather than walking to the next RunPod class while Lambda sits available."""
    from flash.providers import allocator
    from flash.providers.base import Candidate, PollResult
    from flash.providers.lambdalabs import jobs as lambda_jobs
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs

    candidates = (
        Candidate("runpod", "H100", 0.49, 48),  # cheapest -> attempt 0
        Candidate("runpod", "RTX 6000 Ada", 0.50, 48),  # next RunPod class (the WRONG retry target)
        Candidate("lambda", "H100", 0.50, 48),  # the right cross-provider escape
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    monkeypatch.setattr(runpod_api, "cancel_job", lambda e, j: None)
    monkeypatch.setattr(runpod_api, "delete_endpoint", lambda e: True)

    rp_gpus, lam_gpus = [], []

    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, **kw):
        rp_gpus.append(run_spec.gpu.type)
        on_handle(_runpod_handle("ep1", "j1"))
        return PollResult(
            False, failure="no_capacity", detail="job stuck IN_QUEUE (no RunPod capacity)"
        )

    def fake_lam(spec, seed, log=None, on_handle=None, attempt=0, **kw):
        lam_gpus.append(spec.gpu.type)
        if on_handle:
            on_handle(_lambda_handle())
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    monkeypatch.setattr(lambda_jobs, "submit_run_lambda", fake_lam)
    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()
    metrics = orch._submit_seed_supervised(spec, 0, log)
    assert metrics["train_tokens"] == 4096
    assert rp_gpus == ["H100"]  # RunPod tried exactly once...
    assert lam_gpus == ["H100"]  # ...then the retry escaped cross-provider to Lambda
    assert orch.get_status(spec.run_id).remote["provider"] == "lambda"
    assert "walking past the cheapest class" in log.getvalue()


def test_auto_cache_run_gets_cacheless_fallback_at_zero_retries(orch, monkeypatch):
    """The platform auto-attaches the SHARED weight cache, so its endpoint-pinning DC-set
    restriction must not cost the user a GPU-walk retry: a no_capacity on the cached spec earns ONE
    extra, cache-less cross-region attempt even at max_retries=0 (else the auto-cache could fail a
    run a cache-less launch would have won). Regression for the cache-fallback-vs-retry-budget gap.
    """
    from flash.providers import allocator
    from flash.providers.base import Candidate, PollResult
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs
    from flash.runner import WEIGHT_CACHE_VOLUME_NAME

    monkeypatch.setattr(
        allocator,
        "allocate",
        lambda *a, **k: _alloc(candidates=(Candidate("runpod", "H100", 0.49, 48),)),
    )
    monkeypatch.setattr(runpod_api, "cancel_job", lambda e, j: None)
    monkeypatch.setattr(runpod_api, "delete_endpoint", lambda e: True)

    volumes_seen = []

    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, **kw):
        vol = getattr(run_spec.gpu, "network_volume", None)
        volumes_seen.append(vol)
        if on_handle:
            on_handle(_runpod_handle(f"ep{attempt}", f"j{attempt}"))
        # Cache-attached attempt -> no_capacity (the cache's DC set is starved); the cache-less
        # fallback attempt -> success.
        if vol == WEIGHT_CACHE_VOLUME_NAME:
            return PollResult(
                False, failure="no_capacity", detail="IN_QUEUE (cache DC set starved)"
            )
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec(max_retries=0, network_volume=WEIGHT_CACHE_VOLUME_NAME, network_volume_gb=100)
    _seed_status(orch, spec)
    metrics = orch._submit_seed_supervised(spec, 0, io.StringIO())
    assert metrics["train_tokens"] == 4096
    # Exactly two attempts: the cache-attached one (no_capacity) then the bonus cache-less retry.
    assert volumes_seen == [WEIGHT_CACHE_VOLUME_NAME, None]


def test_cache_fallback_does_not_consume_gpu_walk_retry(orch, monkeypatch):
    """The cache-drop fallback is a FREE attempt — it must not spend the user's GPU-walk retry budget.
    With max_retries=1 and both the cache attempt AND the cache-less same-class fallback hitting
    no_capacity, the run must still walk to a DIFFERENT class (its one real retry), not stop after the
    cache drop. Regression for the cache-fallback-steals-the-only-retry gap (the stop check counted the
    cache-drop attempt against max_retries)."""
    from flash.providers import allocator
    from flash.providers.base import Candidate, PollResult
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs
    from flash.runner import WEIGHT_CACHE_VOLUME_NAME

    candidates = (
        Candidate(
            "runpod", "H100", 0.49, 48
        ),  # cheapest -> cache attempt, then cache-less same class
        Candidate(
            "runpod", "RTX 6000 Ada", 0.50, 48
        ),  # the GPU-walk target the real retry must reach
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    monkeypatch.setattr(runpod_api, "cancel_job", lambda e, j: None)
    monkeypatch.setattr(runpod_api, "delete_endpoint", lambda e: True)

    seen = []

    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, **kw):
        vol = getattr(run_spec.gpu, "network_volume", None)
        seen.append((run_spec.gpu.type, vol))
        if on_handle:
            on_handle(_runpod_handle(f"ep{attempt}", f"j{attempt}"))
        # Cache attempt AND its cache-less same-class fallback both starve; only the walk to the OTHER
        # class (the genuine retry that must survive the cache drop) succeeds.
        if run_spec.gpu.type == "RTX 6000 Ada":
            return PollResult(True, metrics={"train_tokens": 4096})
        return PollResult(False, failure="no_capacity", detail="IN_QUEUE (no capacity)")

    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec(max_retries=1, network_volume=WEIGHT_CACHE_VOLUME_NAME, network_volume_gb=100)
    _seed_status(orch, spec)
    metrics = orch._submit_seed_supervised(spec, 0, io.StringIO())
    assert metrics["train_tokens"] == 4096
    # cache attempt -> cache-less SAME class (free fallback) -> GPU-walk to the OTHER class (real retry).
    assert seen == [
        ("H100", WEIGHT_CACHE_VOLUME_NAME),
        ("H100", None),
        ("RTX 6000 Ada", None),
    ]


def test_broken_gpu_preempt_retries_on_other_provider(orch, monkeypatch):
    """Issue 5: a Lambda instance whose CUDA never inits fails job_preempted; the retry must move to
    a DIFFERENT provider (RunPod) instead of re-rolling another broken Lambda instance, and the sick
    instance is torn down before the retry."""
    from flash.providers import allocator
    from flash.providers.base import Candidate, PollResult
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs as lambda_jobs
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs

    candidates = (
        Candidate("lambda", "A10", 0.40, 24),  # cheapest -> attempt 0 (lands on broken instance)
        Candidate("lambda", "H100", 0.45, 48),  # next class on the SAME (sick) provider
        Candidate("runpod", "H100", 0.49, 48),  # the right cross-provider escape
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    terminated = []
    monkeypatch.setattr(
        lambda_api, "terminate_instances", lambda ids: terminated.extend(ids) or list(ids)
    )
    monkeypatch.setattr(runpod_api, "delete_endpoint", lambda e: True)

    lam_gpus, rp_gpus = [], []

    def fake_lam(spec, seed, log=None, on_handle=None, attempt=0, **kw):
        lam_gpus.append(spec.gpu.type)
        if on_handle:
            on_handle(_lambda_handle("i-broken"))
        return PollResult(
            False,
            failure="job_preempted",
            detail="GPU never became ready after 12 tries: cuda not available",
        )

    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, **kw):
        rp_gpus.append(run_spec.gpu.type)
        on_handle(_runpod_handle("ep2", "j2"))
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(lambda_jobs, "submit_run_lambda", fake_lam)
    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()
    metrics = orch._submit_seed_supervised(spec, 0, log)
    assert metrics["train_tokens"] == 4096
    assert lam_gpus == ["A10"]  # broken Lambda instance tried once...
    assert rp_gpus == ["H100"]  # ...then escaped cross-provider to RunPod
    assert "i-broken" in terminated  # sick instance torn down before the retry
    assert orch.get_status(spec.run_id).remote["provider"] == "runpod"


def test_no_liveness_stalled_escapes_to_other_provider(orch, monkeypatch):
    """The new fast first-liveness failover returns failure='stalled' (a sick region where the box
    reached 'active' but the worker never booted). It is infra-shaped, so the retry ESCAPES to a
    different provider rather than re-rolling the same sick substrate — the observed us-east-1 /
    CANADA-1 case, now caught in ~15 min instead of ~50."""
    from flash.providers import allocator
    from flash.providers.base import Candidate, PollResult
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs as lambda_jobs
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs

    candidates = (
        Candidate("lambda", "H100", 0.45, 48),  # cheapest -> attempt 0 (sick region)
        Candidate("runpod", "H100", 0.49, 48),  # the cross-provider escape
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    monkeypatch.setattr(lambda_api, "terminate_instances", lambda ids: list(ids))
    monkeypatch.setattr(runpod_api, "delete_endpoint", lambda e: True)

    lam_gpus, rp_gpus = [], []

    def fake_lam(spec, seed, log=None, on_handle=None, attempt=0, **kw):
        lam_gpus.append(spec.gpu.type)
        if on_handle:
            on_handle(_lambda_handle())
        return PollResult(
            False,
            failure="stalled",
            detail="no worker liveness (boot.log/heartbeat) within 900s of instance active",
        )

    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, **kw):
        rp_gpus.append(run_spec.gpu.type)
        on_handle(_runpod_handle("ep3", "j3"))
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(lambda_jobs, "submit_run_lambda", fake_lam)
    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec()
    _seed_status(orch, spec)
    metrics = orch._submit_seed_supervised(spec, 0, io.StringIO())
    assert metrics["train_tokens"] == 4096
    assert lam_gpus == ["H100"]  # sick region tried once...
    assert rp_gpus == ["H100"]  # ...then escaped cross-provider to RunPod
    assert orch.get_status(spec.run_id).remote["provider"] == "runpod"


def test_genuine_worker_error_does_not_retry(orch, monkeypatch):
    from flash.providers import allocator
    from flash.providers.base import PollResult
    from flash.providers.runpod import jobs as rp_jobs

    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc())
    calls = []

    def fake_runpod_submit(run_spec, seed, log=None, on_handle=None, attempt=0, **kwargs):
        calls.append(attempt)
        return PollResult(False, failure="job_failed", detail="ValueError: bad reward fn")

    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)
    spec = _spec()
    _seed_status(orch, spec)
    with pytest.raises(RuntimeError, match="bad reward fn"):
        orch._submit_seed_supervised(spec, 0, io.StringIO())
    assert calls == [0]


def test_cancel_legacy_handle_defaults_to_runpod(orch, monkeypatch):
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import train as rp_train

    cancelled_jobs, deleted_eps = [], []
    monkeypatch.setattr(runpod_api, "cancel_job", lambda e, j: cancelled_jobs.append((e, j)))
    monkeypatch.setattr(runpod_api, "delete_endpoint", lambda e: deleted_eps.append(e) or True)
    monkeypatch.setattr(rp_train, "terminate_endpoint", lambda *a, **k: [])
    spec = _spec()
    st = _seed_status(orch, spec)
    st.state = "running"
    st.remote = {"endpoint_id": "ep1", "endpoint_name": "n", "job_id": "j1"}
    orch._save_status(st)
    out = orch.cancel_run(spec.run_id)
    assert out.state == "cancelled"
    assert cancelled_jobs == [("ep1", "j1")]
    assert "ep1" in deleted_eps


def test_config_gpu_fields(monkeypatch):
    from flash.schema import spec_from_dict

    base = {
        "model": "Qwen/Qwen3.5-0.8B",
        "algorithm": "sft",
        "train": {"epochs": 1, "max_examples": 8, "hf_repo": "owner/runs"},
        "environment": {"id": "github:owner/repo@main:env/environment.py"},
    }
    spec = spec_from_dict(dict(base), run_id="x")
    assert spec.gpu.type == "RTX 4090"
    again = JobSpec.from_dict(spec.to_dict())
    assert again.gpu.type == "RTX 4090"
    spec = spec_from_dict({**base, "gpu": {"type": "A100 SXM"}}, run_id="x")
    assert spec.gpu.type == "RTX 4090"
