"""Orchestrator RunPod routing: submit/cancel, retry, handle persistence, and cost flow."""

from __future__ import annotations

import io
import threading

import pytest

from flash.spec import JobSpec
from tests._helpers.profile import (
    attach_sft_profile,
    record_sft_profile,
    satisfy_sft_profile,
    stub_revision_geometry,
)

_RUNPOD_FINGERPRINT = "rpk-0123456789ab"


def _spec(run_id="flash-1700000001-rt01", algorithm="sft", **gpu_kw) -> JobSpec:
    """A spec in the shape the lifecycle actually receives: post-``prepare_job``.

    Routing behaviour is algorithm-independent, but sft is the algorithm whose quote is
    profile-backed, so an sft spec has to carry the attached profile its own submit path would have
    resolved. Without it every routing test re-tests the profile gate instead of routing.
    ``algorithm`` is for the few tests whose subject is a submit-path behaviour sft no longer
    reaches; ``attach_sft_profile`` leaves those specs alone.
    """
    gpu = {"type": "RTX 4090", "max_retries": 2}
    gpu.update(gpu_kw)
    return attach_sft_profile(
        JobSpec.from_dict(
            {
                "model": "Qwen/Qwen3.5-0.8B",
                "algorithm": algorithm,
                "run_id": run_id,
                "environment": {"id": "github:owner/repo@main:env/environment.py"},
                "train": {"epochs": 1, "max_examples": 8},
                "gpu": gpu,
            }
        )
    )


def _public_spec(run_id="flash-1700000001-rt01", algorithm="sft") -> JobSpec:
    """The user-authored shape ``submit_job`` receives: env unpinned, no attached profile.

    ``_spec`` is deliberately post-``prepare_job``, which is the wrong input for a test whose
    subject IS submission -- it would arrive already carrying the pins that submission is supposed
    to resolve. Round-tripping through the public serializer drops exactly the platform-managed
    fields (env sha, the profile carrier, managed gpu policy), so this stays one definition of the
    spec instead of a second hand-written copy that could drift from it.
    """
    public = _spec(run_id=run_id, algorithm=algorithm).to_dict()
    public.pop("model_revision", None)  # authored-optional; submission resolves it
    return JobSpec.from_dict({**public, "run_id": run_id})


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


def _lambda_handle(instance_id="i1", attempt=0):
    return {
        "provider": "lambda",
        "instance_id": instance_id,
        "instance_type": "gpu_1x_a100",
        "region": "us-east-1",
        "name": "n",
        "gpu": "A100",
        "hourly_usd": 1.0,
        "attempt": attempt,
        "started_ts": float(attempt + 1),
    }


def _runpod_handle(endpoint_id="ep", job_id="j", attempt=0):
    return {
        "provider": "runpod",
        "endpoint_id": endpoint_id,
        "endpoint_name": f"{endpoint_id}-name",
        "key_fingerprint": _RUNPOD_FINGERPRINT,
        "job_id": job_id,
        "attempt": attempt,
        "started_ts": float(attempt + 1),
    }


@pytest.fixture
def orch(monkeypatch, tmp_path):
    from flash import runner

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    # _spec pins a model revision, which makes the lifecycle's post-allocation quote refresh
    # revision-aware. Left unstubbed it reaches github, and the refresh treats any failure as an
    # infra-shaped transient -- so the whole suite would sit in real retry backoff sleeps.
    stub_revision_geometry(monkeypatch)
    return runner


def _seed_status(orch, spec):
    st = orch.RunStatus(run_id=spec.run_id, state="queued", spec=spec.to_dict())
    orch._save_status(st)
    return st


def test_exact_only_preflight_rejects_unconfigured_provider_set_before_persistence(
    orch, monkeypatch
):
    import flash.providers as providers

    persisted = []
    spec = satisfy_sft_profile(orch, monkeypatch, _spec(type="H200"))
    monkeypatch.setattr(providers, "available_providers", lambda: ("lambda", "vast"))
    monkeypatch.setattr(orch, "_save_status", lambda *args, **kwargs: persisted.append(args))

    with pytest.raises(ValueError, match="no configured provider can provision"):
        orch.submit_job(spec, dry_run=True)

    assert persisted == []


def test_runpod_allocation_routes_to_runpod_submit(orch, monkeypatch):
    from flash.providers import allocator
    from flash.providers.base import PollResult
    from flash.providers.runpod import jobs as rp_jobs

    captured = {}

    def fake_allocate(*args, **kwargs):
        captured["allocate_kwargs"] = kwargs
        return _alloc()

    monkeypatch.setattr(allocator, "allocate", fake_allocate)

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
            on_handle(_runpod_handle(attempt=attempt))
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)
    spec = _spec(provider="runpod", type="RTX 4090")
    _seed_status(orch, spec)
    metrics = orch._submit_seed_supervised(
        spec,
        spec.seed,
        io.StringIO(),
        runtime_secrets={"WANDB_API_KEY": "user-wb"},
    )
    assert metrics["train_tokens"] == 4096
    assert captured["gpu_type"] == "RTX 4090"
    assert captured["runtime_secrets"] == {"WANDB_API_KEY": "user-wb"}
    assert captured["allocate_kwargs"]["provider"] == "runpod"
    assert captured["allocate_kwargs"]["gpu_type"] == "RTX 4090"
    remote = orch.get_status(spec.run_id).remote
    assert remote["provider"] == "runpod"
    assert remote["allocated_gpu"] == "RTX 4090"


def test_auto_gpu_effective_spec_is_transient_and_keeps_base_auto(orch) -> None:
    base = _spec(type="")

    first_attempt = orch._spec_with_gpu(base, "RTX 4090")
    second_attempt = orch._spec_with_gpu(base, "H100")

    assert base.gpu.type == ""
    assert first_attempt.gpu.type == "RTX 4090"
    assert second_attempt.gpu.type == "H100"
    assert first_attempt is not base
    assert second_attempt is not base


def test_effective_spec_carries_the_allocated_card_count(orch) -> None:
    base = _spec(type="")

    # the allocator can satisfy a run with n cards of a smaller class; the count it chose has to reach
    # the spec, because the worker sizes its rank count from gpu.count and the payload rents gpu.count.
    combo = orch._spec_with_gpu(base, "A100 PCIe", 4)
    assert (combo.gpu.type, combo.gpu.count) == ("A100 PCIe", 4)
    assert base.gpu.count == 1

    # omitted/zero count preserves the spec's own count (the historical single-card call shape).
    single = orch._spec_with_gpu(base, "A100 PCIe")
    assert single.gpu.count == 1


def test_terminal_race_before_effective_spec_persistence_skips_provider(orch, monkeypatch):
    from flash.providers import allocator
    from flash.providers.base import PollResult
    from flash.providers.runpod import jobs as rp_jobs

    spec = _spec(max_retries=0)
    orch._save_status(orch.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict()))
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc())

    original_spec_with_gpu = orch._spec_with_gpu

    def cancel_after_allocation(run_spec, gpu_type, gpu_count=0):
        selected = original_spec_with_gpu(run_spec, gpu_type, gpu_count)
        assert orch._update(run_spec.run_id, "cancelled")
        return selected

    provider_calls = []

    def fake_runpod_submit(*args, **kwargs):
        provider_calls.append((args, kwargs))
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(orch, "_spec_with_gpu", cancel_after_allocation)
    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)

    with pytest.raises(orch._RunCancelled):
        orch._submit_seed_supervised(spec, spec.seed, io.StringIO())

    status = orch.get_status(spec.run_id)
    assert status.state == "cancelled"
    assert status.effective_preparation is None
    assert not orch._persist_effective_worker_spec(spec)
    assert provider_calls == []


@pytest.mark.parametrize("first_revocation_fails", [False, True])
def test_cancel_waits_for_durable_provider_handle_then_tears_down(
    orch, monkeypatch, first_revocation_fails
):
    from flash.providers import allocator
    from flash.providers.base import PollResult
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs
    from flash.server import db as server_db

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
    revocation_calls = 0

    def revoke_capabilities(_run_id):
        nonlocal revocation_calls
        revocation_calls += 1
        if first_revocation_fails and revocation_calls == 1:
            raise RuntimeError("teacher revocation store unavailable")
        return 1

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

    def cancel_job(endpoint_id, job_id, **_kw):
        cancelled_handles.append((endpoint_id, job_id))

    def delete_endpoint(endpoint_id, _fingerprint):
        destroyed_handles.append(endpoint_id)
        resource_live["value"] = False
        return True

    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)
    monkeypatch.setattr(server_db, "revoke_teacher_capabilities_for_run", revoke_capabilities)
    monkeypatch.setattr(runpod_api, "cancel_job", cancel_job)
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", delete_endpoint)

    submit_errors = []

    def submit():
        try:
            orch._submit_seed_supervised(spec, spec.seed, io.StringIO())
        except Exception as exc:
            submit_errors.append(exc)

    cancel_results = []
    cancel_errors = []

    def cancel():
        cancellation_started.set()
        try:
            cancel_results.append(orch.cancel_run(spec.run_id))
        except Exception as exc:
            cancel_errors.append(exc)
        finally:
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
    waiting_status = orch.get_status(spec.run_id)
    assert waiting_status.state == "running"
    assert waiting_status.remote is None
    assert resource_live["value"]

    allow_handle.set()
    assert handle_persisted.wait(timeout=5)
    assert polling.wait(timeout=5)
    assert cancellation_finished.wait(timeout=5)
    cancel_thread.join(timeout=5)

    assert not cancel_thread.is_alive()
    assert submit_thread.is_alive()
    if first_revocation_fails:
        assert cancel_results == []
        assert len(cancel_errors) == 1
        assert "teacher revocation store unavailable" in str(cancel_errors[0])
    else:
        assert cancel_errors == []
        assert cancel_results[0].state == "cancelled"
    assert revocation_calls == 2
    status = orch.get_status(spec.run_id)
    assert status.state == "cancelled"
    assert status.remote is None
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
            results[name] = orch._submit_seed_supervised(spec, spec.seed, io.StringIO())
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
        metrics = orch._submit_seed_supervised(spec, spec.seed, io.StringIO())
        assert metrics["train_tokens"] == 4096
    else:
        with pytest.raises(RuntimeError, match="failed after retries"):
            orch._submit_seed_supervised(spec, spec.seed, io.StringIO())

    lock = _deploy_lock(spec.run_id)
    assert lock.acquire(blocking=False)
    lock.release()


def test_sync_submit_persists_resolved_env_sha_before_provider_submission(orch, monkeypatch):
    from dataclasses import replace

    import flash.catalog as catalog
    import flash.envs.loader as env_loader
    from flash.providers import allocator
    from flash.providers.base import PollResult
    from flash.providers.runpod import jobs as rp_jobs
    from flash.runner import lifecycle

    resolved_sha = "a" * 40
    resolved_refs = []
    quote_allocations = []
    submitted = []

    def fake_resolve(parsed, *args, **kwargs):
        resolved_refs.append(parsed.canonical())
        return resolved_sha

    resolved_model_sha = "b" * 40
    monkeypatch.setattr(
        orch,
        "_resolve_model_revision",
        lambda spec, **_kwargs: replace(spec, model_revision=resolved_model_sha),
    )
    monkeypatch.setattr(
        orch,
        "resolve_model",
        lambda model, *args, **kwargs: catalog.MODELS[model],
    )

    def fake_estimate(_spec, *, allocation=None):
        quote_allocations.append(allocation)
        total_usd = 7.0 if allocation is not None else 1.0
        return type("Estimate", (), {"total_usd": total_usd})()

    def fake_runpod_submit(run_spec, seed, **kwargs):
        status = orch.get_status(run_spec.run_id)
        persisted = status.effective_preparation["worker_spec"]
        assert status.estimated_cost_usd == 7.0
        assert quote_allocations[-1] is not None
        assert quote_allocations[-1].gpu == "RTX 5090"
        assert persisted["environment"]["resolved_sha"] == resolved_sha
        assert persisted["gpu"]["type"] == "RTX 5090"
        assert persisted["gpu"]["network_volume"] == "flash-weights"
        assert persisted["model_revision"] == resolved_model_sha
        submitted.append(
            {
                "resolved_sha": run_spec.environment.resolved_sha,
                "gpu_type": run_spec.gpu.type,
                "network_volume": run_spec.gpu.network_volume,
                "model_revision": run_spec.model_revision,
            }
        )
        return PollResult(True, metrics={"train_tokens": 4096, "wall_seconds": 1})

    monkeypatch.setattr(env_loader, "_resolve_ref_sha", fake_resolve)
    monkeypatch.setattr("flash.cost.spec.estimate_for_spec", fake_estimate)
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(gpu="RTX 5090"))
    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)
    monkeypatch.setattr("flash.providers._worker.upload_code", lambda *a, **k: None)
    monkeypatch.setattr(orch, "flash_code_prefix", lambda: "code/test/flash")
    monkeypatch.setattr(orch, "_persist_metrics", lambda *a, **k: 0.0)
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda spec: None)
    monkeypatch.setattr(lifecycle, "_register_checkpoints_best_effort", lambda *a, **k: None)

    public = _public_spec()
    # the profile job for this workload already ran, so preparation reads its record instead of
    # queueing another one. it is keyed on the two shas submission itself resolves above, so it has
    # to be recorded against those -- not against the helper's stand-ins.
    record_sft_profile(
        orch,
        replace(
            public,
            model_revision=resolved_model_sha,
            environment=replace(public.environment, resolved_sha=resolved_sha),
        ),
    )

    status = orch.submit_job(public)

    assert status.state == "done"
    assert resolved_refs == ["github:owner/repo@main:env/environment.py"]
    assert submitted == [
        {
            "resolved_sha": resolved_sha,
            "gpu_type": "RTX 5090",
            "network_volume": "flash-weights",
            "model_revision": resolved_model_sha,
        }
    ]
    stored = orch.get_status(public.run_id)
    # resolved_sha is platform-managed: it is stripped from the public spec, not surfaced empty.
    assert "resolved_sha" not in stored.spec["environment"]
    assert stored.spec["model_revision"] == resolved_model_sha
    worker = stored.effective_preparation["worker_spec"]
    assert worker["environment"]["resolved_sha"] == resolved_sha
    assert worker["gpu"]["type"] == "RTX 5090"
    assert worker["gpu"]["network_volume"] == "flash-weights"


def test_sft_submission_fails_closed_when_the_environment_cannot_be_pinned(orch, monkeypatch):
    """sft never reaches the unpinned-at-submit state the lifecycle fallback below recovers.

    Its workload profile is keyed on the immutable environment revision, so a GitHub blip at submit
    has no pin to key on and no profile can be trusted. The run must not be created at all: pricing
    it would freeze a quote against a ref that can move before the worker resolves it. grpo and opd
    have no such key, so for them the best-effort pin plus lifecycle fallback stays correct.
    """
    import flash.envs.loader as env_loader
    from flash.providers import allocator

    persisted = []

    def blip(_parsed, *_args, **_kwargs):
        raise RuntimeError("github rate limit at submit time")

    from dataclasses import replace

    monkeypatch.setattr(env_loader, "_resolve_ref_sha", blip)
    monkeypatch.setattr(
        orch,
        "_resolve_model_revision",
        lambda spec, **_kw: replace(spec, model_revision="b" * 40),
    )
    monkeypatch.setattr(orch, "_save_status", lambda *a, **k: persisted.append(a))
    monkeypatch.setattr(
        allocator, "allocate", lambda *a, **k: pytest.fail("allocated without a profile")
    )

    with pytest.raises(orch.WorkloadProfileUnavailable, match="immutable resolved environment"):
        orch.submit_job(_public_spec())

    assert persisted == []


def test_lifecycle_fallback_pin_is_persisted_for_recovery(orch, monkeypatch):
    """A pin the lifecycle fallback recovers must survive a control-plane restart.

    Submit's pin is best-effort, so a GitHub blip leaves the run unpinned and
    `_pin_environment_for_run` resolves it instead. That SHA has to reach
    `effective_preparation.worker_spec` before provisioning: recovery reloads only the persisted
    record and calls that helper with `attempt_started=True`, which deliberately refuses to resolve
    again. A pin held in the lifecycle's local `spec` alone would leave recovery unpinned, so a later
    attempt could resolve a moved ref to different code while resuming the first attempt's
    checkpoint (codex[bot]).

    Exercised on grpo because sft can no longer reach this state at all: its profile gate rejects an
    unpinned environment at submit instead of deferring the pin (the fail-closed test above). grpo
    and opd keep the best-effort pin, so the fallback they depend on is still live.
    """
    from dataclasses import replace

    import flash.catalog as catalog
    import flash.envs.loader as env_loader
    from flash.providers import allocator
    from flash.providers.base import PollResult
    from flash.providers.runpod import jobs as rp_jobs
    from flash.runner import lifecycle

    first_sha = "a" * 40
    # every resolution AFTER the fallback returns a different commit, standing in for a push landing
    # mid-run. seeing it anywhere proves something re-resolved a ref that was already pinned.
    moved_sha = "c" * 40
    resolutions = []
    persisted_at_submission = []

    def fake_resolve(_parsed, *_args, **_kwargs):
        resolutions.append(len(resolutions))
        if not resolutions[:-1]:
            raise RuntimeError("github rate limit at submit time")
        return first_sha if len(resolutions) == 2 else moved_sha

    monkeypatch.setattr(
        orch, "_resolve_model_revision", lambda spec, **_kw: replace(spec, model_revision="b" * 40)
    )
    monkeypatch.setattr(orch, "resolve_model", lambda model, *a, **k: catalog.MODELS[model])

    def fake_runpod_submit(run_spec, seed, **kwargs):
        persisted = orch.get_status(run_spec.run_id).effective_preparation["worker_spec"]
        persisted_at_submission.append(persisted["environment"]["resolved_sha"])
        return PollResult(True, metrics={"train_tokens": 4096, "wall_seconds": 1})

    monkeypatch.setattr(env_loader, "_resolve_ref_sha", fake_resolve)
    monkeypatch.setattr(
        "flash.cost.spec.estimate_for_spec",
        lambda _spec, **_kwargs: type("Estimate", (), {"total_usd": 1.0})(),
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(gpu="RTX 5090"))
    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)
    monkeypatch.setattr("flash.providers._worker.upload_code", lambda *a, **k: None)
    monkeypatch.setattr(orch, "flash_code_prefix", lambda: "code/test/flash")
    monkeypatch.setattr(orch, "_persist_metrics", lambda *a, **k: 0.0)
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda spec: None)
    monkeypatch.setattr(lifecycle, "_register_checkpoints_best_effort", lambda *a, **k: None)

    public = _public_spec(algorithm="grpo")
    orch.submit_job(public)

    # the pin must already be persisted when the provider is called, not written back afterwards:
    # a crash between provisioning and a later write is exactly the window recovery reads in.
    assert persisted_at_submission == [first_sha]

    # a control-plane restart keeps nothing in memory -- this is the whole recovery input.
    restarted = orch.get_status(public.run_id)
    assert (
        restarted.effective_preparation["worker_spec"]["environment"]["resolved_sha"] == first_sha
    )
    recovered = orch.reallocation_spec_from_status(restarted, verify_source=True)
    assert recovered.environment.resolved_sha == first_sha
    assert moved_sha not in resolutions


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


def test_a_multi_card_run_is_costed_for_every_card_it_occupied(orch):
    """`hourly_rate` is per CARD, so the measured cost must multiply by the allocated count.

    without it a 4-card run records a quarter of its real spend, and the cost analytics that compare
    estimates against actuals are corrupted by exactly that factor. the count has to come from the
    metrics stamp rather than `spec.gpu.count`, which is only a ceiling -- allocation routinely picks
    fewer cards than authored.
    """
    spec = _spec()
    _seed_status(orch, spec)
    base = {"train_tokens": 4096, "wall_seconds": 1800, "allocated_gpu": "RTX 4090"}

    four_card = orch._persist_metrics(spec, {**base, "allocated_gpu_count": 4})

    assert four_card == pytest.approx(0.345 * 4), "a 4-card run was priced as one card"
    # a record predating the stamp still reads as one card rather than zero or a crash
    assert orch._persist_metrics(spec, dict(base)) == pytest.approx(0.345)


def test_the_allocated_card_count_reaches_the_metrics_the_cost_is_read_from(orch, monkeypatch):
    """The multiply above is inert unless allocation actually stamps the count it chose.

    this drives the real submit path rather than handing `_persist_metrics` a literal, because the
    two halves fail independently: pricing can multiply correctly by a count that is never recorded.
    """
    from flash.providers import allocator
    from flash.providers.base import Candidate, PollResult
    from flash.providers.runpod import jobs as rp_jobs

    monkeypatch.setattr(
        allocator,
        "allocate",
        lambda *a, **k: _alloc(candidates=(Candidate("runpod", "RTX 4090", 0.69, 24, 4),)),
    )

    def fake_runpod_submit(run_spec, seed, log=None, on_handle=None, attempt=0, **_):
        if on_handle:
            on_handle(_runpod_handle(attempt=attempt))
        return PollResult(True, metrics={"train_tokens": 4096, "wall_seconds": 1800})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)
    spec = _spec(provider="runpod", type="RTX 4090", count=4)
    _seed_status(orch, spec)

    metrics = orch._submit_seed_supervised(spec, spec.seed, io.StringIO())

    assert metrics["allocated_gpu_count"] == 4
    # and the stamp is what pricing then reads, so the two halves compose
    assert orch._persist_metrics(spec, metrics) == pytest.approx(0.345 * 4)


def test_infra_retry_walks_to_next_runpod_class_and_deletes_endpoint(orch, monkeypatch):
    from flash.providers import allocator
    from flash.providers.base import Candidate, PollResult
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs

    candidates = (
        Candidate("runpod", "RTX 4090", 0.39, 24),
        Candidate("runpod", "H100", 0.49, 48),
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    cancelled, deleted, submitted_gpus = [], [], []
    monkeypatch.setattr(
        runpod_api,
        "cancel_job",
        lambda e, j, **_kw: cancelled.append((e, j)) or {"id": j, "status": "CANCELLED"},
    )
    monkeypatch.setattr(
        runpod_api,
        "delete_endpoint_for_fingerprint",
        lambda e, _fingerprint: deleted.append(e) or True,
    )

    def fake_runpod_submit(run_spec, seed, log=None, on_handle=None, attempt=0, **kwargs):
        submitted_gpus.append(run_spec.gpu.type)
        if attempt == 0:
            on_handle(_runpod_handle("ep1", "j1", attempt))
            return PollResult(False, failure="stalled", detail="no worker progress")
        on_handle(_runpod_handle("ep2", "j2", attempt))
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)
    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()
    metrics = orch._submit_seed_supervised(spec, spec.seed, log)
    assert metrics["train_tokens"] == 4096
    assert submitted_gpus == ["RTX 4090", "H100"]
    assert cancelled == [("ep1", "j1")]
    assert "ep1" in deleted
    assert "walking past the cheapest class" in log.getvalue()


def test_pinned_gpu_retry_says_there_is_no_untried_class_left(orch, monkeypatch):
    """A pinned gpu.type gives the picker a ONE-ENTRY candidate list, so a no_capacity retry
    re-selects the same unavailable class and burns another full capacity grace on it. That is the
    right call (never strand a run), but it used to be silent: the walk line only prints when the
    class CHANGES, so the operator saw two identical attempts and no reason for either. The fix the
    log has to point at is unpinning gpu.type, not waiting longer."""
    from flash.providers import allocator
    from flash.providers.base import Candidate, PollResult
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs

    # exactly what a pinned spec produces: the allocator only ever offers the pinned class.
    candidates = (Candidate("runpod", "H200", 4.0, 141),)
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    monkeypatch.setattr(runpod_api, "cancel_job", lambda *a, **k: None)
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", lambda *_a, **_k: True)

    submitted_gpus = []

    def fake_runpod_submit(run_spec, seed, log=None, on_handle=None, attempt=0, **kwargs):
        submitted_gpus.append(run_spec.gpu.type)
        on_handle(_runpod_handle(f"ep{attempt}", f"j{attempt}", attempt))
        if attempt == 0:
            return PollResult(
                False,
                failure="no_capacity",
                detail="never scheduled: job stuck IN_QUEUE for 903s",
            )
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)
    spec = _spec(run_id="flash-pinned-gpu-nowalk", type="H200")
    _seed_status(orch, spec)
    log = io.StringIO()
    orch._submit_seed_supervised(spec, spec.seed, log)

    # the retry re-picked the SAME class, so the walk line cannot fire...
    assert submitted_gpus == ["H200", "H200"]
    out = log.getvalue()
    assert "walking past the cheapest class" not in out
    # ...and the operator is told why, plus that the pin is what made the list a singleton.
    assert "no untried class left; re-selecting H200" in out
    assert "gpu.type is pinned" in out


def test_unconfirmed_runpod_teardown_retains_handle_and_blocks_retry(orch, monkeypatch):
    from flash.providers import allocator
    from flash.providers.base import Candidate, PollResult
    from flash.providers.runpod import RunpodProvider
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs

    candidates = (
        Candidate("runpod", "RTX 4090", 0.39, 24),
        Candidate("runpod", "H100", 0.49, 48),
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))

    submitted_attempts = []
    deleted_endpoints = []
    gc_calls = []
    monkeypatch.setattr(runpod_api, "cancel_job", lambda *a, **k: None)
    monkeypatch.setattr(
        runpod_api,
        "delete_endpoint_for_fingerprint",
        lambda endpoint_id, _fingerprint: deleted_endpoints.append(endpoint_id) or False,
    )
    monkeypatch.setattr(
        runpod_api,
        "job_status",
        lambda *_args, **_kwargs: {"status": "IN_PROGRESS"},
    )
    monkeypatch.setattr(
        RunpodProvider,
        "gc",
        lambda self, cleanup_spec: gc_calls.append(cleanup_spec.run_id),
    )

    def fake_runpod_submit(run_spec, seed, log=None, on_handle=None, attempt=0, **kwargs):
        submitted_attempts.append(attempt)
        on_handle(_runpod_handle("ep-unconfirmed", "job-unconfirmed", attempt))
        return PollResult(False, failure="stalled", detail="no worker progress")

    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)
    spec = _spec(run_id="flash-unconfirmed-runpod-teardown")
    _seed_status(orch, spec)
    log = io.StringIO()

    with pytest.raises(RuntimeError, match="teardown could not be confirmed"):
        orch._submit_seed_supervised(spec, spec.seed, log)

    assert submitted_attempts == [0]
    assert deleted_endpoints
    assert set(deleted_endpoints) == {"ep-unconfirmed"}
    assert gc_calls == [spec.run_id]
    status = orch.get_status(spec.run_id)
    assert status.remote["endpoint_id"] == "ep-unconfirmed"
    assert status.remote["job_id"] == "job-unconfirmed"
    assert "teardown unconfirmed" in log.getvalue()


def _oom_candidates():
    from flash.providers.base import Candidate

    return (
        Candidate("runpod", "A100 PCIe", 1.0, 80),
        Candidate("runpod", "RTX Pro 6000", 2.0, 96),
        Candidate("runpod", "B200", 3.0, 180),
    )


def _run_failed_oom_sequence(orch, monkeypatch, failures, *, max_retries):
    from flash.providers import allocator
    from flash.providers.base import PollResult
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs

    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=_oom_candidates()))
    monkeypatch.setattr(
        runpod_api,
        "cancel_job",
        lambda _endpoint_id, job_id, **_kw: {"id": job_id, "status": "CANCELLED"},
    )
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", lambda e, _fingerprint: True)

    submitted = []
    on_last_gpu = []
    failure_iter = iter(failures)

    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, **kwargs):
        submitted.append(run_spec.gpu.type)
        on_last_gpu.append(kwargs.get("on_last_gpu", False))
        on_handle(_runpod_handle(f"ep{attempt}", f"j{attempt}", attempt))
        failure = next(failure_iter)
        detail = "CUDA out of memory" if failure == "oom" else "x"
        return PollResult(False, failure=failure, detail=detail)

    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec(max_retries=max_retries)
    _seed_status(orch, spec)
    with pytest.raises(RuntimeError):
        orch._submit_seed_supervised(spec, spec.seed, io.StringIO())
    return submitted, on_last_gpu


def test_oom_escalates_exactly_once_at_max_retries_1(orch, monkeypatch):
    submitted, _on_last_gpu = _run_failed_oom_sequence(
        orch,
        monkeypatch,
        ["oom", "oom"],
        max_retries=1,
    )
    assert submitted == ["A100 PCIe", "RTX Pro 6000"]


def test_oom_after_infra_failure_still_escalates(orch, monkeypatch):
    submitted, _on_last_gpu = _run_failed_oom_sequence(
        orch,
        monkeypatch,
        ["no_capacity", "oom", "oom"],
        max_retries=1,
    )
    assert submitted == ["A100 PCIe", "RTX Pro 6000", "B200"]


def test_infra_failure_after_oom_uses_infra_budget(orch, monkeypatch):
    submitted, on_last_gpu = _run_failed_oom_sequence(
        orch,
        monkeypatch,
        ["oom", "no_capacity", "oom"],
        max_retries=1,
    )
    assert submitted == ["A100 PCIe", "RTX Pro 6000", "B200"]
    assert on_last_gpu == [False, False, True]


def test_oom_never_escalates_at_max_retries_0(orch, monkeypatch):
    submitted, _on_last_gpu = _run_failed_oom_sequence(
        orch,
        monkeypatch,
        ["oom"],
        max_retries=0,
    )
    assert submitted == ["A100 PCIe"]


def test_select_candidate_escapes_failed_provider_then_walks_classes():
    """The retry picker prefers cheapest first, escapes a failed provider cross-provider on retry,
    and only walks classes within a provider once every provider has been burned."""
    from flash.providers.base import Candidate
    from flash.runner.lifecycle import _select_candidate

    cands = (
        Candidate("runpod", "H100", 0.49, 48),
        Candidate("runpod", "RTX Pro 6000", 0.50, 96),
        Candidate("lambda", "H100", 0.50, 48),
    )
    # Attempt 0 (nothing failed): cheapest overall.
    assert _select_candidate(cands, set(), set()) is cands[0]
    # RunPod burned an infra attempt -> escape to the OTHER provider, not the next RunPod class.
    # the tried set is keyed by SHAPE (provider, class, card count), so a 2-tuple would never match
    # and every candidate would read as untried.
    chosen = _select_candidate(cands, {"runpod"}, {("runpod", "H100", 1)})
    assert (chosen.provider, chosen.gpu) == ("lambda", "H100")
    # Both providers burned -> fall back to the cheapest class NOT yet tried (within-provider walk).
    chosen = _select_candidate(
        cands,
        {"runpod", "lambda"},
        {("runpod", "H100", 1), ("lambda", "H100", 1)},
    )
    assert (chosen.provider, chosen.gpu) == ("runpod", "RTX Pro 6000")


def test_select_candidate_single_provider_walks_classes():
    """With only one provider configured, the picker degrades to the cheapest untried class."""
    from flash.providers.base import Candidate
    from flash.runner.lifecycle import _select_candidate

    cands = (Candidate("runpod", "RTX 4090", 0.39, 24), Candidate("runpod", "H100", 0.49, 48))
    assert _select_candidate(cands, {"runpod"}, {("runpod", "RTX 4090", 1)}).gpu == "H100"


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
    assert _select_candidate(cands, {"runpod"}, {("runpod", "H200", 1)}) is only


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
        Candidate("runpod", "RTX Pro 6000", 0.50, 96),  # next runpod class, the wrong retry target
        Candidate("lambda", "H100", 0.50, 48),  # the right cross-provider escape
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    monkeypatch.setattr(
        runpod_api,
        "cancel_job",
        lambda _endpoint_id, job_id, **_kw: {"id": job_id, "status": "CANCELLED"},
    )
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", lambda e, _fingerprint: True)

    rp_gpus, lam_gpus = [], []

    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, **kw):
        rp_gpus.append(run_spec.gpu.type)
        on_handle(_runpod_handle("ep1", "j1", attempt))
        return PollResult(
            False, failure="no_capacity", detail="job stuck IN_QUEUE (no RunPod capacity)"
        )

    def fake_lam(spec, seed, log=None, on_handle=None, attempt=0, **kw):
        lam_gpus.append(spec.gpu.type)
        if on_handle:
            on_handle(_lambda_handle(attempt=attempt))
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    monkeypatch.setattr(lambda_jobs, "submit_run_lambda", fake_lam)
    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()
    metrics = orch._submit_seed_supervised(spec, spec.seed, log)
    assert metrics["train_tokens"] == 4096
    assert rp_gpus == ["H100"]  # RunPod tried exactly once...
    assert lam_gpus == ["H100"]  # ...then the retry escaped cross-provider to Lambda
    assert orch.get_status(spec.run_id).remote["provider"] == "lambda"
    assert "walking past the cheapest class" in log.getvalue()


@pytest.mark.parametrize("failure", ["no_capacity", "poll_error"])
def test_shared_cache_zero_retries_submits_exactly_once(orch, monkeypatch, failure):
    """max_retries=0 is one provider submission even with the managed shared cache."""
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
    monkeypatch.setattr(
        runpod_api,
        "cancel_job",
        lambda _endpoint_id, job_id, **_kw: {"id": job_id, "status": "CANCELLED"},
    )
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", lambda e, _fingerprint: True)

    volumes_seen = []

    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, **kw):
        vol = getattr(run_spec.gpu, "network_volume", None)
        volumes_seen.append(vol)
        if on_handle:
            on_handle(_runpod_handle(f"ep{attempt}", f"j{attempt}", attempt))
        return PollResult(False, failure=failure, detail="cache-constrained failure")

    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec(max_retries=0, network_volume=WEIGHT_CACHE_VOLUME_NAME, network_volume_gb=100)
    _seed_status(orch, spec)
    with pytest.raises(RuntimeError, match="failed after retries"):
        orch._submit_seed_supervised(spec, spec.seed, io.StringIO())
    assert volumes_seen == [WEIGHT_CACHE_VOLUME_NAME]


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
            "runpod", "RTX Pro 6000", 0.50, 96
        ),  # the GPU-walk target the real retry must reach
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    monkeypatch.setattr(
        runpod_api,
        "cancel_job",
        lambda _endpoint_id, job_id, **_kw: {"id": job_id, "status": "CANCELLED"},
    )
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", lambda e, _fingerprint: True)

    seen = []

    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, **kw):
        vol = getattr(run_spec.gpu, "network_volume", None)
        seen.append((run_spec.gpu.type, vol))
        if on_handle:
            on_handle(_runpod_handle(f"ep{attempt}", f"j{attempt}", attempt))
        # Cache attempt AND its cache-less same-class fallback both starve; only the walk to the OTHER
        # class (the genuine retry that must survive the cache drop) succeeds.
        if run_spec.gpu.type == "RTX Pro 6000":
            return PollResult(True, metrics={"train_tokens": 4096})
        return PollResult(False, failure="no_capacity", detail="IN_QUEUE (no capacity)")

    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec(max_retries=1, network_volume=WEIGHT_CACHE_VOLUME_NAME, network_volume_gb=100)
    _seed_status(orch, spec)
    metrics = orch._submit_seed_supervised(spec, spec.seed, io.StringIO())
    assert metrics["train_tokens"] == 4096
    # cache attempt -> cache-less SAME class (free fallback) -> GPU-walk to the OTHER class (real retry).
    assert seen == [
        ("H100", WEIGHT_CACHE_VOLUME_NAME),
        ("H100", None),
        ("RTX Pro 6000", None),
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
        lambda_api,
        "terminate_instance_confirmed",
        lambda instance_id: terminated.append(instance_id),
    )
    monkeypatch.setattr(lambda_jobs, "run_instances_remaining", lambda _run_id: [])
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", lambda e, _fingerprint: True)

    lam_gpus, rp_gpus = [], []

    def fake_lam(spec, seed, log=None, on_handle=None, attempt=0, **kw):
        lam_gpus.append(spec.gpu.type)
        if on_handle:
            on_handle(_lambda_handle("i-broken", attempt))
        return PollResult(
            False,
            failure="job_preempted",
            detail="GPU never became ready after 12 tries: cuda not available",
        )

    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, **kw):
        rp_gpus.append(run_spec.gpu.type)
        on_handle(_runpod_handle("ep2", "j2", attempt))
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(lambda_jobs, "submit_run_lambda", fake_lam)
    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()
    metrics = orch._submit_seed_supervised(spec, spec.seed, log)
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
    monkeypatch.setattr(lambda_api, "terminate_instance_confirmed", lambda instance_id: None)
    monkeypatch.setattr(lambda_jobs, "run_instances_remaining", lambda _run_id: [])
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", lambda e, _fingerprint: True)

    lam_gpus, rp_gpus = [], []

    def fake_lam(spec, seed, log=None, on_handle=None, attempt=0, **kw):
        lam_gpus.append(spec.gpu.type)
        if on_handle:
            on_handle(_lambda_handle(attempt=attempt))
        return PollResult(
            False,
            failure="stalled",
            detail="no worker liveness (boot.log/heartbeat) within 900s of instance active",
        )

    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, **kw):
        rp_gpus.append(run_spec.gpu.type)
        on_handle(_runpod_handle("ep3", "j3", attempt))
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(lambda_jobs, "submit_run_lambda", fake_lam)
    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec()
    _seed_status(orch, spec)
    metrics = orch._submit_seed_supervised(spec, spec.seed, io.StringIO())
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
        orch._submit_seed_supervised(spec, spec.seed, io.StringIO())
    assert calls == [0]


def test_cancel_rejects_legacy_handle_without_provider_identity(orch, monkeypatch):
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import train as rp_train

    cancelled_jobs, deleted_eps = [], []
    monkeypatch.setattr(
        runpod_api,
        "cancel_job",
        lambda e, j, **_kw: cancelled_jobs.append((e, j)) or {"id": j, "status": "CANCELLED"},
    )
    monkeypatch.setattr(
        runpod_api,
        "delete_endpoint_for_fingerprint",
        lambda e, _fingerprint: deleted_eps.append(e) or True,
    )
    monkeypatch.setattr(rp_train, "terminate_endpoint", lambda *a, **k: [])
    spec = _spec()
    st = _seed_status(orch, spec)
    st.state = "running"
    st.remote = {"endpoint_id": "ep1", "endpoint_name": "n", "job_id": "j1"}
    orch._save_status(st)
    with pytest.raises(RuntimeError, match="exact cleanup target could not be preserved"):
        orch.cancel_run(spec.run_id)
    assert cancelled_jobs == []
    assert deleted_eps == []
    assert orch.get_status(spec.run_id).remote == st.remote


def test_config_gpu_fields(monkeypatch):
    from flash.schema import spec_from_dict

    base = {
        "model": "Qwen/Qwen3.5-0.8B",
        "algorithm": "sft",
        "train": {"epochs": 1, "max_examples": 8},
        "environment": {"id": "github:owner/repo@main:env/environment.py"},
    }
    spec = spec_from_dict(dict(base), run_id="x")
    assert spec.gpu.type == ""
    again = JobSpec.from_dict(spec.to_dict())
    assert again.gpu.type == ""
    spec = spec_from_dict({**base, "gpu": {"type": "A100 SXM"}}, run_id="x")
    assert spec.gpu.type == "A100 SXM"


def _retry_action_line(log_text: str, attempt: int) -> str:
    """The `failed (...); <action>` line for one attempt.

    Assertions about the retry promise must read THIS line only. The allocation summary printed
    each attempt legitimately says `next-best: <class>` -- that is the candidate list, not a claim
    about the retry -- so a whole-log substring check cannot tell the two apart.
    """
    marker = f"attempt={attempt} failed"
    for line in log_text.splitlines():
        if marker in line:
            return line
    raise AssertionError(f"no action line for attempt={attempt} in:\n{log_text}")


def _retry_block(log_text: str, attempt: int) -> str:
    """The action line plus the failure detail printed with it, as one string.

    The two come from a single print and are read together, so a claim in one contradicts a claim in
    the other. This is the unit to assert on when the point is that they AGREE. It stops at the
    closing `---` so the next attempt's allocation summary -- which legitimately says
    `next-best: <class>` about the candidate list -- stays out (see `_retry_action_line`).
    """
    lines = log_text.splitlines()
    marker = f"attempt={attempt} failed"
    for i, line in enumerate(lines):
        if marker in line:
            for j in range(i + 1, len(lines)):
                if lines[j].strip() == "---" and j > i + 1:
                    return "\n".join(lines[i : j + 1])
            return "\n".join(lines[i:])
    raise AssertionError(f"no action line for attempt={attempt} in:\n{log_text}")


def test_no_capacity_retry_message_names_the_class_it_actually_reuses(orch, monkeypatch):
    """LS-008/AT-013: a capacity failure on the LAST fitting class used to say the run was 'retrying
    on the next-best GPU' while the picker had nowhere to walk and re-selected that same class. The
    log must describe the retry that actually happens."""
    from flash.providers import allocator
    from flash.providers.base import Candidate, PollResult
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs

    # exactly one fitting class: the picker can only ever re-pick it.
    candidates = (Candidate("runpod", "H200", 4.0, 141),)
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    monkeypatch.setattr(
        runpod_api,
        "cancel_job",
        lambda _endpoint_id, job_id, **_kw: {"id": job_id, "status": "CANCELLED"},
    )
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", lambda e, _fingerprint: True)

    gpus = []

    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, **kw):
        gpus.append(run_spec.gpu.type)
        on_handle(_runpod_handle("ep1", "j1", attempt))
        if attempt == 0:
            return PollResult(False, failure="no_capacity", detail="job stuck IN_QUEUE")
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()
    orch._submit_seed_supervised(spec, spec.seed, log)

    assert gpus == ["H200", "H200"]  # nowhere to walk: the same class is genuinely reused
    action = _retry_action_line(log.getvalue(), 0)
    assert "expecting to retry on H200 @ runpod again" in action, action
    assert "no untried GPU class fits this run" in action, action
    assert "next-best" not in action, "claimed an escalation that the picker cannot perform"
    # the next attempt re-allocates against live capacity, so this is a projection, not a promise.
    assert "the class may change" in action, action


def test_last_gpu_retry_message_names_the_clamped_back_class_not_the_current_one(orch, monkeypatch):
    """on_last_gpu means no UNTRIED class is left -- NOT that the current class is reused. With two
    fitting classes the walk is PCIe, SXM, then back to the cheaper PCIe, so the message printed on
    the SXM failure must name the PCIe the picker actually selects, not the SXM it is leaving."""
    from flash.providers import allocator
    from flash.providers.base import Candidate, PollResult
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs

    candidates = (
        Candidate("runpod", "A100 PCIe", 1.0, 80),
        Candidate("runpod", "A100 SXM", 2.0, 80),
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    monkeypatch.setattr(
        runpod_api,
        "cancel_job",
        lambda _endpoint_id, job_id, **_kw: {"id": job_id, "status": "CANCELLED"},
    )
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", lambda e, _fingerprint: True)

    gpus = []

    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, **kw):
        gpus.append(run_spec.gpu.type)
        on_handle(_runpod_handle("ep1", "j1", attempt))
        if attempt < 2:
            return PollResult(False, failure="no_capacity", detail="job stuck IN_QUEUE")
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec(max_retries=2)
    _seed_status(orch, spec)
    log = io.StringIO()
    orch._submit_seed_supervised(spec, spec.seed, log)

    # the clamp-back the message has to describe: cheapest, then the untried SXM, then back.
    assert gpus == ["A100 PCIe", "A100 SXM", "A100 PCIe"]
    text = log.getvalue()
    # attempt 1 IS the SXM failure, and it is the one that clamps back to the cheaper PCIe.
    action = _retry_action_line(text, 1)
    assert (
        "expecting to retry on A100 PCIe @ runpod, no untried GPU class fits this run" in action
    ), action
    # the failing attempt was the SXM; promising it back would name hardware the picker skipped.
    assert "retry on A100 SXM" not in action, action
    assert "A100 SXM @ runpod again" not in action, action
    # attempt 0 still has the untried SXM to walk to, so it names that and claims no exhaustion.
    first = _retry_action_line(text, 0)
    assert "expecting to retry on A100 SXM @ runpod" in first, first
    assert "no untried GPU class fits this run" not in first, first


def test_cache_drop_retry_names_the_same_class_it_reselects(orch, monkeypatch):
    """A cache-drop retry deliberately leaves failed_providers and tried_classes untouched, so the
    next attempt removes the volume and reselects the SAME class. That path is not gated on
    on_last_gpu -- with several fitting classes it runs while on_last_gpu is false -- so gating the
    projection on the flag left exactly this escalation described by the generic retry line, the
    same misleading claim the change exists to fix."""
    from flash.providers import allocator
    from flash.providers.base import Candidate, PollResult
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs
    from flash.runner import WEIGHT_CACHE_VOLUME_NAME

    candidates = (
        Candidate("runpod", "H100", 0.49, 48),
        Candidate("runpod", "RTX Pro 6000", 0.50, 96),
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    monkeypatch.setattr(
        runpod_api,
        "cancel_job",
        lambda _endpoint_id, job_id, **_kw: {"id": job_id, "status": "CANCELLED"},
    )
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", lambda e, _fingerprint: True)

    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, **kw):
        on_handle(_runpod_handle(f"ep{attempt}", f"j{attempt}", attempt))
        if getattr(run_spec.gpu, "network_volume", None) == WEIGHT_CACHE_VOLUME_NAME:
            return PollResult(False, failure="no_capacity", detail="IN_QUEUE (no capacity)")
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec(max_retries=1, network_volume=WEIGHT_CACHE_VOLUME_NAME, network_volume_gb=100)
    _seed_status(orch, spec)
    log = io.StringIO()
    orch._submit_seed_supervised(spec, spec.seed, log)

    action = _retry_action_line(log.getvalue(), 0)
    # two classes fit, so the escalation claim must be absent -- but the class is still named.
    assert "expecting to retry on H100 @ runpod again" in action, action
    assert "no untried GPU class fits this run" not in action, action
    assert "next-best" not in action, action


def test_cache_drop_failure_detail_does_not_contradict_the_action_line(orch, monkeypatch):
    """The failure detail and the action line are printed in the SAME log block, so they have to
    agree. A cache-drop retry runs with on_last_gpu false (classes remain untried), and poll_job's
    false branch used to read 'retrying on the next-best GPU' -- while the action line directly
    beneath named the same class being reselected. One block, two contradictory retry targets.

    poll_job holds neither the retry budget nor the candidate list, so it cannot know a cache drop is
    coming; the fix is to make the detail state only the escalation fact and leave the target to the
    supervisor. Asserts on the block as a whole, which is what an operator actually reads."""
    from flash.providers import allocator
    from flash.providers.base import Candidate, PollResult
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs
    from flash.runner import WEIGHT_CACHE_VOLUME_NAME

    candidates = (
        Candidate("runpod", "H100", 0.49, 48),
        Candidate("runpod", "RTX Pro 6000", 0.50, 96),
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    monkeypatch.setattr(
        runpod_api,
        "cancel_job",
        lambda _endpoint_id, job_id, **_kw: {"id": job_id, "status": "CANCELLED"},
    )
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", lambda e, _fingerprint: True)

    # the real poll_job wording, not a hand-written stand-in: build the clause with the production
    # helper so a regression in it fails this test rather than being masked by a copied string.
    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, on_last_gpu=False, **kw):
        on_handle(_runpod_handle(f"ep{attempt}", f"j{attempt}", attempt))
        if getattr(run_spec.gpu, "network_volume", None) == WEIGHT_CACHE_VOLUME_NAME:
            note = rp_jobs.capacity_escalation_note(on_last_gpu)
            return PollResult(
                False,
                failure="no_capacity",
                detail=f"never scheduled: job stuck IN_QUEUE for 900s (no RunPod capacity for the "
                f"pinned GPU class); {note}",
            )
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec(max_retries=1, network_volume=WEIGHT_CACHE_VOLUME_NAME, network_volume_gb=100)
    _seed_status(orch, spec)
    log = io.StringIO()
    orch._submit_seed_supervised(spec, spec.seed, log)

    block = _retry_block(log.getvalue(), 0)
    assert "expecting to retry on H100 @ runpod again" in block, block
    # the whole point: nothing in this block may promise a walk the supervisor is not making. scoped
    # to the block, since the allocation summary outside it names the next-best CANDIDATE legitimately.
    assert "next-best" not in block, block
    assert "GPU-class escalation may follow" in block, block


def test_projected_retry_class_is_worded_as_a_projection_not_a_promise(orch, monkeypatch):
    """The projection reads the CURRENT candidate list, but the next attempt calls allocate() again.
    Providers that rebuild candidates from live capacity can drop the named class or surface a
    cheaper one, so the log must not claim this is the class the retry will certainly use."""
    from flash.providers import allocator
    from flash.providers.base import Candidate, PollResult
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs

    # capacity moves between attempts: the projected H200 is gone by the time the retry allocates.
    allocations = [
        _alloc(candidates=(Candidate("runpod", "H200", 4.0, 141),)),
        _alloc(candidates=(Candidate("runpod", "H100", 0.49, 80),)),
    ]
    monkeypatch.setattr(
        allocator, "allocate", lambda *a, **k: allocations.pop(0) if allocations else allocations
    )
    monkeypatch.setattr(
        runpod_api,
        "cancel_job",
        lambda _endpoint_id, job_id, **_kw: {"id": job_id, "status": "CANCELLED"},
    )
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", lambda e, _fingerprint: True)

    gpus = []

    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, **kw):
        gpus.append(run_spec.gpu.type)
        on_handle(_runpod_handle("ep1", "j1", attempt))
        if attempt == 0:
            return PollResult(False, failure="no_capacity", detail="job stuck IN_QUEUE")
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()
    orch._submit_seed_supervised(spec, spec.seed, log)

    # the projection named H200; reallocation actually produced H100. the wording must survive that.
    assert gpus == ["H200", "H100"]
    action = _retry_action_line(log.getvalue(), 0)
    assert "expecting to retry on H200 @ runpod" in action, action
    assert "the class may change" in action, action
    assert "retrying on H200" not in action, (
        "stated a projected class as the confirmed retry target"
    )


def test_sole_class_cache_drop_does_not_claim_the_class_is_exhausted(orch, monkeypatch):
    """on_last_gpu is not an exhaustion signal, so the escalation clause cannot be read off it.

    With one fitting class, ``len(untried) <= 1`` sets the flag on the FIRST attempt -- before that
    class has been tried at all. A cache-drop retry then deliberately leaves tried_classes untouched
    and reselects the same class cold, so the line read "expecting to retry on H100 @ runpod again,
    no untried GPU class fits this run": naming the untried class in the same clause that denies one
    exists. Derive the clause from the sets the retry will actually see instead (codex[bot])."""
    from flash.providers import allocator
    from flash.providers.base import Candidate, PollResult
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs
    from flash.runner import WEIGHT_CACHE_VOLUME_NAME

    # exactly one fitting class -> untried is a single entry -> on_last_gpu true from attempt 0.
    candidates = (Candidate("runpod", "H100", 0.49, 80),)
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    monkeypatch.setattr(
        runpod_api,
        "cancel_job",
        lambda _endpoint_id, job_id, **_kw: {"id": job_id, "status": "CANCELLED"},
    )
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", lambda e, _fingerprint: True)

    seen_flags = []

    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, on_last_gpu=False, **kw):
        seen_flags.append(on_last_gpu)
        on_handle(_runpod_handle(f"ep{attempt}", f"j{attempt}", attempt))
        if getattr(run_spec.gpu, "network_volume", None) == WEIGHT_CACHE_VOLUME_NAME:
            return PollResult(False, failure="no_capacity", detail="IN_QUEUE (no capacity)")
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec(max_retries=1, network_volume=WEIGHT_CACHE_VOLUME_NAME, network_volume_gb=100)
    _seed_status(orch, spec)
    log = io.StringIO()
    orch._submit_seed_supervised(spec, spec.seed, log)

    # the flag really is set here -- otherwise this test would pass for the wrong reason.
    assert seen_flags[0] is True, seen_flags
    action = _retry_action_line(log.getvalue(), 0)
    assert "expecting to retry on H100 @ runpod again" in action, action
    # the class the very next attempt reuses is still untried, so nothing may claim otherwise.
    assert "no untried GPU class fits this run" not in action, action


def test_sole_class_infra_retry_still_reports_exhaustion(orch, monkeypatch):
    """The complement of the cache-drop case: a plain infra retry DOES mark the class tried, so with
    one fitting class the clause is accurate and must survive. Guards against fixing the false
    positive by deleting the clause outright."""
    from flash.providers import allocator
    from flash.providers.base import Candidate, PollResult
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs

    candidates = (Candidate("runpod", "H100", 0.49, 80),)
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    monkeypatch.setattr(
        runpod_api,
        "cancel_job",
        lambda _endpoint_id, job_id, **_kw: {"id": job_id, "status": "CANCELLED"},
    )
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", lambda e, _fingerprint: True)

    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, **kw):
        on_handle(_runpod_handle(f"ep{attempt}", f"j{attempt}", attempt))
        if attempt == 0:
            return PollResult(False, failure="no_capacity", detail="job stuck IN_QUEUE")
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec(max_retries=2)
    _seed_status(orch, spec)
    log = io.StringIO()
    orch._submit_seed_supervised(spec, spec.seed, log)

    action = _retry_action_line(log.getvalue(), 0)
    assert "expecting to retry on H100 @ runpod again" in action, action
    assert "no untried GPU class fits this run" in action, action


def test_workload_profile_mismatch_fails_fast_instead_of_retrying(orch, monkeypatch):
    """A profile whose identity does not match the spec is terminal, not infrastructure.

    The selected-quote refresh re-derives the profile digest from the effective spec, so a mismatch
    resolves identically on every attempt. Classifying it as infra-shaped burns the run's whole
    retry budget on real ``time.sleep`` backoffs before failing anyway -- the shape that wedged the
    suite for 20 minutes on a single test."""
    from flash.providers import allocator
    from flash.providers.base import PollResult
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs
    from flash.workload_profile import WorkloadProfileMismatch

    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc())
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", lambda e, _fingerprint: True)

    submits = []

    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, **kw):
        submits.append(attempt)
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)

    import flash.cost.spec as cost_spec

    def refuse(*_a, **_kw):
        raise WorkloadProfileMismatch("workload profile input digest does not match")

    monkeypatch.setattr(cost_spec, "estimate_for_spec", refuse)

    # any sleep here means the failure was misclassified as a transient the run should wait out.
    slept = []
    monkeypatch.setattr(orch.lifecycle.time, "sleep", lambda s: slept.append(s))

    spec = _spec(max_retries=2)
    _seed_status(orch, spec)
    with pytest.raises(WorkloadProfileMismatch):
        orch._submit_seed_supervised(spec, spec.seed, io.StringIO())

    assert submits == []  # never reached a provider
    assert slept == []  # and never backed off waiting for it to clear
