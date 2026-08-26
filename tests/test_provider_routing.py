"""Orchestrator RunPod routing: submit/cancel, retry, handle persistence, and cost flow."""

from __future__ import annotations

import io
import threading
from dataclasses import replace

import pytest

import flash.providers._lifecycle.net.worker as provider_worker
import flash.runner.accounting.artifacts as runner_artifacts
import flash.runner.lifecycle.deadlines as runner_deadlines
import flash.runner.lifecycle.preparation as runner_preparation
import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.status as runner_status
import flash.runner.lifecycle.submit as runner_submit
import flash.runner.supervise.deploy as runner_deploy
import flash.runner.supervise.errors as runner_errors
import flash.runner.supervise.lifecycle as runner_lifecycle
import flash.runner.supervise.recovery as runner_recovery
from flash.core.spec import JobSpec
from tests._helpers.profile import (
    attach_sft_profile,
    record_sft_profile,
    satisfy_sft_profile,
    stub_revision_geometry,
)
from tests._helpers.source_snapshot import valid_source_snapshot

_RUNPOD_FINGERPRINT = "rpk-" + "0" * 64
_SOURCE_SNAPSHOT = valid_source_snapshot()


@pytest.fixture(autouse=True)
def _source_snapshot_boundary(monkeypatch):

    monkeypatch.setattr(
        provider_worker, "publish_source_snapshot", lambda _repo=None: _SOURCE_SNAPSHOT
    )
    monkeypatch.setattr(
        runner_status,
        "validate_terminal_source_metrics",
        lambda _status, metrics, expected_attempt=None: (metrics, expected_attempt),
    )


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
                "model": "Qwen/Qwen3.5-9B",
                "algorithm": algorithm,
                "run_id": run_id,
                "environment": {
                    "id": "github:owner/repo@main:env/environment.py",
                    "resolved_sha": "a" * 40,
                    "package": {
                        "artifact_revision": "b" * 40,
                        "archive_sha256": "c" * 64,
                        "manifest_sha256": "d" * 64,
                    },
                },
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
    # submission tests exercise environment and persistence boundaries, not an exact gpu pin.
    public["gpu"]["type"] = ""
    return JobSpec.from_dict({**public, "run_id": run_id})


def _alloc(gpu="RTX 4090", rate=0.69, candidates=None):
    from flash.providers.core.base import Allocation, Candidate

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
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    # _spec pins a model revision, which makes the lifecycle's post-allocation quote refresh
    # revision-aware. Left unstubbed it reaches github, and the refresh treats any failure as an
    # infra-shaped transient -- so the whole suite would sit in real retry backoff sleeps.
    stub_revision_geometry(monkeypatch)


def _seed_status(orch, spec):
    st = runner_state.RunStatus(run_id=spec.run_id, state="queued", spec=spec.to_dict())
    runner_state._save_status(st)
    return st


def test_control_plane_rejects_a_profiler_result_with_the_wrong_identity(orch, monkeypatch):
    import flash.engine.profiling.dataset_profile as dataset_profile

    spec = satisfy_sft_profile(monkeypatch, _spec(type="H200"))
    honest = dataset_profile.profile_packaged_sft_dataset(spec, producer_version="ignored")
    monkeypatch.setattr(
        dataset_profile,
        "profile_packaged_sft_dataset",
        lambda *_args, **_kwargs: replace(honest, input_digest="f" * 64),
    )

    with pytest.raises(ValueError, match="input digest"):
        runner_preparation._require_sft_workload_profile(spec)


def test_exact_only_preflight_rejects_unconfigured_provider_set_before_persistence(
    orch, monkeypatch
):
    from flash.providers.core import registry as providers

    persisted = []
    spec = satisfy_sft_profile(monkeypatch, _spec(type="H200"))
    monkeypatch.setattr(providers, "available_providers", lambda: ("lambda", "vast"))
    monkeypatch.setattr(
        runner_state, "_save_status", lambda *args, **kwargs: persisted.append(args)
    )

    with pytest.raises(ValueError, match="no configured provider can provision"):
        runner_submit.submit_job(spec, dry_run=True)

    assert persisted == []


def test_preflight_rejects_an_unreachable_fallback_class(orch, monkeypatch) -> None:
    from flash.providers.core import registry as providers

    spec = satisfy_sft_profile(
        monkeypatch,
        _spec(type="H100", type_fallbacks=("RTX 5090",)),
    )
    monkeypatch.setattr(providers, "available_providers", lambda: ("lambda",))

    with pytest.raises(ValueError, match=r"gpu\.type 'RTX 5090'"):
        runner_submit.submit_job(spec, dry_run=True)


@pytest.mark.parametrize(
    ("gpu_preferences", "expected_provider", "expected_providers"),
    [
        ({"provider": "runpod"}, "runpod", ()),
        ({"providers": ("runpod", "vast")}, "", ("runpod", "vast")),
    ],
)
def test_runpod_allocation_routes_to_runpod_submit(
    orch, monkeypatch, gpu_preferences, expected_provider, expected_providers
):
    from flash.providers.core import allocator
    from flash.providers.core.base import PollResult
    from flash.providers.runpod.execution import job_execution as rp_jobs

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
    spec = _spec(type="RTX 4090", **gpu_preferences)
    _seed_status(orch, spec)
    metrics = runner_lifecycle._submit_seed_supervised(
        spec,
        spec.seed,
        io.StringIO(),
        runtime_secrets={"WANDB_API_KEY": "user-wb"},
        source_snapshot=_SOURCE_SNAPSHOT,
    )
    assert metrics["train_tokens"] == 4096
    assert captured["gpu_type"] == "RTX 4090"
    assert captured["runtime_secrets"] == {"WANDB_API_KEY": "user-wb"}
    assert captured["allocate_kwargs"]["provider"] == expected_provider
    assert captured["allocate_kwargs"]["providers"] == expected_providers
    assert captured["allocate_kwargs"]["gpu_type"] == "RTX 4090"
    remote = runner_status.get_status(spec.run_id).remote
    assert remote["provider"] == "runpod"
    assert remote["allocated_gpu"] == "RTX 4090"


def test_auto_gpu_effective_spec_is_transient_and_keeps_base_auto(orch) -> None:
    base = _spec(type="")

    first_attempt = runner_lifecycle._spec_with_gpu(base, "RTX 4090")
    second_attempt = runner_lifecycle._spec_with_gpu(base, "H100")

    assert base.gpu.type == ""
    assert first_attempt.gpu.type == "RTX 4090"
    assert second_attempt.gpu.type == "H100"
    assert first_attempt is not base
    assert second_attempt is not base


def test_selected_fallback_preserves_the_authored_acceptable_set(orch) -> None:
    base = _spec(type="RTX 5090", type_fallbacks=("A100 PCIe",))

    selected = runner_lifecycle._spec_with_gpu(base, "A100 PCIe")

    assert selected.gpu.acceptable_types == ("A100 PCIe", "RTX 5090")
    _seed_status(orch, base)
    assert runner_submit._persist_effective_worker_spec(selected)
    restored = runner_status.reallocation_spec_from_status(runner_status.get_status(base.run_id))
    assert restored.gpu.acceptable_types == ("RTX 5090", "A100 PCIe")


def test_terminate_persisted_endpoints_isolates_each_gpu_failure(monkeypatch) -> None:
    import flash.providers.runpod.serverless.endpoints as serverless
    from flash.providers.runpod.execution import provider as runpod

    calls: list[str] = []

    def terminate(gpu_type: str, _run_id: str) -> None:
        calls.append(gpu_type)
        if gpu_type == "RTX 5090":
            raise RuntimeError("first cleanup failed")

    monkeypatch.setattr(serverless, "terminate_endpoint", terminate)

    runpod.terminate_persisted_endpoints(
        {"gpu": {"type": ["RTX 5090", "A100 PCIe"]}}, "flash-cleanup"
    )
    assert calls == ["RTX 5090", "A100 PCIe"]


def test_effective_spec_carries_the_allocated_card_count(orch) -> None:
    base = _spec(type="")

    # the allocator can satisfy a run with n cards of a smaller class; the count it chose has to reach
    # the spec, because the worker sizes its rank count from gpu.count and the payload rents gpu.count.
    combo = runner_lifecycle._spec_with_gpu(base, "A100 PCIe", 4)
    assert (combo.gpu.type, combo.gpu.count) == ("A100 PCIe", 4)
    assert base.gpu.count == 1

    # omitted/zero count preserves the spec's own count (the historical single-card call shape).
    single = runner_lifecycle._spec_with_gpu(base, "A100 PCIe")
    assert single.gpu.count == 1


def test_terminal_race_before_effective_spec_persistence_skips_provider(orch, monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core.base import PollResult
    from flash.providers.runpod.execution import job_execution as rp_jobs

    spec = _spec(max_retries=0)
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict())
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc())

    original_spec_with_gpu = runner_lifecycle._spec_with_gpu

    def cancel_after_allocation(run_spec, gpu_type, gpu_count=0):
        selected = original_spec_with_gpu(run_spec, gpu_type, gpu_count)
        assert runner_status._update(run_spec.run_id, "cancelled")
        return selected

    provider_calls = []

    def fake_runpod_submit(*args, **kwargs):
        provider_calls.append((args, kwargs))
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(runner_lifecycle, "_spec_with_gpu", cancel_after_allocation)
    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)

    with pytest.raises(runner_errors._RunCancelled):
        runner_lifecycle._submit_seed_supervised(
            spec, spec.seed, io.StringIO(), source_snapshot=_SOURCE_SNAPSHOT
        )

    status = runner_status.get_status(spec.run_id)
    assert status.state == "cancelled"
    assert status.effective_preparation is None
    assert not runner_submit._persist_effective_worker_spec(spec)
    assert provider_calls == []


@pytest.mark.parametrize("first_revocation_fails", [False, True])
def test_cancel_waits_for_durable_provider_handle_then_tears_down(
    orch, monkeypatch, first_revocation_fails
):
    from flash.providers.core import allocator
    from flash.providers.core.base import PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs
    from flash.server.platform import db as server_db

    spec = _spec(run_id="flash-provider-handshake-cancel", max_retries=0)
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict())
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc())
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda *a, **k: None)

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
        persisted_status = runner_status.get_status(spec.run_id)
        persisted_remote = persisted_status.remote
        assert persisted_remote["endpoint_id"] == "ep-handshake"
        assert persisted_remote["job_id"] == "job-handshake"
        assert persisted_status.lifecycle_started_attempt == 0
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
            runner_lifecycle._submit_seed_supervised(
                spec, spec.seed, io.StringIO(), source_snapshot=_SOURCE_SNAPSHOT
            )
        except Exception as exc:
            submit_errors.append(exc)

    cancel_results = []
    cancel_errors = []

    def cancel():
        cancellation_started.set()
        try:
            cancel_results.append(runner_deploy.cancel_run(spec.run_id))
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
    waiting_status = runner_status.get_status(spec.run_id)
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
    status = runner_status.get_status(spec.run_id)
    assert status.state == "cancelled"
    assert status.remote is None
    assert cancelled_handles == [("ep-handshake", "job-handshake")]
    assert "ep-handshake" in destroyed_handles
    assert not resource_live["value"]

    allow_poll.set()
    submit_thread.join(timeout=5)
    assert not submit_thread.is_alive()
    assert len(submit_errors) == 1
    assert isinstance(submit_errors[0], runner_errors._RunCancelled)
    assert not resource_live["value"]


def test_concurrent_supervisors_preserve_first_effective_spec_and_provider(orch, monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core.base import PollResult
    from flash.providers.runpod.execution import job_execution as rp_jobs

    spec = _spec(run_id="flash-concurrent-supervisors", max_retries=0)
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict())
    )

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
            results[name] = runner_lifecycle._submit_seed_supervised(
                spec, spec.seed, io.StringIO(), source_snapshot=_SOURCE_SNAPSHOT
            )
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
    assert isinstance(results["second"], runner_errors._RunCancelled)
    assert provider_gpus == ["RTX 4090"]

    status = runner_status.get_status(spec.run_id)
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
    from flash.providers.core import allocator
    from flash.providers.core.base import PollResult
    from flash.providers.runpod.execution import job_execution as rp_jobs
    from flash.server.platform.locks import _deploy_lock

    spec = _spec(run_id=f"flash-lock-release-{failure_mode}", max_retries=0)
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict())
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc())

    original_update = runner_status._update

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

    monkeypatch.setattr(runner_status, "_update", fail_remote_persistence)
    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)

    if failure_mode == "provider_without_callback":
        metrics = runner_lifecycle._submit_seed_supervised(
            spec, spec.seed, io.StringIO(), source_snapshot=_SOURCE_SNAPSHOT
        )
        assert metrics["train_tokens"] == 4096
    else:
        with pytest.raises(RuntimeError, match="failed after retries"):
            runner_lifecycle._submit_seed_supervised(
                spec, spec.seed, io.StringIO(), source_snapshot=_SOURCE_SNAPSHOT
            )

    lock = _deploy_lock(spec.run_id)
    assert lock.acquire(blocking=False)
    lock.release()


def test_sync_submit_persists_resolved_env_sha_before_provider_submission(orch, monkeypatch):
    from dataclasses import replace

    import flash.core.catalog as catalog
    import flash.envs.loading.loader as env_loader
    from flash.providers.core import allocator
    from flash.providers.core.base import PollResult
    from flash.providers.runpod.execution import job_execution as rp_jobs
    from flash.runner.supervise import lifecycle

    resolved_sha = "a" * 40
    resolved_refs = []
    quote_allocations = []
    submitted = []

    def fake_resolve(parsed, *args, **kwargs):
        resolved_refs.append(parsed.canonical())
        return resolved_sha

    resolved_model_sha = "b" * 40
    monkeypatch.setattr(
        runner_preparation,
        "_resolve_model_revision",
        lambda spec, **_kwargs: replace(
            spec, model_revision=resolved_model_sha, model_revision_auto=True
        ),
    )
    monkeypatch.setattr(
        catalog,
        "resolve_model",
        lambda model, *args, **kwargs: catalog.MODELS[model],
    )

    def fake_estimate(_spec, *, allocation=None):
        quote_allocations.append(allocation)
        return type("Estimate", (), {"total_usd": 1.0})()

    def fake_runpod_submit(run_spec, seed, **kwargs):
        status = runner_status.get_status(run_spec.run_id)
        persisted = status.effective_preparation["worker_spec"]
        assert status.estimated_cost_usd == 1.0
        assert quote_allocations == [None]
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
    monkeypatch.setattr(runner_artifacts, "stage_environment_package", lambda spec, **_kwargs: spec)
    monkeypatch.setattr(
        provider_worker, "publish_source_snapshot", lambda _repo=None: _SOURCE_SNAPSHOT
    )
    monkeypatch.setattr(
        runner_status,
        "validate_terminal_source_metrics",
        lambda _status, metrics, expected_attempt=None: (metrics, expected_attempt),
    )
    monkeypatch.setattr(runner_status, "_persist_metrics", lambda *a, **k: 0.0)
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda spec: None)
    monkeypatch.setattr(lifecycle, "_register_checkpoints_best_effort", lambda *a, **k: None)

    public = _public_spec()
    # stub the static dataset estimate against the two revisions submission resolves above, not
    # against the helper's stand-ins.
    record_sft_profile(
        replace(
            public,
            model_revision=resolved_model_sha,
            environment=replace(public.environment, resolved_sha=resolved_sha),
        ),
        monkeypatch,
    )

    status = runner_submit.submit_job(public)

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
    stored = runner_status.get_status(public.run_id)
    # resolved identities are platform-managed: they stay on the internal worker spec only.
    assert "resolved_sha" not in stored.spec["environment"]
    assert "model_revision" not in stored.spec
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
    import flash.envs.loading.loader as env_loader
    from flash.providers.core import allocator

    persisted = []

    def blip(_parsed, *_args, **_kwargs):
        raise RuntimeError("github rate limit at submit time")

    from dataclasses import replace

    monkeypatch.setattr(env_loader, "_resolve_ref_sha", blip)
    monkeypatch.setattr(
        runner_preparation,
        "_resolve_model_revision",
        lambda spec, **_kw: replace(spec, model_revision="b" * 40, model_revision_auto=True),
    )
    monkeypatch.setattr(runner_state, "_save_status", lambda *a, **k: persisted.append(a))
    monkeypatch.setattr(
        allocator, "allocate", lambda *a, **k: pytest.fail("allocated without a profile")
    )

    with pytest.raises(
        runner_preparation.WorkloadProfileUnavailable, match="pinned environment package revision"
    ):
        runner_submit.submit_job(_public_spec())

    assert persisted == []


def _permanent_404(_parsed, *_args, **_kwargs):
    from flash.envs.meta.identity import GitHubPermanentError

    raise GitHubPermanentError("GitHub environment request failed (404): Not Found")


@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("algorithm", ["grpo", "opd"])
def test_permanent_env_404_is_refused_before_a_gpu_is_allocated(
    orch, monkeypatch, algorithm, dry_run
):
    """A nonexistent environment repo must fail AT SUBMIT, not on a rented worker.

    grpo and opd keep the best-effort pin, so a 404 used to be swallowed exactly like a rate-limit
    blip: `--dry-run` answered `state: dry_run, error: null`, and a real submit allocated a GPU that
    existed only to rediscover the 404 the control plane already had. The dry-run case is the point
    of the parametrisation -- that mode's entire job is to answer this without paying.
    """
    import flash.envs.loading.loader as env_loader
    from flash.providers.core import allocator

    monkeypatch.setattr(env_loader, "_github_token", lambda: "ghp_test")
    monkeypatch.setattr(env_loader, "_resolve_ref_sha", _permanent_404)
    monkeypatch.setattr(
        allocator, "allocate", lambda *a, **k: pytest.fail("allocated a GPU for a 404 environment")
    )

    with pytest.raises(
        runner_preparation.EnvironmentRefNotFound, match="could not be resolved on GitHub"
    ):
        runner_submit.submit_job(_public_spec(algorithm=algorithm), dry_run=dry_run)


def test_unauthenticated_404_still_defers_rather_than_refusing(orch, monkeypatch):
    """Without a token a 404 is not evidence of a typo, so it must not refuse the run.

    GitHub answers 404 both for a repo that does not exist and for a private one the caller may not
    see. Every managed-hub environment is private, so a tokenless plane 404s on refs that are
    perfectly valid -- refusing those would ground legitimate runs to catch a typo.
    """
    import flash.envs.loading.loader as env_loader
    from flash.providers.core import allocator

    monkeypatch.setattr(env_loader, "_github_token", lambda: None)
    monkeypatch.setattr(env_loader, "_resolve_ref_sha", _permanent_404)
    monkeypatch.setattr(runner_lifecycle, "_run_job", lambda *a, **k: None)
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc())

    status = runner_submit.submit_job(_public_spec(algorithm="grpo"), dry_run=True)

    assert status.state == "dry_run"


@pytest.mark.parametrize("blip_kind", ["rate-limit", "unavailable", "untyped"])
def test_transient_github_failure_still_defers_the_pin(orch, monkeypatch, blip_kind):
    """The 404 gate must not turn a blip into a refused submit.

    The deferral above this is deliberate: grpo and opd have no profile keyed on the pin, the worker
    resolves the ref itself, and the lifecycle fallback pins it on recovery. Only a PERMANENT answer
    changes that, so every transient class -- and anything untyped, which is unproven -- keeps
    reaching allocation.
    """
    import flash.envs.loading.loader as env_loader
    from flash.envs.meta.identity import GitHubRateLimitError, GitHubUnavailableError
    from flash.providers.core import allocator

    blip = {
        "rate-limit": GitHubRateLimitError("GitHub API rate limit exceeded (429)"),
        "unavailable": GitHubUnavailableError("GitHub server error (503, transient)"),
        "untyped": RuntimeError("some other github failure"),
    }[blip_kind]

    def raise_blip(_parsed, *_args, **_kwargs):
        raise blip

    allocated = []
    monkeypatch.setattr(env_loader, "_resolve_ref_sha", raise_blip)
    monkeypatch.setattr(
        runner_lifecycle, "_run_job", lambda *a, **k: allocated.append("submitted") or None
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc())

    runner_submit.submit_job(_public_spec(algorithm="grpo"))

    assert allocated == ["submitted"], "a transient blip must still defer the pin and submit"


def test_the_gate_keeps_the_sha_it_resolved_instead_of_resolving_twice(orch, monkeypatch):
    """The 404 check and the pin are the same GitHub call, so it must be made once.

    Both ask ``/repos/{repo}/commits/{ref}``. Resolving for the gate, discarding the answer, then
    resolving again for the pin doubles this submit's spend against the secondary rate limit the
    pin exists to protect -- and the two calls can disagree, since a ref can move between them.
    """
    import flash.envs.loading.loader as env_loader
    from flash.providers.core import allocator

    sha = "a" * 40
    calls = []

    def counting_resolve(_parsed, *_args, **_kwargs):
        calls.append(1)
        return sha

    submitted = []
    monkeypatch.setattr(env_loader, "_github_token", lambda: "ghp_test")
    monkeypatch.setattr(env_loader, "_resolve_ref_sha", counting_resolve)
    monkeypatch.setattr(
        runner_lifecycle, "_run_job", lambda spec, **k: submitted.append(spec) or None
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc())

    runner_submit.submit_job(_public_spec(algorithm="grpo"))

    assert len(calls) == 1, f"the env ref was resolved {len(calls)} times, expected once"
    assert submitted
    assert submitted[0].environment.resolved_sha == sha


@pytest.mark.parametrize(
    ("github_error", "expected"),
    [
        pytest.param(
            'GitHub environment request failed (422): {"message":"No commit found for SHA: main"}',
            "No commit found for SHA: main",
            id="wrong-ref",
        ),
        pytest.param(
            "GitHub API rate limit exceeded (403)",
            "rate limit",
            id="rate-limit",
        ),
        pytest.param(
            'GitHub environment request failed (404): {"message":"Not Found"}',
            "404",
            id="private-or-missing",
        ),
    ],
)
def test_unpinnable_sft_environment_reports_githubs_own_diagnosis(
    orch, monkeypatch, github_error, expected
):
    """The pin failure must name WHY, not just that a revision is missing.

    `@main` against a master-default repo made GitHub answer "No commit found for SHA: main" -- the
    ref, the cause and the fix in one line. The plane had that string and threw it away, reporting
    only "sft workload profiling requires a pinned environment package revision", which names none
    of the three. A rate limit, an outage and a private repo the token cannot read all produce the
    identical missing pin and all need different fixes, so the generic text sends every one of them
    to the wrong remedy.
    """
    from dataclasses import replace

    import flash.envs.loading.loader as env_loader
    from flash.providers.core import allocator

    def fail(_parsed, *_args, **_kwargs):
        raise RuntimeError(github_error)

    monkeypatch.setattr(env_loader, "_resolve_ref_sha", fail)
    monkeypatch.setattr(
        runner_preparation,
        "_resolve_model_revision",
        lambda spec, **_kw: replace(spec, model_revision="b" * 40, model_revision_auto=True),
    )
    monkeypatch.setattr(
        allocator, "allocate", lambda *a, **k: pytest.fail("allocated without a profile")
    )

    with pytest.raises(runner_preparation.WorkloadProfileUnavailable) as excinfo:
        runner_submit.submit_job(_public_spec())

    message = str(excinfo.value)
    assert expected in message, message
    # the ref itself has to appear too: the operator has to know WHICH environment to fix.
    assert "github:owner/repo@main:env/environment.py" in message


def test_the_reported_reason_describes_the_resolve_that_was_actually_used(orch, monkeypatch):
    """One resolve, so a retry that would have succeeded cannot blank out the reason.

    Diagnosing by re-resolving is wrong twice over. A transient cause (a rate-limit window that
    resets, a blip that clears) can succeed on the second call, and the caller -- already committed
    to rejecting on the FIRST result -- would then report no reason at all, landing back on the
    generic message this change exists to replace. It also doubles GitHub calls on every failing
    submit, against the same rate limit that is one of the causes.
    """
    from dataclasses import replace

    import flash.envs.loading.loader as env_loader
    from flash.providers.core import allocator

    calls = []

    def flaky(_parsed, *_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("GitHub API rate limit exceeded (403)")
        return "b" * 40  # a second attempt SUCCEEDS; the rejection is already decided

    monkeypatch.setattr(env_loader, "_resolve_ref_sha", flaky)
    monkeypatch.setattr(
        runner_preparation,
        "_resolve_model_revision",
        lambda spec, **_kw: replace(spec, model_revision="b" * 40, model_revision_auto=True),
    )
    monkeypatch.setattr(
        allocator, "allocate", lambda *a, **k: pytest.fail("allocated without a profile")
    )

    with pytest.raises(runner_preparation.WorkloadProfileUnavailable) as excinfo:
        runner_submit.submit_job(_public_spec())

    assert "rate limit" in str(excinfo.value), str(excinfo.value)
    assert calls == [1], f"the pin must be resolved exactly once, got {len(calls)} calls"


def test_controller_staging_is_persisted_before_provider_submission(orch, monkeypatch):
    from dataclasses import replace

    import flash.core.catalog as catalog
    from flash.core.spec import EnvironmentPackageSpec
    from flash.providers.core import allocator
    from flash.providers.core.base import PollResult
    from flash.providers.runpod.execution import job_execution as rp_jobs
    from flash.runner.supervise import lifecycle

    resolved_sha = "e" * 40
    package = EnvironmentPackageSpec(
        artifact_revision="f" * 40,
        archive_sha256="1" * 64,
        manifest_sha256="2" * 64,
    )
    persisted_at_submission = []

    monkeypatch.setattr(
        runner_preparation,
        "_resolve_model_revision",
        lambda spec, **_kw: replace(spec, model_revision="b" * 40, model_revision_auto=True),
    )
    monkeypatch.setattr(catalog, "resolve_model", lambda model, *a, **k: catalog.MODELS[model])
    monkeypatch.setattr(
        runner_artifacts,
        "preflight_validate_environment_ref",
        lambda spec: (spec, True),
    )

    def fake_stage(spec, **_kwargs):
        return replace(
            spec,
            environment=replace(
                spec.environment,
                resolved_sha=resolved_sha,
                package=package,
            ),
        )

    def fake_runpod_submit(run_spec, seed, **kwargs):
        persisted = runner_status.get_status(run_spec.run_id).effective_preparation["worker_spec"]
        persisted_at_submission.append(persisted["environment"])
        return PollResult(True, metrics={"train_tokens": 4096, "wall_seconds": 1})

    monkeypatch.setattr(
        "flash.cost.spec.estimate_for_spec",
        lambda _spec, **_kwargs: type("Estimate", (), {"total_usd": 1.0})(),
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(gpu="RTX 5090"))
    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)
    monkeypatch.setattr(runner_artifacts, "stage_environment_package", fake_stage)
    monkeypatch.setattr(
        provider_worker, "publish_source_snapshot", lambda _repo=None: _SOURCE_SNAPSHOT
    )
    monkeypatch.setattr(
        runner_status,
        "validate_terminal_source_metrics",
        lambda _status, metrics, expected_attempt=None: (metrics, expected_attempt),
    )
    monkeypatch.setattr(runner_status, "_persist_metrics", lambda *a, **k: 0.0)
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda spec: None)
    monkeypatch.setattr(lifecycle, "_register_checkpoints_best_effort", lambda *a, **k: None)

    public = _public_spec(algorithm="grpo")
    runner_submit.submit_job(public)

    assert len(persisted_at_submission) == 1
    persisted_environment = persisted_at_submission[0]
    assert persisted_environment["id"] == public.environment.id
    assert persisted_environment["resolved_sha"] == resolved_sha
    assert persisted_environment["package"] == {
        "artifact_revision": package.artifact_revision,
        "archive_sha256": package.archive_sha256,
        "manifest_sha256": package.manifest_sha256,
    }
    restarted = runner_status.get_status(public.run_id)
    recovered = runner_status.reallocation_spec_from_status(restarted, verify_source=True)
    assert recovered.environment.resolved_sha == resolved_sha
    assert recovered.environment.package == package


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
        runner_preparation._validate_effective_spec(public, worker)


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
        runner_preparation._validate_effective_spec(public, worker)


@pytest.mark.parametrize(
    "resolved_sha",
    [
        pytest.param("", id="removal"),
        pytest.param("b" * 40, id="replacement"),
    ],
)
def test_effective_spec_rejects_changing_existing_resolved_env_sha(resolved_sha):
    from dataclasses import replace

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
        runner_preparation._validate_effective_spec(public, worker)


def test_runpod_cost_projection_flows_into_run_status(orch, monkeypatch):
    spec = _spec()
    _seed_status(orch, spec)
    cost = runner_status._persist_metrics(
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

    four_card = runner_status._persist_metrics(spec, {**base, "allocated_gpu_count": 4})

    assert four_card == pytest.approx(0.345 * 4), "a 4-card run was priced as one card"
    # a record predating the stamp still reads as one card rather than zero or a crash
    assert runner_status._persist_metrics(spec, dict(base)) == pytest.approx(0.345)


def test_the_allocated_card_count_reaches_the_metrics_the_cost_is_read_from(orch, monkeypatch):
    """The multiply above is inert unless allocation actually stamps the count it chose.

    this drives the real submit path rather than handing `_persist_metrics` a literal, because the
    two halves fail independently: pricing can multiply correctly by a count that is never recorded.
    """
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, PollResult
    from flash.providers.runpod.execution import job_execution as rp_jobs

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

    metrics = runner_lifecycle._submit_seed_supervised(
        spec, spec.seed, io.StringIO(), source_snapshot=_SOURCE_SNAPSHOT
    )

    assert metrics["allocated_gpu_count"] == 4
    # and the stamp is what pricing then reads, so the two halves compose
    assert runner_status._persist_metrics(spec, metrics) == pytest.approx(0.345 * 4)


def test_infra_retry_walks_to_next_runpod_class_and_deletes_endpoint(orch, monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs

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
    metrics = runner_lifecycle._submit_seed_supervised(
        spec, spec.seed, log, source_snapshot=_SOURCE_SNAPSHOT
    )
    assert metrics["train_tokens"] == 4096
    assert submitted_gpus == ["RTX 4090", "H100"]
    assert cancelled == [("ep1", "j1")]
    assert "ep1" in deleted
    assert "walking past the cheapest class" in log.getvalue()


def test_ordered_pin_stops_once_the_class_it_would_reuse_has_refused_twice(orch, monkeypatch):
    """A fixed multi-class market stops when the projected class refuses twice."""
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs

    candidates = (
        Candidate("runpod", "A100 PCIe", 1.19, 80),
        Candidate("runpod", "A100 SXM", 1.89, 80),
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    monkeypatch.setattr(runpod_api, "cancel_job", lambda *a, **k: None)
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", lambda *_a, **_k: True)

    submitted_gpus = []

    def fake_runpod_submit(run_spec, seed, log=None, on_handle=None, attempt=0, **kwargs):
        submitted_gpus.append(run_spec.gpu.type)
        on_handle(_runpod_handle(f"ep{attempt}", f"j{attempt}", attempt))
        return PollResult(False, failure="no_capacity", detail="dry")

    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)
    spec = _spec(
        run_id="flash-ordered-pin-stop",
        type="A100 PCIe",
        type_fallbacks=("A100 SXM",),
        provider="runpod",
        max_retries=5,
    )
    _seed_status(orch, spec)
    log = io.StringIO()
    with pytest.raises(RuntimeError, match="failed after retries"):
        runner_lifecycle._submit_seed_supervised(
            spec, spec.seed, log, source_snapshot=_SOURCE_SNAPSHOT
        )

    # pcie, sxm, then pcie again -- and there it stops, because that third submission is pcie's
    # second refusal and pcie is where the retry keeps landing. not the budget's five: the two
    # attempts saved are 900s of capacity grace each. every named class still got a look first,
    # which is what separates this from writing a run off on one refusal.
    assert submitted_gpus == ["A100 PCIe", "A100 SXM", "A100 PCIe"]
    out = log.getvalue()
    assert "has already refused capacity twice" in out
    assert "drop the gpu.type pin" in out


def test_a_named_alternative_still_gets_its_own_look_after_the_first_class_refuses(
    orch, monkeypatch
):
    """Each named class gets one refusal before a repeated class is stopped."""
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs

    candidates = (
        Candidate("runpod", "A100 PCIe", 1.19, 80),
        Candidate("runpod", "A100 SXM", 1.89, 80),
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    monkeypatch.setattr(runpod_api, "cancel_job", lambda *a, **k: None)
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", lambda *_a, **_k: True)

    submitted_gpus = []

    def fake_runpod_submit(run_spec, seed, log=None, on_handle=None, attempt=0, **kwargs):
        submitted_gpus.append(run_spec.gpu.type)
        on_handle(_runpod_handle(f"ep{attempt}", f"j{attempt}", attempt))
        # both classes dry on their first look; pcie rentable when the retry returns to it.
        if run_spec.gpu.type == "A100 PCIe" and submitted_gpus.count("A100 PCIe") > 1:
            return PollResult(True, metrics={"train_tokens": 4096})
        return PollResult(False, failure="no_capacity", detail="dry")

    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)
    spec = _spec(
        run_id="flash-per-class-margin",
        type="A100 PCIe",
        type_fallbacks=("A100 SXM",),
        max_retries=5,
    )
    _seed_status(orch, spec)
    log = io.StringIO()
    metrics = runner_lifecycle._submit_seed_supervised(
        spec, spec.seed, log, source_snapshot=_SOURCE_SNAPSHOT
    )

    # three submissions, not the two a membership set would have allowed: the run survives one
    # refusal from each class and rents pcie on the look the tally kept alive.
    assert metrics["train_tokens"] == 4096
    assert submitted_gpus == ["A100 PCIe", "A100 SXM", "A100 PCIe"]
    assert "has already refused capacity twice" not in log.getvalue()


def test_unconfirmed_runpod_teardown_retains_handle_and_blocks_retry(orch, monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs
    from flash.providers.runpod.execution.provider import RunpodProvider

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
        runner_lifecycle._submit_seed_supervised(
            spec, spec.seed, log, source_snapshot=_SOURCE_SNAPSHOT
        )

    assert submitted_attempts == [0]
    assert deleted_endpoints
    assert set(deleted_endpoints) == {"ep-unconfirmed"}
    assert gc_calls == [spec.run_id]
    status = runner_status.get_status(spec.run_id)
    assert status.remote["endpoint_id"] == "ep-unconfirmed"
    assert status.remote["job_id"] == "job-unconfirmed"
    assert "teardown unconfirmed" in log.getvalue()


def _oom_candidates():
    from flash.providers.core.base import Candidate

    return (
        Candidate("runpod", "A100 PCIe", 1.0, 80),
        Candidate("runpod", "RTX Pro 6000", 2.0, 96),
        Candidate("runpod", "B200", 3.0, 180),
    )


def _run_failed_oom_sequence(orch, monkeypatch, failures, *, max_retries):
    from flash.providers.core import allocator
    from flash.providers.core.base import PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs

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
        runner_lifecycle._submit_seed_supervised(
            spec, spec.seed, io.StringIO(), source_snapshot=_SOURCE_SNAPSHOT
        )
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
    from flash.providers.core.base import Candidate
    from flash.runner.supervise.lifecycle import _select_candidate

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


def test_select_candidate_escapes_a_failed_preferred_provider():
    from flash.providers.core.base import Candidate
    from flash.runner.supervise.lifecycle import _select_candidate

    # the allocator placed the preferred provider first even though vast was cheaper. retry must
    # still demote the failed provider before preserving that incoming preference-ranked order.
    ranked = (
        Candidate("runpod", "H100", 3.00, 80),
        Candidate("vast", "H100", 0.50, 80),
    )

    chosen = _select_candidate(ranked, {"runpod"}, {("runpod", "H100", 1)})

    assert chosen is ranked[1]


def test_select_candidate_keeps_the_allocators_per_step_ranking():
    """The picker must take the allocator's order, not re-price the list by hourly rate.

    ``allocate()`` ranks on the dollars one optimizer step costs, so a faster card can rank first
    while costing more per hour. a measured opd case ranks the $0.99/hr rtx 5090 ahead of the
    $0.69/hr rtx 4090. re-sorting here on total $/hr overrode that on the first paid attempt,
    running the slower card for more total money. Ordering is the allocator's job; this picker only
    demotes failed providers and tried shapes.
    """
    from flash.providers.core.base import Candidate
    from flash.runner.supervise.lifecycle import _select_candidate

    # as allocate() returns them: cheapest PER STEP first, which is NOT cheapest per hour.
    ranked = (
        Candidate("runpod", "RTX 5090", 0.99, 32),
        Candidate("runpod", "RTX 4090", 0.69, 24),
    )
    assert _select_candidate(ranked, set(), set()) is ranked[0]

    # the demotion keys still outrank the allocator's order: once the per-step winner's shape has
    # been tried, the walk moves on rather than re-picking it.
    chosen = _select_candidate(ranked, set(), {("runpod", "RTX 5090", 1)})
    assert chosen.gpu == "RTX 4090"


def test_select_candidate_single_provider_walks_classes():
    """With only one provider configured, the picker degrades to the cheapest untried class."""
    from flash.providers.core.base import Candidate
    from flash.runner.supervise.lifecycle import _select_candidate

    cands = (Candidate("runpod", "RTX 4090", 0.39, 24), Candidate("runpod", "H100", 0.49, 48))
    assert _select_candidate(cands, {"runpod"}, {("runpod", "RTX 4090", 1)}).gpu == "H100"


def test_select_candidate_single_fitting_gpu_never_breaks():
    """A large model with exactly ONE fitting class (e.g. only H200 fits a 35B run) must keep
    re-picking THAT class on every infra retry — the candidate list only ever holds *fitting*
    classes, so the walk can never escape to a card too small to hold the model, and the picker
    must still return it (not None / not raise) even after it's been tried and its provider burned."""
    from flash.providers.core.base import Candidate
    from flash.runner.supervise.lifecycle import _select_candidate

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
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, PollResult
    from flash.providers.lambda_ import jobs as lambda_jobs
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs

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
    metrics = runner_lifecycle._submit_seed_supervised(
        spec, spec.seed, log, source_snapshot=_SOURCE_SNAPSHOT
    )
    assert metrics["train_tokens"] == 4096
    assert rp_gpus == ["H100"]  # RunPod tried exactly once...
    assert lam_gpus == ["H100"]  # ...then the retry escaped cross-provider to Lambda
    assert runner_status.get_status(spec.run_id).remote["provider"] == "lambda"
    assert "walking past the cheapest class" in log.getvalue()


@pytest.mark.parametrize("failure", ["no_capacity", "poll_error"])
def test_shared_cache_zero_retries_submits_exactly_once(orch, monkeypatch, failure):
    """max_retries=0 is one provider submission even with the managed shared cache."""
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME

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
        runner_lifecycle._submit_seed_supervised(
            spec, spec.seed, io.StringIO(), source_snapshot=_SOURCE_SNAPSHOT
        )
    assert volumes_seen == [WEIGHT_CACHE_VOLUME_NAME]


def test_cache_fallback_does_not_consume_gpu_walk_retry(orch, monkeypatch):
    """The cache-drop fallback is a FREE attempt — it must not spend the user's GPU-walk retry budget.
    With max_retries=1 and both the cache attempt AND the cache-less same-class fallback hitting
    no_capacity, the run must still walk to a DIFFERENT class (its one real retry), not stop after the
    cache drop. Regression for the cache-fallback-steals-the-only-retry gap (the stop check counted the
    cache-drop attempt against max_retries)."""
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME

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
    metrics = runner_lifecycle._submit_seed_supervised(
        spec, spec.seed, io.StringIO(), source_snapshot=_SOURCE_SNAPSHOT
    )
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
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, PollResult
    from flash.providers.lambda_ import jobs as lambda_jobs
    from flash.providers.lambda_.client import api as lambda_api
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs

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
    metrics = runner_lifecycle._submit_seed_supervised(
        spec, spec.seed, log, source_snapshot=_SOURCE_SNAPSHOT
    )
    assert metrics["train_tokens"] == 4096
    assert lam_gpus == ["A10"]  # broken Lambda instance tried once...
    assert rp_gpus == ["H100"]  # ...then escaped cross-provider to RunPod
    assert "i-broken" in terminated  # sick instance torn down before the retry
    assert runner_status.get_status(spec.run_id).remote["provider"] == "runpod"


def test_no_liveness_stalled_escapes_to_other_provider(orch, monkeypatch):
    """The new fast first-liveness failover returns failure='stalled' (a sick region where the box
    reached 'active' but the worker never booted). It is infra-shaped, so the retry ESCAPES to a
    different provider rather than re-rolling the same sick substrate — the observed us-east-1 /
    CANADA-1 case, now caught in ~15 min instead of ~50."""
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, PollResult
    from flash.providers.lambda_ import jobs as lambda_jobs
    from flash.providers.lambda_.client import api as lambda_api
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs

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
    metrics = runner_lifecycle._submit_seed_supervised(
        spec, spec.seed, io.StringIO(), source_snapshot=_SOURCE_SNAPSHOT
    )
    assert metrics["train_tokens"] == 4096
    assert lam_gpus == ["H100"]  # sick region tried once...
    assert rp_gpus == ["H100"]  # ...then escaped cross-provider to RunPod
    assert runner_status.get_status(spec.run_id).remote["provider"] == "runpod"


def test_genuine_worker_error_does_not_retry(orch, monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core.base import PollResult
    from flash.providers.runpod.execution import job_execution as rp_jobs

    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc())
    calls = []

    def fake_runpod_submit(run_spec, seed, log=None, on_handle=None, attempt=0, **kwargs):
        calls.append(attempt)
        return PollResult(False, failure="job_failed", detail="ValueError: bad reward fn")

    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)
    spec = _spec()
    _seed_status(orch, spec)
    with pytest.raises(RuntimeError, match="bad reward fn"):
        runner_lifecycle._submit_seed_supervised(
            spec, spec.seed, io.StringIO(), source_snapshot=_SOURCE_SNAPSHOT
        )
    assert calls == [0]


def test_cancel_rejects_legacy_handle_without_provider_identity(orch, monkeypatch):
    import flash.providers.runpod.serverless.endpoints as rp_train
    from flash.providers.runpod.client import api as runpod_api

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
    runner_state._save_status(st)
    with pytest.raises(RuntimeError, match="exact cleanup target could not be preserved"):
        runner_deploy.cancel_run(spec.run_id)
    assert cancelled_jobs == []
    assert deleted_eps == []
    assert runner_status.get_status(spec.run_id).remote == st.remote


def test_config_gpu_fields(monkeypatch):
    from flash.schema import spec_from_dict

    base = {
        "model": "Qwen/Qwen3.5-9B",
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
    log must describe the retry that actually happens. A stalled failure now exercises the same
    last-class retry message because ordinary exhausted capacity stops instead."""
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs

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
            return PollResult(False, failure="stalled", detail="worker stopped reporting progress")
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()
    runner_lifecycle._submit_seed_supervised(spec, spec.seed, log, source_snapshot=_SOURCE_SNAPSHOT)

    assert gpus == ["H200", "H200"]  # nowhere to walk: the same class is genuinely reused
    action = _retry_action_line(log.getvalue(), 0)
    assert "expecting to retry on H200 @ runpod again" in action, action
    assert "no untried GPU class fits this run" in action, action
    assert "next-best" not in action, "claimed an escalation that the picker cannot perform"
    # the next attempt re-allocates against live capacity, so this is a projection, not a promise.
    assert "the class may change" in action, action


def test_retry_message_admits_when_the_projected_provider_already_failed(orch, monkeypatch):
    """A retry that clamps back onto a provider this run already lost on is not a failover.

    ``_select_candidate`` escapes a failed provider only while some candidate has an unfailed one:
    the key is ``provider in failed_providers``, so once every candidate's provider has failed it is
    True for all of them and the escape degrades to plain cheapest-first. The line then reads as
    recovery while the run loops on the substrate that is failing -- the shape of a 46-attempt
    profile loop that printed a failover target it never acted on. The operator's fix is another
    provider, not more waiting, so the message has to say the pool is exhausted. A stalled failure
    keeps the retry-message path active while preserving the same provider topology.
    """
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs

    # one provider, two fitting classes: every candidate's provider fails together.
    candidates = (
        Candidate("runpod", "A100 PCIe", 1.2, 40),
        Candidate("runpod", "A100 SXM", 1.8, 80),
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    monkeypatch.setattr(
        runpod_api,
        "cancel_job",
        lambda _endpoint_id, job_id, **_kw: {"id": job_id, "status": "CANCELLED"},
    )
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", lambda e, _fingerprint: True)

    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, **kw):
        on_handle(_runpod_handle("ep1", "j1", attempt))
        if attempt < 2:
            return PollResult(False, failure="stalled", detail="worker stopped reporting progress")
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec(max_retries=2)
    _seed_status(orch, spec)
    log = io.StringIO()
    runner_lifecycle._submit_seed_supervised(spec, spec.seed, log, source_snapshot=_SOURCE_SNAPSHOT)

    # attempt 0 fails runpod, and the only other candidate is also runpod, so the "failover" is
    # back onto the provider that just failed.
    action = _retry_action_line(log.getvalue(), 0)
    assert "already lost an attempt on" in action, action
    # every fitting candidate really is runpod here, so denying another provider is true.
    assert "no other provider offers a fitting class" in action, action


def test_retry_message_does_not_deny_a_provider_that_is_in_the_candidate_list(orch, monkeypatch):
    """Landing on a failed provider proves no UNFAILED one is left, not that none exists.

    With fitting candidates on two providers that have both failed, ``_select_candidate``'s first
    key is True for either, so the projection lands on a failed provider while another provider sits
    in the very list the message is derived from. Claiming none offers a fitting class would deny
    that provider and push the operator toward the wrong remedy -- the answer there is to unpin or
    wait for one of them to recover, not to go find a provider that is already on the list.
    """
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs
    from flash.providers.vast import jobs as vast_jobs

    # two providers, both of which this run burns through.
    candidates = (
        Candidate("runpod", "A100 PCIe", 1.2, 40),
        Candidate("vast", "A100 SXM", 1.8, 80),
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    monkeypatch.setattr(
        runpod_api,
        "cancel_job",
        lambda _endpoint_id, job_id, **_kw: {"id": job_id, "status": "CANCELLED"},
    )
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", lambda e, _fingerprint: True)

    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, **kw):
        on_handle(_runpod_handle("ep1", "j1", attempt))
        if attempt < 2:
            return PollResult(False, failure="no_capacity", detail="job stuck IN_QUEUE")
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    monkeypatch.setattr(
        vast_jobs,
        "submit_run_vast",
        lambda *_a, **_kw: PollResult(
            False,
            failure="no_capacity",
            detail="job stuck IN_QUEUE",
        ),
    )
    spec = _spec(max_retries=2)
    _seed_status(orch, spec)
    log = io.StringIO()
    runner_lifecycle._submit_seed_supervised(spec, spec.seed, log, source_snapshot=_SOURCE_SNAPSHOT)

    # attempt 1 projects back onto a failed provider while the other one is still in the list.
    action = _retry_action_line(log.getvalue(), 1)
    assert "already lost an attempt on" in action, action
    assert "no other provider offers a fitting class" not in action, action
    assert "has now failed" in action, action
    # and it names them rather than implying the list is empty.
    assert "runpod" in action, action
    assert "vast" in action, action


def test_a_genuine_cross_provider_failover_is_not_labelled_exhausted(orch, monkeypatch):
    """The admission must not fire when the retry really does escape to a different provider."""
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs

    candidates = (
        Candidate("runpod", "A100 PCIe", 1.2, 40),
        Candidate("lambda", "A100 PCIe", 1.5, 40),
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    monkeypatch.setattr(
        runpod_api,
        "cancel_job",
        lambda _endpoint_id, job_id, **_kw: {"id": job_id, "status": "CANCELLED"},
    )
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", lambda e, _fingerprint: True)

    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, **kw):
        on_handle(_runpod_handle("ep1", "j1", attempt))
        return PollResult(False, failure="no_capacity", detail="job stuck IN_QUEUE")

    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec(max_retries=1)
    _seed_status(orch, spec)
    log = io.StringIO()
    with pytest.raises(RuntimeError, match="failed after retries"):
        runner_lifecycle._submit_seed_supervised(
            spec, spec.seed, log, source_snapshot=_SOURCE_SNAPSHOT
        )

    action = _retry_action_line(log.getvalue(), 0)
    assert "@ lambda" in action, action
    assert "already lost an attempt on" not in action, action


def test_last_gpu_retry_message_names_the_clamped_back_class_not_the_current_one(orch, monkeypatch):
    """on_last_gpu means no UNTRIED class is left -- NOT that the current class is reused. With two
    fitting classes the walk is PCIe, SXM, then back to the cheaper PCIe, so the message printed on
    the SXM failure must name the PCIe the picker actually selects, not the SXM it is leaving. Stalls
    retain the full retry walk needed to exercise that clamp-back message."""
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs

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
            return PollResult(False, failure="stalled", detail="worker stopped reporting progress")
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec(max_retries=2)
    _seed_status(orch, spec)
    log = io.StringIO()
    runner_lifecycle._submit_seed_supervised(spec, spec.seed, log, source_snapshot=_SOURCE_SNAPSHOT)

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
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME

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
    runner_lifecycle._submit_seed_supervised(spec, spec.seed, log, source_snapshot=_SOURCE_SNAPSHOT)

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
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME

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
            from flash.providers.runpod.execution.jobs import capacity_escalation_note

            note = capacity_escalation_note(on_last_gpu)
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
    runner_lifecycle._submit_seed_supervised(spec, spec.seed, log, source_snapshot=_SOURCE_SNAPSHOT)

    block = _retry_block(log.getvalue(), 0)
    assert "expecting to retry on H100 @ runpod again" in block, block
    # the whole point: nothing in this block may promise a walk the supervisor is not making. scoped
    # to the block, since the allocation summary outside it names the next-best CANDIDATE legitimately.
    assert "next-best" not in block, block
    assert "GPU-class escalation may follow" in block, block


def test_projected_retry_class_is_worded_as_a_projection_not_a_promise(orch, monkeypatch):
    """The projection reads the CURRENT candidate list, but the next attempt calls allocate() again.
    Providers that rebuild candidates from live capacity can drop the named class or surface a
    cheaper one, so the log must not claim this is the class the retry will certainly use. A stalled
    failure keeps this projection observable after exhausted capacity became terminal."""
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs

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
            return PollResult(False, failure="stalled", detail="worker stopped reporting progress")
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()
    runner_lifecycle._submit_seed_supervised(spec, spec.seed, log, source_snapshot=_SOURCE_SNAPSHOT)

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
    exists. Derive the clause from the sets the retry will actually see instead."""
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME

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
    runner_lifecycle._submit_seed_supervised(spec, spec.seed, log, source_snapshot=_SOURCE_SNAPSHOT)

    # the flag really is set here -- otherwise this test would pass for the wrong reason.
    assert seen_flags[0] is True, seen_flags
    action = _retry_action_line(log.getvalue(), 0)
    assert "expecting to retry on H100 @ runpod again" in action, action
    # the class the very next attempt reuses is still untried, so nothing may claim otherwise.
    assert "no untried GPU class fits this run" not in action, action


def test_sole_class_infra_retry_still_reports_exhaustion(orch, monkeypatch):
    """The complement of the cache-drop case: a plain infra retry DOES mark the class tried, so with
    one fitting class the clause is accurate and must survive. Guards against fixing the false
    positive by deleting the clause outright. A stalled failure exercises this ordinary infra retry.
    """
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs

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
            return PollResult(False, failure="stalled", detail="worker stopped reporting progress")
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec(max_retries=2)
    _seed_status(orch, spec)
    log = io.StringIO()
    runner_lifecycle._submit_seed_supervised(spec, spec.seed, log, source_snapshot=_SOURCE_SNAPSHOT)

    action = _retry_action_line(log.getvalue(), 0)
    assert "expecting to retry on H100 @ runpod again" in action, action
    assert "no untried GPU class fits this run" in action, action


def test_submit_supplies_the_worker_pip_when_the_author_declared_none() -> None:
    """``worker_pip_for_env`` is the baseline every worker needs to run a Freesolo environment.

    Shipping a worker with no Freesolo SDK fails only once a GPU is already rented, so the baseline
    must be present whether or not the author declared anything.
    """
    from flash.providers._lifecycle.instances.instance import build_payload

    spec = _spec()
    assert not spec.environment.pip

    payload = build_payload(
        spec,
        spec.seed,
        0,
        arm="a",
        source_snapshot=_SOURCE_SNAPSHOT,
        deadline_at=1_800_000_000.0,
    )

    assert payload["extra_pip"] == ["freesolo>=0.4.2"]


def test_submit_appends_authored_pip_without_displacing_the_worker_spec() -> None:
    """The author's scorer deps are additional to the worker requirement, never a replacement.

    Substituting instead of appending would drop ``freesolo>=0.4.2`` the moment anyone declared a
    dependency, breaking the worker outright for exactly the users the knob exists to serve.
    """
    from dataclasses import replace

    from flash.providers._lifecycle.instances.instance import build_payload

    spec = _spec()
    spec = replace(spec, environment=replace(spec.environment, pip=("pymongo>=4.6", "rapidfuzz")))

    payload = build_payload(
        spec,
        spec.seed,
        0,
        arm="a",
        source_snapshot=_SOURCE_SNAPSHOT,
        deadline_at=1_800_000_000.0,
    )

    assert payload["extra_pip"] == ["freesolo>=0.4.2", "pymongo>=4.6", "rapidfuzz"]


def test_worker_pip_with_extras_dedupes_and_ignores_blanks() -> None:
    """Restating the worker spec must not install it twice, and blanks must not reach pip."""
    from flash.envs.loading.base import worker_pip_with_extras

    assert worker_pip_with_extras("e", ["freesolo>=0.4.2", "pymongo"]) == [
        "freesolo>=0.4.2",
        "pymongo",
    ]
    assert worker_pip_with_extras("e", ["  pymongo  ", ""]) == ["freesolo>=0.4.2", "pymongo"]
    assert worker_pip_with_extras("e", None) == ["freesolo>=0.4.2"]


def _submit_failure(provider_obj):
    """Drive the real ``_submit_provider`` against a provider whose submit raises.

    Stubs only what stands between the call and the ``except`` clause under test: provider lookup,
    the teacher-secret transport, and the persisted run deadline. The handler itself is untouched.
    """
    import contextlib as _contextlib

    import flash.server.domain.teacher.broker as _broker
    from flash.providers.core import registry as _providers
    from flash.runner.supervise import seed_submission as _ss

    @_contextlib.contextmanager
    def _no_secrets(*_a, **_kw):
        yield {}

    saved = (
        _providers.get_provider,
        _broker.teacher_attempt_transport,
        runner_deadlines._load_run_deadline_at,
        runner_deadlines._worker_deadline_at,
    )
    _providers.get_provider = lambda _name: provider_obj
    _broker.teacher_attempt_transport = _no_secrets
    runner_deadlines._load_run_deadline_at = lambda _run_id: 1.0e12
    runner_deadlines._worker_deadline_at = lambda _run_id, _spec: 1.0e12
    try:
        spec = _spec()
        ctx = _ss._SubmitContext(
            spec=spec,
            seed=spec.seed,
            log=io.StringIO(),
            runtime_secrets={},
            source_snapshot=_SOURCE_SNAPSHOT,
            attempt_start=0,
            infra_budget=1,
            retry_budget=None,
            started_with_shared_cache=False,
        )
        prepared = _ss._PreparedAttempt(
            local_attempt=0, attempt=0, attempt_spec=spec, runtime_secrets={}
        )
        plan = _ss._CandidatePlan(
            allocation=None,
            candidates=(),
            chosen=_Chosen("vast", "RTX 4090"),
            on_last_gpu=False,
            effective_spec=spec,
            run_spec=spec,
        )
        return _ss._submit_provider(ctx, prepared, plan)
    finally:
        (
            _providers.get_provider,
            _broker.teacher_attempt_transport,
            runner_deadlines._load_run_deadline_at,
            runner_deadlines._worker_deadline_at,
        ) = saved


class _Chosen:
    def __init__(self, provider, gpu):
        self.provider = provider
        self.gpu = gpu
        self.gpu_count = 1


def _submit_failure_with_secret(exc):
    class _Boom:
        def submit_run(self, *_a, **_kw):
            raise exc

    return _submit_failure(_Boom())


def test_the_pool_exhaustion_message_survives_supervision():
    """The one submit-side message worth reading has to reach the operator, not just its class.

    Supervision replaced every submit exception with `provider submit failed (<ClassName>)`, which
    is the same line whether the market is dry or this run has burned every host in the class
    itself. Those have different operator fixes -- wait, versus another class or provider -- so an
    error that draws the distinction is worthless if it is discarded one frame later.
    """
    from flash.providers.core.base import RunExhaustedProviderPoolError

    class _Boom:
        def submit_run(self, *_a, **_kw):
            raise RunExhaustedProviderPoolError(
                "no usable vast offers for RTX 4090 outside the 3 machine(s) this run "
                "already rented and lost"
            )

    result, retriable = _submit_failure(_Boom())

    assert retriable
    assert result.failure == "no_capacity"
    assert "already rented and lost" in result.detail, result.detail


def test_generic_provider_exception_text_still_never_reaches_the_run_record():
    """Only the AUTHORED message is surfaced; provider text keeps its class-name-only treatment.

    A provider exception can quote a request body, and this detail is persisted. `sanitize_diagnostic`
    would not save it: that redacts CONFIGURED secrets, so arbitrary provider text passes straight
    through. Surfacing the authored error is safe precisely because Flash writes that string.
    """
    from flash.providers.vast.client.api import VastApiError

    result, _ = _submit_failure_with_secret(
        VastApiError("create rejected: body echoed a token nobody registered")
    )

    assert "body echoed a token nobody registered" not in result.detail, result.detail
    assert result.detail == "provider submit failed (VastApiError)", result.detail


def test_pinned_gpu_out_of_capacity_stops_instead_of_requeueing_on_the_same_class(
    orch, monkeypatch
):
    """A hard gpu.type plus gpu.provider pin gives the picker one fixed market question, so a
    no_capacity retry used to re-select the same unavailable shape and burn another full capacity
    grace on it -- five times, at 900s each, for up to 75 minutes of wall clock. Once that exact
    class-provider shape has refused twice, the market has answered: stop and name the fix.

    Two, not one: `no_capacity` also covers a transient search flake and an exhausted provider pool,
    and a dry market frees cards, so the second refusal is what separates a blip from a wall (see
    `test_pinned_gpu_retries_a_single_capacity_blip_before_giving_up`)."""
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs

    # exactly what hard class and provider pins produce: one fixed class-provider shape.
    candidates = (Candidate("runpod", "H200", 4.0, 141),)
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    monkeypatch.setattr(runpod_api, "cancel_job", lambda *a, **k: None)
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", lambda *_a, **_k: True)

    submitted_gpus = []

    def fake_runpod_submit(run_spec, seed, log=None, on_handle=None, attempt=0, **kwargs):
        submitted_gpus.append(run_spec.gpu.type)
        on_handle(_runpod_handle(f"ep{attempt}", f"j{attempt}", attempt))
        return PollResult(
            False,
            failure="no_capacity",
            detail="never scheduled: job stuck IN_QUEUE for 903s",
        )

    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)
    spec = _spec(run_id="flash-pinned-gpu-nowalk", type="H200", provider="runpod")
    _seed_status(orch, spec)
    log = io.StringIO()
    with pytest.raises(RuntimeError, match="failed after retries"):
        runner_lifecycle._submit_seed_supervised(
            spec, spec.seed, log, source_snapshot=_SOURCE_SNAPSHOT
        )

    # TWO submissions, not the budget's five: the class refused, the retry confirmed it, and the
    # remaining attempts would each have re-asked the settled question at a full capacity grace.
    assert submitted_gpus == ["H200", "H200"]
    out = log.getvalue()
    assert "walking past the cheapest class" not in out
    # and the operator is pointed at the two things that actually widen the search.
    assert "has already refused capacity twice" in out
    assert "drop the gpu.type pin" in out
    assert "drop gpu.provider" in out


def test_pinned_gpu_retries_a_single_capacity_blip_before_giving_up(orch, monkeypatch):
    """One `no_capacity` is a data point, not a verdict, so it must not end the run.

    The label covers a transient search flake and an exhausted provider pool as well as the 900s
    queue-grace expiry, and capacity that was gone a minute ago comes back. Stopping on the first
    refusal would convert every momentary shortage into a failed run -- the opposite defect from the
    75-minute loop, and the more expensive one, because the work is already paid for."""
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs

    candidates = (Candidate("runpod", "H200", 4.0, 141),)
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    monkeypatch.setattr(runpod_api, "cancel_job", lambda *a, **k: None)
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", lambda *_a, **_k: True)

    submitted_gpus = []

    def fake_runpod_submit(run_spec, seed, log=None, on_handle=None, attempt=0, **kwargs):
        submitted_gpus.append(run_spec.gpu.type)
        on_handle(_runpod_handle(f"ep{attempt}", f"j{attempt}", attempt))
        if attempt == 0:
            return PollResult(False, failure="no_capacity", detail="search flaked")
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)
    spec = _spec(run_id="flash-pinned-gpu-blip", type="H200", provider="runpod")
    _seed_status(orch, spec)
    log = io.StringIO()
    metrics = runner_lifecycle._submit_seed_supervised(
        spec, spec.seed, log, source_snapshot=_SOURCE_SNAPSHOT
    )

    assert metrics["train_tokens"] == 4096
    assert submitted_gpus == ["H200", "H200"], "the blip must cost a retry, not the run"
    assert "has already refused capacity twice" not in log.getvalue()


@pytest.mark.parametrize("gpu_type", ["", "H200"], ids=["auto-class", "pinned-class"])
def test_dynamic_provider_search_keeps_the_full_capacity_retry_budget(orch, monkeypatch, gpu_type):
    """Without a hard provider pin, each retry can discover a class on a returning provider."""
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs
    from flash.runner.supervise.lifecycle import INFRA_RETRY_FLOOR

    candidates = (Candidate("runpod", "H200", 4.0, 141),)
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    monkeypatch.setattr(runpod_api, "cancel_job", lambda *a, **k: None)
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", lambda *_a, **_k: True)

    submitted_gpus = []

    def fake_runpod_submit(run_spec, seed, log=None, on_handle=None, attempt=0, **kwargs):
        submitted_gpus.append(run_spec.gpu.type)
        on_handle(_runpod_handle(f"ep{attempt}", f"j{attempt}", attempt))
        return PollResult(False, failure="no_capacity", detail="market search is dry")

    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)
    spec = _spec(run_id=f"flash-dynamic-provider-{gpu_type or 'auto'}", type=gpu_type)
    _seed_status(orch, spec)
    log = io.StringIO()
    with pytest.raises(RuntimeError, match="failed after retries"):
        runner_lifecycle._submit_seed_supervised(
            spec, spec.seed, log, source_snapshot=_SOURCE_SNAPSHOT
        )

    assert submitted_gpus == ["H200"] * (INFRA_RETRY_FLOOR + 1)
    assert "has already refused capacity twice" not in log.getvalue()
    assert "drop the gpu.type pin" not in log.getvalue()


def test_multi_count_pin_keeps_retry_budget_when_another_width_can_reappear(orch, monkeypatch):
    """A count ceiling admits several shapes, even when the current market exposes only one."""
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs
    from flash.runner.supervise.lifecycle import INFRA_RETRY_FLOOR

    candidates = (Candidate("runpod", "H200", 4.0, 141, gpu_count=1),)
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    monkeypatch.setattr(runpod_api, "cancel_job", lambda *a, **k: None)
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", lambda *_a, **_k: True)

    submissions = []

    def fake_runpod_submit(run_spec, seed, log=None, on_handle=None, attempt=0, **kwargs):
        submissions.append(run_spec.gpu.count)
        on_handle(_runpod_handle(f"ep{attempt}", f"j{attempt}", attempt))
        return PollResult(False, failure="no_capacity", detail="only one width is visible")

    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)
    spec = _spec(
        run_id="flash-pinned-multi-count-capacity",
        type="H200",
        provider="runpod",
        count=2,
    )
    _seed_status(orch, spec)
    log = io.StringIO()
    with pytest.raises(RuntimeError, match="failed after retries"):
        runner_lifecycle._submit_seed_supervised(
            spec, spec.seed, log, source_snapshot=_SOURCE_SNAPSHOT
        )

    assert submissions == [1] * (INFRA_RETRY_FLOOR + 1)
    assert "has already refused capacity twice" not in log.getvalue()


def test_allocation_time_sellout_counts_for_a_fixed_shape(orch, monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core.base import CapacityUnavailableError

    calls = 0

    def sold_out(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise CapacityUnavailableError("exact GPU 'H100' currently has no capacity")

    monkeypatch.setattr(allocator, "allocate", sold_out)
    spec = _spec(
        run_id="flash-pinned-allocation-capacity",
        type="H100",
        provider="lambda",
        count=1,
        max_retries=4,
    )
    _seed_status(orch, spec)
    log = io.StringIO()
    with pytest.raises(RuntimeError, match="failed after retries"):
        runner_lifecycle._submit_seed_supervised(
            spec, spec.seed, log, source_snapshot=_SOURCE_SNAPSHOT
        )

    assert calls == 2
    assert "has already refused capacity twice" in log.getvalue()


def test_allocation_lookup_outage_keeps_the_full_retry_budget(orch, monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core.base import CapacityLookupError
    from flash.runner.supervise.lifecycle import INFRA_RETRY_FLOOR

    calls = 0

    def lookup_failed(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise CapacityLookupError("lambda live capacity lookup failed")

    monkeypatch.setattr(allocator, "allocate", lookup_failed)
    spec = _spec(
        run_id="flash-pinned-allocation-lookup",
        type="H100",
        provider="lambda",
        count=1,
    )
    _seed_status(orch, spec)
    log = io.StringIO()
    with pytest.raises(RuntimeError, match="failed after retries"):
        runner_lifecycle._submit_seed_supervised(
            spec, spec.seed, log, source_snapshot=_SOURCE_SNAPSHOT
        )

    assert calls == INFRA_RETRY_FLOOR + 1
    assert "has already refused capacity twice" not in log.getvalue()


def test_a_provisioned_attempt_resets_the_class_capacity_refusals(orch, monkeypatch):
    """A worker-side failure proves the class admitted the run, so an older shortage is stale."""
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs

    candidates = (Candidate("runpod", "H200", 4.0, 141),)
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    monkeypatch.setattr(runpod_api, "cancel_job", lambda *a, **k: None)
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", lambda *_a, **_k: True)

    submitted_gpus = []
    failures = ("no_capacity", "stalled", "no_capacity")

    def fake_runpod_submit(run_spec, seed, log=None, on_handle=None, attempt=0, **kwargs):
        submitted_gpus.append(run_spec.gpu.type)
        on_handle(_runpod_handle(f"ep{attempt}", f"j{attempt}", attempt))
        if attempt < len(failures):
            return PollResult(False, failure=failures[attempt], detail="attempt failed")
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)
    spec = _spec(run_id="flash-capacity-recovers", type="H200", provider="runpod", max_retries=3)
    _seed_status(orch, spec)
    log = io.StringIO()
    metrics = runner_lifecycle._submit_seed_supervised(
        spec, spec.seed, log, source_snapshot=_SOURCE_SNAPSHOT
    )

    assert metrics["train_tokens"] == 4096
    assert submitted_gpus == ["H200", "H200", "H200", "H200"]
    assert "has already refused capacity twice" not in log.getvalue()


def test_dropping_the_weight_cache_gives_the_widened_search_its_own_capacity_looks(
    orch, monkeypatch
):
    """A refusal heard while pinned to the cache volume must not count against the cacheless search.

    The weight-cache volume pins the run to the one region holding those weights, so a refusal there
    answers "any capacity for this class in that region?" -- a narrower question than the cacheless
    retry asks. Carrying the tally across would let one region's shortage plus a single blip in the
    unrestricted pool reach two and stop the run, having heard the wider market refuse only once."""
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME

    candidates = (Candidate("runpod", "H100", 0.49, 48),)
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
        # region-pinned refusal, then ONE blip in the widened pool, then the market comes back.
        if len(seen) >= 3:
            return PollResult(True, metrics={"train_tokens": 4096})
        return PollResult(False, failure="no_capacity", detail="IN_QUEUE (no capacity)")

    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec(
        provider="runpod",
        max_retries=3,
        network_volume=WEIGHT_CACHE_VOLUME_NAME,
        network_volume_gb=100,
    )
    _seed_status(orch, spec)
    metrics = runner_lifecycle._submit_seed_supervised(
        spec, spec.seed, io.StringIO(), source_snapshot=_SOURCE_SNAPSHOT
    )

    # the run survives to its third look. carrying the cache-pinned refusal would have made the
    # cacheless blip the second strike and stopped here with the market never having been asked twice.
    assert metrics["train_tokens"] == 4096
    assert seen == [
        ("H100", WEIGHT_CACHE_VOLUME_NAME),
        ("H100", None),
        ("H100", None),
    ]
