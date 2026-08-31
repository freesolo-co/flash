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

    monkeypatch.setattr(rp_jobs, "submit_attempt", fake_runpod_submit)
    spec = _spec(type="RTX 4090", **gpu_preferences)
    _seed_status(orch, spec)
    metrics = runner_lifecycle._run_attempts_supervised(
        spec,
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
    monkeypatch.setattr(rp_jobs, "submit_attempt", fake_runpod_submit)

    with pytest.raises(runner_errors._RunCancelled):
        runner_lifecycle._run_attempts_supervised(
            spec, io.StringIO(), source_snapshot=_SOURCE_SNAPSHOT
        )

    status = runner_status.get_status(spec.run_id)
    assert status.state == "cancelled"
    assert status.effective_preparation is None
    assert not runner_submit._persist_effective_worker_spec(spec)
    assert provider_calls == []


@pytest.mark.parametrize("first_revocation_fails", [False, True])
def test_cancel_terminalizes_before_handle_and_callback_cleans_resource(
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

    def fake_runpod_submit(run_spec, *, on_handle, **kwargs):
        resource_live["value"] = True
        resource_created.set()
        assert allow_handle.wait(timeout=5)
        on_handle(_runpod_handle("ep-handshake", "job-handshake"))
        persisted_remote = runner_status.get_status(spec.run_id).remote
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

    monkeypatch.setattr(rp_jobs, "submit_attempt", fake_runpod_submit)
    monkeypatch.setattr(server_db, "revoke_teacher_capabilities_for_run", revoke_capabilities)
    monkeypatch.setattr(runpod_api, "cancel_job", cancel_job)
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", delete_endpoint)

    submit_errors = []

    def submit():
        try:
            runner_lifecycle._run_attempts_supervised(
                spec, io.StringIO(), source_snapshot=_SOURCE_SNAPSHOT
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
    assert cancellation_finished.wait(timeout=5)
    cancel_thread.join(timeout=5)
    assert not cancel_thread.is_alive()
    waiting_status = runner_status.get_status(spec.run_id)
    assert waiting_status.state == "cancelled"
    assert waiting_status.remote is None
    assert resource_live["value"]

    allow_handle.set()
    submit_thread.join(timeout=5)
    assert not submit_thread.is_alive()
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
    assert isinstance(submit_errors[0], runner_errors._TerminalHandleRace)
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

    def fake_runpod_submit(run_spec, *, on_handle, **kwargs):
        provider_gpus.append(run_spec.gpu.type)
        resource_created.set()
        assert allow_handle.wait(timeout=5)
        on_handle(_runpod_handle("ep-first", "job-first"))
        polling.set()
        assert allow_poll.wait(timeout=5)
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_attempt", fake_runpod_submit)

    results = {}

    def submit(name):
        try:
            results[name] = runner_lifecycle._run_attempts_supervised(
                spec, io.StringIO(), source_snapshot=_SOURCE_SNAPSHOT
            )
        except Exception as exc:
            results[name] = exc

    first = threading.Thread(target=submit, args=("first",), name="supervisor-a")
    second = threading.Thread(target=submit, args=("second",), name="supervisor-b")
    first.start()
    assert resource_created.wait(timeout=5)
    second.start()
    second.join(timeout=5)
    assert not second.is_alive()
    assert isinstance(results["second"], RuntimeError)
    assert "attempt launch reservation lost ownership" in str(results["second"])

    allow_handle.set()
    assert polling.wait(timeout=5)
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
def test_provider_submission_paths_release_launch_lease(orch, monkeypatch, failure_mode):
    """Every provider outcome must leave the launch lease free for the next attempt."""
    from flash.providers.core import allocator
    from flash.providers.core.base import PollResult
    from flash.providers.runpod.execution import job_execution as rp_jobs
    from flash.runner.lifecycle import attempts as runner_attempts
    from flash.runner.lifecycle import claim_lock

    spec = _spec(run_id=f"flash-lease-release-{failure_mode}", max_retries=0)
    status = _seed_status(orch, spec)
    status.source_snapshot = _SOURCE_SNAPSHOT
    runner_state._save_status(status)
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc())

    def fail_remote_persistence(run_id, claim, persisted_remote):
        if failure_mode == "callback_persistence_exception":
            raise RuntimeError("remote persistence failed")
        return True

    def fake_runpod_submit(run_spec, *, on_handle, **kwargs):
        if failure_mode == "provider_exception":
            raise RuntimeError("provider create failed")
        if failure_mode == "callback_persistence_exception":
            on_handle({"provider": "runpod", "job_id": "job-no-persist"})
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(runner_attempts, "persist_claimed_remote", fail_remote_persistence)
    monkeypatch.setattr(rp_jobs, "submit_attempt", fake_runpod_submit)

    if failure_mode == "provider_without_callback":
        metrics = runner_lifecycle._run_attempts_supervised(
            spec, io.StringIO(), source_snapshot=_SOURCE_SNAPSHOT
        )
        assert metrics["train_tokens"] == 4096
    else:
        with pytest.raises(RuntimeError, match="failed after retries"):
            runner_lifecycle._run_attempts_supervised(
                spec, io.StringIO(), source_snapshot=_SOURCE_SNAPSHOT
            )

    raw = runner_status._load_status_json(spec.run_id)
    assert runner_state._ACTIVE_LAUNCH_CLAIM_KEY not in raw
    fd = claim_lock.try_acquire(spec.run_id)
    assert fd is not None
    claim_lock.close(fd)


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

    def fake_runpod_submit(run_spec, **kwargs):
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
    monkeypatch.setattr(rp_jobs, "submit_attempt", fake_runpod_submit)
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

    def fake_runpod_submit(run_spec, **kwargs):
        persisted = runner_status.get_status(run_spec.run_id).effective_preparation["worker_spec"]
        persisted_at_submission.append(persisted["environment"])
        return PollResult(True, metrics={"train_tokens": 4096, "wall_seconds": 1})

    monkeypatch.setattr(
        "flash.cost.spec.estimate_for_spec",
        lambda _spec, **_kwargs: type("Estimate", (), {"total_usd": 1.0})(),
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(gpu="RTX 5090"))
    monkeypatch.setattr(rp_jobs, "submit_attempt", fake_runpod_submit)
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

    def fake_runpod_submit(run_spec, log=None, on_handle=None, attempt=0, **_):
        if on_handle:
            on_handle(_runpod_handle(attempt=attempt))
        return PollResult(True, metrics={"train_tokens": 4096, "wall_seconds": 1800})

    monkeypatch.setattr(rp_jobs, "submit_attempt", fake_runpod_submit)
    spec = _spec(provider="runpod", type="RTX 4090", count=4)
    _seed_status(orch, spec)

    metrics = runner_lifecycle._run_attempts_supervised(
        spec, io.StringIO(), source_snapshot=_SOURCE_SNAPSHOT
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

    def fake_runpod_submit(run_spec, log=None, on_handle=None, attempt=0, **kwargs):
        submitted_gpus.append(run_spec.gpu.type)
        if attempt == 0:
            on_handle(_runpod_handle("ep1", "j1", attempt))
            return PollResult(False, failure="stalled", detail="no worker progress")
        on_handle(_runpod_handle("ep2", "j2", attempt))
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_attempt", fake_runpod_submit)
    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()
    metrics = runner_lifecycle._run_attempts_supervised(spec, log, source_snapshot=_SOURCE_SNAPSHOT)
    assert metrics["train_tokens"] == 4096
    assert submitted_gpus == ["RTX 4090", "H100"]
    assert cancelled == [("ep1", "j1"), ("ep2", "j2")]
    assert deleted == ["ep1", "ep2"]
    assert "selected strictly larger H100 @ runpod" in log.getvalue()


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

    def fake_runpod_submit(run_spec, log=None, on_handle=None, attempt=0, **kwargs):
        submitted_attempts.append(attempt)
        on_handle(_runpod_handle("ep-unconfirmed", "job-unconfirmed", attempt))
        return PollResult(False, failure="stalled", detail="no worker progress")

    monkeypatch.setattr(rp_jobs, "submit_attempt", fake_runpod_submit)
    spec = _spec(run_id="flash-unconfirmed-runpod-teardown")
    _seed_status(orch, spec)
    log = io.StringIO()

    with pytest.raises(RuntimeError, match="teardown could not be confirmed"):
        runner_lifecycle._run_attempts_supervised(spec, log, source_snapshot=_SOURCE_SNAPSHOT)

    assert submitted_attempts == [0]
    assert deleted_endpoints
    assert set(deleted_endpoints) == {"ep-unconfirmed"}
    assert gc_calls == [spec.run_id]
    status = runner_status.get_status(spec.run_id)
    assert status.remote["endpoint_id"] == "ep-unconfirmed"
    assert status.remote["job_id"] == "job-unconfirmed"
    assert "teardown unconfirmed" in log.getvalue()


def _retry_candidate(provider, gpu, vram, count=1):
    from flash.providers.core.base import Candidate

    return Candidate(provider, gpu, 1.0, vram, count)


@pytest.mark.parametrize("first_failure", ["no_capacity", "poll_error"])
def test_shared_cache_fallback_is_exact_one_shot_then_walks_larger(
    orch, monkeypatch, first_failure
):
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME

    candidates = (
        Candidate("runpod", "H100", 1.0, 80, 2),
        Candidate("runpod", "B200", 2.0, 180, 1),
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    monkeypatch.setattr(runpod_api, "cancel_job", lambda *a, **k: None)
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", lambda *a, **k: True)
    seen = []

    def fake_submit(run_spec, on_handle=None, attempt=0, **kwargs):
        seen.append(
            (
                run_spec.gpu.type,
                run_spec.gpu.count,
                run_spec.gpu.network_volume,
            )
        )
        on_handle(_runpod_handle(f"ep{attempt}", f"j{attempt}", attempt))
        if attempt == 0:
            return PollResult(False, failure=first_failure, detail="cached failure")
        if attempt == 1:
            return PollResult(False, failure="poll_error", detail="cacheless failure")
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_attempt", fake_submit)
    spec = _spec(
        run_id=f"flash-cache-{first_failure}",
        max_retries=1,
        count=2,
        network_volume=WEIGHT_CACHE_VOLUME_NAME,
        network_volume_gb=100,
    )
    _seed_status(orch, spec)

    metrics = runner_lifecycle._run_attempts_supervised(
        spec, io.StringIO(), source_snapshot=_SOURCE_SNAPSHOT
    )

    assert metrics["train_tokens"] == 4096
    assert seen == [
        ("H100", 2, WEIGHT_CACHE_VOLUME_NAME),
        ("H100", 2, None),
        ("B200", 1, None),
    ]


def test_lambda_cacheless_attempt_requires_atomic_cache_authorization(orch, monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core import registry as providers
    from flash.providers.core.base import Candidate, PollResult
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME

    candidate = Candidate("lambda", "H100", 1.0, 80)
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=(candidate,)))
    seen = []

    class Provider:
        supports_weight_cache = True

        def submit_attempt(self, run_spec, *, attempt, **_kwargs):
            snapshot = runner_status._load_status_json(run_spec.run_id)[
                runner_state._RETRY_STATE_KEY
            ]
            seen.append((attempt, run_spec.gpu.network_volume, snapshot["drop_weight_cache"]))
            if attempt == 0:
                raise RuntimeError("cached lambda launch rejected")
            return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(providers, "get_provider", lambda _name: Provider())
    spec = _spec(
        run_id="flash-lambda-atomic-cache",
        max_retries=1,
        network_volume=WEIGHT_CACHE_VOLUME_NAME,
        network_volume_gb=100,
    )
    _seed_status(orch, spec)

    metrics = runner_lifecycle._run_attempts_supervised(
        spec, io.StringIO(), source_snapshot=_SOURCE_SNAPSHOT
    )

    assert metrics["train_tokens"] == 4096
    assert seen == [
        (0, WEIGHT_CACHE_VOLUME_NAME, 0),
        (1, None, 1),
    ]


def test_custom_volume_is_preserved_and_gets_no_cache_fallback(orch, monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, PollResult
    from flash.providers.runpod.execution import job_execution as rp_jobs

    candidate = Candidate("runpod", "H100", 1.0, 80)
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=(candidate,)))
    seen = []

    def fake_submit(run_spec, **kwargs):
        seen.append(run_spec.gpu.network_volume)
        return PollResult(False, failure="no_capacity", detail="dry")

    monkeypatch.setattr(rp_jobs, "submit_attempt", fake_submit)
    spec = _spec(network_volume="org-cache", network_volume_gb=100)
    _seed_status(orch, spec)

    with pytest.raises(RuntimeError, match="failed after retries"):
        runner_lifecycle._run_attempts_supervised(
            spec, io.StringIO(), source_snapshot=_SOURCE_SNAPSHOT
        )

    assert seen == ["org-cache"]


def test_genuine_worker_error_does_not_retry(orch, monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core.base import PollResult
    from flash.providers.runpod.execution import job_execution as rp_jobs

    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc())
    calls = []

    def fake_runpod_submit(run_spec, log=None, on_handle=None, attempt=0, **kwargs):
        calls.append(attempt)
        return PollResult(False, failure="job_failed", detail="ValueError: bad reward fn")

    monkeypatch.setattr(rp_jobs, "submit_attempt", fake_runpod_submit)
    spec = _spec()
    _seed_status(orch, spec)
    with pytest.raises(RuntimeError, match="bad reward fn"):
        runner_lifecycle._run_attempts_supervised(
            spec, io.StringIO(), source_snapshot=_SOURCE_SNAPSHOT
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
    from flash.runner.supervise import attempt_supervision as _ss

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
        from flash.runner.lifecycle.attempts import AttemptLaunchClaim
        from flash.runner.supervise.retry_decision import RetryState

        ctx = _ss._SubmitContext(
            spec=spec,
            log=io.StringIO(),
            runtime_secrets={},
            source_snapshot=_SOURCE_SNAPSHOT,
        )
        state = RetryState(1, 1, 0)
        prepared = (AttemptLaunchClaim(0, "submit-failure-claim"), spec, {}, state)
        candidate_plan = ((), _Chosen("vast", "RTX 4090"), False, spec, spec)
        return _ss._submit_provider(ctx, prepared, candidate_plan)
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
        def submit_attempt(self, *_a, **_kw):
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
        def submit_attempt(self, *_a, **_kw):
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


def test_candidate_less_allocation_poll_error_uses_infrastructure_backoff(orch, monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, CapacityLookupError, PollResult
    from flash.providers.runpod.execution import job_execution as rp_jobs

    calls = 0
    candidate = Candidate("runpod", "H100", 1.0, 80)

    def allocate(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise CapacityLookupError("lookup failed")
        return _alloc(candidates=(candidate,))

    monkeypatch.setattr(allocator, "allocate", allocate)
    monkeypatch.setattr(
        rp_jobs,
        "submit_attempt",
        lambda *a, **k: PollResult(True, metrics={"train_tokens": 4096}),
    )
    sleeps = []
    monkeypatch.setattr(runner_lifecycle.time, "sleep", sleeps.append)
    spec = _spec(run_id="flash-allocation-poll-error")
    _seed_status(orch, spec)

    metrics = runner_lifecycle._run_attempts_supervised(
        spec, io.StringIO(), source_snapshot=_SOURCE_SNAPSHOT
    )

    assert metrics["train_tokens"] == 4096
    assert calls == 3
    assert sleeps == [10, 20]


def test_candidate_less_allocation_no_capacity_stops_immediately(orch, monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core.base import CapacityUnavailableError

    calls = 0

    def allocate(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise CapacityUnavailableError("dry")

    monkeypatch.setattr(allocator, "allocate", allocate)
    spec = _spec(run_id="flash-allocation-no-capacity")
    _seed_status(orch, spec)

    with pytest.raises(RuntimeError, match="failed after retries"):
        runner_lifecycle._run_attempts_supervised(
            spec, io.StringIO(), source_snapshot=_SOURCE_SNAPSHOT
        )

    assert calls == 1


def test_unsupported_gpu_terminal_failure_consumes_prehandle_claim(orch, monkeypatch, tmp_path):
    from flash.providers.core import allocator
    from flash.providers.core.base import UnsupportedGpuError
    from flash.runner.lifecycle import claim_lock

    spec = _spec(run_id="flash-unsupported-prehandle")
    status = _seed_status(orch, spec)
    status.source_snapshot = _SOURCE_SNAPSHOT
    runner_state._save_status(status)
    monkeypatch.setattr(
        runner_artifacts, "stage_environment_package", lambda value, **_kwargs: value
    )
    monkeypatch.setattr(
        allocator,
        "allocate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(UnsupportedGpuError("unsupported")),
    )

    with pytest.raises(UnsupportedGpuError, match="unsupported"):
        runner_lifecycle._run_job_inner(spec, str(tmp_path / "unsupported.log"))

    persisted = runner_status.get_status(spec.run_id)
    assert persisted.state == "failed"
    raw = runner_status._load_status_json(spec.run_id)
    assert runner_state._ACTIVE_LAUNCH_CLAIM_KEY not in raw
    fd = claim_lock.try_acquire(spec.run_id)
    assert fd is not None
    claim_lock.close(fd)


def test_cancel_before_provider_submit_consumes_prehandle_claim(orch, monkeypatch, tmp_path):
    from flash.providers.core import allocator
    from flash.runner.lifecycle import claim_lock
    from flash.runner.supervise import attempt_supervision as submission

    spec = _spec(run_id="flash-cancel-prehandle")
    status = _seed_status(orch, spec)
    status.source_snapshot = _SOURCE_SNAPSHOT
    runner_state._save_status(status)
    monkeypatch.setattr(
        runner_artifacts, "stage_environment_package", lambda value, **_kwargs: value
    )
    original_prepare = submission._prepare_attempt

    def cancel_after_reservation(ctx):
        prepared = original_prepare(ctx)
        assert runner_status._update(spec.run_id, "cancelled")
        return prepared

    monkeypatch.setattr(submission, "_prepare_attempt", cancel_after_reservation)
    monkeypatch.setattr(
        allocator,
        "allocate",
        lambda *_args, **_kwargs: pytest.fail("cancelled run reached provider allocation"),
    )

    runner_lifecycle._run_job_inner(spec, str(tmp_path / "cancelled.log"))

    persisted = runner_status.get_status(spec.run_id)
    assert persisted.state == "cancelled"
    raw = runner_status._load_status_json(spec.run_id)
    assert runner_state._ACTIVE_LAUNCH_CLAIM_KEY not in raw
    fd = claim_lock.try_acquire(spec.run_id)
    assert fd is not None
    claim_lock.close(fd)


def test_reserved_claim_context_failure_preserves_error_and_consumes_claim(orch, monkeypatch):
    from flash.runner.lifecycle import attempts as runner_attempts
    from flash.runner.lifecycle import claim_lock
    from flash.runner.supervise import attempt_supervision as submission

    class SourceContextFailure(RuntimeError):
        pass

    spec = _spec(run_id="flash-reserved-context-failure")
    _seed_status(orch, spec)
    claim = runner_attempts.reserve_verified_attempt_launch(spec.run_id)
    assert claim is not None
    monkeypatch.setattr(
        submission,
        "_build_context",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            SourceContextFailure("source context failed")
        ),
    )

    with pytest.raises(SourceContextFailure, match="source context failed"):
        runner_lifecycle._run_attempts_supervised(
            spec,
            io.StringIO(),
            source_snapshot=_SOURCE_SNAPSHOT,
            reserved_claim=claim,
        )

    raw = runner_status._load_status_json(spec.run_id)
    assert runner_state._ACTIVE_LAUNCH_CLAIM_KEY not in raw
    fd = claim_lock.try_acquire(spec.run_id)
    assert fd is not None
    claim_lock.close(fd)


def test_stale_reserved_attempt_cannot_reach_provider_submit(orch, monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core import registry as providers
    from flash.runner.lifecycle import attempts as runner_attempts
    from flash.runner.supervise.retry_decision import FailureObservation

    spec = _spec(run_id="flash-stale-reserved-launch")
    _seed_status(orch, spec)
    first_claim = runner_attempts.reserve_verified_attempt_launch(spec.run_id)
    assert first_claim is not None
    plan = runner_attempts.decide_attempt_failure(
        spec.run_id,
        claim_token=first_claim.token,
        expected_remote=None,
        observation=FailureObservation("poll_error"),
        attempt=0,
    )
    assert plan is not None
    assert plan.retry
    second_claim = runner_attempts.reserve_verified_attempt_launch(spec.run_id)
    assert second_claim is not None
    assert second_claim.attempt == 1
    monkeypatch.setattr(allocator, "allocate", lambda *_args, **_kwargs: _alloc())

    class Provider:
        def submit_attempt(self, *_args, **_kwargs):
            pytest.fail("stale reserved attempt reached provider submit")

    monkeypatch.setattr(providers, "get_provider", lambda _name: Provider())
    with pytest.raises(RuntimeError, match="claim changed"):
        runner_lifecycle._run_attempts_supervised(
            spec,
            io.StringIO(),
            source_snapshot=_SOURCE_SNAPSHOT,
            reserved_claim=first_claim,
        )


def test_mixed_cache_oom_and_infra_retries_back_off_only_by_infra_ordinal(orch, monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution as rp_jobs
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME

    candidates = (
        Candidate("runpod", "H100", 1.0, 80, 2),
        Candidate("runpod", "B200", 2.0, 180, 1),
        Candidate("runpod", "B200", 2.0, 180, 2),
        Candidate("runpod", "B200", 2.0, 180, 4),
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    monkeypatch.setattr(runpod_api, "cancel_job", lambda *a, **k: None)
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", lambda *a, **k: True)
    sleeps = []
    monkeypatch.setattr(runner_lifecycle.time, "sleep", sleeps.append)
    seen = []

    def fake_submit(run_spec, on_handle=None, attempt=0, **kwargs):
        seen.append((run_spec.gpu.type, run_spec.gpu.count, run_spec.gpu.network_volume))
        if attempt < 2:
            on_handle(_runpod_handle(f"ep{attempt}", f"j{attempt}", attempt))
        if attempt == 0:
            return PollResult(False, failure="no_capacity", detail="cached capacity")
        if attempt == 1:
            return PollResult(False, failure="oom", detail="cacheless oom")
        if attempt in {2, 3}:
            raise RuntimeError("submit transport failed")
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_attempt", fake_submit)
    spec = _spec(
        run_id="flash-mixed-retry-backoff",
        max_retries=2,
        count=4,
        network_volume=WEIGHT_CACHE_VOLUME_NAME,
        network_volume_gb=100,
    )
    _seed_status(orch, spec)

    metrics = runner_lifecycle._run_attempts_supervised(
        spec, io.StringIO(), source_snapshot=_SOURCE_SNAPSHOT
    )

    assert metrics["train_tokens"] == 4096
    assert sleeps == [10, 20]
    assert seen == [
        ("H100", 2, WEIGHT_CACHE_VOLUME_NAME),
        ("H100", 2, None),
        ("B200", 1, None),
        ("B200", 2, None),
        ("B200", 4, None),
    ]


def test_submit_failure_with_no_larger_candidate_does_not_sleep(orch, monkeypatch):
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate
    from flash.providers.runpod.execution import job_execution as rp_jobs

    candidate = Candidate("runpod", "B200", 2.0, 180)
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=(candidate,)))
    monkeypatch.setattr(
        rp_jobs,
        "submit_attempt",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("submit transport failed")),
    )
    sleeps = []
    monkeypatch.setattr(runner_lifecycle.time, "sleep", sleeps.append)
    spec = _spec(run_id="flash-no-larger-no-backoff", max_retries=2)
    _seed_status(orch, spec)

    with pytest.raises(RuntimeError, match="failed after retries"):
        runner_lifecycle._run_attempts_supervised(
            spec, io.StringIO(), source_snapshot=_SOURCE_SNAPSHOT
        )

    assert sleeps == []
