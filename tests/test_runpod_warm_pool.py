from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace

import pytest

_RUNPOD_FINGERPRINT = "rpk-0123456789ab"


@pytest.fixture
def warm_pool_dir(tmp_path, monkeypatch):
    from flash.providers.runpod import warm_pool

    path = tmp_path / "runpod_warm"
    monkeypatch.setattr(warm_pool, "WARM_DIR", str(path))
    return path


def _spec(*, run_id: str = "warm-run"):
    from flash.spec import EnvironmentSpec, GpuSpec, JobSpec, TrainSpec

    return JobSpec(
        run_id=run_id,
        seed=42,
        model="Qwen/Qwen3.5-0.8B",
        model_revision="a" * 40,
        algorithm="sft",
        environment=EnvironmentSpec(
            id="env-a",
            params={"dataset": "dataset-a"},
            pip=("package-a",),
        ),
        train=TrainSpec(epochs=1, max_context_tokens=4096),
        gpu=GpuSpec(
            type="RTX 4090",
            disk_gb=80,
            network_volume="weight-cache",
            network_volume_gb=100,
            max_wall_seconds=3600,
        ),
    )


def _record(
    endpoint_id: str,
    *,
    signature: str = "signature",
    key_fingerprint: str = _RUNPOD_FINGERPRINT,
    released_at: float = 1.0,
    execution_timeout_ms: int = 600_000,
):
    from flash.providers.runpod.warm_pool import WarmEndpoint

    return WarmEndpoint(
        endpoint_id=endpoint_id,
        name=f"flash-4090-{endpoint_id}",
        key_fingerprint=key_fingerprint,
        signature=signature,
        execution_timeout_ms=execution_timeout_ms,
        released_at=released_at,
    )


def test_reuse_signature_excludes_per_job_payload_fields():
    from flash.providers.runpod.warm_pool import reuse_signature
    from flash.spec import EnvironmentSpec

    first = _spec(run_id="run-a")
    second = replace(
        first,
        run_id="run-b",
        seed=987,
        environment=EnvironmentSpec(
            id="env-b",
            params={"dataset": "dataset-b"},
            pip=("different-package",),
        ),
    )

    assert reuse_signature(first, _RUNPOD_FINGERPRINT) == reuse_signature(
        second, _RUNPOD_FINGERPRINT
    )


def test_reuse_signature_changes_for_endpoint_compatibility_fields():
    from flash.providers.runpod.warm_pool import reuse_signature

    base = _spec()
    base_signature = reuse_signature(base, _RUNPOD_FINGERPRINT)
    variants = [
        replace(base, gpu=replace(base.gpu, type="RTX 5090")),
        replace(base, model="Qwen/Qwen3.5-4B"),
        replace(base, model_revision="b" * 40),
        replace(base, train=replace(base.train, max_context_tokens=8192)),
        replace(base, gpu=replace(base.gpu, disk_gb=120)),
        replace(base, gpu=replace(base.gpu, network_volume="other-cache")),
        replace(base, gpu=replace(base.gpu, network_volume_gb=200)),
    ]

    for variant in variants:
        assert reuse_signature(variant, _RUNPOD_FINGERPRINT) != base_signature


def test_registry_round_trip_claim_prune_and_atomic_file(warm_pool_dir):
    from flash.providers.runpod import warm_pool

    assert warm_pool.register(_record("older", released_at=1.0))
    assert warm_pool.register(_record("newer", released_at=2.0))
    assert [record.endpoint_id for record in warm_pool.candidates("signature", 600_000)] == [
        "newer",
        "older",
    ]

    assert warm_pool.claim("newer") is True
    assert warm_pool.claim("newer") is False
    warm_pool.prune("older")
    warm_pool.prune("missing")
    assert warm_pool.candidates("signature", 0) == []

    registry_path = warm_pool_dir / "endpoints.json"
    assert json.loads(registry_path.read_text()) == {}
    assert list(warm_pool_dir.glob("*.tmp")) == []


def test_registry_enforces_hard_cap(warm_pool_dir, monkeypatch):
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import warm_pool

    for index in range(warm_pool.MAX_WARM_ENDPOINTS):
        assert warm_pool.register(_record(f"endpoint-{index}"))
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *_args, **_kwargs: {
            "workers": {"idle": 1, "unhealthy": 0},
            "jobs": {"inQueue": 0, "inProgress": 0},
        },
    )
    assert warm_pool.register(_record("endpoint-over-cap")) is False
    assert len(warm_pool.candidates("signature", 0)) == warm_pool.MAX_WARM_ENDPOINTS


def test_registry_cap_reconcile_prunes_dead_record_and_frees_slot(warm_pool_dir, monkeypatch):
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import warm_pool

    for index in range(warm_pool.MAX_WARM_ENDPOINTS):
        assert warm_pool.register(_record(f"endpoint-{index}"))

    def health(endpoint_id, *_args, **_kwargs):
        if endpoint_id == "endpoint-0":
            raise RuntimeError("endpoint missing")
        return {
            "workers": {"idle": 1, "unhealthy": 0},
            "jobs": {"inQueue": 0, "inProgress": 0},
        }

    monkeypatch.setattr(runpod_api, "endpoint_health_for_fingerprint", health)

    assert warm_pool.register(_record("replacement")) is True
    ids = {record.endpoint_id for record in warm_pool.candidates("signature", 0)}
    assert "endpoint-0" not in ids
    assert "replacement" in ids
    assert len(ids) == warm_pool.MAX_WARM_ENDPOINTS


def test_registry_drops_invalid_records_without_poisoning_valid_records(warm_pool_dir):
    from flash.providers.runpod import warm_pool

    warm_pool_dir.mkdir(parents=True)
    valid = asdict(_record("valid", released_at=2.0))
    invalid_values = {
        "string-timeout": {**valid, "endpoint_id": "string-timeout", "execution_timeout_ms": "1"},
        "negative-timeout": {
            **valid,
            "endpoint_id": "negative-timeout",
            "execution_timeout_ms": -1,
        },
        "non-finite": {**valid, "endpoint_id": "non-finite", "released_at": float("inf")},
        "empty-id": {**valid, "endpoint_id": ""},
        "extra-field": {**valid, "endpoint_id": "extra-field", "unexpected": True},
    }
    raw = {"valid": valid}
    raw.update(invalid_values)
    (warm_pool_dir / "endpoints.json").write_text(json.dumps(raw))

    records = warm_pool.candidates("signature", 0)

    assert [record.endpoint_id for record in records] == ["valid"]


def test_claim_has_single_thread_winner(warm_pool_dir):
    from flash.providers.runpod import warm_pool

    assert warm_pool.register(_record("shared"))
    barrier = threading.Barrier(2)

    def claim_once() -> bool:
        barrier.wait()
        return warm_pool.claim("shared")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: claim_once(), range(2)))

    assert sorted(results) == [False, True]


def _persist_terminal_run(
    tmp_path,
    monkeypatch,
    state: str,
    *,
    include_timeout: bool = True,
    include_signature: bool = True,
):
    from flash.providers.runpod import warm_pool
    from tests._helpers.runner import fresh_runner

    runner = fresh_runner(tmp_path, monkeypatch)
    spec = _spec(run_id=f"release-{state}")
    remote = {
        "provider": "runpod",
        "endpoint_id": f"endpoint-{state}",
        "endpoint_name": f"flash-4090-{state}",
        "key_fingerprint": _RUNPOD_FINGERPRINT,
        "job_id": f"job-{state}",
        "attempt": 0,
        "started_ts": 1.0,
    }
    if include_timeout:
        remote["execution_timeout_ms"] = 700_000
    if include_signature:
        remote["reuse_signature"] = warm_pool.reuse_signature(spec, _RUNPOD_FINGERPRINT)
    runner._save_status(
        runner.RunStatus(run_id=spec.run_id, state=state, spec=spec.to_dict(), remote=remote)
    )
    return runner, spec, remote


def test_successful_run_releases_endpoint_without_deleting(tmp_path, monkeypatch, warm_pool_dir):
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import train, warm_pool

    runner, spec, remote = _persist_terminal_run(tmp_path, monkeypatch, "done")
    monkeypatch.setattr(
        runpod_api,
        "cancel_job",
        lambda *_args, **_kwargs: pytest.fail("successful endpoint must not be cancelled"),
    )
    monkeypatch.setattr(
        runpod_api,
        "delete_endpoint_for_fingerprint",
        lambda *_args, **_kwargs: pytest.fail("successful endpoint must not be deleted"),
    )
    gc_calls = []
    monkeypatch.setattr(
        train,
        "terminate_endpoint",
        lambda *args, **kwargs: gc_calls.append((args, kwargs)) or [],
    )

    monkeypatch.setenv("FLASH_WORKER_IMAGE", "changed-after-deploy")
    runner._gc_run_endpoints(spec)

    signature = remote["reuse_signature"]
    assert warm_pool.reuse_signature(spec, _RUNPOD_FINGERPRINT) != signature
    records = warm_pool.candidates(signature, 700_000)
    assert [record.endpoint_id for record in records] == [remote["endpoint_id"]]
    assert records[0].execution_timeout_ms == 700_000
    assert gc_calls == [
        ((spec.gpu.type, spec.run_id), {"exclude_endpoint_id": remote["endpoint_id"]})
    ]
    stored_remote = runner.get_status(spec.run_id).remote
    assert stored_remote["released_to_warm_pool"] is True


def test_released_owner_handle_cannot_destroy_endpoint_after_claim(
    tmp_path, monkeypatch, warm_pool_dir
):
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import train, warm_pool
    from flash.runner.lifecycle import _strict_teardown_handle

    runner, spec, remote = _persist_terminal_run(tmp_path, monkeypatch, "done")
    monkeypatch.setattr(train, "terminate_endpoint", lambda *_args, **_kwargs: [])
    deleted = []
    monkeypatch.setattr(
        runpod_api,
        "delete_endpoint_for_fingerprint",
        lambda endpoint_id, fingerprint: deleted.append((endpoint_id, fingerprint)) or True,
    )

    runner._gc_run_endpoints(spec)
    assert warm_pool.claim(remote["endpoint_id"])
    released_remote = runner.get_status(spec.run_id).remote

    _strict_teardown_handle(released_remote)

    assert deleted == []


def test_successful_run_deletes_endpoint_when_pool_is_full(tmp_path, monkeypatch, warm_pool_dir):
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import train, warm_pool

    for index in range(warm_pool.MAX_WARM_ENDPOINTS):
        assert warm_pool.register(_record(f"occupied-{index}"))
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *_args, **_kwargs: {
            "workers": {"idle": 1, "unhealthy": 0},
            "jobs": {"inQueue": 0, "inProgress": 0},
        },
    )
    runner, spec, remote = _persist_terminal_run(tmp_path, monkeypatch, "done")
    monkeypatch.setattr(
        runpod_api,
        "cancel_job",
        lambda _endpoint_id, job_id, **_kwargs: {"id": job_id, "status": "CANCELLED"},
    )
    deleted = []
    monkeypatch.setattr(
        runpod_api,
        "delete_endpoint_for_fingerprint",
        lambda endpoint_id, fingerprint: deleted.append((endpoint_id, fingerprint)) or True,
    )
    monkeypatch.setattr(train, "terminate_endpoint", lambda *_args, **_kwargs: [])

    runner._gc_run_endpoints(spec)

    assert deleted == [(remote["endpoint_id"], _RUNPOD_FINGERPRINT)]


def test_legacy_handle_without_timeout_is_destroyed_not_warmed(
    tmp_path, monkeypatch, warm_pool_dir
):
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import train, warm_pool

    runner, spec, remote = _persist_terminal_run(
        tmp_path,
        monkeypatch,
        "done",
        include_timeout=False,
    )
    deleted = []
    monkeypatch.setattr(
        runpod_api,
        "cancel_job",
        lambda _endpoint_id, job_id, **_kwargs: {"id": job_id, "status": "CANCELLED"},
    )
    monkeypatch.setattr(
        runpod_api,
        "delete_endpoint_for_fingerprint",
        lambda endpoint_id, fingerprint: deleted.append((endpoint_id, fingerprint)) or True,
    )
    monkeypatch.setattr(train, "terminate_endpoint", lambda *_args, **_kwargs: [])

    runner._gc_run_endpoints(spec)

    assert warm_pool.candidates(remote["reuse_signature"], 0) == []
    assert deleted == [(remote["endpoint_id"], _RUNPOD_FINGERPRINT)]


@pytest.mark.parametrize("state", ["failed", "cancelled"])
def test_unsuccessful_run_deletes_endpoint_instead_of_releasing(
    tmp_path, monkeypatch, warm_pool_dir, state
):
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import train, warm_pool

    runner, spec, remote = _persist_terminal_run(tmp_path, monkeypatch, state)
    cancelled = []
    deleted = []
    monkeypatch.setattr(
        runpod_api,
        "cancel_job",
        lambda endpoint_id, job_id, **_kwargs: (
            cancelled.append((endpoint_id, job_id)) or {"id": job_id, "status": "CANCELLED"}
        ),
    )
    monkeypatch.setattr(
        runpod_api,
        "delete_endpoint_for_fingerprint",
        lambda endpoint_id, fingerprint: deleted.append((endpoint_id, fingerprint)) or True,
    )
    monkeypatch.setattr(train, "terminate_endpoint", lambda *_args, **_kwargs: [])

    runner._gc_run_endpoints(spec)

    signature = warm_pool.reuse_signature(spec, _RUNPOD_FINGERPRINT)
    assert warm_pool.candidates(signature, 0) == []
    assert cancelled == [(remote["endpoint_id"], remote["job_id"])]
    assert deleted == [(remote["endpoint_id"], _RUNPOD_FINGERPRINT)]


def test_retry_endpoint_gc_selection_preserves_warmed_endpoint():
    from types import SimpleNamespace

    from flash.providers.runpod.train.endpoints import _select_endpoint_resources

    resources = {
        "warm-sdk-uid": SimpleNamespace(id="warm-id", name="live-flash-4090-runhash"),
        "retry-1": SimpleNamespace(id="retry-1", name="live-flash-4090-runhashr1"),
        "retry-2": SimpleNamespace(id="retry-2", name="flash-4090-runhashr2"),
        "other": SimpleNamespace(id="other", name="flash-4090-other"),
    }

    selected = _select_endpoint_resources(
        resources,
        "flash-4090-runhash",
        exclude_endpoint_id="warm-id",
    )

    assert selected == ["retry-1", "retry-2"]


def test_retry_endpoint_gc_delete_loops_preserve_warmed_endpoint(monkeypatch):
    from types import SimpleNamespace

    from flash.providers.runpod import train
    from tests.test_cancel_remote import _install_fake_sdk, target_for

    target = target_for("flash-q-1")
    undeployed = []

    async def undeploy(uid, **_kwargs):
        undeployed.append(uid)
        return {"success": True}

    rest_deleted = []
    ep_mod, _target = _install_fake_sdk(
        monkeypatch,
        resources={
            "warm-sdk-uid": SimpleNamespace(id="warm-id", name=f"live-{target}"),
            "retry-sdk-uid": SimpleNamespace(id="retry-sdk", name=f"live-{target}r1"),
        },
        undeploy=undeploy,
        rest_find=lambda _target: [
            {"id": "warm-id", "name": target},
            {"id": "retry-rest", "name": f"{target}r2"},
        ],
        rest_delete=lambda endpoint_id: rest_deleted.append(endpoint_id) or True,
    )

    train.terminate_endpoint(
        "RTX 5090",
        "flash-q-1",
        exclude_endpoint_id="warm-id",
    )

    assert undeployed == ["retry-sdk-uid"]
    assert rest_deleted == ["retry-rest"]
    assert target in ep_mod._ACQUIRED


def _stub_submit_dependencies(monkeypatch):
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import auth, jobs, train

    submitted = []
    monkeypatch.setattr(auth, "ensure_auth", lambda: "runpod-key")
    monkeypatch.setattr(runpod_api, "key_fingerprint", lambda _key: _RUNPOD_FINGERPRINT)
    monkeypatch.setattr(train, "build_worker_env", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(train, "chalk_extra_pip", lambda _spec: [])
    monkeypatch.setattr(jobs, "weight_cache_endpoint_kwargs", lambda _spec: {"volume": [object()]})
    monkeypatch.setattr(jobs, "build_function_input", lambda payload: payload)
    monkeypatch.setattr(
        runpod_api,
        "submit_job",
        lambda endpoint_id, payload, **_kwargs: submitted.append((endpoint_id, payload)) or "job-1",
    )
    monkeypatch.setattr(
        jobs, "poll_job", lambda *_args, **_kwargs: jobs.PollResult(True, metrics={})
    )
    return jobs, runpod_api, submitted


def _submit_deadline() -> float:
    return time.time() + 600.0


def test_submit_reuses_matching_healthy_warm_endpoint(monkeypatch, warm_pool_dir):
    from flash.providers.runpod import warm_pool

    jobs, runpod_api, submitted = _stub_submit_dependencies(monkeypatch)
    spec = _spec(run_id="reuse-healthy")
    signature = warm_pool.reuse_signature(spec, _RUNPOD_FINGERPRINT)
    assert warm_pool.register(
        _record("warm-endpoint", signature=signature, execution_timeout_ms=700_000)
    )
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *_args, **_kwargs: {
            "workers": {"ready": 1, "idle": 1, "unhealthy": 0},
            "jobs": {"inQueue": 0, "inProgress": 0},
        },
    )
    monkeypatch.setattr(
        jobs,
        "deploy_train_endpoint",
        lambda *_args, **_kwargs: pytest.fail("matching warm endpoint must be reused"),
    )

    events = []

    def submit_job(endpoint_id, payload, **_kwargs):
        events.append(("submit", endpoint_id))
        submitted.append((endpoint_id, payload))
        return "job-1"

    monkeypatch.setattr(runpod_api, "submit_job", submit_job)
    handles = []

    def on_handle(handle):
        events.append(("handle", handle.get("job_id")))
        handles.append(handle)

    result = jobs.submit_run(
        spec,
        spec.seed,
        on_handle=on_handle,
        code_prefix="code/revision/flash",
        deadline_at=_submit_deadline(),
    )

    assert result.ok
    assert submitted[0][0] == "warm-endpoint"
    # the durable handle is persisted once, after submit (job_id set), same as the fresh-deploy path.
    assert events == [("submit", "warm-endpoint"), ("handle", "job-1")]
    assert handles[0]["job_id"] == "job-1"
    assert handles[0]["reuse_signature"] == signature
    assert handles[0]["execution_timeout_ms"] == 700_000
    assert warm_pool.claim("warm-endpoint") is False


def test_acquire_matches_each_records_owning_account(monkeypatch, warm_pool_dir):
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import warm_pool

    other_fingerprint = "rpk-ffffffffffff"
    spec = _spec(run_id="reuse-other-account")
    signature = warm_pool.reuse_signature(spec, other_fingerprint)
    assert warm_pool.register(
        _record(
            "other-account-endpoint",
            signature=signature,
            key_fingerprint=other_fingerprint,
            execution_timeout_ms=700_000,
        )
    )
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda endpoint_id, fingerprint, **_kwargs: (
            {
                "workers": {"idle": 1, "unhealthy": 0},
                "jobs": {"inQueue": 0, "inProgress": 0},
            }
            if (endpoint_id, fingerprint) == ("other-account-endpoint", other_fingerprint)
            else pytest.fail("health must use the record's owning fingerprint")
        ),
    )

    acquired = warm_pool.acquire(spec, 600_000, _submit_deadline(), has_volume=True)

    assert acquired is not None
    assert acquired.key_fingerprint == other_fingerprint


def test_busy_warm_endpoint_is_skipped_without_pruning(monkeypatch, warm_pool_dir):
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import warm_pool

    spec = _spec(run_id="reuse-busy")
    signature = warm_pool.reuse_signature(spec, _RUNPOD_FINGERPRINT)
    assert warm_pool.register(_record("busy-endpoint", signature=signature))
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *_args, **_kwargs: {
            "workers": {"idle": 0, "unhealthy": 0},
            "jobs": {"inQueue": 0, "inProgress": 1},
        },
    )

    acquired = warm_pool.acquire(
        spec,
        1,
        _submit_deadline(),
        has_volume=True,
    )

    assert acquired is None
    assert [record.endpoint_id for record in warm_pool.candidates(signature, 0)] == [
        "busy-endpoint"
    ]


def test_reused_endpoint_is_deleted_when_queue_submit_fails(monkeypatch, warm_pool_dir):
    from flash.providers.runpod import warm_pool

    jobs, runpod_api, _submitted = _stub_submit_dependencies(monkeypatch)
    spec = _spec(run_id="reuse-submit-failure")
    signature = warm_pool.reuse_signature(spec, _RUNPOD_FINGERPRINT)
    assert warm_pool.register(
        _record("warm-endpoint", signature=signature, execution_timeout_ms=700_000)
    )
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *_args, **_kwargs: {
            "workers": {"idle": 1, "unhealthy": 0},
            "jobs": {"inQueue": 0, "inProgress": 0},
        },
    )
    monkeypatch.setattr(
        jobs,
        "deploy_train_endpoint",
        lambda *_args, **_kwargs: pytest.fail("matching warm endpoint must be reused"),
    )
    original = RuntimeError("ambiguous queue submit")
    monkeypatch.setattr(
        runpod_api,
        "submit_job",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(original),
    )
    deleted = []
    monkeypatch.setattr(
        runpod_api,
        "delete_endpoint_for_fingerprint",
        lambda endpoint_id, fingerprint: deleted.append((endpoint_id, fingerprint)) or True,
    )

    with pytest.raises(RuntimeError) as caught:
        jobs.submit_run(
            spec,
            spec.seed,
            code_prefix="code/revision/flash",
            deadline_at=_submit_deadline(),
        )

    assert caught.value is original
    assert deleted == [("warm-endpoint", _RUNPOD_FINGERPRINT)]
    assert warm_pool.claim("warm-endpoint") is False


def test_submit_prunes_stale_candidate_and_creates_fresh(monkeypatch, warm_pool_dir):
    from flash.providers.runpod import warm_pool

    jobs, runpod_api, submitted = _stub_submit_dependencies(monkeypatch)
    spec = _spec(run_id="reuse-stale")
    signature = warm_pool.reuse_signature(spec, _RUNPOD_FINGERPRINT)
    assert warm_pool.register(_record("stale-endpoint", signature=signature))
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("endpoint missing")),
    )
    deploys = []
    monkeypatch.setattr(
        jobs,
        "deploy_train_endpoint",
        lambda *_args, **_kwargs: (
            deploys.append(True) or ("fresh-endpoint", "fresh-name", _RUNPOD_FINGERPRINT)
        ),
    )

    result = jobs.submit_run(
        spec,
        spec.seed,
        code_prefix="code/revision/flash",
        deadline_at=_submit_deadline(),
    )

    assert result.ok
    assert deploys == [True]
    assert submitted[0][0] == "fresh-endpoint"
    assert warm_pool.claim("stale-endpoint") is False


def test_submit_ignores_candidate_with_insufficient_timeout(monkeypatch, warm_pool_dir):
    from flash.providers.runpod import warm_pool

    jobs, runpod_api, submitted = _stub_submit_dependencies(monkeypatch)
    spec = _spec(run_id="reuse-short-timeout")
    signature = warm_pool.reuse_signature(spec, _RUNPOD_FINGERPRINT)
    assert warm_pool.register(
        _record("short-endpoint", signature=signature, execution_timeout_ms=1)
    )
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *_args, **_kwargs: pytest.fail("short timeout candidate must not be offered"),
    )
    deploys = []
    monkeypatch.setattr(
        jobs,
        "deploy_train_endpoint",
        lambda *_args, **_kwargs: (
            deploys.append(True) or ("fresh-endpoint", "fresh-name", _RUNPOD_FINGERPRINT)
        ),
    )

    result = jobs.submit_run(
        spec,
        spec.seed,
        code_prefix="code/revision/flash",
        deadline_at=_submit_deadline(),
    )

    assert result.ok
    assert deploys == [True]
    assert submitted[0][0] == "fresh-endpoint"


def test_submit_without_candidates_uses_existing_fresh_path(monkeypatch, warm_pool_dir):
    jobs, runpod_api, submitted = _stub_submit_dependencies(monkeypatch)
    spec = _spec(run_id="reuse-none")
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *_args, **_kwargs: pytest.fail("no candidate should trigger no health call"),
    )
    deploys = []
    monkeypatch.setattr(
        jobs,
        "deploy_train_endpoint",
        lambda *_args, **_kwargs: (
            deploys.append(True) or ("fresh-endpoint", "fresh-name", _RUNPOD_FINGERPRINT)
        ),
    )

    result = jobs.submit_run(
        spec,
        spec.seed,
        code_prefix="code/revision/flash",
        deadline_at=_submit_deadline(),
    )

    assert result.ok
    assert deploys == [True]
    assert submitted[0][0] == "fresh-endpoint"


def test_fresh_submit_persists_deploy_time_signature_and_timeout(monkeypatch, warm_pool_dir):
    from flash.providers.runpod import warm_pool

    jobs, _runpod_api, _submitted = _stub_submit_dependencies(monkeypatch)
    spec = _spec(run_id="reuse-capture")
    deployed = {}

    def deploy(*_args, **kwargs):
        deployed.update(kwargs)
        return "fresh-endpoint", "fresh-name", _RUNPOD_FINGERPRINT

    monkeypatch.setattr(jobs, "worker_image_for_gpu", lambda *_args, **_kwargs: "image-at-deploy")
    monkeypatch.setattr(jobs, "deploy_train_endpoint", deploy)
    handles = []

    result = jobs.submit_run(
        spec,
        spec.seed,
        on_handle=lambda handle: handles.append(handle),
        code_prefix="code/revision/flash",
        deadline_at=_submit_deadline(),
    )

    assert result.ok
    assert deployed["worker_image"] == "image-at-deploy"
    assert handles[-1]["reuse_signature"] == warm_pool.reuse_signature(
        spec,
        _RUNPOD_FINGERPRINT,
        worker_image="image-at-deploy",
        has_volume=True,
    )
    assert handles[-1]["execution_timeout_ms"] == deployed["execution_timeout_ms"]
