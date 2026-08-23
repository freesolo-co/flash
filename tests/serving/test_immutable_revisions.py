from __future__ import annotations

import asyncio
import itertools
from typing import Any
from urllib.parse import quote

import httpx
import pytest
from fastapi import BackgroundTasks, HTTPException, Request
from fastapi.testclient import TestClient

from flash.serving.src import adapter_routes
from flash.serving.src.adapter_routes import remove_adapter
from flash.serving.src.persistence import (
    PersistenceConflict,
    PersistenceRecordError,
    PersistenceReferenceError,
)
from flash.serving.src.router import AdapterRouter, build_serving_app
from flash.serving.src.schemas import (
    AdapterRecord,
    ImmutableAdapterRegistration,
    PersistedAdapterRecord,
    internal_adapter_payload,
)
from tests.serving.conftest import attest

QWEN = "Qwen/Qwen3.5-0.8B"
QWEN_2B = "Qwen/Qwen3.5-2B"
RUN_ID = "flash-1234567890-abcdef12"
SHA_A = "a" * 40
SHA_B = "b" * 40
REVISION_A = f"{RUN_ID}@step-20.{SHA_A}"
REVISION_B = f"{RUN_ID}@step-40.{SHA_B}"
FINAL_REVISION = f"{RUN_ID}@final.{SHA_A}"
INTERNAL_HEADERS = {"X-Freesolo-Internal-Key": "secret"}


def _registration(step: int = 20, sha: str = SHA_A, **overrides: object) -> dict[str, object]:
    revision_id = f"{RUN_ID}@step-{step}.{sha}"
    payload: dict[str, object] = {
        "adapter_id": revision_id,
        "repo_id": "org/run",
        "base_model": QWEN,
        "subfolder": f"checkpoints/step-{step}",
        "repo_type": "model",
        "org_id": "org-1",
        "url": "https://huggingface.co/org/run",
        "checkpoint": f"{RUN_ID}/step-{step}",
        "private": True,
        "thinking": False,
        "structured_outputs": None,
        "metadata": {
            "record_type": "revision",
            "run_id": RUN_ID,
            "checkpoint_step": step,
            "hf_revision": sha,
        },
    }
    payload.update(overrides)
    return payload


def _final_registration(**overrides: object) -> dict[str, object]:
    payload = _registration(
        adapter_id=FINAL_REVISION,
        checkpoint=RUN_ID,
        metadata={
            "record_type": "revision",
            "run_id": RUN_ID,
            "checkpoint_step": None,
            "hf_revision": SHA_A,
        },
    )
    payload.update(overrides)
    return payload


class MemoryPersistence:
    def __init__(self) -> None:
        self.rows: dict[str, AdapterRecord] = {}
        self.inserts: list[str] = []
        self.replacements: list[str] = []
        self.force_replace_miss = False
        self.force_replace_miss_ids: set[str] = set()
        self.concurrent_disable_once: set[str] = set()
        self.concurrent_redeploy_once: dict[str, str] = {}
        self.get_failure_ids: set[str] = set()
        # `_get_stored` re-raises PersistenceRecordError but maps every *other* storage failure to
        # a 503. a fake that only ever raises the former cannot reach the handlers that recover
        # that 503, so a transport-level outage is modelled separately.
        self.get_transport_failure_ids: set[str] = set()
        self.insert_failure_ids: set[str] = set()
        self.replace_failure_ids: set[str] = set()
        self._clock = itertools.count(1)

    def _stamp(self, record: AdapterRecord) -> AdapterRecord:
        timestamp = f"2026-07-14T00:00:{next(self._clock):02d}+00:00"
        return PersistedAdapterRecord.model_validate(
            {
                **internal_adapter_payload(record),
                "created_at": record.created_at or timestamp,
                "updated_at": timestamp,
                "deployment_generation": record.deployment_generation,
            }
        )

    def get(self, adapter_id: str, _settings: object) -> AdapterRecord | None:
        if adapter_id in self.get_failure_ids:
            raise PersistenceRecordError("storage unavailable")
        if adapter_id in self.get_transport_failure_ids:
            raise OSError("connection reset by peer")
        return self.rows.get(adapter_id)

    def list_run(self, run_id: str, _settings: object) -> list[AdapterRecord]:
        return [record for record in self.rows.values() if record.run_id == run_id]

    def insert(self, record: AdapterRecord, _settings: object) -> AdapterRecord:
        if record.adapter_id in self.insert_failure_ids:
            raise OSError("storage unavailable")
        if record.adapter_id in self.rows:
            raise PersistenceConflict("exists")
        stored = self._stamp(record)
        self.rows[stored.adapter_id] = stored
        self.inserts.append(stored.adapter_id)
        return stored

    def replace(
        self,
        record: AdapterRecord,
        *,
        expected_updated_at: str,
        settings: object,
    ) -> AdapterRecord | None:
        del settings
        if record.adapter_id in self.replace_failure_ids:
            raise PersistenceRecordError("storage unavailable")
        current = self.rows.get(record.adapter_id)
        redeployed_generation = self.concurrent_redeploy_once.pop(record.adapter_id, None)
        if redeployed_generation is not None and current is not None:
            self.rows[record.adapter_id] = self._stamp(
                current.model_copy(
                    update={
                        "status": "ready",
                        "deployment_generation": redeployed_generation,
                    }
                )
            )
            return None
        if record.adapter_id in self.concurrent_disable_once and current is not None:
            self.concurrent_disable_once.remove(record.adapter_id)
            self.rows[record.adapter_id] = self._stamp(
                current.model_copy(update={"status": "disabled"})
            )
            return None
        if (
            self.force_replace_miss
            or record.adapter_id in self.force_replace_miss_ids
            or current is None
            or current.updated_at != expected_updated_at
        ):
            return None
        stored = self._stamp(record)
        self.rows[stored.adapter_id] = stored
        self.replacements.append(stored.adapter_id)
        return stored


class FakePool:
    def __init__(self) -> None:
        self.registered: list[str] = []
        self.registered_generations: list[str | None] = []
        self.generated: list[tuple[str, str, str]] = []
        self.unregistered: list[str] = []
        self.unregistered_generations: list[str | None] = []
        self.loaded_generations: dict[str, str | None] = {}

    async def register(self, base_model: str, record: AdapterRecord) -> None:
        assert base_model == record.base_model
        self.registered.append(record.adapter_id)
        self.registered_generations.append(record.deployment_generation)
        self.loaded_generations[record.adapter_id] = record.deployment_generation

    async def unregister(
        self,
        base_model: str,
        adapter_id: str,
        expected_generation: str | None = None,
    ) -> None:
        del base_model
        self.unregistered.append(adapter_id)
        self.unregistered_generations.append(expected_generation)
        if self.loaded_generations.get(adapter_id) == expected_generation:
            self.loaded_generations.pop(adapter_id, None)

    async def generate(
        self,
        base_model: str,
        payload: Any,
        record: AdapterRecord,
        *,
        expected_checkpoint: str | None = None,
    ) -> dict[str, Any]:
        del expected_checkpoint
        self.generated.append((base_model, payload.adapter_id, record.adapter_id))
        return attest(
            record,
            {
                "ok": True,
                "adapter_id": payload.adapter_id,
                "text": record.adapter_id,
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "checkpoint": record.checkpoint,
            },
        )

    async def stream_generate(
        self,
        base_model: str,
        payload: Any,
        record: AdapterRecord,
        *,
        expected_checkpoint: str | None = None,
    ):
        del base_model, payload, record, expected_checkpoint
        yield {"type": "ready", "checkpoint": ""}
        yield {"type": "final", "prompt_tokens": 1, "completion_tokens": 1}


@pytest.fixture
def setup(monkeypatch):
    persistence = MemoryPersistence()
    pool = FakePool()
    router = AdapterRouter()
    monkeypatch.setattr("flash.serving.src.persistence.get_adapter", persistence.get)
    monkeypatch.setattr("flash.serving.src.persistence.list_run_adapters", persistence.list_run)
    monkeypatch.setattr("flash.serving.src.persistence.insert_adapter", persistence.insert)
    monkeypatch.setattr("flash.serving.src.persistence.replace_adapter_cas", persistence.replace)
    client = TestClient(
        build_serving_app(pool, router, internal_key="secret"),
        headers=INTERNAL_HEADERS,
    )
    return client, pool, router, persistence


def _register(client: TestClient, payload: dict[str, object]) -> Any:
    return client.post("/adapters", json=payload)


def test_legacy_and_direct_alias_registration_fail_without_write(setup) -> None:
    client, _, _, persistence = setup
    legacy = _registration()
    legacy.pop("metadata")
    assert _register(client, legacy).status_code == 422

    direct_alias = _registration()
    direct_alias["metadata"] = {
        "record_type": "alias",
        "run_id": RUN_ID,
        "alias_of": REVISION_A,
    }
    assert _register(client, direct_alias).status_code == 422
    assert persistence.rows == {}


@pytest.mark.parametrize("run_id", ["bad/alias", ".hidden", "trailing-", "a" * 97])
def test_registration_rejects_non_route_safe_run_id_without_write(setup, run_id: str) -> None:
    client, pool, _, persistence = setup
    payload = _registration(
        adapter_id=f"{run_id}@step-20.{SHA_A}",
        checkpoint=f"{run_id}/step-20",
    )
    payload["metadata"] = {**payload["metadata"], "run_id": run_id}

    response = _register(client, payload)
    assert response.status_code == 422
    assert persistence.rows == {}
    assert pool.registered == []


def test_base_model_namespace_collision_rejects_before_write(setup) -> None:
    client, pool, router, persistence = setup
    base_record = AdapterRecord.model_validate(
        {
            "adapter_id": RUN_ID,
            "repo_id": QWEN,
            "base_model": QWEN,
            "serve_base_model": True,
            "thinking": True,
        }
    )
    router.upsert(base_record)

    response = _register(client, _registration())
    assert response.status_code == 409
    assert persistence.rows == {}
    assert persistence.inserts == []
    assert pool.registered == []
    assert router.get(RUN_ID) is base_record


def test_new_registration_creates_disabled_revision_and_alias_then_marks_only_revision_ready(
    setup,
) -> None:
    client, pool, router, persistence = setup
    response = _register(client, _registration())
    assert response.status_code == 200
    assert response.json()["status"] == "disabled"
    assert persistence.inserts == [RUN_ID, REVISION_A]
    assert persistence.rows[RUN_ID].status == "disabled"
    assert persistence.rows[RUN_ID].metadata == {
        "record_type": "alias",
        "run_id": RUN_ID,
        "alias_of": REVISION_A,
    }
    committed = persistence.rows[REVISION_A]
    assert committed.status == "ready"
    generation = committed.deployment_generation
    assert generation is not None
    assert persistence.replacements == [REVISION_A, REVISION_A]
    assert pool.registered == [REVISION_A]
    assert pool.registered_generations == [generation]
    assert pool.unregistered == []
    assert pool.loaded_generations == {REVISION_A: generation}
    assert router.get(REVISION_A) == committed
    assert router.resolve(REVISION_A) is not None
    assert router.resolve(RUN_ID) is None


def test_registration_recovers_disabled_alias_after_revision_insert_outage(setup) -> None:
    client, pool, _, persistence = setup
    persistence.insert_failure_ids.add(REVISION_A)

    failed = _register(client, _registration())

    assert failed.status_code == 503
    orphan = persistence.rows[RUN_ID]
    assert orphan.status == "disabled"
    assert orphan.alias_of == REVISION_A
    persistence.insert_failure_ids.clear()

    recovered = _register(client, _registration(step=40, sha=SHA_B))

    assert recovered.status_code == 200
    assert persistence.rows[RUN_ID].alias_of == REVISION_B
    assert REVISION_B in persistence.rows
    assert pool.registered == [REVISION_B]


def test_registration_promotion_http_error_unloads_unrecorded_adapter(setup, monkeypatch) -> None:
    client, pool, router, persistence = setup
    replace = adapter_routes._replace_stored_cas

    async def fail_promotion(record: AdapterRecord, *, expected_updated_at: str):
        if record.status == "ready":
            raise HTTPException(503, "adapter storage is unavailable")
        return await replace(record, expected_updated_at=expected_updated_at)

    monkeypatch.setattr(adapter_routes, "_replace_stored_cas", fail_promotion)

    response = _register(client, _registration())

    assert response.status_code == 200
    assert persistence.rows[REVISION_A].status == "disabled"
    assert router.resolve(REVISION_A) is None
    assert pool.loaded_generations == {}, "failed promotion left an unrecorded adapter loaded"
    assert pool.unregistered == [REVISION_A]
    assert pool.unregistered_generations == [pool.registered_generations[0]]


def test_registration_promotion_cas_miss_unloads_unrecorded_adapter(setup, monkeypatch) -> None:
    client, pool, router, persistence = setup
    replace = adapter_routes._replace_stored_cas
    missed_promotion = False

    async def miss_promotion(record: AdapterRecord, *, expected_updated_at: str):
        nonlocal missed_promotion
        if record.status == "ready" and not missed_promotion:
            missed_promotion = True
            return None
        return await replace(record, expected_updated_at=expected_updated_at)

    monkeypatch.setattr(adapter_routes, "_replace_stored_cas", miss_promotion)

    response = _register(client, _registration())

    assert response.status_code == 200
    assert persistence.rows[REVISION_A].status == "disabled"
    assert router.resolve(REVISION_A) is None
    assert pool.loaded_generations == {}, "cas loser left an unrecorded adapter loaded"
    assert pool.unregistered == [REVISION_A]
    assert pool.unregistered_generations == [pool.registered_generations[0]]


def test_registration_reconciliation_preserves_concurrent_promotion_winner(
    setup, monkeypatch
) -> None:
    client, pool, router, persistence = setup
    promotion_missed = False
    replace = adapter_routes._replace_stored_cas

    async def concurrent_winner(record: AdapterRecord, *, expected_updated_at: str):
        nonlocal promotion_missed
        if record.status == "ready" and not promotion_missed:
            promotion_missed = True
            persistence.rows[record.adapter_id] = persistence._stamp(record)
            return None
        return await replace(record, expected_updated_at=expected_updated_at)

    monkeypatch.setattr(adapter_routes, "_replace_stored_cas", concurrent_winner)

    response = _register(client, _registration())

    assert response.status_code == 200
    winner = persistence.rows[REVISION_A]
    assert winner.status == "ready"
    assert router.get(REVISION_A) == winner
    assert pool.unregistered == [], "reconciliation unloaded the concurrent promotion winner"
    assert pool.loaded_generations == {REVISION_A: winner.deployment_generation}, (
        "concurrent promotion winner lost its loaded adapter"
    )


def test_registration_reconciliation_survives_unload_failure(setup, monkeypatch) -> None:
    client, pool, _, persistence = setup
    replace = adapter_routes._replace_stored_cas

    async def miss_promotion(record: AdapterRecord, *, expected_updated_at: str):
        if record.status == "ready":
            return None
        return await replace(record, expected_updated_at=expected_updated_at)

    async def fail_unregister(
        _base_model: str,
        _adapter_id: str,
        _expected_generation: str | None = None,
    ) -> None:
        raise RuntimeError("engine unavailable")

    monkeypatch.setattr(adapter_routes, "_replace_stored_cas", miss_promotion)
    pool.unregister = fail_unregister

    response = _register(client, _registration())

    assert response.status_code == 200
    assert persistence.rows[REVISION_A].status == "disabled"
    assert pool.loaded_generations == {REVISION_A: pool.registered_generations[0]}, (
        "failed cleanup disturbed the loaded adapter"
    )


def test_registration_response_loss_reconciles_remote_success(setup) -> None:
    client, pool, router, persistence = setup
    register = pool.register

    async def lose_response(base_model: str, record: AdapterRecord) -> None:
        await register(base_model, record)
        raise RuntimeError("registration response lost")

    pool.register = lose_response

    response = _register(client, _registration())

    assert response.status_code == 200
    assert persistence.rows[REVISION_A].status == "disabled"
    assert persistence.rows[REVISION_A].deployment_generation is None
    assert router.resolve(REVISION_A) is None
    assert pool.loaded_generations == {}, "response loss left an unrecorded adapter loaded"
    assert pool.unregistered == [REVISION_A]
    assert pool.unregistered_generations == [pool.registered_generations[0]]


def test_get_loading_revision_reloads_status_without_making_it_routable(setup) -> None:
    _, pool, _, persistence = setup
    revision = ImmutableAdapterRegistration.model_validate(_registration()).to_record()
    stored = persistence.insert(revision, object())
    loading = persistence.replace(
        stored.model_copy(update={"deployment_generation": stored.updated_at}),
        expected_updated_at=stored.updated_at,
        settings=object(),
    )
    assert loading is not None
    router = AdapterRouter()
    client = TestClient(
        build_serving_app(
            pool,
            router,
            internal_key="secret",
            reload_records=lambda: [
                record for record in persistence.rows.values() if record.status == "ready"
            ],
            lookup_record=lambda adapter_id: persistence.get(adapter_id, object()),
        ),
        headers=INTERNAL_HEADERS,
    )
    encoded_revision = quote(REVISION_A, safe="")

    response = client.get(f"/adapters/{encoded_revision}")

    assert response.status_code == 200
    record = response.json()["adapter"]
    assert record["status"] == "loading"
    assert record["lifecycle_state"] == "loading"
    assert "deployment_generation" not in record
    assert router.get(REVISION_A) is None
    assert router.resolve(REVISION_A) is None
    inference = client.post("/generate", json={"adapter_id": REVISION_A, "prompt": "hello"})
    assert inference.status_code == 404


def test_status_lookup_without_targeted_fetch_keeps_bulk_reload_fallback(setup) -> None:
    _, pool, _, _ = setup
    loading = ImmutableAdapterRegistration.model_validate(_registration()).to_record().model_copy(
        update={"deployment_generation": "generation-1"}
    )
    router = AdapterRouter()
    client = TestClient(
        build_serving_app(
            pool,
            router,
            internal_key="secret",
            reload_records=lambda: [loading],
        ),
        headers=INTERNAL_HEADERS,
    )

    response = client.get(f"/adapters/{quote(REVISION_A, safe='')}")

    assert response.status_code == 200
    assert response.json()["adapter"]["lifecycle_state"] == "loading"
    assert router.get(REVISION_A) == loading
    assert router.resolve(REVISION_A) is None


def test_targeted_status_lookup_does_not_replace_ready_routing_registry(setup) -> None:
    _, pool, _, persistence = setup
    cached_ready = ImmutableAdapterRegistration.model_validate(_registration()).to_record().model_copy(
        update={"status": "ready"}
    )
    loading_revision = ImmutableAdapterRegistration.model_validate(
        _registration(step=40, sha=SHA_B)
    ).to_record()
    stored = persistence.insert(loading_revision, object())
    loading = persistence.replace(
        stored.model_copy(update={"deployment_generation": stored.updated_at}),
        expected_updated_at=stored.updated_at,
        settings=object(),
    )
    assert loading is not None
    reloads = {"count": 0}

    def _reload_ready() -> list[AdapterRecord]:
        reloads["count"] += 1
        return [record for record in persistence.rows.values() if record.status == "ready"]

    router = AdapterRouter([cached_ready])
    client = TestClient(
        build_serving_app(
            pool,
            router,
            internal_key="secret",
            reload_records=_reload_ready,
            lookup_record=lambda adapter_id: persistence.get(adapter_id, object()),
        ),
        headers=INTERNAL_HEADERS,
    )

    response = client.get(f"/adapters/{quote(REVISION_B, safe='')}")

    assert response.status_code == 200
    assert response.json()["adapter"]["lifecycle_state"] == "loading"
    assert reloads["count"] == 0
    assert router.get(REVISION_B) is None
    assert router.resolve(REVISION_A) == (cached_ready, cached_ready)


def test_get_revision_reports_disabled_until_ready_then_returns_full_identity(setup) -> None:
    client, pool, _, persistence = setup
    register = pool.register

    async def fail_register(_base_model: str, _record: AdapterRecord) -> None:
        raise RuntimeError("still loading")

    pool.register = fail_register
    assert _register(client, _registration()).status_code == 200
    assert persistence.rows[REVISION_A].status == "disabled"
    encoded_revision = quote(REVISION_A, safe="")
    disabled = client.get(f"/adapters/{encoded_revision}")
    assert disabled.status_code == 200
    assert disabled.json()["adapter"]["status"] == "disabled"
    assert disabled.json()["adapter"]["lifecycle_state"] == "disabled"

    pool.register = register
    assert _register(client, _registration()).status_code == 200
    response = client.get(f"/adapters/{encoded_revision}")
    assert response.status_code == 200
    assert response.json()["ok"] is True

    record = response.json()["adapter"]
    expected = _registration()
    for field in (
        "adapter_id",
        "repo_id",
        "repo_type",
        "subfolder",
        "base_model",
        "checkpoint",
        "thinking",
    ):
        assert record[field] == expected[field]
    assert record["org_id"] == "org-1"
    assert record["metadata"] == expected["metadata"]
    assert record["lifecycle_state"] == "ready"


def test_get_unknown_adapter_returns_not_found(setup) -> None:
    client, _, _, _ = setup
    assert client.get("/adapters/unknown").status_code == 404


def test_get_adapter_requires_internal_key(setup) -> None:
    _, pool, router, _ = setup
    client = TestClient(build_serving_app(pool, router, internal_key="secret"))
    response = client.get(f"/adapters/{quote(REVISION_A, safe='')}")
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid serving internal key"


def test_retry_completes_a_matching_disabled_alias_claim(setup) -> None:
    client, pool, _, persistence = setup
    revision = ImmutableAdapterRegistration.model_validate(_registration()).to_record()
    alias = revision.model_copy(
        update={
            "adapter_id": RUN_ID,
            "checkpoint": None,
            "metadata": {
                "record_type": "alias",
                "run_id": RUN_ID,
                "alias_of": REVISION_A,
            },
        }
    )
    persistence.insert(alias, object())

    response = _register(client, _registration())
    assert response.status_code == 200
    assert persistence.inserts == [RUN_ID, REVISION_A]
    assert persistence.rows[REVISION_A].status == "ready"
    assert pool.registered == [REVISION_A]


def test_unresolvable_org_is_permanent_not_a_retryable_outage(setup, monkeypatch) -> None:
    """SERVE-020: registering under an org that does not exist must not look like an outage.

    The org_id FK violation arrives as PostgREST 409, the same status as a duplicate row. Treating
    it as a conflict makes the router read back a row that was never inserted, find nothing, and
    answer 503 -- which the flash client retries (it only fails fast below 500). The registration
    can never succeed, so the caller must be told permanently and told which value is wrong.
    """
    client, pool, _, _persistence = setup

    def _reject_unknown_org(record: AdapterRecord, _settings: object) -> AdapterRecord:
        raise PersistenceReferenceError(
            f"hosted LoRA adapter references a row that does not exist (org_id={record.org_id!r}): "
            'Key (org_id)=(dev-org) is not present in table "orgs".'
        )

    # the fixture already bound src.persistence.insert_adapter, so rebind that same symbol --
    # replacing persistence.insert here would leave the router calling the fixture's original.
    monkeypatch.setattr("flash.serving.src.persistence.insert_adapter", _reject_unknown_org)

    response = _register(client, _registration(org_id="dev-org"))

    assert response.status_code == 422, "a permanent misconfiguration must not be a 5xx"
    assert response.status_code < 500, "flash retries 5xx; this can never succeed"
    assert "dev-org" in response.json()["detail"]
    assert pool.registered == []


def test_unclassified_storage_conflict_does_not_trigger_duplicate_readback(
    setup, monkeypatch
) -> None:
    client, pool, _, _ = setup
    request = httpx.Request("POST", "https://example.invalid/hosted_lora_adapters")
    response = httpx.Response(409, request=request)

    def _reject_unclassified_conflict(_record: AdapterRecord, _settings: object) -> AdapterRecord:
        raise httpx.HTTPStatusError("unclassified conflict", request=request, response=response)

    reads = 0

    def _read_required_namespaces(*_args: object, **_kwargs: object) -> None:
        nonlocal reads
        reads += 1
        if reads > 2:
            pytest.fail("unclassified 409 must not enter conflict readback")

    monkeypatch.setattr(
        "flash.serving.src.persistence.insert_adapter", _reject_unclassified_conflict
    )
    monkeypatch.setattr("flash.serving.src.persistence.get_adapter", _read_required_namespaces)

    result = _register(client, _registration())

    assert result.status_code == 503
    assert result.json()["detail"] == "adapter storage is unavailable"
    assert reads == 2, "only the required revision and alias reads should occur"
    assert pool.registered == []


def test_exact_duplicate_is_idempotent_and_disabled_repost_retriggers_load(setup) -> None:
    client, pool, _, persistence = setup
    assert _register(client, _registration()).status_code == 200
    assert _register(client, _registration()).status_code == 200
    assert persistence.inserts == [RUN_ID, REVISION_A]
    assert pool.registered == [REVISION_A]

    disabled = persistence._stamp(
        persistence.rows[REVISION_A].model_copy(update={"status": "disabled"})
    )
    persistence.rows[REVISION_A] = disabled
    assert _register(client, _registration()).status_code == 200
    assert pool.registered == [REVISION_A, REVISION_A]
    assert pool.registered_generations[-1] == disabled.updated_at
    assert persistence.rows[REVISION_A].status == "ready"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repo_id", "org/other"),
        ("base_model", QWEN_2B),
        ("subfolder", "other/path"),
        ("repo_type", "dataset"),
        ("url", "https://huggingface.co/org/other"),
        ("private", False),
        ("thinking", True),
        ("structured_outputs", {"json_object": True}),
    ],
)
def test_every_changed_immutable_field_conflicts(setup, field: str, value: object) -> None:
    client, _, _, persistence = setup
    assert _register(client, _registration()).status_code == 200
    insert_count = len(persistence.inserts)
    response = _register(client, _registration(**{field: value}))
    assert response.status_code == 409
    assert len(persistence.inserts) == insert_count


def test_foreign_owner_is_non_enumerating_not_found(setup) -> None:
    client, _, _, _ = setup
    assert _register(client, _registration()).status_code == 200
    response = _register(client, _registration(org_id="org-2"))
    assert response.status_code == 404


def test_activation_cas_routes_alias_without_disabling_old_revisions(setup) -> None:
    client, pool, router, persistence = setup
    assert _register(client, _registration()).status_code == 200
    assert _register(client, _registration(step=40, sha=SHA_B)).status_code == 200

    first = client.post(
        f"/adapters/{REVISION_A}/activate",
        json={"expected_adapter_revision": None},
    )
    assert first.status_code == 200
    first_body = first.json()
    assert first_body == {
        "adapter_id": RUN_ID,
        "target_adapter_revision": REVISION_A,
        "previous_adapter_revision": None,
        "checkpoint": f"{RUN_ID}/step-20",
        "updated_at": first_body["updated_at"],
    }
    assert isinstance(first_body["updated_at"], str)
    assert first_body["updated_at"]

    stale = client.post(
        f"/adapters/{REVISION_B}/activate",
        json={"expected_adapter_revision": None},
    )
    assert stale.status_code == 409

    promoted = client.post(
        f"/adapters/{REVISION_B}/activate",
        json={"expected_adapter_revision": REVISION_A},
    )
    assert promoted.status_code == 200
    promoted_body = promoted.json()
    assert promoted_body == {
        "adapter_id": RUN_ID,
        "target_adapter_revision": REVISION_B,
        "previous_adapter_revision": REVISION_A,
        "checkpoint": f"{RUN_ID}/step-40",
        "updated_at": promoted_body["updated_at"],
    }
    assert isinstance(promoted_body["updated_at"], str)
    assert promoted_body["updated_at"]

    for adapter_id in (RUN_ID, REVISION_A, REVISION_B):
        response = client.post("/generate", json={"adapter_id": adapter_id, "prompt": "hi"})
        assert response.status_code == 200
        assert response.json()["adapter_id"] == adapter_id
        expected_checkpoint = (
            f"{RUN_ID}/step-40" if adapter_id in {RUN_ID, REVISION_B} else f"{RUN_ID}/step-20"
        )
        assert response.headers["X-Freesolo-Checkpoint"] == expected_checkpoint

    assert pool.generated == [
        (QWEN, REVISION_B, REVISION_B),
        (QWEN, REVISION_A, REVISION_A),
        (QWEN, REVISION_B, REVISION_B),
    ]
    assert persistence.rows[REVISION_A].status == "ready"
    assert persistence.rows[REVISION_B].status == "ready"
    assert pool.unregistered == []
    assert router.resolve(RUN_ID) == (
        persistence.rows[RUN_ID],
        persistence.rows[REVISION_B],
    )


def test_activation_updated_at_cas_miss_returns_conflict(setup) -> None:
    client, _, _, persistence = setup
    assert _register(client, _registration()).status_code == 200
    persistence.force_replace_miss = True
    response = client.post(
        f"/adapters/{REVISION_A}/activate",
        json={"expected_adapter_revision": None},
    )
    assert response.status_code == 409
    assert persistence.rows[RUN_ID].status == "disabled"


def test_delete_run_alias_cascades_ready_alias_and_revisions(setup) -> None:
    client, pool, _, persistence = setup
    assert _register(client, _final_registration()).status_code == 200
    assert (
        client.post(
            f"/adapters/{FINAL_REVISION}/activate",
            json={"expected_adapter_revision": None},
        ).status_code
        == 200
    )

    revision_generation = persistence.rows[FINAL_REVISION].deployment_generation
    assert revision_generation is not None
    reloaded_router = AdapterRouter([persistence.rows[RUN_ID]])
    reloaded_client = TestClient(
        build_serving_app(
            pool,
            reloaded_router,
            internal_key="secret",
            reload_records=lambda: [
                record for record in persistence.rows.values() if record.status == "ready"
            ],
        ),
        headers=INTERNAL_HEADERS,
    )
    response = reloaded_client.delete(f"/adapters/{RUN_ID}")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "removed": RUN_ID,
        "base_model": QWEN,
        "run_id": RUN_ID,
        "disabled_aliases": [RUN_ID],
        "disabled_revisions": [FINAL_REVISION],
    }
    assert all(isinstance(value, str) for value in response.json()["disabled_aliases"])
    assert all(isinstance(value, str) for value in response.json()["disabled_revisions"])
    assert persistence.rows[RUN_ID].status == "disabled"
    assert persistence.rows[FINAL_REVISION].status == "disabled"
    assert pool.unregistered == [RUN_ID, FINAL_REVISION]
    assert pool.unregistered_generations == [None, revision_generation]
    assert (
        reloaded_client.post("/generate", json={"adapter_id": RUN_ID, "prompt": "hi"}).status_code
        == 404
    )
    assert (
        reloaded_client.post(
            "/generate", json={"adapter_id": FINAL_REVISION, "prompt": "hi"}
        ).status_code
        == 404
    )


def test_failed_retry_fences_prior_generation_before_undeploy(setup) -> None:
    client, pool, _, persistence = setup
    assert _register(client, _registration()).status_code == 200
    previous = persistence.rows[REVISION_A]
    disabled = persistence._stamp(previous.model_copy(update={"status": "disabled"}))
    persistence.rows[REVISION_A] = disabled
    attempts: list[AdapterRecord] = []

    async def _fail_register(_base_model: str, record: AdapterRecord) -> None:
        attempts.append(record)
        raise RuntimeError("still loading")

    pool.register = _fail_register
    assert _register(client, _registration()).status_code == 200
    attempt = attempts[0]
    fenced = persistence.rows[REVISION_A]
    assert attempt.deployment_generation == disabled.updated_at
    assert fenced.status == "disabled"
    assert fenced.deployment_generation is None
    assert fenced.updated_at != attempt.updated_at
    stale_promotion = persistence.replace(
        attempt.model_copy(update={"status": "ready"}),
        expected_updated_at=attempt.updated_at,
        settings=object(),
    )
    assert stale_promotion is None, "the retried load retained promotion authority"
    assert pool.unregistered_generations[-1] == attempt.deployment_generation


def test_delete_run_alias_evicts_revision_whose_register_result_was_lost(setup) -> None:
    client, pool, _, persistence = setup
    assert _register(client, _registration()).status_code == 200
    assert (
        client.post(
            f"/adapters/{REVISION_A}/activate",
            json={"expected_adapter_revision": None},
        ).status_code
        == 200
    )
    loading = ImmutableAdapterRegistration.model_validate(
        _registration(step=40, sha=SHA_B)
    ).to_record()
    loading = persistence._stamp(loading)
    loading = loading.model_copy(update={"deployment_generation": loading.updated_at})
    persistence.rows[REVISION_B] = loading
    # the remote engine loaded this generation, but its successful result was lost before promotion.
    pool.loaded_generations[REVISION_B] = loading.updated_at

    response = client.delete(f"/adapters/{RUN_ID}")

    assert response.status_code == 200
    assert response.json()["disabled_revisions"] == [REVISION_A, REVISION_B]
    assert persistence.rows[REVISION_B].updated_at != loading.updated_at
    stale_promotion = persistence.replace(
        loading.model_copy(update={"status": "ready"}),
        expected_updated_at=loading.updated_at,
        settings=object(),
    )
    assert stale_promotion is None, "the loading revision retained authority to become ready"
    assert pool.unregistered == [RUN_ID, REVISION_A, REVISION_B]
    assert pool.unregistered_generations[-1] == loading.updated_at
    assert REVISION_B not in pool.loaded_generations, "fenced loading revision stayed gpu-resident"


def test_delete_run_alias_disables_revisions_when_alias_already_disabled(setup) -> None:
    client, pool, router, persistence = setup
    assert _register(client, _final_registration()).status_code == 200
    assert (
        client.post(
            f"/adapters/{FINAL_REVISION}/activate",
            json={"expected_adapter_revision": None},
        ).status_code
        == 200
    )
    disabled_alias = persistence._stamp(
        persistence.rows[RUN_ID].model_copy(update={"status": "disabled"})
    )
    persistence.rows[RUN_ID] = disabled_alias
    router.upsert(disabled_alias)

    response = client.delete(f"/adapters/{RUN_ID}")

    assert response.status_code == 200
    assert response.json()["run_id"] == RUN_ID
    assert response.json()["disabled_aliases"] == []
    assert response.json()["disabled_revisions"] == [FINAL_REVISION]
    assert persistence.rows[RUN_ID].status == "disabled"
    assert persistence.rows[FINAL_REVISION].status == "disabled"
    assert pool.unregistered == [FINAL_REVISION]


def test_delete_revision_cascades_its_run(setup) -> None:
    client, pool, _, persistence = setup
    assert _register(client, _registration()).status_code == 200
    assert (
        client.post(
            f"/adapters/{REVISION_A}/activate",
            json={"expected_adapter_revision": None},
        ).status_code
        == 200
    )

    response = client.delete(f"/adapters/{REVISION_A}")

    assert response.status_code == 200
    assert response.json()["run_id"] == RUN_ID
    assert response.json()["disabled_aliases"] == [RUN_ID]
    assert response.json()["disabled_revisions"] == [REVISION_A]
    assert persistence.rows[REVISION_A].status == "disabled"
    assert persistence.rows[RUN_ID].status == "disabled"
    assert pool.unregistered == [RUN_ID, REVISION_A]
    delete_run_response = client.delete(f"/adapters/{RUN_ID}")
    assert delete_run_response.status_code == 404


def test_delete_unknown_revision_id_does_not_disable_live_run(setup) -> None:
    client, pool, router, persistence = setup
    assert _register(client, _registration()).status_code == 200
    assert (
        client.post(
            f"/adapters/{REVISION_A}/activate",
            json={"expected_adapter_revision": None},
        ).status_code
        == 200
    )
    unknown_revision = f"{RUN_ID}@final.{SHA_B}"

    response = client.delete(f"/adapters/{unknown_revision}")

    assert response.status_code == 404
    assert persistence.rows[RUN_ID].status == "ready"
    assert persistence.rows[REVISION_A].status == "ready"
    assert router.resolve(RUN_ID) == (
        persistence.rows[RUN_ID],
        persistence.rows[REVISION_A],
    )
    assert pool.unregistered == []


def test_delete_refetches_missing_updated_at_before_cas(setup) -> None:
    client, pool, router, persistence = setup
    assert _register(client, _final_registration()).status_code == 200
    assert (
        client.post(
            f"/adapters/{FINAL_REVISION}/activate",
            json={"expected_adapter_revision": None},
        ).status_code
        == 200
    )
    router.upsert(persistence.rows[RUN_ID].model_copy(update={"updated_at": None}))

    response = client.delete(f"/adapters/{RUN_ID}")

    assert response.status_code == 200
    assert response.json()["disabled_aliases"] == [RUN_ID]
    assert response.json()["disabled_revisions"] == [FINAL_REVISION]
    assert persistence.rows[RUN_ID].status == "disabled"
    assert persistence.rows[FINAL_REVISION].status == "disabled"
    assert pool.unregistered == [RUN_ID, FINAL_REVISION]


def test_delete_cas_miss_is_success_when_concurrent_writer_disabled_row(setup) -> None:
    client, pool, _, persistence = setup
    assert _register(client, _final_registration()).status_code == 200
    assert (
        client.post(
            f"/adapters/{FINAL_REVISION}/activate",
            json={"expected_adapter_revision": None},
        ).status_code
        == 200
    )
    persistence.concurrent_disable_once.add(RUN_ID)

    response = client.delete(f"/adapters/{RUN_ID}")

    assert response.status_code == 200
    assert response.json()["disabled_aliases"] == [RUN_ID]
    assert response.json()["disabled_revisions"] == [FINAL_REVISION]
    assert persistence.rows[RUN_ID].status == "disabled"
    assert persistence.rows[FINAL_REVISION].status == "disabled"
    assert pool.unregistered == [RUN_ID, FINAL_REVISION]


def test_delete_cleans_up_generation_reloaded_after_cas_miss(setup) -> None:
    client, pool, _, persistence = setup
    assert _register(client, _registration()).status_code == 200
    redeployed_generation = "generation-redeployed"
    persistence.concurrent_redeploy_once[REVISION_A] = redeployed_generation

    response = client.delete(f"/adapters/{REVISION_A}")

    assert response.status_code == 200
    assert persistence.rows[REVISION_A].status == "disabled"
    assert persistence.rows[REVISION_A].deployment_generation is None
    assert pool.unregistered_generations[-1] == redeployed_generation


def test_delete_disabled_revision_id_cascades_remaining_ready_siblings(setup) -> None:
    client, pool, router, persistence = setup
    assert _register(client, _registration()).status_code == 200
    assert _register(client, _registration(step=40, sha=SHA_B)).status_code == 200
    assert (
        client.post(
            f"/adapters/{REVISION_B}/activate",
            json={"expected_adapter_revision": None},
        ).status_code
        == 200
    )
    persistence.rows[REVISION_A] = persistence._stamp(
        persistence.rows[REVISION_A].model_copy(
            update={"status": "disabled", "deployment_generation": None}
        )
    )
    router.remove(REVISION_A)

    response = client.delete(f"/adapters/{REVISION_A}")

    assert response.status_code == 200
    assert response.json()["run_id"] == RUN_ID
    assert response.json()["disabled_aliases"] == [RUN_ID]
    assert response.json()["disabled_revisions"] == [REVISION_B]
    assert all(record.status == "disabled" for record in persistence.rows.values())
    assert pool.unregistered == [RUN_ID, REVISION_B]
    unknown = f"unknown-run@final.{SHA_A}"
    unknown_response = client.delete(f"/adapters/{unknown}")
    assert unknown_response.status_code == 404


def test_delete_repeated_cas_miss_returns_conflict(setup) -> None:
    client, pool, _, persistence = setup
    assert _register(client, _final_registration()).status_code == 200
    assert (
        client.post(
            f"/adapters/{FINAL_REVISION}/activate",
            json={"expected_adapter_revision": None},
        ).status_code
        == 200
    )
    persistence.force_replace_miss = True

    response = client.delete(f"/adapters/{RUN_ID}")

    assert response.status_code == 409
    assert persistence.rows[RUN_ID].status == "ready"
    assert persistence.rows[FINAL_REVISION].status == "ready"
    assert pool.unregistered == []


@pytest.mark.parametrize(
    "failure_path",
    [
        "missing_timestamp",
        "missing_timestamp_transport",
        "cas_loser",
        "cas_loser_transport",
        "cas_write",
    ],
)
def test_delete_storage_failure_tears_down_rows_that_already_converged(
    setup, failure_path: str
) -> None:
    client, pool, router, persistence = setup
    assert _register(client, _registration()).status_code == 200
    assert (
        client.post(
            f"/adapters/{REVISION_A}/activate",
            json={"expected_adapter_revision": None},
        ).status_code
        == 200
    )
    if failure_path.startswith("missing_timestamp"):
        # a cached row with no timestamp cannot be compare-and-swapped, so the cascade re-reads it
        # first. that read is one of the two that can fail.
        if failure_path.endswith("_transport"):
            persistence.get_transport_failure_ids.add(REVISION_A)
        else:
            persistence.get_failure_ids.add(REVISION_A)
        router.upsert(persistence.rows[REVISION_A].model_copy(update={"updated_at": None}))
    elif failure_path.startswith("cas_loser"):
        # the row loses its compare-and-swap and the retry's re-read is what hits the outage.
        if failure_path.endswith("_transport"):
            persistence.get_transport_failure_ids.add(REVISION_A)
        else:
            persistence.get_failure_ids.add(REVISION_A)
        persistence.force_replace_miss_ids.add(REVISION_A)
    else:
        # the outage hits the compare-and-swap write itself rather than a read around it. that
        # raises the same 503 from a call that sat outside either recovery block, so the whole
        # cascade unwound and the run alias -- already disabled and unregistered above -- was
        # dropped from the teardown list while its row stayed disabled: a live gpu with no record
        # left to reconcile it.
        persistence.replace_failure_ids.add(REVISION_A)

    response = client.delete(f"/adapters/{RUN_ID}")

    assert response.status_code == 503
    assert response.json() == {"detail": "adapter storage is unavailable"}
    assert persistence.rows[RUN_ID].status == "disabled"
    assert persistence.rows[REVISION_A].status == "ready"
    assert pool.unregistered == [RUN_ID]
    assert router.resolve(RUN_ID) is None


def test_delete_pre_cas_storage_failure_returns_503_without_teardown(setup) -> None:
    _, pool, router, persistence = setup
    persistence.get_failure_ids.add(RUN_ID)
    app = build_serving_app(pool, router, internal_key="secret")
    request = Request(
        {
            "type": "http",
            "method": "DELETE",
            "path": f"/adapters/{RUN_ID}",
            "headers": [(b"x-freesolo-internal-key", b"secret")],
            "query_string": b"",
            "app": app,
        }
    )
    background_tasks = BackgroundTasks()

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            remove_adapter(
                adapter_id=RUN_ID,
                request=request,
                background_tasks=background_tasks,
            )
        )

    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == "adapter storage is unavailable"
    assert background_tasks.tasks == []
    assert pool.unregistered == []


def test_delete_partial_cascade_conflict_tears_down_converged_rows(setup) -> None:
    client, pool, router, persistence = setup
    assert _register(client, _registration()).status_code == 200
    assert (
        client.post(
            f"/adapters/{REVISION_A}/activate",
            json={"expected_adapter_revision": None},
        ).status_code
        == 200
    )
    # the alias disables cleanly but the revision's cas never converges, so the run is only
    # half-disabled in persistence.
    persistence.force_replace_miss_ids = {REVISION_A}

    response = client.delete(f"/adapters/{RUN_ID}")

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["run_id"] == RUN_ID
    assert detail["disabled_aliases"] == [RUN_ID]
    assert detail["disabled_revisions"] == []
    assert detail["stuck"] == [REVISION_A]
    # the converged alias is torn down immediately even though a sibling is stuck: _reload() only
    # rehydrates ready rows, so deferring its teardown would orphan the gpu registration forever
    # (a retry could never re-enumerate the already-disabled alias). the stuck revision keeps
    # serving until an idempotent retry disables and tears it down too.
    assert pool.unregistered == [RUN_ID]
    assert persistence.rows[REVISION_A].status == "ready"
    # the torn-down alias no longer resolves for inference (in-memory registry matches persistence),
    # so this container stops serving it immediately instead of waiting for a retry or reload.
    assert router.resolve(RUN_ID) is None

    # the transient conflict clears; an idempotent retry disables and tears down the remaining stuck
    # revision without re-tearing-down the already-converged alias, and returns success.
    persistence.force_replace_miss_ids = set()
    retry = client.delete(f"/adapters/{RUN_ID}")
    assert retry.status_code == 200
    assert persistence.rows[REVISION_A].status == "disabled"
    assert pool.unregistered == [RUN_ID, REVISION_A]


def test_malformed_and_cross_owner_alias_targets_fail_closed() -> None:
    target = PersistedAdapterRecord.model_validate(
        {
            **_registration(),
            "status": "ready",
            "created_at": "2026-07-14T00:00:00+00:00",
            "updated_at": "2026-07-14T00:00:00+00:00",
        }
    )
    alias = PersistedAdapterRecord.model_validate(
        {
            **internal_adapter_payload(target),
            "adapter_id": RUN_ID,
            "checkpoint": None,
            "status": "ready",
            "metadata": {
                "record_type": "alias",
                "run_id": RUN_ID,
                "alias_of": REVISION_A,
            },
        }
    )
    cross_owner = target.model_copy(update={"org_id": "org-2"})
    assert AdapterRouter([alias, cross_owner]).resolve(RUN_ID) is None

    malformed = alias.model_copy(update={"metadata": {**alias.metadata, "unexpected": "legacy"}})
    assert AdapterRouter([malformed, target]).resolve(RUN_ID) is None


def test_health_reports_exact_capabilities(setup) -> None:
    client, _, _, _ = setup
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["capabilities"] == [
        "immutable_adapter_revisions",
        "alias_compare_and_swap",
        "revision_provenance",
        "thinking_structured_outputs_deferred_v1",
    ]


def test_chat_completion_against_revision_emits_provenance(setup) -> None:
    # this is the exact deploy-smoke path: flash chats the immutable revision id and verifies the
    # response carries revision provenance before flipping the run alias.
    client, _, _, _ = setup
    assert _register(client, _registration()).status_code == 200
    checkpoint = f"{RUN_ID}/step-20"
    response = client.post(
        "/v1/chat/completions",
        json={"model": REVISION_A, "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Freesolo-Expected-Checkpoint": checkpoint},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"].strip()
    assert body["freesolo"] == {
        "adapter_revision": REVISION_A,
        "checkpoint": checkpoint,
        "hf_revision": SHA_A,
    }
    assert response.headers["X-Freesolo-Adapter-Revision"] == REVISION_A
    assert response.headers["X-Freesolo-Checkpoint"] == checkpoint
    assert response.headers["X-Freesolo-HF-Revision"] == SHA_A


def test_chat_completion_via_run_alias_reports_active_revision(setup) -> None:
    # a client addressing the mutable run alias gets the provenance of whichever revision is active.
    client, _, _, _ = setup
    assert _register(client, _registration()).status_code == 200
    assert (
        client.post(
            f"/adapters/{REVISION_A}/activate", json={"expected_adapter_revision": None}
        ).status_code
        == 200
    )
    checkpoint = f"{RUN_ID}/step-20"
    response = client.post(
        "/v1/chat/completions",
        json={"model": RUN_ID, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == RUN_ID
    assert body["freesolo"] == {
        "adapter_revision": REVISION_A,
        "checkpoint": checkpoint,
        "hf_revision": SHA_A,
    }
    assert response.headers["X-Freesolo-Adapter-Revision"] == REVISION_A
    assert response.headers["X-Freesolo-Checkpoint"] == checkpoint
    assert response.headers["X-Freesolo-HF-Revision"] == SHA_A


def test_chat_completion_stream_emits_provenance_headers(setup) -> None:
    client, pool, _, _ = setup
    assert _register(client, _registration()).status_code == 200
    checkpoint = f"{RUN_ID}/step-20"

    async def _stream(base_model, payload, record, *, expected_checkpoint=None):
        del base_model, payload, expected_checkpoint
        yield {"type": "ready", "checkpoint": record.checkpoint}
        yield {"type": "final", "prompt_tokens": 1, "completion_tokens": 1}

    pool.stream_generate = _stream
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": REVISION_A,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["X-Freesolo-Adapter-Revision"] == REVISION_A
        assert response.headers["X-Freesolo-Checkpoint"] == checkpoint
        assert response.headers["X-Freesolo-HF-Revision"] == SHA_A
        response.read()


def test_generate_routes_emit_revision_provenance(setup) -> None:
    # the /generate and per-adapter /adapters/{id}/generate routes carry the same revision
    # provenance (body block + headers) as chat, for both direct-revision and per-adapter addressing.
    client, _, _, _ = setup
    assert _register(client, _registration()).status_code == 200
    checkpoint = f"{RUN_ID}/step-20"
    expected = {"adapter_revision": REVISION_A, "checkpoint": checkpoint, "hf_revision": SHA_A}

    direct = client.post("/generate", json={"adapter_id": REVISION_A, "prompt": "hi"})
    assert direct.status_code == 200
    assert direct.json()["freesolo"] == expected
    assert direct.headers["X-Freesolo-Adapter-Revision"] == REVISION_A
    assert direct.headers["X-Freesolo-Checkpoint"] == checkpoint
    assert direct.headers["X-Freesolo-HF-Revision"] == SHA_A

    encoded_revision = quote(REVISION_A, safe="")
    per_adapter = client.post(f"/adapters/{encoded_revision}/generate", json={"prompt": "hi"})
    assert per_adapter.status_code == 200
    assert per_adapter.json()["freesolo"] == expected
    assert per_adapter.headers["X-Freesolo-Adapter-Revision"] == REVISION_A
    assert per_adapter.headers["X-Freesolo-Checkpoint"] == checkpoint
    assert per_adapter.headers["X-Freesolo-HF-Revision"] == SHA_A


def test_generate_base_model_response_carries_no_revision_provenance(setup) -> None:
    # base-model serving is not a revision: no freesolo body block and none of the revision headers,
    # so _inference_json_response falls back to the unchanged checkpoint-only behaviour.
    client, _, router, _ = setup
    base = AdapterRecord.model_validate(
        {
            "adapter_id": QWEN_2B,
            "repo_id": QWEN_2B,
            "base_model": QWEN_2B,
            "serve_base_model": True,
            "thinking": True,
            "org_id": None,
            "status": "ready",
        }
    )
    router.upsert(base)
    response = client.post("/generate", json={"adapter_id": QWEN_2B, "prompt": "hi"})
    assert response.status_code == 200
    assert "freesolo" not in response.json()
    assert "X-Freesolo-Adapter-Revision" not in response.headers
    assert "X-Freesolo-HF-Revision" not in response.headers
    assert "X-Freesolo-Checkpoint" not in response.headers


def test_chat_completion_stream_replay_path_reports_revision_checkpoint(setup) -> None:
    # the real engine always leads a stream with a ready event, but the router also supports stream
    # sources that do not. on that fallback path a revision reports its own immutable checkpoint
    # (record.checkpoint = the exact value the engine enforces), so provenance stays correct.
    client, pool, _, _ = setup
    assert _register(client, _registration()).status_code == 200
    checkpoint = f"{RUN_ID}/step-20"

    async def _stream_without_ready(base_model, payload, record, *, expected_checkpoint=None):
        del base_model, payload, record, expected_checkpoint
        yield {"type": "final", "prompt_tokens": 1, "completion_tokens": 1}

    pool.stream_generate = _stream_without_ready
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": REVISION_A,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["X-Freesolo-Adapter-Revision"] == REVISION_A
        assert response.headers["X-Freesolo-Checkpoint"] == checkpoint
        assert response.headers["X-Freesolo-HF-Revision"] == SHA_A
        response.read()
