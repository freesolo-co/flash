from __future__ import annotations

from typing import Any, ClassVar

import httpx
import pytest

from flash.serve.contract.provenance import immutable_binding_fingerprint
from flash.serving.src.io.schemas import AdapterRecord, PersistedAdapterRecord
from flash.serving.src.store.persistence import (
    PERSISTED_COLUMNS,
    PersistenceConflict,
    PersistenceRecordError,
    PersistenceReferenceError,
    _adapter_to_row,
    _row_to_adapter,
    get_adapter,
    insert_adapter,
    list_run_adapters,
    load_adapters,
    replace_adapter_cas,
)
from flash.serving.src.store.settings import Settings

SHA = "a" * 40
DIGEST = "b" * 64
RUN_ID = "flash-1234567890-abcdef12"
ORG_ID = "org-1"
CHECKPOINT_ID = f"{RUN_ID}/step-20"


def _settings() -> Settings:
    return Settings(
        SUPABASE_URL="https://supabase.test",
        SUPABASE_SERVICE_ROLE_KEY="sb_secret_service_role",
    )


def _record_values(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "adapter_id": CHECKPOINT_ID,
        "repo_id": "org/run",
        "org_id": ORG_ID,
        "url": "https://huggingface.co/org/run",
        "base_model": "Qwen/Qwen3.5-9B",
        "subfolder": "checkpoints/step-20/adapter",
        "repo_type": "dataset",
        "checkpoint": CHECKPOINT_ID,
        "private": True,
        "thinking": False,
        "status": "disabled",
        "run_id": RUN_ID,
        "checkpoint_step": 20,
        "artifact_revision": SHA,
        "artifact_digest": DIGEST,
        "lora_rank": 32,
        "created_at": "2026-07-14T00:00:00+00:00",
        "updated_at": "2026-07-14T00:00:01+00:00",
    }
    values.update(overrides)
    values["artifact_fingerprint"] = immutable_binding_fingerprint(values)
    return values


def _record(**overrides: object) -> AdapterRecord:
    return AdapterRecord.model_validate(_record_values(**overrides))


def _row(**overrides: object) -> dict[str, object]:
    record = _record()
    row = _adapter_to_row(record, "2026-07-14T00:00:01+00:00")
    row.update(overrides)
    return row


class FakeClient:
    requests: ClassVar[list[dict[str, Any]]] = []
    get_rows: ClassVar[list[dict[str, object]]] = []
    post_status = 201
    post_rows: ClassVar[list[dict[str, object]]] = []
    post_body: dict[str, object] | None = None
    patch_rows: ClassVar[list[dict[str, object]]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    @classmethod
    def reset(cls) -> None:
        cls.requests = []
        cls.get_rows = []
        cls.post_status = 201
        cls.post_rows = []
        cls.post_body = None
        cls.patch_rows = []

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        self.requests.append({"method": "GET", "url": url, **kwargs})
        return httpx.Response(200, json=self.get_rows, request=httpx.Request("GET", url))

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        self.requests.append({"method": "POST", "url": url, **kwargs})
        request = httpx.Request("POST", url)
        if self.post_status >= 400:
            if self.post_body is None:
                return httpx.Response(self.post_status, text="upstream failure", request=request)
            return httpx.Response(self.post_status, json=self.post_body, request=request)
        return httpx.Response(self.post_status, json=self.post_rows, request=request)

    def patch(self, url: str, **kwargs: Any) -> httpx.Response:
        self.requests.append({"method": "PATCH", "url": url, **kwargs})
        return httpx.Response(200, json=self.patch_rows, request=httpx.Request("PATCH", url))


@pytest.fixture(autouse=True)
def fake_http(monkeypatch) -> None:
    FakeClient.reset()
    monkeypatch.setattr("flash.serving.src.store.persistence.httpx.Client", FakeClient)


def test_load_uses_explicit_projected_columns_and_ready_filter() -> None:
    FakeClient.get_rows = [_row(status="ready")]

    [record] = load_adapters(_settings())

    request = FakeClient.requests[0]
    assert request["params"]["select"] == PERSISTED_COLUMNS
    assert request["params"]["status"] == "eq.ready"
    assert record.adapter_id == CHECKPOINT_ID
    assert record.org_id == ORG_ID


def test_run_listing_reads_disabled_checkpoints_with_tenant_scope() -> None:
    FakeClient.get_rows = [_row(status="disabled")]

    [record] = list_run_adapters(ORG_ID, RUN_ID, _settings())

    params = FakeClient.requests[0]["params"]
    assert params["org_id"] == f"eq.{ORG_ID}"
    assert params["run_id"] == f"eq.{RUN_ID}"
    assert "status" not in params
    assert record.status == "disabled"


def test_targeted_get_reads_exact_tenant_checkpoint() -> None:
    FakeClient.get_rows = [_row(status="disabled")]

    record = get_adapter(ORG_ID, CHECKPOINT_ID, _settings())

    assert record is not None
    params = FakeClient.requests[0]["params"]
    assert params["org_id"] == f"eq.{ORG_ID}"
    assert params["checkpoint_id"] == f"eq.{CHECKPOINT_ID}"
    assert "status" not in params


def test_insert_binds_immutable_checkpoint_and_returns_authoritative_row() -> None:
    FakeClient.post_rows = [_row()]

    stored = insert_adapter(_record(), _settings())

    request = FakeClient.requests[0]
    assert request["url"].endswith("/rest/v1/rpc/bind_hosted_checkpoint")
    assert request["json"] == {
        "p_org_id": ORG_ID,
        "p_run_id": RUN_ID,
        "p_checkpoint_id": CHECKPOINT_ID,
        "p_source_repo_type": "dataset",
        "p_source_repository": "org/run",
        "p_source_revision": SHA,
        "p_source_subfolder": "checkpoints/step-20/adapter",
        "p_artifact_digest": DIGEST,
        "p_artifact_fingerprint": _record().artifact_fingerprint,
        "p_base_model": "Qwen/Qwen3.5-9B",
        "p_lora_config": {"rank": 32},
        "p_serving_defaults": {"thinking": False, "structured_outputs": None},
    }
    assert stored.updated_at == "2026-07-14T00:00:01+00:00"


def test_insert_conflict_is_explicit() -> None:
    FakeClient.post_status = 409
    FakeClient.post_body = {"code": "23505", "message": "duplicate key value"}

    with pytest.raises(PersistenceConflict):
        insert_adapter(_record(), _settings())


def test_foreign_key_violation_is_not_reported_as_a_duplicate_row() -> None:
    FakeClient.post_status = 409
    FakeClient.post_body = {
        "code": "23503",
        "details": f"Key (org_id)=({ORG_ID}) is not present in table orgs.",
        "message": "insert or update violates foreign key constraint",
    }

    with pytest.raises(PersistenceReferenceError, match=ORG_ID):
        insert_adapter(_record(), _settings())


@pytest.mark.parametrize(
    "body",
    [None, {"code": "23514", "message": "check constraint violation"}],
)
def test_unclassified_409_does_not_enter_duplicate_reconciliation(body: object) -> None:
    FakeClient.post_status = 409
    FakeClient.post_body = body

    with pytest.raises(RuntimeError, match="409"):
        insert_adapter(_record(), _settings())


def test_replace_adapter_uses_tenant_checkpoint_cas() -> None:
    replacement = _record(status="ready")
    FakeClient.patch_rows = [_row(status="ready", updated_at="2026-07-14T00:00:02+00:00")]

    committed = replace_adapter_cas(
        replacement,
        expected_updated_at="2026-07-14T00:00:01+00:00",
        settings=_settings(),
    )

    request = FakeClient.requests[0]
    assert request["params"] == {
        "select": PERSISTED_COLUMNS,
        "org_id": f"eq.{ORG_ID}",
        "checkpoint_id": f"eq.{CHECKPOINT_ID}",
        "updated_at": "eq.2026-07-14T00:00:01+00:00",
    }
    assert committed is not None
    assert committed.status == "ready"


def test_cas_miss_returns_none() -> None:
    FakeClient.patch_rows = []

    assert (
        replace_adapter_cas(
            _record(),
            expected_updated_at="stale",
            settings=_settings(),
        )
        is None
    )


def test_hydration_skips_legacy_rows_and_targeted_reads_reject_them() -> None:
    legacy = {"adapter_id": "legacy-alias", "metadata": {"run_id": RUN_ID}}
    FakeClient.get_rows = [legacy]
    assert load_adapters(_settings()) == []

    with pytest.raises(PersistenceRecordError, match="invalid checkpoint row"):
        get_adapter(ORG_ID, CHECKPOINT_ID, _settings())


def test_row_round_trip_preserves_exact_source_identity() -> None:
    record = _record().model_copy(update={"deployment_generation": "generation-1"})
    row = _adapter_to_row(record, "2026-07-14T00:00:03+00:00")
    hydrated = PersistedAdapterRecord.model_validate(_row_to_adapter(row))

    assert hydrated.repo_id == record.repo_id
    assert hydrated.repo_type == record.repo_type
    assert hydrated.artifact_revision == SHA
    assert hydrated.artifact_digest == DIGEST
    assert hydrated.artifact_fingerprint == record.artifact_fingerprint
    assert hydrated.subfolder == record.subfolder
    assert hydrated.url == record.url
    assert hydrated.deployment_generation == "generation-1"
    assert row["checkpoint_id"] == CHECKPOINT_ID
    assert row["checkpoint"] == "step-20"
