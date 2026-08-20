from __future__ import annotations

from typing import Any

import httpx
import pytest

from flash.serving.src.persistence import (
    _PERSISTED_COLUMNS,
    PersistenceConflict,
    PersistenceRecordError,
    PersistenceReferenceError,
    _adapter_to_row,
    _row_to_adapter,
    get_adapter,
    insert_adapter,
    load_adapters,
    replace_adapter_cas,
    update_adapter_status_cas,
)
from flash.serving.src.schemas import AdapterRecord, PersistedAdapterRecord
from flash.serving.src.settings import Settings

SHA = "a" * 40
RUN_ID = "flash-1234567890-abcdef12"
REVISION_ID = f"{RUN_ID}@step-20.{SHA}"


def _settings() -> Settings:
    return Settings(
        SUPABASE_URL="https://supabase.test",
        SUPABASE_SERVICE_ROLE_KEY="service-role",
    )


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "adapter_id": REVISION_ID,
        "repo_id": "org/run",
        "org_id": "org-1",
        "url": "https://huggingface.co/org/run",
        "base_model": "Qwen/Qwen3.5-0.8B",
        "subfolder": "checkpoints/step-20",
        "repo_type": "model",
        "checkpoint": f"{RUN_ID}/step-20",
        "private": True,
        "status": "disabled",
        "metadata": {
            "record_type": "revision",
            "run_id": RUN_ID,
            "checkpoint_step": 20,
            "hf_revision": SHA,
            "thinking": False,
        },
        "created_at": "2026-07-14T00:00:00+00:00",
        "updated_at": "2026-07-14T00:00:01+00:00",
    }
    row.update(overrides)
    return row


def _record(**overrides: object) -> AdapterRecord:
    return PersistedAdapterRecord.model_validate(_row_to_adapter(_row(**overrides)))


class FakeClient:
    requests: list[dict[str, Any]] = []
    get_rows: list[dict[str, object]] = []
    post_status = 201
    post_rows: list[dict[str, object]] = []
    # error payload for a failing POST. `None` sends a non-JSON body, which is how a proxy or
    # gateway error arrives -- the code must not read a sqlstate out of it.
    post_body: dict[str, object] | None = None
    patch_rows: list[dict[str, object]] = []

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
        return httpx.Response(
            200,
            json=self.get_rows,
            request=httpx.Request("GET", url),
        )

    @staticmethod
    def _mutation_rows(
        rows: list[dict[str, object]], kwargs: dict[str, Any]
    ) -> list[dict[str, object]]:
        if kwargs.get("params", {}).get("select"):
            return rows
        return [{**row, "id": index + 1} for index, row in enumerate(rows)]

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        self.requests.append({"method": "POST", "url": url, **kwargs})
        request = httpx.Request("POST", url)
        if self.post_status >= 400:
            if self.post_body is None:
                return httpx.Response(self.post_status, text="upstream failure", request=request)
            return httpx.Response(self.post_status, json=self.post_body, request=request)
        return httpx.Response(
            self.post_status,
            json=self._mutation_rows(self.post_rows, kwargs),
            request=request,
        )

    def patch(self, url: str, **kwargs: Any) -> httpx.Response:
        self.requests.append({"method": "PATCH", "url": url, **kwargs})
        return httpx.Response(
            200,
            json=self._mutation_rows(self.patch_rows, kwargs),
            request=httpx.Request("PATCH", url),
        )


@pytest.fixture(autouse=True)
def fake_http(monkeypatch) -> None:
    FakeClient.reset()
    monkeypatch.setattr("flash.serving.src.persistence.httpx.Client", FakeClient)


def test_load_uses_explicit_projected_columns_and_ready_filter() -> None:
    FakeClient.get_rows = [_row(status="ready")]
    [record] = load_adapters(_settings())
    request = FakeClient.requests[0]
    assert request["params"]["select"] != "*"
    assert "adapter_id" in request["params"]["select"]
    assert request["params"]["status"] == "eq.ready"
    assert record.adapter_id == REVISION_ID


def test_targeted_get_reads_disabled_revision_without_status_filter() -> None:
    FakeClient.get_rows = [_row(status="disabled")]
    record = get_adapter(REVISION_ID, _settings())
    assert record is not None and record.status == "disabled"
    assert "status" not in FakeClient.requests[0]["params"]


def test_insert_is_non_upserting_and_returns_authoritative_row() -> None:
    FakeClient.post_rows = [_row()]
    stored = insert_adapter(_record(), _settings())
    request = FakeClient.requests[0]
    assert request["params"] == {"select": _PERSISTED_COLUMNS}
    assert request["headers"]["Prefer"] == "return=representation"
    assert request["json"]["url"] == "https://huggingface.co/org/run"
    assert request["json"]["private"] is True
    assert stored.updated_at == "2026-07-14T00:00:01+00:00"


def test_insert_conflict_is_explicit() -> None:
    FakeClient.post_status = 409
    FakeClient.post_body = {"code": "23505", "message": "duplicate key value"}
    with pytest.raises(PersistenceConflict):
        insert_adapter(_record(), _settings())


def test_foreign_key_violation_is_not_reported_as_a_duplicate_row() -> None:
    """A 409 carrying 23503 means org_id names no org -- the opposite of "already exists".

    Reporting it as a conflict sends the caller to read back a row that was never inserted, so a
    permanent misconfiguration surfaces as a retryable 503 (SERVE-020).
    """
    FakeClient.post_status = 409
    FakeClient.post_body = {
        "code": "23503",
        "details": 'Key (org_id)=(00000000-0000-0000-0000-000000000000) is not present in table "orgs".',
        "message": "insert or update violates foreign key constraint",
    }
    with pytest.raises(PersistenceReferenceError) as caught:
        insert_adapter(_record(org_id="00000000-0000-0000-0000-000000000000"), _settings())
    # the message must name the unresolvable value, or the operator still has to guess
    assert "00000000-0000-0000-0000-000000000000" in str(caught.value)
    assert not isinstance(caught.value, PersistenceConflict)


@pytest.mark.parametrize(
    "body",
    [
        None,
        {"code": "23514", "message": "check constraint violation"},
    ],
)
def test_unclassified_409_does_not_enter_duplicate_reconciliation(body: object) -> None:
    FakeClient.post_status = 409
    FakeClient.post_body = body
    with pytest.raises(RuntimeError, match="Failed to insert hosted LoRA adapter: 409"):
        insert_adapter(_record(), _settings())


def test_replace_adapter_uses_updated_at_cas() -> None:
    replacement = _record(status="ready")
    FakeClient.patch_rows = [_row(status="ready", updated_at="2026-07-14T00:00:02+00:00")]
    committed = replace_adapter_cas(
        replacement,
        expected_updated_at="2026-07-14T00:00:01+00:00",
        settings=_settings(),
    )
    request = FakeClient.requests[0]
    assert request["params"] == {
        "select": _PERSISTED_COLUMNS,
        "adapter_id": f"eq.{REVISION_ID}",
        "updated_at": "eq.2026-07-14T00:00:01+00:00",
    }
    assert committed is not None and committed.status == "ready"


def test_status_cas_patches_only_status_and_updated_at() -> None:
    FakeClient.patch_rows = [_row(status="ready", updated_at="2026-07-14T00:00:02+00:00")]
    committed = update_adapter_status_cas(
        REVISION_ID,
        "ready",
        expected_updated_at="2026-07-14T00:00:01+00:00",
        settings=_settings(),
    )
    request = FakeClient.requests[0]
    assert request["params"]["select"] == _PERSISTED_COLUMNS
    assert set(request["json"]) == {"status", "updated_at"}
    assert request["json"]["status"] == "ready"
    assert committed is not None and committed.status == "ready"


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
    FakeClient.get_rows = [_row(id=123)]
    assert load_adapters(_settings()) == []
    with pytest.raises(PersistenceRecordError, match="invalid adapter row"):
        get_adapter(REVISION_ID, _settings())


def test_row_round_trip_preserves_exact_source_identity() -> None:
    record = _record().model_copy(update={"deployment_generation": "generation-1"})
    row = _adapter_to_row(record, "2026-07-14T00:00:03+00:00")
    hydrated = PersistedAdapterRecord.model_validate(_row_to_adapter(row))
    assert hydrated.repo_id == record.repo_id
    assert hydrated.repo_type == record.repo_type
    assert hydrated.hf_revision == SHA
    assert hydrated.subfolder == record.subfolder
    assert hydrated.url == record.url
    assert hydrated.private is True
    assert row["deployment_generation"] == "generation-1"
    assert "deployment_generation" not in row["metadata"]
    assert hydrated.deployment_generation == "generation-1"
    assert "deployment_generation" not in hydrated.model_dump()
