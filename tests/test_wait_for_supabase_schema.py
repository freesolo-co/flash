from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "wait_for_supabase_schema.py"
SPEC = importlib.util.spec_from_file_location("wait_for_supabase_schema", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
schema_wait = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(schema_wait)


@pytest.mark.parametrize("status", [0, 400, 404, 406, 500, 503, 599])
def test_schema_status_classifies_migration_races_as_retryable(status: int) -> None:
    assert schema_wait.classify_schema_status(status) == "retry"


@pytest.mark.parametrize("status", [401, 403])
def test_schema_status_classifies_authorization_errors_as_terminal(status: int) -> None:
    assert schema_wait.classify_schema_status(status) == "authorization"


@pytest.mark.parametrize("status", [201, 301, 409, 422, 600])
def test_schema_status_classifies_unrelated_statuses_as_terminal(status: int) -> None:
    assert schema_wait.classify_schema_status(status) == "terminal"


def test_schema_wait_retries_then_succeeds() -> None:
    statuses = iter([404, 500, 200])
    clock = [0.0]
    sleeps: list[float] = []

    def sleep(delay: float) -> None:
        sleeps.append(delay)
        clock[0] += delay

    schema_wait.wait_for_schema(
        supabase_url="https://example.supabase.co",
        service_role_key="secret",
        table="hosted_model_readiness_passes",
        columns="model_id,passed_at",
        label="readiness schema",
        timeout_seconds=30.0,
        interval_seconds=10.0,
        clock=lambda: clock[0],
        sleep=sleep,
        probe=lambda *_args: next(statuses),
    )

    assert sleeps == [10.0, 10.0]


def test_schema_wait_fails_authorization_without_sleeping() -> None:
    sleeps: list[float] = []

    with pytest.raises(RuntimeError, match="rejected with HTTP 403"):
        schema_wait.wait_for_schema(
            supabase_url="https://example.supabase.co",
            service_role_key="bad-secret",
            table="hosted_model_readiness_passes",
            columns="model_id",
            label="readiness schema",
            clock=lambda: 0.0,
            sleep=sleeps.append,
            probe=lambda *_args: 403,
        )

    assert sleeps == []


def test_schema_wait_times_out_after_bounded_window() -> None:
    clock = [0.0]

    def sleep(delay: float) -> None:
        clock[0] += delay

    with pytest.raises(RuntimeError, match="within 20 seconds"):
        schema_wait.wait_for_schema(
            supabase_url="https://example.supabase.co",
            service_role_key="secret",
            table="hosted_model_readiness_passes",
            columns="model_id",
            label="readiness schema",
            timeout_seconds=20.0,
            interval_seconds=10.0,
            clock=lambda: clock[0],
            sleep=sleep,
            probe=lambda *_args: 404,
        )

    assert clock[0] == 20.0
