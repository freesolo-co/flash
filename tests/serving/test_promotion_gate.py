"""Gate ordering, bounded accounting, and what reaches the build log on failure."""

from __future__ import annotations

import asyncio
import inspect
import pathlib
from unittest import mock

import pytest

from flash.serving.promotion import gate as gate_module
from flash.serving.promotion.canary import (
    CANARY_TIMEOUT,
    CANARY_TRANSPORT_FAILURE,
    CanaryError,
)
from flash.serving.promotion.evidence import (
    ACCOUNTING_MALFORMED,
    ACCOUNTING_STALLED,
    HEALTH_SHA_MISMATCH,
    STREAM_NO_CONTENT,
    StreamEvidence,
)
from flash.serving.promotion.gate import (
    GATE_CONFIG_INCOMPLETE,
    HEALTH_UNREACHABLE,
    evaluate_promotion,
)
from flash.serving.src.accounting.usage_outbox import DurableUsageOutbox

SHA = "94210a323f9beaa713241e305f178b364848446d"
DEPLOYMENT_ID = "12345-1"


def _Snapshot(pending=0, leased=0, expired_leases=0, oldest_undelivered_age_seconds=None):
    """A backlog RPC body, shaped exactly as `serving_usage_backlog_snapshot` returns it.

    `pending`/`leased` are still sent by the RPC and still shaped here, because the point of the
    test below is that a deployment-wide backlog does NOT fail a promotion. The gate no longer
    reads them; this helper keeps them so the body stays the real wire shape.
    """
    return {
        "states": {"pending": pending, "leased": leased},
        "expired_leases": expired_leases,
        "oldest_undelivered_age_seconds": oldest_undelivered_age_seconds,
    }


def _health(**overrides):
    body = {"ok": True, "deployment_sha": SHA, "deployment_id": DEPLOYMENT_ID, "gpus": 2}
    body.update(overrides)
    return body


def _good_stream() -> StreamEvidence:
    return StreamEvidence(
        content_type_ok=True,
        content_delta_count=2,
        finish_reason="stop",
        completion_tokens=5,
        saw_done_sentinel=True,
    )


def _evaluate(*, health, stream, accounting, **kwargs):
    calls: list[str] = []

    async def health_loader():
        calls.append("health")
        if isinstance(health, BaseException):
            raise health
        return health

    async def stream_runner():
        calls.append("stream")
        if isinstance(stream, BaseException):
            raise stream
        return stream

    async def accounting_loader():
        calls.append("accounting")
        if isinstance(accounting, BaseException):
            raise accounting
        if callable(accounting):
            return accounting()
        return accounting

    async def no_sleep(_seconds: float) -> None:
        return None

    verdict = asyncio.run(
        evaluate_promotion(
            health_loader=health_loader,
            stream_runner=stream_runner,
            accounting_loader=accounting_loader,
            expected_sha=SHA,
            expected_deployment_id=DEPLOYMENT_ID,
            sleep=no_sleep,
            **kwargs,
        )
    )
    return verdict, calls


def test_a_wrong_release_is_never_streamed_against():
    """Streaming against a router that is not this release would prove nothing about it.

    It would also bill a real generation and settle real usage under the WRONG release id, so the
    accounting evidence would be actively misleading rather than merely useless.
    """
    verdict, calls = _evaluate(
        health=_health(deployment_sha="0" * 40), stream=_good_stream(), accounting=_Snapshot()
    )
    assert verdict.reason == HEALTH_SHA_MISMATCH
    assert calls == ["health"]


def test_an_unreachable_router_fails_the_gate_instead_of_crashing_it():
    """A crashed step does not run the rollback that a failed step does."""
    verdict, calls = _evaluate(
        health=RuntimeError("connection refused"), stream=_good_stream(), accounting=_Snapshot()
    )
    assert verdict.reason == HEALTH_UNREACHABLE
    assert calls == ["health"]


def test_a_failed_stream_does_not_wait_on_an_accounting_backlog():
    """There is no usage row to wait for, so polling would only burn the deadline."""
    verdict, calls = _evaluate(
        health=_health(),
        stream=StreamEvidence(
            content_type_ok=True,
            content_delta_count=0,
            finish_reason="stop",
            completion_tokens=5,
            saw_done_sentinel=True,
        ),
        accounting=_Snapshot(),
    )
    assert verdict.reason == STREAM_NO_CONTENT
    assert calls == ["health", "stream"]


def test_a_canary_error_becomes_its_reason_code_not_an_exception():
    verdict, _ = _evaluate(
        health=_health(), stream=CanaryError(CANARY_TIMEOUT), accounting=_Snapshot()
    )
    assert verdict.reason == CANARY_TIMEOUT


def test_a_delivery_loop_that_stays_wedged_fails_within_the_deadline():
    """Delivery is asynchronous, so the gate retries; it must still stop."""
    verdict, calls = _evaluate(
        health=_health(),
        stream=_good_stream(),
        accounting=_Snapshot(expired_leases=1),
        accounting_deadline_seconds=10,
        poll_seconds=5,
    )
    assert verdict.reason == ACCOUNTING_STALLED
    # bounded: three reads at t=0, 5, 10, then the deadline stops it.
    assert calls.count("accounting") == 3


def test_a_transient_stall_that_recovers_is_given_time_to_clear():
    """An expired lease gets recovered by the worker, so one bad read is not a verdict."""
    snapshots = iter([_Snapshot(expired_leases=1), _Snapshot(expired_leases=1), _Snapshot()])
    verdict, calls = _evaluate(
        health=_health(),
        stream=_good_stream(),
        accounting=lambda: next(snapshots),
        accounting_deadline_seconds=60,
        poll_seconds=5,
    )
    assert verdict.ok
    assert calls.count("accounting") == 3


def test_the_accounting_deadline_outlives_a_lease_a_deploy_itself_expires():
    """The deadline is not a round number, it is a function of the outbox's lease.

    `modal deploy` replaces the router containers. A container holding a claimed row when it goes
    away leaves that lease to EXPIRE rather than releasing it, so a deploy manufactures the exact
    `expired_leases > 0` signal `stalled` reads -- on a perfectly healthy release. The replacement
    worker reclaims it on its next sweep, which is why the gate polls instead of failing on the
    first read. But that only saves the promotion if the deadline outlives the lease: at or below
    it, the gate would stop looking before the recovery it is waiting for could possibly have
    happened, and every deploy would roll a healthy release back to its predecessor.

    Asserted against the canonical default rather than a restated number, so raising the outbox's
    lease turns into a red test here instead of a rollback on the next production deploy.
    """
    lease_seconds = (
        inspect.signature(DurableUsageOutbox.__init__).parameters["lease_seconds"].default
    )

    assert lease_seconds * 2 < gate_module._DEFAULT_ACCOUNTING_DEADLINE_SECONDS, (
        f"the accounting deadline is {gate_module._DEFAULT_ACCOUNTING_DEADLINE_SECONDS}s but the "
        f"outbox leases for {lease_seconds}s. A deploy expires leases by replacing the containers "
        "holding them; the gate must outlast at least one full reclaim cycle or it rolls back "
        "healthy releases."
    )


def test_concurrent_traffic_does_not_fail_the_promotion():
    """The snapshot is deployment-wide, so unrelated in-flight rows are always present.

    Requiring a drained backlog here would make every promotion a coin flip on production traffic.
    """
    verdict, _ = _evaluate(
        health=_health(),
        stream=_good_stream(),
        accounting=_Snapshot(pending=12, leased=4),
    )
    assert verdict.ok


def test_an_unreadable_snapshot_never_passes_as_settled():
    verdict, _ = _evaluate(
        health=_health(),
        stream=_good_stream(),
        accounting=RuntimeError("supabase_rpc_500"),
        accounting_deadline_seconds=0,
    )
    assert verdict.reason == ACCOUNTING_MALFORMED


def test_promotion_passes_only_when_all_three_layers_prove_themselves():
    verdict, calls = _evaluate(health=_health(), stream=_good_stream(), accounting=_Snapshot())
    assert verdict.ok
    assert verdict.reason == ""
    assert calls == ["health", "stream", "accounting"]


def test_incomplete_configuration_fails_closed_without_naming_the_value(monkeypatch, capsys):
    """A missing secret must not be diagnosed by echoing anything about it."""
    monkeypatch.setenv("SERVING_BASE_URL", "https://serve.freesolo.co")
    for name in gate_module._REQUIRED_ENV:
        monkeypatch.delenv(name, raising=False)
    assert gate_module.main([]) == 1
    captured = capsys.readouterr()
    assert GATE_CONFIG_INCOMPLETE in captured.err
    assert captured.out == ""


def test_a_missing_base_url_does_not_deploy_a_gate_against_nothing(monkeypatch, capsys):
    """The URL specifically, with EVERY other required variable present.

    Deleting `SERVING_BASE_URL` from an already-empty environment proves nothing: the test passed
    with the base-url check deleted outright, because the other required variables were empty too
    and one of them failed first. The whole environment is populated here so the URL is the only
    thing missing, which is exactly the shape of the real hazard -- a workflow that drops the URL
    from the gate step would otherwise probe `/healthz` at a relative path and read the failure as
    an unhealthy release rather than as its own misconfiguration.
    """
    for name, value in _gate_step_env().items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("SERVING_BASE_URL", raising=False)

    assert gate_module.main([]) == 1
    assert GATE_CONFIG_INCOMPLETE in capsys.readouterr().err


def _gate_step_env() -> dict[str, str]:
    """Exactly the variables the workflow's gate step actually supplies, read from the YAML.

    Deriving this from the workflow rather than restating it keeps the code and the step that runs
    it from drifting apart silently: a variable dropped from the `env:` block turns into a red test
    rather than a broken deploy.
    """
    import yaml

    workflow = yaml.safe_load(
        (pathlib.Path(gate_module.__file__).parents[3] / ".github/workflows/deploy-modal.yml")
        .read_text()
        .replace("\ton", "\t'on'")
    )
    steps = workflow["jobs"]["deploy"]["steps"]
    gate_step = next(s for s in steps if s.get("run", "").strip().endswith("promotion.gate"))
    supplied = set(gate_step.get("env", {}))
    values = {"SUPABASE_SERVICE_ROLE_KEY": "sb_secret_x", "SUPABASE_URL": "https://db.example"}
    return {n: values.get(n, "x") for n in gate_module._REQUIRED_ENV if n in supplied}


def test_the_gate_is_constructible_from_the_environment_the_workflow_supplies():
    """The gate must actually be able to build itself in CI.

    Every test above injects its loaders, so none of them touches the real construction. That is
    exactly where the gate crashed: it built a `DurableUsageOutbox`, which takes a REQUIRED
    keyword-only `worker_id` and additionally refuses to construct without `backend_url` and
    `deployment_id`, none of which the workflow's gate step puts in the environment. A crash there
    is worse than a failed gate -- the rollback step keys off `failure()`, so it would fire on every
    single deploy and roll production back to its predecessor even when the release was fine.
    """
    plan = gate_module._resolve("https://serve.freesolo.co", _gate_step_env())

    assert plan.expected_sha == "x"
    assert plan.expected_deployment_id == "x-x"
    assert callable(plan.read_backlog)


def test_a_malformed_service_role_key_fails_the_gate_instead_of_crashing_it():
    """A misconfigured secret must decline to promote, NOT roll production back.

    `supabase_headers` raises on a key that is not `sb_secret_`-shaped. Letting that escape would
    crash the step, and the rollback fires on `failure()` for a crash exactly as it does for a
    failure -- so a typo in a secret would redeploy the previous release over a healthy one.
    """
    env = {**_gate_step_env(), "SUPABASE_SERVICE_ROLE_KEY": "eyJhbGciOiJIUzI1NiJ9.legacy.key"}

    with pytest.raises(gate_module.GateConfigError):
        gate_module._resolve("https://serve.freesolo.co", env)


def test_an_empty_base_model_catalog_fails_the_gate_instead_of_crashing_it():
    """Same rollback hazard, reached through the other resolvable config fault."""
    with (
        mock.patch("flash.serving.src.engine.model_config.base_models", return_value=[]),
        pytest.raises(gate_module.GateConfigError),
    ):
        gate_module._resolve("https://serve.freesolo.co", _gate_step_env())


def test_the_entrypoint_is_importable_as_a_module(monkeypatch, capsys):
    """`python -m flash.serving.promotion.gate` must reach main().

    Defining the entrypoint below a `__main__` guard makes it dead for `python -m` while every
    in-process test still passes.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "flash.serving.promotion.gate"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(gate_module.__file__).split("/flash/")[0]},
    )
    assert result.returncode == 1
    assert GATE_CONFIG_INCOMPLETE in result.stderr


def test_an_unexpected_canary_exception_fails_the_gate_instead_of_crashing_it():
    """A crash is strictly worse than a failure: `failure()` cannot tell the two apart.

    `run_stream_canary` collapses transport faults into `CanaryError`, but that guarantee lives in
    another module and the runner is injected here. If anything else escapes, the step crashes,
    Actions fires the rollback, and a healthy release is redeployed over with its predecessor --
    the exact outcome the gate exists to prevent. The health load above already catches broadly;
    this is the same hazard twelve lines further down.
    """
    verdict, calls = _evaluate(
        health=_health(),
        stream=RuntimeError("httpx raised something the canary does not wrap"),
        accounting=_Snapshot(),
    )

    assert verdict.ok is False
    assert verdict.reason == CANARY_TRANSPORT_FAILURE
    assert "accounting" not in calls, "a failed stream must not be followed by an accounting wait"
