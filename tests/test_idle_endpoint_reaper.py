"""The control-plane idle-endpoint reaper: it must protect every live run's endpoint and pass the
run-aware protected set + idle grace down to the provider sweep. (Server-side; no fastapi needed.)
"""

from __future__ import annotations

import pytest

import flash.providers.runpod.execution.resources as runpod_resources
import flash.server.asgi.app as app_mod
from flash.providers._lifecycle.net.destructive import DestructiveOperationOutcome
from flash.providers.core.base import canonical_gpu
from flash.providers.runpod.serverless.endpoints import _run_suffix, endpoint_name
from flash.runner.lifecycle.state import RunStatus

# Test run-ids below are plain fixtures: a run id is any string starting with ``flash-`` (the
# server assigns ``flash-<ts>-<rand>``), and these just name the SCENARIO for readability —
# ``flash-live`` is a running run, ``flash-done`` a finished one, ``flash-prov`` a provisioning one.
# They are NOT special names the code keys on (the only real name conventions are the ``flash-`` /
# ``live-flash-`` endpoint forms; see ``canonical_endpoint_name``).


def _derived(gpu: str, run_id: str) -> str:
    return endpoint_name(canonical_gpu(gpu), _run_suffix(run_id))


def test_protected_names_cover_live_runs_only(monkeypatch):
    rows = [{"run_id": r} for r in ("flash-active", "flash-prov", "flash-done")]
    statuses = {
        # running run with a persisted handle + a spec gpu
        "flash-active": RunStatus(
            run_id="flash-active",
            state="running",
            spec={"gpu": {"type": "RTX 5090"}},
            remote={"endpoint_name": "flash-5090-handle", "endpoint_id": "e", "job_id": "j"},
        ),
        # provisioning run with no handle yet (submit -> handle-persist window)
        "flash-prov": RunStatus(
            run_id="flash-prov", state="provisioning", spec={"gpu": {"type": "A100"}}
        ),
        # terminal run: must NOT be protected
        "flash-done": RunStatus(
            run_id="flash-done", state="done", spec={"gpu": {"type": "RTX 5090"}}
        ),
    }
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: rows)
    monkeypatch.setattr(app_mod, "get_status", lambda rid: statuses[rid])

    names = app_mod._protected_train_endpoint_names()

    # active run: the persisted handle name AND the spec-derived name. The set holds the CANONICAL
    # bare form only — the reaper canonicalizes the ``live-flash-...`` names RunPod lists before
    # comparing, so the ``live-`` form is deliberately NOT stored here.
    assert "flash-5090-handle" in names
    assert "live-flash-5090-handle" not in names
    active_derived = _derived("RTX 5090", "flash-active")
    assert active_derived in names
    assert f"live-{active_derived}" not in names
    # provisioning run: protected by its spec-derived name even with no handle.
    assert _derived("A100", "flash-prov") in names
    # terminal run: not protected (its endpoint is reapable once idle).
    assert _derived("RTX 5090", "flash-done") not in names


def test_ordered_gpu_pin_is_protected_before_its_handle_is_persisted(monkeypatch):
    """Every acceptable class is protected before the handle is persisted."""
    rows = [{"run_id": "flash-ordered"}]
    statuses = {
        "flash-ordered": RunStatus(
            run_id="flash-ordered",
            state="provisioning",
            spec={"gpu": {"type": ["A100 PCIe", "A100 SXM"]}},
        )
    }
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: rows)
    monkeypatch.setattr(app_mod, "get_status", lambda rid: statuses[rid])

    names = app_mod._protected_train_endpoint_names()

    assert names, "an ordered-pin run contributed no protected name at all"
    assert _derived("A100 PCIe", "flash-ordered") in names
    assert _derived("A100 SXM", "flash-ordered") in names


def test_an_orphaned_fallback_endpoint_stays_in_the_reapers_scope(monkeypatch):
    """A terminal fallback endpoint remains within the reaper's known scope."""
    rows = [{"run_id": "flash-orphan"}]
    statuses = {
        "flash-orphan": RunStatus(
            run_id="flash-orphan",
            state="failed",
            spec={"gpu": {"type": ["A100 PCIe", "A100 SXM"]}},
        )
    }
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: rows)
    monkeypatch.setattr(app_mod, "get_status", lambda rid: statuses[rid])

    known = app_mod._known_train_endpoint_names()

    # the fallback is the one the head-only index missed, and the one that leaks.
    assert _derived("A100 SXM", "flash-orphan") in known
    assert _derived("A100 PCIe", "flash-orphan") in known


def test_reap_once_passes_protected_set_and_grace(monkeypatch):
    monkeypatch.setattr(app_mod, "_protected_train_endpoint_names", lambda: {"flash-live"})
    # The reaper also passes the KNOWN set (every run this plane has a record of) so it only reaps
    # this plane's own idle endpoints, never another control plane's between-jobs endpoint.
    monkeypatch.setattr(
        app_mod, "_known_train_endpoint_names", lambda: {"flash-live", "flash-done"}
    )
    captured: dict = {}

    def fake_sweep(protected, min_idle_s=0.0, known=None, should_stop=None):
        captured["protected"] = protected
        captured["grace"] = min_idle_s
        captured["known"] = known
        captured["should_stop"] = should_stop
        return runpod_resources.IdleEndpointSweepResult(deleted_ids=("a", "b", "c"))

    def stop() -> bool:
        return False

    monkeypatch.setattr(runpod_resources, "_sweep_idle_flash_endpoints", fake_sweep)
    assert app_mod._reap_idle_endpoints_once(900.0, stop).deleted_count == 3
    assert captured == {
        "protected": {"flash-live"},
        "grace": 900.0,
        "known": {"flash-live", "flash-done"},
        # identity, not truthiness: the loop's stop event is what must reach the sweep. a wrapper
        # that swallowed it and defaulted to None would leave every other assertion here green.
        "should_stop": stop,
    }
    assert captured["should_stop"] is stop


def test_protected_names_skip_unreadable_run(monkeypatch):
    # A run row whose status file vanished (FileNotFoundError) is skipped, not fatal.
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "gone"}, {"run_id": "live"}])

    def get_status(rid):
        if rid == "gone":
            raise FileNotFoundError(rid)
        return RunStatus(run_id="live", state="running", spec={"gpu": {"type": "A100"}})

    monkeypatch.setattr(app_mod, "get_status", get_status)
    names = app_mod._protected_train_endpoint_names()
    assert _derived("A100", "live") in names


# --- the RunPod idle sweep across a multi-account key pool --------------------------------------
# Regression cover for the orphan pile-up: the sweep used the all-or-nothing ``list_endpoints``, so
# one unhealthy pool key (rejected/expired/rate-limited) aborted the WHOLE sweep and idle orphans on
# every OTHER account survived indefinitely. The sweep now lists per account and reaps what responds.


def _idle_health():
    """A warm-idle endpoint with no work — reapable under reap_warm=True."""
    return {
        "workers": {"running": 0, "initializing": 0, "ready": 1, "idle": 1},
        "jobs": {"inQueue": 0, "inProgress": 0},
    }


def test_canonical_endpoint_name_strips_sdk_live_prefix():
    """One endpoint, two names: flash stores the bare ``flash-...`` form, the runpod-flash SDK lists
    it as ``live-flash-...``. ``canonical_endpoint_name`` collapses them to the bare form so every
    comparison site uses one name. Idempotent; a non-``live-`` name passes through unchanged."""
    assert runpod_resources.canonical_endpoint_name("live-flash-5090-abc") == "flash-5090-abc"
    assert runpod_resources.canonical_endpoint_name("flash-5090-abc") == "flash-5090-abc"
    assert (
        runpod_resources.canonical_endpoint_name(
            runpod_resources.canonical_endpoint_name("live-flash-x")
        )
        == "flash-x"
    )
    assert runpod_resources.canonical_endpoint_name("") == ""


@pytest.mark.parametrize(
    "endpoint",
    [
        pytest.param({"name": "live-flash-bad", "id": ""}, id="empty"),
        pytest.param({"name": "live-flash-bad", "id": "   "}, id="whitespace"),
        pytest.param({"name": "live-flash-bad", "id": " ep-bad"}, id="leading-space"),
        pytest.param({"name": "live-flash-bad", "id": "ep-bad "}, id="trailing-space"),
        pytest.param({"name": "live-flash-bad", "id": True}, id="bool"),
        pytest.param({"name": "live-flash-bad", "id": 7}, id="integer"),
        pytest.param({"name": "live-flash-bad", "id": None}, id="none"),
        pytest.param({"name": "live-flash-bad"}, id="missing"),
    ],
)
def test_sweep_rejects_malformed_selected_endpoint_ids_before_provider_calls(monkeypatch, endpoint):
    runpod_resources._idle_since.clear()
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "list_endpoints_by_key",
        lambda **_kwargs: ({"fpA": [endpoint]}, []),
    )
    provider_calls = []
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *args, **kwargs: provider_calls.append(("health", args, kwargs)),
    )
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "_delete_endpoint_for_fingerprint_outcome",
        lambda *args, **kwargs: provider_calls.append(("delete", args, kwargs)),
    )

    result = runpod_resources._sweep_idle_flash_endpoints(
        protected=set(), min_idle_s=0.0, known={"flash-bad"}
    )

    assert result.deleted_count == 0
    assert result.unresolved_count == 1
    assert result.unresolved[0].endpoint_name == "flash-bad"
    assert result.unresolved[0].reason == "invalid selected endpoint identity"
    assert provider_calls == []
    assert runpod_resources._idle_since == {}


def test_sweep_rejects_malformed_selected_owner_before_provider_calls(monkeypatch):
    runpod_resources._idle_since.clear()
    endpoint = {"name": "live-flash-bad-owner", "id": "ep-valid"}
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "list_endpoints_by_key",
        lambda **_kwargs: ({" fpA": [endpoint]}, []),
    )
    provider_calls = []
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *args, **kwargs: provider_calls.append(("health", args, kwargs)),
    )
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "_delete_endpoint_for_fingerprint_outcome",
        lambda *args, **kwargs: provider_calls.append(("delete", args, kwargs)),
    )

    result = runpod_resources._sweep_idle_flash_endpoints(
        protected=set(), min_idle_s=0.0, known={"flash-bad-owner"}
    )

    assert result.deleted_count == 0
    assert result.unresolved_count == 1
    assert result.unresolved[0].observed_endpoint_id == "'ep-valid'"
    assert provider_calls == []
    assert runpod_resources._idle_since == {}


def test_sweep_rejects_endpoint_id_with_ambiguous_owner_before_provider_calls(monkeypatch):
    runpod_resources._idle_since.clear()
    endpoint = {"name": "live-flash-ambiguous", "id": "ep-shared"}
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "list_endpoints_by_key",
        lambda **_kwargs: ({"fpA": [endpoint], "fpB": [endpoint]}, []),
    )
    provider_calls = []
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *args, **kwargs: provider_calls.append(("health", args, kwargs)),
    )
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "_delete_endpoint_for_fingerprint_outcome",
        lambda *args, **kwargs: provider_calls.append(("delete", args, kwargs)),
    )

    result = runpod_resources._sweep_idle_flash_endpoints(
        protected=set(), min_idle_s=0.0, known={"flash-ambiguous"}
    )

    assert result.deleted_count == 0
    assert result.unresolved_count == 1
    assert result.unresolved[0].reason == "endpoint identity appeared under multiple owners"
    assert provider_calls == []
    assert runpod_resources._idle_since == {}


def test_sweep_mixed_valid_and_malformed_owners_blocks_provider_calls(monkeypatch):
    runpod_resources._idle_since.clear()
    endpoint = {"name": "live-flash-mixed-owner", "id": "ep-mixed"}
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "list_endpoints_by_key",
        lambda **_kwargs: ({"fp-valid": [endpoint], " fp-malformed": [endpoint]}, []),
    )
    provider_calls = []
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *args, **kwargs: provider_calls.append(("health", args, kwargs)),
    )
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "_delete_endpoint_for_fingerprint_outcome",
        lambda *args, **kwargs: provider_calls.append(("delete", args, kwargs)),
    )

    result = runpod_resources._sweep_idle_flash_endpoints(
        protected=set(), min_idle_s=0.0, known={"flash-mixed-owner"}
    )

    assert result.deleted_count == 0
    assert {issue.reason for issue in result.unresolved} == {
        "endpoint identity appeared under multiple owners",
        "invalid selected endpoint identity",
    }
    assert provider_calls == []
    assert runpod_resources._idle_since == {}


@pytest.mark.parametrize(
    ("group", "field", "value"),
    [
        pytest.param("workers", "running", True, id="worker-bool"),
        pytest.param("workers", "initializing", -1, id="worker-negative"),
        pytest.param("workers", "ready", 1.0, id="worker-float"),
        pytest.param("workers", "idle", "0", id="worker-string"),
        pytest.param("jobs", "inQueue", False, id="job-bool"),
        pytest.param("jobs", "inProgress", -1, id="job-negative"),
    ],
)
def test_sweep_rejects_malformed_cleanup_health_counters(monkeypatch, group, field, value):
    runpod_resources._idle_since.clear()
    endpoint = {"name": "live-flash-bad-health", "id": "ep-health"}
    health = _idle_health()
    health[group][field] = value
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "list_endpoints_by_key",
        lambda **_kwargs: ({"fpA": [endpoint]}, []),
    )
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *_args, **_kwargs: health,
    )
    deletes = []
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "_delete_endpoint_for_fingerprint_outcome",
        lambda *args, **kwargs: (
            deletes.append((args, kwargs)) or DestructiveOperationOutcome.DELETED
        ),
    )

    result = runpod_resources._sweep_idle_flash_endpoints(
        protected=set(), min_idle_s=0.0, known={"flash-bad-health"}
    )

    assert result.deleted_count == 0
    assert result.unresolved_count == 1
    assert result.unresolved[0].reason == "health evidence unavailable"
    assert deletes == []
    assert runpod_resources._idle_since == {}


def test_sweep_rejects_incomplete_cleanup_health(monkeypatch):
    runpod_resources._idle_since.clear()
    endpoint = {"name": "live-flash-incomplete-health", "id": "ep-health"}
    health = _idle_health()
    del health["jobs"]["inProgress"]
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "list_endpoints_by_key",
        lambda **_kwargs: ({"fpA": [endpoint]}, []),
    )
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *_args, **_kwargs: health,
    )
    deletes = []
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "_delete_endpoint_for_fingerprint_outcome",
        lambda *args, **kwargs: (
            deletes.append((args, kwargs)) or DestructiveOperationOutcome.DELETED
        ),
    )

    result = runpod_resources._sweep_idle_flash_endpoints(
        protected=set(), min_idle_s=0.0, known={"flash-incomplete-health"}
    )

    assert result.deleted_count == 0
    assert result.unresolved_count == 1
    assert result.unresolved[0].reason == "health evidence unavailable"
    assert deletes == []


def test_sweep_deduplicates_inventory_before_health_and_delete(monkeypatch):
    runpod_resources._idle_since.clear()
    endpoint = {"name": "live-flash-duplicate", "id": "ep-duplicate"}
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "list_endpoints_by_key",
        lambda **_kwargs: ({"fpA": [endpoint, dict(endpoint), endpoint]}, []),
    )
    health_calls = []
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *args, **kwargs: health_calls.append((args, kwargs)) or _idle_health(),
    )
    deletes = []
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "_delete_endpoint_for_fingerprint_outcome",
        lambda *args, **kwargs: (
            deletes.append((args, kwargs)) or DestructiveOperationOutcome.DELETED
        ),
    )

    result = runpod_resources._sweep_idle_flash_endpoints(
        protected=set(), min_idle_s=0.0, known={"flash-duplicate"}
    )

    assert result.deleted_ids == ("ep-duplicate",)
    assert len(health_calls) == 1
    assert len(deletes) == 1


def test_idle_sweep_result_rejects_duplicate_evidence() -> None:
    issue = runpod_resources.IdleEndpointSweepIssue("fpA", "flash-dup", "ep-dup", "failed")

    with pytest.raises(ValueError, match="deleted_ids must be unique"):
        runpod_resources.IdleEndpointSweepResult(deleted_ids=("ep-dup", "ep-dup"))
    with pytest.raises(ValueError, match="unresolved must be unique"):
        runpod_resources.IdleEndpointSweepResult(unresolved=(issue, issue))
    with pytest.raises(ValueError, match="failed_owner_fingerprints must be unique"):
        runpod_resources.IdleEndpointSweepResult(failed_owner_fingerprints=("fpA", "fpA"))


def test_sweep_reaps_responsive_account_when_one_pool_key_fails(monkeypatch):
    """One pool key fails to list this cycle; the responding account's idle orphan is still reaped,
    using that account's OWN key — and the failure is surfaced at WARNING (not a silent DEBUG)."""
    runpod_resources._idle_since.clear()
    orphan = {"id": "ep-b1", "name": "live-flash-5090-orphan"}
    # fpA failed to list; fpB returned the orphan. (Accounts are identified by non-secret
    # fingerprints, never the raw key.)
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "list_endpoints_by_key",
        lambda **_kwargs: ({"fpB": [orphan]}, ["fpA"]),
    )
    health_calls = []

    def health(eid, fp):
        health_calls.append((eid, fp))
        return _idle_health()

    deletes = []
    monkeypatch.setattr(runpod_resources.runpod_api, "endpoint_health_for_fingerprint", health)
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "_delete_endpoint_for_fingerprint_outcome",
        lambda eid, fp: deletes.append((eid, fp)) or DestructiveOperationOutcome.DELETED,
    )
    warnings = []
    monkeypatch.setattr(runpod_resources.logger, "warning", lambda *a, **k: warnings.append(a))

    # min_idle_s=0 -> a first idle observation is immediately reapable.
    deleted = runpod_resources._sweep_idle_flash_endpoints(protected=set(), min_idle_s=0.0)

    assert deleted.deleted_count == 1
    assert deleted.failed_owner_fingerprints == ("fpA",)
    assert health_calls == [("ep-b1", "fpB")]  # account-scoped: queried with the OWNING fingerprint
    assert deletes == [("ep-b1", "fpB")]  # ...and deleted with it, no failover waterfall
    assert warnings, "a failed pool account must be surfaced at WARNING, not swallowed at DEBUG"


def test_sweep_skips_endpoints_outside_known_scope(monkeypatch):
    """Multi-plane safety for RunPod: with a ``known`` scope, the reaper deletes only idle endpoints
    THIS plane has a record of. An idle endpoint owned by another control plane on the same account
    (its name absent from ``known``) is left alone, even though it is idle and unprotected."""
    runpod_resources._idle_since.clear()
    mine = {"id": "ep-mine", "name": "live-flash-mine-idle"}
    theirs = {"id": "ep-theirs", "name": "live-flash-theirs-idle"}
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "list_endpoints_by_key",
        lambda **_kwargs: ({"fpA": [mine, theirs]}, []),
    )
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "endpoint_health_for_fingerprint",
        lambda eid, fp: _idle_health(),
    )
    deletes = []
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "_delete_endpoint_for_fingerprint_outcome",
        lambda eid, fp: deletes.append(eid) or DestructiveOperationOutcome.DELETED,
    )

    # known carries only OUR endpoint name (bare form); the reaper compares both bare and live- forms.
    deleted = runpod_resources._sweep_idle_flash_endpoints(
        protected=set(), min_idle_s=0.0, known={"flash-mine-idle"}
    )

    assert deleted.deleted_count == 1
    assert deletes == ["ep-mine"]  # only ours; the other plane's idle endpoint is untouched


def test_sweep_preserves_grace_for_unlisted_account(monkeypatch):
    """A partial outage must not reset the idle-grace clock for the account it couldn't list — else
    a flaky account's orphan restarts its 15-min grace every sweep and never ages out."""
    runpod_resources._idle_since.clear()
    runpod_resources._idle_since["ep-a1"] = (
        1.0,
        "fpA",
    )  # orphan on account A, observed idle long ago
    # This cycle account A fails to list; account B responds with nothing.
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "list_endpoints_by_key",
        lambda **_kwargs: ({"fpB": []}, ["fpA"]),
    )
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "endpoint_health_for_fingerprint",
        lambda eid, fp: _idle_health(),
    )
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "_delete_endpoint_for_fingerprint_outcome",
        lambda eid, fp: DestructiveOperationOutcome.DELETED,
    )
    monkeypatch.setattr(runpod_resources.logger, "warning", lambda *a, **k: None)

    deleted = runpod_resources._sweep_idle_flash_endpoints(protected=set(), min_idle_s=900.0)

    assert deleted.deleted_count == 0
    # grace timer (owned by the FAILED account) SURVIVED the partial outage
    assert runpod_resources._idle_since.get("ep-a1") == (1.0, "fpA")


def test_sweep_partial_view_prunes_vanished_timer_for_responsive_account(monkeypatch):
    """During a partial outage, a stale grace timer whose OWNING account responded this cycle must
    still be pruned — the endpoint genuinely vanished from a healthy account. (The earlier
    listed-ids-only prune leaked it: any one failing account kept every responsive account's vanished
    timers alive forever.) A timer owned by the FAILED account is still preserved."""
    runpod_resources._idle_since.clear()
    runpod_resources._idle_since["gone-b"] = (
        1.0,
        "fpB",
    )  # vanished from account B, which responds this cycle
    runpod_resources._idle_since["stay-a"] = (
        1.0,
        "fpA",
    )  # owned by account A, which fails this cycle
    # B responds (no longer lists gone-b); A fails.
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "list_endpoints_by_key",
        lambda **_kwargs: ({"fpB": []}, ["fpA"]),
    )
    monkeypatch.setattr(runpod_resources.logger, "warning", lambda *a, **k: None)

    deleted = runpod_resources._sweep_idle_flash_endpoints(protected=set(), min_idle_s=900.0)

    assert deleted.deleted_count == 0
    assert (
        "gone-b" not in runpod_resources._idle_since
    )  # responsive account's vanished timer pruned (the fix)
    assert runpod_resources._idle_since.get("stay-a") == (
        1.0,
        "fpA",
    )  # failed account's timer preserved


def test_sweep_full_view_prunes_vanished_grace_timer(monkeypatch):
    """With a complete fleet view (no failed account), a grace timer for an endpoint that is no
    longer present is pruned — the original behavior, unchanged."""
    runpod_resources._idle_since.clear()
    runpod_resources._idle_since["ghost"] = (1.0, "fpA")  # endpoint that has since vanished
    monkeypatch.setattr(
        runpod_resources.runpod_api, "list_endpoints_by_key", lambda **_kwargs: ({"fpA": []}, [])
    )
    monkeypatch.setattr(runpod_resources.logger, "warning", lambda *a, **k: None)

    deleted = runpod_resources._sweep_idle_flash_endpoints(protected=set(), min_idle_s=900.0)

    assert deleted.deleted_count == 0
    assert "ghost" not in runpod_resources._idle_since  # full view -> stale timer pruned


def test_sweep_halts_between_deletes_on_stop_signal(monkeypatch):
    """The sweep runs in a worker thread that task.cancel() cannot interrupt, so the lifespan
    signals it with a stop callback instead. Once set, no further endpoint is deleted."""
    runpod_resources._idle_since.clear()
    endpoints = [{"name": f"live-flash-ep{n}", "id": f"ep-{n}"} for n in (1, 2, 3)]
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "list_endpoints_by_key",
        lambda **_kwargs: ({"fpA": endpoints}, []),
    )
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *a, **k: _idle_health(),
    )
    deletes: list[str] = []
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "_delete_endpoint_for_fingerprint_outcome",
        lambda *a, **k: (
            deletes.append(a[1] if len(a) > 1 else "?") or DestructiveOperationOutcome.DELETED
        ),
    )
    monkeypatch.setattr(runpod_resources.logger, "info", lambda *a, **k: None)

    result = runpod_resources._sweep_idle_flash_endpoints(
        protected=set(),
        min_idle_s=0.0,
        known={"flash-ep1", "flash-ep2", "flash-ep3"},
        should_stop=lambda: len(deletes) >= 1,
    )

    assert len(deletes) == 1  # halted after the first delete
    assert result.deleted_count == 1


def test_runpod_stop_during_delete_retry_prevents_second_request(monkeypatch):
    """A stop raised inside the delete halts the sweep; it is not endpoint-level evidence.

    The delete honours the same stop signal, so it returns false the moment the signal lands
    mid-retry. Reporting that as a ``delete was not confirmed`` endpoint issue would blame the
    endpoint for the operator's shutdown -- exactly the sweep-level-condition-as-endpoint-record
    lie the halt flag replaces. ``unresolved_count`` alone cannot tell the two apart: it counts
    ``halted`` too, so it reads 1 under either representation. Assert the shape, not the count.
    """
    from flash.providers._lifecycle.net import http as lifecycle_http
    from flash.providers.runpod.client import api as runpod_api

    runpod_resources._idle_since.clear()
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "list_endpoints_by_key",
        lambda **_kwargs: ({"fpA": [{"name": "live-flash-ep1", "id": "ep-1"}]}, []),
    )
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *_args, **_kwargs: _idle_health(),
    )
    monkeypatch.setattr(runpod_api, "_key_for_fingerprint", lambda _fingerprint: "test-key")
    monkeypatch.setattr(lifecycle_http.time, "sleep", lambda _delay: None)
    attempts: list[str] = []
    stopping: list[bool] = []

    def fail_first_delete(_target, *, method, **_kwargs):
        attempts.append(method)
        stopping.append(True)
        raise OSError("transient delete failure")

    monkeypatch.setattr(runpod_api._CLIENT, "request", fail_first_delete)

    result = runpod_resources._sweep_idle_flash_endpoints(
        protected=set(),
        min_idle_s=0.0,
        known={"flash-ep1"},
        should_stop=lambda: bool(stopping),
    )

    assert attempts == ["DELETE"], f"expected one destructive request, got {attempts}"
    assert result.deleted_count == 0
    assert result.halted, "a stop inside the delete must report the sweep-level halt"
    assert not result.unresolved, f"halt must not fabricate endpoint evidence: {result.unresolved}"
    assert result.unresolved_count == 1


def test_delete_failure_is_not_reclassified_by_a_later_stop_signal(monkeypatch):
    """The delete attempt must preserve its own failure provenance after it returns."""
    from flash.providers.runpod.client import api as runpod_api

    runpod_resources._idle_since.clear()
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "list_endpoints_by_key",
        lambda **_kwargs: ({"fpA": [{"name": "live-flash-ep1", "id": "ep-1"}]}, []),
    )
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *_args, **_kwargs: _idle_health(),
    )
    monkeypatch.setattr(runpod_api, "_key_for_fingerprint", lambda _fingerprint: "test-key")

    def fail_delete(*_args, **_kwargs):
        raise runpod_api.RunpodApiError("delete denied")

    monkeypatch.setattr(runpod_api._CLIENT, "request_with_retries_for_key", fail_delete)
    stop_checks = 0

    def should_stop() -> bool:
        nonlocal stop_checks
        stop_checks += 1
        return stop_checks >= 3

    result = runpod_resources._sweep_idle_flash_endpoints(
        protected=set(),
        min_idle_s=0.0,
        known={"flash-ep1"},
        should_stop=should_stop,
    )

    assert should_stop()  # the unrelated stop arrives only after the failed attempt returned
    assert stop_checks == 3
    assert not result.halted
    assert result.deleted_count == 0
    assert len(result.unresolved) == 1
    assert result.unresolved[0].reason == "delete was not confirmed"


def test_stop_during_the_health_lookup_prevents_the_delete_that_follows(monkeypatch):
    """The health call is a blocking round-trip, so shutdown most often lands DURING it -- the
    widest window in the sweep. A loop-head check alone would clear before the request and still
    delete after it returns, so the stop must be re-read at the destructive boundary itself."""
    runpod_resources._idle_since.clear()
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "list_endpoints_by_key",
        lambda **_kwargs: ({"fpA": [{"name": "live-flash-ep1", "id": "ep-1"}]}, []),
    )
    stopping = []

    def health_then_stop(*_a, **_k):
        stopping.append(True)  # shutdown arrives while this request is in flight
        return _idle_health()

    monkeypatch.setattr(
        runpod_resources.runpod_api, "endpoint_health_for_fingerprint", health_then_stop
    )
    deletes: list[object] = []
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "_delete_endpoint_for_fingerprint_outcome",
        lambda *a, **k: deletes.append(a) or DestructiveOperationOutcome.DELETED,
    )
    monkeypatch.setattr(runpod_resources.logger, "info", lambda *a, **k: None)

    result = runpod_resources._sweep_idle_flash_endpoints(
        protected=set(),
        min_idle_s=0.0,
        known={"flash-ep1"},
        should_stop=lambda: bool(stopping),
    )

    assert not deletes  # the stop landed mid-request; nothing may be destroyed after it
    assert result.deleted_count == 0
    assert result.halted  # and the halt is retained, not reported as a clean sweep
    assert not result.unresolved  # the endpoint itself produced no evidence: it was never visited
    assert result.unresolved_count == 1


def test_a_halted_sweep_is_distinguishable_from_a_complete_one(monkeypatch):
    """A halt leaves selected inventory unvisited. Without a sweep-level halt flag the result is
    byte-identical to a complete sweep that simply found one endpoint to reap, so the server's
    warning path could never tell an interrupted cleanup from a finished one."""
    runpod_resources._idle_since.clear()
    endpoints = [{"name": f"live-flash-ep{n}", "id": f"ep-{n}"} for n in (1, 2, 3)]
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "list_endpoints_by_key",
        lambda **_kwargs: ({"fpA": endpoints}, []),
    )
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *a, **k: _idle_health(),
    )
    deletes: list[object] = []
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "_delete_endpoint_for_fingerprint_outcome",
        lambda *a, **k: deletes.append(a) or DestructiveOperationOutcome.DELETED,
    )
    monkeypatch.setattr(runpod_resources.logger, "info", lambda *a, **k: None)

    result = runpod_resources._sweep_idle_flash_endpoints(
        protected=set(),
        min_idle_s=0.0,
        known={"flash-ep1", "flash-ep2", "flash-ep3"},
        should_stop=lambda: len(deletes) >= 1,
    )

    assert result.deleted_count == 1
    assert result.halted
    # the halt is a property of the SWEEP, not of any endpoint: no endpoint-shaped record is
    # fabricated for inventory that was simply never reached.
    assert not result.unresolved
    # but it still counts as unresolved, so the server's existing warning path keeps firing.
    assert result.unresolved_count == 1


def test_halted_sweep_preserves_grace_for_endpoints_it_never_visited(monkeypatch):
    """A halt leaves the rest of the inventory unvisited, so their absence from ``still_idle``
    means "not reached", not "no longer idle". Pruning there would discard accumulated grace and
    restart the 15-minute idle window on the next boot, silently extending every orphan's life."""
    runpod_resources._idle_since.clear()
    endpoints = [{"name": f"live-flash-ep{n}", "id": f"ep-{n}"} for n in (1, 2, 3)]
    # ep-2 and ep-3 have already accrued grace from earlier cycles.
    runpod_resources._idle_since["ep-2"] = (1.0, "fpA")
    runpod_resources._idle_since["ep-3"] = (1.0, "fpA")
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "list_endpoints_by_key",
        lambda **_kwargs: ({"fpA": endpoints}, []),
    )
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *a, **k: _idle_health(),
    )
    deletes: list[object] = []
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "_delete_endpoint_for_fingerprint_outcome",
        lambda *a, **k: deletes.append(a) or DestructiveOperationOutcome.DELETED,
    )
    monkeypatch.setattr(runpod_resources.logger, "info", lambda *a, **k: None)

    runpod_resources._sweep_idle_flash_endpoints(
        protected=set(),
        min_idle_s=0.0,
        known={"flash-ep1", "flash-ep2", "flash-ep3"},
        should_stop=lambda: len(deletes) >= 1,
    )

    # unvisited endpoints keep their accrued grace; a naive prune would have dropped both
    assert runpod_resources._idle_since.get("ep-2") == (1.0, "fpA")
    assert runpod_resources._idle_since.get("ep-3") == (1.0, "fpA")


def test_a_stop_raised_inside_the_inventory_listing_is_a_halt_not_a_dead_pool(monkeypatch):
    """The listing's own retries allow three 30s attempts per pool key. When the stop reaches the
    transport the call raises, and reading that raise as ``inventory_unavailable`` would blame a
    perfectly healthy pool for a shutdown the server itself requested -- and would keep the reaper
    logging a provider warning on every clean stop."""
    runpod_resources._idle_since.clear()

    def fake_list(**kwargs):
        stop = kwargs.get("should_stop")
        assert stop is not None
        stop()  # the lifespan's stop lands while the listing's retries are in flight
        raise RuntimeError("listing ended by the stop")

    monkeypatch.setattr(runpod_resources.runpod_api, "list_endpoints_by_key", fake_list)
    monkeypatch.setattr(runpod_resources.logger, "info", lambda *a, **k: None)

    result = runpod_resources._sweep_idle_flash_endpoints(
        protected=set(), min_idle_s=0.0, known=set(), should_stop=lambda: True
    )

    assert result.halted
    assert not result.inventory_unavailable  # the pool was never observed to be unreachable
    assert result.unresolved_count == 1


def test_a_partial_listing_under_a_stop_is_not_a_complete_inventory(monkeypatch):
    """``list_endpoints_by_key`` breaks its per-key waterfall on the stop, so it can RETURN
    normally while covering only the keys it reached. Trusting that as the whole pool would report
    a clean sweep over accounts that were never listed at all."""
    runpod_resources._idle_since.clear()

    def fake_list(**kwargs):
        stop = kwargs.get("should_stop")
        assert stop is not None
        stop()  # the stop landed after key A, so keys B and C were never listed
        return ({"fpA": []}, [])

    monkeypatch.setattr(runpod_resources.runpod_api, "list_endpoints_by_key", fake_list)
    monkeypatch.setattr(runpod_resources.logger, "info", lambda *a, **k: None)

    result = runpod_resources._sweep_idle_flash_endpoints(
        protected=set(), min_idle_s=0.0, known=set(), should_stop=lambda: True
    )

    assert result.halted
    assert not result.deleted_ids
    assert result.unresolved_count == 1


def test_a_stop_inside_the_health_lookup_is_a_halt_not_a_provider_fault(monkeypatch):
    """The health lookup inherits five 30s attempts plus backoffs. When the stop ends it the call
    raises, and the generic handler would file a per-endpoint ``provider operation failed`` issue:
    a fabricated fault record naming a healthy endpoint, on every shutdown mid-sweep."""
    runpod_resources._idle_since.clear()
    endpoints = [{"name": "live-flash-ep1", "id": "ep-1"}]
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "list_endpoints_by_key",
        lambda **_kwargs: ({"fpA": endpoints}, []),
    )

    # the stop must still be clear at the loop head, or the sweep breaks there and the health
    # lookup is never reached -- which would leave the handler below completely untested.
    stopping = False

    def fake_health(*_a, **kwargs):
        nonlocal stopping
        stop = kwargs.get("should_stop")
        assert stop is not None
        stopping = True
        stop()  # the stop lands inside the blocking lookup, which then gives up
        raise RuntimeError("health lookup ended by the stop")

    monkeypatch.setattr(runpod_resources.runpod_api, "endpoint_health_for_fingerprint", fake_health)
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "_delete_endpoint_for_fingerprint_outcome",
        lambda *a, **k: pytest.fail("a halted health lookup must not authorize a delete"),
    )
    monkeypatch.setattr(runpod_resources.logger, "info", lambda *a, **k: None)

    result = runpod_resources._sweep_idle_flash_endpoints(
        protected=set(), min_idle_s=0.0, known={"flash-ep1"}, should_stop=lambda: stopping
    )

    assert stopping, "the health lookup must actually run; a loop-head break proves nothing here"

    assert result.halted
    # no endpoint-shaped fault is invented for an endpoint the stop simply cut short
    assert not result.unresolved
    assert result.unresolved_count == 1
