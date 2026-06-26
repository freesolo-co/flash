"""The control-plane idle-endpoint reaper: it must protect every live run's endpoint and pass the
run-aware protected set + idle grace down to the provider sweep. (Server-side; no fastapi needed.)
"""

from __future__ import annotations

import flash.providers.runpod.jobs as jobs
import flash.server.app as app_mod
from flash.providers.base import canonical_gpu
from flash.providers.runpod.train import _run_suffix, endpoint_name
from flash.runner import RunStatus


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

    # active run: the persisted handle name AND the spec-derived name, both registered forms.
    assert {"flash-5090-handle", "live-flash-5090-handle"} <= names
    active_derived = _derived("RTX 5090", "flash-active")
    assert {active_derived, f"live-{active_derived}"} <= names
    # provisioning run: protected by its spec-derived name even with no handle.
    assert _derived("A100", "flash-prov") in names
    # terminal run: not protected (its endpoint is reapable once idle).
    assert _derived("RTX 5090", "flash-done") not in names


def test_reap_once_passes_protected_set_and_grace(monkeypatch):
    monkeypatch.setattr(app_mod, "_protected_train_endpoint_names", lambda: {"flash-live"})
    captured: dict = {}

    def fake_sweep(protected, min_idle_s=0.0):
        captured["protected"] = protected
        captured["grace"] = min_idle_s
        return 3

    monkeypatch.setattr(jobs, "_sweep_idle_flash_endpoints", fake_sweep)
    assert app_mod._reap_idle_endpoints_once(900.0) == 3
    assert captured == {"protected": {"flash-live"}, "grace": 900.0}


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


# --- the in-lifetime instance orphan sweep (Lambda/Hyperstack) ---------------------------------


class _FakeProvider:
    """A configured provider whose ``sweep_orphans`` records the active set it was handed and
    returns a fixed list of torn-down ids (or raises, to model an API blip). It RESOLVES a callable
    ``active_labels`` exactly as the real instance providers do (after they list), so the recorded
    set reflects the post-listing resolution the periodic sweep relies on."""

    def __init__(self, name, torn=(), raises=False):
        self.name = name
        self._torn = list(torn)
        self._raises = raises
        self.seen_active = None

    def sweep_orphans(self, active_labels=None):
        if self._raises:
            raise RuntimeError(f"{self.name} api blip")
        self.seen_active = active_labels() if callable(active_labels) else active_labels
        return self._torn


def test_active_run_ids_covers_live_runs_only(monkeypatch):
    rows = [
        {"run_id": r}
        for r in ("flash-run", "flash-prov", "flash-done", "flash-failed", "flash-deployed")
    ]
    statuses = {
        "flash-run": RunStatus(run_id="flash-run", state="running", spec={}),
        # provisioning: no handle yet, but its instance may already be launching -> must be protected
        "flash-prov": RunStatus(run_id="flash-prov", state="provisioning", spec={}),
        "flash-done": RunStatus(run_id="flash-done", state="done", spec={}),
        "flash-failed": RunStatus(run_id="flash-failed", state="failed", spec={}),
        # deployed: reachable only from `done`, so the seed loop's finally already tore the training
        # instance down -> the run owns no training worker and must NOT be protected (else a leaked
        # instance under its prefix would be shielded from the sweep forever).
        "flash-deployed": RunStatus(run_id="flash-deployed", state="deployed", spec={}),
    }
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: rows)
    monkeypatch.setattr(app_mod, "get_status", lambda rid: statuses[rid])

    assert app_mod._active_run_ids() == {"flash-run", "flash-prov"}


def test_active_run_ids_skips_unreadable_run(monkeypatch):
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "gone"}, {"run_id": "live"}])

    def get_status(rid):
        if rid == "gone":
            raise FileNotFoundError(rid)
        return RunStatus(run_id="live", state="running", spec={})

    monkeypatch.setattr(app_mod, "get_status", get_status)
    assert app_mod._active_run_ids() == {"live"}


def test_sweep_instances_dispatches_active_set_and_sums(monkeypatch):
    monkeypatch.setattr(app_mod, "_active_run_ids", lambda: {"flash-live"})
    lam = _FakeProvider("lambda", torn=["i-1", "i-2"])
    hyp = _FakeProvider("hyperstack", torn=["vm-9"])
    rp = _FakeProvider("runpod", torn=[])  # no-op for RunPod, still dispatched
    monkeypatch.setattr(
        "flash.providers.configured_providers", lambda: [rp, lam, hyp], raising=False
    )

    # 2 lambda + 1 hyperstack + 0 runpod torn down.
    assert app_mod._sweep_orphan_instances_once() == 3
    # Every provider got the SAME live-run protection set.
    assert lam.seen_active == {"flash-live"}
    assert hyp.seen_active == {"flash-live"}
    assert rp.seen_active == {"flash-live"}


def test_sweep_instances_one_provider_blip_does_not_skip_others(monkeypatch):
    monkeypatch.setattr(app_mod, "_active_run_ids", lambda: set())
    boom = _FakeProvider("lambda", raises=True)
    ok = _FakeProvider("hyperstack", torn=["vm-1", "vm-2"])
    monkeypatch.setattr(
        "flash.providers.configured_providers", lambda: [boom, ok], raising=False
    )

    # The raising provider is swallowed; the other still reaps.
    assert app_mod._sweep_orphan_instances_once() == 2


def test_instance_providers_configured_gating(monkeypatch):
    monkeypatch.setattr(
        "flash.providers.available_providers", lambda: ("runpod",), raising=False
    )
    assert app_mod._instance_providers_configured() is False

    monkeypatch.setattr(
        "flash.providers.available_providers", lambda: ("runpod", "lambda"), raising=False
    )
    assert app_mod._instance_providers_configured() is True

    monkeypatch.setattr(
        "flash.providers.available_providers", lambda: ("hyperstack",), raising=False
    )
    assert app_mod._instance_providers_configured() is True


def test_sweep_end_to_end_reaps_orphans_protects_live_run(monkeypatch):
    """End-to-end through the REAL Lambda + Hyperstack ``sweep_orphans`` (only the provider REST
    layer is faked): a periodic sweep tears down each provider's leaked instance while the live
    run's instance — named from the SAME run id the server reports as active — is protected.

    Exercises the full path the lifespan loop runs: ``_sweep_orphan_instances_once`` ->
    ``configured_providers`` -> ``LambdaProvider/HyperstackProvider.sweep_orphans`` -> the real
    name<->run matching -> the (faked) terminate call. No ``sweep_orphans`` mock anywhere."""
    from flash.providers.hyperstack import api as hs_api
    from flash.providers.hyperstack import jobs as hs_jobs
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs as lambda_jobs
    from flash.runner import RunStatus

    # One live run; its instance on each provider is named from this exact run id.
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "flash-live"}])
    monkeypatch.setattr(
        app_mod, "get_status", lambda rid: RunStatus(run_id="flash-live", state="running", spec={})
    )
    # Make both instance providers "configured" so the real ones are dispatched (RunPod absent here).
    monkeypatch.setattr(
        "flash.providers.available_providers", lambda: ("lambda", "hyperstack"), raising=False
    )

    lam_instances = [
        {"id": "i-live", "name": lambda_jobs.instance_label("flash-live", 0, 0)},  # live -> KEEP
        {"id": "i-orphan", "name": lambda_jobs.instance_label("flash-dead", 0, 0)},  # leaked -> kill
        {"id": "i-foreign", "name": "not-ours"},  # never touch
    ]
    hs_vms = [
        {"id": "vm-live", "name": hs_jobs.instance_label("flash-live", 0, 0)},  # live -> KEEP
        {"id": "vm-orphan", "name": hs_jobs.instance_label("flash-gone", 0, 0)},  # leaked -> kill
    ]
    terminated, deleted = [], []
    monkeypatch.setattr(lambda_api, "list_instances", lambda: lam_instances)
    monkeypatch.setattr(
        lambda_api, "terminate_instances", lambda ids: terminated.extend(ids) or list(ids)
    )
    monkeypatch.setattr(hs_api, "list_vms", lambda: hs_vms)
    monkeypatch.setattr(hs_api, "delete_vms", lambda ids: deleted.extend(ids) or list(ids))

    torn = app_mod._sweep_orphan_instances_once()

    assert torn == 2  # one Lambda instance + one Hyperstack VM
    assert terminated == ["i-orphan"]  # leaked reaped, live + foreign untouched
    assert deleted == ["vm-orphan"]


def test_sweep_resolves_active_labels_after_listing(monkeypatch):
    """Launch-race fix: when ``active_labels`` is a callable, the real provider resolves it AFTER it
    lists instances. A run that only enters the live set concurrently with the sweep therefore still
    shields its fresh worker, instead of having it reaped as a phantom orphan."""
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    events = []
    fresh = jobs.instance_label("flash-fresh", 0, 0)
    orphan = jobs.instance_label("flash-old", 0, 0)

    def fake_list():
        events.append("list")
        return [{"id": "i-fresh", "name": fresh}, {"id": "i-orphan", "name": orphan}]

    def active_fn():
        events.append("active")
        return {"flash-fresh"}  # the fresh run is live only at RESOLUTION time (post-listing)

    terminated = []
    monkeypatch.setattr(lambda_api, "list_instances", fake_list)
    monkeypatch.setattr(
        lambda_api, "terminate_instances", lambda ids: terminated.extend(ids) or list(ids)
    )

    out = jobs.sweep_orphans(active_labels=active_fn)

    assert events == ["list", "active"]  # protection set resolved AFTER the instance list
    assert out == ["i-orphan"]  # fresh worker protected, orphan reaped
    assert terminated == ["i-orphan"]


def test_sweep_skips_when_active_set_resolution_raises(monkeypatch):
    """If resolving a callable ``active_labels`` raises (e.g. a db/status read error), the sweep must
    SKIP (return []) — never fall through to an empty protection set, which would treat every live
    run's instance as an orphan and reap it. Honors the 'never raises' contract."""
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    terminated = []
    monkeypatch.setattr(
        lambda_api,
        "list_instances",
        lambda: [{"id": "i-live", "name": jobs.instance_label("flash-live", 0, 0)}],
    )
    monkeypatch.setattr(
        lambda_api, "terminate_instances", lambda ids: terminated.extend(ids) or list(ids)
    )

    def boom():
        raise RuntimeError("db read failed")

    out = jobs.sweep_orphans(active_labels=boom)

    assert out == []  # skipped, did NOT raise
    assert terminated == []  # and crucially did NOT reap the live instance


# --- the RunPod idle sweep across a multi-account key pool --------------------------------------
# Regression cover for the orphan pile-up: the sweep used the all-or-nothing ``list_endpoints``, so
# one unhealthy pool key (rejected/expired/rate-limited) aborted the WHOLE sweep and idle orphans on
# every OTHER account survived indefinitely. The sweep now lists per account and reaps what responds.


def _idle_health():
    """A warm-idle endpoint with no work — reapable under reap_warm=True."""
    return {"workers": {"ready": 1, "idle": 1}, "jobs": {"inQueue": 0, "inProgress": 0}}


def test_sweep_reaps_responsive_account_when_one_pool_key_fails(monkeypatch):
    """One pool key fails to list this cycle; the responding account's idle orphan is still reaped,
    using that account's OWN key — and the failure is surfaced at WARNING (not a silent DEBUG)."""
    jobs._idle_since.clear()
    orphan = {"id": "ep-b1", "name": "live-flash-5090-orphan"}
    # keyA failed to list; keyB returned the orphan.
    monkeypatch.setattr(
        jobs.runpod_api, "list_endpoints_by_key", lambda: ({"keyB": [orphan]}, ["keyA"])
    )
    health_calls = []

    def health(eid, key):
        health_calls.append((eid, key))
        return _idle_health()

    deletes = []
    monkeypatch.setattr(jobs.runpod_api, "endpoint_health_for_key", health)
    monkeypatch.setattr(
        jobs.runpod_api, "delete_endpoint_for_key", lambda eid, key: deletes.append((eid, key)) or True
    )
    warnings = []
    monkeypatch.setattr(jobs.logger, "warning", lambda *a, **k: warnings.append(a))

    # min_idle_s=0 -> a first idle observation is immediately reapable.
    deleted = jobs._sweep_idle_flash_endpoints(protected=set(), min_idle_s=0.0)

    assert deleted == 1
    assert health_calls == [("ep-b1", "keyB")]  # account-scoped: queried with the OWNING key
    assert deletes == [("ep-b1", "keyB")]  # ...and deleted with it, no failover waterfall
    assert warnings, "a failed pool account must be surfaced at WARNING, not swallowed at DEBUG"


def test_sweep_preserves_grace_for_unlisted_account(monkeypatch):
    """A partial outage must not reset the idle-grace clock for the account it couldn't list — else
    a flaky account's orphan restarts its 15-min grace every sweep and never ages out."""
    jobs._idle_since.clear()
    jobs._idle_since["ep-a1"] = 1.0  # orphan on account A, observed idle long ago
    # This cycle account A fails to list; account B responds with nothing.
    monkeypatch.setattr(jobs.runpod_api, "list_endpoints_by_key", lambda: ({"keyB": []}, ["keyA"]))
    monkeypatch.setattr(jobs.runpod_api, "endpoint_health_for_key", lambda eid, key: _idle_health())
    monkeypatch.setattr(jobs.runpod_api, "delete_endpoint_for_key", lambda eid, key: True)
    monkeypatch.setattr(jobs.logger, "warning", lambda *a, **k: None)

    deleted = jobs._sweep_idle_flash_endpoints(protected=set(), min_idle_s=900.0)

    assert deleted == 0
    assert jobs._idle_since.get("ep-a1") == 1.0  # grace timer SURVIVED the partial outage


def test_sweep_full_view_prunes_vanished_grace_timer(monkeypatch):
    """With a complete fleet view (no failed account), a grace timer for an endpoint that is no
    longer present is pruned — the original behavior, unchanged."""
    jobs._idle_since.clear()
    jobs._idle_since["ghost"] = 1.0  # endpoint that has since vanished
    monkeypatch.setattr(jobs.runpod_api, "list_endpoints_by_key", lambda: ({"keyA": []}, []))
    monkeypatch.setattr(jobs.logger, "warning", lambda *a, **k: None)

    deleted = jobs._sweep_idle_flash_endpoints(protected=set(), min_idle_s=900.0)

    assert deleted == 0
    assert "ghost" not in jobs._idle_since  # full view -> stale timer pruned
