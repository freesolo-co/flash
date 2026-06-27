"""The control-plane idle-endpoint reaper: it must protect every live run's endpoint and pass the
run-aware protected set + idle grace down to the provider sweep. (Server-side; no fastapi needed.)
"""

from __future__ import annotations

import flash.providers.runpod.jobs as jobs
import flash.server.app as app_mod
from flash.providers.base import canonical_gpu
from flash.providers.runpod.train import _run_suffix, endpoint_name
from flash.runner import RunStatus

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


def test_reap_once_passes_protected_set_and_grace(monkeypatch):
    monkeypatch.setattr(app_mod, "_protected_train_endpoint_names", lambda: {"flash-live"})
    # The reaper also passes the KNOWN set (every run this plane has a record of) so it only reaps
    # this plane's own idle endpoints, never another control plane's between-jobs endpoint.
    monkeypatch.setattr(
        app_mod, "_known_train_endpoint_names", lambda: {"flash-live", "flash-done"}
    )
    captured: dict = {}

    def fake_sweep(protected, min_idle_s=0.0, known=None):
        captured["protected"] = protected
        captured["grace"] = min_idle_s
        captured["known"] = known
        return 3

    monkeypatch.setattr(jobs, "_sweep_idle_flash_endpoints", fake_sweep)
    assert app_mod._reap_idle_endpoints_once(900.0) == 3
    assert captured == {
        "protected": {"flash-live"},
        "grace": 900.0,
        "known": {"flash-live", "flash-done"},
    }


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


# --- the in-lifetime instance orphan sweep (Lambda) ---------------------------------


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
        self.seen_known = None

    def sweep_orphans(self, active_labels=None, known_labels=None):
        if self._raises:
            raise RuntimeError(f"{self.name} api blip")
        self.seen_active = active_labels() if callable(active_labels) else active_labels
        self.seen_known = known_labels() if callable(known_labels) else known_labels
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
    monkeypatch.setattr(app_mod, "_known_run_ids", lambda: {"flash-live", "flash-done"})
    lam = _FakeProvider("lambda", torn=["i-1", "i-2"])
    rp = _FakeProvider("runpod", torn=[])  # no-op for RunPod, still dispatched
    monkeypatch.setattr(
        "flash.providers.configured_providers", lambda: [rp, lam], raising=False
    )

    # 2 lambda + 0 runpod torn down.
    assert app_mod._sweep_orphan_instances_once() == 2
    # Every provider got the SAME live-run protection set AND the same known-run scope.
    assert lam.seen_active == {"flash-live"}
    assert rp.seen_active == {"flash-live"}
    assert lam.seen_known == {"flash-live", "flash-done"}
    assert rp.seen_known == {"flash-live", "flash-done"}


def test_sweep_instances_one_provider_blip_does_not_skip_others(monkeypatch):
    monkeypatch.setattr(app_mod, "_active_run_ids", lambda: set())
    monkeypatch.setattr(app_mod, "_known_run_ids", lambda: set())
    boom = _FakeProvider("lambda", raises=True)
    ok = _FakeProvider("runpod", torn=["vm-1", "vm-2"])
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
        "flash.providers.available_providers", lambda: ("lambda",), raising=False
    )
    assert app_mod._instance_providers_configured() is True


def test_sweep_end_to_end_reaps_orphans_protects_live_run(monkeypatch):
    """End-to-end through the REAL Lambda ``sweep_orphans`` (only the provider REST
    layer is faked): a periodic sweep tears down the provider's leaked instance while the live
    run's instance — named from the SAME run id the server reports as active — is protected.

    Exercises the full path the lifespan loop runs: ``_sweep_orphan_instances_once`` ->
    ``configured_providers`` -> ``LambdaProvider.sweep_orphans`` -> the real
    name<->run matching -> the (faked) terminate call. No ``sweep_orphans`` mock anywhere."""
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs as lambda_jobs
    from flash.runner import RunStatus

    # Two runs THIS plane knows: one live (running), one finished (terminal) whose teardown leaked
    # an instance. Both appear in the registry, so both are in the KNOWN scope; only the live one is
    # in the ACTIVE (protected) set.
    statuses = {
        "flash-live": RunStatus(run_id="flash-live", state="running", spec={}),
        "flash-dead": RunStatus(run_id="flash-dead", state="done", spec={}),
    }
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": r} for r in statuses])
    monkeypatch.setattr(app_mod, "get_status", lambda rid: statuses[rid])
    # Make the instance provider "configured" so the real one is dispatched (RunPod absent here).
    monkeypatch.setattr(
        "flash.providers.available_providers", lambda: ("lambda",), raising=False
    )

    lam_instances = [
        {"id": "i-live", "name": lambda_jobs.instance_label("flash-live", 0, 0)},  # live -> KEEP
        {"id": "i-orphan", "name": lambda_jobs.instance_label("flash-dead", 0, 0)},  # our leak -> kill
        {"id": "i-foreign", "name": "not-ours"},  # non-flash name -> never touch
    ]
    terminated = []
    monkeypatch.setattr(lambda_api, "list_instances", lambda: lam_instances)
    monkeypatch.setattr(
        lambda_api, "terminate_instances", lambda ids: terminated.extend(ids) or list(ids)
    )

    torn = app_mod._sweep_orphan_instances_once()

    assert torn == 1  # one leaked Lambda instance, owned by OUR terminal run
    assert terminated == ["i-orphan"]  # leaked reaped, live + foreign untouched


def test_sweep_spares_other_control_planes_live_instances(monkeypatch):
    """Multi-plane safety: two control planes sharing one Lambda account. This plane must NEVER
    terminate a box belonging to a run it has no record of — that box is another plane's, very
    possibly a LIVE training instance. Before the ``known_labels`` scope, this plane saw the other's
    box, found its run id absent from ITS active set, and reaped it (the planes mutually executed
    each other's live runs every sweep)."""
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs as lambda_jobs
    from flash.runner import RunStatus

    # This plane knows exactly ONE run (live). The other plane's run id is absent from our registry.
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "flash-mine"}])
    monkeypatch.setattr(
        app_mod, "get_status", lambda rid: RunStatus(run_id="flash-mine", state="running", spec={})
    )
    monkeypatch.setattr(
        "flash.providers.available_providers", lambda: ("lambda",), raising=False
    )

    lam_instances = [
        {"id": "i-mine", "name": lambda_jobs.instance_label("flash-mine", 0, 0)},  # ours, live
        # Another control plane's box, named with ITS run id — we have no record of it.
        {"id": "i-theirs", "name": lambda_jobs.instance_label("flash-theirs", 0, 0)},
    ]
    terminated = []
    monkeypatch.setattr(lambda_api, "list_instances", lambda: lam_instances)
    monkeypatch.setattr(
        lambda_api, "terminate_instances", lambda ids: terminated.extend(ids) or list(ids)
    )

    torn = app_mod._sweep_orphan_instances_once()

    assert torn == 0  # nothing reaped
    assert terminated == []  # the other plane's live box is left strictly alone


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


def test_canonical_endpoint_name_strips_sdk_live_prefix():
    """One endpoint, two names: flash stores the bare ``flash-...`` form, the runpod-flash SDK lists
    it as ``live-flash-...``. ``canonical_endpoint_name`` collapses them to the bare form so every
    comparison site uses one name. Idempotent; a non-``live-`` name passes through unchanged."""
    assert jobs.canonical_endpoint_name("live-flash-5090-abc") == "flash-5090-abc"
    assert jobs.canonical_endpoint_name("flash-5090-abc") == "flash-5090-abc"
    assert jobs.canonical_endpoint_name(jobs.canonical_endpoint_name("live-flash-x")) == "flash-x"
    assert jobs.canonical_endpoint_name("") == ""


def test_sweep_reaps_responsive_account_when_one_pool_key_fails(monkeypatch):
    """One pool key fails to list this cycle; the responding account's idle orphan is still reaped,
    using that account's OWN key — and the failure is surfaced at WARNING (not a silent DEBUG)."""
    jobs._idle_since.clear()
    orphan = {"id": "ep-b1", "name": "live-flash-5090-orphan"}
    # fpA failed to list; fpB returned the orphan. (Accounts are identified by non-secret
    # fingerprints, never the raw key.)
    monkeypatch.setattr(
        jobs.runpod_api, "list_endpoints_by_key", lambda: ({"fpB": [orphan]}, ["fpA"])
    )
    health_calls = []

    def health(eid, fp):
        health_calls.append((eid, fp))
        return _idle_health()

    deletes = []
    monkeypatch.setattr(jobs.runpod_api, "endpoint_health_for_fingerprint", health)
    monkeypatch.setattr(
        jobs.runpod_api,
        "delete_endpoint_for_fingerprint",
        lambda eid, fp: deletes.append((eid, fp)) or True,
    )
    warnings = []
    monkeypatch.setattr(jobs.logger, "warning", lambda *a, **k: warnings.append(a))

    # min_idle_s=0 -> a first idle observation is immediately reapable.
    deleted = jobs._sweep_idle_flash_endpoints(protected=set(), min_idle_s=0.0)

    assert deleted == 1
    assert health_calls == [("ep-b1", "fpB")]  # account-scoped: queried with the OWNING fingerprint
    assert deletes == [("ep-b1", "fpB")]  # ...and deleted with it, no failover waterfall
    assert warnings, "a failed pool account must be surfaced at WARNING, not swallowed at DEBUG"


def test_sweep_skips_endpoints_outside_known_scope(monkeypatch):
    """Multi-plane safety for RunPod: with a ``known`` scope, the reaper deletes only idle endpoints
    THIS plane has a record of. An idle endpoint owned by another control plane on the same account
    (its name absent from ``known``) is left alone, even though it is idle and unprotected."""
    jobs._idle_since.clear()
    mine = {"id": "ep-mine", "name": "live-flash-mine-idle"}
    theirs = {"id": "ep-theirs", "name": "live-flash-theirs-idle"}
    monkeypatch.setattr(
        jobs.runpod_api, "list_endpoints_by_key", lambda: ({"fpA": [mine, theirs]}, [])
    )
    monkeypatch.setattr(
        jobs.runpod_api, "endpoint_health_for_fingerprint", lambda eid, fp: _idle_health()
    )
    deletes = []
    monkeypatch.setattr(
        jobs.runpod_api,
        "delete_endpoint_for_fingerprint",
        lambda eid, fp: deletes.append(eid) or True,
    )

    # known carries only OUR endpoint name (bare form); the reaper compares both bare and live- forms.
    deleted = jobs._sweep_idle_flash_endpoints(
        protected=set(), min_idle_s=0.0, known={"flash-mine-idle"}
    )

    assert deleted == 1
    assert deletes == ["ep-mine"]  # only ours; the other plane's idle endpoint is untouched


def test_sweep_preserves_grace_for_unlisted_account(monkeypatch):
    """A partial outage must not reset the idle-grace clock for the account it couldn't list — else
    a flaky account's orphan restarts its 15-min grace every sweep and never ages out."""
    jobs._idle_since.clear()
    jobs._idle_since["ep-a1"] = (1.0, "fpA")  # orphan on account A, observed idle long ago
    # This cycle account A fails to list; account B responds with nothing.
    monkeypatch.setattr(jobs.runpod_api, "list_endpoints_by_key", lambda: ({"fpB": []}, ["fpA"]))
    monkeypatch.setattr(
        jobs.runpod_api, "endpoint_health_for_fingerprint", lambda eid, fp: _idle_health()
    )
    monkeypatch.setattr(jobs.runpod_api, "delete_endpoint_for_fingerprint", lambda eid, fp: True)
    monkeypatch.setattr(jobs.logger, "warning", lambda *a, **k: None)

    deleted = jobs._sweep_idle_flash_endpoints(protected=set(), min_idle_s=900.0)

    assert deleted == 0
    # grace timer (owned by the FAILED account) SURVIVED the partial outage
    assert jobs._idle_since.get("ep-a1") == (1.0, "fpA")


def test_sweep_partial_view_prunes_vanished_timer_for_responsive_account(monkeypatch):
    """During a partial outage, a stale grace timer whose OWNING account responded this cycle must
    still be pruned — the endpoint genuinely vanished from a healthy account. (The earlier
    listed-ids-only prune leaked it: any one failing account kept every responsive account's vanished
    timers alive forever.) A timer owned by the FAILED account is still preserved."""
    jobs._idle_since.clear()
    jobs._idle_since["gone-b"] = (1.0, "fpB")  # vanished from account B, which responds this cycle
    jobs._idle_since["stay-a"] = (1.0, "fpA")  # owned by account A, which fails this cycle
    # B responds (no longer lists gone-b); A fails.
    monkeypatch.setattr(jobs.runpod_api, "list_endpoints_by_key", lambda: ({"fpB": []}, ["fpA"]))
    monkeypatch.setattr(jobs.logger, "warning", lambda *a, **k: None)

    deleted = jobs._sweep_idle_flash_endpoints(protected=set(), min_idle_s=900.0)

    assert deleted == 0
    assert "gone-b" not in jobs._idle_since  # responsive account's vanished timer pruned (the fix)
    assert jobs._idle_since.get("stay-a") == (1.0, "fpA")  # failed account's timer preserved


def test_sweep_full_view_prunes_vanished_grace_timer(monkeypatch):
    """With a complete fleet view (no failed account), a grace timer for an endpoint that is no
    longer present is pruned — the original behavior, unchanged."""
    jobs._idle_since.clear()
    jobs._idle_since["ghost"] = (1.0, "fpA")  # endpoint that has since vanished
    monkeypatch.setattr(jobs.runpod_api, "list_endpoints_by_key", lambda: ({"fpA": []}, []))
    monkeypatch.setattr(jobs.logger, "warning", lambda *a, **k: None)

    deleted = jobs._sweep_idle_flash_endpoints(protected=set(), min_idle_s=900.0)

    assert deleted == 0
    assert "ghost" not in jobs._idle_since  # full view -> stale timer pruned
