"""Instance-provider orphan sweep selection, dispatch, and stop behavior."""

from __future__ import annotations

import flash.server.asgi.app as app_mod
from flash.runner.lifecycle.state import RunStatus


class _FakeProvider:
    """A configured provider whose ``sweep_orphans`` records the active set it was handed and
    returns a fixed list of torn-down ids (or raises, to model an API blip). It RESOLVES a callable
    ``active_labels`` exactly as the real instance providers do (after they list), so the recorded
    set reflects the post-listing resolution the periodic sweep relies on."""

    def __init__(self, name, torn=(), raises=False, halted=False):
        from flash.providers.core.capabilities import ProviderCapabilities

        self.name = name
        self._torn = list(torn)
        self._raises = raises
        self._halted = halted
        self.seen_active = None
        self.seen_known = None
        self.seen_should_stop = None
        self.capabilities = ProviderCapabilities(False, True, None, self._sweep_orphans)

    def _sweep_orphans(self, active_labels=None, known_labels=None, should_stop=None):
        from flash.providers.core.capabilities import CleanupOutcome, CleanupResult

        self.seen_should_stop = should_stop
        if self._raises:
            raise RuntimeError(f"{self.name} api blip")
        self.seen_active = active_labels() if callable(active_labels) else active_labels
        self.seen_known = known_labels() if callable(known_labels) else known_labels
        if self._halted:
            outcome = CleanupOutcome.UNCONFIRMED if self._torn else CleanupOutcome.RETRYABLE
        else:
            outcome = CleanupOutcome.DELETED if self._torn else CleanupOutcome.ABSENT
        return CleanupResult(
            outcome,
            confirmed_deleted_ids=tuple(self._torn),
            halted=self._halted,
        )


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
    rp = _FakeProvider("runpod", torn=[])
    monkeypatch.setattr(
        "flash.providers.core.registry.configured_providers", lambda: [rp, lam], raising=False
    )

    # 2 lambda + 0 runpod torn down.
    assert app_mod._sweep_orphan_instances_once().deleted_count == 2
    # Every provider got the SAME live-run protection set AND the same known-run scope.
    assert lam.seen_active == {"flash-live"}
    assert rp.seen_active == {"flash-live"}
    assert lam.seen_known == {"flash-live", "flash-done"}
    assert rp.seen_known == {"flash-live", "flash-done"}


def test_sweep_instances_one_provider_blip_warns_and_does_not_skip_others(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(
        app_mod._log,
        "warning",
        lambda message, *args, **_kwargs: warnings.append(message % args),
    )
    monkeypatch.setattr(app_mod, "_active_run_ids", lambda: set())
    monkeypatch.setattr(app_mod, "_known_run_ids", lambda: set())
    boom = _FakeProvider("lambda", raises=True)
    ok = _FakeProvider("runpod", torn=["vm-1", "vm-2"])
    monkeypatch.setattr(
        "flash.providers.core.registry.configured_providers", lambda: [boom, ok], raising=False
    )

    # The dispatcher converts the exception to RETRYABLE; the aggregate must still expose it.
    assert app_mod._sweep_orphan_instances_once().deleted_count == 2
    assert warnings == [
        "instance orphan sweep was inconclusive for provider 'lambda'; retrying next cycle"
    ]


def test_instance_providers_configured_gating(monkeypatch):
    monkeypatch.setattr(
        "flash.providers.core.registry.available_providers", lambda: ("runpod",), raising=False
    )
    assert app_mod._instance_providers_configured() is False

    monkeypatch.setattr(
        "flash.providers.core.registry.available_providers",
        lambda: ("runpod", "lambda"),
        raising=False,
    )
    assert app_mod._instance_providers_configured() is True

    monkeypatch.setattr(
        "flash.providers.core.registry.available_providers", lambda: ("lambda",), raising=False
    )
    assert app_mod._instance_providers_configured() is True


def test_sweep_end_to_end_reaps_orphans_protects_live_run(monkeypatch):
    """End-to-end through the REAL Lambda ``sweep_orphans`` (only the provider REST
    layer is faked): a periodic sweep tears down the provider's leaked instance while the live
    run's instance — named from the SAME run id the server reports as active — is protected.

    Exercises the full path the lifespan loop runs: ``_sweep_orphan_instances_once`` ->
    ``configured_providers`` -> ``LambdaProvider.sweep_orphans`` -> the real
    name<->run matching -> the (faked) terminate call. No ``sweep_orphans`` mock anywhere."""
    from flash.providers.lambda_ import jobs as lambda_jobs
    from flash.providers.lambda_.client import api as lambda_api
    from flash.runner.lifecycle.state import RunStatus

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
        "flash.providers.core.registry.available_providers", lambda: ("lambda",), raising=False
    )

    lam_instances = [
        {"id": "i-live", "name": lambda_jobs.instance_label("flash-live", 0, 0)},  # live -> KEEP
        {
            "id": "i-orphan",
            "name": lambda_jobs.instance_label("flash-dead", 0, 0),
        },  # our leak -> kill
        {"id": "i-foreign", "name": "not-ours"},  # non-flash name -> never touch
    ]
    terminated = []
    monkeypatch.setattr(lambda_api, "list_instances", lambda **_: lam_instances)
    monkeypatch.setattr(
        lambda_api, "terminate_instances", lambda ids, **_: terminated.extend(ids) or list(ids)
    )

    torn = app_mod._sweep_orphan_instances_once()

    assert torn.deleted_count == 1  # one leaked Lambda instance, owned by OUR terminal run
    assert terminated == ["i-orphan"]  # leaked reaped, live + foreign untouched


def test_sweep_spares_other_control_planes_live_instances(monkeypatch):
    """Multi-plane safety: two control planes sharing one Lambda account. This plane must NEVER
    terminate a box belonging to a run it has no record of — that box is another plane's, very
    possibly a LIVE training instance. Before the ``known_labels`` scope, this plane saw the other's
    box, found its run id absent from ITS active set, and reaped it (the planes mutually executed
    each other's live runs every sweep)."""
    from flash.providers.lambda_ import jobs as lambda_jobs
    from flash.providers.lambda_.client import api as lambda_api
    from flash.runner.lifecycle.state import RunStatus

    # This plane knows exactly ONE run (live). The other plane's run id is absent from our registry.
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "flash-mine"}])
    monkeypatch.setattr(
        app_mod, "get_status", lambda rid: RunStatus(run_id="flash-mine", state="running", spec={})
    )
    monkeypatch.setattr(
        "flash.providers.core.registry.available_providers", lambda: ("lambda",), raising=False
    )

    lam_instances = [
        {"id": "i-mine", "name": lambda_jobs.instance_label("flash-mine", 0, 0)},  # ours, live
        # Another control plane's box, named with ITS run id — we have no record of it.
        {"id": "i-theirs", "name": lambda_jobs.instance_label("flash-theirs", 0, 0)},
    ]
    terminated = []
    monkeypatch.setattr(lambda_api, "list_instances", lambda **_: lam_instances)
    monkeypatch.setattr(
        lambda_api, "terminate_instances", lambda ids, **_: terminated.extend(ids) or list(ids)
    )

    torn = app_mod._sweep_orphan_instances_once()

    assert torn.deleted_count == 0  # nothing reaped
    assert terminated == []  # the other plane's live box is left strictly alone


def test_sweep_resolves_active_labels_after_listing(monkeypatch):
    """Launch-race fix: when ``active_labels`` is a callable, the real provider resolves it AFTER it
    lists instances. A run that only enters the live set concurrently with the sweep therefore still
    shields its fresh worker, instead of having it reaped as a phantom orphan."""
    from flash.providers.lambda_ import jobs
    from flash.providers.lambda_.client import api as lambda_api

    events = []
    fresh = jobs.instance_label("flash-fresh", 0, 0)
    orphan = jobs.instance_label("flash-old", 0, 0)

    def fake_list(**_):
        events.append("list")
        return [{"id": "i-fresh", "name": fresh}, {"id": "i-orphan", "name": orphan}]

    def active_fn():
        events.append("active")
        return {"flash-fresh"}  # the fresh run is live only at RESOLUTION time (post-listing)

    terminated = []
    monkeypatch.setattr(lambda_api, "list_instances", fake_list)
    monkeypatch.setattr(
        lambda_api, "terminate_instances", lambda ids, **_: terminated.extend(ids) or list(ids)
    )

    out = jobs.sweep_orphans(active_labels=active_fn)

    assert events == ["list", "active"]  # protection set resolved AFTER the instance list
    assert out.confirmed_deleted_ids == ("i-orphan",)  # fresh worker protected, orphan reaped
    assert terminated == ["i-orphan"]


def test_sweep_skips_when_active_set_resolution_raises(monkeypatch):
    """If resolving a callable ``active_labels`` raises (e.g. a db/status read error), the sweep must
    SKIP (return []) — never fall through to an empty protection set, which would treat every live
    run's instance as an orphan and reap it. Honors the 'never raises' contract."""
    from flash.providers.lambda_ import jobs
    from flash.providers.lambda_.client import api as lambda_api

    terminated = []
    monkeypatch.setattr(
        lambda_api,
        "list_instances",
        lambda: [{"id": "i-live", "name": jobs.instance_label("flash-live", 0, 0)}],
    )
    monkeypatch.setattr(
        lambda_api, "terminate_instances", lambda ids, **_: terminated.extend(ids) or list(ids)
    )

    def boom():
        raise RuntimeError("db read failed")

    out = jobs.sweep_orphans(active_labels=boom)

    from flash.providers.core.capabilities import CleanupOutcome

    assert out.outcome is CleanupOutcome.RETRYABLE
    assert terminated == []  # and crucially did NOT reap the live instance


def test_sweep_instances_forwards_the_stop_callback_to_every_provider(monkeypatch):
    """The callback must actually REACH the provider. The capabilities dispatcher wraps the
    callback in a bare ``except Exception`` that turns a stale two-arg signature into a
    RETRYABLE result, so "nothing raised" proves nothing on its own."""
    monkeypatch.setattr(app_mod, "_active_run_ids", lambda: set())
    monkeypatch.setattr(app_mod, "_known_run_ids", lambda: set())
    lam = _FakeProvider("lambda", torn=["i-1"])
    rp = _FakeProvider("runpod", torn=[])
    monkeypatch.setattr(
        "flash.providers.core.registry.configured_providers", lambda: [rp, lam], raising=False
    )

    def never_stop() -> bool:
        return False

    assert app_mod._sweep_orphan_instances_once(never_stop).deleted_count == 1
    assert lam.seen_should_stop is never_stop
    assert rp.seen_should_stop is never_stop


def test_sweep_instances_halts_between_providers_on_stop(monkeypatch):
    """Without a between-providers check, a stop arriving during provider A's sweep would still
    let provider B's entire sweep start from scratch."""
    monkeypatch.setattr(app_mod, "_active_run_ids", lambda: set())
    monkeypatch.setattr(app_mod, "_known_run_ids", lambda: set())
    first = _FakeProvider("runpod", torn=["vm-1"])
    second = _FakeProvider("lambda", torn=["i-9"])
    monkeypatch.setattr(
        "flash.providers.core.registry.configured_providers",
        lambda: [first, second],
        raising=False,
    )

    torn = app_mod._sweep_orphan_instances_once(lambda: first.seen_should_stop is not None)

    assert torn.deleted_count == 1  # only the first provider ran
    assert second.seen_should_stop is None  # second was never dispatched


def test_sweep_instances_stops_dispatch_after_provider_reports_halt(monkeypatch):
    monkeypatch.setattr(app_mod, "_active_run_ids", lambda: set())
    monkeypatch.setattr(app_mod, "_known_run_ids", lambda: set())
    first = _FakeProvider("vast", torn=["vm-1"], halted=True)
    second = _FakeProvider("lambda", torn=["i-9"])
    monkeypatch.setattr(
        "flash.providers.core.registry.configured_providers",
        lambda: [first, second],
        raising=False,
    )

    result = app_mod._sweep_orphan_instances_once(lambda: False)

    assert result.deleted_count == 1
    assert result.halted
    assert second.seen_should_stop is None


def test_sweep_instances_with_no_stop_signal_counts_every_teardown(monkeypatch):
    """The ``should_stop=None`` default is a live production path, not just a test convenience:
    ``platform/runtime.py`` sweeps without a stop callback. This pins that the default reaches the
    provider as ``None`` -- the capabilities dispatcher forwards it POSITIONALLY into a bare
    ``except Exception``, so a provider that never learned the parameter would be reported as a
    silent RETRYABLE rather than raising. The teardown count is the unhalted baseline the halt
    test above reads against."""
    monkeypatch.setattr(app_mod, "_active_run_ids", lambda: set())
    monkeypatch.setattr(app_mod, "_known_run_ids", lambda: set())
    lam = _FakeProvider("lambda", torn=["i-1", "i-2"])
    monkeypatch.setattr(
        "flash.providers.core.registry.configured_providers", lambda: [lam], raising=False
    )

    assert app_mod._sweep_orphan_instances_once().deleted_count == 2
    assert lam.seen_should_stop is None
