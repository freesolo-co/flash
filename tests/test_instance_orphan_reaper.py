"""Instance-provider orphan sweep selection, dispatch, and stop behavior."""

from __future__ import annotations

import pytest

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


def test_vast_sweep_refuses_clean_absence_from_a_halted_listing(monkeypatch):
    """A stop landing mid-pagination is a FALSE-CLEAN hazard, not just wasted latency.

    Vast's ``list_instances`` walks pages; when a later page's retry loop observes the stop it
    raises, and the lenient (non-strict) path catches that and RETURNS the pages already collected.
    The sweep cannot tell that short list from a complete one. If the collected pages happen to
    hold no orphan, the ``not selected`` early return reports ``ABSENT`` -- a positive claim that
    teardown is confirmed clean, built on pages that were never fetched. An unread page can hold a
    live billable box. The sweep must instead report RETRYABLE/halted so the next sweep re-reads."""
    from flash.providers.core.capabilities import CleanupOutcome
    from flash.providers.vast import jobs
    from flash.providers.vast.client import api as vast_api

    destroyed = []

    def fake_list(should_stop=None, **_):
        # page 1 succeeded and held no orphan; page 2's retry loop saw the stop, so the real
        # client returns just the collected prefix rather than raising.
        assert should_stop is not None, "the sweep must forward a stop callback into the listing"
        should_stop()
        return [{"id": 1, "label": jobs.instance_label("flash-live", 0, 0)}]

    monkeypatch.setattr(vast_api, "list_instances", fake_list)
    monkeypatch.setattr(
        vast_api, "destroy_instance", lambda iid, **_: destroyed.append(iid) or True
    )

    out = jobs.sweep_orphans(active_labels={"flash-live"}, should_stop=lambda: True)

    # before the fix this was CleanupOutcome.ABSENT with halted=False: absence claimed from a
    # partial inventory.
    assert out.outcome is CleanupOutcome.RETRYABLE
    assert out.halted
    assert destroyed == []


def test_vast_sweep_keeps_absence_evidence_when_the_listing_completed(monkeypatch):
    """The converse of the halted-listing guard, and the reason it observes the stop the LISTING
    saw rather than re-reading the flag afterwards. A listing that ran to completion is
    authoritative even if a stop is raised immediately after it; discarding its absence evidence
    would make every sweep racing a shutdown report RETRYABLE and never confirm a clean account."""
    from flash.providers.core.capabilities import CleanupOutcome
    from flash.providers.vast import jobs
    from flash.providers.vast.client import api as vast_api

    stop_raised = False

    def fake_list(should_stop=None, **_):
        # every page fetched cleanly: the retry loops polled the stop and saw it unset. the
        # shutdown then begins the instant the last page lands.
        nonlocal stop_raised
        assert should_stop is not None
        assert should_stop() is False
        stop_raised = True
        return [{"id": 1, "label": jobs.instance_label("flash-live", 0, 0)}]

    monkeypatch.setattr(vast_api, "list_instances", fake_list)
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid, **_: True)

    out = jobs.sweep_orphans(active_labels={"flash-live"}, should_stop=lambda: stop_raised)

    # a naive re-read of the flag after the listing would report RETRYABLE/halted here and throw
    # away a complete inventory's absence evidence, so every sweep racing a shutdown would fail to
    # confirm a clean account.
    assert stop_raised  # the stop really is set by the time the sweep inspects its result
    assert out.outcome is CleanupOutcome.ABSENT
    assert not out.halted


def test_lambda_sweep_reports_a_halted_listing_as_halted(monkeypatch):
    """Lambda's listing is a SINGLE unpaginated request, so unlike Vast it cannot hand back a
    partial inventory: a stop during its retries surfaces as a raise. That already returned
    RETRYABLE, so there is no false-clean here -- but it was reported as an ordinary retryable
    failure, indistinguishable from an API blip. Stamping ``halted`` lets the caller tell a
    deliberate shutdown apart from a provider error it should alarm on."""
    from flash.providers.core.capabilities import CleanupOutcome
    from flash.providers.lambda_ import jobs
    from flash.providers.lambda_.client import api as lambda_api

    def fake_list(should_stop=None, **_):
        assert should_stop is not None, "the sweep must forward a stop callback into the listing"
        should_stop()
        raise lambda_api.LambdaApiError("listing aborted by stop")

    monkeypatch.setattr(lambda_api, "list_instances", fake_list)

    out = jobs.sweep_orphans(should_stop=lambda: True)

    assert out.outcome is CleanupOutcome.RETRYABLE
    assert out.halted  # before the fix: halted=False, a shutdown misread as a provider blip


def test_vast_destroy_forwards_the_stop_into_its_absence_confirmation(monkeypatch):
    """When the DELETE itself fails, absence is confirmed by a second lookup that allows two 30s
    attempts plus a backoff. Without the stop reaching that lookup, every shutdown racing a destroy
    waits out the extra minute on the joined sweep thread -- once per instance still in the sweep.

    It also pins the ORDER: a lookup that ran to completion and found the instance gone is
    authoritative even when the stop lands immediately afterwards. Reading the halt first would
    throw away real deletion evidence and leave a confirmed-gone instance recorded as unconfirmed.
    """
    from flash.providers._lifecycle.net.destructive import DestructiveOperationOutcome
    from flash.providers.vast.client import api as vast_api

    stopping = False
    lookup_stops: list[object] = []

    def fake_req(path, method="GET", body=None, retries=4, base_delay=2.0, deadline_at=None, **kw):
        nonlocal stopping
        if method == "DELETE":
            raise vast_api.VastApiError("destroy blipped")
        # the absence lookup: its retry loop polls the stop, which the lifespan raises mid-call
        lookup_stops.append(kw.get("should_stop"))
        stopping = True
        return {"success": True, "instances": None}

    monkeypatch.setattr(vast_api, "request_with_retries", fake_req)

    out = vast_api._destroy_instance_outcome(7, should_stop=lambda: stopping)

    assert len(lookup_stops) == 1
    assert lookup_stops[0] is not None  # before the fix: None, so the lookup ran unstoppable
    assert stopping  # the stop really is set by the time the outcome is decided
    assert out is DestructiveOperationOutcome.DELETED


def test_vast_destroy_reports_a_stop_cut_absence_lookup_as_halted(monkeypatch):
    """The converse: a lookup the stop cut short produced no absence evidence at all. Calling that
    NOT_CONFIRMED would file a fabricated provider fault for an instance nothing is actually wrong
    with, on every shutdown that lands mid-destroy.

    The stop must land in the LOOKUP, not the DELETE. A stop the DELETE itself saw is caught by the
    earlier halt check, so a fake that stops during both would leave this branch untested.
    """
    from flash.providers._lifecycle.net.destructive import DestructiveOperationOutcome
    from flash.providers.vast.client import api as vast_api

    stopping = False

    def fake_req(path, method="GET", body=None, retries=4, base_delay=2.0, deadline_at=None, **kw):
        nonlocal stopping
        if method == "DELETE":
            # a plain transport blip: the stop is not raised yet, so the destroy is not a halt
            raise vast_api.VastApiError("destroy blipped")
        stop = kw.get("should_stop")
        assert stop is not None
        stopping = True
        stop()  # the transport polls the stop and gives up on this attempt
        raise vast_api.VastApiError("aborted by stop")

    monkeypatch.setattr(vast_api, "request_with_retries", fake_req)

    out = vast_api._destroy_instance_outcome(7, should_stop=lambda: stopping)

    assert stopping, (
        "the absence lookup must run; a stop seen by the DELETE proves a different branch"
    )
    assert out is DestructiveOperationOutcome.HALTED


def test_vast_destroy_stopped_during_the_delete_does_not_open_a_second_lookup(monkeypatch):
    """The other halt branch. When the DELETE itself observed the stop, the absence lookup is
    already known to be pointless: it would spend two more 30s attempts to answer a question the
    caller has stopped caring about, on the joined sweep thread that shutdown is waiting for."""
    from flash.providers._lifecycle.net.destructive import DestructiveOperationOutcome
    from flash.providers.vast.client import api as vast_api

    methods: list[str] = []

    def fake_req(path, method="GET", body=None, retries=4, base_delay=2.0, deadline_at=None, **kw):
        methods.append(method)
        stop = kw.get("should_stop")
        assert stop is not None
        stop()  # the destroy's own retry loop polls the stop and gives up
        raise vast_api.VastApiError("aborted by stop")

    monkeypatch.setattr(vast_api, "request_with_retries", fake_req)

    out = vast_api._destroy_instance_outcome(7, should_stop=lambda: True)

    assert methods == ["DELETE"]  # before the fix: a second, pointless GET lookup
    assert out is DestructiveOperationOutcome.HALTED


def test_vast_sweep_records_a_halt_from_an_all_successful_paginated_listing(monkeypatch):
    """The false-clean hazard reaches the sweep even when NOTHING fails.

    The sibling guard above fakes ``list_instances`` outright, so it only proves the sweep reacts
    to a listing that already observed the stop. A real account whose pages all succeed on the
    first attempt never polls the stop inside the retry loop, so before the page-boundary check the
    sweep's ``_observe_listing_stop`` wrapper was never invoked: the walk ran to the end and its
    partial-then-complete result was indistinguishable from a clean account. Exercise the real
    ``list_instances`` against a fake transport to prove the halt is recorded end to end.
    """
    from flash.providers.core.capabilities import CleanupOutcome
    from flash.providers.vast import jobs
    from flash.providers.vast.client import api as vast_api

    fetched: list[int] = []
    shutting_down = False

    def fake_request(path, **kwargs):
        nonlocal shutting_down
        fetched.append(len(fetched) + 1)
        shutting_down = True  # the lifespan shutdown begins while page 1 is in flight
        # page 1 holds only a live run's box, so a walk that stopped here and reported a complete
        # listing would claim ABSENT -- teardown confirmed clean -- from one unread page.
        return {
            "instances": [{"id": 1, "label": jobs.instance_label("flash-live", 0, 0)}],
            "next_token": "t2",
        }

    monkeypatch.setattr(vast_api._CLIENT, "request_with_retries", fake_request)
    monkeypatch.setattr(
        vast_api, "destroy_instance", lambda iid, **_: pytest.fail(f"destroyed {iid} during a stop")
    )

    out = jobs.sweep_orphans(active_labels={"flash-live"}, should_stop=lambda: shutting_down)

    assert fetched == [1], "the walk must end at the first page boundary, not run to the cap"
    assert out.outcome is CleanupOutcome.RETRYABLE
    assert out.halted
