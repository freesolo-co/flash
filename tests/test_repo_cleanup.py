"""The always-on repo GC deletes per-run HF artifact repos that aren't currently deployed once they
age out — and must NEVER touch a deployed repo, a young/in-flight repo, or delete blind when the
serving live set can't be confirmed. These tests pin those invariants and the fixed policy."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from flash.server import repo_cleanup as rc

# Real implementations, captured before the autouse fixture swaps in test defaults, so the unit
# tests below can exercise the genuine functions.
_REAL_KNOWN_RUN_REPO_IDS = rc._known_run_repo_ids
_REAL_HOLD_RUN_LOCK = rc._hold_run_lock
_REAL_INFLIGHT_REPO_IDS = rc._inflight_repo_ids
# The global offline conftest stubs run_scheduled_cleanup to a no-op (so the always-on GC sweep
# never reaches serving/HF in offline TestClient startups). Capture the genuine function so this
# file's fixture can restore it — these unit tests exercise the real sweep.
_REAL_RUN_SCHEDULED_CLEANUP = rc.run_scheduled_cleanup

NS = rc._ARTIFACT_NAMESPACE
NOW = datetime(2026, 6, 28, 12, 0, 0, tzinfo=UTC)


class _DummyHeld:
    """Stand-in for a held per-run lock: the sweep only ever calls ``release()`` on it."""

    def release(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _fast_and_frozen(monkeypatch):
    # Freeze now() so fixture ages are deterministic, and drop the inter-delete sleep.
    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW

    monkeypatch.setattr(rc, "datetime", _Frozen)
    monkeypatch.setattr(rc, "_DELETE_SLEEP_S", 0)
    # Default: no in-flight runs (isolate the sweep tests from the local run registry / db). The
    # in-flight guard is exercised explicitly below.
    monkeypatch.setattr(rc, "_inflight_repo_ids", lambda: set())
    # Default: this plane "knows" every repo (single-plane) — so the known-runs scope never filters
    # anything. The multi-plane scope is exercised explicitly below.
    monkeypatch.setattr(rc, "_known_run_repo_ids", lambda: _Everything())
    # Default: the per-run deploy/export lock is always free (nothing in progress) — the sweep
    # acquires-and-holds a dummy. The acquire-and-hold guard is exercised explicitly below.
    monkeypatch.setattr(rc, "_hold_run_lock", lambda repo_id: _DummyHeld())
    # The global offline conftest stubs run_scheduled_cleanup to a no-op (keeps the GC sweep off the
    # network in offline TestClient startups); restore the genuine function so these unit tests
    # exercise the real sweep. Mirrors the _REAL_* restores above.
    monkeypatch.setattr(rc, "run_scheduled_cleanup", _REAL_RUN_SCHEDULED_CLEANUP)
    # Clear HF_TOKEN so the enablement check is deterministic (the policy itself has no env knobs).
    monkeypatch.delenv("HF_TOKEN", raising=False)


class _Everything:
    """A set-like that contains everything — the test default for the 'runs this plane knows' scope
    (single-plane: every repo belongs to a run this plane issued)."""

    def __contains__(self, item: object) -> bool:
        return True


def _days_ago(d: float) -> datetime:
    return NOW - timedelta(days=d)


class _DS:
    def __init__(self, id_: str, last_modified=None):
        self.id = id_
        self.last_modified = last_modified


class _Info:
    def __init__(self, last_modified):
        self.last_modified = last_modified


class FakeApi:
    """Stand-in for HfApi: serves a fixed dataset listing and records deletes."""

    def __init__(self, datasets: list[_DS], info_lm: dict | None = None):
        self._datasets = datasets
        self._info_lm = info_lm or {}  # repo_id -> last_modified for the repo_info fallback
        self.deleted: list[str] = []

    def list_datasets(self, author=None):
        return list(self._datasets)

    def repo_info(self, repo_id, repo_type=None):
        return _Info(self._info_lm[repo_id])

    def delete_repo(self, repo_id=None, repo_type=None, missing_ok=None):
        self.deleted.append(repo_id)


# ---- gating -----------------------------------------------------------------------------------

def test_enabled_requires_hf_token(monkeypatch):
    assert rc.repo_cleanup_enabled() is False  # no HF_TOKEN -> never runs
    monkeypatch.setenv("HF_TOKEN", "tok")
    assert rc.repo_cleanup_enabled() is True  # credential present -> always on (no off switch)


def test_sweep_noops_when_huggingface_hub_unavailable(monkeypatch):
    # huggingface_hub is an OPTIONAL server extra. On a plane without it the always-on GC must
    # degrade to a logged no-op (returns 0, touches no serving/HF) instead of crashing the loop with
    # ModuleNotFoundError every cycle — mirroring _worker_artifacts()/_validate_hf_repo_id().
    monkeypatch.setattr(rc, "HfApi", None)
    monkeypatch.setattr(rc, "_warned_hf_unavailable", False)  # reset the warn-once latch
    # If the sweep tried to confirm the live set or list repos it would call these; make them blow up
    # so a regression that doesn't short-circuit is caught rather than silently passing.
    monkeypatch.setattr(rc, "_confirm_live_set", lambda: (_ for _ in ()).throw(AssertionError("called")))
    monkeypatch.setattr(rc, "list_run_repos", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    assert rc.run_scheduled_cleanup(dry_run=False, api=None) == 0  # no-op, no network


# ---- the policy predicate ---------------------------------------------------------------------

AGE = rc.DELETE_AGE_SECONDS / 86400.0  # the fixed 30-day threshold, in days


def test_deletable_old_undeployed_is_deleted():
    assert rc._deletable(f"{NS}/flashrun-a", _days_ago(AGE + 1), set(), NOW, AGE * 86400)


def test_deletable_skips_deployed_even_if_ancient():
    rid = f"{NS}/flashrun-b"
    assert rc._deletable(rid, _days_ago(400), {rid}, NOW, AGE * 86400) is False


def test_deletable_skips_young():
    assert rc._deletable(f"{NS}/flashrun-c", _days_ago(AGE - 1), set(), NOW, AGE * 86400) is False


def test_deletable_skips_unknown_age():
    assert rc._deletable(f"{NS}/flashrun-d", None, set(), NOW, AGE * 86400) is False


# ---- listing / allowlist ----------------------------------------------------------------------

def test_list_run_repos_allowlist_and_lm_fallback():
    api = FakeApi(
        datasets=[
            _DS(f"{NS}/flashrun-x", _days_ago(40)),
            _DS(f"{NS}/flashrun-y", None),          # listing lacks lm -> repo_info fallback
            _DS(f"{NS}/paper-gsm8k", _days_ago(40)),  # not flashrun-* -> ignored
            _DS(f"{NS}/some-env", _days_ago(40)),     # not flashrun-* -> ignored
        ],
        info_lm={f"{NS}/flashrun-y": _days_ago(99)},
    )
    out = dict(rc.list_run_repos(api, NS))
    assert set(out) == {f"{NS}/flashrun-x", f"{NS}/flashrun-y"}
    assert out[f"{NS}/flashrun-y"] == _days_ago(99)  # came from the repo_info fallback


# ---- end-to-end sweep -------------------------------------------------------------------------

def _sweep_api():
    return FakeApi([
        _DS(f"{NS}/flashrun-old-undeployed", _days_ago(45)),   # -> delete
        _DS(f"{NS}/flashrun-old-deployed", _days_ago(45)),     # deployed -> keep
        _DS(f"{NS}/flashrun-young", _days_ago(3)),             # young -> keep
        _DS(f"{NS}/paper-bench", _days_ago(99)),               # not flashrun -> keep
    ])


def test_sweep_deletes_only_old_undeployed(monkeypatch):
    api = _sweep_api()
    monkeypatch.setattr(rc, "deployed_repo_ids", lambda: ({f"{NS}/flashrun-old-deployed"}, True))
    n = rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert n == 1
    assert api.deleted == [f"{NS}/flashrun-old-undeployed"]


def test_dry_run_deletes_nothing(monkeypatch):
    api = _sweep_api()
    monkeypatch.setattr(rc, "deployed_repo_ids", lambda: (set(), True))
    n = rc.run_scheduled_cleanup(dry_run=True, api=api)
    assert n == 0
    assert api.deleted == []


# ---- fail-closed safety -----------------------------------------------------------------------

def test_aborts_and_deletes_nothing_when_serving_unreachable(monkeypatch):
    api = _sweep_api()

    def _boom():
        raise RuntimeError("serving down")

    monkeypatch.setattr(rc, "deployed_repo_ids", _boom)
    with pytest.raises(rc.CleanupAborted):
        rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert api.deleted == []


def test_aborts_on_incomplete_live_set(monkeypatch):
    # A live record without a repo id -> complete=False -> refuse to delete anything.
    api = _sweep_api()
    monkeypatch.setattr(rc, "deployed_repo_ids", lambda: ({f"{NS}/flashrun-old-deployed"}, False))
    with pytest.raises(rc.CleanupAborted):
        rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert api.deleted == []


def test_inflight_run_repo_is_protected_regardless_of_age(monkeypatch):
    # An old, undeployed repo belonging to a still-in-flight run must NOT be deleted (its repo
    # predates any worker and is never in serving's deployed set).
    rid = f"{NS}/flashrun-stuck-provisioning"
    api = FakeApi([_DS(rid, _days_ago(120))])
    monkeypatch.setattr(rc, "deployed_repo_ids", lambda: (set(), True))
    monkeypatch.setattr(rc, "_inflight_repo_ids", lambda: {rid})
    n = rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert n == 0
    assert api.deleted == []


def test_fails_closed_when_run_state_unenumerable(monkeypatch):
    # If the in-flight set can't be built (run registry unreadable), delete NOTHING.
    api = _sweep_api()
    monkeypatch.setattr(rc, "deployed_repo_ids", lambda: (set(), True))

    def _boom():
        raise RuntimeError("runs db unreadable")

    monkeypatch.setattr(rc, "_inflight_repo_ids", _boom)
    with pytest.raises(RuntimeError):
        rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert api.deleted == []


def test_per_delete_recheck_spares_repo_deployed_midsweep(monkeypatch):
    # Two old undeployed repos at enumeration; the first becomes deployed before its delete.
    api = FakeApi([
        _DS(f"{NS}/flashrun-1", _days_ago(45)),
        _DS(f"{NS}/flashrun-2", _days_ago(45)),
    ])
    calls = {"n": 0}

    def _live():
        # enumeration sees neither deployed; the pre-delete re-check for flashrun-1 sees it deployed.
        calls["n"] += 1
        if calls["n"] == 1:
            return (set(), True)
        return ({f"{NS}/flashrun-1"}, True)

    monkeypatch.setattr(rc, "deployed_repo_ids", _live)
    rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert api.deleted == [f"{NS}/flashrun-2"]  # 1 was spared by the mid-sweep re-check


def test_midsweep_unconfirmable_live_set_aborts_whole_sweep(monkeypatch):
    # If the live set becomes unconfirmable BETWEEN deletes (serving blip, or a live adapter with no
    # repo id), the WHOLE sweep must abort — never press on to delete later repos while an
    # unidentified live adapter may be backed by one of them.
    api = FakeApi([
        _DS(f"{NS}/flashrun-1", _days_ago(45)),
        _DS(f"{NS}/flashrun-2", _days_ago(45)),
    ])
    calls = {"n": 0}

    def _live():
        calls["n"] += 1
        if calls["n"] <= 2:
            return (set(), True)   # enumeration + pre-delete re-check for flashrun-1: OK -> delete
        return (set(), False)      # pre-delete re-check for flashrun-2: incomplete -> abort sweep

    monkeypatch.setattr(rc, "deployed_repo_ids", _live)
    with pytest.raises(rc.CleanupAborted):
        rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert api.deleted == [f"{NS}/flashrun-1"]  # 1 deleted before the abort; 2 was NOT deleted


# ---- multi-plane scope (only delete repos this plane issued) -----------------------------------

def test_known_run_repo_ids_built_from_local_runs(monkeypatch):
    from flash.server import db

    monkeypatch.setattr(db, "all_runs", lambda: [{"run_id": "r1"}, {"run_id": "r2"}])
    assert _REAL_KNOWN_RUN_REPO_IDS() == {f"{NS}/flashrun-r1", f"{NS}/flashrun-r2"}


def test_inflight_repo_ids_protects_warmstart_source(monkeypatch):
    # An in-flight GRPO run warm-starting (init_from_adapter) off an OLD SFT run must protect BOTH
    # its own repo AND that SFT source repo — the worker snapshot_downloads the source at boot, so
    # the GC must not delete it out from under the still-queued/running dependent run.
    from flash.server import db

    class _St:
        def __init__(self, state, spec):
            self.state = state
            self.spec = spec

    statuses = {
        "grpo1": _St("running", {"train": {
            "hf_repo": f"{NS}/flashrun-grpo1",
            "init_from_adapter": f"{NS}/flashrun-sft0:sft/sft0",
        }}),
        "done1": _St("done", {"train": {"hf_repo": f"{NS}/flashrun-done1"}}),
    }
    monkeypatch.setattr(db, "all_runs", lambda: [{"run_id": "grpo1"}, {"run_id": "done1"}])
    monkeypatch.setattr("flash.runner.get_status", lambda rid: statuses[rid])

    ids = _REAL_INFLIGHT_REPO_IDS()
    assert f"{NS}/flashrun-grpo1" in ids   # the in-flight run's own repo
    assert f"{NS}/flashrun-sft0" in ids    # its warm-start SOURCE repo (the fix)
    assert f"{NS}/flashrun-done1" not in ids  # a terminal run is not protected


def test_sweep_skips_repos_for_runs_this_plane_doesnt_know(monkeypatch):
    # A repo for a run this plane has no record of (e.g. launched by a sibling control plane) must
    # NOT be deleted even when old + undeployed — only repos this plane issued are reapable.
    ours = f"{NS}/flashrun-ours"
    theirs = f"{NS}/flashrun-theirs"
    api = FakeApi([_DS(ours, _days_ago(45)), _DS(theirs, _days_ago(45))])
    monkeypatch.setattr(rc, "deployed_repo_ids", lambda: (set(), True))
    monkeypatch.setattr(rc, "_known_run_repo_ids", lambda: {ours})
    n = rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert n == 1
    assert api.deleted == [ours]  # the sibling plane's repo was spared


# ---- in-progress deploy/export guard (acquire-and-hold the per-run lock) -----------------------

def test_sweep_skips_repo_with_deploy_or_export_in_progress(monkeypatch):
    # A repo whose run is mid-deploy/mid-export (holding the per-run lock, before serving reports it
    # live) must be spared — deleting its HF source would break the in-flight registration/download.
    # The GC's non-blocking acquire fails for such a repo, so _hold_run_lock returns None -> skip.
    busy = f"{NS}/flashrun-busy"
    other = f"{NS}/flashrun-other"
    api = FakeApi([_DS(busy, _days_ago(45)), _DS(other, _days_ago(45))])
    monkeypatch.setattr(rc, "deployed_repo_ids", lambda: (set(), True))
    monkeypatch.setattr(
        rc, "_hold_run_lock", lambda repo_id: None if repo_id == busy else _DummyHeld()
    )
    n = rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert n == 1
    assert api.deleted == [other]


def test_hold_run_lock_blocked_by_held_deploy_lock():
    # The real helper parses the run id from the repo id and does a NON-BLOCKING acquire of the same
    # per-run lock deploy/undeploy/export take: free -> returns a held lock; already held -> None.
    from flash.server import _locks

    rid = f"{NS}/flashrun-deploy-lock"
    held = _REAL_HOLD_RUN_LOCK(rid)  # lock free -> acquired and returned
    assert held is not None
    held.release()
    with _locks._deploy_lock("deploy-lock"):            # a deploy/export holds the run's lock
        assert _REAL_HOLD_RUN_LOCK(rid) is None         # GC can't acquire -> spare the repo
    again = _REAL_HOLD_RUN_LOCK(rid)                     # released -> acquirable again
    assert again is not None
    again.release()


def test_sweep_holds_deploy_lock_across_delete(monkeypatch):
    # The destructive delete must run WHILE the per-run lock is held, so a deploy/export is mutually
    # excluded from the delete window (not merely observed). Assert the lock is un-acquirable from
    # the outside exactly at delete time, using the REAL acquire-and-hold helper.
    from flash.server import _locks

    rid = f"{NS}/flashrun-held"
    seen: dict[str, bool] = {}

    class _CheckApi(FakeApi):
        def delete_repo(self, repo_id=None, repo_type=None, missing_ok=None):
            # A non-blocking acquire from outside must FAIL (return False) while the GC holds it.
            lk = _locks._deploy_lock("held")
            acquired = lk.acquire(blocking=False)
            seen["held_during_delete"] = not acquired
            if acquired:
                lk.release()
            super().delete_repo(repo_id=repo_id, repo_type=repo_type, missing_ok=missing_ok)

    api = _CheckApi([_DS(rid, _days_ago(45))])
    monkeypatch.setattr(rc, "deployed_repo_ids", lambda: (set(), True))
    monkeypatch.setattr(rc, "_hold_run_lock", _REAL_HOLD_RUN_LOCK)  # undo the fixture's dummy default
    n = rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert n == 1
    assert seen["held_during_delete"] is True
    # The lock is released after the sweep, so a later deploy can take it. Keep a strong ref across
    # acquire+release (the WeakValueDictionary would otherwise hand back a different object).
    lk = _locks._deploy_lock("held")
    assert lk.acquire(blocking=False) is True
    lk.release()


# ---- cooperative shutdown stop --------------------------------------------------------------

def test_sweep_stops_between_deletes_when_should_stop_set(monkeypatch):
    # The sweep checks should_stop BETWEEN targets and bails promptly (so a shutdown can't keep it
    # churning destructive deletes). Stop AFTER the first delete: first check False, second True.
    api = FakeApi([
        _DS(f"{NS}/flashrun-1", _days_ago(45)),
        _DS(f"{NS}/flashrun-2", _days_ago(45)),
    ])
    monkeypatch.setattr(rc, "deployed_repo_ids", lambda: (set(), True))
    calls = {"n": 0}

    def _stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    n = rc.run_scheduled_cleanup(dry_run=False, api=api, should_stop=_stop)
    assert n == 1
    assert api.deleted == [f"{NS}/flashrun-1"]  # flashrun-2 never reached after the stop


def test_cleanup_loop_signals_worker_stop_on_cancel(monkeypatch):
    # On shutdown the lifespan cancels the loop task; that only cancels the await on to_thread, so the
    # loop must SET a stop Event the in-flight worker sweep observes. Verify the loop threads a real
    # stop callback into the sweep and sets it once the cancel propagates.
    import asyncio

    from flash.server import _runtime

    seen: dict[str, object] = {}

    def fake_sweep(*, should_stop=None, **_kw):
        seen["should_stop"] = should_stop
        seen["stopped_while_running"] = should_stop()  # not yet set during the sweep
        raise asyncio.CancelledError  # emulate shutdown arriving mid-sweep

    # The loop imports run_scheduled_cleanup from flash.server.repo_cleanup at call time, so patch it
    # on that source module (rc), not on _runtime.
    monkeypatch.setattr(rc, "run_scheduled_cleanup", fake_sweep)

    calls = {"sleep": 0}

    async def fake_sleep(_interval):
        calls["sleep"] += 1  # never reached: the sweep runs FIRST and cancels before any sleep

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_runtime._repo_cleanup_loop())

    assert callable(seen["should_stop"])
    assert seen["stopped_while_running"] is False  # the worker saw a live (un-set) stop flag
    assert seen["should_stop"]() is True           # the loop set it on cancel -> worker will halt


def test_cleanup_loop_sweeps_on_startup_before_sleeping(monkeypatch):
    # The loop must run its FIRST sweep immediately on startup, then sleep between subsequent sweeps —
    # so a control plane that restarts (or crash-loops) more often than the 24h interval still reclaims
    # repos instead of always being cancelled before its first sleep elapses. Assert the sweep fires
    # before any sleep, and that the post-sweep sleep is what runs next.
    import asyncio

    from flash.server import _runtime

    order: list[str] = []

    def fake_sweep(*, should_stop=None, **_kw):
        order.append("sweep")
        return 0  # nothing deleted

    monkeypatch.setattr(rc, "run_scheduled_cleanup", fake_sweep)

    async def fake_sleep(_interval):
        order.append("sleep")
        raise asyncio.CancelledError  # stop the loop after one full sweep+sleep cycle

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_runtime._repo_cleanup_loop())

    # Sweep happened FIRST (startup), then the interval sleep — not the other way around.
    assert order == ["sweep", "sleep"]


# ---- malformed live-record fail-closed (deployed_repo_ids) -------------------------------------

def test_deployed_repo_ids_fail_closed_on_blank_or_nonstr(monkeypatch):
    from flash.serve import deploy

    monkeypatch.setattr(
        deploy,
        "list_deployed_adapters",
        lambda: [
            {"repoId": f"{NS}/flashrun-good"},
            {"repoId": "   "},   # whitespace-only -> incomplete, not coerced
            {"repoId": 12345},   # non-str -> incomplete, not coerced into a bogus id
        ],
    )
    ids, complete = rc.deployed_repo_ids()
    assert ids == {f"{NS}/flashrun-good"}
    assert complete is False  # a live repo couldn't be identified -> caller fails closed
