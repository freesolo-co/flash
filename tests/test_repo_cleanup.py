"""The always-on artifact GC deletes a terminal run's ``<phase>/<run_id>`` prefix inside the shared
per-environment HF repo once it ages out and isn't serving — and must NEVER touch a deployed run, an
in-flight run (or its warm-start source), a sibling plane's run, or delete blind when the live set
can't be confirmed. These tests pin those invariants and the fixed 7-day policy."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from flash.server import repo_cleanup as rc

# Real implementations captured before the autouse fixture (and the offline conftest stub) swap in
# test defaults, so the unit tests below exercise the genuine functions.
_REAL_KNOWN_RUN_IDS = rc._known_run_ids
_REAL_INFLIGHT = rc._inflight_protected_prefixes
_REAL_DEPLOYED = rc.deployed_prefixes
_REAL_TERMINAL_TARGETS = rc._terminal_run_targets
_REAL_HOLD_RUN_LOCK = rc._hold_run_lock
# The global offline conftest stubs run_scheduled_cleanup to a no-op (so the always-on GC never
# reaches serving/HF in offline TestClient startups). Restore the genuine sweep for this file.
_REAL_RUN_SCHEDULED_CLEANUP = rc.run_scheduled_cleanup

NS = rc._ARTIFACT_NAMESPACE
_UNSET = object()
NOW = 1_800_000_000.0
DAY = 86400.0
AGE_DAYS = rc.DELETE_AGE_SECONDS / DAY  # the fixed 7-day threshold, in days


def _ago(days: float) -> float:
    return NOW - days * DAY


@pytest.fixture(autouse=True)
def _frozen(monkeypatch):
    monkeypatch.setattr(rc, "_now", lambda: NOW)
    monkeypatch.setattr(rc, "_DELETE_SLEEP_S", 0)
    monkeypatch.setattr(rc, "run_scheduled_cleanup", _REAL_RUN_SCHEDULED_CLEANUP)
    monkeypatch.delenv("HF_TOKEN", raising=False)


def _managed(slug: str) -> str:
    return f"{NS}/flashrun-{slug}"


class _DummyHeld:
    """Stand-in for a held per-run lock: the sweep only ever calls ``release()`` on it."""

    def release(self) -> None:
        pass


class _St:
    """Minimal ``RunStatus`` stand-in (only the fields the GC reads)."""

    def __init__(
        self,
        run_id,
        state,
        *,
        hf_repo=None,
        algorithm="sft",
        deployment=None,
        finished_at=None,
        updated_at=None,
        created_at=None,
        init_from_adapter=None,
        spec=_UNSET,
    ):
        self.run_id = run_id
        self.state = state
        if spec is _UNSET:
            train: dict = {}
            if hf_repo is not None:
                train["hf_repo"] = hf_repo
            if init_from_adapter is not None:
                train["init_from_adapter"] = init_from_adapter
            spec = {"algorithm": algorithm, "train": train}
        self.spec = spec
        self.deployment = deployment
        self.finished_at = finished_at
        self.updated_at = updated_at
        self.created_at = created_at


def _target(run_id="flash-1-a", *, slug="env", phase="sft", age_days=AGE_DAYS + 1) -> rc._RunTarget:
    return rc._RunTarget(
        run_id=run_id,
        repo_id=_managed(slug),
        prefix=f"{phase}/{run_id}",
        age_ts=_ago(age_days),
    )


class _Commit:
    def __init__(self, date):
        self.date = date


class _Entry:
    def __init__(self, date):
        self.last_commit = _Commit(date)


class FakeApi:
    """Stand-in for HfApi: serves per-prefix commit dates and records ``delete_folder`` calls."""

    def __init__(self, tree: dict | None = None):
        # (repo_id, prefix) -> list[datetime] commit dates. Absent -> empty listing.
        self._tree = tree or {}
        self.deleted: list[tuple[str, str]] = []

    def list_repo_tree(
        self, repo_id=None, repo_type=None, path_in_repo=None, recursive=False, expand=False
    ):
        return [_Entry(d) for d in self._tree.get((repo_id, path_in_repo), [])]

    def delete_folder(self, path_in_repo=None, repo_id=None, repo_type=None):
        self.deleted.append((repo_id, path_in_repo))


def _wire(monkeypatch, *, targets, deployed=None, inflight=frozenset(), known=None, hold=None):
    """Patch the sweep's seams so the end-to-end tests isolate policy from the registry/HF."""
    monkeypatch.setattr(rc, "_terminal_run_targets", lambda: list(targets))
    monkeypatch.setattr(
        rc, "deployed_prefixes", (lambda: (set(), True)) if deployed is None else deployed
    )
    monkeypatch.setattr(rc, "_inflight_protected_prefixes", lambda: set(inflight))
    monkeypatch.setattr(
        rc, "_known_run_ids", lambda: {t.run_id for t in targets} if known is None else set(known)
    )
    monkeypatch.setattr(rc, "_hold_run_lock", hold or (lambda run_id: _DummyHeld()))


# ---- gating -----------------------------------------------------------------------------------


def test_enabled_requires_hf_token(monkeypatch):
    assert rc.repo_cleanup_enabled() is False
    monkeypatch.setenv("HF_TOKEN", "tok")
    assert rc.repo_cleanup_enabled() is True


def test_sweep_noops_when_huggingface_hub_unavailable(monkeypatch):
    monkeypatch.setattr(rc, "HfApi", None)
    monkeypatch.setattr(rc, "_warned_hf_unavailable", False)
    # If it didn't short-circuit it would call these; make them explode so a regression is caught.
    monkeypatch.setattr(
        rc, "_confirm_live_set", lambda: (_ for _ in ()).throw(AssertionError("called"))
    )
    monkeypatch.setattr(
        rc, "_terminal_run_targets", lambda: (_ for _ in ()).throw(AssertionError("called"))
    )
    assert rc.run_scheduled_cleanup(dry_run=False, api=None) == 0


# ---- helpers ----------------------------------------------------------------------------------


def test_is_managed_env_repo_allowlist():
    assert rc._is_managed_env_repo(_managed("x")) is True
    assert rc._is_managed_env_repo(f"{NS}/some-env") is False  # not flashrun-*
    assert rc._is_managed_env_repo(f"{NS}/paper-gsm8k") is False
    assert rc._is_managed_env_repo("other-org/flashrun-x") is False  # wrong namespace
    assert rc._is_managed_env_repo(None) is False


def test_source_repo_prefix_parses_internal_ref():
    assert rc._source_repo_prefix(f"{NS}/flashrun-e:sft/flash-9-s") == (
        _managed("e"),
        "sft/flash-9-s",
    )
    # a checkpoint-step source still yields just the run prefix
    assert rc._source_repo_prefix(f"{NS}/flashrun-e:rl/flash-9-s/checkpoints/step-5") == (
        _managed("e"),
        "rl/flash-9-s",
    )
    assert rc._source_repo_prefix("flash-9-s/step-5") is None  # public form, no repo
    assert rc._source_repo_prefix("") is None
    assert rc._source_repo_prefix(None) is None


def test_run_repo_prefix_from_status():
    st = _St("flash-1-a", "done", hf_repo=_managed("e"), algorithm="grpo")
    assert rc._run_repo_prefix(st) == (_managed("e"), "rl/flash-1-a")  # grpo -> rl phase
    st2 = _St("flash-2-b", "done", hf_repo=_managed("e"), algorithm="opd")
    assert rc._run_repo_prefix(st2) == (_managed("e"), "opd/flash-2-b")
    assert rc._run_repo_prefix(_St("r", "done", hf_repo=None)) is None  # no repo


def test_run_id_epoch():
    assert rc._run_id_epoch("flash-1782280298-13db58fa") == 1782280298.0
    assert rc._run_id_epoch("weird") is None


# ---- the policy predicate ---------------------------------------------------------------------


def test_deletable_old_undeployed_known_is_deleted():
    t = _target("r1")
    assert rc._deletable(t, set(), {"r1"}, NOW, rc.DELETE_AGE_SECONDS) is True


def test_deletable_skips_deployed_even_if_ancient():
    t = _target("r1", age_days=400)
    assert rc._deletable(t, {(t.repo_id, t.prefix)}, {"r1"}, NOW, rc.DELETE_AGE_SECONDS) is False


def test_deletable_skips_young():
    t = _target("r1", age_days=AGE_DAYS - 1)
    assert rc._deletable(t, set(), {"r1"}, NOW, rc.DELETE_AGE_SECONDS) is False


def test_deletable_skips_unknown_age():
    t = rc._RunTarget("r1", _managed("e"), "sft/r1", age_ts=None)
    assert rc._deletable(t, set(), {"r1"}, NOW, rc.DELETE_AGE_SECONDS) is False


def test_deletable_skips_run_this_plane_doesnt_know():
    t = _target("r1")
    assert rc._deletable(t, set(), set(), NOW, rc.DELETE_AGE_SECONDS) is False  # not in known


# ---- end-to-end sweep -------------------------------------------------------------------------


def test_sweep_deletes_only_old_undeployed(monkeypatch):
    delete = _target("flash-1-old", slug="e1")
    deployed = _target("flash-2-dep", slug="e2")
    young = _target("flash-3-young", slug="e3", age_days=1)
    _wire(
        monkeypatch,
        targets=[delete, deployed, young],
        deployed=lambda: ({(deployed.repo_id, deployed.prefix)}, True),
    )
    api = FakeApi()
    n = rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert n == 1
    assert api.deleted == [(delete.repo_id, delete.prefix)]


def test_dry_run_deletes_nothing(monkeypatch):
    _wire(monkeypatch, targets=[_target("r1"), _target("r2")])
    api = FakeApi()
    assert rc.run_scheduled_cleanup(dry_run=True, api=api) == 0
    assert api.deleted == []


def test_sweep_skips_runs_this_plane_doesnt_know(monkeypatch):
    ours = _target("flash-1-ours")
    theirs = _target("flash-2-theirs")
    _wire(monkeypatch, targets=[ours, theirs], known={"flash-1-ours"})
    api = FakeApi()
    n = rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert n == 1
    assert api.deleted == [(ours.repo_id, ours.prefix)]  # sibling plane's run spared


def test_inflight_prefix_protected_regardless_of_age(monkeypatch):
    stuck = _target("flash-1-stuck", age_days=120)
    _wire(monkeypatch, targets=[stuck], inflight={(stuck.repo_id, stuck.prefix)})
    api = FakeApi()
    assert rc.run_scheduled_cleanup(dry_run=False, api=api) == 0
    assert api.deleted == []


# ---- fail-closed safety -----------------------------------------------------------------------


def test_aborts_and_deletes_nothing_when_live_set_unreachable(monkeypatch):
    def _boom():
        raise RuntimeError("registry down")

    _wire(monkeypatch, targets=[_target("r1")], deployed=_boom)
    api = FakeApi()
    with pytest.raises(rc.CleanupAborted):
        rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert api.deleted == []


def test_aborts_on_incomplete_live_set(monkeypatch):
    _wire(monkeypatch, targets=[_target("r1")], deployed=lambda: ({_managed("x")}, False))
    api = FakeApi()
    with pytest.raises(rc.CleanupAborted):
        rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert api.deleted == []


def test_fails_closed_when_inflight_unenumerable(monkeypatch):
    _wire(monkeypatch, targets=[_target("r1")])

    def _boom():
        raise RuntimeError("runs db unreadable")

    monkeypatch.setattr(rc, "_inflight_protected_prefixes", _boom)
    api = FakeApi()
    with pytest.raises(RuntimeError):
        rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert api.deleted == []


def test_fails_closed_when_known_runs_unenumerable(monkeypatch):
    _wire(monkeypatch, targets=[_target("r1")])

    def _boom():
        raise RuntimeError("db unreadable")

    monkeypatch.setattr(rc, "_known_run_ids", _boom)
    api = FakeApi()
    with pytest.raises(RuntimeError):
        rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert api.deleted == []


def test_per_delete_recheck_spares_prefix_deployed_midsweep(monkeypatch):
    t1 = _target("flash-1-a")
    t2 = _target("flash-2-b")
    calls = {"n": 0}

    def _live():
        calls["n"] += 1
        # up-front confirm + t2 pre-delete: nothing live; t1 pre-delete: t1 is now live.
        if calls["n"] == 2:
            return ({(t1.repo_id, t1.prefix)}, True)
        return (set(), True)

    _wire(monkeypatch, targets=[t1, t2], deployed=_live)
    api = FakeApi()
    rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert api.deleted == [(t2.repo_id, t2.prefix)]  # t1 spared by the mid-sweep re-check


def test_midsweep_unconfirmable_live_set_aborts_whole_sweep(monkeypatch):
    t1 = _target("flash-1-a")
    t2 = _target("flash-2-b")
    calls = {"n": 0}

    def _live():
        calls["n"] += 1
        if calls["n"] <= 2:
            return (set(), True)  # up-front + t1 pre-delete: OK -> delete t1
        return (set(), False)  # t2 pre-delete: incomplete -> abort the whole sweep

    _wire(monkeypatch, targets=[t1, t2], deployed=_live)
    api = FakeApi()
    with pytest.raises(rc.CleanupAborted):
        rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert api.deleted == [(t1.repo_id, t1.prefix)]  # t1 deleted before the abort; t2 not


def test_per_delete_recheck_spares_run_that_became_inflight(monkeypatch):
    # A run not in-flight at the up-front snapshot but that becomes a warm-start source (or goes
    # in-flight) before its delete must be spared by the per-delete in-flight re-check.
    t = _target("flash-1-a")
    calls = {"n": 0}

    def _inflight():
        calls["n"] += 1
        return (
            set() if calls["n"] == 1 else {(t.repo_id, t.prefix)}
        )  # up-front clear; pre-delete protected

    _wire(monkeypatch, targets=[t])
    monkeypatch.setattr(rc, "_inflight_protected_prefixes", _inflight)
    api = FakeApi()
    assert rc.run_scheduled_cleanup(dry_run=False, api=api) == 0
    assert api.deleted == []


def test_per_delete_inflight_recheck_error_skips_target(monkeypatch):
    # If the in-flight set can't be re-read right before a delete, fail SAFE: skip that target (an
    # unreadable registry is not proof the prefix is free) — but don't abort the whole sweep.
    t = _target("flash-1-a")
    calls = {"n": 0}

    def _inflight():
        calls["n"] += 1
        if calls["n"] == 1:
            return set()
        raise RuntimeError("registry blip")

    _wire(monkeypatch, targets=[t])
    monkeypatch.setattr(rc, "_inflight_protected_prefixes", _inflight)
    api = FakeApi()
    assert rc.run_scheduled_cleanup(dry_run=False, api=api) == 0
    assert api.deleted == []


# ---- in-progress deploy/export guard ----------------------------------------------------------


def test_sweep_skips_run_with_deploy_or_export_in_progress(monkeypatch):
    busy = _target("flash-1-busy")
    other = _target("flash-2-other")
    _wire(
        monkeypatch,
        targets=[busy, other],
        hold=lambda run_id: None if run_id == "flash-1-busy" else _DummyHeld(),
    )
    api = FakeApi()
    n = rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert n == 1
    assert api.deleted == [(other.repo_id, other.prefix)]


def test_cooperative_stop_halts_before_deleting(monkeypatch):
    _wire(monkeypatch, targets=[_target("r1"), _target("r2")])
    api = FakeApi()
    n = rc.run_scheduled_cleanup(dry_run=False, api=api, should_stop=lambda: True)
    assert n == 0
    assert api.deleted == []


# ---- per-prefix re-stat -----------------------------------------------------------------------


def test_recheck_spares_prefix_written_since_enumeration(monkeypatch):
    t = _target("flash-1-a")  # old by the db age gate...
    _wire(monkeypatch, targets=[t])
    # ...but a file under the prefix was committed 1 day ago -> within the 7-day window -> spare it.
    api = FakeApi({(t.repo_id, t.prefix): [datetime.fromtimestamp(_ago(1), UTC)]})
    assert rc.run_scheduled_cleanup(dry_run=False, api=api) == 0
    assert api.deleted == []


def test_recheck_proceeds_when_prefix_is_old(monkeypatch):
    t = _target("flash-1-a")
    _wire(monkeypatch, targets=[t])
    api = FakeApi({(t.repo_id, t.prefix): [datetime.fromtimestamp(_ago(30), UTC)]})  # 30d old
    assert rc.run_scheduled_cleanup(dry_run=False, api=api) == 1
    assert api.deleted == [(t.repo_id, t.prefix)]


def test_recheck_skips_target_when_restat_errors(monkeypatch):
    t = _target("flash-1-a")
    _wire(monkeypatch, targets=[t])

    class _BoomApi(FakeApi):
        def list_repo_tree(self, **kwargs):
            raise RuntimeError("HF blip")

    api = _BoomApi()
    assert rc.run_scheduled_cleanup(dry_run=False, api=api) == 0  # skipped to stay safe
    assert api.deleted == []


# ---- reconstruction from the run registry (real functions) ------------------------------------


def test_deployed_prefixes_built_from_run_status(monkeypatch):
    from flash.server import db

    statuses = {
        "d1": _St(
            "d1",
            "deployed",
            hf_repo=_managed("e1"),
            algorithm="grpo",
            deployment={"state": "ready"},
        ),
        "dep1": _St(
            "dep1",
            "done",
            hf_repo=_managed("e4"),
            algorithm="opd",
            deployment={"state": "deploying"},
        ),  # in-progress -> protected
        "u1": _St("u1", "done", hf_repo=_managed("e2"), deployment={"state": "undeployed"}),
        "f1": _St(
            "f1", "done", hf_repo=_managed("e5"), deployment={"state": "failed"}
        ),  # failed deploy serves nothing -> reclaimable
        "n1": _St("n1", "done", hf_repo=_managed("e3"), deployment=None),
    }
    monkeypatch.setattr(db, "all_runs", lambda: [{"run_id": r} for r in statuses])
    monkeypatch.setattr("flash.runner.get_status", lambda rid: statuses[rid])
    ids, complete = _REAL_DEPLOYED()
    assert ids == {
        (_managed("e1"), "rl/d1"),
        (_managed("e4"), "opd/dep1"),
    }  # live + in-progress only
    assert complete is True


def test_deployed_prefixes_incomplete_when_live_run_unmappable(monkeypatch):
    from flash.server import db

    statuses = {"d1": _St("d1", "deployed", hf_repo=None, deployment={"state": "ready"})}
    monkeypatch.setattr(db, "all_runs", lambda: [{"run_id": "d1"}])
    monkeypatch.setattr("flash.runner.get_status", lambda rid: statuses[rid])
    ids, complete = _REAL_DEPLOYED()
    assert complete is False  # a live run with no repo id -> fail closed
    assert ids == set()


def test_inflight_protects_own_and_warmstart_source(monkeypatch):
    from flash.server import db

    statuses = {
        # PUBLIC init_from_adapter form (a bare source RUN ID) — exactly what submit_job persists.
        "grpo1": _St(
            "grpo1", "running", hf_repo=_managed("e"), algorithm="grpo", init_from_adapter="sft0"
        ),
        "sft0": _St(
            "sft0", "done", hf_repo=_managed("e"), algorithm="sft"
        ),  # the warm-start source
        "done1": _St("done1", "done", hf_repo=_managed("e")),
    }
    monkeypatch.setattr(db, "all_runs", lambda: [{"run_id": r} for r in statuses])
    monkeypatch.setattr("flash.runner.get_status", lambda rid: statuses[rid])
    ids = _REAL_INFLIGHT()
    assert (_managed("e"), "rl/grpo1") in ids  # the in-flight run's own prefix
    assert (
        _managed("e"),
        "sft/sft0",
    ) in ids  # its warm-start SOURCE prefix (resolved via get_status)
    assert (_managed("e"), "sft/done1") not in ids  # a terminal run is not protected


def test_warmstart_source_prefix_public_and_internal_forms():
    src = _St("sft9", "done", hf_repo=_managed("e"), algorithm="sft")
    # Public form (what's persisted): a bare source run id resolved via get_status.
    assert rc._warmstart_source_prefix("sft9", lambda rid: src) == (_managed("e"), "sft/sft9")
    # Public form with a step suffix.
    assert rc._warmstart_source_prefix("sft9/step-5", lambda rid: src) == (
        _managed("e"),
        "sft/sft9",
    )
    # Internal colon form is still accepted defensively (no get_status needed).
    assert rc._warmstart_source_prefix(f"{_managed('e')}:sft/sft9", _raise) == (
        _managed("e"),
        "sft/sft9",
    )

    def _missing(rid):
        raise FileNotFoundError(rid)

    # A cross-plane source not in this plane's registry -> None, never a crash.
    assert rc._warmstart_source_prefix("othersrc", _missing) is None
    assert rc._warmstart_source_prefix("", lambda rid: None) is None
    assert rc._warmstart_source_prefix(None, lambda rid: None) is None


def _raise(rid):
    raise AssertionError("get_status must not be called for the internal colon form")


def test_known_run_ids_from_db(monkeypatch):
    from flash.server import db

    monkeypatch.setattr(db, "all_runs", lambda: [{"run_id": "r1"}, {"run_id": "r2"}])
    assert _REAL_KNOWN_RUN_IDS() == {"r1", "r2"}


def test_terminal_targets_filters_and_ages(monkeypatch):
    from flash.server import db

    statuses = {
        "done1": _St("done1", "done", hf_repo=_managed("e"), finished_at=_ago(10)),
        "run1": _St("run1", "running", hf_repo=_managed("e")),  # not terminal -> excluded
        "ext1": _St("ext1", "done", hf_repo="someuser/my-dataset"),  # not managed -> excluded
    }
    monkeypatch.setattr(db, "all_runs", lambda: [{"run_id": r} for r in statuses])
    monkeypatch.setattr("flash.runner.get_status", lambda rid: statuses[rid])
    targets = _REAL_TERMINAL_TARGETS()
    assert [t.run_id for t in targets] == ["done1"]
    assert targets[0].prefix == "sft/done1"
    assert targets[0].age_ts == _ago(10)  # from finished_at


def test_terminal_targets_skips_missing_run(monkeypatch):
    from flash.server import db

    def _get(rid):
        raise FileNotFoundError(rid)

    monkeypatch.setattr(db, "all_runs", lambda: [{"run_id": "gone"}])
    monkeypatch.setattr("flash.runner.get_status", _get)
    assert _REAL_TERMINAL_TARGETS() == []


# ---- the real per-run lock --------------------------------------------------------------------


def test_hold_run_lock_is_nonblocking_and_mutually_exclusive():
    from flash.server._locks import _deploy_lock

    held = _REAL_HOLD_RUN_LOCK("flash-lock-a")  # free -> acquired and returned
    assert held is not None
    # A non-blocking acquire from outside must FAIL while the GC holds it (same run id, same lock).
    assert _deploy_lock("flash-lock-a").acquire(blocking=False) is False
    held.release()
    reacquired = _deploy_lock("flash-lock-a").acquire(blocking=False)  # released -> free again
    assert reacquired is True
    _deploy_lock("flash-lock-a").release()
