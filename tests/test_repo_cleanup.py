"""The always-on repo GC deletes per-run HF artifact repos that aren't currently deployed once they
age out — and must NEVER touch a deployed repo, a young/in-flight repo, or delete blind when the
serving live set can't be confirmed. These tests pin those invariants and the fixed policy."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from flash.server import repo_cleanup as rc

NS = rc._ARTIFACT_NAMESPACE
NOW = datetime(2026, 6, 28, 12, 0, 0, tzinfo=UTC)


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
    # A clean env so FLASH_REPO_GC_* from the host can't perturb thresholds/gating.
    for k in ("FLASH_REPO_GC_ENABLED", "FLASH_REPO_GC_DELETE_AGE_DAYS", "HF_TOKEN"):
        monkeypatch.delenv(k, raising=False)


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
    assert rc.repo_cleanup_enabled() is False  # no HF_TOKEN
    monkeypatch.setenv("HF_TOKEN", "tok")
    assert rc.repo_cleanup_enabled() is True
    monkeypatch.setenv("FLASH_REPO_GC_ENABLED", "0")
    assert rc.repo_cleanup_enabled() is False


# ---- the policy predicate ---------------------------------------------------------------------

AGE = rc.DEFAULT_DELETE_AGE_DAYS


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


def test_age_threshold_env_override(monkeypatch):
    # Lower the age to 2d: the 3-day-old "young" repo now ages out too.
    monkeypatch.setenv("FLASH_REPO_GC_DELETE_AGE_DAYS", "2")
    api = _sweep_api()
    monkeypatch.setattr(rc, "deployed_repo_ids", lambda: ({f"{NS}/flashrun-old-deployed"}, True))
    rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert set(api.deleted) == {f"{NS}/flashrun-old-undeployed", f"{NS}/flashrun-young"}


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
