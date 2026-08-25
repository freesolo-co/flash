"""The always-on artifact GC is CROSS-PLANE: it reaps a run's ``<phase>/<run_id>`` prefix inside the
shared per-environment HF repo once it ages out and isn't in the GLOBAL serving set — never touching a
deployed adapter (on any plane), a recently-written (in-flight) prefix, a warm-start source with a
recent ``referenced_by`` marker, ``code/``/unknown dirs, or deleting blind when the serving registry
can't be confirmed. These tests pin those invariants and the fixed 7-day policy."""

from __future__ import annotations

import os
import subprocess
import sys
import types
from datetime import UTC, datetime

import pytest

from flash.runner.accounting.artifacts import _DEFAULT_ARTIFACT_NAMESPACE
from flash.serve.contract import urls as serving_urls
from flash.serve.request import transport as serving_transport
from flash.server.domain.ops import repo_cleanup as rc

# Real implementations captured before the offline conftest stub swaps in a no-op sweep.
_REAL_DEPLOYED = rc.deployed_prefixes
_REAL_HOLD_RUN_LOCK = rc._hold_run_lock
_REAL_RUN_SCHEDULED_CLEANUP = rc.run_scheduled_cleanup

# The DEFAULT managed namespace, not whatever the ambient env resolves to. This is read at IMPORT
# time while `artifact_namespace()` reads FLASH_HF_NAMESPACE at CALL time, so deriving it from the
# live env would bind it before the offline fixture scrubs that var -- every expectation built from
# NS would then name a different namespace than the code under test. The `_frozen` fixture below
# makes the two agree for real by forcing the env to this value for the duration of each test.
NS = _DEFAULT_ARTIFACT_NAMESPACE
NOW = 1_800_000_000.0
DAY = 86400.0
AGE_DAYS = rc.DELETE_AGE_SECONDS / DAY  # the fixed 7-day threshold, in days
OLD = AGE_DAYS + 5  # comfortably past the age gate
YOUNG = AGE_DAYS - 1  # inside the age gate


def _ago(days: float) -> float:
    return NOW - days * DAY


def _managed(slug: str) -> str:
    return f"{NS}/flashrun-{slug}"


@pytest.fixture(autouse=True)
def _frozen(monkeypatch):
    # Pin the namespace the code under test resolves to the same constant the expectations above
    # were built from. `artifact_namespace()` reads FLASH_HF_NAMESPACE on every call, so without
    # this an operator shell that exports it (per SELF_HOSTING.md) would make the sweep scan
    # `<their-ns>/...` while every fixture repo id says `Freesolo-Co/...`, and the allowlist would
    # match nothing -- failing locally while CI, whose env is clean, stayed green.
    monkeypatch.setenv("FLASH_HF_NAMESPACE", NS)
    monkeypatch.setattr(rc, "_now", lambda: NOW)
    monkeypatch.setattr(rc, "_DELETE_SLEEP_S", 0)
    monkeypatch.setattr(rc, "run_scheduled_cleanup", _REAL_RUN_SCHEDULED_CLEANUP)
    monkeypatch.delenv("HF_TOKEN", raising=False)


class _DummyHeld:
    """Stand-in for a held per-run lock: the sweep only ever calls ``release()`` on it."""

    def release(self) -> None:
        pass


class _Commit:
    def __init__(self, date):
        self.date = date


class _Entry:
    def __init__(self, path, size, date):
        self.path = path
        self.size = size
        self.last_commit = _Commit(date) if date is not None else None


class FakeApi:
    """Stand-in for HfApi driving the whole sweep: enumerates repos, serves per-file commit dates for
    both the scan and the pre-delete re-stat, and records ``delete_folder`` calls.

    ``repos`` maps ``repo_id -> [(path, size, age_days_or_None), ...]``; an ``age_days`` of ``None``
    models a file with no commit date (older ``huggingface_hub``)."""

    def __init__(self, repos: dict | None = None):
        self._repos = repos or {}
        self.deleted: list[tuple[str, str]] = []

    def list_datasets(self, author=None):
        return [types.SimpleNamespace(id=r) for r in self._repos]

    def list_repo_tree(
        self, repo_id=None, repo_type=None, path_in_repo=None, recursive=False, expand=False
    ):
        out = []
        for path, size, age in self._repos.get(repo_id, []):
            if path_in_repo and not (path == path_in_repo or path.startswith(path_in_repo + "/")):
                continue
            date = datetime.fromtimestamp(_ago(age), UTC) if age is not None else None
            out.append(_Entry(path, size, date))
        return out

    def delete_folder(self, path_in_repo=None, repo_id=None, repo_type=None):
        self.deleted.append((repo_id, path_in_repo))


# A single live adapter in an unrelated repo, so the fail-closed "zero live adapters" guard passes in
# tests that are really about the aged-undeployed path. Deliberately disjoint from any target prefix.
_LIVE_SENTINEL = ({(_managed("live"), "rl/flash-live-0")}, set(), True)


def _wire(monkeypatch, *, deployed=None, hold=None):
    """Patch the sweep's non-HF seams: the global serving set and the per-run lock. The HF side is
    driven entirely by the FakeApi passed to ``run_scheduled_cleanup``."""
    monkeypatch.setattr(
        rc, "deployed_prefixes", deployed if deployed is not None else (lambda: _LIVE_SENTINEL)
    )
    monkeypatch.setattr(rc, "_hold_run_lock", hold or (lambda run_id: _DummyHeld()))


def _adapter(path: str, age: float, size: int = 1000):
    return (path, size, age)


# ---- gating -----------------------------------------------------------------------------------


def test_enabled_requires_hf_token(monkeypatch):
    assert rc.repo_cleanup_enabled() is False
    monkeypatch.setenv("HF_TOKEN", "tok")
    assert rc.repo_cleanup_enabled() is True


@pytest.mark.parametrize("value", ["1", "64", "0", "-1", "not-a-number"])
def test_scan_workers_ignores_environment(value):
    env = os.environ.copy()
    env["FLASH_GC_SCAN_WORKERS"] = value
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from flash.server.domain.ops.repo_cleanup import _SCAN_WORKERS; print(_SCAN_WORKERS)",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "8"


def test_sweep_noops_when_huggingface_hub_unavailable(monkeypatch):
    monkeypatch.setattr(rc, "HfApi", None)
    monkeypatch.setattr(rc, "_warned_hf_unavailable", False)
    # If it didn't short-circuit it would confirm the live set; make that explode to catch a regression.
    monkeypatch.setattr(
        rc, "_confirm_live_set", lambda: (_ for _ in ()).throw(AssertionError("called"))
    )
    assert rc.run_scheduled_cleanup(dry_run=False, api=None) == 0


# ---- helpers ----------------------------------------------------------------------------------


def test_is_managed_env_repo_allowlist():
    assert rc._is_managed_env_repo(_managed("x")) is True
    assert rc._is_managed_env_repo(f"{NS}/some-env") is False  # not flashrun-*
    assert rc._is_managed_env_repo(f"{NS}/paper-gsm8k") is False
    assert rc._is_managed_env_repo("other-org/flashrun-x") is False  # wrong namespace
    assert rc._is_managed_env_repo(None) is False


def test_scan_repo_classifies_paths():
    api = FakeApi(
        {
            _managed("e"): [
                _adapter("sft/flash-1-a/adapter/w.safetensors", OLD),
                _adapter("sft/flash-1-a/checkpoints/step-3/adapter/w", OLD + 3),  # older sibling
                _adapter("rl/flash-2-b/adapter/w", YOUNG),
                _adapter("recomb/flash-2-b/adapter/w", YOUNG),
                _adapter("code/deadbeef/flash/x.py", OLD),  # snapshot -> ignored
                _adapter("referenced_by/flash-9-child", OLD, size=0),  # lineage marker
                _adapter(
                    "telemetry/events.jsonl", OLD
                ),  # unknown top-level -> reported, not reaped
            ]
        }
    )
    prefixes, ref_recent_ts, unknown = rc._scan_repo(api, _managed("e"))
    assert set(prefixes) == {"sft/flash-1-a", "rl/flash-2-b", "recomb/flash-2-b"}
    # newest commit across the prefix's files wins (the adapter at OLD, not the older checkpoint).
    assert prefixes["sft/flash-1-a"][1] == pytest.approx(_ago(OLD))
    assert ref_recent_ts == pytest.approx(_ago(OLD))  # from the referenced_by marker
    assert unknown == {"telemetry"}


def test_private_opd_retry_markers_are_never_cleanup_targets(monkeypatch):
    from flash.teacher.retry_contract import opd_optimizer_start_marker_path

    repo = _managed("opd-retry")
    marker_path = opd_optimizer_start_marker_path("flash-1-a", 0)
    api = FakeApi(
        {
            repo: [
                _adapter("opd/flash-1-a/adapter/w.safetensors", OLD),
                _adapter(marker_path, OLD, size=123),
            ]
        }
    )

    prefixes, _ref_recent_ts, unknown = rc._scan_repo(api, repo)
    assert set(prefixes) == {"opd/flash-1-a"}
    assert unknown == {"_opd_retry"}

    _wire(monkeypatch)
    assert rc.run_scheduled_cleanup(dry_run=False, api=api) == 1
    assert api.deleted == [(repo, "opd/flash-1-a")]
    assert all(marker_path != deleted_path for _repo, deleted_path in api.deleted)


# ---- the global serving set (deployed_prefixes) -----------------------------------------------


def _serving(monkeypatch, records):
    """Point deployed_prefixes' serving call at a canned ``GET /adapters`` payload."""
    monkeypatch.setattr(serving_urls, "serving_base_url", lambda: "https://serving.test")
    resp = types.SimpleNamespace(json=lambda: {"adapters": records})
    monkeypatch.setattr(serving_transport, "serving_request", lambda method, url: resp)


def test_deployed_prefixes_from_serving_registry(monkeypatch):
    _serving(
        monkeypatch,
        [
            {"repo_id": _managed("e1"), "subfolder": "rl/flash-1-a"},
            {"repo_id": _managed("e2"), "subfolder": "sft/flash-2-b/checkpoints/step-4"},
        ],
    )
    prefixes, whole, complete = _REAL_DEPLOYED()
    assert prefixes == {(_managed("e1"), "rl/flash-1-a"), (_managed("e2"), "sft/flash-2-b")}
    assert whole == set()
    assert complete is True


def test_deployed_prefixes_incomplete_when_record_has_no_repo(monkeypatch):
    _serving(monkeypatch, [{"subfolder": "rl/flash-1-a"}])  # live adapter, no repo id
    prefixes, _whole, complete = _REAL_DEPLOYED()
    assert complete is False
    assert prefixes == set()


def test_deployed_prefixes_protects_whole_repo_when_subfolder_unmappable(monkeypatch):
    _serving(monkeypatch, [{"repo_id": _managed("e1"), "subfolder": "toplevel-only"}])
    prefixes, whole, complete = _REAL_DEPLOYED()
    assert whole == {_managed("e1")}
    assert prefixes == set()
    assert complete is True


# ---- fail-closed confirmation -----------------------------------------------------------------


def test_confirm_live_set_aborts_when_unreachable(monkeypatch):
    def _boom():
        raise RuntimeError("registry down")

    _wire(monkeypatch, deployed=_boom)
    api = FakeApi({_managed("e"): [_adapter("sft/flash-1-a/w", OLD)]})
    with pytest.raises(rc.CleanupAborted):
        rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert api.deleted == []


def test_confirm_live_set_aborts_when_incomplete(monkeypatch):
    _wire(monkeypatch, deployed=lambda: (set(), set(), False))  # an unmappable live adapter
    api = FakeApi({_managed("e"): [_adapter("sft/flash-1-a/w", OLD)]})
    with pytest.raises(rc.CleanupAborted):
        rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert api.deleted == []


def test_confirm_live_set_aborts_when_registry_empty(monkeypatch):
    # Zero live adapters almost always means a broken/empty query -> refuse rather than sweep against
    # an empty do-not-touch set.
    _wire(monkeypatch, deployed=lambda: (set(), set(), True))
    api = FakeApi({_managed("e"): [_adapter("sft/flash-1-a/w", OLD)]})
    with pytest.raises(rc.CleanupAborted):
        rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert api.deleted == []


# ---- end-to-end sweep -------------------------------------------------------------------------


def test_sweep_deletes_only_old_undeployed(monkeypatch):
    deployed_pfx = (_managed("e2"), "rl/flash-2-dep")
    _wire(monkeypatch, deployed=lambda: ({deployed_pfx}, set(), True))
    api = FakeApi(
        {
            _managed("e1"): [_adapter("sft/flash-1-old/adapter/w", OLD)],  # -> deleted
            _managed("e2"): [_adapter("rl/flash-2-dep/adapter/w", OLD)],  # deployed -> spared
            _managed("e3"): [_adapter("sft/flash-3-young/adapter/w", YOUNG)],  # young -> spared
        }
    )
    n = rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert n == 1
    assert api.deleted == [(_managed("e1"), "sft/flash-1-old")]


def test_dry_run_deletes_nothing(monkeypatch):
    _wire(monkeypatch)
    api = FakeApi(
        {
            _managed("e1"): [_adapter("sft/flash-1-old/w", OLD)],
            _managed("e2"): [_adapter("rl/flash-2-old/w", OLD)],
        }
    )
    assert rc.run_scheduled_cleanup(dry_run=True, api=api) == 0
    assert api.deleted == []


def test_sweep_never_deletes_code_or_unknown_dirs(monkeypatch):
    _wire(monkeypatch)
    api = FakeApi(
        {
            _managed("e"): [
                _adapter("code/deadbeef/flash/x.py", OLD),  # shared snapshot
                _adapter("telemetry/events.jsonl", OLD),  # unknown phase
                _adapter("sft/flash-1-old/adapter/w", OLD),  # the only real target
            ]
        }
    )
    n = rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert n == 1
    assert api.deleted == [(_managed("e"), "sft/flash-1-old")]  # code/ + telemetry/ untouched


def test_sweep_protects_warmstart_source_with_recent_marker(monkeypatch):
    # An OLD sft that a run recently warm-started from (recent referenced_by marker) is a source for a
    # possibly-in-flight child -> spared, even though the artifact itself is aged.
    _wire(monkeypatch)
    api = FakeApi(
        {
            _managed("src"): [
                _adapter("sft/flash-1-src/adapter/w", OLD),
                _adapter("referenced_by/flash-9-child", YOUNG, size=0),  # recent lineage marker
            ]
        }
    )
    assert rc.run_scheduled_cleanup(dry_run=False, api=api) == 0
    assert api.deleted == []


def test_sweep_reaps_source_whose_marker_is_old(monkeypatch):
    # An OLD referenced_by marker = the child finished long ago (already baked its own recomb) -> the
    # source is no longer needed and is reaped.
    _wire(monkeypatch)
    api = FakeApi(
        {
            _managed("src"): [
                _adapter("sft/flash-1-src/adapter/w", OLD),
                _adapter("referenced_by/flash-9-child", OLD, size=0),  # stale lineage marker
            ]
        }
    )
    n = rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert n == 1
    assert api.deleted == [(_managed("src"), "sft/flash-1-src")]


def test_sweep_protects_whole_repo_when_live_unmappable(monkeypatch):
    _wire(monkeypatch, deployed=lambda: (_LIVE_SENTINEL[0], {_managed("murky")}, True))
    api = FakeApi(
        {
            _managed("murky"): [_adapter("sft/flash-1-old/w", OLD)],  # whole repo protected
            _managed("clean"): [_adapter("sft/flash-2-old/w", OLD)],  # reaped
        }
    )
    n = rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert n == 1
    assert api.deleted == [(_managed("clean"), "sft/flash-2-old")]


def test_sweep_skips_undatable_prefix(monkeypatch):
    # A prefix whose files carry NO commit dates is undatable: submit time (the run-id epoch) is not
    # last-activity, so the sweep never deletes it — a still-writing in-flight run whose fresh commits
    # the Hub returned without dates must not be reaped on submit time (fail closed; retry next cycle).
    _wire(monkeypatch)
    old_epoch = int(NOW - OLD * DAY)  # submitted long ago, but the listing carries no commit date
    api = FakeApi({_managed("e1"): [_adapter(f"sft/flash-{old_epoch}-a/w", None)]})
    assert rc.run_scheduled_cleanup(dry_run=False, api=api) == 0
    assert api.deleted == []


def test_sweep_never_touches_non_flashrun_repos(monkeypatch):
    # The hard allowlist (flashrun-* only) must hold end-to-end through the sweep: env/eval packages in
    # the same namespace are never delete targets even with old artifact-shaped paths.
    _wire(monkeypatch)
    api = FakeApi(
        {
            f"{NS}/paper-gsm8k": [
                _adapter("sft/flash-1-old/adapter/w", OLD)
            ],  # eval pkg, NOT a run repo
            f"{NS}/some-env": [_adapter("sft/flash-2-old/adapter/w", OLD)],  # env package
            _managed("real"): [
                _adapter("sft/flash-3-old/adapter/w", OLD)
            ],  # the only reapable repo
        }
    )
    n = rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert n == 1
    assert api.deleted == [(_managed("real"), "sft/flash-3-old")]


def test_scan_failure_on_one_repo_is_not_fatal(monkeypatch):
    _wire(monkeypatch)

    class _FlakyApi(FakeApi):
        def list_repo_tree(self, repo_id=None, **kwargs):
            if repo_id == _managed("bad") and kwargs.get("path_in_repo") is None:
                raise RuntimeError("HF 429")
            return super().list_repo_tree(repo_id=repo_id, **kwargs)

    api = _FlakyApi(
        {
            _managed("bad"): [
                _adapter("sft/flash-1-old/w", OLD)
            ],  # scan raises -> skipped this cycle
            _managed("good"): [_adapter("sft/flash-2-old/w", OLD)],  # reaped
        }
    )
    n = rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert n == 1
    assert api.deleted == [(_managed("good"), "sft/flash-2-old")]


# ---- TOCTOU re-confirmation -------------------------------------------------------------------


def test_per_delete_recheck_spares_prefix_deployed_midsweep(monkeypatch):
    target = (_managed("e"), "sft/flash-1-a")
    calls = {"n": 0}

    def _live():
        calls["n"] += 1
        # up-front confirm (1): only the sentinel is live. The pre-delete re-confirm (2+) now sees the
        # target as live -> it must be spared even though enumeration qualified it.
        if calls["n"] >= 2:
            return (_LIVE_SENTINEL[0] | {target}, set(), True)
        return _LIVE_SENTINEL

    _wire(monkeypatch, deployed=_live)
    api = FakeApi({target[0]: [_adapter(target[1] + "/w", OLD)]})
    assert rc.run_scheduled_cleanup(dry_run=False, api=api) == 0
    assert api.deleted == []


def test_midsweep_unconfirmable_live_set_aborts_whole_sweep(monkeypatch):
    calls = {"n": 0}

    def _live():
        calls["n"] += 1
        if calls["n"] <= 2:  # up-front + first pre-delete: OK
            return _LIVE_SENTINEL
        return (set(), set(), False)  # second pre-delete: incomplete -> abort the whole sweep

    _wire(monkeypatch, deployed=_live)
    api = FakeApi(
        {
            _managed("e1"): [_adapter("sft/flash-1-a/w", OLD)],
            _managed("e2"): [_adapter("sft/flash-2-b/w", OLD)],
        }
    )
    with pytest.raises(rc.CleanupAborted):
        rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert len(api.deleted) == 1  # one deleted before the abort; the sweep then stopped


def test_per_delete_recheck_spares_whole_repo_unmappable_midsweep(monkeypatch):
    # The mid-sweep re-confirm's whole-repo branch: a repo that gains a live-but-unmappable adapter
    # between enumeration and delete is protected wholesale.
    target = (_managed("e"), "sft/flash-1-a")
    calls = {"n": 0}

    def _live():
        calls["n"] += 1
        if calls["n"] >= 2:  # pre-delete re-confirm: the target's repo is now live-but-unmappable
            return (_LIVE_SENTINEL[0], {target[0]}, True)
        return _LIVE_SENTINEL

    _wire(monkeypatch, deployed=_live)
    api = FakeApi({target[0]: [_adapter(target[1] + "/w", OLD)]})
    assert rc.run_scheduled_cleanup(dry_run=False, api=api) == 0
    assert api.deleted == []


def test_one_delete_failure_does_not_abort_sweep(monkeypatch):
    # A single delete_folder failure (concurrent delete / rate-limit) is logged and skipped; the sweep
    # continues and reaps the remaining targets.
    _wire(monkeypatch)

    class _FlakyDeleteApi(FakeApi):
        def delete_folder(self, path_in_repo=None, repo_id=None, repo_type=None):
            if path_in_repo == "sft/flash-1-old":
                raise RuntimeError("HF 429 concurrent delete")
            super().delete_folder(path_in_repo=path_in_repo, repo_id=repo_id, repo_type=repo_type)

    api = _FlakyDeleteApi(
        {
            _managed("e1"): [
                _adapter("sft/flash-1-old/w", OLD)
            ],  # delete raises -> logged, continue
            _managed("e2"): [_adapter("sft/flash-2-old/w", OLD)],  # still reaped
        }
    )
    n = rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert n == 1
    assert api.deleted == [(_managed("e2"), "sft/flash-2-old")]


# ---- per-prefix re-stat -----------------------------------------------------------------------


def test_restat_spares_prefix_written_since_enumeration(monkeypatch):
    _wire(monkeypatch)
    monkeypatch.setattr(
        rc, "_prefix_written_within", lambda *a, **k: True
    )  # a fresh write appeared
    api = FakeApi({_managed("e"): [_adapter("sft/flash-1-old/w", OLD)]})
    assert rc.run_scheduled_cleanup(dry_run=False, api=api) == 0
    assert api.deleted == []


def test_restat_skips_target_when_it_errors(monkeypatch):
    _wire(monkeypatch)

    def _boom(*a, **k):
        raise RuntimeError("HF blip")

    monkeypatch.setattr(rc, "_prefix_written_within", _boom)
    api = FakeApi({_managed("e"): [_adapter("sft/flash-1-old/w", OLD)]})
    assert rc.run_scheduled_cleanup(dry_run=False, api=api) == 0  # skipped to stay safe
    assert api.deleted == []


def test_prefix_written_within_reads_newest_commit():
    api = FakeApi(
        {
            _managed("e"): [
                _adapter("sft/flash-1-a/old", 30),
                _adapter("sft/flash-1-a/fresh", 1),  # newest -> within 7d
            ]
        }
    )
    assert (
        rc._prefix_written_within(api, _managed("e"), "sft/flash-1-a", NOW, rc.DELETE_AGE_SECONDS)
        is True
    )


def test_prefix_written_within_false_when_all_old_or_undated():
    api = FakeApi({_managed("e"): [_adapter("sft/flash-1-a/w", 30)]})
    assert (
        rc._prefix_written_within(api, _managed("e"), "sft/flash-1-a", NOW, rc.DELETE_AGE_SECONDS)
        is False
    )
    undated = FakeApi({_managed("e"): [_adapter("sft/flash-1-a/w", None)]})
    assert (
        rc._prefix_written_within(
            undated, _managed("e"), "sft/flash-1-a", NOW, rc.DELETE_AGE_SECONDS
        )
        is False
    )


# ---- in-progress deploy/export guard ----------------------------------------------------------


def test_sweep_skips_run_with_deploy_or_export_in_progress(monkeypatch):
    _wire(monkeypatch, hold=lambda run_id: None if run_id == "flash-1-busy" else _DummyHeld())
    api = FakeApi(
        {
            _managed("e1"): [_adapter("sft/flash-1-busy/w", OLD)],  # locked -> skipped
            _managed("e2"): [_adapter("sft/flash-2-other/w", OLD)],  # reaped
        }
    )
    n = rc.run_scheduled_cleanup(dry_run=False, api=api)
    assert n == 1
    assert api.deleted == [(_managed("e2"), "sft/flash-2-other")]


def test_cooperative_stop_halts_before_deleting(monkeypatch):
    _wire(monkeypatch)
    api = FakeApi(
        {
            _managed("e1"): [_adapter("sft/flash-1-old/w", OLD)],
            _managed("e2"): [_adapter("sft/flash-2-old/w", OLD)],
        }
    )
    n = rc.run_scheduled_cleanup(dry_run=False, api=api, should_stop=lambda: True)
    assert n == 0
    assert api.deleted == []


# ---- the real per-run lock --------------------------------------------------------------------


def test_hold_run_lock_is_nonblocking_and_mutually_exclusive():
    from flash.server.platform.locks import _deploy_lock

    held = _REAL_HOLD_RUN_LOCK("flash-lock-a")  # free -> acquired and returned
    assert held is not None
    # A non-blocking acquire from outside must FAIL while the GC holds it (same run id, same lock).
    assert _deploy_lock("flash-lock-a").acquire(blocking=False) is False
    held.release()
    reacquired = _deploy_lock("flash-lock-a").acquire(blocking=False)  # released -> free again
    assert reacquired is True
    _deploy_lock("flash-lock-a").release()
