"""Operator GC for per-run HF artifact repos must reclaim space without ever touching an adapter
that serving is currently using. These tests pin the safety gates (deployed / in-flight / age /
allowlist), the per-tier actions, dry-run inertness, and the fail-safe abort when the serving live
set can't be confirmed."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from flash.server import repo_cleanup as rc

NS = "Freesolo-Co"
NOW = datetime(2026, 6, 27, 12, 0, 0, tzinfo=UTC)


def _days_ago(d: float) -> datetime:
    return NOW - timedelta(days=d)


def _hours_ago(h: float) -> datetime:
    return NOW - timedelta(hours=h)


class _Sib:
    def __init__(self, rfilename: str, size: int):
        self.rfilename = rfilename
        self.size = size


class _Info:
    def __init__(self, last_modified, files):
        self.last_modified = last_modified
        self.siblings = [_Sib(p, s) for p, s in files]


class _DS:
    def __init__(self, id_: str):
        self.id = id_


class FakeApi:
    """Minimal stand-in for HfApi: serves a fixed repo table and records mutations."""

    def __init__(self, repos: dict):
        # repos: {repo_id: (last_modified, [(path, size), ...])}
        self.repos = repos
        self.deleted_repos: list[str] = []
        self.deleted_folders: list[tuple[str, str]] = []

    def list_datasets(self, author=None):
        return [_DS(rid) for rid in self.repos]

    def repo_info(self, repo_id, repo_type=None, files_metadata=None):
        lm, files = self.repos[repo_id]
        return _Info(lm, files)

    def delete_repo(self, repo_id=None, repo_type=None, missing_ok=None):
        self.deleted_repos.append(repo_id)

    def delete_folder(self, path_in_repo=None, repo_id=None, repo_type=None, commit_message=None):
        self.deleted_folders.append((repo_id, path_in_repo))


# ---- fixtures: a few representative repo shapes -----------------------------------------------

SUCCEEDED = (  # has final adapter + code + checkpoints, old, not deployed
    [("code/flash/__init__.py", 700_000), ("adapter/adapter_model.safetensors", 50_000_000),
     ("checkpoints/step-1/adapter/adapter_model.safetensors", 50_000_000),
     ("checkpoints/step-2/adapter/adapter_model.safetensors", 50_000_000), ("metrics.json", 2_000)]
)
FAILED = (  # never produced an adapter (failed/cancelled), old
    [("code/flash/__init__.py", 700_000), ("console_train.txt", 64_000)]
)


def _view(repo_id, last_modified, files):
    return rc.RepoView(repo_id=repo_id, last_modified=last_modified, files=files)


def _cfg(**kw):
    return rc.Config(**kw)


# ---- KEEP-set gates ---------------------------------------------------------------------------

def test_deployed_repo_is_never_touched_even_if_old():
    v = _view(f"{NS}/flashrun-a", _days_ago(400), SUCCEEDED)
    cfg = _cfg(code=True, checkpoints=True, repos=True, delete_with_adapter=True)
    actions = rc.classify(v, deployed={f"{NS}/flashrun-a"}, cfg=cfg, now=NOW)
    assert len(actions) == 1
    assert actions[0].skipped
    assert "deployed" in actions[0].reason


def test_recently_written_repo_is_skipped_as_maybe_in_flight():
    v = _view(f"{NS}/flashrun-b", _hours_ago(1), SUCCEEDED)  # < 6h default inactive window
    actions = rc.classify(v, deployed=set(), cfg=_cfg(code=True, checkpoints=True, repos=True), now=NOW)
    assert len(actions) == 1
    assert actions[0].skipped
    assert "in-flight" in actions[0].reason


# ---- T1: code purge ---------------------------------------------------------------------------

def test_code_purge_targets_only_code_flash():
    v = _view(f"{NS}/flashrun-c", _days_ago(30), SUCCEEDED)
    actions = rc.classify(v, deployed=set(), cfg=_cfg(code=True), now=NOW)
    assert [(a.kind, a.path) for a in actions] == [("delete_folder", rc.CODE_PATH)]
    assert actions[0].reclaim_bytes == 700_000  # only the code bytes, not the adapter


def test_code_purge_runs_even_on_deployed_is_blocked_by_deployed_gate():
    # The deployed gate wins: even though code/ is serving-safe, a deployed repo is left entirely
    # untouched (simplest invariant — we never write to a live repo at all).
    v = _view(f"{NS}/flashrun-c2", _days_ago(30), SUCCEEDED)
    actions = rc.classify(v, deployed={f"{NS}/flashrun-c2"}, cfg=_cfg(code=True), now=NOW)
    assert actions[0].skipped


# ---- T2: checkpoint trim ----------------------------------------------------------------------

def test_checkpoint_trim_keeps_adapter_and_drops_checkpoints():
    v = _view(f"{NS}/flashrun-d", _days_ago(30), SUCCEEDED)  # > 14d trim age
    actions = rc.classify(v, deployed=set(), cfg=_cfg(code=False, checkpoints=True), now=NOW)
    paths = [(a.kind, a.path) for a in actions]
    assert ("delete_folder", rc.CHECKPOINTS_PATH) in paths
    assert ("delete_folder", rc.ADAPTER_PATH) not in paths  # adapter is preserved
    ckpt = next(a for a in actions if a.path == rc.CHECKPOINTS_PATH)
    assert ckpt.reclaim_bytes == 100_000_000  # both step checkpoints, not the final adapter


def test_checkpoint_trim_respects_trim_age():
    v = _view(f"{NS}/flashrun-e", _days_ago(5), SUCCEEDED)  # younger than 14d → no trim
    actions = rc.classify(v, deployed=set(), cfg=_cfg(code=False, checkpoints=True), now=NOW)
    assert len(actions) == 1
    assert actions[0].skipped


# ---- T3: whole-repo deletion ------------------------------------------------------------------

def test_failed_run_without_adapter_is_whole_deleted():
    v = _view(f"{NS}/flashrun-f", _days_ago(90), FAILED)  # no adapter/, > 60d
    actions = rc.classify(v, deployed=set(), cfg=_cfg(code=True, repos=True), now=NOW)
    assert [(a.kind, a.path) for a in actions] == [("delete_repo", None)]
    assert actions[0].reclaim_bytes == 764_000  # whole repo


def test_repo_with_adapter_not_whole_deleted_by_default():
    v = _view(f"{NS}/flashrun-g", _days_ago(90), SUCCEEDED)  # has adapter/ → protected from T3
    actions = rc.classify(v, deployed=set(), cfg=_cfg(code=True, checkpoints=True, repos=True), now=NOW)
    kinds = {a.kind for a in actions}
    assert "delete_repo" not in kinds  # adapter-bearing repo is never whole-deleted by default
    assert ("delete_folder", rc.CODE_PATH) in [(a.kind, a.path) for a in actions]


def test_delete_with_adapter_opt_in_whole_deletes_old_undeployed():
    v = _view(f"{NS}/flashrun-h", _days_ago(90), SUCCEEDED)
    cfg = _cfg(code=True, checkpoints=True, repos=True, delete_with_adapter=True)
    actions = rc.classify(v, deployed=set(), cfg=cfg, now=NOW)
    assert [(a.kind, a.path) for a in actions] == [("delete_repo", None)]


# ---- allowlist + end-to-end -------------------------------------------------------------------

def test_non_flashrun_repos_are_ignored():
    api = FakeApi({
        f"{NS}/flashrun-x": (_days_ago(90), FAILED),
        f"{NS}/paper-gsm8k": (_days_ago(90), [("data/train.parquet", 9_000_000)]),
        f"{NS}/some-env-package": (_days_ago(90), [("env.py", 1_000)]),
    })
    views = rc.list_run_repos(api, NS)
    assert [v.repo_id for v in views] == [f"{NS}/flashrun-x"]


def test_dry_run_makes_no_mutations(monkeypatch):
    api = FakeApi({f"{NS}/flashrun-f": (_days_ago(90), FAILED)})
    monkeypatch.setattr(rc, "deployed_repo_ids", lambda: set())
    plan = rc.run(_cfg(code=True, repos=True), dry_run=True, sleep=0, api=api)
    assert plan.actions
    assert plan.actions[0].kind == "delete_repo"
    assert api.deleted_repos == []  # nothing actually deleted
    assert api.deleted_folders == []


def test_apply_executes_planned_actions(monkeypatch):
    api = FakeApi({
        f"{NS}/flashrun-f": (_days_ago(90), FAILED),       # → delete_repo
        f"{NS}/flashrun-d": (_days_ago(30), SUCCEEDED),    # → trim checkpoints + code
    })
    monkeypatch.setattr(rc, "deployed_repo_ids", lambda: set())
    rc.run(_cfg(code=True, checkpoints=True, repos=True), dry_run=False, sleep=0, api=api)
    assert api.deleted_repos == [f"{NS}/flashrun-f"]
    assert (f"{NS}/flashrun-d", rc.CHECKPOINTS_PATH) in api.deleted_folders
    assert (f"{NS}/flashrun-d", rc.CODE_PATH) in api.deleted_folders


# ---- fail-safe: serving live set unavailable --------------------------------------------------

def _raise_serving():
    raise RuntimeError("serving unreachable")


def test_abort_when_live_set_unconfirmed_and_destructive_tier_requested(monkeypatch):
    api = FakeApi({f"{NS}/flashrun-d": (_days_ago(30), SUCCEEDED)})
    monkeypatch.setattr(rc, "deployed_repo_ids", _raise_serving)
    with pytest.raises(rc.CleanupAborted):
        rc.run(_cfg(code=True, checkpoints=True), dry_run=True, sleep=0, api=api)
    assert api.deleted_folders == []
    assert api.deleted_repos == []


def test_code_only_tier_proceeds_when_live_set_unavailable(monkeypatch):
    # code/ is never read by serving, so its purge is safe even without the live set.
    api = FakeApi({f"{NS}/flashrun-d": (_days_ago(30), SUCCEEDED)})
    monkeypatch.setattr(rc, "deployed_repo_ids", _raise_serving)
    plan = rc.run(_cfg(code=True, checkpoints=False, repos=False), dry_run=True, sleep=0, api=api)
    assert [(a.kind, a.path) for a in plan.actions] == [("delete_folder", rc.CODE_PATH)]


# ---- nested upload layout (worker writes under hf_prefix = {phase}/{run_id}/seed{N}/) ----------

# The real repo tree never has code/flash, adapter, checkpoints at the ROOT — every artifact nests
# under hf_prefix(). A root-anchored matcher would silently match nothing and the GC would no-op.
NESTED_SUCCEEDED = [
    ("rl/flash-9-zzzz/seed0/code/flash/__init__.py", 700_000),
    ("rl/flash-9-zzzz/seed0/adapter/adapter_model.safetensors", 50_000_000),
    ("rl/flash-9-zzzz/seed0/checkpoints/step-1/adapter/adapter_model.safetensors", 50_000_000),
    ("rl/flash-9-zzzz/seed0/checkpoints/step-2/adapter/adapter_model.safetensors", 50_000_000),
    ("rl/flash-9-zzzz/seed0/metrics.json", 2_000),
]
NESTED_FAILED = [
    ("sft/flash-9-yyyy/seed0/code/flash/__init__.py", 700_000),
    ("sft/flash-9-yyyy/seed0/console_train.txt", 64_000),
]


def test_nested_code_purge_targets_the_real_folder_path():
    v = _view(f"{NS}/flashrun-n", _days_ago(30), NESTED_SUCCEEDED)
    actions = rc.classify(v, deployed=set(), cfg=_cfg(code=True), now=NOW)
    assert [(a.kind, a.path) for a in actions] == [
        ("delete_folder", "rl/flash-9-zzzz/seed0/code/flash")
    ]
    assert actions[0].reclaim_bytes == 700_000


def test_nested_checkpoint_trim_targets_real_path_and_keeps_adapter():
    v = _view(f"{NS}/flashrun-n2", _days_ago(30), NESTED_SUCCEEDED)
    actions = rc.classify(v, deployed=set(), cfg=_cfg(code=False, checkpoints=True), now=NOW)
    assert [(a.kind, a.path) for a in actions] == [
        ("delete_folder", "rl/flash-9-zzzz/seed0/checkpoints")
    ]
    assert actions[0].reclaim_bytes == 100_000_000  # both step checkpoints, not the final adapter


def test_nested_adapter_detected_so_t3_protects_repo():
    v = _view(f"{NS}/flashrun-n3", _days_ago(90), NESTED_SUCCEEDED)
    cfg = _cfg(code=True, checkpoints=True, repos=True)
    actions = rc.classify(v, deployed=set(), cfg=cfg, now=NOW)
    assert "delete_repo" not in {a.kind for a in actions}  # nested adapter/ is seen → protected


def test_nested_failed_run_without_adapter_is_whole_deleted():
    v = _view(f"{NS}/flashrun-n4", _days_ago(90), NESTED_FAILED)
    actions = rc.classify(v, deployed=set(), cfg=_cfg(code=True, repos=True), now=NOW)
    assert [(a.kind, a.path) for a in actions] == [("delete_repo", None)]


def test_multiple_seeds_each_get_their_own_delete_folder():
    files = [
        ("rl/flash-9-multi/seed0/code/flash/a.py", 100),
        ("rl/flash-9-multi/seed1/code/flash/a.py", 200),
    ]
    v = _view(f"{NS}/flashrun-n5", _days_ago(30), files)
    actions = rc.classify(v, deployed=set(), cfg=_cfg(code=True), now=NOW)
    paths = sorted(a.path for a in actions)
    assert paths == ["rl/flash-9-multi/seed0/code/flash", "rl/flash-9-multi/seed1/code/flash"]


def test_bare_file_named_like_artifact_is_not_treated_as_folder():
    # A file literally named "adapter" (no children) is not an adapter FOLDER.
    v = _view(f"{NS}/flashrun-n6", _days_ago(90), [("rl/r/seed0/adapter", 10)])
    assert v.has(rc.ADAPTER_PATH) is False


# ---- CLI: --delete-with-adapter requires --repos ----------------------------------------------

def test_delete_with_adapter_does_not_force_live_set_on_its_own():
    # On its own it acts on nothing, so it must not require the serving live set.
    assert _cfg(delete_with_adapter=True).needs_live_set is False
    assert _cfg(repos=True, delete_with_adapter=True).needs_live_set is True


def test_main_rejects_delete_with_adapter_without_repos(capsys):
    rc_code = rc.main(["--delete-with-adapter"])
    assert rc_code == 2
    assert "--repos" in capsys.readouterr().err


def test_main_accepts_delete_with_adapter_with_repos(monkeypatch):
    # Should pass validation and run (dry-run); we stub the live set + HF api.
    monkeypatch.setattr(rc, "deployed_repo_ids", lambda: set())
    monkeypatch.setattr(rc, "list_run_repos", lambda api, ns: [])
    rc_code = rc.main(["--repos", "--delete-with-adapter"])
    assert rc_code == 0
