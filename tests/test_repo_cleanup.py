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


@pytest.fixture(autouse=True)
def _frozen_now(monkeypatch):
    # run()/main() compute age from datetime.now(UTC); freeze it to NOW so the fixtures' relative
    # ages are deterministic and never depend on the wall clock (classify tests pass now=NOW
    # explicitly and are unaffected).
    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW

    monkeypatch.setattr(rc, "datetime", _Frozen)


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

SUCCEEDED = (  # has a COMPLETE final adapter (config+weights) + code + checkpoints, old, not deployed
    [("code/flash/__init__.py", 700_000),
     ("adapter/adapter_config.json", 500), ("adapter/adapter_model.safetensors", 50_000_000),
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
    v = _view(f"{NS}/flashrun-c", _days_ago(90), SUCCEEDED)
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
    monkeypatch.setattr(rc, "deployed_repo_ids", lambda: (set(), True))
    plan = rc.run(_cfg(code=True, repos=True), dry_run=True, sleep=0, api=api)
    assert plan.actions
    assert plan.actions[0].kind == "delete_repo"
    assert api.deleted_repos == []  # nothing actually deleted
    assert api.deleted_folders == []


def test_apply_executes_planned_actions(monkeypatch):
    api = FakeApi({
        f"{NS}/flashrun-f": (_days_ago(90), FAILED),       # → delete_repo
        f"{NS}/flashrun-d": (_days_ago(90), SUCCEEDED),    # → trim checkpoints + code
    })
    monkeypatch.setattr(rc, "deployed_repo_ids", lambda: (set(), True))
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
    api = FakeApi({f"{NS}/flashrun-d": (_days_ago(90), SUCCEEDED)})
    monkeypatch.setattr(rc, "deployed_repo_ids", _raise_serving)
    plan = rc.run(_cfg(code=True, checkpoints=False, repos=False), dry_run=True, sleep=0, api=api)
    assert [(a.kind, a.path) for a in plan.actions] == [("delete_folder", rc.CODE_PATH)]


# ---- nested upload layout (worker writes under hf_prefix = {phase}/{run_id}/seed{N}/) ----------

# The real repo tree never has code/flash, adapter, checkpoints at the ROOT — every artifact nests
# under hf_prefix(). A root-anchored matcher would silently match nothing and the GC would no-op.
NESTED_SUCCEEDED = [
    ("rl/flash-9-zzzz/seed0/code/flash/__init__.py", 700_000),
    ("rl/flash-9-zzzz/seed0/adapter/adapter_config.json", 500),
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
    v = _view(f"{NS}/flashrun-n", _days_ago(90), NESTED_SUCCEEDED)
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
    v = _view(f"{NS}/flashrun-n5", _days_ago(90), files)  # delete-age old → shared code reclaimable
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
    monkeypatch.setattr(rc, "deployed_repo_ids", lambda: (set(), True))
    monkeypatch.setattr(rc, "list_run_repos", lambda api, ns: [])
    rc_code = rc.main(["--repos", "--delete-with-adapter"])
    assert rc_code == 0


# ---- queued-run protection: don't purge code from a repo that hasn't booted -------------------

CODE_ONLY = [("code/flash/__init__.py", 700_000)]  # control plane uploaded code; worker not booted


def test_code_not_purged_before_delete_age_even_when_booted():
    # The shared root code/flash is needed by every seed + resume for the whole run lifetime, so it
    # is held until delete-age regardless of run output — a later seed / inter-seed-gap resume still
    # needs it. (SUCCEEDED has adapter+metrics, i.e. clearly booted, yet code is NOT purged at 30d.)
    v = _view(f"{NS}/flashrun-q", _days_ago(30), SUCCEEDED)  # past inactive window, < 60d delete age
    actions = rc.classify(v, deployed=set(), cfg=_cfg(code=True), now=NOW)
    assert ("delete_folder", rc.CODE_PATH) not in [(a.kind, a.path) for a in actions]


def test_code_purged_once_delete_age_old():
    v = _view(f"{NS}/flashrun-q2", _days_ago(90), SUCCEEDED)  # > 60d delete age → run provably done
    actions = rc.classify(v, deployed=set(), cfg=_cfg(code=True), now=NOW)
    assert ("delete_folder", rc.CODE_PATH) in [(a.kind, a.path) for a in actions]


def test_code_only_queued_repo_is_held_before_delete_age():
    # A repo with ONLY code/flash may be a run still queued for capacity; hold its code until
    # delete-age so its worker can still boot.
    v = _view(f"{NS}/flashrun-q3", _days_ago(30), CODE_ONLY)
    actions = rc.classify(v, deployed=set(), cfg=_cfg(code=True), now=NOW)
    assert len(actions) == 1
    assert actions[0].skipped


# ---- checkpoint-only repos are servable (deployable via `flash deploy --step N`) ---------------

CHECKPOINT_ONLY = [  # no final adapter/, but a COMPLETE (config+weights) per-step checkpoint adapter
    ("rl/flash-9-ck/seed0/code/flash/__init__.py", 700_000),
    ("rl/flash-9-ck/seed0/checkpoints/step-1/adapter/adapter_config.json", 500),
    ("rl/flash-9-ck/seed0/checkpoints/step-1/adapter/adapter_model.safetensors", 50_000_000),
    ("rl/flash-9-ck/seed0/metrics.json", 2_000),
]


PARTIAL_FINAL_ADAPTER = [  # final adapter upload only partially landed (config, NO weights) + ckpts
    ("rl/flash-9-pf/seed0/adapter/adapter_config.json", 500),  # no adapter_model.* → not deployable
    ("rl/flash-9-pf/seed0/checkpoints/step-1/adapter/adapter_config.json", 500),
    ("rl/flash-9-pf/seed0/checkpoints/step-1/adapter/adapter_model.safetensors", 50_000_000),
    ("rl/flash-9-pf/seed0/metrics.json", 2_000),
]


def test_incomplete_final_adapter_does_not_trigger_checkpoint_trim():
    # The final adapter is config-only (weights upload failed). Trimming checkpoints would leave NO
    # deployable artifact, so T2 must not fire — the checkpoint adapter is the only servable one.
    v = _view(f"{NS}/flashrun-pf", _days_ago(30), PARTIAL_FINAL_ADAPTER)
    actions = rc.classify(v, deployed=set(), cfg=_cfg(code=False, checkpoints=True), now=NOW)
    assert "delete_folder" not in {a.kind for a in actions}  # no checkpoint trim
    assert not v.has_final_adapter()  # config-only final adapter is not "complete"
    assert v.has_servable_adapter()  # the complete checkpoint adapter still makes the repo servable


def test_checkpoint_only_repo_is_not_whole_deleted_as_unservable():
    v = _view(f"{NS}/flashrun-ck", _days_ago(90), CHECKPOINT_ONLY)
    actions = rc.classify(v, deployed=set(), cfg=_cfg(code=True, repos=True), now=NOW)
    assert "delete_repo" not in {a.kind for a in actions}  # checkpoint adapter is servable → kept


def test_checkpoint_only_repo_is_not_checkpoint_trimmed():
    # Trimming a checkpoint-only repo's checkpoints would strip its sole servable content.
    v = _view(f"{NS}/flashrun-ck2", _days_ago(30), CHECKPOINT_ONLY)
    actions = rc.classify(v, deployed=set(), cfg=_cfg(code=False, checkpoints=True), now=NOW)
    assert "delete_folder" not in {a.kind for a in actions}


def test_checkpoint_only_repo_whole_deleted_under_delete_with_adapter():
    v = _view(f"{NS}/flashrun-ck3", _days_ago(90), CHECKPOINT_ONLY)
    cfg = _cfg(code=True, repos=True, delete_with_adapter=True)
    actions = rc.classify(v, deployed=set(), cfg=cfg, now=NOW)
    assert [(a.kind, a.path) for a in actions] == [("delete_repo", None)]


def test_repoview_adapter_helpers():
    final = _view("x", NOW, SUCCEEDED)
    assert final.has_servable_adapter()
    assert final.has_final_adapter()
    ckpt = _view("x", NOW, CHECKPOINT_ONLY)
    assert ckpt.has_servable_adapter()
    assert not ckpt.has_final_adapter()  # only a checkpoint adapter, no final adapter/
    bare = _view("x", NOW, CODE_ONLY)
    assert not bare.has_servable_adapter()
    assert not bare.has_final_adapter()


# ---- NqG: re-confirm the live set right before applying destructive actions --------------------

def test_apply_skips_repo_that_became_deployed_after_enumeration(monkeypatch):
    # First sample (enumeration) sees nothing deployed; the pre-apply re-check sees flashrun-d live.
    api = FakeApi({f"{NS}/flashrun-d": (_days_ago(30), SUCCEEDED)})
    calls = {"n": 0}

    def fake_deployed():
        calls["n"] += 1
        return (set(), True) if calls["n"] == 1 else ({f"{NS}/flashrun-d"}, True)

    monkeypatch.setattr(rc, "deployed_repo_ids", fake_deployed)
    rc.run(_cfg(code=True, checkpoints=True, repos=True), dry_run=False, sleep=0, api=api)
    assert api.deleted_folders == []  # the now-deployed repo's actions were dropped
    assert api.deleted_repos == []
    assert calls["n"] == 2  # sampled once, re-confirmed once before apply


# ---- live keep-set extraction: best-effort set + completeness flag ----------------------------

def test_deployed_repo_ids_flags_record_without_repo_id_as_incomplete(monkeypatch):
    from flash.serve import deploy

    # A record with no repo id can't be mapped; the set keeps the mappable ones but complete=False.
    monkeypatch.setattr(
        deploy, "list_deployed_adapters",
        lambda: [{"repoId": "Freesolo-Co/flashrun-a"}, {"adapterId": "x"}],  # 2nd has no repoId
    )
    ids, complete = rc.deployed_repo_ids()
    assert ids == {"Freesolo-Co/flashrun-a"}
    assert complete is False


def test_deployed_repo_ids_flags_malformed_truthy_repo_id_as_incomplete(monkeypatch):
    from flash.serve import deploy

    # Schema drift: a truthy but non-string repoId (here a dict) must NOT be str()-ified into the
    # keep-set — that token can't match the real Freesolo-Co/flashrun-* id, so the live repo would
    # look un-deployed and become delete-eligible. Fail closed (complete=False) like a missing key.
    monkeypatch.setattr(
        deploy, "list_deployed_adapters",
        lambda: [
            {"repoId": "Freesolo-Co/flashrun-a"},
            {"repoId": {"name": "flashrun-b"}},  # truthy non-string
            {"repoId": "   "},  # blank
        ],
    )
    ids, complete = rc.deployed_repo_ids()
    assert ids == {"Freesolo-Co/flashrun-a"}  # only the well-formed id, no "{'name': ...}" token
    assert complete is False


def test_deployed_repo_ids_collects_repo_ids(monkeypatch):
    from flash.serve import deploy

    monkeypatch.setattr(
        deploy, "list_deployed_adapters",
        lambda: [{"repoId": "Freesolo-Co/flashrun-a"}, {"repo_id": "Freesolo-Co/flashrun-b"}],
    )
    ids, complete = rc.deployed_repo_ids()
    assert ids == {"Freesolo-Co/flashrun-a", "Freesolo-Co/flashrun-b"}
    assert complete is True


def test_incomplete_live_set_aborts_destructive_tier_but_not_code(monkeypatch):
    # The WiB regression: an unmappable live record must NOT collapse the code-only keep-set to
    # empty. Destructive tiers abort; --code proceeds with the best-effort (still-protective) set.
    from flash.serve import deploy

    monkeypatch.setattr(
        deploy, "list_deployed_adapters",
        lambda: [{"repoId": f"{NS}/flashrun-live"}, {"adapterId": "unmappable"}],
    )
    # destructive tier → abort
    api = FakeApi({f"{NS}/flashrun-d": (_days_ago(90), SUCCEEDED)})
    with pytest.raises(rc.CleanupAborted):
        rc.run(_cfg(code=True, checkpoints=True), dry_run=True, sleep=0, api=api)
    # code-only → proceeds, and the identifiable deployed repo is still protected (skipped)
    api2 = FakeApi({f"{NS}/flashrun-live": (_days_ago(90), SUCCEEDED)})
    plan = rc.run(_cfg(code=True, checkpoints=False, repos=False), dry_run=True, sleep=0, api=api2)
    assert plan.actions == []  # flashrun-live is in the best-effort keep-set → not purged
    assert any("deployed" in s.reason for s in plan.skips)


def test_incomplete_live_set_reports_incomplete_not_confirmed(monkeypatch, capsys):
    # The report must not say "confirmed" when the keep-set is incomplete (false sense of safety).
    from flash.serve import deploy

    monkeypatch.setattr(
        deploy, "list_deployed_adapters",
        lambda: [{"repoId": f"{NS}/flashrun-live"}, {"adapterId": "unmappable"}],
    )
    api = FakeApi({f"{NS}/flashrun-c": (_days_ago(90), SUCCEEDED)})
    rc.run(_cfg(code=True), dry_run=True, sleep=0, api=api)
    out = capsys.readouterr().out
    assert "INCOMPLETE" in out
    assert "live set: confirmed" not in out
    # cfg has --code on, so the consequence names it
    assert "--code still proceeds" in out


def test_unavailable_live_set_report_no_code_does_not_imply_code_tier(monkeypatch, capsys):
    # With --no-code and no destructive tiers, an unavailable live set must NOT print the old
    # hard-coded "code-only tier" — it should report the consequence for the tiers actually
    # requested (destructive tiers need the live set and would abort, so they can't co-occur here).
    from flash.serve import deploy

    def _raise(*_a, **_k):
        raise RuntimeError("serving down")

    monkeypatch.setattr(deploy, "list_deployed_adapters", _raise)
    api = FakeApi({f"{NS}/flashrun-c": (_days_ago(90), SUCCEEDED)})
    rc.run(_cfg(code=False, checkpoints=False, repos=False), dry_run=True, sleep=0, api=api)
    out = capsys.readouterr().out
    assert "UNAVAILABLE" in out
    assert "--code" not in out  # never imply the code tier the operator disabled
    assert "no tiers active" in out


# ---- unknown age + invalid retention windows --------------------------------------------------

def test_repo_with_no_last_modified_is_skipped_for_safety():
    # HF returning last_modified=None (fresh/queued repo) must not look "infinitely old" and become
    # eligible for deletion. It's left untouched.
    v = _view(f"{NS}/flashrun-u", None, FAILED)  # no adapter, but unknown age
    actions = rc.classify(v, deployed=set(), cfg=_cfg(code=True, checkpoints=True, repos=True), now=NOW)
    assert len(actions) == 1
    assert actions[0].skipped
    assert "unknown age" in actions[0].reason


def test_age_seconds_treats_missing_timestamp_as_just_written():
    assert _view("x", None, FAILED).age_seconds(NOW) == 0.0


def test_main_rejects_negative_retention_windows(capsys):
    assert rc.main(["--delete-age-days", "-1"]) == 2
    assert "--delete-age-days must be >= 0" in capsys.readouterr().err
    assert rc.main(["--checkpoints", "--trim-age-days", "-5"]) == 2
    assert "--trim-age-days must be >= 0" in capsys.readouterr().err


# ---- T2 also trims the singular resume checkpoint; per-action live re-check --------------------

# A successful run that also left the full-trainer resume checkpoint under the singular path.
SUCCEEDED_WITH_RESUME = [
    *SUCCEEDED,
    ("checkpoint/checkpoint-5/optimizer.pt", 80_000_000),
    ("checkpoint/checkpoint-5/adapter_model.safetensors", 50_000_000),
]


def test_plural_checkpoints_trimmed_at_trim_age_resume_held_to_delete_age():
    # Plural deployable snapshots trim at trim-age; the singular resume checkpoint is held until
    # delete-age (a delayed preemption retry could still re-download it).
    v = _view(f"{NS}/flashrun-rs", _days_ago(30), SUCCEEDED_WITH_RESUME)  # > trim-age, < delete-age
    actions = rc.classify(v, deployed=set(), cfg=_cfg(code=False, checkpoints=True), now=NOW)
    paths = {(a.kind, a.path) for a in actions}
    assert ("delete_folder", rc.CHECKPOINTS_PATH) in paths  # plural deployable snapshots trimmed
    assert ("delete_folder", rc.RESUME_CHECKPOINT_PATH) not in paths  # resume state HELD (< delete-age)
    assert ("delete_folder", rc.ADAPTER_PATH) not in paths  # final adapter kept


def test_resume_checkpoint_trimmed_at_delete_age():
    v = _view(f"{NS}/flashrun-rs", _days_ago(90), SUCCEEDED_WITH_RESUME)  # > delete-age
    actions = rc.classify(v, deployed=set(), cfg=_cfg(code=False, checkpoints=True), now=NOW)
    paths = {(a.kind, a.path) for a in actions}
    assert ("delete_folder", rc.RESUME_CHECKPOINT_PATH) in paths  # resume state trimmed once abandoned


def test_resume_checkpoint_trimmed_even_without_final_adapter_at_delete_age():
    # The singular resume checkpoint is never servable, so trim it even for a checkpoint-only repo —
    # but only at delete-age.
    files = [*CHECKPOINT_ONLY, ("rl/flash-9-ck/seed0/checkpoint/checkpoint-3/optimizer.pt", 80_000_000)]
    v = _view(f"{NS}/flashrun-rs2", _days_ago(90), files)
    actions = rc.classify(v, deployed=set(), cfg=_cfg(code=False, checkpoints=True), now=NOW)
    paths = {(a.kind, a.path) for a in actions}
    assert ("delete_folder", "rl/flash-9-ck/seed0/checkpoint") in paths  # resume state trimmed
    # the deployable plural checkpoints are NOT trimmed (no final adapter to fall back to)
    assert not any(a.path and a.path.endswith("checkpoints") for a in actions)


def test_multiseed_checkpoint_trim_is_per_prefix():
    # seed0 has a complete final adapter; seed1 has only deployable checkpoint adapters (no final).
    # Only seed0's checkpoints may be trimmed — seed1's are its sole servable content.
    files = [
        ("rl/run/seed0/adapter/adapter_config.json", 500),
        ("rl/run/seed0/adapter/adapter_model.safetensors", 50_000_000),
        ("rl/run/seed0/checkpoints/step-1/adapter/adapter_model.safetensors", 50_000_000),
        ("rl/run/seed1/checkpoints/step-1/adapter/adapter_config.json", 500),
        ("rl/run/seed1/checkpoints/step-1/adapter/adapter_model.safetensors", 50_000_000),
    ]
    v = _view(f"{NS}/flashrun-ms", _days_ago(30), files)
    actions = rc.classify(v, deployed=set(), cfg=_cfg(code=False, checkpoints=True), now=NOW)
    trimmed = {a.path for a in actions if a.kind == "delete_folder"}
    assert "rl/run/seed0/checkpoints" in trimmed  # seed0 has a final adapter → trimmable
    assert "rl/run/seed1/checkpoints" not in trimmed  # seed1 has no final adapter → kept


def test_apply_per_action_recheck_skips_repo_deployed_mid_apply(monkeypatch):
    # A repo not deployed at the pre-apply sample but deployed by the time its action runs must be
    # skipped (per-action just-in-time re-check), not deleted.
    api = FakeApi({
        f"{NS}/flashrun-f": (_days_ago(90), FAILED),   # → delete_repo
        f"{NS}/flashrun-d": (_days_ago(90), SUCCEEDED),  # → checkpoint trim + code purge
    })
    calls = {"n": 0}

    def fake_deployed():
        calls["n"] += 1
        # samples 1 (initial) and 2 (pre-apply bulk) see nothing; from the per-action re-check on,
        # flashrun-f is live.
        return (set(), True) if calls["n"] <= 2 else ({f"{NS}/flashrun-f"}, True)

    monkeypatch.setattr(rc, "deployed_repo_ids", fake_deployed)
    rc.run(_cfg(code=True, checkpoints=True, repos=True), dry_run=False, sleep=0, api=api)
    assert api.deleted_repos == []  # flashrun-f was skipped by the per-action re-check
    assert all(r != f"{NS}/flashrun-f" for r, _ in api.deleted_folders)


def test_apply_per_action_recheck_failure_skips_to_stay_safe(monkeypatch):
    api = FakeApi({f"{NS}/flashrun-f": (_days_ago(90), FAILED)})
    calls = {"n": 0}

    def fake_deployed():
        calls["n"] += 1
        if calls["n"] <= 2:
            return (set(), True)  # initial + pre-apply bulk succeed
        raise RuntimeError("serving down mid-apply")  # per-action re-check fails

    monkeypatch.setattr(rc, "deployed_repo_ids", fake_deployed)
    rc.run(_cfg(repos=True), dry_run=False, sleep=0, api=api)
    assert api.deleted_repos == []  # skipped to stay safe, not deleted


def test_apply_per_action_recheck_incomplete_set_skips(monkeypatch):
    # An incomplete keep-set on the per-action re-check must fail closed (don't delete against a
    # best-effort set during a destructive apply).
    api = FakeApi({f"{NS}/flashrun-f": (_days_ago(90), FAILED)})
    calls = {"n": 0}

    def fake_deployed():
        calls["n"] += 1
        # initial + pre-apply bulk are complete; the per-action re-check returns an incomplete set
        return (set(), True) if calls["n"] <= 2 else (set(), False)

    monkeypatch.setattr(rc, "deployed_repo_ids", fake_deployed)
    rc.run(_cfg(repos=True), dry_run=False, sleep=0, api=api)
    assert api.deleted_repos == []  # incomplete set → skipped, not deleted


def test_apply_recheck_refetches_per_action_after_failure(monkeypatch):
    # The live set is re-fetched for EVERY destructive action, so a failed re-check on one action
    # never lets a LATER action delete against a stale/empty set — each one re-checks fresh.
    api = FakeApi({
        f"{NS}/flashrun-f1": (_days_ago(90), FAILED),
        f"{NS}/flashrun-f2": (_days_ago(90), FAILED),
    })
    calls = {"n": 0}

    def fake_deployed():
        calls["n"] += 1
        # 1=initial, 2=pre-apply bulk; 3=first action's re-check FAILS; 4=second action re-fetches
        # (proving the cache was invalidated) and reports f2 as now-live.
        if calls["n"] == 3:
            raise RuntimeError("transient serving blip")
        if calls["n"] >= 4:
            return ({f"{NS}/flashrun-f2"}, True)
        return (set(), True)

    monkeypatch.setattr(rc, "deployed_repo_ids", fake_deployed)
    rc.run(_cfg(repos=True), dry_run=False, sleep=0, api=api)
    assert api.deleted_repos == []  # f1 skipped (re-check failed), f2 skipped (re-fetched → live)
    assert calls["n"] >= 4  # the 2nd action re-fetched rather than trusting the stale cache
