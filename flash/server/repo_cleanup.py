"""Operator-side garbage collection for per-run HF artifact repos (``Freesolo-Co/flashrun-*``).

Every managed run creates a *private* HF dataset repo ``Freesolo-Co/flashrun-<run_id>`` (see
``flash.runner._assign_managed_hf_repo``) that holds, under one repo:

- ``code/flash/``               the flash source snapshot the worker ran (needed only *during* training)
- ``adapter/``                  the final trained LoRA adapter (serving reads this)
- ``checkpoints/step-N/...``    per-step deployable checkpoints (resume + ``flash deploy --step N``)
- metrics / heartbeat / ``console_*.txt``   telemetry

Nothing ever deletes these, so they accumulate against the org's private-storage quota. This tool
reclaims that space **without ever removing an adapter that serving is currently using**.

Safety model
------------
The freesolo serving app pulls adapter weights straight from these dataset repos
(``flash/serve/deploy.py`` registers ``{repoId}:{subfolder}``). So the *live serving set* is the
authoritative do-not-touch set:

* A repo whose ``repoId`` is registered with serving is **never** modified or deleted.
* If the live set cannot be confirmed (serving unreachable), any tier that removes servable content
  **aborts** rather than guessing. The ``code/`` purge (which serving never reads) is the one
  exception and may still proceed.

Every repo is classified from three self-contained signals — no backend DB required:

1. the serving live set (``deploy.list_deployed_adapters``),
2. the repo's own file tree (does it have ``adapter/``? ``checkpoints/``? ``code/flash/``?),
3. the repo's ``last_modified`` age (a run still training writes heartbeats/checkpoints constantly,
   so a repo untouched for ``--inactive-age-hours`` is definitely not in-flight).

Tiers (conservative defaults; each opt-in beyond ``--code``)
-----------------------------------------------------------
* ``--code``        (default ON)  T1: delete ``code/flash`` from terminal repos. Serving never reads
                                  it, so this is safe even for deployed repos; reclaims little but is
                                  the safe default.
* ``--checkpoints`` (default off) T2: on terminal + undeployed + older than ``--trim-age-days``,
                                  delete ``checkpoints/`` while **keeping** ``adapter/``. The big
                                  byte win; you lose only ``flash deploy --step N`` for old runs.
* ``--repos``       (default off) T3: ``delete_repo`` for terminal + undeployed + older than
                                  ``--delete-age-days`` repos that have **no** ``adapter/`` (i.e.
                                  failed/cancelled runs that never produced a servable adapter).
* ``--delete-with-adapter`` (default off)  let T3 also whole-delete *old undeployed* repos that DO
                                  carry a final adapter. Off by default so a redeployable adapter is
                                  never dropped implicitly.

Operator-only: needs the ``Freesolo-Co`` operator ``HF_TOKEN`` (owns the repos) and
``FREESOLO_INTERNAL_KEY`` (queries serving). Run from the control-plane host::

    python -m flash.server.repo_cleanup                 # dry-run report, deletes nothing
    python -m flash.server.repo_cleanup --checkpoints --repos          # full plan, still dry-run
    python -m flash.server.repo_cleanup --checkpoints --repos --apply  # actually delete
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from flash._logging import get_logger
from flash.runner import _ARTIFACT_NAMESPACE

logger = get_logger(__name__)

RUN_REPO_PREFIX = "flashrun-"
CODE_PATH = "code/flash"
CHECKPOINTS_PATH = "checkpoints"
ADAPTER_PATH = "adapter"


class CleanupAborted(RuntimeError):
    """A destructive tier was requested but its safety precondition could not be met."""


@dataclass
class RepoView:
    """The facts about one ``flashrun-*`` repo needed to classify it, fetched in a single
    ``repo_info(files_metadata=True)`` call."""

    repo_id: str
    last_modified: datetime | None
    files: list[tuple[str, int]]  # (rfilename, size_bytes)

    @property
    def total_bytes(self) -> int:
        return sum(size for _, size in self.files)

    def _under(self, prefix: str) -> list[tuple[str, int]]:
        return [(f, s) for f, s in self.files if f == prefix or f.startswith(prefix + "/")]

    def has(self, prefix: str) -> bool:
        return bool(self._under(prefix))

    def bytes_under(self, prefix: str) -> int:
        return sum(s for _, s in self._under(prefix))

    def age_seconds(self, now: datetime) -> float:
        if self.last_modified is None:
            # No timestamp → treat as "very old" so the age gate never *protects* it; the deployed
            # and content gates still apply, so this can't cause an unsafe delete on its own.
            return float("inf")
        return (now - self.last_modified).total_seconds()


@dataclass
class Action:
    """One planned mutation against a repo."""

    repo_id: str
    kind: str  # "delete_folder" | "delete_repo"
    path: str | None  # folder for delete_folder, None for delete_repo
    reclaim_bytes: int
    reason: str
    skipped: bool = False  # set when a repo is intentionally left untouched (kind == "skip")


@dataclass
class Plan:
    actions: list[Action] = field(default_factory=list)
    skips: list[Action] = field(default_factory=list)

    @property
    def reclaim_bytes(self) -> int:
        return sum(a.reclaim_bytes for a in self.actions)


@dataclass
class Config:
    namespace: str = _ARTIFACT_NAMESPACE
    code: bool = True
    checkpoints: bool = False
    repos: bool = False
    delete_with_adapter: bool = False
    inactive_age_hours: float = 6.0
    trim_age_days: float = 14.0
    delete_age_days: float = 60.0

    @property
    def needs_live_set(self) -> bool:
        """Tiers that remove servable content require the live serving set to be confirmed."""
        return self.checkpoints or self.repos or self.delete_with_adapter


def _human_bytes(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.1f}{unit}"
        f /= 1024
    return f"{f:.1f}TB"


def list_run_repos(api, namespace: str) -> list[RepoView]:
    """Enumerate every ``flashrun-*`` dataset under ``namespace`` with its file tree + age."""
    views: list[RepoView] = []
    for ds in api.list_datasets(author=namespace):
        repo_id = ds.id
        name = repo_id.split("/", 1)[-1]
        if not name.startswith(RUN_REPO_PREFIX):
            continue  # hard allowlist: never touch env packages, paper-*, oracle/eval sets
        try:
            info = api.repo_info(repo_id, repo_type="dataset", files_metadata=True)
        except Exception as exc:
            logger.warning("skipping %s: repo_info failed: %s", repo_id, exc)
            continue
        files = [
            (s.rfilename, int(getattr(s, "size", None) or 0)) for s in (info.siblings or [])
        ]
        views.append(RepoView(repo_id=repo_id, last_modified=info.last_modified, files=files))
    return views


def deployed_repo_ids() -> set[str]:
    """The set of HF repo ids serving is currently loading adapters from (the live keep-set)."""
    from flash.serve import deploy

    ids: set[str] = set()
    for rec in deploy.list_deployed_adapters():
        repo = rec.get("repoId") or rec.get("repo_id")
        if repo:
            ids.add(str(repo))
    return ids


def classify(view: RepoView, deployed: set[str], cfg: Config, now: datetime) -> list[Action]:
    """Decide what to do with one repo. Returns the planned actions (possibly a single
    ``kind="skip"`` action explaining why nothing is done)."""
    if view.repo_id in deployed:
        return [Action(view.repo_id, "skip", None, 0, "deployed (serving live set)", skipped=True)]

    age = view.age_seconds(now)
    if age < cfg.inactive_age_hours * 3600:
        return [Action(view.repo_id, "skip", None, 0, "recently written (maybe in-flight)", skipped=True)]

    # T3 first: a repo with NO final adapter never produced anything servable (failed/cancelled/empty).
    # Whole-deleting it reclaims everything and can't compromise serving. A repo WITH an adapter is
    # whole-deleted only under the explicit --delete-with-adapter opt-in.
    has_adapter = view.has(ADAPTER_PATH)
    if cfg.repos and age >= cfg.delete_age_days * 86400:
        if not has_adapter:
            return [Action(view.repo_id, "delete_repo", None, view.total_bytes, "terminal, no servable adapter")]
        if cfg.delete_with_adapter:
            return [Action(view.repo_id, "delete_repo", None, view.total_bytes, "old undeployed adapter (--delete-with-adapter)")]

    actions: list[Action] = []
    if cfg.code and view.has(CODE_PATH):
        actions.append(Action(view.repo_id, "delete_folder", CODE_PATH, view.bytes_under(CODE_PATH), "training source snapshot"))
    if cfg.checkpoints and age >= cfg.trim_age_days * 86400 and view.has(CHECKPOINTS_PATH):
        actions.append(Action(view.repo_id, "delete_folder", CHECKPOINTS_PATH, view.bytes_under(CHECKPOINTS_PATH), "intermediate checkpoints (final adapter kept)"))

    if not actions:
        return [Action(view.repo_id, "skip", None, 0, "nothing eligible", skipped=True)]
    return actions


def build_plan(views: list[RepoView], deployed: set[str], cfg: Config, now: datetime) -> Plan:
    plan = Plan()
    for view in views:
        for action in classify(view, deployed, cfg, now):
            (plan.skips if action.skipped else plan.actions).append(action)
    return plan


def apply_plan(api, plan: Plan, *, dry_run: bool, sleep: float) -> list[dict]:
    """Execute (or, when ``dry_run``, only record) every action. Returns a manifest of results."""
    manifest: list[dict] = []
    for action in plan.actions:
        record = {
            "repo_id": action.repo_id,
            "kind": action.kind,
            "path": action.path,
            "reclaim_bytes": action.reclaim_bytes,
            "reason": action.reason,
            "applied": False,
        }
        if dry_run:
            manifest.append(record)
            continue
        try:
            if action.kind == "delete_repo":
                api.delete_repo(repo_id=action.repo_id, repo_type="dataset", missing_ok=True)
            elif action.kind == "delete_folder":
                api.delete_folder(
                    path_in_repo=action.path,
                    repo_id=action.repo_id,
                    repo_type="dataset",
                    commit_message=f"repo_cleanup: drop {action.path} ({action.reason})",
                )
            record["applied"] = True
            logger.info("%s %s %s (-%s)", action.kind, action.repo_id, action.path or "", _human_bytes(action.reclaim_bytes))
        except Exception as exc:
            record["error"] = str(exc)
            logger.warning("failed %s %s: %s", action.kind, action.repo_id, exc)
        manifest.append(record)
        if sleep:
            time.sleep(sleep)  # be gentle with HF's repo-mutation rate limit
    return manifest


def _print_report(plan: Plan, cfg: Config, *, dry_run: bool, live_set_known: bool) -> None:
    out = sys.stdout
    by_kind: dict[str, int] = {}
    for a in plan.actions:
        by_kind[a.kind if a.path is None else f"{a.kind}:{a.path}"] = by_kind.get(a.kind if a.path is None else f"{a.kind}:{a.path}", 0) + 1
    print(f"\n=== flashrun-* repo cleanup ({'DRY-RUN' if dry_run else 'APPLY'}) ===", file=out)
    print(f"namespace={cfg.namespace}  tiers: code={cfg.code} checkpoints={cfg.checkpoints} repos={cfg.repos}", file=out)
    print(f"serving live set: {'confirmed' if live_set_known else 'UNAVAILABLE (code-only tier)'}", file=out)
    print(f"skipped (untouched): {len(plan.skips)}   planned actions: {len(plan.actions)}", file=out)
    for label, n in sorted(by_kind.items()):
        print(f"  {label}: {n}", file=out)
    print(f"reclaimable: {_human_bytes(plan.reclaim_bytes)}", file=out)
    for a in plan.actions:
        tgt = a.path or "(whole repo)"
        print(f"  {'would ' if dry_run else ''}{a.kind:<13} {a.repo_id}  {tgt}  -{_human_bytes(a.reclaim_bytes)}  [{a.reason}]", file=out)


def run(cfg: Config, *, dry_run: bool = True, sleep: float = 0.5, manifest_path: str | None = None, api=None) -> Plan:
    """Top-level entry: enumerate, classify, report, and (unless ``dry_run``) apply."""
    from huggingface_hub import HfApi

    api = api or HfApi()

    # Resolve the live serving keep-set first. Tiers that remove servable content REQUIRE it; if it
    # can't be confirmed they abort. The code-only tier is serving-safe and may proceed without it.
    live_set_known = True
    try:
        deployed = deployed_repo_ids()
    except Exception as exc:
        live_set_known = False
        deployed = set()
        if cfg.needs_live_set:
            raise CleanupAborted(
                f"cannot confirm the serving live set ({exc}); refusing to run checkpoint/repo "
                "deletion. Re-run with only --code (serving-safe), or restore serving access."
            ) from exc
        logger.warning("serving live set unavailable (%s); proceeding with code-only purge", exc)

    now = datetime.now(UTC)
    views = list_run_repos(api, cfg.namespace)
    plan = build_plan(views, deployed, cfg, now)
    _print_report(plan, cfg, dry_run=dry_run, live_set_known=live_set_known)

    manifest = apply_plan(api, plan, dry_run=dry_run, sleep=sleep)
    if manifest_path:
        with open(manifest_path, "w") as f:
            json.dump({"dry_run": dry_run, "actions": manifest}, f, indent=2)
        print(f"\nmanifest → {manifest_path}", file=sys.stdout)
    return plan


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m flash.server.repo_cleanup",
        description="Operator GC for per-run HF artifact repos (Freesolo-Co/flashrun-*). "
        "Dry-run by default; serving-safe.",
    )
    p.add_argument("--apply", action="store_true", help="actually delete (default: dry-run report only)")
    p.add_argument("--namespace", default=_ARTIFACT_NAMESPACE, help=f"HF org owning the run repos (default: {_ARTIFACT_NAMESPACE})")
    p.add_argument("--no-code", dest="code", action="store_false", help="do NOT purge code/flash (default: purge it)")
    p.add_argument("--checkpoints", action="store_true", help="T2: trim checkpoints/ from old undeployed repos (keeps adapter/)")
    p.add_argument("--repos", action="store_true", help="T3: delete whole repos for old undeployed runs with no servable adapter")
    p.add_argument("--delete-with-adapter", action="store_true", help="let T3 also delete old undeployed repos that DO have a final adapter")
    p.add_argument("--inactive-age-hours", type=float, default=6.0, help="min idle time before a repo is considered not in-flight (default: 6)")
    p.add_argument("--trim-age-days", type=float, default=14.0, help="min age for checkpoint trimming (default: 14)")
    p.add_argument("--delete-age-days", type=float, default=60.0, help="min age for whole-repo deletion (default: 60)")
    p.add_argument("--sleep", type=float, default=0.5, help="seconds between mutations (HF rate-limit courtesy; default: 0.5)")
    p.add_argument("--manifest", default=None, help="write a JSON manifest of (planned/applied) actions to this path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = Config(
        namespace=args.namespace,
        code=args.code,
        checkpoints=args.checkpoints,
        repos=args.repos,
        delete_with_adapter=args.delete_with_adapter,
        inactive_age_hours=args.inactive_age_hours,
        trim_age_days=args.trim_age_days,
        delete_age_days=args.delete_age_days,
    )
    try:
        run(cfg, dry_run=not args.apply, sleep=args.sleep, manifest_path=args.manifest)
    except CleanupAborted as exc:
        print(f"ABORTED: {exc}", file=sys.stderr)
        return 2
    if not args.apply:
        print("\n(dry-run — nothing was deleted. Re-run with --apply to execute.)", file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
