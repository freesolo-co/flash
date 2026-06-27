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

* When the live set is known, a repo whose ``repoId`` is registered with serving is **never**
  modified or deleted (it is skipped wholesale before any tier runs).
* If the live set cannot be confirmed (serving unreachable) or is incomplete (a live record had no
  repo id), any tier that removes servable content **aborts** rather than guessing. The ``code/``
  purge is the one exception and still proceeds — serving never reads ``code/flash``, so deleting it
  is harmless even from a deployed repo. NOTE the consequence: with ``--code`` and an
  unavailable/incomplete live set, ``code/flash`` MAY be purged from a currently-deployed repo
  (serving keeps running off the already-loaded adapter; only the unused source snapshot is lost).

Every repo is classified from three self-contained signals — no backend DB required:

1. the serving live set (``deploy.list_deployed_adapters``),
2. the repo's own file tree (does it have ``adapter/``? ``checkpoints/``? ``code/flash/``?),
3. the repo's ``last_modified`` age (a run still training writes heartbeats/checkpoints constantly,
   so a repo untouched for ``--inactive-age-hours`` is definitely not in-flight).

Tiers (conservative defaults; each opt-in beyond ``--code``)
-----------------------------------------------------------
* ``--code``        (default ON)  T1: delete ``code/flash`` from repos older than
                                  ``--delete-age-days``. The single ``code/flash`` snapshot is
                                  uploaded once per run (pre-boot) and shared by EVERY seed and by
                                  ``resume_run`` (no re-upload), so it must survive until the whole
                                  run is provably finished — a later seed or a resume after a long
                                  inter-seed capacity gap still needs it. Deployed repos are skipped
                                  wholesale, and serving never reads ``code/flash`` regardless.
                                  Reclaims little; safe.
* ``--checkpoints`` (default off) T2: on terminal + undeployed + older than ``--trim-age-days`` and
                                  carrying a FINAL ``adapter/``, delete ``checkpoints/`` while
                                  **keeping** that final adapter. The big byte win; you lose only
                                  ``flash deploy --step N`` for old runs. A checkpoint-only repo
                                  (no final adapter) is left intact — its checkpoints are its only
                                  servable content.
* ``--repos``       (default off) T3: ``delete_repo`` for terminal + undeployed + older than
                                  ``--delete-age-days`` repos with **no servable adapter at all**
                                  (neither a final ``adapter/`` nor a deployable
                                  ``checkpoints/step-N/adapter``) — i.e. failed/cancelled runs that
                                  never produced anything deployable.
* ``--delete-with-adapter`` (default off)  let T3 also whole-delete *old undeployed* repos that DO
                                  carry a servable adapter (final or checkpoint). Off by default so a
                                  redeployable adapter is never dropped implicitly; requires
                                  ``--repos`` (it only widens T3 and does nothing on its own).

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
CHECKPOINTS_PATH = "checkpoints"  # plural: per-step DEPLOYABLE adapter snapshots
# singular: the full-trainer resume checkpoint (optimizer/scheduler/rng + weights) the worker keeps
# under <prefix>/checkpoint/checkpoint-N for resume-on-preemption (flash.engine.worker.hf). It is
# never servable and far larger than a deployable adapter, so an old terminal run never needs it.
RESUME_CHECKPOINT_PATH = "checkpoint"
ADAPTER_PATH = "adapter"
# A PEFT adapter folder is only *deployable* when it carries adapter_config.json AND a weights file
# — the exact signal serving / flash.runner.checkpoints.list_checkpoints use, so the GC agrees with
# them on what counts as a servable artifact (a half-uploaded config-only folder is NOT one).
_ADAPTER_CONFIG = "adapter_config.json"
_ADAPTER_WEIGHT_FILES = frozenset({"adapter_model.safetensors", "adapter_model.bin"})


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

    def _artifact_folders(self, name: str) -> list[str]:
        """Actual folder paths whose trailing path segments are ``name``, found *anywhere* in the
        tree. The layout is mixed: the control plane uploads ``code/flash`` at the repo ROOT
        (``providers.runpod.train.upload_code``, before the worker boots), while the worker nests
        the adapter and checkpoints under ``hf_prefix() = {phase}/{run_id}/seed{N}/``
        (``flash.engine.worker.hf``). A root-anchored match would silently miss the nested ones, so
        match the segment-aligned ``name`` at any depth. A folder is only reported when at least one
        file lives beneath it (a bare file *named* ``adapter`` is not a folder)."""
        segs = name.split("/")
        n = len(segs)
        folders: set[str] = set()
        for f, _ in self.files:
            parts = f.split("/")
            # leave >=1 trailing component so the match is a folder containing this file
            for i in range(len(parts) - n):
                if parts[i : i + n] == segs:
                    folders.add("/".join(parts[: i + n]))
        return sorted(folders)

    def _files_in(self, folder: str) -> list[tuple[str, int]]:
        return [(f, s) for f, s in self.files if f == folder or f.startswith(folder + "/")]

    def has(self, name: str) -> bool:
        return bool(self._artifact_folders(name))

    def bytes_in(self, folder: str) -> int:
        return sum(s for _, s in self._files_in(folder))

    def bytes_under(self, name: str) -> int:
        return sum(self.bytes_in(folder) for folder in self._artifact_folders(name))

    def adapter_folders(self) -> list[str]:
        """Every ``adapter`` folder: the final ``<prefix>/adapter`` AND each checkpoint
        ``<prefix>/checkpoints/step-N/adapter``."""
        return self._artifact_folders(ADAPTER_PATH)

    def _is_deployable_adapter(self, folder: str) -> bool:
        """A folder is a *deployable* adapter only if it carries adapter_config.json AND a weights
        file — matching serving / ``list_checkpoints``. A half-uploaded config-only folder is not
        servable, so it must not stand in for a real adapter."""
        names = {f.rsplit("/", 1)[-1] for f, _ in self._files_in(folder)}
        return _ADAPTER_CONFIG in names and not names.isdisjoint(_ADAPTER_WEIGHT_FILES)

    def deployable_adapter_folders(self) -> list[str]:
        return [f for f in self.adapter_folders() if self._is_deployable_adapter(f)]

    def has_servable_adapter(self) -> bool:
        """True if the repo holds ANY *deployable* adapter — final or per-checkpoint. A repo with
        only checkpoint adapters (no final ``adapter/``) is still servable and must not be
        whole-deleted as 'unservable'."""
        return bool(self.deployable_adapter_folders())

    def has_final_adapter(self) -> bool:
        """True only for a *complete* final trained adapter at ``<prefix>/adapter`` — never one
        nested under ``checkpoints/``. T2 trims checkpoints only when this remains, so it never
        strips a checkpoint-only repo's sole servable content, nor trims when the final adapter
        upload only partially landed (config without weights)."""
        return any(
            "checkpoints" not in folder.split("/") for folder in self.deployable_adapter_folders()
        )

    def age_seconds(self, now: datetime) -> float:
        if self.last_modified is None:
            # Unknown age (HF returned no last_modified — e.g. a freshly created / queued run repo).
            # Treat it as just-written (age 0) so the inactive-age gate PROTECTS it: an ambiguous
            # timestamp must never make a brand-new repo look old enough to delete. ``classify`` also
            # skips ``last_modified is None`` explicitly with a clear reason.
            return 0.0
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
        """Tiers that remove servable content require the live serving set to be confirmed.

        Only the tiers that actually *act* count: ``--checkpoints`` (T2) and ``--repos`` (T3).
        ``--delete-with-adapter`` is a *modifier* of T3 — on its own it deletes nothing, so it
        must not by itself force the live-set requirement (and is rejected without ``--repos`` in
        ``main``)."""
        return self.checkpoints or self.repos


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


def deployed_repo_ids() -> tuple[set[str], bool]:
    """The HF repo ids serving is currently loading adapters from (the live keep-set), plus a
    ``complete`` flag.

    ``complete`` is ``False`` when a live record could NOT be mapped to a repo id (missing
    ``repoId``/``repo_id`` — a schema drift). The two callers treat that differently:
    destructive tiers (``--checkpoints``/``--repos``) abort on an incomplete set (an unidentifiable
    live repo could be deleted), while the serving-safe ``--code`` purge proceeds with the
    best-effort set — it still protects every identifiable deployed repo, and code/flash is never
    read by serving anyway. (A hard raise here regressed code-only to an EMPTY keep-set.)
    """
    from flash.serve import deploy

    ids: set[str] = set()
    complete = True
    for rec in deploy.list_deployed_adapters():
        repo = rec.get("repoId") or rec.get("repo_id")
        if not repo:
            complete = False  # an unidentifiable live repo — caller decides whether to abort
            continue
        ids.add(str(repo))
    return ids, complete


def classify(view: RepoView, deployed: set[str], cfg: Config, now: datetime) -> list[Action]:
    """Decide what to do with one repo. Returns the planned actions (possibly a single
    ``kind="skip"`` action explaining why nothing is done)."""
    if view.repo_id in deployed:
        return [Action(view.repo_id, "skip", None, 0, "deployed (serving live set)", skipped=True)]

    if view.last_modified is None:
        # No timestamp at all — can't prove the repo is old/terminal (it may be freshly created or
        # queued). Leave it untouched rather than guess; a missing age must never green-light a
        # delete. (Operators can investigate a repo that genuinely lost its timestamp.)
        return [Action(view.repo_id, "skip", None, 0, "unknown age (no last_modified) — left for safety", skipped=True)]

    age = view.age_seconds(now)
    if age < cfg.inactive_age_hours * 3600:
        return [Action(view.repo_id, "skip", None, 0, "recently written (maybe in-flight)", skipped=True)]

    # T3 first: a repo with NO servable adapter never produced anything deployable
    # (failed/cancelled/empty). Whole-deleting it reclaims everything and can't compromise serving.
    # "Servable" includes per-step CHECKPOINT adapters, not just the final adapter/ — a repo whose
    # only adapter is checkpoints/step-N/adapter is still one-command-deployable, so it is NOT
    # treated as unservable. A repo with any servable adapter is whole-deleted only under the
    # explicit --delete-with-adapter opt-in.
    has_servable = view.has_servable_adapter()
    if cfg.repos and age >= cfg.delete_age_days * 86400:
        if not has_servable:
            return [Action(view.repo_id, "delete_repo", None, view.total_bytes, "terminal, no servable adapter")]
        if cfg.delete_with_adapter:
            return [Action(view.repo_id, "delete_repo", None, view.total_bytes, "old undeployed adapter (--delete-with-adapter)")]

    actions: list[Action] = []
    if cfg.code and age >= cfg.delete_age_days * 86400:
        # The single ``code/flash`` snapshot is uploaded ONCE per run (control plane, pre-boot) and
        # is shared by EVERY seed and by ``resume_run`` (which re-downloads it with no re-upload). A
        # repo-wide signal can't prove the whole run is finished — a later seed or a resume after a
        # long inter-seed capacity gap still needs that code — so the shared snapshot is only purged
        # once the repo is delete-age old (provably abandoned, no seed/resume can still be pending).
        actions.extend(
            Action(view.repo_id, "delete_folder", folder, view.bytes_in(folder), "training source snapshot")
            for folder in view._artifact_folders(CODE_PATH)
        )
    if cfg.checkpoints and age >= cfg.trim_age_days * 86400:
        # Trim a prefix's plural DEPLOYABLE checkpoint snapshots only when THAT prefix has its own
        # complete final ``adapter/`` to keep serving. A repo can mix prefixes (multi-seed): seed0
        # may have a final adapter while seed1 only has deployable checkpoint adapters — a repo-wide
        # gate would wrongly strip seed1's only servable content. Match each ``<prefix>/checkpoints``
        # against ``<prefix>/adapter``.
        deployable = set(view.deployable_adapter_folders())
        for cp in view._artifact_folders(CHECKPOINTS_PATH):
            prefix = cp.rsplit("/", 1)[0] if "/" in cp else ""
            final_adapter = f"{prefix}/{ADAPTER_PATH}" if prefix else ADAPTER_PATH
            if final_adapter in deployable:
                actions.append(Action(view.repo_id, "delete_folder", cp, view.bytes_in(cp), "intermediate checkpoints (final adapter kept)"))
    if cfg.checkpoints and age >= cfg.delete_age_days * 86400:
        # The singular full-trainer resume checkpoint is what ``hf_resume_checkpoint`` re-downloads
        # for a REPLACEMENT worker after preemption. A retry/recovery can be delayed well past
        # ``--trim-age-days`` (e.g. waiting for capacity), so — like the shared code snapshot — only
        # purge it once the repo is delete-age old (provably abandoned, no retry can still be pending).
        actions.extend(
            Action(view.repo_id, "delete_folder", folder, view.bytes_in(folder), "resume checkpoint (full trainer state)")
            for folder in view._artifact_folders(RESUME_CHECKPOINT_PATH)
        )

    if not actions:
        return [Action(view.repo_id, "skip", None, 0, "nothing eligible", skipped=True)]
    return actions


def build_plan(views: list[RepoView], deployed: set[str], cfg: Config, now: datetime) -> Plan:
    plan = Plan()
    for view in views:
        for action in classify(view, deployed, cfg, now):
            (plan.skips if action.skipped else plan.actions).append(action)
    return plan


def apply_plan(api, plan: Plan, *, dry_run: bool, sleep: float, refresh_live=None) -> list[dict]:
    """Execute (or, when ``dry_run``, only record) every action. Returns a manifest of results.

    ``refresh_live`` (used only for destructive tiers) returns the current ``(deployed_ids,
    complete)``; it is consulted JUST BEFORE EVERY mutation so a repo deployed mid-apply — at any
    instant after the pre-apply sample — is skipped instead of deleted. If the re-check fails or is
    incomplete, the action is skipped to stay safe rather than risk a live repo."""
    manifest: list[dict] = []
    live: set[str] = set()
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
        if refresh_live is not None:
            # Re-fetch the live set for EVERY destructive action (no caching): a repo can be deployed
            # at any instant during apply, so the only fully-safe check is immediately before each
            # mutation. Mutations are already paced by ``--sleep`` and the read is lightweight.
            try:
                live, live_complete = refresh_live()
            except Exception as exc:
                record["skipped"] = f"live re-check failed ({exc}); skipped to stay safe"
                logger.warning("skipping %s %s: live re-check failed: %s", action.kind, action.repo_id, exc)
                manifest.append(record)
                continue
            if not live_complete:
                # An unmappable live record means an unidentifiable live repo could be the one we're
                # about to delete. Fail closed: skip.
                record["skipped"] = "live set incomplete on re-check (a record lacked a repo id); skipped to stay safe"
                logger.warning("skipping %s %s: live set incomplete on re-check", action.kind, action.repo_id)
                manifest.append(record)
                continue
            if action.repo_id in live:
                record["skipped"] = "repo became deployed during apply"
                logger.warning("skipping %s %s: repo became deployed during apply", action.kind, action.repo_id)
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


def _print_report(plan: Plan, cfg: Config, *, dry_run: bool, live_set_known: bool, live_set_complete: bool = True) -> None:
    out = sys.stdout
    by_kind: dict[str, int] = {}
    for a in plan.actions:
        by_kind[a.kind if a.path is None else f"{a.kind}:{a.path}"] = by_kind.get(a.kind if a.path is None else f"{a.kind}:{a.path}", 0) + 1
    if not live_set_known:
        live_set_status = "UNAVAILABLE (code-only tier)"
    elif not live_set_complete:
        live_set_status = "INCOMPLETE — a live record had no repo id (code-only tier)"
    else:
        live_set_status = "confirmed"
    print(f"\n=== flashrun-* repo cleanup ({'DRY-RUN' if dry_run else 'APPLY'}) ===", file=out)
    print(f"namespace={cfg.namespace}  tiers: code={cfg.code} checkpoints={cfg.checkpoints} repos={cfg.repos}", file=out)
    print(f"serving live set: {live_set_status}", file=out)
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
    live_set_complete = True
    try:
        deployed, complete = deployed_repo_ids()
    except Exception as exc:
        live_set_known = False
        live_set_complete = False
        deployed = set()
        if cfg.needs_live_set:
            raise CleanupAborted(
                f"cannot confirm the serving live set ({exc}); refusing to run checkpoint/repo "
                "deletion. Re-run with only --code (serving-safe), or restore serving access."
            ) from exc
        logger.warning("serving live set unavailable (%s); proceeding with code-only purge", exc)
    else:
        if not complete:
            # Serving answered, but a live record couldn't be mapped to a repo id. Destructive tiers
            # abort (an unidentifiable live repo could be deleted); code-only proceeds with the
            # best-effort set (still protects every identifiable deployed repo; code is serving-safe).
            # Report it as incomplete (NOT "confirmed") so operators aren't given false assurance.
            live_set_complete = False
            if cfg.needs_live_set:
                raise CleanupAborted(
                    "serving returned a live adapter record with no repo id; refusing checkpoint/repo "
                    "deletion against an incomplete keep-set. Re-run with only --code, or fix serving."
                )
            logger.warning("serving live set is incomplete (a record lacked a repo id); code-only purge proceeds")

    now = datetime.now(UTC)
    views = list_run_repos(api, cfg.namespace)
    plan = build_plan(views, deployed, cfg, now)

    # The live set was sampled before enumeration; a deploy may have landed during enumeration +
    # classification (minutes, for a large namespace). Re-confirm it immediately before mutating and
    # drop any action now targeting a live repo, so we never write to a repo that became deployed in
    # that window. Done BEFORE the report so --apply output reflects what is actually applied (the
    # dry-run path skips this and reports the full preview plan, mutating nothing).
    if not dry_run and plan.actions and live_set_known:
        try:
            fresh, fresh_complete = deployed_repo_ids()
        except Exception as exc:
            if cfg.needs_live_set:
                raise CleanupAborted(
                    f"could not re-confirm the serving live set before applying ({exc}); aborting "
                    "rather than risk deleting a newly-deployed repo."
                ) from exc
            logger.warning("could not re-confirm live set before apply (%s); proceeding (code-only)", exc)
            fresh = set()
        else:
            if not fresh_complete and cfg.needs_live_set:
                raise CleanupAborted(
                    "serving returned an incomplete live set on re-confirm (a record lacked a repo "
                    "id); aborting rather than risk deleting a newly-deployed repo."
                )
        if fresh:
            kept = [a for a in plan.actions if a.repo_id not in fresh]
            dropped = len(plan.actions) - len(kept)
            if dropped:
                logger.warning("dropping %d planned action(s) against repos deployed since enumeration", dropped)
                plan.actions = kept

    _print_report(plan, cfg, dry_run=dry_run, live_set_known=live_set_known, live_set_complete=live_set_complete)
    # For destructive tiers, re-fetch the live set just before EVERY mutation (no caching) so a repo
    # deployed mid-apply is skipped — the pre-apply sample above only catches deploys up to this
    # point. Code-only purge is serving-safe and needs no per-action re-check.
    refresh_live = deployed_repo_ids if (not dry_run and cfg.needs_live_set) else None
    manifest = apply_plan(api, plan, dry_run=dry_run, sleep=sleep, refresh_live=refresh_live)
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
    p.add_argument("--delete-with-adapter", action="store_true", help="let T3 also delete old undeployed repos that DO have a servable adapter (final OR checkpoint); requires --repos")
    p.add_argument("--inactive-age-hours", type=float, default=6.0, help="min idle time before a repo is considered not in-flight (default: 6)")
    p.add_argument("--trim-age-days", type=float, default=14.0, help="min age for checkpoint trimming (default: 14)")
    p.add_argument("--delete-age-days", type=float, default=60.0, help="min age for whole-repo deletion (default: 60)")
    p.add_argument("--sleep", type=float, default=0.5, help="seconds between mutations (HF rate-limit courtesy; default: 0.5)")
    p.add_argument("--manifest", default=None, help="write a JSON manifest of (planned/applied) actions to this path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.delete_with_adapter and not args.repos:
        print(
            "--delete-with-adapter only widens T3 (whole-repo deletion) and does nothing on its "
            "own; pass --repos as well, or drop it.",
            file=sys.stderr,
        )
        return 2
    for name, value in (
        ("--inactive-age-hours", args.inactive_age_hours),
        ("--trim-age-days", args.trim_age_days),
        ("--delete-age-days", args.delete_age_days),
    ):
        # A negative window would make its age gate true for (nearly) every repo — a typo must not
        # silently widen deletion. Reject it.
        if value < 0:
            print(f"{name} must be >= 0 (got {value})", file=sys.stderr)
            return 2
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
