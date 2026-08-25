"""Always-on GC for aged run artifacts in per-environment ``flashrun-*`` HF repos.

The daily cross-plane sweep deletes non-serving ``<phase>/<run_id>`` prefixes whose newest commit is
older than seven days. It never deletes whole repos, shared code snapshots, or unknown phases.

Serving registry data is the fail-closed do-not-delete set and is rechecked before each delete.
Recent warm-start markers, undatable prefixes, and in-flight writes are protected. The loop requires
an operator ``HF_TOKEN`` and has no manual CLI or policy knobs.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from flash._internal.logging import get_logger
from flash.runner.accounting.artifacts import artifact_namespace
from flash.serve.contract import urls as serving_urls
from flash.serve.request import transport as serving_transport

logger = get_logger(__name__)

# huggingface_hub is an OPTIONAL server extra (see pyproject ``[server]``). The rest of the control
# plane treats it as optional, so mirror that: without the extra there is no way to list or delete
# repos, so the always-on GC degrades to a logged no-op instead of crashing the loop with a
# ``ModuleNotFoundError`` every cycle.
try:
    from huggingface_hub import HfApi
except ModuleNotFoundError:  # pragma: no cover - the test venv always has the server extra
    HfApi = None  # type: ignore[assignment,misc]

# Latch so a plane missing the extra warns ONCE (the sweep runs daily — don't spam the log).
_warned_hf_unavailable = False

# Hard allowlist: only <artifact namespace>/flashrun-* dataset repos are ever touched (never env
# packages, paper-*/oracle/eval sets, or a user's own datasets).
RUN_REPO_PREFIX = "flashrun-"
# Delete an undeployed run's prefix once its newest file was committed this long ago (fixed: 7 days).
DELETE_AGE_SECONDS = 7.0 * 86400.0
_DELETE_SLEEP_S = 0.5  # pause between deletes — HF repo-mutation rate-limit courtesy
_SCAN_WORKERS = 8  # hf tree-listing concurrency
# expanded HF tree listings hit origin and rate-limit below the general quota.
# pace every call globally so _SCAN_WORKERS concurrency cannot create request bursts.
_TREE_LIST_MIN_INTERVAL_S = 0.5  # cap ~2 req/s across the whole sweep, workers included
_tree_list_lock = threading.Lock()
_tree_list_last_call = 0.0


def _throttle_tree_list() -> None:
    global _tree_list_last_call
    with _tree_list_lock:
        wait = _tree_list_last_call + _TREE_LIST_MIN_INTERVAL_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _tree_list_last_call = time.monotonic()


# Only these top-level dirs are deletable model-artifact phases (``JobSpec.phase``: grpo -> "rl",
# plus the VL warm-start ``recomb`` recombined adapter). Anything else — ``code/`` snapshots,
# ``referenced_by/`` lineage markers, telemetry, or an unrecognized future phase — is NEVER a delete
# target. Allowlist, not denylist, so a new artifact kind can't be silently reaped.
ARTIFACT_PHASES = frozenset({"sft", "rl", "opd", "dpo", "grpo", "recomb"})
# 0-byte warm-start lineage back-reference written into the SOURCE repo: ``referenced_by/<child_run_id>``.
REF_MARKER = "referenced_by"


class CleanupAborted(RuntimeError):
    """The global live set could not be confirmed, so the sweep deleted nothing (fails closed)."""


@dataclass(frozen=True)
class _RunTarget:
    repo_id: str
    prefix: str  # "<phase>/<run_id>"
    run_id: str
    age_ts: float | None  # newest-commit epoch of the prefix (fallback: run-id epoch)


def _now() -> float:
    """Wall-clock epoch seconds. A seam so tests can freeze time deterministically."""
    return time.time()


def repo_cleanup_enabled() -> bool:
    """Whether the always-on artifact GC should run on this control plane.

    Require an operator ``HF_TOKEN`` and never run standalone: the hosted serving registry is not a
    self-hosted authority and querying it would expose ``FREESOLO_INTERNAL_KEY``.
    """
    from flash.server.platform.auth import standalone

    return bool((os.environ.get("HF_TOKEN") or "").strip()) and not standalone()


def _is_managed_env_repo(repo_id) -> bool:
    """True only for ``<artifact namespace>/flashrun-*`` dataset repos (the hard allowlist).

    Reads the namespace through ``artifact_namespace()`` so the allowlist always names the same
    place ``managed_hf_repo_for_environment`` creates repos in; a fixed constant here would stop
    matching the moment an operator sets ``FLASH_HF_NAMESPACE``, silently disabling the GC."""
    if not isinstance(repo_id, str):
        return False
    return repo_id.startswith(f"{artifact_namespace()}/") and repo_id.split("/", 1)[-1].startswith(
        RUN_REPO_PREFIX
    )


def deployed_prefixes() -> tuple[set[tuple[str, str]], set[str], bool]:
    """Return the global serving do-not-delete set from ``GET /adapters``.

    The tuple contains exact prefixes, whole repos with unmappable live records, and a completeness
    flag; callers fail closed when any live adapter cannot be identified.
    """
    resp = serving_transport.serving_request("GET", f"{serving_urls.serving_base_url()}/adapters")
    records = resp.json().get("adapters") or []
    prefixes: set[tuple[str, str]] = set()
    whole: set[str] = set()
    complete = True
    for rec in records:
        # snake_case is the wire spelling: freesolo's `AdapterRecord` declares `repo_id`
        # with no serialization alias, and its persistence layer reads and writes the same
        # key. a camelCase reading here would never match and would fail this closed.
        repo = rec.get("repo_id")
        if not isinstance(repo, str) or not repo.strip():
            complete = False  # a live adapter we can't even attribute to a repo -> fail closed
            continue
        repo = repo.strip()
        sub = rec.get("subfolder")
        segs = [s for s in str(sub).split("/") if s] if isinstance(sub, str) else []
        if len(segs) >= 2:
            prefixes.add((repo, f"{segs[0]}/{segs[1]}"))
        else:
            whole.add(repo)  # live, but which prefix? protect the whole repo
    return prefixes, whole, complete


def _confirm_live_set() -> tuple[set[tuple[str, str]], set[str]]:
    """Return ``(live_prefixes, protected_whole_repos)`` or raise ``CleanupAborted``. Called before
    listing AND before every delete; every failure mode fails closed so a blind delete is impossible."""
    try:
        prefixes, whole, complete = deployed_prefixes()
    except Exception as exc:  # any failure to read the live set must fail closed
        raise CleanupAborted(f"serving registry unreachable ({exc}); deleting nothing") from exc
    if not complete:
        raise CleanupAborted("a live adapter could not be mapped to a repo; deleting nothing")
    if not prefixes and not whole:
        # Zero live adapters almost always means a broken/empty query rather than a genuinely empty
        # fleet; refuse rather than risk deleting every artifact against an empty do-not-touch set.
        raise CleanupAborted("serving registry returned zero live adapters; refusing to sweep")
    return prefixes, whole


def _scan_repo(api, repo_id: str) -> tuple[dict[str, list], float | None, set[str]]:
    """List one ``flashrun-*`` repo. Returns ``(prefixes, ref_recent_ts, unknown_tops)`` where
    ``prefixes`` maps ``"<phase>/<run_id>"`` -> ``[total_size, newest_commit_ts]`` for ARTIFACT_PHASES
    only, ``ref_recent_ts`` is the newest commit among ``referenced_by/`` markers (warm-start-source
    signal), and ``unknown_tops`` are unrecognized top-level dirs (reported, never deleted)."""
    _throttle_tree_list()
    entries = api.list_repo_tree(repo_id=repo_id, repo_type="dataset", recursive=True, expand=True)
    prefixes: dict[str, list] = {}
    ref_recent_ts: float | None = None
    unknown: set[str] = set()
    for entry in entries:
        segs = entry.path.split("/")
        if len(segs) < 2:
            continue
        top = segs[0]
        commit = getattr(entry, "last_commit", None)
        date = getattr(commit, "date", None) if commit is not None else None
        ts = date.timestamp() if date is not None else None
        if top == REF_MARKER:  # lineage marker — track recency, never a delete target
            if ts is not None and (ref_recent_ts is None or ts > ref_recent_ts):
                ref_recent_ts = ts
            continue
        if top == "code":
            continue
        if top not in ARTIFACT_PHASES:
            unknown.add(top)
            continue
        if not segs[1].startswith("flash-"):
            continue
        pfx = f"{top}/{segs[1]}"
        size = getattr(entry, "size", 0) or 0
        cur = prefixes.get(pfx)
        if cur is None:
            prefixes[pfx] = [size, ts]
        else:
            cur[0] += size
            if ts is not None and (cur[1] is None or ts > cur[1]):
                cur[1] = ts
    return prefixes, ref_recent_ts, unknown


def _hold_run_lock(run_id: str):
    """Non-blocking acquire-and-HOLD of the per-run deploy/export lock for ``run_id``.

    Returns the held lock — the caller MUST ``release()`` it once the delete is done — or ``None`` when
    a deploy/undeploy/export currently holds it. Held ACROSS the delete so the destructive mutation is
    mutually exclusive with THIS plane's adapter registration/export of the same run. In-process and
    best-effort (it can't mutex a SIBLING plane — the serving-set re-confirm below covers that); never
    blocks the sweep waiting on a slow deploy — it just spares the run this cycle."""
    from flash.server.platform.locks import _deploy_lock

    lock = _deploy_lock(run_id)
    if not lock.acquire(blocking=False):
        return None
    return lock


def _prefix_written_within(api, repo_id: str, prefix: str, now: float, max_age_s: float) -> bool:
    """True if any file under ``prefix`` was committed within ``max_age_s`` (=> recently active, spare
    it). Closes the window between enumeration and delete: a warm-start touch, a late checkpoint, or a
    redeploy read-back that landed after enumeration must spare the prefix. Raises on an API error (the
    caller then skips this target to stay safe). If the listing carries no commit dates (older
    ``huggingface_hub``), returns ``False`` — the enumeration age gate already qualified it."""
    _throttle_tree_list()
    entries = api.list_repo_tree(
        repo_id=repo_id, repo_type="dataset", path_in_repo=prefix, recursive=True, expand=True
    )
    newest: float | None = None
    for entry in entries:
        commit = getattr(entry, "last_commit", None)
        date = getattr(commit, "date", None) if commit is not None else None
        if date is None:
            continue
        ts = date.timestamp()
        if newest is None or ts > newest:
            newest = ts
    if newest is None:
        return False
    return (now - newest) < max_age_s


def _collect_targets(api, live, whole, now: float, max_age_s: float) -> list[_RunTarget]:
    """Enumerate every ``flashrun-*`` repo and return the aged, undeployed, non-warm-start-source run
    prefixes to delete. Per-repo scan failures (HF rate-limit / transient) are skipped and logged, not
    fatal — the sweep reaps what it could read and tries the rest next cycle."""
    repos = [
        d.id for d in api.list_datasets(author=artifact_namespace()) if _is_managed_env_repo(d.id)
    ]
    targets: list[_RunTarget] = []
    unknown_tops: set[str] = set()
    scanned = errored = 0
    with ThreadPoolExecutor(max_workers=_SCAN_WORKERS) as pool:
        futs = {pool.submit(_scan_repo, api, r): r for r in repos}
        for fut in as_completed(futs):
            repo = futs[fut]
            scanned += 1
            if repo in whole:
                continue  # a live-but-unmappable adapter here -> protect the whole repo
            try:
                prefixes, ref_recent_ts, unknown = fut.result()
            except Exception as exc:  # transient list failure -> skip this repo, retry next cycle
                errored += 1
                logger.warning("repo GC: scan of %s failed; skipping this cycle (%s)", repo, exc)
                continue
            unknown_tops |= unknown
            # A RECENT (<=age) referenced_by marker => warm-start source for a possibly in-flight child
            # that still needs it -> protect ALL of this repo's artifacts. Older markers = finished.
            warm_src = ref_recent_ts is not None and (now - ref_recent_ts) <= max_age_s
            for pfx, (_size, ts) in prefixes.items():
                if (repo, pfx) in live:
                    continue  # currently serving on some plane
                if ts is None:
                    # No commit date => LAST ACTIVITY is unknown. The run-id epoch is SUBMIT time, not
                    # activity — a run submitted >7d ago can still be training (long RunPod queue) or
                    # have fresh commits the Hub hasn't dated yet. Never delete on submit time; skip and
                    # retry next cycle (fail closed, matching the module's "can't date -> leave alone").
                    continue
                if (now - ts) <= max_age_s:
                    continue  # recently active (in-flight or just finished)
                if warm_src:
                    continue  # warm-start source for a recent child (referenced_by marker)
                targets.append(
                    _RunTarget(repo_id=repo, prefix=pfx, run_id=pfx.split("/", 1)[1], age_ts=ts)
                )
    if unknown_tops:
        logger.info(
            "repo GC: skipped unrecognized non-artifact dirs (never deleted): %s",
            sorted(unknown_tops),
        )
    logger.info(
        "repo GC: scanned %d/%d repos (%d unreadable), %d aged undeployed prefix(es) to reap",
        scanned - errored,
        len(repos),
        errored,
        len(targets),
    )
    return targets


def run_scheduled_cleanup(*, dry_run: bool = False, api=None, should_stop=None) -> int:
    """One sweep of the fixed policy. Returns the number of run prefixes deleted (0 in dry-run).

    Raise ``CleanupAborted`` before deleting if the serving set is unconfirmed. ``should_stop`` is
    checked between deletes because cancelling the caller cannot interrupt the worker thread.
    """
    global _warned_hf_unavailable
    if api is None:
        if HfApi is None:
            # No HF client on this plane -> nothing to list or delete. Degrade to a no-op (warn ONCE)
            # rather than crash the always-on loop with ModuleNotFoundError every cycle.
            if not _warned_hf_unavailable:
                logger.warning(
                    "repo GC: huggingface_hub not installed (server extra absent); skipping sweeps"
                )
                _warned_hf_unavailable = True
            return 0
        api = HfApi()
    max_age_s = DELETE_AGE_SECONDS

    live, whole = _confirm_live_set()  # fail closed before we even list
    now = _now()
    targets = _collect_targets(api, live, whole, now, max_age_s)

    if dry_run:
        logger.info(
            "repo GC (dry-run): %d aged undeployed run prefix(es) would be deleted", len(targets)
        )
        return 0

    deleted = 0
    for target in targets:
        # Cooperative shutdown: honor the stop signal BETWEEN deletes so a large in-flight sweep can't
        # keep deleting after the lifespan started tearing down.
        if should_stop is not None and should_stop():
            logger.info("repo GC: stop requested; halting sweep after %d delete(s)", deleted)
            break
        # Acquire-and-HOLD this plane's per-run deploy/export lock across the delete so the destructive
        # mutation is mutually exclusive with a concurrent deploy/undeploy/export of this run. Non-
        # blocking: if one already holds it, spare the run this cycle rather than wait.
        held = _hold_run_lock(target.run_id)
        if held is None:
            logger.warning(
                "repo GC: %s has a deploy/undeploy/export in progress; skipping", target.run_id
            )
            continue
        try:
            # Re-stat the prefix: its newest file may have been written since enumeration (a cross-plane
            # warm-start touch, a late checkpoint, a redeploy read-back). Do this FIRST — it is a slow
            # recursive HF list — so the live-set re-confirm below is the LAST thing before the delete
            # and its confirm->delete window is ~microseconds, not a full HF round-trip.
            try:
                if _prefix_written_within(api, target.repo_id, target.prefix, now, max_age_s):
                    logger.warning(
                        "repo GC: %s was written since enumeration (now within age); skipping",
                        target.prefix,
                    )
                    continue
            except Exception as exc:  # a re-stat failure must not delete blind
                logger.warning(
                    "repo GC: re-stat of %s:%s failed; skipping to stay safe (%s)",
                    target.repo_id,
                    target.prefix,
                    exc,
                )
                continue
            # Re-confirm the GLOBAL live set immediately before the delete — now UNDER the held lock and
            # right ahead of delete_folder, so a deploy can't register this run in the confirm->delete
            # gap. If it can't be confirmed now, abort the WHOLE sweep (re-raise) rather than press on
            # deleting other prefixes while an unidentified live adapter may be backed by one of them.
            fresh, fresh_whole = (
                _confirm_live_set()
            )  # raises -> finally releases, sweep fails closed
            if (target.repo_id, target.prefix) in fresh or target.repo_id in fresh_whole:
                logger.warning("repo GC: %s became deployed mid-sweep; skipping", target.prefix)
                continue
            try:
                # delete_folder has no missing_ok; a concurrent delete / already-gone prefix raises.
                api.delete_folder(
                    path_in_repo=target.prefix, repo_id=target.repo_id, repo_type="dataset"
                )
                deleted += 1
                logger.info("repo GC: deleted %s:%s", target.repo_id, target.prefix)
            except Exception as exc:  # one failed delete must not abort the sweep
                logger.warning(
                    "repo GC: failed to delete %s:%s: %s", target.repo_id, target.prefix, exc
                )
        finally:
            held.release()
        time.sleep(_DELETE_SLEEP_S)  # HF repo-mutation rate-limit courtesy
    return deleted
