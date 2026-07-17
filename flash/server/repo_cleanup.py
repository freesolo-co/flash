"""Always-on GC for aged run artifacts inside the per-environment HF repos (``Freesolo-Co/flashrun-*``).

Every managed run stores its code snapshot, adapter, checkpoints, and telemetry under a per-run prefix
``<phase>/<run_id>/`` inside a *private* HF dataset repo that is shared by every run of an environment
(``managed_hf_repo_for_environment`` -> ``Freesolo-Co/flashrun-<slug>-<digest>``). The deployable
per-step adapters (``.../checkpoints/step-N/adapter``) are kept forever by the trainer and nothing else
deletes old runs, so these repos grow without bound against the org's private-storage quota.

An earlier GC (PR #311) deleted whole ``flashrun-<run_id>`` repos, but that predated the switch to
environment-scoped repos (#346): a shared env repo holds live runs alongside stale ones, so a
whole-repo delete is unsafe. This is its environment-scoped replacement — it deletes an aged run's
PREFIX (``delete_folder`` on ``<phase>/<run_id>``), never the whole repo.

CROSS-PLANE by design. The control plane runs this sweep AUTOMATICALLY
(``flash.server._runtime._repo_cleanup_loop`` calls ``run_scheduled_cleanup`` daily). Unlike a
plane-local scan, it does not consult this plane's run database at all — it works entirely from two
GLOBAL sources of truth, so it reclaims artifacts left by EVERY plane/session, not just this one:

* **What's live** = the serving registry (``GET /adapters`` on the Freesolo serving app). An adapter
  that isn't in the registry isn't loadable/serving *anywhere*, so "not in the registry" is a stronger,
  cross-plane "not deployed" than any single plane's SQLite db.
* **What exists** = every ``<phase>/<run_id>/`` prefix under every ``flashrun-*`` dataset repo, listed
  straight from HF.

ONE fixed, opinionated policy:

    Delete every ``<phase>/<run_id>/`` artifact prefix that is NOT in the global serving set and whose
    newest file was committed more than the GC age (fixed: 7 days) ago.

There is intentionally **no manual CLI and no tiers/flags**. A redeployable-but-undeployed run is not
spared and contents are not trimmed selectively — the whole run prefix (including its final adapter)
goes once it ages out and isn't serving. The shared ``code/<digest>/`` snapshot is content-addressed
and is never touched; neither is anything outside ``ARTIFACT_PHASES``.

Safety — the sweep NEVER deletes blind:

* **Serving set is the do-not-touch set.** A prefix any plane is serving from is excluded, re-confirmed
  immediately before every delete (TOCTOU). If the registry is unreachable, returns an empty set, or
  lists a live adapter that can't be mapped to a repo/prefix, the sweep deletes NOTHING and retries
  next cycle (fails closed).
* **Age is last-activity, not submit time.** A prefix is reaped only if its newest COMMIT is >7d old,
  so an in-flight run (still writing checkpoints) is protected without needing any run registry.
* **Warm-start sources are protected.** When a run ``init_from_adapter``s off another,
  ``flash.runner._mark_warmstart_source`` writes a 0-byte ``referenced_by/<child_run_id>`` marker into
  the SOURCE repo at submit (and re-writes it on recovery). If a repo carries a marker committed within
  the GC age, its artifacts are a source for a possibly-still-training child and are spared. Older
  markers = finished children (which already baked a self-contained ``recomb`` adapter) and do not
  protect. Residual: a child that trains continuously for longer than the age window WITHOUT any
  control-plane restart lets its source's marker age out; the fail-closed-on-undatable age gate (a
  still-writing source keeps recent commits) is the backstop there.
* **Only ``flashrun-*`` repos, only ``ARTIFACT_PHASES``** (hard allowlists); a prefix whose age can't be
  determined is left alone; the destructive delete holds this plane's per-run deploy/export lock; and
  the prefix is re-stat'd for recent writes right before deletion.

There is nothing to configure: the policy (7-day age, daily sweep) is fixed. The loop only runs on a
plane with an operator ``HF_TOKEN`` — without it the sweep cannot delete operator-owned repos at all.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from flash._logging import get_logger
from flash.runner import _ARTIFACT_NAMESPACE

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

# Hard allowlist: only Freesolo-Co/flashrun-* dataset repos are ever touched (never env packages,
# paper-*/oracle/eval sets, or a user's own datasets).
RUN_REPO_PREFIX = "flashrun-"
# Delete an undeployed run's prefix once its newest file was committed this long ago (fixed: 7 days).
DELETE_AGE_SECONDS = 7.0 * 86400.0
_DELETE_SLEEP_S = 0.5  # pause between deletes — HF repo-mutation rate-limit courtesy
_SCAN_WORKERS = 8  # hf tree-listing concurrency
# ``list_repo_tree(..., expand=True)`` is HF's most expensive listing mode (expand bypasses the
# CDN and hits origin for a last_commit lookup per entry) -- it rate-limits well below the
# general API quota. _scan_repo (one call per repo, fanned across _SCAN_WORKERS) and
# _prefix_written_within (one call per delete target) both hit it; with no shared pacing, the
# worker pool lets up to _SCAN_WORKERS calls land on HF in the same instant, which is enough on
# its own to trip the rate limit on a sweep over more than a handful of repos. This floor paces
# EVERY list_repo_tree call across the whole sweep (not per-thread), so concurrency no longer
# translates into a request burst.
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

    Requires an operator ``HF_TOKEN`` — the sweep deletes operator-owned ``flashrun-*`` dataset
    prefixes, impossible (and meaningless) without it — so a plane without the token never schedules
    the loop. This is a credential check, not a knob: there is no on/off env switch."""
    return bool((os.environ.get("HF_TOKEN") or "").strip())


def _is_managed_env_repo(repo_id) -> bool:
    """True only for ``Freesolo-Co/flashrun-*`` dataset repos (the hard allowlist)."""
    if not isinstance(repo_id, str):
        return False
    return repo_id.startswith(f"{_ARTIFACT_NAMESPACE}/") and repo_id.split("/", 1)[-1].startswith(
        RUN_REPO_PREFIX
    )


def deployed_prefixes() -> tuple[set[tuple[str, str]], set[str], bool]:
    """The GLOBAL live serving set from the serving registry (``GET /adapters``): every adapter live on
    ANY control plane. Returns ``(prefixes, whole_repos, complete)`` where:

    * ``prefixes`` — exact ``(repo_id, "<phase>/<run_id>")`` do-not-delete set.
    * ``whole_repos`` — repos with a live record whose *subfolder* couldn't be mapped to a prefix; the
      caller protects the ENTIRE repo (can't tell which prefix is live).
    * ``complete`` — ``False`` if a live record had no ``repoId`` at all (unidentifiable live adapter);
      the caller fails closed.

    An adapter absent from this registry is not loadable/serving anywhere, so this is a strictly
    stronger cross-plane "not deployed" than any single plane's run db."""
    from flash.serve import deploy as _sd

    resp = _sd._serving_request("GET", f"{_sd.serving_base_url()}/adapters")
    records = resp.json().get("adapters") or []
    prefixes: set[tuple[str, str]] = set()
    whole: set[str] = set()
    complete = True
    for rec in records:
        repo = rec.get("repoId") or rec.get("repo_id")
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
    entries = api.list_repo_tree(
        repo_id=repo_id, repo_type="dataset", recursive=True, expand=True
    )
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
    from flash.server._locks import _deploy_lock

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
    repos = [d.id for d in api.list_datasets(author=_ARTIFACT_NAMESPACE) if _is_managed_env_repo(d.id)]
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
        logger.info("repo GC: skipped unrecognized non-artifact dirs (never deleted): %s", sorted(unknown_tops))
    logger.info(
        "repo GC: scanned %d/%d repos (%d unreadable), %d aged undeployed prefix(es) to reap",
        scanned - errored, len(repos), errored, len(targets),
    )
    return targets


def run_scheduled_cleanup(*, dry_run: bool = False, api=None, should_stop=None) -> int:
    """One sweep of the fixed policy. Returns the number of run prefixes deleted (0 in dry-run).

    Fails closed: raises ``CleanupAborted`` (deleting nothing) when the global serving set can't be
    confirmed up front. The HF + serving calls are blocking, so callers offload this to a thread.

    ``should_stop`` is an optional cooperative-cancel callback checked BETWEEN deletes. The sweep runs
    in a worker thread that ``task.cancel()`` cannot interrupt, so at shutdown the caller sets a stop
    flag and this loop halts promptly instead of churning through more destructive deletes long after
    the server was told to stop (mirrors the completion-charge retry sweeps)."""
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
            fresh, fresh_whole = _confirm_live_set()  # raises -> finally releases, sweep fails closed
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
