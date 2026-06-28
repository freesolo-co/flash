"""Always-on GC for per-run HF artifact repos (``Freesolo-Co/flashrun-*``).

Every managed run creates a *private* HF dataset repo ``Freesolo-Co/flashrun-<run_id>`` (see
``flash.runner._assign_managed_hf_repo``) holding the run's code snapshot, adapter, checkpoints, and
telemetry. Nothing else deletes them, so they accumulate against the org's private-storage quota.

The control plane runs this sweep AUTOMATICALLY — ``flash.server._runtime._repo_cleanup_loop`` calls
``run_scheduled_cleanup`` on an interval (default daily). There is ONE fixed, opinionated policy:

    Delete every ``flashrun-*`` repo that is NOT currently deployed, once it is older than the GC age
    (default 30 days). Currently-deployed repos — the ones serving reads — are the only exception.

There is intentionally **no manual CLI and no tiers/flags**: a redeployable-but-undeployed adapter
is not spared and contents are not trimmed in place — the whole repo just goes once it ages out and
isn't serving.

Safety. Serving pulls adapter weights straight from these repos, so the live serving set is the
do-not-touch set:

* A deployed repo is never deleted.
* If the live set can't be confirmed — serving unreachable, or a live record carrying no repo id —
  the sweep deletes NOTHING and retries next cycle (fails closed; never deletes blind).
* The live set is re-confirmed immediately before every delete, so a repo deployed mid-sweep is
  spared (TOCTOU).
* Only ``flashrun-*`` repos are ever touched (hard prefix allowlist), and a repo whose age can't be
  determined is left alone.

There is nothing to configure: the policy (30-day age, daily sweep) is fixed, with no env knobs or
flags. The loop only runs on a plane that has an operator ``HF_TOKEN`` — without it the sweep cannot
delete operator-owned repos at all.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime

from flash._logging import get_logger
from flash.runner import _ARTIFACT_NAMESPACE

logger = get_logger(__name__)

RUN_REPO_PREFIX = "flashrun-"
DELETE_AGE_SECONDS = 30.0 * 86400.0  # delete an undeployed repo once it is this old (fixed: 30 days)
_DELETE_SLEEP_S = 0.5  # pause between deletes — HF repo-mutation rate-limit courtesy


class CleanupAborted(RuntimeError):
    """The serving live set could not be confirmed, so the sweep deleted nothing (fails closed)."""


def repo_cleanup_enabled() -> bool:
    """Whether the always-on repo GC should run on this control plane.

    Requires an operator ``HF_TOKEN`` — the sweep deletes operator-owned ``flashrun-*`` dataset
    repos, impossible (and meaningless) without it — so a plane without the token never schedules the
    loop. This is a credential check, not a knob: there is no on/off env switch."""
    return bool((os.environ.get("HF_TOKEN") or "").strip())


def deployed_repo_ids() -> tuple[set[str], bool]:
    """The HF repo ids serving is currently loading adapters from (the do-not-delete set), plus a
    ``complete`` flag that is ``False`` when a live record carried no repo id (schema drift). The
    sweep refuses to delete against an incomplete or unreachable set."""
    from flash.serve import deploy

    ids: set[str] = set()
    complete = True
    for rec in deploy.list_deployed_adapters():
        repo = rec.get("repoId") or rec.get("repo_id")
        if not repo:
            complete = False  # an unidentifiable live repo — caller fails closed
            continue
        ids.add(str(repo))
    return ids, complete


def _confirm_live_set() -> set[str]:
    """Return the live serving set, or raise ``CleanupAborted``. Called before listing AND before
    every delete; both must fail closed so a blind delete is impossible."""
    try:
        deployed, complete = deployed_repo_ids()
    except Exception as exc:
        raise CleanupAborted(f"serving live set unreachable ({exc}); deleting nothing") from exc
    if not complete:
        raise CleanupAborted("serving returned a live adapter with no repo id; deleting nothing")
    return deployed


def _inflight_repo_ids() -> set[str]:
    """Repos of runs still IN FLIGHT (queued/provisioning/running) — protected regardless of age.

    The deployed-set + age gates alone cannot shield an in-flight run: its repo is created at submit
    (``upload_code`` -> ``create_repo``) BEFORE any GPU worker, so during a long provisioning/capacity
    wait nothing commits and ``last_modified`` stays frozen at upload time — and an in-flight run is
    never in serving's ``/adapters`` set. So an aged-out-but-still-running repo would be deletable.
    This mirrors how the sibling reapers shield ``_RECOVERABLE`` runs. Raises (the sweep fails closed,
    deleting nothing) if run state can't be enumerated."""
    from flash.runner import get_status
    from flash.server import db
    from flash.server._runtime import _RECOVERABLE

    ids: set[str] = set()
    for row in db.all_runs():
        try:
            status = get_status(row["run_id"])
        except FileNotFoundError:
            continue
        if status.state in _RECOVERABLE:
            repo = ((status.spec or {}).get("train") or {}).get("hf_repo")
            ids.add(repo or f"{_ARTIFACT_NAMESPACE}/{RUN_REPO_PREFIX}{row['run_id']}")
    return ids


def list_run_repos(api, namespace: str) -> list[tuple[str, datetime | None]]:
    """Every ``flashrun-*`` dataset under ``namespace`` as ``(repo_id, last_modified)``."""
    out: list[tuple[str, datetime | None]] = []
    for ds in api.list_datasets(author=namespace):
        repo_id = ds.id
        if not repo_id.split("/", 1)[-1].startswith(RUN_REPO_PREFIX):
            continue  # hard allowlist: never touch env packages, paper-*, oracle/eval sets
        last_modified = getattr(ds, "last_modified", None)
        if last_modified is None:
            # The listing didn't carry the timestamp — fetch it directly rather than guess the age
            # (a repo we can't age is never deleted, so a fetch failure just skips it this cycle).
            try:
                last_modified = api.repo_info(repo_id, repo_type="dataset").last_modified
            except Exception as exc:
                logger.warning("repo GC: skipping %s (no last_modified: %s)", repo_id, exc)
                continue
        out.append((repo_id, last_modified))
    return out


def _deletable(repo_id: str, last_modified, protected: set[str], now: datetime, max_age_s: float) -> bool:
    if repo_id in protected:
        return False  # currently serving OR an in-flight run — never touch
    if last_modified is None:
        return False  # unknown age — never delete on a guess
    return (now - last_modified).total_seconds() >= max_age_s


def run_scheduled_cleanup(*, dry_run: bool = False, api=None) -> int:
    """One sweep of the fixed policy. Returns the number of repos deleted (0 in dry-run).

    Fails closed: raises ``CleanupAborted`` (deleting nothing) when the serving live set can't be
    confirmed up front. The HF + serving calls are blocking, so callers offload this to a thread."""
    from huggingface_hub import HfApi

    api = api or HfApi()
    max_age_s = DELETE_AGE_SECONDS

    deployed = _confirm_live_set()  # fail closed before we even list
    # In-flight runs are protected regardless of age (their repo predates any worker and isn't in the
    # deployed set). Snapshotted once per sweep like the sibling reapers; a run that goes in-flight
    # AFTER this has a brand-new repo (young -> not a target anyway). Raises -> sweep fails closed.
    protected = deployed | _inflight_repo_ids()
    now = datetime.now(UTC)
    targets = [
        repo_id
        for repo_id, last_modified in list_run_repos(api, _ARTIFACT_NAMESPACE)
        if _deletable(repo_id, last_modified, protected, now, max_age_s)
    ]
    if dry_run:
        logger.info("repo GC (dry-run): %d undeployed repo(s) past the GC age would be deleted", len(targets))
        return 0

    deleted = 0
    for repo_id in targets:
        # Re-confirm the live set immediately before EACH delete: a repo deployed since enumeration
        # must be spared, and if serving is momentarily unconfirmable we skip this repo rather than
        # risk a blind delete (it'll be reconsidered next cycle).
        try:
            fresh = _confirm_live_set()
        except CleanupAborted as exc:
            logger.warning("repo GC: live set unconfirmable before deleting %s; skipping (%s)", repo_id, exc)
            continue
        if repo_id in fresh:
            logger.warning("repo GC: %s became deployed mid-sweep; skipping", repo_id)
            continue
        try:
            api.delete_repo(repo_id=repo_id, repo_type="dataset", missing_ok=True)
            deleted += 1
            logger.info("repo GC: deleted %s", repo_id)
        except Exception as exc:
            logger.warning("repo GC: failed to delete %s: %s", repo_id, exc)
        time.sleep(_DELETE_SLEEP_S)  # HF repo-mutation rate-limit courtesy
    return deleted
