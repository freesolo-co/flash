"""Always-on GC for aged run artifacts inside the per-environment HF repos (``Freesolo-Co/flashrun-*``).

Every managed run stores its code snapshot, adapter, checkpoints, and telemetry under a per-run prefix
``<phase>/<run_id>/`` inside a *private* HF dataset repo that is now shared by every run of an
environment (``managed_hf_repo_for_environment`` -> ``Freesolo-Co/flashrun-<slug>-<digest>``). The
deployable per-step adapters (``.../checkpoints/step-N/adapter``) are kept forever by the trainer and
nothing else deletes old runs, so these repos grow without bound against the org's private-storage
quota.

An earlier GC (PR #311) deleted whole ``flashrun-<run_id>`` repos, but that predated the switch to
environment-scoped repos (#346, "Remove legacy HF repo cleanup pressure"): a shared env repo can hold
a live run alongside stale ones, so a *whole-repo* delete is no longer safe and the GC was removed.
This is its environment-scoped replacement — it deletes the aged run's PREFIX (``delete_folder`` on
``<phase>/<run_id>``), never the whole repo, so an env's live runs are untouched.

The control plane runs this sweep AUTOMATICALLY — ``flash.server._runtime._repo_cleanup_loop`` calls
``run_scheduled_cleanup`` on an interval (daily). There is ONE fixed, opinionated policy:

    Delete every managed run's ``<phase>/<run_id>/`` prefix that is NOT currently deployed and NOT
    in-flight, once the run finished more than the GC age (fixed: 7 days) ago.

There is intentionally **no manual CLI and no tiers/flags**: a redeployable-but-undeployed run is not
spared and contents are not trimmed selectively — the whole run prefix (including its final adapter)
goes once it ages out and isn't serving. The shared ``code/<digest>/`` snapshot is content-addressed
across runs and is never touched.

Safety. Serving pulls adapter weights straight from these prefixes, so the live serving set is the
do-not-touch set:

* A prefix a run is currently serving from is never deleted.
* If the live set can't be confirmed — run registry unreadable, or a live run whose repo/prefix can't
  be mapped — the sweep deletes NOTHING and retries next cycle (fails closed; never deletes blind).
* The live set is re-confirmed immediately before every delete, so a run deployed mid-sweep is spared
  (TOCTOU), and the deleted prefix is re-checked for recent writes right before deletion so a
  warm-start touch or a late checkpoint spares it.
* Only ``flashrun-*`` repos are ever touched (hard prefix allowlist), only TERMINAL runs are reaped,
  and a run whose age can't be determined is left alone.
* Only runs THIS control plane issued are reaped (``_known_run_ids``), so a sibling plane's run
  sharing the same env repo is never deleted.

There is nothing to configure: the policy (7-day age, daily sweep) is fixed, with no env knobs. The
loop only runs on a plane that has an operator ``HF_TOKEN`` — without it the sweep cannot delete
operator-owned repos at all.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from flash._logging import get_logger
from flash.runner import _ARTIFACT_NAMESPACE

logger = get_logger(__name__)

# huggingface_hub is an OPTIONAL server extra (see pyproject ``[server]``). The rest of the control
# plane already treats it as optional, so mirror that here: without the extra there is no way to list
# or delete repos, so the always-on GC degrades to a logged no-op instead of crashing the loop with a
# ``ModuleNotFoundError`` on every cycle.
try:
    from huggingface_hub import HfApi
except ModuleNotFoundError:  # pragma: no cover - the test venv always has the server extra
    HfApi = None  # type: ignore[assignment,misc]

# Latch so a plane missing the extra warns ONCE (the sweep runs daily — don't spam the log).
_warned_hf_unavailable = False

# Hard allowlist: only Freesolo-Co/flashrun-* dataset repos are ever touched (never env packages,
# paper-*/oracle/eval sets, or a user's own datasets).
RUN_REPO_PREFIX = "flashrun-"
# Delete an undeployed, terminal run's prefix once it finished this long ago (fixed: 7 days).
DELETE_AGE_SECONDS = 7.0 * 86400.0
_DELETE_SLEEP_S = 0.5  # pause between deletes — HF repo-mutation rate-limit courtesy

# Deployment-record states that mean NOTHING is currently serving from the run's prefix, so the GC may
# reap an aged run despite the record: an explicit undeploy, a dry-run, or a FAILED deploy.
# ``mark_deployment_failed`` only lands on ``failed`` when there is no live previous deployment to
# restore (a redeploy over a live adapter restores the old one instead), and ``recover_deployments``
# turns restart-interrupted deploys into ``failed`` too — so ``failed`` provably serves nothing. Every
# OTHER state (``ready``/``deployed``/``deploying``/unknown) is a conservative do-not-touch: it may back
# a live or in-progress serving container. A mid-deploy run is additionally shielded by its held lock.
_INACTIVE_DEPLOYMENT_STATES = frozenset({"undeployed", "dry_run", "failed"})


class CleanupAborted(RuntimeError):
    """The live/known set could not be confirmed, so the sweep deleted nothing (fails closed)."""


@dataclass(frozen=True)
class _RunTarget:
    run_id: str
    repo_id: str
    prefix: str  # "<phase>/<run_id>"
    age_ts: float | None  # epoch seconds of the run's terminal time, or None if undeterminable


def _now() -> float:
    """Wall-clock epoch seconds. A seam so tests can freeze time deterministically."""
    return time.time()


def repo_cleanup_enabled() -> bool:
    """Whether the always-on artifact GC should run on this control plane.

    Requires an operator ``HF_TOKEN`` — the sweep deletes operator-owned ``flashrun-*`` dataset
    prefixes, impossible (and meaningless) without it — so a plane without the token never schedules
    the loop. This is a credential check, not a knob: there is no on/off env switch."""
    return bool((os.environ.get("HF_TOKEN") or "").strip())


def _is_managed_env_repo(repo_id: str) -> bool:
    """True only for ``Freesolo-Co/flashrun-*`` dataset repos (the hard allowlist)."""
    if not isinstance(repo_id, str):
        return False
    return repo_id.startswith(f"{_ARTIFACT_NAMESPACE}/") and repo_id.split("/", 1)[-1].startswith(
        RUN_REPO_PREFIX
    )


def _run_repo_prefix(status) -> tuple[str, str] | None:
    """``(repo_id, "<phase>/<run_id>")`` for a run, or ``None`` when it can't be mapped.

    Reads the persisted spec dict directly (no ``JobSpec`` construction) so a single malformed spec
    never crashes the sweep: a run we can't map is treated as unidentifiable by the caller (fail
    closed for the live set, skip for a delete target)."""
    spec = status.spec or {}
    train = spec.get("train") or {}
    repo = train.get("hf_repo")
    if not isinstance(repo, str) or not repo.strip():
        return None
    algorithm = str(spec.get("algorithm") or "")
    # Mirror flash.spec.JobSpec.phase: grpo runs live under "rl", everything else under its algorithm.
    phase = "rl" if algorithm == "grpo" else algorithm
    run_id = getattr(status, "run_id", None)
    if not phase or not run_id:
        return None
    return (repo.strip(), f"{phase}/{run_id}")


def _source_repo_prefix(ref) -> tuple[str, str] | None:
    """Parse a resolved ``init_from_adapter`` storage ref into the SOURCE ``(repo, "<phase>/<run_id>")``.

    The worker downloads a warm-start source's adapter at boot, so a still-running dependent run must
    keep its source alive. The resolved ref is ``<owner>/<repo>:<phase>/<run_id>[/checkpoints/step-N]``
    (``flash.schema.checkpoint_storage_ref``); the public ``<run_id>[/step-N]`` form (no ``:``) has no
    repo to protect and is ignored. Returns ``None`` for anything unparseable."""
    if not isinstance(ref, str) or ":" not in ref:
        return None
    repo, _, tail = ref.partition(":")
    segments = [s for s in tail.split("/") if s]
    if not repo or len(segments) < 2:
        return None
    return (repo, f"{segments[0]}/{segments[1]}")


def _warmstart_source_prefix(ref, get_status) -> tuple[str, str] | None:
    """The SOURCE ``(repo, "<phase>/<run_id>")`` a run warm-starts from (``init_from_adapter``), or None.

    The PERSISTED spec carries the PUBLIC ref ``<run_id>[/step-N]`` — ``submit_job`` persists the
    pre-resolution spec, and the internal ``<repo>:<phase>/<run_id>`` storage ref is handed only to the
    worker — so the common case is a bare source RUN ID that must be resolved to its repo/prefix via
    ``get_status``. The internal colon form is also accepted defensively. Returns None for a blank ref,
    or for a cross-plane source whose run record isn't on this plane (``get_status`` raises
    ``FileNotFoundError``): that source can't be resolved here — a documented limitation the per-delete
    re-stat narrows."""
    if not isinstance(ref, str) or not ref.strip():
        return None
    if ":" in ref:
        return _source_repo_prefix(ref)
    source_run_id = ref.split("/", 1)[0].strip()
    if not source_run_id:
        return None
    try:
        source_status = get_status(source_run_id)
    except FileNotFoundError:
        return None
    return _run_repo_prefix(source_status)


def deployed_prefixes() -> tuple[set[tuple[str, str]], bool]:
    """The protected ``(repo_id, "<phase>/<run_id>")`` prefixes for runs with a live-or-recent
    deployment (the do-not-delete set), plus a ``complete`` flag that is ``False`` when such a run
    couldn't be mapped to a repo/prefix (schema drift / malformed spec). The sweep refuses to delete
    against an incomplete or unreachable set.

    Reconstructed from run status (the old ``serve.deploy.list_deployed_adapters`` no longer exists):
    a run is protected iff it carries a ``deployment`` record whose state is NOT one of the
    provably-not-serving states (``_INACTIVE_DEPLOYMENT_STATES``: ``undeployed``/``dry_run``/
    ``failed``). This is a deliberately CONSERVATIVE superset of "currently serving" — a mid-transition
    ``deploying`` (or any unknown) state is protected too (it may already back a warm serving
    container), erring toward keeping rather than a blind delete — while a ``failed`` deploy, which
    provably serves nothing, is NOT pinned out of GC forever. Whole-run in-flight protection is handled
    separately by ``_inflight_protected_prefixes``."""
    from flash.runner import get_status
    from flash.server import db

    ids: set[tuple[str, str]] = set()
    complete = True
    for row in db.all_runs():
        try:
            status = get_status(row["run_id"])
        except FileNotFoundError:
            continue
        deployment = getattr(status, "deployment", None)
        if not deployment or deployment.get("state") in _INACTIVE_DEPLOYMENT_STATES:
            continue
        target = _run_repo_prefix(status)
        # A live run whose repo/prefix can't be resolved is NOT coerced — that would drop a serving
        # prefix from the protected set. Treat it as an unidentifiable live run and fail closed.
        if target is None:
            complete = False
            continue
        ids.add(target)
    return ids, complete


def _confirm_live_set() -> set[tuple[str, str]]:
    """Return the live serving prefix set, or raise ``CleanupAborted``. Called before listing AND
    before every delete; both must fail closed so a blind delete is impossible."""
    try:
        deployed, complete = deployed_prefixes()
    except Exception as exc:  # noqa: BLE001 - any failure to read the live set must fail closed
        raise CleanupAborted(f"serving live set unreachable ({exc}); deleting nothing") from exc
    if not complete:
        raise CleanupAborted("a live run could not be mapped to a repo/prefix; deleting nothing")
    return deployed


def _inflight_protected_prefixes() -> set[tuple[str, str]]:
    """Prefixes PROTECTED regardless of age because an IN-FLIGHT (queued/provisioning/running) run
    still needs them: the run's OWN prefix AND any warm-start SOURCE prefix it will download.

    The deployed-set + age gates alone can't shield an in-flight run: its prefix is created before any
    checkpoint lands, so ``finished_at`` is unset and it is never in serving's deployed set. A run can
    also ``init_from_adapter`` off another run (e.g. GRPO continuing an SFT run); the worker
    ``snapshot_download``s that SOURCE at boot, so an old undeployed source must survive until the
    dependent run has fetched it. Mirrors how the sibling reapers shield ``_RECOVERABLE`` runs. Raises
    (the sweep fails closed) if run state can't be enumerated.

    NOTE: a warm-start dependent launched by ANOTHER control plane is invisible to this plane's
    registry, so a cross-plane source is not protected here — the deliberate cost of a purely local
    scan (same trade-off as the sibling instance reapers). The per-delete re-stat below still spares a
    source whose prefix was written recently, which narrows that window."""
    from flash.runner import get_status
    from flash.server import db
    from flash.server._runtime import _RECOVERABLE

    ids: set[tuple[str, str]] = set()
    for row in db.all_runs():
        try:
            status = get_status(row["run_id"])
        except FileNotFoundError:
            continue
        if status.state not in _RECOVERABLE:
            continue
        own = _run_repo_prefix(status)
        if own is not None:
            ids.add(own)
        source = _warmstart_source_prefix(
            ((status.spec or {}).get("train") or {}).get("init_from_adapter"), get_status
        )
        if source is not None:
            ids.add(source)
    return ids


def _known_run_ids() -> set[str]:
    """The run ids THIS control plane has a record of, in ANY state — the only runs the GC may reap.

    Mirrors the instance reaper's multi-plane guard (``flash.server.app._known_run_ids``): two control
    planes sharing one HF org carry DISJOINT server-assigned run ids, and after #346 they may even
    share an env repo. A run launched by ANOTHER plane is absent from this SQLite db and not yet in
    serving, so the age gate alone would make its prefix a target — this plane would reap a sibling's
    run out of the shared repo. Restricting deletes to runs this plane issued shields it. Raises if the
    registry can't be enumerated, so the sweep fails closed."""
    from flash.server import db

    return {row["run_id"] for row in db.all_runs()}


def _hold_run_lock(run_id: str):
    """Non-blocking acquire-and-HOLD of the per-run deploy/export lock for ``run_id``.

    Returns the held lock — the caller MUST ``release()`` it once the delete is done — or ``None`` when
    a deploy/undeploy/export currently holds it. The GC holds this lock (the SAME one
    ``/v1/runs/{run_id}/deploy``, ``/undeploy`` and ``/export`` take) ACROSS the delete so the
    destructive mutation is mutually exclusive with adapter registration AND export. Best-effort /
    in-process; never blocks the sweep waiting on a slow deploy/export — it just spares the run this
    cycle."""
    from flash.server._locks import _deploy_lock

    lock = _deploy_lock(run_id)
    if not lock.acquire(blocking=False):
        return None
    return lock


def _terminal_run_targets() -> list[_RunTarget]:
    """Every TERMINAL run this plane issued whose artifacts live in a managed ``flashrun-*`` repo.

    In-flight (``_RECOVERABLE``) and live-``deployed`` runs are excluded here and are additionally
    protected by the in-flight / deployed sets; only genuinely finished runs are reap candidates. The
    age is the run's terminal time (``finished_at``), falling back to ``updated_at``/``created_at`` and
    finally the epoch embedded in the run id, so a legacy run without ``finished_at`` is still dateable
    (and a run we cannot date at all is skipped by ``_deletable``). Raises if the registry can't be
    enumerated, so the sweep fails closed."""
    from flash.runner import TERMINAL_STATES, get_status
    from flash.server import db

    out: list[_RunTarget] = []
    for row in db.all_runs():
        try:
            status = get_status(row["run_id"])
        except FileNotFoundError:
            continue
        if status.state not in TERMINAL_STATES:
            continue
        mapped = _run_repo_prefix(status)
        if mapped is None:
            continue
        repo_id, prefix = mapped
        if not _is_managed_env_repo(repo_id):
            continue
        age_ts = (
            getattr(status, "finished_at", None)
            or getattr(status, "updated_at", None)
            or getattr(status, "created_at", None)
            or _run_id_epoch(status.run_id)
        )
        out.append(_RunTarget(run_id=status.run_id, repo_id=repo_id, prefix=prefix, age_ts=age_ts))
    return out


def _run_id_epoch(run_id: str) -> float | None:
    """The submit epoch embedded in ``flash-<epoch>-<hex>`` (``new_run_id``), or ``None``."""
    parts = (run_id or "").split("-")
    if len(parts) >= 3 and parts[0] == "flash" and parts[1].isdigit():
        return float(parts[1])
    return None


def _deletable(
    target: _RunTarget,
    protected: set[tuple[str, str]],
    known: set[str],
    now: float,
    max_age_s: float,
) -> bool:
    if target.run_id not in known:
        return False  # a sibling plane's run sharing this repo — never touch
    if (target.repo_id, target.prefix) in protected:
        return False  # currently serving OR an in-flight run / warm-start source
    if target.age_ts is None:
        return False  # unknown age — never delete on a guess
    return (now - target.age_ts) >= max_age_s


def _prefix_written_within(api, repo_id: str, prefix: str, now: float, max_age_s: float) -> bool:
    """True if any file under ``prefix`` was committed within ``max_age_s`` (=> recently active, spare
    it). Closes the window between enumeration and delete: a warm-start touch, a late checkpoint, or a
    redeploy read-back that landed after enumeration must spare the prefix. Raises on an API error (the
    caller then skips this target to stay safe). If the listing carries no commit dates (older
    ``huggingface_hub``), returns ``False`` — the ``finished_at`` age gate already qualified it."""
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


def run_scheduled_cleanup(*, dry_run: bool = False, api=None, should_stop=None) -> int:
    """One sweep of the fixed policy. Returns the number of run prefixes deleted (0 in dry-run).

    Fails closed: raises ``CleanupAborted`` (deleting nothing) when the serving live set can't be
    confirmed up front. The HF + registry calls are blocking, so callers offload this to a thread.

    ``should_stop`` is an optional cooperative-cancel callback checked BETWEEN deletes. The sweep runs
    in a worker thread that ``task.cancel()`` cannot interrupt, so at shutdown the caller sets a stop
    flag and this loop halts promptly instead of churning through more destructive deletes long after
    the server was told to stop (mirrors the completion-charge retry sweeps)."""
    global _warned_hf_unavailable
    if api is None:
        if HfApi is None:
            # No HF client on this plane -> nothing to list or delete. Degrade to a no-op (warn ONCE
            # so a misconfigured plane is visible without the daily sweep spamming the log) rather than
            # crash the always-on loop with ModuleNotFoundError every cycle.
            if not _warned_hf_unavailable:
                logger.warning(
                    "repo GC: huggingface_hub not installed (server extra absent); skipping sweeps"
                )
                _warned_hf_unavailable = True
            return 0
        api = HfApi()
    max_age_s = DELETE_AGE_SECONDS

    deployed = _confirm_live_set()  # fail closed before we even list
    # In-flight runs (and their warm-start sources) are protected regardless of age. Snapshotted once
    # per sweep like the sibling reapers; raises -> the sweep fails closed.
    protected = deployed | _inflight_protected_prefixes()
    # Multi-plane guard: only reap runs THIS plane issued. Raises -> fails closed.
    known = _known_run_ids()
    now = _now()
    targets = [
        t for t in _terminal_run_targets() if _deletable(t, protected, known, now, max_age_s)
    ]
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
        # Acquire-and-HOLD the per-run deploy/export lock across the delete so the destructive mutation
        # is mutually exclusive with a concurrent deploy/undeploy/export of this run. Non-blocking: if
        # one already holds it, spare the run this cycle rather than wait. None => skip.
        held = _hold_run_lock(target.run_id)
        if held is None:
            logger.warning(
                "repo GC: %s has a deploy/undeploy/export in progress; skipping", target.run_id
            )
            continue
        try:
            # Re-confirm the live set immediately before the delete — now UNDER the held lock, so a
            # deploy can't register this run between the confirm and the delete. If it can't be
            # confirmed now, abort the WHOLE sweep (re-raise) rather than press on deleting other
            # prefixes while an unidentified live run may be backed by one of them.
            fresh = _confirm_live_set()  # raises CleanupAborted -> finally releases, sweep fails closed
            if (target.repo_id, target.prefix) in fresh:
                logger.warning("repo GC: %s became deployed mid-sweep; skipping", target.prefix)
                continue
            # Re-check the in-flight / warm-start-source set too: a run submitted mid-sweep (after the
            # up-front snapshot) could warm-start off THIS aged source, and its worker fetches the
            # source at boot. Fail SAFE (skip this target, don't abort the sweep) if the registry can't
            # be re-read — an unreadable registry is not proof the target is free to delete.
            try:
                if (target.repo_id, target.prefix) in _inflight_protected_prefixes():
                    logger.warning(
                        "repo GC: %s became in-flight / a warm-start source mid-sweep; skipping",
                        target.prefix,
                    )
                    continue
            except Exception as exc:  # noqa: BLE001 - can't confirm it's free -> don't delete blind
                logger.warning(
                    "repo GC: in-flight re-check for %s failed; skipping to stay safe (%s)",
                    target.prefix,
                    exc,
                )
                continue
            # Re-stat the prefix right before deleting: its newest file may have been written since
            # enumeration (a cross-plane warm-start touch, a late checkpoint, a redeploy read-back).
            try:
                if _prefix_written_within(api, target.repo_id, target.prefix, now, max_age_s):
                    logger.warning(
                        "repo GC: %s was written since enumeration (now within age); skipping",
                        target.prefix,
                    )
                    continue
            except Exception as exc:  # noqa: BLE001 - a re-stat failure must not delete blind
                logger.warning(
                    "repo GC: re-stat of %s:%s failed; skipping to stay safe (%s)",
                    target.repo_id,
                    target.prefix,
                    exc,
                )
                continue
            try:
                # delete_folder has no missing_ok; a concurrent delete / already-gone prefix raises.
                api.delete_folder(
                    path_in_repo=target.prefix, repo_id=target.repo_id, repo_type="dataset"
                )
                deleted += 1
                logger.info("repo GC: deleted %s:%s", target.repo_id, target.prefix)
            except Exception as exc:  # noqa: BLE001 - one failed delete must not abort the sweep
                logger.warning(
                    "repo GC: failed to delete %s:%s: %s", target.repo_id, target.prefix, exc
                )
        finally:
            held.release()
        time.sleep(_DELETE_SLEEP_S)  # HF repo-mutation rate-limit courtesy
    return deleted
