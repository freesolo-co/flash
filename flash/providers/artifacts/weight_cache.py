"""Preload (warm) the shared weight-cache volumes with the catalog's base-model weights.

Covers BOTH substrates that hold a shared cache -- RunPod network volumes (``warm_weight_cache``)
and Lambda filesystems (``warm_instances``) -- which is why this sits at the provider-neutral level
rather than under one provider package. ``main`` is a single CLI over both: ``--gpu`` documents a
per-mode default for each, and ``--teardown`` reclaims storage on every provider.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from flash._internal.logging import configure_logging, get_logger
from flash.providers._lifecycle.instances.poll import preload_instance_run_id
from flash.providers._lifecycle.net.deadline import deadline_kwargs
from flash.providers.artifacts.hf import make_hf_text_reader

# imported at the top rather than re-exported at the bottom: these are default argument
# values below, and a default evaluates when the `def` runs, long before a bottom import.
# the rest of the runpod preload surface is re-exported here for the same reason, one step
# further along: `main()` runs from the `__main__` guard, which sits ABOVE where these used to be
# imported, so every CLI invocation died on `NameError: catalog_model_ids` while the library path
# stayed green (importing the module executes the whole file, bottom import included). tests that
# patch `weight_cache.<name>` keep working because the names still land in this module's namespace.
from flash.providers.artifacts.preload_runpod import (  # noqa: F401
    _NO_CAPACITY_GRACE_S,
    _PRELOAD_GPU,
    _PRELOAD_TIMEOUT_S,
    _QUEUED,
    _THROTTLED_GRACE_S,
    _UNHEALTHY_GRACE_S,
    NoCapacityError,
    _any_worker,
    _has_worker,
    _only_unhealthy_workers,
    _poll_until_done,
    _preload_one_dc,
    _throttled_workers,
    _worker_counts,
    catalog_model_ids,
    teardown_lambda_filesystems,
    teardown_weight_cache,
    warm_weight_cache,
)
from flash.providers.core.base import UnreconciledCreateError
from flash.providers.runpod.client import api as runpod_api  # noqa: F401

# several names below have no call site here since the runpod half moved to `.preload_runpod`,
# but they are kept imported on purpose: the preload tests patch them on THIS module and that
# half reads them back through it. an autofix that drops them as unused breaks those tests.
from flash.providers.runpod.execution.job_execution import deploy_train_endpoint  # noqa: F401
from flash.providers.runpod.execution.jobs import GraceTimer, decode_output  # noqa: F401
from flash.providers.runpod.execution.resources import (  # noqa: F401
    weight_cache_datacenters,
    weight_cache_volume_name,
)

logger = get_logger(__name__)


def _run_async(coro):
    """Run a coroutine from sync code even if an event loop is already running."""
    import asyncio as _asyncio

    try:
        _asyncio.get_running_loop()
    except RuntimeError:
        return _asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_asyncio.run, coro).result()


_LAMBDA_PRELOAD_GPU = "A10"
# Cheapest-first fallback ladder for warming. A10 alone reaches only the regions that happen to stock
# it, which left most of the fleet permanently cold: the filesystem exists in every region, but the
# warm path could never launch a box there, so those regions were never warmed even once. Preload only
# downloads weights, so any class works and the cheapest that a region actually stocks is the right one.
# Ordered by Lambda list price (see lambdalabs/pricing.py); an explicit --gpu skips the ladder entirely.
_LAMBDA_PRELOAD_GPU_LADDER = ("A10", "A100 SXM 40GB", "H100", "B200")
# Wall budget for planning the whole warm, shared by every class in the ladder. Generous enough that
# a healthy Lambda answers all four classes well inside it, tight enough that a hung /instance-types
# cannot hold the warm hostage for the length of the per-class retry budgets stacked end to end.
_LAMBDA_PLANNING_BUDGET_S = 180.0
# Separate and much smaller: the filesystem snapshot is optional reporting that runs after the
# planning budget is already spent, so it must not extend the pre-launch phase by another
# retry-and-backoff cycle. Losing it only costs a summary line.
_LAMBDA_SNAPSHOT_BUDGET_S = 30.0
# Also separate, and for a stronger reason: the per-region filesystem pre-check must not be charged
# to the run deadline the launch and poll get. Sharing one deadline made a slow pre-check both eat
# into the provider's 60s create allowance (so classes reported "no capacity" untested) and end the
# driver's poll before the instance wall cap it is watching. Its own budget bounds a hung Lambda
# without taking anything from the warm.
_FS_PRECHECK_BUDGET_S = 120.0
_PRELOAD_STATUS_REPO_NAME = "flash-weight-preload"


def _preload_status_repo() -> str:
    """The HF dataset repo the warm boxes report completion into.

    Derived from ``artifact_namespace()`` for the same reason run artifacts are: this repo is
    CREATED with the operator's ``HF_TOKEN``, and a hardcoded ``Freesolo-Co`` made the warm path
    unusable for a self-hoster whose token cannot write there -- with an error telling them to fix
    an ``HF_TOKEN`` that was already correct.
    """
    from flash.runner.accounting.artifacts import artifact_namespace

    return f"{artifact_namespace()}/{_PRELOAD_STATUS_REPO_NAME}"


class IncompleteWarmPlanError(RuntimeError):
    """Some regions warmed, but a class went unanswered so the fleet was never fully measured.

    Carries the results of the launches that DID run: they are real, paid, completed work, and a
    bare raise would throw them away along with the record of which regions are now warm. The caller
    reports them and then treats the run as unfinished rather than as a clean sweep.
    """

    def __init__(self, message: str, *, results: list[dict]):
        super().__init__(message)
        self.results = results


def _lambda_warm_targets(lambda_jobs, gpu: str | None) -> tuple[list[list], bool]:
    """``(one cheapest-first candidate list per Lambda region, planning was complete)``.

    Ranked by each candidate's own ``price_usd_hr``, which ``usable_instances`` fills from the live
    Lambda rate (falling back to the static table only when the live lookup fails). Ranking on the
    ladder's fixed order instead would keep claiming regions in a stale June price order and could
    launch the more expensive class after a Lambda discount.
    """
    classes = [gpu] if gpu else list(_LAMBDA_PRELOAD_GPU_LADDER)
    complete = True
    # ONE deadline across the whole ladder, not one per class. Each usable_instances does a live price
    # lookup and a capacity lookup, and each of those retries internally, so an /instance-types endpoint
    # that accepts connections and then hangs would burn its full retry budget four times over -- turning
    # a single-class stall into ~20 minutes before this can report "no targets". The budget is for
    # planning the fleet, so it belongs to the ladder as a whole.
    deadline = time.time() + _LAMBDA_PLANNING_BUDGET_S
    by_region: dict[str, list] = {}
    for cls in classes:
        if time.time() >= deadline:
            logger.warning(
                "warm lambda: capacity planning budget (%ds) exhausted; skipping remaining "
                "class(es) %s",
                int(_LAMBDA_PLANNING_BUDGET_S),
                ", ".join(classes[classes.index(cls) :]),
            )
            complete = False
            break
        try:
            candidates = lambda_jobs.usable_instances(
                cls,
                **deadline_kwargs(lambda_jobs.usable_instances, deadline),
            )
        except Exception as exc:
            logger.warning("warm lambda: usable_instances(%s) failed (skipping): %s", cls, exc)
            complete = False
            continue
        for c in candidates:
            by_region.setdefault(c.region, []).append(c)
    # stable price ties preserve the static cheapest-first ladder when live pricing is unavailable.
    targets = [
        sorted(cands, key=lambda c: getattr(c, "price_usd_hr", None) or math.inf)
        for _region, cands in sorted(by_region.items())
    ]
    return targets, complete


def _lambda_provisioned_regions() -> set[str]:
    """Regions where the weight-cache filesystem exists, per the Lambda API. Empty if unreadable.

    Deadline-bounded because this runs AFTER ``_lambda_warm_targets`` has already spent the shared
    planning budget, and ``list_filesystems`` retries internally: an endpoint that accepts
    connections then hangs would otherwise add several minutes of attempts and backoff to planning,
    for what is only a reporting nicety. Losing the snapshot degrades the summary; blocking on it
    delays warming.
    """
    from flash.providers.lambda_.client import api as lambda_api
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME

    try:
        fses = lambda_api.list_filesystems(
            **deadline_kwargs(lambda_api.list_filesystems, time.time() + _LAMBDA_SNAPSHOT_BUDGET_S),
        )
    except Exception as exc:
        logger.warning(
            "warm lambda: list_filesystems failed, cannot report unreachable regions: %s", exc
        )
        return set()
    return {
        (fs.get("region") or {}).get("name")
        for fs in fses
        if fs.get("name") == WEIGHT_CACHE_VOLUME_NAME and (fs.get("region") or {}).get("name")
    }


def _ensure_status_repo(token: str | None) -> None:
    """Create the preload status dataset repo if absent. RAISES on failure — call before launching."""
    from huggingface_hub import HfApi

    HfApi(token=token).create_repo(
        _preload_status_repo(), repo_type="dataset", exist_ok=True, private=True
    )


def _preload_instance_spec(gpu: str, run_id: str, wall_s: int = 1800):
    """Minimal download-only spec with cache volume attached and wall cap set to the warm timeout."""
    from flash.core.spec import JobSpec
    from flash.runner.accounting.weight_cache import (
        WEIGHT_CACHE_VOLUME_GB,
        WEIGHT_CACHE_VOLUME_NAME,
    )

    return JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "sft",
            "run_id": run_id,
            "train": {
                "hf_repo": _preload_status_repo(),
                "credit_assignment": "per_episode",
            },
            "gpu": {
                "type": gpu,
                "max_wall_seconds": max(60, int(wall_s)),
                "network_volume": WEIGHT_CACHE_VOLUME_NAME,
                "network_volume_gb": WEIGHT_CACHE_VOLUME_GB,
            },
        }
    )


def _region_filesystem_is_listed(region: str, deadline: float) -> bool:
    """True when this region's weight-cache filesystem is VISIBLE in the account listing.

    Visibility, not a successful create, is the property that matters. Every ``ensure_filesystem``
    call begins by listing and returns early on a match, so once the filesystem is listed no later
    caller can reach the non-idempotent create path for it.
    """
    from flash.providers.lambda_.client import api as lambda_api
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME

    fses = lambda_api.list_filesystems(
        **deadline_kwargs(lambda_api.list_filesystems, deadline),
    )
    return any(
        fs.get("name") == WEIGHT_CACHE_VOLUME_NAME
        and (fs.get("region") or {}).get("name") == region
        for fs in fses
    )


def _ensure_region_filesystem(region: str, deadline: float) -> str:
    """Confirm this region's weight-cache filesystem exists before any paid launch.

    ``launch_and_submit`` calls ``ensure_filesystem`` on every attempt and ``create_filesystem`` is
    NOT idempotent, so the filesystem must be settled before the ladder runs or a rejection on one
    class is followed by a second create for the same name and region -- duplicate storage, billed
    forever. Creating is not enough on its own: a filesystem that exists but has not yet appeared in
    ``list_filesystems()`` makes the launcher's own listing miss and submit that duplicate, so only
    a LISTED filesystem counts as settled. Matching on error text cannot substitute, because
    ``launch_and_submit`` wraps timeouts and real capacity rejections in the same "no capacity"
    message.

    Returns one of:
      ``"listed"``      -- confirmed present, so every later ensure reuses it and cannot create.
      ``"unreachable"`` -- Lambda was never reached, so no create can exist and nothing is at risk.
      ``"doubtful"``    -- reached Lambda but cannot confirm; launching could pay for a second
                           filesystem forever, so the caller must skip the region.
    """
    from flash.providers.lambda_.client import api as lambda_api
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME

    try:
        # Already listed: nothing to create, so skip the create path entirely rather than trusting it
        # to no-op. This is the steady state once provisioning has run.
        if _region_filesystem_is_listed(region, deadline):
            return "listed"
        lambda_api.ensure_filesystem(
            WEIGHT_CACHE_VOLUME_NAME,
            region,
            **deadline_kwargs(lambda_api.ensure_filesystem, deadline),
        )
        if _region_filesystem_is_listed(region, deadline):
            return "listed"
        # Created, but the launcher's listing would still miss it. One cold cycle for this region is
        # recoverable; a duplicate filesystem is billed until someone notices it by hand.
        logger.warning(
            "warm lambda/%s: filesystem created but not yet listed; skipping this cycle "
            "so the launcher cannot create a duplicate",
            region,
        )
        return "doubtful"
    except Exception as exc:
        # No credentials means no request was ever sent, so nothing can have been created. Treating
        # that as doubt would skip every region on any host without Lambda creds -- turning a missing
        # key into a silent no-op instead of the launcher's own explicit failure.
        if not _lambda_is_reachable(exc):
            logger.info("warm lambda/%s: skipping filesystem pre-check (%s)", region, exc)
            return "unreachable"
        logger.warning(
            "warm lambda/%s: filesystem could not be confirmed (%s); skipping this cycle "
            "so the launcher cannot create a duplicate",
            region,
            exc,
        )
        return "doubtful"


def _lambda_is_reachable(exc: Exception) -> bool:
    """False when the failure proves no Lambda API call was ever issued (so no create can exist).

    The text comes from ``RestClient.missing_key_message`` (``_http.py``), the single place a missing
    key is reported, and is matched on the substring both halves of that message share.
    """
    return "not configured" not in str(exc).lower()


def _warm_one_lambda_instance(
    lambda_jobs, candidates: list, models: list, timeout_s: int, poll_interval_s: float
) -> dict:
    """Launch a download-only preload instance in one Lambda region, poll its status marker, then

    ``candidates`` is that region's cheapest-first class list. The GPU class is read off the
    candidate that actually launched, so a mixed-class fleet warm reports what each region really
    cost.
    """
    region = getattr(candidates[0], "region", "?")
    gpu = getattr(candidates[0], "gpu", None) or _LAMBDA_PRELOAD_GPU
    effective_s = max(60, int(timeout_s))
    run_id = None

    def _result(status: str, **extra) -> dict:
        return {"provider": "lambda", "region": region, "gpu": gpu, "status": status, **extra}

    try:
        # Settle the filesystem before any class runs, so every per-attempt
        # ensure_filesystem inside the launcher only ever reuses and can never reach the
        # non-idempotent create path. Give it its OWN budget: charging the pre-check to the run
        # deadline made the driver give up before the box it watches, and it silently ate into the
        # provider's 60s create allowance, so every class in the ladder failed the allowance check
        # inside launch_and_submit and the region reported "no capacity" for classes never tried.
        fs_state = _ensure_region_filesystem(region, time.time() + _FS_PRECHECK_BUDGET_S)
        if fs_state == "doubtful":
            # Launching now would let the launcher's own listing miss and create a duplicate that is
            # billed forever. A region left cold this cycle just downloads on first use.
            return _result(
                "error", error="filesystem unconfirmed; skipped to avoid a duplicate create"
            )
        # One anchor for everything downstream: the driver's poll deadline, the reap deadline
        # embedded in the run_id, and the instance's own wall cap all start here and all run for
        # effective_s, so no clock is ahead of another.
        deadline = time.time() + effective_s
        # Embed reap deadline in the run_id so orphan sweep can free the box if this driver dies.
        run_id = preload_instance_run_id("lambda", region, int(deadline), uuid.uuid4().hex[:6])
        spec = launch_err = None
        for cand in candidates:
            # Rebuild per class: the spec carries the GPU, so reusing the cheap one's spec would
            # launch the very mismatch this fallback exists to avoid.
            gpu = getattr(cand, "gpu", None) or gpu
            spec = _preload_instance_spec(gpu, run_id, wall_s=effective_s)
            try:
                lambda_jobs.launch_and_submit(
                    spec,
                    instances=[cand],
                    attempt=0,
                    mode="preload",
                    models=models,
                    deadline_at=deadline,
                )
                launch_err = None
                break
            except UnreconciledCreateError as exc:
                # An ambiguous create means Lambda may have billed a box we cannot see, and every
                # class here shares one run_id -- launching again could pay for two. This error
                # exists precisely to forbid another create, so it must stop the ladder, not walk it.
                launch_err = exc
                logger.warning(
                    "warm lambda/%s: ambiguous create, not trying another class: %s", region, exc
                )
                break
            except Exception as exc:
                # no capacity / launch reject. Deciding this from the error text
                # is impossible -- ensure_filesystem leaves its reconciliation
                # listing unguarded, and launch_and_submit wraps a filesystem
                # failure and a genuine capacity rejection in the same "no
                # capacity" message -- which is why it is settled before the
                # ladder instead.
                launch_err = exc
                logger.info("warm lambda/%s: %s rejected (%s); trying next class", region, gpu, exc)
        if launch_err is not None or spec is None:
            return _result("error", error=f"launch: {launch_err}")
        prefix = f"{spec.phase}/{run_id}"
        # bound once: both readers must watch the repo the launched spec actually writes to.
        status_repo = spec.train.hf_repo
        reader = make_hf_text_reader(
            status_repo,
            f"{prefix}/preload_result.json",
            min_interval_s=max(5.0, poll_interval_s),
        )
        # Also watch the attempt marker: if the box dies early the failmark is the only signal (avoids
        # polling to full timeout on a dead box). Completion file is authoritative when present.
        fail_reader = make_hf_text_reader(
            status_repo,
            f"{prefix}/lambda_attempt0.json",
            min_interval_s=max(5.0, poll_interval_s),
        )
        logger.info("warm lambda/%s: launched preload (%d models)", region, len(models))
        text = None
        while time.time() < deadline:
            text = reader(force=True)
            if text:
                break
            # No completion file yet — the terminal attempt marker is the backstop: ok=false means the
            # box already died (stop polling, free it now), ok=true means the download SUCCEEDED but
            # only the preload_result.json upload had a transient Hub blip (the worker still wrote a
            # terminal ok=true marker), so the box is ALREADY warmed — short-circuit the wait instead
            # of polling to the full budget then terminating a warmed box and reporting it timed out.
            fail_text = fail_reader(force=True)
            if fail_text:
                try:
                    fail = json.loads(fail_text)
                except Exception:
                    fail = {}
                if fail.get("ok") is True:
                    bad = fail.get("error") or fail.get("failed")
                    return _result("partial" if bad else "ok", result=fail)
                if not fail.get("ok", True):
                    # Completion file is authoritative: a partial run writes it before the fail marker,
                    # so re-check once before reporting early death.
                    text = reader(force=True)
                    if text:
                        break
                    return _result(
                        "error", error=f"box failed early: {fail.get('error') or 'see boot log'}"
                    )
            time.sleep(max(5.0, poll_interval_s))
        if not text:
            return _result("timeout")
        result = json.loads(text)
        bad = result.get("error") or result.get("failed")
        return _result("partial" if bad else "ok", result=result)
    except Exception as exc:
        return _result("error", error=str(exc))
    finally:
        # None means the pre-check returned or raised before a run_id existed, so no launch was ever
        # attempted and there is nothing to reap. Sweeping on None would be a terminate call with no
        # run to scope it to.
        if run_id is not None:
            with contextlib.suppress(Exception):
                lambda_jobs.terminate_run_instances(run_id)


def warm_instances(
    models: list | None = None,
    gpu: str | None = None,
    timeout_s: int = _PRELOAD_TIMEOUT_S,
    poll_interval_s: float = 20.0,
    max_workers: int = 4,
) -> list[dict]:
    """Warm Lambda caches: one download-only launch per region with capacity. Returns status per region."""
    models = models or catalog_model_ids()
    token = os.environ.get("HF_TOKEN")

    from flash.providers.lambda_ import jobs as lambda_jobs

    targets, planned = _lambda_warm_targets(lambda_jobs, gpu)
    # Read before launching: a region with no capacity in any class never becomes a target, so the
    # provisioned set is the only place its name still exists.
    provisioned = _lambda_provisioned_regions()
    if not targets:
        # An empty plan means one of two very different things, and only the first is a healthy
        # no-op: every class answered and none had capacity, or we never got an answer. Reporting
        # the second as "no capacity" would let a Lambda outage exit successfully while the whole
        # fleet stays cold.
        _log_unreachable_lambda_regions(provisioned, [], planned=planned)
        if not planned:
            # Raise rather than return empty: an empty list is indistinguishable from a healthy
            # "nothing to do" at every call site, including the CLI, which would print "no capacity"
            # and exit 0 over a total Lambda outage. The caller must see this as a failure.
            raise RuntimeError(
                "could not determine Lambda capacity: at least one instance-type lookup failed or "
                "was cut off by the planning budget, and the classes that did answer reported no "
                "capacity, so no region was warmed. This is NOT the same as a measured zero-capacity "
                "fleet -- regions reachable only through an unanswered class are unexamined, not "
                "known cold. Check the warnings above for which class(es) went unanswered."
            )
        logger.warning("warm: no Lambda capacity right now (nothing to warm)")
        return []
    logger.info(
        "warm lambda: %d region(s) -> %s",
        len(targets),
        ", ".join(
            f"{getattr(cands[0], 'region', '?')}={'/'.join(getattr(c, 'gpu', '?') for c in cands)}"
            for cands in targets
        ),
    )
    # Fail fast before launching paid GPUs: status repo is the only completion signal.
    try:
        _ensure_status_repo(token)
    except Exception as exc:
        raise RuntimeError(
            f"preload status repo {_preload_status_repo()!r} unavailable ({exc}); set a valid HF_TOKEN "
            "with write access before warming (refusing to launch paid GPUs that can't report)."
        ) from exc
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        # Each region gets its whole cheapest-first list, not one pre-picked class, so the launcher
        # can fall through to a pricier class rather than leaving the region cold.
        futs = [
            ex.submit(
                _warm_one_lambda_instance, lambda_jobs, cands, models, timeout_s, poll_interval_s
            )
            for cands in targets
        ]
        results = [f.result() for f in as_completed(futs)]
    _log_unreachable_lambda_regions(provisioned, results, planned=planned)
    if not planned:
        # The reachable launches above are real work and are kept -- but the fleet was
        # never fully measured, so this run cannot be reported as a finished one. Without
        # this the mixed case (one class unanswered, another still yielding targets)
        # printed "N/N regions warmed" and exited 0, where N counted only the regions we
        # managed to look at.
        raise IncompleteWarmPlanError(
            f"examined {len(results)} region(s), but at least one instance-type lookup failed or "
            "was cut off by the planning budget, so the fleet was not fully measured. Regions "
            "reachable only through an unanswered class were never examined -- they are unmeasured, "
            "not known warm. Re-run once Lambda is answering to cover them.",
            results=results,
        )
    return results


def _unreachable_lambda_regions(provisioned: set[str], results: list[dict]) -> list[str]:
    """Regions provisioned but never launched: no capacity in any ladder class, so no result at all.

    This is the silent fleet gap the summary exists to expose -- a report built from results alone
    cannot see it, because these regions never became targets.
    """
    return sorted(provisioned - {r["region"] for r in results})


def _cold_lambda_regions(provisioned: set[str], results: list[dict]) -> tuple[list[str], int]:
    """``(cold regions, fleet size)``. Cold = did not finish ``ok``, however it got that way.

    Two ways a region ends up cold and a summary built from results alone only sees the first: it
    warmed but did not finish (``timeout``/``partial``/``error``), or it had no capacity in any
    ladder class and never produced a result. The second is reported from the provisioned
    filesystems instead.
    """
    # Anything not "ok" left that region's cache incomplete, so a timeout and a partial are as
    # actionable as an error -- reporting only errors would read as success on a half-warmed fleet.
    incomplete = {r["region"] for r in results if r.get("status") != "ok"}
    unreachable = _unreachable_lambda_regions(provisioned, results)
    # Union, not just the pre-launch snapshot: eager provisioning can succeed in only a subset of
    # regions and launch-time ensure_filesystem backstops the rest, so results may name regions the
    # snapshot never had. Sizing the fleet off the snapshot alone under-counts it -- and could print
    # a denominator smaller than the numerator it is being compared against.
    total = len(provisioned | {r["region"] for r in results})
    return sorted(incomplete | set(unreachable)), total


def _log_unreachable_lambda_regions(
    provisioned: set[str],
    results: list[dict],
    *,
    planned: bool = True,
) -> list[str]:
    """Warn about every region whose cache is not fully warm. Returns them sorted, for printing.

    Returned as well as logged because this is a library module: the ``flash`` logger carries only a
    NullHandler until an app calls ``configure_logging``, so a caller that has not opted in would
    otherwise lose the one message naming regions with no capacity in any class. ``planned`` is
    False when some class went unanswered.
    """
    cold, total = _cold_lambda_regions(provisioned, results)
    if not cold:
        return []
    unreachable = set(_unreachable_lambda_regions(provisioned, results))
    label = "no capacity" if planned else "capacity unknown"
    detail = ", ".join(f"{r} ({label})" if r in unreachable else r for r in cold)
    logger.warning("warm lambda: %d of %d region(s) not fully warmed: %s", len(cold), total, detail)
    return cold


def provision_lambda_filesystems(name: str | None = None) -> list[str]:
    """Eagerly create the weight-cache filesystem in every Lambda region (idempotent, GPU-free).

    Best-effort: zero-capacity regions are covered by the launch-time ensure_filesystem backstop.
    """
    from flash.providers.lambda_.client import api as lambda_api
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME

    target = name or WEIGHT_CACHE_VOLUME_NAME
    done: list[str] = []
    try:
        regions = lambda_api.all_regions()
    except Exception as exc:
        logger.warning("provision: lambda all_regions failed (skipping): %s", exc)
        return done
    for region in regions:
        try:
            lambda_api.ensure_filesystem(target, region, deadline_at=time.time() + 300.0)
            done.append(f"lambda:{region}")
        except Exception as exc:
            logger.warning(
                "provision: lambda ensure_filesystem(%s, %s) failed: %s", target, region, exc
            )
    return done


def _resolve_cli_selection(ap, args) -> tuple[list[str], list[str], bool, int | None]:
    selected_modes = [
        name
        for name, on in (
            ("--provision", args.provision),
            ("--warm-instances", args.warm_instances),
            ("--teardown", args.teardown),
        )
        if on
    ]
    if len(selected_modes) > 1:
        ap.error(f"{', '.join(selected_modes)} are mutually exclusive — pass exactly one mode")

    catalog = catalog_model_ids()
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else catalog
    # Reject off-catalog ids before any download: private/gated weights must not land on the shared cache.
    if args.models and not args.teardown and not args.provision:
        off_catalog = [m for m in models if m not in set(catalog)]
        if off_catalog:
            print(
                "--models: refusing to preload off-catalog model id(s) into the shared cache: "
                f"{', '.join(off_catalog)} — only public catalog models may be warmed (private/gated "
                "repos would leak onto the platform-wide shared volume). They download cold on first "
                "use instead."
            )
            return models, [], False, 2
    # `--datacenters ""` must error, not silently widen to a full fleet teardown.
    dcs_given = args.datacenters is not None
    parsed_dcs = [d.strip() for d in args.datacenters.split(",") if d.strip()] if dcs_given else []
    if dcs_given and not parsed_dcs:
        print(
            "--datacenters was given but parsed to no datacenter ids — refusing to run "
            "(an empty scope would delete the WHOLE RunPod fleet); drop --datacenters for a full "
            "teardown, or pass real DC ids."
        )
        return models, parsed_dcs, False, 2
    return models, parsed_dcs, bool(parsed_dcs), None


def _default_dcs() -> list[str]:
    # Lazy: weight_cache_datacenters() imports runpod_flash; avoid importing it on instance-only hosts.
    return [dc.value for dc in weight_cache_datacenters()]


def _run_lambda_mode(args, models: list[str]) -> int | None:
    if args.provision:
        if args.dry_run:
            print("would provision Lambda filesystems in every region")
            return 0
        provisioned = provision_lambda_filesystems()
        print(
            f"provisioned {len(provisioned)} Lambda filesystem(s): "
            f"{', '.join(provisioned) or '(none: no Lambda key or no regions)'}"
        )
        return 0
    if not args.warm_instances:
        return None
    if args.dry_run:
        print("would warm Lambda caches (one download-only launch per region with capacity)")
        return 0
    incomplete = ""
    try:
        results = warm_instances(
            models=models, gpu=args.gpu, timeout_s=args.timeout_s, max_workers=args.max_workers
        )
    except IncompleteWarmPlanError as exc:
        # Still print the per-region lines below: those launches ran and were paid for, and a
        # bare traceback would hide which regions are now warm. The run is reported as
        # unfinished at the end instead of as a clean sweep.
        results, incomplete = exc.results, str(exc)
    except RuntimeError as exc:
        # A total planning outage or an unusable status repo aborts before any launch, so there
        # is nothing paid to report. Exit non-zero with the message instead of a traceback: this
        # is an operator-actionable condition (Lambda down, HF_TOKEN missing), not a crash.
        print(f"0 regions warmed — {exc}")
        return 1
    if not results:
        if incomplete:
            print(f"0 regions warmed — {incomplete}")
            return 1
        print(
            "0 regions warmed — no Lambda region had capacity to warm right now "
            "(weights download cold on first run). Nothing launched."
        )
        return 0
    failed = [r for r in results if r.get("status") not in ("ok",)]
    for r in results:
        # Name the GPU: without --gpu the ladder picks per region, so this is the only place the
        # operator can see which paid class each region actually billed.
        gpu_note = f" on {r['gpu']}" if r.get("gpu") else ""
        print(
            f"  {r['provider']}/{r['region']}: {r['status']}{gpu_note}"
            + (f" ({r.get('error')})" if r.get("error") else "")
        )
    print(f"{len(results) - len(failed)}/{len(results)} regions warmed")
    if incomplete:
        # NOT "N/N regions warmed ... exit 0": the denominator counts only the regions we could
        # see, so a perfect ratio here means the gap is invisible, not absent.
        print(f"WARNING: {incomplete}")
        return 1
    return 1 if failed else 0


def _run_teardown_mode(args, parsed_dcs: list[str], scoped: bool) -> int | None:
    if not args.teardown:
        return None
    if scoped:
        from runpod_flash.core.resources.datacenter import DataCenter

        bad = []
        for d in parsed_dcs:
            try:
                DataCenter.from_string(d)
            except Exception:
                bad.append(d)
        if bad:
            print(
                f"--teardown --datacenters: invalid datacenter id(s): {', '.join(bad)} "
                "— refusing to run (nothing deleted)"
            )
            return 2
    if args.dry_run:
        scope_desc = (
            f"{len(parsed_dcs)} datacenter(s): {', '.join(parsed_dcs)}"
            if scoped
            else "every RunPod storage datacenter"
        )
        print(
            f"would delete the RunPod weight-cache volumes in {scope_desc}"
            + ("" if scoped else " + every Lambda filesystem named flash-weights")
        )
        return 0
    deleted: list[str] = []
    try:
        deleted += teardown_weight_cache(parsed_dcs or None)
    except Exception as exc:
        logger.warning("teardown: RunPod cache teardown failed (continuing): %s", exc)
    # Scoped (--datacenters) teardown is RunPod-only; Lambda regions are a different namespace.
    if not scoped:
        try:
            deleted += teardown_lambda_filesystems()
        except Exception as exc:
            logger.warning("teardown: Lambda cache teardown failed (continuing): %s", exc)
    else:
        print("scoped teardown (--datacenters): RunPod-only; Lambda caches left intact")
    print(f"deleted {len(deleted)} weight-cache volume(s): {', '.join(deleted) or '(none)'}")
    return 0


def _run_runpod_warm(args, models: list[str], parsed_dcs: list[str]) -> int:
    dcs = parsed_dcs or _default_dcs()
    if args.dry_run:
        print(f"would warm {len(dcs)} datacenter(s): {', '.join(dcs)}")
        print(f"with {len(models)} model(s): {', '.join(models)}")
        return 0

    results = warm_weight_cache(
        models=models,
        datacenters=dcs,
        gpu=args.gpu or _PRELOAD_GPU,
        timeout_s=args.timeout_s,
        max_workers=args.max_workers,
    )
    failed = [r for r in results if r.get("status") != "ok"]
    for r in results:
        print(
            f"  {r['datacenter']}: {r['status']}"
            + (f" ({r.get('error')})" if r.get("error") else "")
        )
    print(f"{len(results) - len(failed)}/{len(results)} datacenters warmed")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    # This module is a library: the `flash` logger carries only a NullHandler until an app opts in,
    # so every logger.warning here -- including the one naming regions with no capacity in any class,
    # which has no other output path -- is discarded when run as __main__. An operator running the
    # documented entry point would see "N/N regions warmed" and exit 0 over a half-cold fleet.
    configure_logging()
    ap = argparse.ArgumentParser(description="Preload the flash weight-cache volumes.")
    ap.add_argument("--models", help="comma-separated HF model ids (default: whole catalog)")
    ap.add_argument("--datacenters", help="comma-separated DC ids (default: all storage DCs)")
    ap.add_argument(
        "--gpu",
        default=None,
        help="GPU class for the preload worker. Defaults are per-mode: RunPod warm -> "
        f"{_PRELOAD_GPU!r}; --warm-instances -> the cheapest class each region actually stocks, "
        f"tried in the order {' -> '.join(_LAMBDA_PRELOAD_GPU_LADDER)}, so a region with no "
        "cheap capacity is warmed on a pricier class instead of being skipped. Pass this to "
        "pin ONE class everywhere and disable that fallback (a region that does not stock it "
        "is then left cold). Defaulting to None (not a sentinel string) lets you explicitly "
        "pick even a default GPU without it being mistaken for 'no override'.",
    )
    ap.add_argument(
        "--timeout-s",
        type=int,
        default=_PRELOAD_TIMEOUT_S,
        help="per-DC job timeout (default sized for a fully cold whole-catalog warm)",
    )
    ap.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="datacenters warmed concurrently. Each one deploys a preload endpoint, so this MUST stay "
        "under your RunPod endpoint/worker quota (the documented default is 5); the default of 4 "
        "leaves a 1-slot buffer. Raise it only if your account quota is higher.",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the plan, provision nothing")
    ap.add_argument(
        "--provision",
        action="store_true",
        help="CREATE the Lambda weight-cache filesystem in every region (pure API, no GPU) and "
        "exit; RunPod volumes are auto-created by the eager deploy/warm. Run before --teardown's "
        "inverse to set up all storage up front.",
    )
    ap.add_argument(
        "--warm-instances",
        action="store_true",
        help="WARM the Lambda caches: one download-only GPU launch per region with "
        "capacity now (needs the merged worker image carrying the bootstrap preload branch).",
    )
    ap.add_argument(
        "--teardown",
        action="store_true",
        help="DELETE the weight-cache storage on every provider (reclaim standing storage) and exit. "
        "With --datacenters it is SCOPED to that RunPod-DC subset only (Lambda caches "
        "are left intact, since DC ids don't map to their region namespace).",
    )
    args = ap.parse_args(argv)
    models, parsed_dcs, scoped, selection_error = _resolve_cli_selection(ap, args)
    if selection_error is not None:
        return selection_error

    lambda_result = _run_lambda_mode(args, models)
    if lambda_result is not None:
        return lambda_result
    teardown_result = _run_teardown_mode(args, parsed_dcs, scoped)
    if teardown_result is not None:
        return teardown_result
    return _run_runpod_warm(args, models, parsed_dcs)


if __name__ == "__main__":
    raise SystemExit(main())
