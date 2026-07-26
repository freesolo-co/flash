"""Preload (warm) the shared weight-cache volumes with the catalog's base-model weights.

Covers BOTH substrates that hold a shared cache -- RunPod network volumes (``warm_weight_cache``)
and Lambda filesystems (``warm_instances``) -- which is why this sits at the provider-neutral level
rather than under one provider package. ``main`` is a single CLI over both: ``--gpu`` documents a
per-mode default for each, and ``--teardown`` reclaims storage on every provider.

Run it::

    python -m flash.providers.weight_cache_preload                 # all catalog models, all DCs
    python -m flash.providers.weight_cache_preload --datacenters US-CA-2,EU-RO-1 --models Qwen/Qwen3.5-4B
    python -m flash.providers.weight_cache_preload --dry-run       # print the plan, provision nothing
    python -m flash.providers.weight_cache_preload --provision     # CREATE lambda filesystems, no GPU
    python -m flash.providers.weight_cache_preload --warm-instances  # warm the lambda caches
    python -m flash.providers.weight_cache_preload --teardown      # DELETE the cache volumes (reclaim $)
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

from flash._logging import configure_logging, get_logger
from flash.providers._deadline import deadline_kwargs
from flash.providers._hf_artifacts import make_hf_text_reader
from flash.providers._poll import preload_instance_run_id
from flash.providers.base import UnreconciledCreateError
from flash.providers.runpod import api as runpod_api
from flash.providers.runpod.jobs import (
    GraceTimer,
    build_function_input,
    decode_output,
    deploy_train_endpoint,
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


_HF_HOME = "/runpod-volume/hf-cache"
_PRELOAD_GPU = "RTX 4090"
_TERMINAL_OK = {"COMPLETED"}
_TERMINAL_FAIL = {"FAILED", "CANCELLED", "TIMED_OUT"}
# Only a QUEUED job can be starved. Any other KNOWN status means RunPod already handed the job to a
# worker, so a zero-worker health reading there is a health-reporting artifact, not starvation -- and
# acting on it would delete the endpoint mid-download. Statuses outside these sets are not evidence
# in either direction: they are treated as unknown rather than as proof the job left the queue.
_QUEUED = "IN_QUEUE"
_RUNNING = {"IN_PROGRESS"}
# Not every storage DC stocks every GPU class -- US-KS-2/US-MO-2/US-NC-2/US-NE-1/US-WA-1 carry no
# RTX 4090 at all. A preload pinned to a class the DC cannot serve never allocates a worker: the job
# just sits queued until the timeout, so the DC stays silently cold and we pay for the wait. Give up
# on a DC once it has held a queued job this long with no worker in any state.
_NO_CAPACITY_GRACE_S = 420.0
# A worker that RunPod allocated and then marked unhealthy is a broken image, not a cold DC, and no
# amount of waiting fixes it. Without its own timer that case reads as "capacity found" forever and
# holds a paid endpoint for the whole _PRELOAD_TIMEOUT_S. Matches poll_job's unhealthy_grace_s, which
# calls the same condition a failed image pull and retries on a fresh endpoint.
_UNHEALTHY_GRACE_S = 240.0


class NoCapacityError(RuntimeError):
    """A datacenter never allocated a worker, i.e. it cannot serve the requested GPU class."""


# Per-DC job budget for a preload. A fully cold volume must download the WHOLE catalog, so this is
# sized off the measured worst case rather than left at a round number: a real cold 35B pull ran
# 70 GB in ~870s (~0.08 GB/s), which puts the full ~159 GB catalog near 2000s -- already past a
# 1800s budget before any retry or slow-mirror variance. Cold warms are rare and a too-short budget
# throws away everything downloaded so far, so the asymmetry favours the larger number.
_PRELOAD_TIMEOUT_S = 5400


def _worker_counts(endpoint_id: str, key_fingerprint: str, deadline: float) -> dict | None:
    """Per-state worker counts for the endpoint, or None when health could not be read.

    None is deliberately not an empty dict: an empty dict is a positive "this endpoint has no
    workers" answer, which the caller escalates into NoCapacityError and an endpoint teardown. A
    health API that is merely unreachable must never look like that.
    """
    try:
        health = runpod_api.endpoint_health_for_fingerprint(
            endpoint_id,
            key_fingerprint,
            deadline_at=deadline,
        )
    except Exception:
        return None
    if health is None:
        return None
    return health.get("workers") or {}


def _any_worker(workers: dict | None, *states: str) -> bool:
    return any(int((workers or {}).get(state) or 0) > 0 for state in states)


def _has_worker(workers: dict | None) -> bool | None:
    """True once a datacenter has actually given us a box, in any state it can then be in.

    - ``initializing`` / ``ready`` / ``running`` / ``idle`` -- allocated and fine
    - ``unhealthy`` -- allocated, then the image failed to start. ``jobs.py`` reads this as a failed
      image pull and retries on a fresh endpoint, so counting it as "no capacity" would blame the
      datacenter for a broken image and tell the operator to change GPU class, which cannot help.
      ``_only_unhealthy_workers`` is what separates that case out on its own timer.

    ``throttled`` is deliberately NOT counted. ``jobs.py`` classifies a sustained throttled worker as
    ``no_capacity`` ("retrying on the next-best GPU"), which is exactly the condition this poller
    exists to catch: RunPod is not scheduling the pinned class here. Treating it as capacity would
    make a preload sit the full timeout on a datacenter that will never run it.

    Returns None when health could not be read. That is deliberately NOT False: the caller escalates
    a sustained False into NoCapacityError and tears the endpoint down, so a health endpoint that is
    merely unreachable would look identical to a starved datacenter and could kill a download that is
    running fine. Unknown must stay unknown -- only a positive "no workers" answer is evidence.
    """
    if workers is None:
        return None
    return _any_worker(workers, "initializing", "ready", "running", "idle", "unhealthy")


def _only_unhealthy_workers(workers: dict | None) -> bool:
    """True when every box this datacenter gave us failed to start.

    Same predicate ``poll_job`` runs while IN_QUEUE: unhealthy with nothing usable and nothing still
    coming up. A box that is initializing may yet come good, and one that is ready/running/idle
    already has, so neither is a broken image.

    ``throttled`` blocks this too. A throttled box is capacity contention, not a failed image: it may
    still become runnable, and ``poll_job`` gives it its own longer grace. Calling a mixed
    unhealthy+throttled endpoint a broken image would tear it down at the shorter grace and blame a
    failed image pull for what is actually a busy datacenter.
    """
    if not workers:
        return False
    return _any_worker(workers, "unhealthy") and not _any_worker(
        workers, "initializing", "ready", "running", "idle", "throttled"
    )


def catalog_model_ids() -> list[str]:
    """Catalog base models that fit the weight-cache volume, LARGEST FIRST.

    Largest-first only buys fail-fast: the biggest model is the one whose cold download costs the
    most and is the likeliest to run out of room, so trying it before spending 20 minutes on the
    small ones surfaces the failure early.

    It is NOT what makes the catalog fit, and it must not be mistaken for a capacity fix. The volume
    is persistent and preload never evicts, so on any DC warmed even once the largest model meets a
    volume already holding everything else no matter what order this returns. Capacity comes from
    sizing the volume for the whole resident catalog plus the largest model's in-transit scratch --
    see ``flash.runner.weight_cache_catalog_peak_gb``.
    """
    from flash.catalog import MODELS
    from flash.runner import _fits_weight_cache

    fitting = [(mid, info) for mid, info in MODELS.items() if _fits_weight_cache(info)]
    fitting.sort(key=lambda pair: (-(pair[1].params_b or 0.0), pair[0]))
    return [mid for mid, _ in fitting]


def _grow_existing_cache_volumes(wanted: dict[str, int]) -> None:
    """Raise any already-provisioned cache volume in ``wanted`` to the managed size, fleet-wide.

    A NetworkVolume is only sized on CREATE: the SDK matches an existing volume by name+datacenter
    and hands it back untouched, so every volume provisioned before a size bump stays at the old
    size and the bump is a silent no-op. The warm then attaches an under-sized mount and the
    download dies with "Disk quota exceeded" -- the exact failure the bump was meant to fix.

    Unlike the deploy path, this walks the WHOLE pool: the warm has not picked an account yet, so
    the volume may belong to any of them. Each account therefore gets its own short budget, and the
    pool as a whole is capped -- an unreachable account would otherwise spend three request timeouts
    plus backoff against the deadline the launch needs and make deploy_train_endpoint reject a
    launch the healthy owning account could have served.

    The cap is carved from the front rather than by reserving a slice of the deadline: a short
    --timeout-s floors at 60s all by itself, so subtracting the create allowance from it would
    silently skip reconciliation entirely and reintroduce the under-sized mount.

    Best-effort throughout: a volume that cannot be grown still gets attached, which is no worse
    than not trying, so nothing here fails a warm.
    """
    from flash.providers.runpod import keys as rp_keys
    from flash.providers.runpod.jobs import WEIGHT_CACHE_GROW_BUDGET_S

    reconcile_until = time.time() + WEIGHT_CACHE_GROW_BUDGET_S
    for i, key in enumerate(rp_keys.keys()):
        if time.time() >= reconcile_until:
            logger.warning("weight cache: out of reconciliation budget; attaching volume(s) as-is")
            return
        try:
            grown = runpod_api.grow_network_volumes_for_key(key, wanted, deadline_at=reconcile_until)
        except Exception as exc:
            logger.warning(
                "weight cache: could not grow volume(s) on RunPod account %d (%s); attaching as-is",
                i, exc,
            )
            continue
        for name, size in sorted(grown.items()):
            logger.info("weight cache: grew %s to %d GB (was under the managed size)", name, size)


def _preload_one_dc(
    dc_id: str,
    models: list[str],
    token: str | None,
    gpu: str,
    timeout_s: int,
    poll_interval_s: float,
) -> dict:
    """Warm one datacenter's volume: deploy (pinned to that DC) -> preload job -> teardown."""
    from runpod_flash import NetworkVolume
    from runpod_flash.core.resources.datacenter import DataCenter

    from flash.runner import WEIGHT_CACHE_VOLUME_GB, WEIGHT_CACHE_VOLUME_NAME

    dc = DataCenter.from_string(dc_id)
    vol_name = weight_cache_volume_name(WEIGHT_CACHE_VOLUME_NAME, dc)
    # Pass a factory, not a prebuilt dict: SDK stamps an account-scoped id onto NetworkVolume, so each
    # failover attempt must build a fresh volume.
    def _endpoint_kwargs():
        return {
            "volume": [NetworkVolume(name=vol_name, size=WEIGHT_CACHE_VOLUME_GB, datacenter=dc)],
            "datacenter": [dc],
        }

    endpoint_id = None
    key_fingerprint = None
    deadline_at = time.time() + timeout_s
    try:
        _grow_existing_cache_volumes({vol_name: WEIGHT_CACHE_VOLUME_GB})
        endpoint_id, _name, key_fingerprint = deploy_train_endpoint(
            gpu,
            execution_timeout_ms=timeout_s * 1000,
            name_suffix=f"preload-{dc_id.lower()}-{uuid.uuid4().hex[:6]}",
            spec=None,
            endpoint_kwargs=_endpoint_kwargs,
            deadline_at=deadline_at,
        )
        payload = {
            "mode": "preload",
            "models": models,
            "env": {"HF_HOME": _HF_HOME, **({"HF_TOKEN": token} if token else {})},
        }
        job_id = runpod_api.submit_job(
            endpoint_id,
            build_function_input(payload),
            key_fingerprint=key_fingerprint,
            deadline_at=deadline_at,
        )
        logger.info("preload %s: job %s submitted (%d models)", dc_id, job_id, len(models))
        result = _poll_until_done(
            endpoint_id,
            job_id,
            key_fingerprint,
            timeout_s,
            poll_interval_s,
        )
        if result.get("error"):
            return {"datacenter": dc_id, "status": "error", "error": result["error"], "result": result}
        if result.get("failed"):
            return {"datacenter": dc_id, "status": "partial", "result": result}
        return {"datacenter": dc_id, "status": "ok", "result": result}
    except NoCapacityError as exc:
        # Distinct from "error": nothing is broken, this DC just has no GPU of this class.
        logger.warning("preload %s NO CAPACITY for %s: %s", dc_id, gpu, exc)
        return {"datacenter": dc_id, "status": "no_capacity", "gpu": gpu, "error": str(exc)}
    except Exception as exc:  # one region failing must not abort the others
        logger.warning("preload %s FAILED: %s", dc_id, exc)
        return {"datacenter": dc_id, "status": "error", "error": str(exc)}
    finally:
        if endpoint_id and key_fingerprint:
            with contextlib.suppress(Exception):
                runpod_api.delete_endpoint_for_fingerprint(endpoint_id, key_fingerprint)


def _poll_until_done(
    endpoint_id: str,
    job_id: str,
    key_fingerprint: str,
    timeout_s: int,
    poll_interval_s: float,
) -> dict:
    deadline = time.time() + timeout_s
    # Same GraceTimer poll_job runs, for the same reason: each grace is measured over an UNBROKEN
    # run of confirmed readings, so it arms on the first such reading rather than at launch. Timing
    # from launch would let an unreadable health API age a timer silently, and the first definite
    # reading after that would fire instantly -- deleting an endpoint whose download may be
    # progressing.
    starved = GraceTimer()
    unhealthy = GraceTimer()
    while time.time() < deadline:
        st = runpod_api.job_status(
            endpoint_id,
            job_id,
            key_fingerprint=key_fingerprint,
            deadline_at=deadline,
        )
        status = (st or {}).get("status")
        # Only a status that PROVES the job left the queue breaks the run. `!= _QUEUED` would also
        # match None and any unrecognized string, so one flaky or empty job_status response would
        # reset the grace window and keep NoCapacityError from ever firing -- the DC would stay
        # silently cold for the full timeout, which is the failure this poller exists to catch.
        # An unknown status is unknown: it proves nothing either way, so it leaves both anchors as
        # they are and the next definite reading decides.
        left_queue = status is not None and (
            status in _TERMINAL_OK or status in _TERMINAL_FAIL or status in _RUNNING
        )
        if left_queue:
            # A job that reached IN_PROGRESS was allocated a worker, so if it is later re-queued
            # after an interruption it must serve a FRESH grace window: carrying the old anchor
            # forward would charge the whole running interval to starvation and the first
            # zero-worker reading after the re-queue would delete an endpoint that never actually
            # waited on capacity. Same reasoning as poll_job clearing its in-queue timers.
            starved.since = unhealthy.since = None
        # Restricted to IN_QUEUE: any other nonterminal status proves a worker was allocated, so
        # zero-worker health then is a reporting artifact and never evidence of a starved DC.
        #
        # Health is re-read on EVERY queued poll rather than latched off after the first worker
        # sighting. A box that is reported and then reclaimed while the job is still queued would
        # otherwise suppress all later probes, and because the job never leaves the queue nothing
        # would ever clear the latch -- the preload would burn the full timeout on a datacenter that
        # had lost the worker. The grace timers below, not the probe, are what keep a transient
        # blip from tearing down a healthy endpoint.
        elif status == _QUEUED:
            now = time.time()
            workers = _worker_counts(endpoint_id, key_fingerprint, deadline)
            # Only a definite "no workers" arms or holds the timer. None means health was
            # unreadable, and treating that as starvation would delete the endpoint out from under
            # a live download -- a broken health API must not look like a dead DC.
            if starved.expired(_has_worker(workers) is False, now, _NO_CAPACITY_GRACE_S):
                raise NoCapacityError(
                    f"job {job_id} sat queued {_NO_CAPACITY_GRACE_S:.0f}s with no worker in any "
                    "state: this datacenter cannot serve the requested GPU class"
                )
            # A box that was allocated and then died counts as capacity above, so without its own
            # timer it clears the starvation one every poll and the preload holds a paid endpoint
            # for the whole timeout before reporting a bare TimeoutError. The image is broken; say so.
            if unhealthy.expired(_only_unhealthy_workers(workers), now, _UNHEALTHY_GRACE_S):
                raise RuntimeError(
                    f"preload job {job_id} sat queued {_UNHEALTHY_GRACE_S:.0f}s with every worker "
                    "unhealthy: the worker image failed to start (likely a failed image pull), "
                    "which no datacenter or GPU class can fix"
                )
        if status in _TERMINAL_OK:
            output = (st or {}).get("output")
            if not output:
                # COMPLETED with no output = broken worker image or API mismatch, not a warmed region.
                return {"error": f"preload job {job_id} completed with no output"}
            try:
                return decode_output(output) or {}
            except Exception as exc:
                return {"error": str(exc)}
        if status in _TERMINAL_FAIL:
            raise RuntimeError(f"preload job {job_id} ended {status}: {(st or {}).get('error')}")
        time.sleep(poll_interval_s)
    raise TimeoutError(f"preload job {job_id} did not finish within {timeout_s}s")


def warm_weight_cache(
    models: list[str] | None = None,
    datacenters: list[str] | None = None,
    gpu: str = _PRELOAD_GPU,
    timeout_s: int = _PRELOAD_TIMEOUT_S,
    max_workers: int = 4,
    poll_interval_s: float = 10.0,
    token: str | None = None,
) -> list[dict]:
    """Warm every datacenter volume with the given models. Returns one result dict per DC.

    ``max_workers`` MUST stay under the RunPod endpoint quota (default 5); the default of 4 leaves a
    buffer.
    """
    from runpod_flash.core.resources.datacenter import DataCenter

    models = models or catalog_model_ids()
    dc_ids = datacenters or [dc.value for dc in weight_cache_datacenters()]
    # Validate all DC ids up front so a bad id fails before any paid endpoint launches.
    for d in dc_ids:
        DataCenter.from_string(d)
    token = token or os.environ.get("HF_TOKEN")
    logger.info("warming %d datacenter(s) with %d model(s)", len(dc_ids), len(models))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {
            pool.submit(_preload_one_dc, dc, models, token, gpu, timeout_s, poll_interval_s): dc
            for dc in dc_ids
        }
        results: list[dict] = [fut.result() for fut in as_completed(futs)]
    ok = sum(1 for r in results if r.get("status") == "ok")
    starved = [r["datacenter"] for r in results if r.get("status") == "no_capacity"]
    logger.info("preload complete: %d/%d datacenters warmed", ok, len(results))
    # A "partial" DC downloaded some models and failed others. The worker already reports which ones
    # and why, but that detail died inside the result dict -- without it "partial" is unactionable.
    for r in results:
        if r.get("status") != "partial":
            continue
        failed = ((r.get("result") or {}).get("failed") or {})
        for model_id, detail in sorted(failed.items()):
            logger.warning("preload %s: %s FAILED: %s", r["datacenter"], model_id, detail)
    if starved:
        # Actionable: these stay cold until re-run with a class the DC actually stocks.
        logger.warning(
            "no %s capacity in %s -- re-run those with --datacenters %s --gpu <class>",
            gpu, ", ".join(starved), ",".join(starved),
        )
    return results


def teardown_weight_cache(datacenters: list[str] | None = None) -> list[str]:
    """Delete per-DC weight-cache volumes. Sweeps every account in the RUNPOD_API_KEY pool.

    ``datacenters=None`` → whole fleet; ``[]`` → no-op (never widened to all — that's a footgun).
    """

    from flash.providers.runpod import keys as rp_keys
    from flash.runner import WEIGHT_CACHE_VOLUME_NAME

    # Explicit [] is a no-op — never widen zero DCs to the whole fleet.
    if datacenters is not None and not datacenters:
        logger.info("teardown: empty datacenter scope — nothing to reclaim (refusing to widen to all)")
        return []
    pool = rp_keys.keys()
    if not pool:
        logger.info("teardown: RUNPOD_API_KEY not configured — skipping RunPod cache teardown")
        return []
    # Import SDK after early returns: may be absent on instance-only control planes.
    from runpod_flash.core.api.runpod import RunpodRestClient
    from runpod_flash.core.resources.datacenter import DataCenter
    from runpod_flash.core.urls import RUNPOD_REST_API_URL

    dc_ids = datacenters if datacenters else [dc.value for dc in weight_cache_datacenters()]
    targets = {
        weight_cache_volume_name(WEIGHT_CACHE_VOLUME_NAME, DataCenter.from_string(d)) for d in dc_ids
    }

    async def _names(client) -> set:
        res = await client.list_network_volumes()
        vols = res if isinstance(res, list) else res.get("networkVolumes", [])
        return {v.get("name") for v in vols}

    async def _go_one(api_key) -> list[str]:
        client = RunpodRestClient(api_key=api_key) if api_key else RunpodRestClient()
        res = await client.list_network_volumes()
        vols = res if isinstance(res, list) else res.get("networkVolumes", [])
        to_delete = {v["name"]: v["id"] for v in vols if v.get("name") in targets and v.get("id")}
        for vid in to_delete.values():
            # SDK's _execute_rest chokes on 204 No Content; swallow and confirm by re-listing.
            with contextlib.suppress(Exception):
                await client._execute_rest("DELETE", f"{RUNPOD_REST_API_URL}/networkvolumes/{vid}")
        remaining = await _names(client)
        gone = [name for name in to_delete if name not in remaining]
        still = [name for name in to_delete if name in remaining]
        if still:
            logger.warning("teardown: %d cache volume(s) FAILED to delete (still present): %s",
                           len(still), ", ".join(sorted(still)))
        return gone

    multi = len(pool) > 1
    deleted: list[str] = []
    failed_accounts: list[str] = []
    for i, key in enumerate(pool):
        try:
            names = _run_async(_go_one(key))
        except Exception as exc:
            failed_accounts.append(f"acct{i}")
            logger.warning("teardown: RunPod account %d sweep FAILED (continuing): %s", i, exc)
            continue
        deleted.extend((f"acct{i}:{n}" if multi else n) for n in names)
    if failed_accounts:
        logger.warning(
            "teardown: %d of %d RunPod account(s) failed to sweep (%s) — their cache volumes may "
            "still be billed; re-run teardown once the key(s) are valid",
            len(failed_accounts), len(pool), ", ".join(failed_accounts),
        )
    return deleted


def teardown_lambda_filesystems(name: str | None = None) -> list[str]:
    """Delete Lambda weight-cache filesystems across all regions. Best-effort and idempotent."""
    from flash.providers.lambdalabs import api as lambda_api
    from flash.runner import WEIGHT_CACHE_VOLUME_NAME

    target = name or WEIGHT_CACHE_VOLUME_NAME
    deleted: list[str] = []
    try:
        fses = lambda_api.list_filesystems()
    except Exception as exc:
        logger.warning("teardown: lambda list_filesystems failed (skipping): %s", exc)
        return deleted
    for fs in fses:
        if fs.get("name") == target and fs.get("id") and lambda_api.delete_filesystem(fs["id"]):
            region = (fs.get("region") or {}).get("name") or "?"
            deleted.append(f"lambda:{region}/{target}")
    return deleted


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
_PRELOAD_STATUS_REPO = "Freesolo-Co/flash-weight-preload"


class IncompleteWarmPlanError(RuntimeError):
    """Some regions warmed, but a class went unanswered so the fleet was never fully measured.

    Carries the results of the launches that DID run: they are real, paid, completed work, and a
    bare raise would throw them away along with the record of which regions are now warm. The
    caller reports them and then treats the run as unfinished rather than as a clean sweep.
    """

    def __init__(self, message: str, *, results: list[dict]):
        super().__init__(message)
        self.results = results


def _lambda_warm_targets(lambda_jobs, gpu: str | None) -> tuple[list[list], bool]:
    """``(one cheapest-first candidate list per Lambda region, planning was complete)``.

    An explicit ``gpu`` pins the class and is never second-guessed. Otherwise every ladder class is
    asked which regions have capacity, and each region keeps ALL the classes that can reach it, so a
    region with no A10 is still warmed on A100/H100/B200 instead of being silently skipped. Capacity
    is a live, per-region property, so this is a lookup and not a constant.

    The whole per-region list is kept rather than just its cheapest entry because preload mode
    deliberately never refreshes candidates: handing the launcher a single candidate means one clean
    capacity rejection leaves that region cold even though a pricier class from the same inventory
    snapshot was sitting right there. The alternative to a fallback is not a cheaper box, it is no box.

    Ranked by each candidate's own ``price_usd_hr``, which ``usable_instances`` fills from the live
    Lambda rate (falling back to the static table only when the live lookup fails). Ranking on the
    ladder's fixed order instead would keep claiming regions in a stale June price order and could
    launch the more expensive class after a Lambda discount.

    The second element is False when any class went unanswered -- a lookup that failed or was cut off
    by the planning budget. "No capacity" and "we never got an answer" are different facts, and only
    the first may be reported as one: a region reachable solely through an unanswered class is not
    known to be cold. The caller needs the distinction to avoid printing a definitive fleet summary
    over a Lambda outage.
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
                "class(es) %s", int(_LAMBDA_PLANNING_BUDGET_S), ", ".join(classes[classes.index(cls):]),
            )
            complete = False
            break
        try:
            candidates = lambda_jobs.usable_instances(
                cls, **deadline_kwargs(lambda_jobs.usable_instances, deadline),
            )
        except Exception as exc:
            logger.warning("warm lambda: usable_instances(%s) failed (skipping): %s", cls, exc)
            complete = False
            continue
        for c in candidates:
            by_region.setdefault(c.region, []).append(c)
    # Ties keep ladder order, which is the static cheapest-first sequence: a stable sort means an
    # unavailable live price degrades to the old behaviour instead of to an arbitrary one.
    targets = [
        sorted(cands, key=lambda c: getattr(c, "price_usd_hr", None) or math.inf)
        for _region, cands in sorted(by_region.items())
    ]
    return targets, complete


def _lambda_provisioned_regions() -> set[str]:
    """Regions where the weight-cache filesystem exists, per the Lambda API. Empty if unreadable.

    The warm summary needs this because a region with no capacity in ANY ladder class never becomes a
    target and so never produces a result -- it would be invisible in a report built only from
    results, which is exactly the silent fleet gap this change exists to expose.

    Deadline-bounded because this runs AFTER ``_lambda_warm_targets`` has already spent the shared
    planning budget, and ``list_filesystems`` retries internally: an endpoint that accepts connections
    then hangs would otherwise add several minutes of attempts and backoff to planning, for what is
    only a reporting nicety. Losing the snapshot degrades the summary; blocking on it delays warming.
    """
    from flash.providers.lambdalabs import api as lambda_api
    from flash.runner import WEIGHT_CACHE_VOLUME_NAME

    try:
        fses = lambda_api.list_filesystems(
            **deadline_kwargs(
                lambda_api.list_filesystems, time.time() + _LAMBDA_SNAPSHOT_BUDGET_S
            ),
        )
    except Exception as exc:
        logger.warning("warm lambda: list_filesystems failed, cannot report unreachable regions: %s", exc)
        return set()
    return {
        (fs.get("region") or {}).get("name")
        for fs in fses
        if fs.get("name") == WEIGHT_CACHE_VOLUME_NAME and (fs.get("region") or {}).get("name")
    }


def _ensure_status_repo(token: str | None) -> None:
    """Create the preload status dataset repo if absent. RAISES on failure — call before launching."""
    from huggingface_hub import HfApi

    HfApi(token=token).create_repo(_PRELOAD_STATUS_REPO, repo_type="dataset", exist_ok=True, private=True)


def _preload_instance_spec(gpu: str, run_id: str, wall_s: int = 1800):
    """Minimal download-only spec with cache volume attached and wall cap set to the warm timeout."""
    from flash.runner import WEIGHT_CACHE_VOLUME_GB, WEIGHT_CACHE_VOLUME_NAME
    from flash.spec import JobSpec

    return JobSpec.from_dict({
        "model": "Qwen/Qwen3.5-0.8B", "algorithm": "sft", "run_id": run_id,
        "train": {"hf_repo": _PRELOAD_STATUS_REPO},
        "gpu": {"type": gpu, "max_wall_seconds": max(60, int(wall_s)),
                "network_volume": WEIGHT_CACHE_VOLUME_NAME, "network_volume_gb": WEIGHT_CACHE_VOLUME_GB},
    })


def _region_filesystem_is_listed(region: str, deadline: float) -> bool:
    """True when this region's weight-cache filesystem is VISIBLE in the account listing.

    Visibility, not a successful create, is the property that matters. Every ``ensure_filesystem``
    call begins by listing and returns early on a match, so once the filesystem is listed no later
    caller can reach the non-idempotent create path for it.
    """
    from flash.providers.lambdalabs import api as lambda_api
    from flash.runner import WEIGHT_CACHE_VOLUME_NAME

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

    ``launch_and_submit`` calls ``ensure_filesystem`` itself on every attempt and
    ``create_filesystem`` is not idempotent, so the filesystem has to be settled before the ladder
    runs or a rejection on the cheap class can be followed by a second create for the same name and
    region on the next -- duplicate storage, billed forever.

    Creating here is not by itself enough, which is the subtle part. ``ensure_filesystem`` returning
    only proves the create call succeeded; the launcher then does its own listing, and a filesystem
    that exists but has not yet appeared in ``list_filesystems()`` makes that listing miss and submit
    the very duplicate this is meant to prevent. So the create is followed by an explicit visibility
    check, and only a listed filesystem counts as settled.

    Matching on error text instead cannot work. ``ensure_filesystem`` guards its create but not the
    reconciliation listing inside its own except block (api.py), so when that listing times out the
    raw error propagates, and ``launch_and_submit`` then wraps *every* failure -- real capacity
    rejections included -- in the same "all N region(s) rejected ... (no capacity)" message. The two
    are indistinguishable downstream, so the duplicate has to be prevented upstream.

    Returns one of:
      ``"listed"``      -- confirmed present, so every later ensure reuses it and cannot create.
      ``"unreachable"`` -- Lambda was never reached, so no create can exist and nothing is at risk.
      ``"doubtful"``    -- we reached Lambda and cannot confirm the outcome; launching now could pay
                           for a second filesystem forever, so the caller must skip the region.
    """
    from flash.providers.lambdalabs import api as lambda_api
    from flash.runner import WEIGHT_CACHE_VOLUME_NAME

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
        logger.warning("warm lambda/%s: filesystem created but not yet listed; skipping this cycle "
                       "so the launcher cannot create a duplicate", region)
        return "doubtful"
    except Exception as exc:
        # No credentials means no request was ever sent, so nothing can have been created. Treating
        # that as doubt would skip every region on any host without Lambda creds -- turning a missing
        # key into a silent no-op instead of the launcher's own explicit failure.
        if not _lambda_is_reachable(exc):
            logger.info("warm lambda/%s: skipping filesystem pre-check (%s)", region, exc)
            return "unreachable"
        logger.warning("warm lambda/%s: filesystem could not be confirmed (%s); skipping this cycle "
                       "so the launcher cannot create a duplicate", region, exc)
        return "doubtful"


def _lambda_is_reachable(exc: Exception) -> bool:
    """False when the failure proves no Lambda API call was ever issued (so no create can exist).

    The text comes from ``RestClient.missing_key_message`` (``_http.py``), the single place a missing
    key is reported, and is matched on the substring both halves of that message share.
    """
    return "not configured" not in str(exc).lower()


def _warm_one_lambda_instance(lambda_jobs, candidates: list, models: list,
                              timeout_s: int, poll_interval_s: float) -> dict:
    """Launch a download-only preload instance in one Lambda region, poll its status marker, then
    ALWAYS terminate. One region failing never aborts the others.

    ``candidates`` is that region's cheapest-first class list. Each is tried until one launches, so a
    capacity rejection on the cheap class falls through to a pricier one instead of leaving the region
    cold -- preload mode never refreshes candidates itself. The GPU class is read off the candidate
    that actually launched, so a mixed-class fleet warm reports what each region really cost.
    """
    region = getattr(candidates[0], "region", "?")
    gpu = getattr(candidates[0], "gpu", None) or _LAMBDA_PRELOAD_GPU
    effective_s = max(60, int(timeout_s))
    run_id = None

    def _result(status: str, **extra) -> dict:
        return {"provider": "lambda", "region": region, "gpu": gpu, "status": status, **extra}

    try:
        # Settle the filesystem before any class runs, so every per-attempt ensure_filesystem inside
        # the launcher only ever reuses and can never reach the non-idempotent create path.
        #
        # On its OWN budget, not the launch/poll one. Charging it to the run deadline while the
        # instance wall cap and the reap deadline still got the full effective_s made the driver
        # give up before the box it is watching, so a warm that was still downloading was reported
        # as timed out. It also silently ate into the provider's 60s create allowance: a pre-check
        # that left less than that made every class in the ladder fail the allowance check inside
        # launch_and_submit, and the region reported "no capacity" for classes never actually tried.
        fs_state = _ensure_region_filesystem(region, time.time() + _FS_PRECHECK_BUDGET_S)
        if fs_state == "doubtful":
            # Launching now would let the launcher's own listing miss and create a duplicate that is
            # billed forever. A region left cold this cycle just downloads on first use.
            return _result("error", error="filesystem unconfirmed; skipped to avoid a duplicate create")
        # One anchor for everything downstream: the driver's poll deadline, the reap deadline
        # embedded in the run_id, and the instance's own wall cap all start here and all run for
        # effective_s, so no clock is ahead of another.
        deadline = time.time() + effective_s
        # Embed reap deadline in the run_id so orphan sweep can free the box if this driver dies.
        run_id = preload_instance_run_id(
            "lambda", region, int(deadline), uuid.uuid4().hex[:6]
        )
        spec = launch_err = None
        for cand in candidates:
            # Rebuild per class: the spec carries the GPU, so reusing the cheap one's spec would
            # launch the very mismatch this fallback exists to avoid.
            gpu = getattr(cand, "gpu", None) or gpu
            spec = _preload_instance_spec(gpu, run_id, wall_s=effective_s)
            try:
                lambda_jobs.launch_and_submit(
                    spec,
                    seed=spec.seed,
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
                logger.warning("warm lambda/%s: ambiguous create, not trying another class: %s",
                               region, exc)
                break
            except Exception as exc:
                # no capacity / launch reject. Walking to the next class is safe here: the doubtful
                # case already returned above, so the filesystem is either listed (every per-class
                # ensure_filesystem reuses it and cannot create) or Lambda was never reachable at all
                # (no create can exist). Deciding this from the error text is impossible --
                # ensure_filesystem leaves its reconciliation listing unguarded, and launch_and_submit
                # wraps a filesystem failure and a genuine capacity rejection in the same "no
                # capacity" message -- which is why it is settled before the ladder instead.
                launch_err = exc
                logger.info("warm lambda/%s: %s rejected (%s); trying next class", region, gpu, exc)
        if launch_err is not None or spec is None:
            return _result("error", error=f"launch: {launch_err}")
        prefix = f"{spec.phase}/{run_id}"
        reader = make_hf_text_reader(_PRELOAD_STATUS_REPO, f"{prefix}/preload_result.json",
                                     min_interval_s=max(5.0, poll_interval_s))
        # Also watch the attempt marker: if the box dies early the failmark is the only signal (avoids
        # polling to full timeout on a dead box). Completion file is authoritative when present.
        fail_reader = make_hf_text_reader(_PRELOAD_STATUS_REPO, f"{prefix}/lambda_attempt0.json",
                                          min_interval_s=max(5.0, poll_interval_s))
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
                    return _result("error", error=f"box failed early: {fail.get('error') or 'see boot log'}")
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


def warm_instances(models: list | None = None, gpu: str | None = None,
                   timeout_s: int = _PRELOAD_TIMEOUT_S, poll_interval_s: float = 20.0,
                   max_workers: int = 4) -> list[dict]:
    """Warm Lambda caches: one download-only launch per region with capacity. Returns status per region."""
    models = models or catalog_model_ids()
    token = os.environ.get("HF_TOKEN")

    from flash.providers.lambdalabs import jobs as lambda_jobs

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
        "warm lambda: %d region(s) -> %s", len(targets),
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
            f"preload status repo {_PRELOAD_STATUS_REPO!r} unavailable ({exc}); set a valid HF_TOKEN "
            "with write access before warming (refusing to launch paid GPUs that can't report)."
        ) from exc
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        # Each region gets its whole cheapest-first list, not one pre-picked class, so the launcher
        # can fall through to a pricier class rather than leaving the region cold.
        futs = [
            ex.submit(_warm_one_lambda_instance, lambda_jobs, cands, models,
                      timeout_s, poll_interval_s)
            for cands in targets
        ]
        results = [f.result() for f in as_completed(futs)]
    _log_unreachable_lambda_regions(provisioned, results, planned=planned)
    if not planned:
        # The reachable launches above are real work and are kept -- but the fleet was never fully
        # measured, so this run cannot be reported as a finished one. Without this the mixed case
        # (one class unanswered, another still yielding targets) printed "N/N regions warmed" and
        # exited 0, where N counted only the regions we managed to look at. A region reachable
        # solely through the unanswered class is missing from both the numerator AND the
        # denominator, so the ratio looks perfect precisely because the gap is invisible.
        # "examined", not "warmed": results holds every launched region whatever its status, so
        # counting them as warmed would contradict the ok-only "X/Y regions warmed" line the CLI
        # prints right above this one.
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
    provisioned: set[str], results: list[dict], *, planned: bool = True,
) -> list[str]:
    """Warn about every region whose cache is not fully warm. Returns them sorted, for printing.

    Returned as well as logged because this is a library module: the ``flash`` logger carries only a
    NullHandler until an app calls ``configure_logging``, so a caller that has not opted in would
    otherwise lose the one message naming regions with no capacity in any class.

    ``planned`` is False when some class went unanswered. A region that never became a target is
    then not known to be cold, only unexamined, so it is labelled as such: claiming "no capacity"
    off a lookup that never returned would report an outage as a finished measurement.
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
    from flash.providers.lambdalabs import api as lambda_api
    from flash.runner import WEIGHT_CACHE_VOLUME_NAME

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
            logger.warning("provision: lambda ensure_filesystem(%s, %s) failed: %s", target, region, exc)
    return done


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
        "--gpu", default=None,
        help="GPU class for the preload worker. Defaults are per-mode: RunPod warm -> "
             f"{_PRELOAD_GPU!r}; --warm-instances -> the cheapest class each region actually stocks, "
             f"tried in the order {' -> '.join(_LAMBDA_PRELOAD_GPU_LADDER)}, so a region with no "
             "cheap capacity is warmed on a pricier class instead of being skipped. Pass this to "
             "pin ONE class everywhere and disable that fallback (a region that does not stock it "
             "is then left cold). Defaulting to None (not a sentinel string) lets you explicitly "
             "pick even a default GPU without it being mistaken for 'no override'.",
    )
    ap.add_argument("--timeout-s", type=int, default=_PRELOAD_TIMEOUT_S,
                    help="per-DC job timeout (default sized for a fully cold whole-catalog warm)")
    ap.add_argument(
        "--max-workers", type=int, default=4,
        help="datacenters warmed concurrently. Each one deploys a preload endpoint, so this MUST stay "
             "under your RunPod endpoint/worker quota (the documented default is 5); the default of 4 "
             "leaves a 1-slot buffer. Raise it only if your account quota is higher.",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the plan, provision nothing")
    ap.add_argument(
        "--provision", action="store_true",
        help="CREATE the Lambda weight-cache filesystem in every region (pure API, no GPU) and "
             "exit; RunPod volumes are auto-created by the eager deploy/warm. Run before --teardown's "
             "inverse to set up all storage up front.",
    )
    ap.add_argument(
        "--warm-instances", action="store_true",
        help="WARM the Lambda caches: one download-only GPU launch per region with "
             "capacity now (needs the merged worker image carrying the bootstrap preload branch).",
    )
    ap.add_argument(
        "--teardown", action="store_true",
        help="DELETE the weight-cache storage on every provider (reclaim standing storage) and exit. "
             "With --datacenters it is SCOPED to that RunPod-DC subset only (Lambda caches "
             "are left intact, since DC ids don't map to their region namespace).",
    )
    args = ap.parse_args(argv)

    selected_modes = [
        name for name, on in (
            ("--provision", args.provision),
            ("--warm-instances", args.warm_instances),
            ("--teardown", args.teardown),
        ) if on
    ]
    if len(selected_modes) > 1:
        ap.error(f"{', '.join(selected_modes)} are mutually exclusive — pass exactly one mode")

    catalog = catalog_model_ids()
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else catalog
    # Reject off-catalog ids before any download: private/gated weights must not land on the shared cache.
    if args.models and not args.teardown and not args.provision:
        off_catalog = [m for m in models if m not in set(catalog)]
        if off_catalog:
            print("--models: refusing to preload off-catalog model id(s) into the shared cache: "
                  f"{', '.join(off_catalog)} — only public catalog models may be warmed (private/gated "
                  "repos would leak onto the platform-wide shared volume). They download cold on first "
                  "use instead.")
            return 2
    # `--datacenters ""` must error, not silently widen to a full fleet teardown.
    dcs_given = args.datacenters is not None
    parsed_dcs = (
        [d.strip() for d in args.datacenters.split(",") if d.strip()] if dcs_given else []
    )
    if dcs_given and not parsed_dcs:
        print("--datacenters was given but parsed to no datacenter ids — refusing to run "
              "(an empty scope would delete the WHOLE RunPod fleet); drop --datacenters for a full "
              "teardown, or pass real DC ids.")
        return 2
    scoped = bool(parsed_dcs)

    # Lazy: weight_cache_datacenters() imports runpod_flash; avoid importing it on instance-only hosts.
    def _default_dcs() -> list[str]:
        return [dc.value for dc in weight_cache_datacenters()]

    if args.provision:
        if args.dry_run:
            print("would provision Lambda filesystems in every region")
            return 0
        provisioned = provision_lambda_filesystems()
        print(f"provisioned {len(provisioned)} Lambda filesystem(s): "
              f"{', '.join(provisioned) or '(none: no Lambda key or no regions)'}")
        return 0
    if args.warm_instances:
        if args.dry_run:
            print("would warm Lambda caches (one download-only launch per region with capacity)")
            return 0
        incomplete = ""
        try:
            results = warm_instances(models=models, gpu=args.gpu,
                                     timeout_s=args.timeout_s, max_workers=args.max_workers)
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
            print("0 regions warmed — no Lambda region had capacity to warm right now "
                  "(weights download cold on first run). Nothing launched.")
            return 0
        failed = [r for r in results if r.get("status") not in ("ok",)]
        for r in results:
            # Name the GPU: without --gpu the ladder picks per region, so this is the only place the
            # operator can see which paid class each region actually billed.
            gpu_note = f" on {r['gpu']}" if r.get("gpu") else ""
            print(f"  {r['provider']}/{r['region']}: {r['status']}{gpu_note}"
                  + (f" ({r.get('error')})" if r.get("error") else ""))
        print(f"{len(results) - len(failed)}/{len(results)} regions warmed")
        if incomplete:
            # NOT "N/N regions warmed ... exit 0": the denominator counts only the regions we could
            # see, so a perfect ratio here means the gap is invisible, not absent.
            print(f"WARNING: {incomplete}")
            return 1
        return 1 if failed else 0
    if args.teardown:
        if scoped:
            from runpod_flash.core.resources.datacenter import DataCenter
            bad = []
            for d in parsed_dcs:
                try:
                    DataCenter.from_string(d)
                except Exception:
                    bad.append(d)
            if bad:
                print(f"--teardown --datacenters: invalid datacenter id(s): {', '.join(bad)} "
                      "— refusing to run (nothing deleted)")
                return 2
        if args.dry_run:
            scope_desc = (f"{len(parsed_dcs)} datacenter(s): {', '.join(parsed_dcs)}"
                          if scoped else "every RunPod storage datacenter")
            print(f"would delete the RunPod weight-cache volumes in {scope_desc}"
                  + ("" if scoped else " + every Lambda filesystem named flash-weights"))
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
    dcs = parsed_dcs or _default_dcs()
    if args.dry_run:
        print(f"would warm {len(dcs)} datacenter(s): {', '.join(dcs)}")
        print(f"with {len(models)} model(s): {', '.join(models)}")
        return 0

    results = warm_weight_cache(
        models=models, datacenters=dcs, gpu=args.gpu or _PRELOAD_GPU,
        timeout_s=args.timeout_s, max_workers=args.max_workers,
    )
    failed = [r for r in results if r.get("status") != "ok"]
    for r in results:
        print(f"  {r['datacenter']}: {r['status']}" + (f" ({r.get('error')})" if r.get("error") else ""))
    print(f"{len(results) - len(failed)}/{len(results)} datacenters warmed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
