"""Warm and tear down the RunPod weight cache, one datacenter at a time.

A cold datacenter makes every run there re-download the base weights. This preloads each
volume-capable DC by deploying a throwaway endpoint that pulls the catalog models onto the shared
volume, polls it to completion through RunPod's queue/throttle states, and tears the endpoint back
down. `NoCapacityError` is the expected outcome when a DC has no free workers, not a failure.

Split out of `flash.providers.artifacts.weight_cache` to keep that module under the file-size limit.
"""

from __future__ import annotations

import contextlib
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from flash.providers.runpod.client import api as runpod_api
from flash.providers.runpod.execution.jobs import GraceTimer
from flash.providers.runpod.execution.resources import weight_cache_volume_name


def _preload():
    """The parent module, imported lazily because it imports this one.

    The preload tests address this whole subsystem through `weight_cache`: they patch the grace
    budgets, `deploy_train_endpoint`, `decode_output`, and even `weight_cache.time` as a module.
    Each is read back through the parent so those patches keep reaching these callers; binding them
    here would freeze the originals at import time.
    """
    from flash.providers.artifacts import weight_cache

    return weight_cache


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
# Boxes held throttled with nothing usable are RunPod declining to schedule the pinned class here.
# Matches poll_job's throttled_grace_s, which calls the same condition no_capacity and walks to the
# next-best GPU. Longer than the unhealthy grace because throttling can still come good.
_THROTTLED_GRACE_S = 300.0


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

    - ``initializing`` / ``ready`` / ``running`` / ``idle`` -- allocated and fine - ``unhealthy`` --
    allocated, then the image failed to start. ``jobs.py`` classifies a sustained throttled worker
    as ``no_capacity`` (a capacity failure on the pinned class), which is exactly the condition this
    exists to catch: RunPod is not scheduling the pinned class here.
    """
    if workers is None:
        return None
    return _any_worker(workers, "initializing", "ready", "running", "idle", "unhealthy")


def _only_unhealthy_workers(workers: dict | None) -> bool:
    """True when every box this datacenter gave us failed to start.

    A throttled box is capacity contention, not a failed image: it may still become runnable, and
    ``poll_job`` gives it its own longer grace. Calling a mixed unhealthy+throttled endpoint a
    broken image would tear it down at the shorter grace and blame a failed image pull for what is
    actually a busy datacenter.
    """
    if not workers:
        return False
    return _any_worker(workers, "unhealthy") and not _any_worker(
        workers, "initializing", "ready", "running", "idle", "throttled"
    )


def _throttled_workers(workers: dict | None) -> bool:
    """True while RunPod is holding boxes throttled with nothing usable -- the same call ``poll_job``

    A mixed unhealthy+throttled endpoint sits in the one gap the other two timers leave:
    ``_has_worker`` counts the unhealthy box as allocated capacity so the starvation timer resets,
    and ``_only_unhealthy_workers`` is blocked by the throttled box so the broken-image timer
    resets. Nothing would ever fire and the preload would hold a paid endpoint for the whole timeout
    before reporting a bare TimeoutError.
    """
    if not workers:
        return False
    return _any_worker(workers, "throttled") and not _any_worker(
        workers, "initializing", "ready", "running", "idle"
    )


def catalog_model_ids() -> list[str]:
    """Catalog base models that fit the weight-cache volume, LARGEST FIRST.

    Largest-first only buys fail-fast: the biggest model is the one whose cold download costs the
    most and is the likeliest to run out of room, so trying it before spending 20 minutes on the
    small ones surfaces the failure early. It is NOT what makes the catalog fit, and it must not be
    mistaken for a capacity fix.
    """
    from flash.core.catalog import MODELS
    from flash.runner.accounting.weight_cache import _fits_weight_cache

    fitting = [(mid, info) for mid, info in MODELS.items() if _fits_weight_cache(info)]
    fitting.sort(key=lambda pair: (-(pair[1].params_b or 0.0), pair[0]))
    return [mid for mid, _ in fitting]


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

    from flash.runner.accounting.weight_cache import (
        WEIGHT_CACHE_VOLUME_GB,
        WEIGHT_CACHE_VOLUME_NAME,
    )

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
    try:
        # timeout_s is the budget for this DC's job; the best-effort reconciliation in
        # front of it gets its own bounded budget on top, never a slice carved out of the
        # job's. Without the headroom a short --timeout-s floors at the 60s create
        # allowance all by itself, the grow yields to that allowance, and reconciliation
        # is skipped entirely -- reintroducing the under-sized mount this whole path
        # exists to prevent.
        from flash.providers.runpod.execution.resources import weight_cache_grow_headroom_s

        deadline_at = _preload().time.time() + timeout_s + weight_cache_grow_headroom_s()
        # The warm attaches its own volume (spec=None), so the deploy cannot derive what to
        # reconcile -- name it here. Reconciling per deploy attempt rather than sweeping the pool
        # up front is what keeps a quota/balance failover correct: the attempt grows the volume
        # owned by the account it is about to attach, so the account failover lands on is never the
        # one left holding a stale, under-sized mount.
        endpoint_id, _name, key_fingerprint = _preload().deploy_train_endpoint(
            gpu,
            execution_timeout_ms=timeout_s * 1000,
            name_suffix=f"preload-{dc_id.lower()}-{uuid.uuid4().hex[:6]}",
            spec=None,
            endpoint_kwargs=_endpoint_kwargs,
            deadline_at=deadline_at,
            cache_volumes={vol_name: WEIGHT_CACHE_VOLUME_GB},
        )
        payload = {
            "mode": "preload",
            "models": models,
            "env": {"HF_HOME": _HF_HOME, **({"HF_TOKEN": token} if token else {})},
        }
        job_id = runpod_api.submit_job(
            endpoint_id,
            payload,
            key_fingerprint=key_fingerprint,
            deadline_at=deadline_at,
        )
        _preload().logger.info(
            "preload %s: job %s submitted (%d models)", dc_id, job_id, len(models)
        )
        result = _preload()._poll_until_done(
            endpoint_id,
            job_id,
            key_fingerprint,
            timeout_s,
            poll_interval_s,
        )
        if result.get("error"):
            return {
                "datacenter": dc_id,
                "status": "error",
                "error": result["error"],
                "result": result,
            }
        if result.get("failed"):
            return {"datacenter": dc_id, "status": "partial", "result": result}
        return {"datacenter": dc_id, "status": "ok", "result": result}
    except NoCapacityError as exc:
        # Distinct from "error": nothing is broken, this DC just has no GPU of this class.
        _preload().logger.warning("preload %s NO CAPACITY for %s: %s", dc_id, gpu, exc)
        return {"datacenter": dc_id, "status": "no_capacity", "gpu": gpu, "error": str(exc)}
    except Exception as exc:  # one region failing must not abort the others
        _preload().logger.warning("preload %s FAILED: %s", dc_id, exc)
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
    deadline = _preload().time.time() + timeout_s
    # Same GraceTimer poll_job runs, for the same reason: each grace is measured over an UNBROKEN
    # run of confirmed readings, so it arms on the first such reading rather than at launch. Timing
    # from launch would let an unreadable health API age a timer silently, and the first definite
    # reading after that would fire instantly -- deleting an endpoint whose download may be
    # progressing.
    starved = GraceTimer()
    unhealthy = GraceTimer()
    throttled = GraceTimer()
    while _preload().time.time() < deadline:
        st = runpod_api.job_status(
            endpoint_id,
            job_id,
            key_fingerprint=key_fingerprint,
            deadline_at=deadline,
        )
        status = (st or {}).get("status")
        # Only a status that PROVES the job left the queue breaks the run. `!= _QUEUED`
        # would also match None and any unrecognized string, so one flaky or empty
        # job_status response would reset the grace window and keep NoCapacityError from
        # ever firing -- the DC would stay silently cold for the full timeout, which is
        # the failure this poller exists to catch.
        left_queue = status is not None and (
            status in _TERMINAL_OK or status in _TERMINAL_FAIL or status in _RUNNING
        )
        if left_queue:
            # A job that reached IN_PROGRESS was allocated a worker, so if it is later re-queued
            # after an interruption it must serve a FRESH grace window: carrying the old anchor
            # forward would charge the whole running interval to starvation and the first
            # zero-worker reading after the re-queue would delete an endpoint that never actually
            # waited on capacity. Same reasoning as poll_job clearing its in-queue timers.
            starved.since = unhealthy.since = throttled.since = None
        # Health is re-read on EVERY queued poll rather than latched off after the first
        # worker sighting. A box that is reported and then reclaimed while the job is
        # still queued would otherwise suppress all later probes, and because the job
        # never leaves the queue nothing would ever clear the latch -- the preload would
        # burn the full timeout on a datacenter that had lost the worker.
        elif status == _QUEUED:
            now = _preload().time.time()
            workers = _preload()._worker_counts(endpoint_id, key_fingerprint, deadline)
            if workers is None:
                # Every predicate below reads None as inactive, so running them
                # would CLEAR all three anchors -- one failed health call per
                # grace window would restart every window and a genuinely starved,
                # broken or throttled DC would hold a paid endpoint for the whole
                # timeout before reporting a bare TimeoutError. The job being
                # confirmed queued says nothing about which of those it is
                # suffering from; only health does, and health is what went dark.
                starved.unknown(now)
                unhealthy.unknown(now)
                throttled.unknown(now)
            else:
                # Only a definite "no workers" arms or holds the timer -- see the None branch above
                # for why an unreadable health API must never look like a starved DC.
                if starved.expired(
                    _has_worker(workers) is False, now, _preload()._NO_CAPACITY_GRACE_S
                ):
                    raise NoCapacityError(
                        f"job {job_id} sat queued {_preload()._NO_CAPACITY_GRACE_S:.0f}s with no worker in any "
                        "state: this datacenter cannot serve the requested GPU class"
                    )
                # A box that was allocated and then died counts as capacity above, so without its
                # own timer it clears the starvation one every poll and the preload holds a paid
                # endpoint for the whole timeout before reporting a bare TimeoutError. The image is
                # broken; say so.
                if unhealthy.expired(
                    _only_unhealthy_workers(workers), now, _preload()._UNHEALTHY_GRACE_S
                ):
                    raise RuntimeError(
                        f"preload job {job_id} sat queued {_preload()._UNHEALTHY_GRACE_S:.0f}s with every "
                        "worker unhealthy: the worker image failed to start (likely a failed image "
                        "pull), which no datacenter or GPU class can fix"
                    )
                # Throttled boxes count as allocated capacity above and block the unhealthy timer,
                # so without this a mixed unhealthy+throttled endpoint clears both every poll and
                # burns the whole timeout. Same call poll_job makes: RunPod is not scheduling here.
                if throttled.expired(
                    _throttled_workers(workers), now, _preload()._THROTTLED_GRACE_S
                ):
                    raise NoCapacityError(
                        f"job {job_id} sat queued {_preload()._THROTTLED_GRACE_S:.0f}s with every worker "
                        "throttled and none usable: RunPod is not scheduling the requested GPU "
                        "class here"
                    )
        else:
            # Status unreadable. Neither branch above ran, so without this the armed anchors keep
            # aging on wall _preload().time the poller could not see -- and a run that ended and was re-queued
            # inside that blackout would be torn down on its first queued reading, on a grace it
            # never actually served. Hold the confirmed duration; the gap itself proves nothing.
            now = _preload().time.time()
            starved.unknown(now)
            unhealthy.unknown(now)
            throttled.unknown(now)
        if status in _TERMINAL_OK:
            output = (st or {}).get("output")
            if not output:
                # COMPLETED with no output = broken worker image or API mismatch, not a warmed region.
                return {"error": f"preload job {job_id} completed with no output"}
            try:
                return _preload().decode_output(output) or {}
            except Exception as exc:
                return {"error": str(exc)}
        if status in _TERMINAL_FAIL:
            raise RuntimeError(f"preload job {job_id} ended {status}: {(st or {}).get('error')}")
        _preload().time.sleep(poll_interval_s)
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

    models = models or _preload().catalog_model_ids()
    dc_ids = datacenters or [dc.value for dc in _preload().weight_cache_datacenters()]
    # Validate all DC ids up front so a bad id fails before any paid endpoint launches.
    for d in dc_ids:
        DataCenter.from_string(d)
    token = token or os.environ.get("HF_TOKEN")
    _preload().logger.info("warming %d datacenter(s) with %d model(s)", len(dc_ids), len(models))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {
            pool.submit(
                _preload()._preload_one_dc, dc, models, token, gpu, timeout_s, poll_interval_s
            ): dc
            for dc in dc_ids
        }
        results: list[dict] = [fut.result() for fut in as_completed(futs)]
    ok = sum(1 for r in results if r.get("status") == "ok")
    starved = [r["datacenter"] for r in results if r.get("status") == "no_capacity"]
    _preload().logger.info("preload complete: %d/%d datacenters warmed", ok, len(results))
    # A "partial" DC downloaded some models and failed others. The worker already reports which ones
    # and why, but that detail died inside the result dict -- without it "partial" is unactionable.
    for r in results:
        if r.get("status") != "partial":
            continue
        failed = (r.get("result") or {}).get("failed") or {}
        for model_id, detail in sorted(failed.items()):
            _preload().logger.warning(
                "preload %s: %s FAILED: %s", r["datacenter"], model_id, detail
            )
    if starved:
        # Actionable: these stay cold until re-run with a class the DC actually stocks.
        _preload().logger.warning(
            "no %s capacity in %s -- re-run those with --datacenters %s --gpu <class>",
            gpu,
            ", ".join(starved),
            ",".join(starved),
        )
    return results


def teardown_weight_cache(datacenters: list[str] | None = None) -> list[str]:
    """Delete per-DC weight-cache volumes. Sweeps every account in the RUNPOD_API_KEY pool.

    ``datacenters=None`` → whole fleet; ``[]`` → no-op (never widened to all — that's a footgun).
    """

    from flash.providers.runpod.client import auth as rp_keys
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME

    # Explicit [] is a no-op — never widen zero DCs to the whole fleet.
    if datacenters is not None and not datacenters:
        _preload().logger.info(
            "teardown: empty datacenter scope — nothing to reclaim (refusing to widen to all)"
        )
        return []
    pool = rp_keys.keys()
    if not pool:
        _preload().logger.info(
            "teardown: RUNPOD_API_KEY not configured — skipping RunPod cache teardown"
        )
        return []
    # Import SDK after early returns: may be absent on instance-only control planes.
    from runpod_flash.core.api.runpod import RunpodRestClient
    from runpod_flash.core.resources.datacenter import DataCenter
    from runpod_flash.core.urls import RUNPOD_REST_API_URL

    dc_ids = (
        datacenters if datacenters else [dc.value for dc in _preload().weight_cache_datacenters()]
    )
    targets = {
        weight_cache_volume_name(WEIGHT_CACHE_VOLUME_NAME, DataCenter.from_string(d))
        for d in dc_ids
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
            _preload().logger.warning(
                "teardown: %d cache volume(s) FAILED to delete (still present): %s",
                len(still),
                ", ".join(sorted(still)),
            )
        return gone

    multi = len(pool) > 1
    deleted: list[str] = []
    failed_accounts: list[str] = []
    for i, key in enumerate(pool):
        try:
            names = _preload()._run_async(_go_one(key))
        except Exception as exc:
            failed_accounts.append(f"acct{i}")
            _preload().logger.warning(
                "teardown: RunPod account %d sweep FAILED (continuing): %s", i, exc
            )
            continue
        deleted.extend((f"acct{i}:{n}" if multi else n) for n in names)
    if failed_accounts:
        _preload().logger.warning(
            "teardown: %d of %d RunPod account(s) failed to sweep (%s) — their cache volumes may "
            "still be billed; re-run teardown once the key(s) are valid",
            len(failed_accounts),
            len(pool),
            ", ".join(failed_accounts),
        )
    return deleted


def teardown_lambda_filesystems(name: str | None = None) -> list[str]:
    """Delete Lambda weight-cache filesystems across all regions. Best-effort and idempotent."""
    from flash.providers.lambda_.client import api as lambda_api
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME

    target = name or WEIGHT_CACHE_VOLUME_NAME
    deleted: list[str] = []
    try:
        fses = lambda_api.list_filesystems()
    except Exception as exc:
        _preload().logger.warning("teardown: lambda list_filesystems failed (skipping): %s", exc)
        return deleted
    for fs in fses:
        if fs.get("name") == target and fs.get("id") and lambda_api.delete_filesystem(fs["id"]):
            region = (fs.get("region") or {}).get("name") or "?"
            deleted.append(f"lambda:{region}/{target}")
    return deleted
