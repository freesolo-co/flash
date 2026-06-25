"""Preload (warm) the shared weight-cache volumes with the catalog's base-model weights.

The weight cache (a same-named ``flash-weights`` network volume in every storage datacenter) is
populated lazily by real runs: the first run to land in a region downloads the model onto that
region's volume, and every later run there hits the cache. This module makes that first hit free by
PRE-warming every region up front — an operator/setup action, not a user one (the cache is fully
managed, so there is no user-facing knob).

Mechanism: for each datacenter, deploy a short-lived worker with ONLY that region's volume attached
(pinned to that single DC, so the worker provably lands there), run the baked handler in ``preload``
mode (download-only, ``HF_HOME`` -> the mounted volume), then tear the endpoint down. Reuses the
existing baked worker image + deploy/submit/quota machinery; the only new worker code is the
``preload`` branch in ``train.endpoints._train_body``.

COST / GC NOTE: the fleet is permanent, billed standing storage. A run (or a full preload) creates a
``flash-weights-<dc>`` volume in EVERY storage datacenter (one per DataCenter.all() entry — currently
~11 x 100 GB ~= 1.1 TB, ~$77/mo; grows by one volume if the SDK adds a storage region), and RunPod
network volumes are NOT auto-deleted — there is no GC. Reclaim them with ``--teardown`` (deletes
every per-DC weight-cache volume across ALL pool accounts via the RunPod REST API).

Run it::

    python -m flash.providers.runpod.preload                 # all catalog models, all DCs
    python -m flash.providers.runpod.preload --datacenters US-CA-2,EU-RO-1 --models Qwen/Qwen3.5-4B
    python -m flash.providers.runpod.preload --dry-run       # print the plan, provision nothing
    python -m flash.providers.runpod.preload --teardown      # DELETE the cache volumes (reclaim $)
"""

from __future__ import annotations

import argparse
import contextlib
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from flash._logging import get_logger
from flash.providers.runpod import api as runpod_api
from flash.providers.runpod.jobs import (
    build_function_input,
    deploy_train_endpoint,
    weight_cache_datacenters,
    weight_cache_volume_name,
)

logger = get_logger(__name__)


def _run_async(coro):
    """Run a coroutine to completion from sync code, even if an event loop is already running.

    teardown is normally a sync CLI/operator entrypoint (asyncio.run is fine), but it may also be
    called from an async context (a notebook, a FastAPI handler) where ``asyncio.run`` raises
    "cannot be called from a running event loop". In that case run it on a worker thread instead.
    """
    import asyncio as _asyncio

    try:
        _asyncio.get_running_loop()
    except RuntimeError:
        return _asyncio.run(coro)  # no running loop — the normal CLI/sync path
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_asyncio.run, coro).result()


_HF_HOME = "/runpod-volume/hf-cache"
# Cheapest broadly-available class; preload only downloads (no compute), so the GPU is incidental —
# the job is short, so the cost is a few cents per region.
_PRELOAD_GPU = "RTX 4090"
_TERMINAL_OK = {"COMPLETED"}
_TERMINAL_FAIL = {"FAILED", "CANCELLED", "TIMED_OUT"}


def catalog_model_ids() -> list[str]:
    """The public base models to warm: every curated catalog entry (the cache holds public weights).

    Open-model-policy (``allow``) runs may use arbitrary/private models that aren't worth — or safe —
    to pre-warm globally; those simply download cold on first use and then cache like any other.
    """
    from flash.catalog import MODELS

    return list(MODELS)


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
    # SAME per-DC physical name the training path uses (weight_cache_volume_name), so preload warms
    # exactly the volume a later run in this DC will mount.
    vol_name = weight_cache_volume_name(WEIGHT_CACHE_VOLUME_NAME, dc)
    # Pass a FACTORY (not a prebuilt dict): deploy_train_endpoint may fail over across accounts under
    # a multi-key pool, and the SDK stamps an account-scoped id onto a NetworkVolume — so each account
    # attempt must build a fresh volume, else the next account reuses the first's stale id and the
    # single-DC preload fails.
    def _endpoint_kwargs():
        return {
            "volume": [NetworkVolume(name=vol_name, size=WEIGHT_CACHE_VOLUME_GB, datacenter=dc)],
            "datacenter": [dc],
        }

    endpoint_id = None
    try:
        endpoint_id, _name = deploy_train_endpoint(
            gpu,
            execution_timeout_ms=timeout_s * 1000,
            # Unique per invocation: RunPod reuses an endpoint by name, so a stable suffix could
            # resolve a stale (deleted) endpoint id from a prior preload's persisted SDK state on a
            # long-lived control plane. A fresh suffix each run sidesteps that.
            name_suffix=f"preload-{dc_id.lower()}-{uuid.uuid4().hex[:6]}",
            spec=None,
            endpoint_kwargs=_endpoint_kwargs,
        )
        # HF_HUB_ENABLE_HF_TRANSFER is exported by the worker image (Dockerfile.worker ENV), so it is
        # not passed here — only HF_HOME (the per-region mount) and the token need overriding.
        payload = {
            "mode": "preload",
            "models": models,
            "env": {"HF_HOME": _HF_HOME, **({"HF_TOKEN": token} if token else {})},
        }
        job_id = runpod_api.submit_job(endpoint_id, build_function_input(payload))
        logger.info("preload %s: job %s submitted (%d models)", dc_id, job_id, len(models))
        result = _poll_until_done(endpoint_id, job_id, timeout_s, poll_interval_s)
        # The job COMPLETED, but the handler reports per-model failures (and a hard error if the
        # volume wasn't mounted) inside its result — a completed job is NOT necessarily a warmed
        # region. Surface those so the driver/CLI don't count a no-op (or partial) warm as success.
        if result.get("error"):
            return {"datacenter": dc_id, "status": "error", "error": result["error"], "result": result}
        # Warmed this DC's volume -> record it so the LAZY training path attaches+uses it (operator
        # preload is how a DC enters the used set without waiting for a cold run to land there first).
        from flash.runner import record_weight_cache_dc

        record_weight_cache_dc(dc_id)
        if result.get("failed"):
            return {"datacenter": dc_id, "status": "partial", "result": result}
        return {"datacenter": dc_id, "status": "ok", "result": result}
    except Exception as exc:  # one region failing must not abort the others
        logger.warning("preload %s FAILED: %s", dc_id, exc)
        return {"datacenter": dc_id, "status": "error", "error": str(exc)}
    finally:
        if endpoint_id:
            with contextlib.suppress(Exception):
                runpod_api.delete_endpoint(endpoint_id)


def _poll_until_done(
    endpoint_id: str, job_id: str, timeout_s: int, poll_interval_s: float
) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        st = runpod_api.job_status(endpoint_id, job_id)
        status = (st or {}).get("status")
        if status in _TERMINAL_OK:
            return (st or {}).get("output") or {}
        if status in _TERMINAL_FAIL:
            raise RuntimeError(f"preload job {job_id} ended {status}: {(st or {}).get('error')}")
        time.sleep(poll_interval_s)
    raise TimeoutError(f"preload job {job_id} did not finish within {timeout_s}s")


def warm_weight_cache(
    models: list[str] | None = None,
    datacenters: list[str] | None = None,
    gpu: str = _PRELOAD_GPU,
    timeout_s: int = 1800,
    max_workers: int = 8,
    poll_interval_s: float = 10.0,
    token: str | None = None,
) -> list[dict]:
    """Warm every (datacenter) volume with the given models. Returns one result dict per DC.

    Datacenters are warmed concurrently (bounded by ``max_workers``). A region that errors is
    reported in its result dict and does not abort the others.
    """
    models = models or catalog_model_ids()
    dc_ids = datacenters or [dc.value for dc in weight_cache_datacenters()]
    token = token or os.environ.get("HF_TOKEN")
    logger.info("warming %d datacenter(s) with %d model(s)", len(dc_ids), len(models))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {
            pool.submit(_preload_one_dc, dc, models, token, gpu, timeout_s, poll_interval_s): dc
            for dc in dc_ids
        }
        results: list[dict] = [fut.result() for fut in as_completed(futs)]
    ok = sum(1 for r in results if r.get("status") == "ok")
    logger.info("preload complete: %d/%d datacenters warmed", ok, len(results))
    return results


def teardown_weight_cache(datacenters: list[str] | None = None) -> list[str]:
    """Delete the per-DC ``flash-weights-<dc>`` cache volumes to reclaim the standing storage.

    RunPod network volumes are never auto-GC'd, so this is the only way to stop the monthly bill
    short of the console. Returns the names deleted (``account:name`` when a multi-account pool is
    configured). Targets ONLY this fleet's per-DC names (built from ``WEIGHT_CACHE_VOLUME_NAME``),
    never other volumes.

    Sweeps EVERY account in the ``RUNPOD_API_KEY`` pool: ``deploy_train_endpoint`` fails over to
    another account on a quota error, so a cache volume may have been created under any pool key —
    a single-account teardown would leak the volumes the failover created elsewhere.
    """

    from runpod_flash.core.api.runpod import RunpodRestClient
    from runpod_flash.core.resources.datacenter import DataCenter
    from runpod_flash.core.urls import RUNPOD_REST_API_URL

    from flash.providers.runpod import keys as rp_keys
    from flash.runner import WEIGHT_CACHE_VOLUME_NAME

    dc_ids = datacenters or [dc.value for dc in weight_cache_datacenters()]
    targets = {
        weight_cache_volume_name(WEIGHT_CACHE_VOLUME_NAME, DataCenter.from_string(d)) for d in dc_ids
    }
    pool = rp_keys.keys() or [None]  # [None] -> RunpodRestClient() resolves the key from the env

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
            # RunPod's DELETE /networkvolumes/{id} returns 204 No Content, which the SDK's
            # _execute_rest chokes on (it always await response.json()). Swallow that — we confirm
            # the actual outcome by RE-LISTING below, not by trusting the delete's parsed response.
            with contextlib.suppress(Exception):
                await client._execute_rest("DELETE", f"{RUNPOD_REST_API_URL}/networkvolumes/{vid}")
        remaining = await _names(client)
        gone = [name for name in to_delete if name not in remaining]  # provably gone (confirmed)
        # A target still present after its delete means a REAL failure (auth/permission/5xx/network)
        # that the 204-tolerant suppress() above hid — surface it so a failed reclaim isn't silent.
        still = [name for name in to_delete if name in remaining]
        if still:
            logger.warning("teardown: %d cache volume(s) FAILED to delete (still present): %s",
                           len(still), ", ".join(sorted(still)))
        return gone

    multi = len(pool) > 1
    deleted: list[str] = []
    for i, key in enumerate(pool):
        names = _run_async(_go_one(key))
        deleted.extend((f"acct{i}:{n}" if multi else n) for n in names)
    return deleted


def teardown_lambda_filesystems(name: str | None = None) -> list[str]:
    """Delete the Lambda persistent filesystems named ``name`` (default ``flash-weights``) across ALL
    regions, reclaiming the standing NFS cache storage.

    Best-effort and idempotent: Lambda refuses to delete a filesystem that is still in use (an
    instance is mounting it), so a live run keeps its cache — re-run teardown once the run finishes.
    Returns ``lambda:<region>/<name>`` per filesystem deleted. A missing/empty Lambda key is not an
    error (nothing to reclaim) — it logs and returns ``[]``.
    """
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


def teardown_hyperstack_volumes(name: str | None = None) -> list[str]:
    """Delete the Hyperstack cache volumes named ``name`` (default ``flash-weights``) across ALL
    environments, reclaiming the standing block storage.

    Best-effort and idempotent: a volume attached to a live VM won't delete — re-run once the run
    finishes. Returns ``hyperstack:<env>/<name>`` per volume deleted. A missing Hyperstack key is not
    an error — it logs and returns ``[]``.
    """
    from flash.providers.hyperstack import api as hs_api
    from flash.runner import WEIGHT_CACHE_VOLUME_NAME

    target = name or WEIGHT_CACHE_VOLUME_NAME
    deleted: list[str] = []
    try:
        vols = hs_api.list_volumes()
    except Exception as exc:
        logger.warning("teardown: hyperstack list_volumes failed (skipping): %s", exc)
        return deleted
    for v in vols:
        if v.get("name") == target and v.get("id") and hs_api.delete_volume(v["id"]):
            env = (v.get("environment") or {}).get("name") or "?"
            deleted.append(f"hyperstack:{env}/{target}")
    return deleted


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Preload the flash weight-cache volumes.")
    ap.add_argument("--models", help="comma-separated HF model ids (default: whole catalog)")
    ap.add_argument("--datacenters", help="comma-separated DC ids (default: all storage DCs)")
    ap.add_argument("--gpu", default=_PRELOAD_GPU, help="GPU class for the preload worker")
    ap.add_argument("--timeout-s", type=int, default=1800, help="per-DC job timeout")
    ap.add_argument("--max-workers", type=int, default=8, help="datacenters warmed concurrently")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, provision nothing")
    ap.add_argument(
        "--teardown", action="store_true",
        help="DELETE the per-DC weight-cache volumes (reclaim standing storage) and exit",
    )
    args = ap.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else catalog_model_ids()
    dcs = (
        [d.strip() for d in args.datacenters.split(",") if d.strip()]
        if args.datacenters
        else [dc.value for dc in weight_cache_datacenters()]
    )
    if args.teardown:
        # Reclaim the cache storage on EVERY provider: RunPod network volumes, Lambda filesystems,
        # and Hyperstack block volumes (each best-effort; a provider with no configured key is a no-op).
        deleted = teardown_weight_cache(dcs)
        deleted += teardown_lambda_filesystems()
        deleted += teardown_hyperstack_volumes()
        print(f"deleted {len(deleted)} weight-cache volume(s): {', '.join(deleted) or '(none)'}")
        return 0
    if args.dry_run:
        print(f"would warm {len(dcs)} datacenter(s): {', '.join(dcs)}")
        print(f"with {len(models)} model(s): {', '.join(models)}")
        return 0

    results = warm_weight_cache(
        models=models, datacenters=dcs, gpu=args.gpu,
        timeout_s=args.timeout_s, max_workers=args.max_workers,
    )
    failed = [r for r in results if r.get("status") != "ok"]
    for r in results:
        print(f"  {r['datacenter']}: {r['status']}" + (f" ({r.get('error')})" if r.get("error") else ""))
    print(f"{len(results) - len(failed)}/{len(results)} datacenters warmed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
