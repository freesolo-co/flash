"""Preload (warm) the shared weight-cache volumes with the catalog's base-model weights.

Run it::

    python -m flash.providers.runpod.preload                 # all catalog models, all DCs
    python -m flash.providers.runpod.preload --datacenters US-CA-2,EU-RO-1 --models Qwen/Qwen3.5-4B
    python -m flash.providers.runpod.preload --dry-run       # print the plan, provision nothing
    python -m flash.providers.runpod.preload --teardown      # DELETE the cache volumes (reclaim $)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from flash._logging import get_logger
from flash.providers._poll import preload_instance_run_id
from flash.providers.runpod import api as runpod_api
from flash.providers.runpod.jobs import (
    build_function_input,
    decode_output,
    deploy_train_endpoint,
    make_hf_text_reader,
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


def catalog_model_ids() -> list[str]:
    """Catalog base models that fit the weight-cache volume."""
    from flash.catalog import MODELS
    from flash.runner import _fits_weight_cache

    return [mid for mid, info in MODELS.items() if _fits_weight_cache(info)]


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
    try:
        endpoint_id, _name = deploy_train_endpoint(
            gpu,
            execution_timeout_ms=timeout_s * 1000,
            name_suffix=f"preload-{dc_id.lower()}-{uuid.uuid4().hex[:6]}",
            spec=None,
            endpoint_kwargs=_endpoint_kwargs,
        )
        payload = {
            "mode": "preload",
            "models": models,
            "env": {"HF_HOME": _HF_HOME, **({"HF_TOKEN": token} if token else {})},
        }
        job_id = runpod_api.submit_job(endpoint_id, build_function_input(payload))
        logger.info("preload %s: job %s submitted (%d models)", dc_id, job_id, len(models))
        result = _poll_until_done(endpoint_id, job_id, timeout_s, poll_interval_s)
        if result.get("error"):
            return {"datacenter": dc_id, "status": "error", "error": result["error"], "result": result}
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
    timeout_s: int = 1800,
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
    logger.info("preload complete: %d/%d datacenters warmed", ok, len(results))
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


_PRELOAD_INSTANCE_GPU = os.environ.get("FLASH_PRELOAD_INSTANCE_GPU") or "A10"
_PRELOAD_GPU_BY_PROVIDER = {"lambda": "A10"}
_PRELOAD_STATUS_REPO = os.environ.get("FLASH_PRELOAD_STATUS_REPO") or "Freesolo-Co/flash-weight-preload"


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
        "train": {"hf_repo": _PRELOAD_STATUS_REPO, "seeds": [0]},
        "gpu": {"type": gpu, "max_wall_seconds": max(60, int(wall_s)),
                "network_volume": WEIGHT_CACHE_VOLUME_NAME, "network_volume_gb": WEIGHT_CACHE_VOLUME_GB},
    })


def _warm_one_instance(provider: str, jobs_mod, candidate, models: list, gpu: str,
                       token: str | None, timeout_s: int, poll_interval_s: float) -> dict:
    """Launch a download-only preload instance pinned to ``candidate``'s region, poll its status
    marker, then ALWAYS terminate. One region failing never aborts the others."""
    region = getattr(candidate, "region", "?")
    effective_s = max(60, int(timeout_s))
    # Embed reap deadline in the run_id so orphan sweep can free the box if this driver process dies.
    reap_deadline = int(time.time()) + effective_s
    run_id = preload_instance_run_id(provider, region, reap_deadline, uuid.uuid4().hex[:6])
    spec = _preload_instance_spec(gpu, run_id, wall_s=effective_s)
    prefix = f"{spec.phase}/{run_id}/seed0"
    reader = make_hf_text_reader(_PRELOAD_STATUS_REPO, f"{prefix}/preload_result.json",
                                 min_interval_s=max(5.0, poll_interval_s))
    # Also watch the attempt marker: if the box dies early the failmark is the only signal (avoids
    # polling to full timeout on a dead box). Completion file is authoritative when present.
    fail_reader = make_hf_text_reader(_PRELOAD_STATUS_REPO, f"{prefix}/{provider}_attempt0.json",
                                      min_interval_s=max(5.0, poll_interval_s))
    try:
        try:
            jobs_mod.launch_and_submit(spec, seed=0, instances=[candidate], attempt=0,
                                       mode="preload", models=models)
        except Exception as exc:  # no capacity / launch reject — skip this region (warm-on-first-run covers it)
            return {"provider": provider, "region": region, "status": "error", "error": f"launch: {exc}"}
        logger.info("warm %s/%s: launched preload (%d models)", provider, region, len(models))
        deadline = time.time() + effective_s
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
                    return {"provider": provider, "region": region,
                            "status": "partial" if bad else "ok", "result": fail}
                if not fail.get("ok", True):
                    # Completion file is authoritative: a partial run writes it before the fail marker,
                    # so re-check once before reporting early death.
                    text = reader(force=True)
                    if text:
                        break
                    return {"provider": provider, "region": region, "status": "error",
                            "error": f"box failed early: {fail.get('error') or 'see boot log'}"}
            time.sleep(max(5.0, poll_interval_s))
        if not text:
            return {"provider": provider, "region": region, "status": "timeout"}
        result = json.loads(text)
        bad = result.get("error") or result.get("failed")
        return {"provider": provider, "region": region,
                "status": "partial" if bad else "ok", "result": result}
    except Exception as exc:
        return {"provider": provider, "region": region, "status": "error", "error": str(exc)}
    finally:
        with contextlib.suppress(Exception):
            jobs_mod.terminate_run_instances(run_id)


def warm_instances(models: list | None = None, gpu: str | None = None,
                   providers: list | None = None, timeout_s: int = 1800,
                   poll_interval_s: float = 20.0, max_workers: int = 4) -> list[dict]:
    """Warm Lambda caches: one download-only launch per region with capacity. Returns status per region."""
    models = models or catalog_model_ids()
    providers = providers or ["lambda"]
    token = os.environ.get("HF_TOKEN")

    from flash.providers.lambdalabs import jobs as lambda_jobs

    mods = {"lambda": lambda_jobs}
    region_ok: dict = {}
    targets: list = []
    for provider in providers:
        jobs_mod = mods.get(provider)
        if jobs_mod is None:
            continue
        provider_gpu = gpu or _PRELOAD_GPU_BY_PROVIDER.get(provider, _PRELOAD_INSTANCE_GPU)
        cache_capable = region_ok.get(provider)
        seen_regions: set = set()
        try:
            candidates = jobs_mod.usable_instances(provider_gpu)
        except Exception as exc:
            logger.warning("warm %s: usable_instances(%s) failed (skipping): %s", provider, provider_gpu, exc)
            continue
        for c in candidates:
            if c.region in seen_regions:
                continue
            if cache_capable is not None and not cache_capable(c.region):
                logger.info("warm %s: skipping cache-incapable region %s", provider, c.region)
                seen_regions.add(c.region)
                continue
            seen_regions.add(c.region)
            targets.append((provider, jobs_mod, c, provider_gpu))
    if not targets:
        logger.warning("warm: no Lambda capacity right now (nothing to warm)")
        return []
    # Fail fast before launching paid GPUs: status repo is the only completion signal.
    try:
        _ensure_status_repo(token)
    except Exception as exc:
        raise RuntimeError(
            f"preload status repo {_PRELOAD_STATUS_REPO!r} unavailable ({exc}); set a valid HF_TOKEN "
            "with write access before warming (refusing to launch paid GPUs that can't report)."
        ) from exc
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(_warm_one_instance, provider, jobs_mod, c, models, provider_gpu, token, timeout_s, poll_interval_s)
            for (provider, jobs_mod, c, provider_gpu) in targets
        ]
        return [f.result() for f in as_completed(futs)]


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
            lambda_api.ensure_filesystem(target, region)
            done.append(f"lambda:{region}")
        except Exception as exc:
            logger.warning("provision: lambda ensure_filesystem(%s, %s) failed: %s", target, region, exc)
    return done


def provision_all() -> list[str]:
    """Eagerly create instance-provider cache storage in every region (GPU-free). RunPod volumes are
    created automatically by endpoint deploy, so only Lambda is provisioned here."""
    return provision_lambda_filesystems()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Preload the flash weight-cache volumes.")
    ap.add_argument("--models", help="comma-separated HF model ids (default: whole catalog)")
    ap.add_argument("--datacenters", help="comma-separated DC ids (default: all storage DCs)")
    ap.add_argument(
        "--gpu", default=None,
        help="GPU class for the preload worker. Defaults are per-mode (RunPod warm -> "
             f"{_PRELOAD_GPU!r}; --warm-instances -> {_PRELOAD_INSTANCE_GPU!r}); pass this to override "
             "either. Defaulting to None (not a sentinel string) lets you explicitly pick even the "
             "per-mode default GPU without it being mistaken for 'no override'.",
    )
    ap.add_argument("--timeout-s", type=int, default=1800, help="per-DC job timeout")
    ap.add_argument(
        "--max-workers", type=int, default=4,
        help="datacenters warmed concurrently. Each one deploys a preload endpoint, so this MUST stay "
             "under your RunPod endpoint/worker quota (the documented default is 5); the default of 4 "
             "leaves a 1-slot buffer. Raise it only if your account quota is higher.",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the plan, provision nothing")
    ap.add_argument(
        "--provision", action="store_true",
        help="CREATE the Lambda cache storage in every region (pure API, no GPU) and "
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
        provisioned = provision_all()
        print(f"provisioned {len(provisioned)} instance-provider cache store(s): "
              f"{', '.join(provisioned) or '(none — no Lambda key, or no regions)'}")
        return 0
    if args.warm_instances:
        if args.dry_run:
            print("would warm Lambda caches (one download-only launch per region with capacity)")
            return 0
        results = warm_instances(models=models, gpu=args.gpu,
                                 timeout_s=args.timeout_s, max_workers=args.max_workers)
        if not results:
            print("0 regions warmed — no Lambda region had capacity to warm right now "
                  "(weights download cold on first run). Nothing launched.")
            return 0
        failed = [r for r in results if r.get("status") not in ("ok",)]
        for r in results:
            print(f"  {r['provider']}/{r['region']}: {r['status']}"
                  + (f" ({r.get('error')})" if r.get("error") else ""))
        print(f"{len(results) - len(failed)}/{len(results)} regions warmed")
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
