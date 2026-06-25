"""Preload (warm) the shared weight-cache volumes with the catalog's base-model weights.

The weight cache is the eager ``flash-weights-<dc>`` network-volume fleet (one volume per storage
datacenter, created-or-attached on every endpoint deploy). This module WARMS that fleet — downloads
the catalog's base-model weights onto each region's volume up front — so the very first run in any
region is already warm. An operator/setup action, not a user one (the cache is fully managed, so
there is no user-facing knob).

Mechanism: for each datacenter, deploy a short-lived worker with ONLY that region's volume attached
(pinned to that single DC, so the worker provably lands there), run the baked handler in ``preload``
mode (download-only, ``HF_HOME`` -> the mounted volume), then tear the endpoint down. Reuses the
existing baked worker image + deploy/submit/quota machinery; the only new worker code is the
``preload`` branch in ``train.endpoints._train_body``.

COST / GC NOTE: the fleet is permanent, billed standing storage. Eager provisioning means a
``flash-weights-<dc>`` volume exists in EVERY storage datacenter (one per DataCenter.all() entry —
currently ~11 x 100 GB ~= 1.1 TB, ~$77/mo; grows by one volume if the SDK adds a storage region),
created by the first endpoint deploy (or ``--provision`` / a full preload), and RunPod network
volumes are NOT auto-deleted — there is no GC. Reclaim them with ``--teardown`` (deletes every
per-DC weight-cache volume across ALL pool accounts via the RunPod REST API).

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
        # Warmed this DC's volume. Nothing else to record: the training path attaches the eager fleet
        # (a volume in every storage DC) regardless, so the warm weights are picked up automatically.
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
            # RunPod may surface a completed job's ``output`` as a JSON STRING (not a dict); decode it
            # to the handler's metrics dict (the same helper the normal poller uses) so the caller's
            # ``result.get(...)`` never crashes on a str and mis-reports a warmed region as failed.
            output = (st or {}).get("output")
            return (decode_output(output) or {}) if output else {}
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
    """Warm every (datacenter) volume with the given models. Returns one result dict per DC.

    Datacenters are warmed concurrently (bounded by ``max_workers``). Each concurrent warm deploys a
    preload endpoint, so ``max_workers`` MUST stay under the RunPod endpoint/worker quota (documented
    default 5) — the default of 4 leaves a buffer so extra deploys don't fail on quota. A region that
    errors is reported in its result dict and does not abort the others.
    """
    from runpod_flash.core.resources.datacenter import DataCenter

    models = models or catalog_model_ids()
    dc_ids = datacenters or [dc.value for dc in weight_cache_datacenters()]
    # Validate the WHOLE --datacenters scope to concrete DataCenter values BEFORE submitting any
    # futures: the per-DC parse otherwise runs inside _preload_one_dc on a worker thread, so a single
    # bad id would raise through fut.result() only AFTER the valid DCs already deployed paid preload
    # endpoints — aborting the command with money already spent. Parse up front so an invalid id fails
    # the whole command (naming the bad id + listing valid ones via DataCenter.from_string) before any
    # endpoint launches.
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
    """Delete the per-DC ``flash-weights-<dc>`` cache volumes to reclaim the standing storage.

    RunPod network volumes are never auto-GC'd, so this is the only way to stop the monthly bill
    short of the console. Returns the names deleted (``account:name`` when a multi-account pool is
    configured). Targets ONLY this fleet's per-DC names (built from ``WEIGHT_CACHE_VOLUME_NAME``),
    never other volumes.

    Sweeps EVERY account in the ``RUNPOD_API_KEY`` pool: ``deploy_train_endpoint`` fails over to
    another account on a quota error, so a cache volume may have been created under any pool key —
    a single-account teardown would leak the volumes the failover created elsewhere.

    ``datacenters`` semantics: ``None`` (the default) = the WHOLE storage-DC fleet; a non-empty list =
    just those DCs; an EXPLICIT empty list ``[]`` = nothing (returns ``[]``). The empty-list case must
    NOT widen to the full fleet — a caller that resolved a scope down to zero DCs intends a no-op, and
    silently nuking every cache there would be a destructive footgun.
    """

    from flash.providers.runpod import keys as rp_keys
    from flash.runner import WEIGHT_CACHE_VOLUME_NAME

    # An EXPLICIT empty scope ([]) is a no-op, NOT "all" — never widen zero DCs to the whole fleet.
    if datacenters is not None and not datacenters:
        logger.info("teardown: empty datacenter scope — nothing to reclaim (refusing to widen to all)")
        return []
    pool = rp_keys.keys()
    if not pool:
        # No RunPod key configured (e.g. an instance-only control plane): this is a best-effort
        # no-op, NOT an error — RunpodRestClient() would raise on a missing key and (under a chained
        # `--teardown`) could abort the Lambda/Hyperstack reclaim. Mirror the instance providers'
        # missing-key behavior: log and return nothing reclaimed.
        logger.info("teardown: RUNPOD_API_KEY not configured — skipping RunPod cache teardown")
        return []
    # Import the runpod_flash SDK only AFTER the empty-scope / no-key early returns: on an
    # instance-only control plane the SDK may be unavailable, and importing it at the top would defeat
    # the intended best-effort no-op (a missing-key teardown must not raise on an absent SDK).
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
    failed_accounts: list[str] = []
    for i, key in enumerate(pool):
        # One bad key (expired / revoked / network) must NOT abort the sweep: the cache volume a
        # failover created under a LATER account would otherwise stay billed forever. Catch, record,
        # and keep going so every other account is still reclaimed.
        try:
            names = _run_async(_go_one(key))
        except Exception as exc:
            failed_accounts.append(f"acct{i}")
            logger.warning("teardown: RunPod account %d sweep FAILED (continuing): %s", i, exc)
            continue
        deleted.extend((f"acct{i}:{n}" if multi else n) for n in names)
    if failed_accounts:
        # Surface the aggregate so a fully-failed (or partially-failed) sweep is observable, not silent
        # — the caller logs/returns `deleted`, which would otherwise hide that some accounts never ran.
        logger.warning(
            "teardown: %d of %d RunPod account(s) failed to sweep (%s) — their cache volumes may "
            "still be billed; re-run teardown once the key(s) are valid",
            len(failed_accounts), len(pool), ", ".join(failed_accounts),
        )
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

    base = name or WEIGHT_CACHE_VOLUME_NAME
    deleted: list[str] = []
    try:
        vols = hs_api.list_volumes()
    except Exception as exc:
        logger.warning("teardown: hyperstack list_volumes failed (skipping): %s", exc)
        return deleted
    # Allowlist of EXACT deterministic cache-fleet names — the per-region ``flash-weights-<region>``
    # this code provisions, PLUS the legacy bare ``flash-weights`` from before per-region naming. A
    # broad ``startswith(base + "-")`` prefix would also nuke unrelated user volumes like
    # ``flash-weights-backup`` / ``flash-weights-test``, so match exact names only.
    fleet = {base}
    try:
        fleet |= {hs_api.cache_volume_name(base, r) for r in hs_api.cache_regions()}
    except Exception as exc:
        # cache_regions() failed (API down / auth) — we genuinely cannot enumerate the canonical
        # region set, so we CANNOT distinguish a fleet volume ``flash-weights-us-1`` from a user volume
        # ``flash-weights-backup-1`` / ``flash-weights-test-1`` (both are region-shaped). FAVOR DATA
        # SAFETY: do NOT guess-delete per-region volumes by pattern — a missed cache volume is just
        # recoverable leftover billing, but deleting a user's volume is unrecoverable data loss. Delete
        # ONLY the unambiguous legacy bare ``base`` name, and warn LOUDLY that the per-region cache
        # volumes could not be enumerated and were LEFT INTACT (re-run once regions are reachable, or
        # clean them manually). This still satisfies "failure is loud/observable, never silently
        # narrowed" without the over-broad deletion.
        logger.warning(
            "teardown: hyperstack cache_regions failed (%s) — could NOT enumerate per-region cache "
            "volumes; deleting only the legacy bare %r and LEAVING any per-region "
            "flash-weights-<region> volumes INTACT (re-run teardown once regions are reachable, or "
            "delete them manually). Refusing to pattern-match region-shaped names to avoid deleting "
            "unrelated user volumes.",
            exc, base,
        )
    for v in vols:
        vname = v.get("name") or ""
        if vname in fleet and v.get("id") and hs_api.delete_volume(v["id"]):
            env = (v.get("environment") or {}).get("name") or "?"
            deleted.append(f"hyperstack:{env}/{vname}")
    return deleted


# Instance-provider WARM (Lambda + Hyperstack). RunPod warms via the serverless preload above; the
# instance providers have no serverless API, so a warm is a real (cheap, short) GPU launch in download
# -only mode: the bootstrap pulls the catalog into the mounted cache and exits (no worker). The box
# self-reports completion by uploading ``preload_result.json`` to a shared status repo, which the
# driver polls; the instance is ALWAYS terminated in a finally. Cheap class by default (the work is a
# download, not compute) — override with FLASH_PRELOAD_INSTANCE_GPU.
# Per-provider default warm GPU: a cheap class that the provider actually offers. A10 is LAMBDA-ONLY
# (no hyperstack_name), so using it for Hyperstack makes usable_instances("A10") empty/raise and
# Hyperstack is silently never warmed — pick L40 (a cheap Hyperstack datacenter card) there. An
# explicit --gpu / FLASH_PRELOAD_INSTANCE_GPU overrides BOTH.
_PRELOAD_INSTANCE_GPU = os.environ.get("FLASH_PRELOAD_INSTANCE_GPU") or "A10"
_PRELOAD_GPU_BY_PROVIDER = {"lambda": "A10", "hyperstack": "L40"}
# Shared dataset repo the preload boxes upload their status marker to (the driver polls it). The
# warmed WEIGHTS go to the per-region cache volume, NOT here — this holds only tiny status JSON.
_PRELOAD_STATUS_REPO = os.environ.get("FLASH_PRELOAD_STATUS_REPO") or "Freesolo-Co/flash-weight-preload"


def _ensure_status_repo(token: str | None) -> None:
    """Create the preload status dataset repo if absent (the boxes upload their marker there).

    RAISES on failure (missing/invalid HF_TOKEN, no access): the repo is the ONLY completion signal
    — without it every launched box runs until timeout_s with no preload_result.json, so the warm
    burns paid GPUs and reports nothing. Fail fast BEFORE launching instead of swallowing the error.
    """
    from huggingface_hub import HfApi

    HfApi(token=token).create_repo(_PRELOAD_STATUS_REPO, repo_type="dataset", exist_ok=True, private=True)


def _preload_instance_spec(gpu: str, run_id: str, wall_s: int = 1800):
    """A minimal download-only preload spec: cache attached, status marker repo, placeholder model
    (the bootstrap warms ``payload['models']``, not ``spec.model``). ``wall_s`` is the worker wall cap
    — thread the warm timeout in so a long catalog warm isn't killed at the hard-coded 30 min while
    the driver is still polling."""
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
    run_id = f"flash-preload-{provider}-{region.lower()}-{uuid.uuid4().hex[:6]}"
    # ONE effective budget shared by the worker wall cap AND the driver poll, so the two can't disagree.
    # The worker spec floors the wall cap at 60s (a sub-minute cap can't even boot+download), so the
    # driver must poll for that SAME floored budget — otherwise a `--timeout-s` under 60 would have the
    # driver report timeout + terminate the box at e.g. 30s while the worker still had ~60s to finish,
    # aborting an in-progress preload.
    effective_s = max(60, int(timeout_s))
    spec = _preload_instance_spec(gpu, run_id, wall_s=effective_s)
    prefix = f"{spec.phase}/{run_id}/seed0"
    reader = make_hf_text_reader(_PRELOAD_STATUS_REPO, f"{prefix}/preload_result.json",
                                 min_interval_s=max(5.0, poll_interval_s))
    # ALSO watch the attempt-failure marker (<arm>_attempt0.json): if the box dies BEFORE run_preload
    # uploads preload_result.json (docker/GPU never ready, image pull fails, the bootstrap crashes
    # early), the worker/host failmark uploader still writes this terminal ok=false marker. Without
    # watching it the driver would poll to the full effective_s on an already-dead box, burning paid
    # GPU. The completion file is authoritative when present (success or partial), so check it FIRST.
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
                    # Terminal SUCCESS marker, completion file lost to a transient upload failure. Treat
                    # the marker itself as the result (the completion file is still authoritative when
                    # present, but it never landed here). "partial" if the marker carries an
                    # error/failed field, else "ok".
                    bad = fail.get("error") or fail.get("failed")
                    return {"provider": provider, "region": region,
                            "status": "partial" if bad else "ok", "result": fail}
                if not fail.get("ok", True):
                    # The completion file (preload_result.json) is authoritative when present: a
                    # partial/failed-download run uploads it AND THEN writes the ok=false fail marker,
                    # so the marker can be visible an iteration before the completion file. Re-check
                    # the completion file ONE more time; if it's now there, fall through to the normal
                    # completion handling (-> "partial"/"ok") instead of mislabeling a completed
                    # (partial) preload as an early box death. Only the genuinely-still-absent case
                    # returns the early-death error.
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
    """WARM the Lambda + Hyperstack caches: one download-only launch per region that currently has
    capacity (regions with no capacity now are skipped — warm-on-first-run covers them). Each launch
    is pinned to its region, polled to completion, and terminated. Best-effort: a provider with no key
    / no capacity contributes nothing. Returns a status dict per region attempted.

    NB: the preload logic itself ships in the cloud-init ``user_data`` — ``_instance.build_user_data``
    reads the current ``_instance_bootstrap.py`` from the repo and embeds it, so every launch runs the
    latest bootstrap (no image rebuild needed for the preload code). The only image requirement is the
    HF download deps (huggingface_hub + hf_transfer), which the worker image already carries.
    """
    models = models or catalog_model_ids()
    providers = providers or ["lambda", "hyperstack"]
    token = os.environ.get("HF_TOKEN")

    from flash.providers.hyperstack import api as hs_api
    from flash.providers.hyperstack import jobs as hs_jobs
    from flash.providers.lambdalabs import jobs as lambda_jobs

    mods = {"lambda": lambda_jobs, "hyperstack": hs_jobs}
    # Per-provider "can this region host the cache?" predicate. Skipping a cache-incapable region (e.g.
    # Hyperstack CANADA-2, excluded from cache_regions()) BEFORE launching avoids burning a paid GPU
    # whose preload just reports "weight cache not supported in region" — which main() then counts as a
    # failed warm, so the default --warm-instances would fail even when every cache-capable region
    # succeeded. Lambda exposes no such filter (every region hosts filesystems), so it stays unfiltered.
    region_ok = {"hyperstack": hs_api.region_supports_cache}
    # One launch per region (dedupe so two candidates in a region don't double-launch — block volumes
    # are single-attach anyway). Each entry carries its provider's resolved GPU (an explicit override
    # applies to all; otherwise the per-provider default — so A10 doesn't silently skip Hyperstack).
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
            # Skip regions that can't host the cache for this provider — the preload would just report
            # "weight cache not supported in region" and be counted as a failed warm.
            if cache_capable is not None and not cache_capable(c.region):
                logger.info("warm %s: skipping cache-incapable region %s", provider, c.region)
                seen_regions.add(c.region)
                continue
            seen_regions.add(c.region)
            targets.append((provider, jobs_mod, c, provider_gpu))
    if not targets:
        logger.warning("warm: no Lambda/Hyperstack capacity right now (nothing to warm)")
        return []
    # Fail fast BEFORE launching any paid GPU: the status repo is the only completion signal, so if it
    # can't be created/accessed (missing/invalid HF_TOKEN) every box would just run to timeout warming
    # nothing observable. Surface a clear error instead of silently burning instances. Done only AFTER
    # the target list is built and the no-targets early-return above, so an empty warm (no capacity /
    # provider not configured) stays a harmless no-op and doesn't hard-fail on a missing HF_TOKEN.
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
    """Eagerly create the ``flash-weights`` filesystem in every Lambda region ``all_regions()`` can
    enumerate (create-if-absent), so the cache storage exists before runs land — pure control-plane
    API, no GPU.

    NB: Lambda has no standalone region list, so ``all_regions()`` is the UNION of regions currently
    advertising capacity across instance types — a region advertising ZERO capacity right now won't be
    covered here. That's fine: the launch-time ``ensure_filesystem`` backstop creates the FS the moment
    a run actually lands in such a region. So this is a best-effort eager warm, not a hard guarantee of
    coverage in literally every region Lambda might ever expose.

    Idempotent (``ensure_filesystem`` reuses an existing same-name FS). Returns ``lambda:<region>``
    per region provisioned. A missing/empty Lambda key is not an error (logs + returns ``[]``); a
    per-region failure is logged and skipped so one bad region never aborts the rest.
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


def provision_hyperstack_volumes(name: str | None = None, size_gb: int | None = None) -> list[str]:
    """Eagerly create the ``flash-weights`` block volume in EVERY Hyperstack environment
    (create-if-absent), so the cache storage exists before any run lands — pure control-plane API, no
    GPU.

    Idempotent (``ensure_volume`` reuses an existing same-name volume in the env). Returns
    ``hyperstack:<env>`` per environment provisioned. A missing Hyperstack key is not an error; a
    per-environment failure is logged and skipped.
    """
    from flash.providers.hyperstack import api as hs_api
    from flash.runner import WEIGHT_CACHE_VOLUME_GB, WEIGHT_CACHE_VOLUME_NAME

    base = name or WEIGHT_CACHE_VOLUME_NAME
    gb = int(size_gb or WEIGHT_CACHE_VOLUME_GB)
    done: list[str] = []
    try:
        # cache_regions() drops volume-incapable regions (e.g. CANADA-2) so we don't burn a
        # guaranteed-400 create on a region that can't host the cache anyway.
        regions = hs_api.cache_regions()
    except Exception as exc:
        logger.warning("provision: hyperstack cache_regions failed (skipping): %s", exc)
        return done
    # One PER-REGION volume (Hyperstack names are globally unique — see cache_volume_name), created in
    # that region's default environment.
    for region in regions:
        try:
            env = hs_api.environment_for_region(region)
            vol_name = hs_api.cache_volume_name(base, region)
            vol_id = hs_api.ensure_volume(vol_name, env, gb)
            # ensure_volume returns the volume id; a falsy id means create-or-confirm did NOT yield a
            # real volume (e.g. the API responded without an id). Don't record that region as
            # provisioned — otherwise --provision reports success and the launch path treats a
            # never-created region as warm.
            if not vol_id:
                logger.warning("provision: hyperstack ensure_volume(%s, %s) returned no id — region not "
                               "provisioned", vol_name, region)
                continue
            done.append(f"hyperstack:{region}")
        except Exception as exc:
            logger.warning("provision: hyperstack ensure_volume(%s, %s) failed: %s", base, region, exc)
    return done


def provision_all() -> list[str]:
    """Eagerly create the cache storage on every instance provider, in every region/environment
    (pure control-plane API, no GPU). RunPod's per-DC network volumes are NOT provisioned here: they
    are create-or-attached automatically by the eager endpoint deploy (jobs.weight_cache_volumes
    covers every storage DC) and warmed by ``warm_weight_cache`` — there is no GPU-free RunPod
    volume-create in the SDK. Returns ``provider:<region/env>`` per storage created/confirmed."""
    provisioned = provision_lambda_filesystems()
    provisioned += provision_hyperstack_volumes()
    return provisioned


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
        help="CREATE the Lambda/Hyperstack cache storage in every region/env (pure API, no GPU) and "
             "exit; RunPod volumes are auto-created by the eager deploy/warm. Run before --teardown's "
             "inverse to set up all storage up front.",
    )
    ap.add_argument(
        "--warm-instances", action="store_true",
        help="WARM the Lambda + Hyperstack caches: one download-only GPU launch per region with "
             "capacity now (needs the merged worker image carrying the bootstrap preload branch).",
    )
    ap.add_argument(
        "--teardown", action="store_true",
        help="DELETE the weight-cache storage on every provider (reclaim standing storage) and exit. "
             "With --datacenters it is SCOPED to that RunPod-DC subset only (Lambda/Hyperstack caches "
             "are left intact, since DC ids don't map to their region/env namespace).",
    )
    args = ap.parse_args(argv)

    # The mode flags are MUTUALLY EXCLUSIVE: each selects a different exit-early branch below, and the
    # branch order (provision -> warm-instances -> teardown -> default RunPod warm) silently picks ONE
    # when several are set — e.g. `--teardown --warm-instances` would launch paid warm jobs (the warm
    # branch runs first) instead of deleting caches, AND bypass the off-catalog --models check (the
    # teardown exemption short-circuits it). Reject the conflict up front so the off-catalog gate always
    # applies to whichever warm branch actually executes. The default RunPod warm has no flag, so it's
    # only reachable when NONE of these are set — it can't conflict.
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
    # Confidentiality gate: an explicit --models override may ONLY name public catalog ids on the paths
    # that actually DOWNLOAD weights into the shared cache (the default RunPod warm + --warm-instances).
    # Warming an arbitrary (private/gated) repo with the operator HF_TOKEN would leave those weights on
    # the platform-wide WRITABLE shared cache for every other tenant — bypassing the same catalog gate
    # the normal run path enforces. Reject any off-catalog id BEFORE launching any preload worker.
    # --teardown (only deletes) and --provision (only CREATES empty storage — downloads NOTHING) are
    # both exempt: neither reaches a download path, so an off-catalog id there is harmless.
    if args.models and not args.teardown and not args.provision:
        off_catalog = [m for m in models if m not in set(catalog)]
        if off_catalog:
            print("--models: refusing to preload off-catalog model id(s) into the shared cache: "
                  f"{', '.join(off_catalog)} — only public catalog models may be warmed (private/gated "
                  "repos would leak onto the platform-wide shared volume). They download cold on first "
                  "use instead.")
            return 2
    # Parse --datacenters ONCE. `scoped` means the operator actually narrowed to >=1 real DC id — NOT
    # merely that the flag was present. A flag that parses to NOTHING (e.g. `--datacenters ""`, all
    # whitespace/commas, or an all-invalid list) must be an ERROR, never a silent full teardown: it
    # would otherwise (a) hit teardown_weight_cache's `datacenters or <all>` fallback and delete EVERY
    # RunPod cache, while (b) the present-but-empty flag skipped the instance-provider cleanup.
    # argparse default is None when --datacenters is OMITTED; an empty/whitespace/all-comma STRING is
    # still "provided" (`is not None`) but parses to zero ids. Use `is not None` — NOT truthiness — so
    # `--datacenters ""` is caught too (bool("") is False).
    dcs_given = args.datacenters is not None
    parsed_dcs = (
        [d.strip() for d in args.datacenters.split(",") if d.strip()] if dcs_given else []
    )
    if dcs_given and not parsed_dcs:
        print("--datacenters was given but parsed to no datacenter ids — refusing to run "
              "(an empty scope would delete the WHOLE RunPod fleet); drop --datacenters for a full "
              "teardown, or pass real DC ids.")
        return 2
    scoped = bool(parsed_dcs)  # a real RunPod-DC subset -> RunPod-only scope

    # LAZY default RunPod-DC list. weight_cache_datacenters() imports runpod_flash, so resolving it
    # eagerly here would crash --provision / --warm-instances / --teardown --dry-run on an instance-only
    # control plane (no/broken RunPod SDK) — modes that never touch a RunPod DC, or are non-destructive.
    # Resolve it ONLY inside the branches that actually warm or tear down RunPod without an explicit
    # --datacenters scope. When `scoped`, `parsed_dcs` is used directly and this is never called.
    def _default_dcs() -> list[str]:
        return [dc.value for dc in weight_cache_datacenters()]

    if args.provision:
        # Eagerly create the instance-provider cache storage in every region/env (GPU-free). RunPod's
        # per-DC fleet materializes on the next eager endpoint deploy / warm, so it's not created here.
        if args.dry_run:
            print("would provision Lambda filesystems + Hyperstack volumes in every region/env")
            return 0
        provisioned = provision_all()
        print(f"provisioned {len(provisioned)} instance-provider cache store(s): "
              f"{', '.join(provisioned) or '(none — no Lambda/Hyperstack key, or no regions)'}")
        return 0
    if args.warm_instances:
        if args.dry_run:
            print("would warm Lambda + Hyperstack caches (one download-only launch per region with capacity)")
            return 0
        # gpu=None lets warm_instances apply its own per-mode default (_PRELOAD_INSTANCE_GPU). Passing
        # args.gpu directly (no sentinel comparison) means an explicit --gpu, even RTX 4090, overrides.
        results = warm_instances(models=models, gpu=args.gpu,
                                 timeout_s=args.timeout_s, max_workers=args.max_workers)
        failed = [r for r in results if r.get("status") not in ("ok",)]
        for r in results:
            print(f"  {r['provider']}/{r['region']}: {r['status']}"
                  + (f" ({r.get('error')})" if r.get("error") else ""))
        print(f"{len(results) - len(failed)}/{len(results)} regions warmed")
        return 1 if failed else 0
    if args.teardown:
        # Validate any scoped DC ids BEFORE deleting anything: an invalid id (typo in teardown
        # automation) must fail loudly with a non-zero exit, NOT get swallowed by the best-effort
        # catch below and report success while deleting nothing / leaving the billed fleet in place.
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
            # `--teardown --dry-run` must only PRINT the plan — never call the destructive helpers AND
            # never resolve the full RunPod DC list (weight_cache_datacenters imports runpod_flash):
            # describe the scope abstractly when unscoped so this stays usable on an instance-only host.
            scope_desc = (f"{len(parsed_dcs)} datacenter(s): {', '.join(parsed_dcs)}"
                          if scoped else "every RunPod storage datacenter")
            print(f"would delete the RunPod weight-cache volumes in {scope_desc}"
                  + ("" if scoped else " + every Lambda filesystem + Hyperstack volume named flash-weights"))
            return 0
        # Reclaim the cache storage on EVERY provider: RunPod network volumes, Lambda filesystems,
        # and Hyperstack block volumes. Each provider is guarded INDEPENDENTLY so one provider's
        # failure (e.g. RunPod auth absent/broken on an instance-only control plane, or a RunPod
        # outage) never aborts the others' best-effort cleanup — otherwise their billed caches would
        # leak behind a single RunPod error. A provider with no configured key is already a no-op.
        deleted: list[str] = []
        try:
            # Pass the scoped list, or None for a full teardown — teardown_weight_cache resolves the
            # default DC fleet itself (lazily, and only AFTER it confirms a RunPod key is configured),
            # so an instance-only control plane never imports the RunPod SDK here.
            deleted += teardown_weight_cache(parsed_dcs or None)
        except Exception as exc:
            logger.warning("teardown: RunPod cache teardown failed (continuing): %s", exc)
        # `--datacenters` is a RunPod-DC subset and has no meaning for the instance providers (Lambda
        # regions / Hyperstack envs are a different namespace), so a SCOPED teardown stays RunPod-only
        # rather than unexpectedly deleting every Lambda/Hyperstack `flash-weights` cache too. Only a
        # FULL teardown (no --datacenters) reclaims the instance-provider caches.
        if not scoped:
            try:
                deleted += teardown_lambda_filesystems()
            except Exception as exc:
                logger.warning("teardown: Lambda cache teardown failed (continuing): %s", exc)
            try:
                deleted += teardown_hyperstack_volumes()
            except Exception as exc:
                logger.warning("teardown: Hyperstack cache teardown failed (continuing): %s", exc)
        else:
            print("scoped teardown (--datacenters): RunPod-only; Lambda/Hyperstack caches left intact")
        print(f"deleted {len(deleted)} weight-cache volume(s): {', '.join(deleted) or '(none)'}")
        return 0
    # Default mode = warm the RunPod serverless fleet. This is the ONE path that genuinely needs the
    # RunPod DC list (and the SDK), so resolve the lazy default here rather than eagerly above.
    dcs = parsed_dcs or _default_dcs()
    if args.dry_run:
        print(f"would warm {len(dcs)} datacenter(s): {', '.join(dcs)}")
        print(f"with {len(models)} model(s): {', '.join(models)}")
        return 0

    results = warm_weight_cache(
        # args.gpu defaults to None -> fall back to the RunPod warm default here so None never reaches
        # _preload_one_dc / deploy_train_endpoint; an explicit --gpu (incl. RTX 4090) still overrides.
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
