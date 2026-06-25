"""RunPod Flash endpoint lifecycle: provision/cache/teardown + the worker handler.

The live ("ad-hoc") endpoint deploy/cache (``get_train_endpoint``), the worker-side
training handler it registers (``_train_body``), the per-run endpoint naming/suffix,
the runpod_flash SDK state-isolation + backoff patches, and cross-process teardown
(``terminate_endpoint``). Imports the dependency stack + worker env builder from
``.deps`` (the leaf).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading
import time
from typing import Any

from flash.providers.base import canonical_gpu, gpu_short
from flash.providers.runpod.gpus import flash_gpu
from flash.providers.runpod.train.deps import (
    DEFAULT_EXECUTION_TIMEOUT_MS,
    WORKER_SYSTEM_DEPS,
    logger,
    resolve_worker_deps,
)

# The control plane runs each training run in its own thread. All runpod_flash deploy/
# undeploy work goes through a shared asyncio singleton whose Lock binds to the first event
# loop that touches it; two threads each calling asyncio.run() get distinct loops and the
# second fails with "Lock ... is bound to a different event loop". Serialize every Flash SDK
# async section (deploy AND undeploy) behind this one process-wide lock. Deploys/teardowns
# are infrequent vs training, so the serialization cost is negligible.
FLASH_SDK_LOCK = threading.Lock()

# Quota: cap in-flight endpoints under RunPod's account-wide max-workers quota (30; the cap of
# 28 leaves a 2-slot buffer for endpoints deployed by other tools/the CLI). Every endpoint this
# process creates claims one slot; terminate_endpoint releases it once the remote endpoint is
# provably torn down. Runs that find the quota full QUEUE (block) here instead of failing with
# RunPod's "Max workers across all endpoints must not exceed your workers quota (30)" error.
#
# The store is CROSS-PROCESS when an operator internal key is configured: the claim is an
# advisory-locked atomic op in Postgres (via the freesolo backend), so >1 control-plane replica
# can never together exceed the cap, and a startup reconcile recovers the true in-use count after
# a crash. Without an internal key (local/dev single process) we fall back to an in-process
# semaphore. Releases route by how the slot was acquired (recorded in ``_ACQUIRED``), so a slot
# claimed against the shared store is always released there.
RUNPOD_ENDPOINT_SLOT_CAP = 28
# How long to wait before re-checking a full shared quota (queue-don't-crash), and how many
# consecutive store errors to tolerate before falling back to the local semaphore so a single
# process stays capped rather than deadlocking on a persistently-unreachable backend.
_SLOT_QUEUE_WAIT_S = 10.0
_SLOT_STORE_MAX_ERRORS = 6

# Local (single-process) fallback, used when no internal key is configured or the shared store is
# persistently unreachable. Resets on restart — fine, because a fresh process holds no slots.
_LOCAL_SLOTS = threading.Semaphore(RUNPOD_ENDPOINT_SLOT_CAP)
# name -> "shared" | "local": how this process acquired each held slot, so release routes to the
# same store. Makes release idempotent — terminate_endpoint may run more than once for a name
# (e.g. a retry after a failed undeploy), and only the call that still finds the name releases.
_ACQUIRED: dict[str, str] = {}
_ACQUIRED_LOCK = threading.Lock()

_ENDPOINT_CACHE: dict[str, Any] = {}


def _acquire_local_slot(name: str) -> None:
    """Claim an in-process semaphore slot (no internal key / store-unreachable fallback)."""
    if not _LOCAL_SLOTS.acquire(blocking=False):
        logger.info(
            "Quota full (%d/%d slots) — waiting for a free slot...",
            RUNPOD_ENDPOINT_SLOT_CAP,
            RUNPOD_ENDPOINT_SLOT_CAP,
        )
        _LOCAL_SLOTS.acquire()
    with _ACQUIRED_LOCK:
        _ACQUIRED[name] = "local"


def _acquire_endpoint_slot(name: str) -> None:
    """Claim a quota slot for ``name``, QUEUEING (blocking) until one is free. Idempotent per name.

    Uses the shared cross-process store when an internal key is set, else the in-process
    semaphore. On persistent store errors it falls back to the local semaphore so a single
    process stays capped instead of deadlocking or oversubscribing.
    """
    with _ACQUIRED_LOCK:
        if name in _ACQUIRED:
            return  # already hold a slot for this name (e.g. _ENDPOINT_CACHE re-entry)
    from flash.providers.runpod import slots

    if slots.internal_key() is None:
        _acquire_local_slot(name)
        return

    errors = 0
    queued = False
    while True:
        try:
            claimed, in_use = slots.claim(
                name, cap=RUNPOD_ENDPOINT_SLOT_CAP, claimed_by=slots.claimed_by_ident()
            )
        except slots.SlotStoreError as exc:
            errors += 1
            logger.warning(
                "slot-store claim failed for %s (%s) [%d/%d]",
                name,
                exc,
                errors,
                _SLOT_STORE_MAX_ERRORS,
            )
            if errors >= _SLOT_STORE_MAX_ERRORS:
                logger.error(
                    "slot store unreachable; falling back to the in-process cap for %s", name
                )
                _acquire_local_slot(name)
                return
            time.sleep(_SLOT_QUEUE_WAIT_S)
            continue
        errors = 0
        if claimed:
            if queued:
                logger.info(
                    "RunPod endpoint slot acquired for %s after queueing (%d/%d in use)",
                    name,
                    in_use,
                    RUNPOD_ENDPOINT_SLOT_CAP,
                )
            with _ACQUIRED_LOCK:
                _ACQUIRED[name] = "shared"
            return
        if not queued:
            logger.info(
                "RunPod quota full (%d/%d) — queueing for a free slot...",
                in_use,
                RUNPOD_ENDPOINT_SLOT_CAP,
            )
            queued = True
        time.sleep(_SLOT_QUEUE_WAIT_S)


def _release_endpoint_slot(name: str) -> bool:
    """Release the quota slot for ``name``, routed to the store it was claimed from.

    The ``_ACQUIRED`` map records how this process claimed the slot (``"local"`` | ``"shared"``)
    and makes the common path idempotent — only the first call for a name this process claimed
    releases. When a shared store is configured, a teardown running on a DIFFERENT replica than
    the one that claimed the slot (e.g. a ``flash cancel`` routed to another control-plane
    replica) has no ``_ACQUIRED`` entry but must STILL drop the shared row, or the slot leaks
    until the next reconcile and the queue is throttled even though the endpoint is gone. The
    callers only invoke this once the remote endpoint is provably gone, and server-side release
    is idempotent, so the best-effort cross-replica release is safe.

    Returns ``True`` if a slot was (or, on the shared path, may have been) released; ``False`` for
    a no-op (nothing tracked locally and no shared store to release against).
    """
    with _ACQUIRED_LOCK:
        mode = _ACQUIRED.pop(name, None)
    if mode == "local":
        _LOCAL_SLOTS.release()
        return True
    from flash.providers.runpod import slots

    # mode == "shared": this process holds the row. mode is None: either a cross-replica teardown
    # (release the shared row anyway) or — with no shared store at all — genuinely nothing to do.
    cross_replica = mode is None
    if cross_replica and slots.internal_key() is None:
        return False

    try:
        released = slots.release(name)
    except slots.SlotStoreError as exc:
        # A transient release failure can't leak the slot permanently: the endpoint is gone, so
        # the next startup reconcile reclaims its row against the live endpoint list.
        logger.warning(
            "slot-store release failed for %s (%s); reconcile will reclaim it on restart",
            name,
            exc,
        )
        return not cross_replica
    return released if cross_replica else True


def reconcile_endpoint_slots() -> None:
    """Reconcile the shared slot store against the live RunPod endpoint list (control-plane
    startup), so a crash can't leak slots: rows for endpoints that no longer exist are reclaimed
    and the in-use count recovers to the truth. No-op without an internal key (the local
    semaphore needs no reconciliation — a fresh process starts empty). Best-effort: never raises.
    """
    from flash.providers.runpod import slots

    if slots.internal_key() is None:
        return
    try:
        from flash.providers.runpod import api as runpod_api

        live = [
            name
            for e in runpod_api.list_endpoints()
            if (name := (e.get("name") or "")).startswith("flash-")
        ]
    except Exception as exc:  # listing failed: do NOT reconcile against a partial/empty list
        logger.warning("slot reconcile skipped: could not list RunPod endpoints (%s)", exc)
        return
    try:
        result = slots.reconcile(live)
        logger.info(
            "RunPod slot reconcile: %s in use, %s reclaimed",
            result.get("inUse"),
            result.get("reclaimed"),
        )
    except slots.SlotStoreError as exc:
        logger.warning("slot reconcile failed (%s)", exc)


def _train_body(input_data: dict) -> dict:
    """Runs ON the RunPod GPU worker: fetch code, train (phase), return metrics.

    NOTE: Flash serializes this handler and runs it standalone, so every name it uses
    must be imported INSIDE the function body (module-level imports are not in scope).
    """
    import contextlib
    import json
    import os
    import subprocess
    import sys

    from huggingface_hub import snapshot_download

    # Weight-cache PRELOAD mode: download-only, no code repo / no training subprocess. Runs on a
    # worker whose `flash-weights` volume for ONE datacenter is mounted at /runpod-volume; with
    # HF_HOME pointed there (passed in env), each snapshot_download warms that region's cache so the
    # first real run in that region is a cache hit. Returns the repos it cached. Kept here (in the
    # baked handler) so preload reuses the existing image/handler — no separate worker image.
    if input_data.get("mode") == "preload":
        overrides = {k: str(v) for k, v in (input_data.get("env") or {}).items()}
        os.environ.update(overrides)
        # NB: HF_HUB_ENABLE_HF_TRANSFER is NOT set here — the worker image already exports it
        # (Dockerfile.worker ENV), same as the training path relies on (see deps.py).
        tok = overrides.get("HF_TOKEN")
        # CRITICAL: huggingface_hub froze HF_HUB_CACHE from HF_HOME at IMPORT time (the
        # `from huggingface_hub import snapshot_download` above, before this branch), so the
        # HF_HOME we just set in os.environ is ignored by snapshot_download. Pass cache_dir
        # EXPLICITLY = <HF_HOME>/hub (HF's own layout) so the download lands on the mounted volume
        # instead of the worker's ephemeral default cache. (The training path is immune: it spawns a
        # subprocess that imports huggingface_hub fresh with HF_HOME already in its env.)
        hf_home = overrides.get("HF_HOME")
        # The whole point of preload is to write onto the per-region network volume. If it isn't
        # actually mounted (e.g. the endpoint was deployed without the volume / RunPod didn't mount
        # it), downloading would silently warm the worker's EPHEMERAL disk and report success while
        # warming nothing. Fail loudly instead so the driver records this region as failed.
        if hf_home and hf_home.startswith("/runpod-volume") and not os.path.isdir("/runpod-volume"):
            return {
                "preloaded": [], "already_cached": [], "failed": {},
                "error": f"weight-cache volume not mounted at /runpod-volume (HF_HOME={hf_home})",
                "hf_home": hf_home,
            }
        cache_dir = os.path.join(hf_home, "hub") if hf_home else None
        # Same exclusions as the worker prefetch (engine/worker.prefetch_model), the image bake, and
        # the instance-provider preload (_instance_bootstrap.run_preload): weights + tokenizer/config
        # only, never the large unused artifacts. Inlined (this handler is baked self-contained, so it
        # can't import a shared constant). Keeps the warmed cache byte-for-byte what workers fetch,
        # so the 100GB sizing holds and the local_files_only `already_cached` probe stays consistent.
        ignore_patterns = ["*.pth", "*.gguf", "original/*", "*.onnx", "*.msgpack", "*.h5"]
        done, already, failed = [], [], {}
        for repo_id in input_data.get("models") or []:
            try:
                # Idempotent: if the volume already has this snapshot, skip the download. The
                # local_files_only probe is also the persistence signal the live test reads —
                # ``already_cached`` proves the weights survived a previous, separate deployment.
                try:
                    snapshot_download(
                        repo_id=repo_id, token=tok, cache_dir=cache_dir,
                        ignore_patterns=ignore_patterns, local_files_only=True,
                    )
                    already.append(repo_id)
                    done.append(repo_id)
                    continue
                except Exception:
                    pass
                snapshot_download(
                    repo_id=repo_id, token=tok, cache_dir=cache_dir, ignore_patterns=ignore_patterns
                )
                done.append(repo_id)
            except Exception as exc:  # one bad/gated repo must not abort warming the rest
                failed[repo_id] = str(exc)[:300]
        return {
            "preloaded": done,
            "already_cached": already,
            "failed": failed,
            "hf_home": os.environ.get("HF_HOME"),
        }

    # NB: the Hopper fla fast-path setup lives in flash.engine.worker._ensure_fla_fastpath_on_hopper (runs
    # in the worker process AFTER all installs, before any model import) — doing it here would be
    # undone by a later extra_pip / `prime env install`, and depends on a handler redeploy.

    # Extra pip deps for Freesolo environments (installed per-run).
    extra_pip = input_data.get("extra_pip") or []
    if extra_pip:
        # check=True: a deterministic dependency failure should fail fast here,
        # not after model download + worker startup with a less actionable error.
        subprocess.run([sys.executable, "-m", "pip", "install", *extra_pip], check=True)

    # NB: fla is kept on ALL arches. On Hopper (sm90) fla's GDN backward is miscomputed with
    # Triton>=3.4 (#640); the fix is fla's tilelang backend, so flash.engine.worker._ensure_fla_fastpath_on_hopper
    # makes fla+tilelang live at worker startup (instead of dropping fla) for ~4-13x faster + ~2x
    # lighter Hopper GDN training than the pure-PyTorch delta fallback.

    overrides = {k: str(v) for k, v in (input_data.get("env") or {}).items()}
    snapshot_download(
        repo_id=input_data["hf_repo"],
        repo_type="dataset",
        allow_patterns=["code/**"],
        local_dir="/runcode",
        token=overrides.get("HF_TOKEN"),
    )
    code_dir = "/runcode/code"

    env = dict(os.environ)
    env.update(overrides)
    # If the weight-cache volume isn't actually mounted (cold/no-volume run, or the attach degraded
    # to {}), don't leave HF_HOME pointing at a missing /runpod-volume path — fall back to the
    # default ephemeral cache. INLINED (not a flash import): this handler is extracted standalone
    # into the baked image's rp_handler (docker/make_rp_handler.py), where flash is NOT importable,
    # so it must stay self-contained. Mirrors deps.drop_unmounted_cache_env (unit-tested there).
    if not os.path.isdir("/runpod-volume"):
        for _k in [k for k, v in env.items() if str(v).startswith("/runpod-volume")]:
            env.pop(_k, None)
    # Always pass the spec via a file (FLASH_JOB_SPEC_PATH): a large inline spec can blow past the
    # ~128 KiB per-env-string exec limit ("Argument list too long"), and a file is ONE code path for
    # every size (cheap write). load_job_spec_from_env reads it.
    spec_path = "/tmp/job_spec.json"
    with open(spec_path, "w") as sf:
        sf.write(input_data["job_spec_json"])
    env["FLASH_JOB_SPEC_PATH"] = spec_path
    env.pop("FLASH_JOB_SPEC_JSON", None)
    env["PHASE"] = input_data["phase"]
    env["SEED"] = str(input_data["seed"])
    env["PYTHONPATH"] = code_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    def _upload_console(mode: str) -> None:
        """Upload the captured console tail for ``mode`` to ``{phase_ns}/{run_id}/seed{n}/
        console_<mode>.txt`` in the run repo. Idempotent and best-effort, so it is safe to call
        from both the subprocess-failure path and the missing-metrics crash path: a worker killed
        without a Python exception (OOM/SIGKILL, segfault, or a silent early exit) writes NO
        ``error_<mode>.txt``, so the captured console is then the only root-cause record — and a
        crash that exits 0 would otherwise skip the upload entirely, leaving the failure opaque."""
        console = f"/tmp/console_{mode}.txt"
        if not os.path.exists(console):
            return
        try:
            from huggingface_hub import HfApi

            spec = json.loads(input_data["job_spec_json"])
            phase_ns = "rl" if spec.get("algorithm") == "grpo" else spec["algorithm"]
            prefix = f"{phase_ns}/{spec['run_id']}/seed{input_data['seed']}"
            # Read only the last 64 KB (seek from the end) — the console can be very large on long
            # runs, so f.read()[-64_000:] would pull the whole file into memory just to slice it.
            tail_bytes = 64_000
            with open(console, "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - tail_bytes))
                tail = f.read().decode("utf-8", "replace")
            # utf-8 + replace so a non-ASCII console tail can't raise UnicodeEncodeError under a
            # minimal-container ASCII locale (LANG=C); the tail itself was decoded utf-8/replace.
            with open(console + ".tail", "w", encoding="utf-8", errors="replace") as f:
                f.write(tail)
            HfApi(token=env.get("HF_TOKEN")).upload_file(
                path_or_fileobj=console + ".tail",
                path_in_repo=f"{prefix}/console_{mode}.txt",
                repo_id=input_data["hf_repo"],
                repo_type="dataset",
            )
        except Exception as up_err:
            print("console upload warn:", up_err)

    def run_mode(mode: str, check: bool) -> int:
        """Run one worker process, tee its console to a file, and upload the tail to HF as
        console_<mode>.txt on failure — the engine-core root cause of crashes like vLLM
        EngineDeadError only ever appears on the subprocess console, never in the Python
        traceback. With FLASH_UPLOAD_CONSOLE=1 (forwarded via build_worker_env) the console
        is also uploaded on SUCCESS, so an operator can verify which optimizations engaged."""
        console = f"/tmp/console_{mode}.txt"
        with open(console, "w") as cf:
            proc = subprocess.Popen(
                [sys.executable, "-m", "flash.engine.worker"],
                cwd=code_dir,
                env={**env, "RUN_MODE": mode},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in proc.stdout:
                print(line, end="")  # keep streaming to the platform console
                cf.write(line)
            proc.wait()
        # Console is uploaded on FAILURE (crash root-cause). FLASH_UPLOAD_CONSOLE=1 also uploads it
        # on SUCCESS so an operator can verify which optimizations engaged — LoRA+/8-bit-AdamW/
        # Liger/PiSSA/rsLoRA/fla/chalk all log their engagement (or fallback) to the console.
        _force_console = env.get("FLASH_UPLOAD_CONSOLE", "").strip().lower() not in (
            "", "0", "false", "no", "off",
        )
        if proc.returncode != 0 or _force_console:
            _upload_console(mode)
        if proc.returncode != 0 and check:
            raise RuntimeError(
                f"worker mode '{mode}' exited {proc.returncode}; see console_{mode}.txt "
                f"and error_{mode}.txt in the HF dataset repo"
            )
        return proc.returncode

    # A warm worker can carry a previous seed's metrics files; a stale metrics.json
    # would let a crashed train phase report the previous run's numbers. Clear before
    # training.
    for stale in ("/tmp/train_meta.json", "/tmp/metrics.json"):
        with contextlib.suppress(FileNotFoundError):
            os.remove(stale)
    # Train. check=False — RL's colocated vLLM can segfault at interpreter exit AFTER
    # the adapter + metrics.json + DONE are saved; don't treat that as a failure.
    run_mode(input_data["phase"], check=False)
    # The train phase writes metrics.json + the DONE sentinel itself (RunPod can also
    # redeliver a completed job, whose worker restores metrics.json from DONE). If it
    # is missing, the train phase crashed before finishing — fail fast with the real
    # cause (full traceback in error_<phase>.txt / console_<phase>.txt in the HF repo).
    if not os.path.exists("/tmp/metrics.json"):
        phase = input_data["phase"]
        # run_mode skips the console upload when the worker exits 0 (and a hard OOM/segfault kill
        # may have raced it), so force it here — otherwise this exact "crashed before finishing"
        # failure is undebuggable: no metrics.json, often no error_<phase>.txt, and the message
        # below points operators at a console_<phase>.txt that was never uploaded.
        _upload_console(phase)
        raise RuntimeError(
            f"train phase '{phase}' produced no /tmp/metrics.json (it crashed before "
            f"finishing); see error_{phase}.txt and console_{phase}.txt in the HF "
            f"dataset repo for the full traceback"
        )
    with open("/tmp/metrics.json") as f:
        return json.load(f)


def isolate_flash_state(scope: str | None = None) -> None:
    """Point the Flash SDK's resource registry at a per-process/private directory.

    The SDK persists its registry to ``./.flash/resources.pkl`` — shared, whole-dict,
    last-writer-wins across every process in the CWD. Observed failure modes: stale
    entries resurrecting long-dead endpoints on later syncs, and concurrent processes
    clobbering each other's bookkeeping. Each Flash process gets its own registry
    under ``~/.flash/flash-state/<scope>``; remote cleanup never relies on the
    registry anyway (REST by id/name — see api.py).
    """
    try:
        from pathlib import Path

        import runpod_flash.core.resources.resource_manager as rm

        scope = scope or f"pid{os.getpid()}"
        state_dir = Path.home() / ".flash" / "flash-state" / scope
        state_dir.mkdir(parents=True, exist_ok=True)
        rm.FLASH_STATE_DIR = state_dir
        rm.RESOURCE_STATE_FILE = state_dir / "resources.pkl"
    except Exception as exc:  # never block a run on this
        logger.warning("flash state isolation skipped: %s", exc)


def _patch_runpod_backoff() -> None:
    """Work around a runpod_flash bug that aborts long-running jobs.

    The SDK polls a synchronous job with exponential backoff computed as
    ``base * (2 ** attempt)`` and only clamps to ``max_seconds`` afterwards. On a long
    run the poll ``attempt`` grows without bound, so ``2 ** attempt`` becomes a huge int
    and the float multiply raises ``OverflowError: int too large to convert to float``
    (observed ~80 min in), killing an otherwise-healthy job mid-run. We patch the symbol
    to cap the exponent before the power so the delay still saturates at ``max_seconds``.
    """
    try:
        import math
        import random

        from runpod_flash.core.utils import backoff as _bo

        if getattr(_bo, "_flash_backoff_patched", False):
            return

        def _safe_get_backoff_delay(
            attempt,
            base=0.1,
            max_seconds=10.0,
            jitter=0.2,
            strategy=_bo.BackoffStrategy.EXPONENTIAL,
        ):
            a = min(int(attempt), 30)  # cap exponent: 2**30 is plenty; delay saturates anyway
            if strategy == _bo.BackoffStrategy.EXPONENTIAL:
                delay = base * (2**a)
            elif strategy == _bo.BackoffStrategy.LINEAR:
                delay = base + (attempt * base)
            elif strategy == _bo.BackoffStrategy.LOGARITHMIC:
                delay = base * math.log2(attempt + 2)
            else:
                raise ValueError(f"Unsupported backoff strategy: {strategy}")
            delay = min(delay, max_seconds)
            return delay * random.uniform(1 - jitter, 1 + jitter)

        _bo.get_backoff_delay = _safe_get_backoff_delay
        _bo._flash_backoff_patched = True
        # serverless.py did `from ..utils.backoff import get_backoff_delay`, so patch its ref too.
        try:
            from runpod_flash.core.resources import serverless as _sl

            _sl.get_backoff_delay = _safe_get_backoff_delay
        except Exception:
            # serverless.py may not import the symbol in this SDK version; the primary
            # patch above still applies, so a missing alias is fine to ignore.
            pass
    except Exception as exc:  # never let the patch break submission
        logger.warning("runpod backoff patch skipped: %s", exc)


def min_cuda_for(friendly_gpu: str) -> str:
    """Minimum host CUDA (driver) version for this GPU class on the active stack.

    Blackwell classes (sm_120 — RTX 5090, RTX Pro 6000): pypi wheels for
    the modern stack (vllm 0.19) ship no Blackwell SASS, so every custom CUDA kernel
    is PTX-JIT'd by the driver — and their PTX is built with a newer toolchain than
    CUDA-12.8-era drivers can JIT (observed: "the provided PTX was compiled with an
    unsupported toolchain" on driver 570.x). CUDA-13 drivers JIT it fine, so those
    classes are pinned to >=13.0 on the modern stack (per-GPU ``min_cuda_modern`` in
    providers.base.GPU_INFO). Ampere/Ada/Hopper have SASS in the wheels and run on 12.8.
    Fully managed per-GPU (no override).
    """
    from flash.providers.base import min_cuda_modern

    return min_cuda_modern(friendly_gpu)


def endpoint_name(friendly_gpu: str, suffix: str | None = None) -> str:
    """Flash endpoint/template name for a GPU class, optionally made unique per run.

    A fixed name (``flash-5090``) collides across back-to-back runs: runpod_flash's
    ``get_or_deploy_resource`` finds the prior run's still-registered resource and tries to
    *update* it, which fails with ``GraphQL errors: Template name must be unique`` (there is
    no endpoint GC/reuse). A per-run ``suffix`` (the run id tail) gives each run its own
    endpoint so it deploys fresh instead of colliding. RunPod scales each to zero when idle.
    """
    base = f"flash-{gpu_short(friendly_gpu)}"
    if not suffix:
        return base
    safe = "".join(c for c in str(suffix) if c.isalnum() or c == "-").strip("-")[:24]
    return f"{base}-{safe}" if safe else base


def get_train_endpoint(
    friendly_gpu: str,
    execution_timeout_ms: int | None = None,
    name_suffix: str | None = None,
    disk_gb: int | None = None,
    spec=None,
):
    """Build (and cache) the live Flash endpoint handler for a GPU class."""
    # Live ("ad-hoc") provisioning: provision on call, no separate `flash deploy`.
    os.environ["FLASH_IS_LIVE_PROVISIONING"] = "true"
    from runpod_flash import Endpoint

    from flash.providers.runpod.auth import ensure_auth

    ensure_auth()
    _patch_runpod_backoff()
    isolate_flash_state(name_suffix)

    friendly = canonical_gpu(friendly_gpu)
    name = endpoint_name(friendly, name_suffix)
    if name in _ENDPOINT_CACHE:
        # Slot was already acquired when the entry was first created; don't re-acquire.
        return _ENDPOINT_CACHE[name]
    # Claim a quota slot before creating the endpoint. This QUEUES (blocks) when the quota is
    # full, making new runs wait here instead of failing with RunPod's worker-quota error. The
    # slot is released by terminate_endpoint once the remote endpoint is provably torn down.
    _acquire_endpoint_slot(name)
    try:
        kwargs = {
            "name": name,
            "gpu": flash_gpu(friendly),
            "gpu_count": 1,
            "min_cuda_version": min_cuda_for(friendly),
            "execution_timeout_ms": execution_timeout_ms or DEFAULT_EXECUTION_TIMEOUT_MS,
            "workers": (0, 1),  # one dedicated worker per run; scale to zero when idle
        }
        # RunPod Flash needs its serverless runtime baked into the worker image. Boot-install
        # WORKER_DEPS on Flash's default template instead (cached as a Flash artifact). Optional
        # FLASH_WORKER_IMAGE override for a RunPod-serverless-compatible image.
        image = os.environ.get("FLASH_WORKER_IMAGE")
        if image:
            kwargs["image"] = image
        else:
            kwargs["dependencies"] = resolve_worker_deps()
            kwargs["system_dependencies"] = WORKER_SYSTEM_DEPS
        # Parity with the baked-image deploy_train_endpoint path: attach the multi-region weight
        # cache (best-effort {} on no-cache/error). Local import avoids a jobs<->endpoints import
        # cycle (jobs imports this module at load), same as apply_disk_gb below.
        from flash.providers.runpod.jobs import weight_cache_endpoint_kwargs

        kwargs.update(weight_cache_endpoint_kwargs(spec))
        ep = Endpoint(**kwargs)
        handler = ep(_train_body)  # register the queue-based handler; returns the callable
        # The resource config is cached on the Endpoint, so raising the disk on it here
        # carries through to the deploy that the first handler call triggers.
        from flash.providers.runpod.jobs import apply_disk_gb

        cfg = ep._build_resource_config()
        apply_disk_gb(cfg, disk_gb)
        _ENDPOINT_CACHE[name] = handler
        return handler
    except Exception:
        # Endpoint creation failed — release the slot so other runs are not permanently blocked.
        _release_endpoint_slot(name)
        raise


def _run_suffix(run_id: str | None) -> str | None:
    """Short, COLLISION-FREE per-run endpoint suffix.

    Must be unique per run_id: the endpoint name is ``endpoint_name(friendly, suffix)`` and
    RunPod reuses an endpoint by name -- two runs with the same suffix share one endpoint (and
    its cached image/deps/registry-auth/template), so a later run silently reuses the earlier
    one's config. The old ``run_id.split("-")[-1]`` only worked for hash-tailed default ids; a
    descriptive run_id ending in e.g. the card name (``...-a100``) collided across every run.
    Use a stable short hash of the WHOLE run_id, with a sanitized prefix for readability."""
    if not run_id:
        return None
    import hashlib
    import re

    h = hashlib.sha1(run_id.encode()).hexdigest()[:8]
    prefix = re.sub(r"[^a-z0-9]", "", run_id.lower())[-12:]
    return f"{prefix}{h}" if prefix else h


def stop_endpoint(friendly_gpu: str, name: str | None = None) -> None:
    """Best-effort: scale cached endpoint(s) to zero / drop them.

    With ``name`` only that run's cached endpoint is dropped; without it, every
    cached endpoint of the GPU class is — so a per-run teardown passes ``name``
    to avoid evicting a concurrent run's handler in the same process.

    NOTE: this only touches THIS process's in-memory cache, so it does nothing in a fresh
    ``flash cancel`` process. Use ``terminate_endpoint`` to actually delete the remote endpoint.
    """
    friendly = canonical_gpu(friendly_gpu)
    prefix = f"flash-{gpu_short(friendly)}"
    if name:
        match = [k for k in _ENDPOINT_CACHE if k == name]
    else:
        match = [k for k in _ENDPOINT_CACHE if k.startswith(prefix)]
    for key in match:
        handler = _ENDPOINT_CACHE.pop(key, None)
        ep = getattr(handler, "__self__", None) or getattr(handler, "endpoint", None)
        for meth in ("scale_to_zero", "stop", "delete"):
            fn = getattr(ep, meth, None)
            if callable(fn):
                try:
                    fn()
                    break
                except Exception:
                    continue


def _select_endpoint_resources(resources: dict, target: str) -> list[str]:
    """Resource ids whose resource ``.name`` contains ``target``.

    The live-provisioned resource is named ``live-<endpoint_name>``, so we match by substring
    to catch the prefix. ``target`` is the endpoint name (``flash-<gpu>[-<run>]``).
    """
    if not target:
        return []
    out = []
    for uid, res in (resources or {}).items():
        name = str(getattr(res, "name", "") or "")
        if target in name:
            out.append(uid)
    return out


def terminate_endpoint(friendly_gpu: str, run_id: str | None = None) -> list[dict]:
    """Reliably tear down the remote Flash endpoint(s) for a run — cross-process.

    Unlike ``stop_endpoint`` (which only touches this process's in-memory cache), this looks
    the endpoint up by name in runpod_flash's *persisted* resource registry and deletes it via
    the RunPod API (``ResourceManager.undeploy_resource`` -> ``delete_endpoint``), which stops
    any running worker. Best-effort: never raises. Returns the per-resource undeploy results.

    With ``run_id`` it targets exactly that run's uniquely-named endpoint; without it, the
    bare ``flash-<gpu>`` prefix matches every endpoint of that GPU class.
    """
    friendly = canonical_gpu(friendly_gpu)
    target = endpoint_name(friendly, _run_suffix(run_id))
    # Hold FLASH_SDK_LOCK across the ENTIRE Flash critical section, not just the undeploy.
    # isolate_flash_state() swaps runpod_flash's process-wide registry globals and
    # ResourceManager shares the SDK's asyncio singleton, so a concurrent deploy/undeploy on
    # another thread could swap the registry scope between our lookup and our undeploy and tear
    # down the wrong run's resources. Serialize isolation + lookup + undeploy together.
    with FLASH_SDK_LOCK:
        try:
            from flash.providers.runpod.auth import ensure_auth

            ensure_auth()
            isolate_flash_state(_run_suffix(run_id))
            from runpod_flash.core.resources.resource_manager import ResourceManager
        except Exception as exc:  # SDK/auth unavailable
            return [{"success": False, "name": target, "message": f"flash unavailable: {exc}"}]

        try:
            rm = ResourceManager()
            resources = rm.list_all_resources()
            uids = _select_endpoint_resources(resources, target)
        except Exception as exc:
            return [{"success": False, "name": target, "message": f"resource lookup failed: {exc}"}]

        async def _undeploy_all() -> list:
            out = []
            for uid in uids:
                res = resources.get(uid)
                name = getattr(res, "name", None)
                try:
                    out.append(
                        await rm.undeploy_resource(uid, resource_name=name, force_remove=True)
                    )
                except Exception as exc:
                    out.append({"success": False, "name": name, "message": str(exc)})
            return out

        try:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                # No running event loop — asyncio.run() works directly.
                results = asyncio.run(_undeploy_all())
            else:
                # Running event loop (e.g. FastAPI lifespan) — asyncio.run() would raise;
                # daemon=True so a hung undeploy cannot prevent process shutdown.
                _out: list = []
                _err: list = []

                def _run_undeploy() -> None:
                    try:
                        _out.append(asyncio.run(_undeploy_all()))
                    except Exception as _e:
                        _err.append(_e)

                _t = threading.Thread(target=_run_undeploy, daemon=True)
                _t.start()
                _t.join(timeout=30)
                if _err:
                    raise _err[0]
                if not _out:
                    raise TimeoutError("undeploy timed out after 30s")
                results = _out[0]
        except Exception as exc:
            results = [{"success": False, "name": target, "message": str(exc)}]

    # Registry-less fallback: isolate_flash_state() keeps the Flash SDK's resource
    # registry per-process under ~/.flash, so a recreated container (or a crash before
    # on_handle() persisted the endpoint id) leaves the live endpoint invisible to the
    # lookup above. Delete it via the RunPod REST API by its reconstructed name so it
    # can't keep a paid worker alive.  ``rest_confirmed_absent`` records that the REST
    # lookup actually RAN and found no endpoint of this name — i.e. we positively verified
    # there is nothing to tear down (distinct from the API being unreachable, where we
    # cannot tell and must keep holding the slot).
    rest_confirmed_absent = False
    if not uids:
        try:
            from flash.providers.runpod import api as runpod_api

            matches = [
                e for e in runpod_api.find_endpoints_by_name(target) if e.get("name") == target
            ]
            for ep in matches:
                if runpod_api.delete_endpoint(ep["id"]):
                    results.append(
                        {"success": True, "name": target, "message": "deleted via REST API"}
                    )
            # No endpoint of this name exists remotely — positively verified absent.
            rest_confirmed_absent = not matches
        except Exception as exc:
            # REST API unreachable: we cannot prove the endpoint is gone, so do NOT treat this
            # as "absent" — releasing on an unverified absence risks oversubscribing the quota.
            logger.debug("REST endpoint lookup failed for %s: %s", target, exc)

    # Release the quota slot for this run's endpoint.  Releases the slot this process acquired,
    # or — when a shared store is configured — the row a different replica claimed (a ``flash
    # cancel`` may run on another replica than the one that deployed).  Release ONLY when
    # the remote endpoint is provably gone: (a) at least one undeploy/delete succeeded, or
    # (b) we positively verified no remote endpoint exists (registry returned no uids AND the
    # REST lookup confirmed none — e.g. the endpoint never finished deploying, so its slot
    # would otherwise leak forever and deadlock the queue).  We deliberately do NOT release on
    # an undeploy/delete *failure*: the endpoint may still be live and counting against the
    # RunPod quota, so releasing would oversubscribe it — the slot stays held until a later
    # teardown confirms the endpoint is gone.  (stop_endpoint no longer releases the slot.)
    if any(r.get("success") for r in results) or (not uids and rest_confirmed_absent):
        _release_endpoint_slot(target)

    # also drop the in-process cached handler for THIS run only (a class-wide
    # drop would evict a concurrent run's endpoint on the same GPU class).
    with contextlib.suppress(Exception):
        stop_endpoint(friendly, name=target)
    return results
