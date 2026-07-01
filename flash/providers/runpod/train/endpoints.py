"""RunPod Flash endpoint lifecycle: provision/cache/teardown + the worker handler."""

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
    worker_image_for_gpu,
)

# runpod_flash asyncio singleton is bound to one event loop; serialize all deploy/undeploy.
FLASH_SDK_LOCK = threading.Lock()

# 28 leaves a 2-slot buffer under RunPod's 30-worker account quota. Shared via Postgres when an
# internal key is set; falls back to in-process semaphore otherwise. Releases only after the
# remote endpoint is provably gone.
RUNPOD_ENDPOINT_SLOT_CAP = 28
_SLOT_QUEUE_WAIT_S = 10.0
_SLOT_STORE_MAX_ERRORS = 6
_CONSOLE_UPLOAD_INTERVAL_S = 3600.0

_LOCAL_SLOTS = threading.Semaphore(RUNPOD_ENDPOINT_SLOT_CAP)
# name -> "shared"|"local": tracks how this process acquired each slot so release routes correctly.
_ACQUIRED: dict[str, str] = {}
_ACQUIRED_LOCK = threading.Lock()

_ENDPOINT_CACHE: dict[str, Any] = {}


def _acquire_local_slot(name: str) -> None:
    """Claim an in-process semaphore slot."""
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
    """Claim a quota slot for ``name``, blocking until one is free. Idempotent per name."""
    with _ACQUIRED_LOCK:
        if name in _ACQUIRED:
            return
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

    Returns True if a slot was released, False for a no-op.
    """
    with _ACQUIRED_LOCK:
        mode = _ACQUIRED.pop(name, None)
    if mode == "local":
        _LOCAL_SLOTS.release()
        return True
    from flash.providers.runpod import slots

    cross_replica = mode is None
    if cross_replica and slots.internal_key() is None:
        return False

    try:
        released = slots.release(name)
    except slots.SlotStoreError as exc:
        # Transient failure: reconcile on next startup will reclaim stale rows.
        logger.warning(
            "slot-store release failed for %s (%s); reconcile will reclaim it on restart",
            name,
            exc,
        )
        return not cross_replica
    return released if cross_replica else True


def reconcile_endpoint_slots() -> None:
    """Reconcile the shared slot store against live RunPod endpoints on startup. Best-effort."""
    from flash.providers.runpod import slots

    if slots.internal_key() is None:
        return
    try:
        from flash.providers.runpod import api as runpod_api
        from flash.providers.runpod.jobs import _is_flash_endpoint

        live = [
            name
            for e in runpod_api.list_endpoints()
            if _is_flash_endpoint(name := (e.get("name") or ""))
        ]
    except Exception as exc:
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
    """Runs ON the RunPod GPU worker: fetch code, train, return metrics.

    All imports must be inside the function body — this handler is serialized standalone.
    """
    import contextlib
    import json
    import os
    import subprocess
    import sys
    import tempfile
    import threading
    import time
    from datetime import UTC, datetime
    from email.utils import parsedate_to_datetime

    from huggingface_hub import snapshot_download

    if input_data.get("mode") == "preload":
        overrides = {k: str(v) for k, v in (input_data.get("env") or {}).items()}
        os.environ.update(overrides)
        tok = overrides.get("HF_TOKEN")
        # CRITICAL: huggingface_hub froze HF_HUB_CACHE at import time, so pass cache_dir
        # explicitly; os.environ update above is ignored by snapshot_download.
        hf_home = overrides.get("HF_HOME")
        # Refuse a preload not rooted at /runpod-volume — HF_HOME elsewhere means nothing
        # gets persisted to the volume (phantom warm).
        if not hf_home or not hf_home.startswith("/runpod-volume"):
            return {
                "preloaded": [],
                "already_cached": [],
                "failed": {},
                "error": f"preload requires HF_HOME rooted at /runpod-volume (got HF_HOME={hf_home!r})",
                "hf_home": hf_home,
            }
        if not os.path.isdir("/runpod-volume"):
            return {
                "preloaded": [],
                "already_cached": [],
                "failed": {},
                "error": f"weight-cache volume not mounted at /runpod-volume (HF_HOME={hf_home})",
                "hf_home": hf_home,
            }
        cache_dir = os.path.join(hf_home, "hub")
        # Inlined (handler is baked standalone); keep in sync with worker prefetch exclusions.
        ignore_patterns = ["*.pth", "*.gguf", "original/*", "*.onnx", "*.msgpack", "*.h5"]
        done, already, failed = [], [], {}
        for repo_id in input_data.get("models") or []:
            try:
                try:
                    snapshot_download(
                        repo_id=repo_id,
                        token=tok,
                        cache_dir=cache_dir,
                        ignore_patterns=ignore_patterns,
                        local_files_only=True,
                    )
                    already.append(repo_id)
                    continue
                except Exception:
                    pass
                snapshot_download(
                    repo_id=repo_id, token=tok, cache_dir=cache_dir, ignore_patterns=ignore_patterns
                )
                done.append(repo_id)
            except Exception as exc:
                failed[repo_id] = str(exc)[:300]
        return {
            "preloaded": done,
            "already_cached": already,
            "failed": failed,
            "hf_home": os.environ.get("HF_HOME"),
        }

    overrides = {k: str(v) for k, v in (input_data.get("env") or {}).items()}

    def _extra_pip_env() -> tuple[dict[str, str], str | None]:
        env = dict(os.environ)
        env.update(overrides)
        env["GIT_TERMINAL_PROMPT"] = "0"
        askpass = None
        if env.get("GITHUB_TOKEN"):
            fd, askpass = tempfile.mkstemp(prefix="flash-github-askpass-", suffix=".sh")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(
                    "#!/bin/sh\n"
                    'case "$1" in\n'
                    '*Username*) printf "%s\\n" "x-access-token" ;;\n'
                    '*) printf "%s\\n" "$GITHUB_TOKEN" ;;\n'
                    "esac\n"
                )
            os.chmod(askpass, 0o700)
            env["GIT_ASKPASS"] = askpass
        return env, askpass

    extra_pip = input_data.get("extra_pip") or []
    if extra_pip:
        extra_env, askpass = _extra_pip_env()
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", *extra_pip],
                check=True,
                env=extra_env,
            )
        finally:
            if askpass:
                with contextlib.suppress(OSError):
                    os.remove(askpass)

    def _code_prefix() -> str:
        raw = input_data.get("code_prefix")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("missing code_prefix")
        prefix = raw.strip().strip("/")
        parts = prefix.split("/")
        digest = parts[1] if len(parts) == 3 else ""
        if (
            len(parts) != 3
            or parts[0] != "code"
            or parts[2] != "flash"
            or len(digest) != 32
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError(f"invalid code_prefix: {prefix!r}")
        return prefix

    def _hf_status_code(exc: BaseException) -> int | None:
        response = getattr(exc, "response", None)
        code = getattr(response, "status_code", None)
        try:
            return int(code)
        except (TypeError, ValueError):
            return None

    def _hf_retry_after(exc: BaseException) -> float | None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None) or {}
        value = headers.get("retry-after") if hasattr(headers, "get") else None
        if not value and hasattr(headers, "items"):
            for key, candidate in headers.items():
                if str(key).lower() == "retry-after":
                    value = candidate
                    break
        if not value:
            return None
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(str(value))
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=UTC)
                seconds = (retry_at - datetime.now(UTC)).total_seconds()
            except (TypeError, ValueError):
                return None
        return min(60.0, max(0.0, seconds))

    def _hf_call(call, label: str):
        retry_delays = (1.0, 3.0, 8.0, 20.0, 60.0)
        transient_status_codes = {429, 500, 502, 503, 504}
        for attempt in range(len(retry_delays) + 1):
            try:
                return call()
            except Exception as exc:
                if _hf_status_code(exc) not in transient_status_codes or attempt >= len(
                    retry_delays
                ):
                    raise
                retry_after = _hf_retry_after(exc)
                delay = retry_after if retry_after is not None else retry_delays[attempt]
                print(
                    f"{label} transient Hugging Face error; retrying in {delay:.0f}s: {exc}",
                    flush=True,
                )
                time.sleep(delay)
        raise AssertionError("unreachable")

    def _download_code_prefix(repo_id: str, prefix: str, token: str | None) -> None:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi(token=token)
        files = [
            entry.path
            for entry in _hf_call(
                lambda: list(
                    api.list_repo_tree(
                        repo_id=repo_id,
                        repo_type="dataset",
                        path_in_repo=prefix,
                        recursive=True,
                        token=token,
                    )
                ),
                f"list flash code under {repo_id}:{prefix}",
            )
            if getattr(entry, "path", None) and getattr(entry, "size", None) is not None
        ]
        if not files:
            raise RuntimeError(f"no flash code files found under {repo_id}:{prefix}")
        for filename in files:
            _hf_call(
                lambda filename=filename: hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=filename,
                    local_dir="/runcode",
                    token=token,
                ),
                f"download flash code file {repo_id}:{filename}",
            )

    code_prefix = _code_prefix()
    _download_code_prefix(input_data["hf_repo"], code_prefix, overrides.get("HF_TOKEN"))
    code_dir = os.path.join("/runcode", os.path.dirname(code_prefix) or ".")

    env = dict(os.environ)
    env.update(overrides)
    # INLINED: handler is baked standalone (flash not importable). Mirrors deps.drop_unmounted_cache_env.
    if not os.path.isdir("/runpod-volume"):
        for _k in [k for k, v in env.items() if str(v).startswith("/runpod-volume")]:
            env.pop(_k, None)
    # Pass spec via file to avoid ~128 KiB per-env-string exec limit.
    spec_path = "/tmp/job_spec.json"
    with open(spec_path, "w") as sf:
        sf.write(input_data["job_spec_json"])
    env["FLASH_JOB_SPEC_PATH"] = spec_path
    env.pop("FLASH_JOB_SPEC_JSON", None)
    env["PHASE"] = input_data["phase"]
    env["SEED"] = str(input_data["seed"])
    env["PYTHONPATH"] = code_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    def _upload_console(mode: str) -> None:
        """Upload the captured console tail for ``mode`` to ``{phase_ns}/{run_id}/
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
            prefix = f"{phase_ns}/{spec['run_id']}"
            # Read only the last 64 KB (seek from the end) — the console can be very large on long
            # runs, so f.read()[-64_000:] would pull the whole file into memory just to slice it.
            tail_bytes = 64_000
            with open(console, "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - tail_bytes))
                tail = f.read().decode("utf-8", "replace")
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
        """Run worker subprocess, tee console to file, upload tail periodically and on exit."""
        console = f"/tmp/console_{mode}.txt"
        interval = 3600.0
        stop_upload = threading.Event()

        def _upload_loop() -> None:
            while not stop_upload.wait(interval):
                _upload_console(mode)  # best-effort; swallows its own errors

        with open(console, "w", buffering=1) as cf:  # line-buffered so uploader sees each line
            proc = subprocess.Popen(
                [sys.executable, "-m", "flash.engine.worker"],
                cwd=code_dir,
                env={**env, "RUN_MODE": mode},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            uploader = threading.Thread(target=_upload_loop, daemon=True)
            uploader.start()
            try:
                for line in proc.stdout:
                    print(line, end="")
                    cf.write(line)
                proc.wait()
            finally:
                stop_upload.set()
                uploader.join(timeout=10)
        _upload_console(mode)
        if proc.returncode != 0 and check:
            raise RuntimeError(
                f"worker mode '{mode}' exited {proc.returncode}; see console_{mode}.txt "
                f"and error_{mode}_attempt*.txt in the HF dataset repo"
            )
        return proc.returncode

    # Clear stale metrics from a previous seed so a crash can't report wrong numbers.
    for stale in ("/tmp/train_meta.json", "/tmp/metrics.json"):
        with contextlib.suppress(FileNotFoundError):
            os.remove(stale)
    # check=False: RL's colocated vLLM can segfault at interpreter exit after saving — not a failure.
    run_mode(input_data["phase"], check=False)
    if not os.path.exists("/tmp/metrics.json"):
        phase = input_data["phase"]
        _upload_console(phase)
        raise RuntimeError(
            f"train phase '{phase}' produced no /tmp/metrics.json (it crashed before "
            f"finishing); see error_{phase}_attempt*.txt and console_{phase}.txt in the HF "
            f"dataset repo for the full traceback"
        )
    with open("/tmp/metrics.json") as f:
        return json.load(f)


def isolate_flash_state(scope: str | None = None) -> None:
    """Point the Flash SDK's resource registry at a per-process directory under ~/.flash/flash-state/."""
    try:
        from pathlib import Path

        import runpod_flash.core.resources.resource_manager as rm

        scope = scope or f"pid{os.getpid()}"
        state_dir = Path.home() / ".flash" / "flash-state" / scope
        state_dir.mkdir(parents=True, exist_ok=True)
        rm.FLASH_STATE_DIR = state_dir
        rm.RESOURCE_STATE_FILE = state_dir / "resources.pkl"
    except Exception as exc:
        logger.warning("flash state isolation skipped: %s", exc)


def _patch_runpod_backoff() -> None:
    """Cap the backoff exponent before the power to prevent OverflowError on long runs (~80 min+)."""
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
            a = min(int(attempt), 30)
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
        # serverless.py imported the symbol directly; patch its ref too.
        try:
            from runpod_flash.core.resources import serverless as _sl

            _sl.get_backoff_delay = _safe_get_backoff_delay
        except Exception:
            pass
    except Exception as exc:
        logger.warning("runpod backoff patch skipped: %s", exc)


def min_cuda_for(friendly_gpu: str) -> str:
    """Minimum host CUDA driver version for this GPU class (Blackwell requires >=13.0)."""
    from flash.providers.base import min_cuda_modern

    return min_cuda_modern(friendly_gpu)


def endpoint_name(friendly_gpu: str, suffix: str | None = None) -> str:
    """Flash endpoint name for a GPU class, with a per-run suffix to avoid template name collisions."""
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
    os.environ["FLASH_IS_LIVE_PROVISIONING"] = "true"
    from runpod_flash import Endpoint

    from flash.providers.runpod.auth import ensure_auth

    ensure_auth()
    _patch_runpod_backoff()
    isolate_flash_state(name_suffix)

    friendly = canonical_gpu(friendly_gpu)
    name = endpoint_name(friendly, name_suffix)
    if name in _ENDPOINT_CACHE:
        return _ENDPOINT_CACHE[name]
    _acquire_endpoint_slot(name)
    try:
        kwargs = {
            "name": name,
            "gpu": flash_gpu(friendly),
            "gpu_count": 1,
            "min_cuda_version": min_cuda_for(friendly),
            "execution_timeout_ms": execution_timeout_ms or DEFAULT_EXECUTION_TIMEOUT_MS,
            "workers": (0, 1),
        }
        image = worker_image_for_gpu(friendly, allow_default=False)
        if image:
            kwargs["image"] = image
        else:
            kwargs["dependencies"] = resolve_worker_deps()
            kwargs["system_dependencies"] = WORKER_SYSTEM_DEPS
        # Local import: avoids a jobs<->endpoints import cycle (jobs imports this module).
        from flash.providers.runpod.jobs import weight_cache_endpoint_kwargs

        kwargs.update(weight_cache_endpoint_kwargs(spec))
        ep = Endpoint(**kwargs)
        handler = ep(_train_body)
        from flash.providers.runpod.jobs import apply_disk_gb

        cfg = ep._build_resource_config()
        apply_disk_gb(cfg, disk_gb)
        _ENDPOINT_CACHE[name] = handler
        return handler
    except Exception:
        _release_endpoint_slot(name)
        raise


def _run_suffix(run_id: str | None) -> str | None:
    """Stable, collision-free per-run endpoint suffix: sha1(run_id)[:8] with a readable prefix.

    Using only the last segment of run_id collides when run_ids end in a GPU name.
    """
    if not run_id:
        return None
    import hashlib
    import re

    h = hashlib.sha1(run_id.encode()).hexdigest()[:8]
    prefix = re.sub(r"[^a-z0-9]", "", run_id.lower())[-12:]
    return f"{prefix}{h}" if prefix else h


def stop_endpoint(friendly_gpu: str, name: str | None = None) -> None:
    """Scale cached endpoint(s) to zero. Only touches in-process cache; use terminate_endpoint for cross-process teardown."""
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
    """Return resource ids whose name contains ``target`` (live resources are prefixed with ``live-``)."""
    if not target:
        return []
    out = []
    for uid, res in (resources or {}).items():
        name = str(getattr(res, "name", "") or "")
        if target in name:
            out.append(uid)
    return out


def terminate_endpoint(friendly_gpu: str, run_id: str | None = None) -> list[dict]:
    """Delete the remote Flash endpoint(s) for a run via the RunPod API. Best-effort, never raises."""
    friendly = canonical_gpu(friendly_gpu)
    target = endpoint_name(friendly, _run_suffix(run_id))
    # Serialize isolation + lookup + undeploy: isolate_flash_state swaps process-wide globals,
    # and a concurrent call could swap the registry scope between our lookup and undeploy.
    with FLASH_SDK_LOCK:
        try:
            from flash.providers.runpod.auth import ensure_auth

            ensure_auth()
            isolate_flash_state(_run_suffix(run_id))
            from runpod_flash.core.resources.resource_manager import ResourceManager
        except Exception as exc:
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
                results = asyncio.run(_undeploy_all())
            else:
                # Running event loop (FastAPI lifespan etc) — run in a daemon thread.
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

    # Registry-less fallback: per-process state means a fresh container can't see the endpoint.
    # Delete by reconstructed name via REST so the worker doesn't stay live.
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
            rest_confirmed_absent = not matches
        except Exception as exc:
            # Can't prove endpoint is gone — do NOT treat as absent; slot stays held.
            logger.debug("REST endpoint lookup failed for %s: %s", target, exc)

    # Release only when the endpoint is provably gone; do NOT release on failure (may still be live).
    if any(r.get("success") for r in results) or (not uids and rest_confirmed_absent):
        _release_endpoint_slot(target)

    with contextlib.suppress(Exception):
        stop_endpoint(friendly, name=target)
    return results
