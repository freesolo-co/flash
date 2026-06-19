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

_ENDPOINT_CACHE: dict[str, Any] = {}


def _train_body(input_data: dict) -> dict:
    """Runs ON the RunPod GPU worker: fetch code, train (phase), return metrics.

    NOTE: Flash serializes this handler and runs it standalone, so every name it uses
    must be imported INSIDE the function body (module-level imports are not in scope).
    """
    import contextlib
    import json
    import os
    import shutil
    import subprocess
    import sys

    from huggingface_hub import snapshot_download

    # NB: the Hopper fla guard lives in engine.worker._drop_fla_on_hopper (runs in the worker
    # process AFTER all installs, before any model import) — doing it here would be undone by a
    # later extra_pip / `prime env install` that pulls fla back, and depends on a handler redeploy.

    # Extra pip deps for verifiers / Prime Hub environments (installed per-run).
    extra_pip = input_data.get("extra_pip") or []
    if extra_pip:
        # check=True: a deterministic dependency failure should fail fast here,
        # not after model download + worker startup with a less actionable error.
        subprocess.run([sys.executable, "-m", "pip", "install", *extra_pip], check=True)

    # NB: fla is dropped on Hopper (sm90) automatically — resolve_worker_deps omits it from the
    # install list, and engine.worker._drop_fla_on_hopper removes any baked-in copy at worker
    # startup (fla's GDN backward is miscomputed on sm90, #640). No env toggle: fla only ever runs
    # on the consumer archs where its Triton kernel is correct.

    # Install the run's verifiers environment(s) from the Prime Hub via the authenticated
    # `prime` CLI. The public pip index does not serve PRIVATE env wheels, so a plain pip
    # install can't fetch them; `prime env install` pulls/builds/installs public + private
    # alike, authenticated by PRIME_API_KEY forwarded from the control plane.
    hub_env_ids = input_data.get("hub_env_ids") or []
    if hub_env_ids:
        worker_env = {k: str(v) for k, v in (input_data.get("env") or {}).items()}
        prime_key = worker_env.get("PRIME_API_KEY") or os.environ.get("PRIME_API_KEY")
        if not prime_key:
            raise RuntimeError(
                "PRIME_API_KEY is required to install the Prime Hub environment on the worker"
            )
        # Only install `prime` when it isn't already on the worker (it's often baked into
        # the worker image) — an unconditional install adds latency and a per-run PyPI
        # failure point every run.
        if shutil.which("prime") is None:
            subprocess.run([sys.executable, "-m", "pip", "install", "prime"], check=True)
        # --with pip: install the env into THIS (the trainer's) python via pip. The default
        # (`--with uv`) installs into prime's own isolated uv env, so the trainer then can't
        # import the env module (ModuleNotFoundError at load_environment). PIP_BREAK_SYSTEM_PACKAGES
        # lets pip write to a PEP-668 "externally-managed" base python (the worker image's).
        install_env = {
            **os.environ,
            "PRIME_API_KEY": prime_key,
            "PRIME_DISABLE_VERSION_CHECK": "1",
            "PIP_BREAK_SYSTEM_PACKAGES": "1",
        }
        for env_id in hub_env_ids:
            subprocess.run(
                ["prime", "env", "install", env_id, "--with", "pip"], check=True, env=install_env
            )

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
            try:
                from huggingface_hub import HfApi

                spec = json.loads(input_data["job_spec_json"])
                phase_ns = "rl" if spec.get("algorithm") == "grpo" else spec["algorithm"]
                prefix = f"{phase_ns}/{spec['run_id']}/seed{input_data['seed']}"
                with open(console) as f:
                    tail = f.read()[-64_000:]
                with open(console + ".tail", "w") as f:
                    f.write(tail)
                HfApi(token=env.get("HF_TOKEN")).upload_file(
                    path_or_fileobj=console + ".tail",
                    path_in_repo=f"{prefix}/console_{mode}.txt",
                    repo_id=input_data["hf_repo"],
                    repo_type="dataset",
                )
            except Exception as up_err:
                print("console upload warn:", up_err)
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
    from flash.providers.runpod.jobs import volume_endpoint_kwargs

    ensure_auth()
    _patch_runpod_backoff()
    isolate_flash_state(name_suffix)

    friendly = canonical_gpu(friendly_gpu)
    name = endpoint_name(friendly, name_suffix)
    if name in _ENDPOINT_CACHE:
        return _ENDPOINT_CACHE[name]
    kwargs = dict(
        name=name,
        gpu=flash_gpu(friendly),
        # GPUs per worker (= trainer + inference_gpus); >1 for disaggregated async GRPO, else 1.
        gpu_count=max(1, int(getattr(getattr(spec, "gpu", None), "count", 1))),
        min_cuda_version=min_cuda_for(friendly),
        execution_timeout_ms=execution_timeout_ms or DEFAULT_EXECUTION_TIMEOUT_MS,
        workers=(0, 1),  # one dedicated worker per run; scale to zero when idle
        **volume_endpoint_kwargs(spec),
    )
    # RunPod Flash needs its serverless runtime baked into the worker image; the prebuilt
    # WORKER_IMAGE is Vast's cold-start image (no Flash runtime) and leaves the worker unhealthy if
    # forced. RunPod boot-installs WORKER_DEPS on Flash's default template instead (cached as a
    # Flash artifact). Optional FLASH_WORKER_IMAGE override for a RunPod-serverless-compatible image.
    image = (os.environ.get("FLASH_WORKER_IMAGE") or "").strip()
    if image:
        kwargs["image"] = image
    else:
        kwargs["dependencies"] = resolve_worker_deps(friendly)
        kwargs["system_dependencies"] = WORKER_SYSTEM_DEPS
    ep = Endpoint(**kwargs)
    handler = ep(_train_body)  # register the queue-based handler; returns the callable
    # The resource config is cached on the Endpoint, so raising the disk on it here
    # carries through to the deploy that the first handler call triggers.
    from flash.providers.runpod.jobs import apply_disk_gb

    cfg = ep._build_resource_config()
    apply_disk_gb(cfg, disk_gb)
    _ENDPOINT_CACHE[name] = handler
    return handler


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
    ``slm cancel`` process. Use ``terminate_endpoint`` to actually delete the remote endpoint.
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
            results = asyncio.run(_undeploy_all())
        except Exception as exc:
            results = [{"success": False, "name": target, "message": str(exc)}]

    # Registry-less fallback: isolate_flash_state() keeps the Flash SDK's resource
    # registry per-process under ~/.flash, so a recreated container (or a crash before
    # on_handle() persisted the endpoint id) leaves the live endpoint invisible to the
    # lookup above. Delete it via the RunPod REST API by its reconstructed name so it
    # can't keep a paid worker alive.
    if not uids:
        with contextlib.suppress(Exception):
            from flash.providers.runpod import api as runpod_api

            for ep in runpod_api.find_endpoints_by_name(target):
                if ep.get("name") == target and runpod_api.delete_endpoint(ep["id"]):
                    results.append(
                        {"success": True, "name": target, "message": "deleted via REST API"}
                    )

    # also drop the in-process cached handler for THIS run only (a class-wide
    # drop would evict a concurrent run's endpoint on the same GPU class).
    with contextlib.suppress(Exception):
        stop_endpoint(friendly, name=target)
    return results
