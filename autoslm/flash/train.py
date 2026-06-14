"""RunPod Flash fine-tuning endpoints (queue-based, one dedicated GPU per run).

Flash provisions a dedicated RunPod GPU (RTX 4090 / 5090, no Docker), installs
``WORKER_DEPS``, runs the handler, returns the metrics dict, and scales to zero.

Flash's live ("ad-hoc") provisioning does not bundle local project code, so the
handler fetches the ``autoslm`` package from the HF dataset repo (uploaded by
``upload_code`` before submit), adds it to ``PYTHONPATH``, and runs
``autoslm.engine.worker`` to train. The worker streams the adapter + checkpoints to
the same HF repo for serving and preemption-resilient resume.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import os
from typing import Any

from autoslm._logging import get_logger
from autoslm.flash.gpus import canonical_gpu, flash_gpu, gpu_short
from autoslm.worker_spec import JobSpec

logger = get_logger(__name__)


# Worker stack: trl 1.x (colocate default), vllm 0.19.1 (Qwen3.5/3.6 archs,
# native RL APIs, transformers-5
# compatible metadata), transformers 5.x (qwen3_5/qwen3_5_moe model types),
# bitsandbytes (QLoRA tier for the 35B-A3B MoE).
# Resolver/driver notes: vllm 0.17/0.18 hard-pin transformers<5 (uv refuses the
# combo), so the first transformers-5-compatible vllm line is 0.19.1. vllm >=0.20
# pins torch 2.11 whose default pypi wheels are CUDA-13 builds — RunPod 4090/5090
# hosts filtered at min_cuda 12.8 often run 12.8/12.9 drivers where cu13 torch sees
# NO GPU (observed: "cuda not available" + vLLM "cumem allocator not supported").
# vllm 0.19.1 pins torch 2.10 (cu128 default) which matches those drivers.
# trl's *optional* [vllm] extra caps at 0.18, but we install plain trl, so the only
# constraint that matters is runtime API compat — validated per-model on real
# RTX 4090/5090 workers before promotion to default (see bench/results/phase1).
WORKER_DEPS = [
    "torch==2.10.0",
    "transformers>=5.6,<5.11",
    "trl>=1.5,<1.6",
    "peft>=0.19",
    "vllm==0.19.1",
    "bitsandbytes>=0.49",
    "datasets>=4.7,<6",
    "huggingface_hub>=0.25",
    "accelerate>=1.4",
    # Fused Triton kernels for Gated-DeltaNet (Qwen3.5/3.6 family): without this,
    # transformers falls back to a pure-PyTorch delta rule and GRPO trainer steps are
    # 2-3x slower (measured A/B on Qwen3.5-2B: ~65 s/step -> ~20 s/step steady).
    "flash-linear-attention==0.5.0",
]
# NOTE on download speed: Flash's runtime already ships hf_transfer and exports
# HF_HUB_ENABLE_HF_TRANSFER=1 on workers (measured: Qwen3-4B's ~8 GB pulled in 6.3 s,
# NIC-saturated — bench/results/phase6). Adding hf_transfer here is redundant; don't.
# Override the whole pinned stack per-run with AUTOSLM_WORKER_DEPS="pkgA==1 pkgB>=2"
# (whitespace-separated, or a JSON list for specs containing commas).
WORKER_SYSTEM_DEPS = ["build-essential"]  # Triton/Inductor need a C compiler


def resolve_worker_deps() -> list[str]:
    """The dependency list Flash installs on the GPU worker for this run.

    Precedence: AUTOSLM_WORKER_DEPS (explicit list) > the pinned ``WORKER_DEPS``.
    """
    explicit = os.environ.get("AUTOSLM_WORKER_DEPS")
    if explicit:
        # JSON list (use this for specs containing commas, e.g.
        # "transformers>=5.6,<5.11") or a whitespace-separated string.
        if explicit.strip().startswith("["):
            import json as _json

            deps = [str(d).strip() for d in _json.loads(explicit) if str(d).strip()]
        else:
            # shlex (whitespace) splitting, NOT comma: a comma is part of a PEP 440
            # range like `transformers>=5.6,<5.11` and must not be split.
            import shlex

            deps = [d for d in shlex.split(explicit) if d.strip()]
        if deps:
            return deps
    deps = WORKER_DEPS
    # Additive per-run extras (e.g. liger-kernel for the SFT_LIGER A/B) without
    # restating the whole pinned stack the way AUTOSLM_WORKER_DEPS requires.
    extra = os.environ.get("AUTOSLM_WORKER_EXTRA_DEPS")
    if extra:
        import shlex

        deps = deps + [d for d in shlex.split(extra) if d.strip()]
    return deps


DEFAULT_EXECUTION_TIMEOUT_MS = int(
    os.environ.get("AUTOSLM_EXECUTION_TIMEOUT_MS", str(6 * 3600 * 1000))
)

_ENDPOINT_CACHE: dict[str, Any] = {}


def upload_code() -> str:
    """Upload the ``autoslm`` package (+ the sibling ``examples/`` tree) to ``HF_REPO``.

    The worker downloads ``code/**`` to ``/runcode`` and the registry resolves
    built-in example environments (gsm8k/math) from ``code/examples`` next to
    the package, so the examples tree must travel with the code or those runs
    fail with FileNotFoundError after provisioning.
    """
    from huggingface_hub import HfApi

    import autoslm

    repo = os.environ.get("HF_REPO")
    if not repo:
        raise RuntimeError("HF_REPO must be set (HF dataset repo for code + artifacts)")
    token = os.environ.get("HUGGINGFACE_TOKEN")
    pkg_dir = os.path.dirname(os.path.abspath(autoslm.__file__))
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", exist_ok=True, private=True)
    api.upload_folder(
        folder_path=pkg_dir,
        path_in_repo="code/autoslm",
        repo_id=repo,
        repo_type="dataset",
        ignore_patterns=["__pycache__/*", "*.pyc"],
    )
    examples_dir = os.path.join(os.path.dirname(pkg_dir), "examples")
    if os.path.isdir(examples_dir):
        api.upload_folder(
            folder_path=examples_dir,
            path_in_repo="code/examples",
            repo_id=repo,
            repo_type="dataset",
            ignore_patterns=["__pycache__/*", "*.pyc"],
        )
    return repo


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

    # Extra pip deps for verifiers / Prime Hub environments (installed per-run).
    extra_pip = input_data.get("extra_pip") or []
    if extra_pip:
        # check=True: a deterministic dependency failure should fail fast here,
        # not after model download + worker startup with a less actionable error.
        subprocess.run([sys.executable, "-m", "pip", "install", *extra_pip], check=True)

    overrides = {k: str(v) for k, v in (input_data.get("env") or {}).items()}
    snapshot_download(
        repo_id=input_data["hf_repo"],
        repo_type="dataset",
        allow_patterns=["code/**"],
        local_dir="/runcode",
        token=overrides.get("HUGGINGFACE_TOKEN"),
    )
    code_dir = "/runcode/code"

    env = dict(os.environ)
    env.update(overrides)
    # A Freesolo bridge spec embeds the whole dataset in records, which can blow
    # past the ~128 KiB per-env-string exec limit ("Argument list too long").
    # Pass a large spec via a file (AUTOSLM_JOB_SPEC_PATH); keep the inline env
    # var for small specs. load_job_spec_from_env reads either.
    spec_json = input_data["job_spec_json"]
    if len(spec_json) > 96_000:
        spec_path = "/tmp/job_spec.json"
        with open(spec_path, "w") as sf:
            sf.write(spec_json)
        env["AUTOSLM_JOB_SPEC_PATH"] = spec_path
        env.pop("AUTOSLM_JOB_SPEC_JSON", None)
    else:
        env["AUTOSLM_JOB_SPEC_JSON"] = spec_json
    env["PHASE"] = input_data["phase"]
    env["SEED"] = str(input_data["seed"])
    env["PYTHONPATH"] = code_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    def run_mode(mode: str, check: bool) -> int:
        """Run one worker process, tee its console to a file, and on failure upload the
        tail to HF as console_<mode>.txt — the engine-core root cause of crashes like
        vLLM EngineDeadError only ever appears on the subprocess console, never in the
        Python traceback."""
        console = f"/tmp/console_{mode}.txt"
        with open(console, "w") as cf:
            proc = subprocess.Popen(
                [sys.executable, "-m", "autoslm.engine.worker"],
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
        if proc.returncode != 0:
            try:
                from huggingface_hub import HfApi

                spec = json.loads(input_data["job_spec_json"])
                phase_ns = "rl" if spec.get("algorithm") == "grpo" else spec["algorithm"]
                prefix = f"{phase_ns}/{spec['run_id']}/seed{input_data['seed']}"
                with open(console) as f:
                    tail = f.read()[-64_000:]
                with open(console + ".tail", "w") as f:
                    f.write(tail)
                HfApi(token=env.get("HUGGINGFACE_TOKEN")).upload_file(
                    path_or_fileobj=console + ".tail",
                    path_in_repo=f"{prefix}/console_{mode}.txt",
                    repo_id=input_data["hf_repo"],
                    repo_type="dataset",
                )
            except Exception as up_err:
                print("console upload warn:", up_err)
            if check:
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
    clobbering each other's bookkeeping. Each AutoSLM process gets its own registry
    under ``~/.autoslm/flash-state/<scope>``; remote cleanup never relies on the
    registry anyway (REST by id/name — see runpod_api).
    """
    try:
        from pathlib import Path

        import runpod_flash.core.resources.resource_manager as rm

        scope = scope or f"pid{os.getpid()}"
        state_dir = Path.home() / ".autoslm" / "flash-state" / scope
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

        if getattr(_bo, "_autoslm_backoff_patched", False):
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
        _bo._autoslm_backoff_patched = True
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

    Blackwell classes (sm_120/sm_100 — RTX 5090, RTX Pro 6000, B200): pypi wheels for
    the modern stack (vllm 0.19) ship no Blackwell SASS, so every custom CUDA kernel
    is PTX-JIT'd by the driver — and their PTX is built with a newer toolchain than
    CUDA-12.8-era drivers can JIT (observed: "the provided PTX was compiled with an
    unsupported toolchain" on driver 570.x). CUDA-13 drivers JIT it fine, so those
    classes are pinned to >=13.0 on the modern stack (per-GPU ``min_cuda_modern`` in
    flash.gpus.GPU_INFO). Ampere/Ada/Hopper have SASS in the wheels and run on 12.8.
    Override with AUTOSLM_MIN_CUDA.
    """
    explicit = os.environ.get("AUTOSLM_MIN_CUDA")
    if explicit:
        return explicit
    from autoslm.flash.gpus import min_cuda_modern

    return min_cuda_modern(friendly_gpu)


def endpoint_name(friendly_gpu: str, suffix: str | None = None) -> str:
    """Flash endpoint/template name for a GPU class, optionally made unique per run.

    A fixed name (``autoslm-train-5090``) collides across back-to-back runs: runpod_flash's
    ``get_or_deploy_resource`` finds the prior run's still-registered resource and tries to
    *update* it, which fails with ``GraphQL errors: Template name must be unique`` (there is
    no endpoint GC/reuse). A per-run ``suffix`` (the run id tail) gives each run its own
    endpoint so it deploys fresh instead of colliding. RunPod scales each to zero when idle.
    """
    base = f"autoslm-train-{gpu_short(friendly_gpu)}"
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

    from autoslm.flash.auth import ensure_auth
    from autoslm.flash.durable import volume_endpoint_kwargs

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
        gpu_count=1,
        min_cuda_version=min_cuda_for(friendly),
        execution_timeout_ms=execution_timeout_ms or DEFAULT_EXECUTION_TIMEOUT_MS,
        workers=(0, 1),  # one dedicated worker per run; scale to zero when idle
        **volume_endpoint_kwargs(spec),
    )
    # Optional prebuilt image (deps baked in) cuts the cold-start dep install. Otherwise
    # Flash installs WORKER_DEPS on first use (cached as an artifact across calls).
    image = os.environ.get("AUTOSLM_WORKER_IMAGE")
    if image:
        kwargs["image"] = image
    else:
        kwargs["dependencies"] = resolve_worker_deps()
        kwargs["system_dependencies"] = WORKER_SYSTEM_DEPS
    ep = Endpoint(**kwargs)
    handler = ep(_train_body)  # register the queue-based handler; returns the callable
    # The resource config is cached on the Endpoint, so raising the disk on it here
    # carries through to the deploy that the first handler call triggers.
    from autoslm.flash.durable import apply_disk_gb

    apply_disk_gb(ep._build_resource_config(), disk_gb)
    _ENDPOINT_CACHE[name] = handler
    return handler


def _run_suffix(run_id: str | None) -> str | None:
    """Short, unique-per-run endpoint suffix derived from the run id tail."""
    if not run_id:
        return None
    return run_id.split("-")[-1]


def stop_endpoint(friendly_gpu: str, name: str | None = None) -> None:
    """Best-effort: scale cached endpoint(s) to zero / drop them.

    With ``name`` only that run's cached endpoint is dropped; without it, every
    cached endpoint of the GPU class is — so a per-run teardown passes ``name``
    to avoid evicting a concurrent run's handler in the same process.

    NOTE: this only touches THIS process's in-memory cache, so it does nothing in a fresh
    ``slm cancel`` process. Use ``terminate_endpoint`` to actually delete the remote endpoint.
    """
    friendly = canonical_gpu(friendly_gpu)
    prefix = f"autoslm-train-{gpu_short(friendly)}"
    if name:
        match = [k for k in _ENDPOINT_CACHE if k == name]
    else:
        match = [k for k in _ENDPOINT_CACHE if k == friendly or k.startswith(prefix)]
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


def _select_endpoint_resources(resources: dict, target: str) -> list:
    """Resource ids whose resource ``.name`` contains ``target``.

    The live-provisioned resource is named ``live-<endpoint_name>``, so we match by substring
    to catch the prefix. ``target`` is the endpoint name (``autoslm-train-<gpu>[-<run>]``).
    """
    if not target:
        return []
    out = []
    for uid, res in (resources or {}).items():
        name = str(getattr(res, "name", "") or "")
        if target in name:
            out.append(uid)
    return out


def terminate_endpoint(friendly_gpu: str, run_id: str | None = None) -> list:
    """Reliably tear down the remote Flash endpoint(s) for a run — cross-process.

    Unlike ``stop_endpoint`` (which only touches this process's in-memory cache), this looks
    the endpoint up by name in runpod_flash's *persisted* resource registry and deletes it via
    the RunPod API (``ResourceManager.undeploy_resource`` -> ``delete_endpoint``), which stops
    any running worker. Best-effort: never raises. Returns the per-resource undeploy results.

    With ``run_id`` it targets exactly that run's uniquely-named endpoint; without it, the
    bare ``autoslm-train-<gpu>`` prefix matches every endpoint of that GPU class.
    """
    friendly = canonical_gpu(friendly_gpu)
    target = endpoint_name(friendly, _run_suffix(run_id))
    try:
        from autoslm.flash.auth import ensure_auth

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
                out.append(await rm.undeploy_resource(uid, resource_name=name, force_remove=True))
            except Exception as exc:
                out.append({"success": False, "name": name, "message": str(exc)})
        return out

    try:
        results = asyncio.run(_undeploy_all())
    except Exception as exc:
        results = [{"success": False, "name": target, "message": str(exc)}]

    # Registry-less fallback: isolate_flash_state() keeps the Flash SDK's resource
    # registry per-process under ~/.autoslm, so a recreated container (or a crash before
    # on_handle() persisted the endpoint id) leaves the live endpoint invisible to the
    # lookup above. Delete it via the RunPod REST API by its reconstructed name so it
    # can't keep a paid worker alive.
    if not uids:
        with contextlib.suppress(Exception):
            from autoslm.flash import runpod_api

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


def build_worker_env(spec: JobSpec, seed: int) -> dict:
    """Per-run env passed to the worker (secrets + recipe overrides)."""
    env: dict[str, str] = {
        "RUN_ID": spec.run_id,
        "BENCH_HF_MODEL": spec.model,
        # Colocate (TRL trainer + vLLM on one GPU) is memory-tight and fragments over a long
        # run, OOMing late (e.g. in vLLM's sampler) with memory "reserved but unallocated".
        # expandable_segments lets the allocator reclaim that fragmentation. Overridable.
        # torch >= 2.10 renamed the env to PYTORCH_ALLOC_CONF — set BOTH names so the
        # setting applies on either stack.
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
            "PYTORCH_ALLOC_CONF",
            os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        ),
        "PYTORCH_ALLOC_CONF": os.environ.get(
            "PYTORCH_ALLOC_CONF",
            os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        ),
        # Escape hatch for torch.compile/inductor spikes (Qwen3.5 DeltaNet kernels
        # compile at first forward and can OOM a tight colocate budget).
        **(
            {"TORCHDYNAMO_DISABLE": os.environ["TORCHDYNAMO_DISABLE"]}
            if os.environ.get("TORCHDYNAMO_DISABLE")
            else {}
        ),
    }
    for key in ("HF_REPO", "HUGGINGFACE_TOKEN"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    # snapshot_download / from_pretrained / load_dataset / vLLM read HF_TOKEN, not
    # HUGGINGFACE_TOKEN, so private/gated model+data pulls need it under that name.
    if os.environ.get("HUGGINGFACE_TOKEN") and not env.get("HF_TOKEN"):
        env["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]
    # Opt-in network volume: point the whole HF cache at the persistent mount so
    # model weights survive across runs (the download becomes a one-time cost per
    # volume instead of per run).
    if getattr(spec.gpu, "network_volume", None):
        env["HF_HOME"] = "/runpod-volume/hf-cache"
    if spec.train.steps is not None:
        env["RL_STEPS"] = str(spec.train.steps)
    if spec.train.epochs is not None:
        env["SFT_EPOCHS"] = str(spec.train.epochs)
    # Forward the documented worker-tuning knobs so they actually reach the GPU worker.
    # RL_VLLM_GPU_UTIL / RL_PER_DEVICE_PROMPTS are the colocate-memory knobs the docs tell
    # users to set to fix vLLM OOM/KV-cache errors; they were previously dropped here.
    for k in (
        "SFT_MAX_STEPS",
        "SFT_MAX_EXAMPLES",
        "SFT_PER_DEVICE_BS",
        "SFT_SAVE_STEPS",
        "SFT_PACKING",
        "SFT_LIGER",
        "RL_MAX_COMPLETION",
        "RL_VLLM_GPU_UTIL",
        "RL_VLLM_SLEEP",
        "RL_VLLM_MAX_LEN",
        "RL_PER_DEVICE_PROMPTS",
        "RL_PROMPTS_PER_STEP",
        "RL_GROUP_SIZE",
        "RL_SAVE_STEPS",
        "VLLM_USE_V1",
        # Attention-backend escape hatch: vllm's bundled flash-attn PTX can be newer
        # than the host driver's JIT (sm_120 + 12.8 drivers); TRITON_ATTN/FLASHINFER
        # sidestep it without restricting the host pool to CUDA-13 drivers.
        "VLLM_ATTENTION_BACKEND",
        "AUTOSLM_QUANT",
        "AUTOSLM_QUANT_REPO",
        "AUTOSLM_THINKING",
        "LORA_TARGETS",
    ):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


def submit_train(spec: JobSpec, seed: int, log=None) -> dict:
    """Provision a dedicated GPU via Flash, run training, return the metrics dict."""
    timeout_s = max(60, int(spec.gpu.max_wall_seconds))
    from autoslm.envs.registry import worker_pip_for_env

    handler = get_train_endpoint(
        spec.gpu.type,
        execution_timeout_ms=timeout_s * 1000,
        name_suffix=_run_suffix(spec.run_id),
        disk_gb=spec.gpu.disk_gb,
        spec=spec,
    )
    payload = {
        "hf_repo": os.environ.get("HF_REPO", ""),
        "job_spec_json": spec.to_json(),
        "phase": spec.phase,
        "seed": int(seed),
        "env": build_worker_env(spec, seed),
        "extra_pip": list(spec.environment.pip)
        or worker_pip_for_env(spec.environment.id, spec.environment.params),
    }
    if log is not None:
        print(
            f"submitting Flash job: gpu={spec.gpu.type} phase={spec.phase} "
            f"seed={seed} model={spec.model}",
            file=log,
            flush=True,
        )

    async def _call():
        res = handler(payload)
        if inspect.isawaitable(res):
            res = await res
        return res

    out = asyncio.run(_call())
    if not isinstance(out, dict):
        raise RuntimeError(f"flash job returned no metrics: {out!r}")
    return out
