"""Modal app: multi-LoRA serving with one GPU container per base model. ``LoraEngine`` is a
Modal class parametrized by ``base_model`` (one GPU per base model, sharing its adapters via
``enable_lora``); ``router`` is the CPU front door that dispatches to the right engine. See
README to deploy.
"""

import asyncio
import contextlib
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import modal

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - only relevant for global modal installs
    load_dotenv = None


SERVING_DIR = Path(__file__).resolve().parent
# flash/serving/app/ -> flash/serving/ -> flash/ -> repo root. the app used to sit one level below
# its repo root; after the moves it is three, so walking up too few parents would look for .env
# inside the flash package and load nothing, leaving a production deploy silently unconfigured.
REPO_DIR = SERVING_DIR.parent.parent.parent

# deployment identity is resolved before dotenv so development can never inherit production wiring.
# modal.is_local() is true in remote child processes too, so the remote marker must also be absent.
_validate_deploy_wiring = modal.is_local() and os.environ.get("MODAL_IS_REMOTE") != "1"
_ALLOWED_SERVING_DEPLOYMENT_MODES = frozenset({"production", "development"})
_requested_deployment_mode = os.environ.get("SERVING_DEPLOYMENT_MODE", "").strip()
SERVING_DEPLOYMENT_MODE = _requested_deployment_mode or "production"
MODAL_ENVIRONMENT = str(modal.config.config.get("environment") or "").strip()
if _validate_deploy_wiring:
    if SERVING_DEPLOYMENT_MODE not in _ALLOWED_SERVING_DEPLOYMENT_MODES:
        raise ValueError(
            "SERVING_DEPLOYMENT_MODE must be 'production' or 'development', "
            f"not {SERVING_DEPLOYMENT_MODE!r}"
        )
    if SERVING_DEPLOYMENT_MODE == "development" and MODAL_ENVIRONMENT != "dev":
        raise ValueError("development serving must target Modal environment 'dev'")
    if SERVING_DEPLOYMENT_MODE == "production" and MODAL_ENVIRONMENT == "dev":
        raise ValueError("production serving must not target Modal environment 'dev'")
    if load_dotenv is not None and SERVING_DEPLOYMENT_MODE == "production":
        load_dotenv(REPO_DIR / ".env")

APP_NAME = "freesolo-lora-serving"  # hardcoded; no deploy-time knob
# branded serving hostname, served alongside the default *.modal.run url once dns + tls resolve.
# this remains environment-driven because custom domains must already be verified in the target modal
# workspace. production keeps the domain optional for local and fork workspaces, while the official
# development deployment fails closed on its isolated public domain.
_DEVELOPMENT_SERVING_DOMAIN = "serve-dev.freesolo.co"
SERVING_CUSTOM_DOMAIN = os.environ.get("SERVING_CUSTOM_DOMAIN", "").strip()
if (
    _validate_deploy_wiring
    and SERVING_DEPLOYMENT_MODE == "development"
    and SERVING_CUSTOM_DOMAIN != _DEVELOPMENT_SERVING_DOMAIN
):
    raise ValueError(f"development SERVING_CUSTOM_DOMAIN must be {_DEVELOPMENT_SERVING_DOMAIN}")
if _validate_deploy_wiring and SERVING_DEPLOYMENT_MODE == "development":
    required = (
        "FREESOLO_INTERNAL_KEY",
        "PLATFORM_BACKEND_URL",
        "SUPABASE_PROJECT_REF",
        "SUPABASE_PROJECT_REF_DEV",
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
    )
    missing = [name for name in required if not os.environ.get(name, "").strip()]
    if not os.environ.get("HF_TOKEN", "").strip():
        missing.append("HF_TOKEN")
    if missing:
        raise ValueError(
            "development serving requires explicit environment wiring: " + ", ".join(missing)
        )
    backend_url = os.environ.get("PLATFORM_BACKEND_URL", "").strip().rstrip("/")
    if backend_url != "https://api-dev.freesolo.co":
        raise ValueError("development PLATFORM_BACKEND_URL must be https://api-dev.freesolo.co")
    production_ref = os.environ["SUPABASE_PROJECT_REF"].strip()
    dev_ref = os.environ["SUPABASE_PROJECT_REF_DEV"].strip()
    if production_ref == dev_ref:
        raise ValueError("production and development Supabase project refs must differ")
    expected_supabase_url = f"https://{dev_ref}.supabase.co"
    if os.environ["SUPABASE_URL"].strip().rstrip("/") != expected_supabase_url:
        raise ValueError("development SUPABASE_URL must match SUPABASE_PROJECT_REF_DEV")
HF_CACHE_VOLUME_NAME = "freesolo-lora-serving-hf-cache"
HOSTING_CACHE_MOUNT = "/vol/hosting-cache"


# ── Autoscaling / container config: HARDCODED (no deploy-time knobs) ──────────────────────────────
# Like src/settings.py, these are plain constants, not env vars — a config either exists here or it
# doesn't; there is nothing to tune at deploy time. Every value is the proven default. (The wiring
# below — HF/internal key, backend + Supabase URLs — stays env-configured because it is per-deploy
# credentials, not tuning.)
TIMEOUT_SECONDS = 600
# 2700s (45 min): the 35B bf16/H200 cold boot takes ~1010s of engine init alone (67 GiB load +
# ~377s torch.compile + graph capture + warmup) on top of image pull, which blows past the old 1200s
# ceiling — Modal SIGTERMs the container mid-warmup and it cold-cycles forever. A higher ceiling is
# harmless for the fast tiers (they still boot in a minute; this is only a kill-if-stuck bound).
STARTUP_TIMEOUT_SECONDS = 2700
# The router awaits the engine call, which on a cold base model includes vLLM startup, so its
# timeout must cover startup + a request (else it's killed mid cold-start).
ROUTER_TIMEOUT_SECONDS = STARTUP_TIMEOUT_SECONDS + TIMEOUT_SECONDS
# Maximum idle window before a gpu model container scales down, PER GPU TIER.
#
# Sizing rule: an idle container bills at the full GPU rate, so holding one is only worth it while
# waiting costs less than re-paying the cold boot. Boot is a fixed cost (weights + torch.compile +
# graph capture + warmup); idle is a rate. The break-even idle window is therefore ~the tier's own
# cold-boot time — hold roughly as long as a restart would cost, then release the card.
#
# A single flat 1800s applied that same 30-minute hold to every tier, so a cheap L4 that boots in
# ~60s still burned 30 GPU-minutes after its last request. Per Modal's rates the idle tax per cold
# wake was $0.40 (L4) / $0.97 (L40S) / $1.98 (H100) / $2.27 (H200) regardless of how much work the
# request did.
#
# The H200 keeps a LONGER hold than break-even alone would suggest: the 35B's boot is ~1010s of
# engine init (67 GiB of weights + ~377s torch.compile + graph capture), so a miss is a ~17-minute
# user-visible stall. Cost and latency point the same way there, and the window stays at 1800s.
DEFAULT_SCALEDOWN_WINDOW_SECONDS = 1800
SCALEDOWN_WINDOW_SECONDS_BY_GPU: dict[str, int] = {
    # ~60s cold boot (small FP8 checkpoints, cached compile artifacts on the shared volume).
    "L4": 300,
    # ~120s cold boot (9B FP8 at 32k).
    "L40S": 420,
    # ~400s cold boot (27B FP8 at 32k, graph capture).
    "H100": 900,
    # ~1010s cold boot (35B bf16, 67 GiB + ~377s torch.compile). Break-even AND a ~17-min
    # user-visible stall on a miss both argue for keeping the full window here.
    "H200": 1800,
}


def scaledown_window_for(gpu: str) -> int:
    """Idle seconds before ``gpu``'s engine containers scale down (see the table above)."""
    return SCALEDOWN_WINDOW_SECONDS_BY_GPU.get(gpu, DEFAULT_SCALEDOWN_WINDOW_SECONDS)


# each advertised model and the cpu router keep one warm container plus one buffer container.
MIN_CONTAINERS = 1
BUFFER_CONTAINERS = 1
# router concurrency is per cpu replica and independent of the number of gpu engine replicas.
ROUTER_MAX_INPUTS = 36
ROUTER_TARGET_INPUTS = 27

# engines enable trust_remote_code, so their secret uses an allowlist: future credentials
# default to staying router-only. engines hydrate adapter state per request from the router-forwarded
# record; only the hf token is needed to download private base weights and adapters.
_ENGINE_SECRET_NAMES = ("HF_TOKEN",)
_RUNTIME_SECRET_NAMES = (
    *_ENGINE_SECRET_NAMES,
    "SERVING_DEPLOYMENT_MODE",
    "SERVING_CUSTOM_DOMAIN",
    "PLATFORM_BACKEND_URL",
    "FREESOLO_INTERNAL_KEY",
    # immutable deployment provenance and attempt identity used by the public readiness endpoint.
    "FREESOLO_DEPLOYMENT_SHA",
    "FREESOLO_DEPLOYMENT_ID",
    "SUPABASE_PROJECT_REF",
    "SUPABASE_PROJECT_REF_DEV",
    "SUPABASE_URL",
    "SUPABASE_SERVICE_ROLE_KEY",
    # only credentials and wiring are forwarded. autoscaling and engine configuration are constants.
)


def _runtime_values() -> dict[str, str]:
    return {name: value for name in _RUNTIME_SECRET_NAMES if (value := os.environ.get(name))}


def _secret_from(values: dict[str, str]) -> modal.Secret | None:
    return modal.Secret.from_dict(values) if values else None


def _runtime_secret() -> modal.Secret | None:
    return _secret_from(_runtime_values())


def _engine_secret() -> modal.Secret | None:
    return _secret_from(
        {name: value for name in _ENGINE_SECRET_NAMES if (value := os.environ.get(name))}
    )


runtime_secret = _runtime_secret()
runtime_secrets = [runtime_secret] if runtime_secret is not None else []

engine_secret = _engine_secret()
engine_secrets = [engine_secret] if engine_secret is not None else []

image = (
    # cuda 13, not 12.8: `serve-runtime` below pins vllm 0.23.0, which requires torch 2.11.0, whose
    # linux wheels depend on the -cu13 nvidia packages -- and vllm's own `_C` extension links
    # libcudart.so.13. a 12.8 base still deploys clean, because pip resolves the cuda 13 stack
    # regardless; the mismatch only surfaces later as `ImportError: libcudart.so.13` on a rented
    # gpu. `Dockerfile.serve` pairs this same pin with a cuda 13 base and asserts the linkage at
    # build time -- this image resolves the same bounds, so it needs the same pairing.
    modal.Image.from_registry(
        # digest-pinned so modal rebuilds use the exact cuda runtime validated here.
        "nvidia/cuda:13.0.0-devel-ubuntu22.04@sha256:1470d2d7904fac4e5cb3bdfd4993305c46d3ee76deb0213eaaf248e5cf9c7400",
        add_python="3.12",
    )
    .apt_install("build-essential", "git", "ninja-build")
    # the app's own pyproject went away with the move; flash's `serve-runtime` extra now carries the
    # same model-path bounds (vllm pinned exactly, transformers ranged), and `serving` carries the
    # app's own deps. the image resolves from these bounds, NOT uv.lock, so the bounds are the
    # contract -- see the notes on both extras in pyproject.toml.
    .pip_install_from_pyproject(
        str(REPO_DIR / "pyproject.toml"),
        optional_dependencies=["serve-runtime", "serving"],
    )
    # copy the fail-closed repair into the image before running it. this is a build-time source
    # backport, not a runtime hook: it verifies exact vllm 0.23.0 pre/post hashes without importing
    # vllm or torch. a separate pass rejects a successful no-op, then the script is removed.
    .add_local_file(
        str(REPO_DIR / "docker" / "patch_vllm_moe_lora.py"),
        remote_path="/root/patch_vllm_moe_lora.py",
        copy=True,
    )
    .run_commands(
        "python /root/patch_vllm_moe_lora.py && "
        "python /root/patch_vllm_moe_lora.py --verify && "
        "rm /root/patch_vllm_moe_lora.py"
    )
    # no in-engine kernel-patching hook is installed. under vllm v1 the model runs in a separate
    # enginecore process, so patches applied in this process never reach the model. a prior attempt's
    # extra cuda context and gpu self-tests also stole the post-init slack flashinfer needs for its
    # lazily allocated decode workspace. the build-time repair above runs before either process.
    # no quantization package is needed: fp8 weights and kv are built into vllm. the old bitsandbytes
    # qlora serving path is gone because nothing references it.
    .env(
        {
            "HF_HOME": HOSTING_CACHE_MOUNT,
            "HF_HUB_CACHE": f"{HOSTING_CACHE_MOUNT}/hub",
            "TRANSFORMERS_CACHE": f"{HOSTING_CACHE_MOUNT}/transformers",
            # vLLM's own cache root, on the SAME persistent volume as the weight caches above.
            # It defaults to ~/.cache/vllm (vllm/envs.py), which is ephemeral container storage, so
            # every replacement container re-compiled the model from scratch: vllm writes its
            # torch.compile artifacts to vllm_cache_root/torch_compile_cache/<hash>/
            # (vllm/compilation/backends.py), plus modelinfos/ and the gpu p2p cache. without the
            # persistent cache that cost is paid on every replacement or recovery cold start. it is the torch.compile +
            # graph-capture portion of the ~1010s 35B engine init measured for STARTUP_TIMEOUT_SECONDS.
            # vLLM's own startup benchmark DEFINES a cold start this way (benchmarks/startup.py
            # cold_startup() points VLLM_CACHE_ROOT at a fresh mkdtemp), so leaving it unset on
            # ephemeral storage reproduces that worst case involuntarily.
            # Reuse is safe by construction: the cache directory name is a sha256 over
            # [env_hash, config_hash, code_hash, compiler_hash], so a different vLLM version, engine
            # config, or model code lands in a DIFFERENT directory rather than loading a stale graph.
            # Production and development share HF_CACHE_VOLUME_NAME but Modal environment isolation
            # namespaces the volume, so the two deployments cannot read each other's entries.
            "VLLM_CACHE_ROOT": f"{HOSTING_CACHE_MOUNT}/vllm",
            # Disable the HF Xet/CAS checkpoint-download path. It intermittently fails engine init
            # with "CAS Client Error: Format error: I/O error: error decoding response body"
            # (huggingface_hub xet_get inside vLLM download_weights_from_hf) and crash-looped 27B FP8
            # serving boots (2 of 3 attempts). HF_HUB_DISABLE_XET=1 forces the deterministic HTTP
            # downloader; a matched re-serve with it set booted the engine and served cleanly. This
            # supersedes the prior HF_XET_HIGH_PERFORMANCE tuning, which is moot once Xet is disabled.
            "HF_HUB_DISABLE_XET": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            # Expandable segments let the CUDA allocator grow existing allocations rather than
            # fragmenting into many small blocks. Reduces peak VRAM and OOM frequency for
            # multi-LoRA serving where adapter load/unload creates allocation pressure.
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            # Keep DeepGEMM out of the FP8 fused-MoE backend race. The validated 35B path
            # leaves moe_backend unset/auto and lets vLLM select a working FP8-MoE backend; this
            # MoE-only env keeps the image away from the DeepGEMM path that previously crash-looped.
            "VLLM_MOE_USE_DEEP_GEMM": "0",
        }
    )
    # Ship the package under its REAL import path. This used to be
    # `add_local_dir(src, remote_path="/root/src")`, which was correct when the app was its own
    # repo and its modules imported each other as `src.X`. After the move they import each other
    # as `flash.serving.src.X`, and a bare `/root/src` tree cannot satisfy that -- the container
    # would raise `ModuleNotFoundError: No module named 'flash.serving'` on the first engine call,
    # long after `modal deploy` reported success. `add_local_python_source` mounts `flash/` into
    # /root (which is on the container PYTHONPATH), so the in-container import path matches the
    # one the test suite exercises. Only .py files are included, per the method's default `ignore`.
    .add_local_python_source("flash")
)

app = modal.App(APP_NAME, image=image)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)


# ``_LoraEngineImpl`` lives in ``flash.serving.src.engine.lora_engine`` (kept free of any ``modal``
# import so it registers nothing, and inside the ``flash`` package so the image's
# ``add_local_python_source("flash")`` ships it to the remote container under the same import path
# it has here), re-exported so ``_build_engine`` can subclass it.
# ---- One Modal LoraEngine class per GPU tier -----------------------------------------------------
# model_config is a pure-stdlib module (no heavy deps), so importing it at module scope is safe for
# `modal deploy` (which imports modal_app.py locally) — unlike the vllm/transformers imports, which
# stay lazy inside the engine methods.
from flash.serving.src.engine.dispatch import PreHeaderDispatchExpired  # noqa: E402
from flash.serving.src.engine.lora_engine import _LoraEngineImpl  # noqa: E402
from flash.serving.src.engine.model_config import (  # noqa: E402
    HostedTrafficPolicy,
    base_models,
    gpu_for,
    hosted_traffic_policy_for,
)


def _engine_concurrency(base_model: str) -> tuple[int, int]:
    policy = hosted_traffic_policy_for(base_model)
    return policy.max_inputs, policy.target_inputs


def _policy_contract(base_model: str) -> dict[str, Any]:
    policy = hosted_traffic_policy_for(base_model)
    return {
        "base_model": base_model,
        "gpu": gpu_for(base_model),
        "scaledown_window": scaledown_window_for(gpu_for(base_model)),
        "startup_timeout": STARTUP_TIMEOUT_SECONDS,
        "timeout": TIMEOUT_SECONDS,
        "min_containers": policy.min_containers,
        "buffer_containers": policy.buffer_containers,
        "max_inputs": policy.max_inputs,
        "target_inputs": policy.target_inputs,
    }


def _engine_class_name(base_model: str) -> str:
    """Deterministic identity for one exact model and every decorator-affecting policy value."""
    contract = _policy_contract(base_model)
    digest = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    model_name = base_model.rsplit("/", 1)[-1]
    safe_model = "".join(ch if ch.isalnum() else "_" for ch in model_name)[:32]
    return f"LoraEngine_{safe_model}_{digest}"


def _build_engine(base_model: str, class_name: str, policy: HostedTrafficPolicy) -> Any:
    """Register one Modal class for one exact advertised model and traffic policy."""
    gpu = gpu_for(base_model)
    exact_base_model = base_model

    class _Engine(_LoraEngineImpl):
        base_model: str = exact_base_model

        @modal.enter()
        async def load(self) -> None:
            await self._load()

        @modal.method()
        async def register(
            self,
            record_dict: dict[str, Any],
            deployment_generation: str | None = None,
        ) -> dict[str, Any]:
            return await self._register(record_dict, deployment_generation)

        @modal.method()
        async def generate(
            self,
            payload_dict: dict[str, Any],
            record_dict: dict[str, Any] | None = None,
            expected_checkpoint: str | None = None,
            generation_id: str | None = None,
            pre_header_dispatch_deadline: float | None = None,
        ) -> dict[str, Any]:
            return await self._generate(
                payload_dict,
                record_dict,
                expected_checkpoint,
                generation_id,
                pre_header_dispatch_deadline,
            )

        @modal.method(is_generator=True)
        async def stream_generate(
            self,
            payload_dict: dict[str, Any],
            record_dict: dict[str, Any] | None = None,
            expected_checkpoint: str | None = None,
            generation_id: str | None = None,
            pre_header_dispatch_deadline: float | None = None,
        ):
            async for event in self._stream_generate(
                payload_dict,
                record_dict,
                expected_checkpoint,
                generation_id,
                pre_header_dispatch_deadline,
            ):
                yield event

        @modal.method()
        async def stream_generate_call(
            self,
            payload_dict: dict[str, Any],
            record_dict: dict[str, Any] | None,
            expected_checkpoint: str | None,
            generation_id: str,
            dispatch_deadline_unix: float,
            queue_id: str,
            invocation_nonce: str,
        ) -> dict[str, Any]:
            from flash.serving.src.stream_channel.engine import stream_generate_call

            return await stream_generate_call(
                self,
                payload_dict,
                record_dict,
                expected_checkpoint,
                generation_id,
                dispatch_deadline_unix,
                queue_id,
                invocation_nonce,
            )

        @modal.method()
        async def unregister(
            self,
            adapter_id: str,
            expected_generation: str | None = None,
        ) -> dict[str, Any]:
            return await self._unregister(adapter_id, expected_generation)

        @modal.method()
        def health(self) -> dict[str, Any]:
            return self._health()

    # Record the GPU this class was actually pinned to (a plain class attribute, NOT a
    # modal.parameter() field). _health() reports THIS rather than re-deriving the tier from the base
    # model name, so a base model accidentally routed onto the wrong tier's class surfaces in health
    # instead of being masked by the expected-tier lookup.
    _Engine.pinned_gpu = gpu
    # Give the REAL class its distinct, module-level identity BEFORE decorating. A clean (no-``<locals>``)
    # __qualname__ satisfies Modal's global-scope validation, and binding it as a module attribute under
    # that name makes ``getattr(module, class_name)`` resolve the class Modal re-imports in the container.
    _Engine.__name__ = class_name
    _Engine.__qualname__ = class_name
    globals()[class_name] = _Engine
    engine = app.cls(
        gpu=gpu,
        secrets=engine_secrets,
        volumes={HOSTING_CACHE_MOUNT: hf_cache_volume},
        scaledown_window=scaledown_window_for(gpu),
        startup_timeout=STARTUP_TIMEOUT_SECONDS,
        timeout=TIMEOUT_SECONDS,
        min_containers=policy.min_containers,
        buffer_containers=policy.buffer_containers,
    )(
        modal.concurrent(
            max_inputs=policy.max_inputs,
            target_inputs=policy.target_inputs,
        )(_Engine)
    )
    # Rebind the module name to the decorated handle, matching the normal ``@app.cls class X`` pattern
    # where the module attribute ends up referring to the decorated class.
    globals()[class_name] = engine
    return engine


ENGINE_BY_MODEL: dict[str, Any] = {
    model: _build_engine(model, _engine_class_name(model), hosted_traffic_policy_for(model))
    for model in base_models()
}


def _engine_cls_for(base_model: str) -> Any:
    hosted_traffic_policy_for(base_model)
    return ENGINE_BY_MODEL[base_model]


def _remaining_pre_header_dispatch_time(deadline: float) -> float:
    remaining = deadline - time.time()
    if remaining <= 0:
        raise PreHeaderDispatchExpired("request expired before gpu generation began")
    return remaining


async def _await_task_before_deadline(task: asyncio.Task[Any], deadline: float) -> Any:
    done, _ = await asyncio.wait(
        {task},
        timeout=_remaining_pre_header_dispatch_time(deadline),
    )
    if task not in done:
        raise PreHeaderDispatchExpired("request expired before gpu generation began")
    return task.result()


async def _cancel_modal_call(call: Any) -> None:
    with contextlib.suppress(BaseException):
        await call.cancel.aio()


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    with contextlib.suppress(BaseException):
        task.result()


_MODAL_CLEANUP_TASKS: set[asyncio.Task[Any]] = set()


def _start_modal_call_cleanup(call: Any) -> None:
    cleanup = asyncio.create_task(_cancel_modal_call(call))
    _MODAL_CLEANUP_TASKS.add(cleanup)
    cleanup.add_done_callback(_MODAL_CLEANUP_TASKS.discard)


def _cancel_modal_call_when_spawned(spawn: asyncio.Task[Any]) -> None:
    def cancel_if_created(task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            return
        with contextlib.suppress(BaseException):
            call = task.result()
            _start_modal_call_cleanup(call)

    spawn.add_done_callback(cancel_if_created)


async def _spawn_modal_call(method: Any, deadline: float, *args: Any) -> Any:
    """Acquire a cancellable handle within the remaining pre-header dispatch budget."""
    spawn = asyncio.create_task(method.spawn.aio(*args))
    try:
        return await _await_task_before_deadline(spawn, deadline)
    except PreHeaderDispatchExpired:
        _cancel_modal_call_when_spawned(spawn)
        spawn.cancel()
        raise
    except asyncio.CancelledError:
        _cancel_modal_call_when_spawned(spawn)
        spawn.cancel()
        raise


async def _await_modal_call(call: Any, deadline: float) -> Any:
    """Await one spawned modal call within the remaining pre-header dispatch budget."""
    result = asyncio.create_task(call.get.aio())
    try:
        return await _await_task_before_deadline(result, deadline)
    except BaseException:
        result.cancel()
        result.add_done_callback(_consume_task_result)
        await _cancel_modal_call(call)
        raise


async def _close_remote_iterator(remote_stream: Any, iterator: Any) -> None:
    close = getattr(iterator, "aclose", None)
    if close is None:
        close = getattr(remote_stream, "aclose", None)
    if close is not None:
        with contextlib.suppress(BaseException):
            await close()


class _ModalEnginePool:
    """Dispatch each model directly to its dedicated Modal engine class."""

    @staticmethod
    def _record_payload(record: Any) -> dict[str, Any]:
        payload = record.model_dump(by_alias=True)
        generation = getattr(record, "deployment_generation", None)
        if generation is not None:
            payload["deployment_generation"] = generation
        return payload

    async def generate(
        self,
        base_model: str,
        payload: Any,
        record: Any,
        *,
        expected_checkpoint: str | None = None,
    ) -> dict[str, Any]:
        generation_id = payload.generation_id
        if not generation_id:
            raise RuntimeError("generation id is required before modal dispatch")
        deadline = payload._pre_header_dispatch_deadline
        if deadline is None:
            raise RuntimeError("pre-header dispatch deadline is required before modal dispatch")
        engine = _engine_cls_for(base_model)()
        call = await _spawn_modal_call(
            engine.generate,
            deadline,
            payload.model_dump(by_alias=True),
            self._record_payload(record),
            expected_checkpoint,
            generation_id,
            deadline,
        )
        return await _await_modal_call(call, deadline)

    async def stream_generate(
        self,
        base_model: str,
        payload: Any,
        record: Any,
        *,
        expected_checkpoint: str | None = None,
    ):
        generation_id = payload.generation_id
        if not generation_id:
            raise RuntimeError("generation id is required before modal dispatch")
        deadline = payload._pre_header_dispatch_deadline
        if deadline is None:
            raise RuntimeError("pre-header dispatch deadline is required before modal dispatch")
        engine = _engine_cls_for(base_model)()
        remote_stream = engine.stream_generate.remote_gen.aio(
            payload.model_dump(by_alias=True),
            self._record_payload(record),
            expected_checkpoint,
            generation_id,
            deadline,
        )
        iterator = remote_stream.__aiter__()
        first = asyncio.create_task(anext(iterator))
        try:
            try:
                yield await _await_task_before_deadline(first, deadline)
            except StopAsyncIteration:
                return
            except BaseException:
                first.cancel()
                first.add_done_callback(_consume_task_result)
                raise
            async for event in iterator:
                yield event
        finally:
            await _close_remote_iterator(remote_stream, iterator)

    def stream_generate_cancellable(
        self,
        base_model: str,
        payload: Any,
        record: Any,
        *,
        expected_checkpoint: str | None = None,
        dispatch_deadline_unix: float | None = None,
    ) -> Any:
        """Build the additive channel transport without changing the rolling default."""
        from flash.serving.src.stream_channel.client import CancellableStreamChannel

        generation_id = payload.generation_id
        if not generation_id:
            raise RuntimeError("generation id is required before modal dispatch")
        engine = _engine_cls_for(base_model)(base_model=base_model)
        deadline = (
            dispatch_deadline_unix
            if dispatch_deadline_unix is not None
            else time.time() + ROUTER_TIMEOUT_SECONDS
        )
        return CancellableStreamChannel(
            spawn_method=engine.stream_generate_call,
            payload_dict=payload.model_dump(by_alias=True),
            record_dict=self._record_payload(record),
            expected_checkpoint=expected_checkpoint,
            generation_id=generation_id,
            dispatch_deadline_unix=deadline,
            invocation_nonce=uuid.uuid4().hex,
        )

    async def register(self, base_model: str, record: Any) -> None:
        engine = _engine_cls_for(base_model)()
        await engine.register.remote.aio(
            self._record_payload(record),
            getattr(record, "deployment_generation", None),
        )

    async def unregister(
        self,
        base_model: str,
        adapter_id: str,
        expected_generation: str | None = None,
    ) -> None:
        engine = _engine_cls_for(base_model)()
        await engine.unregister.remote.aio(adapter_id, expected_generation)


def _build_usage_outbox(settings: Any) -> Any:
    """Build mandatory durable accounting for the hosted serving application."""
    from flash.serving.src.accounting.usage_outbox import DurableUsageOutbox

    return DurableUsageOutbox(
        settings,
        worker_id=f"serving-{settings.deployment_id}-{os.getpid()}-{uuid.uuid4().hex}",
    )


# Short-TTL positive-authorization cache + single-flight for the serving->backend authorize call.
# That call runs on EVERY external chat request; an eval fires many requests with the SAME
# (api_key, adapter_id), so without this a thundering herd of identical auth lookups stampedes the
# backend's Supabase auth path — it 5xx's under the load, which the caller sees as a serving error
# (the intermittent HTTP 502 burst observed under concurrent evals). Coalescing concurrent identical
# lookups into ONE backend call and reusing a SUCCESSFUL result for a short window removes that
# self-inflicted load (no retry). Only ALLOW results are cached; denials (401/402/403) and transient
# failures are never cached, so a revoked key / drained balance is re-checked within at most the TTL.
# Hardcoded (no deploy-time knobs), consistent with src/settings.py.
_AUTH_CACHE_TTL_SECONDS = 60.0
_AUTH_CACHE_MAX_ENTRIES = 4096


def _build_chat_authorizer(settings: Any) -> Any:
    """Build the serving->backend chat authorizer, or None when not configured.

    Returns an async ``authorize(api_key, adapter_id)`` that POSTs to the backend's
    ``/api/serving/authorize`` (authenticated with the shared internal key) and raises an
    ``HTTPException`` (401/402/403/503) when the Freesolo API key's org does not own the adapter or
    the backend can't be reached. Shares one persistent ``httpx.AsyncClient`` to keep the backend
    connection warm across calls. Chat auth is always enforced, so this must be configured in any
    real deployment (backend URL + internal key) — when it returns None, non-internal chat fails
    closed (503).

    Successful authorizations are cached per API-key digest and adapter ID for
    ``_AUTH_CACHE_TTL_SECONDS`` and concurrent identical lookups are coalesced into a single backend
    call, so an eval's many same-key requests don't stampede the backend auth path into transient
    5xx failures. Any backend 5xx maps to a retryable 503 (never a 502) so it isn't read as a
    permanent upstream error.
    """
    base = (settings.backend_url or "").rstrip("/")
    key = settings.internal_key
    if not base or not key:
        return None
    url = f"{base}/api/serving/authorize"

    import hashlib
    import time

    import httpx
    from fastapi import HTTPException, status

    _client = httpx.AsyncClient(timeout=10.0, headers={"Authorization": f"Bearer {key}"})

    # (api_key_sha256, adapter_id) -> (expires_at_monotonic, org_id); populated only on a successful
    # authorization. Per-router-container in-memory (each container reduces its own backend load).
    _cache: dict[tuple[str, str], tuple[float, str | None]] = {}
    # In-flight single-flight tasks so concurrent identical misses share ONE backend call.
    _inflight: dict[tuple[str, str], asyncio.Task[str | None]] = {}

    async def _authorize_backend(api_key: str, adapter_id: str) -> "str | None":
        try:
            resp = await _client.post(url, json={"apiKey": api_key, "adapterId": adapter_id})
        except Exception as exc:  # backend unreachable -> fail closed, never serve unauthorized
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, "serving auth backend unreachable"
            ) from exc
        if resp.status_code == 200:
            # Return the authorized billing org so the router can meter a base-model serve to the
            # caller (a base model has no adapter owner). A malformed response cannot authorize an
            # external request because that would let base-model usage escape billing.
            try:
                org_id = resp.json().get("orgId")
            except Exception as exc:
                raise HTTPException(
                    status.HTTP_503_SERVICE_UNAVAILABLE,
                    "serving auth backend returned malformed response",
                ) from exc
            if not isinstance(org_id, str) or not org_id:
                raise HTTPException(
                    status.HTTP_503_SERVICE_UNAVAILABLE,
                    "serving auth backend returned malformed response",
                )
            return org_id
        if resp.status_code == 401:
            # The backend returns 401 for BOTH an invalid *user* key (our authorize route, body
            # code "invalid_api_key") AND a rejected serving *machine* bearer (require_internal_token,
            # decided before the user key is even read). Only the former is the caller's fault; the
            # latter is a serving misconfiguration we must not report as the user's bad key.
            code = ""
            try:
                detail = resp.json().get("detail")
                if isinstance(detail, dict):
                    code = detail.get("code") or ""
            except Exception:  # a non-JSON 401 is an internal-auth failure (below)
                code = ""
            if code == "invalid_api_key":
                raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid Freesolo API key")
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, "serving auth is misconfigured"
            )
        # Org has no budget to serve (zero balance and no card+auto-topup). The backend gates this
        # before generation so a broke org can't serve for free; surface it as a clean 402 rather
        # than letting it fall through to the retryable 503 below.
        if resp.status_code == 402:
            raise HTTPException(
                status.HTTP_402_PAYMENT_REQUIRED,
                "insufficient balance: top up or enable auto-topup to serve this model",
            )
        # Unknown adapter (404) and org mismatch (403) both collapse to 403 so we don't leak
        # which adapters exist to an unauthorized caller.
        if resp.status_code in (403, 404):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN, "API key is not authorized for this adapter"
            )
        # Any backend 5xx (500/502/503/504) or other unexpected status is a transient auth-lookup
        # infra failure (Supabase down, backend overloaded). Surface it as a RETRYABLE 503, never a
        # 502: a client/load-balancer must be free to retry rather than treat it as a permanent
        # upstream error. The cache+single-flight above is what makes this failure rare under load.
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "serving auth backend unavailable")

    def _prune(now: float) -> None:
        for expired in [k for k, (exp, _) in _cache.items() if exp <= now]:
            _cache.pop(expired, None)
        if len(_cache) > _AUTH_CACHE_MAX_ENTRIES:
            n_drop = len(_cache) - _AUTH_CACHE_MAX_ENTRIES
            for k in sorted(_cache, key=lambda c: _cache[c][0])[:n_drop]:
                _cache.pop(k, None)

    async def _authorize_and_cache(
        ck: tuple[str, str], api_key: str, adapter_id: str
    ) -> "str | None":
        try:
            org = await _authorize_backend(api_key, adapter_id)
        finally:
            # Drop the single-flight slot regardless of outcome so a failed lookup re-checks next
            # time (failures are never cached) and a success can be re-driven once the cache expires.
            _inflight.pop(ck, None)
        now = time.monotonic()
        # Cache the ALLOW only — reached iff _authorize_backend did not raise above.
        _cache[ck] = (now + _AUTH_CACHE_TTL_SECONDS, org)
        _prune(now)
        return org

    async def authorize(api_key: str, adapter_id: str) -> "str | None":
        # do not retain raw user credentials as live dict keys beyond the request that supplied them.
        ck = (hashlib.sha256(api_key.encode("utf-8")).hexdigest(), adapter_id)
        cached = _cache.get(ck)
        now = time.monotonic()
        if cached is not None:
            if cached[0] > now:
                return cached[1]
            _cache.pop(ck, None)
        task = _inflight.get(ck)
        if task is None:
            task = asyncio.ensure_future(_authorize_and_cache(ck, api_key, adapter_id))
            # Retrieve the task's outcome even if every awaiter is cancelled (all identical clients
            # disconnect), so an orphaned single-flight failure doesn't warn "exception never retrieved".
            task.add_done_callback(lambda t: t.cancelled() or t.exception())
            _inflight[ck] = task
        return await asyncio.shield(task)

    authorize.aclose = _client.aclose
    return authorize


def _base_model_records() -> list:
    """One open, no-LoRA record per served base model, seeded into the router IN MEMORY.

    Each base model already has its weights loaded by the per-model engine, so a base serve needs no
    LoRA download and no registration/DB row — we just make the base model addressable by name. These
    records are never persisted and never org-owned; ``serve_base_model`` marks them so the engine
    generates against the base weights (lora_request=None) and the router serves them openly.
    """
    from flash.serving.src.engine.model_config import base_models
    from flash.serving.src.io.schemas import AdapterRecord

    return [
        AdapterRecord(
            adapter_id=m,
            repo_id=m,
            base_model=m,
            serve_base_model=True,
            thinking=False,
            org_id=None,
            status="ready",
        )
        for m in base_models()
    ]


@app.function(
    secrets=runtime_secrets,
    min_containers=MIN_CONTAINERS,
    buffer_containers=BUFFER_CONTAINERS,
    timeout=ROUTER_TIMEOUT_SECONDS,
)
@modal.concurrent(
    max_inputs=ROUTER_MAX_INPUTS,
    target_inputs=ROUTER_TARGET_INPUTS,
)
@modal.asgi_app(
    label=APP_NAME,
    # Claim the branded domain only when one is configured for this workspace; empty => omit it so the
    # app deploys on its default *.modal.run url (see SERVING_CUSTOM_DOMAIN).
    custom_domains=[SERVING_CUSTOM_DOMAIN] if SERVING_CUSTOM_DOMAIN else None,
)
def router():
    from flash.serving.src.http.router import AdapterRouter, build_serving_app
    from flash.serving.src.store import settings as cfg
    from flash.serving.src.store.persistence import get_adapter, load_adapters
    from flash.serving.src.store.settings import Settings

    settings = Settings()
    # Seed the base-model records alongside the persisted LoRA adapters so every base model is
    # reachable by name with no adapter. Re-seed on every reload (hydrate replaces the registry).
    adapter_router = AdapterRouter(load_adapters(settings) + _base_model_records())
    pool = _ModalEnginePool()
    return build_serving_app(
        pool,
        adapter_router,
        internal_key=settings.internal_key,
        deployment_sha=settings.deployment_sha,
        deployment_id=settings.deployment_id,
        # Reload on a routing miss so a stale router still resolves a just-registered adapter.
        reload_records=lambda: load_adapters(settings) + _base_model_records(),
        lookup_record=lambda adapter_id: get_adapter(adapter_id, settings),
        reload_interval_seconds=cfg.RELOAD_INTERVAL_SECONDS,
        # durable capture is required in configured serving deployments.
        usage_store=_build_usage_outbox(settings),
        # External chat auth is ALWAYS enforced: a request needs a Freesolo API key whose org owns
        # the adapter (the backend authorizes), or the shared internal key to bypass. The authorizer
        # must be wired (backend URL + internal key) or non-internal chat fails closed.
        chat_authorizer=_build_chat_authorizer(settings),
    )


@app.local_entrypoint()
def start_all(base_model: str | None = None) -> None:
    """Explicitly boot deployed gpu engines and wait for them to report healthy.

    Normal deploys keep one warm container and one buffer container for every advertised model.
    This manual diagnostic verifies one model with ``--base-model`` or every catalog model without
    changing that Modal-managed scaling policy.
    """
    from flash.serving.src.engine.model_config import base_models

    started = {}
    failures: list[str] = []
    models = [base_model] if base_model else list(base_models())
    for bm in models:
        # Each base model's engine lives on its (GPU tier, concurrency) class.
        engine = modal.Cls.from_name(APP_NAME, _engine_class_name(bm))
        instance = engine()
        started[bm] = instance.health.spawn()
    for bm, handle in started.items():
        try:
            print(f"started {bm}: {handle.get(timeout=1800)}", flush=True)
        except Exception as exc:  # report per-model startup failures, keep going
            failure = f"{bm}: {type(exc).__name__}: {exc}"
            failures.append(failure)
            print(f"FAILED  {failure}", flush=True)
    if failures:
        raise RuntimeError(
            f"serving warm failed for {len(failures)} model(s): {'; '.join(failures)}"
        )
