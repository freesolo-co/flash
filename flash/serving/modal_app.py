"""Modal app: multi-LoRA serving with one immutable engine class per base model.

Each exact-model engine owns one GPU container and shares that model's adapters via
``enable_lora``. ``router`` is the CPU front door that dispatches to the matching engine.
"""

import asyncio
import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import modal

from flash.serving.src.capacity import (
    CAPACITY_POLL_INTERVAL_SECONDS,
    CAPACITY_REFRESH_TIMEOUT_SECONDS,
    CAPACITY_SNAPSHOT_MAX_AGE_SECONDS,
    CapacitySnapshot,
    fixed_local_active_limit,
)
from flash.serving.src.lora_engine import _LoraEngineImpl
from flash.serving.src.model_config import (
    HostedTrafficPolicy,
    base_models,
    configured_router_async_capacity,
    gpu_for,
    hosted_traffic_policy_for,
)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - only relevant for global modal installs
    load_dotenv = None


SERVING_DIR = Path(__file__).resolve().parent
# flash/serving/ -> flash/ -> repo root. the app used to sit one level below its repo root; after the
# move it is two, so walking up a single parent would look for .env inside the flash package and load
# nothing, leaving a production deploy silently unconfigured.
REPO_DIR = SERVING_DIR.parent.parent

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
_USAGE_REPORT_RETRY_DELAYS_SECONDS = (0.1, 0.25, 0.5)
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


# hosted traffic policy is derived per model from its real max_num_seqs. every advertised model keeps
# one warm engine, scales to at most two, and exposes exactly two application-side queue positions.
# the cpu router remains a singleton and is bounded to the aggregate model hard limits plus queues.

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
    # No in-engine kernel-patching hook is installed in this image (see the note at the engine call
    # site): under vLLM V1 the model runs in a separate EngineCore process, so patches applied in THIS
    # process never reach the model, and a prior attempt's extra CUDA context + GPU self-tests stole
    # the post-init slack the engine needs for FlashInfer's lazily-allocated decode workspace
    # (2026-07-05 35B outage: first real request OOM-killed the engine).
    # No quantization package needed: FP8 (weights + KV) is built into vLLM and quantizes the bf16
    # checkpoint online at load (see settings.QUANTIZATION). The old bitsandbytes QLoRA-serving path
    # is gone — bitsandbytes was the slowest inference quant and nothing references it anymore.
    .env(
        {
            "HF_HOME": HOSTING_CACHE_MOUNT,
            "HF_HUB_CACHE": f"{HOSTING_CACHE_MOUNT}/hub",
            "TRANSFORMERS_CACHE": f"{HOSTING_CACHE_MOUNT}/transformers",
            # vllm's own cache root, on the same persistent volume as the weight caches above.
            # normal deploys keep one warm container and may scale each model to two containers, so
            # normal requests avoid container cold startup. persisting the cache also lets replacement
            # and second containers reuse torch.compile artifacts, modelinfos, and the gpu p2p cache
            # instead of recompiling and repeating graph capture during engine initialization.
            # vllm writes compiled artifacts under vllm_cache_root/torch_compile_cache/<hash>/
            # (vllm/compilation/backends.py). its startup benchmark defines a cold startup by pointing
            # vllm_cache_root at a fresh mkdtemp (benchmarks/startup.py), so leaving the cache on
            # ephemeral storage involuntarily repeats the worst-case initialization path.
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


# ``_LoraEngineImpl`` is import-light and registers no modal resources. the serving modules imported
# above keep vllm and transformers lazy, while the image ships the same ``flash`` package path used
# locally. each generated modal subclass below bakes one exact model and traffic policy.


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
        "max_containers": policy.max_containers,
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
        ) -> dict[str, Any]:
            return await self._generate(payload_dict, record_dict, expected_checkpoint)

        @modal.method(is_generator=True)
        async def stream_generate(
            self,
            payload_dict: dict[str, Any],
            record_dict: dict[str, Any] | None = None,
            expected_checkpoint: str | None = None,
        ):
            async for event in self._stream_generate(
                payload_dict, record_dict, expected_checkpoint
            ):
                yield event

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

    _Engine.pinned_gpu = gpu
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
        max_containers=policy.max_containers,
        buffer_containers=policy.buffer_containers,
    )(
        modal.concurrent(
            max_inputs=policy.max_inputs,
            target_inputs=policy.target_inputs,
        )(_Engine)
    )
    globals()[class_name] = engine
    return engine


ENGINE_BY_MODEL: dict[str, Any] = {
    model: _build_engine(model, _engine_class_name(model), hosted_traffic_policy_for(model))
    for model in base_models()
}


def _engine_cls_for(base_model: str) -> Any:
    hosted_traffic_policy_for(base_model)
    return ENGINE_BY_MODEL[base_model]


def _capacity_deployment_identity(base_model: str) -> str:
    return f"modal-app-class:{APP_NAME}/{_engine_class_name(base_model)}"


class _ModalEnginePool:
    """Dispatch and live capacity for each model's dedicated Modal engine class."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._capacity: dict[str, CapacitySnapshot] = {}
        self._capacity_locks: dict[str, asyncio.Lock] = {}
        self._capacity_changed: Callable[[str], None] | None = None

    def bind_capacity_changed(self, callback: Callable[[str], None]) -> None:
        self._capacity_changed = callback

    async def capacity_snapshot(self, model: str, observed_local_active: int) -> CapacitySnapshot:
        now = self._clock()
        current = self._capacity.get(model)
        if current is not None and current.is_fresh(
            now, max_age_seconds=CAPACITY_POLL_INTERVAL_SECONDS
        ):
            return current
        lock = self._capacity_locks.setdefault(model, asyncio.Lock())
        async with lock:
            now = self._clock()
            current = self._capacity.get(model)
            if current is not None and current.is_fresh(
                now, max_age_seconds=CAPACITY_POLL_INTERVAL_SECONDS
            ):
                return current
            snapshot = await self._fetch_capacity_snapshot(model, observed_local_active)
            self._capacity[model] = snapshot
            if self._capacity_changed is not None:
                self._capacity_changed(model)
            return snapshot

    async def _fetch_capacity_snapshot(
        self, model: str, observed_local_active: int
    ) -> CapacitySnapshot:
        observed_at = self._clock()
        deployment_identity = _capacity_deployment_identity(model)
        policy = hosted_traffic_policy_for(model)
        hard_limit = policy.max_inputs * policy.max_containers
        try:
            engine = _engine_cls_for(model)()
            stats = await asyncio.wait_for(
                engine.generate.get_current_stats.aio(),
                timeout=CAPACITY_REFRESH_TIMEOUT_SECONDS,
            )
            counts = {
                "total_runners": stats.num_total_runners,
                "running_inputs": stats.num_running_inputs,
                "input_headroom": stats.input_headroom,
                "backlog": stats.backlog,
            }
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in counts.values()
            ):
                raise ValueError("invalid Modal function stats")
            if counts["total_runners"] <= 0:
                return CapacitySnapshot.unavailable_snapshot(
                    model,
                    deployment_identity,
                    observed_at,
                    "no_warm_runners",
                    observed_local_active=observed_local_active,
                    **counts,
                )
            return CapacitySnapshot(
                model=model,
                deployment_identity=deployment_identity,
                observed_at=observed_at,
                observed_local_active=observed_local_active,
                local_active_limit=fixed_local_active_limit(
                    observed_local_active,
                    counts["running_inputs"],
                    counts["input_headroom"],
                    hard_limit,
                ),
                **counts,
            )
        except Exception:
            return CapacitySnapshot.unavailable_snapshot(
                model,
                deployment_identity,
                observed_at,
                "stats_unavailable",
                observed_local_active=observed_local_active,
            )

    def current_dispatch_capacity(self, model: str) -> int:
        snapshot = self._capacity.get(model)
        now = self._clock()
        deployment_identity = _capacity_deployment_identity(model)
        if snapshot is None or not snapshot.is_dispatchable(
            now,
            model=model,
            deployment_identity=deployment_identity,
            max_age_seconds=CAPACITY_SNAPSHOT_MAX_AGE_SECONDS,
        ):
            return 0
        return snapshot.local_active_limit

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
        engine = _engine_cls_for(base_model)()
        return await engine.generate.remote.aio(
            payload.model_dump(by_alias=True),
            self._record_payload(record),
            expected_checkpoint,
        )

    async def stream_generate(
        self,
        base_model: str,
        payload: Any,
        record: Any,
        *,
        expected_checkpoint: str | None = None,
    ):
        engine = _engine_cls_for(base_model)()
        remote_stream = engine.stream_generate.remote_gen.aio(
            payload.model_dump(by_alias=True),
            self._record_payload(record),
            expected_checkpoint,
        )
        try:
            async for event in remote_stream:
                yield event
        finally:
            close = getattr(remote_stream, "aclose", None)
            if close is not None:
                await close()

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


def _build_usage_reporter(settings: Any) -> Any:
    """Build the serving->backend usage reporter, or None when not configured.

    Returns an async ``report(usage)`` that POSTs one request's token + GPU-time usage to the
    backend's ``/api/billing/serving-usage`` (authenticated with the shared internal key).
    The router calls it as a managed detached task, so a failure here never affects serving.
    When ``internal_key`` is unset, returns None to disable reporting entirely.

    A single persistent ``httpx.AsyncClient`` is shared across all report calls: it keeps the
    TCP/TLS connection to the backend alive between calls, eliminating per-report handshake
    overhead (typically 20-80 ms per TLS round-trip on the first call of each burst).
    """
    base = (settings.backend_url or "").rstrip("/")
    key = settings.internal_key
    if not base or not key:
        return None
    url = f"{base}/api/billing/serving-usage"

    import httpx

    _client = httpx.AsyncClient(
        timeout=10.0,
        headers={"Authorization": f"Bearer {key}"},
    )

    async def report(usage: dict[str, Any]) -> None:
        stable_request_id = bool(str(usage.get("requestId") or "").strip())
        for attempt in range(len(_USAGE_REPORT_RETRY_DELAYS_SECONDS) + 1):
            try:
                resp = await _client.post(url, json=usage)
            except httpx.TransportError:
                if not stable_request_id or attempt >= len(_USAGE_REPORT_RETRY_DELAYS_SECONDS):
                    raise
            else:
                status_code = getattr(resp, "status_code", 200)
                retryable_status = status_code in {408, 429} or status_code >= 500
                if not retryable_status or not stable_request_id:
                    resp.raise_for_status()
                    return
                if attempt >= len(_USAGE_REPORT_RETRY_DELAYS_SECONDS):
                    resp.raise_for_status()
                    return
            await asyncio.sleep(_USAGE_REPORT_RETRY_DELAYS_SECONDS[attempt])

    # Expose the client's aclose so the app can close it on shutdown (avoids leaked sockets /
    # "Unclosed client" ResourceWarnings); build_serving_app wires this to a FastAPI shutdown hook.
    report.aclose = _client.aclose
    return report


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


ROUTER_ASYNC_CAPACITY = configured_router_async_capacity()


@app.function(
    secrets=runtime_secrets,
    min_containers=1,
    max_containers=1,
    buffer_containers=0,
    timeout=ROUTER_TIMEOUT_SECONDS,
)
@modal.concurrent(
    max_inputs=ROUTER_ASYNC_CAPACITY,
    target_inputs=ROUTER_ASYNC_CAPACITY,
)
@modal.asgi_app(
    label=APP_NAME,
    # Claim the branded domain only when one is configured for this workspace; empty => omit it so the
    # app deploys on its default *.modal.run url (see SERVING_CUSTOM_DOMAIN).
    custom_domains=[SERVING_CUSTOM_DOMAIN] if SERVING_CUSTOM_DOMAIN else None,
)
def router():
    from flash.serving.src import settings as cfg
    from flash.serving.src.persistence import get_adapter
    from flash.serving.src.readiness import load_routing_snapshot
    from flash.serving.src.router import AdapterRouter, build_serving_app
    from flash.serving.src.settings import Settings

    settings = Settings()
    try:
        initial_records = load_routing_snapshot(settings)
    except Exception as exc:
        # readiness is fail-closed, but existing persisted adapter routing remains available.
        from flash.serving.src.persistence import load_adapters

        print(f"hosted model readiness startup skipped: {exc!r}", flush=True)
        initial_records = load_adapters(settings)
    adapter_router = AdapterRouter(initial_records, require_base_qualification=True)
    pool = _ModalEnginePool()
    return build_serving_app(
        pool,
        adapter_router,
        internal_key=settings.internal_key,
        deployment_sha=settings.deployment_sha,
        deployment_id=settings.deployment_id,
        # one atomic storage snapshot drives both persisted adapters and qualified base models.
        reload_records=lambda: load_routing_snapshot(settings),
        lookup_record=lambda adapter_id: get_adapter(adapter_id, settings),
        reload_interval_seconds=cfg.RELOAD_INTERVAL_SECONDS,
        # Meter each generation to the backend (fire-and-forget); None disables it.
        usage_reporter=_build_usage_reporter(settings),
        # External chat auth is ALWAYS enforced: a request needs a Freesolo API key whose org owns
        # the adapter (the backend authorizes), or the shared internal key to bypass. The authorizer
        # must be wired (backend URL + internal key) or non-internal chat fails closed.
        chat_authorizer=_build_chat_authorizer(settings),
        capacity_provider=pool,
    )


@app.local_entrypoint()
def start_all(base_model: str | None = None) -> None:
    """Explicitly boot deployed gpu engines and wait for them to report healthy.

    Normal deploys keep one warm container for every advertised model and may scale each model to
    two containers. This manual diagnostic verifies one model with ``--base-model`` or every catalog
    model without changing that one-warm/two-max policy.
    """
    from flash.serving.src.model_config import base_models

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
