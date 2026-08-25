"""Modal app: multi-LoRA serving with one GPU container per base model. ``LoraEngine`` is a
Modal class parametrized by ``base_model`` (one GPU per base model, sharing its adapters via
``enable_lora``); ``router`` is the CPU front door that dispatches to the right engine. See
README to deploy.
"""

import asyncio
import os
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


# gpu model engines scale to zero. inference and adapter registration remote calls start the matching
# parameter-bound engine on demand.
MIN_CONTAINERS = 0
# no autoscaling cap per base-model engine; modal adds capacity as concurrency demands.
MAX_CONTAINERS = None
# Concurrent requests packed onto one base-model GPU before Modal autoscales a new (costly) one.
# A real-GPU sweep (scripts/gpu_canary.py::sweep_concurrency on A10G/Qwen2.5-1.5B) showed vLLM
# throughput scaling near-linearly with no saturation through 128 concurrent, while TTFT stayed
# <60 ms — so the old default of 32 left ~2x of each GPU idle. 64 packs ~2.3x the throughput per
# GPU for ~+14% per-request latency (now blunted by the fp8 KV cache, which is on for every base —
# see settings.KV_CACHE_DTYPE), roughly halving GPU count for the same load.
# Per-engine concurrency is sized by _engine_concurrency from each model's max_num_seqs; these are
# the router/global ceiling. TARGET_INPUTS auto-derives to 48 (= 64*3//4).
MAX_INPUTS = 64
TARGET_INPUTS = max(1, MAX_INPUTS * 3 // 4)

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
            # vLLM's own cache root, on the SAME persistent volume as the weight caches above.
            # It defaults to ~/.cache/vllm (vllm/envs.py), which is ephemeral container storage, so
            # every scale-from-zero re-compiled the model from scratch: vLLM writes its torch.compile
            # artifacts to VLLM_CACHE_ROOT/torch_compile_cache/<hash>/ (vllm/compilation/backends.py),
            # plus modelinfos/ and the GPU p2p cache. With MIN_CONTAINERS = 0 and a 30-minute
            # scaledown window that cost is paid on every cold start — it is the torch.compile +
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


# ``_LoraEngineImpl`` lives in ``flash.serving.src.lora_engine`` (kept free of any ``modal`` import
# so it registers nothing, and inside the ``flash`` package so the image's
# ``add_local_python_source("flash")`` ships it to the remote container under the same import path
# it has here), re-exported so ``_build_engine`` can subclass it.
from flash.serving.src.engine.lora_engine import _LoraEngineImpl  # noqa: E402

# ---- One Modal LoraEngine class per GPU tier -----------------------------------------------------
# model_config is a pure-stdlib module (no heavy deps), so importing it at module scope is safe for
# `modal deploy` (which imports modal_app.py locally) — unlike the vllm/transformers imports, which
# stay lazy inside the engine methods.
from flash.serving.src.engine.model_config import (  # noqa: E402
    base_models,
    engine_overrides_for,
    gpu_for,
)


def _engine_concurrency(base_model: str) -> tuple[int, int]:
    """(max_inputs, target_inputs) sized to the model's REAL vLLM concurrency (``max_num_seqs``).

    Modal's ``max_inputs`` is how many requests it packs onto ONE container before it must add
    another. If it far exceeds the engine's ``max_num_seqs`` (e.g. the global 64 on the 35B, which
    decodes only 8 at a time), Modal piles requests 9..64 INSIDE the container instead of autoscaling
    — high latency and no scale-out until ~target_inputs are packed. So cap ``max_inputs`` near the
    engine's capacity with a small boot buffer (2x, so a cold-booting replacement doesn't reject
    bursts), bounded by the global ``MAX_INPUTS``; scale out at 3/4 of that. Models that leave
    ``max_num_seqs`` at the vLLM default keep the global sizing."""
    seqs = int(engine_overrides_for(base_model).get("max_num_seqs", MAX_INPUTS))
    max_inputs = max(8, min(MAX_INPUTS, seqs * 2))
    target_inputs = max(1, max_inputs * 3 // 4)
    return max_inputs, target_inputs


def _engine_class_name(gpu: str, max_inputs: int) -> str:
    """Deterministic, Modal-safe class name for a (GPU tier, concurrency) engine — e.g.
    ('A100-80GB', 16) -> 'LoraEngine_A100_80GB_c16'. The concurrency is part of the identity because
    ``modal.concurrent`` is fixed per class, so tiers sharing a GPU but needing different max_inputs
    must be distinct classes."""
    base = "LoraEngine_" + "".join(ch if ch.isalnum() else "_" for ch in gpu)
    return f"{base}_c{max_inputs}"


def _build_engine(
    gpu: str,
    class_name: str,
    max_inputs: int,
    target_inputs: int,
) -> Any:
    """Register one Modal ``@app.cls`` LoraEngine pinned to ``gpu``.

    Modal fixes a class's GPU at decoration time, so the A100-80GB 35B model and the L4 models need
    distinct classes. The Modal entrypoints are defined fresh here (so each class owns its own method
    objects) and forward to the shared ``_``-prefixed impl on ``_LoraEngineImpl``.

    Class identity (name/qualname + module-global binding) is fixed BEFORE the class-level decorators
    run. With the real Modal SDK ``@modal.concurrent`` returns a wrapper that holds the user class
    separately, so renaming the bound name AFTER the decorator would rename the wrapper, not the class
    Modal registers — every tier would then register under ``_Engine`` and the ``<locals>`` qualname
    would fail Modal's global-scope validation. So we apply ``modal.concurrent``/``app.cls`` in call
    form on an already-renamed, already-module-global class."""

    class _Engine(_LoraEngineImpl):
        base_model: str = modal.parameter()

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
        min_containers=MIN_CONTAINERS,
        max_containers=MAX_CONTAINERS,
    )(modal.concurrent(max_inputs=max_inputs, target_inputs=target_inputs)(_Engine))
    # Rebind the module name to the decorated handle, matching the normal ``@app.cls class X`` pattern
    # where the module attribute ends up referring to the decorated class.
    globals()[class_name] = engine
    return engine


def _engine_key(base_model: str) -> tuple[str, int]:
    """(gpu, max_inputs) — the identity of the engine CLASS a base model runs on."""
    return gpu_for(base_model), _engine_concurrency(base_model)[0]


def _distinct_engine_keys() -> list[tuple[str, int]]:
    """Distinct (gpu, max_inputs) engine classes needed across the catalog, order-stable."""
    keys: dict[tuple[str, int], None] = {}
    for bm in base_models():
        keys.setdefault(_engine_key(bm), None)
    return list(keys)


# One engine class per distinct (GPU tier, concurrency) — see _engine_concurrency for why max_inputs
# is part of the class identity (Modal fixes concurrency per class).
ENGINE_BY_KEY: dict[tuple[str, int], Any] = {
    (gpu, mi): _build_engine(gpu, _engine_class_name(gpu, mi), mi, max(1, mi * 3 // 4))
    for (gpu, mi) in _distinct_engine_keys()
}


def _engine_cls_for(base_model: str) -> Any:
    """The Modal LoraEngine class to run ``base_model`` on (its (GPU, concurrency) class)."""
    return ENGINE_BY_KEY[_engine_key(base_model)]


class _ModalEnginePool:
    """Dispatches router calls to each base model's per-gpu-tier ``LoraEngine`` container."""

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
        engine = _engine_cls_for(base_model)(base_model=base_model)
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
        engine = _engine_cls_for(base_model)(base_model=base_model)
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
        engine = _engine_cls_for(base_model)(base_model=base_model)
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
        engine = _engine_cls_for(base_model)(base_model=base_model)
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
            thinking=True,
            org_id=None,
            status="ready",
        )
        for m in base_models()
    ]


@app.function(
    secrets=runtime_secrets,
    min_containers=1,  # one warm CPU front door (hardcoded; no deploy-time knob)
    timeout=ROUTER_TIMEOUT_SECONDS,  # must cover engine cold start (see ROUTER_TIMEOUT_SECONDS)
)
@modal.concurrent(max_inputs=MAX_INPUTS, target_inputs=TARGET_INPUTS)
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
        # Meter each generation to the backend (fire-and-forget); None disables it.
        usage_reporter=_build_usage_reporter(settings),
        # External chat auth is ALWAYS enforced: a request needs a Freesolo API key whose org owns
        # the adapter (the backend authorizes), or the shared internal key to bypass. The authorizer
        # must be wired (backend URL + internal key) or non-internal chat fails closed.
        chat_authorizer=_build_chat_authorizer(settings),
    )


@app.local_entrypoint()
def start_all(base_model: str | None = None) -> None:
    """Explicitly boot deployed gpu engines and wait for them to report healthy.

    Normal deploys leave gpu engines at zero until inference or adapter registration reaches the
    matching base model. This manual diagnostic can boot one model with ``--base-model`` or every
    catalog model without changing the scale-to-zero deployment.
    """
    from flash.serving.src.engine.model_config import base_models, gpu_for

    started = {}
    failures: list[str] = []
    models = [base_model] if base_model else list(base_models())
    for bm in models:
        # Each base model's engine lives on its (GPU tier, concurrency) class.
        engine = modal.Cls.from_name(
            APP_NAME, _engine_class_name(gpu_for(bm), _engine_concurrency(bm)[0])
        )
        instance = engine(base_model=bm)
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
