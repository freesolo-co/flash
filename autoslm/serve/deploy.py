"""Serve a trained LoRA adapter for OpenAI-style chat via a managed RunPod Flash GPU.

Two deployment modes (the cost/latency trade-off is explicit and user-chosen):

- ``dev`` (default): scale-to-zero. ``workers=(0,1)`` with a configurable idle timeout
  and FlashBoot — you accept a cold start after inactivity, and pay $0 while idle.
- ``always-on``: ``workers=(1,1)`` — one worker stays warm 24/7 (no cold starts,
  continuous billing). ``slm deployments`` shows the projected $/day so the cost is
  never a surprise.

Each run gets its OWN uniquely-named serve endpoint (``autoslm-serve-<gpu>-<run>``), so
deployments never fight over a shared endpoint config and ``slm undeploy <run_id>``
can tear down exactly one deployment (via the REST API, from any process).

The handler boots vLLM with the base model + the LoRA adapter (pulled from the HF
dataset repo the trainer streamed it to) and returns an OpenAI-shaped chat-completion.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from autoslm._logging import get_logger
from autoslm.providers.base import canonical_gpu, gpu_short
from autoslm.providers.runpod.gpus import flash_gpu

logger = get_logger(__name__)


def _invoke_handler(handler, payload: dict) -> dict:
    """Call a Flash serve handler, awaiting it if the live path returns a coroutine."""
    import asyncio
    import inspect

    async def _call():
        res = handler(payload)
        if inspect.isawaitable(res):
            res = await res
        return res

    return asyncio.run(_call())


# Serving deps mirror the worker stack minus the trainer bits.
SERVE_DEPS = [
    "torch==2.10.0",
    "vllm==0.19.1",
    "transformers>=5.6,<5.11",
    "huggingface_hub>=0.25",
    "peft>=0.19",
    "accelerate>=1.4",
]
SERVE_SYSTEM_DEPS = ["build-essential"]

_ENDPOINT_CACHE: dict[str, object] = {}

# Serving cold start (image pull + vLLM/PEFT install + ~8 GB base model + adapter) can
# exceed 10 min on a fresh host; default the serve execution cap to 25 min
# (env-overridable) so the first `slm chat` on a cold worker doesn't fail with
# "executionTimeout exceeded".
_DEFAULT_SERVE_TIMEOUT_MS = 25 * 60 * 1000

MODES = ("dev", "always-on")
DEFAULT_IDLE_TIMEOUT_S = 300

# Projected always-on cost uses live RunPod rates (static fallback):
# providers/runpod/pricing.py (hourly_rate).


def serve_execution_timeout_ms() -> int:
    return int(os.environ.get("AUTOSLM_SERVE_TIMEOUT_MS", str(_DEFAULT_SERVE_TIMEOUT_MS)))


def resolve_serve_deps() -> list[str]:
    explicit = os.environ.get("AUTOSLM_SERVE_DEPS")
    if explicit:
        # JSON list (use this for specs containing commas, e.g.
        # "transformers>=5.6,<5.11") or a whitespace-separated string. NOT comma-split:
        # a comma is part of a PEP 440 range and must not become two pip arguments
        # (mirrors providers/runpod/train.resolve_worker_deps).
        if explicit.strip().startswith("["):
            import json as _json

            deps = [str(d).strip() for d in _json.loads(explicit) if str(d).strip()]
        else:
            import shlex

            deps = [d for d in shlex.split(explicit) if d.strip()]
        if deps:
            return deps
    return SERVE_DEPS


@dataclass
class Deployment:
    run_id: str
    model: str
    adapter_hf_prefix: str
    gpu: str
    openai_model: str
    endpoint_name: str
    mode: str = "dev"
    idle_timeout_s: int = DEFAULT_IDLE_TIMEOUT_S
    est_idle_cost_usd_per_day: float = 0.0
    state: str = "ready"

    def to_dict(self) -> dict:
        return asdict(self)


def _language_model_only(model: str) -> bool:
    """Natively-multimodal checkpoints are served text-only (the family-name check
    mirrors the worker's config-based one; the client can't load the HF config)."""
    return any(s in model for s in ("Qwen3.5", "Qwen3.6"))


def serve_endpoint_name(friendly_gpu: str, run_id: str) -> str:
    tail = (run_id or "").split("-")[-1][:24]
    base = f"autoslm-serve-{gpu_short(canonical_gpu(friendly_gpu))}"
    return f"{base}-{tail}" if tail else base


def servable_gpu(gpu_name: str, model: str) -> str:
    """Serving runs on RunPod Flash only: a run trained on a class that is not
    RunPod-validated (a Vast-only class like L40S/RTX Pro 4000, OR a class that has a
    RunPod enum member but was validated only on Vast, e.g. RTX 3090) is served from the
    cheapest RunPod-VALIDATED class with at least the trained class's VRAM — NOT directly
    on the unvalidated RunPod substrate (which can fail on first chat) and NOT the
    catalog default (32 GB for open models, too small for the >32 GB class the allocator
    proved was needed)."""
    from autoslm.providers.base import GPU_INFO, UnsupportedGpuError, cheapest_gpu

    friendly = canonical_gpu(gpu_name)
    info = GPU_INFO[friendly]
    # Enum presence is not enough: serve directly only on a RunPod-VALIDATED class.
    if "runpod" in info.validated_on:
        return friendly
    try:
        # Prefer a RunPod-validated class big enough; only if none exists fall back
        # to an unvalidated one (better than refusing to serve at all).
        fallback = cheapest_gpu(info.vram_gb)
    except UnsupportedGpuError:
        fallback = cheapest_gpu(info.vram_gb, include_unvalidated=True)
    logger.warning(
        "%s is not RunPod-validated; serving %s on %s (>= %d GB)",
        friendly,
        model,
        fallback,
        info.vram_gb,
    )
    return fallback


def _serve_body(input_data: dict) -> dict:
    """Runs ON the GPU worker: forward a chat request to a persistent local vLLM server.

    vLLM runs as a LONG-LIVED SUBPROCESS (the OpenAI api_server on localhost), not in
    the handler process: importing torch inside the Flash handler process crashes
    (torch dynamo config-module assertion against the runpod runtime's module state -
    observed live), and a subprocess keeps the engine warm across requests anyway.
    The handler boots it on first request (the cold start) and proxies afterwards.

    NOTE: Flash serializes this handler and runs it standalone - all imports live
    inside the body, and state is cached in module globals while the worker is warm.
    """
    import json as _json
    import os
    import subprocess
    import sys
    import time
    import urllib.error
    import urllib.request

    g = globals()
    base = "http://127.0.0.1:8199"

    def _tail_serve_log(limit: int = 3000) -> str:
        # Defined IN the body: Flash serializes _serve_body and runs it standalone, so
        # the module-level helper of the same name is out of scope on the worker — a
        # bare reference would NameError on the boot-failure path and hide the vLLM log.
        try:
            with open("/tmp/vllm_serve.log") as f:
                return f.read()[-limit:]
        except OSError:
            return "(no serve log)"

    _thinking = input_data.get("thinking", False)
    # Robust bool: input_data may arrive loosely typed (serialized handler payload), and
    # bool("false") is True — treat the usual falsey strings as False.
    thinking = (
        _thinking.strip().lower() not in ("", "0", "false", "no", "off", "none")
        if isinstance(_thinking, str)
        else bool(_thinking)
    )
    # thinking is part of the engine key: it changes --max-model-len, and the vLLM
    # subprocess must be rebooted to change that.
    key = (input_data["model"], input_data["adapter_prefix"], thinking)

    def server_alive() -> bool:
        proc = g.get("_AUTOSLM_PROC")
        if proc is None or proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"{base}/health", timeout=3)
            return True
        except Exception:
            return False

    if g.get("_AUTOSLM_KEY") != key or not server_alive():
        from huggingface_hub import snapshot_download

        prefix = input_data["adapter_prefix"]
        snapshot_download(
            repo_id=input_data["hf_repo"],
            repo_type="dataset",
            allow_patterns=[f"{prefix}/adapter/*"],
            local_dir="/adapter",
            token=input_data.get("token"),
        )
        adapter_dir = f"/adapter/{prefix}/adapter"
        old = g.get("_AUTOSLM_PROC")
        if old is not None and old.poll() is None:
            old.kill()
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--host",
            "127.0.0.1",
            "--port",
            "8199",
            "--model",
            input_data["model"],
            "--served-model-name",
            "base",
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "4096" if thinking else "2048",  # <think> blocks need completion headroom
            "--gpu-memory-utilization",
            "0.85",
            "--enable-lora",
            "--max-lora-rank",
            str(int(input_data.get("max_lora_rank") or 64)),
            "--lora-modules",
            f"adapter={adapter_dir}",
            "--trust-remote-code",
        ]
        if input_data.get("language_model_only"):
            cmd.append("--language-model-only")
        env = dict(os.environ)
        env.setdefault("HF_TOKEN", input_data.get("token") or "")
        # Popen dups the fd into the child, so the parent handle can close
        # immediately while vLLM keeps writing to the log.
        with open("/tmp/vllm_serve.log", "w") as log:
            g["_AUTOSLM_PROC"] = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT, env=env
            )
        # Align the vLLM boot budget with the endpoint execution cap: large MoE
        # tiers can take 15-25 min to load on a cold host, and a hard-coded 900 s
        # would 502 a first chat / fail an always-on warmup while the RunPod request
        # still has time budget. Leave ~60 s of the window for the first generation.
        # Read the env DIRECTLY (with the same default as serve_execution_timeout_ms):
        # Flash serializes _serve_body and runs it standalone, so the module-level
        # helper is out of scope on the worker (see _tail_serve_log) and a bare call
        # would NameError before vLLM gets a chance to boot.
        serve_timeout_ms = int(os.environ.get("AUTOSLM_SERVE_TIMEOUT_MS", str(25 * 60 * 1000)))
        default_boot = max(900, serve_timeout_ms // 1000 - 60)
        deadline = time.time() + float(input_data.get("boot_timeout_s") or default_boot)
        while time.time() < deadline:
            if g["_AUTOSLM_PROC"].poll() is not None:
                tail = _tail_serve_log()
                raise RuntimeError(f"vLLM server exited during boot:\n{tail}")
            try:
                urllib.request.urlopen(f"{base}/health", timeout=3)
                break
            except Exception:
                time.sleep(2)
        else:
            tail = _tail_serve_log()
            raise RuntimeError(f"vLLM server did not become healthy in time:\n{tail}")
        g["_AUTOSLM_KEY"] = key

    # always-on warmup: pay the cold start (download + vLLM boot) at deploy time
    # so the user's first real chat is warm, then return without a completion.
    if input_data.get("warmup"):
        return {"ok": True, "warmed": True}

    body = {
        "model": "adapter",
        "messages": input_data.get("messages") or [],
        "temperature": float(input_data.get("temperature", 0.0)),
        "top_p": float(input_data.get("top_p", 1.0)),
        "max_tokens": int(input_data.get("max_tokens", 512)),
        # Serve with the run's training-time thinking flag (decoding parity). Thinking
        # responses carry the raw <think>...</think> block in message.content.
        "chat_template_kwargs": {"enable_thinking": thinking},
    }

    def post(payload: dict) -> dict:
        req = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=_json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            return _json.loads(resp.read())

    try:
        out = post(body)
    except urllib.error.HTTPError as e:
        if e.code == 400:
            # A vLLM build that rejects the kwarg falls back to the chat template's own
            # default (thinking ON for hybrid Qwen3 — correct for thinking runs, a logged
            # degradation for non-thinking ones).
            body.pop("chat_template_kwargs", None)
            print("vLLM rejected chat_template_kwargs; retrying without enable_thinking")
            out = post(body)
        else:
            raise
    out["model"] = input_data.get("served_model", "autoslm-adapter")
    return out


def _get_serve_endpoint(
    friendly_gpu: str,
    run_id: str,
    mode: str = "dev",
    idle_timeout_s: int = DEFAULT_IDLE_TIMEOUT_S,
):
    os.environ["FLASH_IS_LIVE_PROVISIONING"] = "true"
    from runpod_flash import Endpoint

    from autoslm.providers.runpod.auth import ensure_auth
    from autoslm.providers.runpod.train import FLASH_SDK_LOCK, isolate_flash_state, min_cuda_for

    ensure_auth()
    friendly = canonical_gpu(friendly_gpu)
    name = serve_endpoint_name(friendly, run_id)
    cache_key = f"{name}:{mode}:{idle_timeout_s}"

    # Serialize against training deploy/teardown on the same process: isolate_flash_state()
    # swaps runpod_flash's process-wide registry globals and Endpoint() touches the SDK's
    # asyncio singleton, so a concurrent terminate_endpoint()/always-on warmup on another
    # thread could race the registry scope. Hold the same lock across isolation + construction.
    with FLASH_SDK_LOCK:
        isolate_flash_state(f"serve-{run_id.split('-')[-1]}")
        if cache_key in _ENDPOINT_CACHE:
            return _ENDPOINT_CACHE[cache_key]
        kwargs = {
            "name": name,
            "gpu": flash_gpu(friendly),
            "gpu_count": 1,
            "min_cuda_version": min_cuda_for(friendly),
            # dev: scale to zero after idle_timeout (cold start accepted, $0 idle).
            # always-on: one permanently warm worker (no cold start, 24/7 billing).
            "workers": (0, 1) if mode == "dev" else (1, 1),
            "idle_timeout": int(idle_timeout_s),
            "flashboot": True,
            "execution_timeout_ms": serve_execution_timeout_ms(),
        }
        image = os.environ.get("AUTOSLM_WORKER_IMAGE")
        if image:
            kwargs["image"] = image
        else:
            kwargs["dependencies"] = resolve_serve_deps()
            kwargs["system_dependencies"] = SERVE_SYSTEM_DEPS
        ep = Endpoint(**kwargs)
        handler = ep(_serve_body)
        _ENDPOINT_CACHE[cache_key] = handler
        return handler


def _reject_qlora_serving(model: str) -> None:
    """vLLM serving boots the base in bf16; a 4-bit-QLoRA-only tier won't fit.

    Eval/train use the 4-bit transformers path for those models, but there is
    no quantized vLLM serving path, so reject up front instead of provisioning
    an endpoint that fails on first chat."""
    from autoslm.catalog import MODELS

    info = MODELS.get(model)
    if info is not None and info.quant == "4bit-qlora":
        raise ValueError(
            f"{model} is a 4-bit-QLoRA-only tier and cannot be served bf16 by vLLM; "
            "merge the adapter into a full model (`build_hf_model`) and deploy that, "
            "or use a smaller bf16 model."
        )


def deploy_adapter(
    run_id: str,
    model: str,
    hf_repo: str,
    adapter_prefix: str,
    gpu_name: str = "RTX 5090",
    mode: str = "dev",
    idle_timeout_s: int = DEFAULT_IDLE_TIMEOUT_S,
    dry_run: bool = False,
    lora_rank: int = 64,
    thinking: bool = False,
) -> Deployment:
    """Provision a serving endpoint for a trained adapter (managed, no Docker)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    _reject_qlora_serving(model)
    friendly = servable_gpu(gpu_name, model)
    from autoslm.runner import _gpu_rate

    rate = _gpu_rate(friendly)
    dep = Deployment(
        run_id=run_id,
        model=model,
        adapter_hf_prefix=adapter_prefix,
        gpu=friendly,
        openai_model=f"autoslm-{run_id}",
        endpoint_name=serve_endpoint_name(friendly, run_id),
        mode=mode,
        idle_timeout_s=idle_timeout_s,
        est_idle_cost_usd_per_day=0.0 if mode == "dev" else round(rate * 24, 2),
        state="dry_run" if dry_run else "ready",
    )
    if dry_run:
        return dep
    handler = _get_serve_endpoint(friendly, run_id, mode=mode, idle_timeout_s=idle_timeout_s)
    # always-on promises no cold start: warm the worker now (download + vLLM boot)
    # BEFORE returning ready, so the user's first chat is genuinely warm. dev mode
    # is scale-to-zero by design, so it warms lazily on first chat.
    if mode == "always-on":
        warmup = {
            "hf_repo": hf_repo,
            "model": model,
            "adapter_prefix": adapter_prefix,
            "token": os.environ.get("HUGGINGFACE_TOKEN", ""),
            "max_lora_rank": max(64, int(lora_rank)),
            "language_model_only": _language_model_only(model),
            # warm the engine with the run's thinking flag so the first real chat (same
            # flag) reuses the warmed subprocess instead of rebooting for --max-model-len.
            "thinking": thinking,
            "warmup": True,
        }
        try:
            _invoke_handler(handler, warmup)
        except Exception:
            # The warmup invocation is what actually provisions the always-on worker;
            # if adapter download / vLLM boot fails the endpoint is registered (and may
            # bill) but no deployment is persisted. Tear it down before propagating.
            import contextlib

            with contextlib.suppress(Exception):
                undeploy_adapter(run_id, gpu_name=friendly)
            raise
    return dep


def undeploy_adapter(run_id: str, gpu_name: str = "RTX 5090") -> list[str]:
    """Tear down the run's serve endpoint via the REST API (works from any process)."""
    from autoslm.providers.runpod import api as runpod_api

    name = serve_endpoint_name(gpu_name, run_id)
    # find_endpoints_by_name is a substring filter, so guard with an exact-name
    # check (mirrors providers/runpod/train.py terminate_endpoint) — otherwise a
    # name that is a substring of another run's endpoint would over-delete and
    # mis-report the returned `deleted` list.
    deleted = [
        ep["name"]
        for ep in runpod_api.find_endpoints_by_name(name)
        if ep.get("name") == name and runpod_api.delete_endpoint(ep["id"])
    ]
    # Drop in-process handler cache entries for this endpoint (keyed name:mode:idle)
    # so a later redeploy constructs a fresh endpoint instead of reusing a handler
    # pointing at the just-deleted one.
    for key in [k for k in _ENDPOINT_CACHE if k.startswith(f"{name}:")]:
        _ENDPOINT_CACHE.pop(key, None)
    return deleted


def chat(
    run_id: str,
    messages: list[dict],
    model: str,
    hf_repo: str,
    adapter_prefix: str,
    gpu_name: str = "RTX 5090",
    temperature: float = 0.0,
    max_tokens: int = 512,
    mode: str = "dev",
    idle_timeout_s: int = DEFAULT_IDLE_TIMEOUT_S,
    lora_rank: int = 64,
    thinking: bool = False,
) -> dict:
    """Send an OpenAI-style chat request to the adapter's managed Flash GPU."""
    handler = _get_serve_endpoint(
        servable_gpu(gpu_name, model), run_id, mode=mode, idle_timeout_s=idle_timeout_s
    )
    # Natively-multimodal checkpoints are served text-only (no transformers needed
    # client-side; the family-name check mirrors the worker's config-based one).
    language_model_only = _language_model_only(model)
    payload = {
        "hf_repo": hf_repo,
        "model": model,
        "adapter_prefix": adapter_prefix,
        "token": os.environ.get("HUGGINGFACE_TOKEN", ""),
        "served_model": f"autoslm-{run_id}",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        # vLLM rejects an adapter whose rank exceeds --max-lora-rank; cover the
        # run's configured rank, not just the 64 default.
        "max_lora_rank": max(64, int(lora_rank)),
        "language_model_only": language_model_only,
        "thinking": thinking,
    }
    return _invoke_handler(handler, payload)
