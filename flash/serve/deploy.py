"""Serve a trained LoRA adapter via the freesolo platform's multi-LoRA serving app.

Flash no longer runs its own per-run vLLM endpoint. Instead the control plane is a
thin client of the freesolo serving service (a Modal multi-LoRA app that serves every
adapter on a single GPU per base model, scaling to zero when idle — so there is no
flash-side idle billing to track). The same CLI commands and control-plane endpoints
(`deploy`/`undeploy`/`chat`/`deployments`) stay; only what they do under the hood
changed.

The serving service exposes:

- ``POST {FREESOLO_SERVING_URL}/adapters`` — register/deploy an adapter (auth header).
- ``DELETE {FREESOLO_SERVING_URL}/adapters/{adapterId}`` — undeploy (auth header).
- ``POST {FREESOLO_SERVING_URL}/v1/chat/completions`` — OpenAI-style chat (no auth).
- ``GET {FREESOLO_SERVING_URL}/healthz`` / ``GET .../adapters`` — health / list.

The registration/teardown calls carry the shared ``X-Freesolo-Internal-Key`` header
(the same internal credential flash already holds, ``FREESOLO_INTERNAL_KEY``); chat is
unauthenticated.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import httpx

from flash._logging import get_logger
from flash.providers.base import canonical_gpu, gpu_short

logger = get_logger(__name__)

# Default freesolo serving base URL (the Modal multi-LoRA app). Overridable per-env.
DEFAULT_FREESOLO_SERVING_URL = "https://clado-ai--freesolo-lora-serving.modal.run"

# These remain so callers/tests that imported them keep resolving; they are cosmetic now
# that serving is delegated to freesolo (no flash-owned endpoint to size or warm).
MODES = ("dev", "always-on")
DEFAULT_IDLE_TIMEOUT_S = 300
_ENDPOINT_CACHE: dict[str, object] = {}
# The serving deps used to live on the flash worker image; serving is now external.
SERVE_DEPS: list[str] = []


def serving_base_url() -> str:
    """The freesolo serving base URL (env-overridable, trailing slash stripped)."""
    return (os.environ.get("FREESOLO_SERVING_URL") or DEFAULT_FREESOLO_SERVING_URL).rstrip("/")


def _internal_key_header() -> dict[str, str]:
    key = os.environ.get("FREESOLO_INTERNAL_KEY") or ""
    return {"X-Freesolo-Internal-Key": key} if key else {}


def resolve_serve_deps() -> list[str]:
    """Kept for back-compat; serving deps are external (freesolo), so this is empty.

    An explicit ``FLASH_SERVE_DEPS`` override is still honored for any caller that
    inspects it, but flash no longer installs a serving stack of its own.
    """
    explicit = os.environ.get("FLASH_SERVE_DEPS")
    if explicit:
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
    # freesolo serving scales to zero per base model, so flash never bills for idle
    # serving — there is no flash-side per-run endpoint to keep warm.
    est_idle_cost_usd_per_day: float = 0.0
    state: str = "ready"

    def to_dict(self) -> dict:
        return asdict(self)


def _language_model_only(model: str) -> bool:
    """Cosmetic: the family-name guard the old flash worker used for text-only serving.

    The freesolo serving app makes its own multimodal decisions; kept so any importer
    keeps resolving."""
    return "Qwen3.5" in model or "Qwen3.6" in model


def serve_endpoint_name(friendly_gpu: str, run_id: str) -> str:
    """Cosmetic endpoint label (the freesolo app serves all adapters on one endpoint)."""
    tail = (run_id or "").split("-")[-1][:24]
    base = f"flash-serve-{gpu_short(canonical_gpu(friendly_gpu))}"
    return f"{base}-{tail}" if tail else base


def servable_gpu(gpu_name: str, model: str) -> str:
    """Resolve a friendly GPU class for the deployment record.

    Serving is delegated to freesolo (one GPU per base model, chosen there), so this is
    now informational. We still canonicalize the name and fall back to the cheapest
    RunPod-validated class big enough when the trained class isn't RunPod-validated, so
    the recorded ``gpu`` is a sensible, valid class (and junk GPU names still raise)."""
    from flash.providers.base import GPU_INFO, UnsupportedGpuError, cheapest_gpu

    friendly = canonical_gpu(gpu_name)
    info = GPU_INFO[friendly]
    if "runpod" in info.validated_on:
        return friendly
    try:
        return cheapest_gpu(info.vram_gb)
    except UnsupportedGpuError:
        return cheapest_gpu(info.vram_gb, include_unvalidated=True)


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
    """Register the trained adapter with the freesolo serving app.

    The adapter artifacts already live in the run's HF dataset repo (the trainer
    streamed them there); freesolo serving pulls them from
    ``{hf_repo}:{adapter_prefix}/adapter``. ``dry_run`` validates/shapes the deployment
    without making the network call.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    friendly = servable_gpu(gpu_name, model)
    subfolder = f"{adapter_prefix}/adapter"
    dep = Deployment(
        run_id=run_id,
        model=model,
        adapter_hf_prefix=subfolder,
        gpu=friendly,
        openai_model=run_id,
        endpoint_name=serving_base_url(),
        mode=mode,
        idle_timeout_s=idle_timeout_s,
        est_idle_cost_usd_per_day=0.0,
        state="dry_run" if dry_run else "ready",
    )
    if dry_run:
        return dep
    base = serving_base_url()
    body = {
        "adapterId": run_id,
        "repoId": hf_repo,
        "baseModel": model,
        "subfolder": subfolder,
        "status": "ready",
    }
    resp = httpx.post(
        f"{base}/adapters",
        json=body,
        headers=_internal_key_header(),
        timeout=60.0,
    )
    resp.raise_for_status()
    logger.info("registered adapter %s with freesolo serving (%s)", run_id, base)
    return dep


def undeploy_adapter(run_id: str, gpu_name: str = "RTX 5090") -> list[str]:
    """Deregister the run's adapter from the freesolo serving app.

    Returns ``[run_id]`` when the adapter was removed (200), ``[]`` when it was already
    gone (404). Other statuses raise so the caller can surface a transient failure.
    """
    base = serving_base_url()
    resp = httpx.delete(
        f"{base}/adapters/{run_id}",
        headers=_internal_key_header(),
        timeout=60.0,
    )
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    logger.info("deregistered adapter %s from freesolo serving (%s)", run_id, base)
    return [run_id]


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
    """Send an OpenAI-style chat request for the run's adapter to freesolo serving.

    The adapter is addressed by ``model=run_id`` (its registered ``adapterId``); the
    response is the parsed OpenAI chat-completion dict, so
    ``resp["choices"][0]["message"]["content"]`` keeps working downstream.
    """
    base = serving_base_url()
    body = {
        "model": run_id,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    # Cold starts (scale-from-zero per base model) can take minutes; give it room.
    resp = httpx.post(f"{base}/v1/chat/completions", json=body, timeout=30 * 60.0)
    resp.raise_for_status()
    return resp.json()
