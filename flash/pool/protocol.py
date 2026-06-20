"""vLLM wire protocol (pure builders — no IO).

The pool's backends are stock **vLLM OpenAI-compatible servers** launched with dynamic LoRA
enabled (``--enable-lora`` + env ``VLLM_ALLOW_RUNTIME_LORA_UPDATING=1``). The router speaks vLLM's
real HTTP surface so the same code path works in tests (against a fake backend) and in production
(against real vLLM):

* ``GET  /health``                          -> 200 when the engine is up
* ``GET  /v1/models``                       -> lists base + loaded LoRA "models"
* ``POST /v1/load_lora_adapter``  {lora_name, lora_path}
* ``POST /v1/unload_lora_adapter`` {lora_name}
* ``POST /v1/chat/completions``   (OpenAI body; ``model`` = the lora_name to apply, or the base id)
* ``POST /v1/completions``        (OpenAI text body)

Keeping these as pure dict/URL builders means the request shapes are unit-testable and there is a
single place to adjust if vLLM's surface drifts.
"""

from __future__ import annotations

HEALTH_PATH = "/health"
MODELS_PATH = "/v1/models"
LOAD_LORA_PATH = "/v1/load_lora_adapter"
UNLOAD_LORA_PATH = "/v1/unload_lora_adapter"
CHAT_PATH = "/v1/chat/completions"
COMPLETIONS_PATH = "/v1/completions"


def join(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def load_lora_body(lora_name: str, lora_path: str) -> dict:
    return {"lora_name": lora_name, "lora_path": lora_path}


def unload_lora_body(lora_name: str) -> dict:
    return {"lora_name": lora_name}


def with_model(body: dict, model: str) -> dict:
    """Return a shallow copy of an OpenAI request body with ``model`` set to ``model`` (the LoRA
    name to apply on the backend, or the base id for un-adaptered generation)."""
    out = dict(body)
    out["model"] = model
    return out
