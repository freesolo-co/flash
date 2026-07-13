"""Deployment URL helpers."""

from __future__ import annotations


def serving_control_url(value: str) -> str:
    """Return the serving control root without a terminal OpenAI ``/v1`` path."""
    url = str(value or "").rstrip("/")
    if url.endswith("/v1"):
        return url[: -len("/v1")].rstrip("/")
    return url


def openai_base_url(control_url: str) -> str:
    """Return the OpenAI-compatible base URL for a serving control root."""
    control = serving_control_url(control_url)
    return f"{control}/v1" if control else ""


_PUBLIC_DEPLOYMENT_FIELDS = frozenset(
    {
        "run_id",
        "model",
        "adapter_hf_prefix",
        "openai_model",
        "endpoint_name",
        "openai_base_url",
        "state",
        "detail",
        "error",
        "checkpoint_step",
        "updated_at",
        "verified_at",
        "verify_latency_s",
        "verify_finish_reason",
        "thinking_tag",
        "verify_sample",
    }
)


def public_deployment(deployment: dict) -> dict:
    """Return the documented public deployment projection."""
    public = {key: value for key, value in deployment.items() if key in _PUBLIC_DEPLOYMENT_FIELDS}
    error = public.get("error")
    if isinstance(error, str):
        for field in ("mutation_id", "repo_revision"):
            value = deployment.get(field)
            if value:
                error = error.replace(str(value), "[redacted]")
        public["error"] = error
    return public
