"""Deployment URL compatibility helpers."""

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


def resolve_openai_base_url(deployment: dict) -> str:
    """Return the usable OpenAI base URL from current or legacy deployment records."""
    for field in ("openai_base_url", "url"):
        value = str(deployment.get(field) or "").rstrip("/")
        if value:
            return value

    return openai_base_url(str(deployment.get("endpoint_name") or ""))


def normalize_deployment_urls(deployment: dict) -> dict:
    """Copy a deployment record with canonical and legacy OpenAI URL fields."""
    out = dict(deployment)
    openai_url = resolve_openai_base_url(out)
    if openai_url:
        out["openai_base_url"] = openai_url
        out["url"] = openai_url
    return out


def public_deployment(deployment: dict) -> dict:
    """Return a normalized public copy without private rollback state."""
    out = normalize_deployment_urls(deployment)
    out.pop("previous_deployment", None)
    return out
