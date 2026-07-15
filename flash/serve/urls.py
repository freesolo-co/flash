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


def public_deployment(deployment: dict) -> dict:
    """Return a public copy without private rollback state or the stale legacy URL field."""
    out = dict(deployment)
    out.pop("previous_deployment", None)
    out.pop("verification_generation", None)
    out.pop("url", None)
    return out
