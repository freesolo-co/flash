"""Thin client for the freesolo multi-LoRA serving app (Modal); no flash-side vLLM."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass

import httpx

from flash._logging import get_logger
from flash.providers.base import canonical_gpu

logger = get_logger(__name__)

DEFAULT_FREESOLO_SERVING_URL = "https://clado-ai--freesolo-lora-serving.modal.run"


class ServingError(RuntimeError):
    """Serving backend rejected a request or was unreachable; carries the upstream status."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _serving_request(
    method: str,
    url: str,
    *,
    json: dict | None = None,
    ok_statuses: tuple[int, ...] = (),
) -> httpx.Response:
    """Issue a request to the serving backend; translates failures into ServingError."""
    # follow_redirects: Modal 303-redirects slow requests to an async-result poll URL.
    kwargs: dict = {"headers": _internal_key_header(), "timeout": 60.0, "follow_redirects": True}
    if json is not None:
        kwargs["json"] = json
    try:
        resp = getattr(httpx, method.lower())(url, **kwargs)
        if resp.status_code in ok_statuses:
            return resp
        resp.raise_for_status()
        return resp
    except httpx.HTTPStatusError as exc:
        raise _serving_status_error(url, exc) from exc
    except httpx.RequestError as exc:
        raise ServingError(f"could not reach the serving backend at {url}: {exc}") from exc


def _post_adapter_or_raise(url: str, body: dict) -> httpx.Response:
    return _serving_request("POST", url, json=body)


def _serving_status_error(url: str, exc: httpx.HTTPStatusError) -> ServingError:
    """Build a ServingError from an upstream HTTP failure with a tailored hint."""
    resp = exc.response
    status = resp.status_code if resp is not None else None
    detail = ((resp.text if resp is not None else "") or "").strip()[:500]
    msg = f"serving backend error for {url}"
    if status is not None:
        msg += f" (HTTP {status})"
    if detail:
        msg += f": {detail}"
    if status is not None and status < 500:
        msg += (
            " — the serving backend rejected the request (4xx); check FREESOLO_INTERNAL_KEY "
            "and the request payload (this is a client/auth error, not a serving outage)"
        )
    else:
        msg += (
            " — the serving backend is unavailable or has no engine for this base model; "
            "an operator must check the freesolo serving deployment"
        )
    return ServingError(msg, status_code=status)


def serving_base_url() -> str:
    """Env-overridable serving base URL."""
    return (os.environ.get("FREESOLO_SERVING_URL") or DEFAULT_FREESOLO_SERVING_URL).rstrip("/")


def _internal_key_header() -> dict[str, str]:
    key = os.environ.get("FREESOLO_INTERNAL_KEY") or ""
    return {"X-Freesolo-Internal-Key": key} if key else {}


@dataclass
class Deployment:
    run_id: str
    model: str
    adapter_hf_prefix: str
    gpu: str
    openai_model: str
    endpoint_name: str
    state: str = "ready"

    def to_dict(self) -> dict:
        return asdict(self)


def servable_gpu(gpu_name: str) -> str:
    """Resolve a canonical RunPod GPU class for the deployment record (informational)."""
    from flash.providers.base import GPU_INFO, cheapest_gpu

    friendly = canonical_gpu(gpu_name)
    info = GPU_INFO[friendly]
    if info.enum_member:  # a RunPod class — serve it directly
        return friendly
    return cheapest_gpu(info.vram_gb)  # else the cheapest RunPod class that fits


def deploy_adapter(
    run_id: str,
    model: str,
    hf_repo: str,
    adapter_prefix: str,
    gpu_name: str = "RTX 5090",
    dry_run: bool = False,
    thinking: bool = False,
    org_id: str | None = None,
) -> Deployment:
    """Register the trained adapter with the freesolo serving app."""
    friendly = servable_gpu(gpu_name)
    subfolder = f"{adapter_prefix}/adapter"
    dep = Deployment(
        run_id=run_id,
        model=model,
        adapter_hf_prefix=subfolder,
        gpu=friendly,
        openai_model=run_id,
        endpoint_name=serving_base_url(),
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
        # Must be "dataset": trainer uploads to a dataset repo; serving defaults to "model" and 404s.
        "repoType": "dataset",
        "status": "ready",
        # Preserves thinking parity: without this, Qwen3.5 defaults to thinking ON regardless of training.
        "thinking": bool(thinking),
    }
    normalized_org_id = (org_id or "").strip()
    if normalized_org_id:
        body["orgId"] = normalized_org_id
    _post_adapter_or_raise(f"{base}/adapters", body)
    logger.info("registered adapter %s with freesolo serving (%s)", run_id, base)
    return dep


def undeploy_adapter(run_id: str) -> list[str]:
    """Deregister the adapter; returns [run_id] on success, [] if already gone (404)."""
    base = serving_base_url()
    url = f"{base}/adapters/{run_id}"
    resp = _serving_request("DELETE", url, ok_statuses=(404,))
    if resp.status_code == 404:
        return []
    logger.info("deregistered adapter %s from freesolo serving (%s)", run_id, base)
    return [run_id]


def chat(
    run_id: str,
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 512,
    thinking: bool = False,
) -> dict:
    """Send an OpenAI-style chat request for the run's adapter to freesolo serving."""
    base = serving_base_url()
    body = {
        "model": run_id,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
    }
    # follow_redirects + max_redirects=100: Modal 303-redirects slow cold-start requests across
    # several poll cycles before the result is ready.
    with httpx.Client(follow_redirects=True, max_redirects=100, timeout=30 * 60.0) as client:
        resp = client.post(f"{base}/v1/chat/completions", json=body, headers=_internal_key_header())
    resp.raise_for_status()
    return resp.json()


def _openai_stream_content(lines: Iterator[str]) -> Iterator[str]:
    for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            break
        if not data:
            continue
        chunk = json.loads(data)
        for choice in chunk.get("choices") or []:
            content = ((choice.get("delta") or {}).get("content")) or ""
            if content:
                yield str(content)


def chat_stream(
    run_id: str,
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 512,
    thinking: bool = False,
) -> Iterator[str]:
    """Yield text deltas from the freesolo OpenAI-compatible streaming endpoint."""
    base = serving_base_url()
    body = {
        "model": run_id,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
        "stream": True,
    }
    with (
        httpx.Client(follow_redirects=True, max_redirects=100, timeout=30 * 60.0) as client,
        client.stream(
            "POST", f"{base}/v1/chat/completions", json=body, headers=_internal_key_header()
        ) as resp,
    ):
        resp.raise_for_status()
        if "application/json" in resp.headers.get("content-type", ""):
            # client.stream() leaves body unread; must call resp.read() before .json().
            resp.read()
            payload = resp.json()
            content = (((payload.get("choices") or [{}])[0].get("message") or {}).get("content"))
            if content:
                yield str(content)
            return
        yield from _openai_stream_content(resp.iter_lines())
