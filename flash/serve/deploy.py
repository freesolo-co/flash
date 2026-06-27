"""Serve a trained LoRA adapter via the freesolo platform's multi-LoRA serving app.

Flash no longer runs its own per-run vLLM endpoint. Instead the control plane is a
thin client of the freesolo serving service (a Modal multi-LoRA app that serves every
adapter on shared base-model capacity — so there is no flash-side idle billing to
track). The same CLI commands and control-plane endpoints
(`deploy`/`undeploy`/`chat`/`deployments`) stay; only what they do under the hood
changed.

The serving service exposes:

- ``POST {FREESOLO_SERVING_URL}/adapters`` — register/deploy an adapter (auth header).
- ``DELETE {FREESOLO_SERVING_URL}/adapters/{adapterId}`` — undeploy (auth header).
- ``POST {FREESOLO_SERVING_URL}/v1/chat/completions`` — OpenAI-style chat.
- ``GET {FREESOLO_SERVING_URL}/healthz`` / ``GET .../adapters`` — health / list.

The registration/teardown calls carry the shared ``X-Freesolo-Internal-Key`` header
(the same internal credential flash already holds, ``FREESOLO_INTERNAL_KEY``). The chat
calls also send it: the control plane is a trusted server-to-server caller (it has already
authorized the user's key on its own ``/v1/runs/{run_id}/chat`` route), so it uses the
serving app's internal-key bypass when serving enforces external chat auth.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass

import httpx

from flash._logging import get_logger
from flash.providers.base import canonical_gpu

logger = get_logger(__name__)

# Default freesolo serving base URL (the Modal multi-LoRA app). Overridable per-env.
DEFAULT_FREESOLO_SERVING_URL = "https://clado-ai--freesolo-lora-serving.modal.run"


class ServingError(RuntimeError):
    """The freesolo serving backend (Modal LoRA app) rejected a request or was unreachable.

    Carries the upstream status (when there was an HTTP response) so the API layer can
    surface a clean ``502 Bad Gateway`` with the real reason instead of letting an
    ``httpx`` exception escape as an unhandled ``500`` + traceback.
    """

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
    """Issue a request to the serving backend, translating any transport- or status-level failure
    into a ``ServingError`` that carries the upstream detail (shared by the deploy POST and the
    undeploy DELETE). A status in ``ok_statuses`` is returned WITHOUT raising so the caller can
    short-circuit it (undeploy treats a 404 as an already-gone no-op).

    Dispatched by name (``httpx.post``/``httpx.delete``) rather than ``httpx.request`` so the call
    is equivalent (``httpx.post(..., json=x)`` == ``httpx.request("POST", ..., json=x)``); ``json``
    is omitted for a bodyless request (e.g. the DELETE)."""
    # follow_redirects: Modal answers a slow request with a 303 to an async-result poll URL
    # (?__modal_function_call_id=...); without following it httpx raises on the 303 (see chat).
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
    """POST an adapter registration to the serving backend, translating any transport- or
    status-level failure into a ``ServingError`` that carries the upstream detail."""
    return _serving_request("POST", url, json=body)


def _serving_status_error(url: str, exc: httpx.HTTPStatusError) -> ServingError:
    """Build a ``ServingError`` from an upstream HTTP failure, carrying the status and a
    4xx-vs-5xx-tailored hint (shared by the deploy POST and the undeploy DELETE)."""
    # raise_for_status() always carries a response, but a hand-built HTTPStatusError may
    # not — guard so error translation can never itself raise.
    resp = exc.response
    status = resp.status_code if resp is not None else None
    detail = ((resp.text if resp is not None else "") or "").strip()[:500]
    msg = f"serving backend error for {url}"
    if status is not None:
        msg += f" (HTTP {status})"
    if detail:
        msg += f": {detail}"
    # Tailor the hint to the upstream status: a 4xx is a client/auth problem with THIS request
    # (e.g. a missing/invalid FREESOLO_INTERNAL_KEY), not a serving outage; a 5xx (or unknown)
    # means the backend itself failed / has no engine for the base model.
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
    """The freesolo serving base URL (env-overridable, trailing slash stripped)."""
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
    """Resolve a friendly GPU class for the deployment record.

    Serving is delegated to freesolo (one GPU per base model, chosen there), so this is
    now informational. We still canonicalize the name and fall back to the cheapest RunPod
    class big enough when the trained class isn't a RunPod class, so the recorded ``gpu`` is
    a sensible, valid class (and junk GPU names still raise)."""
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
    """Register the trained adapter with the freesolo serving app.

    The adapter artifacts already live in the run's HF dataset repo (the trainer
    streamed them there); freesolo serving pulls them from
    ``{hf_repo}:{adapter_prefix}/adapter``. ``dry_run`` validates/shapes the deployment
    without making the network call.
    """
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
        # The trainer always streams the adapter into a *dataset* repo (the worker's
        # hf_upload_folder uses repo_type="dataset"), so serving must pull from the dataset
        # namespace. Without this the serving app defaults repoType to "model" and
        # snapshot_download 404s on the model namespace — deploy returns 200 but the engine
        # warmup fails, the adapter is silently disabled, and the first chat 404s.
        "repoType": "dataset",
        "status": "ready",
        # Per-adapter thinking default: the value this run was trained with. Serving applies it as
        # the chat template's ``enable_thinking`` whenever a chat caller omits chat_template_kwargs
        # (e.g. a raw OpenAI client). Without it, serving falls back to Qwen3.5's template default
        # (thinking ON), so a run trained thinking=false emits a reasoning preamble ("…</think>{json}")
        # for any caller that doesn't pass the flag — diverging from how the adapter was trained.
        "thinking": bool(thinking),
    }
    # Attribute the adapter to the deploying org so serving can authorize external chat by org:
    # the backend maps adapterId -> org via hosted_lora_adapters.org_id, which serving persists
    # from this field. Normalize (strip) and omit when blank (older callers / whitespace) so the
    # registration shape is unchanged and a stray " org " can't mis-attribute the adapter.
    normalized_org_id = (org_id or "").strip()
    if normalized_org_id:
        body["orgId"] = normalized_org_id
    _post_adapter_or_raise(f"{base}/adapters", body)
    logger.info("registered adapter %s with freesolo serving (%s)", run_id, base)
    return dep


def undeploy_adapter(run_id: str) -> list[str]:
    """Deregister the run's adapter from the freesolo serving app.

    Returns ``[run_id]`` when the adapter was removed (200), ``[]`` when it was already
    gone (404). Any other failure — a non-404 HTTP status or a transport error — is
    translated into a ``ServingError`` (carrying the upstream status), exactly like
    ``deploy_adapter``, so callers see a stable error surface (the API maps it to a clean
    502) instead of a raw ``httpx`` exception escaping as an unhandled 500.
    """
    base = serving_base_url()
    url = f"{base}/adapters/{run_id}"
    # Undeploy is idempotent: an already-absent adapter (404) is a no-op success, not an error —
    # ok_statuses=(404,) returns it without raising so it never becomes a ServingError.
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
        # Per-run thinking parity: a run trained with thinking must serve with thinking, so
        # forward the flag to the chat template (enable_thinking is the kwarg the renderer and
        # rollout path use, e.g. multiturn_rollout.build_rollout_func). Without this the served
        # completions diverge from training behavior even though the caller passes thinking=.
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
    }
    # Cold starts (scale-from-zero per base model) can take minutes. Modal serves a slow ASGI
    # request by 303-redirecting to an async-result poll URL (?__modal_function_call_id=...), so
    # the client must follow redirects to retrieve the eventual completion — without this httpx
    # raises on the 303 and the chat fails mid cold-start. max_redirects is raised because a long
    # cold start polls across several redirect cycles before the result is ready.
    with httpx.Client(follow_redirects=True, max_redirects=100, timeout=30 * 60.0) as client:
        # The control plane is a trusted server-to-server caller (it already authorized the user's
        # key on the /v1/runs/{run_id}/chat route), so present the internal key to pass serving's
        # external chat-auth gate. No-op when the gate is off or the key is unset.
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
            # The request was opened with client.stream(), so the body is unread; httpx raises
            # ResponseNotRead on .json()/.content until it's pulled. Read it before parsing the
            # buffered (non-SSE) fallback an older serving app returns for stream=true.
            resp.read()
            payload = resp.json()
            content = (((payload.get("choices") or [{}])[0].get("message") or {}).get("content"))
            if content:
                yield str(content)
            return
        yield from _openai_stream_content(resp.iter_lines())
