"""Thin client for the freesolo multi-LoRA serving app (Modal); no flash-side vLLM."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass

import httpx

from flash._logging import get_logger
from flash.lora_rank import rank_from_adapter_config
from flash.providers.base import canonical_gpu

logger = get_logger(__name__)

DEFAULT_FREESOLO_SERVING_URL = "https://clado-ai--freesolo-lora-serving.modal.run"

# Read-back verification: after POST /adapters, poll the serving registry until the adapter is
# visible at the requested checkpoint, so "ready" is a registry-backed claim rather than an
# assumption (see _verify_adapter_registered).
READBACK_ATTEMPTS = 5
READBACK_DELAY_SECONDS = 2.0


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
    # The OpenAI-compatible base URL. endpoint_name is the bare serving root; OpenAI clients must
    # be pointed at {endpoint_name}/v1 or their /chat/completions calls 404.
    url: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def servable_gpu(gpu_name: str) -> str:
    """Resolve a canonical RunPod GPU class for the deployment record (informational)."""
    from flash.providers.base import GPU_INFO, cheapest_gpu, get_gpu_info

    friendly = canonical_gpu(gpu_name)
    info = get_gpu_info(friendly)
    # A directly servable class serves as-is; other managed classes map to the cheapest serving class
    # that fits their VRAM.
    if friendly in GPU_INFO and info.enum_member:
        return friendly
    return cheapest_gpu(info.vram_gb)


def validate_serving_lora_rank(model: str, lora_rank: int, *, rank_source: str = "adapter") -> None:
    """Fail before registration when a trained adapter rank exceeds serving capacity."""
    from flash.catalog import serving_lora_rank_cap

    max_lora_rank = serving_lora_rank_cap(model)
    if max_lora_rank is None:
        return
    if int(lora_rank) > max_lora_rank:
        raise ValueError(
            f"{model} serving supports max_lora_rank={max_lora_rank}; "
            f"{rank_source} has rank {int(lora_rank)} and cannot be deployed"
        )


def _rank_from_adapter_config(config: dict, *, source: str) -> int:
    return rank_from_adapter_config(config, source=source)


def adapter_artifact_lora_rank(hf_repo: str, subfolder: str) -> int:
    """Read the deployed adapter's actual rank from HF artifact metadata."""
    filename = f"{subfolder.rstrip('/')}/adapter_config.json"
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - package extra is present in supported installs
        raise ServingError(
            "could not verify adapter rank: huggingface_hub is not installed"
        ) from exc
    try:
        local = hf_hub_download(
            repo_id=hf_repo,
            filename=filename,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception as exc:
        raise ServingError(
            f"could not verify adapter rank: failed to read {hf_repo}:{filename}"
        ) from exc
    try:
        with open(local, encoding="utf-8") as f:
            config = json.load(f)
    except Exception as exc:
        raise ValueError(
            f"could not verify adapter rank: invalid JSON in {hf_repo}:{filename}"
        ) from exc
    if not isinstance(config, dict):
        raise ValueError(
            f"could not verify adapter rank: {hf_repo}:{filename} is not a JSON object"
        )
    return _rank_from_adapter_config(config, source=f"{hf_repo}:{filename}")


def deploy_adapter(
    run_id: str,
    model: str,
    hf_repo: str,
    adapter_prefix: str,
    gpu_name: str = "RTX 5090",
    *,
    dry_run: bool = False,
    lora_rank: int = 32,
    thinking: bool = False,
    org_id: str | None = None,
) -> Deployment:
    """Register the trained adapter with the freesolo serving app."""
    validate_serving_lora_rank(model, lora_rank, rank_source="configured train.lora_rank")
    friendly = servable_gpu(gpu_name)
    subfolder = f"{adapter_prefix}/adapter"
    if not dry_run:
        validate_serving_lora_rank(
            model,
            adapter_artifact_lora_rank(hf_repo, subfolder),
            rank_source="adapter artifact",
        )
    base = serving_base_url()
    dep = Deployment(
        run_id=run_id,
        model=model,
        adapter_hf_prefix=subfolder,
        gpu=friendly,
        openai_model=run_id,
        endpoint_name=base,
        state="dry_run" if dry_run else "ready",
        url=f"{base}/v1",
    )
    if dry_run:
        return dep
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
    try:
        _post_adapter_or_raise(f"{base}/adapters", body)
    except ServingError as exc:
        # A 4xx means serving rejected the request outright and nothing changed. Anything else
        # (5xx, timeout, unreachable) is ambiguous: the registry may or may not have switched to
        # the new checkpoint. Read it back and report the actual state instead of guessing.
        if exc.status_code is not None and exc.status_code < 500:
            raise
        record = _registered_adapter(run_id)
        recorded = _record_subfolder(record)
        if record is not None and recorded in (None, subfolder):
            logger.warning(
                "POST /adapters for %s failed (%s) but the serving registry shows the adapter "
                "registered; continuing",
                run_id,
                exc,
            )
        elif recorded is not None:
            raise ServingError(
                f"{exc} — the serving registry still shows adapter {run_id} at the previously "
                f"deployed checkpoint ({recorded!r}, requested {subfolder!r}), so the OLD "
                "checkpoint remains active. Retry `flash deploy` once serving recovers",
                status_code=exc.status_code,
            ) from exc
        else:
            raise
    else:
        _verify_adapter_registered(run_id, subfolder)
    logger.info("registered adapter %s with freesolo serving (%s)", run_id, base)
    return dep


def _record_subfolder(record: dict | None) -> str | None:
    if not isinstance(record, dict):
        return None
    value = record.get("subfolder")
    return str(value) if value is not None else None


def _registered_adapter(run_id: str) -> dict | None:
    """The adapter's record in the serving registry, or None when absent or unreadable."""
    try:
        resp = _serving_request("GET", f"{serving_base_url()}/adapters")
        payload = resp.json()
    except (ServingError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    for record in payload.get("adapters") or []:
        if isinstance(record, dict) and record.get("adapter_id", record.get("adapterId")) == run_id:
            return record
    return None


def _verify_adapter_registered(run_id: str, subfolder: str) -> None:
    """Poll the serving registry until the adapter is visible at the requested checkpoint.

    A 2xx from POST /adapters only proves the registration request was accepted. If the record
    never lands in the registry, /v1/chat/completions 404s with "Unknown adapter id" even though
    the deployment record claims ready. Polling here makes "ready" a registry-backed claim.
    """
    recorded: str | None = None
    seen = False
    for attempt in range(READBACK_ATTEMPTS):
        if attempt:
            time.sleep(READBACK_DELAY_SECONDS * attempt)
        record = _registered_adapter(run_id)
        if record is None:
            continue
        seen = True
        recorded = _record_subfolder(record)
        # Older serving builds may omit subfolder from the record; presence then has to count.
        if recorded is None or recorded == subfolder:
            return
    if seen:
        raise ServingError(
            f"adapter {run_id} is registered at checkpoint {recorded!r} instead of the requested "
            f"{subfolder!r}; the previously deployed checkpoint is still active — retry "
            "`flash deploy`"
        )
    raise ServingError(
        f"adapter {run_id} was accepted by serving but never appeared in its registry; chat "
        "requests would fail with HTTP 404 'Unknown adapter id'. The deployment was not marked "
        "ready — retry `flash deploy`, and if this persists an operator must check the freesolo "
        "serving app"
    )


def undeploy_adapter(run_id: str) -> list[str]:
    """Deregister the adapter; returns [run_id] on success, [] if already gone (404)."""
    base = serving_base_url()
    url = f"{base}/adapters/{run_id}"
    resp = _serving_request("DELETE", url, ok_statuses=(404,))
    if resp.status_code == 404:
        return []
    if _registered_adapter(run_id) is not None:
        logger.warning(
            "adapter %s still appears in the serving registry after DELETE; a stale router may "
            "keep serving it until its next reload",
            run_id,
        )
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
            content = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content")
            if content:
                yield str(content)
            return
        yield from _openai_stream_content(resp.iter_lines())
