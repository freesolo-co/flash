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


def validate_serving_lora_rank(
    model: str, lora_rank: int, *, rank_source: str = "adapter"
) -> None:
    """Fail before registration when a trained adapter rank exceeds serving capacity."""
    from flash.catalog import get_model

    try:
        serving = get_model(model).serving
    except ValueError:
        return
    if serving is None:
        return
    if int(lora_rank) > serving.max_lora_rank:
        raise ValueError(
            f"{model} serving supports max_lora_rank={serving.max_lora_rank}; "
            f"{rank_source} rank {int(lora_rank)} cannot be deployed"
        )


def _rank_from_adapter_config(config: dict, *, source: str) -> int:
    ranks: list[int] = []
    try:
        if config.get("r") is not None:
            ranks.append(int(config["r"]))
        rank_pattern = config.get("rank_pattern") or {}
        if rank_pattern:
            if not isinstance(rank_pattern, dict):
                raise TypeError("rank_pattern is not a mapping")
            ranks.extend(int(v) for v in rank_pattern.values())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"could not verify adapter rank: {source} has invalid rank metadata") from exc
    if not ranks:
        raise ValueError(f"could not verify adapter rank: {source} has no LoRA rank metadata")
    rank = max(ranks)
    if rank <= 0:
        raise ValueError(f"could not verify adapter rank: {source} has non-positive rank {rank}")
    return rank


def adapter_artifact_lora_rank(hf_repo: str, subfolder: str) -> int:
    """Read the deployed adapter's actual rank from HF artifact metadata."""
    filename = f"{subfolder.rstrip('/')}/adapter_config.json"
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - package extra is present in supported installs
        raise ValueError("could not verify adapter rank: huggingface_hub is not installed") from exc
    try:
        local = hf_hub_download(
            repo_id=hf_repo,
            filename=filename,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception as exc:
        raise ValueError(f"could not verify adapter rank: failed to read {hf_repo}:{filename}") from exc
    try:
        with open(local, encoding="utf-8") as f:
            config = json.load(f)
    except Exception as exc:
        raise ValueError(f"could not verify adapter rank: invalid JSON in {hf_repo}:{filename}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"could not verify adapter rank: {hf_repo}:{filename} is not a JSON object")
    return _rank_from_adapter_config(config, source=f"{hf_repo}:{filename}")


def deploy_adapter(
    run_id: str,
    model: str,
    hf_repo: str,
    adapter_prefix: str,
    gpu_name: str = "RTX 5090",
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


def list_deployed_adapters() -> list[dict]:
    """Return the serving app's live adapter registry — the authoritative DO-NOT-DELETE set.

    A record normally carries ``adapterId`` (== run_id), ``repoId`` (the HF dataset repo serving
    pulls weights from), and the ``subfolder``/``adapter_hf_prefix`` it loads, but this function
    only validates that the body is a list of dict records — it does NOT enforce those keys, so
    callers must tolerate a record missing them (e.g. on schema drift). Used by operator repo GC
    (``flash.server.repo_cleanup``), whose ``deployed_repo_ids`` flags an unmappable record rather
    than trusting an incomplete keep-set, to know which per-run repos are serving traffic.

    Raises ``ServingError`` if the serving backend is unreachable, returns non-200, returns a
    non-JSON body, or returns a 200 in an unrecognized shape — a caller gating destructive cleanup
    on this set MUST treat any of those as "unknown, do not delete" rather than "nothing deployed".
    The success body is tolerated as either a bare list or an ``{"adapters": [...]}`` /
    ``{"data": [...]}`` envelope; an empty list (live set genuinely empty) returns ``[]``.
    """
    base = serving_base_url()
    url = f"{base}/adapters"
    try:
        resp = httpx.get(
            url,
            headers=_internal_key_header(),
            timeout=60.0,
            # Modal answers a slow request with a 303 to an async-result poll URL; follow it (see chat).
            follow_redirects=True,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise _serving_status_error(url, exc) from exc
    except httpx.RequestError as exc:
        raise ServingError(f"could not reach the serving backend at {url}: {exc}") from exc
    try:
        data = resp.json()
    except ValueError as exc:
        # A 200 with an un-decodable body is NOT "nothing deployed" — surface it so the GC caller
        # treats the live set as unknown and refuses to delete.
        raise ServingError(f"serving backend returned a non-JSON adapter list from {url}: {exc}") from exc
    if isinstance(data, dict):
        # Accept only the known envelopes. An unrecognized object must NOT silently degrade to an
        # empty keep-set (that would green-light deleting live repos); fail closed instead.
        if "adapters" in data:
            data = data["adapters"]
        elif "data" in data:
            data = data["data"]
        else:
            raise ServingError(
                f"serving backend returned an unrecognized adapter-list envelope from {url} "
                # ``sorted(data, key=str)`` not ``sorted(data)``: JSON object keys are always strings,
                # but a hypothetical mixed-type keyset would make a bare ``sorted`` raise TypeError and
                # MASK this fail-closed ServingError with a less actionable error. Sort defensively.
                f"(keys: {sorted(data, key=str)[:8]}); refusing to treat it as an empty live set"
            )
    if not isinstance(data, list):
        raise ServingError(
            f"serving backend returned a {type(data).__name__}, not an adapter list, from {url}; "
            "refusing to treat it as an empty live set"
        )
    # Every item must be a record. Silently dropping non-dict items (strings/None) would shrink the
    # keep-set toward empty — and this set gates destructive GC — so a malformed item fails closed.
    if any(not isinstance(rec, dict) for rec in data):
        raise ServingError(
            f"serving backend returned a non-record item in the adapter list from {url}; "
            "refusing to treat it as the live set"
        )
    return list(data)


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
