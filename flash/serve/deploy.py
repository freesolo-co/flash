"""Thin client for the freesolo multi-LoRA serving app (Modal); no flash-side vLLM."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass

import httpx

from flash._logging import get_logger
from flash.engine.structured_outputs import parse_structured_outputs
from flash.lora_rank import rank_from_adapter_config
from flash.serve.urls import openai_base_url, serving_control_url

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


class AdapterConfigMissing(ServingError):
    """The adapter's adapter_config.json could not be read from HF (artifact likely absent)."""


class AdapterTensorMissing(ServingError):
    """The adapter artifact has metadata but no loadable LoRA tensor file."""


def _is_adapter_tensor_filename(filename: str) -> bool:
    name = filename.rsplit("/", 1)[-1]
    if name in {"adapter_model.safetensors", "adapter_model.bin"}:
        return True
    return name.startswith("adapter_model-") and name.endswith((".safetensors", ".bin"))


def _is_hf_not_found_error(exc: Exception) -> bool:
    try:
        import huggingface_hub.errors as hf_errors  # type: ignore[import-not-found]

        not_found_types = tuple(
            cls
            for name in (
                "EntryNotFoundError",
                "LocalEntryNotFoundError",
                "RepositoryNotFoundError",
                "RevisionNotFoundError",
            )
            if isinstance((cls := getattr(hf_errors, name, None)), type)
        )
        if not_found_types and isinstance(exc, not_found_types):
            return True
    except Exception:
        pass
    return getattr(getattr(exc, "response", None), "status_code", None) == 404


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
    """Env-overridable serving control root."""
    configured = os.environ.get("FREESOLO_SERVING_URL") or DEFAULT_FREESOLO_SERVING_URL
    return serving_control_url(configured)


def serving_openai_base_url() -> str:
    """OpenAI-compatible base URL for the configured serving backend."""
    return openai_base_url(serving_base_url())


def _internal_key_header() -> dict[str, str]:
    key = os.environ.get("FREESOLO_INTERNAL_KEY") or ""
    return {"X-Freesolo-Internal-Key": key} if key else {}


@dataclass
class Deployment:
    run_id: str
    model: str
    adapter_hf_prefix: str
    openai_model: str
    endpoint_name: str
    openai_base_url: str
    state: str = "ready"

    def to_dict(self) -> dict:
        return asdict(self)


def deployment_record(
    run_id: str,
    model: str,
    adapter_prefix: str,
    *,
    state: str = "ready",
) -> Deployment:
    subfolder = f"{adapter_prefix}/adapter"
    base = serving_base_url()
    openai_url = serving_openai_base_url()
    return Deployment(
        run_id=run_id,
        model=model,
        adapter_hf_prefix=subfolder,
        openai_model=run_id,
        endpoint_name=base,
        openai_base_url=openai_url,
        state=state,
    )


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


def _verify_adapter_artifact_tensors(hf_repo: str, subfolder: str) -> None:
    """Confirm the adapter has tensor weights before registering it with serving."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - package extra is present in supported installs
        raise ServingError(
            "could not verify adapter tensors: huggingface_hub is not installed"
        ) from exc
    try:
        entries = list(
            HfApi().list_repo_tree(
                repo_id=hf_repo,
                path_in_repo=subfolder.rstrip("/"),
                repo_type="dataset",
                recursive=False,
                token=os.environ.get("HF_TOKEN"),
            )
        )
    except Exception as exc:
        message = f"could not verify adapter tensors: failed to list {hf_repo}:{subfolder}"
        if _is_hf_not_found_error(exc):
            raise AdapterTensorMissing(message) from exc
        raise ServingError(message) from exc

    tensor_paths: list[str] = []
    zero_byte_tensor_paths: list[str] = []
    for entry in entries:
        path = str(getattr(entry, "path", "") or "")
        if not _is_adapter_tensor_filename(path):
            continue
        tensor_paths.append(path)
        size = getattr(entry, "size", None)
        try:
            if size is not None and int(size) <= 0:
                zero_byte_tensor_paths.append(path)
                continue
        except (TypeError, ValueError):
            pass

    location = f"{hf_repo}:{subfolder}"
    if zero_byte_tensor_paths:
        raise AdapterTensorMissing(
            f"could not verify adapter tensors: {location} has zero-byte adapter tensor "
            f"file(s): {', '.join(zero_byte_tensor_paths)}"
        )
    if tensor_paths:
        return
    raise AdapterTensorMissing(
        f"could not verify adapter tensors: {location} has no adapter_model tensor file"
    )


def adapter_artifact_lora_rank(hf_repo: str, subfolder: str) -> int:
    """Read rank metadata and verify the adapter has tensor weights."""
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
        message = f"could not verify adapter rank: failed to read {hf_repo}:{filename}"
        if _is_hf_not_found_error(exc):
            raise AdapterConfigMissing(message) from exc
        raise ServingError(message) from exc
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
    _verify_adapter_artifact_tensors(hf_repo, subfolder)
    return rank_from_adapter_config(config, source=f"{hf_repo}:{filename}")


def _structured_outputs_body(
    run_id: str, structured_outputs: str, *, thinking: bool
) -> dict | None:
    """The per-adapter structured-outputs serving default for the registration body, or None.

    Returns the parsed canonical StructuredOutputsParams-kwargs dict when the run configured a
    constraint AND is non-thinking; None otherwise (no default is registered). Non-thinking only:
    serving sets no reasoning-parser, so a per-adapter grammar would bind from the very first
    generated token and force a thinking adapter's ``<think>`` phase to open the schema (e.g. ``{``)
    — strictly worse than serving it unconstrained. Training stays compatible with thinking via a
    DEFERRED grammar (``reasoning_parser="deepseek_r1"`` holds the constraint until ``</think>``),
    but serving has no such deferral, so we skip the serve-time default for thinking runs rather
    than break reasoning. Raises ValueError on a corrupt spec — a wiring bug, since
    ``[train].structured_outputs`` is already canonicalized at run creation (schema/fields.py), so
    this cannot fire for a real run; it fails loudly rather than ever silently serving unconstrained."""
    if not structured_outputs:
        return None
    if thinking:
        logger.info(
            "adapter %s trained with structured_outputs but thinking=True; not registering a "
            "serving guided-decoding default — serving has no reasoning-parser deferral, so it "
            "would bind the grammar from token 0 and suppress the <think> phase",
            run_id,
        )
        return None
    return parse_structured_outputs(structured_outputs)


def deploy_adapter(
    run_id: str,
    model: str,
    hf_repo: str,
    adapter_prefix: str,
    *,
    dry_run: bool = False,
    lora_rank: int = 32,
    thinking: bool = False,
    structured_outputs: str = "",
    org_id: str | None = None,
) -> Deployment:
    """Register the trained adapter with the freesolo serving app.

    ``structured_outputs`` is the run's ``[train].structured_outputs`` spec (canonical
    StructuredOutputsParams-kwargs JSON, or "" for none). When set, it is registered as the
    adapter's per-adapter guided-decoding DEFAULT so every serve request is constrained by the
    SAME grammar the adapter trained under — otherwise a structured-outputs run drifts at
    unconstrained serving (train/serve exposure bias: it over-generates arrays that truncate to
    invalid JSON and emits keys the training grammar had masked). See ``_structured_outputs_body``.
    """
    validate_serving_lora_rank(model, lora_rank, rank_source="configured train.lora_rank")
    subfolder = f"{adapter_prefix}/adapter"
    if not dry_run:
        validate_serving_lora_rank(
            model,
            adapter_artifact_lora_rank(hf_repo, subfolder),
            rank_source="adapter artifact",
        )
    dep = deployment_record(run_id, model, adapter_prefix, state="dry_run" if dry_run else "ready")
    if dry_run:
        return dep
    base = serving_base_url()
    body = {
        "adapter_id": run_id,
        "repo_id": hf_repo,
        "base_model": model,
        "subfolder": subfolder,
        # Must be "dataset": trainer uploads to a dataset repo; serving defaults to "model" and 404s.
        "repo_type": "dataset",
        "status": "ready",
        # Preserves thinking parity: without this, Qwen3.5 defaults to thinking ON regardless of training.
        "thinking": bool(thinking),
    }
    so_default = _structured_outputs_body(run_id, structured_outputs, thinking=thinking)
    if so_default is not None:
        body["structured_outputs"] = so_default
    normalized_org_id = (org_id or "").strip()
    if normalized_org_id:
        body["org_id"] = normalized_org_id
    previous_record = _registered_adapter(run_id)
    try:
        _serving_request("POST", f"{base}/adapters", json=body)
    except ServingError as exc:
        # A 4xx means serving rejected the request outright and nothing changed. Anything else
        # (5xx, timeout, unreachable) is ambiguous: the registry may or may not have switched to
        # the new checkpoint. Read it back and report the actual state instead of guessing.
        if exc.status_code is not None and exc.status_code < 500:
            raise
        record = _registered_adapter(run_id)
        recorded = _record_subfolder(record)
        if record is not None and recorded == subfolder:
            logger.warning(
                "POST /adapters for %s failed (%s) but the serving registry shows the adapter "
                "registered; continuing",
                run_id,
                exc,
            )
        elif record is not None and recorded is None and previous_record is None:
            logger.warning(
                "POST /adapters for %s failed (%s) but the serving registry shows a new adapter "
                "record without subfolder details; continuing because no prior deployment was "
                "registered",
                run_id,
                exc,
            )
        elif record is not None and recorded is None:
            raise ServingError(
                f"{exc} — the serving registry returned adapter {run_id} without a subfolder, "
                f"so it cannot confirm that requested checkpoint {subfolder!r} replaced the "
                "previous deployment. Retry `flash deploy` once serving recovers",
                status_code=exc.status_code,
            ) from exc
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
        if not isinstance(record, dict):
            continue
        adapter_id = record.get("adapter_id") or record.get("adapterId")
        if adapter_id == run_id:
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
    base = serving_openai_base_url()
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
        resp = client.post(f"{base}/chat/completions", json=body, headers=_internal_key_header())
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
    base = serving_openai_base_url()
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
            "POST", f"{base}/chat/completions", json=body, headers=_internal_key_header()
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
