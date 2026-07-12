"""Thin client for the freesolo multi-LoRA serving app (Modal); no flash-side vLLM."""

from __future__ import annotations

import copy
import json
import os
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field

import httpx

from flash._logging import get_logger
from flash.engine.structured_outputs import parse_structured_outputs
from flash.lora_rank import rank_from_adapter_config

logger = get_logger(__name__)

DEFAULT_FREESOLO_SERVING_URL = "https://clado-ai--freesolo-lora-serving.modal.run"

# Read-back verification: after POST /adapters, poll the serving registry until the adapter is
# visible at the requested checkpoint, so "ready" is a registry-backed claim rather than an
# assumption (see _verify_adapter_registered).
READBACK_ATTEMPTS = 5
READBACK_DELAY_SECONDS = 2.0
THINKING_STRUCTURED_OUTPUTS_CAPABILITY = "thinking_structured_outputs_deferred_v1"


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
    openai_model: str
    endpoint_name: str
    state: str = "ready"
    # the openai-compatible base url. endpoint_name is the bare serving root.
    url: str = ""
    # internal lifecycle state used to restore serving after a failed smoke or cas transition.
    previous_registry_record: dict | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("previous_registry_record", None)
        return data


def deployment_record(
    run_id: str,
    model: str,
    adapter_prefix: str,
    *,
    state: str = "ready",
) -> Deployment:
    subfolder = f"{adapter_prefix}/adapter"
    base = serving_base_url()
    return Deployment(
        run_id=run_id,
        model=model,
        adapter_hf_prefix=subfolder,
        openai_model=run_id,
        endpoint_name=base,
        state=state,
        url=f"{base}/v1",
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


def _structured_outputs_body(structured_outputs: str) -> dict | None:
    """Return the canonical parsed serving constraint, or None when unconstrained."""
    if not structured_outputs:
        return None
    return parse_structured_outputs(structured_outputs)


def _require_thinking_structured_outputs_capability(base: str) -> None:
    """Fail closed unless serving advertises deferred thinking constraint support."""
    url = f"{base}/healthz"
    try:
        payload = _serving_request("GET", url).json()
    except (ServingError, ValueError) as exc:
        raise ServingError(
            "cannot safely deploy a thinking adapter with structured outputs because the serving "
            "capability probe was unreachable or malformed. No adapter registry mutation was "
            "attempted; serving must advertise "
            f"{THINKING_STRUCTURED_OUTPUTS_CAPABILITY!r} before Flash can preserve training/serving "
            "exposure safety"
        ) from exc
    capabilities = payload.get("capabilities") if isinstance(payload, dict) else None
    if not isinstance(capabilities, list) or not all(isinstance(item, str) for item in capabilities):
        raise ServingError(
            "cannot safely deploy a thinking adapter with structured outputs because /healthz "
            "returned a malformed capabilities list. No adapter registry mutation was attempted; "
            f"serving must advertise {THINKING_STRUCTURED_OUTPUTS_CAPABILITY!r}"
        )
    if THINKING_STRUCTURED_OUTPUTS_CAPABILITY not in capabilities:
        raise ServingError(
            "cannot safely deploy a thinking adapter with structured outputs because serving does "
            "not advertise deferred structured-output enforcement. No adapter registry mutation "
            "was attempted; deploying unconstrained would reintroduce training/serving exposure "
            f"bias. Required capability: {THINKING_STRUCTURED_OUTPUTS_CAPABILITY}"
        )


def _record_value(record: dict, snake: str, camel: str | None = None):
    value = record.get(snake)
    if value is None and camel:
        value = record.get(camel)
    return value


def _record_mismatches(record: dict | None, expected: dict) -> list[str]:
    if not isinstance(record, dict):
        return ["adapter record is absent"]
    comparisons = {
        "adapter_id": (_record_value(record, "adapter_id", "adapterId"), expected["adapter_id"]),
        "subfolder": (record.get("subfolder"), expected["subfolder"]),
        "base_model": (_record_value(record, "base_model", "baseModel"), expected["base_model"]),
        "repo_id": (_record_value(record, "repo_id", "repoId"), expected["repo_id"]),
        "repo_type": (_record_value(record, "repo_type", "repoType"), expected["repo_type"]),
        "thinking": (record.get("thinking"), expected["thinking"]),
        "structured_outputs": (record.get("structured_outputs"), expected.get("structured_outputs")),
        "structured_outputs_after_reasoning": (
            record.get("structured_outputs_after_reasoning"),
            expected.get("structured_outputs_after_reasoning"),
        ),
    }
    return [
        f"{name}={actual!r}, expected {wanted!r}"
        for name, (actual, wanted) in comparisons.items()
        if actual != wanted
    ]


def snapshot_adapter_record(run_id: str) -> dict | None:
    """Return a deep copy of the exact current registry record."""
    record = _registered_adapter(run_id, strict=True)
    return copy.deepcopy(record) if record is not None else None


def restore_adapter_record(run_id: str, previous_record: dict | None) -> None:
    """Restore the exact pre-deploy registry state and verify the result."""
    base = serving_base_url()
    if previous_record is None:
        _serving_request("DELETE", f"{base}/adapters/{run_id}", ok_statuses=(404,))
        if _registered_adapter(run_id, strict=True) is not None:
            raise ServingError(
                f"adapter {run_id} remains registered after rollback delete; operator cleanup required"
            )
        return

    restored = copy.deepcopy(previous_record)
    _serving_request("POST", f"{base}/adapters", json=restored)
    readback = _registered_adapter(run_id, strict=True)
    if readback != restored:
        raise ServingError(
            f"adapter {run_id} rollback readback did not exactly match the prior registry record; "
            "operator cleanup required"
        )


def _restore_after_failure(run_id: str, previous_record: dict | None, exc: Exception) -> ServingError:
    try:
        restore_adapter_record(run_id, previous_record)
    except Exception as rollback_exc:
        return ServingError(
            f"{exc}; serving rollback failed ({rollback_exc}); operator cleanup required",
            status_code=getattr(exc, "status_code", None),
        )
    return ServingError(
        f"{exc}; serving registry was restored to its exact pre-deploy state",
        status_code=getattr(exc, "status_code", None),
    )


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

    ``structured_outputs`` is the run's ``[train].structured_outputs`` spec as canonical
    ``StructuredOutputsParams`` keyword arguments, or an empty string when disabled. Non-thinking
    adapters register it under ``structured_outputs``. Thinking adapters require the deferred
    serving capability and register it under ``structured_outputs_after_reasoning`` so guided
    decoding begins only after the reasoning boundary. This preserves the training grammar without
    constraining reasoning tokens or reintroducing train/serve exposure bias.
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
    so_default = _structured_outputs_body(structured_outputs)
    if thinking and so_default is not None:
        _require_thinking_structured_outputs_capability(base)
    body = {
        "adapter_id": run_id,
        "repo_id": hf_repo,
        "base_model": model,
        "subfolder": subfolder,
        # trainer artifacts live in dataset repositories.
        "repo_type": "dataset",
        "status": "ready",
        "thinking": bool(thinking),
    }
    if so_default is not None:
        constraint_key = (
            "structured_outputs_after_reasoning" if thinking else "structured_outputs"
        )
        body[constraint_key] = so_default
    normalized_org_id = (org_id or "").strip()
    if normalized_org_id:
        body["org_id"] = normalized_org_id

    previous_record = snapshot_adapter_record(run_id)
    dep.previous_registry_record = copy.deepcopy(previous_record)
    try:
        _serving_request("POST", f"{base}/adapters", json=body)
    except ServingError as exc:
        if exc.status_code is not None and exc.status_code < 500:
            raise _restore_after_failure(run_id, previous_record, exc) from exc
        try:
            record = _registered_adapter(run_id, strict=True)
        except ServingError:
            raise _restore_after_failure(run_id, previous_record, exc) from exc
        mismatches = _record_mismatches(record, body)
        if not mismatches:
            logger.warning(
                "POST /adapters for %s failed (%s) but exact registry readback confirms the "
                "requested adapter metadata; continuing",
                run_id,
                exc,
            )
        else:
            ambiguous = ServingError(
                f"{exc}; ambiguous registration readback for adapter {run_id}: "
                + "; ".join(mismatches),
                status_code=exc.status_code,
            )
            raise _restore_after_failure(run_id, previous_record, ambiguous) from exc
    else:
        try:
            _verify_adapter_registered(run_id, body)
        except ServingError as exc:
            raise _restore_after_failure(run_id, previous_record, exc) from exc
    logger.info("registered adapter %s with freesolo serving (%s)", run_id, base)
    return dep


def _registered_adapter(run_id: str, *, strict: bool = False) -> dict | None:
    """Return the adapter registry record, optionally failing on unreadable metadata."""
    try:
        resp = _serving_request("GET", f"{serving_base_url()}/adapters")
        payload = resp.json()
    except (ServingError, ValueError) as exc:
        if strict:
            raise ServingError(
                f"could not read serving registry while checking adapter {run_id}: {exc}"
            ) from exc
        return None
    records = payload.get("adapters") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        if strict:
            raise ServingError("serving registry returned malformed adapter metadata")
        return None
    for record in records:
        if not isinstance(record, dict):
            if strict:
                raise ServingError("serving registry returned a malformed adapter record")
            continue
        adapter_id = _record_value(record, "adapter_id", "adapterId")
        if adapter_id == run_id:
            return record
    return None


def _verify_adapter_registered(run_id: str, expected: dict) -> None:
    """Poll until every safety-relevant adapter field matches the requested registration."""
    mismatches = ["adapter record is absent"]
    for attempt in range(READBACK_ATTEMPTS):
        if attempt:
            time.sleep(READBACK_DELAY_SECONDS * attempt)
        record = _registered_adapter(run_id, strict=True)
        mismatches = _record_mismatches(record, expected)
        if not mismatches:
            return
    if mismatches == ["adapter record is absent"]:
        raise ServingError(
            f"adapter {run_id} was accepted by serving but never appeared in its registry; chat "
            "requests would fail with HTTP 404 'Unknown adapter id'. The deployment was not marked "
            "ready; retry `flash deploy`, and if this persists an operator must check serving"
        )
    raise ServingError(
        f"adapter {run_id} registry readback does not match the requested metadata: "
        + "; ".join(mismatches)
        + ". The deployment was not marked ready"
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
