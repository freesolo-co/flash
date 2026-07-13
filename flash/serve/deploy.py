"""Thin client for the freesolo multi-LoRA serving app."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field

import httpx

from flash._logging import get_logger
from flash.engine.structured_outputs import parse_structured_outputs
from flash.lora_rank import rank_from_adapter_config
from flash.serve.urls import openai_base_url, serving_control_url

logger = get_logger(__name__)

DEFAULT_FREESOLO_SERVING_URL = "https://clado-ai--freesolo-lora-serving.modal.run"
READBACK_ATTEMPTS = 5
READBACK_DELAY_SECONDS = 2.0
_FULL_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


class ServingError(RuntimeError):
    """Serving backend rejected a request or was unreachable."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class DeploymentSuperseded(ServingError):
    """A newer or different serving mutation won the registry race."""


class AdapterConfigMissing(ServingError):
    """The adapter config is absent at the immutable artifact revision."""


class AdapterTensorMissing(ServingError):
    """The immutable adapter artifact has no loadable tensor file."""


def _is_adapter_tensor_filename(filename: str) -> bool:
    name = filename.rsplit("/", 1)[-1]
    return name in {"adapter_model.safetensors", "adapter_model.bin"} or (
        name.startswith("adapter_model-") and name.endswith((".safetensors", ".bin"))
    )


def _is_hf_not_found_error(exc: Exception) -> bool:
    try:
        import huggingface_hub.errors as hf_errors  # type: ignore[import-not-found]

        types = tuple(
            cls
            for name in (
                "EntryNotFoundError",
                "LocalEntryNotFoundError",
                "RepositoryNotFoundError",
                "RevisionNotFoundError",
            )
            if isinstance((cls := getattr(hf_errors, name, None)), type)
        )
        if types and isinstance(exc, types):
            return True
    except Exception:
        pass
    return getattr(getattr(exc, "response", None), "status_code", None) == 404


def _internal_key_header() -> dict[str, str]:
    key = os.environ.get("FREESOLO_INTERNAL_KEY") or ""
    return {"X-Freesolo-Internal-Key": key} if key else {}


def _serving_status_error(url: str, exc: httpx.HTTPStatusError) -> ServingError:
    response = exc.response
    status = response.status_code if response is not None else None
    detail = ((response.text if response is not None else "") or "").strip()[:500]
    message = f"serving backend error for {url}"
    if status is not None:
        message += f" (HTTP {status})"
    if detail:
        message += f": {detail}"
    return ServingError(message, status_code=status)


def _serving_request(
    method: str,
    url: str,
    *,
    json: dict | None = None,
    headers: dict[str, str] | None = None,
    ok_statuses: tuple[int, ...] = (),
) -> httpx.Response:
    request_headers = {**_internal_key_header(), **(headers or {})}
    kwargs: dict = {"headers": request_headers, "timeout": 60.0, "follow_redirects": True}
    if json is not None:
        kwargs["json"] = json
    try:
        response = getattr(httpx, method.lower())(url, **kwargs)
        if response.status_code in ok_statuses:
            return response
        response.raise_for_status()
        return response
    except httpx.HTTPStatusError as exc:
        raise _serving_status_error(url, exc) from exc
    except httpx.RequestError as exc:
        raise ServingError(f"could not reach the serving backend at {url}: {exc}") from exc


def serving_base_url() -> str:
    configured = os.environ.get("FREESOLO_SERVING_URL") or DEFAULT_FREESOLO_SERVING_URL
    return serving_control_url(configured)


def serving_openai_base_url() -> str:
    return openai_base_url(serving_base_url())


@dataclass
class Deployment:
    run_id: str
    model: str
    adapter_hf_prefix: str
    openai_model: str
    endpoint_name: str
    openai_base_url: str
    state: str = "ready"
    desired_record: dict | None = field(default=None, repr=False)
    prior_revision: int | None = field(default=None, repr=False)
    target_revision: int | None = field(default=None, repr=False)
    mutation_id: str | None = field(default=None, repr=False)
    repo_revision: str | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        data = asdict(self)
        for name in (
            "desired_record",
            "prior_revision",
            "target_revision",
            "mutation_id",
            "repo_revision",
        ):
            data.pop(name, None)
        return data


def deployment_record(
    run_id: str, model: str, adapter_prefix: str, *, state: str = "ready"
) -> Deployment:
    return Deployment(
        run_id=run_id,
        model=model,
        adapter_hf_prefix=f"{adapter_prefix}/adapter",
        openai_model=run_id,
        endpoint_name=serving_base_url(),
        openai_base_url=serving_openai_base_url(),
        state=state,
    )


def validate_serving_lora_rank(model: str, lora_rank: int, *, rank_source: str = "adapter") -> None:
    from flash.catalog import serving_lora_rank_cap

    maximum = serving_lora_rank_cap(model)
    if maximum is not None and int(lora_rank) > maximum:
        raise ValueError(
            f"{model} serving supports max_lora_rank={maximum}; "
            f"{rank_source} has rank {int(lora_rank)} and cannot be deployed"
        )


def resolve_repo_revision(hf_repo: str) -> str:
    """Resolve the uploaded dataset once to a full immutable commit sha."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover
        raise ServingError(
            "could not resolve adapter revision: huggingface_hub is not installed"
        ) from exc
    try:
        info = HfApi().repo_info(
            repo_id=hf_repo,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception as exc:
        raise ServingError(f"could not resolve adapter revision for {hf_repo}") from exc
    revision = str(getattr(info, "sha", "") or "")
    if _FULL_COMMIT_RE.fullmatch(revision) is None:
        raise ServingError(
            f"adapter repository {hf_repo} did not resolve to a full lowercase 40-character commit sha"
        )
    return revision


def _verify_adapter_artifact_tensors(hf_repo: str, subfolder: str, repo_revision: str) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover
        raise ServingError(
            "could not verify adapter tensors: huggingface_hub is not installed"
        ) from exc
    try:
        entries = list(
            HfApi().list_repo_tree(
                repo_id=hf_repo,
                path_in_repo=subfolder.rstrip("/"),
                repo_type="dataset",
                revision=repo_revision,
                recursive=False,
                token=os.environ.get("HF_TOKEN"),
            )
        )
    except Exception as exc:
        message = f"could not verify adapter tensors: failed to list {hf_repo}:{subfolder}"
        if _is_hf_not_found_error(exc):
            raise AdapterTensorMissing(message) from exc
        raise ServingError(message) from exc
    tensors: list[str] = []
    empty: list[str] = []
    for entry in entries:
        path = str(getattr(entry, "path", "") or "")
        if not _is_adapter_tensor_filename(path):
            continue
        tensors.append(path)
        size = getattr(entry, "size", None)
        try:
            if size is not None and int(size) <= 0:
                empty.append(path)
        except (TypeError, ValueError):
            pass
    location = f"{hf_repo}@{repo_revision}:{subfolder}"
    if empty:
        raise AdapterTensorMissing(
            f"could not verify adapter tensors: {location} has zero-byte adapter tensor file(s): "
            + ", ".join(empty)
        )
    if not tensors:
        raise AdapterTensorMissing(
            f"could not verify adapter tensors: {location} has no adapter_model tensor file"
        )


def adapter_artifact_lora_rank(hf_repo: str, subfolder: str, repo_revision: str) -> int:
    filename = f"{subfolder.rstrip('/')}/adapter_config.json"
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise ServingError(
            "could not verify adapter rank: huggingface_hub is not installed"
        ) from exc
    try:
        local = hf_hub_download(
            repo_id=hf_repo,
            filename=filename,
            repo_type="dataset",
            revision=repo_revision,
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception as exc:
        message = f"could not verify adapter rank: failed to read {hf_repo}:{filename}"
        if _is_hf_not_found_error(exc):
            raise AdapterConfigMissing(message) from exc
        raise ServingError(message) from exc
    try:
        with open(local, encoding="utf-8") as handle:
            config = json.load(handle)
    except Exception as exc:
        raise ValueError(
            f"could not verify adapter rank: invalid JSON in {hf_repo}:{filename}"
        ) from exc
    if not isinstance(config, dict):
        raise ValueError(
            f"could not verify adapter rank: {hf_repo}:{filename} is not a JSON object"
        )
    _verify_adapter_artifact_tensors(hf_repo, subfolder, repo_revision)
    return rank_from_adapter_config(config, source=f"{hf_repo}:{filename}")


def _structured_outputs_body(structured_outputs: str) -> dict | None:
    return parse_structured_outputs(structured_outputs) if structured_outputs else None


def intended_checkpoint(run_id: str, adapter_prefix: str) -> str:
    if "/checkpoints/step-" not in adapter_prefix:
        return run_id
    step = adapter_prefix.rsplit("/checkpoints/", 1)[-1].split("/", 1)[0]
    return f"{run_id}/{step}"


def _etag_revision(response: httpx.Response) -> int:
    raw = (response.headers.get("ETag") or "").strip()
    match = re.fullmatch(r'"([1-9][0-9]*)"', raw)
    if match is None:
        raise ServingError("serving response omitted a valid quoted ETag revision")
    return int(match.group(1))


def read_adapter_record(run_id: str) -> dict | None:
    """Read the durable serving row directly, including its CAS identity."""
    response = _serving_request(
        "GET", f"{serving_base_url()}/adapters/{run_id}", ok_statuses=(404,)
    )
    if response.status_code == 404:
        return None
    try:
        record = response.json()
    except ValueError as exc:
        raise ServingError(f"serving adapter record for {run_id} was not valid JSON") from exc
    if not isinstance(record, dict):
        raise ServingError(f"serving adapter record for {run_id} was malformed")
    revision = _etag_revision(response)
    body_revision = record.get("registry_revision")
    if body_revision != revision:
        raise ServingError(
            f"serving adapter record for {run_id} had inconsistent revision metadata"
        )
    return record


def record_matches(record: dict | None, desired: dict, revision: int) -> bool:
    if not isinstance(record, dict) or record.get("registry_revision") != revision:
        return False
    return all(record.get(key) == value for key, value in desired.items())


def _same_observed_record(left: dict | None, right: dict | None) -> bool:
    return left == right


def _readback_target(
    run_id: str,
    desired: dict,
    target_revision: int,
    prior: dict | None,
) -> dict:
    last: dict | None = None
    read_error: ServingError | None = None
    for attempt in range(READBACK_ATTEMPTS):
        if attempt:
            time.sleep(READBACK_DELAY_SECONDS * attempt)
        try:
            last = read_adapter_record(run_id)
        except ServingError as exc:
            read_error = exc
            continue
        if record_matches(last, desired, target_revision):
            return last
        if last is not None and (
            last.get("registry_revision") != (prior or {}).get("registry_revision")
            or last.get("mutation_id") != (prior or {}).get("mutation_id")
        ):
            raise DeploymentSuperseded(
                f"adapter {run_id} deployment was superseded by another registry mutation"
            )
    if read_error is not None:
        raise read_error
    if _same_observed_record(last, prior):
        raise ServingError(f"adapter {run_id} registry mutation did not commit")
    raise ServingError(f"adapter {run_id} registry readback did not confirm the requested mutation")


def deploy_adapter(
    run_id: str,
    model: str,
    hf_repo: str,
    adapter_prefix: str,
    *,
    mutation_id: str,
    dry_run: bool = False,
    lora_rank: int = 32,
    thinking: bool = False,
    structured_outputs: str = "",
    org_id: str | None = None,
    before_registry_mutation: Callable[[int | None, dict, int, str, str], None] | None = None,
) -> Deployment:
    """Create a new immutable serving record through compare-and-swap."""
    validate_serving_lora_rank(model, lora_rank, rank_source="configured train.lora_rank")
    dep = deployment_record(run_id, model, adapter_prefix, state="dry_run" if dry_run else "ready")
    if dry_run:
        return dep
    owner = str(org_id or "").strip()
    if not owner:
        raise ServingError("org_id is required for a persisted serving deployment")
    subfolder = f"{adapter_prefix}/adapter"
    repo_revision = resolve_repo_revision(hf_repo)
    validate_serving_lora_rank(
        model,
        adapter_artifact_lora_rank(hf_repo, subfolder, repo_revision),
        rank_source="adapter artifact",
    )
    mutation_id = str(mutation_id).strip()
    if not mutation_id:
        raise ServingError("mutation_id is required for a deployment attempt")
    desired = {
        "adapter_id": run_id,
        "repo_id": hf_repo,
        "repo_type": "dataset",
        "repo_revision": repo_revision,
        "subfolder": subfolder,
        "base_model": model,
        "org_id": owner,
        "checkpoint": intended_checkpoint(run_id, adapter_prefix),
        "mutation_id": mutation_id,
        "thinking": bool(thinking),
        "status": "ready",
    }
    structured = _structured_outputs_body(structured_outputs)
    if structured is not None:
        desired["structured_outputs"] = structured
    prior = read_adapter_record(run_id)
    prior_revision = prior.get("registry_revision") if prior is not None else None
    if prior is not None and prior.get("org_id") != owner:
        raise ServingError(
            f"adapter {run_id} belongs to another org; replacement cannot change owner"
        )
    target_revision = 1 if prior_revision is None else int(prior_revision) + 1
    if before_registry_mutation is not None:
        before_registry_mutation(
            prior_revision,
            dict(desired),
            target_revision,
            mutation_id,
            repo_revision,
        )
    headers = {"If-Match": str(prior_revision)} if prior_revision is not None else None
    response: httpx.Response | None = None
    mutation_error: ServingError | None = None
    try:
        response = _serving_request(
            "POST", f"{serving_base_url()}/adapters", json=desired, headers=headers
        )
    except ServingError as exc:
        mutation_error = exc
        if (
            exc.status_code is not None
            and exc.status_code < 500
            and exc.status_code not in {409, 412}
        ):
            raise
    _readback_target(run_id, desired, target_revision, prior)
    if response is not None and _etag_revision(response) != target_revision:
        raise ServingError(f"adapter {run_id} POST returned an unexpected registry revision")
    if mutation_error is not None:
        logger.warning(
            "adapter registration for %s returned %s but exact readback confirmed revision %d",
            run_id,
            mutation_error,
            target_revision,
        )
    dep.desired_record = desired
    dep.prior_revision = prior_revision
    dep.target_revision = target_revision
    dep.mutation_id = mutation_id
    dep.repo_revision = repo_revision
    return dep


def disable_owned_adapter(run_id: str, revision: int, mutation_id: str) -> bool:
    """Disable only the exact serving mutation owned by this attempt."""
    before = read_adapter_record(run_id)
    if before is None:
        return False
    if before.get("registry_revision") != revision or before.get("mutation_id") != mutation_id:
        raise DeploymentSuperseded(
            f"adapter {run_id} is no longer owned by this deployment attempt"
        )
    if before.get("status") == "disabled":
        return True
    response: httpx.Response | None = None
    error: ServingError | None = None
    try:
        response = _serving_request(
            "DELETE",
            f"{serving_base_url()}/adapters/{run_id}",
            headers={
                "If-Match": str(revision),
                "X-Freesolo-Expected-Mutation-ID": mutation_id,
            },
        )
    except ServingError as exc:
        error = exc
        if (
            exc.status_code is not None
            and exc.status_code < 500
            and exc.status_code not in {409, 412}
        ):
            raise
    target_revision = revision + 1
    last = None
    read_error: ServingError | None = None
    for attempt in range(READBACK_ATTEMPTS):
        if attempt:
            time.sleep(READBACK_DELAY_SECONDS * attempt)
        try:
            last = read_adapter_record(run_id)
        except ServingError as exc:
            read_error = exc
            continue
        if (
            isinstance(last, dict)
            and last.get("registry_revision") == target_revision
            and last.get("mutation_id") == mutation_id
            and last.get("status") == "disabled"
        ):
            if response is not None and _etag_revision(response) != target_revision:
                raise ServingError(
                    f"adapter {run_id} DELETE returned an unexpected registry revision"
                )
            return True
        if isinstance(last, dict) and (
            last.get("registry_revision") != revision or last.get("mutation_id") != mutation_id
        ):
            raise DeploymentSuperseded(f"adapter {run_id} disable was superseded")
    if read_error is not None:
        raise read_error
    if _same_observed_record(last, before):
        raise error or ServingError(f"adapter {run_id} disable did not commit")
    raise ServingError(f"adapter {run_id} disable readback was inconclusive")


def undeploy_adapter(run_id: str) -> list[str]:
    current = read_adapter_record(run_id)
    if current is None or current.get("status") == "disabled":
        return []
    disable_owned_adapter(run_id, int(current["registry_revision"]), str(current["mutation_id"]))
    return [run_id]


def chat(
    run_id: str,
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 512,
    thinking: bool = False,
    expected_checkpoint: str | None = None,
    expected_registry_revision: int | None = None,
    expected_mutation_id: str | None = None,
) -> dict:
    body = {
        "model": run_id,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
    }
    headers = _internal_key_header()
    fence = (expected_checkpoint, expected_registry_revision, expected_mutation_id)
    if any(value is not None for value in fence):
        if not all(value is not None for value in fence):
            raise ValueError(
                "deployment smoke fence requires checkpoint, registry revision, and mutation id"
            )
        headers.update(
            {
                "X-Freesolo-Expected-Checkpoint": str(expected_checkpoint),
                "X-Freesolo-Expected-Registry-Revision": str(expected_registry_revision),
                "X-Freesolo-Expected-Mutation-ID": str(expected_mutation_id),
            }
        )
    with httpx.Client(follow_redirects=True, max_redirects=100, timeout=30 * 60.0) as client:
        response = client.post(
            f"{serving_openai_base_url()}/chat/completions", json=body, headers=headers
        )
    response.raise_for_status()
    return response.json()


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
            "POST",
            f"{serving_openai_base_url()}/chat/completions",
            json=body,
            headers=_internal_key_header(),
        ) as response,
    ):
        response.raise_for_status()
        if "application/json" in response.headers.get("content-type", ""):
            response.read()
            payload = response.json()
            content = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content")
            if content:
                yield str(content)
            return
        yield from _openai_stream_content(response.iter_lines())
