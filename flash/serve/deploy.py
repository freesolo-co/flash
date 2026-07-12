"""Thin client for the freesolo multi-LoRA serving app (Modal); no flash-side vLLM."""

from __future__ import annotations

import copy
import json
import os
import time
from collections.abc import Callable, Iterator
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
EXPECTED_REASONING_PARSER_BY_MODEL = {"Qwen/Qwen3.6-35B-A3B": "qwen3"}


class ServingError(RuntimeError):
    """Serving backend rejected a request or was unreachable; carries the upstream status."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        reconciliation_required: bool = False,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.reconciliation_required = reconciliation_required


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
    headers: dict[str, str] | None = None,
    ok_statuses: tuple[int, ...] = (),
) -> httpx.Response:
    """Issue a request to the serving backend; translates failures into ServingError."""
    # follow redirects because modal may return an async-result poll location.
    request_headers = {**_internal_key_header(), **(headers or {})}
    kwargs: dict = {"headers": request_headers, "timeout": 60.0, "follow_redirects": True}
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
    previous_registry_snapshot: dict | None = field(default=None, repr=False)
    registry_revision: int | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("previous_registry_snapshot", None)
        data.pop("registry_revision", None)
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


def _require_thinking_structured_outputs_capability(base: str, model: str) -> None:
    """Fail closed unless the requested model is live with the serving parser we support."""
    expected_parser = EXPECTED_REASONING_PARSER_BY_MODEL.get(model)
    if expected_parser is None:
        raise ServingError(
            "cannot safely deploy a thinking adapter with structured outputs because the requested "
            f"model {model!r} has no supported serving reasoning parser. No adapter registry mutation "
            "was attempted"
        )
    try:
        payload = _serving_request("GET", f"{base}/healthz").json()
    except (ServingError, ValueError) as exc:
        raise ServingError(
            "cannot safely deploy a thinking adapter with structured outputs because the serving "
            "capability probe was unreachable or malformed. No adapter registry mutation was "
            "attempted"
        ) from exc
    if not isinstance(payload, dict):
        raise ServingError(
            "cannot safely deploy a thinking adapter with structured outputs because /healthz "
            "returned a malformed response. No adapter registry mutation was attempted"
        )
    # the stable versioned handshake: serving only advertises this once a build has both the
    # parser configured and request-level reasoning gating live, so it must be present before we
    # trust the per-model liveness fields below.
    capabilities = payload.get("capabilities")
    if (
        not isinstance(capabilities, list)
        or THINKING_STRUCTURED_OUTPUTS_CAPABILITY not in capabilities
    ):
        raise ServingError(
            "cannot safely deploy a thinking adapter with structured outputs because serving does "
            f"not advertise the {THINKING_STRUCTURED_OUTPUTS_CAPABILITY!r} capability. No adapter "
            "registry mutation was attempted"
        )
    parsers = payload.get("reasoning_parser_by_model")
    states = payload.get("deferred_structured_outputs_by_model")
    parser = parsers.get(model) if isinstance(parsers, dict) else None
    state = states.get(model) if isinstance(states, dict) else None
    if parser != expected_parser or not isinstance(state, dict):
        raise ServingError(
            "cannot safely deploy a thinking adapter with structured outputs because the requested "
            f"model {model!r} does not advertise parser {expected_parser!r}. No adapter registry "
            "mutation was attempted"
        )
    if state.get("status") != "live" or state.get("verified") is not True:
        raise ServingError(
            "cannot safely deploy a thinking adapter with structured outputs because the requested "
            f"model {model!r} is not live with verified deferred enforcement. No adapter registry "
            "mutation was attempted"
        )


def _snapshot_url(run_id: str) -> str:
    return f"{serving_base_url()}/internal/adapter-registry/{run_id}"


def _etag_revision(response: httpx.Response) -> int:
    raw = (response.headers.get("ETag") or "").strip().strip('"')
    try:
        revision = int(raw)
    except ValueError as exc:
        raise ServingError("serving mutation response omitted a valid ETag revision") from exc
    if revision < 0:
        raise ServingError("serving mutation response returned a negative ETag revision")
    return revision


def _read_registry_snapshot(run_id: str) -> dict | None:
    response = _serving_request("GET", _snapshot_url(run_id), ok_statuses=(404,))
    if response.status_code == 404:
        return None
    try:
        snapshot = response.json()
    except ValueError as exc:
        raise ServingError(f"serving registry snapshot for {run_id} was not valid JSON") from exc
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("adapter"), dict):
        raise ServingError(f"serving registry snapshot for {run_id} was malformed")
    revision = snapshot.get("revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
        raise ServingError(f"serving registry snapshot for {run_id} had an invalid revision")
    return copy.deepcopy(snapshot)


def _record_mismatches(snapshot: dict | None, expected: dict, revision: int | None = None) -> list[str]:
    if not isinstance(snapshot, dict):
        return ["adapter record is absent"]
    record = snapshot.get("adapter")
    if not isinstance(record, dict):
        return ["adapter record is malformed"]
    comparisons = {
        "adapter_id": (record.get("adapter_id"), expected["adapter_id"]),
        "subfolder": (record.get("subfolder"), expected["subfolder"]),
        "base_model": (record.get("base_model"), expected["base_model"]),
        "repo_id": (record.get("repo_id"), expected["repo_id"]),
        "repo_type": (record.get("repo_type"), expected["repo_type"]),
        "status": (record.get("status"), expected["status"]),
        "thinking": (record.get("thinking"), expected["thinking"]),
        "structured_outputs": (record.get("structured_outputs"), expected.get("structured_outputs")),
        "structured_outputs_after_reasoning": (
            record.get("structured_outputs_after_reasoning"),
            expected.get("structured_outputs_after_reasoning"),
        ),
        "org_id": (snapshot.get("org_id"), expected.get("org_id")),
    }
    if revision is not None:
        comparisons["revision"] = (snapshot.get("revision"), revision)
    return [
        f"{name}={actual!r}, expected {wanted!r}"
        for name, (actual, wanted) in comparisons.items()
        if actual != wanted
    ]


def _same_snapshot(left: dict | None, right: dict | None) -> bool:
    return left == right


def _restored_snapshot_mismatches(actual: dict | None, expected: dict, revision: int) -> list[str]:
    if not isinstance(actual, dict):
        return ["adapter record is absent"]
    mismatches: list[str] = []
    if actual.get("org_id") != expected.get("org_id"):
        mismatches.append(
            f"org_id={actual.get('org_id')!r}, expected {expected.get('org_id')!r}"
        )
    if actual.get("revision") != revision:
        mismatches.append(f"revision={actual.get('revision')!r}, expected {revision!r}")
    actual_record = actual.get("adapter")
    expected_record = expected.get("adapter")
    if not isinstance(actual_record, dict) or not isinstance(expected_record, dict):
        return [*mismatches, "adapter record is malformed"]
    for name, wanted in expected_record.items():
        if name in {"created_at", "updated_at"}:
            continue
        if actual_record.get(name) != wanted:
            mismatches.append(
                f"adapter.{name}={actual_record.get(name)!r}, expected {wanted!r}"
            )
    return mismatches


def snapshot_adapter_record(run_id: str) -> dict | None:
    """Return the strongly consistent privileged registry snapshot."""
    return _read_registry_snapshot(run_id)


def registry_snapshot_matches(snapshot: dict | None, expected: dict, revision: int) -> bool:
    """Return whether a privileged snapshot exactly represents a requested mutation."""
    return not _record_mismatches(snapshot, expected, revision)


def restored_snapshot_matches(
    snapshot: dict | None, previous_snapshot: dict | None, revision: int
) -> bool:
    """Return whether a snapshot equals a restored prior record at ``revision``.

    A completed prior-record rollback replays ``previous_snapshot``'s content but lands one
    revision past the mutation and bumps ``updated_at``, so this compares fields rather than
    requiring exact dict equality.
    """
    if previous_snapshot is None:
        return False
    return not _restored_snapshot_mismatches(snapshot, previous_snapshot, revision)


def restore_adapter_record(
    run_id: str,
    previous_snapshot: dict | None,
    current_revision: int,
) -> int | None:
    """Restore a prior snapshot only while the caller still owns current_revision."""
    headers = {"If-Match": f'"{current_revision}"'}
    if previous_snapshot is None:
        response = _serving_request(
            "DELETE", f"{serving_base_url()}/adapters/{run_id}", headers=headers
        )
        committed = _etag_revision(response)
        readback = _read_registry_snapshot(run_id)
        record = readback.get("adapter") if isinstance(readback, dict) else None
        if (
            not isinstance(record, dict)
            or record.get("status") != "disabled"
            or readback.get("revision") != committed
        ):
            raise ServingError(
                f"adapter {run_id} rollback delete did not produce a disabled tombstone at "
                f"revision {committed}; reconciliation required"
            )
        return committed

    response = _serving_request(
        "PUT",
        _snapshot_url(run_id),
        json=copy.deepcopy(previous_snapshot),
        headers=headers,
    )
    committed = _etag_revision(response)
    readback = _read_registry_snapshot(run_id)
    mismatches = _restored_snapshot_mismatches(readback, previous_snapshot, committed)
    if mismatches:
        raise ServingError(
            f"adapter {run_id} rollback readback did not match the prior snapshot at revision "
            f"{committed}: {'; '.join(mismatches)}; reconciliation required"
        )
    return committed


def _restore_after_failure(
    run_id: str,
    previous_snapshot: dict | None,
    current_revision: int,
    exc: Exception,
) -> ServingError:
    try:
        restore_adapter_record(run_id, previous_snapshot, current_revision)
    except Exception as rollback_exc:
        return ServingError(
            f"{exc}; serving rollback failed ({rollback_exc}); reconciliation required",
            status_code=getattr(exc, "status_code", None),
            reconciliation_required=True,
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
    before_registry_mutation: Callable[[dict | None, dict, int], None] | None = None,
) -> Deployment:
    """Register an adapter through the revisioned serving registry contract."""
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
        _require_thinking_structured_outputs_capability(base, model)
    body = {
        "adapter_id": run_id,
        "repo_id": hf_repo,
        "base_model": model,
        "subfolder": subfolder,
        "repo_type": "dataset",
        "status": "ready",
        "thinking": bool(thinking),
    }
    if so_default is not None:
        key = "structured_outputs_after_reasoning" if thinking else "structured_outputs"
        body[key] = so_default

    previous_snapshot = snapshot_adapter_record(run_id)
    previous_revision = previous_snapshot["revision"] if previous_snapshot is not None else None
    previous_org = previous_snapshot.get("org_id") if previous_snapshot is not None else None
    requested_org = (org_id or "").strip() or None
    if previous_snapshot is not None:
        if requested_org is not None and requested_org != previous_org:
            raise ServingError(
                f"adapter {run_id} belongs to org {previous_org!r}; replacement cannot change owner"
            )
        body["org_id"] = previous_org
    elif requested_org is not None:
        body["org_id"] = requested_org

    target_revision = (previous_revision or 0) + 1
    dep.previous_registry_snapshot = copy.deepcopy(previous_snapshot)
    if before_registry_mutation is not None:
        before_registry_mutation(
            copy.deepcopy(previous_snapshot), copy.deepcopy(body), target_revision
        )

    headers = (
        {"If-Match": f'"{previous_revision}"'} if previous_revision is not None else None
    )
    try:
        response = _serving_request(
            "POST", f"{base}/adapters", json=body, headers=headers
        )
    except ServingError as exc:
        if exc.status_code is not None and exc.status_code < 500:
            raise
        try:
            snapshot = _verify_adapter_registered(run_id, body, target_revision)
        except ServingError as readback_exc:
            observed = _read_registry_snapshot(run_id)
            if _same_snapshot(observed, previous_snapshot):
                raise ServingError(
                    f"{exc}; privileged readback confirms the registry mutation did not commit",
                    status_code=exc.status_code,
                ) from exc
            raise ServingError(
                f"{exc}; ambiguous registration outcome: {readback_exc}; reconciliation required",
                status_code=exc.status_code,
                reconciliation_required=True,
            ) from exc
        logger.warning(
            "adapter registration for %s returned an ambiguous error (%s), but the privileged "
            "snapshot confirms revision %d",
            run_id,
            exc,
            target_revision,
        )
    else:
        # the post returned 2xx, so the mutation committed. any failure reading back or
        # confirming it now leaves a live-but-unverified revision, so fail closed to
        # reconciliation rather than a plain error that could restore a stale local record.
        try:
            committed_revision = _etag_revision(response)
        except ServingError as exc:
            raise ServingError(
                f"serving accepted adapter {run_id} but returned an unusable ETag ({exc}); "
                "the mutation may have committed; reconciliation required",
                reconciliation_required=True,
            ) from exc
        if committed_revision != target_revision:
            raise ServingError(
                f"serving committed adapter {run_id} at revision {committed_revision}, expected "
                f"{target_revision}; reconciliation required",
                reconciliation_required=True,
            )
        try:
            snapshot = _verify_adapter_registered(run_id, body, target_revision)
        except Exception as exc:
            raise ServingError(
                f"serving accepted adapter {run_id} at revision {target_revision} but readback "
                f"did not confirm it ({exc}); reconciliation required",
                reconciliation_required=True,
            ) from exc

    dep.registry_revision = snapshot["revision"]
    logger.info(
        "registered adapter %s with freesolo serving at revision %d (%s)",
        run_id,
        dep.registry_revision,
        base,
    )
    return dep


def _verify_adapter_registered(run_id: str, expected: dict, revision: int) -> dict:
    """Poll the strongly consistent snapshot until the requested revision is visible."""
    mismatches = ["adapter record is absent"]
    snapshot = None
    for attempt in range(READBACK_ATTEMPTS):
        if attempt:
            time.sleep(READBACK_DELAY_SECONDS * attempt)
        snapshot = _read_registry_snapshot(run_id)
        mismatches = _record_mismatches(snapshot, expected, revision)
        if not mismatches:
            return snapshot
    raise ServingError(
        f"adapter {run_id} registry snapshot does not match the requested mutation: "
        + "; ".join(mismatches)
    )


def undeploy_adapter(run_id: str) -> list[str]:
    """Disable an adapter through revisioned compare-and-swap deletion."""
    base = serving_base_url()
    snapshot = _read_registry_snapshot(run_id)
    if snapshot is None:
        return []
    record = snapshot.get("adapter")
    if not isinstance(record, dict):
        raise ServingError(f"serving registry snapshot for {run_id} was malformed")
    if record.get("status") == "disabled":
        return []
    current_revision = snapshot["revision"]
    target_revision = current_revision + 1
    headers = {"If-Match": f'"{current_revision}"'}
    mutation_error: ServingError | None = None
    try:
        response = _serving_request(
            "DELETE", f"{base}/adapters/{run_id}", headers=headers, ok_statuses=(404,)
        )
    except ServingError as exc:
        if exc.status_code is not None and exc.status_code < 500:
            raise
        mutation_error = exc
        response = None

    readback = _read_registry_snapshot(run_id)
    readback_record = readback.get("adapter") if isinstance(readback, dict) else None
    committed = _etag_revision(response) if response is not None and response.status_code != 404 else None
    if (
        isinstance(readback_record, dict)
        and readback_record.get("status") == "disabled"
        and readback.get("revision") == target_revision
        and committed in {None, target_revision}
    ):
        logger.info(
            "deregistered adapter %s from freesolo serving at revision %d (%s)",
            run_id,
            target_revision,
            base,
        )
        return [run_id]
    if response is not None and response.status_code == 404 and readback is None:
        return []
    prefix = f"{mutation_error}; " if mutation_error is not None else ""
    raise ServingError(
        f"{prefix}adapter {run_id} delete outcome did not produce the expected disabled revision "
        f"{target_revision}; reconciliation required",
        status_code=mutation_error.status_code if mutation_error is not None else None,
        reconciliation_required=True,
    )


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
