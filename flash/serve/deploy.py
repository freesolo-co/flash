"""Thin client for the freesolo multi-LoRA serving app (Modal); no flash-side vLLM."""

from __future__ import annotations

import atexit
import contextlib
import json
import math
import os
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from urllib.parse import quote

import httpx

from flash._channel import CHANNEL
from flash._logging import get_logger
from flash.adapter_artifacts import ADAPTER_WEIGHT_FILES
from flash.engine.structured_outputs import parse_structured_outputs
from flash.lora_rank import rank_from_adapter_config
from flash.schema import format_adapter_revision
from flash.serve.urls import is_freesolo_hosted_url, openai_base_url, serving_control_url

logger = get_logger(__name__)

PROD_FREESOLO_SERVING_URL = "https://serve.freesolo.co"
DEV_FREESOLO_SERVING_URL = "https://serve-dev.freesolo.co"


def default_serving_url(channel: str = CHANNEL) -> str:
    """Default serving control root for the given release channel.

    The serving plane is per-channel for the same reason the control plane is: each is backed by
    its own Supabase project, and an org row exists in exactly one of them. `flash-dev` sending a
    dev org_id to prod serving fails the prod `org_id` foreign key -- a 23503 no amount of retrying
    or GPU warming can fix, because the row is in the other database.

    This constant was the last hosted endpoint that did not derive from CHANNEL: `client.config`
    has split prod/dev since the channel was introduced, so `flash-dev` talked to
    `flash-dev.freesolo.co` for control and `serve.freesolo.co` for serving. freesolo #667 stood up
    the isolated dev serving plane at `serve-dev.freesolo.co` (Modal env `dev`, dev Supabase,
    `api-dev.freesolo.co`) and named this fallback as the defect; this is the flash half of it.
    """
    return DEV_FREESOLO_SERVING_URL if channel == "dev" else PROD_FREESOLO_SERVING_URL


DEFAULT_FREESOLO_SERVING_URL = default_serving_url()
READBACK_DELAY_SECONDS = 0.5
READBACK_MAX_DELAY_SECONDS = 2.0
REVISION_READY_BUDGET_SECONDS = 5 * 60.0
ACTIVATION_READBACK_ATTEMPTS = 3
ACTIVATION_READBACK_DELAY_SECONDS = 2.0
# smoke-retry fallback when a 503 carries no usable Retry-After: keep the prior 2s default rather
# than the 0.5s readiness backoff base, so cold-start smoke retries don't hammer serving.
SMOKE_RETRY_FALLBACK_DELAY_SECONDS = 2.0
THINKING_STRUCTURED_OUTPUTS_CAPABILITY = "thinking_structured_outputs_deferred_v1"
REVISION_PROVENANCE_CAPABILITY = "revision_provenance"
_RETRYABLE_SMOKE_503_CODES = frozenset({"adapter_loading", "engine_unavailable"})
_HTTP_CLIENT: httpx.Client | None = None
_CHAT_HTTP_CLIENT: httpx.Client | None = None
_STREAM_HTTP_CLIENT: httpx.Client | None = None
_HTTP_CLIENT_LOCK = threading.Lock()


class ServingError(RuntimeError):
    """Serving backend rejected a request or was unreachable; carries the upstream status."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class ActivationOutcomeUnknown(ServingError):
    """Alias activation cannot be resolved to the attempted or previous revision."""

    def __init__(self, run_id: str, attempted_revision: str, *, detail: str | None = None):
        reason = detail or "authoritative alias readback failed"
        super().__init__(
            "alias_activation_unknown: serving may have activated "
            f"{attempted_revision} for {run_id}, but {reason}; retry deployment reconciliation "
            "before treating either revision as authoritative"
        )
        self.run_id = run_id
        self.attempted_revision = attempted_revision


class RetryableServingUnavailable(ServingError):
    """a recognized serving cold-start envelope that may be retried within a caller deadline."""

    def __init__(self, code: str, retry_after_seconds: float):
        super().__init__(
            f"serving_retryable_unavailable: {code}",
            status_code=503,
        )
        self.code = code
        self.retry_after_seconds = retry_after_seconds


class AdapterConfigMissing(ServingError):
    """The adapter's adapter_config.json could not be read from HF (artifact likely absent)."""


class AdapterTensorMissing(ServingError):
    """The adapter artifact has metadata but no loadable LoRA tensor file."""


def _is_adapter_tensor_filename(filename: str) -> bool:
    name = filename.rsplit("/", 1)[-1]
    if name in ADAPTER_WEIGHT_FILES:
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


def _http_client() -> httpx.Client:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        with _HTTP_CLIENT_LOCK:
            if _HTTP_CLIENT is None:
                _HTTP_CLIENT = httpx.Client(follow_redirects=True, max_redirects=100)
    return _HTTP_CLIENT


def _chat_http_client() -> httpx.Client:
    global _CHAT_HTTP_CLIENT
    if _CHAT_HTTP_CLIENT is None:
        with _HTTP_CLIENT_LOCK:
            if _CHAT_HTTP_CLIENT is None:
                _CHAT_HTTP_CLIENT = httpx.Client(
                    follow_redirects=True,
                    max_redirects=100,
                    limits=httpx.Limits(max_connections=None, max_keepalive_connections=100),
                )
    return _CHAT_HTTP_CLIENT


def _stream_http_client() -> httpx.Client:
    global _STREAM_HTTP_CLIENT
    if _STREAM_HTTP_CLIENT is None:
        with _HTTP_CLIENT_LOCK:
            if _STREAM_HTTP_CLIENT is None:
                _STREAM_HTTP_CLIENT = httpx.Client(
                    follow_redirects=True,
                    max_redirects=100,
                    limits=httpx.Limits(max_connections=None, max_keepalive_connections=100),
                )
    return _STREAM_HTTP_CLIENT


def _close_http_client() -> None:
    global _CHAT_HTTP_CLIENT, _HTTP_CLIENT, _STREAM_HTTP_CLIENT
    with _HTTP_CLIENT_LOCK:
        clients = (_HTTP_CLIENT, _CHAT_HTTP_CLIENT, _STREAM_HTTP_CLIENT)
        _HTTP_CLIENT = None
        _CHAT_HTTP_CLIENT = None
        _STREAM_HTTP_CLIENT = None
    for client in clients:
        if client is not None:
            client.close()


atexit.register(_close_http_client)


def _serving_request(
    method: str,
    url: str,
    *,
    json: dict | None = None,
    ok_statuses: tuple[int, ...] = (),
    timeout_s: float | None = None,
) -> httpx.Response:
    """Issue a request to the serving backend; translates failures into ServingError."""
    # follow_redirects: modal 303-redirects slow requests to an async-result poll url.
    timeout = 60.0 if timeout_s is None else min(60.0, max(0.0, float(timeout_s)))
    kwargs: dict = {"headers": _internal_key_header(), "timeout": timeout, "follow_redirects": True}
    if json is not None:
        kwargs["json"] = json
    try:
        resp = _http_client().request(method, url, **kwargs)
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
    headers = getattr(resp, "headers", {}) if resp is not None else {}
    retry_after = headers.get("Retry-After")
    return ServingError(msg, status_code=status, retry_after=retry_after)


def serving_base_url() -> str:
    """Env-overridable serving control root.

    A standalone plane must configure this explicitly, and to a backend it OPERATES. Every
    serving request carries ``FREESOLO_INTERNAL_KEY`` (see ``_internal_key_header``), and on a
    self-hosted plane that key is the credential granting full control of the plane itself.
    Sending it to a third party the operator has no relationship with, on an ordinary
    ``flash deploy`` or ``flash chat``, is the failure this guards.

    Both ways of arriving at the hosted backend are refused, not just the unset one: an operator
    who copied a managed ``.env`` has ``FREESOLO_SERVING_URL`` already SET to it, so a guard on
    the fallback alone would miss the more likely case. Raising here covers every caller
    (including ``serving_openai_base_url``) rather than stripping the header at one call site,
    and the error names the fix.
    """
    # imported lazily: flash.serve is the CLIENT side, and a module-level import would pull
    # flash.server into every CLI invocation.
    from flash.server.auth import standalone

    configured = (os.environ.get("FREESOLO_SERVING_URL") or "").strip()
    if standalone() and (not configured or is_freesolo_hosted_url(configured)):
        raise ServingError(
            f"FREESOLO_SERVING_URL is {'not set' if not configured else 'a Freesolo-hosted URL'}. "
            "A standalone plane has no serving backend of its own, and using the hosted one would "
            "send FREESOLO_INTERNAL_KEY - the key that controls this plane - to a service you do "
            "not operate. Point FREESOLO_SERVING_URL at your own multi-LoRA deployment, or export "
            "the adapter and serve it yourself (see SELF_HOSTING.md). Training does not require "
            "this."
        )
    return serving_control_url(configured or DEFAULT_FREESOLO_SERVING_URL)


def serving_openai_base_url() -> str:
    """OpenAI-compatible base URL for the configured serving backend."""
    return openai_base_url(serving_base_url())


def _internal_key_header() -> dict[str, str]:
    # Stripped for the same reason `authenticate` strips: the two must agree on what the key IS.
    # A trailing newline (routine in a `.env` file) authenticates fine against the plane but is an
    # illegal header value, so httpx rejects the request outright; a stray space authenticates and
    # then presents a DIFFERENT credential to the serving backend. Either way deploy/undeploy/chat
    # break for a config the plane itself accepts. Blank collapses to no header rather than an
    # empty one, which is what an unset key already does.
    key = (os.environ.get("FREESOLO_INTERNAL_KEY") or "").strip()
    return {"X-Freesolo-Internal-Key": key} if key else {}


@dataclass
class Deployment:
    run_id: str
    model: str
    adapter_hf_prefix: str
    openai_model: str
    endpoint_name: str
    openai_base_url: str
    adapter_revision: str | None = None
    checkpoint_step: int | None = None
    verified_at: float | None = None
    state: str = "ready"

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_hf_revision(hf_repo: str) -> str:
    """Resolve the full immutable commit SHA for the uploaded adapter repository."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover
        raise ServingError(
            "could not resolve adapter revision: huggingface_hub is not installed"
        ) from exc
    try:
        revision = str(
            HfApi()
            .repo_info(
                repo_id=hf_repo,
                repo_type="dataset",
                token=os.environ.get("HF_TOKEN"),
            )
            .sha
            or ""
        ).strip()
    except Exception as exc:
        raise ServingError(f"could not resolve adapter revision for {hf_repo}: {exc}") from exc
    if len(revision) != 40 or any(char not in "0123456789abcdefABCDEF" for char in revision):
        raise ServingError(f"could not resolve full Hub commit SHA for {hf_repo}")
    return revision.lower()


def deployment_record(
    run_id: str,
    model: str,
    adapter_prefix: str,
    *,
    state: str = "ready",
    checkpoint_step: int | None = None,
    adapter_revision: str | None = None,
) -> Deployment:
    subfolder = f"{adapter_prefix}/adapter"
    base = serving_base_url()
    openai_url = serving_openai_base_url()
    return Deployment(
        run_id=run_id,
        model=model,
        adapter_hf_prefix=subfolder,
        openai_model=run_id,
        adapter_revision=adapter_revision,
        checkpoint_step=checkpoint_step,
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


def _verify_adapter_artifact_tensors(hf_repo: str, subfolder: str, *, hf_revision: str) -> None:
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
                revision=hf_revision,
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


def adapter_artifact_lora_rank(hf_repo: str, subfolder: str, *, hf_revision: str) -> int:
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
            revision=hf_revision,
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
    _verify_adapter_artifact_tensors(hf_repo, subfolder, hf_revision=hf_revision)
    return rank_from_adapter_config(config, source=f"{hf_repo}:{filename}")


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
    checkpoint_step: int | None = None,
    expected_adapter_revision: str | None = None,
    before_activate: Callable[[str, str], None] | None = None,
) -> Deployment:
    """Register, verify, and atomically activate one immutable adapter revision.

    Thinking adapters with structured outputs require serving to advertise deferred constraint
    support before the immutable revision is registered.
    """
    validate_serving_lora_rank(model, lora_rank, rank_source="configured train.lora_rank")
    subfolder = f"{adapter_prefix}/adapter"
    dep = deployment_record(
        run_id,
        model,
        adapter_prefix,
        state="dry_run" if dry_run else "queued",
        checkpoint_step=checkpoint_step,
    )
    if dry_run:
        return dep

    hf_revision = resolve_hf_revision(hf_repo)
    validate_serving_lora_rank(
        model,
        adapter_artifact_lora_rank(hf_repo, subfolder, hf_revision=hf_revision),
        rank_source="adapter artifact",
    )
    revision = format_adapter_revision(run_id, checkpoint_step, hf_revision)
    dep.adapter_revision = revision
    checkpoint = f"{run_id}/step-{checkpoint_step}" if checkpoint_step is not None else run_id
    so_default = parse_structured_outputs(structured_outputs) if structured_outputs else None
    advertised = _require_serving_capabilities(
        thinking_structured_outputs=thinking and so_default is not None
    )
    require_provenance = REVISION_PROVENANCE_CAPABILITY in advertised

    body = {
        "adapter_id": revision,
        "repo_id": hf_repo,
        "base_model": model,
        "subfolder": subfolder,
        "repo_type": "dataset",
        "checkpoint": checkpoint,
        # NOTE: do NOT send a client-chosen "status" -- the serving backend sets the record status
        # itself on registration and rejects an incoming "status" as an extra field (HTTP 422
        # extra_forbidden). Sending it blocked every deploy against the deployed serving backend.
        "metadata": {
            "record_type": "revision",
            "run_id": run_id,
            "checkpoint_step": checkpoint_step,
            "hf_revision": hf_revision,
        },
        "thinking": bool(thinking),
    }
    if so_default is not None:
        body["structured_outputs"] = so_default
    normalized_org_id = (org_id or "").strip()
    if normalized_org_id:
        body["org_id"] = normalized_org_id

    try:
        registration = _serving_request("POST", f"{serving_base_url()}/adapters", json=body)
        if registration.status_code not in {200, 202}:
            raise ServingError(
                f"serving returned unexpected adapter registration status {registration.status_code}"
            )
    except ServingError as exc:
        if exc.status_code is not None and exc.status_code < 500:
            raise
        try:
            record = _registered_adapter(revision)
        except ServingError as read_exc:
            raise exc from read_exc
        if record is None or not _matches_revision_identity(
            record, body, require_provenance=require_provenance
        ):
            raise exc
        logger.warning(
            "adapter registration response was ambiguous; revision %s exists with matching identity",
            revision,
        )

    _wait_revision_ready(
        revision, subfolder, expected_identity=body, require_provenance=require_provenance
    )
    if before_activate is not None:
        before_activate(revision, checkpoint)
    activation = _activate_revision(
        run_id,
        revision,
        checkpoint,
        expected_adapter_revision=expected_adapter_revision,
    )
    dep.state = "ready"
    dep.openai_model = str(activation.get("adapter_id") or run_id)
    logger.info("activated adapter %s revision %s", run_id, revision)
    return dep


def _adapter_url(adapter_id: str) -> str:
    return f"{serving_base_url()}/adapters/{quote(adapter_id, safe='')}"


def _registered_adapter_response(
    adapter_id: str, *, timeout_s: float | None = None
) -> tuple[dict | None, httpx.Response]:
    """Read one authoritative adapter record and retain its polling headers."""
    resp = _serving_request(
        "GET",
        _adapter_url(adapter_id),
        ok_statuses=(404,),
        timeout_s=timeout_s,
    )
    if resp.status_code == 404:
        return None, resp
    try:
        payload = resp.json()
    except ValueError as exc:
        raise ServingError(
            f"serving returned invalid status JSON for adapter {adapter_id}"
        ) from exc
    if not isinstance(payload, dict):
        raise ServingError(f"serving returned invalid status data for adapter {adapter_id}")
    record = payload.get("adapter") if isinstance(payload.get("adapter"), dict) else payload
    if not isinstance(record, dict):
        raise ServingError(f"serving returned no adapter record for {adapter_id}")
    return record, resp


def _registered_adapter(adapter_id: str, *, timeout_s: float | None = None) -> dict | None:
    """Read one authoritative adapter record, including disabled records."""
    record, _ = _registered_adapter_response(adapter_id, timeout_s=timeout_s)
    return record


def _matches_revision_identity(
    record: dict, expected: dict, *, require_provenance: bool = True
) -> bool:
    scalar_fields = (
        "adapter_id",
        "repo_id",
        "repo_type",
        "subfolder",
        "base_model",
        "checkpoint",
        "thinking",
    )
    if any(record.get(field) != expected.get(field) for field in scalar_fields):
        return False
    if (record.get("org_id") or None) != (expected.get("org_id") or None):
        return False
    if (record.get("structured_outputs") or None) != (expected.get("structured_outputs") or None):
        return False
    if not require_provenance:
        # backends without revision_provenance do not echo provenance metadata; the immutable
        # adapter_id already pins the artifact, so this cross-check is best effort here.
        return True
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    expected_metadata = expected["metadata"]
    return all(
        metadata.get(field) == expected_metadata.get(field)
        for field in ("record_type", "run_id", "checkpoint_step", "hf_revision")
    )


def _require_serving_capabilities(*, thinking_structured_outputs: bool = False) -> set[str]:
    # SAFETY-CRITICAL capabilities the deploy correctness contract genuinely depends on: immutable
    # revisions (a revision id always maps to one artifact) and an atomic alias compare-and-swap
    # (the alias flip can't race). These are hard-required.
    required = {
        "immutable_adapter_revisions",
        "alias_compare_and_swap",
    }
    # PREFERRED (not required): `revision_provenance` only lets the serving backend echo back the
    # run/checkpoint/hf_revision metadata, which is used ONLY on the rare 5xx-during-registration
    # recovery path (`_matches_revision_identity`). Hard-requiring it blocked EVERY deploy org-wide
    # whenever the serving build lagged on advertising it, even though the happy path never uses it.
    # So its absence is a logged warning, not a deploy-blocking error.
    preferred = {REVISION_PROVENANCE_CAPABILITY}
    if thinking_structured_outputs:
        # Genuinely required for this run: thinking + structured outputs needs the serving backend's
        # deferred-constraint support (grammar applied after </think>) or served output is invalid.
        required.add(THINKING_STRUCTURED_OUTPUTS_CAPABILITY)
    url = f"{serving_base_url()}/healthz"
    response = _serving_request("GET", url)
    try:
        payload = response.json()
    except ValueError as exc:
        raise ServingError(
            f"serving_contract_unsupported: serving health check at {url} did not return valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise ServingError(
            f"serving_contract_unsupported: serving health check at {url} returned a non-object payload"
        )
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, list):
        raise ServingError(
            f"serving_contract_unsupported: serving health check at {url} must return a list field "
            "named capabilities"
        )
    if not all(isinstance(capability, str) for capability in capabilities):
        raise ServingError(
            f"serving_contract_unsupported: serving health check at {url} capabilities must be "
            "strings"
        )
    advertised = set(capabilities)
    missing = sorted(required - advertised)
    if missing:
        raise ServingError(
            "serving_contract_unsupported: serving is missing required capabilities "
            + ", ".join(missing)
        )
    missing_preferred = sorted(preferred - advertised)
    if missing_preferred:
        # Not fatal: the happy-path deploy does not use these; only the ambiguous-registration
        # recovery path degrades (best-effort identity check). Surface it so an operator can ship
        # the serving build that advertises them, without blocking customers meanwhile.
        logger.warning(
            "serving backend does not advertise preferred capabilities %s; deploying anyway "
            "(ambiguous-registration recovery will be best-effort)",
            ", ".join(missing_preferred),
        )
    return advertised


def _readback_delay(attempt: int, retry_after: str | None = None) -> float:
    if retry_after is not None:
        try:
            delay = float(retry_after)
        except (TypeError, ValueError):
            pass
        else:
            if math.isfinite(delay) and delay > 0:
                return min(delay, READBACK_MAX_DELAY_SECONDS)
    return min(READBACK_DELAY_SECONDS * (2**attempt), READBACK_MAX_DELAY_SECONDS)


def _wait_revision_ready(
    revision: str,
    subfolder: str,
    *,
    expected_identity: dict | None = None,
    require_provenance: bool = True,
    budget_s: float = REVISION_READY_BUDGET_SECONDS,
) -> dict:
    deadline = time.monotonic() + max(0.0, float(budget_s))
    last_state = "registered"
    last_read_error: ServingError | None = None
    first_read = True
    attempt = 0
    retry_after: str | None = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if not first_read:
            delay = _readback_delay(attempt - 1, retry_after)
            time.sleep(min(delay, remaining))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
        first_read = False
        try:
            record, response = _registered_adapter_response(revision, timeout_s=remaining)
        except ServingError as exc:
            if exc.status_code is not None and exc.status_code < 500:
                raise
            last_read_error = exc
            retry_after = exc.retry_after
            attempt += 1
            continue
        # a read that only completes once the whole budget is spent was not confirmed ready in
        # time; do not honor it, and leave last_state untouched so the timeout is reported.
        if deadline - time.monotonic() <= 0:
            break
        retry_after = response.headers.get("Retry-After")
        attempt += 1
        # a record fetched within the deadline is still inspected for readiness before giving up.
        if record is None:
            continue
        last_read_error = None
        if expected_identity is not None and not _matches_revision_identity(
            record, expected_identity, require_provenance=require_provenance
        ):
            raise ServingError(
                f"adapter revision {revision} resolved to a different immutable identity"
            )
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        last_state = str(
            metadata.get("lifecycle_state") or record.get("lifecycle_state") or "registered"
        )
        failure = metadata.get("failure")
        if last_state == "failed" or record.get("status") == "disabled":
            raise ServingError(
                f"serving failed to load adapter revision {revision}: {failure or 'unknown error'}"
            )
        if last_state == "ready":
            value = record.get("subfolder")
            record_subfolder = str(value) if value is not None else None
            if record_subfolder == subfolder:
                return record
    if last_read_error is not None:
        raise ServingError(
            f"adapter revision {revision} readiness could not be confirmed after transient "
            f"serving errors: {last_read_error}"
        ) from last_read_error
    raise ServingError(
        f"adapter revision {revision} remained {last_state!r}; the previous alias remains available"
    )


def _active_alias_target(record: dict | None) -> str | None:
    if not isinstance(record, dict) or record.get("status") == "disabled":
        return None
    metadata = record.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("alias_of"), str):
        return metadata["alias_of"]
    return None


def adapter_alias_target(run_id: str) -> str | None:
    """Read the authoritative immutable revision targeted by a mutable run alias."""
    record = _registered_adapter(run_id)
    if record is None or record.get("status") == "disabled":
        return None
    target = _active_alias_target(record)
    if target is None:
        raise ServingError(
            f"serving alias {run_id} is not an immutable alias record; legacy aliases are unsupported"
        )
    return target


def _validate_activation_response(
    response: object,
    *,
    run_id: str,
    revision: str,
    checkpoint: str,
    expected_adapter_revision: str | None,
) -> dict:
    if not isinstance(response, dict):
        raise ServingError("serving returned an invalid alias activation response")
    if response.get("adapter_id") != run_id or response.get("target_adapter_revision") != revision:
        raise ServingError("serving returned mismatched committed alias activation provenance")
    if response.get("previous_adapter_revision") != expected_adapter_revision:
        raise ServingError("serving returned mismatched previous alias revision")
    if response.get("checkpoint") != checkpoint:
        raise ServingError("serving returned mismatched committed alias checkpoint")
    if not isinstance(response.get("updated_at"), str) or not response["updated_at"].strip():
        raise ServingError("serving returned committed alias activation without updated_at")
    return response


def _activate_revision(
    run_id: str,
    revision: str,
    checkpoint: str,
    *,
    expected_adapter_revision: str | None,
) -> dict:
    body = {"expected_adapter_revision": expected_adapter_revision}
    try:
        response = _serving_request(
            "POST",
            f"{_adapter_url(revision)}/activate",
            json=body,
        ).json()
        return _validate_activation_response(
            response,
            run_id=run_id,
            revision=revision,
            checkpoint=checkpoint,
            expected_adapter_revision=expected_adapter_revision,
        )
    except (ServingError, ValueError) as exc:
        alias = None
        read_error: ServingError | None = None
        target = None
        for attempt in range(ACTIVATION_READBACK_ATTEMPTS):
            if attempt:
                time.sleep(ACTIVATION_READBACK_DELAY_SECONDS)
            try:
                alias = _registered_adapter(run_id)
                read_error = None
            except ServingError as read_exc:
                read_error = read_exc
                continue
            target = _active_alias_target(alias)
            if target == revision:
                return {
                    "adapter_id": run_id,
                    "target_adapter_revision": revision,
                    "previous_adapter_revision": expected_adapter_revision,
                    "checkpoint": checkpoint,
                    "updated_at": alias.get("updated_at") if alias else None,
                }
            if target not in (None, expected_adapter_revision):
                break
        if read_error is not None:
            raise ActivationOutcomeUnknown(run_id, revision) from read_error
        if expected_adapter_revision is not None and target == expected_adapter_revision:
            raise ServingError(
                f"alias activation was not committed; {run_id} still targets {target!r}"
            ) from exc
        if target is None:
            raise ActivationOutcomeUnknown(
                run_id,
                revision,
                detail=(
                    "alias activation outcome is ambiguous because authoritative readback exposed "
                    "no target"
                ),
            ) from exc
        raise ActivationOutcomeUnknown(
            run_id,
            revision,
            detail=f"alias activation diverged because authoritative readback targets {target!r}",
        ) from exc


def undeploy_adapter(run_id: str) -> dict:
    """Disable the run alias and all immutable revisions without engine eviction."""
    response = _serving_request(
        "DELETE",
        _adapter_url(run_id),
        ok_statuses=(404,),
    )
    if response.status_code == 404:
        return {
            "run_id": run_id,
            "disabled_aliases": [],
            "disabled_revisions": [],
            "serving_deregistered": False,
        }
    try:
        payload = response.json()
    except ValueError as exc:
        raise ServingError("serving returned an invalid undeploy response") from exc
    if not isinstance(payload, dict) or payload.get("run_id") != run_id:
        raise ServingError("serving returned a mismatched undeploy response")
    for field in ("disabled_aliases", "disabled_revisions"):
        value = payload.setdefault(field, [])
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ServingError(f"serving returned invalid {field} in undeploy response")
    payload["serving_deregistered"] = bool(
        payload["disabled_aliases"] or payload["disabled_revisions"]
    )
    return payload


def _retryable_smoke_unavailable(
    response: httpx.Response,
    *,
    requested_model: str,
) -> RetryableServingUnavailable | None:
    if response.status_code != 503:
        return None
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("error"), dict):
        return None
    error = payload["error"]
    code = error.get("code")
    if (
        error.get("type") != "adapter_unavailable"
        or error.get("retryable") is not True
        or code not in _RETRYABLE_SMOKE_503_CODES
        or error.get("requested_model") != requested_model
        or error.get("adapter_revision") != requested_model
    ):
        return None
    raw_delay = response.headers.get("Retry-After") or error.get("retry_after_seconds")
    try:
        retry_after_seconds = float(raw_delay)
    except (TypeError, ValueError):
        retry_after_seconds = SMOKE_RETRY_FALLBACK_DELAY_SECONDS
    if not math.isfinite(retry_after_seconds) or retry_after_seconds <= 0:
        retry_after_seconds = SMOKE_RETRY_FALLBACK_DELAY_SECONDS
    return RetryableServingUnavailable(str(code), retry_after_seconds)


_TAG_CLOSE = "</think>"
_TAG_OPEN = "<think>"


def _inline_reasoning_block(content: str) -> tuple[int, int] | None:
    """Span of a balanced ``<think>...</think>`` in ``content``, or ``None``.

    Order matters: an answer may contain the literal ``<think>`` (a ``choice`` constraint, or JSON
    quoting it), so the close must be found after the open rather than anywhere in the string.
    """
    opened = content.find(_TAG_OPEN)
    if opened < 0:
        return None
    closed = content.find(_TAG_CLOSE, opened + len(_TAG_OPEN))
    if closed < 0:
        return None
    return opened, closed


def _is_sampled_delimiter(prefix: str, reasoning: str) -> bool:
    """Whether the text before an inline ``</think>`` marks it as the block's own close.

    Two shapes prove it: nothing before it (the opener went into the prompt), or text duplicating
    ``reasoning_content`` (a compatibility build emitting the reasoning both inline and on the
    field), with or without its own opener. Any other prefix is answer text mentioning the tag.

    Compared stripped, since the model often samples the delimiter on its own line.
    """
    body = prefix.strip()
    if body.startswith(_TAG_OPEN):
        # an explicit opener means a whole block precedes the close, so it is the repeat only if
        # its body is the reasoning. an empty body is the answer's own pair, not the delimiter.
        return body[len(_TAG_OPEN) :].strip() == reasoning.strip()
    return body in ("", reasoning.strip())


def _find_delimiter(buffer: str, start: int) -> int:
    """Index of the sampled close tag at or after ``start``, or ``-1``.

    A named seam for one `str.find`, so the streamed hold path can be measured for scan cost
    without instrumenting the buffer (deltas are coerced with `str()` before being appended).
    """
    return buffer.find(_TAG_CLOSE, start)


def _retained_delimiter_end(content: str, close: int, reasoning: str) -> int | None:
    """Index just past a retained sampled ``</think>``, or ``None`` if it is the answer itself.

    A compatibility build's retained delimiter and an answer that IS the delimiter are identical on
    the wire, so no prefix test separates them. What does is whether stripping leaves anything: a
    retained delimiter is followed by the answer, an answer that is the tag by nothing.
    """
    if not _is_sampled_delimiter(content[:close], reasoning):
        return None
    end = close + len(_TAG_CLOSE)
    if content[end:].strip():
        return end
    return None


def _duplicates_reasoning(content: str, inline: tuple[int, int], reasoning: str) -> bool:
    """Whether this inline pair is ``reasoning_content`` emitted a second time inline.

    A compatibility build repeats the reasoning where a reasoning phase belongs, ahead of the
    answer, so a pair appearing after answer text is the answer quoting the tag instead. Bodies are
    compared stripped, since the repeat tends to carry the newline the model sampled after it.
    """
    opened, closed = inline
    if content[:opened].strip():
        return False
    return content[opened + len(_TAG_OPEN) : closed].strip() == reasoning.strip()


def _balanced_thinking_content(message: dict, *, thinking: bool) -> str:
    """Fold a split-out ``reasoning_content`` back into a balanced ``<think>...</think>`` block.

    A thinking chat template renders the OPENING ``<think>`` into the *prompt*, so the model only
    samples the closing tag; serving's OpenAI surface then returns the reasoning in
    ``reasoning_content`` with just the answer in ``content``. A caller reading ``content`` alone
    watches the reasoning vanish, and one reading the raw text sees a stray ``</think>`` with no
    opener. Re-open the block so every flash-side consumer reads one balanced string.
    ``reasoning_content`` is left in place for callers that want the split.

    ``thinking`` is the request's own flag, not a guess from the text: this path also backs the
    public non-streaming chat route, where an ordinary answer quoting ``</think>`` must not be
    rewritten into a synthetic reasoning block.
    """
    content = str(message.get("content") or "")
    if not thinking:
        return content
    reasoning = message.get("reasoning_content")
    inline = _inline_reasoning_block(content)
    if not isinstance(reasoning, str):
        # tested by type, not falsiness: an explicitly empty string means the model closed its
        # block immediately, which still needs a pair, while absent means serving never split.
        close = content.find(_TAG_CLOSE)
        # the pair must be the one that matched the FIRST close, not merely present somewhere:
        # an unmatched close followed by an answer-side pair still needs re-opening.
        if inline is not None and inline[1] == close:
            return content
        # a legacy build predating the split leaves `reasoned</think>answer` inline with no field.
        # with nothing to contradict it, an unbalanced close can only be the sampled delimiter.
        if close >= 0:
            return f"{_TAG_OPEN}{content[:close]}{_TAG_CLOSE}{content[close + len(_TAG_CLOSE) :]}"
        # no close either: a plain answer. leave it rather than inventing a block around it.
        return content
    if inline is not None:
        if _duplicates_reasoning(content, inline, reasoning):
            return content
        opened, _ = inline
        close = content.find(_TAG_CLOSE)
        if close >= 0 and close < opened and _is_sampled_delimiter(content[:close], reasoning):
            # the retained close precedes the pair, so the pair belongs to the answer and this
            # close is the block's own. strip it and keep the answer whole.
            return f"{_TAG_OPEN}{reasoning}{_TAG_CLOSE}{content[close + len(_TAG_CLOSE) :]}"
        # a different pair: the answer itself contains a literal block. fold, so the reasoning
        # survives and the answer stays intact behind the delimiter.
        return f"{_TAG_OPEN}{reasoning}{_TAG_CLOSE}{content}"
    close = content.find(_TAG_CLOSE)
    if _is_terminal_reasoning_repeat(content, reasoning):
        # the repeat with nothing behind it, so there is no answer to keep. appending it would
        # hand the smoke `why</think>` as an answer and activate a deployment that answered
        # nothing; folding to the block alone lets the smoke reject it. same failure direction
        # the opener-carrying form already takes.
        return f"{_TAG_OPEN}{reasoning}{_TAG_CLOSE}"
    end = _retained_delimiter_end(content, close, reasoning) if close >= 0 else None
    if end is not None:
        return f"{_TAG_OPEN}{reasoning}{_TAG_CLOSE}{content[end:]}"
    # an empty reasoning string reaches here and still gets a pair: a thinking consumer splits the
    # answer on `</think>`, so a bare answer would read as no answer at all.
    return f"{_TAG_OPEN}{reasoning}{_TAG_CLOSE}{content}"


def _balance_thinking_payload(payload: object, *, thinking: bool) -> None:
    """Rewrite each choice's ``content`` in place so the returned payload is balanced."""
    if not isinstance(payload, dict):
        return
    for choice in payload.get("choices") or []:
        message = choice.get("message") if isinstance(choice, dict) else None
        if isinstance(message, dict):
            message["content"] = _balanced_thinking_content(message, thinking=thinking)


def chat(
    run_id: str,
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 512,
    thinking: bool = False,
    expected_checkpoint: str | None = None,
    timeout_s: float | None = None,
    retry_unavailable: bool = False,
    stop: list[str] | None = None,
) -> dict:
    """Send an OpenAI-style chat request for the run's adapter to freesolo serving.

    ``timeout_s`` overrides the default 30-minute request timeout. deployment smoke also enables
    recognized unavailable-envelope classification so its caller can retry within one deadline.
    ``stop`` carries the run's own stop sequences so a model trained to terminate on a delimiter
    rather than EOS finishes on ``stop`` instead of running to ``max_tokens``.
    """
    base = serving_openai_base_url()
    body = {
        "model": run_id,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
    }
    if stop:
        body["stop"] = [str(value) for value in stop]
    # follow_redirects + max_redirects=100: Modal 303-redirects slow cold-start requests across
    # several poll cycles before the result is ready.
    headers = _internal_key_header()
    if expected_checkpoint:
        headers["X-Freesolo-Expected-Checkpoint"] = expected_checkpoint
    timeout = 30 * 60.0 if timeout_s is None else max(0.0, float(timeout_s))
    client_context = (
        httpx.Client(follow_redirects=True, max_redirects=100)
        if retry_unavailable
        else contextlib.nullcontext(_chat_http_client())
    )
    with client_context as client:
        resp = client.post(f"{base}/chat/completions", json=body, headers=headers, timeout=timeout)
        if retry_unavailable:
            retryable_error = _retryable_smoke_unavailable(resp, requested_model=run_id)
            if retryable_error is not None:
                raise retryable_error
        resp.raise_for_status()
        payload = resp.json()
        _balance_thinking_payload(payload, thinking=thinking)
        if expected_checkpoint and isinstance(payload, dict):
            payload["_freesolo_headers"] = {
                "adapter_revision": resp.headers.get("X-Freesolo-Adapter-Revision"),
                "checkpoint": resp.headers.get("X-Freesolo-Checkpoint"),
                "hf_revision": resp.headers.get("X-Freesolo-HF-Revision"),
            }
        return payload


def _delimiter_may_complete(text: str, reasoning: str) -> bool:
    """Whether ``text`` holds too little to rule out a sampled delimiter still arriving.

    The delimiter has two shapes on the wire (the bare tag, and the tag behind a repeat of
    ``reasoning_content``) and a backend may split either anywhere, so the question is not "is this
    the tag" but "could this still become one".
    """
    stripped = text.strip()
    if _TAG_CLOSE.startswith(stripped) or _TAG_OPEN.startswith(stripped):
        # a strict prefix of either tag, including the empty buffer.
        return True
    if stripped.startswith(_TAG_OPEN):
        # past the repeat's opener, so what remains is judged as the bare repeat is, below.
        stripped = stripped[len(_TAG_OPEN) :].strip()
    body = reasoning.strip()
    # an empty reasoning body is repeated as `<think></think>`, leaving nothing to match past the
    # opener, so what remains is tested against the close tag directly.
    if not body:
        return _TAG_CLOSE.startswith(stripped)
    if not body.startswith(stripped[: len(body)]):
        return False
    # the repeat is still arriving, or it landed whole and the tag behind it has not.
    return _TAG_CLOSE.startswith(stripped[len(body) :].strip())


def _strip_retained_close(text: str, reasoning: str, start: int = 0) -> tuple[str | None, int]:
    """Drop a retained sampled ``</think>`` from the head of the post-reasoning content.

    Returns the remaining answer -- or ``None`` while the tag may still be arriving -- paired with
    the offset a later scan of the same growing buffer may resume from. A compatibility build can
    emit reasoning on its own field AND keep the sampled close at the head of the first content
    delta; synthesising another there yields ``<think>reasoned</think></think>answer``.

    The tag need not arrive whole or bare -- the model samples it on its own line as often as not,
    and a backend may split it across deltas -- so this tolerates the same shapes the non-streaming
    path does. A delimiter with nothing behind it keeps buffering: the answer may still be coming,
    or the answer IS the tag, and only the caller's end-of-stream flush can tell them apart.

    ``start`` resumes where a previous call stopped, as the hold path resumes its own scan: the
    buffer grows a delta at a time and re-reading it whole per delta is quadratic in the repeat's
    length. The resume point is the tag's own index once one is found, since a tag with nothing
    behind it keeps buffering and the next call must land on it again.
    """
    close = _find_delimiter(text, start)
    if close >= 0:
        end = _retained_delimiter_end(text, close, reasoning)
        if end is not None:
            return text[end:], close
        if _is_sampled_delimiter(text[:close], reasoning):
            return None, close
        # answer text that merely mentions the tag, so it stays put.
        return text, close
    # nothing before the last few characters can still become a tag.
    clean = max(0, len(text) - (len(_TAG_CLOSE) - 1))
    if _delimiter_may_complete(text, reasoning):
        return None, clean
    return text, clean


def _is_only_retained_delimiter(text: str, reasoning: str) -> bool:
    """Whether a settled buffer holds the block's own retained close and nothing else.

    Asked once no further content can join the buffer, which is the point at which the ambiguity
    `_strip_retained_close` buffers on becomes decidable.
    """
    close = text.find(_TAG_CLOSE)
    if close < 0 or not _is_sampled_delimiter(text[:close], reasoning):
        return False
    return not text[close + len(_TAG_CLOSE) :].strip()


def _is_terminal_reasoning_repeat(text: str, reasoning: str) -> bool:
    """Whether ``text`` is the reasoning repeated inline with nothing behind its close.

    Narrower than `_is_only_retained_delimiter` in requiring text before the close, with or
    without an opener: a delimiter with nothing at all ahead of it is the answer that IS the tag,
    and a checkpoint answering its grammar that way must still reach the smoke.
    """
    close = text.find(_TAG_CLOSE)
    if close < 0 or not text[:close].strip():
        return False
    return _is_only_retained_delimiter(text, reasoning)


def _openai_stream_content(lines: Iterator[str], *, thinking: bool) -> Iterator[str]:
    # reasoning arrives on its own delta field (see _balanced_thinking_content). re-open the block
    # around it and close it at the answer boundary, so the streamed text matches the balanced
    # string the non-streaming path returns.
    reasoning_open = False
    # whether a block was ever emitted, which is not the same as whether one is open now: a backend
    # serializing `reasoning_content: ""` on every delta must not open a second block.
    reasoning_done = False
    # buffered content after the block closed, while a retained delimiter may still be arriving.
    closing: str | None = None
    # how far into `closing` the delimiter search has already looked.
    closing_scanned = 0
    # what arrived on the reasoning field, kept because a compatibility build repeats it inline
    # ahead of the retained close, and recognising that repeat is how the block's own delimiter is
    # told from answer text that merely mentions the tag.
    reasoning_text = ""
    # a legacy backend predating the split streams the whole block inline on `content`, with no
    # reasoning field to key off. in thinking mode the opener lives in the prompt, so such a stream
    # begins mid-block and the sampled `</think>` arrives with nothing to close. hold content until
    # that delimiter proves a reasoning phase, then re-open around it.
    #
    # holding is confined to the case that needs it: outside thinking mode there is no block, and a
    # reasoning delta proves the backend splits, so `held` is dropped the moment either is known.
    #
    # the deltas are kept as they arrived AND concatenated as they arrive: releasing replays the
    # original deltas so a consumer sees the backend's chunk boundaries, while searching needs one
    # string, and rejoining per delta would copy the whole completion every token.
    held: list[str] | None = [] if thinking else None
    # `held` joined, kept in step with it. never bound to a second name while being appended to,
    # since that blocks the in-place concatenation and restores the copy this avoids.
    held_text = ""
    # how far into `held_text` the delimiter search has already looked.
    held_scanned = 0
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
            delta = (choice.get("delta") or {}) if isinstance(choice, dict) else {}
            raw_reasoning = delta.get("reasoning_content")
            # `thinking` gates this as it gates `_balanced_thinking_content`, and for the same
            # reason: this path also backs the public chat route. tested by type, not falsiness --
            # a model that closed its reasoning immediately streams `reasoning_content: ""`, which
            # still needs a pair.
            if thinking and isinstance(raw_reasoning, str):
                if held:
                    # the backend splits after all, so whatever arrived first was answer text and
                    # not an unopened block. release it untouched, delta by delta as it arrived.
                    yield from held
                held = None
                held_text = ""
                # an empty field after the block closed cannot reopen it: it carries no text to
                # label. a non-empty one still opens a block, or its reasoning streams as answer.
                if not reasoning_open and not (reasoning_done and not raw_reasoning):
                    if closing is not None:
                        # a new block opens here, so no further content can join the buffer and it
                        # is decidable now, exactly as at end of stream.
                        settled = (
                            "" if _is_only_retained_delimiter(closing, reasoning_text) else closing
                        )
                        closing = None
                        if settled:
                            yield settled
                    reasoning_open = True
                    # the text belongs to the block that is opening, not to every block so far.
                    # accumulating across blocks made the duplicate checks below compare a later
                    # block's retained close against both blocks' text, so they stopped recognising
                    # it and streamed the delimiter a second time.
                    reasoning_text = ""
                    yield _TAG_OPEN
                if raw_reasoning:
                    reasoning_text += raw_reasoning
                    yield raw_reasoning
            content = delta.get("content") or ""
            if content:
                content = str(content)
                if reasoning_open:
                    reasoning_open = False
                    reasoning_done = True
                    held = None
                    yield _TAG_CLOSE
                    closing = ""
                    closing_scanned = 0
                if closing is not None:
                    closing += content
                    answer, closing_scanned = _strip_retained_close(
                        closing, reasoning_text, closing_scanned
                    )
                    if answer is None:
                        # the tag may still be completing across deltas. keep buffering.
                        continue
                    closing = None
                    if answer:
                        yield answer
                    continue
                if held is not None:
                    held.append(content)
                    held_text += content
                    # resume where the last scan stopped: rescanning the whole buffer per delta is
                    # quadratic in the completion length, and token-sized deltas make that the
                    # common case. only the last few characters can still be a partial tag.
                    close = _find_delimiter(held_text, held_scanned)
                    if close < 0:
                        held_scanned = max(0, len(held_text) - (len(_TAG_CLOSE) - 1))
                        continue
                    joined = held_text
                    held = None
                    held_text = ""
                    inline = _inline_reasoning_block(joined)
                    if inline is not None and inline[1] == close:
                        # already balanced: the legacy stream carried its own opener, so re-opening
                        # would nest one block inside another. the pair must be the one that
                        # MATCHED this close, or an answer-side pair further down would disable the
                        # re-open for the very shape it exists for.
                        yield joined
                        continue
                    yield (
                        f"{_TAG_OPEN}{joined[:close]}{_TAG_CLOSE}"
                        f"{joined[close + len(_TAG_CLOSE) :]}"
                    )
                    continue
                yield content
    if reasoning_open:
        # generation stopped inside the block (a length cap, usually). still close it: an
        # unbalanced opener is the same defect as the unbalanced closer, mirrored.
        yield _TAG_CLOSE
    # the buffer holds the block's own retained close and nothing else, in either the bare or the
    # opener-carrying form. only decidable at end of stream, since nothing more can arrive. any
    # other buffer was answer text after all, covering the answer that IS the delimiter.
    if closing and not _is_terminal_reasoning_repeat(closing, reasoning_text):
        yield closing
    if held:
        # no delimiter ever arrived, so nothing marked a reasoning phase: a plain answer. release
        # it as sent rather than wrapping it, which would label a valid answer as reasoning.
        yield from held


def chat_stream(
    run_id: str,
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 512,
    thinking: bool = False,
    stop: list[str] | None = None,
) -> Iterator[str]:
    """Yield text deltas from the freesolo OpenAI-compatible streaming endpoint.

    ``stop`` carries the run's own stop sequences, as in ``chat``.
    """
    base = serving_openai_base_url()
    body = {
        "model": run_id,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
        "stream": True,
    }
    if stop:
        body["stop"] = [str(value) for value in stop]
    with _stream_http_client().stream(
        "POST",
        f"{base}/chat/completions",
        json=body,
        headers=_internal_key_header(),
        timeout=30 * 60.0,
    ) as resp:
        resp.raise_for_status()
        if "application/json" in resp.headers.get("content-type", ""):
            # client.stream() leaves body unread; must call resp.read() before .json().
            resp.read()
            payload = resp.json()
            content = _balanced_thinking_content(
                (payload.get("choices") or [{}])[0].get("message") or {}, thinking=thinking
            )
            if content:
                yield str(content)
            return
        yield from _openai_stream_content(resp.iter_lines(), thinking=thinking)
