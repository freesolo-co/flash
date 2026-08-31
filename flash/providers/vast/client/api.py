"""Thin Vast.ai REST client (no SDK state): offer search + instance lifecycle.

Only verified datacenter offers are searched (run secrets ship to the box); callers
re-check hosting_type + verification + the reliability floor client-side.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import time
import urllib.error
import urllib.parse
from typing import Any

from flash._internal.logging import get_logger
from flash.providers._lifecycle.net.deadline import (
    remaining_seconds,
    require_create_allowance,
    require_deadline_at,
)
from flash.providers._lifecycle.net.http import RestClient, is_not_found
from flash.providers.vast.client.result import VastResultError, fetch_result

logger = get_logger(__name__)

VAST_BASE = "https://console.vast.ai/api"


class VastApiError(RuntimeError):
    pass


class VastAmbiguousCreate(VastApiError):
    """a create failure that may have left a billed contract behind."""


class VastCreateRejected(VastApiError):
    """an explicit ``success: false`` create rejection that allocated nothing."""


class VastCreateNotSent(VastApiError):
    """a local failure (e.g. missing api key) raised before the create request was sent."""


# env-only key (like runpod_api_key): never written to config files or shipped to workers.
_CLIENT = RestClient(env_var="VAST_API_KEY", error_cls=VastApiError, base_url=VAST_BASE)


def request_with_retries(
    path: str,
    method: str = "GET",
    body: dict | None = None,
    retries: int = 4,
    base_delay: float = 2.0,
    deadline_at: float | None = None,
) -> Any:
    """REST call hardened against transient network/5xx blips (jittered backoff)."""
    return _CLIENT.request_with_retries(
        path,
        method=method,
        body=body,
        retries=retries,
        base_delay=base_delay,
        deadline_at=deadline_at,
    )


# ---------------------------------------------------------------------------
# Offer search
# ---------------------------------------------------------------------------
def search_offers(
    min_vram_mb: int,
    *,
    max_vram_mb: int = 0,
    min_disk_gb: float = 0,
    min_reliability: float = 0.95,
    min_duration_seconds: float = 0,
    limit: int = 64,
    gpu_names: tuple[str, ...] = (),
    num_gpus: int = 1,
    deadline_at: float | None = None,
) -> list[dict]:
    """Rentable offers from verified datacenter hosts, cheapest first.

    `num_gpus` must match the host because Vast bakes card count into the offer. A positive
    `min_duration_seconds` excludes offers that expire before the run.
    """
    # Server-side datacenter-only filter so the price-sorted page isn't filled with community
    # offers usable_offers would reject (run secrets ship to the box); callers still re-check.
    q: dict[str, Any] = {
        "verified": {"eq": True},
        "datacenter": {"eq": True},
        "rentable": {"eq": True},
        "num_gpus": {"eq": int(num_gpus)},
        "gpu_ram": {"gte": int(min_vram_mb)},
        "reliability2": {"gte": float(min_reliability)},
        "type": "ask",
        "order": [["dph_total", "asc"]],
        "limit": int(limit),
    }
    if max_vram_mb > 0:
        q["gpu_ram"]["lte"] = int(max_vram_mb)
    if gpu_names:
        q["gpu_name"] = {"in": list(dict.fromkeys(gpu_names))}
    if min_disk_gb:
        q["disk_space"] = {"gte": float(min_disk_gb)}
    if min_duration_seconds > 0:
        # Keep only offers Vast says are available for at least the run's deadline.
        q["duration"] = {"gte": float(min_duration_seconds)}
    out = request_with_retries(
        "/v0/search/asks/",
        method="PUT",
        body={"q": q},
        deadline_at=deadline_at,
    )
    offers = out.get("offers") if isinstance(out, dict) else None
    return offers if isinstance(offers, list) else []


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------
def _usable_contract_id(value: Any) -> int | None:
    # total and non-throwing: str.isdigit() accepts unicode digits int() rejects (e.g. "²"),
    # so the int() itself stays guarded — an unusable id must flow to ambiguous, not raise.
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not (text.isascii() and text.isdigit()):
            return None
        try:
            parsed = int(text)
        except ValueError:
            return None
    else:
        return None
    return parsed if parsed > 0 else None


def _rejection_from_body(offer_id: int, body: dict) -> VastApiError | None:
    """classify an explicit ``success: false`` create body; shared by the 200 and non-2xx paths.

    a false body carrying ANY non-empty ``new_contract`` evidence is contradictory — the
    contract may exist and bill — so it must be ambiguous, never a definitive rejection."""
    if body.get("success") is not False:
        return None
    contract = body.get("new_contract")
    if contract:
        err = VastAmbiguousCreate(
            f"create_instance({offer_id}) returned success=false with contract evidence "
            "(contradictory response; possible billed contract)"
        )
        err.contract_id = _usable_contract_id(contract)
        return err
    return VastCreateRejected(
        f"create_instance({offer_id}) rejected (success=false with no contract evidence)"
    )


def _http_error_response(error: VastApiError) -> dict | None:
    cause = getattr(error, "__cause__", None)
    if not isinstance(cause, urllib.error.HTTPError):
        return None
    if cause.code >= 500 or cause.code == 429:
        return None

    # _http attaches the full consumed body as structured metadata (the message truncates it);
    # fall back to reading the response only for errors raised outside that wrapper.
    raw: bytes | str | None = getattr(error, "response_body", None)
    if not raw:
        with contextlib.suppress(Exception):
            raw = cause.read()
    if not raw:
        return None

    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return decoded if isinstance(decoded, dict) else None


def create_instance(
    offer_id: int,
    *,
    image: str,
    disk_gb: float,
    env: dict[str, str],
    onstart: str,
    label: str,
    deadline_at: float | None = None,
) -> int:
    """Rent an offer -> instance id. Raises VastApiError on rejection (offer taken).

    ``args`` runtype: the script is the container command (``bash -c``), so no SSH key is
    needed and the container lifecycle == the job lifecycle.
    """
    deadline = require_deadline_at(deadline_at) if deadline_at is not None else None
    if deadline is not None:
        require_create_allowance(deadline)
    body = {
        "client_id": "me",
        "image": image,
        "disk": float(disk_gb),
        "env": dict(env),
        "label": label,
        "runtype": "args",
        # Worker image is public: no docker-login / pull token shipped to the untrusted host.
        "args": ["bash", "-c", onstart],
    }
    # non-idempotent: put /asks/{id} rents a new instance on every success, so never retried
    # (blind retry on a lost response = double-provision + double-bill).
    target = f"/v0/asks/{int(offer_id)}/"
    try:
        out = request_with_retries(
            target,
            method="PUT",
            body=body,
            retries=0,
            **({} if deadline is None else {"deadline_at": deadline}),
        )
    except VastApiError as e:
        if getattr(e, "__cause__", None) is None and str(e) == _CLIENT.missing_key_message:
            # raised locally before any http request: the create was definitively never sent,
            # so surface it as an actionable config error rather than a possibly-billed create.
            raise VastCreateNotSent(str(e)) from None
        response = _http_error_response(e)
        if response is not None:
            classified = _rejection_from_body(offer_id, response)
            if classified is not None:
                raise classified from None
        cause = getattr(e, "__cause__", None)
        if isinstance(cause, urllib.error.HTTPError):
            raise VastAmbiguousCreate(
                f"create_instance({offer_id}) HTTP {cause.code} on {target} was not a definitive "
                "rejection (possible billed contract)"
            ) from None
        if isinstance(cause, (json.JSONDecodeError, UnicodeDecodeError, http.client.HTTPException)):
            raise VastAmbiguousCreate(
                f"create_instance({offer_id}) response unreadable "
                f"({type(cause).__name__}; possible billed contract)"
            ) from None
        raise VastAmbiguousCreate(
            f"create_instance({offer_id}) failed without a definitive rejection classification "
            f"on {target} ({type(e).__name__}; possible billed contract)"
        ) from None
    except (json.JSONDecodeError, UnicodeDecodeError, http.client.HTTPException) as exc:
        # an unreadable response may mean vast billed a contract before the response failed.
        # surface an ambiguous create so the caller reconciles instead of leaking the contract.
        raise VastAmbiguousCreate(
            f"create_instance({offer_id}) response unreadable "
            f"({type(exc).__name__}; possible billed contract)"
        ) from None
    if not isinstance(out, dict):
        raise VastAmbiguousCreate(
            f"create_instance({offer_id}) returned a non-object response "
            f"({type(out).__name__}; possible billed contract)"
        )
    classified = _rejection_from_body(offer_id, out)
    if classified is not None:
        raise classified
    parsed_id = _usable_contract_id(out.get("new_contract"))
    if out.get("success") is not True:
        raise VastAmbiguousCreate(
            f"create_instance({offer_id}) returned an ambiguous success classification "
            "(possible billed contract)"
        )
    if parsed_id is None:
        raise VastAmbiguousCreate(
            f"create_instance({offer_id}) returned no usable instance id (possible billed contract)"
        )
    return parsed_id


def create_error_is_ambiguous(err: Exception) -> bool:
    """return false only when the create provably allocated nothing.

    that is an explicit ``success: false`` rejection with no contract evidence, or a local
    failure raised before the request was sent. everything else may have billed a contract."""
    return not isinstance(err, (VastCreateRejected, VastCreateNotSent))


def get_instance(
    instance_id: int,
    *,
    deadline_at: float | None = None,
) -> dict | None:
    """Instance detail dict, or None once it no longer exists (destroyed).

    The v0 detail route answers 200 with ``{"instances": null}`` for unknown ids — that is
    the "gone" signal, not a 404.
    """
    try:
        out = request_with_retries(
            f"/v0/instances/{int(instance_id)}/",
            deadline_at=deadline_at,
        )
    except VastApiError as e:
        # Status-code authoritative (via is_not_found), not a "404" substring: a non-404 body
        # embedding an id like "4040" must not be misread as a disappearance.
        if is_not_found(e):
            return None
        raise
    if isinstance(out, dict):
        if "instances" in out:
            inst = out["instances"]
            return inst if isinstance(inst, dict) else None
        # A dict WITHOUT "instances" that carries success:false is an error envelope, not instance
        # detail. Returning it would read as a live-but-"unknown" instance (its .get("actual_status")
        # is None), silently RESETTING the missing streak and masking a real disappearance. Raise so
        # the poller counts it as a bounded, retryable poll_error instead of a false healthy read.
        if out.get("success") is False:
            raise VastApiError(
                f"vast get_instance({int(instance_id)}) malformed response: "
                f"classification=error_envelope returned_type={type(out).__name__}"
            )
        return out
    return None


def list_instances(
    strict: bool = False,
    *,
    deadline_at: float | None = None,
) -> list[dict]:
    # list via paginated v1 (`next_token` -> `after_token`, limit 25); detail/destroy remain v0.
    # walk every page. with `strict=True`, any fetch, shape, or page-cap failure raises so an
    # incomplete listing cannot prove an instance is gone.
    instances: list[dict] = []
    after_token: str | None = None
    for page_no in range(
        200
    ):  # runaway guard: 200 pages x 25 = 5000 instances, beyond any real account
        path = "/v1/instances/"
        if after_token:
            path += f"?after_token={urllib.parse.quote(str(after_token))}"
        try:
            out = request_with_retries(path, deadline_at=deadline_at)
        except Exception:
            # A later page failed after earlier ones succeeded: return the partial list rather than
            # discard it — lenient consumers only act on instances they see, and the next sweep
            # retries the rest. A first-page failure or a strict caller re-raises (nothing useful /
            # cannot accept an incomplete list).
            if instances and not strict:
                logger.warning(
                    "vast instance listing truncated at page %d (using %d instance(s) collected so far)",
                    page_no,
                    len(instances),
                )
                return instances
            raise
        if not isinstance(out, dict):
            if strict:
                raise VastApiError(
                    "vast instance listing returned a non-dict page; listing incomplete"
                )
            break
        page = out.get("instances")
        if isinstance(page, list):
            instances.extend(page)
        elif strict:
            # A 200 lacking an ``instances`` list (e.g. an error envelope) would otherwise look like
            # a complete empty page; a strict caller must treat it as incomplete, not "none remain".
            raise VastApiError(
                "vast instance listing page has no 'instances' list; listing incomplete"
            )
        after_token = out.get("next_token")
        if not after_token:
            break
    if strict and after_token:
        # Fell off the page-cap runaway guard with more pages pending -> the listing is incomplete.
        raise VastApiError("vast instance listing exceeded the page cap; listing incomplete")
    return instances


def instance_logs(instance_id: int, *, deadline_at: float | None = None) -> str | None:
    """Container log tail via the logs API (request -> poll the result URL).

    The only place early-bootstrap failures are visible. Best-effort: returns None when logs
    are unavailable (e.g. the instance is already destroyed); never raises.
    """
    try:
        absolute_deadline = require_deadline_at(deadline_at) if deadline_at is not None else None
        out = request_with_retries(
            f"/v0/instances/request_logs/{int(instance_id)}/",
            method="PUT",
            body={"tail": "400"},
            retries=1,
            deadline_at=absolute_deadline,
        )
        url = out.get("result_url") if isinstance(out, dict) else None
        if not url:
            return None
        poll_deadline = time.time() + 20.0
        if absolute_deadline is not None:
            poll_deadline = min(poll_deadline, absolute_deadline)
        while True:
            remaining = remaining_seconds(poll_deadline)
            if remaining <= 0:
                break
            try:
                result = fetch_result(url, timeout=min(15.0, remaining))
            except VastResultError:
                return None
            if result is not None:
                body = result.decode(errors="replace")
                if body.strip():
                    return body
            sleep_for = min(2.0, remaining_seconds(poll_deadline))
            if sleep_for <= 0:
                break
            time.sleep(sleep_for)
    except Exception:
        return None
    return None


def _genuine_http_not_found(exc: Exception) -> bool:
    """return true only for an actual http 404 from the exact-instance request."""
    cause = getattr(exc, "__cause__", None)
    return bool(
        (isinstance(exc, urllib.error.HTTPError) and exc.code == 404)
        or (isinstance(cause, urllib.error.HTTPError) and cause.code == 404)
    )


def _exact_instance_absent(instance_id: int, *, deadline_at: float | None = None) -> bool:
    """confirm absence only from the exact-instance route's documented null or 404 signal."""
    try:
        out = request_with_retries(
            f"/v0/instances/{int(instance_id)}/",
            retries=1,
            deadline_at=deadline_at,
        )
    except Exception as exc:
        return _genuine_http_not_found(exc)
    if not isinstance(out, dict) or "error" in out or "detail" in out:
        return False
    if "success" in out and out.get("success") is not True:
        return False
    return "instances" in out and out["instances"] is None


def destroy_instance(
    instance_id: int,
    *,
    deadline_at: float | None = None,
) -> bool:
    """destroy an instance and return true only when provider-confirmed absent."""
    try:
        out = request_with_retries(
            f"/v0/instances/{int(instance_id)}/",
            method="DELETE",
            retries=2,
            deadline_at=deadline_at,
        )
    except Exception as exc:
        if _genuine_http_not_found(exc):
            return True
        cause = getattr(exc, "__cause__", None)
        if isinstance(cause, urllib.error.HTTPError) and cause.code < 500 and cause.code != 429:
            return False
        return _exact_instance_absent(instance_id, deadline_at=deadline_at)
    if isinstance(out, dict) and out.get("success") is True:
        return True
    if isinstance(out, dict) and (out.get("success") is False or "error" in out or "detail" in out):
        return False
    return _exact_instance_absent(instance_id, deadline_at=deadline_at)
