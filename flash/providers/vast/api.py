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
import urllib.request
from typing import Any

from flash._logging import get_logger
from flash.providers._http import RestClient, is_not_found

logger = get_logger(__name__)

VAST_BASE = "https://console.vast.ai/api"


class VastApiError(RuntimeError):
    pass


class VastAmbiguousCreate(VastApiError):
    """a create failure that may have left a billed contract behind."""


class VastCreateRejected(VastApiError):
    """an explicit ``success: false`` create rejection that allocated nothing."""


# env-only key (like runpod_api_key): never written to config files or shipped to workers.
_CLIENT = RestClient(
    env_var="VAST_API_KEY",
    error_cls=VastApiError,
    base_url=VAST_BASE,
    missing_key_message=("VAST_API_KEY not configured on the control-plane host"),
)


def request_with_retries(
    path: str,
    method: str = "GET",
    body: dict | None = None,
    retries: int = 4,
    base_delay: float = 2.0,
) -> Any:
    """REST call hardened against transient network/5xx blips (jittered backoff)."""
    return _CLIENT.request_with_retries(
        path, method=method, body=body, retries=retries, base_delay=base_delay
    )


# ---------------------------------------------------------------------------
# Offer search
# ---------------------------------------------------------------------------
def search_offers(
    min_vram_mb: int,
    *,
    min_disk_gb: float = 0,
    min_reliability: float = 0.95,
    min_duration_seconds: float = 0,
    limit: int = 64,
) -> list[dict]:
    """Rentable single-GPU offers from verified datacenter hosts, cheapest first.

    ``min_duration_seconds`` applies Vast's ``duration`` filter (offer available for at least
    this long from now); prevents renting a short-lived offer that preempts mid-run. 0 = off.
    """
    # Server-side datacenter-only filter so the price-sorted page isn't filled with community
    # offers usable_offers would reject (run secrets ship to the box); callers still re-check.
    q: dict[str, Any] = {
        "verified": {"eq": True},
        "datacenter": {"eq": True},
        "rentable": {"eq": True},
        "num_gpus": {"eq": 1},
        "gpu_ram": {"gte": int(min_vram_mb)},
        "reliability2": {"gte": float(min_reliability)},
        "type": "ask",
        "order": [["dph_total", "asc"]],
        "limit": int(limit),
    }
    if min_disk_gb:
        q["disk_space"] = {"gte": float(min_disk_gb)}
    if min_duration_seconds > 0:
        # Keep only offers Vast says are available for at least the run's deadline.
        q["duration"] = {"gte": float(min_duration_seconds)}
    out = request_with_retries("/v0/search/asks/", method="PUT", body={"q": q})
    offers = out.get("offers") if isinstance(out, dict) else None
    return offers if isinstance(offers, list) else []


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------
def _usable_contract_id(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
    else:
        return None
    return parsed if parsed > 0 else None


def _http_error_response(error: VastApiError, *, method: str, target: str) -> dict | None:
    cause = getattr(error, "__cause__", None)
    if not isinstance(cause, urllib.error.HTTPError):
        return None
    if cause.code >= 500 or cause.code == 429:
        return None

    raw: bytes | str | None = None
    with contextlib.suppress(Exception):
        raw = cause.read()
    if not raw:
        prefix = f"{method} {target} -> HTTP {cause.code}: {cause.reason}"
        message = str(error)
        if not message.startswith(prefix):
            return None
        suffix = message[len(prefix) :]
        if not suffix.startswith(": "):
            return None
        raw = suffix[2:]

    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
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
) -> int:
    """Rent an offer -> instance id. Raises VastApiError on rejection (offer taken).

    ``args`` runtype: the script is the container command (``bash -c``), so no SSH key is
    needed and the container lifecycle == the job lifecycle.
    """
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
        out = request_with_retries(target, method="PUT", body=body, retries=0)
    except VastApiError as e:
        response = _http_error_response(e, method="PUT", target=target)
        if response is not None and response.get("success") is False:
            parsed_id = _usable_contract_id(response.get("new_contract"))
            if parsed_id is not None:
                raise VastAmbiguousCreate(
                    f"create_instance({offer_id}) returned contradictory rejection with contract "
                    f"{parsed_id} (possible billed contract): {response}"
                ) from getattr(e, "__cause__", e)
            raise VastCreateRejected(
                f"create_instance({offer_id}) rejected: {response}"
            ) from getattr(e, "__cause__", e)
        cause = getattr(e, "__cause__", None)
        if isinstance(cause, (json.JSONDecodeError, UnicodeDecodeError, http.client.HTTPException)):
            raise VastAmbiguousCreate(
                f"create_instance({offer_id}) response unreadable (possible billed contract): {cause}"
            ) from cause
        raise
    except (json.JSONDecodeError, UnicodeDecodeError, http.client.HTTPException) as e:
        # unreadable 200 body on this non-idempotent create may mean vast billed a contract while
        # the response leg failed. these decode errors aren't oserrors so _http doesn't wrap them;
        # surface as vastambiguouscreate so the caller reconciles rather than leaking the contract.
        raise VastAmbiguousCreate(
            f"create_instance({offer_id}) response unreadable (possible billed contract): {e}"
        ) from e
    if not isinstance(out, dict):
        raise VastAmbiguousCreate(
            f"create_instance({offer_id}) returned an ambiguous response "
            f"(possible billed contract): {out}"
        )
    parsed_id = _usable_contract_id(out.get("new_contract"))
    if out.get("success") is False:
        if parsed_id is not None:
            raise VastAmbiguousCreate(
                f"create_instance({offer_id}) returned contradictory rejection with contract "
                f"{parsed_id} (possible billed contract): {out}"
            )
        raise VastCreateRejected(f"create_instance({offer_id}) rejected: {out}")
    if out.get("success") is not True:
        raise VastAmbiguousCreate(
            f"create_instance({offer_id}) returned an ambiguous response "
            f"(possible billed contract): {out}"
        )
    if parsed_id is None:
        raise VastAmbiguousCreate(
            f"create_instance({offer_id}): no instance id in response "
            f"(unparseable new_contract {out.get('new_contract')!r}, possible billed contract): {out}"
        )
    return parsed_id


def create_error_is_ambiguous(err: Exception) -> bool:
    """return false only for an explicit ``success: false`` create response."""
    return not isinstance(err, VastCreateRejected)


def get_instance(instance_id: int) -> dict | None:
    """Instance detail dict, or None once it no longer exists (destroyed).

    The v0 detail route answers 200 with ``{"instances": null}`` for unknown ids — that is
    the "gone" signal, not a 404.
    """
    try:
        out = request_with_retries(f"/v0/instances/{int(instance_id)}/")
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
            raise VastApiError(f"vast instance-detail error envelope for {int(instance_id)}: {out}")
        return out
    return None


def list_instances(strict: bool = False) -> list[dict]:
    # v0 list is deprecated (410 "use /api/v1/instances/"); detail/destroy stay on v0. The v1 list
    # is keyset-paginated (limit max 25; pass the prior page's ``next_token`` as ``after_token``,
    # null on the last page). Must walk every page or a flash-labeled orphan on a later page (which
    # every label-keyed cleanup path reads this list for) bills forever.
    #
    # ``strict`` (default False) is for callers concluding from an ABSENCE (e.g. "no instance for
    # this run remains"): any incompleteness — a page fetch failure, a malformed page, or exhausting
    # the page cap — RAISES instead of returning a partial list, so "couldn't enumerate completely"
    # is never read as "gone".
    instances: list[dict] = []
    after_token: str | None = None
    for page_no in range(200):  # runaway guard: 200 pages x 25 = 5000 instances, beyond any real account
        path = "/v1/instances/"
        if after_token:
            path += f"?after_token={urllib.parse.quote(str(after_token))}"
        try:
            out = request_with_retries(path)
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
                raise VastApiError("vast instance listing returned a non-dict page; listing incomplete")
            break
        page = out.get("instances")
        if isinstance(page, list):
            instances.extend(page)
        elif strict:
            # A 200 lacking an ``instances`` list (e.g. an error envelope) would otherwise look like
            # a complete empty page; a strict caller must treat it as incomplete, not "none remain".
            raise VastApiError("vast instance listing page has no 'instances' list; listing incomplete")
        after_token = out.get("next_token")
        if not after_token:
            break
    if strict and after_token:
        # Fell off the page-cap runaway guard with more pages pending -> the listing is incomplete.
        raise VastApiError("vast instance listing exceeded the page cap; listing incomplete")
    return instances


def instance_logs(instance_id: int) -> str | None:
    """Container log tail via the logs API (request -> poll the result URL).

    The only place early-bootstrap failures are visible. Best-effort: returns None when logs
    are unavailable (e.g. the instance is already destroyed); never raises.
    """
    try:
        out = request_with_retries(
            f"/v0/instances/request_logs/{int(instance_id)}/",
            method="PUT",
            body={"tail": "400"},
            retries=1,
        )
        url = out.get("result_url") if isinstance(out, dict) else None
        if not url:
            return None
        deadline = time.time() + 20.0
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=15) as resp:
                    body = resp.read().decode(errors="replace")
                if body.strip():
                    return body
            except urllib.error.HTTPError as e:
                if e.code != 404:  # 404 = not materialized yet
                    return None
            time.sleep(2.0)
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


def _exact_instance_absent(instance_id: int) -> bool:
    """confirm absence only from the exact-instance route's documented null or 404 signal."""
    try:
        out = request_with_retries(f"/v0/instances/{int(instance_id)}/", retries=1)
    except Exception as exc:
        return _genuine_http_not_found(exc)
    if not isinstance(out, dict) or "error" in out or "detail" in out:
        return False
    if "success" in out and out.get("success") is not True:
        return False
    return "instances" in out and out["instances"] is None


def destroy_instance(instance_id: int) -> bool:
    """destroy an instance and return true only when provider-confirmed absent."""
    try:
        out = request_with_retries(f"/v0/instances/{int(instance_id)}/", method="DELETE", retries=2)
    except Exception as exc:
        if _genuine_http_not_found(exc):
            return True
        cause = getattr(exc, "__cause__", None)
        if isinstance(cause, urllib.error.HTTPError) and cause.code < 500 and cause.code != 429:
            return False
        return _exact_instance_absent(instance_id)
    if isinstance(out, dict) and out.get("success") is True:
        return True
    if isinstance(out, dict) and (
        out.get("success") is False or "error" in out or "detail" in out
    ):
        return False
    return _exact_instance_absent(instance_id)
