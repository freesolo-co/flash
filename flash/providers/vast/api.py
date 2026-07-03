"""Thin Vast.ai REST client (no SDK state): offer search + instance lifecycle.

Only verified datacenter offers are searched (run secrets ship to the box); callers
re-check hosting_type + verification + the reliability floor client-side.
"""

from __future__ import annotations

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
    """A ``create_instance`` failure that MIGHT have left a billed contract behind — the non-idempotent
    PUT /asks may have been accepted while the response was lost or carried no usable id. Raised so
    ``create_error_is_ambiguous`` classifies it by TYPE (not a message substring), and the caller
    reconciles by label instead of renting a duplicate offer."""


# Env-only key (like RUNPOD_API_KEY): never written to config files or shipped to workers.
_CLIENT = RestClient(
    env_var="VAST_API_KEY",
    error_cls=VastApiError,
    base_url=VAST_BASE,
    missing_key_message=("VAST_API_KEY not configured on the control-plane host"),
)


def _api_key() -> str:
    return _CLIENT.api_key()


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
    # NON-IDEMPOTENT: PUT /asks/{id} rents a new instance on every success, so never retried
    # (blind retry on a lost response = double-provision + double-bill).
    try:
        out = request_with_retries(f"/v0/asks/{int(offer_id)}/", method="PUT", body=body, retries=0)
    except VastApiError as e:
        cause = getattr(e, "__cause__", None)
        if isinstance(cause, (json.JSONDecodeError, UnicodeDecodeError, http.client.HTTPException)):
            raise VastAmbiguousCreate(
                f"create_instance({offer_id}) response unreadable (possible billed contract): {cause}"
            ) from cause
        raise
    except (json.JSONDecodeError, UnicodeDecodeError, http.client.HTTPException) as e:
        # Unreadable 200 body on this non-idempotent create may mean Vast billed a contract while
        # the response leg failed. These decode errors aren't OSErrors so _http doesn't wrap them;
        # surface as VastAmbiguousCreate so the caller reconciles rather than leaking the contract.
        raise VastAmbiguousCreate(
            f"create_instance({offer_id}) response unreadable (possible billed contract): {e}"
        ) from e
    if not isinstance(out, dict) or not out.get("success"):
        raise VastApiError(f"create_instance({offer_id}) rejected: {out}")
    instance_id = out.get("new_contract")
    if not instance_id:
        raise VastAmbiguousCreate(f"create_instance({offer_id}): no instance id in response: {out}")
    try:
        return int(instance_id)
    except (TypeError, ValueError) as e:
        # Truthy but non-numeric new_contract: Vast accepted the create (a contract may be billing)
        # but gave an unusable id. Surface as VastAmbiguousCreate so the caller reconciles by label,
        # rather than letting int()'s ValueError escape and leak the contract.
        raise VastAmbiguousCreate(
            f"create_instance({offer_id}): no instance id usable in response "
            f"(unparseable new_contract {instance_id!r}, possible billed contract): {out}"
        ) from e


def create_error_is_ambiguous(err: Exception) -> bool:
    """True when a ``create_instance`` failure MIGHT have left a billed contract behind, so the
    caller must reconcile by label before renting another offer.

    DEFINITIVE (nothing created): a 4xx rejection (chained ``HTTPError``, ``code < 500``, not 429)
    or a ``success: false`` body (no chained cause). AMBIGUOUS (a contract may exist): a 5xx, a 429,
    any socket-level transient (timeout / reset / DNS), or a ``success`` body with no usable id.

    ``OSError`` is the right boundary — ``URLError``, ``TimeoutError`` and ``ConnectionError`` are
    all ``OSError`` subclasses (a bare read-phase ``TimeoutError`` is not a ``URLError``, so keying
    off ``URLError`` alone would leak the instance). ``HTTPError`` is checked first so 4xx stays
    definitive.
    """
    # Our create path raises VastAmbiguousCreate for every case a contract may have billed (no usable
    # id in a success body; unreadable response) — classify by TYPE, not a message substring.
    if isinstance(err, VastAmbiguousCreate):
        return True
    cause = getattr(err, "__cause__", None)
    if isinstance(cause, urllib.error.HTTPError):  # subclass of OSError -> check first (4xx stays False)
        return cause.code >= 500 or cause.code == 429
    if isinstance(cause, OSError):  # URLError / TimeoutError / ConnectionError + any other socket error
        return True
    # Defensive: a bare / plain-wrapped decode error (truncated / non-JSON / invalid UTF-8) is also an
    # unreadable body -> ambiguous. These aren't OSErrors, so they miss the branch above.
    _unreadable = (json.JSONDecodeError, UnicodeDecodeError, http.client.HTTPException)
    return isinstance(err, _unreadable) or isinstance(cause, _unreadable)


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


def destroy_instance(instance_id: int) -> bool:
    """Destroy (and stop billing for) an instance. Best-effort: never raises.

    Vast's 200 DELETE carries a ``success`` bool — ``success: false`` means the box is still
    billable, so we must not report it destroyed. A body with no ``success`` key is treated as
    success.
    """
    try:
        out = request_with_retries(f"/v0/instances/{int(instance_id)}/", method="DELETE", retries=2)
        if isinstance(out, dict) and "success" in out:
            return bool(out.get("success"))
        return True
    except Exception as exc:
        # A 404 is a confirmed non-billing state (already destroyed / preempted), so report it
        # destroyed (True) rather than failing the run over an instance that provably isn't billing.
        # Every other failure (success:false, 5xx, socket breakdown) may still bill -> False.
        return is_not_found(exc)
