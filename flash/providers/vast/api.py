"""Thin Vast.ai REST client (no SDK state): offer search + instance lifecycle.

Mirrors ``providers/runpod/api.py``: stdlib urllib only, hardened retries, and nothing
persisted locally — a fresh process can list/destroy any instance using only the
persisted ids + VAST_API_KEY. Only ``verified`` DATACENTER offers are searched (the
server-side ``datacenter`` hosting-type filter is applied, since run secrets ship to the
box); callers re-check hosting_type + verification + the reliability floor client-side.
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


# Shared urllib client (path form: callers pass paths joined onto VAST_BASE).
# Env-only by design, like RUNPOD_API_KEY: the operator sets VAST_API_KEY on the
# control-plane host; it is never written to config files or shipped to workers.
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
    extra_q: dict | None = None,
) -> list[dict]:
    """Rentable single-GPU offers from verified DATACENTER hosts, cheapest first.

    ``datacenter`` here is Vast's hosting-type filter (professional datacenters vs
    consumer/hobbyist machines); results additionally carry ``hosting_type`` which
    callers re-check (``usable_offers``) — never trust one filter layer alone.

    ``min_duration_seconds`` applies Vast's documented ``duration`` filter ("the offer must be
    available for at least this long from now", in seconds): a run whose wall cap exceeds an offer's
    remaining availability would otherwise rent a short-lived offer that expires/preempts mid-run,
    burning retries (fatal for ``max_retries=0``) while longer offers/providers went unused. 0 = off.
    """
    # Apply Vast's server-side datacenter-only filter (hosting_type==1). usable_offers now rejects
    # community/marketplace hosts unconditionally (run secrets ship to the box), so a mixed search
    # would risk filling the price-sorted limit=64 page with community offers and reporting "no
    # usable offers" even when verified-datacenter capacity exists just past the page. Filtering
    # server-side keeps the page full of datacenter offers; usable_offers still re-checks
    # hosting_type + verification + the reliability floor (belt and suspenders).
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
    if min_duration_seconds and min_duration_seconds > 0:
        # Same operator-dict form as every other numeric filter above (gpu_ram/disk_space/reliability2):
        # keep only offers Vast says are available for at least the run's deadline.
        q["duration"] = {"gte": float(min_duration_seconds)}
    if extra_q:
        q.update(extra_q)
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
    runtype: str = "args",
) -> int:
    """Rent an offer -> instance id. Raises VastApiError on rejection (offer taken).

    Default ``args`` runtype (verified live): the script IS the container command
    (``bash -c``), so the job needs no SSH key on the account, the container's
    lifecycle is the job's lifecycle, and the Vast-injected CONTAINER_API_KEY /
    CONTAINER_ID env vars are available for the self-destroy backstop. ``ssh``
    runtype requires an SSH key attached to the Vast account.
    """
    body = {
        "client_id": "me",
        "image": image,
        "disk": float(disk_gb),
        "env": dict(env),
        "label": label,
        "runtype": runtype,
    }
    # The worker image is PUBLIC, so Vast pulls it with no docker-login (no image_login / pull
    # token is ever shipped to the untrusted host).
    if runtype == "args":
        body["args"] = ["bash", "-c", onstart]
    else:
        body["onstart"] = onstart
    # NON-IDEMPOTENT: ``PUT /asks/{id}`` rents a NEW instance every time it succeeds.
    # A blind retry on a timeout where Vast actually accepted the first request would
    # double-provision (two billed instances, one invisible to our handle). So this
    # call is NOT retried — a transient failure surfaces to deploy_and_submit, which
    # walks to the next offer, and to the orchestrator, which consumes a run retry; a
    # duplicate paid instance is the worse failure. (Idempotent calls — search,
    # detail, destroy — keep their retries.)
    try:
        out = request_with_retries(f"/v0/asks/{int(offer_id)}/", method="PUT", body=body, retries=0)
    except (json.JSONDecodeError, UnicodeDecodeError, http.client.HTTPException) as e:
        # A 200 whose body is truncated / non-JSON / invalid-UTF8 on this NON-IDEMPOTENT create: Vast
        # may have already accepted the PUT and billed a contract while the RESPONSE leg failed, so we
        # have a phantom instance with no returned id. JSONDecodeError (ValueError), UnicodeDecodeError
        # (a SIBLING of JSONDecodeError under ValueError — raised by ``json.loads`` on bytes with
        # invalid UTF-8, so the JSONDecodeError clause alone would miss it), and IncompleteRead
        # (HTTPException) are NOT OSErrors, so the _http retry wrapper neither catches nor wraps them
        # — they'd otherwise escape as a raw decode error past deploy_and_submit's ``except
        # VastApiError`` and skip the ambiguous-create reconcile (leaking the contract). Re-raise as a
        # VastApiError chaining the cause so create_error_is_ambiguous classifies it AMBIGUOUS and the
        # adopt-by-label / destroy-and-abort path runs.
        raise VastApiError(
            f"create_instance({offer_id}) response unreadable (possible billed contract): {e}"
        ) from e
    if not isinstance(out, dict) or not out.get("success"):
        raise VastApiError(f"create_instance({offer_id}) rejected: {out}")
    instance_id = out.get("new_contract")
    if not instance_id:
        raise VastApiError(f"create_instance({offer_id}): no instance id in response: {out}")
    return int(instance_id)


def create_error_is_ambiguous(err: Exception) -> bool:
    """True when a ``create_instance`` failure MIGHT have left a billed contract behind, so the caller
    must reconcile by label before renting another offer (the non-idempotent PUT /asks can succeed on
    the host while the response is lost).

    DEFINITIVE (created nothing -> safe to walk to the next offer): a 4xx client rejection (offer
    taken / bad request, ``code < 500`` and not 429) carries a chained ``HTTPError``; a
    ``success: false`` body is raised with NO chained cause. AMBIGUOUS (a contract may exist): a 5xx,
    a 429 (rate-limit — the request may have been accepted then throttled on the response, so a billed
    instance can exist without a returned id; mirrors Lambda's launch path), ANY socket-level
    transient (timeout / connection reset / DNS), or a ``success`` body that carried no instance id.

    The socket-level case must match every cause ``RestClient`` can chain, not just ``URLError``:
    urllib raises a BARE ``TimeoutError`` (== ``socket.timeout``) / ``ConnectionError`` when a request
    times out or drops on the RESPONSE leg (after the host already accepted the non-idempotent
    ``PUT /asks/{id}`` and billed a contract) — those are NOT ``URLError`` subclasses, so keying off
    ``URLError`` alone let a read-phase timeout look like a clean rejection and leak the instance.
    ``OSError`` is exactly the right boundary: ``URLError``, ``TimeoutError`` and ``ConnectionError``
    are all ``OSError`` subclasses, matching the ``_http`` wrapper's transient except-tuple precisely.
    ``HTTPError`` (also an ``OSError``) is checked FIRST so a 4xx stays definitive."""
    cause = getattr(err, "__cause__", None)
    if isinstance(cause, urllib.error.HTTPError):  # subclass of OSError -> check first (4xx stays False)
        return cause.code >= 500 or cause.code == 429
    if isinstance(cause, OSError):  # URLError / TimeoutError / ConnectionError + any other socket error
        return True
    # A 200 whose body could not be read/parsed on the non-idempotent create (truncated read /
    # non-JSON / invalid UTF-8): the host may have billed a contract while the RESPONSE was lost, so
    # this is ambiguous, NOT a clean rejection. JSONDecodeError (ValueError), UnicodeDecodeError (its
    # SIBLING under ValueError — ``json.loads`` on bytes with invalid UTF-8 raises THIS, not
    # JSONDecodeError) and IncompleteRead/HTTPException are not OSErrors so they miss the branches
    # above; create_instance wraps them as a VastApiError chaining the cause, but match a bare one too
    # (defensive).
    _unreadable = (json.JSONDecodeError, UnicodeDecodeError, http.client.HTTPException)
    if isinstance(err, _unreadable) or isinstance(cause, _unreadable):
        return True
    return "no instance id" in str(err)  # success body without a contract id -> may be billing


def get_instance(instance_id: int) -> dict | None:
    """Instance detail dict, or None once it no longer exists (destroyed).

    The v0 detail route answers 200 with ``{"instances": null}`` for unknown ids
    (verified live) — that is the "gone" signal, not a 404.
    """
    try:
        out = request_with_retries(f"/v0/instances/{int(instance_id)}/")
    except VastApiError as e:
        # Status-CODE-authoritative (the chained HTTPError), NOT a bare "404" substring: a non-404
        # 4xx whose body embeds an id like "4040" must not be misread as a disappearance/preemption.
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
    # The v0 list route is deprecated (410 "use /api/v1/instances/", verified live); detail/destroy
    # remain on v0. The v1 list is KEYSET-PAGINATED (limit default/max 25; pass the prior page's
    # ``next_token`` as ``after_token``; ``next_token`` is null on the last page). A single read would
    # cap at 25 and MISS flash-labeled orphans on later pages — and EVERY label-keyed path
    # (_adopt_instance_by_label / destroy_run_instances / sweep_orphans) reads this list, so an
    # unseen orphan bills forever. Walk every page until next_token is exhausted.
    #
    # ``strict`` (default False) is for callers that need a COMPLETE listing to draw a sound conclusion
    # from an ABSENCE (e.g. run_instances_remaining: "no instance for this run remains"). A truncated
    # page set would let such a caller read "absent" as "gone" and act on it. In strict mode any
    # incompleteness — a page fetch failure (even after earlier pages succeeded), a malformed page, or
    # exhausting the page cap with pages still pending — RAISES instead of returning a partial list, so
    # the caller can treat "couldn't enumerate completely" as "could not confirm" (Cursor).
    instances: list[dict] = []
    after_token: str | None = None
    for page_no in range(200):  # runaway guard: 200 pages x 25 = 5000 instances, beyond any real account
        path = "/v1/instances/"
        if after_token:
            path += f"?after_token={urllib.parse.quote(str(after_token))}"
        try:
            out = request_with_retries(path)
        except Exception:
            # A LATER page failed after earlier pages succeeded: return what we already have rather than
            # discarding it. A partial list is safe for the lenient consumers — destroy_run_instances /
            # sweep_orphans only act on instances they SEE (guarded per-instance by the active/known
            # sets), and _adopt_instance_by_label missing a target leads to destroy-by-label + abort,
            # never a double-provision; the next sweep retries the unfetched pages. If the FIRST page
            # failed we have nothing useful -> re-raise so callers' existing try/except treats a total
            # listing outage exactly as before (skip the sweep). A strict caller NEVER accepts a partial
            # list (an unseen page could hide the very instance it is trying to rule out).
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
            # A 200 whose body lacks an ``instances`` LIST (e.g. an error envelope like
            # ``{"success": false}`` with no ``next_token``) would otherwise fall through as a COMPLETE
            # empty page -> a strict caller would read it as "no instance remains" and act on that false
            # clear. Treat a missing/non-list page as an incomplete listing (Codex).
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

    The only place early-bootstrap failures (pip/env errors before the worker can
    reach HF) are visible. Best-effort: returns None when logs are unavailable
    (e.g. the instance is already destroyed); never raises.
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

    Vast's 200 DELETE carries a ``success`` bool — a ``success: false`` means the box is STILL
    billable, so we must not report it destroyed (``destroy_run_instances``/``sweep_orphans`` would
    count it reaped and stop the immediate cleanup). An older/empty body shape (no ``success`` key) on
    a non-error response is treated as success, preserving prior behavior.
    """
    try:
        out = request_with_retries(f"/v0/instances/{int(instance_id)}/", method="DELETE", retries=2)
        if isinstance(out, dict) and "success" in out:
            return bool(out.get("success"))
        return True
    except Exception as exc:
        # A 404 means the instance no longer exists — already destroyed, or host-preempted/disappeared.
        # That is a CONFIRMED non-billing state, so report it destroyed (True): the retry loop and
        # VastProvider.destroy() treat the box as gone and proceed, instead of raising "unconfirmed
        # teardown" and failing the run over an instance that is provably not billing (Codex). Reserve
        # False for failures where the box MAY still bill (success:false, 5xx, socket breakdown).
        # is_not_found is True ONLY for a genuine 404 -> confirmed-gone; every other failure -> False.
        return is_not_found(exc)
