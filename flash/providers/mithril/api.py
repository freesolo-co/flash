"""Own a Spot bid for each attempt, including its queued and replacement instances."""

import base64
import contextlib
import hashlib
import ipaddress
import json
import re
import time

from flash.providers.core.base import CapacityUnavailableError, UnreconciledCreateError
from flash.providers.mithril.client import MithrilApiError, call, pages
from flash.providers.mithril.config import settings
from flash.providers.mithril.pricing import SPOT_GPUS

CUSTOM_SESSION_PORT = 8000
# After an ambiguous create failure, how long to wait before listing the attempt's bids again.
CREATE_RECHECK_SECONDS = 10
_OWNER_PREFIX = "# flash-mithril-owner: "
_STOPPED = frozenset(
    {"STATUS_TERMINATED", "STATUS_PREEMPTED", "STATUS_REPLACED", "STATUS_PAUSED", "STATUS_ERROR"}
)


def _ownership(bid):
    config = settings()
    if bid.get("project") != config["project_id"] or not bid.get("region"):
        return None
    script = (bid.get("launch_specification") or {}).get("startup_script") or ""
    lines = script.splitlines()
    if len(lines) < 2 or not lines[1].startswith(_OWNER_PREFIX):
        return None
    try:
        labels = json.loads(base64.b64decode(lines[1][len(_OWNER_PREFIX) :], validate=True))
        if not isinstance(labels, dict) or labels.get("flash-owner") != "flash-training":
            return None
        if not all(
            isinstance(labels.get(key), str) and labels[key]
            for key in (
                "flash-run",
                "flash-attempt",
                "flash-spec",
                "flash-source",
            )
        ):
            return None
        return labels
    except (ValueError, TypeError):
        return None


def list_instances(*, deadline_at=None):
    """Expose owned bids to Flash's existing resource reconciliation interface."""
    result = []
    for bid in pages(
        "/spot/bids", params={"project": settings()["project_id"]}, deadline_at=deadline_at
    ):
        labels = _ownership(bid)
        if not labels:
            continue
        # Bid cancellation precedes VM shutdown. Handle-less recovery must still see
        # these resources until the same instance-state check as terminate() passes.
        if bid["status"] == "Terminated" and all(
            row["status"] in _STOPPED for row in _instances(bid["fid"], deadline_at=deadline_at)
        ):
            continue
        result.append({"metadata": {"id": bid["fid"], "name": bid["name"], "labels": labels}})
    return result


def _bid(bid_id, *, deadline_at=None):
    if not isinstance(bid_id, str) or not re.fullmatch(r"bid_[A-Za-z0-9]+", bid_id):
        raise ValueError("invalid Mithril bid id; drain legacy Spot runs before migration")
    try:
        bid = call("GET", f"/spot/bids/{bid_id}", deadline_at=deadline_at)
    except MithrilApiError as error:
        if error.status == 404:
            return None
        raise
    if not isinstance(bid, dict) or not _ownership(bid) or bid.get("fid") != bid_id:
        raise MithrilApiError("Mithril bid does not belong to this project")
    return bid


def _instances(bid_id, *, deadline_at=None):
    config = settings()
    rows = list(
        pages(
            "/instances",
            params={"project": config["project_id"], "bid_fid_in": bid_id},
            deadline_at=deadline_at,
        )
    )
    if any(row.get("bid") != bid_id or row.get("project") != config["project_id"] for row in rows):
        raise MithrilApiError("Mithril returned an instance outside the bid")
    return rows


def get_instance(bid_id, *, deadline_at=None):
    bid = _bid(bid_id, deadline_at=deadline_at)
    rows = _instances(bid_id, deadline_at=deadline_at)
    states = {row["status"] for row in rows}
    if bid is None:
        return None if all(s in _STOPPED for s in states) else {"id": bid_id, "status": "stopping"}
    if bid["status"] == "Terminated":
        state = "stopped" if all(s in _STOPPED for s in states) else "stopping"
    elif bid["status"] in {"Preempting", "Paused", "Relocating"} or states & {
        "STATUS_PREEMPTING",
        "STATUS_RELOCATING",
        *_STOPPED,
    }:
        # End this attempt and cancel the bid. Flash, not Mithril's automatic rebidding,
        # owns checkpoint restoration and the next attempt's artifact namespace.
        state = "preempting"
    elif "STATUS_RUNNING" in states:
        state = "running"
    else:
        state = "starting"
    return {"id": bid_id, "status": state}


def public_ip(bid_id, *, deadline_at=None):
    if _bid(bid_id, deadline_at=deadline_at) is None:
        return None
    rows = _instances(bid_id, deadline_at=deadline_at)
    for row in rows:
        if row["status"] != "STATUS_RUNNING":
            continue
        destination = row.get("ssh_destination") or ""
        # Mithril returns user@IPv4 (optionally :port). Reject private and malformed endpoints.
        host = destination.rsplit("@", 1)[-1].split(":", 1)[0]
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            continue
        if address.version == 4 and address.is_global:
            return str(address)
    return None


def terminate(bid_id):
    deadline = time.time() + 180
    if _bid(bid_id, deadline_at=deadline) is not None:
        with contextlib.suppress(MithrilApiError):
            call("DELETE", f"/spot/bids/{bid_id}", deadline_at=deadline)
    while time.time() < deadline:
        bid = _bid(bid_id, deadline_at=deadline)
        rows = _instances(bid_id, deadline_at=deadline)
        if (bid is None or bid["status"] == "Terminated") and all(
            row["status"] in _STOPPED for row in rows
        ):
            return
        time.sleep(min(2, max(0, deadline - time.time())))
    raise MithrilApiError("Mithril bid cancellation and instance termination are not confirmed")


def create(*, name, offer, payload, image_reference, gpu, gpu_count, capsule, disk_gb, deadline_at):
    from flash.providers.mithril.bootstrap import startup_script

    config = settings()
    if offer.gpu != gpu or offer.count != gpu_count or offer.disk_gb < disk_gb:
        raise ValueError("Mithril offer does not fit the requested GPU and disk shape")
    if not offer.region:
        raise ValueError("Mithril offer has no region")
    labels = {
        "flash-owner": "flash-training",
        "flash-run": payload["run_id"],
        "flash-attempt": str(payload["attempt"]),
        "flash-spec": hashlib.sha256(payload["job_spec_json"].encode()).hexdigest(),
        "flash-source": hashlib.sha256(
            json.dumps(payload["source_snapshot"], sort_keys=True).encode()
        ).hexdigest(),
    }
    cap = SPOT_GPUS[gpu].keep_running_usd * gpu_count

    def reconcile():
        matches = [
            row
            for row in pages(
                "/spot/bids",
                params={"project": config["project_id"], "name": name},
                deadline_at=deadline_at,
            )
            if row.get("name") == name
        ]
        if len(matches) > 1:
            raise UnreconciledCreateError("Multiple Mithril bids exist for this attempt")
        if not matches:
            return None
        bid = matches[0]
        from flash.providers.mithril.prices import dollars

        if (
            _ownership(bid) != labels
            or bid["instance_type"] != offer.instance_type
            or bid["instance_quantity"] != 1
            or dollars(bid["limit_price"]) != cap
            or bid["status"] == "Terminated"
        ):
            raise UnreconciledCreateError(
                "Mithril attempt name belongs to another or terminated launch"
            )
        return bid["fid"]

    existing = reconcile()
    if existing:
        return existing
    owner_line = (
        _OWNER_PREFIX + base64.b64encode(json.dumps(labels, sort_keys=True).encode()).decode()
    )
    script = startup_script(
        payload, capsule, image_reference, owner_line=owner_line, deadline_at=deadline_at
    )
    launch = {
        "volumes": [],
        "ssh_keys": [key.strip() for key in config["ssh_key_ids"].split(",") if key.strip()],
        "startup_script": script,
    }
    if config["image_version"]:
        launch["image_version"] = config["image_version"]
    body = {
        "project": config["project_id"],
        # Bid where the admitted (cheapest qualifying) offer is.
        "region": offer.region,
        "instance_type": offer.instance_type,
        "limit_price": f"${cap:.2f}",
        "instance_quantity": 1,
        "name": name,
        "launch_specification": launch,
    }
    try:
        result = call("POST", "/spot/bids", body=body, deadline_at=deadline_at)
        bid_id = result["fid"]
        if not isinstance(bid_id, str) or not re.fullmatch(r"bid_[A-Za-z0-9]+", bid_id):
            raise ValueError("invalid bid identity")
        return bid_id
    except (MithrilApiError, KeyError, TypeError, ValueError) as error:
        status = error.status if isinstance(error, MithrilApiError) else 0
        if status:
            cause = f"http {status}"
        elif isinstance(error, MithrilApiError):
            cause = "transport error or timeout"
        else:
            cause = "invalid create response"

        def reconciled():
            try:
                return reconcile()
            except (MithrilApiError, TimeoutError):
                raise UnreconciledCreateError(
                    "Mithril launch reconciliation unavailable; refusing duplicate allocation",
                    cause=cause,
                ) from None

        existing = reconciled()
        if existing:
            return existing
        if 400 <= status < 500:
            # A 4xx is a definitive refusal, and reconciliation found no bid under this attempt's
            # name, so nothing was created: requeue (a new attempt name) instead of failing.
            # Cool down this exact region/instance type so the requeue waits for an offer Mithril
            # accepts (often another interchangeable class) instead of rebidding the refused one.
            if status != 429:
                from flash.providers.mithril.region_health import mark_refused

                mark_refused(offer.region, offer.instance_type)
            raise CapacityUnavailableError(
                f"Mithril refused the Spot bid ({cause}); requeuing", cause=cause
            ) from None
        # A 5xx, transport failure, timeout or unreadable 2xx may still have created the bid after
        # the listing above. Look once more after a bounded pause and adopt a late bid, but never
        # retry an unknown POST outcome: the next attempt's new name would leave a late bid from
        # this one running beside it until the run ends.
        time.sleep(max(0, min(CREATE_RECHECK_SECONDS, deadline_at - time.time())))
        existing = reconciled()
        if existing:
            return existing
        raise UnreconciledCreateError(
            "Mithril launch could not be reconciled; refusing duplicate allocation", cause=cause
        ) from None
