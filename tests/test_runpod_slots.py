"""Shared RunPod endpoint-slot quota: the backend client + the acquire/release/reconcile policy.

The cap is enforced cross-process via the freesolo backend store when an operator internal key is
configured (claim queues on a full cap, releases route to the store, startup reconcile reclaims
leaked slots), and falls back to an in-process semaphore otherwise. These are CPU/offline tests:
the network boundary (urlopen) and the RunPod endpoint list are stubbed.
"""

from __future__ import annotations

import json
import threading
import urllib.error

import pytest

import flash.providers.runpod.train.endpoints as ep
from flash.providers.runpod import api as runpod_api
from flash.providers.runpod import slots

# --------------------------------------------------------------------------- client


def test_slots_claim_posts_and_parses(monkeypatch):
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "ikey")
    captured: dict = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"claimed": true, "inUse": 3}'

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data)
        captured["auth"] = req.get_header("Authorization")
        return _Resp()

    monkeypatch.setattr(slots.urllib.request, "urlopen", fake_urlopen)

    claimed, in_use = slots.claim("flash-x", cap=28, claimed_by="h:1")
    assert (claimed, in_use) == (True, 3)
    assert captured["url"].endswith("/api/runpod/internal/slots/claim")
    assert captured["body"] == {"name": "flash-x", "cap": 28, "claimedBy": "h:1"}
    assert captured["auth"] == "Bearer ikey"


def test_slots_claim_without_internal_key_raises(monkeypatch):
    monkeypatch.delenv("FREESOLO_INTERNAL_KEY", raising=False)
    assert slots.internal_key() is None
    with pytest.raises(slots.SlotStoreError):
        slots.claim("flash-x", cap=28)


def test_slots_claim_wraps_network_error(monkeypatch):
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "ikey")

    def boom(req, timeout=None):
        raise urllib.error.URLError("down")

    monkeypatch.setattr(slots.urllib.request, "urlopen", boom)
    with pytest.raises(slots.SlotStoreError):
        slots.claim("flash-x", cap=28)


# --------------------------------------------------------------------------- acquire


def test_acquire_shared_records_shared_mode(monkeypatch):
    monkeypatch.setattr(ep, "_ACQUIRED", {})
    monkeypatch.setattr(slots, "internal_key", lambda: "k")
    seen = []
    monkeypatch.setattr(
        slots, "claim", lambda name, *, cap, claimed_by=None: (seen.append((name, cap)) or (True, 1))
    )
    ep._acquire_endpoint_slot("flash-5090-abc")
    assert ep._ACQUIRED["flash-5090-abc"] == "shared"
    assert seen == [("flash-5090-abc", ep.RUNPOD_ENDPOINT_SLOT_CAP)]


def test_acquire_shared_queues_until_slot_free(monkeypatch):
    monkeypatch.setattr(ep, "_ACQUIRED", {})
    monkeypatch.setattr(ep.time, "sleep", lambda *_: None)
    monkeypatch.setattr(slots, "internal_key", lambda: "k")
    seq = iter([(False, 28), (False, 28), (True, 28)])
    monkeypatch.setattr(slots, "claim", lambda name, *, cap, claimed_by=None: next(seq))
    ep._acquire_endpoint_slot("flash-x")  # queues twice, then claims — must not raise
    assert ep._ACQUIRED["flash-x"] == "shared"


def test_acquire_is_idempotent_for_held_name(monkeypatch):
    monkeypatch.setattr(ep, "_ACQUIRED", {"flash-x": "shared"})
    # A held name short-circuits before touching the store at all.
    monkeypatch.setattr(
        slots, "claim", lambda *a, **k: pytest.fail("must not re-claim a held name")
    )
    ep._acquire_endpoint_slot("flash-x")


def test_acquire_falls_back_to_local_after_persistent_store_errors(monkeypatch):
    monkeypatch.setattr(ep, "_ACQUIRED", {})
    monkeypatch.setattr(ep, "_LOCAL_SLOTS", threading.Semaphore(ep.RUNPOD_ENDPOINT_SLOT_CAP))
    monkeypatch.setattr(ep, "_SLOT_STORE_MAX_ERRORS", 2)
    monkeypatch.setattr(ep.time, "sleep", lambda *_: None)
    monkeypatch.setattr(slots, "internal_key", lambda: "k")

    def boom(*a, **k):
        raise slots.SlotStoreError("store down")

    monkeypatch.setattr(slots, "claim", boom)
    ep._acquire_endpoint_slot("flash-x")
    # Falls back to the local semaphore (single-process cap) rather than deadlocking.
    assert ep._ACQUIRED["flash-x"] == "local"


def test_acquire_local_without_internal_key(monkeypatch):
    monkeypatch.setattr(ep, "_ACQUIRED", {})
    monkeypatch.setattr(ep, "_LOCAL_SLOTS", threading.Semaphore(2))
    monkeypatch.setattr(slots, "internal_key", lambda: None)
    monkeypatch.setattr(slots, "claim", lambda *a, **k: pytest.fail("no store without a key"))
    ep._acquire_endpoint_slot("flash-x")
    assert ep._ACQUIRED["flash-x"] == "local"


# --------------------------------------------------------------------------- release


def test_release_shared_calls_store_and_is_idempotent(monkeypatch):
    monkeypatch.setattr(ep, "_ACQUIRED", {"flash-x": "shared"})
    released = []
    monkeypatch.setattr(slots, "release", lambda name: (released.append(name) or True))
    assert ep._release_endpoint_slot("flash-x") is True
    assert released == ["flash-x"]
    assert "flash-x" not in ep._ACQUIRED
    # Second call: never acquired now -> no-op (no second store call).
    assert ep._release_endpoint_slot("flash-x") is False
    assert released == ["flash-x"]


def test_release_shared_swallows_store_error(monkeypatch):
    monkeypatch.setattr(ep, "_ACQUIRED", {"flash-x": "shared"})

    def boom(name):
        raise slots.SlotStoreError("down")

    monkeypatch.setattr(slots, "release", boom)
    # The slot is dropped from tracking and reconcile will reclaim the row; must not raise.
    assert ep._release_endpoint_slot("flash-x") is True
    assert "flash-x" not in ep._ACQUIRED


def test_release_local_returns_semaphore_slot(monkeypatch):
    sem = threading.Semaphore(1)
    sem.acquire()  # 0 free
    monkeypatch.setattr(ep, "_LOCAL_SLOTS", sem)
    monkeypatch.setattr(ep, "_ACQUIRED", {"flash-x": "local"})
    assert ep._release_endpoint_slot("flash-x") is True
    assert sem.acquire(blocking=False) is True  # the slot was returned


# --------------------------------------------------------------------------- reconcile


def test_reconcile_filters_flash_endpoints_and_calls_store(monkeypatch):
    monkeypatch.setattr(slots, "internal_key", lambda: "k")
    monkeypatch.setattr(
        runpod_api,
        "list_endpoints",
        lambda: [
            {"name": "flash-5090-abc"},
            {"name": "other-thing"},
            {"name": "flash-4090-xyz"},
            {"name": None},
        ],
    )
    got: dict = {}

    def _reconcile(live):
        got["live"] = live
        return {"inUse": 2, "reclaimed": 1}

    monkeypatch.setattr(slots, "reconcile", _reconcile)
    ep.reconcile_endpoint_slots()
    assert got["live"] == ["flash-5090-abc", "flash-4090-xyz"]


def test_reconcile_noop_without_internal_key(monkeypatch):
    monkeypatch.setattr(slots, "internal_key", lambda: None)
    monkeypatch.setattr(
        runpod_api, "list_endpoints", lambda: pytest.fail("must not hit RunPod without a key")
    )
    ep.reconcile_endpoint_slots()


def test_reconcile_skips_when_endpoint_list_fails(monkeypatch):
    monkeypatch.setattr(slots, "internal_key", lambda: "k")

    def boom():
        raise RuntimeError("RunPod API down")

    monkeypatch.setattr(runpod_api, "list_endpoints", boom)
    # Must NOT reconcile against a partial/empty list (would wrongly reclaim live slots).
    monkeypatch.setattr(slots, "reconcile", lambda live: pytest.fail("must not reconcile blind"))
    ep.reconcile_endpoint_slots()  # best-effort: swallows the error
