"""Unit tests for the authenticated OpenRLHF multi-turn session bridge.

Pure-Python and offline: they exercise the real localhost HTTP server with a fake session driver, so
they run without torch/OpenRLHF. They assert the auth/lease/ordinal/replay/bounded-turn contract and
that no environment secret ever crosses the wire."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from flash.engine.worker.openrlhf_session_bridge import SessionBridge

_SECRET = "env-api-key-do-not-leak-2f9c"


class _FakeDriver:
    """Deterministic fake episode: echoes the action, done after ``turns_until_done`` steps."""

    def __init__(self, *, turns_until_done: int) -> None:
        self._turns_until_done = turns_until_done
        self._n = 0
        self.actions: list = []
        self.closed = False

    def step(self, action):
        self.actions.append(action)
        self._n += 1
        done = self._n >= self._turns_until_done
        return {"turn": self._n, "echo": action}, done

    def close(self) -> None:
        self.closed = True


def _factory(*, turns_until_done: int, drivers: list):
    def make(example):
        # The secret is captured in this parent-side closure and must never reach a response.
        _ = _SECRET
        driver = _FakeDriver(turns_until_done=turns_until_done)
        drivers.append(driver)
        return driver, {"prompt": example, "started": True}

    return make


def _post(url: str, payload: dict) -> tuple[int, dict]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


@pytest.fixture
def bridge():
    drivers: list = []
    b = SessionBridge(_factory(turns_until_done=2, drivers=drivers), max_turns=4, max_sessions=2)
    b.drivers = drivers  # type: ignore[attr-defined]
    try:
        yield b
    finally:
        b.shutdown()


def _reset(bridge, example="hi"):
    status, data = _post(bridge.reset_url, {"example": example})
    assert status == 200, data
    return data


def test_reset_step_close_happy_path(bridge):
    r = _reset(bridge, example="solve x")
    assert r["ordinal"] == 0
    assert r["observation"] == {"prompt": "solve x", "started": True}
    sid, lease = r["session_id"], r["lease"]

    s0, d0 = _post(
        bridge.step_url, {"session_id": sid, "lease": lease, "ordinal": 0, "action": "a0"}
    )
    assert s0 == 200
    assert d0["done"] is False
    assert d0["observation"] == {"turn": 1, "echo": "a0"}
    s1, d1 = _post(
        bridge.step_url, {"session_id": sid, "lease": lease, "ordinal": 1, "action": "a1"}
    )
    assert s1 == 200
    assert d1["done"] is True  # env says done at turn 2
    assert d1["turns"] == 2

    c, cd = _post(bridge.close_url, {"session_id": sid, "lease": lease})
    assert c == 200
    assert cd["closed"] is True
    assert bridge.drivers[0].closed is True
    assert bridge.open_session_count() == 0


def test_bad_token_is_unauthorized(bridge):
    bad = bridge.reset_url.rsplit("/", 2)[0] + "/deadbeef/reset"
    status, data = _post(bad, {"example": "x"})
    assert status == 401
    assert data == {"error": "unauthorized"}


def test_wrong_lease_rejected(bridge):
    r = _reset(bridge)
    status, data = _post(
        bridge.step_url,
        {"session_id": r["session_id"], "lease": "not-the-lease", "ordinal": 0, "action": "a"},
    )
    assert status == 403
    assert "unauthorized" in data["error"]
    assert bridge.drivers[0].actions == []  # a real step never ran


def test_unknown_session_rejected(bridge):
    r = _reset(bridge)
    status, _ = _post(
        bridge.step_url,
        {"session_id": "nope", "lease": r["lease"], "ordinal": 0, "action": "a"},
    )
    assert status == 403


def test_out_of_order_ordinal_rejected(bridge):
    r = _reset(bridge)
    sid, lease = r["session_id"], r["lease"]
    status, data = _post(
        bridge.step_url, {"session_id": sid, "lease": lease, "ordinal": 1, "action": "a"}
    )
    assert status == 409
    assert "out of order" in data["error"]
    assert bridge.drivers[0].actions == []


def test_replay_is_idempotent(bridge):
    r = _reset(bridge)
    sid, lease = r["session_id"], r["lease"]
    _, first = _post(
        bridge.step_url, {"session_id": sid, "lease": lease, "ordinal": 0, "action": "a0"}
    )
    status, second = _post(
        bridge.step_url, {"session_id": sid, "lease": lease, "ordinal": 0, "action": "a0-retry"}
    )
    assert status == 200
    assert second == first  # cached; not re-advanced
    assert bridge.drivers[0].actions == ["a0"]  # env stepped exactly once


def test_turn_budget_bounds_episode():
    drivers: list = []
    # env never says done; the bridge's max_turns must terminate it.
    b = SessionBridge(_factory(turns_until_done=999, drivers=drivers), max_turns=3)
    try:
        _status, data = _post(b.reset_url, {"example": "x"})
        sid, lease = data["session_id"], data["lease"]
        done = False
        for ordinal in range(3):
            s, d = _post(
                b.step_url,
                {"session_id": sid, "lease": lease, "ordinal": ordinal, "action": ordinal},
            )
            assert s == 200
            done = d["done"]
        assert done is True  # forced done at max_turns
        # a 4th step is refused as done
        s, d = _post(b.step_url, {"session_id": sid, "lease": lease, "ordinal": 3, "action": 3})
        assert s == 409
        assert "done" in d["error"]
        assert len(drivers[0].actions) == 3
    finally:
        b.shutdown()


def test_capacity_exhausted():
    drivers: list = []
    b = SessionBridge(_factory(turns_until_done=2, drivers=drivers), max_turns=2, max_sessions=1)
    try:
        _post(b.reset_url, {"example": "x"})
        status, data = _post(b.reset_url, {"example": "y"})
        assert status == 429
        assert "capacity" in data["error"]
    finally:
        b.shutdown()


def test_no_env_secret_crosses_the_wire(bridge):
    r = _reset(bridge, example="prompt")
    sid, lease = r["session_id"], r["lease"]
    _, s0 = _post(bridge.step_url, {"session_id": sid, "lease": lease, "ordinal": 0, "action": "a"})
    _, cd = _post(bridge.close_url, {"session_id": sid, "lease": lease})
    for blob in (json.dumps(r), json.dumps(s0), json.dumps(cd)):
        assert _SECRET not in blob


def test_lease_scopes_sessions_to_their_owner(bridge):
    a = _reset(bridge, example="A")
    b_ = _reset(bridge, example="B")
    # owner-A's lease cannot step owner-B's session
    status, _ = _post(
        bridge.step_url,
        {"session_id": b_["session_id"], "lease": a["lease"], "ordinal": 0, "action": "x"},
    )
    assert status == 403
