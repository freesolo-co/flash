"""Authenticated localhost session bridge for multi-turn / tool GRPO+OPD on OpenRLHF.

OpenRLHF rollout actors run inside Ray and must never hold Flash environment secrets. This bridge
runs in the parent (Flash) process, owns the live multi-turn/tool environment *sessions* (and any
secrets they carry), and exposes a minimal authenticated ``reset``/``step``/``close`` RPC over
localhost that a Ray actor calls to drive exactly one episode at a time. It mirrors the reward
bridge's auth model (a per-run capability token compared with :func:`secrets.compare_digest` on a
localhost-only ``ThreadingHTTPServer``) and adds per-session leases plus a strict per-session step
ordinal so one actor cannot advance, replay, or close another actor's session.

This module is transport/session infrastructure only. It carries opaque JSON observations and
actions and drives an injected ``session_factory``; it wires no training mode into any worker. The
concrete ``EnvironmentMultiTurn`` adapter and the multi-turn rollout that uses this bridge land in a
later PR. Environment construction (and therefore every environment secret) happens exclusively in
``session_factory``, which the parent supplies; nothing but the session id, the lease, and opaque
observations ever crosses the wire, so no environment secret reaches Ray.
"""

from __future__ import annotations

import contextlib
import json
import secrets
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol

# Bound request bodies so a bad or hostile local caller cannot exhaust parent memory. Observations
# and actions are small transcript fragments; 4 MiB is generous headroom over the reward bridge.
_SESSION_MAX_BODY_BYTES = 4 * 1024 * 1024


class SessionDriver(Protocol):
    """One live environment episode, owned parent-side by :class:`SessionBridge`.

    ``step`` consumes one opaque model action and returns ``(observation, done)``; ``close`` releases
    the episode. Implementations hold the real environment and its secrets and are never serialized
    into Ray."""

    def step(self, action: Any) -> tuple[Any, bool]:  # (observation, done)
        ...

    def close(self) -> None: ...


# ``session_factory(reset_payload) -> (driver, initial_observation)``. Runs in the parent only.
SessionFactory = Callable[[Any], "tuple[SessionDriver, Any]"]


class _SessionHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _Session:
    """Server-side state for one episode: its driver, lease, and monotonic step ordinal."""

    __slots__ = (
        "closed",
        "done",
        "driver",
        "last_ordinal",
        "last_payload",
        "lease",
        "lock",
        "max_turns",
        "next_ordinal",
        "turns",
    )

    def __init__(self, driver: SessionDriver, lease: str, max_turns: int) -> None:
        self.driver = driver
        self.lease = lease
        self.max_turns = int(max_turns)
        self.next_ordinal = 0
        self.turns = 0
        self.done = False
        self.closed = False
        # Cache the most recent step response so an at-least-once retry of the same ordinal replays
        # the identical result instead of double-advancing the environment.
        self.last_ordinal: int | None = None
        self.last_payload: dict[str, Any] | None = None
        self.lock = threading.Lock()


class SessionError(Exception):
    """Bridge-level rejection carrying the HTTP status to return."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = int(status)
        self.message = message


class SessionBridge:
    """Authenticated localhost bridge exposing reset/step/close for one env per session.

    ``max_turns`` bounds the number of steps any single session may take. ``max_sessions`` bounds the
    number of concurrently-open sessions. ``token`` is the per-run capability token; when omitted a
    fresh 256-bit token is generated. The ``reset_url``/``step_url``/``close_url`` (which embed the
    token) are the only values that need to cross to Ray."""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        max_turns: int,
        max_sessions: int = 1024,
        token: str | None = None,
    ) -> None:
        if int(max_turns) <= 0:
            raise ValueError("session bridge max_turns must be positive")
        if int(max_sessions) <= 0:
            raise ValueError("session bridge max_sessions must be positive")
        self._factory = session_factory
        self._max_turns = int(max_turns)
        self._max_sessions = int(max_sessions)
        self._token = token or secrets.token_urlsafe(32)
        self._reset_path = f"/session/{self._token}/reset"
        self._step_path = f"/session/{self._token}/step"
        self._close_path = f"/session/{self._token}/close"
        self._sessions: dict[str, _Session] = {}
        self._sessions_lock = threading.Lock()
        bridge = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, _format, *_args):  # silence default stderr logging
                return

            def _send(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                route = bridge._route(self.path)
                if route is None:
                    # Constant-time-compared token failed: reveal nothing.
                    self._send(401, {"error": "unauthorized"})
                    return
                try:
                    raw_length = self.headers.get("Content-Length")
                    if raw_length is None:
                        raise SessionError(411, "missing content length")
                    length = int(raw_length)
                    if length <= 0 or length > _SESSION_MAX_BODY_BYTES:
                        raise SessionError(413, "invalid session request size")
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    if not isinstance(payload, dict):
                        raise SessionError(400, "session request must be an object")
                    status, result = bridge._dispatch(route, payload)
                except SessionError as exc:
                    self._send(exc.status, {"error": exc.message})
                    return
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    self._send(400, {"error": "invalid session request"})
                    return
                except Exception as exc:  # pragma: no cover - defensive
                    print(
                        f"[rl-openrlhf] session bridge failed ({type(exc).__name__}: {exc})",
                        flush=True,
                    )
                    self._send(500, {"error": "session driver failed"})
                    return
                self._send(status, result)

        self._server = _SessionHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="openrlhf-session-bridge",
            daemon=True,
        )
        self._thread.start()
        port = int(self._server.server_address[1])
        base = f"http://127.0.0.1:{port}"
        self.reset_url = f"{base}{self._reset_path}"
        self.step_url = f"{base}{self._step_path}"
        self.close_url = f"{base}{self._close_path}"

    # -- routing / auth --------------------------------------------------------------------------

    def _route(self, path: str) -> str | None:
        """Constant-time-match the authenticated path to a route name, else ``None``."""
        for name, authed in (
            ("reset", self._reset_path),
            ("step", self._step_path),
            ("close", self._close_path),
        ):
            if secrets.compare_digest(path, authed):
                return name
        return None

    def _dispatch(self, route: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if route == "reset":
            return self._reset(payload)
        if route == "step":
            return self._step(payload)
        return self._close(payload)

    def _lookup_leased(self, payload: dict[str, Any]) -> _Session:
        """Resolve a session by id and verify its lease (constant-time), else raise."""
        session_id = payload["session_id"]
        lease = payload["lease"]
        if not isinstance(session_id, str) or not isinstance(lease, str):
            raise SessionError(400, "session_id and lease must be strings")
        with self._sessions_lock:
            session = self._sessions.get(session_id)
        # Compare the lease even when the session is missing (against a dummy) to avoid leaking
        # existence via timing; an unknown id then still fails as unauthorized.
        expected = session.lease if session is not None else secrets.token_urlsafe(32)
        if not secrets.compare_digest(lease, expected) or session is None:
            raise SessionError(403, "unauthorized session")
        return session

    # -- endpoints -------------------------------------------------------------------------------

    def _reset(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        example = payload.get("example")
        with self._sessions_lock:
            if len(self._sessions) >= self._max_sessions:
                raise SessionError(429, "session capacity exhausted")
        # Build the environment (and consume any secrets) entirely parent-side.
        driver, observation = self._factory(example)
        session_id = secrets.token_urlsafe(24)
        lease = secrets.token_urlsafe(32)
        session = _Session(driver, lease, self._max_turns)
        with self._sessions_lock:
            self._sessions[session_id] = session
        return 200, {
            "session_id": session_id,
            "lease": lease,
            "observation": observation,
            "ordinal": 0,
            "max_turns": self._max_turns,
        }

    def _step(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        session = self._lookup_leased(payload)
        ordinal = payload["ordinal"]
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise SessionError(400, "ordinal must be an integer")
        action = payload.get("action")
        with session.lock:
            if session.closed:
                raise SessionError(409, "session closed")
            # Idempotent at-least-once retry: an exact re-send of the last committed ordinal replays
            # the cached response and does not advance the environment again.
            if session.last_ordinal is not None and ordinal == session.last_ordinal:
                assert session.last_payload is not None
                return 200, session.last_payload
            if ordinal != session.next_ordinal:
                raise SessionError(409, "session step out of order")
            if session.done:
                raise SessionError(409, "session already done")
            if session.turns >= session.max_turns:
                raise SessionError(409, "session turn budget exhausted")
            observation, done = session.driver.step(action)
            done = bool(done)
            session.turns += 1
            # The final permitted turn always terminates the episode even if the env did not.
            if session.turns >= session.max_turns:
                done = True
            session.done = done
            committed = session.next_ordinal
            session.next_ordinal += 1
            result = {
                "observation": observation,
                "done": done,
                "ordinal": committed,
                "turns": session.turns,
            }
            session.last_ordinal = committed
            session.last_payload = result
            return 200, result

    def _close(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        session = self._lookup_leased(payload)
        session_id = payload["session_id"]
        with session.lock:
            already = session.closed
            session.closed = True
            if not already:
                try:
                    session.driver.close()
                finally:
                    with self._sessions_lock:
                        self._sessions.pop(session_id, None)
        return 200, {"ok": True, "closed": True}

    # -- lifecycle -------------------------------------------------------------------------------

    @property
    def token(self) -> str:
        return self._token

    def open_session_count(self) -> int:
        with self._sessions_lock:
            return len(self._sessions)

    def shutdown(self) -> None:
        """Close every open session and stop the server thread."""
        with self._sessions_lock:
            sessions = list(self._sessions.items())
            self._sessions.clear()
        for _session_id, session in sessions:
            with session.lock:
                if not session.closed:
                    session.closed = True
                    # best-effort teardown
                    with contextlib.suppress(Exception):
                        session.driver.close()
        self._server.shutdown()
        self._server.server_close()
