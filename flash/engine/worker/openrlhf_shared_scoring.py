"""bounded non-blocking scoring futures for shared OpenRLHF runs."""

from __future__ import annotations

import copy
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ScoringRegistryError(RuntimeError):
    """base error for invalid shared scoring operations."""


class ScoringCapacityError(ScoringRegistryError):
    """raised when the bounded scoring pool has no submission capacity."""


class ScoringIdentityError(ScoringRegistryError):
    """raised when a result is requested under the wrong scoring identity."""


class ScoringKind(StrEnum):
    """algorithm-specific scoring operation performed by one run bridge."""

    REWARD = "reward"
    TEACHER = "teacher"


@dataclass(frozen=True, slots=True)
class ScoringBatchIdentity:
    """immutable identity of one run step and rollout batch."""

    run_id: str
    step: int
    batch_id: str

    def __post_init__(self) -> None:
        normalized_run_id = str(self.run_id).strip()
        normalized_batch_id = str(self.batch_id).strip()
        if not normalized_run_id:
            raise ValueError("run_id must not be empty")
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ValueError("scoring step must be a non-negative integer")
        if not normalized_batch_id:
            raise ValueError("batch_id must not be empty")
        object.__setattr__(self, "run_id", normalized_run_id)
        object.__setattr__(self, "step", int(self.step))
        object.__setattr__(self, "batch_id", normalized_batch_id)


@dataclass(frozen=True, slots=True)
class ScoringResult:
    """bridge result bound to the exact submission identity and scoring kind."""

    identity: ScoringBatchIdentity
    kind: ScoringKind
    value: Any


ScoringBridge = Callable[[dict[str, Any]], Any]
BridgeRequest = Callable[[str, dict[str, Any]], Any]


@dataclass(frozen=True, slots=True)
class _RunScorer:
    kind: ScoringKind
    bridge: ScoringBridge


def bind_scoring_bridge(url: str, request: BridgeRequest) -> ScoringBridge:
    """bind one existing localhost bridge client to its run-local endpoint.

    the wrapper deliberately adds no retries, error translation, or response shaping.
    reward fail-closed behavior and teacher retry classification therefore remain owned
    by the existing bridge request callable.
    """

    endpoint = str(url).strip()
    if not endpoint:
        raise ValueError("scoring bridge url must not be empty")
    if not callable(request):
        raise TypeError("scoring bridge request must be callable")

    def score(payload: dict[str, Any]) -> Any:
        return request(endpoint, payload)

    return score


def _score_submission(
    identity: ScoringBatchIdentity,
    scorer: _RunScorer,
    payload: dict[str, Any],
) -> ScoringResult:
    return ScoringResult(identity, scorer.kind, scorer.bridge(payload))


class ScoringFuture:
    """future whose result can only be retrieved for its originating identity."""

    __slots__ = ("_future", "_identity", "_kind")

    def __init__(
        self,
        identity: ScoringBatchIdentity,
        kind: ScoringKind,
        future: Future[ScoringResult],
    ) -> None:
        self._identity = identity
        self._kind = kind
        self._future = future

    @property
    def identity(self) -> ScoringBatchIdentity:
        """return the immutable originating run, step, and batch identity."""

        return self._identity

    @property
    def kind(self) -> ScoringKind:
        """return the registered reward or teacher scoring kind."""

        return self._kind

    def done(self) -> bool:
        """return whether bridge scoring reached a terminal result."""

        return self._future.done()

    def cancelled(self) -> bool:
        """return whether scoring was cancelled before execution."""

        return self._future.cancelled()

    def cancel(self) -> bool:
        """cancel scoring if its worker has not started."""

        return self._future.cancel()

    def result_for(
        self,
        identity: ScoringBatchIdentity,
        timeout: float | None = None,
    ) -> ScoringResult:
        """return the result only when the caller supplies the exact identity."""

        if identity != self.identity:
            raise ScoringIdentityError(
                "scoring future identity does not match the requested run, step, and batch"
            )
        result = self._future.result(timeout=timeout)
        if result.identity != self.identity or result.kind is not self.kind:
            raise ScoringIdentityError("scoring worker returned a mismatched result envelope")
        return result


class SharedScoringPool:
    """submit isolated per-run bridge calls without blocking on remote scoring.

    ``pool_size`` bounds both worker threads and unconsumed submissions. submission
    fails immediately with :class:`ScoringCapacityError` at capacity, providing
    backpressure without blocking the training thread. the later scheduler can retry
    admission after consuming or cancelling a completed future.
    """

    def __init__(self, pool_size: int) -> None:
        if isinstance(pool_size, bool) or int(pool_size) < 1:
            raise ValueError("scoring pool size must be positive")
        self._pool_size = int(pool_size)
        self._executor = ThreadPoolExecutor(
            max_workers=self._pool_size,
            thread_name_prefix="openrlhf-scoring",
        )
        self._lock = threading.Lock()
        self._runs: dict[str, _RunScorer] = {}
        self._known_run_ids: set[str] = set()
        self._futures: dict[ScoringBatchIdentity, ScoringFuture] = {}
        self._closed = False

    @property
    def pool_size(self) -> int:
        """return the configured worker and outstanding-submission bound."""

        return self._pool_size

    @property
    def outstanding_count(self) -> int:
        """return the number of submitted results not yet consumed or cancelled."""

        with self._lock:
            return len(self._futures)

    @property
    def pending_identities(self) -> tuple[ScoringBatchIdentity, ...]:
        """return tracked submission identities in admission order."""

        with self._lock:
            return tuple(self._futures)

    @property
    def registered_run_ids(self) -> tuple[str, ...]:
        """return active run bridge registrations in registration order."""

        with self._lock:
            return tuple(self._runs)

    def register_run(
        self,
        run_id: str,
        *,
        kind: ScoringKind,
        bridge: ScoringBridge,
    ) -> None:
        """register one immutable reward or teacher bridge for a logical run."""

        normalized_run_id = self._normalize_run_id(run_id)
        resolved_kind = ScoringKind(kind)
        if not callable(bridge):
            raise TypeError("scoring bridge must be callable")
        with self._lock:
            self._require_open()
            if normalized_run_id in self._known_run_ids:
                raise ScoringRegistryError(
                    f"scoring run id was already registered: {normalized_run_id}"
                )
            self._runs[normalized_run_id] = _RunScorer(resolved_kind, bridge)
            self._known_run_ids.add(normalized_run_id)

    def submit(
        self,
        identity: ScoringBatchIdentity,
        payload: Mapping[str, Any],
    ) -> ScoringFuture:
        """submit a snapshotted bridge payload and return its future immediately."""

        if not isinstance(identity, ScoringBatchIdentity):
            raise TypeError("scoring identity must be ScoringBatchIdentity")
        if not isinstance(payload, Mapping):
            raise TypeError("scoring payload must be a mapping")
        payload_snapshot = copy.deepcopy(dict(payload))

        with self._lock:
            self._require_open()
            try:
                scorer = self._runs[identity.run_id]
            except KeyError as exc:
                raise ScoringRegistryError(f"unknown scoring run: {identity.run_id}") from exc
            if identity in self._futures:
                raise ScoringRegistryError("scoring identity already has an outstanding future")
            if len(self._futures) >= self._pool_size:
                raise ScoringCapacityError(
                    f"scoring pool is full at {self._pool_size} outstanding submissions"
                )
            raw_future = self._executor.submit(
                _score_submission,
                identity,
                scorer,
                payload_snapshot,
            )
            future = ScoringFuture(identity, scorer.kind, raw_future)
            self._futures[identity] = future
            return future

    def consume(
        self,
        identity: ScoringBatchIdentity,
        future: ScoringFuture,
        *,
        timeout: float | None = None,
    ) -> ScoringResult:
        """consume one exact future, preserving the bridge result or exception."""

        with self._lock:
            tracked = self._futures.get(identity)
            if tracked is None:
                raise ScoringIdentityError(
                    "no outstanding scoring future matches the requested identity"
                )
            if tracked is not future:
                raise ScoringIdentityError(
                    "the supplied future is not registered for the requested identity"
                )
        try:
            return future.result_for(identity, timeout=timeout)
        finally:
            if future.done():
                with self._lock:
                    if self._futures.get(identity) is future:
                        self._futures.pop(identity)

    def cancel_run(self, run_id: str) -> int:
        """remove one run bridge and reject all of its current or late results."""

        normalized_run_id = self._normalize_run_id(run_id)
        with self._lock:
            if normalized_run_id not in self._runs:
                raise ScoringRegistryError(f"unknown scoring run: {normalized_run_id}")
            self._runs.pop(normalized_run_id)
            futures = []
            for identity, future in tuple(self._futures.items()):
                if identity.run_id == normalized_run_id:
                    self._futures.pop(identity)
                    futures.append(future)
        for future in futures:
            future.cancel()
        return len(futures)

    def shutdown(self, *, wait: bool = True) -> None:
        """close submission and stop the worker pool."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            futures = tuple(self._futures.values())
            self._futures.clear()
            self._runs.clear()
        for future in futures:
            future.cancel()
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def __enter__(self) -> SharedScoringPool:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.shutdown()

    def _require_open(self) -> None:
        if self._closed:
            raise ScoringRegistryError("scoring pool is closed")

    @staticmethod
    def _normalize_run_id(run_id: str) -> str:
        normalized_run_id = str(run_id).strip()
        if not normalized_run_id:
            raise ValueError("run_id must not be empty")
        return normalized_run_id
