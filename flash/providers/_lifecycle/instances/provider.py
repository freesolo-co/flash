"""Shared template for rent-a-box (instance) providers.

Lambda and Vast are ~identical thin delegators over the shared ``base.Provider`` interface: they
provision a GPU instance, boot it, and detect completion from the worker's HF artifacts. The only
differences are per-substrate details (which auth/pricing/jobs module to call, the handle class, the
reattach deadline formula, and how a reattached instance is torn down).

``InstanceProvider`` folds the identical method bodies here and defers each per-substrate detail to a
small hook the subclass overrides. "Add a rent-a-box provider" becomes "subclass ``InstanceProvider``
and define the hooks below" — no need to re-derive the poll/submit/gc plumbing.

top-level project imports are confined to shared ``_instance`` and ``base``. ``_hf_artifacts``,
``contextlib``, and substrate-specific modules are imported lazily inside methods. hooks therefore
resolve their targets at call time for monkeypatch seams such as
``vast.jobs.submit_attempt_vast`` and ``lambda_api.terminate_instances``.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any

from flash.providers._lifecycle.instances.instance import InstanceJobHandle
from flash.providers.core.base import GpuClass, JobHandle, PollResult


class InstanceProvider(abc.ABC):
    """Shared ``base.Provider`` template for rent-a-box instance substrates (Lambda, Vast).

    The hooks below are ``@abstractmethod``, so the ABC refuses to instantiate a subclass that
    forgets one — a mis-wired new provider fails at construction, not at a billing-critical teardown.
    """

    # Optional capability (read via getattr, kept off the runtime_checkable Protocol like
    # supports_weight_cache): these substrates price and stock from a LIVE market, so "no candidate
    # right now" is a transient capacity miss the allocator should let a run retry, not proof the
    # class is unsupported. Static-table providers omit it -> False.
    live_capacity = True

    # --- subclass contract: class attrs + abstract hooks each substrate supplies ---
    name: str
    _gpu_identity_attr: str  # ``GpuClass`` identity field: "vast_name" / "lambda_name".

    @property
    @abc.abstractmethod
    def _handle_cls(self) -> type[InstanceJobHandle]:
        """The substrate's ``InstanceJobHandle`` subclass (lazy: a property so importing it stays deferred)."""

    @abc.abstractmethod
    def _load_api_key(self) -> Any: ...

    @abc.abstractmethod
    def _missing_credentials(self, require_hf: bool) -> list[str]: ...

    @abc.abstractmethod
    def _hourly_rate(self, gpu: str) -> float: ...

    @abc.abstractmethod
    def _submit_attempt(
        self,
        spec,
        *,
        log: Any,
        on_handle: Any,
        attempt: int,
        runtime_secrets: dict[str, str] | None,
        source_snapshot: dict | None,
        deadline_at: float | None,
    ) -> PollResult: ...

    @abc.abstractmethod
    def _poll_job(
        self,
        handle: JobHandle,
        spec,
        *,
        log: Any,
        heartbeat_reader: Any,
        deadline_at: float | None,
    ) -> PollResult: ...

    @abc.abstractmethod
    def _teardown_reattached(self, handle: JobHandle, spec) -> None:
        """Destroy an instance recovered via ``poll_attempt`` (attach has no submit teardown)."""

    @abc.abstractmethod
    def _gc(self, run_id: str) -> None: ...

    @abc.abstractmethod
    def _sweep_orphans(
        self,
        *,
        active_labels: set[str] | Callable[[], set[str]] | None,
        known_labels: set[str] | Callable[[], set[str]] | None,
    ) -> list: ...

    # --- shared ``base.Provider`` surface ---------------------------------
    def is_configured(self) -> bool:
        return self._load_api_key() is not None

    def preflight(self, require_hf: bool = True) -> list[str]:
        return self._missing_credentials(require_hf)

    def gpu_classes(self) -> list[GpuClass]:
        from flash.providers.core import base

        return base.gpu_classes_for(self._gpu_identity_attr)

    def hourly_rate(self, gpu: str) -> float:
        return self._hourly_rate(gpu)

    def submit_attempt(
        self,
        spec,
        *,
        log: Any = None,
        on_handle: Any = None,
        attempt: int = 0,
        runtime_secrets: dict[str, str] | None = None,
        on_last_gpu: bool = False,
        source_snapshot: dict | None = None,
        _deadline_at: float | None = None,
    ) -> PollResult:
        # ``on_last_gpu`` is unused: the instance providers use a uniform per-gpu wait (kept for interface parity).
        return self._submit_attempt(
            spec,
            log=log,
            on_handle=on_handle,
            attempt=attempt,
            runtime_secrets=runtime_secrets,
            source_snapshot=source_snapshot,
            deadline_at=_deadline_at,
        )

    def poll_attempt(
        self,
        handle: JobHandle,
        spec,
        *,
        log: Any = None,
        _deadline_at: float | None = None,
    ) -> PollResult:
        import contextlib

        from flash.providers.artifacts.hf import heartbeat_reader_for

        reader = heartbeat_reader_for(spec, deadline_at=_deadline_at)
        h = self._handle_cls.from_dict(handle.to_dict())
        if log is not None:
            print(f"attaching: {self.name} instance={h.instance_id}", file=log, flush=True)
        try:
            return self._poll_job(
                h,
                spec,
                log=log,
                heartbeat_reader=reader,
                deadline_at=_deadline_at,
            )
        finally:
            # attach_run has no submit-time teardown; destroy the reattached instance here so a recovered run stops billing.
            with contextlib.suppress(Exception):
                self._teardown_reattached(h, spec)

    def gc(self, spec) -> None:
        self._gc(spec.run_id)

    def sweep_orphans(
        self,
        active_labels: set[str] | Callable[[], set[str]] | None = None,
        known_labels: set[str] | Callable[[], set[str]] | None = None,
    ) -> list:
        """Crash-recovery sweep (called via the provider object at startup).

        ``known_labels`` scopes the sweep to this control plane's own runs (multi-plane safety)."""
        return self._sweep_orphans(active_labels=active_labels, known_labels=known_labels)
