"""Shared template for rent-a-box (single-GPU instance) providers.

Lambda and Vast are ~identical thin delegators over the shared ``base.Provider`` interface: they
provision one GPU instance, boot it, and detect completion from the worker's HF artifacts. The only
differences are per-substrate details (which auth/pricing/jobs module to call, the handle class, the
reattach deadline formula, and how a reattached instance is torn down).

``InstanceProvider`` folds the identical method bodies here and defers each per-substrate detail to a
small hook the subclass overrides. "Add a rent-a-box provider" becomes "subclass ``InstanceProvider``
and define the hooks below" — no need to re-derive the poll/submit/gc plumbing.

Imports stay confined to the shared kernel (``base``, ``_hf_artifacts``, ``contextlib``); every
substrate call is a lazy import inside a hook so this module is import-side-effect-free and the hooks
resolve their targets at call time (tests monkeypatch e.g. ``vast.jobs.submit_run_vast`` /
``lambda_api.terminate_instances`` — binding at class-definition would defeat those seams).
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any

from flash.providers.base import GpuClass, JobHandle, PollResult


class InstanceProvider(abc.ABC):
    """Shared ``base.Provider`` template for single-GPU instance substrates (Lambda, Vast).

    The hooks below are ``@abstractmethod``, so the ABC refuses to instantiate a subclass that
    forgets one — a mis-wired new provider fails at construction, not at a billing-critical teardown.
    """

    # --- subclass contract: class attrs + abstract hooks each substrate supplies ---
    name: str
    _gpu_identity_attr: str  # ``GpuClass`` identity field: "vast_name" / "lambda_name".

    @property
    @abc.abstractmethod
    def _handle_cls(self) -> type[JobHandle]:
        """The substrate's ``JobHandle`` subclass (lazy: a property so importing it stays deferred)."""

    @abc.abstractmethod
    def _load_api_key(self) -> Any: ...

    @abc.abstractmethod
    def _missing_credentials(self, require_hf: bool) -> list[str]: ...

    @abc.abstractmethod
    def _hourly_rate(self, gpu: str) -> float: ...

    @abc.abstractmethod
    def _submit_run(
        self,
        spec,
        seed: int,
        *,
        log: Any,
        on_handle: Any,
        attempt: int,
        runtime_secrets: dict[str, str] | None,
        code_prefix: str | None,
    ) -> PollResult: ...

    @abc.abstractmethod
    def _poll_job(
        self,
        handle: JobHandle,
        spec,
        seed: int,
        *,
        log: Any,
        heartbeat_reader: Any,
        deadline_s: float,
    ) -> PollResult: ...

    @abc.abstractmethod
    def _reattach_deadline(self, spec) -> float:
        """Launch-relative wall deadline for a reattached poll (per-substrate; the formulas differ)."""

    @abc.abstractmethod
    def _teardown_reattached(self, handle: JobHandle, spec) -> None:
        """Destroy an instance recovered via ``poll`` (attach has no submit teardown to lean on)."""

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
        from flash.providers import base

        return base.gpu_classes_for(self._gpu_identity_attr)

    def hourly_rate(self, gpu: str) -> float:
        return self._hourly_rate(gpu)

    def submit_run(
        self,
        spec,
        seed: int,
        *,
        log: Any = None,
        on_handle: Any = None,
        attempt: int = 0,
        runtime_secrets: dict[str, str] | None = None,
        on_last_gpu: bool = False,
        code_prefix: str | None = None,
    ) -> PollResult:
        # ``on_last_gpu`` is unused: the instance providers use a uniform per-GPU wait (kept for interface parity).
        return self._submit_run(
            spec,
            seed,
            log=log,
            on_handle=on_handle,
            attempt=attempt,
            runtime_secrets=runtime_secrets,
            code_prefix=code_prefix,
        )

    def poll(self, handle: JobHandle, spec, seed: int, *, log: Any = None) -> PollResult:
        import contextlib

        from flash.providers._hf_artifacts import make_hf_heartbeat_reader

        hf_repo = spec.train.hf_repo
        prefix = f"{spec.phase}/{spec.run_id}"
        reader = make_hf_heartbeat_reader(hf_repo, prefix) if hf_repo else None
        h = self._handle_cls.from_dict(handle.to_dict())
        if log is not None:
            print(f"attaching: {self.name} instance={h.instance_id}", file=log, flush=True)
        # Deadline is launch-relative, not reattach-relative: resetting on recovery would extend the billable window unbounded.
        deadline = self._reattach_deadline(spec)
        try:
            return self._poll_job(
                h,
                spec,
                seed,
                log=log,
                heartbeat_reader=reader,
                deadline_s=deadline,
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
