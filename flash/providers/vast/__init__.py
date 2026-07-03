"""Vast.ai provider: verified-datacenter single-GPU containers (REST only).

Opt-in live-market substrate (available only when ``VAST_API_KEY`` is set); detects completion
from the worker's HF artifacts. Implements the shared ``base.Provider`` interface, so the
allocator treats it interchangeably with RunPod/Lambda.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from flash.providers.base import GpuClass, JobHandle, PollResult, Provider


class VastProvider:
    """``base.Provider`` for the Vast.ai verified-datacenter substrate."""

    name = "vast"

    def is_configured(self) -> bool:
        from flash.providers.vast.auth import load_api_key

        return load_api_key() is not None

    def preflight(self, require_hf: bool = True) -> list[str]:
        from flash.providers.vast.preflight import missing_credentials

        return missing_credentials(require_hf=require_hf)

    def gpu_classes(self) -> list[GpuClass]:
        from flash.providers.vast.gpus import gpu_classes

        return gpu_classes()

    def hourly_rate(self, gpu: str) -> float:
        from flash.providers.vast.pricing import hourly_rate

        return hourly_rate(gpu)

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
        from flash.providers.vast.jobs import submit_run_vast

        return submit_run_vast(
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

        from flash.providers.runpod.jobs import make_hf_heartbeat_reader
        from flash.providers.vast.jobs import (
            PROVISION_GRACE_S,
            VastJobHandle,
            poll_vast_job,
        )

        hf_repo = spec.train.hf_repo
        prefix = f"{spec.phase}/{spec.run_id}"
        reader = make_hf_heartbeat_reader(hf_repo, prefix) if hf_repo else None
        vh = VastJobHandle.from_dict(handle.to_dict())
        if log is not None:
            print(f"attaching: vast instance={vh.instance_id}", file=log, flush=True)
        # Deadline is launch-relative (anchored to handle.started_ts), not reattach-relative: resetting on recovery would extend the billable window unbounded.
        deadline = max(60, int(spec.gpu.max_wall_seconds)) + PROVISION_GRACE_S
        try:
            return poll_vast_job(
                vh,
                spec,
                seed,
                log=log,
                heartbeat_reader=reader,
                deadline_s=deadline,
            )
        finally:
            # attach_run has no submit_run_vast teardown; destroy the reattached instance here so a recovered run stops billing.
            from flash.providers.vast.jobs import _best_effort_destroy, destroy_run_instances

            with contextlib.suppress(Exception):
                if not _best_effort_destroy(vh.instance_id, context="poll recovery teardown"):
                    # Unconfirmed teardown: the active-run sweep shields this label, so escalate to a
                    # run-scoped reap by label (re-lists + retries, not active-shielded), mirroring the
                    # submit_run_vast teardown finally.
                    destroy_run_instances(spec.run_id)

    def cancel(self, handle: JobHandle) -> None:
        from flash.providers.vast.jobs import cancel

        cancel(handle.to_dict())

    def destroy(self, handle: JobHandle) -> None:
        from flash._logging import get_logger
        from flash.providers.vast import api as vast_api

        d = handle.to_dict()
        iid = d.get("instance_id")
        if not iid:
            return
        # ``destroy_instance`` returns False on ``success:false`` / breakdown — the box is STILL billable.
        # Warn + raise instead of recording a false success (best-effort callers catch it and fall back to
        # sweep_orphans). Pass ``iid`` through unconverted: destroy_instance does the ``int()`` internally,
        # so converting here would raise instead of surfacing the False -> raise path.
        if not vast_api.destroy_instance(iid):
            get_logger(__name__).warning(
                "vast destroy_instance(%s) returned unconfirmed (success:false / breakdown); "
                "instance may still be billing — relying on sweep_orphans backstop",
                iid,
            )
            raise vast_api.VastApiError(
                f"vast destroy_instance({iid}) unconfirmed (success:false); instance may still bill"
            )

    def gc(self, spec) -> None:
        from flash.providers.vast.jobs import destroy_run_instances

        destroy_run_instances(spec.run_id)

    def run_instances_remaining(self, run_id: str) -> list[int]:
        """Instance ids still carrying ``run_id``'s label after ``gc``. Empty == confirmed clear;
        non-empty == a possibly-live instance survives. RAISES on a listing failure so the caller can't
        mistake "couldn't list" for "clear" (``gc`` returns an empty list, not an error, on an
        unconfirmed DELETE)."""
        from flash.providers.vast.jobs import run_instances_remaining

        return run_instances_remaining(run_id)

    def sweep_orphans(
        self,
        active_labels: set[str] | Callable[[], set[str]] | None = None,
        known_labels: set[str] | Callable[[], set[str]] | None = None,
    ) -> list[int]:
        """Vast crash-recovery sweep (called via the provider object at startup).

        ``known_labels`` scopes the sweep to this control plane's own runs (multi-plane safety)."""
        from flash.providers.vast.jobs import sweep_orphans

        return sweep_orphans(active_labels=active_labels, known_labels=known_labels)


PROVIDER: Provider = VastProvider()
