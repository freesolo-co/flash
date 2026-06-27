"""Vast.ai provider: verified-datacenter single-GPU CONTAINERS (REST only).

Vast rents a single-GPU container from a verified-datacenter offer (the prebuilt ``WORKER_IMAGE`` IS
the container), ships the shared instance bootstrap as the container command, and detects completion
purely from the worker's HF artifacts (no inbound network, no serverless queue, no VM to cloud-init —
unlike Lambda). It implements the SAME ``base.Provider`` interface as RunPod/Lambda, so the
orchestrator/allocator treat them interchangeably.

Vast is the live-market instance complement: like Lambda it is opt-in (available only when
``VAST_API_KEY`` is set), and it is capacity-aware (the allocator offers a class only when the
verified-datacenter market actually has a fitting offer right now).

``PROVIDER`` is the ``base.Provider`` implementation the registry hands out.
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

        # Opt-in live-market substrate: available only when its operator key is present. Without
        # VAST_API_KEY (tests / CI / RunPod-only operators) allocation degrades deterministically to
        # the other providers.
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
    ) -> PollResult:
        # ``on_last_gpu`` is accepted for the shared Provider interface (RunPod stretches its grace on
        # the last GPU); the instance providers use a UNIFORM per-GPU wait, so it is intentionally unused.
        from flash.providers.vast.jobs import submit_run_vast

        return submit_run_vast(
            spec,
            seed,
            log=log,
            on_handle=on_handle,
            attempt=attempt,
            runtime_secrets=runtime_secrets,
        )

    def poll(self, handle: JobHandle, spec, seed: int, *, log: Any = None) -> PollResult:
        import contextlib

        from flash.providers.runpod.jobs import make_hf_heartbeat_reader
        from flash.providers.vast import api as vast_api
        from flash.providers.vast.jobs import (
            PROVISION_GRACE_S,
            VastJobHandle,
            poll_vast_job,
        )

        hf_repo = spec.train.hf_repo
        prefix = f"{spec.phase}/{spec.run_id}/seed{seed}"
        reader = make_hf_heartbeat_reader(hf_repo, prefix) if hf_repo else None
        vh = VastJobHandle.from_dict(handle.to_dict())
        if log is not None:
            print(f"attaching: vast instance={vh.instance_id}", file=log, flush=True)
        # The wall-cap deadline counts from the instance's LAUNCH, not this reattach (Vast has no
        # server-side execution timeout, so resetting on every recovery would extend the billable
        # window unbounded). poll_vast_job anchors its deadline check to handle.started_ts.
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
            # Recovery (attach_run) has no submit_run_vast teardown ``finally``; destroy the reattached
            # instance here so a finished/abandoned recovered seed stops billing immediately.
            with contextlib.suppress(Exception):
                vast_api.destroy_instance(vh.instance_id)

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
        # ``destroy_instance`` returns False on a ``success: false`` / network breakdown — the box is
        # STILL billable. Dropping that bool (the prior behavior) let the best-effort callers log
        # "terminated …" and clear the handle while the instance kept billing, leaving only the slow
        # ``sweep_orphans`` backstop to notice. Surface it: warn here, then raise so the caller does not
        # record a FALSE success (the deploy/cancel callers catch it and still run their endpoint GC /
        # later sweep; a future non-suppressing caller sees the real failure instead of a clean return).
        if not vast_api.destroy_instance(int(iid)):
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
