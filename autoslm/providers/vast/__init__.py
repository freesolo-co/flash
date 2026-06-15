"""Vast.ai provider: verified-datacenter single-GPU instances (REST only).

The Vast substrate rents a single-GPU instance from a verified-datacenter offer, ships
a self-contained bootstrap through the onstart script, and detects completion purely
from the worker's HF artifacts (no inbound network, no serverless queue). It implements
the SAME ``base.Provider`` interface behind the SAME module layout as RunPod, so the
orchestrator/allocator treat the two interchangeably.

``PROVIDER`` is the ``base.Provider`` implementation the registry hands out.
"""

from __future__ import annotations

import os
from typing import Any

from autoslm.providers.base import GpuClass, JobHandle, PollResult, Provider


class VastProvider:
    """``base.Provider`` for the Vast.ai verified-datacenter substrate."""

    name = "vast"

    def is_configured(self) -> bool:
        from autoslm.providers.vast.auth import load_api_key

        # Vast needs its operator key AND a live network path: it is a live-market
        # substrate (offer search), so AUTOSLM_SKIP_NET (offline/CI) disables Vast
        # entirely; offline allocation then degrades deterministically to RunPod's
        # static catalog.
        if os.environ.get("AUTOSLM_SKIP_NET"):
            return False
        return load_api_key() is not None

    def preflight(self, require_hf: bool = True) -> list[str]:
        from autoslm.providers.vast.preflight import missing_credentials

        return missing_credentials(require_hf=require_hf)

    def gpu_classes(self) -> list[GpuClass]:
        from autoslm.providers.vast.gpus import gpu_classes

        return gpu_classes()

    def hourly_rate(self, gpu: str) -> float:
        from autoslm.providers.vast.pricing import hourly_rate

        return hourly_rate(gpu)

    def submit_run(
        self,
        spec,
        seed: int,
        *,
        log: Any = None,
        on_handle: Any = None,
        attempt: int = 0,
        offers: Any = None,
        exclude_machine_ids: Any = frozenset(),
    ) -> PollResult:
        from autoslm.providers.vast.jobs import submit_run_vast

        return submit_run_vast(
            spec,
            seed,
            log=log,
            on_handle=on_handle,
            attempt=attempt,
            offers=offers,
            exclude_machine_ids=exclude_machine_ids,
        )

    def poll(self, handle: JobHandle, spec, seed: int, *, log: Any = None) -> PollResult:
        from autoslm.providers.runpod.jobs import make_hf_heartbeat_reader
        from autoslm.providers.vast.jobs import VastJobHandle, poll_vast_job

        hf_repo = spec.train.hf_repo
        prefix = f"{spec.phase}/{spec.run_id}/seed{seed}"
        reader = make_hf_heartbeat_reader(hf_repo, prefix) if hf_repo else None
        vh = VastJobHandle.from_dict(handle.to_dict())
        if log is not None:
            print(f"attaching: vast instance={vh.instance_id}", file=log, flush=True)
        # Reattach must apply the SAME stall tuning + wall-cap deadline as submit_run_vast
        # (see jobs.py), mirroring RunPod's reattach (runpod/__init__.py). Vast has no
        # server-side execution timeout, so a recovered run that dropped the client-side
        # deadline could bill unbounded.
        stall = float(os.environ.get("AUTOSLM_STALL_AFTER_S", "1500"))
        deadline = max(60, int(spec.gpu.max_wall_seconds)) + 1800
        return poll_vast_job(
            vh,
            spec,
            seed,
            log=log,
            heartbeat_reader=reader,
            stall_after_s=stall,
            deadline_s=deadline,
        )

    def cancel(self, handle: JobHandle) -> None:
        from autoslm.providers.vast.jobs import cancel

        cancel(handle.to_dict())

    def destroy(self, handle: JobHandle) -> None:
        from autoslm.providers.vast import api as vast_api

        d = handle.to_dict()
        if d.get("instance_id"):
            vast_api.destroy_instance(int(d["instance_id"]))

    def gc(self, spec) -> None:
        from autoslm.providers.vast.jobs import destroy_run_instances

        destroy_run_instances(spec.run_id)

    def sweep_orphans(self, active_labels: set[str] | None = None) -> list[int]:
        """Vast-only crash-recovery sweep (called via the provider object at startup)."""
        from autoslm.providers.vast.jobs import sweep_orphans

        return sweep_orphans(active_labels=active_labels)


PROVIDER: Provider = VastProvider()
