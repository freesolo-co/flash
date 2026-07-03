"""Vast.ai provider: verified-datacenter single-GPU containers (REST only).

Opt-in live-market substrate (available only when ``VAST_API_KEY`` is set); detects completion
from the worker's HF artifacts. Implements the shared ``base.Provider`` interface, so the
allocator treats it interchangeably with RunPod/Lambda.
"""

from __future__ import annotations

from typing import Any

from flash.providers._instance_provider import InstanceProvider
from flash.providers.base import (
    AllocationConstraints,
    Candidate,
    CapacityLookupError,
    JobHandle,
    PollResult,
    Provider,
)


class VastProvider(InstanceProvider):
    """``base.Provider`` for the Vast.ai verified-datacenter substrate."""

    name = "vast"
    _gpu_identity_attr = "vast_name"

    @property
    def _handle_cls(self) -> type[JobHandle]:
        from flash.providers.vast.jobs import VastJobHandle

        return VastJobHandle

    def _load_api_key(self) -> Any:
        from flash.providers.vast.auth import load_api_key

        return load_api_key()

    def _missing_credentials(self, require_hf: bool) -> list[str]:
        from flash.providers.vast.preflight import missing_credentials

        return missing_credentials(require_hf=require_hf)

    def _hourly_rate(self, gpu: str) -> float:
        from flash.providers.vast.pricing import hourly_rate

        return hourly_rate(gpu)

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
    ) -> PollResult:
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

    def _poll_job(
        self,
        handle: JobHandle,
        spec,
        seed: int,
        *,
        log: Any,
        heartbeat_reader: Any,
        deadline_s: float,
    ) -> PollResult:
        from flash.providers.vast.jobs import poll_vast_job

        return poll_vast_job(
            handle,
            spec,
            seed,
            log=log,
            heartbeat_reader=heartbeat_reader,
            deadline_s=deadline_s,
        )

    def _reattach_deadline(self, spec) -> float:
        from flash.providers.vast.jobs import PROVISION_GRACE_S

        return max(60, int(spec.gpu.max_wall_seconds)) + PROVISION_GRACE_S

    def _teardown_reattached(self, handle: JobHandle, spec) -> None:
        from flash.providers.vast.jobs import _best_effort_destroy, destroy_run_instances

        if not _best_effort_destroy(handle.instance_id, context="poll recovery teardown"):
            # Unconfirmed teardown: the active-run sweep shields this label, so escalate to a
            # run-scoped reap by label (re-lists + retries, not active-shielded), mirroring the
            # submit_run_vast teardown finally.
            destroy_run_instances(spec.run_id)

    def _gc(self, run_id: str) -> None:
        from flash.providers.vast.jobs import destroy_run_instances

        destroy_run_instances(run_id)

    def _sweep_orphans(
        self,
        *,
        active_labels,
        known_labels,
    ) -> list[int]:
        from flash.providers.vast.jobs import sweep_orphans

        return sweep_orphans(active_labels=active_labels, known_labels=known_labels)

    def live_candidates(
        self, need_vram_gb: int, constraints: AllocationConstraints
    ) -> list[Candidate]:
        """Vast's fitting classes that currently have a LIVE verified-datacenter offer, priced live.

        Capacity-aware like Lambda: a Vast class with no fitting offer on the market right now is EXCLUDED,
        so the allocator never hands the runner a Vast class that would immediately fail to rent. ONE market
        search covers every class (offers carry their own gpu_name -> class), so we search once at the
        smallest fitting class's VRAM and bucket the returned offers by class. A capacity-lookup failure
        (market/API blip) raises ``CapacityLookupError`` -> ``allocate`` degrades to the other providers,
        failing the run retryably (not terminally) only if Vast was the sole fitting source.

        ``constraints.disk_gb`` and ``constraints.max_wall_seconds`` are the run's requested disk and wall
        cap; the Vast package prices against the SAME effective disk/duration floors the submit path
        provisions with, so a high-disk or long run isn't advertised capacity it couldn't actually rent (an
        impossible attempt a max_retries=0 run never escapes).
        """
        from flash.providers.vast.pricing import live_candidate_rates

        fitting = [g for g in self.gpu_classes() if g.vram_gb >= need_vram_gb]
        if not fitting:
            return []
        try:
            # Search once at the smallest fitting class's VRAM; the market covers every class at/above it.
            rates = live_candidate_rates(
                min(g.vram_gb for g in fitting),
                constraints.disk_gb,
                constraints.max_wall_seconds,
            )
        except Exception as exc:
            # Transient market/API blip -> signal allocate() (see CapacityLookupError): a Vast-only run must
            # infra-retry the outage, not terminally fail as if no GPU fit.
            raise CapacityLookupError("vast live capacity lookup failed") from exc
        fitting_names = {g.name: g.vram_gb for g in fitting}
        return [
            Candidate("vast", name, rate, fitting_names[name])
            for name, rate in rates.items()
            if name in fitting_names
        ]

    def cancel(self, handle: JobHandle) -> None:
        from flash.providers.vast.jobs import cancel

        cancel(handle.to_dict())

    def destroy(self, handle: JobHandle) -> None:
        from flash.providers.vast import api as vast_api
        from flash.providers.vast.jobs import _best_effort_destroy

        iid = handle.to_dict().get("instance_id")
        if not iid:
            return
        # Reuse the canonical best-effort teardown (it warns on an unconfirmed success:false/breakdown and
        # passes iid through unconverted), but RE-RAISE on failure so this suppress-wrapped path falls back
        # to sweep_orphans rather than recording a false success.
        if not _best_effort_destroy(iid, context="provider.destroy"):
            raise vast_api.VastApiError(
                f"vast destroy_instance({iid}) unconfirmed (success:false); instance may still bill"
            )

    def run_instances_remaining(self, run_id: str) -> list[int]:
        """Instance ids still carrying ``run_id``'s label after ``gc``. Empty == confirmed clear;
        non-empty == a possibly-live instance survives. RAISES on a listing failure so the caller can't
        mistake "couldn't list" for "clear" (``gc`` returns an empty list, not an error, on an
        unconfirmed DELETE)."""
        from flash.providers.vast.jobs import run_instances_remaining

        return run_instances_remaining(run_id)


PROVIDER: Provider = VastProvider()
