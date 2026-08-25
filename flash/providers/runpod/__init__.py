"""RunPod Secure Cloud Pod provider for managed training."""

from __future__ import annotations

import time
from typing import Any

from flash.providers._lifecycle.instance import InstanceJobHandle
from flash.providers._lifecycle.provider import InstanceProvider
from flash.providers.base import (
    AllocationConstraints,
    Candidate,
    JobHandle,
    PollResult,
    Provider,
    rentable_gpu_counts,
)


def terminate_persisted_endpoints(spec: Any, run_id: str) -> None:
    """Compatibility name for raw-spec recovery; the managed resource is now a Pod."""
    from flash.providers.runpod.pods import destroy_run_pods

    destroy_run_pods(run_id)


class RunpodProvider(InstanceProvider):
    name = "runpod"
    _gpu_identity_attr = "runpod_gpu_type_id"
    # runpod accepts a card count directly; its static class table defines rentable shapes.
    live_capacity = False
    supports_weight_cache = True

    @property
    def _handle_cls(self) -> type[InstanceJobHandle]:
        from flash.providers.runpod.pods import RunpodPodHandle

        return RunpodPodHandle

    def is_configured(self) -> bool:
        from flash.providers.runpod.auth import keys

        return bool(keys())

    def _load_api_key(self) -> Any:
        from flash.providers.runpod.auth import load_api_key

        return load_api_key()

    def _missing_credentials(self, require_hf: bool) -> list[str]:
        from flash.providers.runpod.preflight import missing_credentials

        return missing_credentials(require_hf=require_hf)

    def _hourly_rate(self, gpu: str) -> float:
        from flash.providers.runpod.pricing import hourly_rate

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
        source_snapshot: dict | None,
        deadline_at: float | None,
    ) -> PollResult:
        from flash.providers.runpod.pods import submit_runpod_pod

        return submit_runpod_pod(
            spec,
            seed,
            log=log,
            on_handle=on_handle,
            attempt=attempt,
            runtime_secrets=runtime_secrets,
            source_snapshot=source_snapshot,
            deadline_at=deadline_at,
        )

    def _poll_job(
        self,
        handle: JobHandle,
        spec,
        seed: int,
        *,
        log: Any,
        heartbeat_reader: Any,
        deadline_at: float | None,
    ) -> PollResult:
        from flash.providers.runpod.pods import poll_runpod_pod

        return poll_runpod_pod(
            handle,
            spec,
            seed,
            log=log,
            heartbeat_reader=heartbeat_reader,
            deadline_at=deadline_at,
        )

    def _teardown_reattached(self, handle: JobHandle, spec) -> None:
        from flash.providers.runpod.pods import terminate_handle

        terminate_handle(handle, deadline_at=time.time() + 120.0)

    def poll(
        self,
        handle: JobHandle,
        spec,
        seed: int,
        *,
        log: Any = None,
        _deadline_at: float | None = None,
    ) -> PollResult:
        """Poll a recovered Pod and require complete Pod plus secret teardown."""
        from flash.core.spec import require_matching_seed
        from flash.providers.artifacts.hf import heartbeat_reader_for

        seed = require_matching_seed(spec, seed)
        reader = heartbeat_reader_for(spec, deadline_at=_deadline_at)
        strict = self._handle_cls.from_dict(handle.to_dict())
        if strict.pending:
            raise ValueError(
                "pending RunPod Pod handles must be resolved and persisted before polling"
            )
        if log is not None:
            print(f"attaching: runpod instance={strict.instance_id}", file=log, flush=True)
        try:
            result = self._poll_job(
                strict,
                spec,
                seed,
                log=log,
                heartbeat_reader=reader,
                deadline_at=_deadline_at,
            )
        except BaseException:
            try:
                self._teardown_reattached(strict, spec)
            except Exception:
                from flash.runner import _record_cleanup_remote

                _record_cleanup_remote(spec.run_id, strict.to_dict())
            raise
        try:
            self._teardown_reattached(strict, spec)
        except Exception:
            from flash.runner import _record_cleanup_remote

            if not _record_cleanup_remote(spec.run_id, strict.to_dict()):
                raise RuntimeError("runpod cleanup target could not be persisted") from None
        return result

    def _gc(self, run_id: str) -> None:
        from flash.providers.runpod.pods import destroy_run_pods

        destroy_run_pods(run_id)

    def _sweep_orphans(self, *, active_labels, known_labels) -> list[str]:
        from flash.providers.runpod.pods import sweep_orphan_pods

        return sweep_orphan_pods(active_labels=active_labels, known_labels=known_labels)

    def live_candidates(
        self, need_vram_gb: int, constraints: AllocationConstraints
    ) -> list[Candidate]:
        return [
            Candidate("runpod", gpu.name, self.hourly_rate(gpu.name), gpu.vram_gb, count)
            for gpu in self.gpu_classes()
            if gpu.vram_gb >= need_vram_gb and gpu.validated
            for count in rentable_gpu_counts(constraints.max_gpu_count)
        ]

    def cancel(self, handle: JobHandle) -> None:
        from flash.providers.runpod.pods import terminate_handle

        strict = self._handle_cls.from_dict(handle.to_dict())
        terminate_handle(strict, deadline_at=time.time() + 120.0)

    destroy = cancel

    def run_instances_remaining(self, run_id: str) -> list[str]:
        from flash.providers.runpod.pods import run_pods_remaining

        return run_pods_remaining(run_id)


PROVIDER: Provider = RunpodProvider()
