"""Modal GPU training provider backed by strictly pinned Sandboxes."""

from __future__ import annotations

from typing import Any

from flash.providers._lifecycle.instance import InstanceJobHandle
from flash.providers._lifecycle.provider import InstanceProvider
from flash.providers.base import (
    AllocationConstraints,
    Candidate,
    JobHandle,
    PollResult,
    Provider,
    combined_vram_gb,
    rentable_gpu_counts,
)


class ModalProvider(InstanceProvider):
    """``base.Provider`` for Modal training Sandboxes."""

    name = "modal"
    _gpu_identity_attr = "modal_name"
    live_capacity = False

    @property
    def _handle_cls(self) -> type[InstanceJobHandle]:
        from flash.providers.modal.jobs import ModalJobHandle

        return ModalJobHandle

    def _load_api_key(self) -> Any:
        from flash.providers.modal.auth import load_credentials

        return load_credentials()

    def _missing_credentials(self, require_hf: bool) -> list[str]:
        from flash.providers.modal.preflight import missing_credentials

        return missing_credentials(require_hf=require_hf)

    def _hourly_rate(self, gpu: str) -> float:
        from flash.providers.modal.pricing import hourly_rate

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
        from flash.providers.modal.jobs import submit_run_modal

        return submit_run_modal(
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
        from flash.providers.modal.jobs import poll_modal_job

        return poll_modal_job(
            handle,
            spec,
            seed,
            log=log,
            heartbeat_reader=heartbeat_reader,
            deadline_at=deadline_at,
        )

    def _teardown_reattached(self, handle: JobHandle, spec) -> None:
        from flash.providers.modal.api import terminate_sandbox

        terminate_sandbox(str(handle.instance_id))

    def _gc(self, run_id: str) -> None:
        from flash.providers.modal.jobs import terminate_run_sandboxes

        terminate_run_sandboxes(run_id)

    def _sweep_orphans(self, *, active_labels, known_labels) -> list[str]:
        from flash.providers.modal.jobs import sweep_orphans

        return sweep_orphans(active_labels=active_labels, known_labels=known_labels)

    def live_candidates(
        self, need_vram_gb: int, constraints: AllocationConstraints
    ) -> list[Candidate]:
        """Return every fitting Modal shape without issuing a fake capacity query.

        Modal exposes no scarcity signal and the verified Sandbox substrate accepts these classes as
        always-available requests. ``live_capacity`` is therefore false: an empty result means no
        supported shape fits the constraints, not a transient capacity miss that should be retried.
        """
        out: list[Candidate] = []
        for gpu in self.gpu_classes():
            if constraints.gpu_type and gpu.name != constraints.gpu_type:
                continue
            if gpu.vram_gb < need_vram_gb:
                continue
            max_count = min(constraints.max_gpu_count, 4 if gpu.name == "A10" else 8)
            for count in rentable_gpu_counts(max_count):
                if (
                    constraints.required_vram_gb
                    and combined_vram_gb(gpu.vram_gb, count) < constraints.required_vram_gb
                ):
                    continue
                out.append(
                    Candidate("modal", gpu.name, self.hourly_rate(gpu.name), gpu.vram_gb, count)
                )
        return out

    def cancel(self, handle: JobHandle) -> None:
        from flash.providers.modal.jobs import cancel

        strict = self._handle_cls.from_dict(handle.to_dict())
        cancel(strict.to_dict())

    destroy = cancel

    def run_instances_remaining(self, run_id: str) -> list[str]:
        from flash.providers.modal.jobs import run_instances_remaining

        return run_instances_remaining(run_id)


PROVIDER: Provider = ModalProvider()
