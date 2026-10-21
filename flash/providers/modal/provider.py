"""Modal-only GPU training provider."""

from flash.providers.core.base import GPU_INFO, Candidate, rentable_gpu_counts
from flash.providers.modal.auth import credentials
from flash.providers.modal.handles import ModalJobHandle, require_scope
from flash.providers.modal.pricing import hourly_rate


class ModalProvider:
    name = "modal"
    supports_weight_cache = True

    def is_configured(self):
        return credentials() is not None

    def preflight(self, require_hf=True):
        import os

        problems = []
        if not self.is_configured():
            problems.append("MODAL_TOKEN_ID and MODAL_TOKEN_SECRET are required")
        if require_hf and not os.environ.get("HF_TOKEN", "").strip():
            problems.append("HF_TOKEN is required")
        return problems

    def gpu_classes(self):
        return list(GPU_INFO.values())

    def hourly_rate(self, gpu):
        return hourly_rate(gpu)

    def live_candidates(self, need_vram_gb, constraints):
        if constraints.disk_gb > 512:
            raise ValueError("Modal Sandboxes support at most 512 GiB ephemeral disk")
        # Modal queues GPU reservations; there is no public live inventory API.
        return [
            Candidate(self.name, g.name, hourly_rate(g.name), g.vram_gb, count)
            for g in GPU_INFO.values()
            if g.validated and g.vram_gb >= need_vram_gb
            for count in rentable_gpu_counts(min(g.max_count, constraints.max_gpu_count))
        ]

    def submit_attempt(
        self,
        spec,
        *,
        log=None,
        on_handle=None,
        attempt=0,
        runtime_secrets=None,
        source_snapshot=None,
        _deadline_at=None,
    ):
        from flash.providers.modal.jobs import submit_attempt

        return submit_attempt(
            spec,
            log=log,
            on_handle=on_handle,
            attempt=attempt,
            runtime_secrets=runtime_secrets,
            source_snapshot=source_snapshot,
            deadline_at=_deadline_at,
        )

    def poll_attempt(self, handle, spec, *, log=None, _deadline_at=None):
        from flash.providers.modal.jobs import poll

        return poll(handle, spec, log=log, deadline_at=_deadline_at)

    def destroy(self, handle):
        from flash.providers.modal.api import terminate

        h = ModalJobHandle.from_dict(handle.to_dict())
        require_scope(h)
        terminate(h.instance_id)
        if h.cpu_uploads:
            from flash.providers.modal.checkpoint_upload import cleanup

            cleanup(h)

    def cancel(self, handle):
        self.destroy(handle)

    def gc(self, spec):
        self.terminate_run(spec.run_id)

    def terminate_run(self, run_id):
        from flash.providers.modal import api

        for instance in api.list_instances():
            if instance["tags"].get("flash_run") == run_id:
                api.terminate(instance["id"])

    def run_instances_remaining(self, run_id):
        from flash.providers.modal import api

        return [
            item["id"] for item in api.list_instances() if item["tags"].get("flash_run") == run_id
        ]

    def sweep_orphans(self, active_labels=None, known_labels=None):
        from flash.providers.modal import api

        instances = api.list_instances()
        active = active_labels() if callable(active_labels) else active_labels
        known = known_labels() if callable(known_labels) else known_labels
        deleted = []
        for item in instances:
            label = item["tags"].get("flash_run", "")
            if (known is not None and label not in known) or (active and label in active):
                continue
            api.terminate(item["id"])
            deleted.append(item["id"])
        return deleted


PROVIDER = ModalProvider()


def terminate_persisted_resources(spec, run_id):
    """Reconcile a launch even when its public job spec can no longer be parsed."""
    PROVIDER.terminate_run(run_id)
