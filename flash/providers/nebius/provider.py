"""Nebius-only Spot training provider."""

from flash.providers.core.base import GPU_INFO, Candidate
from flash.providers.nebius.config import configured
from flash.providers.nebius.handles import SpotJobHandle, require_scope
from flash.providers.nebius.pricing import SPOT_GPUS, hourly_rate


class SpotProvider:
    name = "spot"
    supports_weight_cache = False
    live_capacity = True

    def is_configured(self):
        return configured()

    def preflight(self, require_hf=True):
        from flash.artifacts.config import configured as artifacts_configured

        problems = []
        if not self.is_configured():
            problems.append(
                "Configure the Nebius CLI and FLASH_NEBIUS_PROJECT_ID/REGION/SUBNET_ID/IMAGE_ID"
            )
        if require_hf and not artifacts_configured():
            problems.append("Configure private Azure artifact storage")
        return problems

    def gpu_classes(self):
        return [GPU_INFO[name] for name in SPOT_GPUS]

    def hourly_rate(self, gpu):
        return hourly_rate(gpu)

    def live_candidates(self, need_vram_gb, constraints):
        from flash.providers.nebius.config import settings
        from flash.providers.nebius.prices import affordable_gpus

        prices = affordable_gpus(settings()["region"])
        return [
            Candidate(self.name, name, prices[name], GPU_INFO[name].vram_gb, count)
            for name, shape in SPOT_GPUS.items()
            if name in prices and GPU_INFO[name].vram_gb >= need_vram_gb
            for count in shape.counts
            if count <= constraints.max_gpu_count
        ]

    def submit_attempt(
        self,
        spec,
        *,
        log=None,
        on_handle=None,
        attempt=0,
        runtime_secrets=None,
        on_last_gpu=False,
        source_snapshot=None,
        _deadline_at=None,
    ):
        from flash.providers.nebius.jobs import submit_attempt

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
        from flash.providers.nebius.jobs import poll

        return poll(handle, spec, log=log, deadline_at=_deadline_at)

    def destroy(self, handle):
        from flash.providers.nebius.api import terminate

        h = SpotJobHandle.from_dict(handle.to_dict())
        require_scope(h)
        terminate(h.instance_id)

    def cancel(self, handle):
        self.destroy(handle)

    def gc(self, spec):
        self.terminate_run(spec.run_id)

    def terminate_run(self, run_id):
        from flash.providers.nebius import api

        for instance in api.list_instances():
            if instance["metadata"]["labels"].get("flash-run") == run_id:
                api.terminate(instance["metadata"]["id"])

    def run_instances_remaining(self, run_id):
        from flash.providers.nebius import api

        return [
            item["metadata"]["id"]
            for item in api.list_instances()
            if item["metadata"]["labels"].get("flash-run") == run_id
        ]

    def sweep_orphans(self, active_labels=None, known_labels=None):
        from flash.providers.nebius import api

        instances = api.list_instances()
        active = active_labels() if callable(active_labels) else active_labels
        known = known_labels() if callable(known_labels) else known_labels
        deleted = []
        for item in instances:
            label = item["metadata"]["labels"].get("flash-run", "")
            if (known is not None and label not in known) or (active and label in active):
                continue
            api.terminate(item["metadata"]["id"])
            deleted.append(item["metadata"]["id"])
        return deleted


PROVIDER = SpotProvider()


def terminate_persisted_resources(spec, run_id):
    """Reconcile a launch even when its public job spec can no longer be parsed."""
    PROVIDER.terminate_run(run_id)
