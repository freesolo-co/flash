"""Account-scoped RunPod network-volume reconciliation for managed Pod caches."""

from __future__ import annotations

from flash.providers.runpod import api as runpod_api


def weight_cache_datacenters(
    fingerprint: str,
    *,
    deadline_at: float,
) -> list[str]:
    """Return storage-capable data centers visible to one exact account."""
    return runpod_api.list_storage_datacenters_for_fingerprint(
        fingerprint,
        deadline_at=deadline_at,
    )


def weight_cache_volume_name(base: str, data_center_id: str) -> str:
    """Return the physical cache-volume name for one data center."""
    return f"{base}-{data_center_id.lower()}"


def ensure_account_volume(
    fingerprint: str,
    *,
    base: str,
    data_center_id: str,
    size_gb: int,
    deadline_at: float,
):
    """Return one exact account and data-center volume after strict reconciliation."""
    if data_center_id not in weight_cache_datacenters(fingerprint, deadline_at=deadline_at):
        raise runpod_api.RunpodApiError(
            f"runpod data center {data_center_id} does not support account network volumes"
        )
    name = weight_cache_volume_name(base, data_center_id)
    volumes = runpod_api.list_network_volumes_for_fingerprint(
        fingerprint,
        deadline_at=deadline_at,
    )
    matches = [
        volume
        for volume in volumes
        if volume.name == name and volume.data_center_id == data_center_id
    ]
    if len(matches) > 1:
        raise runpod_api.RunpodApiError(
            f"runpod network volume {name} is duplicated in {data_center_id}"
        )
    if not matches:
        try:
            volume = runpod_api.create_network_volume_for_fingerprint(
                fingerprint,
                name=name,
                size_gb=size_gb,
                data_center_id=data_center_id,
                deadline_at=deadline_at,
            )
        except runpod_api.RunpodMutationAmbiguous:
            refreshed = runpod_api.list_network_volumes_for_fingerprint(
                fingerprint,
                deadline_at=deadline_at,
            )
            reconciled = [
                item
                for item in refreshed
                if item.name == name and item.data_center_id == data_center_id
            ]
            if len(reconciled) != 1:
                from flash.providers.base import UnreconciledCreateError

                raise UnreconciledCreateError(
                    f"RunPod network volume creation could not be reconciled in {data_center_id}"
                ) from None
            volume = reconciled[0]
    else:
        volume = matches[0]
    if volume.size_gb >= size_gb:
        return volume
    runpod_api.grow_network_volumes_for_fingerprint(
        fingerprint,
        {name: size_gb},
        deadline_at=deadline_at,
    )
    refreshed = runpod_api.list_network_volumes_for_fingerprint(
        fingerprint,
        deadline_at=deadline_at,
    )
    grown = [
        item for item in refreshed if item.name == name and item.data_center_id == data_center_id
    ]
    if len(grown) != 1 or grown[0].size_gb < size_gb:
        raise runpod_api.RunpodApiError(f"runpod network volume {name} did not reach {size_gb} GB")
    return grown[0]
