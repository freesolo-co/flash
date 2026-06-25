"""Azure Blob storage for managed environment packages.

The control plane stores each published environment's ``.tar.gz`` as a single blob and hands
workers a short-lived **read-only SAS URL** to download it — so the GPU worker needs no Azure
SDK or credentials, just an HTTP GET (see ``flash/envs/adapter``). Only the control plane (the
``server`` extra) imports ``azure-storage-blob``; imports are lazy so CPU tests and the client
don't require it.

Config (control-plane env):
  * ``FLASH_ENV_BLOB_CONNECTION_STRING`` — Azure Storage account connection string (contains the
    account key, which is required to mint SAS tokens).
  * ``FLASH_ENV_BLOB_CONTAINER`` — container name (default ``flash-environments``).
"""

from __future__ import annotations

import contextlib
import os
from datetime import UTC, datetime, timedelta

CONNECTION_STRING_ENV = "FLASH_ENV_BLOB_CONNECTION_STRING"
CONTAINER_ENV = "FLASH_ENV_BLOB_CONTAINER"
_DEFAULT_CONTAINER = "flash-environments"

# Default read-SAS lifetime. Generous on purpose: a run can sit IN_QUEUE for a long time waiting
# on GPU capacity before the worker downloads the env, and the SAS is resolved once at submit and
# threaded through the spec. 48h comfortably outlives realistic queue waits.
DEFAULT_SAS_TTL_SECONDS = 48 * 60 * 60


class AzureBlobNotConfigured(RuntimeError):
    """The control plane has no Azure Blob connection string configured."""


def container_name() -> str:
    return (os.environ.get(CONTAINER_ENV) or _DEFAULT_CONTAINER).strip() or _DEFAULT_CONTAINER


def _connection_string() -> str:
    conn = (os.environ.get(CONNECTION_STRING_ENV) or "").strip()
    if not conn:
        raise AzureBlobNotConfigured(
            f"{CONNECTION_STRING_ENV} is not set; the control plane cannot store environment "
            f"packages in Azure Blob storage"
        )
    return conn


def _service_client():  # type: ignore[no-untyped-def]
    # Lazy import: only the control plane installs azure-storage-blob.
    from azure.storage.blob import BlobServiceClient

    return BlobServiceClient.from_connection_string(_connection_string())


def upload_package(blob_key: str, data: bytes) -> None:
    """Upload (overwrite) a packaged environment tarball at ``blob_key`` in the env container."""
    client = _service_client().get_blob_client(container=container_name(), blob=blob_key)
    client.upload_blob(data, overwrite=True)


def read_sas_url(blob_key: str, *, ttl_seconds: int = DEFAULT_SAS_TTL_SECONDS) -> str:
    """Return a time-limited, read-only HTTPS URL the worker can GET to download the package."""
    from azure.storage.blob import BlobSasPermissions, generate_blob_sas

    service = _service_client()
    account_key = getattr(service.credential, "account_key", None)
    if not account_key:
        # SAS minting requires the shared account key (present in a standard connection string).
        raise AzureBlobNotConfigured(
            f"{CONNECTION_STRING_ENV} has no account key; cannot mint a read SAS URL for "
            f"environment packages"
        )
    container = container_name()
    sas = generate_blob_sas(
        account_name=service.account_name,
        container_name=container,
        blob_name=blob_key,
        account_key=account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.now(UTC) + timedelta(seconds=ttl_seconds),
    )
    blob_client = service.get_blob_client(container=container, blob=blob_key)
    return f"{blob_client.url}?{sas}"


def delete(blob_key: str) -> None:
    """Best-effort delete of a package blob (used by cleanup / re-publish paths)."""
    from azure.core.exceptions import ResourceNotFoundError

    client = _service_client().get_blob_client(container=container_name(), blob=blob_key)
    with contextlib.suppress(ResourceNotFoundError):
        client.delete_blob()


def ensure_container() -> None:
    """Create the env container if it does not exist (idempotent; for provisioning/backfill)."""
    from azure.core.exceptions import ResourceExistsError

    with contextlib.suppress(ResourceExistsError):
        _service_client().create_container(container_name())
