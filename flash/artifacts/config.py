"""Azure credentials stay separate from optional upstream model credentials."""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from urllib.parse import urlsplit

ENV_KEYS = (
    "FLASH_ARTIFACT_BLOB_ACCOUNT_URL",
    "FLASH_ARTIFACT_BLOB_CONTAINER",
    "FLASH_ARTIFACT_BLOB_CONNECTION_STRING",
    "FLASH_ARTIFACT_BLOB_SAS_URL",
)


def configured() -> bool:
    return any(os.environ.get(key, "").strip() for key in ENV_KEYS if key != ENV_KEYS[1])


def validate_sas_url(value: str) -> str:
    if not re.fullmatch(r"https://[a-z0-9]{3,24}\.blob\.core\.windows\.net/[a-z0-9-]+\?.+", value):
        raise ValueError("artifact credential must be an Azure container SAS URL")
    parsed = urlsplit(value)
    expected_container = os.environ.get("FLASH_ARTIFACT_BLOB_CONTAINER", "").strip()
    expected_account = os.environ.get("FLASH_ARTIFACT_BLOB_ACCOUNT_URL", "").strip().rstrip("/")
    if (expected_container and parsed.path != "/" + expected_container) or (
        expected_account and f"{parsed.scheme}://{parsed.netloc}" != expected_account
    ):
        raise ValueError(
            "artifact SAS destination does not match the configured Azure account/container"
        )
    return value


def runtime_environment() -> dict[str, str]:
    """Give a 24-hour sandbox a bounded container capability, never an account key."""
    from datetime import UTC, datetime, timedelta

    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import (
        BlobServiceClient,
        ContainerSasPermissions,
        generate_container_sas,
    )

    supplied = os.environ.get("FLASH_ARTIFACT_BLOB_SAS_URL", "").strip()
    if supplied:
        return {"FLASH_ARTIFACT_BLOB_SAS_URL": validate_sas_url(supplied)}
    if not configured():
        return {}
    start = datetime.now(UTC) - timedelta(minutes=5)
    expiry = start + timedelta(hours=27)
    container = os.environ.get("FLASH_ARTIFACT_BLOB_CONTAINER", "artifacts").strip()
    connection = os.environ.get("FLASH_ARTIFACT_BLOB_CONNECTION_STRING", "").strip()
    permissions = ContainerSasPermissions(read=True, write=True, create=True, list=True)
    if connection:
        with BlobServiceClient.from_connection_string(connection) as service:
            sas = generate_container_sas(
                service.account_name,
                container,
                account_key=service.credential.account_key,
                permission=permissions,
                start=start,
                expiry=expiry,
                protocol="https",
            )
            url = service.get_container_client(container).url
    else:
        url = os.environ.get("FLASH_ARTIFACT_BLOB_ACCOUNT_URL", "").strip().rstrip("/")
        if not re.fullmatch(r"https://[a-z0-9]{3,24}\.blob\.core\.windows\.net", url):
            raise ValueError("invalid Azure artifact account URL")
        with DefaultAzureCredential() as credential, BlobServiceClient(url, credential) as service:
            key = service.get_user_delegation_key(start, expiry)
            sas = generate_container_sas(
                service.account_name,
                container,
                user_delegation_key=key,
                permission=permissions,
                start=start,
                expiry=expiry,
                protocol="https",
            )
            url = service.get_container_client(container).url
    return {"FLASH_ARTIFACT_BLOB_SAS_URL": validate_sas_url(f"{url}?{sas}")}


@contextmanager
def open_container(token: str | None = None):
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import ContainerClient

    container = os.environ.get("FLASH_ARTIFACT_BLOB_CONTAINER", "artifacts").strip()
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{1,61})[a-z0-9]", container):
        raise ValueError("invalid FLASH_ARTIFACT_BLOB_CONTAINER")
    options = {
        "connection_timeout": 10,
        "read_timeout": 60,
        "retry_total": 2,
        "logging_enable": False,
    }
    sas_url = token or os.environ.get("FLASH_ARTIFACT_BLOB_SAS_URL", "").strip()
    connection = os.environ.get("FLASH_ARTIFACT_BLOB_CONNECTION_STRING", "").strip()
    if sas_url:
        with ContainerClient.from_container_url(validate_sas_url(sas_url), **options) as client:
            yield client
    elif connection:
        with ContainerClient.from_connection_string(connection, container, **options) as client:
            yield client
    else:
        url = os.environ.get("FLASH_ARTIFACT_BLOB_ACCOUNT_URL", "").strip().rstrip("/")
        if not re.fullmatch(r"https://[a-z0-9]{3,24}\.blob\.core\.windows\.net", url):
            raise ValueError(
                "FLASH_ARTIFACT_BLOB_ACCOUNT_URL must name an HTTPS Azure Blob account"
            )
        with (
            DefaultAzureCredential() as credential,
            ContainerClient(url, container, credential=credential, **options) as client,
        ):
            yield client
