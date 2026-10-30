"""Immutable environment archives and atomic named heads in Azure Blob Storage.

The archive is written before its revision record, and the head is committed last.
Readers never observe a partially uploaded revision. Deletion removes only the head;
already pinned jobs can continue to read their immutable package.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass

from flash.envs.meta.identity import _is_safe_github_path_parts, is_commit_sha

MAX_PACKAGE_BYTES = 256 * 1024 * 1024
MAX_RECORD_BYTES = 4096


class RegistryError(RuntimeError):
    def __init__(self, message: str, *, status: int = 502):
        super().__init__(message)
        self.status = status


def enabled() -> bool:
    mode = os.environ.get("FLASH_ENV_STORE", "github").strip()
    if mode not in {"github", "azure"}:
        raise RegistryError("FLASH_ENV_STORE must be github or azure", status=503)
    return mode == "azure"


def validate_slug(slug: str) -> str:
    parts = slug.split("/")
    if len(parts) != 3 or not _is_safe_github_path_parts(parts):
        raise RegistryError("environment id must be namespace/project/name", status=400)
    return slug


@dataclass(frozen=True)
class Revision:
    slug: str
    revision: str
    sha256: str

    def encode(self) -> bytes:
        return json.dumps(vars(self), sort_keys=True, separators=(",", ":")).encode()

    @classmethod
    def decode(cls, data: bytes, slug: str, revision: str = "") -> Revision:
        try:
            obj = json.loads(data)
            record = cls(**obj)
            if (
                record.slug != slug
                or not is_commit_sha(record.revision)
                or record.revision != record.revision.lower()
                or not re.fullmatch(r"[0-9a-f]{64}", record.sha256)
                or (revision and record.revision != revision)
            ):
                raise ValueError("inconsistent revision")
        except (ValueError, TypeError, KeyError) as exc:
            raise RegistryError("invalid environment registry revision") from exc
        return record


class BlobRegistry:
    def __init__(self, container, *, deadline_at: float | None = None):
        self.container = container
        self.deadline_at = deadline_at

    def _budget(self) -> dict:
        remaining = 60.0 if self.deadline_at is None else self.deadline_at - time.time()
        if remaining <= 0:
            raise RegistryError("environment registry deadline exceeded", status=503)
        # Bound each I/O as well as checking the overall deadline between operations.
        return {"connection_timeout": min(10.0, remaining), "read_timeout": min(60.0, remaining)}

    def _read(self, name: str, limit: int) -> tuple[bytes, str]:
        download = self.container.get_blob_client(name).download_blob(**self._budget())
        if download.size > limit:
            raise RegistryError("environment registry object exceeds size limit")
        data = bytearray()
        for chunk in download.chunks():
            self._budget()
            if len(data) + len(chunk) > limit:
                raise RegistryError("environment registry object exceeds size limit")
            data.extend(chunk)
        return bytes(data), download.properties.etag

    def resolve(self, slug: str, revision: str = "") -> Revision:
        validate_slug(slug)
        if revision and not is_commit_sha(revision):
            raise RegistryError("invalid immutable environment revision", status=400)
        name = f"revisions/{slug}/{revision}.json" if revision else f"heads/{slug}.json"
        data, _ = self._read(name, MAX_RECORD_BYTES)
        return Revision.decode(data, slug, revision)

    def download(self, record: Revision) -> bytes:
        validate_slug(record.slug)
        data, _ = self._read(f"archives/{record.slug}/{record.sha256}.tar.gz", MAX_PACKAGE_BYTES)
        if hashlib.sha256(data).hexdigest() != record.sha256:
            raise RegistryError("environment archive checksum mismatch")
        return data

    def _immutable(self, name: str, data: bytes) -> None:
        from azure.core.exceptions import ResourceExistsError

        try:
            self.container.get_blob_client(name).upload_blob(
                data, overwrite=False, **self._budget()
            )
        except ResourceExistsError:
            existing, _ = self._read(name, len(data))
            if existing != data:
                raise RegistryError(
                    "immutable environment object already has different content"
                ) from None

    def publish(
        self, slug: str, package: bytes, *, create_only: bool = False, source_revision: str = ""
    ) -> Revision:
        from azure.core import MatchConditions
        from azure.core.exceptions import ResourceNotFoundError

        validate_slug(slug)
        if not package or len(package) > MAX_PACKAGE_BYTES:
            raise RegistryError("environment package is empty or too large", status=413)
        if source_revision and not is_commit_sha(source_revision):
            raise RegistryError("invalid source revision", status=400)
        name = f"heads/{slug}.json"
        try:
            head, etag = self._read(name, MAX_RECORD_BYTES)
        except ResourceNotFoundError:
            head, etag = b"", ""
        if create_only and head:
            return Revision.decode(head, slug)
        digest = hashlib.sha256(package).hexdigest()
        # Preserve the existing 40-hex wire format; integrity uses the FULL SHA-256.
        record = Revision(slug, digest[:40], digest)
        self._immutable(f"archives/{slug}/{digest}.tar.gz", package)
        self._immutable(f"revisions/{slug}/{record.revision}.json", record.encode())
        if source_revision:
            alias = Revision(slug, source_revision.lower(), digest)
            self._immutable(f"revisions/{slug}/{alias.revision}.json", alias.encode())
        options = {"etag": etag, "match_condition": MatchConditions.IfNotModified} if etag else {}
        self.container.get_blob_client(name).upload_blob(
            record.encode(), overwrite=bool(etag), **options, **self._budget()
        )
        return record

    def delete(self, slug: str) -> bool:
        from azure.core import MatchConditions
        from azure.core.exceptions import ResourceNotFoundError

        name = f"heads/{validate_slug(slug)}.json"
        try:
            _, etag = self._read(name, MAX_RECORD_BYTES)
        except ResourceNotFoundError:
            return False
        try:
            self.container.get_blob_client(name).delete_blob(
                etag=etag, match_condition=MatchConditions.IfNotModified, **self._budget()
            )
        except ResourceNotFoundError:
            return False
        return True

    def list(self, namespace: str) -> list[str]:
        if not _is_safe_github_path_parts((namespace,)):
            raise RegistryError("invalid environment namespace", status=400)
        slugs = []
        for blob in self.container.list_blobs(
            name_starts_with=f"heads/{namespace}/", **self._budget()
        ):
            self._budget()
            if not blob.name.endswith(".json"):
                raise RegistryError("invalid environment registry head")
            slugs.append(validate_slug(blob.name[len("heads/") : -len(".json")]))
        return sorted(slugs)


@contextmanager
def open_registry(*, deadline_at: float | None = None):
    """Use Entra credentials (managed identity in production, Azure CLI locally)."""
    from azure.core.exceptions import AzureError
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import ContainerClient

    url = os.environ.get("FLASH_ENV_BLOB_ACCOUNT_URL", "").strip().rstrip("/")
    container = os.environ.get("FLASH_ENV_BLOB_CONTAINER", "environments").strip()
    if not re.fullmatch(r"https://[a-z0-9]{3,24}\.blob\.core\.windows\.net", url):
        raise RegistryError(
            "FLASH_ENV_BLOB_ACCOUNT_URL must name an HTTPS Azure Blob account", status=503
        )
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{1,61})[a-z0-9]", container):
        raise RegistryError("invalid FLASH_ENV_BLOB_CONTAINER", status=503)
    try:
        with (
            DefaultAzureCredential() as credential,
            ContainerClient(
                url,
                container,
                credential=credential,
                retry_total=0 if deadline_at is not None else 2,
                connection_timeout=10,
                read_timeout=60,
                max_single_get_size=1024 * 1024,
                max_chunk_get_size=1024 * 1024,
                logging_enable=False,
            ) as client,
        ):
            yield BlobRegistry(client, deadline_at=deadline_at)
    except AzureError as exc:
        status = getattr(exc, "status_code", None)
        if status == 404:
            raise RegistryError("environment package or revision not found", status=404) from exc
        if status in {409, 412}:
            raise RegistryError(
                "environment changed concurrently; retry the operation", status=409
            ) from exc
        # Do not expose SDK messages: they can contain URLs, credentials or account details.
        raise RegistryError("Azure environment storage is unavailable", status=503) from exc
