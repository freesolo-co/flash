"""Stream content to Azure, then atomically publish its revision with an ETag CAS."""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

if __package__:
    from .manifest import (
        MAX_MANIFEST_BYTES,
        ArtifactNotFoundError,
        decode_manifest,
        encode_manifest,
        repo_prefix,
        revision_for,
        validate_path,
        validate_revision,
    )
else:
    from artifact_manifest import (
        MAX_MANIFEST_BYTES,
        ArtifactNotFoundError,
        decode_manifest,
        encode_manifest,
        repo_prefix,
        revision_for,
        validate_path,
        validate_revision,
    )


class BlobArtifacts:
    def __init__(self, container):
        self.container = container

    def read(self, name, limit):
        from azure.core.exceptions import ResourceNotFoundError

        try:
            download = self.container.get_blob_client(name).download_blob()
            if download.size > limit:
                raise ValueError("artifact metadata exceeds size limit")
            data = bytearray()
            for chunk in download.chunks():
                data.extend(chunk)
                if len(data) > limit:
                    raise ValueError("artifact metadata exceeds size limit")
            return bytes(data), download.properties.etag
        except ResourceNotFoundError:
            raise ArtifactNotFoundError("artifact or revision is unavailable") from None

    def manifest(self, repo, revision=None):
        prefix = repo_prefix(repo)
        etag = None
        if not revision or revision == "main":
            head, etag = self.read(f"{prefix}/HEAD", 40)
            revision = head.decode("ascii")
        validate_revision(revision)
        try:
            raw, _ = self.read(f"{prefix}/revisions/{revision}.json", MAX_MANIFEST_BYTES)
        except ArtifactNotFoundError:
            alias, _ = self.read(f"{prefix}/aliases/{revision}", 40)
            revision = validate_revision(alias.decode("ascii"))
            raw, _ = self.read(f"{prefix}/revisions/{revision}.json", MAX_MANIFEST_BYTES)
        if revision_for(raw) != revision:
            raise ValueError("artifact manifest checksum mismatch")
        return decode_manifest(raw, repo), revision, etag

    def immutable(self, name, data):
        from azure.core.exceptions import ResourceExistsError

        try:
            self.container.get_blob_client(name).upload_blob(data, overwrite=False)
        except ResourceExistsError:
            existing, _ = self.read(name, len(data))
            if existing != data:
                raise ValueError("immutable artifact revision collision") from None

    def put_file(self, repo, source, *, immutable=False):
        from azure.core.exceptions import ResourceExistsError

        # Finalized checkpoint callers guarantee stable files for the entire upload. Stream
        # those from the original descriptor; a second multi-GB copy can exhaust worker disk.
        # Mutable artifacts (including live logs) still hash and send one private snapshot.
        with contextlib.ExitStack() as stack:
            stream = stack.enter_context(open(source, "rb"))
            if not immutable:
                snapshot = stack.enter_context(tempfile.TemporaryFile())
                shutil.copyfileobj(stream, snapshot, length=1024 * 1024)
                stream = snapshot
                stream.seek(0)
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            size = os.fstat(stream.fileno()).st_size
            stream.seek(0)
            name = f"{repo_prefix(repo)}/objects/{digest}"
            try:
                self.container.get_blob_client(name).upload_blob(
                    stream,
                    length=size,
                    overwrite=False,
                    max_concurrency=4,
                    metadata={"sha256": digest},
                )
            except ResourceExistsError:
                props = self.container.get_blob_client(name).get_blob_properties()
                if props.size != size or props.metadata.get("sha256") != digest:
                    raise ValueError("immutable artifact object collision") from None
        return {"sha256": digest, "size": size}

    def commit(self, repo, additions, *, delete=None, expected_revision=None):
        from azure.core import MatchConditions
        from azure.core.exceptions import ResourceExistsError, ResourceModifiedError

        prefix = repo_prefix(repo)
        for _ in range(12):
            try:
                head, etag = self.read(f"{prefix}/HEAD", 40)
            except ArtifactNotFoundError:
                files, current, etag = {}, None, None
            else:
                current = validate_revision(head.decode("ascii"))
                # A missing/corrupt referenced revision is not an empty repository.
                files, _, _ = self.manifest(repo, current)
            if expected_revision is not None and expected_revision != current:
                raise RuntimeError("artifact head changed concurrently")
            if delete is not None:
                files = {path: entry for path, entry in files.items() if not delete(path)}
            files.update(additions)
            raw = encode_manifest(repo, files)
            revision = revision_for(raw)
            self.immutable(f"{prefix}/revisions/{revision}.json", raw)
            options = (
                {"etag": etag, "match_condition": MatchConditions.IfNotModified} if etag else {}
            )
            try:
                self.container.get_blob_client(f"{prefix}/HEAD").upload_blob(
                    revision.encode(), overwrite=bool(etag), **options
                )
                return revision
            except (ResourceExistsError, ResourceModifiedError):
                continue
        raise RuntimeError("artifact head remained busy; retry publication")

    def download(self, repo, path, entry, destination):
        validate_path(path)
        root = Path(destination).resolve()
        target = root / path
        if not target.resolve().is_relative_to(root):
            raise ValueError("artifact download escapes its destination")
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".artifact-", dir=target.parent)
        try:
            digest = hashlib.sha256()
            size = 0
            blob = f"{repo_prefix(repo)}/objects/{entry['sha256']}"
            with os.fdopen(fd, "wb") as stream:
                download = self.container.get_blob_client(blob).download_blob()
                for chunk in download.chunks():
                    size += len(chunk)
                    if size > entry["size"]:
                        raise ValueError("artifact size mismatch")
                    digest.update(chunk)
                    stream.write(chunk)
            if size != entry["size"] or digest.hexdigest() != entry["sha256"]:
                raise ValueError("artifact checksum mismatch")
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return str(target)
