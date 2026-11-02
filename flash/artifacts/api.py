"""Run artifact operations backed exclusively by private Azure Blob Storage.

Repository IDs and revision descriptors remain stable across the cutover. The
``repo_type`` spelling is retained in persisted records; it is not a Hub backend.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

if __package__:
    from .config import open_container
    from .manifest import ArtifactNotFoundError, repo_prefix, validate_path
    from .storage import BlobArtifacts
else:
    from artifact_config import open_container
    from artifact_manifest import ArtifactNotFoundError, repo_prefix, validate_path
    from artifact_storage import BlobArtifacts


@dataclass(frozen=True)
class ArtifactFile:
    path: str
    size: int
    blob_id: str
    modified: float
    lfs: object = None

    @property
    def last_commit(self):
        return SimpleNamespace(date=datetime.fromtimestamp(self.modified, UTC))


def _matches(path, patterns):
    if isinstance(patterns, str):
        patterns = [patterns]
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns or [])


class ArtifactStore:
    def __init__(self, *, token=None):
        self.token = token

    def repo_info(self, repo_id, *, repo_type="dataset", revision=None, token=None):
        with open_container(token or self.token) as container:
            _, sha, _ = BlobArtifacts(container).manifest(repo_id, revision)
        return SimpleNamespace(sha=sha, oid=sha, private=True)

    def create_repo(self, repo_id, *, repo_type="dataset", exist_ok=True, private=True):
        if not private:
            raise ValueError("Flash artifact storage is private")
        with open_container(self.token) as container:
            # Containers are provisioned privately by the operator, never by a job.
            if container.get_container_properties().public_access:
                raise ValueError("Flash artifact container must deny public access")
            store = BlobArtifacts(container)
            try:
                _, revision, _ = store.manifest(repo_id)
                if not exist_ok:
                    raise FileExistsError("artifact repository already exists")
            except ArtifactNotFoundError:
                revision = store.commit(repo_id, {})
        return SimpleNamespace(oid=revision)

    def upload_file(
        self,
        *,
        path_or_fileobj,
        path_in_repo,
        repo_id,
        repo_type="dataset",
        commit_message=None,
        immutable=False,
    ):
        # immutable=True is reserved for finalized files that cannot change until this call returns.
        validate_path(path_in_repo)
        with tempfile.TemporaryDirectory(prefix="flash-artifact-upload-") as temporary:
            source = path_or_fileobj
            if not isinstance(source, (str, os.PathLike)):
                source = Path(temporary) / "file"
                with source.open("wb") as output:
                    if isinstance(path_or_fileobj, bytes):
                        output.write(path_or_fileobj)
                    else:
                        import shutil

                        shutil.copyfileobj(path_or_fileobj, output, length=1024 * 1024)
            with open_container(self.token) as container:
                store = BlobArtifacts(container)
                entry = {
                    **store.put_file(repo_id, source, immutable=immutable),
                    "modified": time.time(),
                }
                revision = store.commit(repo_id, {path_in_repo: entry})
        return SimpleNamespace(oid=revision)

    def upload_folder(
        self,
        *,
        folder_path,
        repo_id,
        path_in_repo="",
        repo_type="dataset",
        ignore_patterns=None,
        allow_patterns=None,
        delete_patterns=None,
        commit_message=None,
        parent_commit=None,
        immutable=False,
    ):
        # immutable=True requires the caller to keep every file stable for the entire publication.
        if path_in_repo:
            validate_path(path_in_repo)
        root = Path(folder_path).resolve()
        if not root.is_dir():
            raise FileNotFoundError("artifact upload folder is unavailable")
        prefix = path_in_repo + "/" if path_in_repo else ""
        with open_container(self.token) as container:
            store = BlobArtifacts(container)
            additions = {}
            for path in sorted(root.rglob("*")):
                if path.is_symlink():
                    raise ValueError("artifact uploads cannot contain symlinks")
                if not path.is_file():
                    continue
                relative = path.relative_to(root).as_posix()
                if _matches(relative, ignore_patterns) or (
                    allow_patterns and not _matches(relative, allow_patterns)
                ):
                    continue
                additions[prefix + relative] = {
                    **store.put_file(repo_id, path, immutable=immutable),
                    "modified": time.time(),
                }
            revision = store.commit(
                repo_id,
                additions,
                expected_revision=parent_commit,
                delete=lambda path: (
                    path.startswith(prefix) and _matches(path[len(prefix) :], delete_patterns)
                ),
            )
        return SimpleNamespace(oid=revision)

    def list_repo_tree(
        self,
        repo_id,
        *,
        path_in_repo="",
        revision=None,
        recursive=False,
        repo_type="dataset",
        expand=False,
        token=None,
    ):
        if path_in_repo:
            validate_path(path_in_repo)
        prefix = path_in_repo + "/" if path_in_repo else ""
        with open_container(token or self.token) as container:
            files, _, _ = BlobArtifacts(container).manifest(repo_id, revision)
        return [
            ArtifactFile(
                path, entry["size"], entry.get("source_blob_id", entry["sha256"]), entry["modified"]
            )
            for path, entry in sorted(files.items())
            if path.startswith(prefix) and (recursive or "/" not in path[len(prefix) :])
        ]

    def list_repo_files(self, repo_id, *, revision=None, repo_type="dataset", token=None):
        return [
            entry.path
            for entry in self.list_repo_tree(
                repo_id, revision=revision, recursive=True, token=token
            )
        ]

    def file_exists(self, repo_id, filename, *, repo_type="dataset", revision=None):
        validate_path(filename)
        try:
            with open_container(self.token) as container:
                files, _, _ = BlobArtifacts(container).manifest(repo_id, revision)
                return filename in files
        except ArtifactNotFoundError:
            return False

    def delete_folder(self, *, repo_id, path_in_repo, repo_type="dataset"):
        validate_path(path_in_repo)
        with open_container(self.token) as container:
            revision = BlobArtifacts(container).commit(
                repo_id, {}, delete=lambda path: path.startswith(path_in_repo + "/")
            )
        return SimpleNamespace(oid=revision)

    def delete_file(self, *, repo_id, path_in_repo, repo_type="dataset"):
        validate_path(path_in_repo)
        with open_container(self.token) as container:
            revision = BlobArtifacts(container).commit(
                repo_id, {}, delete=lambda path: path == path_in_repo
            )
        return SimpleNamespace(oid=revision)

    def list_datasets(self, *, author):
        validate_path(author)
        with open_container(self.token) as container:
            return [
                SimpleNamespace(id=blob.name[len("repos/") : -len("/HEAD")])
                for blob in container.list_blobs(name_starts_with=f"repos/{author}/")
                if blob.name.endswith("/HEAD")
            ]


def _destination(repo, revision, cache_dir):
    cache = Path(cache_dir or Path.home() / ".cache" / "flash" / "artifacts")
    return cache / hashlib.sha256(repo.encode()).hexdigest() / revision


def download_file(
    repo_id,
    filename,
    *,
    revision=None,
    local_dir=None,
    cache_dir=None,
    token=None,
    repo_type="dataset",
    force_download=False,
):
    validate_path(filename)
    with open_container(token) as container:
        store = BlobArtifacts(container)
        files, revision, _ = store.manifest(repo_id, revision)
        if filename not in files:
            raise ArtifactNotFoundError("artifact file is unavailable")
        destination = local_dir or _destination(repo_id, revision, cache_dir)
        return store.download(repo_id, filename, files[filename], destination)


def download_snapshot(
    repo_id,
    *,
    revision=None,
    local_dir=None,
    cache_dir=None,
    token=None,
    repo_type="dataset",
    allow_patterns=None,
    ignore_patterns=None,
):
    repo_prefix(repo_id)
    with open_container(token) as container:
        store = BlobArtifacts(container)
        files, revision, _ = store.manifest(repo_id, revision)
        destination = Path(local_dir or _destination(repo_id, revision, cache_dir))
        destination.mkdir(parents=True, exist_ok=True)
        for path, entry in files.items():
            if _matches(path, ignore_patterns) or (
                allow_patterns and not _matches(path, allow_patterns)
            ):
                continue
            store.download(repo_id, path, entry, destination)
    return str(destination)
