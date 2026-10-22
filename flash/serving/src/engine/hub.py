"""Fetch immutable adapter directories without scanning unrelated training artifacts."""

import math
import re
from pathlib import Path, PurePosixPath


def download_adapter(
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    local_dir: str,
    token: str | None,
    subfolder: str | None,
) -> str:
    from huggingface_hub import HfApi, RepoFile, hf_hub_download, snapshot_download

    args = {"repo_id": repo_id, "repo_type": repo_type, "revision": revision, "token": token}
    if not subfolder:
        return snapshot_download(**args, local_dir=local_dir)
    prefix = PurePosixPath(subfolder)
    if prefix.is_absolute() or ".." in prefix.parts:
        raise ValueError("adapter subfolder must stay inside its repository")
    # snapshot_download(allow_patterns=...) still enumerates the entire repository.
    # Training datasets can contain thousands of native checkpoint shards; scope the
    # tree request itself, retaining the same immutable SHA for every file download.
    for entry in HfApi().list_repo_tree(**args, path_in_repo=subfolder, recursive=True):
        if not isinstance(entry, RepoFile):
            continue
        path = PurePosixPath(entry.path)
        if path.is_absolute() or ".." in path.parts or not path.is_relative_to(prefix):
            raise ValueError("Hub returned a file outside the adapter directory")
        target = Path(local_dir) / entry.path
        if not target.resolve().is_relative_to(Path(local_dir).resolve()):
            raise ValueError("adapter download escapes its exact-SHA source cache")
        hf_hub_download(**args, filename=entry.path, local_dir=local_dir)
    return local_dir


def download_retry_delay(error: Exception, attempt: int) -> float:
    """Honor Hub throttling within a bounded five-minute retry window."""
    fallback = 0.5 * (2**attempt)
    response = getattr(error, "response", None)
    if response is None or response.status_code != 429:
        return fallback
    headers = response.headers
    reset = re.search(r"(?:^|;)\s*t=(\d+)", headers.get("ratelimit", ""))
    candidates = [headers.get("retry-after", ""), reset.group(1) if reset else ""]
    delays = []
    for value in candidates:
        try:
            delay = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(delay) and delay >= 0:
            delays.append(delay)
    return min(300.0, max(fallback, max(delays, default=30.0) + 1.0))
