"""Bounded upload concurrency through Hugging Face's public commit API."""

from __future__ import annotations

import os


def upload_threads() -> int:
    """Allow image operators to tune parallel LFS files without unbounded thread pools."""
    value = os.environ.get("FLASH_HF_UPLOAD_THREADS", "8")
    try:
        count = int(value)
    except ValueError:
        raise ValueError("FLASH_HF_UPLOAD_THREADS must be an integer from 1 to 16") from None
    if not 1 <= count <= 16:
        raise ValueError("FLASH_HF_UPLOAD_THREADS must be an integer from 1 to 16")
    return count


def upload_api(*, token: str | None):
    # Keep this import lazy: worker startup disables disk-hungry Xet staging before Hub imports.
    from huggingface_hub import HfApi

    threads = upload_threads()

    class UploadApi(HfApi):
        def create_commit(self, *args, **kwargs):
            # upload_folder delegates to this public method. Preserve its atomic commit,
            # delete-pattern filtering and retries rather than uploading each shard separately.
            kwargs.setdefault("num_threads", threads)
            return super().create_commit(*args, **kwargs)

    return UploadApi(token=token)
