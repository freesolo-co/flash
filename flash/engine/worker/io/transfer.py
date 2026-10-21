"""Bounded upload concurrency through Hugging Face's public commit API."""

from __future__ import annotations


def upload_api(*, token: str | None):
    # Keep this import lazy: worker startup disables disk-hungry Xet staging before Hub imports.
    from huggingface_hub import HfApi

    class UploadApi(HfApi):
        def create_commit(self, *args, **kwargs):
            # upload_folder delegates to this public method. Preserve its atomic commit,
            # delete-pattern filtering and retries rather than uploading each shard separately.
            kwargs.setdefault("num_threads", 8)
            return super().create_commit(*args, **kwargs)

    return UploadApi(token=token)
