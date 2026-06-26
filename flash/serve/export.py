"""Export a run's trained LoRA adapter into a user-owned HuggingFace repo.

``flash export`` copies a trained adapter out of the managed platform and into a repo the
user owns. The adapter artifacts live in the run's ``[train].hf_repo`` — a PRIVATE HF *dataset*
repo created with the operator token (see ``flash.providers.runpod.train``) — so only the
control plane can read them. This runs SERVER-side for exactly that reason: it downloads the
adapter folder with the operator ``HF_TOKEN`` and re-uploads it to the destination repo with
the USER's token (which only needs write access to their own repo). The user therefore never
needs read access to the internal source repo.

It mirrors the deploy seam: the same ``adapter_prefix`` / ``checkpoint_adapter_prefix`` locates
the source subfolder, so ``flash export --step N`` exports the exact per-step checkpoint that
``flash deploy --step N`` would serve.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from flash._logging import get_logger
from flash.serve.deploy import ServingError

logger = get_logger(__name__)

# Stale-artifact patterns cleared on (re-)export so a previous, DIFFERENT adapter in the destination
# repo can't leave weights behind: upload_folder only overwrites matching paths, so an old
# ``adapter_model.safetensors`` could otherwise survive next to a freshly-exported bin-only adapter and
# be loaded as stale weights. SCOPED to the LoRA adapter's own filenames (``adapter_model.*`` covers
# the .safetensors/.bin weight files; ``adapter_config.json`` is its config) — deliberately NOT broad
# globs like ``*.safetensors`` / ``*.bin``, which in a reused destination repo could delete unrelated
# base-model weights (``model.safetensors`` / ``pytorch_model.bin``) or other user files.
_STALE_ADAPTER_PATTERNS = [
    "adapter_model.*",
    "adapter_config.json",
]


def _hf_api():
    """Import huggingface_hub lazily (it's a server extra, not a base CLI dependency)."""
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ModuleNotFoundError as exc:  # pragma: no cover - the server always has the extra
        raise ServingError(
            "huggingface_hub is required to export adapters; install the server extras "
            "(pip install 'flash[server]')"
        ) from exc
    return HfApi, snapshot_download


def export_adapter(
    *,
    source_repo: str,
    source_subfolder: str,
    dest_repo: str,
    dest_token: str,
    source_token: str | None = None,
    private: bool = True,
) -> str:
    """Copy adapter ``source_repo:{source_subfolder}`` into ``dest_repo`` and return its URL.

    The source is a private HF *dataset* repo (the run's artifact store), read with the operator
    token (``source_token`` or ``HF_TOKEN``). The destination is a HF *model* repo owned by the
    user, written with ``dest_token``; it is created (private unless ``private=False``) when it
    does not already exist. The copy is a download-then-upload so the bytes pass through the
    control plane and the user never needs read access to the internal source.

    Raises :class:`ValueError` when the source has no adapter artifacts (nothing to export) and
    :class:`~flash.serve.deploy.ServingError` on an HF transport/permission failure, so the route
    can map them to a clean 404 / 502.
    """
    HfApi, snapshot_download = _hf_api()
    read_token = source_token or os.environ.get("HF_TOKEN")
    with tempfile.TemporaryDirectory(prefix="flash-export-") as tmp:
        try:
            # Pull ONLY this adapter's folder (not the whole multi-run/-checkpoint dataset repo).
            # fnmatch's ``*`` spans ``/``, so a single trailing-glob pattern also captures any
            # nested files under the adapter folder.
            snapshot_download(
                repo_id=source_repo,
                repo_type="dataset",
                allow_patterns=[f"{source_subfolder}/*"],
                local_dir=tmp,
                token=read_token,
            )
        except Exception as exc:
            raise ServingError(
                f"could not download adapter {source_repo}:{source_subfolder}: {exc}"
            ) from exc
        adapter_dir = Path(tmp) / source_subfolder
        files = (
            sorted(p for p in adapter_dir.rglob("*") if p.is_file())
            if adapter_dir.is_dir()
            else []
        )
        if not files:
            raise ValueError(
                f"no adapter artifacts found at {source_repo}:{source_subfolder} "
                "(nothing to export)"
            )
        api = HfApi(token=dest_token)
        try:
            api.create_repo(
                repo_id=dest_repo, repo_type="model", private=private, exist_ok=True
            )
            # create_repo(exist_ok=True) does NOT change the visibility of a repo that already
            # exists, so a default-private export into a pre-existing PUBLIC repo would otherwise
            # stay public (and vice-versa). Enforce the requested visibility explicitly.
            api.update_repo_settings(repo_id=dest_repo, repo_type="model", private=private)
            api.upload_folder(
                repo_id=dest_repo,
                repo_type="model",
                folder_path=str(adapter_dir),
                commit_message=f"Export Freesolo adapter ({source_subfolder})",
                # Clear stale weights/metadata a previous adapter left in this repo (see
                # _STALE_ADAPTER_PATTERNS) so a re-export can't serve a mix of old + new files.
                delete_patterns=_STALE_ADAPTER_PATTERNS,
            )
        except Exception as exc:
            raise ServingError(f"could not upload adapter to {dest_repo}: {exc}") from exc
    logger.info(
        "exported %s:%s -> %s (%d files)", source_repo, source_subfolder, dest_repo, len(files)
    )
    return f"https://huggingface.co/{dest_repo}"
