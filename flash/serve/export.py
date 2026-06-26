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

# Repo files we never delete when re-exporting into an existing destination repo: these are user/repo
# furniture, not adapter artifacts. Everything else already in the repo that this export does NOT
# re-upload is treated as a stale leftover and removed (see ``_orphan_files``) — so the destination
# always mirrors exactly the exported adapter and can't serve a mix of old + new files (e.g. an old
# ``adapter_model.safetensors`` surviving next to a freshly-exported bin-only adapter, which the
# loader would prefer as stale weights).
_PRESERVE_ON_EXPORT = frozenset({"README.md", ".gitattributes", ".gitignore"})


def _orphan_files(api, dest_repo: str, uploaded: set[str]) -> list[str]:
    """Files already in ``dest_repo`` that this export is NOT replacing (stale leftovers).

    Computed as ``existing - uploaded`` so it is serialization-format-agnostic — it can never miss a
    new/sharded weight filename the way a fixed extension/pattern list would. Repo furniture
    (``_PRESERVE_ON_EXPORT``) and the HF cache dir are kept. Returns ``[]`` for a brand-new or
    not-yet-listable repo (nothing to clean)."""
    try:
        existing = set(api.list_repo_files(repo_id=dest_repo, repo_type="model"))
    except Exception:
        return []
    return sorted(
        f
        for f in existing - uploaded
        if f not in _PRESERVE_ON_EXPORT and not f.startswith(".cache/")
    )


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
        uploaded = {p.relative_to(adapter_dir).as_posix() for p in files}
        try:
            api.create_repo(
                repo_id=dest_repo, repo_type="model", private=private, exist_ok=True
            )
            # create_repo(exist_ok=True) does NOT change the visibility of a repo that already
            # exists, so a default-private export into a pre-existing PUBLIC repo would otherwise
            # stay public (and vice-versa). Enforce the requested visibility explicitly.
            api.update_repo_settings(repo_id=dest_repo, repo_type="model", private=private)
            # Mirror the adapter folder into the repo: upload this adapter's files and, in the SAME
            # atomic commit, delete any leftover files from a previous export so stale weights can't
            # linger (see _orphan_files). delete_patterns runs against the repo's pre-commit state,
            # and these are exact relative paths (valid fnmatch patterns).
            api.upload_folder(
                repo_id=dest_repo,
                repo_type="model",
                folder_path=str(adapter_dir),
                commit_message=f"Export Freesolo adapter ({source_subfolder})",
                delete_patterns=_orphan_files(api, dest_repo, uploaded),
            )
        except Exception as exc:
            raise ServingError(f"could not upload adapter to {dest_repo}: {exc}") from exc
    logger.info(
        "exported %s:%s -> %s (%d files)", source_repo, source_subfolder, dest_repo, len(files)
    )
    return f"https://huggingface.co/{dest_repo}"
