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

# Stale ADAPTER artifacts to delete (in the same commit as the upload) when re-exporting into an
# existing repo, so an old weight file can't linger next to a freshly-exported one — e.g. a previous
# ``adapter_model.safetensors`` surviving next to a new ``.bin``, which the loader would then prefer as
# stale weights. Scoped to PEFT adapter filenames ONLY (NEVER the user's unrelated repo files): the
# destination is a user-owned repo that may legitimately hold other content, so a full ``existing -
# uploaded`` mirror would delete files this export has no business touching. Static patterns (no
# ``list_repo_files`` call), so a listing failure can't silently skip the cleanup either. The
# ``adapter_model*`` glob covers single-file, sharded, and ``.index.json`` variants regardless of
# extension; HF evaluates ``delete_patterns`` against the repo's pre-commit state, and the freshly
# uploaded files in the SAME commit win, so a re-export of the same format is a no-op delete.
_STALE_ADAPTER_DELETE_PATTERNS = ["adapter_model*", "adapter_config.json"]


def _hf_api():
    """Import huggingface_hub lazily (it's a server extra, not a base CLI dependency)."""
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ModuleNotFoundError as exc:  # pragma: no cover - the server always has the extra
        # A missing server extra is an internal MISCONFIGURATION, not an upstream gateway/transport
        # failure — raise a plain RuntimeError (NOT ServingError, which the route maps to 502) so it
        # surfaces as a 500. ServingError is reserved for real HF download/upload failures.
        raise RuntimeError(
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
            # Always create/ensure the repo PRIVATE first: a brand-new public repo would otherwise be
            # exposed (empty, then partial) for the whole upload, and a failed upload would leave an
            # empty PUBLIC repo behind. Visibility is enforced AFTER the upload succeeds (below).
            api.create_repo(
                repo_id=dest_repo, repo_type="model", private=True, exist_ok=True
            )
            # Upload this adapter's files and, in the SAME atomic commit, delete only the STALE ADAPTER
            # artifacts of a prior export (see _STALE_ADAPTER_DELETE_PATTERNS) so old weights can't
            # linger — without touching the user's unrelated repo files. delete_patterns runs against
            # the repo's pre-commit state; the files uploaded in this commit win, so re-exporting the
            # same format is a no-op delete.
            api.upload_folder(
                repo_id=dest_repo,
                repo_type="model",
                folder_path=str(adapter_dir),
                commit_message=f"Export Freesolo adapter ({source_subfolder})",
                delete_patterns=_STALE_ADAPTER_DELETE_PATTERNS,
            )
            # Now that the adapter is committed, enforce the requested visibility. We created/ensured
            # the repo private above and create_repo(exist_ok=True) never changes an existing repo's
            # visibility, so this single call covers both a public export (exposed only now that it has
            # content) and a re-export into a pre-existing repo whose visibility differs from the
            # request.
            api.update_repo_settings(repo_id=dest_repo, repo_type="model", private=private)
        except Exception as exc:
            raise ServingError(f"could not upload adapter to {dest_repo}: {exc}") from exc
    logger.info(
        "exported %s:%s -> %s (%d files)", source_repo, source_subfolder, dest_repo, len(files)
    )
    return f"https://huggingface.co/{dest_repo}"
