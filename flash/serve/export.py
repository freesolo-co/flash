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
    except ModuleNotFoundError as exc:
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

    Raises :class:`ValueError` when the source has no adapter artifacts (nothing to export),
    :class:`~flash.serve.deploy.ServingError` on an HF transport/permission failure (route -> 502),
    and :class:`RuntimeError` when no operator read token is configured — an internal
    misconfiguration the route surfaces as a 500, not a 502.
    """
    HfApi, snapshot_download = _hf_api()
    read_token = source_token or os.environ.get("HF_TOKEN")
    if not read_token:
        # The operator token to READ the private source artifact repo is an internal server invariant
        # (the control plane always supplies HF_TOKEN). Its absence is a MISCONFIGURATION, not an
        # upstream HF transport/permission failure — raise a plain RuntimeError (the route maps it to a
        # 500, like _hf_api's missing-extra case) NOT ServingError, which becomes a 502 and looks like
        # an upstream gateway issue. Surfacing it HERE, before snapshot_download, also keeps the real
        # cause from being buried under a generic token=None auth error wrapped as ServingError.
        raise RuntimeError(
            "no operator HF token available to read the source adapter; set HF_TOKEN on the "
            "server (or pass source_token)"
        )
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
        except OSError:
            # A LOCAL filesystem/tempdir failure (e.g. disk full writing the snapshot) is an INTERNAL
            # error, not an upstream HF transport/permission failure — let it propagate as a 500 rather
            # than wrapping it in ServingError (which the route maps to a misleading 502 "upstream").
            raise
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
        # A LOADABLE LoRA needs BOTH adapter_config.json AND an adapter_model* weight. A source folder
        # with only tokenizer/config files (the same non-deployable shape checkpoint-publishing filters
        # out) must NOT be "exported": the upload's delete_patterns would clear any prior destination
        # weights and commit a repo that can't load the adapter while reporting success. Reject up front.
        names = {p.name for p in files}
        has_weight = any(n.startswith("adapter_model") for n in names)
        if not has_weight or "adapter_config.json" not in names:
            raise ValueError(
                f"no loadable LoRA adapter at {source_repo}:{source_subfolder} "
                "(need adapter_config.json + an adapter_model* weight; nothing to export)"
            )
        api = HfApi(token=dest_token)
        try:
            # Always create/ensure the repo PRIVATE first: a brand-new public repo would otherwise be
            # exposed (empty, then partial) for the whole upload, and a failed upload would leave an
            # empty PUBLIC repo behind.
            api.create_repo(
                repo_id=dest_repo, repo_type="model", private=True, exist_ok=True
            )
            # If the caller wants PRIVATE, enforce it BEFORE the upload: create_repo(exist_ok=True)
            # won't change a PRE-EXISTING public repo's visibility, so without this the new weights
            # would commit while that repo is still public. A PUBLIC export instead defers the flip to
            # AFTER the upload (below) so an empty/partial repo is never exposed.
            if private:
                api.update_repo_settings(repo_id=dest_repo, repo_type="model", private=True)
            # Upload this adapter's files and, in the SAME atomic commit, delete only the STALE ADAPTER
            # artifacts of a prior export (see _STALE_ADAPTER_DELETE_PATTERNS) so old weights can't
            # linger — without touching the user's unrelated repo files. delete_patterns runs against
            # the repo's pre-commit state; the files uploaded in this commit win, so re-exporting the
            # same format is a no-op delete.
            #
            # Concurrent-export guard: pass the repo's CURRENT head as parent_commit. Our
            # delete_patterns were computed against this tree, so if a racing export to the SAME dest
            # commits in between, HF rejects ours (412) instead of layering on a now-stale base and
            # possibly leaving MIXED weights (an old .bin our delete didn't catch + the new
            # .safetensors). A rejected race surfaces as a ServingError the caller can retry — far
            # better than a false success with a repo loaders can't reliably resolve.
            parent_commit = api.repo_info(repo_id=dest_repo, repo_type="model").sha
            api.upload_folder(
                repo_id=dest_repo,
                repo_type="model",
                folder_path=str(adapter_dir),
                commit_message=f"Export Freesolo adapter ({source_subfolder})",
                delete_patterns=_STALE_ADAPTER_DELETE_PATTERNS,
                parent_commit=parent_commit,
            )
            # PUBLIC export: flip to public only now that the adapter is committed, so the repo is never
            # exposed empty/partial.
            if not private:
                api.update_repo_settings(repo_id=dest_repo, repo_type="model", private=False)
        except OSError:
            # Local I/O failure reading the staged adapter dir is INTERNAL (-> 500), not an upstream HF
            # error (-> 502). HF transport/permission failures are not OSError, so they still wrap below.
            raise
        except Exception as exc:
            raise ServingError(f"could not upload adapter to {dest_repo}: {exc}") from exc
    logger.info(
        "exported %s:%s -> %s (%d files)", source_repo, source_subfolder, dest_repo, len(files)
    )
    return f"https://huggingface.co/{dest_repo}"
