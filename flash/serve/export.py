"""Export a run's trained LoRA adapter into a user-owned HuggingFace repo.

Runs server-side: operator token reads the private source dataset repo, user token writes
the destination model repo, so the user never needs access to internal artifact storage.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from flash._logging import get_logger
from flash.serve.deploy import ServingError

logger = get_logger(__name__)

# Delete stale adapter artifacts on re-export so old weights can't linger beside new ones.
# Scoped to PEFT filenames only — never touch the user's unrelated repo files.
_STALE_ADAPTER_DELETE_PATTERNS = ["adapter_model*", "adapter_config.json"]


def _normalize_base_model(base_model: str | None) -> str | None:
    base_model = (base_model or "").strip()
    return base_model or None


def _rewrite_adapter_config_base_model(adapter_dir: Path, base_model: str) -> bool:
    path = adapter_dir / "adapter_config.json"
    try:
        config = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(config, dict):
        return False

    changed = config.get("base_model_name_or_path") != base_model
    config["base_model_name_or_path"] = base_model
    if "base_model" in config and config.get("base_model") != base_model:
        config["base_model"] = base_model
        changed = True
    if changed:
        path.write_text(json.dumps(config, indent=2) + "\n")
    return changed


def _rewrite_readme_base_model(adapter_dir: Path, base_model: str) -> bool:
    path = adapter_dir / "README.md"
    try:
        text = path.read_text()
    except OSError:
        return False
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return False
    end = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end = i
            break
    if end is None:
        return False

    metadata = lines[1:end]
    rewritten: list[str] = []
    found = False
    changed = False
    j = 0
    top_level_key = re.compile(r"^[A-Za-z0-9_-]+:\s*")
    while j < len(metadata):
        line = metadata[j]
        if line.startswith("base_model:"):
            found = True
            replacement = f"base_model: {base_model}\n"
            rewritten.append(replacement)
            changed = changed or line != replacement
            j += 1
            while j < len(metadata) and not top_level_key.match(metadata[j]):
                changed = True
                j += 1
            continue
        rewritten.append(line)
        j += 1

    if not found:
        rewritten.insert(0, f"base_model: {base_model}\n")
        changed = True
    if changed:
        path.write_text("".join([lines[0], *rewritten, *lines[end:]]))
    return changed


def _repair_export_metadata(adapter_dir: Path, base_model: str | None) -> None:
    base_model = _normalize_base_model(base_model)
    if not base_model:
        return
    changed = 0
    changed += int(_rewrite_adapter_config_base_model(adapter_dir, base_model))
    changed += int(_rewrite_readme_base_model(adapter_dir, base_model))
    if changed:
        logger.info("repaired exported adapter metadata base_model=%s", base_model)


def _hf_api():
    """Import huggingface_hub lazily (it's a server extra, not a base CLI dependency)."""
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ModuleNotFoundError as exc:
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
    base_model: str | None = None,
) -> str:
    """Copy adapter ``source_repo:{source_subfolder}`` into ``dest_repo`` and return its URL."""
    HfApi, snapshot_download = _hf_api()
    read_token = source_token or os.environ.get("HF_TOKEN")
    if not read_token:
        raise RuntimeError(
            "no operator HF token available to read the source adapter; set HF_TOKEN on the "
            "server (or pass source_token)"
        )
    with tempfile.TemporaryDirectory(prefix="flash-export-") as tmp:
        try:
            snapshot_download(
                repo_id=source_repo,
                repo_type="dataset",
                allow_patterns=[f"{source_subfolder}/*"],
                local_dir=tmp,
                token=read_token,
            )
        except OSError:
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
        names = {p.name for p in files}
        has_weight = any(n.startswith("adapter_model") for n in names)
        if not has_weight or "adapter_config.json" not in names:
            raise ValueError(
                f"no loadable LoRA adapter at {source_repo}:{source_subfolder} "
                "(need adapter_config.json + an adapter_model* weight; nothing to export)"
            )
        _repair_export_metadata(adapter_dir, base_model)
        api = HfApi(token=dest_token)
        try:
            # Always create private first so the repo is never transiently exposed empty/partial.
            api.create_repo(
                repo_id=dest_repo, repo_type="model", private=True, exist_ok=True
            )
            if private:
                api.update_repo_settings(repo_id=dest_repo, repo_type="model", private=True)
            # parent_commit guards against concurrent exports leaving mixed weights in the same dest repo.
            parent_commit = api.repo_info(repo_id=dest_repo, repo_type="model").sha
            api.upload_folder(
                repo_id=dest_repo,
                repo_type="model",
                folder_path=str(adapter_dir),
                commit_message=f"Export Freesolo adapter ({source_subfolder})",
                delete_patterns=_STALE_ADAPTER_DELETE_PATTERNS,
                parent_commit=parent_commit,
            )
            if not private:
                api.update_repo_settings(repo_id=dest_repo, repo_type="model", private=False)
        except OSError:
            raise
        except Exception as exc:
            raise ServingError(f"could not upload adapter to {dest_repo}: {exc}") from exc
    logger.info(
        "exported %s:%s -> %s (%d files)", source_repo, source_subfolder, dest_repo, len(files)
    )
    return f"https://huggingface.co/{dest_repo}"
