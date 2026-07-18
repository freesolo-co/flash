"""Export a run's trained LoRA adapter into a user-owned HuggingFace repo.

Runs server-side: operator token reads the private source dataset repo, user token writes
the destination model repo, so the user never needs access to internal artifact storage.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import struct
import tempfile
from pathlib import Path

from flash._logging import get_logger
from flash.serve.deploy import ServingError

logger = get_logger(__name__)

# Delete stale adapter artifacts on re-export so old weights can't linger beside new ones.
# Scoped to PEFT filenames only — never touch the user's unrelated repo files.
_STALE_ADAPTER_DELETE_PATTERNS = ["adapter_model*", "adapter_config.json"]
_TEMP_MERGED_BASE_MODEL_RE = re.compile(
    r"(?:/[^\s\"'`,\]\){}]+)*/flash_sft_merged_[^\s\"'`,\]\){}]+"
)
_MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024

# module-path segments outside the language model; mirrors _VL_EXCLUDE_SEGMENTS in
# flash/engine/worker/lora.py. today's managed adapters never touch these (the worker
# excludes them from lora), so their presence means a genuinely multimodal adapter whose
# vision keys the text-only strip cannot represent. normalizing only the lm keys of such
# an adapter would produce a mixed namespace that matches no transformers class, so the
# normalization must skip the file entirely and export it as trained.
_NON_LM_KEY_SEGMENTS = (".visual.", ".vision_tower.", ".multi_modal_projector.", ".mtp.")


def _references_non_lm_modules(key: str) -> bool:
    return any(segment in key for segment in _NON_LM_KEY_SEGMENTS)


def _strip_language_model_infix(key: str) -> str:
    # mirrors _LANGUAGE_MODEL_INFIX namespace semantics in flash/engine/worker/lora.py
    infix = ".language_model."
    index = key.find(infix)
    if index == -1:
        return key
    return key[:index] + "." + key[index + len(infix) :]


def _json_object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate safetensors JSON key {key!r}")
        value[key] = item
    return value


def _normalize_export_adapter_keys(adapter_dir: Path) -> bool:
    path = adapter_dir / "adapter_model.safetensors"
    if not path.is_file():
        logger.warning("exported adapter has no safetensors weights; leaving weights unchanged")
        return False

    temp_path: Path | None = None
    try:
        file_size = path.stat().st_size
        with path.open("rb") as source:
            length_bytes = source.read(8)
            if len(length_bytes) != 8:
                raise ValueError("file is too small to contain a safetensors header")
            (header_length,) = struct.unpack("<Q", length_bytes)
            if (
                header_length > file_size - 8
                or header_length > _MAX_SAFETENSORS_HEADER_BYTES
            ):
                raise ValueError(
                    f"declared header length {header_length} is implausible for {file_size}-byte file"
                )
            header_bytes = source.read(header_length)
            if len(header_bytes) != header_length:
                raise ValueError("safetensors header is truncated")
            header = json.loads(
                header_bytes,
                object_pairs_hook=_json_object_without_duplicate_keys,
            )
            if not isinstance(header, dict):
                raise ValueError("safetensors header is not a JSON object")

            non_lm = [
                key
                for key in header
                if key != "__metadata__" and _references_non_lm_modules(key)
            ]
            if non_lm:
                logger.info(
                    "exported adapter touches non-lm modules (e.g. %s); keeping the "
                    "multimodal key namespace unchanged",
                    non_lm[0],
                )
                return False

            normalized: dict[str, object] = {}
            remapped = 0
            for key, value in header.items():
                normalized_key = (
                    key if key == "__metadata__" else _strip_language_model_infix(key)
                )
                if normalized_key in normalized:
                    raise ValueError(
                        f"key {key!r} collides after stripping the '.language_model.' infix"
                    )
                normalized[normalized_key] = value
                remapped += int(normalized_key != key)
            if remapped == 0:
                return False

            new_header = json.dumps(
                normalized,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            new_header += b" " * (-len(new_header) % 8)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                delete=False,
            ) as destination:
                temp_path = Path(destination.name)
                destination.write(struct.pack("<Q", len(new_header)))
                destination.write(new_header)
                shutil.copyfileobj(source, destination)
                destination.flush()
                os.fsync(destination.fileno())
        os.replace(temp_path, path)
        temp_path = None
    except Exception as exc:
        logger.warning("could not normalize exported adapter keys; leaving weights unchanged: %s", exc)
        return False
    finally:
        if temp_path is not None:
            with contextlib.suppress(OSError):
                temp_path.unlink(missing_ok=True)

    logger.info("normalized %d exported adapter weight keys", remapped)
    return True


def _rewrite_adapter_config_base_model(
    adapter_dir: Path, base_model: str, base_model_revision: str = ""
) -> bool:
    path = adapter_dir / "adapter_config.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(config, dict):
        return False

    existing_base = str(config.get("base_model_name_or_path") or "").strip()
    if existing_base and existing_base != base_model and not _TEMP_MERGED_BASE_MODEL_RE.fullmatch(
        existing_base
    ):
        raise ValueError(
            f"adapter base_model_name_or_path {existing_base!r} does not match run model {base_model!r}"
        )
    existing_revision = str(config.get("revision") or "").strip()
    if existing_revision and existing_revision != base_model_revision:
        raise ValueError("adapter revision does not match the run's validated base-model commit")
    changed = existing_base != base_model or existing_revision != base_model_revision
    config["base_model_name_or_path"] = base_model
    config["revision"] = base_model_revision or None
    if "base_model" in config and config.get("base_model") != base_model:
        config["base_model"] = base_model
        changed = True
    if changed:
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return changed


def _rewrite_readme_temp_base_model(adapter_dir: Path, base_model: str) -> bool:
    path = adapter_dir / "README.md"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    repaired = _TEMP_MERGED_BASE_MODEL_RE.sub(lambda _match: base_model, text)
    if repaired == text:
        return False
    path.write_text(repaired, encoding="utf-8")
    return True


def _repair_export_metadata(
    adapter_dir: Path, base_model: str, base_model_revision: str = ""
) -> None:
    changed = 0
    changed += int(
        _rewrite_adapter_config_base_model(adapter_dir, base_model, base_model_revision)
    )
    changed += int(_rewrite_readme_temp_base_model(adapter_dir, base_model))
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


def _hub_error_types() -> tuple[type[BaseException], ...]:
    error_types: list[type[BaseException]] = []
    try:
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:
        pass
    else:
        error_types.append(HfHubHTTPError)
    try:
        from requests import RequestException
    except ImportError:
        pass
    else:
        error_types.append(RequestException)
    return tuple(error_types)


def export_adapter(
    *,
    source_repo: str,
    source_subfolder: str,
    dest_repo: str,
    dest_token: str,
    base_model: str,
    base_model_revision: str = "",
    source_token: str | None = None,
    private: bool = True,
) -> str:
    """Copy adapter ``source_repo:{source_subfolder}`` into ``dest_repo`` and return its URL."""
    if not isinstance(base_model, str) or not base_model.strip():
        raise RuntimeError("base_model is required to export adapter metadata")
    base_model = base_model.strip()
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
        except Exception as exc:
            if isinstance(exc, OSError) and not isinstance(exc, _hub_error_types()):
                raise
            raise ServingError(
                f"could not download adapter {source_repo}:{source_subfolder}: {exc}"
            ) from exc
        adapter_dir = Path(tmp) / source_subfolder
        files = (
            sorted(p for p in adapter_dir.rglob("*") if p.is_file()) if adapter_dir.is_dir() else []
        )
        names = {p.name for p in files}
        has_weight = any(n.startswith("adapter_model") for n in names)
        if not has_weight or "adapter_config.json" not in names:
            raise ValueError(
                f"no loadable LoRA adapter at {source_repo}:{source_subfolder} "
                "(need adapter_config.json + an adapter_model* weight; nothing to export)"
            )
        _repair_export_metadata(adapter_dir, base_model, base_model_revision)
        _normalize_export_adapter_keys(adapter_dir)
        api = HfApi(token=dest_token)
        try:
            # Always create private first so the repo is never transiently exposed empty/partial.
            api.create_repo(repo_id=dest_repo, repo_type="model", private=True, exist_ok=True)
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
        except Exception as exc:
            if isinstance(exc, OSError) and not isinstance(exc, _hub_error_types()):
                raise
            raise ServingError(f"could not upload adapter to {dest_repo}: {exc}") from exc
    logger.info(
        "exported %s:%s -> %s (%d files)", source_repo, source_subfolder, dest_repo, len(files)
    )
    return f"https://huggingface.co/{dest_repo}"
