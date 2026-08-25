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
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, BinaryIO

from flash._internal.fileio import reject_duplicate_keys
from flash._internal.logging import get_logger
from flash.adapters.artifacts import (
    ADAPTER_SHARD_PREFIX,
    has_loadable_adapter_weights,
    is_adapter_weight_filename,
)
from flash.adapters.lora_rank import (
    DeclaredLoraRanks,
    _rank_for_module,
    lora_tensor_rank_disagrees,
    strict_declared_lora_ranks,
)
from flash.serve.contract.errors import ServingError

logger = get_logger(__name__)

# Delete stale adapter artifacts on re-export so old weights can't linger beside new ones.
# Scoped to PEFT filenames only — never touch the user's unrelated repo files.
_STALE_ADAPTER_DELETE_PATTERNS = ["adapter_model*", "adapter_config.json"]
_TEMP_MERGED_BASE_MODEL_RE = re.compile(
    r"(?:/[^\s\"'`,\]\){}]+)*/flash_sft_merged_[^\s\"'`,\]\){}]+"
)
_MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024

# exclude non-language-model paths like `flash/engine/worker/model/lora.py` does. preserve mixed
# namespaces only for a nonzero LoRA-B contribution or an unrecognized non-LM tensor; inert entries
# alone do not prove those modules were trained.
_NON_LM_KEY_SEGMENTS = (".visual.", ".vision_tower.", ".multi_modal_projector.", ".mtp.")


def _references_non_lm_modules(key: str) -> bool:
    return any(segment in key for segment in _NON_LM_KEY_SEGMENTS)


def _non_lm_liveness_from_key(key: str) -> bool | None:
    """True/False when the key name alone settles liveness, None when the tensor data decides."""
    is_lora_a = ".lora_A." in key
    is_lora_b = ".lora_B." in key
    if is_lora_a and not is_lora_b:
        return False
    if not is_lora_b or is_lora_a:
        return True
    return None


def _non_lm_tensor_is_live(
    source: BinaryIO,
    key: str,
    descriptor: object,
    *,
    data_start: int,
    file_size: int,
) -> bool:
    decided = _non_lm_liveness_from_key(key)
    if decided is not None:
        return decided
    if not isinstance(descriptor, dict):
        raise ValueError(f"safetensors descriptor for {key!r} is not an object")
    offsets = descriptor.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(not isinstance(offset, int) or isinstance(offset, bool) for offset in offsets)
    ):
        raise ValueError(f"safetensors data_offsets for {key!r} are invalid")
    lo, hi = offsets
    data_size = file_size - data_start
    if lo < 0 or hi < lo or hi > data_size:
        raise ValueError(
            f"safetensors data_offsets for {key!r} are outside the {data_size}-byte data section"
        )
    source.seek(data_start + lo)
    tensor_bytes = source.read(hi - lo)
    if len(tensor_bytes) != hi - lo:
        raise ValueError(f"safetensors tensor data for {key!r} is truncated")
    return any(tensor_bytes)


def _strip_language_model_infix(key: str) -> str:
    # mirrors the peft `.language_model.` namespace semantics for warm-start vl adapters
    infix = ".language_model."
    index = key.find(infix)
    if index == -1:
        return key
    return key[:index] + "." + key[index + len(infix) :]


_json_object_without_duplicate_keys = reject_duplicate_keys(
    lambda key: ValueError(f"duplicate safetensors JSON key {key!r}")
)


@dataclass
class _WeightScan:
    """One adapter weight file: its tensor keys, and whatever pins it to the multimodal namespace.

    the safetensors header is key metadata, so holding every shard's header is cheap while the
    tensor payload stays on disk until the rewrite copies it.
    """

    path: Path
    keys: tuple[str, ...]
    header: dict[str, object]
    data_start: int
    live_non_lm_key: str | None = None


@contextlib.contextmanager
def _atomic_replacement(path: Path) -> Iterator[IO[bytes]]:
    """Build a replacement for `path` beside it and swap it in only once it is complete."""
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as destination:
            temp_path = Path(destination.name)
            yield destination
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            with contextlib.suppress(OSError):
                temp_path.unlink(missing_ok=True)


def _adapter_weight_paths(adapter_dir: Path) -> list[Path]:
    """the safetensors weight files peft will actually load, single-file or sharded."""
    if not adapter_dir.is_dir():
        return []
    candidates = sorted(
        p for p in adapter_dir.rglob("*") if p.is_file() and is_adapter_weight_filename(p.name)
    )
    single = [p for p in candidates if not p.name.startswith(ADAPTER_SHARD_PREFIX)]
    if single:
        return single
    # the index names the current shard set. a retry can leave same-suffix shards from a previous
    # attempt beside the live ones, so scanning every shard could bind stale keys or false collisions.
    return _index_referenced_shards(adapter_dir, candidates)


def _index_referenced_shards(adapter_dir: Path, candidates: list[Path]) -> list[Path]:
    """the shards the safetensors index points at, or [] when it names none we can read."""
    index_path = adapter_dir / "adapter_model.safetensors.index.json"
    if not index_path.is_file():
        return []
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict):
            return []
        referenced = {str(shard) for shard in weight_map.values()}
    except (OSError, ValueError):
        return []
    by_name = {path.name: path for path in candidates}
    if not referenced or not referenced <= by_name.keys():
        return []
    return [by_name[name] for name in sorted(referenced)]


def _read_safetensors_header(path: Path) -> tuple[dict[str, object], int, int]:
    """Return the parsed header, the offset its tensor data starts at, and the file size."""
    file_size = path.stat().st_size
    with path.open("rb") as source:
        length_bytes = source.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"{path.name} is too small to contain a safetensors header")
        (header_length,) = struct.unpack("<Q", length_bytes)
        if header_length > file_size - 8 or header_length > _MAX_SAFETENSORS_HEADER_BYTES:
            raise ValueError(
                f"{path.name}: declared header length {header_length} is implausible for "
                f"{file_size}-byte file"
            )
        header_bytes = source.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError(f"{path.name}: safetensors header is truncated")
    header = json.loads(header_bytes, object_pairs_hook=_json_object_without_duplicate_keys)
    if not isinstance(header, dict):
        raise ValueError(f"{path.name}: safetensors header is not a JSON object")
    return header, 8 + header_length, file_size


def _scan_safetensors(path: Path) -> _WeightScan:
    header, data_start, file_size = _read_safetensors_header(path)
    live: str | None = None
    with path.open("rb") as source:
        for key, descriptor in header.items():
            if key == "__metadata__" or not _references_non_lm_modules(key):
                continue
            if _non_lm_tensor_is_live(
                source, key, descriptor, data_start=data_start, file_size=file_size
            ):
                live = key
                break
    return _WeightScan(
        path=path,
        keys=tuple(k for k in header if k != "__metadata__"),
        live_non_lm_key=live,
        header=header,
        data_start=data_start,
    )


def _plan_key_renames(scans: list[_WeightScan]) -> dict[str, str]:
    """Map every infixed key to its stripped form, refusing a rename that would shadow another key.

    Planned across all shards at once: sharding splits one key namespace over several files, so a
    collision is only visible in their union.
    """
    seen: dict[str, str] = {}
    renames: dict[str, str] = {}
    for scan in scans:
        for key in scan.keys:
            normalized = _strip_language_model_infix(key)
            collided = seen.get(normalized)
            if collided is not None:
                raise ValueError(
                    f"key {key!r} collides with {collided!r} after stripping the "
                    "'.language_model.' infix"
                )
            seen[normalized] = key
            if normalized != key:
                renames[key] = normalized
    return renames


def _rewrite_safetensors_keys(scan: _WeightScan, renames: dict[str, str]) -> None:
    header = scan.header
    normalized = {
        (key if key == "__metadata__" else renames.get(key, key)): value
        for key, value in header.items()
    }
    new_header = json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    new_header += b" " * (-len(new_header) % 8)
    with scan.path.open("rb") as source:
        source.seek(scan.data_start)
        with _atomic_replacement(scan.path) as destination:
            destination.write(struct.pack("<Q", len(new_header)))
            destination.write(new_header)
            shutil.copyfileobj(source, destination)


def _rewrite_weight_index_keys(adapter_dir: Path, renames: dict[str, str]) -> None:
    """keep the safetensors shard index in step with the shards it points at."""
    path = adapter_dir / "adapter_model.safetensors.index.json"
    if not path.is_file():
        return
    index = json.loads(path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict):
        raise ValueError(f"{path.name}: no weight_map object to normalize")
    index["weight_map"] = {renames.get(key, key): value for key, value in weight_map.items()}
    path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


def _normalize_adapter_key_namespace(adapter_dir: Path) -> str:
    paths = _adapter_weight_paths(adapter_dir)
    if not paths:
        raise ValueError("no adapter_model safetensors weight file to normalize")
    scans = [_scan_safetensors(path) for path in paths]
    # before any rewrite: the shapes are what they are, and an adapter whose tensors contradict
    # its own config is unloadable regardless of which key namespace it ends up in.
    _verify_export_tensor_ranks(adapter_dir, scans)
    pinned = next((scan for scan in scans if scan.live_non_lm_key), None)
    if pinned is not None:
        logger.info(
            "exported adapter has live non-lm weights (e.g. %s); keeping the multimodal key "
            "namespace unchanged",
            pinned.live_non_lm_key,
        )
        return "multimodal"

    renames = _plan_key_renames(scans)
    if not renames:
        return "text_only"
    for scan in scans:
        if any(key in renames for key in scan.keys):
            _rewrite_safetensors_keys(scan, renames)
    _rewrite_weight_index_keys(adapter_dir, renames)
    logger.info(
        "normalized %d exported adapter weight keys across %d file(s)", len(renames), len(paths)
    )
    return "text_only"


def _normalize_export_targeting(adapter_dir: Path, namespace: str) -> None:
    """drop the conditional-model exclusion after text keys enter the causal-lm namespace."""
    if namespace != "text_only":
        return
    path = adapter_dir / "adapter_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "exclude_modules" not in config:
        return
    config.pop("exclude_modules")
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _normalize_export_adapter_keys(adapter_dir: Path) -> str:
    """Rewrite exported adapter keys into the namespace vanilla peft and vLLM load.

    Returns the resulting namespace: ``"text_only"`` once the ``.language_model.`` infix is gone,
    ``"multimodal"`` for an adapter whose live non-LM weights mean it must keep that namespace.

    Raises when normalization is needed and cannot be applied. peft does not error on mismatched
    adapter keys, it emits a UserWarning and applies nothing, so an export that ships unnormalized
    weights hands the user a base model they will benchmark as their adapter. A failed export is
    the only signal that ever reaches them.
    """
    try:
        return _normalize_adapter_key_namespace(adapter_dir)
    except _AdapterRankMismatch:
        # a self-contradicting adapter, not an unreadable key namespace. reported as itself.
        raise
    except Exception as exc:
        raise ServingError(
            f"could not normalize exported adapter keys ({exc}); refusing to export weights "
            "vanilla peft would load as a no-op"
        ) from exc


class _AdapterRankMismatch(ValueError):
    """The adapter's tensors contradict its own config, so it is unloadable as published.

    Distinct from the errors `_normalize_export_adapter_keys` wraps: that wrapper reports a key
    namespace it could not normalize, which is a different defect with a different remedy, and
    letting this one be reported as that would send the user looking in the wrong place. A
    ValueError so it reaches the export route as a 404 beside the sibling "no loadable LoRA
    adapter" refusal, rather than as a 502 that blames the Hub for a bad artifact.
    """


def _declared_export_ranks(adapter_dir: Path) -> DeclaredLoraRanks:
    """The rank structure ``adapter_config.json`` declares, or an empty one we cannot judge from.

    Empty rather than an error: a config carrying no readable rank is the pre-existing shape of
    every fixture and of adapters PEFT wrote without one, and export has never required it. The
    check below exists to catch a config and its tensors DISAGREEING, which needs both halves.
    """
    try:
        config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return DeclaredLoraRanks()
    if not isinstance(config, dict):
        return DeclaredLoraRanks()
    try:
        return strict_declared_lora_ranks(config)
    except ValueError as exc:
        raise _AdapterRankMismatch(str(exc)) from exc


def _verify_export_tensor_ranks(adapter_dir: Path, scans: list[_WeightScan]) -> None:
    """Refuse an export whose LoRA tensors do not carry the rank its config declares.

    A serving engine sizes its LoRA slots from ``adapter_config.json`` and then binds the tensors,
    so the two disagreeing is not a cosmetic metadata bug: it is an adapter that cannot be loaded.
    That is not hypothetical -- a 35B MoE SFT run published `r: 32` beside expert tensors export
    checked only for presence and key-namespace readability, so it succeeded and printed a Hub URL
    while deploying the identical artifact failed. Export exiting 0 is what made a broken adapter
    look like a finished deliverable.

    `lora_tensor_rank_disagrees` resolves the rank PER MODULE the way PEFT does, so a
    `rank_pattern` override is judged against its own rank rather than a summary of the config.

    safetensors headers carry the shapes as metadata and are already parsed here, so validation does
    not materialize tensor payloads or require the worker's gpu dependencies.
    """
    declared = _declared_export_ranks(adapter_dir)

    def rank_is_invalid(key: str, descriptor: object) -> bool:
        if not isinstance(descriptor, dict):
            return False
        infix = ".lora_A." if ".lora_A." in key else ".lora_B." if ".lora_B." in key else None
        if infix is None:
            return False
        module = key.partition(infix)[0]
        return _rank_for_module(module, declared) is None or lora_tensor_rank_disagrees(
            key, descriptor.get("shape"), declared
        )

    mismatched = [
        f"{key} has shape {descriptor.get('shape')}"
        for scan in scans
        for key, descriptor in scan.header.items()
        if key != "__metadata__" and rank_is_invalid(key, descriptor)
    ]
    if not mismatched:
        return
    shown = ", ".join(sorted(mismatched)[:3])
    raise _AdapterRankMismatch(
        f"adapter_config.json declares r={declared.default} but {len(mismatched)} LoRA tensor(s) "
        f"do not carry the rank configured for their module (e.g. {shown}); a serving engine sizes "
        "its LoRA slots from the config, so this adapter cannot be loaded and exporting it would "
        "publish a broken artifact"
    )


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
    if (
        existing_base
        and existing_base != base_model
        and not _TEMP_MERGED_BASE_MODEL_RE.fullmatch(existing_base)
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
    changed += int(_rewrite_adapter_config_base_model(adapter_dir, base_model, base_model_revision))
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
        if not has_loadable_adapter_weights(names) or "adapter_config.json" not in names:
            raise ValueError(
                f"no loadable LoRA adapter at {source_repo}:{source_subfolder} "
                "(need adapter_config.json + an adapter_model weight, or a complete "
                "index-referenced shard set; nothing to export)"
            )
        _repair_export_metadata(adapter_dir, base_model, base_model_revision)
        namespace = _normalize_export_adapter_keys(adapter_dir)
        _normalize_export_targeting(adapter_dir, namespace)
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
        "exported %s:%s -> %s (%d files, %s key namespace)",
        source_repo,
        source_subfolder,
        dest_repo,
        len(files),
        namespace,
    )
    return f"https://huggingface.co/{dest_repo}"
