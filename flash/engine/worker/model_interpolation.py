"""materialize deterministic full-model weight interpolations from safetensors parents."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from flash.spec import ModelInterpolationSpec

_FORMULA = "W=(1-alpha)*W_base+alpha*W_instruct"
_DEFAULT_MAX_SHARD_BYTES = 1024**3
_MANIFEST_NAME = "flash_interpolation_manifest.json"


@dataclass(frozen=True)
class TensorLocation:
    file: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class MaterializedInterpolation:
    source: str
    manifest: dict[str, Any]

    @property
    def output_fingerprint(self) -> str:
        return str(self.manifest["output_fingerprint"])


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _snapshot_commit(path: str) -> str | None:
    parts = Path(path).resolve().parts
    try:
        index = parts.index("snapshots")
    except ValueError:
        return None
    if index + 1 >= len(parts):
        return None
    return parts[index + 1]


def _load_config(repo: Path) -> dict[str, Any]:
    path = repo / "config.json"
    if not path.is_file():
        raise ValueError(f"interpolation parent {repo} has no config.json")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"interpolation parent {repo} has a non-object config.json")
    return value


_CONFIG_PROVENANCE_KEYS = frozenset(
    {
        "_name_or_path",
        "name_or_path",
        "transformers_version",
        "torch_dtype",
        "dtype",
    }
)


def _architecture_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _architecture_config(child)
            for key, child in sorted(value.items())
            if key not in _CONFIG_PROVENANCE_KEYS
        }
    if isinstance(value, list):
        return [_architecture_config(child) for child in value]
    return value


def config_fingerprint(repo: str | Path) -> str:
    config = _architecture_config(_load_config(Path(repo)))
    return _sha256_bytes(_canonical_json(config))


def _tensor_framework() -> str:
    try:
        import torch  # noqa: F401
    except ImportError:
        return "np"
    return "pt"


def _weight_map(repo: Path) -> dict[str, str]:
    index_path = repo / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as handle:
            index = json.load(handle)
        mapping = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"invalid safetensors index in {index_path}")
        out = {str(key): str(filename) for key, filename in mapping.items()}
        missing = sorted({filename for filename in out.values() if not (repo / filename).is_file()})
        if missing:
            raise ValueError(f"safetensors index references missing shard(s): {', '.join(missing)}")
        return out

    single = repo / "model.safetensors"
    candidates = [single] if single.is_file() else sorted(repo.glob("*.safetensors"))
    if not candidates:
        raise ValueError(f"interpolation parent {repo} has no safetensors model weights")

    from safetensors import safe_open

    out: dict[str, str] = {}
    for path in candidates:
        with safe_open(path, framework=_tensor_framework(), device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118 - safe_open is not iterable
                if key in out:
                    raise ValueError(f"duplicate tensor key {key!r} across safetensors shards")
                out[key] = path.name
    return out


def _tensor_inventory(repo: Path, mapping: dict[str, str]) -> dict[str, TensorLocation]:
    from safetensors import safe_open

    inventory: dict[str, TensorLocation] = {}
    by_file: dict[str, list[str]] = {}
    for key, filename in mapping.items():
        by_file.setdefault(filename, []).append(key)
    for filename, expected_keys in sorted(by_file.items()):
        with safe_open(repo / filename, framework=_tensor_framework(), device="cpu") as handle:
            actual = set(handle.keys())
            expected = set(expected_keys)
            if actual != expected:
                missing = sorted(expected - actual)
                extra = sorted(actual - expected)
                raise ValueError(
                    f"safetensors index mismatch in {filename}: missing={missing}, extra={extra}"
                )
            for key in sorted(expected):
                tensor = handle.get_tensor(key)
                inventory[key] = TensorLocation(
                    file=filename,
                    shape=tuple(int(dim) for dim in tensor.shape),
                    dtype=str(tensor.dtype),
                )
    return inventory


def _validate_inventories(
    base: dict[str, TensorLocation], instruct: dict[str, TensorLocation]
) -> None:
    base_keys = set(base)
    instruct_keys = set(instruct)
    if base_keys != instruct_keys:
        missing = sorted(base_keys - instruct_keys)
        extra = sorted(instruct_keys - base_keys)
        raise ValueError(f"checkpoint tensor sets differ: missing={missing}, extra={extra}")
    for key in sorted(base):
        left = base[key]
        right = instruct[key]
        if left.shape != right.shape:
            raise ValueError(
                f"checkpoint tensor shape mismatch for {key}: {left.shape} != {right.shape}"
            )
        if left.dtype != right.dtype:
            raise ValueError(
                f"checkpoint tensor dtype mismatch for {key}: {left.dtype} != {right.dtype}"
            )


def _copy_source_tree(source: Path, output: Path, weight_files: set[str]) -> None:
    excluded = set(weight_files) | {"model.safetensors.index.json", _MANIFEST_NAME}

    def ignore(_root: str, names: list[str]) -> set[str]:
        stale_weights = {
            name
            for name in names
            if name == "pytorch_model.bin"
            or name == "pytorch_model.bin.index.json"
            or (name.startswith("pytorch_model-") and name.endswith(".bin"))
        }
        return {
            name
            for name in names
            if name in excluded or name in stale_weights or name in {".cache", ".git"}
        }

    shutil.copytree(source, output, ignore=ignore)


def _tensor_bytes(tensor: Any) -> int:
    if hasattr(tensor, "numel"):
        return int(tensor.numel()) * int(tensor.element_size())
    return int(tensor.nbytes)


def _tensor_digest(tensor: Any) -> str:
    if hasattr(tensor, "detach"):
        import torch

        raw = tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
    else:
        raw = tensor.tobytes(order="C")
    return _sha256_bytes(raw)


def _interpolate_tensor(base: Any, instruct: Any, alpha: float) -> Any:
    if hasattr(base, "detach"):
        import torch

        if not base.dtype.is_floating_point:
            if not torch.equal(base, instruct):
                raise ValueError("non-floating tensor mismatch")
            return base
        if alpha == 0.0:
            return base
        if alpha == 1.0:
            return instruct
        original_dtype = base.dtype
        if original_dtype in {torch.float16, torch.bfloat16}:
            base = base.float()
            instruct = instruct.float()
        base.mul_(1.0 - alpha)
        base.add_(instruct, alpha=alpha)
        return base.to(dtype=original_dtype)

    import numpy as np

    if not np.issubdtype(base.dtype, np.floating):
        if not np.array_equal(base, instruct):
            raise ValueError("non-floating tensor mismatch")
        return base
    if alpha == 0.0:
        return base
    if alpha == 1.0:
        return instruct
    original_dtype = base.dtype
    accumulator = base.astype(np.float32, copy=True)
    accumulator *= 1.0 - alpha
    accumulator += instruct.astype(np.float32, copy=False) * alpha
    return accumulator.astype(original_dtype, copy=False)


def _tied_pairs(keys: set[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for head in sorted(key for key in keys if key.endswith("lm_head.weight")):
        prefix = head[: -len("lm_head.weight")]
        candidates = (
            f"{prefix}model.embed_tokens.weight",
            f"{prefix}embed_tokens.weight",
            head.replace("lm_head.weight", "model.language_model.embed_tokens.weight"),
        )
        match = next((candidate for candidate in candidates if candidate in keys), None)
        if match is not None:
            pairs.append((match, head))
    return pairs


def _manifest_fingerprint(manifest: dict[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "output_fingerprint"}
    return _sha256_bytes(_canonical_json(payload))


def _materialized_files_valid(output: Path, manifest: dict[str, Any]) -> bool:
    if manifest.get("output_fingerprint") != _manifest_fingerprint(manifest):
        return False
    shards = manifest.get("shards")
    if not isinstance(shards, dict) or not (output / "model.safetensors.index.json").is_file():
        return False
    return all(
        isinstance(expected, str)
        and (path := output / str(filename)).is_file()
        and _sha256_file(path) == expected
        for filename, expected in shards.items()
    )


def materialize_interpolation_from_paths(
    spec: ModelInterpolationSpec,
    *,
    base_path: str,
    instruct_path: str,
    output_dir: str,
    base_commit: str | None = None,
    instruct_commit: str | None = None,
    max_shard_bytes: int = _DEFAULT_MAX_SHARD_BYTES,
) -> MaterializedInterpolation:
    """materialize one interpolation from two local snapshots."""
    from safetensors import safe_open

    if _tensor_framework() == "pt":
        from safetensors.torch import save_file
    else:
        from safetensors.numpy import save_file

    if max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive")
    base_repo = Path(base_path)
    instruct_repo = Path(instruct_path)
    output = Path(output_dir)
    if output.exists():
        shutil.rmtree(output)

    base_config_fp = config_fingerprint(base_repo)
    instruct_config_fp = config_fingerprint(instruct_repo)
    if base_config_fp != instruct_config_fp:
        raise ValueError(
            "interpolation parents have different architecture/config fingerprints: "
            f"{base_config_fp} != {instruct_config_fp}"
        )

    base_map = _weight_map(base_repo)
    instruct_map = _weight_map(instruct_repo)
    base_inventory = _tensor_inventory(base_repo, base_map)
    instruct_inventory = _tensor_inventory(instruct_repo, instruct_map)
    _validate_inventories(base_inventory, instruct_inventory)

    source_repo = base_repo if spec.tokenizer_config_source == "base" else instruct_repo
    source_map = base_map if spec.tokenizer_config_source == "base" else instruct_map
    _copy_source_tree(source_repo, output, set(source_map.values()))

    keys = sorted(base_inventory)
    shard: dict[str, Any] = {}
    shard_bytes = 0
    temp_shards: list[tuple[Path, tuple[str, ...]]] = []
    total_size = 0
    tied_digests: dict[str, str] = {}
    tied_keys = {key for pair in _tied_pairs(set(keys)) for key in pair}

    def flush_shard() -> None:
        nonlocal shard, shard_bytes
        if not shard:
            return
        path = output / f".flash-shard-{len(temp_shards) + 1:05d}.safetensors"
        shard_keys = tuple(shard)
        save_file(shard, path, metadata={"format": "pt"})
        temp_shards.append((path, shard_keys))
        shard.clear()
        shard = {}
        shard_bytes = 0

    for key in keys:
        left_location = base_inventory[key]
        right_location = instruct_inventory[key]
        with safe_open(
            base_repo / left_location.file,
            framework=_tensor_framework(),
            device="cpu",
        ) as left_file:
            left = left_file.get_tensor(key)
        with safe_open(
            instruct_repo / right_location.file,
            framework=_tensor_framework(),
            device="cpu",
        ) as right_file:
            right = right_file.get_tensor(key)
        try:
            value = _interpolate_tensor(left, right, spec.alpha)
        except ValueError as exc:
            raise ValueError(f"{exc} for {key}") from exc
        value_bytes = _tensor_bytes(value)
        if shard and shard_bytes + value_bytes > max_shard_bytes:
            flush_shard()
        shard[key] = value
        shard_bytes += value_bytes
        total_size += value_bytes
        if key in tied_keys:
            tied_digests[key] = _tensor_digest(value)
    flush_shard()

    tied_pairs = _tied_pairs(set(keys))
    source_config = _load_config(source_repo)
    if source_config.get("tie_word_embeddings") is True:
        for embedding, head in tied_pairs:
            if tied_digests[embedding] != tied_digests[head]:
                raise ValueError(
                    f"tied embedding tensors diverged after interpolation: {embedding} != {head}"
                )

    weight_map: dict[str, str] = {}
    shard_hashes: dict[str, str] = {}
    shard_count = len(temp_shards)
    for index, (temp_path, shard_keys) in enumerate(temp_shards, start=1):
        filename = (
            "model.safetensors"
            if shard_count == 1
            else f"model-{index:05d}-of-{shard_count:05d}.safetensors"
        )
        path = output / filename
        temp_path.replace(path)
        shard_hashes[filename] = _sha256_file(path)
        for key in shard_keys:
            weight_map[key] = filename

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": {key: weight_map[key] for key in sorted(weight_map)},
    }
    with (output / "model.safetensors.index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")

    manifest: dict[str, Any] = {
        "version": 1,
        "formula": _FORMULA,
        "spec": asdict(spec),
        "parents": {
            "base": {
                "model": spec.base_model,
                "requested_revision": spec.base_revision or None,
                "commit": base_commit or _snapshot_commit(base_path),
                "config_fingerprint": base_config_fp,
            },
            "instruct": {
                "model": spec.instruct_model,
                "requested_revision": spec.instruct_revision or None,
                "commit": instruct_commit or _snapshot_commit(instruct_path),
                "config_fingerprint": instruct_config_fp,
            },
        },
        "tokenizer_config_source": spec.tokenizer_config_source,
        "tensor_count": len(keys),
        "total_size": total_size,
        "shards": {name: shard_hashes[name] for name in sorted(shard_hashes)},
    }
    manifest["output_fingerprint"] = _manifest_fingerprint(manifest)
    with (output / _MANIFEST_NAME).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return MaterializedInterpolation(source=str(output), manifest=manifest)


def materialize_model_interpolation(
    spec: ModelInterpolationSpec,
    *,
    output_root: str = "/tmp/flash-model-interpolations",
    cache_dir: str | None = None,
) -> MaterializedInterpolation:
    """prefetch both parents, then materialize a deterministic concrete source path."""
    from huggingface_hub import snapshot_download

    key = _sha256_bytes(_canonical_json(asdict(spec)))[:24]
    output = Path(output_root) / key
    manifest_path = output / _MANIFEST_NAME
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if _materialized_files_valid(output, manifest):
            return MaterializedInterpolation(source=str(output), manifest=manifest)
        shutil.rmtree(output)

    common = {
        "ignore_patterns": ["*.pth", "*.gguf", "original/*", "*.onnx", "*.msgpack", "*.h5"],
    }
    if cache_dir is not None:
        common["cache_dir"] = cache_dir
    from flash.engine.worker.heartbeat import liveness_heartbeat

    with liveness_heartbeat("model_interpolation_prefetching"):
        base_path = snapshot_download(
            repo_id=spec.base_model,
            revision=spec.base_revision or None,
            **common,
        )
        instruct_path = snapshot_download(
            repo_id=spec.instruct_model,
            revision=spec.instruct_revision or None,
            **common,
        )
    with liveness_heartbeat("model_interpolation_materializing"):
        return materialize_interpolation_from_paths(
            spec,
            base_path=base_path,
            instruct_path=instruct_path,
            output_dir=str(output),
        )


def interpolation_disk_gb(params_b: float) -> int:
    """ephemeral disk floor for two bf16 parents, one output, and download headroom."""
    if not math.isfinite(params_b) or params_b <= 0:
        return 0
    return math.ceil(params_b * 2.0 * 3.0 * 1.2)


def main(argv: list[str] | None = None) -> int:
    """materialize local parent directories for offline verification and operations."""
    import argparse

    parser = argparse.ArgumentParser(description="materialize a model weight interpolation")
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--instruct-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--instruct-model", required=True)
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--tokenizer-config-source", choices=("base", "instruct"), default="instruct")
    parser.add_argument("--base-revision", default="")
    parser.add_argument("--instruct-revision", default="")
    parser.add_argument("--max-shard-bytes", type=int, default=_DEFAULT_MAX_SHARD_BYTES)
    args = parser.parse_args(argv)
    spec = ModelInterpolationSpec(
        base_model=args.base_model,
        instruct_model=args.instruct_model,
        alpha=args.alpha,
        tokenizer_config_source=args.tokenizer_config_source,
        base_revision=args.base_revision,
        instruct_revision=args.instruct_revision,
    )
    result = materialize_interpolation_from_paths(
        spec,
        base_path=args.base_path,
        instruct_path=args.instruct_path,
        output_dir=args.output_dir,
        max_shard_bytes=args.max_shard_bytes,
    )
    print(json.dumps(result.manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
