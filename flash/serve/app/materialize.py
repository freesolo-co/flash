"""exact data-only adapter hydration and cache revalidation."""

from __future__ import annotations

import contextlib
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import shutil
import stat
import uuid
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from flash.adapters.fused_experts import (
    has_complete_fused_expert_tensors,
    lora_target_parameters,
    validate_fused_expert_adapter_config,
)
from flash.adapters.lora_rank import (
    lora_tensor_rank_disagrees,
    rank_from_adapter_config,
    strict_declared_lora_ranks,
)
from flash.engine.worker.model.lora import _read_safetensors_tensor_metadata
from flash.serve.contract.protocol import reject_non_finite_json_constant

from .manifest import ArtifactFile, ManifestAdapter, ServingManifest
from .progress import emit_boot_progress

_ARTIFACT_TOKEN_FD_ENV = "FLASH_ARTIFACT_TOKEN_FD"
_CONFIG_NAME = "adapter_config.json"
_WEIGHTS_NAME = "adapter_model.safetensors"
_COPY_CHUNK_BYTES = 1024 * 1024
# the served base model is a private Freesolo-Co repo, so vllm cannot fetch it anonymously. the
# artifact token only exists during bootstrap, so the weights are pulled into the persistent
# volume's hub cache while it is still available and every later start reads them from there.
BASE_WEIGHTS_CACHE_DIRNAME = "hub"
_DIRECTORY_FLAGS = (
    getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
)
_UNSUPPORTED_DIRECTORY_SYNC = {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}
_LIBC = ctypes.CDLL(None, use_errno=True)
_AT_EMPTY_PATH = 0x1000
_AT_STATX_FORCE_SYNC = 0x2000
_STATX_BASIC_STATS = 0x7FF


class MaterializationError(RuntimeError):
    """an adapter could not be safely hydrated or revalidated."""


def adapter_cache_path(cache_root: str | os.PathLike[str], adapter: ManifestAdapter) -> Path:
    """return the digest-addressed runtime directory for one adapter."""

    return Path(cache_root) / "adapters" / adapter.aggregate_sha256


def base_weights_cache_path(cache_root: str | os.PathLike[str]) -> Path:
    """return the huggingface hub cache directory shared by every start of one deployment."""

    return Path(cache_root) / BASE_WEIGHTS_CACHE_DIRNAME


def _file_sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _snapshot_inventory(snapshot_path: object) -> str:
    """one stable line per file: relative path, byte size, and sha-256 digest.

    this intentionally reads every byte. path and metadata checks cannot detect same-size content
    replacement, and this cache volume does not expose a trusted filesystem content digest.
    """

    root = Path(str(snapshot_path))
    entries = [
        f"{path.relative_to(root).as_posix()}\t{path.stat().st_size}\t{_file_sha256(path)}"
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    return "\n".join(entries)


def _snapshot_matches_inventory(snapshot_path: object, recorded: str) -> bool:
    """whether the snapshot on disk still matches what the marker vouched for."""

    try:
        return _snapshot_inventory(snapshot_path) == recorded
    except OSError:
        return False


def _hydration_marker(
    cache_dir: str | os.PathLike[str], repo_id: str, revision: str | None
) -> Path:
    """locate proof that one exact hub snapshot finished downloading.
    exact-sha offline resolution accepts partial snapshots, so readiness also needs this marker.
    it is co-located with the cache it vouches for, preserving path spelling and wipe semantics.
    """

    digest = hashlib.sha256(f"{repo_id}@{revision or ''}".encode()).hexdigest()[:32]
    return Path(cache_dir) / f".flash-complete-{digest}"


def base_weights_are_cached(
    manifest: ServingManifest,
    cache_root: str | os.PathLike[str],
    *,
    snapshot_download_fn: Callable[..., str] | None = None,
) -> bool:
    """report whether the exact served base revision resolves offline from the volume cache.

    readiness is the AND of two independent checks, because neither is sufficient alone:

    * a completion marker this module wrote, which is the only positive evidence that a download
      finished rather than merely started -- see `_hydration_marker` for why the hub's own offline
      resolution cannot answer that for a commit-sha revision;
    * the hub's offline resolution, which is what notices the cache being emptied or evicted
      underneath a marker that survived.
    """

    if type(manifest) is not ServingManifest:
        raise MaterializationError("base weight lookup requires an exact ServingManifest")
    if snapshot_download_fn is None:
        from huggingface_hub import snapshot_download

        snapshot_download_fn = snapshot_download
    cache_dir = str(base_weights_cache_path(cache_root))
    for repo_id, revision in _base_weight_sources(manifest):
        marker = _hydration_marker(cache_dir, repo_id, revision)
        try:
            recorded = marker.read_text(encoding="utf-8")
        except OSError:
            return False
        try:
            snapshot_path = snapshot_download_fn(
                repo_id=repo_id,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=True,
            )
        except Exception:
            return False
        # the marker can outlive an evicted shard, so compare it with the current snapshot bytes.
        if not _snapshot_matches_inventory(snapshot_path, recorded):
            return False
    return True


def _base_weight_sources(manifest: ServingManifest) -> tuple[tuple[str, str | None], ...]:
    """return each distinct (repo, revision) vllm loads for this engine identity.

    the tokenizer is usually the same repo as the model, but the profile states them separately,
    so both are hydrated and deduplicated rather than assuming they agree.
    """

    engine = manifest.engine
    sources: list[tuple[str, str | None]] = [(engine.served_model, engine.model_revision)]
    tokenizer_model = getattr(engine, "tokenizer_model", None) or engine.served_model
    tokenizer_revision = getattr(engine, "tokenizer_revision", None)
    sources.append((tokenizer_model, tokenizer_revision))
    seen: dict[tuple[str, str | None], None] = {}
    for source in sources:
        seen.setdefault(source, None)
    return tuple(seen)


def hydrate_base_weights(
    manifest: ServingManifest,
    cache_root: str | os.PathLike[str],
    *,
    token_fd: int | None = None,
    snapshot_download_fn: Callable[..., str] | None = None,
) -> None:
    """download the served base model into the volume hub cache with the bootstrap token."""

    if type(manifest) is not ServingManifest:
        raise MaterializationError("base weight hydration requires an exact ServingManifest")
    token = read_artifact_token_fd(token_fd)
    try:
        if snapshot_download_fn is None:
            from huggingface_hub import snapshot_download

            snapshot_download_fn = snapshot_download
        root = _prepare_cache_root(cache_root)
        cache_dir = root / BASE_WEIGHTS_CACHE_DIRNAME
        cache_dir.mkdir(mode=0o755, exist_ok=True)
        for repo_id, revision in _base_weight_sources(manifest):
            emit_boot_progress("artifact-download-starting", repo=repo_id, revision=revision)
            try:
                snapshot_path = snapshot_download_fn(
                    repo_id=repo_id,
                    revision=revision,
                    token=token,
                    cache_dir=str(cache_dir),
                )
            except Exception:
                # the repo id is manifest data and safe to name, but the exception can carry the
                # request headers, so it is dropped rather than chained.
                raise MaterializationError(f"base weight download failed for {repo_id}") from None
            # record exact bytes only after a complete download; interrupted or later-evicted
            # snapshots must rehydrate while the bootstrap token still exists.
            marker = _hydration_marker(cache_dir, repo_id, revision)
            marker.write_text(_snapshot_inventory(snapshot_path), encoding="utf-8")
    finally:
        token = ""


def read_artifact_token_fd(fd: int | None = None) -> str:
    """read and close one short-lived token descriptor selected by a numeric value."""

    if fd is None:
        raw_fd = os.environ.get(_ARTIFACT_TOKEN_FD_ENV)
        if raw_fd is None or not raw_fd.isdecimal():
            raise MaterializationError("artifact token fd is not configured")
        fd = int(raw_fd)
    if type(fd) is not int or fd < 0:
        raise MaterializationError("artifact token fd must be a non-negative integer")
    try:
        with os.fdopen(fd, "r", encoding="utf-8", closefd=True) as token_file:
            token = token_file.read()
    except OSError as exc:
        raise MaterializationError("artifact token fd could not be read") from exc
    token = token.strip()
    if not token:
        raise MaterializationError("artifact token fd was empty")
    return token


def hydrate_manifest(
    manifest: ServingManifest,
    cache_root: str | os.PathLike[str],
    *,
    token_fd: int | None = None,
    snapshot_download_fn: Callable[..., str] | None = None,
) -> dict[str, Path]:
    """materialize every adapter, then return detached digest-addressed paths."""

    if type(manifest) is not ServingManifest:
        raise MaterializationError("hydrate requires an exact validated ServingManifest")
    token = read_artifact_token_fd(token_fd)
    try:
        return _materialize_all(
            manifest,
            cache_root,
            token=token,
            snapshot_download_fn=snapshot_download_fn,
        )
    finally:
        token = ""


def _materialize_all(
    manifest: ServingManifest,
    cache_root: str | os.PathLike[str],
    *,
    token: str,
    snapshot_download_fn: Callable[..., str] | None = None,
) -> dict[str, Path]:
    """hydrate all exact adapters with one request-scoped token value."""

    if type(token) is not str or not token:
        raise MaterializationError("artifact token is required for hydration")
    if snapshot_download_fn is None:
        from huggingface_hub import snapshot_download

        snapshot_download_fn = snapshot_download
    root = _prepare_cache_root(cache_root)
    hydrated: dict[str, Path] = {}
    for adapter in manifest.adapters:
        hydrated[adapter.checkpoint_id] = _materialize_adapter(
            adapter,
            manifest,
            root,
            token=token,
            snapshot_download_fn=snapshot_download_fn,
        )
    return hydrated


def _materialize_adapter(
    adapter: ManifestAdapter,
    manifest: ServingManifest,
    cache_root: str | os.PathLike[str],
    *,
    token: str,
    snapshot_download_fn: Callable[..., str],
) -> Path:
    """hydrate one adapter under a per-digest process lock and atomic rename."""

    root = _prepare_cache_root(cache_root)
    destination = adapter_cache_path(root, adapter)
    lock_path = root / ".locks" / f"{adapter.aggregate_sha256}.lock"
    with _digest_lock(lock_path):
        if _path_exists(destination):
            validate_materialized_adapter(adapter, manifest, destination)
            return destination

        stage = root / "adapters" / f".stage-{adapter.aggregate_sha256}-{uuid.uuid4().hex}"
        stage.mkdir(mode=0o700)
        stage_identity = _directory_identity(stage)
        try:
            hf_cache = root / ".hf-cache"
            patterns = [f"{adapter.source_subfolder}/{entry.path}" for entry in adapter.files]
            emit_boot_progress(
                "artifact-download-starting",
                repo=adapter.repo_id,
                revision=adapter.source_revision,
            )
            try:
                snapshot = snapshot_download_fn(
                    repo_id=adapter.repo_id,
                    repo_type=adapter.repo_type,
                    revision=adapter.source_revision,
                    allow_patterns=patterns,
                    token=token,
                    cache_dir=str(hf_cache),
                )
            except Exception:
                raise MaterializationError("artifact download failed") from None
            _copy_declared_files(Path(snapshot), hf_cache, adapter, stage)
            # settle child-entry metadata before validation captures its first directory identity.
            _synchronize_directory(stage)
            validate_materialized_adapter(adapter, manifest, stage)
            try:
                os.rename(stage, destination)
            except OSError:
                if _path_exists(destination):
                    validate_materialized_adapter(adapter, manifest, destination)
                else:
                    raise
            else:
                # rename changes both the moved directory's inode metadata and its parent's entries.
                _synchronize_directory(destination)
                _synchronize_directory(destination.parent)
            validate_materialized_adapter(adapter, manifest, destination)
            return destination
        except BaseException:
            _clean_invocation_path(stage, root, stage_identity)
            raise
        finally:
            if _path_exists(stage):
                _clean_invocation_path(stage, root, stage_identity)


@contextlib.contextmanager
def locked_manifest_cache(
    manifest: ServingManifest,
    cache_root: str | os.PathLike[str],
) -> Iterator[dict[str, Path]]:
    """hold every digest lock through validation and the caller's registration work."""

    if type(manifest) is not ServingManifest:
        raise MaterializationError("cache validation requires an exact ServingManifest")
    root = _prepare_cache_root(cache_root)
    adapters = tuple(sorted(manifest.adapters, key=lambda adapter: adapter.aggregate_sha256))
    with contextlib.ExitStack() as locks:
        for adapter in adapters:
            lock_path = root / ".locks" / f"{adapter.aggregate_sha256}.lock"
            locks.enter_context(_digest_lock(lock_path))
        paths = {
            adapter.checkpoint_id: adapter_cache_path(root, adapter)
            for adapter in manifest.adapters
        }
        for adapter in manifest.adapters:
            validate_materialized_adapter(adapter, manifest, paths[adapter.checkpoint_id])
        try:
            yield paths
        except BaseException:
            raise
        else:
            for adapter in manifest.adapters:
                validate_materialized_adapter(adapter, manifest, paths[adapter.checkpoint_id])


def validate_manifest_cache(
    manifest: ServingManifest,
    cache_root: str | os.PathLike[str],
) -> dict[str, Path]:
    """revalidate every cached adapter without contacting Hugging Face."""

    with locked_manifest_cache(manifest, cache_root) as paths:
        return dict(paths)


def validate_materialized_adapter(
    adapter: ManifestAdapter,
    manifest: ServingManifest,
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    """fully revalidate exact files, PEFT metadata, and safetensors structure."""

    directory = Path(path)
    before, contents = _read_exact_regular_files(directory, adapter.files)
    config = _load_strict_config(contents[_CONFIG_NAME])
    _validate_adapter_config(config, adapter, manifest)
    validate_adapter_weight_structure(directory / _WEIGHTS_NAME, config, adapter.base_model)
    after, _ = _read_exact_regular_files(directory, adapter.files)
    if after != before:
        raise MaterializationError("adapter cache entry changed during validation")
    return config


def validate_adapter_weight_structure(
    weights_path: str | os.PathLike[str], config: Mapping[str, Any], base_model: str
) -> None:
    """validate the safetensors and LoRA contract shared by resolution and hydration."""

    try:
        tensors = _read_safetensors_tensor_metadata(os.fspath(weights_path))
    except (OSError, ValueError) as exc:
        raise MaterializationError("adapter safetensors structure is invalid") from exc
    if not tensors:
        raise MaterializationError("adapter safetensors contains no tensors")
    _validate_lora_pairs(tensors)
    # the strict reader rejects malformed rank declarations instead of skipping them, which is what
    # a serving load boundary needs. it raises ValueError, so translate like the fused-expert call.
    try:
        declared = strict_declared_lora_ranks(config)
    except ValueError as exc:
        raise MaterializationError("adapter_config.json rank declarations are invalid") from exc
    contradictions = [
        key for key, shape in tensors.items() if lora_tensor_rank_disagrees(key, shape, declared)
    ]
    if contradictions:
        raise MaterializationError(
            "adapter safetensors tensor ranks contradict adapter_config.json"
        )
    if not any(".lora_A." in key for key in tensors) or not any(
        ".lora_B." in key for key in tensors
    ):
        raise MaterializationError("adapter safetensors has no complete LoRA factor evidence")
    if lora_target_parameters(base_model):
        try:
            validate_fused_expert_adapter_config(config, base_model)
        except ValueError as exc:
            raise MaterializationError("fused-expert adapter config is invalid") from exc
        if not has_complete_fused_expert_tensors(tensors, config, base_model):
            raise MaterializationError("fused-expert adapter tensors are incomplete")


@contextlib.contextmanager
def _digest_lock(path: Path):
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise MaterializationError("adapter digest lock could not be opened safely") from exc
    try:
        details = os.fstat(fd)
        # the lock file lives under the cache root, so its directory decides whether the mode it
        # was created with can survive at all.
        _validate_regular_stat(
            details,
            "adapter digest lock",
            enforce_mode=_directory_enforces_permission_bits(path.parent),
        )
        emit_boot_progress("digest-lock-acquiring", path=path)
        fcntl.flock(fd, fcntl.LOCK_EX)
        emit_boot_progress("digest-lock-acquired", path=path)
        if os.fstat(fd) != details:
            raise MaterializationError("adapter digest lock changed while it was opened")
        yield
    finally:
        os.close(fd)


def _prepare_cache_root(cache_root: str | os.PathLike[str]) -> Path:
    raw = os.fspath(cache_root)
    if not raw:
        raise MaterializationError("cache root must not be empty")
    root = Path(os.path.abspath(os.path.expanduser(raw)))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open("/", flags)
    current = Path("/")
    try:
        _validate_cache_ancestor_stat(
            os.fstat(directory_fd),
            "cache ancestor /",
            directory_fd=directory_fd,
        )
        for part in root.parts[1:]:
            try:
                child_fd = os.open(part, flags, dir_fd=directory_fd)
            except OSError as exc:
                if exc.errno != errno.ENOENT:
                    raise MaterializationError("cache root contains an unsafe parent") from exc
                try:
                    os.mkdir(part, mode=0o700, dir_fd=directory_fd)
                except FileExistsError:
                    pass
                except OSError as create_exc:
                    raise MaterializationError(
                        "cache root could not be created safely"
                    ) from create_exc
                try:
                    child_fd = os.open(part, flags, dir_fd=directory_fd)
                except OSError as open_exc:
                    raise MaterializationError("cache root contains an unsafe parent") from open_exc
            os.close(directory_fd)
            directory_fd = child_fd
            current /= part
            _validate_cache_ancestor_stat(
                os.fstat(directory_fd),
                f"cache ancestor {current}",
                directory_fd=directory_fd,
            )
        root_details = os.fstat(directory_fd)
        enforce_mode = _permission_bits_are_enforceable(directory_fd)
        _validate_trusted_directory_stat(root_details, "cache root", enforce_mode=enforce_mode)
        for name in ("adapters", ".locks", ".hf-cache"):
            _ensure_trusted_child_directory(directory_fd, name, flags, enforce_mode=enforce_mode)
    finally:
        os.close(directory_fd)
    return root


def _ensure_trusted_child_directory(
    parent_fd: int,
    name: str,
    flags: int,
    *,
    enforce_mode: bool = True,
) -> None:
    try:
        child_fd = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        if exc.errno != errno.ENOENT:
            raise MaterializationError("cache directory is unsafe") from exc
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
        except OSError as create_exc:
            raise MaterializationError(
                "cache directory could not be created safely"
            ) from create_exc
        try:
            child_fd = os.open(name, flags, dir_fd=parent_fd)
        except OSError as open_exc:
            raise MaterializationError("cache directory is unsafe") from open_exc
    try:
        _validate_trusted_directory_stat(
            os.fstat(child_fd),
            f"cache directory {name}",
            enforce_mode=enforce_mode,
        )
    finally:
        os.close(child_fd)


def _validate_cache_ancestor_stat(
    details: os.stat_result,
    name: str,
    *,
    directory_fd: int | None = None,
) -> None:
    if not stat.S_ISDIR(details.st_mode):
        raise MaterializationError(f"{name} is not a directory")
    owner = details.st_uid
    writable = bool(details.st_mode & 0o022)
    # root-owned shared directories may be sticky or expose fixed world-writable modes. only root can
    # replace their entries, while every inner cache object must still be owned by us.
    # filesystems that cannot store modes rely on the manifest digest and before/after identity
    # checks instead; _permission_bits_are_enforceable distinguishes those mounts below.
    root_shared = owner == 0
    if owner not in {0, os.getuid()}:
        raise MaterializationError(f"{name} is owned by an untrusted uid")
    if not writable or root_shared:
        return
    # probe only at rejection: a mode-preserving mount is unsafe, while a modeless mount reports a
    # writable bit it cannot change and must rely on the content and identity checks instead.
    if directory_fd is not None and not _permission_bits_are_enforceable(directory_fd):
        return
    raise MaterializationError(f"{name} is group or world writable")


def _permission_bits_are_enforceable(directory_fd: int) -> bool:
    """report whether this filesystem preserves a private mode on a probe directory.

    some mounted filesystems report fixed 0777/0666 modes, so a capability probe avoids an
    unsatisfiable assertion without coupling the policy to a provider or filesystem name.
    """

    probe = f".flash-mode-probe-{os.getpid()}"
    with contextlib.suppress(OSError):
        os.rmdir(probe, dir_fd=directory_fd)
    try:
        os.mkdir(probe, mode=0o700, dir_fd=directory_fd)
    except OSError:
        # cannot tell, so assume the strict reading: an unwritable directory fails elsewhere anyway.
        return True
    try:
        try:
            probe_fd = os.open(probe, os.O_RDONLY | _DIRECTORY_FLAGS, dir_fd=directory_fd)
        except OSError:
            return True
        try:
            return not (os.fstat(probe_fd).st_mode & 0o022)
        finally:
            os.close(probe_fd)
    finally:
        # every exit after mkdir must remove the entry before an exact-set validation.
        with contextlib.suppress(OSError):
            os.rmdir(probe, dir_fd=directory_fd)


def _directory_enforces_permission_bits(directory: Path) -> bool:
    """probe a directory by path rather than by descriptor, for callers that only hold a path."""

    try:
        directory_fd = os.open(directory, os.O_RDONLY | _DIRECTORY_FLAGS)
    except OSError:
        return True
    try:
        return _permission_bits_are_enforceable(directory_fd)
    finally:
        os.close(directory_fd)


def _validate_trusted_directory_stat(
    details: os.stat_result,
    name: str,
    *,
    enforce_mode: bool = True,
) -> None:
    if not stat.S_ISDIR(details.st_mode):
        raise MaterializationError(f"{name} is not a directory")
    if details.st_uid != os.getuid():
        raise MaterializationError(f"{name} is not owned by the current uid")
    if enforce_mode and details.st_mode & 0o022:
        raise MaterializationError(f"{name} is group or world writable")


def _validate_regular_stat(
    details: os.stat_result,
    name: str,
    *,
    enforce_mode: bool = True,
) -> None:
    if not stat.S_ISREG(details.st_mode):
        raise MaterializationError(f"{name} is not a regular file")
    if details.st_uid != os.getuid():
        raise MaterializationError(f"{name} is not owned by the current uid")
    if enforce_mode and details.st_mode & 0o022:
        raise MaterializationError(f"{name} is group or world writable")
    if details.st_nlink != 1:
        raise MaterializationError(f"{name} must have exactly one hard link")


def _copy_declared_files(
    snapshot: Path,
    hf_cache: Path,
    adapter: ManifestAdapter,
    stage: Path,
) -> None:
    snapshot_root = snapshot.expanduser().resolve(strict=True)
    cache_root = hf_cache.expanduser().resolve(strict=True)
    try:
        snapshot_root.relative_to(cache_root)
    except ValueError as exc:
        raise MaterializationError(
            "snapshot path escaped the configured Hugging Face cache"
        ) from exc
    source_root = snapshot_root / adapter.source_subfolder
    for declaration in adapter.files:
        source = source_root / declaration.path
        try:
            resolved = source.resolve(strict=True)
            resolved.relative_to(cache_root)
        except (OSError, ValueError) as exc:
            raise MaterializationError("downloaded artifact file escaped the cache") from exc
        source_stat = resolved.stat()
        if not stat.S_ISREG(source_stat.st_mode):
            raise MaterializationError("downloaded artifact entry is not a regular file")
        destination = stage / declaration.path
        _copy_regular_file(resolved, destination)


def _copy_regular_file(source: Path, destination: Path) -> None:
    read_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    write_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    write_flags |= getattr(os, "O_NOFOLLOW", 0)
    source_fd = os.open(source, read_flags)
    destination_fd = os.open(destination, write_flags, 0o600)
    try:
        source_before = os.fstat(source_fd)
        if not stat.S_ISREG(source_before.st_mode):
            raise MaterializationError("downloaded artifact entry is not a regular file")
        with (
            os.fdopen(source_fd, "rb", closefd=False) as source_file,
            os.fdopen(destination_fd, "wb", closefd=False) as destination_file,
        ):
            shutil.copyfileobj(source_file, destination_file, length=_COPY_CHUNK_BYTES)
            destination_file.flush()
            os.fsync(destination_fd)
        if _stable_file_identity(os.fstat(source_fd)) != _stable_file_identity(source_before):
            raise MaterializationError("downloaded artifact changed while it was copied")
    finally:
        os.close(destination_fd)
        os.close(source_fd)


def _synchronize_directory(path: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | _DIRECTORY_FLAGS)
        os.fsync(descriptor)
    except OSError as exc:
        # unsupported directory fsync is a filesystem capability; every other error may mean the
        # populated cache was not persisted, so only the explicit capability errors are ignored.
        if descriptor is None or exc.errno not in _UNSUPPORTED_DIRECTORY_SYNC:
            raise MaterializationError("adapter cache directory could not be synchronized") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


# fuse fsync does not invalidate cached attrs, so force server-backed times at each identity read.
def _forced_directory_times(descriptor: int) -> tuple[int, int] | None:
    statx = getattr(_LIBC, "statx", None)
    if statx is None:
        return None
    buffer = ctypes.create_string_buffer(256)
    flags = _AT_EMPTY_PATH | _AT_STATX_FORCE_SYNC
    if statx(descriptor, b"", flags, _STATX_BASIC_STATS, buffer) != 0:
        error = ctypes.get_errno()
        if error in _UNSUPPORTED_DIRECTORY_SYNC:
            return None
        raise MaterializationError("adapter cache directory metadata could not be synchronized")
    # linux statx has fixed ctime and mtime timestamp offsets in its stable 256-byte abi.
    ctime_seconds = ctypes.c_int64.from_buffer(buffer, 96).value
    ctime_nanos = ctypes.c_uint32.from_buffer(buffer, 104).value
    mtime_seconds = ctypes.c_int64.from_buffer(buffer, 112).value
    mtime_nanos = ctypes.c_uint32.from_buffer(buffer, 120).value
    return mtime_seconds * 1_000_000_000 + mtime_nanos, ctime_seconds * 1_000_000_000 + ctime_nanos


def _open_directory_snapshot(descriptor: int) -> tuple[os.stat_result, tuple[int, ...]]:
    forced_times = _forced_directory_times(descriptor)
    details = os.fstat(descriptor)
    return details, _stable_directory_identity(details, times=forced_times)


def _read_exact_regular_files(
    directory: Path,
    declarations: tuple[ArtifactFile, ...],
) -> tuple[tuple[object, ...], dict[str, bytes]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(directory, flags)
    except OSError as exc:
        raise MaterializationError("adapter cache entry is not a safe directory") from exc
    try:
        directory_stat, directory_identity = _open_directory_snapshot(directory_fd)
        # probe the parent because probing here would mutate the exact entry set under validation.
        enforce_mode = _directory_enforces_permission_bits(directory.parent)
        _validate_trusted_directory_stat(
            directory_stat,
            "adapter cache entry",
            enforce_mode=enforce_mode,
        )
        expected = {entry.path: entry for entry in declarations}
        try:
            names = set(os.listdir(directory_fd))
        except OSError as exc:
            raise MaterializationError("adapter cache entry cannot be inspected") from exc
        if names != set(expected):
            raise MaterializationError("adapter cache file set does not exactly match the manifest")
        snapshots: list[object] = [directory_identity]
        contents: dict[str, bytes] = {}
        for declaration in declarations:
            snapshot, content = _read_declared_file(
                directory_fd,
                declaration,
                enforce_mode=enforce_mode,
            )
            snapshots.append(snapshot)
            if content is not None:
                contents[declaration.path] = content
        if _open_directory_snapshot(directory_fd)[1] != snapshots[0]:
            raise MaterializationError("adapter cache directory changed during validation")
        return tuple(snapshots), contents
    finally:
        os.close(directory_fd)


def _read_declared_file(
    directory_fd: int,
    declaration: ArtifactFile,
    *,
    enforce_mode: bool = True,
) -> tuple[tuple[object, ...], bytes | None]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(declaration.path, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise MaterializationError("adapter cache file could not be opened safely") from exc
    try:
        before = os.fstat(fd)
        _validate_regular_stat(before, "adapter cache file", enforce_mode=enforce_mode)
        if before.st_size != declaration.size:
            raise MaterializationError("adapter cache file size does not match the manifest")
        digest = hashlib.sha256()
        captured = bytearray() if declaration.path == _CONFIG_NAME else None
        while chunk := os.read(fd, _COPY_CHUNK_BYTES):
            digest.update(chunk)
            if captured is not None:
                captured.extend(chunk)
        after = os.fstat(fd)
        if _stable_file_identity(after) != _stable_file_identity(before):
            raise MaterializationError("adapter cache file changed while it was read")
        actual_digest = digest.hexdigest()
        if actual_digest != declaration.sha256:
            raise MaterializationError("adapter cache file digest does not match the manifest")
        snapshot = (declaration.path, *_stable_file_identity(after), actual_digest)
        return snapshot, None if captured is None else bytes(captured)
    finally:
        os.close(fd)


def _stable_directory_identity(
    details: os.stat_result,
    *,
    times: tuple[int, int] | None = None,
) -> tuple[int, ...]:
    mtime_ns, ctime_ns = times or (details.st_mtime_ns, details.st_ctime_ns)
    return (
        details.st_dev,
        details.st_ino,
        details.st_mode,
        details.st_uid,
        details.st_nlink,
        mtime_ns,
        ctime_ns,
    )


def _stable_file_identity(details: os.stat_result) -> tuple[int, ...]:
    return (*_stable_directory_identity(details), details.st_size)


def _load_strict_config(raw: bytes) -> dict[str, Any]:
    try:
        config = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_non_finite_json_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise MaterializationError("adapter_config.json is not strict utf-8 json") from exc
    if type(config) is not dict:
        raise MaterializationError("adapter_config.json must contain a json object")
    return config


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MaterializationError("adapter_config.json contains a duplicate key")
        result[key] = value
    return result


def _validate_lora_pairs(tensors: Mapping[str, tuple[int, ...]]) -> None:
    pairs: dict[tuple[str, str], dict[str, tuple[int, ...]]] = {}
    for key, shape in tensors.items():
        matches = [
            (factor, infix)
            for factor, infix in (("A", ".lora_A."), ("B", ".lora_B."))
            if infix in key
        ]
        if len(matches) != 1:
            raise MaterializationError("adapter safetensors contains a non-LoRA tensor")
        factor, infix = matches[0]
        module, _, leaf = key.partition(infix)
        # the adapter name between the infix and "weight" is optional. peft writes
        # "<module>.lora_A.<adapter>.weight" when saving a named adapter, but verl's model_merger
        # -- which is what produces every adapter flash serves -- writes "<module>.lora_A.weight"
        # with no name at all. demanding exactly two leaf parts rejected every real flash adapter
        # while accepting the hand-built ".default.weight" fixtures, so the whole materialize
        # suite passed against a key shape flash does not actually produce. the adapter name is
        # only part of the pairing identity, so an absent one pairs under "".
        leaf_parts = leaf.split(".")
        if not module or not leaf_parts or leaf_parts[-1] != "weight" or len(leaf_parts) > 2:
            raise MaterializationError("adapter safetensors contains a malformed LoRA tensor key")
        adapter_name = leaf_parts[0] if len(leaf_parts) == 2 else ""
        if len(leaf_parts) == 2 and not adapter_name:
            raise MaterializationError("adapter safetensors contains a malformed LoRA tensor key")
        pair = pairs.setdefault((module, adapter_name), {})
        if factor in pair:
            raise MaterializationError("adapter safetensors contains duplicate LoRA factors")
        pair[factor] = shape
    if not pairs:
        raise MaterializationError("adapter safetensors has no LoRA tensor pairs")
    for pair in pairs.values():
        if set(pair) != {"A", "B"}:
            raise MaterializationError(
                "adapter safetensors contains an incomplete LoRA tensor pair"
            )
        a_shape = pair["A"]
        b_shape = pair["B"]
        # peft writes a convolution LoRA as `A (r, in_ch, *kernel)` / `B (out_ch, r, 1, 1, 1)`,
        # so factors are not always 2-D. flash's own `all-linear` training produces exactly one
        # such pair on image models -- `visual.patch_embed.proj` is a Conv3d -- and demanding
        # rank-2 rejected the whole adapter over it, making every image adapter undeployable.
        # vllm loads the pair, warns that it cannot wrap a convolution, and leaves it unapplied,
        # so refusing to serve the other 346 pairs is strictly worse than what the engine does.
        # the rank agreement below holds for both shapes: rank is `A[0]` and `B[1]` either way.
        if (
            len(a_shape) < 2
            or len(b_shape) < 2
            or any(dimension <= 0 for dimension in (*a_shape, *b_shape))
            or a_shape[0] != b_shape[1]
        ):
            raise MaterializationError(
                "adapter safetensors contains incompatible LoRA factor shapes"
            )


def _validate_adapter_config(
    config: Mapping[str, Any],
    adapter: ManifestAdapter,
    manifest: ServingManifest,
) -> None:
    if config.get("peft_type") != "LORA":
        raise MaterializationError("adapter_config.json peft_type must be LORA")
    if config.get("task_type") not in {None, "CAUSAL_LM"}:
        raise MaterializationError("adapter_config.json task_type must be absent or CAUSAL_LM")
    if config.get("base_model_name_or_path") != adapter.base_model:
        raise MaterializationError("adapter logical base model does not match the manifest")
    revision = config.get("revision")
    if revision is not None and revision != "" and revision != adapter.base_model_revision:
        raise MaterializationError("adapter logical base revision does not match the manifest")
    modules_to_save = config.get("modules_to_save")
    if modules_to_save is not None and modules_to_save != []:
        raise MaterializationError("modules_to_save adapters are not supported")
    try:
        declared_rank = rank_from_adapter_config(config, source="adapter_config.json")
    except ValueError as exc:
        raise MaterializationError("adapter_config.json has invalid LoRA rank metadata") from exc
    if declared_rank != adapter.lora_rank:
        raise MaterializationError("adapter declared rank does not match the manifest")
    if declared_rank > manifest.engine.max_lora_rank:
        raise MaterializationError("adapter declared rank exceeds the engine rank ceiling")


def _path_exists(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise MaterializationError("cache path could not be inspected safely") from exc
    return True


def _directory_identity(path: Path) -> tuple[int, int]:
    try:
        details = os.lstat(path)
    except OSError as exc:
        raise MaterializationError("cache directory could not be inspected safely") from exc
    _validate_trusted_directory_stat(
        details,
        "cache directory",
        # probe the parent: probing a stage would mutate the exact entry set being validated, and a
        # delayed fuse removal could then reject an otherwise valid adapter.
        enforce_mode=_directory_enforces_permission_bits(path.parent),
    )
    return details.st_dev, details.st_ino


def _clean_invocation_path(
    path: Path,
    root: Path,
    expected_identity: tuple[int, int],
) -> None:
    with contextlib.suppress(OSError, MaterializationError):
        parent_identity = _directory_identity(path.parent)
        expected_parent_identity = _directory_identity(root / "adapters")
        if parent_identity != expected_parent_identity or not path.name.startswith(".stage-"):
            return
        details = os.lstat(path)
        if (
            not stat.S_ISDIR(details.st_mode)
            or (details.st_dev, details.st_ino) != expected_identity
        ):
            return
        shutil.rmtree(path)
