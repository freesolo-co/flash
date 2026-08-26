"""shared pid 1 launcher for one immutable serving deployment."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import sys
import threading
import uuid
from collections.abc import MutableMapping
from pathlib import Path
from types import FrameType, SimpleNamespace
from typing import Any, Protocol, Self

from .progress import emit_boot_progress, emit_filesystem_usage

_MANIFEST_ENV = "FLASH_SERVING_MANIFEST"
_MANIFEST_ID_ENV = "FLASH_SERVING_MANIFEST_ID"
_IMAGE_DIGEST_ENV = "FLASH_SERVING_IMAGE_DIGEST"
_CACHE_ROOT_ENV = "FLASH_SERVING_CACHE_ROOT"
_HOST_ENV = "FLASH_SERVING_HOST"
_PORT_ENV = "FLASH_SERVING_PORT"
_INFERENCE_TOKEN_ENV = "FLASH_INFERENCE_TOKEN"
_ARTIFACT_TOKEN_ENV = "FLASH_ARTIFACT_TOKEN"
_INFERENCE_TOKEN_FD_ENV = "FLASH_INFERENCE_TOKEN_FD"
_ARTIFACT_TOKEN_FD_ENV = "FLASH_ARTIFACT_TOKEN_FD"
_DEFAULT_CACHE_ROOT = "/var/lib/flash-serving"
# subdirectory of the cache root that holds vllm's compiled-graph cache.
_VLLM_CACHE_DIRNAME = "vllm"
_DEFAULT_HOST = "0.0.0.0"
_DEFAULT_PORT = 8000
_CHILD_STOP_TIMEOUT_SECONDS = 5.0
_BOOTSTRAP_PATH = "/app/serve_launch.py"
_SAFE_CHILD_COPIED_ENV_NAMES = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "FLASH_SERVING_CACHE_ROOT",
        "FLASH_SERVING_HOST",
        "FLASH_SERVING_IMAGE_DIGEST",
        "FLASH_SERVING_MANIFEST",
        "FLASH_SERVING_MANIFEST_ID",
        "FLASH_SERVING_PORT",
        # the hub cache location, so the child engine reads the weights bootstrap hydrated into
        # the volume instead of looking in ephemeral storage and reaching for the network.
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_HUB_DISABLE_XET",
        # a child imports the hub fresh, so it reads this from the environment rather than
        # inheriting the constant this process already flipped.
        "HF_HUB_OFFLINE",
        "LANG",
        "LC_ALL",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_VISIBLE_DEVICES",
        "TZ",
        # the compiled-graph cache, so the child engine reuses the volume's graphs instead of
        # recompiling the model into ephemeral storage on every start.
        "VLLM_CACHE_ROOT",
    }
)
_FIXED_CHILD_ENVIRONMENT = {
    "PATH": "/opt/flash-venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
    "PYTHONUNBUFFERED": "1",
}
# engine settings the vllm process needs regardless of which provider started the container.
#
# without the spawn method vllm forks its EngineCore after this process has already touched cuda,
# and the child dies with "Cannot re-initialize CUDA in forked subprocess" before anything binds a
# port. externally that is another silent boot failure, indistinguishable from a slow image pull.
#
# they belong to the runtime rather than to a provider image: the engine's requirements do not
# change with the machine that rented the gpu, and a second provider must not have to rediscover
# them. applied to the running process and to the child, since the engine reads them at import.
_ENGINE_RUNTIME_ENVIRONMENT = {
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    # the xet download path intermittently fails engine init, so force the http downloader.
    "HF_HUB_DISABLE_XET": "1",
    # expandable segments reduce fragmentation from repeated adapter load and unload.
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    # keep deepgemm out of the fp8 fused-moe backend selection.
    "VLLM_MOE_USE_DEEP_GEMM": "0",
}

ServingRuntimeSecrets: Any = None
MaterializationError: Any = None
adapter_cache_path: Any = None
base_weights_are_cached: Any = None
base_weights_cache_path: Any = None
decode_manifest_environment: Any = None
hydrate_base_weights: Any = None
hydrate_manifest: Any = None
validate_manifest_cache: Any = None
_prepare_cache_root: Any = None
_serve: Any = None


class LaunchError(RuntimeError):
    """the sanitized serving launcher rejected its startup inputs."""


class StartupTerminated(LaunchError):
    """startup was interrupted before the runtime assumed signal ownership."""

    def __init__(self, exit_code: int) -> None:
        self.exit_code = exit_code
        super().__init__("serving startup was terminated")


class _PopenFactory(Protocol):
    def __call__(self, args: list[str], **kwargs: Any) -> subprocess.Popen[Any]: ...


class _SignalGuardHandoff(Protocol):
    def release_for_handoff(self) -> dict[int, Any]: ...


def _load_project_boundaries() -> None:
    global MaterializationError
    global ServingRuntimeSecrets
    global _prepare_cache_root
    global _serve
    global adapter_cache_path
    global base_weights_are_cached
    global base_weights_cache_path
    global decode_manifest_environment
    global hydrate_base_weights
    global hydrate_manifest
    global validate_manifest_cache

    from flash.serve.app.__main__ import _serve as loaded_serve
    from flash.serve.app.materialize import (
        MaterializationError as LoadedMaterializationError,
    )
    from flash.serve.app.materialize import (
        _prepare_cache_root as loaded_prepare_cache_root,
    )
    from flash.serve.app.materialize import adapter_cache_path as loaded_adapter_cache_path
    from flash.serve.app.materialize import (
        base_weights_are_cached as loaded_base_weights_are_cached,
    )
    from flash.serve.app.materialize import (
        base_weights_cache_path as loaded_base_weights_cache_path,
    )
    from flash.serve.app.materialize import hydrate_base_weights as loaded_hydrate_base_weights
    from flash.serve.app.materialize import hydrate_manifest as loaded_hydrate_manifest
    from flash.serve.app.materialize import (
        validate_manifest_cache as loaded_validate_manifest_cache,
    )
    from flash.serve.provisioning import ServingRuntimeSecrets as LoadedServingRuntimeSecrets
    from flash.serve.provisioning import (
        decode_manifest_environment as loaded_decode_manifest_environment,
    )

    if MaterializationError is None:
        MaterializationError = LoadedMaterializationError
    if ServingRuntimeSecrets is None:
        ServingRuntimeSecrets = LoadedServingRuntimeSecrets
    if _prepare_cache_root is None:
        _prepare_cache_root = loaded_prepare_cache_root
    if _serve is None:
        _serve = loaded_serve
    if adapter_cache_path is None:
        adapter_cache_path = loaded_adapter_cache_path
    if base_weights_are_cached is None:
        base_weights_are_cached = loaded_base_weights_are_cached
    if base_weights_cache_path is None:
        base_weights_cache_path = loaded_base_weights_cache_path
    if decode_manifest_environment is None:
        decode_manifest_environment = loaded_decode_manifest_environment
    if hydrate_base_weights is None:
        hydrate_base_weights = loaded_hydrate_base_weights
    if hydrate_manifest is None:
        hydrate_manifest = loaded_hydrate_manifest
    if validate_manifest_cache is None:
        validate_manifest_cache = loaded_validate_manifest_cache


def _required_environment(environment: MutableMapping[str, str], name: str) -> str:
    value = environment.get(name)
    if type(value) is not str or not value:
        raise LaunchError(f"{name} is required")
    return value


def _read_secret_descriptor(raw_fd: str | None, name: str) -> str | None:
    if raw_fd is None:
        return None
    if not raw_fd.isdecimal():
        raise LaunchError(f"{name} descriptor is invalid")
    try:
        with os.fdopen(int(raw_fd), "r", encoding="utf-8", closefd=True) as source:
            value = source.read().strip()
    except OSError as exc:
        raise LaunchError(f"{name} descriptor could not be read") from exc
    if not value:
        raise LaunchError(f"{name} descriptor was empty")
    return value


def _close_raw_descriptor(raw_fd: str | None) -> None:
    if raw_fd is not None and raw_fd.isdecimal():
        with contextlib.suppress(OSError):
            os.close(int(raw_fd))


def _validate_secret(value: str | None, name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if type(value) is not str or not value or value != value.strip():
        raise LaunchError(f"{name} is invalid")
    return value


def _pop_runtime_secrets(
    environment: MutableMapping[str, str],
) -> tuple[str, str | None]:
    inference = environment.pop(_INFERENCE_TOKEN_ENV, None)
    artifact = environment.pop(_ARTIFACT_TOKEN_ENV, None)
    inference_fd = environment.pop(_INFERENCE_TOKEN_FD_ENV, None)
    artifact_fd = environment.pop(_ARTIFACT_TOKEN_FD_ENV, None)
    try:
        if inference is not None and inference_fd is not None:
            raise LaunchError("inference token has multiple sources")
        if artifact is not None and artifact_fd is not None:
            raise LaunchError("artifact token has multiple sources")
        if inference is None:
            selected_fd, inference_fd = inference_fd, None
            inference = _read_secret_descriptor(selected_fd, "inference token")
        if artifact is None:
            selected_fd, artifact_fd = artifact_fd, None
            artifact = _read_secret_descriptor(selected_fd, "artifact token")
        inference = _validate_secret(inference, "inference token")
        artifact = _validate_secret(artifact, "artifact token", optional=True)
        assert inference is not None
        return inference, artifact
    finally:
        _close_raw_descriptor(inference_fd)
        _close_raw_descriptor(artifact_fd)


class _StartupSignalGuard:
    def __init__(self) -> None:
        self._previous: dict[int, Any] = {}
        self._active = False

    def __enter__(self) -> Self:
        for signum in (signal.SIGTERM, signal.SIGINT):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._terminate)
        self._active = True
        return self

    def _terminate(self, signum: int, _frame: FrameType | None) -> None:
        raise StartupTerminated(128 + signum)

    def restore(self) -> None:
        if not self._active:
            return
        for signum, previous in self._previous.items():
            signal.signal(signum, previous)
        self._active = False

    def adopt_previous(self, previous_guard: _SignalGuardHandoff) -> None:
        self._previous = previous_guard.release_for_handoff()

    def release_for_handoff(self) -> dict[int, Any]:
        previous = dict(self._previous)
        self._active = False
        return previous

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.restore()


def _verify_external_bindings(manifest: Any, environment: MutableMapping[str, str]) -> None:
    manifest_id = _required_environment(environment, _MANIFEST_ID_ENV)
    image_digest = _required_environment(environment, _IMAGE_DIGEST_ENV)
    if manifest.manifest_id != manifest_id:
        raise LaunchError("serving manifest id does not match its external binding")
    if manifest.expected_oci_digest != image_digest:
        raise LaunchError("serving image digest does not match its external binding")


def _atomic_write_manifest(manifest: Any, cache_root: str) -> Path:
    # the walk below opens, creates and mode-probes every component of the cache root on the
    # provider's network volume, so it is the first place startup can block on remote storage.
    emit_boot_progress("cache-root-preparing", path=cache_root)
    root = _prepare_cache_root(cache_root)
    emit_boot_progress("cache-root-prepared", path=root)
    target = root / "serving-manifest.json"
    temporary_name = f".serving-manifest-{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(root, directory_flags)
    temporary_fd: int | None = None
    try:
        temporary_fd = os.open(temporary_name, flags, 0o600, dir_fd=directory_fd)
        payload = manifest.canonical_json().encode("utf-8")
        with os.fdopen(temporary_fd, "wb", closefd=False) as output:
            output.write(payload)
            output.flush()
            os.fsync(temporary_fd)
        os.replace(temporary_name, target.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=directory_fd)
        raise
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        os.close(directory_fd)
    emit_boot_progress("manifest-written", path=target, manifest_id=manifest.manifest_id)
    return target


def _cache_has_missing_adapter(manifest: Any, cache_root: str) -> bool:
    for adapter in manifest.adapters:
        try:
            os.lstat(adapter_cache_path(cache_root, adapter))
        except FileNotFoundError:
            return True
        except OSError:
            return False
    return False


def _write_all(fd: int, value: str) -> None:
    payload = value.encode("utf-8")
    offset = 0
    try:
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise OSError("secret pipe write failed")
            offset += written
    except OSError as exc:
        raise LaunchError("secret descriptor could not be populated") from exc


@contextlib.contextmanager
def _secret_descriptor(value: str):
    read_fd, write_fd = os.pipe()
    errors: list[BaseException] = []
    value_holder = [value]
    del value

    def populate() -> None:
        secret = value_holder.pop()
        try:
            _write_all(write_fd, secret)
        except BaseException as exc:
            errors.append(exc)
        finally:
            # the writer may block until a large value is consumed, so drop its final raw reference
            # as soon as the write finishes rather than retaining it for the long-lived with-body.
            del secret
            with contextlib.suppress(OSError):
                os.close(write_fd)

    writer = threading.Thread(target=populate, name="flash-secret-pipe", daemon=True)
    writer.start()
    try:
        yield read_fd
    finally:
        with contextlib.suppress(OSError):
            os.close(read_fd)
        writer.join()
    if errors:
        raise LaunchError("secret descriptor could not be populated") from errors[0]


def _prepare_cache(manifest: Any, cache_root: str, artifact_token: str | None) -> None:
    """make both the adapters and the base weights present before the engine starts.

    the served base model is a private repo, and the artifact token is deleted at the end of
    bootstrap, so the weights have to reach the persistent volume while the token still exists.
    every later start -- including the finalized redeploy that carries no token at all -- then
    resolves them offline from that volume.
    """

    emit_boot_progress("cache-validation-starting", path=cache_root)
    adapters_missing = True
    try:
        validate_manifest_cache(manifest, cache_root)
        adapters_missing = False
        emit_boot_progress("adapters-validated", count=len(manifest.adapters))
    except MaterializationError as exc:
        if not _cache_has_missing_adapter(manifest, cache_root):
            raise LaunchError("serving cache validation failed") from exc
    weights_missing = not base_weights_are_cached(manifest, cache_root)
    if not adapters_missing and not weights_missing:
        emit_boot_progress("cache-validated", result="hit")
        return
    emit_boot_progress(
        "hydration-starting",
        path=cache_root,
        repo=manifest.engine.served_model,
        revision=manifest.engine.model_revision,
        weights=weights_missing,
        adapters=adapters_missing,
    )
    if artifact_token is None:
        raise LaunchError("artifact token is required when serving cache hydration is missing")
    try:
        if weights_missing:
            with _secret_descriptor(artifact_token) as token_fd:
                hydrate_base_weights(manifest, cache_root, token_fd=token_fd)
        if adapters_missing:
            with _secret_descriptor(artifact_token) as token_fd:
                hydrate_manifest(manifest, cache_root, token_fd=token_fd)
            emit_boot_progress("adapters-validated", count=len(manifest.adapters))
    except MaterializationError as exc:
        raise LaunchError("serving cache hydration failed") from exc
    emit_boot_progress("hydration-complete", path=cache_root)


def _bind_hub_cache(environment: MutableMapping[str, str], cache_root: str) -> None:
    """point huggingface_hub at the persistent volume for both hydration and engine start.

    hydration and vllm are separate processes, and vllm resolves the base model through the hub's
    own cache lookup. without this they use different directories: hydration writes to the volume
    and the engine then looks in ephemeral container storage, finds nothing, and tries the network
    with no token. setting it here -- before hydration -- keeps writer and reader on one path.
    """

    hub_cache = str(base_weights_cache_path(cache_root))
    # vllm's compile cache defaults to ephemeral container storage, so leaving this unset makes
    # every cold start recompile the model. it belongs here rather than in a provider image for
    # the same reason as the engine settings above: it derives from the cache root the runtime
    # already knows, and a second provider must not have to rediscover it.
    vllm_cache = str(Path(cache_root) / _VLLM_CACHE_DIRNAME)
    environment["HF_HOME"] = cache_root
    environment["HF_HUB_CACHE"] = hub_cache
    environment["VLLM_CACHE_ROOT"] = vllm_cache
    # the download path this image uses; xet intermittently fails engine init on a cold volume.
    environment["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HOME"] = cache_root
    os.environ["HF_HUB_CACHE"] = hub_cache
    os.environ["VLLM_CACHE_ROOT"] = vllm_cache
    os.environ["HF_HUB_DISABLE_XET"] = "1"


def _seal_hub_offline(environment: MutableMapping[str, str]) -> None:
    """stop the engine from reaching the hub once the cache is complete.

    vllm probes the served repo for optional files -- an image processor config among them -- and
    passes no local_files_only, so those calls default to the hub's own offline flag. the served repo is private and the artifact token is deleted at the end of
    bootstrap, so the probe authenticates as nobody, and a private repo answers "not found"
    rather than "no such file". transformers re-raises exactly that as a hard OSError even though
    the caller asked it not to raise for missing entries, so an optional file kills startup.

    the environment variable alone is not enough. huggingface_hub reads it once at import and
    binds the result to a module constant, and hydration has already imported it by the time this
    runs, so setting only the variable is inert. the constant is what vllm actually reads, through
    a live attribute lookup, so both are set: the constant for this process, the variable for any
    child that imports the hub fresh.

    this must run after hydration. the flag disables outgoing traffic globally, so setting it any
    earlier would break the very download that fills the cache.
    """

    environment["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        import huggingface_hub.constants as hub_constants

        hub_constants.HF_HUB_OFFLINE = True
    except Exception as exc:
        # the engine cannot serve without the hub library, so a failure here is not recoverable
        # by continuing: it would reach the network with no token and fail later and less clearly.
        raise LaunchError("hub offline mode could not be applied") from exc


def _apply_engine_runtime_environment(environment: MutableMapping[str, str]) -> None:
    """set the engine settings this process and its children need, on every provider.

    vllm reads these at import time, so they have to be in place before the engine is constructed
    rather than passed to it. the serving process starts the engine itself, so setting only the
    child environment would leave the in-process path on the fork start method and reproduce the
    cuda re-initialization failure this exists to prevent.
    """

    for key, value in _ENGINE_RUNTIME_ENVIRONMENT.items():
        environment[key] = value
        os.environ[key] = value


def _port(environment: MutableMapping[str, str]) -> int:
    raw = environment.get(_PORT_ENV, str(_DEFAULT_PORT))
    if not raw.isdecimal():
        raise LaunchError("serving port is invalid")
    port = int(raw)
    if not 0 < port <= 65535:
        raise LaunchError("serving port is invalid")
    return port


def _run_with_secrets(
    environment: MutableMapping[str, str],
    raw_inference: str,
    raw_artifact: str | None,
) -> tuple[str, SimpleNamespace, Any]:
    """hydrate while the artifact credential exists, then return only serving inputs."""

    # first marker of the process. everything below it -- the cache-root walk in particular --
    # touches the provider's network volume, so without a marker here a hang there is
    # indistinguishable from a container that never ran python at all.
    emit_boot_progress("launcher-entered")
    _load_project_boundaries()
    try:
        secrets = ServingRuntimeSecrets(raw_inference, raw_artifact)
    except ValueError as exc:
        raise LaunchError("runtime secret input is invalid") from exc
    encoded_manifest = _required_environment(environment, _MANIFEST_ENV)
    try:
        manifest = decode_manifest_environment(encoded_manifest)
    except ValueError as exc:
        raise LaunchError("encoded serving manifest is invalid") from exc
    _verify_external_bindings(manifest, environment)
    cache_root = environment.get(_CACHE_ROOT_ENV, _DEFAULT_CACHE_ROOT)
    if type(cache_root) is not str or not cache_root:
        raise LaunchError("serving cache root is invalid")
    _atomic_write_manifest(manifest, cache_root)
    _apply_engine_runtime_environment(environment)
    _bind_hub_cache(environment, cache_root)
    inference_token, artifact_token = secrets._reveal_for_launch()
    _prepare_cache(manifest, cache_root, artifact_token)
    emit_filesystem_usage("cache-prepared", cache_root)
    _seal_hub_offline(environment)
    args = SimpleNamespace(
        cache_root=cache_root,
        host=environment.get(_HOST_ENV, _DEFAULT_HOST),
        port=_port(environment),
    )
    return inference_token, args, manifest


def _prepare_explicit_secrets(
    environment: MutableMapping[str, str],
    inference_token: str,
    artifact_token: str | None,
) -> tuple[str, SimpleNamespace, Any]:
    inference = _validate_secret(inference_token, "inference token")
    artifact = _validate_secret(artifact_token, "artifact token", optional=True)
    assert inference is not None
    return _run_with_secrets(environment, inference, artifact)


def _run_prepared(prepared: list[Any], startup_signals: _StartupSignalGuard) -> None:
    inference_token = prepared.pop(0)
    args, manifest = prepared
    emit_boot_progress("entering-serve", host=args.host, port=args.port)
    with _secret_descriptor(inference_token) as inference_fd:
        # the descriptor owns the write now, so this frame must not retain the raw token while serving.
        del inference_token
        asyncio.run(
            _serve(
                args,
                manifest,
                inference_token_fd=inference_fd,
                on_signals_installed=startup_signals.release_for_handoff,
            )
        )


def run_launcher_with_secrets(
    inference_token_holder: list[str],
    artifact_token_holder: list[str | None],
    *,
    environment: MutableMapping[str, str] | None = None,
    previous_signal_guard: _SignalGuardHandoff | None = None,
) -> None:
    """accept bootstrap-popped secrets under a nested startup signal guard."""

    environment = os.environ if environment is None else environment
    with _StartupSignalGuard() as startup_signals:
        if previous_signal_guard is not None:
            startup_signals.adopt_previous(previous_signal_guard)
        if type(inference_token_holder) is not list or len(inference_token_holder) != 1:
            raise LaunchError("inference token holder is invalid")
        if type(artifact_token_holder) is not list or len(artifact_token_holder) != 1:
            raise LaunchError("artifact token holder is invalid")
        inference_token = inference_token_holder.pop()
        artifact_token = artifact_token_holder.pop()
        prepared = list(_prepare_explicit_secrets(environment, inference_token, artifact_token))
        # strings cannot be zeroed, so the only meaningful boundary is removing every live
        # bootstrap reference before the long-lived server frame exists.
        del inference_token
        del artifact_token
        _run_prepared(prepared, startup_signals)


def _scrub_child_environment(environment: MutableMapping[str, str]) -> dict[str, str]:
    child = dict(_FIXED_CHILD_ENVIRONMENT)
    child.update(
        {key: environment[key] for key in _SAFE_CHILD_COPIED_ENV_NAMES if key in environment}
    )
    # after the copy, so the engine's requirements win over an inherited value. a provider image
    # that already sets these agrees with them; one that sets something else would break the child.
    child.update(_ENGINE_RUNTIME_ENVIRONMENT)
    return child


def _close_descriptors(descriptors: list[int]) -> None:
    for fd in descriptors:
        with contextlib.suppress(OSError):
            os.close(fd)


def _terminate_and_reap(process: subprocess.Popen[Any]) -> None:
    with contextlib.suppress(Exception):
        process.terminate()
    try:
        process.wait(timeout=_CHILD_STOP_TIMEOUT_SECONDS)
        return
    except Exception:
        pass
    with contextlib.suppress(Exception):
        process.kill()
    try:
        process.wait(timeout=_CHILD_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(Exception):
            process.wait()
    except Exception:
        pass


def start_launcher_process(
    environment: MutableMapping[str, str] | None = None,
    *,
    popen_factory: _PopenFactory = subprocess.Popen,
) -> subprocess.Popen[Any]:
    """start the launcher with secrets only on inherited numeric descriptors."""

    environment = os.environ if environment is None else environment
    raw_inference, raw_artifact = _pop_runtime_secrets(environment)
    _load_project_boundaries()
    try:
        secrets = ServingRuntimeSecrets(raw_inference, raw_artifact)
    except ValueError as exc:
        raise LaunchError("runtime secret input is invalid") from exc
    inference_token, artifact_token = secrets._reveal_for_launch()
    child_environment = _scrub_child_environment(environment)
    read_fds: list[int] = []
    write_fds: list[int] = []
    process: subprocess.Popen[Any] | None = None
    try:
        inference_read, inference_write = os.pipe()
        read_fds.append(inference_read)
        write_fds.append(inference_write)
        child_environment[_INFERENCE_TOKEN_FD_ENV] = str(inference_read)
        if artifact_token is not None:
            artifact_read, artifact_write = os.pipe()
            read_fds.append(artifact_read)
            write_fds.append(artifact_write)
            child_environment[_ARTIFACT_TOKEN_FD_ENV] = str(artifact_read)
        process = popen_factory(
            [sys.executable, _BOOTSTRAP_PATH],
            env=child_environment,
            pass_fds=tuple(read_fds),
            close_fds=True,
        )
        _close_descriptors(read_fds)
        read_fds.clear()
        _write_all(write_fds[0], inference_token)
        if artifact_token is not None:
            _write_all(write_fds[1], artifact_token)
        return process
    except BaseException:
        if process is not None:
            _terminate_and_reap(process)
        raise
    finally:
        _close_descriptors(read_fds)
        _close_descriptors(write_fds)
