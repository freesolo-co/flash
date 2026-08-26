"""shared launcher cache, descriptor, binding, and process boundaries."""

from __future__ import annotations

import gc
import hashlib
import inspect
import os
import subprocess
import sys
import time
import types
import uuid
import weakref
from pathlib import Path

import pytest

from flash.serve.app import launch
from flash.serve.app.manifest import build_serving_manifest
from flash.serve.app.materialize import MaterializationError, read_artifact_token_fd
from flash.serve.provisioning import encode_manifest_environment
from tests.test_serve_app_manifest import _spec_and_inputs

INFERENCE_TOKEN = "inference-token-sentinel"
ARTIFACT_TOKEN = "artifact-token-sentinel"


def _manifest():
    return build_serving_manifest(*_spec_and_inputs())


def _environment(tmp_path: Path, *, artifact: bool = True) -> dict[str, str]:
    manifest = _manifest()
    environment = {
        "FLASH_SERVING_MANIFEST": encode_manifest_environment(manifest),
        "FLASH_SERVING_MANIFEST_ID": manifest.manifest_id,
        "FLASH_SERVING_IMAGE_DIGEST": manifest.expected_oci_digest,
        "FLASH_SERVING_CACHE_ROOT": str(tmp_path / "cache"),
        "FLASH_INFERENCE_TOKEN": INFERENCE_TOKEN,
    }
    if artifact:
        environment["FLASH_ARTIFACT_TOKEN"] = ARTIFACT_TOKEN
    return environment


def _run_with_environment(environment: dict[str, str]) -> None:
    inference_token = environment.pop("FLASH_INFERENCE_TOKEN")
    artifact_token = environment.pop("FLASH_ARTIFACT_TOKEN", None)
    launch.run_launcher_with_secrets([inference_token], [artifact_token], environment=environment)


@pytest.fixture(autouse=True)
def _base_weights_present(monkeypatch):
    """treat the base weights as already on the volume unless a test says otherwise.

    these tests all run against an empty tmp cache and are about adapters, descriptors, and
    signals. the real lookup would call huggingface_hub for every one of them, so it is stubbed
    here rather than reaching the network. the tests that own base-weight behaviour replace this
    with their own stub, which takes precedence because monkeypatch applies in call order.
    """

    monkeypatch.setattr(launch, "base_weights_are_cached", lambda *_a, **_k: True)
    monkeypatch.setattr(
        launch,
        "hydrate_base_weights",
        lambda *_a, **_k: pytest.fail("base weights were already present"),
    )


def _successful_serve(captured: dict[str, object]):
    async def serve(args, manifest, *, inference_token_fd: int, on_signals_installed):
        captured["args"] = args
        captured["manifest"] = manifest
        captured["inference_fd"] = inference_token_fd
        captured["inference_token"] = read_artifact_token_fd(inference_token_fd)
        for signum in (launch.signal.SIGTERM, launch.signal.SIGINT):
            launch.signal.signal(signum, lambda *_args: None)
        restore_handlers = on_signals_installed()
        for signum, handler in restore_handlers.items():
            launch.signal.signal(signum, handler)

    return serve


def test_launcher_uses_cache_first_without_reading_artifact_token(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    environment = _environment(tmp_path, artifact=False)
    calls: list[str] = []
    captured: dict[str, object] = {}

    def validate(manifest, cache_root):
        calls.append("validate")
        return {manifest.adapters[0].checkpoint_id: Path(cache_root) / "cached"}

    def hydrate(*_args, **_kwargs):
        raise AssertionError("cache-first restart must not hydrate")

    real_seal_hub_offline = launch._seal_hub_offline

    def seal_hub_offline(environment):
        calls.append("seal")
        real_seal_hub_offline(environment)

    monkeypatch.setattr(launch, "validate_manifest_cache", validate)
    monkeypatch.setattr(launch, "hydrate_manifest", hydrate)
    monkeypatch.setattr(
        launch,
        "emit_filesystem_usage",
        lambda stage, cache_root: calls.append(f"usage:{stage}:{cache_root}"),
    )
    monkeypatch.setattr(launch, "_seal_hub_offline", seal_hub_offline)
    monkeypatch.setattr(launch, "_serve", _successful_serve(captured))

    _run_with_environment(environment)

    assert calls == [
        "validate",
        f"usage:cache-prepared:{environment['FLASH_SERVING_CACHE_ROOT']}",
        "seal",
    ]
    assert captured["inference_token"] == INFERENCE_TOKEN
    assert "FLASH_INFERENCE_TOKEN" not in environment
    assert "FLASH_ARTIFACT_TOKEN" not in environment
    assert (tmp_path / "cache" / "serving-manifest.json").read_text() == (
        _manifest().canonical_json()
    )
    output = capsys.readouterr().out
    assert output.index("phase=adapters-validated") < output.index(
        'phase=cache-validated result="hit"'
    )
    assert INFERENCE_TOKEN not in output
    assert ARTIFACT_TOKEN not in output


def test_bootstrap_handoff_installs_inner_guard_before_restoring_outer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path, artifact=False)
    environment.pop("FLASH_INFERENCE_TOKEN")
    observed = False

    previous_handlers = {
        signum: launch.signal.getsignal(signum)
        for signum in (launch.signal.SIGTERM, launch.signal.SIGINT)
    }

    class PreviousGuard:
        def release_for_handoff(self) -> dict[int, object]:
            nonlocal observed
            handler = launch.signal.getsignal(launch.signal.SIGTERM)
            assert getattr(handler, "__self__", None).__class__ is launch._StartupSignalGuard
            observed = True
            return previous_handlers

    monkeypatch.setattr(launch, "validate_manifest_cache", lambda *_args: {})
    monkeypatch.setattr(launch, "_serve", _successful_serve({}))

    launch.run_launcher_with_secrets(
        [INFERENCE_TOKEN],
        [None],
        environment=environment,
        previous_signal_guard=PreviousGuard(),
    )

    assert observed is True


def test_launcher_streams_secret_values_larger_than_pipe_capacity(
    monkeypatch,
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path, artifact=False)
    large_token = "x" * (128 * 1024)
    environment["FLASH_INFERENCE_TOKEN"] = large_token
    captured: dict[str, object] = {}
    monkeypatch.setattr(launch, "validate_manifest_cache", lambda *_args: {})
    monkeypatch.setattr(launch, "_serve", _successful_serve(captured))

    _run_with_environment(environment)

    assert captured["inference_token"] == large_token


def test_launcher_cache_miss_requires_artifact_token_before_serving(
    monkeypatch,
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path, artifact=False)
    serve_calls = 0

    def missing(*_args):
        raise MaterializationError("missing cache")

    async def serve(*_args, **_kwargs):
        nonlocal serve_calls
        serve_calls += 1

    monkeypatch.setattr(launch, "validate_manifest_cache", missing)
    monkeypatch.setattr(launch, "_serve", serve)

    with pytest.raises(launch.LaunchError, match="artifact token is required") as exc_info:
        _run_with_environment(environment)

    assert serve_calls == 0
    assert INFERENCE_TOKEN not in str(exc_info.value)
    assert "FLASH_INFERENCE_TOKEN" not in environment


def test_launcher_hydrates_missing_cache_through_closed_descriptor(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    environment = _environment(tmp_path)
    captured: dict[str, object] = {}

    def missing(*_args):
        raise MaterializationError("missing cache")

    def hydrate(_manifest, _cache_root, *, token_fd: int):
        captured["artifact_fd"] = token_fd
        captured["artifact_token"] = read_artifact_token_fd(token_fd)
        return {}

    monkeypatch.setattr(launch, "validate_manifest_cache", missing)
    monkeypatch.setattr(launch, "hydrate_manifest", hydrate)
    monkeypatch.setattr(launch, "_serve", _successful_serve(captured))

    _run_with_environment(environment)

    assert captured["artifact_token"] == ARTIFACT_TOKEN
    assert captured["inference_token"] == INFERENCE_TOKEN
    for key in ("artifact_fd", "inference_fd"):
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(captured[key])
    output = capsys.readouterr().out
    phases = [
        "cache-root-prepared",
        "manifest-written",
        "cache-validation-starting",
        "hydration-starting",
        "adapters-validated",
        "hydration-complete",
        "entering-serve",
    ]
    positions = [output.index(f"phase={phase}") for phase in phases]
    assert positions == sorted(positions)
    assert f'repo="{_manifest().engine.served_model}"' in output
    assert f'revision="{_manifest().engine.model_revision}"' in output
    assert f'path="{tmp_path / "cache"}"' in output
    assert all(line.startswith("flash-serving boot elapsed=") for line in output.splitlines())
    assert INFERENCE_TOKEN not in output
    assert ARTIFACT_TOKEN not in output


def _value_contains_secret(value: object, fingerprint: bytes, seen: set[int]) -> bool:
    if isinstance(value, str):
        return hashlib.sha256(value.encode()).digest() == fingerprint
    if value is None or isinstance(value, (bool, int, float, bytes)):
        return False
    identity = id(value)
    if identity in seen:
        return False
    seen.add(identity)
    if isinstance(value, dict):
        return any(
            _value_contains_secret(item, fingerprint, seen)
            for pair in value.items()
            for item in pair
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_value_contains_secret(item, fingerprint, seen) for item in value)
    if isinstance(value, (types.ModuleType, types.FunctionType, types.MethodType, type)):
        return False
    namespace = getattr(value, "__dict__", None)
    if isinstance(namespace, dict) and _value_contains_secret(namespace, fingerprint, seen):
        return True
    for cls in type(value).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            try:
                child = getattr(value, slot)
            except (AttributeError, TypeError):
                continue
            if _value_contains_secret(child, fingerprint, seen):
                return True
    return False


def _secret_reachability_probe(
    fingerprint: bytes,
    expected_frames: set[str],
    *,
    inference_fingerprint: bytes | None = None,
):
    async def serve(_args, _manifest, *, inference_token_fd: int, on_signals_installed):
        inference_token = read_artifact_token_fd(inference_token_fd)
        if inference_fingerprint is None:
            assert inference_token == INFERENCE_TOKEN
        else:
            assert hashlib.sha256(inference_token.encode()).digest() == inference_fingerprint
        del inference_token
        frames: dict[str, object] = {}
        frame = inspect.currentframe()
        while frame is not None:
            if frame.f_code.co_filename.endswith(
                ("/flash/serve/app/launch.py", "/serve_launch.py")
            ):
                frames[frame.f_code.co_name] = frame
            frame = frame.f_back

        assert expected_frames <= frames.keys(), frames.keys()
        assert "_run_with_secrets" not in frames
        assert "_prepare_explicit_secrets" not in frames
        for name, active in frames.items():
            for local_name, value in active.f_locals.items():
                assert not _value_contains_secret(value, fingerprint, set()), (
                    f"artifact token remains reachable from {name}.{local_name}"
                )

        for signum in (launch.signal.SIGTERM, launch.signal.SIGINT):
            launch.signal.signal(signum, lambda *_args: None)
        restore_handlers = on_signals_installed()
        for signum, handler in restore_handlers.items():
            launch.signal.signal(signum, handler)

    return serve


def _unpredictable_artifact_token() -> tuple[str, bytes]:
    token = f"artifact-{uuid.uuid4().hex}"
    return token, hashlib.sha256(token.encode()).digest()


def test_secret_descriptor_releases_its_source_after_the_consumer_reads() -> None:
    class WeakToken(str):
        __slots__ = ("__weakref__",)

    expected = f"inference-{uuid.uuid4().hex}"
    token = WeakToken(expected)
    token_reference = weakref.ref(token)

    with launch._secret_descriptor(token) as token_fd:
        del token
        assert read_artifact_token_fd(token_fd) == expected
        deadline = time.monotonic() + 1
        while token_reference() is not None and time.monotonic() < deadline:
            gc.collect()
            time.sleep(0.001)
        assert token_reference() is None


def test_secret_descriptor_chains_the_original_write_failure(monkeypatch) -> None:
    failure = OSError("write failed")

    def fail_write(_fd: int, _value: str) -> None:
        raise failure

    monkeypatch.setattr(launch, "_write_all", fail_write)
    with (
        pytest.raises(
            launch.LaunchError, match=r"^secret descriptor could not be populated$"
        ) as exc_info,
        launch._secret_descriptor(INFERENCE_TOKEN) as token_fd,
    ):
        assert os.read(token_fd, 1) == b""

    assert exc_info.value.__cause__ is failure


def test_explicit_launcher_drops_the_artifact_token_before_the_server_loop(
    monkeypatch, tmp_path: Path
) -> None:
    artifact_token, fingerprint = _unpredictable_artifact_token()
    environment = _environment(tmp_path, artifact=False)
    environment.pop("FLASH_INFERENCE_TOKEN")
    monkeypatch.setattr(launch, "validate_manifest_cache", lambda *_args: {})
    monkeypatch.setattr(
        launch,
        "_serve",
        _secret_reachability_probe(fingerprint, {"run_launcher_with_secrets", "_run_prepared"}),
    )

    holder = [artifact_token]
    del artifact_token
    launch.run_launcher_with_secrets([INFERENCE_TOKEN], holder, environment=environment)


def test_root_bootstrap_drops_the_inference_token_before_the_server_loop(
    monkeypatch, tmp_path: Path
) -> None:
    import runpy

    inference_token = f"inference-{uuid.uuid4().hex}"
    fingerprint = hashlib.sha256(inference_token.encode()).digest()
    environment = _environment(tmp_path, artifact=False)
    environment["FLASH_INFERENCE_TOKEN"] = inference_token
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    del inference_token
    monkeypatch.setattr(launch, "validate_manifest_cache", lambda *_args: {})
    monkeypatch.setattr(
        launch,
        "_serve",
        _secret_reachability_probe(
            fingerprint,
            {"_run", "run_launcher_with_secrets", "_run_prepared"},
            inference_fingerprint=fingerprint,
        ),
    )
    bootstrap = runpy.run_path(
        Path(__file__).resolve().parents[1] / "serve_launch.py",
        run_name="serve_launch_inference_retention_probe",
    )

    bootstrap["_run"]()


def test_root_bootstrap_drops_the_artifact_token_before_the_server_loop(
    monkeypatch, tmp_path: Path
) -> None:
    import runpy

    artifact_token, fingerprint = _unpredictable_artifact_token()
    environment = _environment(tmp_path)
    environment["FLASH_ARTIFACT_TOKEN"] = artifact_token
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(launch, "validate_manifest_cache", lambda *_args: {})
    monkeypatch.setattr(
        launch,
        "_serve",
        _secret_reachability_probe(fingerprint, {"_run", "_run_prepared"}),
    )
    bootstrap = runpy.run_path(
        Path(__file__).resolve().parents[1] / "serve_launch.py",
        run_name="serve_launch_retention_probe",
    )

    bootstrap["_run"]()


def test_bootstrap_hydrates_base_weights_while_the_artifact_token_still_exists(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # the served base model is a private repo, so vllm cannot fetch it anonymously. the artifact
    # token exists only during bootstrap, so if the weights are not pulled onto the volume here
    # they can never be pulled at all: the finalized redeploy carries no token. every offline test
    # passed because they all stubbed adapter hydration and no test asserted on the base model,
    # so both providers deployed successfully and then died inside the container with
    # "Freesolo-Co/Qwen3.5-4B-FP8 is not a local folder", after billing had already started.
    environment = _environment(tmp_path)
    captured: dict[str, object] = {}

    def weights_absent(*_args, **_kwargs):
        return False

    def hydrate_weights(_manifest, _cache_root, *, token_fd: int):
        captured["weights_token"] = read_artifact_token_fd(token_fd)

    monkeypatch.setattr(launch, "validate_manifest_cache", lambda *_args: {})
    monkeypatch.setattr(launch, "base_weights_are_cached", weights_absent)
    monkeypatch.setattr(launch, "hydrate_base_weights", hydrate_weights)
    monkeypatch.setattr(launch, "_serve", _successful_serve(captured))

    _run_with_environment(environment)

    assert captured["weights_token"] == ARTIFACT_TOKEN


def test_missing_base_weights_without_a_token_fail_before_the_engine_starts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # a finalized container with a cold volume cannot recover: it has no token, and vllm would
    # otherwise reach the network anonymously and fail deep inside engine construction. failing
    # here keeps the reason attributable instead of surfacing as a transformers OSError.
    environment = _environment(tmp_path, artifact=False)
    serve_calls = 0

    async def serve(*_args, **_kwargs):
        nonlocal serve_calls
        serve_calls += 1

    monkeypatch.setattr(launch, "validate_manifest_cache", lambda *_args: {})
    monkeypatch.setattr(launch, "base_weights_are_cached", lambda *_a, **_k: False)
    monkeypatch.setattr(launch, "_serve", serve)

    with pytest.raises(launch.LaunchError, match="artifact token is required"):
        _run_with_environment(environment)

    assert serve_calls == 0


def test_cached_base_weights_are_not_downloaded_again_on_a_warm_restart(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # scale-from-zero is the common case and carries no artifact token. re-downloading would make
    # every cold start pay for the full base model, and on a finalized container it would fail.
    environment = _environment(tmp_path, artifact=False)
    captured: dict[str, object] = {}

    monkeypatch.setattr(launch, "validate_manifest_cache", lambda *_args: {})
    monkeypatch.setattr(launch, "base_weights_are_cached", lambda *_a, **_k: True)
    monkeypatch.setattr(
        launch,
        "hydrate_base_weights",
        lambda *_a, **_k: pytest.fail("a warm volume must not re-download the base weights"),
    )
    monkeypatch.setattr(launch, "_serve", _successful_serve(captured))

    _run_with_environment(environment)

    assert captured["inference_token"] == INFERENCE_TOKEN


def test_hub_cache_is_bound_to_the_volume_for_the_engine_child(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # hydration and vllm are different processes. hydration writes into the volume's hub cache,
    # and the engine resolves the base model through huggingface_hub's own cache lookup, so if
    # the two disagree the engine finds nothing and tries the network with no token. the child
    # environment is scrubbed to an allowlist, so the names must survive that scrub too.
    environment = _environment(tmp_path, artifact=False)
    cache_root = environment["FLASH_SERVING_CACHE_ROOT"]
    captured: dict[str, object] = {}

    monkeypatch.setattr(launch, "validate_manifest_cache", lambda *_args: {})
    monkeypatch.setattr(launch, "base_weights_are_cached", lambda *_a, **_k: True)
    monkeypatch.setattr(launch, "_serve", _successful_serve(captured))

    _run_with_environment(environment)

    assert environment["HF_HOME"] == cache_root
    assert environment["HF_HUB_CACHE"] == str(launch.base_weights_cache_path(cache_root))
    scrubbed = launch._scrub_child_environment(environment)
    assert scrubbed["HF_HUB_CACHE"] == str(launch.base_weights_cache_path(cache_root))


def test_engine_runtime_environment_is_applied_on_every_provider(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # vllm forks its EngineCore, and the serving process has already touched cuda by then, so
    # without the spawn start method the child dies with "Cannot re-initialize CUDA in forked
    # subprocess" before anything binds a port. measured on a live runpod l4.
    #
    # these were previously set only by the modal image, so modal served and runpod did not. the
    # launcher must set them itself, because it is the only piece both providers share.
    environment = _environment(tmp_path, artifact=False)
    captured: dict[str, object] = {}

    monkeypatch.setattr(launch, "validate_manifest_cache", lambda *_args: {})
    monkeypatch.setattr(launch, "base_weights_are_cached", lambda *_a, **_k: True)
    monkeypatch.setattr(launch, "_serve", _successful_serve(captured))
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)

    _run_with_environment(environment)

    # the engine runs in this process, so the live environment is what actually decides the start
    # method. asserting only the child mapping would pass while the real path still forked.
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert environment["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert launch._scrub_child_environment(environment)["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_engine_runtime_environment_overrides_a_conflicting_inherited_value(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # a provider image that sets fork would otherwise win over the runtime and reintroduce the
    # crash, so the engine's requirement has to be applied after anything inherited.
    environment = _environment(tmp_path, artifact=False)
    environment["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
    captured: dict[str, object] = {}

    monkeypatch.setattr(launch, "validate_manifest_cache", lambda *_args: {})
    monkeypatch.setattr(launch, "base_weights_are_cached", lambda *_a, **_k: True)
    monkeypatch.setattr(launch, "_serve", _successful_serve(captured))

    _run_with_environment(environment)

    assert environment["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert launch._scrub_child_environment(environment)["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_compile_cache_is_bound_to_the_volume_on_every_provider(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # vllm's compile cache defaults to ephemeral container storage, so an unset cache root makes
    # every cold start recompile the model. it was set only by the modal image, like the engine
    # settings above, so runpod recompiled on each start while modal reused the volume's graphs.
    environment = _environment(tmp_path, artifact=False)
    captured: dict[str, object] = {}

    monkeypatch.setattr(launch, "validate_manifest_cache", lambda *_args: {})
    monkeypatch.setattr(launch, "base_weights_are_cached", lambda *_a, **_k: True)
    monkeypatch.setattr(launch, "_serve", _successful_serve(captured))
    monkeypatch.delenv("VLLM_CACHE_ROOT", raising=False)

    _run_with_environment(environment)

    expected = str(Path(environment["FLASH_SERVING_CACHE_ROOT"]) / "vllm")
    # under the cache root, so the graphs land on the persistent volume rather than in the
    # container. the engine runs in-process, so os.environ is the mapping that actually decides it.
    assert os.environ["VLLM_CACHE_ROOT"] == expected
    assert environment["VLLM_CACHE_ROOT"] == expected
    # and the child must inherit it, or a forked engine recompiles into ephemeral storage anyway.
    assert launch._scrub_child_environment(environment)["VLLM_CACHE_ROOT"] == expected


def test_engine_start_is_sealed_offline_only_after_hydration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # vllm probes the served repo for optional files and passes no local_files_only, so those
    # calls follow the hub's offline flag. the served repo is private and the token is gone by
    # then, so the probe authenticates as nobody and a private repo answers "not found" rather
    # than "no such file" -- which transformers turns into a hard OSError that kills startup.
    # both live canary arms died exactly there.
    environment = _environment(tmp_path)
    order: list[str] = []
    captured: dict[str, object] = {}

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(launch, "validate_manifest_cache", lambda *_args: {})
    monkeypatch.setattr(launch, "base_weights_are_cached", lambda *_a, **_k: False)

    def hydrate(*_args, **_kwargs):
        # the flag disables outgoing traffic globally, so sealing before this point would break
        # the very download that fills the cache. hydration must still see it unset.
        order.append("hydrate")
        assert os.environ.get("HF_HUB_OFFLINE") is None

    async def serve(*_args, **_kwargs):
        import huggingface_hub.constants as hub_constants

        order.append("serve")
        # the environment variable alone is inert: huggingface_hub binds it to a module constant
        # at import, and hydration already imported it. the constant is what vllm reads.
        captured["constant"] = hub_constants.HF_HUB_OFFLINE
        captured["variable"] = os.environ.get("HF_HUB_OFFLINE")

    monkeypatch.setattr(launch, "hydrate_base_weights", hydrate)
    monkeypatch.setattr(launch, "hydrate_manifest", hydrate)
    monkeypatch.setattr(launch, "_serve", serve)

    _run_with_environment(environment)

    assert order == ["hydrate", "serve"]
    assert captured["constant"] is True
    assert captured["variable"] == "1"
    # a child imports the hub fresh, so it needs the name to survive the allowlist scrub.
    assert launch._scrub_child_environment(environment)["HF_HUB_OFFLINE"] == "1"


def test_sigterm_during_hydration_aborts_and_closes_startup_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    captured_fd: int | None = None
    serve_calls = 0

    def missing(*_args):
        raise MaterializationError("missing cache")

    def terminate_during_hydration(*_args, token_fd: int, **_kwargs):
        nonlocal captured_fd
        captured_fd = token_fd
        os.kill(os.getpid(), launch.signal.SIGTERM)

    async def serve(*_args, **_kwargs):
        nonlocal serve_calls
        serve_calls += 1

    monkeypatch.setattr(launch, "validate_manifest_cache", missing)
    monkeypatch.setattr(launch, "hydrate_manifest", terminate_during_hydration)
    monkeypatch.setattr(launch, "_serve", serve)

    with pytest.raises(launch.StartupTerminated, match="startup was terminated") as exc_info:
        _run_with_environment(environment)

    assert exc_info.value.exit_code == 128 + launch.signal.SIGTERM
    assert serve_calls == 0
    assert captured_fd is not None
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(captured_fd)
    assert list((tmp_path / "cache").glob(".serving-manifest-*.tmp")) == []
    assert "FLASH_INFERENCE_TOKEN" not in environment
    assert "FLASH_ARTIFACT_TOKEN" not in environment


def test_sigint_during_bootstrap_closes_inference_descriptor(
    monkeypatch,
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path, artifact=False)
    captured_fd: int | None = None

    async def terminate_during_bootstrap(
        _args,
        _manifest,
        *,
        inference_token_fd: int,
        on_signals_installed,
    ):
        nonlocal captured_fd
        captured_fd = inference_token_fd
        assert on_signals_installed
        os.kill(os.getpid(), launch.signal.SIGINT)

    monkeypatch.setattr(launch, "validate_manifest_cache", lambda *_args: {})
    monkeypatch.setattr(launch, "_serve", terminate_during_bootstrap)

    with pytest.raises(launch.StartupTerminated, match="startup was terminated") as exc_info:
        _run_with_environment(environment)

    assert exc_info.value.exit_code == 128 + launch.signal.SIGINT
    assert captured_fd is not None
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(captured_fd)
    assert list((tmp_path / "cache").glob(".serving-manifest-*.tmp")) == []


def test_sigterm_during_hydration_aborts_in_subprocess(tmp_path: Path) -> None:
    cache_root = tmp_path / "subprocess-cache"
    probe = f"""
import os
from flash.serve.app import launch
from flash.serve.app.manifest import build_serving_manifest
from flash.serve.app.materialize import MaterializationError
from flash.serve.provisioning import encode_manifest_environment
from tests.test_serve_app_manifest import _spec_and_inputs

manifest = build_serving_manifest(*_spec_and_inputs())
environment = {{
    "FLASH_SERVING_MANIFEST": encode_manifest_environment(manifest),
    "FLASH_SERVING_MANIFEST_ID": manifest.manifest_id,
    "FLASH_SERVING_IMAGE_DIGEST": manifest.expected_oci_digest,
    "FLASH_SERVING_CACHE_ROOT": {str(cache_root)!r},
    "FLASH_INFERENCE_TOKEN": "subprocess-inference-sentinel",
    "FLASH_ARTIFACT_TOKEN": "subprocess-artifact-sentinel",
}}
launch._load_project_boundaries()


def missing(*_args):
    raise MaterializationError("missing")


def terminate(*_args, **_kwargs):
    os.kill(os.getpid(), launch.signal.SIGTERM)


async def serve(*_args, **_kwargs):
    raise AssertionError("serve must not run")


launch.validate_manifest_cache = missing
launch.hydrate_manifest = terminate
# this probe is about the adapter hydration signal window, so the base weights are declared
# present rather than reaching huggingface_hub from a subprocess with an empty cache.
launch.base_weights_are_cached = lambda *_a, **_k: True
launch._serve = serve
try:
    inference_token = environment.pop("FLASH_INFERENCE_TOKEN")
    artifact_token = environment.pop("FLASH_ARTIFACT_TOKEN", None)
    launch.run_launcher_with_secrets([inference_token], [artifact_token], environment=environment)
except launch.StartupTerminated:
    pass
else:
    raise AssertionError("startup termination was not raised")
assert "FLASH_INFERENCE_TOKEN" not in environment
assert "FLASH_ARTIFACT_TOKEN" not in environment
assert not list(__import__("pathlib").Path({str(cache_root)!r}).glob(".serving-manifest-*.tmp"))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert INFERENCE_TOKEN not in result.stdout + result.stderr
    assert ARTIFACT_TOKEN not in result.stdout + result.stderr


def test_sigterm_after_bootstrap_before_uvicorn_capture_has_no_window(tmp_path: Path) -> None:
    cache_root = tmp_path / "handoff-cache"
    probe = f"""
import os
from flash.serve.app import __main__ as app_main
from flash.serve.app import launch
from flash.serve.app.manifest import build_serving_manifest
from flash.serve.provisioning import encode_manifest_environment
from tests.test_serve_app_manifest import _spec_and_inputs

manifest = build_serving_manifest(*_spec_and_inputs())
environment = {{
    "FLASH_SERVING_MANIFEST": encode_manifest_environment(manifest),
    "FLASH_SERVING_MANIFEST_ID": manifest.manifest_id,
    "FLASH_SERVING_IMAGE_DIGEST": manifest.expected_oci_digest,
    "FLASH_SERVING_CACHE_ROOT": {str(cache_root)!r},
    "FLASH_INFERENCE_TOKEN": "handoff-inference-sentinel",
}}
state = {{"closed": False, "bootstrap_complete": False}}


class Owner:
    async def close(self):
        state["closed"] = True


async def bootstrap(*_args, **_kwargs):
    state["bootstrap_complete"] = True
    return Owner()


def create_app(*_args, **_kwargs):
    assert state["bootstrap_complete"] is True
    os.kill(os.getpid(), launch.signal.SIGTERM)
    raise AssertionError("signal handler did not abort app construction")


launch._load_project_boundaries()
launch.validate_manifest_cache = lambda *_args: {{}}
# this probe is about the signal window between bootstrap and uvicorn, so the base weights are
# declared present rather than reaching huggingface_hub from a subprocess with an empty cache.
launch.base_weights_are_cached = lambda *_a, **_k: True
app_main.bootstrap_serving = bootstrap
app_main.create_app = create_app
launch._serve = app_main._serve
try:
    inference_token = environment.pop("FLASH_INFERENCE_TOKEN")
    artifact_token = environment.pop("FLASH_ARTIFACT_TOKEN", None)
    launch.run_launcher_with_secrets([inference_token], [artifact_token], environment=environment)
except launch.StartupTerminated as exc:
    assert exc.exit_code == 128 + launch.signal.SIGTERM
else:
    raise AssertionError("startup termination was not raised")
assert state == {{"closed": True, "bootstrap_complete": True}}
assert "FLASH_INFERENCE_TOKEN" not in environment
assert not list(__import__("pathlib").Path({str(cache_root)!r}).glob(".serving-manifest-*.tmp"))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "handoff-inference-sentinel" not in result.stdout + result.stderr


def test_launcher_does_not_rehydrate_a_present_corrupt_cache(monkeypatch, tmp_path: Path) -> None:
    # launch.adapter_cache_path is a deferred global: it stays None until the boundaries load, so
    # reaching through `launch` without this call only works when some earlier test in the same
    # process happened to load them first. that ordering dependency made this test pass alone and
    # fail under -n auto, which is the configuration ci runs.
    launch._load_project_boundaries()
    environment = _environment(tmp_path)
    manifest = _manifest()
    destination = launch.adapter_cache_path(tmp_path / "cache", manifest.adapters[0])
    destination.mkdir(parents=True, mode=0o700)
    (tmp_path / "cache").chmod(0o700)
    (tmp_path / "cache" / "adapters").chmod(0o700)
    destination.chmod(0o700)
    hydrate_calls = 0

    def corrupt(*_args):
        raise MaterializationError("corrupt cache")

    def hydrate(*_args, **_kwargs):
        nonlocal hydrate_calls
        hydrate_calls += 1

    monkeypatch.setattr(launch, "validate_manifest_cache", corrupt)
    monkeypatch.setattr(launch, "hydrate_manifest", hydrate)

    with pytest.raises(launch.LaunchError, match="cache validation failed"):
        _run_with_environment(environment)
    assert hydrate_calls == 0


def test_launcher_rejects_external_binding_before_cache_or_serving(
    monkeypatch,
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    environment["FLASH_SERVING_MANIFEST_ID"] = "0" * 64
    monkeypatch.setattr(
        launch,
        "validate_manifest_cache",
        lambda *_args: (_ for _ in ()).throw(AssertionError("cache must not be read")),
    )

    with pytest.raises(launch.LaunchError, match="external binding"):
        _run_with_environment(environment)
    assert not (tmp_path / "cache").exists()


def test_atomic_manifest_write_replaces_complete_file_and_cleans_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    cache = tmp_path / "cache"
    path = launch._atomic_write_manifest(manifest, str(cache))
    assert path.read_text() == manifest.canonical_json()
    assert path.stat().st_mode & 0o777 == 0o600

    real_replace = launch.os.replace

    def fail_replace(*_args, **_kwargs):
        raise OSError("replace failed")

    monkeypatch.setattr(launch.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        launch._atomic_write_manifest(manifest, str(cache))
    assert list(cache.glob(".serving-manifest-*.tmp")) == []
    assert path.read_text() == manifest.canonical_json()
    monkeypatch.setattr(launch.os, "replace", real_replace)


class _FakeProcess:
    def __init__(self, *, resist_terminate: bool = False) -> None:
        self.terminated = False
        self.killed = False
        self.resist_terminate = resist_terminate
        self.wait_calls: list[float | None] = []

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        if self.resist_terminate and not self.killed:
            raise launch.subprocess.TimeoutExpired("launcher", timeout)
        return -9 if self.killed else -15


def test_parent_helper_scrubs_environment_argv_and_closes_descriptors(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    environment.update(
        {
            "PROVIDER_API_KEY": "provider-secret-sentinel",
            "FLASH_SERVING_PROVIDER_SECRET": "prefixed-secret-sentinel",
            "PATH": "/tmp/injected-bin",
            "PYTHONPATH": "/tmp/injected-python",
            "PYTHONHOME": "/tmp/injected-home",
            "LD_PRELOAD": "/tmp/injected.so",
            "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/tmp/injected-native",
            "VLLM_WORKER_MULTIPROC_METHOD": "arbitrary-module-control",
        }
    )
    captured: dict[str, object] = {}
    process = _FakeProcess()

    def popen(args, **kwargs):
        captured["args"] = args
        captured["env"] = kwargs["env"]
        captured["pass_fds"] = kwargs["pass_fds"]
        captured["close_fds"] = kwargs["close_fds"]
        captured["reader_copies"] = tuple(os.dup(fd) for fd in kwargs["pass_fds"])
        return process

    returned = launch.start_launcher_process(environment, popen_factory=popen)

    assert returned is process
    assert captured["args"] == [sys.executable, "/app/serve_launch.py"]
    encoded_call = repr(captured["args"]) + repr(captured["env"])
    assert INFERENCE_TOKEN not in encoded_call
    assert ARTIFACT_TOKEN not in encoded_call
    assert "provider-secret-sentinel" not in encoded_call
    assert "prefixed-secret-sentinel" not in encoded_call
    for rejected in (
        "PROVIDER_API_KEY",
        "FLASH_SERVING_PROVIDER_SECRET",
        "PYTHONPATH",
        "PYTHONHOME",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
    ):
        assert rejected not in captured["env"]
    # the engine settings are not inherited from the ambient environment either: the child gets the
    # runtime's own fixed value, whatever the provider image happened to set. the launcher applies
    # these because without the spawn start method vllm's forked EngineCore cannot re-initialize
    # cuda, which is what killed every runpod deployment while modal's image happened to set it.
    assert environment["VLLM_WORKER_MULTIPROC_METHOD"] == "arbitrary-module-control"
    assert captured["env"]["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert captured["env"]["FLASH_SERVING_MANIFEST_ID"] == _manifest().manifest_id
    assert captured["env"]["FLASH_SERVING_CACHE_ROOT"] == str(tmp_path / "cache")
    assert captured["env"]["PATH"] == (
        "/opt/flash-venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
    )
    assert captured["close_fds"] is True
    assert len(captured["pass_fds"]) == 2
    assert read_artifact_token_fd(captured["reader_copies"][0]) == INFERENCE_TOKEN
    assert read_artifact_token_fd(captured["reader_copies"][1]) == ARTIFACT_TOKEN
    for fd in captured["pass_fds"]:
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(fd)
    assert "FLASH_INFERENCE_TOKEN" not in environment
    assert "FLASH_ARTIFACT_TOKEN" not in environment


def test_parent_helper_closes_every_descriptor_when_process_start_fails(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    passed: tuple[int, ...] = ()

    def fail_start(_args, **kwargs):
        nonlocal passed
        passed = kwargs["pass_fds"]
        raise OSError("start failed")

    with pytest.raises(OSError, match="start failed") as exc_info:
        launch.start_launcher_process(environment, popen_factory=fail_start)
    assert INFERENCE_TOKEN not in str(exc_info.value)
    assert ARTIFACT_TOKEN not in str(exc_info.value)
    for fd in passed:
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(fd)


def test_parent_helper_terminates_child_and_cleans_up_on_pipe_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    process = _FakeProcess()
    passed: tuple[int, ...] = ()

    def popen(_args, **kwargs):
        nonlocal passed
        passed = kwargs["pass_fds"]
        return process

    def fail_write(_fd: int, _value: str) -> None:
        raise launch.LaunchError("secret descriptor could not be populated")

    monkeypatch.setattr(launch, "_write_all", fail_write)
    with pytest.raises(launch.LaunchError, match="could not be populated") as exc_info:
        launch.start_launcher_process(environment, popen_factory=popen)
    assert process.terminated is True
    assert process.killed is False
    assert process.wait_calls == [launch._CHILD_STOP_TIMEOUT_SECONDS]
    assert INFERENCE_TOKEN not in str(exc_info.value)
    for fd in passed:
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(fd)


def test_parent_helper_kills_and_reaps_a_termination_resistant_child(
    monkeypatch,
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    process = _FakeProcess(resist_terminate=True)

    monkeypatch.setattr(
        launch,
        "_write_all",
        lambda *_args: (_ for _ in ()).throw(launch.LaunchError("pipe failed")),
    )

    with pytest.raises(launch.LaunchError, match="pipe failed") as exc_info:
        launch.start_launcher_process(environment, popen_factory=lambda *_args, **_kwargs: process)

    assert process.terminated is True
    assert process.killed is True
    assert process.wait_calls == [
        launch._CHILD_STOP_TIMEOUT_SECONDS,
        launch._CHILD_STOP_TIMEOUT_SECONDS,
    ]
    assert INFERENCE_TOKEN not in str(exc_info.value)
    assert ARTIFACT_TOKEN not in str(exc_info.value)


def test_secret_source_conflict_closes_every_inherited_descriptor(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    inference_read, inference_write = os.pipe()
    artifact_read, artifact_write = os.pipe()
    os.close(inference_write)
    os.close(artifact_write)
    environment["FLASH_INFERENCE_TOKEN_FD"] = str(inference_read)
    environment["FLASH_ARTIFACT_TOKEN_FD"] = str(artifact_read)

    with pytest.raises(launch.LaunchError, match="multiple sources"):
        launch.start_launcher_process(
            environment,
            popen_factory=lambda *_args, **_kwargs: pytest.fail("must reject before process start"),
        )

    for fd in (inference_read, artifact_read):
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(fd)


def test_launcher_errors_and_output_never_include_plaintext_secrets(
    capsys,
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    environment["FLASH_SERVING_MANIFEST"] = "not-base64"

    with pytest.raises(launch.LaunchError, match="encoded serving manifest is invalid") as exc_info:
        _run_with_environment(environment)

    captured = capsys.readouterr()
    rendered = str(exc_info.value) + repr(exc_info.value) + captured.out + captured.err
    assert INFERENCE_TOKEN not in rendered
    assert ARTIFACT_TOKEN not in rendered
    assert "FLASH_INFERENCE_TOKEN" not in environment
    assert "FLASH_ARTIFACT_TOKEN" not in environment
