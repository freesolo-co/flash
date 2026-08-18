"""shared launcher cache, descriptor, binding, and process boundaries."""

from __future__ import annotations

import os
import subprocess
import sys
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
    monkeypatch, tmp_path: Path
) -> None:
    environment = _environment(tmp_path, artifact=False)
    calls: list[str] = []
    captured: dict[str, object] = {}

    def validate(manifest, cache_root):
        calls.append("validate")
        return {manifest.adapters[0].adapter_revision: Path(cache_root) / "cached"}

    def hydrate(*_args, **_kwargs):
        raise AssertionError("cache-first restart must not hydrate")

    monkeypatch.setattr(launch, "validate_manifest_cache", validate)
    monkeypatch.setattr(launch, "hydrate_manifest", hydrate)
    monkeypatch.setattr(launch, "_serve", _successful_serve(captured))

    launch.run_launcher(environment)

    assert calls == ["validate"]
    assert captured["inference_token"] == INFERENCE_TOKEN
    assert "FLASH_INFERENCE_TOKEN" not in environment
    assert "FLASH_ARTIFACT_TOKEN" not in environment
    assert (tmp_path / "cache" / "serving-manifest.json").read_text() == (
        _manifest().canonical_json()
    )


def test_direct_launcher_installs_signal_guard_before_secret_access(
    monkeypatch,
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path, artifact=False)
    real_pop = launch._pop_runtime_secrets
    observed = False

    def pop_with_signal_assertion(selected_environment):
        nonlocal observed
        handler = launch.signal.getsignal(launch.signal.SIGTERM)
        assert getattr(handler, "__self__", None).__class__ is launch._StartupSignalGuard
        observed = True
        return real_pop(selected_environment)

    monkeypatch.setattr(launch, "_pop_runtime_secrets", pop_with_signal_assertion)
    monkeypatch.setattr(launch, "validate_manifest_cache", lambda *_args: {})
    monkeypatch.setattr(launch, "_serve", _successful_serve({}))

    launch.run_launcher(environment)

    assert observed is True


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
        INFERENCE_TOKEN,
        None,
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

    launch.run_launcher(environment)

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
        launch.run_launcher(environment)

    assert serve_calls == 0
    assert INFERENCE_TOKEN not in str(exc_info.value)
    assert "FLASH_INFERENCE_TOKEN" not in environment


def test_launcher_hydrates_missing_cache_through_closed_descriptor(
    monkeypatch,
    tmp_path: Path,
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

    launch.run_launcher(environment)

    assert captured["artifact_token"] == ARTIFACT_TOKEN
    assert captured["inference_token"] == INFERENCE_TOKEN
    for key in ("artifact_fd", "inference_fd"):
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(captured[key])


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
        launch.run_launcher(environment)

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
        launch.run_launcher(environment)

    assert exc_info.value.exit_code == 128 + launch.signal.SIGINT
    assert captured_fd is not None
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(captured_fd)
    assert list((tmp_path / "cache").glob(".serving-manifest-*.tmp")) == []


def test_launcher_main_preserves_signal_exit_status(monkeypatch) -> None:
    monkeypatch.setattr(
        launch,
        "run_launcher",
        lambda: (_ for _ in ()).throw(launch.StartupTerminated(143)),
    )

    with pytest.raises(SystemExit) as exc_info:
        launch.main()

    assert exc_info.value.code == 143


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
launch._serve = serve
try:
    launch.run_launcher(environment)
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
app_main.bootstrap_serving = bootstrap
app_main.create_app = create_app
launch._serve = app_main._serve
try:
    launch.run_launcher(environment)
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
        launch.run_launcher(environment)
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
        launch.run_launcher(environment)
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
        "VLLM_WORKER_MULTIPROC_METHOD",
    ):
        assert rejected not in captured["env"]
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
        launch.run_launcher(environment)

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
        launch.run_launcher(environment)

    captured = capsys.readouterr()
    rendered = str(exc_info.value) + repr(exc_info.value) + captured.out + captured.err
    assert INFERENCE_TOKEN not in rendered
    assert ARTIFACT_TOKEN not in rendered
    assert "FLASH_INFERENCE_TOKEN" not in environment
    assert "FLASH_ARTIFACT_TOKEN" not in environment
