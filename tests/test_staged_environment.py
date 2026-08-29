from __future__ import annotations

import concurrent.futures
import copy
import gzip
import hashlib
import importlib
import json
import shutil
import sys
import tarfile
import threading
import time
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import flash.engine.worker.entry.sft as worker_sft
import flash.engine.worker.entry.worker as worker_entry
import flash.engine.worker.io.heartbeat as worker_heartbeat
import flash.engine.worker.io.hf as worker_hf
import flash.engine.worker.perf as worker_perf
import flash.engine.worker.runtime.kernel_warmup as worker_kernel_warmup
import flash.engine.worker.runtime.state as worker_state
import flash.runner.accounting.artifacts as runner_artifacts
import flash.runner.accounting.reconciliation as runner_reconciliation
import flash.runner.lifecycle.preparation as runner_preparation
import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.status as runner_status
import flash.runner.supervise.lifecycle as runner_lifecycle
from flash.core.spec import JobSpec
from flash.envs.loading.staged import (
    ResolvedEnvironmentSource,
    StagedEnvironmentTransientError,
    archive_path_for_digest,
    encode_manifest,
    manifest_path_for_digest,
    manifest_payload,
    write_environment_archive,
)
from tests._helpers.source_snapshot import valid_source_snapshot

_SHA = "a" * 40
_OTHER_SHA = "b" * 40
_REVISION = "c" * 40


def _spec(
    *,
    package: dict | None = None,
    env_id: str = "org/project/env",
    resolved_sha: str = _SHA,
    run_id: str = "flash-staged-env-test",
) -> JobSpec:
    environment = {"id": env_id, "resolved_sha": resolved_sha}
    if package is not None:
        environment["package"] = package
    return JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "sft",
            "run_id": run_id,
            "environment": environment,
            "train": {"hf_repo": "owner/run-artifacts", "max_examples": 1},
            "gpu": {"type": "H100", "max_retries": 1},
        }
    )


def _source(tmp_path: Path, *, sha: str = _SHA, name: str = "source") -> ResolvedEnvironmentSource:
    root = tmp_path / name
    package = root / "envpkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "helper.py").write_text("VALUE = 'sidecar-ok'\n", encoding="utf-8")
    (package / "data.txt").write_text("sidecar-data\n", encoding="utf-8")
    (package / "environment.py").write_text(
        "from .helper import VALUE\n"
        "from pathlib import Path\n"
        "SIDECAR = Path(__file__).with_name('data.txt').read_text().strip()\n",
        encoding="utf-8",
    )
    return ResolvedEnvironmentSource("org/project/env", sha, root, "envpkg/environment.py")


def _staged_files(
    tmp_path: Path,
    *,
    sha: str = _SHA,
) -> tuple[dict, Path, Path, str, str]:
    source = _source(tmp_path, sha=sha)
    archive = tmp_path / "package.tar.gz"
    digest = write_environment_archive(source, archive)
    archive_path = archive_path_for_digest(digest)
    manifest = tmp_path / "manifest.json"
    manifest_bytes = encode_manifest(manifest_payload(source, digest, archive_path))
    manifest.write_bytes(manifest_bytes)
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    package = {
        "artifact_revision": _REVISION,
        "archive_sha256": digest,
        "manifest_sha256": manifest_digest,
    }
    return package, manifest, archive, manifest_path_for_digest(manifest_digest), archive_path


def _hf_downloads(
    monkeypatch,
    package: dict,
    manifest: Path,
    archive: Path,
) -> list[dict]:
    calls: list[dict] = []
    monkeypatch.setenv("HF_TOKEN", "hf-test")

    def fake_download(**kwargs):
        calls.append(kwargs)
        if kwargs["filename"] == manifest_path_for_digest(package["manifest_sha256"]):
            return str(manifest)
        return str(archive)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    return calls


def _disable_github(monkeypatch) -> None:
    from flash.envs.loading import loader

    def boom(*_args, **_kwargs):
        raise AssertionError("worker must not call github")

    for name in (
        "_resolve_ref_sha",
        "_resolve_github_environment_file",
        "_download_github_directory",
        "_extract_github_tarball",
        "_urlopen",
    ):
        monkeypatch.setattr(loader, name, boom)


def _prepared_status(public: JobSpec, worker: JobSpec, *, state: str = "provisioning"):

    return runner_state.RunStatus(
        run_id=worker.run_id,
        state=state,
        spec=public.to_dict(),
        effective_preparation={
            "worker_spec": worker.to_internal_dict(),
            "version": 1,
            "preparation_digest": runner_preparation._preparation_digest(public, worker, None),
        },
    )


def test_controller_acquires_fresh_managed_tree_without_reading_poisoned_cache(
    monkeypatch, tmp_path
) -> None:
    from flash.envs.loading import loader
    from flash.envs.loading.staged import resolve_environment_source

    poisoned = tmp_path / "cache" / "poisoned"
    poisoned.mkdir(parents=True)
    (poisoned / "environment.py").write_text("POISON = True\n", encoding="utf-8")
    monkeypatch.setattr(loader, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(loader, "_resolve_ref_sha", lambda *_args, **_kwargs: _SHA)
    monkeypatch.setattr(loader, "_managed_hub_package_root", lambda _ref: "org/project/env")
    monkeypatch.setattr(
        loader,
        "_resolve_github_environment_file",
        lambda *_args, **_kwargs: pytest.fail("staging must not read the executable env cache"),
    )

    def fresh_download(_ref, repo_dir, dest, **_kwargs):
        assert repo_dir == "org/project/env"
        root = dest / "repo"
        package = root / repo_dir
        package.mkdir(parents=True)
        (package / "environment.py").write_text("FRESH = True\n", encoding="utf-8")
        (package / "sidecar.txt").write_text("fresh\n", encoding="utf-8")
        return root

    monkeypatch.setattr(loader, "_download_github_directory", fresh_download)
    source = resolve_environment_source("org/project/env")
    try:
        assert source.root != loader._CACHE_ROOT
        assert (source.root / source.entrypoint).read_text(encoding="utf-8") == "FRESH = True\n"
        assert (source.root / "org/project/env/sidecar.txt").is_file()
    finally:
        assert source.staging_root is not None
        shutil.rmtree(source.staging_root, ignore_errors=True)


@pytest.mark.parametrize(
    ("env_id", "pinned_sha"),
    [
        (f"github:someone/custom@{'A' * 40}:nested/environment.py", ""),
        ("github:someone/custom@main:nested/environment.py", "A" * 40),
    ],
)
def test_resolved_git_sha_is_lowercase_before_exact_acquisition(
    monkeypatch, tmp_path, env_id, pinned_sha
) -> None:
    from flash.envs.loading import loader
    from flash.envs.loading.staged import resolve_environment_source

    acquired: list[str] = []

    def fake_extract(ref, dest, **_kwargs):
        acquired.append(ref.ref)
        root = dest / "repo"
        (root / "nested").mkdir(parents=True)
        (root / "nested/environment.py").write_text("VALUE = 1\n", encoding="utf-8")
        return root

    monkeypatch.setattr(loader, "_extract_github_tarball", fake_extract)
    source = resolve_environment_source(env_id, pinned_sha)
    try:
        assert source.resolved_sha == _SHA
        assert acquired == [_SHA]
    finally:
        assert source.staging_root is not None
        shutil.rmtree(source.staging_root, ignore_errors=True)


def test_staging_deadline_stops_before_request_attempt(monkeypatch) -> None:
    from flash.envs.loading import loader

    attempted = False

    def fake_urlopen(*_args, **_kwargs):
        nonlocal attempted
        attempted = True
        raise AssertionError("request must not start after deadline")

    monkeypatch.setattr(loader.time, "time", lambda: 100.0)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    request = urllib.request.Request("https://api.github.com/repos/example/repo")
    with pytest.raises(loader.GitHubUnavailableError, match="authoritative run deadline"):
        loader._urlopen(request, deadline_at=100.0)
    assert not attempted


def test_staging_request_timeout_and_backoff_are_capped_to_deadline(monkeypatch) -> None:
    from flash.envs.loading import loader

    clock = iter((100.0, 100.0, 103.0, 105.0))
    timeouts: list[float] = []
    sleeps: list[float] = []

    def fake_time():
        return next(clock)

    def fake_urlopen(_request, *, timeout):
        timeouts.append(timeout)
        raise urllib.error.HTTPError(
            "https://api.github.com",
            503,
            "unavailable",
            {},
            None,
        )

    monkeypatch.setattr(loader.time, "time", fake_time)
    monkeypatch.setattr(loader.time, "sleep", lambda delay: sleeps.append(delay))
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    request = urllib.request.Request("https://api.github.com/repos/example/repo")
    with pytest.raises(loader.GitHubUnavailableError, match="authoritative run deadline"):
        loader._urlopen(request, timeout=60.0, deadline_at=105.0)
    assert timeouts == [5.0, 2.0]
    assert sleeps == [5.0]


def test_archive_permissions_are_deterministic(tmp_path) -> None:
    source = _source(tmp_path)
    executable = source.root / "envpkg" / "tool.py"
    executable.write_text("pass\n", encoding="utf-8")
    executable.chmod(0o711)
    (source.root / "envpkg" / "helper.py").chmod(0o600)
    archive = tmp_path / "permissions.tar.gz"
    write_environment_archive(source, archive)

    with gzip.open(archive, "rb") as raw, tarfile.open(fileobj=raw, mode="r|") as tar:
        modes = {member.name: member.mode for member in tar if member.name}
    assert modes["envpkg"] == 0o755
    assert modes["envpkg/helper.py"] == 0o644
    assert modes["envpkg/tool.py"] == 0o755


def test_load_staged_environment_uses_hf_only_and_cleans_owned_tree(monkeypatch, tmp_path) -> None:
    import flash.envs.loading.adapter as adapter
    from flash.envs.loading import loader
    from flash.envs.loading.staged import load_staged_freesolo_environment

    package, manifest, archive, _manifest_path, _archive_path = _staged_files(tmp_path)
    calls = _hf_downloads(monkeypatch, package, manifest, archive)
    _disable_github(monkeypatch)

    class FakeMultiTurn:
        pass

    class FakeSdkEnv:
        def __init__(self, value, sidecar):
            self.dataset = [{"prompt": "hello"}]
            self.value = value
            self.sidecar = sidecar

    def fake_load(reference, **_params):
        entrypoint = Path(reference)
        monkeypatch.syspath_prepend(str(entrypoint.parents[1]))
        sys.modules.pop("envpkg.environment", None)
        sys.modules.pop("envpkg", None)
        module = importlib.import_module("envpkg.environment")
        return FakeSdkEnv(module.VALUE, module.SIDECAR)

    tools = {
        "EnvironmentEpisode": object,
        "EnvironmentMultiTurn": FakeMultiTurn,
        "EnvironmentTurn": object,
        "load_environment": fake_load,
        "load_task_examples": lambda value: value,
        "task_example_from_record": lambda value: value,
    }
    monkeypatch.setattr(loader, "_import_freesolo_environment_tools", lambda: tools)
    monkeypatch.setattr(adapter, "_import_freesolo_environment_tools", lambda: tools)

    loaded, materialization = load_staged_freesolo_environment(
        _spec(package=package).environment,
        {},
        hf_repo="owner/run-artifacts",
    )
    assert loaded.id == "org/project/env"
    assert loaded._env.value == "sidecar-ok"
    assert loaded._env.sidecar == "sidecar-data"
    assert materialization.root.is_dir()
    assert [call["revision"] for call in calls] == [_REVISION, _REVISION]
    materialization.cleanup()
    assert not materialization.root.exists()


def test_materialization_cleans_partial_tree_on_extraction_failure(monkeypatch, tmp_path) -> None:
    from flash.envs.loading.staged import materialize_staged_environment

    package, manifest, archive, _manifest_path, _archive_path = _staged_files(tmp_path)
    archive.write_bytes(b"not a gzip archive")
    package["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["archive_sha256"] = package["archive_sha256"]
    payload["archive_path"] = archive_path_for_digest(package["archive_sha256"])
    raw = encode_manifest(payload)
    manifest.write_bytes(raw)
    package["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    _hf_downloads(monkeypatch, package, manifest, archive)
    owned = tmp_path / "owned-materialization"
    monkeypatch.setattr(
        "flash.envs.loading.staged.tempfile.mkdtemp",
        lambda **_kwargs: str(owned),
    )

    with pytest.raises((gzip.BadGzipFile, EOFError, tarfile.TarError)):
        materialize_staged_environment(
            _spec(package=package).environment,
            hf_repo="owner/run-artifacts",
        )
    assert not owned.exists()


@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError("timed out"),
        ConnectionError("connection reset"),
        httpx.ConnectError("connect failed", request=httpx.Request("GET", "https://hf.co")),
        type("RateLimited", (Exception,), {"response": SimpleNamespace(status_code=429)})(),
        type("ServerFailure", (Exception,), {"response": SimpleNamespace(status_code=503)})(),
    ],
)
def test_actual_staged_download_classifies_transient_failures(
    monkeypatch, tmp_path, failure
) -> None:
    from flash.envs.loading.staged import verify_staged_environment

    package, _manifest, _archive, _manifest_path, _archive_path = _staged_files(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "hf-test")
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda **_kwargs: (_ for _ in ()).throw(failure),
    )
    with pytest.raises(StagedEnvironmentTransientError):
        verify_staged_environment(
            _spec(package=package).environment,
            hf_repo="owner/run-artifacts",
        )


def test_missing_immutable_revision_is_terminal(monkeypatch, tmp_path) -> None:
    from flash.envs.loading.staged import verify_staged_environment

    package, _manifest, _archive, _manifest_path, _archive_path = _staged_files(tmp_path)

    class MissingRevision(Exception):
        response = SimpleNamespace(status_code=404)

    monkeypatch.setenv("HF_TOKEN", "hf-test")
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda **_kwargs: (_ for _ in ()).throw(MissingRevision()),
    )
    with pytest.raises(RuntimeError, match="unavailable at its immutable revision"):
        verify_staged_environment(
            _spec(package=package).environment,
            hf_repo="owner/run-artifacts",
        )


def test_worker_failure_boundary_marks_staged_transient_retriable() -> None:
    from flash.envs.meta.identity import GitHubUnavailableError

    assert worker_entry._worker_failure_flags(StagedEnvironmentTransientError("temporary")) == {
        "retriable": True,
        "oom": False,
    }
    # the staged type is NEW, so this must not narrow the boundary to it: a github outage or quota
    # exhaustion still arrives here as GitHubTransientError, and the worker's answer to either is
    # the same -- reschedule. classifying only the staged type would fail those runs permanently.
    assert worker_entry._worker_failure_flags(GitHubUnavailableError("github outage")) == {
        "retriable": True,
        "oom": False,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("canonical_environment_id", "other/project/env"),
        ("resolved_sha", _OTHER_SHA),
        ("entrypoint", "envpkg/other.py"),
        ("archive_sha256", "d" * 64),
        ("archive_path", "environment-packages/archives/sha256/other/package.tar.gz"),
    ],
)
def test_manifest_identity_tampering_fails_before_environment_import(
    monkeypatch, tmp_path, field, value
) -> None:
    from flash.envs.loading import loader
    from flash.envs.loading.staged import load_staged_freesolo_environment

    package, manifest, archive, _manifest_path, _archive_path = _staged_files(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload[field] = value
    raw = encode_manifest(payload)
    manifest.write_bytes(raw)
    package["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    _hf_downloads(monkeypatch, package, manifest, archive)
    _disable_github(monkeypatch)
    monkeypatch.setattr(
        loader,
        "_load_resolved_freesolo_environment",
        lambda *_args, **_kwargs: pytest.fail("tampered package must fail before import"),
    )
    with pytest.raises(RuntimeError, match=r"manifest|entrypoint"):
        load_staged_freesolo_environment(
            _spec(package=package).environment,
            {},
            hf_repo="owner/run-artifacts",
        )


def test_controller_stage_uploads_completion_last_and_verifies_returned_revision(
    monkeypatch, tmp_path
) -> None:
    import huggingface_hub

    import flash.envs.loading.staged as staged
    import flash.providers._lifecycle.net.worker as provider_worker
    from flash.runner.accounting.artifacts import stage_environment_package

    source = _source(tmp_path)
    uploads: list[dict] = []
    verified: list[tuple[str, str]] = []

    class FakeApi:
        def __init__(self, token):
            assert token == "hf-controller"

        def upload_file(self, **kwargs):
            uploads.append(kwargs)
            return SimpleNamespace(oid=_REVISION)

    def fake_verify(environment, *, hf_repo, token):
        assert environment.package is not None
        verified.append((environment.package.artifact_revision, hf_repo))
        assert token == "hf-controller"
        return SimpleNamespace()

    monkeypatch.setenv("HF_TOKEN", "hf-controller")
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(staged, "resolve_environment_source", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(staged, "verify_staged_environment", fake_verify)
    monkeypatch.setattr(provider_worker, "_ensure_private_artifact_repo", lambda *_a, **_k: None)
    monkeypatch.setattr(provider_worker, "_hf_call", lambda call, *_a, **_k: call())

    staged_spec = stage_environment_package(_spec())
    package = staged_spec.environment.package
    assert package is not None
    assert uploads[0]["path_in_repo"] == archive_path_for_digest(package.archive_sha256)
    assert uploads[1]["path_in_repo"] == manifest_path_for_digest(package.manifest_sha256)
    manifest_bytes = uploads[1]["path_or_fileobj"].getvalue()
    assert hashlib.sha256(manifest_bytes).hexdigest() == package.manifest_sha256
    assert json.loads(manifest_bytes)["resolved_sha"] == _SHA
    assert verified == [(_REVISION, "owner/run-artifacts")]

    assert stage_environment_package(staged_spec) is staged_spec
    assert verified == [
        (_REVISION, "owner/run-artifacts"),
        (_REVISION, "owner/run-artifacts"),
    ]
    assert len(uploads) == 2


def test_identical_archives_for_different_git_shas_use_distinct_manifests_concurrently(
    monkeypatch, tmp_path
) -> None:
    import huggingface_hub

    import flash.envs.loading.staged as staged
    import flash.providers._lifecycle.net.worker as provider_worker
    from flash.runner.accounting.artifacts import stage_environment_package

    sources = {
        _SHA: _source(tmp_path, sha=_SHA, name="source-a"),
        _OTHER_SHA: _source(tmp_path, sha=_OTHER_SHA, name="source-b"),
    }
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    stored_manifests: dict[tuple[str, str], bytes] = {}
    revisions = {_SHA: "d" * 40, _OTHER_SHA: "e" * 40}

    class FakeApi:
        def __init__(self, token):
            assert token == "hf-controller"

        def upload_file(self, **kwargs):
            value = kwargs["path_or_fileobj"]
            if isinstance(value, str):
                return SimpleNamespace(oid="f" * 40)
            raw = value.getvalue()
            payload = json.loads(raw)
            revision = revisions[payload["resolved_sha"]]
            barrier.wait(timeout=5)
            with lock:
                stored_manifests[(revision, kwargs["path_in_repo"])] = raw
            return SimpleNamespace(oid=revision)

    def fake_verify(environment, *, hf_repo, token):
        package = environment.package
        assert package is not None
        path = manifest_path_for_digest(package.manifest_sha256)
        raw = stored_manifests[(package.artifact_revision, path)]
        payload = json.loads(raw)
        assert payload["resolved_sha"] == environment.resolved_sha
        assert hf_repo == "owner/run-artifacts"
        assert token == "hf-controller"
        return SimpleNamespace()

    monkeypatch.setenv("HF_TOKEN", "hf-controller")
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(
        staged,
        "resolve_environment_source",
        lambda _env_id, resolved_sha, **_kwargs: sources[resolved_sha],
    )
    monkeypatch.setattr(staged, "verify_staged_environment", fake_verify)
    monkeypatch.setattr(provider_worker, "_ensure_private_artifact_repo", lambda *_a, **_k: None)
    monkeypatch.setattr(provider_worker, "_hf_call", lambda call, *_a, **_k: call())

    specs = [_spec(resolved_sha=_SHA), _spec(resolved_sha=_OTHER_SHA)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        staged_specs = list(pool.map(stage_environment_package, specs))

    first = staged_specs[0].environment.package
    second = staged_specs[1].environment.package
    assert first is not None
    assert second is not None
    assert first.archive_sha256 == second.archive_sha256
    assert first.manifest_sha256 != second.manifest_sha256
    assert manifest_path_for_digest(first.manifest_sha256) != manifest_path_for_digest(
        second.manifest_sha256
    )
    assert first.artifact_revision != second.artifact_revision


def test_a_stripped_package_restages_from_the_same_pin_instead_of_running_unstaged(
    tmp_path,
) -> None:
    """Dropping the package from a snapshot must not silently produce an unstaged run.

    `_validate_effective_spec` cannot reject this shape: `to_internal_dict` omits the key when
    unset, so a stripped package is byte-identical to a run that has not staged yet -- the state of
    every run between submit and allocation. The containment is at the consumer instead, and the
    property that matters is that the recovered spec keeps the digest-bound `resolved_sha`, so a
    restage rebuilds the SAME commit rather than resolving a newer one.
    """

    package, _manifest, _archive, _manifest_path, _archive_path = _staged_files(tmp_path)
    public = _spec(package=None, resolved_sha="")
    worker = _spec(package=package)
    snapshot = {
        "worker_spec": worker.to_internal_dict(),
        "version": 1,
        "preparation_digest": runner_preparation._preparation_digest(public, worker, None),
    }
    tampered = copy.deepcopy(snapshot)
    tampered["worker_spec"]["environment"].pop("package")
    status = runner_state.RunStatus(
        run_id=worker.run_id,
        state="provisioning",
        spec=public.to_dict(),
        effective_preparation=tampered,
    )

    recovered = runner_status.effective_spec_from_status(status)
    assert recovered.environment.package is None
    # the pin survives, so a restage cannot drift to a different commit than the one staged.
    assert recovered.environment.resolved_sha == worker.environment.resolved_sha


# a SUBSTITUTED package -- the case that must fail closed -- is covered by
# test_manifest_identity_tampering_fails_before_environment_import above, which asserts the
# specific digest/entrypoint rejection rather than merely that something raised.


def test_initial_lifecycle_defers_transient_staging_before_training(monkeypatch, tmp_path) -> None:
    from flash.runner.supervise import lifecycle

    package, _manifest, _archive, _manifest_path, _archive_path = _staged_files(tmp_path)
    public = _spec(package=None, resolved_sha="")
    unstaged = _spec(package=None, resolved_sha="")
    staged_spec = _spec(package=package)
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    status = _prepared_status(public, unstaged)
    runner_state._save_status(
        status,
        _run_deadline_at=status.created_at + unstaged.gpu.max_wall_seconds,
    )
    calls = {"stage": 0, "train": 0}

    def fake_stage(_spec, **_kwargs):
        calls["stage"] += 1
        if calls["stage"] == 1:
            raise StagedEnvironmentTransientError("connect failed")
        return staged_spec

    monkeypatch.setattr(runner_artifacts, "stage_environment_package", fake_stage)
    monkeypatch.setattr(
        runner_lifecycle,
        "_run_training",
        lambda *_args, **_kwargs: calls.__setitem__("train", calls["train"] + 1),
    )
    monkeypatch.setattr("flash.content.multimodal.preflight_validate_image_opd", lambda _spec: None)
    monkeypatch.setattr(lifecycle.time, "sleep", lambda _delay: None)

    lifecycle._run_job(unstaged)
    assert calls == {"stage": 2, "train": 1}
    assert runner_status.get_status(unstaged.run_id).state == "provisioning"


def test_staging_deadline_prevents_training_and_provider_allocation(monkeypatch, tmp_path) -> None:
    from flash.runner.supervise import lifecycle

    public = _spec(package=None, resolved_sha="")
    unstaged = _spec(package=None, resolved_sha="")
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    status = _prepared_status(public, unstaged)
    status.created_at = time.time() - unstaged.gpu.max_wall_seconds - 1
    runner_state._save_status(
        status,
        _run_deadline_at=status.created_at + unstaged.gpu.max_wall_seconds,
    )
    calls = {"train": 0}
    monkeypatch.setattr(
        runner_artifacts,
        "stage_environment_package",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            StagedEnvironmentTransientError("timed out")
        ),
    )
    monkeypatch.setattr(
        runner_lifecycle,
        "_run_training",
        lambda *_args, **_kwargs: calls.__setitem__("train", calls["train"] + 1),
    )
    monkeypatch.setattr("flash.content.multimodal.preflight_validate_image_opd", lambda _spec: None)

    with pytest.raises(RuntimeError, match="deadline exhausted during environment staging"):
        lifecycle._run_job(unstaged)
    assert calls == {"train": 0}
    assert runner_status.get_status(unstaged.run_id).state == "failed"


def test_attach_boundary_schedules_reconciliation_for_staged_transient(
    monkeypatch, tmp_path
) -> None:
    from flash.providers.core.base import PollResult
    from flash.runner.supervise import attach

    package, _manifest, _archive, _manifest_path, _archive_path = _staged_files(tmp_path)
    public = _spec(package=None, resolved_sha="")
    worker = _spec(package=package)
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    remote = {
        "provider": "runpod",
        "endpoint_id": "ep",
        "job_id": "job",
        "key_fingerprint": "rpk-0123456789ab",
        "attempt": 0,
        "started_ts": time.time(),
    }
    status = _prepared_status(public, worker, state="running")
    status.remote = remote
    runner_state._save_status(
        status,
        _run_deadline_at=status.created_at + worker.gpu.max_wall_seconds,
    )
    context = SimpleNamespace(
        worker_spec=worker,
        persisted_remote=remote,
        handle=SimpleNamespace(provider="runpod", data={}),
        seed=worker.seed,
        recovered_attempt=0,
        next_attempt=1,
        source_snapshot=valid_source_snapshot(),
        allocated_gpu=None,
        allocated_gpu_count=None,
    )
    scheduled: list[tuple] = []
    monkeypatch.setattr(attach, "_build_attach_context", lambda *_args: context)
    monkeypatch.setattr(
        "flash.providers.core.registry.get_provider",
        lambda _name: SimpleNamespace(
            poll_attempt=lambda *_args, **_kwargs: PollResult(
                False, failure="stalled", detail="lost"
            )
        ),
    )
    monkeypatch.setattr(
        attach,
        "_handle_failed_attach_poll",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            StagedEnvironmentTransientError("httpx.ConnectError")
        ),
    )
    monkeypatch.setattr(
        attach,
        "_schedule_attach_reconciliation",
        lambda *args: scheduled.append(args) or True,
    )

    returned = attach.attach_run(worker.run_id)
    assert returned.state == "running"
    assert len(scheduled) == 1
    assert scheduled[0][0] == worker.run_id


def test_confirmed_teardown_staging_transient_defers_without_clearing_or_allocating(
    monkeypatch, tmp_path
) -> None:
    from flash.providers.core.base import PollResult
    from flash.runner.supervise import attach
    from flash.runner.supervise import lifecycle as supervise_lifecycle

    package, _manifest, _archive, _manifest_path, _archive_path = _staged_files(tmp_path)
    public = _spec(package=None, resolved_sha="")
    worker = _spec(package=package)
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    remote = {
        "provider": "runpod",
        "endpoint_id": "ep",
        "job_id": "job",
        "key_fingerprint": "rpk-0123456789ab",
        "attempt": 0,
        "started_ts": time.time(),
        # the launch persist writes the authorizing token and the allocation stamp together; retry
        # reconstructs its candidate from the stamp, so a fixture without it never reaches staging.
        "launch_claim_token": "token-staged-0",
        "allocated_gpu": worker.gpu.type,
        "allocated_gpu_count": 1,
        "allocated_usable_vram_gb": 32.0,
    }
    status = _prepared_status(public, worker, state="running")
    status.remote = remote
    deadline_at = status.created_at + worker.gpu.max_wall_seconds
    # attempt 0 already holds a handle, so the reserved counter production would have written is 1.
    runner_state._save_status(status, _run_deadline_at=deadline_at, _next_attempt=1)
    context = SimpleNamespace(
        worker_spec=worker,
        persisted_remote=remote,
        handle=SimpleNamespace(provider="runpod", data={}),
        seed=worker.seed,
        recovered_attempt=0,
        next_attempt=1,
        source_snapshot=valid_source_snapshot(),
        launch_claim_token="token-staged-0",
        allocated_gpu=None,
        allocated_gpu_count=None,
    )
    calls = {"stage": 0, "clear": 0, "fail": 0, "train": 0, "allocate": 0}
    scheduled: list[tuple] = []
    deadline_at_expected = deadline_at

    def transient_stage(spec, *, deadline_at):
        calls["stage"] += 1
        assert spec.environment.package == worker.environment.package
        assert deadline_at == pytest.approx(deadline_at_expected)
        raise StagedEnvironmentTransientError("artifact store temporarily unavailable")

    def record_clear(*_args, **_kwargs):
        calls["clear"] += 1
        return False

    def record_fail(*_args, **_kwargs):
        calls["fail"] += 1
        return False

    def forbidden_training(*_args, **_kwargs):
        calls["train"] += 1
        raise AssertionError("training must not start before staged environment verification")

    def forbidden_allocation(*_args, **_kwargs):
        calls["allocate"] += 1
        raise AssertionError(
            "provider allocation must not start before staged environment verification"
        )

    monkeypatch.setattr(attach, "_build_attach_context", lambda *_args: context)
    monkeypatch.setattr(
        "flash.providers.core.registry.get_provider",
        lambda _name: SimpleNamespace(
            poll_attempt=lambda *_args, **_kwargs: PollResult(
                False, failure="stalled", detail="lost"
            )
        ),
    )
    monkeypatch.setattr(supervise_lifecycle, "_runpod_completed_metrics", lambda *_a, **_k: None)
    monkeypatch.setattr(supervise_lifecycle, "_strict_teardown_handle", lambda *_a, **_k: True)
    monkeypatch.setattr(runner_artifacts, "stage_environment_package", transient_stage)
    monkeypatch.setattr(runner_reconciliation, "_compare_and_clear_remote", record_clear)
    monkeypatch.setattr(runner_reconciliation, "_compare_and_fail_remote", record_fail)
    monkeypatch.setattr(runner_lifecycle, "_run_training", forbidden_training)
    monkeypatch.setattr("flash.providers.core.allocator.allocate", forbidden_allocation)
    monkeypatch.setattr(
        attach,
        "_schedule_attach_reconciliation",
        lambda *args: scheduled.append(args) or True,
    )

    returned = attach.attach_run(worker.run_id)

    assert calls == {"stage": 1, "clear": 0, "fail": 0, "train": 0, "allocate": 0}
    assert returned.state == "running"
    assert returned.remote == remote
    assert len(scheduled) == 1
    assert scheduled[0][0] == worker.run_id
    assert scheduled[0][1] == remote
    assert "temporarily unavailable" in scheduled[0][-1]


@pytest.mark.parametrize("handler_fails", [False, True])
def test_worker_cleans_materialized_package_after_training_handler(
    monkeypatch, tmp_path, handler_fails
) -> None:
    from flash.envs.loading.staged import StagedEnvironmentMaterialization

    class WorkerExited(BaseException):
        pass

    root = tmp_path / "materialized"
    root.mkdir()
    entrypoint = root / "environment.py"
    entrypoint.write_text("VALUE = 1\n", encoding="utf-8")
    worker_state.ACTIVE_ENV_PACKAGE = StagedEnvironmentMaterialization(root, entrypoint)

    def handler() -> None:
        assert root.is_dir()
        if handler_fails:
            raise RuntimeError("training failed")

    monkeypatch.setattr(worker_state, "RUN_MODE", "sft")
    monkeypatch.setattr(worker_state, "JOB_SPEC", None)
    monkeypatch.setattr(worker_state, "HF_REPO", "")
    monkeypatch.setattr(worker_sft, "run_sft", handler)
    monkeypatch.setattr(worker_hf, "_disable_xet_upload_staging", lambda: None)
    monkeypatch.setattr(worker_state, "_remaining_worker_wall_seconds", lambda: None)
    monkeypatch.setattr(worker_entry, "_preflight_gpu_occupancy_for_spec", lambda: None)
    monkeypatch.setattr(worker_perf, "_force_fla_triton_gdn_on_sm100", lambda: None)
    monkeypatch.setattr(worker_perf, "_ensure_fla_fastpath_on_hopper", lambda: None)
    monkeypatch.setattr(worker_perf, "_restrict_fla_gdn_autotune_on_blackwell", lambda: None)
    monkeypatch.setattr(worker_perf, "gpu_diagnostics", lambda **_kwargs: {})
    monkeypatch.setattr(worker_heartbeat, "heartbeat", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker_kernel_warmup, "load_mega_cache", lambda: None)
    monkeypatch.setattr(
        worker_entry.os,
        "_exit",
        lambda _code: (_ for _ in ()).throw(WorkerExited()),
    )

    expected = RuntimeError if handler_fails else WorkerExited
    with pytest.raises(expected):
        worker_entry._run_worker_mode()
    assert worker_state.ACTIVE_ENV_PACKAGE is None
    assert not root.exists()


def test_package_descriptor_contains_only_immutable_transport_values(tmp_path) -> None:
    package, _manifest, _archive, _manifest_path, _archive_path = _staged_files(tmp_path)
    spec = _spec(package=package)
    assert set(spec.to_internal_dict()["environment"]["package"]) == {
        "artifact_revision",
        "archive_sha256",
        "manifest_sha256",
    }


def test_artifact_repository_identity_keeps_raw_environment_spelling(monkeypatch) -> None:
    from flash.runner.accounting.artifacts import managed_hf_repo_for_environment

    monkeypatch.setenv("FLASH_HF_NAMESPACE", "owner")
    managed = "org/project/env"
    url = "https://github.com/freesolo-co/environment-hub/blob/main/org/project/env/environment.py"
    assert managed_hf_repo_for_environment(managed) != managed_hf_repo_for_environment(url)
