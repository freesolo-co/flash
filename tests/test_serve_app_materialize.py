"""artifact hydration, cache integrity, token, and current adapter contract."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from flash.serve.app import materialize as materialize_module
from flash.serve.app.manifest import ArtifactFile, build_serving_manifest
from flash.serve.app.materialize import (
    MaterializationError,
    _load_strict_config,
    _validate_cache_ancestor_stat,
    adapter_cache_path,
    base_weights_are_cached,
    base_weights_cache_path,
    hydrate_base_weights,
    hydrate_manifest,
    locked_manifest_cache,
    validate_manifest_cache,
)
from tests.test_serve_app_manifest import BASE_REVISION, _spec_and_inputs

TOKEN = "scoped-artifact-token-sentinel"


def _artifact_bytes(
    tmp_path: Path, *, adapter_name: str = "default", **config_overrides: object
) -> tuple[bytes, bytes]:
    """build one adapter, optionally without the peft adapter-name segment.

    ``adapter_name=""`` produces the "<module>.lora_A.weight" shape verl's model_merger writes,
    which is what every adapter flash actually serves looks like.
    """

    source = tmp_path / "source"
    source.mkdir()
    infix = f".{adapter_name}." if adapter_name else "."
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": "Qwen/Qwen3.5-9B",
        "revision": BASE_REVISION,
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj"],
        **config_overrides,
    }
    config_bytes = json.dumps(config, sort_keys=True).encode()
    config_path = source / "adapter_config.json"
    config_path.write_bytes(config_bytes)
    weights_path = source / "adapter_model.safetensors"
    module = "base_model.model.layers.0.self_attn.q_proj"
    save_file(
        {
            f"{module}.lora_A{infix}weight": np.zeros((16, 2), dtype=np.float32),
            f"{module}.lora_B{infix}weight": np.zeros((2, 16), dtype=np.float32),
        },
        weights_path,
    )
    return config_bytes, weights_path.read_bytes()


def _manifest(config_bytes: bytes, weights_bytes: bytes):
    files = (
        ArtifactFile(
            "adapter_config.json",
            len(config_bytes),
            hashlib.sha256(config_bytes).hexdigest(),
        ),
        ArtifactFile(
            "adapter_model.safetensors",
            len(weights_bytes),
            hashlib.sha256(weights_bytes).hexdigest(),
        ),
    )
    return build_serving_manifest(*_spec_and_inputs(files=files))


def _download_stub(config_bytes: bytes, weights_bytes: bytes, calls: list[dict]):
    def download(**kwargs):
        calls.append(kwargs)
        snapshot = Path(kwargs["cache_dir"]) / "snapshot"
        subfolder = snapshot / "sft/run-1/adapter"
        subfolder.mkdir(parents=True, exist_ok=True)
        (subfolder / "adapter_config.json").write_bytes(config_bytes)
        (subfolder / "adapter_model.safetensors").write_bytes(weights_bytes)
        return str(snapshot)

    return download


def _token_fd() -> int:
    read_fd, write_fd = os.pipe()
    os.write(write_fd, TOKEN.encode())
    os.close(write_fd)
    return read_fd


class _DirectoryTimestampStat:
    def __init__(self, details: os.stat_result, *, offset: int) -> None:
        self._details = details
        self.st_mtime_ns = details.st_mtime_ns + offset
        self.st_ctime_ns = details.st_ctime_ns + offset

    def __getattr__(self, name: str):
        return getattr(self._details, name)


def _descriptor_path(descriptor: int) -> str:
    return os.readlink(f"/proc/self/fd/{descriptor}")


def test_hydration_synchronizes_directory_metadata_before_validation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_bytes, weights_bytes = _artifact_bytes(tmp_path)
    manifest = _manifest(config_bytes, weights_bytes)
    cache = tmp_path / "cache"
    real_fstat = os.fstat
    real_fsync = os.fsync
    fsynced: set[str] = set()
    stale_reads: set[str] = set()
    refreshed: set[str] = set()

    def fsync(descriptor: int) -> None:
        details = real_fstat(descriptor)
        if stat.S_ISDIR(details.st_mode):
            fsynced.add(_descriptor_path(descriptor))
        real_fsync(descriptor)

    def fstat(descriptor: int):
        details = real_fstat(descriptor)
        path = _descriptor_path(descriptor)
        is_adapter = "/adapters/.stage-" in path or path.endswith(
            manifest.adapters[0].aggregate_sha256
        )
        if stat.S_ISDIR(details.st_mode) and is_adapter and path not in stale_reads:
            stale_reads.add(path)
            return _DirectoryTimestampStat(details, offset=-1)
        return details

    def forced_times(descriptor: int) -> tuple[int, int]:
        details = real_fstat(descriptor)
        refreshed.add(_descriptor_path(descriptor))
        return details.st_mtime_ns, details.st_ctime_ns

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "fstat", fstat)
    monkeypatch.setattr(materialize_module, "_forced_directory_times", forced_times)

    paths = hydrate_manifest(
        manifest,
        cache,
        token_fd=_token_fd(),
        snapshot_download_fn=_download_stub(config_bytes, weights_bytes, []),
    )

    destination = str(next(iter(paths.values())))
    assert destination in stale_reads
    assert destination in refreshed
    assert destination in fsynced
    assert str(Path(destination).parent) in fsynced


def test_unsupported_directory_fsync_does_not_block_hydration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_bytes, weights_bytes = _artifact_bytes(tmp_path)
    manifest = _manifest(config_bytes, weights_bytes)
    real_fstat = os.fstat
    real_fsync = os.fsync

    def fsync(descriptor: int) -> None:
        if stat.S_ISDIR(real_fstat(descriptor).st_mode):
            raise OSError(errno.EINVAL, "directory fsync is unsupported")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(materialize_module, "_forced_directory_times", lambda _descriptor: None)

    assert hydrate_manifest(
        manifest,
        tmp_path / "cache",
        token_fd=_token_fd(),
        snapshot_download_fn=_download_stub(config_bytes, weights_bytes, []),
    )


def test_directory_fsync_io_failure_blocks_hydration(monkeypatch, tmp_path: Path) -> None:
    config_bytes, weights_bytes = _artifact_bytes(tmp_path)
    manifest = _manifest(config_bytes, weights_bytes)
    real_fstat = os.fstat
    real_fsync = os.fsync

    def fsync(descriptor: int) -> None:
        if stat.S_ISDIR(real_fstat(descriptor).st_mode):
            raise OSError(errno.EIO, "directory fsync failed")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fsync)

    with pytest.raises(MaterializationError, match="could not be synchronized"):
        hydrate_manifest(
            manifest,
            tmp_path / "cache",
            token_fd=_token_fd(),
            snapshot_download_fn=_download_stub(config_bytes, weights_bytes, []),
        )


def test_transient_directory_mutation_during_validation_is_detected(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_bytes, weights_bytes = _artifact_bytes(tmp_path)
    manifest = _manifest(config_bytes, weights_bytes)
    cache = tmp_path / "cache"
    paths = hydrate_manifest(
        manifest,
        cache,
        token_fd=_token_fd(),
        snapshot_download_fn=_download_stub(config_bytes, weights_bytes, []),
    )
    destination = next(iter(paths.values()))
    real_fstat = os.fstat
    real_validate = materialize_module.validate_adapter_weight_structure
    mutation_complete = False

    def mutate_between_passes(*args, **kwargs) -> None:
        nonlocal mutation_complete
        real_validate(*args, **kwargs)
        marker = destination / ".transient-entry"
        marker.write_bytes(b"transient")
        marker.unlink()
        mutation_complete = True

    def forced_times(descriptor: int) -> tuple[int, int]:
        details = real_fstat(descriptor)
        offset = int(mutation_complete and _descriptor_path(descriptor) == str(destination))
        return details.st_mtime_ns + offset, details.st_ctime_ns + offset

    monkeypatch.setattr(
        materialize_module, "validate_adapter_weight_structure", mutate_between_passes
    )
    monkeypatch.setattr(materialize_module, "_forced_directory_times", forced_times)

    with pytest.raises(MaterializationError, match="changed during validation"):
        validate_manifest_cache(manifest, cache)


def test_hydrate_forwards_exact_source_patterns_and_closes_token_fd(tmp_path: Path, capsys) -> None:
    config_bytes, weights_bytes = _artifact_bytes(tmp_path)
    manifest = _manifest(config_bytes, weights_bytes)
    calls: list[dict] = []
    read_fd = _token_fd()

    paths = hydrate_manifest(
        manifest,
        tmp_path / "cache",
        token_fd=read_fd,
        snapshot_download_fn=_download_stub(config_bytes, weights_bytes, calls),
    )
    output = capsys.readouterr().out
    acquiring = output.index("phase=digest-lock-acquiring")
    acquired = output.index("phase=digest-lock-acquired")
    assert acquiring < acquired
    assert str(tmp_path / "cache" / ".locks") in output
    assert "phase=artifact-download-starting" in output
    assert 'repo="flash-owned/run-artifacts"' in output
    assert f'revision="{manifest.adapters[0].source_revision}"' in output
    assert TOKEN not in output

    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(read_fd)
    assert len(calls) == 1
    assert calls[0] == {
        "repo_id": "flash-owned/run-artifacts",
        "repo_type": "model",
        "revision": "4" * 40,
        "allow_patterns": [
            "sft/run-1/adapter/adapter_config.json",
            "sft/run-1/adapter/adapter_model.safetensors",
        ],
        "token": TOKEN,
        "cache_dir": str((tmp_path / "cache").resolve() / ".hf-cache"),
    }
    destination = paths[manifest.adapters[0].checkpoint_id]
    assert destination == adapter_cache_path(tmp_path / "cache", manifest.adapters[0])
    assert {path.name for path in destination.iterdir()} == {
        "adapter_config.json",
        "adapter_model.safetensors",
    }
    assert validate_manifest_cache(manifest, tmp_path / "cache") == paths
    assert TOKEN not in repr(manifest)
    assert TOKEN not in repr(paths)


def test_serve_revalidation_rejects_corruption_without_downloading(tmp_path: Path) -> None:
    config_bytes, weights_bytes = _artifact_bytes(tmp_path)
    manifest = _manifest(config_bytes, weights_bytes)
    hydrate_manifest(
        manifest,
        tmp_path / "cache",
        token_fd=_token_fd(),
        snapshot_download_fn=_download_stub(config_bytes, weights_bytes, []),
    )
    destination = adapter_cache_path(tmp_path / "cache", manifest.adapters[0])
    (destination / "adapter_model.safetensors").write_bytes(b"corrupt")

    with pytest.raises(MaterializationError, match=r"size|digest"):
        validate_manifest_cache(manifest, tmp_path / "cache")


@pytest.mark.parametrize(
    "constant",
    [
        pytest.param("NaN", id="nan"),
        pytest.param("Infinity", id="positive-infinity"),
        pytest.param("-Infinity", id="negative-infinity"),
    ],
)
def test_non_finite_adapter_config_constants_are_not_strict_json(constant: str) -> None:
    raw = f'{{"peft_type":"LORA","r":16,"lora_alpha":{constant}}}'.encode()

    with pytest.raises(
        MaterializationError, match=r"adapter_config\.json is not strict utf-8 json"
    ):
        _load_strict_config(raw)


def test_invalid_current_adapter_metadata_never_publishes_cache(tmp_path: Path) -> None:
    invalid_configs = (
        {"peft_type": "PREFIX_TUNING"},
        {"task_type": "SEQ_2_SEQ_LM"},
        {"base_model_name_or_path": "other/model"},
        {"revision": "9" * 40},
        {"r": 65},
        {"modules_to_save": ["lm_head"]},
    )
    for index, changes in enumerate(invalid_configs):
        case = tmp_path / f"case-{index}"
        case.mkdir()
        config_bytes, weights_bytes = _artifact_bytes(case, **changes)
        manifest = _manifest(config_bytes, weights_bytes)
        with pytest.raises(MaterializationError):
            hydrate_manifest(
                manifest,
                case / "cache",
                token_fd=_token_fd(),
                snapshot_download_fn=_download_stub(config_bytes, weights_bytes, []),
            )
        destination = adapter_cache_path(case / "cache", manifest.adapters[0])
        assert not destination.exists()
        assert list((case / "cache" / "adapters").glob(".stage-*")) == []


def test_download_failure_cleans_only_invocation_stage(tmp_path: Path) -> None:
    config_bytes, weights_bytes = _artifact_bytes(tmp_path)
    manifest = _manifest(config_bytes, weights_bytes)
    cache = tmp_path / "cache"
    preexisting = cache / "adapters" / "preexisting-user-file"
    preexisting.parent.mkdir(parents=True)
    preexisting.write_text("keep")

    def incomplete(**kwargs):
        snapshot = Path(kwargs["cache_dir"]) / "snapshot"
        subfolder = snapshot / "sft/run-1/adapter"
        subfolder.mkdir(parents=True)
        (subfolder / "adapter_config.json").write_bytes(config_bytes)
        return str(snapshot)

    with pytest.raises(MaterializationError):
        hydrate_manifest(
            manifest,
            cache,
            token_fd=_token_fd(),
            snapshot_download_fn=incomplete,
        )
    assert preexisting.read_text() == "keep"
    assert list((cache / "adapters").glob(".stage-*")) == []
    assert not adapter_cache_path(cache, manifest.adapters[0]).exists()


def test_download_errors_do_not_expose_the_artifact_token(tmp_path: Path) -> None:
    config_bytes, weights_bytes = _artifact_bytes(tmp_path)
    manifest = _manifest(config_bytes, weights_bytes)

    def failed_download(**_kwargs):
        raise RuntimeError(f"remote echoed {TOKEN}")

    with pytest.raises(MaterializationError, match="artifact download failed") as exc_info:
        hydrate_manifest(
            manifest,
            tmp_path / "cache",
            token_fd=_token_fd(),
            snapshot_download_fn=failed_download,
        )
    assert TOKEN not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_extra_downloaded_code_is_never_copied_or_executed(tmp_path: Path) -> None:
    config_bytes, weights_bytes = _artifact_bytes(tmp_path)
    manifest = _manifest(config_bytes, weights_bytes)
    marker = tmp_path / "executed"

    def with_code(**kwargs):
        snapshot = Path(kwargs["cache_dir"]) / "snapshot"
        subfolder = snapshot / "sft/run-1/adapter"
        subfolder.mkdir(parents=True)
        (subfolder / "adapter_config.json").write_bytes(config_bytes)
        (subfolder / "adapter_model.safetensors").write_bytes(weights_bytes)
        (subfolder / "plugin.py").write_text(f"open({str(marker)!r}, 'w').write('bad')")
        return str(snapshot)

    paths = hydrate_manifest(
        manifest,
        tmp_path / "cache",
        token_fd=_token_fd(),
        snapshot_download_fn=with_code,
    )
    destination = next(iter(paths.values()))
    assert not (destination / "plugin.py").exists()
    assert not marker.exists()
    shutil.rmtree(tmp_path / "cache" / ".hf-cache")
    assert validate_manifest_cache(manifest, tmp_path / "cache")


def test_adapter_config_revision_missing_null_and_empty_are_unspecified(tmp_path: Path) -> None:
    for index, revision in enumerate(("missing", None, "")):
        case = tmp_path / f"revision-{index}"
        case.mkdir(mode=0o700)
        case.chmod(0o700)
        config_bytes, weights_bytes = _artifact_bytes(case)
        config = json.loads(config_bytes)
        if revision == "missing":
            config.pop("revision")
        else:
            config["revision"] = revision
        config_bytes = json.dumps(config, sort_keys=True).encode()
        manifest = _manifest(config_bytes, weights_bytes)

        paths = hydrate_manifest(
            manifest,
            case / "cache",
            token_fd=_token_fd(),
            snapshot_download_fn=_download_stub(config_bytes, weights_bytes, []),
        )

        assert validate_manifest_cache(manifest, case / "cache") == paths


def _directory_stat(*, uid: int, permissions: int) -> os.stat_result:
    return os.stat_result((stat.S_IFDIR | permissions, 1, 1, 2, uid, uid, 0, 0, 0, 0))


def test_cache_ancestor_policy_rejects_foreign_intermediate_before_current_uid_child() -> None:
    foreign_uid = 1 if os.getuid() != 1 else 2
    chain = (
        ("foreign writable intermediate", _directory_stat(uid=foreign_uid, permissions=0o777)),
        ("current uid cache child", _directory_stat(uid=os.getuid(), permissions=0o700)),
    )

    def validate_chain() -> None:
        for name, details in chain:
            _validate_cache_ancestor_stat(details, name)

    with pytest.raises(MaterializationError, match="untrusted uid"):
        validate_chain()


def test_cache_ancestor_policy_allows_narrow_root_directories() -> None:
    _validate_cache_ancestor_stat(
        _directory_stat(uid=0, permissions=0o755),
        "root non-writable ancestor",
    )
    _validate_cache_ancestor_stat(
        _directory_stat(uid=0, permissions=0o1777),
        "root sticky shared ancestor",
    )
    # a root-owned world-writable ancestor is accepted with OR without the sticky bit. runpod
    # mounts its network volume as root-owned 0777 and non-sticky (measured on a live pod), and
    # rejecting that killed the container about two seconds into every startup, forever.
    _validate_cache_ancestor_stat(
        _directory_stat(uid=0, permissions=0o777),
        "root writable non-sticky ancestor",
    )


def test_cache_ancestor_policy_still_rejects_a_non_root_writable_ancestor() -> None:
    # widening the root case must not widen this one: an ancestor owned by another unprivileged
    # user is what the traversal actually has to defend against, and it stays rejected whether or
    # not it carries the sticky bit.
    foreign_uid = 1 if os.getuid() != 1 else 2
    for permissions in (0o777, 0o1777, 0o770):
        with pytest.raises(MaterializationError, match="untrusted uid"):
            _validate_cache_ancestor_stat(
                _directory_stat(uid=foreign_uid, permissions=permissions),
                "foreign writable ancestor",
            )
    # and an ancestor owned by us still may not be group or world writable, because we are not
    # root: anyone in the group could swap a component of the path.
    with pytest.raises(MaterializationError, match="group or world writable"):
        _validate_cache_ancestor_stat(
            _directory_stat(uid=os.getuid(), permissions=0o777),
            "current uid writable ancestor",
        )


def test_cache_rejects_symlinked_parent_and_unsafe_mode(tmp_path: Path) -> None:
    config_bytes, weights_bytes = _artifact_bytes(tmp_path)
    manifest = _manifest(config_bytes, weights_bytes)
    actual_parent = tmp_path / "actual-parent"
    actual_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(actual_parent, target_is_directory=True)

    with pytest.raises(MaterializationError, match="unsafe parent"):
        hydrate_manifest(
            manifest,
            linked_parent / "cache",
            token_fd=_token_fd(),
            snapshot_download_fn=_download_stub(config_bytes, weights_bytes, []),
        )

    unsafe_parent = tmp_path / "unsafe-parent"
    unsafe_parent.mkdir()
    unsafe_parent.chmod(0o770)
    with pytest.raises(MaterializationError, match="group or world writable"):
        hydrate_manifest(
            manifest,
            unsafe_parent / "cache",
            token_fd=_token_fd(),
            snapshot_download_fn=_download_stub(config_bytes, weights_bytes, []),
        )

    cache = tmp_path / "mode-cache"
    hydrate_manifest(
        manifest,
        cache,
        token_fd=_token_fd(),
        snapshot_download_fn=_download_stub(config_bytes, weights_bytes, []),
    )
    cache.chmod(0o770)
    with pytest.raises(MaterializationError, match="group or world writable"):
        validate_manifest_cache(manifest, cache)


def _simulate_modeless_filesystem(monkeypatch, root: Path) -> None:
    """make every path under root read back as 0777/0666 no matter what mode it was created with.

    this is what runpod's moosefs volume does: `mkdir -m 700` and an explicit `chmod 700` both read
    back as 777, and files read back as 666 (measured on a live pod). a mocked stat_result cannot
    exercise this, because the behaviour under test is the interaction between creating a file and
    reading its mode back -- so patch at the syscall boundary and let the real code run.
    """

    real_stat = os.stat
    real_fstat = os.fstat
    real_lstat = os.lstat
    # compare as plain strings. Path.resolve() calls os.stat, which is the function being patched,
    # so using it inside the wrapper recurses until the stack runs out.
    root_prefix = os.path.abspath(root)

    def _force_shared_mode(details: os.stat_result) -> os.stat_result:
        fields = list(details)
        if stat.S_ISDIR(details.st_mode):
            fields[0] = stat.S_IFDIR | 0o777
        elif stat.S_ISREG(details.st_mode):
            fields[0] = stat.S_IFREG | 0o666
        return os.stat_result(tuple(fields))

    def _under_root(path: object) -> bool:
        # only the cache root and its contents live on the simulated volume. the ancestors above it
        # are the test machine's own filesystem, and forcing 0777 onto those would trip the
        # ancestor gate instead -- a different check, guarding a different thing.
        if isinstance(path, int):
            return _fd_under_root(path)
        try:
            return os.path.abspath(os.fsdecode(path)).startswith(root_prefix)
        except (TypeError, ValueError):
            return False

    def _fd_under_root(fd: int) -> bool:
        # resolve the descriptor back to a path through procfs, so an fd opened on an ancestor is
        # not mistaken for one inside the cache tree.
        try:
            return os.readlink(f"/proc/self/fd/{fd}").startswith(root_prefix)
        except OSError:
            return False

    def _patched(real):
        def wrapper(path, *args, **kwargs):
            details = real(path, *args, **kwargs)
            dir_fd = kwargs.get("dir_fd")
            inside = _fd_under_root(dir_fd) if dir_fd is not None else _under_root(path)
            return _force_shared_mode(details) if inside else details

        return wrapper

    def _patched_fstat(fd: int) -> os.stat_result:
        details = real_fstat(fd)
        return _force_shared_mode(details) if _fd_under_root(fd) else details

    monkeypatch.setattr(os, "stat", _patched(real_stat))
    monkeypatch.setattr(os, "lstat", _patched(real_lstat))
    monkeypatch.setattr(os, "fstat", _patched_fstat)


def test_hydration_survives_a_filesystem_that_cannot_store_modes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # runpod mounts its network volume from moosefs, which reports a fixed 0777 for directories and
    # 0666 for files and silently discards chmod. the mode assertion can therefore NEVER be
    # satisfied there, and it rejected every runpod deployment about two seconds into startup --
    # a permanent restart loop that externally looks like a slow image pull.
    #
    # the payload's integrity does not rest on the mode bit: every adapter file is verified against
    # the manifest's sha256 with a before/after identity check around the read.
    config_bytes, weights_bytes = _artifact_bytes(tmp_path)
    manifest = _manifest(config_bytes, weights_bytes)
    cache = tmp_path / "modeless-cache"
    cache.mkdir()
    _simulate_modeless_filesystem(monkeypatch, cache)

    paths = hydrate_manifest(
        manifest,
        cache,
        token_fd=_token_fd(),
        snapshot_download_fn=_download_stub(config_bytes, weights_bytes, []),
    )

    assert paths
    # and a later start must revalidate from that same cache rather than failing the gate.
    validate_manifest_cache(manifest, cache)


def test_mode_check_still_bites_where_modes_are_honoured(tmp_path: Path) -> None:
    # the probe must not become a blanket amnesty: on a filesystem that does store modes, a
    # group-writable cache root is still a real finding and still has to be rejected. without this,
    # the modeless carve-out above would silently disable the check everywhere.
    config_bytes, weights_bytes = _artifact_bytes(tmp_path)
    manifest = _manifest(config_bytes, weights_bytes)
    cache = tmp_path / "honoured-cache"
    hydrate_manifest(
        manifest,
        cache,
        token_fd=_token_fd(),
        snapshot_download_fn=_download_stub(config_bytes, weights_bytes, []),
    )

    cache.chmod(0o770)
    with pytest.raises(MaterializationError, match="group or world writable"):
        validate_manifest_cache(manifest, cache)


def test_mode_probe_leaves_no_entry_when_its_removal_does_not_take(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # the probe creates and removes a child directory. hydration stages into a directory whose
    # exact entry set is then compared against the manifest, so a probe entry that outlives its own
    # rmdir -- which a fuse mount may well do -- fails that comparison and aborts hydration with
    # "adapter cache file set does not exactly match the manifest". probing the parent keeps the
    # directory under verification untouched.
    config_bytes, weights_bytes = _artifact_bytes(tmp_path)
    manifest = _manifest(config_bytes, weights_bytes)
    real_rmdir = os.rmdir

    def rmdir_that_does_not_take(path, *, dir_fd=None):
        if ".flash-mode-probe-" in str(path):
            return None  # report success, leave the entry behind
        if dir_fd is not None:
            return real_rmdir(path, dir_fd=dir_fd)
        return real_rmdir(path)

    monkeypatch.setattr(os, "rmdir", rmdir_that_does_not_take)

    paths = hydrate_manifest(
        manifest,
        tmp_path / "cache",
        token_fd=_token_fd(),
        snapshot_download_fn=_download_stub(config_bytes, weights_bytes, []),
    )

    destination = paths[manifest.adapters[0].checkpoint_id]
    assert {path.name for path in destination.iterdir()} == {
        "adapter_config.json",
        "adapter_model.safetensors",
    }


def test_cache_rejects_hardlinked_materialized_files(tmp_path: Path) -> None:
    config_bytes, weights_bytes = _artifact_bytes(tmp_path)
    manifest = _manifest(config_bytes, weights_bytes)
    cache = tmp_path / "cache"
    paths = hydrate_manifest(
        manifest,
        cache,
        token_fd=_token_fd(),
        snapshot_download_fn=_download_stub(config_bytes, weights_bytes, []),
    )
    destination = next(iter(paths.values())) / "adapter_config.json"
    outside = tmp_path / "same-config.json"
    outside.write_bytes(destination.read_bytes())
    outside.chmod(0o600)
    destination.unlink()
    os.link(outside, destination)

    with pytest.raises(MaterializationError, match="exactly one hard link"):
        validate_manifest_cache(manifest, cache)


def test_locked_cache_revalidates_replacement_before_release(tmp_path: Path) -> None:
    config_bytes, weights_bytes = _artifact_bytes(tmp_path)
    manifest = _manifest(config_bytes, weights_bytes)
    cache = tmp_path / "cache"
    hydrate_manifest(
        manifest,
        cache,
        token_fd=_token_fd(),
        snapshot_download_fn=_download_stub(config_bytes, weights_bytes, []),
    )

    def replace_after_validation() -> None:
        with locked_manifest_cache(manifest, cache) as paths:
            destination = next(iter(paths.values())) / "adapter_config.json"
            destination.write_bytes(b"replaced after validation")

    with pytest.raises(MaterializationError, match=r"size|digest|changed"):
        replace_after_validation()


def test_adapter_without_a_peft_adapter_name_segment_validates(tmp_path: Path) -> None:
    # verl's model_merger produces every adapter flash serves, and it writes
    # "<module>.lora_A.weight" with no adapter-name segment. the validator demanded exactly two
    # leaf parts, so it rejected all of them with "malformed LoRA tensor key" -- after the
    # provider had created and started billing for the app, volume, secret, and pod. the whole
    # materialize suite passed because every fixture here hand-wrote ".default.weight", a shape
    # flash's own exporter never emits.
    config_bytes, weights_bytes = _artifact_bytes(tmp_path, adapter_name="")
    manifest = _manifest(config_bytes, weights_bytes)
    cache = tmp_path / "cache"

    hydrate_manifest(
        manifest,
        cache,
        token_fd=_token_fd(),
        snapshot_download_fn=_download_stub(config_bytes, weights_bytes, []),
    )

    validate_manifest_cache(manifest, cache)


def test_convolution_lora_factors_validate(tmp_path: Path) -> None:
    # flash trains with peft `all-linear`, which on an image model wraps `visual.patch_embed.proj`
    # -- a Conv3d. peft writes that pair as A (r, in_ch, *kernel) / B (out_ch, r, 1, 1, 1), so the
    # factors are 5-D. requiring rank 2 rejected the ENTIRE adapter over that single pair, which
    # made every image adapter undeployable: observed live as a crash loop on both a Modal app and
    # a $0.49/hr RunPod L4, after the provider had already allocated and started billing.
    # vllm loads the pair, warns it cannot wrap a convolution, and leaves it unapplied, so
    # refusing the other 346 pairs is strictly worse than what the engine itself does.
    source = tmp_path / "source"
    source.mkdir()
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": "Qwen/Qwen3.5-9B",
        "revision": BASE_REVISION,
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "proj"],
    }
    config_bytes = json.dumps(config, sort_keys=True).encode()
    (source / "adapter_config.json").write_bytes(config_bytes)
    linear = "base_model.model.layers.0.self_attn.q_proj"
    conv = "base_model.model.model.visual.patch_embed.proj"
    weights_path = source / "adapter_model.safetensors"
    save_file(
        {
            f"{linear}.lora_A.weight": np.zeros((16, 2), dtype=np.float32),
            f"{linear}.lora_B.weight": np.zeros((2, 16), dtype=np.float32),
            f"{conv}.lora_A.weight": np.zeros((16, 3, 2, 4, 4), dtype=np.float32),
            f"{conv}.lora_B.weight": np.zeros((8, 16, 1, 1, 1), dtype=np.float32),
        },
        weights_path,
    )
    weights_bytes = weights_path.read_bytes()
    manifest = _manifest(config_bytes, weights_bytes)
    cache = tmp_path / "cache"

    hydrate_manifest(
        manifest,
        cache,
        token_fd=_token_fd(),
        snapshot_download_fn=_download_stub(config_bytes, weights_bytes, []),
    )

    validate_manifest_cache(manifest, cache)


def test_convolution_lora_factors_with_mismatched_rank_are_refused(tmp_path: Path) -> None:
    # accepting >2-D factors must not stop enforcing rank agreement: rank is A[0] and B[1] for
    # both the 2-D and the convolution shape, so a disagreeing pair is still a broken adapter.
    source = tmp_path / "source"
    source.mkdir()
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": "Qwen/Qwen3.5-9B",
        "revision": BASE_REVISION,
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["proj"],
    }
    config_bytes = json.dumps(config, sort_keys=True).encode()
    (source / "adapter_config.json").write_bytes(config_bytes)
    conv = "base_model.model.model.visual.patch_embed.proj"
    weights_path = source / "adapter_model.safetensors"
    save_file(
        {
            f"{conv}.lora_A.weight": np.zeros((16, 3, 2, 4, 4), dtype=np.float32),
            f"{conv}.lora_B.weight": np.zeros((8, 15, 1, 1, 1), dtype=np.float32),
        },
        weights_path,
    )
    weights_bytes = weights_path.read_bytes()
    manifest = _manifest(config_bytes, weights_bytes)
    cache = tmp_path / "cache"

    with pytest.raises(MaterializationError, match="incompatible LoRA factor shapes"):
        hydrate_manifest(
            manifest,
            cache,
            token_fd=_token_fd(),
            snapshot_download_fn=_download_stub(config_bytes, weights_bytes, []),
        )


def test_lora_tensor_key_with_extra_leaf_segments_is_still_refused(tmp_path: Path) -> None:
    # relaxing the leaf count must not turn into accepting anything: a key carrying more than an
    # optional adapter name before "weight" is not a shape peft or verl produces, and pairing on
    # it would silently merge distinct tensors under one identity.
    config_bytes, weights_bytes = _artifact_bytes(tmp_path, adapter_name="default.extra")
    manifest = _manifest(config_bytes, weights_bytes)
    cache = tmp_path / "cache"

    with pytest.raises(MaterializationError, match="malformed LoRA tensor key"):
        hydrate_manifest(
            manifest,
            cache,
            token_fd=_token_fd(),
            snapshot_download_fn=_download_stub(config_bytes, weights_bytes, []),
        )


def _base_download_stub(landed: list[tuple[str, str | None]], *, fail_on: str | None = None):
    """stand in for snapshot_download, recording what it was asked to fetch."""

    def download(**kwargs):
        repo_id = kwargs["repo_id"]
        if repo_id == fail_on:
            raise RuntimeError("connection reset")
        landed.append((repo_id, kwargs.get("revision")))
        return str(Path(kwargs["cache_dir"]) / "snapshot")

    return download


def _offline_stub(available: set[str]):
    """stand in for the hub's offline resolution, which only sees whether the repo is present."""

    def download(**kwargs):
        if kwargs["repo_id"] not in available:
            raise OSError("not cached")
        return str(Path(kwargs["cache_dir"]) / "snapshot")

    return download


def test_interrupted_base_weight_download_is_not_reported_as_cached(tmp_path: Path) -> None:
    # the failure this guards: the tokenizer lands, the model download dies mid-flight, and the
    # next start reads the cache back as ready. it then seals the engine offline with the
    # bootstrap token already gone, so vllm dies on the missing shard on every restart.
    config_bytes, weights_bytes = _artifact_bytes(tmp_path)
    manifest = _manifest(config_bytes, weights_bytes)
    cache = tmp_path / "cache"
    landed: list[tuple[str, str | None]] = []

    with pytest.raises(MaterializationError, match="base weight download failed"):
        hydrate_base_weights(
            manifest,
            cache,
            token_fd=_token_fd(),
            snapshot_download_fn=_base_download_stub(landed, fail_on="flash-owned/tokenizer"),
        )

    assert landed == [("flash-owned/served-checkpoint", "1" * 40)]
    # the hub resolves both repos offline -- a partial snapshot directory is exactly what it
    # returns for a commit-sha revision -- so only the missing marker can catch this.
    assert not base_weights_are_cached(
        manifest,
        cache,
        snapshot_download_fn=_offline_stub(
            {"flash-owned/served-checkpoint", "flash-owned/tokenizer"}
        ),
    )


def test_completed_base_weight_download_is_reported_as_cached(tmp_path: Path) -> None:
    config_bytes, weights_bytes = _artifact_bytes(tmp_path)
    manifest = _manifest(config_bytes, weights_bytes)
    cache = tmp_path / "cache"
    landed: list[tuple[str, str | None]] = []

    hydrate_base_weights(
        manifest,
        cache,
        token_fd=_token_fd(),
        snapshot_download_fn=_base_download_stub(landed),
    )

    assert landed == [
        ("flash-owned/served-checkpoint", "1" * 40),
        ("flash-owned/tokenizer", "2" * 40),
    ]
    assert base_weights_are_cached(
        manifest,
        cache,
        snapshot_download_fn=_offline_stub(
            {"flash-owned/served-checkpoint", "flash-owned/tokenizer"}
        ),
    )


def test_marker_alone_does_not_survive_an_emptied_cache(tmp_path: Path) -> None:
    # the marker is a claim about a download, not about the bytes still being there. an evicted or
    # wiped volume must still read as not-cached, which is what the hub check contributes.
    config_bytes, weights_bytes = _artifact_bytes(tmp_path)
    manifest = _manifest(config_bytes, weights_bytes)
    cache = tmp_path / "cache"

    hydrate_base_weights(
        manifest,
        cache,
        token_fd=_token_fd(),
        snapshot_download_fn=_base_download_stub([]),
    )

    assert not base_weights_are_cached(
        manifest,
        cache,
        snapshot_download_fn=_offline_stub({"flash-owned/tokenizer"}),
    )


def test_hub_accepts_a_partial_commit_sha_snapshot_as_complete(tmp_path: Path) -> None:
    # this is the upstream behavior the marker exists for, pinned against the real library so a
    # future huggingface_hub cannot quietly make the guard look unnecessary. for a *branch*
    # revision the same call raises, which is why this is easy to disprove with a casual probe.
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    revision = "d" * 40
    repo_dir = tmp_path / "hub" / "models--acme--partial"
    snapshot = repo_dir / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (repo_dir / "refs").mkdir()

    resolved = snapshot_download(
        repo_id="acme/partial",
        revision=revision,
        cache_dir=str(tmp_path / "hub"),
        local_files_only=True,
    )
    assert Path(resolved) == snapshot

    with pytest.raises(LocalEntryNotFoundError):
        snapshot_download(
            repo_id="acme/partial",
            revision="main",
            cache_dir=str(tmp_path / "hub"),
            local_files_only=True,
        )


def _materializing_stub(landed: list[tuple[str, str | None]], files: dict[str, bytes]):
    """stand in for snapshot_download, but actually write the snapshot it claims to have fetched.

    the other stubs return a path that never exists, so a completeness check that reads the
    snapshot sees an empty directory either way and cannot fail. proving the marker is bound to
    real contents needs real files.
    """

    def download(**kwargs):
        snapshot = Path(kwargs["cache_dir"]) / "snapshot" / kwargs["repo_id"].replace("/", "--")
        snapshot.mkdir(parents=True, exist_ok=True)
        for name, payload in files.items():
            (snapshot / name).write_bytes(payload)
        landed.append((kwargs["repo_id"], kwargs.get("revision")))
        return str(snapshot)

    return download


def test_a_shard_lost_after_hydration_is_not_reported_as_cached(tmp_path: Path) -> None:
    """the marker must vouch for the snapshot's contents, not merely for its own existence.

    `snapshot_download(local_files_only=True)` resolves the directory for an exact commit sha even
    when only part of it is present, so a shard evicted or truncated after the marker was written
    used to read back as hydrated. the launcher then skips rehydration and seals the hub offline
    -- and a finalized deployment has already had its artifact secret removed, so vllm crash-loops
    on every restart with no token left to recover.
    """
    config_bytes, weights_bytes = _artifact_bytes(tmp_path)
    manifest = _manifest(config_bytes, weights_bytes)
    cache = tmp_path / "cache"
    files = {"config.json": b"{}", "model-00001.safetensors": b"x" * 64}

    hydrate_base_weights(
        manifest,
        cache,
        token_fd=_token_fd(),
        snapshot_download_fn=_materializing_stub([], files),
    )

    def offline(**kwargs):
        # resolve the same directory hydration wrote, without repopulating it.
        return str(Path(kwargs["cache_dir"]) / "snapshot" / kwargs["repo_id"].replace("/", "--"))

    assert base_weights_are_cached(manifest, cache, snapshot_download_fn=offline)

    # lose one shard, exactly as an eviction or a partial volume restore would.
    served = (
        base_weights_cache_path(cache)
        / "snapshot"
        / "flash-owned--served-checkpoint"
        / "model-00001.safetensors"
    )
    assert served.is_file(), "the fixture must have written the shard for this to test anything"
    served.unlink()

    assert not base_weights_are_cached(manifest, cache, snapshot_download_fn=offline)


def test_same_length_shard_corruption_is_not_reported_as_cached(tmp_path: Path) -> None:
    config_bytes, weights_bytes = _artifact_bytes(tmp_path)
    manifest = _manifest(config_bytes, weights_bytes)
    cache = tmp_path / "cache"
    files = {"config.json": b"{}", "model-00001.safetensors": b"x" * 64}

    hydrate_base_weights(
        manifest,
        cache,
        token_fd=_token_fd(),
        snapshot_download_fn=_materializing_stub([], files),
    )

    def offline(**kwargs):
        return str(Path(kwargs["cache_dir"]) / "snapshot" / kwargs["repo_id"].replace("/", "--"))

    assert base_weights_are_cached(manifest, cache, snapshot_download_fn=offline)

    served = (
        base_weights_cache_path(cache)
        / "snapshot"
        / "flash-owned--served-checkpoint"
        / "model-00001.safetensors"
    )
    original_size = served.stat().st_size
    served.write_bytes(b"y" * original_size)
    assert served.stat().st_size == original_size

    assert not base_weights_are_cached(manifest, cache, snapshot_download_fn=offline)
