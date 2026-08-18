"""artifact hydration, cache integrity, token, and current adapter contract."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from flash.serve.app.manifest import ArtifactFile, build_serving_manifest
from flash.serve.app.materialize import (
    MaterializationError,
    _validate_cache_ancestor_stat,
    adapter_cache_path,
    hydrate_manifest,
    locked_manifest_cache,
    validate_manifest_cache,
)
from tests.test_serve_app_manifest import BASE_REVISION, _spec_and_inputs

TOKEN = "scoped-artifact-token-sentinel"


def _artifact_bytes(tmp_path: Path, **config_overrides: object) -> tuple[bytes, bytes]:
    source = tmp_path / "source"
    source.mkdir()
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": "Qwen/Qwen3.5-4B",
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
    save_file(
        {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight": np.zeros(
                (16, 2), dtype=np.float32
            ),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight": np.zeros(
                (2, 16), dtype=np.float32
            ),
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


def test_hydrate_forwards_exact_source_patterns_and_closes_token_fd(tmp_path: Path) -> None:
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
    destination = paths[manifest.adapters[0].adapter_revision]
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


def test_adapter_revision_missing_null_and_empty_are_unspecified(tmp_path: Path) -> None:
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
    with pytest.raises(MaterializationError, match="group or world writable"):
        _validate_cache_ancestor_stat(
            _directory_stat(uid=0, permissions=0o777),
            "root writable non-sticky ancestor",
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
