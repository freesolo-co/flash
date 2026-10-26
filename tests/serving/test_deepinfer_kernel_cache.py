"""A stale compiler seed must restore exhaustive startup coverage."""

import hashlib
import json

import pytest

from flash.serving.deepinfer.runtime import kernel_cache


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def qualified_seed(tmp_path, monkeypatch):
    source = "flash/serving/deepinfer/runtime/worker.py"
    source_path = tmp_path / "root" / source
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"qualified runtime source")
    artifact = "/root/.dj/cache/kernel/kernel.cubin"
    artifact_path = tmp_path / artifact.lstrip("/")
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"qualified compiled kernel")
    manifest = {
        "libraries": {"torch": "qualified-torch"},
        "cuda": "qualified-nvcc",
        "runtime_sources": {source: digest(source_path.read_bytes())},
        "files": {
            artifact: {
                "size": artifact_path.stat().st_size,
                "sha256": digest(artifact_path.read_bytes()),
            }
        },
    }
    manifest_path = tmp_path / kernel_cache.MANIFEST_PATH.lstrip("/")
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        kernel_cache, "MANIFEST_SHA256", digest(json.dumps(manifest, sort_keys=True).encode())
    )
    monkeypatch.setattr(kernel_cache, "version", lambda name: "qualified-torch")
    monkeypatch.setattr(kernel_cache.subprocess, "check_output", lambda *a, **kw: "qualified-nvcc")
    proof = {
        "native_sha256": kernel_cache.NATIVE_SHA256,
        "revision": kernel_cache.config.RUNTIME_REVISION,
        "gpu": "NVIDIA B200",
        "cuda": "13.2",
        "torch": "qualified-torch",
        "batch_invariant": False,
        "nextn": 5,
        "zero_overhead_scheduling": True,
    }
    return tmp_path, {0: dict(proof), 1: dict(proof)}, source_path, artifact_path


def test_complete_qualified_seed_allows_representative_warmup(qualified_seed):
    root, proof, _, _ = qualified_seed
    assert kernel_cache.verified_compiler_cache(proof, filesystem_root=root)


@pytest.mark.parametrize("damage", ["missing", "truncated", "changed", "symlink"])
def test_changed_kernel_artifact_restores_exhaustive_warmup(qualified_seed, damage):
    root, proof, _, artifact = qualified_seed
    original = artifact.read_bytes()
    artifact.unlink()
    if damage == "truncated":
        artifact.write_bytes(original[:-1])
    elif damage == "changed":
        artifact.write_bytes(b"X" + original[1:])
    elif damage == "symlink":
        target = artifact.with_suffix(".elsewhere")
        target.write_bytes(original)
        artifact.symlink_to(target)
    assert not kernel_cache.verified_compiler_cache(proof, filesystem_root=root)


@pytest.mark.parametrize("new_module", [False, True])
def test_runtime_source_changes_restore_exhaustive_warmup(qualified_seed, new_module):
    root, proof, source, _ = qualified_seed
    if new_module:
        source = source.with_name("new_kernel_settings.py")
    source.write_text("changed source")
    assert not kernel_cache.verified_compiler_cache(proof, filesystem_root=root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("native_sha256", "new-native-build"),
        ("gpu", "NVIDIA H100"),
        ("nextn", 4),
        ("batch_invariant", True),
    ],
)
def test_each_driver_must_match_qualified_runtime(qualified_seed, field, value):
    root, proof, _, _ = qualified_seed
    proof[1][field] = value
    assert not kernel_cache.verified_compiler_cache(proof, filesystem_root=root)


def test_kernel_geometry_change_restores_exhaustive_warmup(qualified_seed, monkeypatch):
    root, proof, _, _ = qualified_seed
    monkeypatch.setattr(kernel_cache.config, "CONTEXT_PARALLEL_SIZE", 4)
    assert not kernel_cache.verified_compiler_cache(proof, filesystem_root=root)


def test_library_or_compiler_change_restores_exhaustive_warmup(qualified_seed, monkeypatch):
    root, proof, _, _ = qualified_seed
    monkeypatch.setattr(kernel_cache, "version", lambda name: "different-library")
    assert not kernel_cache.verified_compiler_cache(proof, filesystem_root=root)
    monkeypatch.setattr(kernel_cache, "version", lambda name: "qualified-torch")
    monkeypatch.setattr(kernel_cache.subprocess, "check_output", lambda *a, **kw: "different-nvcc")
    assert not kernel_cache.verified_compiler_cache(proof, filesystem_root=root)


def test_manifest_tampering_or_missing_seed_restores_exhaustive_warmup(qualified_seed):
    root, proof, _, _ = qualified_seed
    path = root / kernel_cache.MANIFEST_PATH.lstrip("/")
    path.write_text("{}")
    assert not kernel_cache.verified_compiler_cache(proof, filesystem_root=root)
    path.unlink()
    assert not kernel_cache.verified_compiler_cache(proof, filesystem_root=root)


@pytest.mark.parametrize("damage", ["renamed", "missing"])
def test_source_paths_must_match_manifest_exactly(qualified_seed, damage):
    root, proof, source, _ = qualified_seed
    if damage == "renamed":
        source.rename(source.with_name("renamed_worker.py"))
    else:
        source.unlink()
    assert not kernel_cache.verified_compiler_cache(proof, filesystem_root=root)


@pytest.mark.parametrize("damage", ["missing", "unexpected", "malformed"])
def test_incomplete_or_invalid_driver_proof_uses_full_warmup(qualified_seed, damage):
    root, proof, _, _ = qualified_seed
    if damage == "missing":
        proof.pop(1)
    elif damage == "unexpected":
        proof[2] = dict(proof[0])
    else:
        proof[1] = None
    assert not kernel_cache.verified_compiler_cache(proof, filesystem_root=root)


@pytest.mark.parametrize("dependency", ["library", "compiler"])
def test_unavailable_toolchain_uses_full_warmup(qualified_seed, monkeypatch, dependency):
    root, proof, _, _ = qualified_seed

    def unavailable(*args, **kwargs):
        if dependency == "library":
            raise kernel_cache.PackageNotFoundError("torch")
        raise FileNotFoundError("nvcc")

    if dependency == "library":
        monkeypatch.setattr(kernel_cache, "version", unavailable)
    else:
        monkeypatch.setattr(kernel_cache.subprocess, "check_output", unavailable)
    assert not kernel_cache.verified_compiler_cache(proof, filesystem_root=root)


def test_invalid_manifest_json_uses_full_warmup(qualified_seed):
    root, proof, _, _ = qualified_seed
    (root / kernel_cache.MANIFEST_PATH.lstrip("/")).write_text("{truncated")
    assert not kernel_cache.verified_compiler_cache(proof, filesystem_root=root)


@pytest.mark.parametrize(
    "name", [None, "/root/other/kernel.cubin", "/root/.dj/cache/../kernel.cubin"]
)
def test_empty_or_out_of_bounds_artifact_manifest_uses_full_warmup(
    qualified_seed, monkeypatch, name
):
    root, proof, _, _ = qualified_seed
    path = root / kernel_cache.MANIFEST_PATH.lstrip("/")
    manifest = json.loads(path.read_text())
    manifest["files"] = {}
    if name is not None:
        artifact = root / name.lstrip("/")
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"valid bytes at an invalid location")
        manifest["files"][name] = {
            "size": artifact.stat().st_size,
            "sha256": digest(artifact.read_bytes()),
        }
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        kernel_cache, "MANIFEST_SHA256", digest(json.dumps(manifest, sort_keys=True).encode())
    )
    assert not kernel_cache.verified_compiler_cache(proof, filesystem_root=root)
