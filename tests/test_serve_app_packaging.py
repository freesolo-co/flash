"""packaged app import and immutable serving image contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from flash.serve.app.__main__ import _bound_manifest
from flash.serve.app.manifest import build_serving_manifest
from tests.test_serve_app_manifest import _spec_and_inputs

ROOT = Path(__file__).resolve().parents[1]


def test_hydrate_import_does_not_import_vllm_or_provider_sdks() -> None:
    probe = r"""
import builtins
import sys

blocked = ("vllm", "modal", "runpod", "runpod_flash", "supabase")
real_import = builtins.__import__


def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if name == blocked or name.startswith(tuple(item + "." for item in blocked)):
        raise ModuleNotFoundError(name)
    return real_import(name, globals, locals, fromlist, level)


builtins.__import__ = guarded
from flash.serve.app.manifest import load_serving_manifest
from flash.serve.app.materialize import hydrate_manifest

assert load_serving_manifest
assert hydrate_manifest
for name in blocked:
    assert name not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", probe],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT), "PYTHONNOUSERSITE": "1", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_manifest_requires_external_manifest_and_image_bindings(
    monkeypatch, tmp_path: Path
) -> None:
    manifest = build_serving_manifest(*_spec_and_inputs())
    path = tmp_path / "manifest.json"
    path.write_text(manifest.canonical_json())
    monkeypatch.setenv("FLASH_SERVING_MANIFEST_ID", manifest.manifest_id)
    monkeypatch.setenv("FLASH_SERVING_IMAGE_DIGEST", manifest.expected_oci_digest)
    assert _bound_manifest(str(path)) == manifest

    monkeypatch.setenv("FLASH_SERVING_MANIFEST_ID", "0" * 64)
    with pytest.raises(RuntimeError, match="external binding"):
        _bound_manifest(str(path))


def test_dockerfile_serve_uses_existing_cuda_family_and_frozen_lock() -> None:
    source = (ROOT / "Dockerfile.serve").read_text()
    worker = (ROOT / "Dockerfile.worker").read_text()
    base = next(line for line in worker.splitlines() if line.startswith("FROM "))

    assert base in source
    assert "python -c 'import sys; assert sys.version_info[:2] == (3, 12)'" in source
    assert "uv sync --frozen --no-dev --extra serve-runtime" in source
    assert 'CMD ["python", "-m", "flash.serve.app", "serve"]' in source
    assert "EXPOSE 8000" in source
    for forbidden in (
        "arg hf_token",
        "env hf_token",
        "copy serving-manifest",
        "modal",
        "runpod",
        "supabase",
        "freesolo_internal_key",
    ):
        assert forbidden not in source.lower()
