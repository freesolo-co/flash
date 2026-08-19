"""packaged app import and immutable serving image contract."""

from __future__ import annotations

import ast
import runpy
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from flash.serve.app.__main__ import _bound_manifest
from flash.serve.app.manifest import build_serving_manifest
from flash.serve.provisioning import LAUNCHER_ABI_ID
from tests.test_serve_app_manifest import _spec_and_inputs

ROOT = Path(__file__).resolve().parents[1]


def test_hydrate_import_does_not_import_vllm_or_provider_sdks() -> None:
    probe = r"""
import builtins
import sys

blocked = ("vllm", "modal", "runpod", "runpod_flash", "supabase")
real_import = builtins.__import__


intercepted = []


def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if name in blocked or name.startswith(tuple(item + "." for item in blocked)):
        intercepted.append(name)
        raise ModuleNotFoundError(name)
    return real_import(name, globals, locals, fromlist, level)


builtins.__import__ = guarded
try:
    __import__("modal")
except ModuleNotFoundError:
    pass
assert intercepted == ["modal"]
intercepted.clear()
from flash.serve.app.manifest import load_serving_manifest
from flash.serve.app.materialize import hydrate_manifest
from flash.serve.provisioning import ServingImage

assert load_serving_manifest
assert hydrate_manifest
assert ServingImage
assert intercepted == []
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


def test_app_package_import_is_project_empty() -> None:
    probe = r"""
import sys
import flash.serve.app

loaded = {
    name
    for name in sys.modules
    if name.startswith("flash.serve.app.")
}
assert loaded == set(), loaded
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


def test_root_bootstrap_has_only_stdlib_module_imports() -> None:
    tree = ast.parse((ROOT / "serve_launch.py").read_text())
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".", 1)[0] if node.module else "")
    assert imported == {"__future__", "contextlib", "os", "signal", "types", "typing"}


def test_root_bootstrap_installs_signal_guard_before_secret_access() -> None:
    bootstrap = runpy.run_path(ROOT / "serve_launch.py", run_name="serve_launch_signal_probe")

    class Observed(Exception):
        pass

    def observe_secret_access():
        handler = signal.getsignal(signal.SIGTERM)
        assert getattr(handler, "__self__", None).__class__.__name__ == "_StartupSignalGuard"
        raise Observed

    bootstrap["_run"].__globals__["_pop_runtime_secrets"] = observe_secret_access
    with pytest.raises(Observed):
        bootstrap["_run"]()


def test_root_bootstrap_pops_secrets_before_every_project_import() -> None:
    probe = r"""
import builtins
import os
import runpy

inference = "inference-import-sentinel"
artifact = "artifact-import-sentinel"
os.environ["FLASH_INFERENCE_TOKEN"] = inference
os.environ["FLASH_ARTIFACT_TOKEN"] = artifact
os.environ["FLASH_SERVING_MANIFEST"] = "invalid"
os.environ["FLASH_SERVING_MANIFEST_ID"] = "0" * 64
os.environ["FLASH_SERVING_IMAGE_DIGEST"] = "sha256:" + "0" * 64
real_import = builtins.__import__
observed = []


class ProjectImportObserved(Exception):
    pass


def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "flash" or name.startswith("flash."):
        assert "FLASH_INFERENCE_TOKEN" not in os.environ
        assert "FLASH_ARTIFACT_TOKEN" not in os.environ
        observed.append(name)
        if name == "flash.serve.app.__main__":
            raise ProjectImportObserved(name)
    return real_import(name, globals, locals, fromlist, level)


builtins.__import__ = guarded
bootstrap = runpy.run_path("serve_launch.py", run_name="serve_launch_probe")
try:
    bootstrap["_run"]()
except ProjectImportObserved:
    pass
else:
    raise AssertionError("project import probe did not fire")
assert "flash._internal.channel" in observed
assert "flash.serve.app.__main__" in observed
assert inference not in repr(observed)
assert artifact not in repr(observed)
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
    # The shared base ships /usr/lib/python3.12/EXTERNALLY-MANAGED, so a pip install into the
    # system interpreter fails with "externally-managed-environment" (PEP 668) unless this is
    # set. Dockerfile.worker sets it for exactly that reason; Dockerfile.serve did not, and the
    # image could never build past its uv bootstrap. Nothing in CI builds this image, so only
    # this assertion keeps the two files' PEP 668 handling from drifting apart again.
    assert "PIP_BREAK_SYSTEM_PACKAGES=1" in worker
    assert "PIP_BREAK_SYSTEM_PACKAGES=1" in source
    assert "python -c 'import sys; assert sys.version_info[:2] == (3, 12)'" in source
    assert "uv sync --frozen --no-dev --extra serve-runtime" in source
    assert "COPY serve_launch.py ./serve_launch.py" in source
    assert 'CMD ["python", "/app/serve_launch.py"]' in source
    assert "python -m flash" not in source
    assert "EXPOSE 8000" in source
    assert len(LAUNCHER_ABI_ID) <= 63
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
