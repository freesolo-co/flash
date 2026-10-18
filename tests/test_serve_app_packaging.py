"""packaged app import and immutable serving image contract."""

from __future__ import annotations

import ast
import runpy
import signal
import subprocess
import sys
from pathlib import Path

import pytest

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


def test_dockerfile_serve_uses_existing_cuda_family_and_frozen_lock() -> None:
    source = (ROOT / "Dockerfile.serve").read_text()
    worker = (ROOT / "Dockerfile.worker").read_text()
    base = next(line for line in source.splitlines() if line.startswith("FROM "))

    # the two images deliberately no longer share one base. the worker is pinned to torch 2.10 /
    # cu12.8 because that is what its FlashAttention wheels are built for; the serving image runs
    # vllm 0.23.0, whose compiled vllm/_C extension links libcudart.so.13, so a cuda 12 base fails
    # at import with "ImportError: libcudart.so.13". asserting the two FROM lines were identical
    # therefore asserted something that cannot be true -- what actually has to hold is that each
    # image states a pytorch base whose cuda matches its own stack.
    assert base.startswith("FROM pytorch/pytorch:")
    assert "-cuda13." in base, "serving needs a cuda 13 base for vllm's compiled extension"
    assert next(line for line in worker.splitlines() if line.startswith("FROM ")).startswith(
        "FROM pytorch/pytorch:"
    )
    assert "libcudart.so.13" in source, "the cuda pairing must be asserted at build time"
    # The shared base ships /usr/lib/python3.12/EXTERNALLY-MANAGED, so a pip install into the
    # system interpreter fails with "externally-managed-environment" (PEP 668) unless this is
    # set. Dockerfile.worker sets it for exactly that reason; Dockerfile.serve did not, and the
    # image could never build past its uv bootstrap. Nothing in CI builds this image, so only
    # this assertion keeps the two files' PEP 668 handling from drifting apart again.
    assert "PIP_BREAK_SYSTEM_PACKAGES=1" in worker
    assert "PIP_BREAK_SYSTEM_PACKAGES=1" in source
    assert "python -c 'import sys; assert sys.version_info[:2] == (3, 12)'" in source
    sync = "RUN uv sync --frozen --no-dev --extra serve-runtime"
    repair = "RUN /opt/flash-venv/bin/python /app/docker/patch_vllm_moe_lora.py"
    verify = "/opt/flash-venv/bin/python /app/docker/patch_vllm_moe_lora.py --verify"
    cleanup = "rm /app/docker/patch_vllm_moe_lora.py"
    assert sync in source
    assert "COPY docker/patch_vllm_moe_lora.py ./docker/patch_vllm_moe_lora.py" in source
    assert source.index(sync) < source.index(repair) < source.index(verify) < source.index(cleanup)
    assert "COPY serve_launch.py ./serve_launch.py" in source
    assert 'CMD ["python", "/app/serve_launch.py"]' in source
    assert "python -m flash" not in source
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


def test_modal_wrapper_is_packaged_under_the_image_import_root() -> None:
    source = (ROOT / "Dockerfile.serve").read_text()
    lines = source.splitlines()
    assert "COPY flash ./flash" in lines, "digest-pinned image must package the modal wrapper"
    copy_index = lines.index("COPY flash ./flash")
    assert lines.index("WORKDIR /app") < copy_index
    assert copy_index < lines.index("RUN uv sync --frozen --no-dev --extra serve-runtime")
    assert "UV_PROJECT_ENVIRONMENT=/opt/flash-venv" in source
    assert "PATH=/opt/flash-venv/bin:${PATH}" in source
    assert "--no-editable" not in source
    assert (ROOT / "flash/serve/provisioning/modal/planning/wrapper.py").is_file()

    lock = (ROOT / "uv.lock").read_text()
    project_start = lock.index('name = "freesolo-flash"')
    project_end = lock.index("[[package]]", project_start)
    assert 'source = { editable = "." }' in lock[project_start:project_end]
