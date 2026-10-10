"""optional dependency metadata and wheel contents for the serving runtime."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

from flash.engine.worker.train.entry.backend_common import TRANSFORMERS_REQUIREMENT

ROOT = Path(__file__).resolve().parents[1]
# the packaged serving app runs from these modules, so a wheel that omits them leaves
# `flash serve deploy` broken on every install that is not an editable checkout.
CONTROL_FILES = {
    "flash/serve/control/__init__.py",
    "flash/serve/control/_canonical.py",
    "flash/serve/control/_serialization.py",
    "flash/serve/control/_urls.py",
    "flash/serve/control/credentials.py",
    "flash/serve/control/planning.py",
    "flash/serve/control/types.py",
}
APP_FILES = {
    "flash/serve/app/__init__.py",
    "flash/serve/app/__main__.py",
    "flash/serve/app/bootstrap.py",
    "flash/serve/app/http.py",
    "flash/serve/app/manifest.py",
    "flash/serve/app/materialize.py",
    "flash/serve/app/openai.py",
}
RUNTIME_FILES = {
    "flash/serve/runtime/__init__.py",
    "flash/serve/runtime/adapters.py",
    "flash/serve/runtime/engine.py",
    "flash/serve/runtime/errors.py",
    "flash/serve/runtime/multimodal.py",
    "flash/serve/runtime/prompt.py",
    "flash/serve/runtime/structured_outputs.py",
    "flash/serve/runtime/types.py",
}


def test_serve_runtime_extra_is_independent_and_pinned() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extras = project["project"]["optional-dependencies"]

    assert project["project"]["dependencies"] == []
    assert extras["serve-runtime"] == [
        "fastapi>=0.136,<0.137",
        "uvicorn>=0.52,<1",
        "huggingface-hub>=1.2.0",
        "safetensors>=0.8,<1",
        "pillow>=11.0.0",
        TRANSFORMERS_REQUIREMENT,
        "vllm==0.23.0",
    ]
    assert "vllm==0.19.1" in extras["gpu"]
    assert project["tool"]["uv"]["conflicts"] == [[{"extra": "gpu"}, {"extra": "serve-runtime"}]]
    lock = (ROOT / "uv.lock").read_text()
    assert 'name = "modal"\nversion = "1.5.4"' in lock
    # the lockfile must actually RESOLVE the extra, not merely mention it: Dockerfile.serve runs
    # `uv sync --frozen --extra serve-runtime`, and --frozen refuses to re-resolve. adding the extra
    # to pyproject without re-locking leaves provides-extras stale, and the image build dies at that
    # step even though every assertion above still passes.
    assert 'provides-extras = ["gpu", "server", "serving", "serve-runtime", "dev"]' in lock
    # both vllm builds are recorded side by side; that is what the gpu/serve-runtime conflict buys.
    assert (
        '{ name = "vllm", marker = "extra == \'serve-runtime\'", specifier = "==0.23.0" }' in lock
    )
    assert '{ name = "vllm", marker = "extra == \'gpu\'", specifier = "==0.19.1" }' in lock


def test_wheel_contains_runtime_and_declares_extra(tmp_path: Path) -> None:
    output = tmp_path / "dist"
    env = os.environ.copy()
    result = subprocess.run(
        ["uv", "build", "--out-dir", str(output)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = list(output.glob("*.whl"))
    assert len(wheels) == 1

    with zipfile.ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())
        assert names >= APP_FILES | RUNTIME_FILES | CONTROL_FILES
        # derive the complete python module inventory from the source tree, so lazy imports cannot
        # hide omitted wheel files and the same test works from a checkout or an extracted sdist.
        source_modules = {
            path.relative_to(ROOT).as_posix() for path in (ROOT / "flash").rglob("*.py")
        }
        assert names >= source_modules
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = wheel.read(metadata_name).decode()

    # import the changed cli and profile paths from the built wheel with site packages disabled,
    # so the relocation cannot pass by leaking modules in from the editable checkout.
    import_env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(wheels[0].resolve()),
    }
    imported = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import flash.cli.commands.serving.deploy; import flash.serve.deployment.profiles",
        ],
        cwd=tmp_path,
        env=import_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert imported.returncode == 0, imported.stdout + imported.stderr

    assert "Provides-Extra: serve-runtime" in metadata
    for dependency in ("fastapi", "uvicorn", "huggingface-hub", "safetensors"):
        assert f"Requires-Dist: {dependency}" in metadata
    assert "Requires-Dist: vllm==0.23.0; extra == 'serve-runtime'" in metadata
    assert "Requires-Dist: vllm==0.19.1; extra == 'gpu'" in metadata
