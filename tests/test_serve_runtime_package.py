"""optional dependency metadata and wheel contents for the serving runtime."""

from __future__ import annotations

import os
import subprocess
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
        "pillow>=11.0.0",
        "transformers>=5.0.0",
        "vllm==0.23.0",
    ]
    assert all("serve-runtime" not in dependency for dependency in extras["serve-modal"])
    assert "vllm==0.19.1" in extras["gpu"]
    assert project["tool"]["uv"]["conflicts"] == [[{"extra": "gpu"}, {"extra": "serve-runtime"}]]


def test_wheel_contains_runtime_and_declares_extra(tmp_path: Path) -> None:
    output = tmp_path / "dist"
    env = os.environ.copy()
    result = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(output)],
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
        assert names >= RUNTIME_FILES
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = wheel.read(metadata_name).decode()

    assert "Provides-Extra: serve-runtime" in metadata
    assert "Requires-Dist: vllm==0.23.0; extra == 'serve-runtime'" in metadata
    assert "Requires-Dist: vllm==0.19.1; extra == 'gpu'" in metadata
