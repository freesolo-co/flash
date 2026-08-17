"""optional dependency metadata and wheel contents for the serving runtime."""

from __future__ import annotations

import os
import subprocess
import tomllib
import zipfile
from pathlib import Path

from flash.engine.worker.backend_common import TRANSFORMERS_REQUIREMENT

ROOT = Path(__file__).resolve().parents[1]
# the generated self-hosted app is rendered from this resource at runtime, so a wheel that omits it
# leaves `flash serve setup` broken on every install that is not an editable checkout.
TEMPLATE_RESOURCE = "flash/serve/backend/templates/modal_app.py.tmpl"
CONTROL_FILES = {
    "flash/serve/control/__init__.py",
    "flash/serve/control/_canonical.py",
    "flash/serve/control/_serialization.py",
    "flash/serve/control/_urls.py",
    "flash/serve/control/credentials.py",
    "flash/serve/control/planning.py",
    "flash/serve/control/types.py",
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
        "pillow>=11.0.0",
        TRANSFORMERS_REQUIREMENT,
        "vllm==0.23.0",
    ]
    assert extras["serve-modal"] == ["modal>=1.0", "fastapi"]
    assert "serve-control" not in extras
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
        assert names >= RUNTIME_FILES | CONTROL_FILES
        # the template is data, not python, so nothing imports it and a packaging miss would only
        # surface the first time a user ran `flash serve setup` off a real install.
        assert TEMPLATE_RESOURCE in names
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = wheel.read(metadata_name).decode()

    assert "Provides-Extra: serve-runtime" in metadata
    assert "Requires-Dist: vllm==0.23.0; extra == 'serve-runtime'" in metadata
    assert "Requires-Dist: vllm==0.19.1; extra == 'gpu'" in metadata


def test_the_shipped_template_renders_an_app_that_pins_this_build() -> None:
    """`render_app()` works from packaged resources and pins the generating distribution.

    reading the template through `importlib.resources` is what an installed wheel does, so this
    covers the path a user hits rather than the source tree the other tests read directly.
    """
    from flash.core.catalog import MODELS
    from flash.serve.backend.generate import flash_requirement, render_app

    source = render_app(MODELS["Qwen/Qwen3.5-4B"])
    requirement = flash_requirement()
    assert f"FLASH_REQUIREMENT = {requirement!r}" in source
    assert "[serve-runtime]==" in requirement
    # the app delegates to the runtime, so it must not carry its own vllm engine mechanics.
    for symbol in ("AsyncEngineArgs", "AsyncLLMEngine", "SamplingParams", "LoRARequest"):
        assert symbol not in source
