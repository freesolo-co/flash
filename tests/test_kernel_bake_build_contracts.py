"""Exercise cache cleanup and CI input handling without allocating a GPU."""

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _bake_job():
    return yaml.safe_load((ROOT / ".github/workflows/bake-kernel-cache.yml").read_text())["jobs"][
        "bake"
    ]


@pytest.mark.parametrize(
    ("enabled", "metadata"), [(False, True), (False, False), (True, True), (True, False)]
)
def test_worker_build_removes_all_unrequested_cache_artifacts(tmp_path, enabled, metadata):
    from flash.engine.worker.runtime.kernel_warmup import KERNEL_CACHE_ENV_SUBDIRS

    cache = tmp_path / "kernelcache"
    for directory in {*KERNEL_CACHE_ENV_SUBDIRS.values(), ".hidden", "future_backend"}:
        target = cache / directory / "nested/stale.so"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"stale compiled code")
    (cache / "flashinfer/aot").mkdir()
    (cache / "flashinfer/aot/manifest.json").write_text('{"stale": true}')
    if metadata:
        (cache / "mega_cache.json").write_text('{"sm": "sm90"}')
    before = {p.relative_to(cache): p.read_bytes() for p in cache.rglob("*") if p.is_file()}
    dockerfile = (ROOT / "Dockerfile.worker").read_text()
    block = dockerfile[dockerfile.index('RUN if [ "${BUILD_KERNEL_CACHE}"') :].split("\n\n", 1)[0]
    script = block.removeprefix("RUN ").replace("\\\n", "\n")
    script = script.replace("/opt/flash/kernelcache", shlex.quote(str(cache)))
    subprocess.run(
        ["sh", "-eu", "-c", script],
        env={**os.environ, "BUILD_KERNEL_CACHE": str(enabled).lower()},
        check=True,
        capture_output=True,
    )
    after = {p.relative_to(cache): p.read_bytes() for p in cache.rglob("*") if p.is_file()}
    assert after == (before if enabled and metadata else {})


@pytest.mark.parametrize(
    "deadline", ["1", "30", "90", "0", "91", "120", "1440", "-1", "1.5", "", "1e2"]
)
def test_bake_gate_rejects_deadlines_that_outlive_ci(tmp_path, deadline):
    job = _bake_job()
    gate = next(step for step in job["steps"] if step.get("id") == "gate")
    assert gate["env"]["BAKE_DEADLINE_MINUTES"] == "${{ inputs.deadline_minutes }}"
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", "-eu", "-c", gate["run"].replace("${{ matrix.sm }}", "sm90")],
        env={
            **os.environ,
            "REQUESTED_SMS": "sm90",
            "BAKE_DEADLINE_MINUTES": deadline,
            "GITHUB_OUTPUT": str(output),
        },
        capture_output=True,
        text=True,
    )
    if deadline in {"1", "30", "90"}:
        assert result.returncode == 0, result.stderr
        assert output.read_text() == "run=true\n"
        assert int(deadline) + 30 <= job["timeout-minutes"]
    else:
        assert result.returncode != 0
        assert "must be an integer from 1 to 90" in result.stdout
        assert not output.exists()


@pytest.mark.skipif(shutil.which("jq") is None, reason="Workflow resolver uses jq")
@pytest.mark.parametrize(
    ("base", "repository"),
    [
        ("ghcr.io/org/worker:cu128", "ghcr.io/org/worker"),
        ("ghcr.io/org/worker@sha256:" + "a" * 64, "ghcr.io/org/worker"),
        ("registry.example:5000/org/worker:cu128", "registry.example:5000/org/worker"),
        ("registry.example:5000/org/worker", "registry.example:5000/org/worker"),
    ],
)
def test_manual_bake_preserves_full_digest_reference(tmp_path, base, repository):
    reference = repository + "@sha256:" + "a" * 64
    labels = {
        "co.freesolo.flash.kernelcache.fp_cache": "cache",
        "co.freesolo.flash.kernelcache.fp_base": "base",
        "org.opencontainers.image.revision": "source",
    }
    crane = tmp_path / "crane"
    crane.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[1] == 'digest':\n"
        "    assert sys.argv[2:] == ['--full-ref', os.environ['BASE_REF']]\n"
        "    print(os.environ['RESOLVED_REF'])\n"
        "elif sys.argv[1] == 'config':\n"
        "    assert sys.argv[2:] == [os.environ['RESOLVED_REF']]\n"
        f"    print(json.dumps({{'config': {{'Labels': {labels!r}}}}}))\n"
        "else:\n"
        "    raise AssertionError(sys.argv)\n"
    )
    crane.chmod(0o755)
    resolver = next(step for step in _bake_job()["steps"] if step.get("id") == "base")
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", "-eu", "-c", resolver["run"]],
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "BASE_REF": base,
            "RESOLVED_REF": reference,
            "FP_CACHE": "",
            "GITHUB_OUTPUT": str(output),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert dict(line.split("=", 1) for line in output.read_text().splitlines()) == {
        "ref": reference,
        "fp_cache": "cache",
        "fp_base": "base",
        "revision": "source",
    }
