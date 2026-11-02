"""exercise complete file-level selection and the required ci gate."""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / ".github/scripts/ci_sharding.py"
spec = importlib.util.spec_from_file_location("ci_sharding", PLUGIN)
sharding = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sharding)


def test_new_files_are_partitioned_exactly_once_and_deterministically():
    paths = {f"tests/test_new_{index}.py" for index in range(17)}
    assigned = sharding.assign_files(paths, 2)
    assert set(assigned) == paths
    assert assigned == sharding.assign_files(sorted(paths, reverse=True), 2)
    assert set(assigned.values()) == {0, 1}
    assert abs(list(assigned.values()).count(0) - list(assigned.values()).count(1)) == 1


def test_measured_heavy_files_are_balanced_without_dropping_new_files(monkeypatch):
    monkeypatch.setattr(sharding, "FILE_SECONDS", {"slow-a.py": 40, "slow-b.py": 35})
    assigned = sharding.assign_files({"slow-a.py", "slow-b.py", "new.py"}, 2)
    assert assigned["slow-a.py"] != assigned["slow-b.py"]
    assert set(assigned) == {"slow-a.py", "slow-b.py", "new.py"}


@pytest.mark.parametrize(("index", "count"), [(None, 2), (-1, 2), (2, 2), (0, 0)])
def test_invalid_shard_fails_before_collection(index, count):
    class Config:
        @staticmethod
        def getoption(name):
            return {"--ci-shard": index, "--ci-shard-count": count}[name]

    with pytest.raises(pytest.UsageError):
        sharding.pytest_configure(Config())


@pytest.mark.parametrize("marker_args", [[], ["-m", "not excluded"]])
def test_real_pytest_discovery_keeps_files_whole_and_all_test_ids(tmp_path, marker_args):
    for name in ("a", "b", "new"):
        (tmp_path / f"test_{name}.py").write_text(
            "import pytest\n"
            "@pytest.mark.parametrize('value', [1, 2])\n"
            "def test_discovered(value): assert value > 0\n"
            + ("@pytest.mark.excluded\ndef test_excluded(): assert False\n" if marker_args else "")
        )
    collected = []
    for shard in range(2):
        report = tmp_path / f"report-{shard}.xml"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "ci_sharding",
                "--ci-shard",
                str(shard),
                "--ci-shard-count",
                "2",
                f"--junitxml={report}",
                *marker_args,
            ],
            cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": str(PLUGIN.parent)},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        collected.append(
            {
                (case.attrib["classname"], case.attrib["name"])
                for case in ET.parse(report).iter("testcase")
            }
        )
    assert not collected[0] & collected[1]
    assert collected[0] | collected[1] == {
        (f"test_{name}", f"test_discovered[{value}]")
        for name in ("a", "b", "new")
        for value in (1, 2)
    }
    assert not {name for name, _ in collected[0]} & {name for name, _ in collected[1]}


@pytest.mark.parametrize("bad_result", ["failure", "cancelled", "skipped", ""])
def test_required_gate_rejects_any_unsuccessful_group(bad_result):
    source = (ROOT / ".github/workflows/ci.yml").read_text()
    gate = source.split("  test-result:\n", 1)[1]
    assert "    needs: [lint, test]\n" in gate
    assert "    if: always()\n" in gate
    for name in ("lint", "test"):
        assert "${{ needs." + name + ".result }}" in gate
    script = textwrap.dedent(gate.split("        run: |\n", 1)[1])
    for key in ("LINT", "TEST"):
        env = {"LINT": "success", "TEST": "success", key: bad_result}
        result = subprocess.run(["bash", "-eu", "-c", script], env=env, capture_output=True)
        assert result.returncode != 0
    result = subprocess.run(
        ["bash", "-eu", "-c", script],
        env={"LINT": "success", "TEST": "success"},
        capture_output=True,
    )
    assert result.returncode == 0


def test_ci_covers_both_python_versions_and_every_shard_without_intra_runner_parallelism():
    source = (ROOT / ".github/workflows/ci.yml").read_text()
    assert 'python-version: ["3.11", "3.12"]' in source
    assert "shard: [0, 1]" in source
    assert "--ci-shard ${{ matrix.shard }}" in source
    assert "--ci-shard-count 2" in source
    assert " -n " not in source
    assert source.count("branches: [main, dev]") == 2
    assert "cancel-in-progress: true" in source
    assert "uv sync --extra server --extra serving --dev --locked" in source
    assert "uv run ruff check ." in source
    assert "uv run ruff format --check ." in source
    assert 'HF_HUB_OFFLINE: "1"' in source
    for reference in re.findall(r"uses: ([^\s]+)", source):
        assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", reference)


def test_control_plane_os_layers_precede_source_and_entrypoint_permissions_follow_it():
    source = (ROOT / "Dockerfile").read_text()
    assert source.index("RUN apt-get update") < source.index("COPY . .")
    assert source.index('RUN if [ "${INSTALL_INFISICAL}"') < source.index("COPY . .")
    assert source.index("COPY . .") < source.index(
        "RUN chmod +x /usr/local/bin/flash-infisical-entrypoint"
    )
    assert source.index("COPY . .") < source.index('RUN pip install --no-cache-dir ".[server]"')
