"""Exercise real capsule starts, source verification, workers and spawned uploaders offline."""

from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from flash.providers._lifecycle.bootstrapping.attempt import AttemptState
from flash.runtime_capsule import build_capsule
from flash.runtime_capsule.profiles import get_profile
from flash.snapshot.archive import build_source_archive, descriptor_for_archive


@pytest.fixture
def capsule_run(tmp_path):
    # include /tmp/ even on macos, where pytest normally uses /private/var/folders/.
    tmp_path = tmp_path / "tmp" / "worker"
    tmp_path.mkdir(parents=True)
    root = tmp_path / "build"
    repo = Path(__file__).resolve().parents[1]
    for source, _member in get_profile("instance-bootstrap").sources:
        target = root / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / source, target)
    payload_path = tmp_path / "payload.json"
    code_root = tmp_path / "runcode"
    metrics = tmp_path / "metrics.json"
    bootstrap = root / "flash/providers/_lifecycle/bootstrapping/bootstrap.py"
    text = bootstrap.read_text()
    # redirect original scratch paths before inserting paths that may themselves contain /tmp/.
    text = text.replace("/tmp/", str(tmp_path) + "/")
    text = text.replace('"/root/flash/payload.json"', repr(str(payload_path)))
    text = text.replace('"/runcode"', repr(str(code_root)))
    bootstrap.write_text(text)
    capsule = tmp_path / "capsule.pyz"
    capsule.write_bytes(build_capsule("instance-bootstrap", root=root)[0])

    package = tmp_path / "source/flash"
    (package / "engine/support").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "engine/__init__.py").write_text("")
    (package / "engine/support/__init__.py").write_text("")
    worker = f"""
import json
import time
from pathlib import Path
root = Path({str(tmp_path)!r})
with (root / 'worker-starts').open('a') as output:
    output.write('started\\n')
(root / 'started').touch()
deadline = time.monotonic() + 15
while not (root / 'release').exists():
    if time.monotonic() >= deadline:
        raise RuntimeError('test worker timed out')
    time.sleep(0.01)
Path({str(metrics)!r}).write_text(json.dumps({{'step': 2}}))
print('NODE DONE', flush=True)
"""
    (package / "engine/support/worker_entrypoint.py").write_text(worker)
    archive = build_source_archive(package_dir=package)
    archive_path = tmp_path / "source.zip"
    archive_path.write_bytes(archive)
    descriptor = descriptor_for_archive(archive, "a" * 40).to_dict()
    created = time.time()
    payload = {
        "run_id": "run-reentry",
        "attempt": 0,
        "phase": "sft",
        "seed": 17,
        "flash_arm": "modal",
        "hf_repo": "test/repo",
        "hf_prefix": "sft/run-reentry",
        "env": {},
        "extra_pip": [],
        "job_spec_json": "{}",
        "source_snapshot": descriptor,
        "run_created_at": created,
        "run_max_wall_seconds": 60,
        "deadline_at": created + 60,
    }
    payload_path.write_text(json.dumps(payload))
    # only the artifact transport is substituted; bootstrap and subprocess orchestration are real.
    (tmp_path / "huggingface_hub.py").write_text(f"""
import json
import os
from pathlib import Path
root = Path({str(tmp_path)!r})
def hf_hub_download(**kwargs):
    return str(root / 'source.zip')
class HfApi:
    def __init__(self, **kwargs):
        pass
    def upload_file(self, path_or_fileobj, path_in_repo, **kwargs):
        if path_in_repo.endswith('modal_attempt0.json'):
            if (root / 'fail-upload').exists():
                raise RuntimeError('simulated artifact outage')
            marker = Path(path_or_fileobj).read_text()
            with (root / 'published-markers').open('a') as output:
                output.write(json.dumps(json.loads(marker)) + '\\n')
""")
    processes = []

    def start():
        process = subprocess.Popen(
            [sys.executable, str(capsule), "bootstrap"],
            cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": str(tmp_path)},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append(process)
        return process

    yield tmp_path, payload, start
    for process in processes:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)


def _finish(process, expected=0):
    output, _ = process.communicate(timeout=25)
    assert process.returncode == expected, output
    return output


def _markers(root):
    return [json.loads(line) for line in (root / "published-markers").read_text().splitlines()]


@pytest.mark.parametrize("concurrent", [False, True])
def test_duplicate_capsule_runs_train_once_and_preserve_attested_success(capsule_run, concurrent):
    root, payload, start = capsule_run
    first = start()
    if concurrent:
        deadline = time.monotonic() + 10
        while not (root / "started").exists():
            assert first.poll() is None, first.communicate()[0]
            assert time.monotonic() < deadline, "worker did not start"
            time.sleep(0.01)
        second = start()
        assert select.select([second.stdout], [], [], 10)[0], "duplicate did not reach the lock"
        assert "duplicate start waiting" in second.stdout.readline()
        assert first.poll() is None
        assert second.poll() is None
    (root / "release").touch()
    assert "NODE DONE" in _finish(first)
    original = _markers(root)[0]
    if not concurrent:
        second = start()
    assert "replaying the persisted terminal result" in _finish(second)
    assert (root / "worker-starts").read_text().splitlines() == ["started"]
    assert _markers(root) == [original, original]
    assert original["ok"] is True
    assert original["source_attestation"]["sha256"] == payload["source_snapshot"]["sha256"]


def test_reentry_republishes_after_upload_failure_even_after_deadline(capsule_run):
    root, payload, start = capsule_run
    (root / "release").touch()
    (root / "fail-upload").touch()
    _finish(start())
    assert not (root / "published-markers").exists()
    state = AttemptState(str(root / "runcode/run-reentry-attempt-0"), payload)
    original = state.read()
    (root / "fail-upload").unlink()
    payload["run_created_at"] = 1
    payload["deadline_at"] = 61
    (root / "payload.json").write_text(json.dumps(payload))
    _finish(start())
    assert _markers(root) == [original]
    assert (root / "worker-starts").read_text().splitlines() == ["started"]


def test_reentry_preserves_failure_without_running_worker(capsule_run):
    root, _payload, start = capsule_run
    (root / "source.zip").write_bytes(b"corrupt archive")
    _finish(start(), expected=1)
    original = _markers(root)[0]
    _finish(start(), expected=1)
    assert _markers(root) == [original, original]
    assert original["ok"] is False
    assert "source_attestation" not in original
    assert not (root / "worker-starts").exists()


def test_reentry_rejects_changed_snapshot_without_overwriting_result(capsule_run):
    root, payload, start = capsule_run
    (root / "release").touch()
    _finish(start())
    original = _markers(root)[0]
    payload["source_snapshot"]["revision"] = "b" * 40
    (root / "payload.json").write_text(json.dumps(payload))
    assert "different attempt payload" in _finish(start(), expected=1)
    assert _markers(root) == [original]


def test_first_terminal_result_survives_later_writer(tmp_path):
    payload = {"run_id": "run-1", "attempt": 0}
    state = AttemptState(str(tmp_path / "run-1-attempt-0"), payload)
    success = {"run_id": "run-1", "attempt": 0, "ok": True}
    failure = {**success, "ok": False}
    assert state.save(success) == success
    assert state.save(failure) == success
