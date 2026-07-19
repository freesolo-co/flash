from __future__ import annotations

import json
import multiprocessing
import subprocess
import sys
from pathlib import Path

import pytest


def _complete_checkpoint(run_id: str, revision: str, barrier) -> None:
    import flash.runner as runner

    barrier.wait(timeout=10)
    runner.mark_checkpoint_deployed(
        run_id,
        {
            "state": "ready",
            "endpoint_name": "https://serve.example",
            "adapter_revision": revision,
        },
        verification_generation=runner.verified_adapter_revision_generation(run_id),
    )


def _write_stale_status(run_id: str, loaded, release) -> None:
    import flash.runner as runner

    status = runner.get_status(run_id)
    loaded.set()
    if not release.wait(timeout=10):
        raise TimeoutError("stale status writer was not released")
    status.last_heartbeat = {"step": 1}
    runner._save_status(status)


def _read_revisions(run_id: str, output) -> None:
    import flash.runner as runner

    output.put(tuple(sorted(runner.read_verified_adapter_revisions(run_id))))


def _new_status(runner, run_id: str):
    return runner.RunStatus(
        run_id=run_id,
        state="done",
        spec={"model": "Qwen/Qwen3.5-0.8B", "algorithm": "sft"},
    )


@pytest.mark.parametrize("helper_name", ["mark_deployed", "mark_checkpoint_deployed"])
@pytest.mark.parametrize(
    "revision",
    [
        None,
        "ready-commit@final.short",
        "another-run@final." + "a" * 40,
    ],
)
def test_ready_commit_helpers_reject_noncanonical_revision(
    monkeypatch, tmp_path, helper_name, revision
):
    import flash.runner as runner

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    run_id = "ready-commit"
    runner._save_status(_new_status(runner, run_id))
    deployment = {"state": "ready", "endpoint_name": "https://serve.example"}
    if revision is not None:
        deployment["adapter_revision"] = revision

    with pytest.raises(ValueError, match="full same-run adapter revision"):
        getattr(runner, helper_name)(
            run_id,
            deployment,
            verification_generation=runner.verified_adapter_revision_generation(run_id),
        )

    assert runner.get_status(run_id).deployment is None
    assert runner.read_verified_adapter_revisions(run_id) == frozenset()


@pytest.mark.parametrize(
    "raw",
    [
        ["legacy-revision"],
        {"generation": 0, "revisions": ["legacy-revision"]},
    ],
)
def test_verified_revision_ledger_rejects_legacy_shapes(monkeypatch, tmp_path, raw):
    import flash.runner as runner

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    run_id = "strict-revision-ledger"
    path = Path(runner.runs_file_path(run_id, ".verified-revisions"))
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="invalid verified"):
        runner.read_verified_adapter_revisions(run_id)


def test_stale_generation_cannot_commit_ready_revision(monkeypatch, tmp_path):
    import flash.runner as runner

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    run_id = "stale-generation"
    runner._save_status(_new_status(runner, run_id))
    revision = f"{run_id}@final." + "e" * 40
    stale_generation = runner.verified_adapter_revision_generation(run_id)

    assert runner.clear_verified_adapter_revisions(run_id) == stale_generation + 1
    status = runner.mark_checkpoint_deployed(
        run_id,
        {
            "state": "ready",
            "endpoint_name": "https://serve.example",
            "adapter_revision": revision,
        },
        verification_generation=stale_generation,
    )

    assert status.deployment is None
    assert runner.read_verified_adapter_revisions(run_id) == frozenset()


def test_concurrent_checkpoint_completions_preserve_both_revisions(monkeypatch, tmp_path):
    import flash.runner as runner

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    run_id = "concurrent-revisions"
    runner._save_status(_new_status(runner, run_id))
    revisions = [f"{run_id}@step-20." + "a" * 40, f"{run_id}@step-40." + "b" * 40]
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(3)
    processes = [
        context.Process(target=_complete_checkpoint, args=(run_id, revision, barrier))
        for revision in revisions
    ]

    for process in processes:
        process.start()
    barrier.wait(timeout=10)
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    assert runner.read_verified_adapter_revisions(run_id) == frozenset(revisions)


def test_stale_status_process_cannot_erase_verified_revision(monkeypatch, tmp_path):
    import flash.runner as runner

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    run_id = "stale-status-revision"
    runner._save_status(_new_status(runner, run_id))
    revision = f"{run_id}@step-20." + "c" * 40
    context = multiprocessing.get_context("fork")
    loaded = context.Event()
    release = context.Event()
    process = context.Process(target=_write_stale_status, args=(run_id, loaded, release))
    process.start()
    assert loaded.wait(timeout=10)

    runner.mark_checkpoint_deployed(
        run_id,
        {
            "state": "ready",
            "endpoint_name": "https://serve.example",
            "adapter_revision": revision,
        },
        verification_generation=runner.verified_adapter_revision_generation(run_id),
    )
    release.set()
    process.join(timeout=10)

    assert process.exitcode == 0
    assert runner.get_status(run_id).last_heartbeat == {"step": 1}
    assert runner.read_verified_adapter_revisions(run_id) == frozenset({revision})
    assert "verified_adapter_revisions" not in runner.get_status(run_id).to_dict()


def test_verified_revision_ledger_survives_new_process(monkeypatch, tmp_path):
    import flash.runner as runner

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    run_id = "restart-revision"
    revision = f"{run_id}@final." + "d" * 40
    runner.add_verified_adapter_revision(
        run_id,
        revision,
        expected_generation=runner.verified_adapter_revision_generation(run_id),
    )
    context = multiprocessing.get_context("fork")
    output = context.Queue()
    process = context.Process(target=_read_revisions, args=(run_id, output))

    process.start()
    process.join(timeout=10)

    assert process.exitcode == 0
    assert output.get(timeout=2) == (revision,)


def test_verified_revision_ledger_fails_closed_without_fcntl(monkeypatch):
    # fcntl is unix-only; on platforms without it the ledger must fail closed
    # rather than silently skip its cross-process lock.
    from flash.runner import verified_revisions

    monkeypatch.setattr(verified_revisions, "fcntl", None)
    with pytest.raises(RuntimeError, match="verified-revision locking is unavailable"):
        verified_revisions.read_verified_adapter_revisions("run-without-fcntl")


def test_cli_import_chain_survives_missing_fcntl():
    # reproduces the windows `flash login` crash: fcntl is unix-only, so the cli
    # import chain (flash.cli -> flash.runner -> verified_revisions) must not
    # hard-depend on it. blocking fcntl mimics a platform that lacks the module.
    code = (
        "import sys; sys.modules['fcntl'] = None; "
        "import flash.runner.verified_revisions; "
        "import flash.cli"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
