from __future__ import annotations

import multiprocessing


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
    runner.add_verified_adapter_revision(run_id, revision)
    context = multiprocessing.get_context("fork")
    output = context.Queue()
    process = context.Process(target=_read_revisions, args=(run_id, output))

    process.start()
    process.join(timeout=10)

    assert process.exitcode == 0
    assert output.get(timeout=2) == (revision,)
