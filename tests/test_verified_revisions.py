from __future__ import annotations

import pytest

import flash.runner.lifecycle.state as runner_state
import flash.runner.results.verified_revisions as verified


def _use_runs_dir(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))


def test_verified_checkpoint_ledger_starts_empty(monkeypatch, tmp_path) -> None:
    _use_runs_dir(monkeypatch, tmp_path)

    assert verified.read_verified_checkpoints("run-a") == frozenset()
    assert verified.verified_checkpoint_generation("run-a") == 0


def test_exact_checkpoint_commits_are_idempotent(monkeypatch, tmp_path) -> None:
    _use_runs_dir(monkeypatch, tmp_path)

    verified.add_verified_checkpoint(
        "run-a", "run-a/final", expected_generation=verified.verified_checkpoint_generation("run-a")
    )
    verified.add_verified_checkpoint(
        "run-a", "run-a/final", expected_generation=verified.verified_checkpoint_generation("run-a")
    )

    assert verified.read_verified_checkpoints("run-a") == frozenset({"run-a/final"})


def test_sibling_checkpoints_are_independent(monkeypatch, tmp_path) -> None:
    _use_runs_dir(monkeypatch, tmp_path)
    verified.add_verified_checkpoint(
        "run-a",
        "run-a/step-20",
        expected_generation=verified.verified_checkpoint_generation("run-a"),
    )
    verified.add_verified_checkpoint(
        "run-a",
        "run-a/step-40",
        expected_generation=verified.verified_checkpoint_generation("run-a"),
    )

    verified.remove_verified_checkpoint("run-a", "run-a/step-20", commit=lambda _retained: None)

    assert verified.read_verified_checkpoints("run-a") == frozenset({"run-a/step-40"})


def test_wrong_run_and_noncanonical_values_are_rejected(monkeypatch, tmp_path) -> None:
    _use_runs_dir(monkeypatch, tmp_path)

    for checkpoint_id in ("run-a", "run-b/final", "run-a@final." + "a" * 40):
        with pytest.raises(ValueError, match="checkpoint"):
            verified.add_verified_checkpoint("run-a", checkpoint_id, expected_generation=0)


def test_generation_fence_rejects_stale_commit(monkeypatch, tmp_path) -> None:
    _use_runs_dir(monkeypatch, tmp_path)
    generation = verified.verified_checkpoint_generation("run-a")
    verified.invalidate_verified_checkpoints("run-a", commit=lambda: None)
    committed = []

    assert not verified.commit_verified_checkpoint(
        "run-a",
        "run-a/final",
        expected_generation=generation,
        commit=lambda: committed.append(True),
    )
    assert committed == []
    assert verified.read_verified_checkpoints("run-a") == frozenset()


def test_successful_fenced_commit_persists_checkpoint_and_state(monkeypatch, tmp_path) -> None:
    _use_runs_dir(monkeypatch, tmp_path)
    generation = verified.verified_checkpoint_generation("run-a")
    committed = []

    assert verified.commit_verified_checkpoint(
        "run-a",
        "run-a/final",
        expected_generation=generation,
        commit=lambda: committed.append(True),
    )
    assert committed == [True]
    assert verified.read_verified_checkpoints("run-a") == frozenset({"run-a/final"})


def test_run_cleanup_invalidates_all_checkpoints_and_advances_generation(
    monkeypatch, tmp_path
) -> None:
    _use_runs_dir(monkeypatch, tmp_path)
    verified.add_verified_checkpoint(
        "run-a", "run-a/final", expected_generation=verified.verified_checkpoint_generation("run-a")
    )
    verified.add_verified_checkpoint(
        "run-a",
        "run-a/step-20",
        expected_generation=verified.verified_checkpoint_generation("run-a"),
    )

    verified.invalidate_verified_checkpoints("run-a", commit=lambda: None)

    assert verified.read_verified_checkpoints("run-a") == frozenset()
    assert verified.verified_checkpoint_generation("run-a") == 1
