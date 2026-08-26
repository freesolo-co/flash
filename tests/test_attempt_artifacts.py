from __future__ import annotations

import json

import pytest

from flash.providers.artifacts import attempts
from flash.runner.lifecycle.protocol import (
    ProgressRecord,
    ResultManifest,
    canonical_bytes,
    digest_record,
    progress_path,
    result_path,
)

SOURCE = {
    "kind": "flash-source-snapshot",
    "format_version": 1,
    "archive_path": "source/" + "a" * 64 + "/flash-source.zip",
    "sha256": "a" * 64,
    "size": 1,
    "revision": "b" * 40,
}
ATTESTATION = {
    "kind": "flash-source-attestation",
    "format_version": 1,
    "sha256": "a" * 64,
    "revision": "b" * 40,
    "run_id": "run-1",
    "attempt": 2,
    "fence": 9,
}


def _progress(sequence: int, previous: ProgressRecord | None = None) -> ProgressRecord:
    return ProgressRecord(
        run_id="run-1",
        phase_namespace="rl",
        attempt_id=2,
        fence=9,
        sequence=sequence,
        previous_digest=digest_record(previous.to_dict()) if previous else None,
        occurred_at=100.0 + sequence,
        kind="attempt_started" if sequence == 1 else "progressed",
        phase="boot" if sequence == 1 else "rl_step",
        training_entered=sequence > 1,
        completed_steps=max(0, sequence - 1),
    )


def _result(**updates) -> ResultManifest:
    values = {
        "run_id": "run-1",
        "phase_namespace": "rl",
        "attempt_id": 2,
        "fence": 9,
        "outcome": "succeeded",
        "failure_class": None,
        "started_at": 100.0,
        "finished_at": 120.0,
        "training_entered": True,
        "completed_steps": 1,
        "metrics": {"step": 1},
        "checkpoint": {},
        "artifacts": {"adapter": "published"},
        "source_attestation": ATTESTATION,
        "diagnostics": {},
    }
    values.update(updates)
    return ResultManifest(**values)


def test_snapshot_lists_exact_attempt_prefix_at_pinned_revision(monkeypatch) -> None:
    import huggingface_hub

    calls = []
    prefix = "rl/run-1/attempts/2-9"

    class Api:
        def __init__(self, *, token):
            calls.append(("init", token))

        @staticmethod
        def repo_info(**kwargs):
            calls.append(("repo_info", kwargs))
            return type("Info", (), {"sha": "c" * 40})()

        @staticmethod
        def list_repo_tree(**kwargs):
            calls.append(("list_repo_tree", kwargs))
            return [
                huggingface_hub.RepoFile(
                    path=f"{prefix}/progress/nested/ignored.json", size=1, oid="a"
                ),
                huggingface_hub.RepoFile(path=f"{prefix}/progress/record.json", size=1, oid="b"),
                type("RepoFolder", (), {"path": f"{prefix}/progress"})(),
                huggingface_hub.RepoFile(path="other/file.json", size=1, oid="c"),
            ]

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    revision, paths = attempts._repo_snapshot("org/repo", prefix=prefix)

    assert revision == "c" * 40
    assert paths == [
        f"{prefix}/progress/nested/ignored.json",
        f"{prefix}/progress/record.json",
    ]
    assert calls[-1] == (
        "list_repo_tree",
        {
            "repo_id": "org/repo",
            "path_in_repo": prefix,
            "recursive": True,
            "revision": "c" * 40,
            "repo_type": "dataset",
        },
    )


def test_snapshot_missing_attempt_directory_is_empty(monkeypatch) -> None:
    import httpx
    import huggingface_hub
    from huggingface_hub.errors import RemoteEntryNotFoundError

    class Api:
        def __init__(self, *, token):
            del token

        @staticmethod
        def repo_info(**_kwargs):
            return type("Info", (), {"sha": "c" * 40})()

        @staticmethod
        def list_repo_tree(**_kwargs):
            response = httpx.Response(404, request=httpx.Request("GET", "https://example.test"))
            raise RemoteEntryNotFoundError("missing", response=response)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)

    assert attempts._repo_snapshot("org/repo", prefix="rl/run-1/attempts/2-9") == (
        "c" * 40,
        [],
    )


@pytest.mark.parametrize(
    "error",
    [PermissionError("authentication denied"), OSError("transport down")],
)
def test_snapshot_propagates_listing_failures(monkeypatch, error) -> None:
    import huggingface_hub

    class Api:
        def __init__(self, *, token):
            del token

        @staticmethod
        def repo_info(**_kwargs):
            return type("Info", (), {"sha": "c" * 40})()

        @staticmethod
        def list_repo_tree(**_kwargs):
            raise error

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)

    with pytest.raises(type(error), match=str(error)):
        attempts._repo_snapshot("org/repo", prefix="rl/run-1/attempts/2-9")


def test_reads_latest_verified_progress_and_single_result(monkeypatch) -> None:
    first = _progress(1)
    second = _progress(2, first)
    result = _result()
    payloads = {
        progress_path(first): canonical_bytes(first.to_dict()),
        progress_path(second): canonical_bytes(second.to_dict()),
        result_path(result): canonical_bytes(result.to_dict()),
    }
    monkeypatch.setattr(
        attempts, "_repo_snapshot", lambda _repo, *, prefix: ("c" * 40, list(payloads))
    )
    monkeypatch.setattr(
        attempts,
        "_download_bytes",
        lambda _repo, path, *, revision: payloads[path],
    )
    monkeypatch.setattr(attempts.time, "time", lambda: 130.0)

    observed = attempts.read_attempt_artifacts(
        "org/repo",
        phase="rl",
        run_id="run-1",
        attempt_id=2,
        fence=9,
        source_snapshot=SOURCE,
    )

    assert observed.progress is None
    assert observed.result["outcome"] == "succeeded"
    assert observed.result["receipt"]["path"] == result_path(result)


def test_repeated_polls_download_result_before_progress(monkeypatch) -> None:
    first = _progress(1)
    second = _progress(2, first)
    result = _result()
    payloads = {
        progress_path(first): canonical_bytes(first.to_dict()),
        progress_path(second): canonical_bytes(second.to_dict()),
        result_path(result): canonical_bytes(result.to_dict()),
    }
    paths = list(payloads)
    monkeypatch.setattr(attempts, "_repo_snapshot", lambda _repo, *, prefix: ("c" * 40, paths))
    downloads = []

    def download(_repo, path, *, revision):
        downloads.append(path)
        return payloads[path]

    monkeypatch.setattr(attempts, "_download_bytes", download)

    for _ in range(2):
        observed = attempts.read_attempt_artifacts(
            "org/repo",
            phase="rl",
            run_id="run-1",
            attempt_id=2,
            fence=9,
            source_snapshot=SOURCE,
        )
        assert observed.progress is None
        assert observed.result["outcome"] == "succeeded"

    assert downloads == [result_path(result), result_path(result)]


def test_poll_without_result_replays_strict_progress_chain(monkeypatch) -> None:
    first = _progress(1)
    second = _progress(2, first)
    payloads = {
        progress_path(first): canonical_bytes(first.to_dict()),
        progress_path(second): canonical_bytes(second.to_dict()),
    }
    downloads = []
    monkeypatch.setattr(
        attempts, "_repo_snapshot", lambda _repo, *, prefix: ("c" * 40, list(payloads))
    )

    def download(_repo, path, *, revision):
        downloads.append(path)
        return payloads[path]

    monkeypatch.setattr(attempts, "_download_bytes", download)

    observed = attempts.read_attempt_artifacts(
        "org/repo",
        phase="rl",
        run_id="run-1",
        attempt_id=2,
        fence=9,
        source_snapshot=SOURCE,
    )

    assert downloads == [progress_path(first), progress_path(second)]
    assert observed.progress["sequence"] == 2
    assert observed.result is None


def test_result_download_transport_error_remains_retriable(monkeypatch) -> None:
    result = _result()
    path = result_path(result)
    monkeypatch.setattr(attempts, "_repo_snapshot", lambda _repo, *, prefix: ("c" * 40, [path]))
    monkeypatch.setattr(
        attempts,
        "_download_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("temporary transport failure")),
    )

    with pytest.raises(OSError, match="temporary transport failure"):
        attempts.read_attempt_artifacts(
            "org/repo",
            phase="rl",
            run_id="run-1",
            attempt_id=2,
            fence=9,
            source_snapshot=SOURCE,
        )


def test_downloaded_malformed_result_is_attempt_artifact_error(monkeypatch) -> None:
    progress = _progress(1)
    result = _result()
    path = result_path(result)
    payloads = {progress_path(progress): canonical_bytes(progress.to_dict()), path: b"not-json"}
    monkeypatch.setattr(
        attempts, "_repo_snapshot", lambda _repo, *, prefix: ("c" * 40, list(payloads))
    )
    downloads = []

    def download(_repo, artifact_path, *, revision):
        downloads.append(artifact_path)
        return payloads[artifact_path]

    monkeypatch.setattr(attempts, "_download_bytes", download)

    with pytest.raises(attempts.AttemptArtifactError, match="invalid or unverifiable"):
        attempts.read_attempt_artifacts(
            "org/repo",
            phase="rl",
            run_id="run-1",
            attempt_id=2,
            fence=9,
            source_snapshot=SOURCE,
        )
    assert downloads == [path]


def test_rejects_conflicting_results(monkeypatch) -> None:
    first = _result(outcome="failed", failure_class="worker", metrics={})
    second = _result(outcome="failed", failure_class="oom", metrics={})
    payloads = {
        result_path(first): canonical_bytes(first.to_dict()),
        result_path(second): canonical_bytes(second.to_dict()),
    }
    monkeypatch.setattr(
        attempts, "_repo_snapshot", lambda _repo, *, prefix: ("c" * 40, list(payloads))
    )
    monkeypatch.setattr(
        attempts,
        "_download_bytes",
        lambda _repo, path, *, revision: payloads[path],
    )

    with pytest.raises(attempts.AttemptArtifactError, match="conflicting"):
        attempts.read_attempt_artifacts(
            "org/repo",
            phase="rl",
            run_id="run-1",
            attempt_id=2,
            fence=9,
            source_snapshot=SOURCE,
        )


def test_stale_fence_records_are_not_visible(monkeypatch) -> None:
    stale = ProgressRecord(
        **{
            **_progress(1).to_dict(),
            "fence": 8,
        }
    )
    payloads = {progress_path(stale): json.dumps(stale.to_dict()).encode()}
    monkeypatch.setattr(
        attempts, "_repo_snapshot", lambda _repo, *, prefix: ("c" * 40, list(payloads))
    )
    monkeypatch.setattr(
        attempts,
        "_download_bytes",
        lambda _repo, path, *, revision: payloads[path],
    )

    observed = attempts.read_attempt_artifacts(
        "org/repo",
        phase="rl",
        run_id="run-1",
        attempt_id=2,
        fence=9,
        source_snapshot=SOURCE,
    )

    assert observed.progress is None
    assert observed.result is None
