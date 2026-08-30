"""Controller staging is the only runtime environment pin boundary."""

from __future__ import annotations

import io

import flash.runner.accounting.artifacts as runner_artifacts
from flash.core.spec import JobSpec
from tests._helpers.source_snapshot import valid_source_snapshot

_SHA = "a" * 40
_REVISION = "b" * 40
_ARCHIVE = "c" * 64
_MANIFEST = "d" * 64


def _staged_spec() -> JobSpec:
    return JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "grpo",
            "seed": 0,
            "run_id": "flash-staged-pin-test",
            "environment": {
                "id": "org/project/env",
                "resolved_sha": _SHA,
                "package": {
                    "artifact_revision": _REVISION,
                    "archive_sha256": _ARCHIVE,
                    "manifest_sha256": _MANIFEST,
                },
            },
            "train": {"hf_repo": "owner/runs", "max_examples": 8},
            "gpu": {"type": "RTX 4090", "max_retries": 2},
        }
    )


def test_worker_retry_path_has_no_github_pin_fallback() -> None:
    from flash.runner.supervise import lifecycle

    assert not hasattr(lifecycle, "_pin_environment_for_run")
    assert not hasattr(lifecycle, "_spec_with_resolved_env_sha")


def test_submit_context_preserves_controller_staged_identity_without_resolving(monkeypatch) -> None:
    from flash.runner.supervise import attempt_supervision

    monkeypatch.setattr(
        runner_artifacts,
        "_assign_resolved_env_sha",
        lambda _spec: (_ for _ in ()).throw(AssertionError("worker retry must not resolve github")),
    )
    spec = _staged_spec()
    context = attempt_supervision._build_context(
        spec,
        io.StringIO(),
        None,
        valid_source_snapshot(),
        None,
    )
    assert context.spec.environment.package == spec.environment.package
    assert context.spec.environment.resolved_sha == _SHA


def test_attempt_shape_rebuild_preserves_staged_package() -> None:
    from flash.runner.supervise.lifecycle import _spec_with_gpu
    from flash.runner.supervise.retry_decision import _drop_weight_cache

    spec = _staged_spec()
    shaped = _spec_with_gpu(spec, "H100", 2)
    assert shaped.environment == spec.environment
    assert _drop_weight_cache(shaped).environment == spec.environment
