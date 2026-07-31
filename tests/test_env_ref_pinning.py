"""LS-010: the environment ref must stay pinned to one commit across retry attempts.

Submit pins best-effort. When that pin fails (GitHub outage, rate limit) the spec reaches the retry
loop unpinned, and a managed slug resolves through ``environment-hub@main`` -- a moving ref. Without
a run-scoped pin, attempt 2 can resolve to a newer push and train on different environment code with
no error and nothing in the run record showing it happened.

The end-to-end test at the bottom is the regression: it drives the real retry loop and asserts both
attempts were handed the same commit. It deliberately imports no new helper, so it fails on behavior
rather than on an import error if the fix is reverted.
"""

from __future__ import annotations

import io

import pytest

from flash.spec import JobSpec

_SHA = "a" * 40
_NEWER_SHA = "b" * 40


def _env_spec(resolved_sha: str = "", **extra) -> JobSpec:
    return JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "grpo",
            "environment": {"id": "org/env", "resolved_sha": resolved_sha},
            **extra,
        }
    )


def _with_sha(spec: JobSpec, sha: str) -> JobSpec:
    """Pin a spec locally, so these tests never depend on the helper under test."""
    d = spec.to_internal_dict()
    d["environment"] = {**d["environment"], "resolved_sha": sha}
    return JobSpec.from_dict(d)


# ============================================================================================
# the two helpers, in isolation
# ============================================================================================
def test_spec_with_resolved_env_sha_applies_the_pin() -> None:
    from flash.runner.lifecycle import _spec_with_resolved_env_sha

    assert _spec_with_resolved_env_sha(_env_spec(), _SHA).environment.resolved_sha == _SHA


def test_spec_with_resolved_env_sha_is_a_noop_without_a_sha() -> None:
    from flash.runner.lifecycle import _spec_with_resolved_env_sha

    spec = _env_spec()
    assert _spec_with_resolved_env_sha(spec, "") is spec  # no copy when there is nothing to pin


def test_spec_with_resolved_env_sha_is_a_noop_when_already_pinned() -> None:
    from flash.runner.lifecycle import _spec_with_resolved_env_sha

    spec = _env_spec(_SHA)
    assert _spec_with_resolved_env_sha(spec, _SHA) is spec


def test_pin_keeps_an_existing_pin_without_re_resolving(monkeypatch) -> None:
    """An already-pinned spec must not hit GitHub again; re-resolving is what moves the ref."""
    import flash.runner as runner
    from flash.runner.lifecycle import _pin_environment_for_run

    def _boom(_spec):
        raise AssertionError("must not re-resolve a spec that is already pinned")

    monkeypatch.setattr(runner, "_assign_resolved_env_sha", _boom)
    spec = _env_spec(_SHA)
    assert _pin_environment_for_run(spec, io.StringIO(), attempt_started=False) is spec


def test_pin_resolves_and_reports_when_submit_left_it_unpinned(monkeypatch) -> None:
    import flash.runner as runner
    from flash.runner.lifecycle import _pin_environment_for_run

    monkeypatch.setattr(runner, "_assign_resolved_env_sha", lambda spec: _with_sha(spec, _SHA))
    log = io.StringIO()
    out = _pin_environment_for_run(_env_spec(), log, attempt_started=False)
    assert out.environment.resolved_sha == _SHA
    assert _SHA in log.getvalue()  # the pin is visible in the run log the user reads


def test_pin_warns_in_the_run_log_when_it_cannot_resolve(monkeypatch) -> None:
    # submit's failure only reaches a server-side logger; the user sees the run log, so warn there.
    import flash.runner as runner
    from flash.runner.lifecycle import _pin_environment_for_run

    monkeypatch.setattr(runner, "_assign_resolved_env_sha", lambda spec: spec)
    log = io.StringIO()
    out = _pin_environment_for_run(_env_spec(), log, attempt_started=False)
    assert out.environment.resolved_sha == ""
    assert "could not pin" in log.getvalue()
    assert "retry" in log.getvalue()


def test_pin_fails_closed_once_an_attempt_has_already_run_unpinned(monkeypatch) -> None:
    """Resolving after an unpinned attempt would pin a commit that attempt may not have used.

    An attempt that ran unpinned resolved the symbolic ref on its own worker, and that commit is not
    knowable here. Pinning now can only guess -- and since a retry resumes from the unpinned attempt's
    checkpoint, guessing wrong splits one run across two commits. Staying symbolic is the consistent
    answer.
    """
    import flash.runner as runner
    from flash.runner.lifecycle import _pin_environment_for_run

    def _boom(_spec):
        raise AssertionError("must not resolve the ref after an attempt has already run unpinned")

    monkeypatch.setattr(runner, "_assign_resolved_env_sha", _boom)
    log = io.StringIO()

    out = _pin_environment_for_run(_env_spec(), log, attempt_started=True)

    assert out.environment.resolved_sha == ""
    assert "ran unpinned on an earlier attempt" in log.getvalue()


def test_pin_still_applies_a_submit_time_pin_after_an_attempt_has_run(monkeypatch) -> None:
    """Failing closed must not discard a pin that already exists.

    Post-restart recovery reloads a spec that submit DID pin. That pin is the run's one commit and
    every later attempt must keep using it; the fail-closed branch is only about refusing to resolve
    a ref that is still symbolic.
    """
    import flash.runner as runner
    from flash.runner.lifecycle import _pin_environment_for_run

    monkeypatch.setattr(
        runner,
        "_assign_resolved_env_sha",
        lambda _spec: pytest.fail("must not re-resolve an already-pinned spec"),
    )
    log = io.StringIO()

    out = _pin_environment_for_run(_env_spec(_SHA), log, attempt_started=True)

    assert out.environment.resolved_sha == _SHA
    assert log.getvalue() == ""


# ============================================================================================
# end to end: the real retry loop, driven the way test_resume_on_retry drives it
# ============================================================================================
def _retry_spec() -> JobSpec:
    return JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "grpo",
            "seed": 0,
            "run_id": "flash-1700000001-pin1",
            "train": {"epochs": 1, "hf_repo": "owner/runs"},
            "gpu": {"type": "RTX 4090", "max_retries": 2},
            "environment": {"id": "org/env"},
        }
    )


@pytest.fixture
def orch(monkeypatch, tmp_path):
    """The runner package on a tmp run-store, with the inter-attempt teardown kept off-network."""
    from flash import runner
    from flash.providers import allocator
    from flash.providers.base import Allocation, Candidate
    from flash.providers.runpod import api as runpod_api

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    candidate = Candidate("runpod", "RTX 4090", 0.69, 24)
    monkeypatch.setattr(
        allocator,
        "allocate",
        lambda *a, **k: Allocation("runpod", "RTX 4090", 0.69, 12, (candidate,)),
    )
    monkeypatch.setattr(
        runpod_api,
        "cancel_job",
        lambda _endpoint_id, job_id, **_kw: {"id": job_id, "status": "CANCELLED"},
    )
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", lambda *a, **k: True)
    return runner


def test_retry_trains_on_the_same_environment_commit_as_the_first_attempt(
    orch, monkeypatch
) -> None:
    """LS-010 end to end: attempt 2 must not silently pick up a push that landed mid-run.

    Submit left the spec unpinned (its pin is best-effort). The first attempt fails infra-shaped and
    relaunches. Both attempts must be handed the same environment SHA -- otherwise the retry trains
    on different environment code, the run reports success, and nothing records that it happened.
    """
    from flash.providers.base import PollResult
    from flash.providers.runpod import jobs as rp_jobs

    resolved: list[str] = []

    def _moving_ref(spec):
        # environment-hub@main advanced between attempt 1 and attempt 2.
        sha = _SHA if not resolved else _NEWER_SHA
        resolved.append(sha)
        return _with_sha(spec, sha)

    monkeypatch.setattr(orch, "_assign_resolved_env_sha", _moving_ref)

    seen_shas: list[str] = []

    def fake_submit(run_spec, seed, log=None, on_handle=None, attempt=0, **_):
        seen_shas.append(run_spec.environment.resolved_sha)
        on_handle(
            {
                "provider": "runpod",
                "endpoint_id": f"ep{attempt}",
                "endpoint_name": f"ep{attempt}-name",
                "key_fingerprint": "rpk-0123456789ab",
                "job_id": f"j{attempt}",
                "attempt": attempt,
                "started_ts": float(attempt + 1),
            }
        )
        if attempt == 0:
            return PollResult(False, failure="stalled", detail="infra")
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_submit)

    spec = _retry_spec()
    orch._save_status(orch.RunStatus(run_id=spec.run_id, state="queued", spec=spec.to_dict()))
    orch._submit_seed_supervised(spec, 0, io.StringIO())

    assert len(seen_shas) == 2, "the infra failure should have produced a second attempt"
    assert seen_shas == [_SHA, _SHA], (
        f"retry trained on a different environment commit: {seen_shas}"
    )
    assert resolved == [_SHA], f"the ref must be resolved once per run, not per attempt: {resolved}"


def test_a_recovered_resolve_cannot_pin_a_newer_commit_mid_run(orch, monkeypatch) -> None:
    """The pin must not start working halfway through a run.

    Both submit and the run-start resolve fail transiently, so attempt 1 runs unpinned and resolves
    environment-hub@main on its own worker. If GitHub then recovers, resolving before attempt 2 pins
    whatever main points at NOW -- and a push in between means the retry, which resumes from attempt
    1's checkpoint, trains on different environment code. Exactly the corruption the pin exists to
    prevent, just moved one attempt later.
    """
    from flash.providers.base import PollResult
    from flash.providers.runpod import jobs as rp_jobs

    calls: list[str] = []

    def _outage_then_recovery(spec):
        # attempt 1 gets nothing; by attempt 2 github is back and main has advanced.
        calls.append("resolve")
        return spec if len(calls) == 1 else _with_sha(spec, _NEWER_SHA)

    monkeypatch.setattr(orch, "_assign_resolved_env_sha", _outage_then_recovery)

    seen_shas: list[str] = []

    def fake_submit(run_spec, seed, log=None, on_handle=None, attempt=0, **_):
        seen_shas.append(run_spec.environment.resolved_sha)
        on_handle(
            {
                "provider": "runpod",
                "endpoint_id": f"ep{attempt}",
                "endpoint_name": f"ep{attempt}-name",
                "key_fingerprint": "rpk-0123456789ab",
                "job_id": f"j{attempt}",
                "attempt": attempt,
                "started_ts": float(attempt + 1),
            }
        )
        if attempt == 0:
            return PollResult(False, failure="stalled", detail="infra")
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_submit)

    spec = _retry_spec()
    orch._save_status(orch.RunStatus(run_id=spec.run_id, state="queued", spec=spec.to_dict()))
    log = io.StringIO()
    orch._submit_seed_supervised(spec, 0, log)

    assert len(seen_shas) == 2, "the infra failure should have produced a second attempt"
    assert seen_shas == ["", ""], (
        f"the retry pinned a commit the first attempt never used: {seen_shas}"
    )
    assert calls == ["resolve"], f"the ref must not be re-resolved per attempt: {calls}"
    assert "could not pin" in log.getvalue()
