"""Coverage for control-plane env publishing helpers (`flash.server.envs`) and the serving
route helpers (`flash.server.routes.serving`).

These target error / edge branches the existing suite leaves uncovered: git-subprocess
failure translation, archive-extraction guards, publish/slug input validation, and the pure
deployment-lifecycle helpers behind the deploy/chat routes. Everything stays hermetic — no
real git/network/GPU — matching the direct-call + monkeypatch style of tests/test_env_publish.py
and the offline conftest.
"""

from __future__ import annotations

import io
import subprocess
import tarfile
import types

import pytest

from flash.server import envs

pytest.importorskip("fastapi")
from fastapi import HTTPException

import flash.server.routes.serving as serving
from flash.serve.deploy import ServingError


def _targz(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for info, data in members:
            if data is None:
                tar.addfile(info)
            else:
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


# ===========================================================================
# flash.server.envs
# ===========================================================================


def test_pure_url_and_redact_helpers():
    # _credentialed_repo_url percent-encodes the token into the https remote.
    url = envs._credentialed_repo_url("owner/repo", "tok/with:chars")
    assert url == "https://x-access-token:tok%2Fwith%3Achars@github.com/owner/repo.git"

    # _redact with an empty token is a no-op (the early-return branch).
    assert envs._redact("nothing to redact", "") == "nothing to redact"


def test_is_retryable_git_publish_error_classifies_markers():
    # Permanent auth/not-found failures are never retried.
    for permanent in ("authentication failed", "Repository not found", "HTTP 403 Forbidden"):
        assert envs._is_retryable_git_publish_error(permanent) is False
    # Transient push races are retryable.
    for transient in ("failed to push some refs", "non-fast-forward update", "cannot lock ref"):
        assert envs._is_retryable_git_publish_error(transient) is True
    # An unclassified message defaults to non-retryable.
    assert envs._is_retryable_git_publish_error("some totally novel git error") is False


def test_publish_slug_and_input_validation(monkeypatch):
    # A namespaced name must be exactly two non-empty segments.
    with pytest.raises(envs.EnvPublishError, match="namespace/name"):
        envs._publish_slug_for_name("a/b/c", {"org_slug": "acme"})
    with pytest.raises(envs.EnvPublishError, match="namespace/name"):
        envs._publish_slug_for_name("acme/", {"org_slug": "acme"})

    # publish_package validates argument TYPES before touching storage.
    monkeypatch.setattr(envs, "_github_publish_once", lambda **_k: None)
    with pytest.raises(envs.EnvPublishError, match="env name must be a string"):
        envs.publish_package(package_b64="x", name=123, key={})  # type: ignore[arg-type]
    with pytest.raises(envs.EnvPublishError, match="base64 string"):
        envs.publish_package(package_b64=123, name="e", key={})  # type: ignore[arg-type]

    # canonical_env_id / _validate_slug reject a non-string id.
    with pytest.raises(envs.EnvPublishError, match="env id must be a string"):
        envs.canonical_env_id(123)  # type: ignore[arg-type]


def test_safe_extract_error_branches(tmp_path, monkeypatch):
    # Corrupt (non-gzip) bytes are surfaced as a clean EnvPublishError, not a raw OSError/TarError.
    with pytest.raises(envs.EnvPublishError, match="env package"):
        envs._safe_extract(b"this is not a gzip archive at all", tmp_path)

    # A symlink member is rejected outright.
    link = tarfile.TarInfo("pkg/evil-link")
    link.type = tarfile.SYMTYPE
    link.linkname = "environment.py"
    with pytest.raises(envs.EnvPublishError, match="links are not allowed"):
        envs._safe_extract(_targz([(link, None)]), tmp_path)

    # A "." path entry contributes no segments and is skipped, while the real file still extracts.
    dot = tarfile.TarInfo(".")
    dot.type = tarfile.DIRTYPE
    good = _targz([(dot, None), (tarfile.TarInfo("pkg/environment.py"), b"ok")])
    envs._safe_extract(good, tmp_path)
    assert (tmp_path / "pkg" / "environment.py").read_bytes() == b"ok"

    # Exceeding the per-archive scan cap aborts before extracting everything.
    monkeypatch.setattr(envs, "_MAX_SCAN_MEMBERS", 1)
    two = _targz(
        [
            (tarfile.TarInfo("a.txt"), b"a"),
            (tarfile.TarInfo("b.txt"), b"b"),
        ]
    )
    with pytest.raises(envs.EnvPublishError, match="too many entries to scan"):
        envs._safe_extract(two, tmp_path / "scan")


def test_safe_extract_enforces_member_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(envs, "_MAX_MEMBERS", 1)
    payload = _targz(
        [
            (tarfile.TarInfo("one.txt"), b"1"),
            (tarfile.TarInfo("two.txt"), b"2"),
        ]
    )
    with pytest.raises(envs.EnvPublishError, match="too many members"):
        envs._safe_extract(payload, tmp_path)


def test_package_checkout_directory_error_branches(tmp_path, monkeypatch):
    # No environment.py in the checkout -> 404 "environment package not found".
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(envs.EnvPublishError) as missing:
        envs._package_checkout_directory(empty)
    assert missing.value.status == 404

    # A repo-control top-level dir inside the package is a 502.
    repo_control = tmp_path / "repoctl"
    repo_control.mkdir()
    (repo_control / "environment.py").write_text("def load_environment(**k): pass\n")
    (repo_control / ".git").mkdir()
    (repo_control / ".git" / "config").write_text("[core]\n")
    with pytest.raises(envs.EnvPublishError, match="top-level paths") as blocked:
        envs._package_checkout_directory(repo_control)
    assert blocked.value.status == 502

    # A symlink inside the package is a 502.
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "environment.py").write_text("def load_environment(**k): pass\n")
    (linked / "zzz-link.py").symlink_to(linked / "environment.py")
    with pytest.raises(envs.EnvPublishError, match="links are not allowed") as sym:
        envs._package_checkout_directory(linked)
    assert sym.value.status == 502

    # Too many members trips the member cap.
    many = tmp_path / "many"
    many.mkdir()
    (many / "environment.py").write_text("def load_environment(**k): pass\n")
    (many / "extra.py").write_text("x = 1\n")
    monkeypatch.setattr(envs, "_MAX_MEMBERS", 1)
    with pytest.raises(envs.EnvPublishError, match="too many members"):
        envs._package_checkout_directory(many)


def test_run_git_translates_missing_binary_and_nonzero_exit(tmp_path, monkeypatch):
    # git binary absent -> 503 phrased for the download verb.
    def _missing(*_a, **_k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(envs.subprocess, "run", _missing)
    with pytest.raises(envs.EnvPublishError) as no_git:
        envs._run_git(tmp_path, ["clone"], token="tok", operation="download")
    assert no_git.value.status == 503
    assert "download" in str(no_git.value)
    assert "from Freesolo" in str(no_git.value)

    # A nonzero exit -> 502, with the credential scrubbed out of the surfaced output.
    def _fail(*_a, **_k):
        return subprocess.CompletedProcess(["git"], 128, "", "fatal: ghp-secret rejected")

    monkeypatch.setattr(envs.subprocess, "run", _fail)
    with pytest.raises(envs.EnvPublishError) as failed:
        envs._run_git(tmp_path, ["push"], token="ghp-secret", operation="upload")
    assert failed.value.status == 502
    assert "ghp-secret" not in str(failed.value)
    assert "<redacted>" in str(failed.value)


def test_staged_has_changes_timeout_is_504(tmp_path, monkeypatch):
    def _timeout(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd=["git"], timeout=180)

    monkeypatch.setattr(envs.subprocess, "run", _timeout)
    with pytest.raises(envs.EnvPublishError) as exc:
        envs._staged_has_changes(tmp_path)
    assert exc.value.status == 504
    assert "timed out" in str(exc.value)


def test_github_publish_does_not_retry_permanent_error(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    (tmp_path / "environment.py").write_text("def load_environment(**k): pass\n")
    calls = {"n": 0}

    def fake_once(**_kwargs):
        calls["n"] += 1
        raise envs.EnvPublishError("authentication failed", status=502)

    monkeypatch.setattr(envs, "_github_publish_once", fake_once)
    # Guard: if the code ever retried, this sleep would be exercised — no-op it either way.
    monkeypatch.setattr(
        envs.time, "sleep", lambda _s: pytest.fail("permanent error must not retry")
    )

    with pytest.raises(envs.EnvPublishError, match="authentication failed"):
        envs._github_publish(tmp_path, name="e", key={"org_slug": "acme"})
    assert calls["n"] == 1


def test_github_download_wrapper_uses_default_repo(monkeypatch):
    seen: dict[str, object] = {}

    def fake_once(*, repo, token, publish_root):
        seen.update(repo=repo, token=token, publish_root=publish_root)
        return b"package-bytes"

    monkeypatch.setattr(envs, "_github_download_once", fake_once)
    assert envs._github_download("ns/env", token="tok") == b"package-bytes"
    assert seen == {
        "repo": envs._DEFAULT_GITHUB_REPO,
        "token": "tok",
        "publish_root": "ns/env",
    }


# ===========================================================================
# flash.server.routes.serving
# ===========================================================================


def test_deployment_state_and_public_deployment():
    original = {"a": 1, "state": "deploying"}
    out = serving._deployment_state(original, "ready", detail="done")
    assert out["state"] == "ready"
    assert out["detail"] == "done"
    assert out["a"] == 1
    assert isinstance(out["updated_at"], float)
    # the input is not mutated in place.
    assert original == {"a": 1, "state": "deploying"}

    pub = serving._public_deployment(
        {
            "state": "failed",
            "detail": "operator action required",
            "error": "mutation mutation-5 used repo " + "a" * 40,
            "endpoint_name": "https://serve.example",
            "openai_base_url": "https://serve.example/v1",
            "url": "https://stale.example/v1",
            "desired_record": {"adapter_id": "r1"},
            "prior_revision": 4,
            "target_revision": 5,
            "mutation_id": "mutation-5",
            "repo_revision": "a" * 40,
            "requested_at": 123.0,
            "verify": True,
            "unknown_future_internal": "secret",
        }
    )
    assert pub == {
        "state": "failed",
        "detail": "operator action required",
        "error": "mutation [redacted] used repo [redacted]",
        "endpoint_name": "https://serve.example",
        "openai_base_url": "https://serve.example/v1",
    }


def test_deployment_attempt_is_stale_branches():
    assert serving._deployment_attempt_is_stale({"state": "ready"}) is False
    assert serving._deployment_attempt_is_stale({"state": "deploying"}) is True
    fresh = {"state": "deploying", "updated_at": 1000.0}
    assert serving._deployment_attempt_is_stale(fresh, now=1010.0) is False
    old = {"state": "deploying", "requested_at": 1000.0}
    assert (
        serving._deployment_attempt_is_stale(old, now=1000.0 + serving._DEPLOYMENT_STALE_SECONDS)
        is True
    )


def test_chat_messages_from_payload_validation():
    assert serving._chat_messages_from_payload({}) == []
    assert serving._chat_messages_from_payload({"messages": None}) == []
    valid = [{"role": "user", "content": "hi"}]
    assert serving._chat_messages_from_payload({"messages": valid}) is valid
    with pytest.raises(HTTPException) as not_list:
        serving._chat_messages_from_payload({"messages": "nope"})
    assert not_list.value.status_code == 400
    with pytest.raises(HTTPException) as bad_item:
        serving._chat_messages_from_payload({"messages": [{"role": "user"}, "oops"]})
    assert bad_item.value.status_code == 400


def test_validate_hf_repo_id_accepts_valid_and_rejects_malformed():
    serving._validate_hf_repo_id("owner/name")
    with pytest.raises(HTTPException) as exc:
        serving._validate_hf_repo_id("bad//id")
    assert exc.value.status_code == 400


def test_resolve_deploy_step_branches(monkeypatch):
    monkeypatch.setattr(serving._app, "list_checkpoints", lambda spec: [{"step": 20}, {"step": 40}])
    assert serving._resolve_deploy_step("run-1", object(), None) is None
    assert serving._resolve_deploy_step("run-1", object(), 20) == 20
    assert serving._resolve_deploy_step("run-1", object(), 40.0) == 40
    assert serving._resolve_deploy_step("run-1", object(), "40") == 40
    with pytest.raises(HTTPException) as not_found:
        serving._resolve_deploy_step("run-1", object(), 999)
    assert not_found.value.status_code == 404
    for bad in (True, 20.5, -5, "-5", "abc", "1.5"):
        with pytest.raises(HTTPException) as exc:
            serving._resolve_deploy_step("run-1", object(), bad)
        assert exc.value.status_code == 400, bad


def _recovering_status(deployment, deployment_attempt=None, deployment_cleanup=None):
    return types.SimpleNamespace(
        run_id="run-1",
        state="done",
        spec={"model": "Qwen/Qwen3.5-0.8B"},
        deployment=deployment,
        deployment_attempt=deployment_attempt,
        deployment_cleanup=deployment_cleanup,
    )


def test_recovery_retries_persisted_cleanup_after_disable_failure(monkeypatch):
    cleanup = {
        "target": {"revision": 7, "mutation_id": "old"},
        "prior": None,
        "requested_at": 1.0,
    }
    status = _recovering_status(
        {"state": "undeployed", "mutation_id": "old"}, deployment_cleanup=cleanup
    )
    monkeypatch.setattr(serving._app, "get_status", lambda _run_id: status)
    monkeypatch.setattr(
        serving._app,
        "reconcile_owned_adapter_cleanup",
        lambda *_args: (_ for _ in ()).throw(ServingError("delete failed")),
    )
    monkeypatch.setattr(
        serving,
        "complete_deployment_cleanup",
        lambda *_args: pytest.fail("failed cleanup must remain retryable"),
    )

    assert serving._recover_deployment("run-1") is False


def test_recovery_clears_cleanup_after_forward_supersession(monkeypatch):
    cleanup = {
        "target": {"revision": 7, "mutation_id": "old"},
        "prior": None,
        "requested_at": 1.0,
    }
    newer = {"state": "deploying", "target_revision": 8, "mutation_id": "new"}
    status = _recovering_status(newer, deployment_cleanup=cleanup)
    completed = []
    monkeypatch.setattr(serving._app, "get_status", lambda _run_id: status)
    monkeypatch.setattr(
        serving._app,
        "reconcile_owned_adapter_cleanup",
        lambda run_id, owned: (run_id, owned) == ("run-1", cleanup),
    )
    monkeypatch.setattr(
        serving,
        "complete_deployment_cleanup",
        lambda run_id, owned: completed.append((run_id, owned)),
    )

    assert serving._recover_deployment("run-1") is True
    assert completed == [("run-1", cleanup)]
    assert status.deployment == newer


def test_recovery_preserves_active_slow_pre_intent_worker(monkeypatch):
    queued = {"state": "deploying", "mutation_id": "mine", "requested_at": 1.0}
    attempt = {"phase": "initial", "deployment": queued, "active_deployment": None}
    status = _recovering_status(queued, attempt)
    monkeypatch.setattr(serving._app, "get_status", lambda _run_id: status)
    monkeypatch.setattr(
        serving,
        "mark_deployment_pre_intent_failed",
        lambda *_args: pytest.fail("active deployment worker must retain ownership"),
    )

    serving._mark_deployment_worker_active("run-1", "mine")
    try:
        assert serving._recover_deployment("run-1") is False
    finally:
        serving._clear_deployment_worker_active("run-1", "mine")


def test_recovery_fails_orphaned_pre_intent_attempt(monkeypatch):
    queued = {"state": "deploying", "mutation_id": "orphan", "requested_at": 1.0}
    attempt = {"phase": "initial", "deployment": queued, "active_deployment": None}
    status = _recovering_status(queued, attempt)
    marked = []
    monkeypatch.setattr(serving._app, "get_status", lambda _run_id: status)
    monkeypatch.setattr(
        serving,
        "mark_deployment_pre_intent_failed",
        lambda run_id, owned_attempt, failed: marked.append((run_id, owned_attempt, failed)),
    )

    assert serving._recover_deployment("run-1") is True
    assert marked[0][0] == "run-1"
    assert marked[0][1] is attempt
    assert marked[0][2]["state"] == "failed"
    assert "before registry intent" in marked[0][2]["error"]


@pytest.mark.parametrize(
    "worker_error", [None, RuntimeError("worker stopped")], ids=["success", "error"]
)
def test_deployment_worker_liveness_clears_on_all_exits(monkeypatch, worker_error):
    attempt = {
        "phase": "initial",
        "deployment": {"state": "deploying", "mutation_id": "mine"},
        "active_deployment": None,
    }

    def finish(**_kwargs):
        assert serving._deployment_worker_is_active("run-1", "mine") is True
        if worker_error is not None:
            raise worker_error

    monkeypatch.setattr(serving, "_finish_deployment_unlocked", finish)
    serving._mark_deployment_worker_active("run-1", "mine")

    if worker_error is None:
        serving._finish_deployment(run_id="run-1", deployment_attempt=attempt)
    else:
        with pytest.raises(RuntimeError, match="worker stopped"):
            serving._finish_deployment(run_id="run-1", deployment_attempt=attempt)
    assert serving._deployment_worker_is_active("run-1", "mine") is False


def test_recovery_preserves_active_post_intent_worker(monkeypatch):
    deployment = {
        "state": "deploying",
        "requested_at": 1.0,
        "desired_record": {"adapter_id": "run-1", "mutation_id": "mine"},
        "prior_revision": None,
        "target_revision": 1,
        "mutation_id": "mine",
    }
    monkeypatch.setattr(
        serving._app, "get_status", lambda _run_id: _recovering_status(deployment)
    )
    monkeypatch.setattr(
        serving._app,
        "read_adapter_record",
        lambda _run_id: pytest.fail("active deployment worker owns post-intent recovery"),
    )

    serving._mark_deployment_worker_active("run-1", "mine")
    try:
        assert serving._recover_deployment("run-1") is False
    finally:
        serving._clear_deployment_worker_active("run-1", "mine")


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (None, "did not commit"),
        ({"registry_revision": 4, "mutation_id": "old", "status": "ready"}, "did not commit"),
        ({"registry_revision": 6, "mutation_id": "mine", "status": "disabled"}, "disabled"),
        ({"registry_revision": 5, "mutation_id": "other", "status": "ready"}, "superseded"),
    ],
)
def test_worker_death_recovery_classifies_non_resumable_records(monkeypatch, record, expected):
    deployment = {
        "state": "deploying",
        "requested_at": 1.0,
        "desired_record": {"adapter_id": "run-1", "mutation_id": "mine", "status": "ready"},
        "prior_revision": 4,
        "target_revision": 5,
        "mutation_id": "mine",
    }
    monkeypatch.setattr(serving._app, "get_status", lambda _run_id: _recovering_status(deployment))
    monkeypatch.setattr(serving._app, "read_adapter_record", lambda _run_id: record)
    monkeypatch.setattr(serving._app, "record_matches", lambda *_args: False)
    marked = []
    monkeypatch.setattr(
        serving, "mark_deployment_failed", lambda _run_id, failed: marked.append(failed)
    )
    assert serving._recover_deployment("run-1") is True
    assert expected in marked[0]["error"]


def test_post_intent_orphan_resumes_without_live_worker(monkeypatch):
    desired = {"adapter_id": "run-1", "mutation_id": "mine", "status": "ready"}
    deployment = {
        "state": "deploying",
        "requested_at": 1.0,
        "desired_record": desired,
        "prior_revision": None,
        "target_revision": 1,
        "mutation_id": "mine",
    }
    status = _recovering_status(deployment)
    resumed = []
    monkeypatch.setattr(serving._app, "get_status", lambda _run_id: status)
    monkeypatch.setattr(
        serving._app, "read_adapter_record", lambda _run_id: {**desired, "registry_revision": 1}
    )
    monkeypatch.setattr(serving._app, "record_matches", lambda *_args: True)
    monkeypatch.setattr(
        serving, "_resume_registered_deployment", lambda *args: resumed.append(args)
    )
    assert serving._deployment_worker_is_active("run-1", "mine") is False
    assert serving._recover_deployment("run-1") is True
    assert resumed


def test_worker_death_recovery_retries_transient_readback(monkeypatch):
    desired = {"adapter_id": "run-1", "mutation_id": "mine", "status": "ready"}
    deployment = {
        "state": "deploying",
        "requested_at": 1.0,
        "desired_record": desired,
        "prior_revision": None,
        "target_revision": 1,
        "mutation_id": "mine",
    }
    calls = {"count": 0}

    def read(_run_id):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("transient outage")
        return {**desired, "registry_revision": 1}

    monkeypatch.setattr(serving, "_RECOVERY_RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(serving._app, "get_status", lambda _run_id: _recovering_status(deployment))
    monkeypatch.setattr(serving._app, "read_adapter_record", read)
    monkeypatch.setattr(serving._app, "record_matches", lambda *_args: True)
    resumed = []
    monkeypatch.setattr(
        serving, "_resume_registered_deployment", lambda *args: resumed.append(args)
    )

    assert serving._recover_deployment("run-1") is True
    assert calls["count"] == 2
    assert resumed


def test_worker_death_recovery_leaves_deploying_on_readback_outage(monkeypatch):
    monkeypatch.setattr(serving, "_RECOVERY_RETRY_DELAY_SECONDS", 0)
    deployment = {
        "state": "deploying",
        "requested_at": 1.0,
        "desired_record": {"mutation_id": "mine"},
        "prior_revision": None,
        "target_revision": 1,
        "mutation_id": "mine",
    }
    monkeypatch.setattr(serving._app, "get_status", lambda _run_id: _recovering_status(deployment))
    monkeypatch.setattr(
        serving._app,
        "read_adapter_record",
        lambda _run_id: (_ for _ in ()).throw(RuntimeError("outage")),
    )
    monkeypatch.setattr(serving, "mark_deployment_failed", lambda *_args: pytest.fail("must defer"))
    assert serving._recover_deployment("run-1") is False


@pytest.mark.parametrize("requested_at", [None, "not-a-number", float("inf")])
def test_worker_death_recovery_fails_malformed_attempt_metadata(monkeypatch, requested_at):
    deployment = {
        "state": "deploying",
        "desired_record": {"mutation_id": "mine"},
        "target_revision": 1,
        "mutation_id": "mine",
    }
    if requested_at is not None:
        deployment["requested_at"] = requested_at
    monkeypatch.setattr(serving._app, "get_status", lambda _run_id: _recovering_status(deployment))
    monkeypatch.setattr(
        serving._app, "read_adapter_record", lambda _run_id: pytest.fail("must not read serving")
    )
    marked = []
    monkeypatch.setattr(
        serving, "mark_deployment_failed", lambda _run_id, failed: marked.append(failed)
    )

    assert serving._recover_deployment("run-1") is True
    assert marked[0]["state"] == "failed"
    assert marked[0]["error"] == "deployment attempt metadata is malformed"


def test_failed_attempt_stays_recoverable_when_disable_readback_is_unavailable(monkeypatch):
    deployment = {
        "state": "deploying",
        "requested_at": 1.0,
        "desired_record": {"mutation_id": "mine"},
        "target_revision": 1,
        "mutation_id": "mine",
    }
    status = _recovering_status(deployment)
    monkeypatch.setattr(serving.JobSpec, "from_dict", lambda _spec: object())
    monkeypatch.setattr(serving._app, "get_status", lambda _run_id: status)
    monkeypatch.setattr(
        serving,
        "_finalize_registered_deployment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ServingError("smoke failed")),
    )
    monkeypatch.setattr(
        serving._app,
        "disable_owned_adapter",
        lambda *_args: (_ for _ in ()).throw(ServingError("readback unavailable")),
    )
    monkeypatch.setattr(serving, "mark_deployment_failed", lambda *_args: pytest.fail("must defer"))

    serving._resume_registered_deployment("run-1", {}, deployment)


def test_registry_intent_write_rejects_a_replaced_local_attempt(monkeypatch):
    old = {"state": "deploying", "mutation_id": "old", "requested_at": 1.0}
    newer = {"state": "deploying", "mutation_id": "new", "requested_at": 2.0}
    old_attempt = {"phase": "initial", "deployment": old, "active_deployment": None}
    newer_attempt = {"phase": "initial", "deployment": newer, "active_deployment": None}
    status = _recovering_status(newer)
    status.deployment_attempt = newer_attempt
    seen = {}

    def deploy_adapter(**kwargs):
        kwargs["before_registry_mutation"](None, {"mutation_id": "old"}, 1, "old", "a" * 40)
        pytest.fail("superseded local attempt must stop before POST")

    def mark_intent(_run_id, _intent, **kwargs):
        seen.update(kwargs)
        return status

    monkeypatch.setattr(serving.JobSpec, "from_dict", lambda _spec: object())
    monkeypatch.setattr(serving._app, "deploy_adapter", deploy_adapter)
    monkeypatch.setattr(serving._app, "get_status", lambda _run_id: status)
    monkeypatch.setattr(serving, "mark_deployment_intent", mark_intent)
    monkeypatch.setattr(
        serving, "mark_deployment_failed", lambda *_args: pytest.fail("must not fail newer")
    )

    serving._finish_deployment_unlocked(
        run_id="run-1",
        spec_dict={},
        checkpoint_step=None,
        is_checkpoint=False,
        deploy_kwargs={"mutation_id": "old"},
        deployment_attempt=old_attempt,
        prev_state="done",
        verify=True,
    )

    assert seen == {"expect_attempt": old_attempt}


def test_lifespan_runs_recovery_after_readiness_and_awaits_shutdown(monkeypatch):
    import asyncio
    import threading

    import flash.providers.preflight as preflight
    import flash.providers.runpod.train.endpoints as endpoints
    import flash.server.app as app_mod
    import flash.server.billing_retry as billing_retry
    import flash.server.reconcile as reconcile
    import flash.server.repo_cleanup as repo_cleanup

    started = threading.Event()
    finished = threading.Event()
    received_stop_event = None

    def recover_deployments(*, stop_event):
        nonlocal received_stop_event
        received_stop_event = stop_event
        started.set()
        stop_event.wait(1)
        finished.set()
        return 0

    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setattr(preflight, "check_run_preflight", lambda: None)
    monkeypatch.setattr(app_mod, "recover_runs", lambda: None)
    monkeypatch.setattr(serving, "recover_deployments", recover_deployments)
    monkeypatch.setattr(billing_retry, "charge_retry_enabled", lambda: False)
    monkeypatch.setattr(reconcile, "reconcile_enabled", lambda: False)
    monkeypatch.setattr(endpoints, "reconcile_endpoint_slots", lambda: None)
    monkeypatch.setattr(app_mod, "_instance_providers_configured", lambda: False)
    monkeypatch.setattr(repo_cleanup, "repo_cleanup_enabled", lambda: False)
    application = app_mod.create_app()

    async def exercise_lifespan():
        async with application.router.lifespan_context(application):
            assert await asyncio.to_thread(started.wait, 1)
            assert received_stop_event is not None
            assert not received_stop_event.is_set()
            assert not finished.is_set()
        assert received_stop_event.is_set()
        assert finished.is_set()

    asyncio.run(exercise_lifespan())


def test_recovery_workers_are_bounded_and_honor_shutdown_signal(monkeypatch):
    import threading
    import time

    monkeypatch.setattr(
        serving.db, "all_runs", lambda: [{"run_id": f"r-{index}"} for index in range(6)]
    )
    lock = threading.Lock()
    active = 0
    peak = 0

    def recover(_run_id):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return True

    monkeypatch.setattr(serving, "_recover_deployment", recover)
    assert serving.recover_deployments(max_workers=2) == 6
    assert peak == 2

    stopped = threading.Event()
    stopped.set()
    assert serving.recover_deployments(max_workers=2, stop_event=stopped) == 0


def test_transient_final_registry_read_preserves_deploying_for_recovery(monkeypatch):
    desired = {"adapter_id": "run-1", "mutation_id": "mine", "status": "ready"}
    deployment = {
        "state": "deploying",
        "requested_at": 1.0,
        "desired_record": desired,
        "prior_revision": None,
        "target_revision": 1,
        "mutation_id": "mine",
        "verify": True,
    }
    status = _recovering_status(deployment)
    reads = []
    monkeypatch.setattr(serving, "_RECOVERY_RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(serving.JobSpec, "from_dict", lambda _spec: object())
    monkeypatch.setattr(serving._app, "get_status", lambda _run_id: status)
    monkeypatch.setattr(serving, "_smoke_with_retries", lambda *_args: {"verified_at": 2.0})

    def transient_read(run_id):
        reads.append(run_id)
        raise ServingError("registry unavailable")

    monkeypatch.setattr(serving._app, "read_adapter_record", transient_read)
    monkeypatch.setattr(
        serving._app,
        "disable_owned_adapter",
        lambda *_args: pytest.fail("inconclusive readback must not disable"),
    )
    monkeypatch.setattr(
        serving,
        "mark_deployment_failed",
        lambda *_args: pytest.fail("inconclusive readback must remain deploying"),
    )

    serving._resume_registered_deployment("run-1", {}, deployment)

    assert reads == ["run-1"] * serving._RECOVERY_READ_ATTEMPTS
    assert status.deployment["state"] == "deploying"


def test_authoritative_final_registry_mismatch_uses_exact_owned_cleanup(monkeypatch):
    desired = {"adapter_id": "run-1", "mutation_id": "mine", "status": "ready"}
    deployment = {
        "state": "deploying",
        "requested_at": 1.0,
        "desired_record": desired,
        "prior_revision": None,
        "target_revision": 5,
        "mutation_id": "mine",
        "verify": True,
    }
    status = _recovering_status(deployment)
    disabled = []
    failed = []
    monkeypatch.setattr(serving.JobSpec, "from_dict", lambda _spec: object())
    monkeypatch.setattr(serving._app, "get_status", lambda _run_id: status)
    monkeypatch.setattr(serving, "_smoke_with_retries", lambda *_args: {"verified_at": 2.0})
    monkeypatch.setattr(
        serving._app,
        "read_adapter_record",
        lambda _run_id: {
            "adapter_id": "run-1",
            "mutation_id": "newer",
            "registry_revision": 6,
            "status": "ready",
        },
    )
    monkeypatch.setattr(serving._app, "record_matches", lambda *_args: False)

    def reject_superseded(run_id, revision, mutation_id):
        disabled.append((run_id, revision, mutation_id))
        raise serving._app.DeploymentSuperseded("newer mutation owns the registry row")

    monkeypatch.setattr(serving._app, "disable_owned_adapter", reject_superseded)
    monkeypatch.setattr(
        serving, "mark_deployment_failed", lambda _run_id, record: failed.append(record)
    )

    serving._resume_registered_deployment("run-1", {}, deployment)

    assert disabled == [("run-1", 5, "mine")]
    assert failed[0]["state"] == "failed"
    assert "registry changed" in failed[0]["error"]


def test_run_deployment_smoke_binds_expected_checkpoint(monkeypatch):
    spec = types.SimpleNamespace(thinking=False, train=types.SimpleNamespace(structured_outputs=""))
    seen = {}

    def fake_serve_chat(**kwargs):
        seen.update(kwargs)
        return {"choices": [{"message": {"content": "4"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(serving._app, "serve_chat", fake_serve_chat)
    serving._run_deployment_smoke(
        "run-1",
        spec,
        expected_checkpoint="run-1/step-40",
        expected_registry_revision=7,
        expected_mutation_id="mutation-7",
    )
    assert seen["expected_checkpoint"] == "run-1/step-40"


def test_run_deployment_smoke_success_and_empty(monkeypatch):
    spec = types.SimpleNamespace(thinking=False, train=types.SimpleNamespace(structured_outputs=""))

    def fake_serve_chat(**kwargs):
        assert kwargs["run_id"] == "run-1"
        assert kwargs["temperature"] == 0.0
        return {"choices": [{"message": {"content": "The answer is 4"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(serving._app, "serve_chat", fake_serve_chat)
    out = serving._run_deployment_smoke(
        "run-1",
        spec,
        expected_checkpoint="run-1",
        expected_registry_revision=1,
        expected_mutation_id="mutation-1",
    )
    assert out["verify_finish_reason"] == "stop"
    assert out["verify_sample"] == "The answer is 4"
    assert out["thinking_tag"] is False
    assert out["verify_latency_s"] >= 0.0

    # Thinking tags in the sample are detected.
    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **_k: {"choices": [{"message": {"content": "<think>hmm</think> 4"}}]},
    )
    assert (
        serving._run_deployment_smoke(
            "run-1",
            spec,
            expected_checkpoint="run-1",
            expected_registry_revision=1,
            expected_mutation_id="mutation-1",
        )["thinking_tag"]
        is True
    )

    # Empty generation is a ServingError, not a silent "ready".
    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **_k: {"choices": [{"message": {"content": "   "}, "finish_reason": "stop"}]},
    )
    with pytest.raises(ServingError, match="no answer content"):
        serving._run_deployment_smoke(
            "run-1",
            spec,
            expected_checkpoint="run-1",
            expected_registry_revision=1,
            expected_mutation_id="mutation-1",
        )

    # A generation truncated at the token limit is rejected before the empty-answer check, because
    # a length-capped sample cannot prove the adapter emits a complete, well-formed answer.
    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **_k: {"choices": [{"message": {"content": "4"}, "finish_reason": "length"}]},
    )
    with pytest.raises(ServingError, match="truncated at the maximum token length"):
        serving._run_deployment_smoke(
            "run-1",
            spec,
            expected_checkpoint="run-1",
            expected_registry_revision=1,
            expected_mutation_id="mutation-1",
        )


@pytest.mark.parametrize(
    ("constraint", "content", "sample"),
    [
        ({"json_object": True}, '<think>{"bad":</think>{"answer": 4}', '{"answer": 4}'),
        (
            {"json": {"type": "object", "required": ["answer"]}},
            '<think>x</think>{"answer": 4}',
            '{"answer": 4}',
        ),
        ({"choice": ["4", "four"]}, "<think>2+2</think>4", "4"),
        ({"regex": "[0-9]+"}, "<think>2+2</think>4", "4"),
        ({"json_object": True}, '<think>x</think>{"literal": "<think>"}', '{"literal": "<think>"}'),
        ({"choice": ["<think>"]}, "<think>x</think><think>", "<think>"),
        ({"regex": "<think>"}, "<think>x</think><think>", "<think>"),
    ],
)
def test_run_deployment_smoke_validates_only_answer_after_final_think(
    monkeypatch, constraint, content, sample
):
    import json

    spec = types.SimpleNamespace(
        thinking=True,
        train=types.SimpleNamespace(structured_outputs=json.dumps(constraint)),
    )
    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **_k: {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]},
    )
    out = serving._run_deployment_smoke(
        "run-1",
        spec,
        expected_checkpoint="run-1",
        expected_registry_revision=1,
        expected_mutation_id="mutation-1",
    )
    assert out["verify_sample"] == sample
    assert out["verify_finish_reason"] == "stop"
    assert out["verify_latency_s"] >= 0.0


@pytest.mark.parametrize(
    ("constraint", "content"),
    [
        ({"json_object": True}, "<think>x</think>[]"),
        ({"json": {"type": "object", "required": ["answer"]}}, "<think>x</think>{}"),
        ({"choice": ["4"]}, "<think>x</think>5"),
        ({"regex": "[0-9]+"}, "<think>x</think>four"),
    ],
)
def test_run_deployment_smoke_rejects_structured_answer_violation(monkeypatch, constraint, content):
    import json

    spec = types.SimpleNamespace(
        thinking=True,
        train=types.SimpleNamespace(structured_outputs=json.dumps(constraint)),
    )
    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **_k: {"choices": [{"message": {"content": content}}]},
    )
    with pytest.raises(ServingError, match="structured smoke output"):
        serving._run_deployment_smoke(
            "run-1",
            spec,
            expected_checkpoint="run-1",
            expected_registry_revision=1,
            expected_mutation_id="mutation-1",
        )


def test_mutation_ownership_loss_before_smoke_returns_without_finalization(monkeypatch):
    spec = types.SimpleNamespace(thinking=False, train=types.SimpleNamespace(structured_outputs=""))
    deployment = {
        "state": "deploying",
        "requested_at": 1.0,
        "desired_record": {"checkpoint": "run-1", "mutation_id": "m1"},
        "target_revision": 1,
        "mutation_id": "m1",
    }
    newer = types.SimpleNamespace(
        deployment={"state": "deploying", "mutation_id": "m2", "requested_at": 2.0}
    )
    monkeypatch.setattr(serving._app, "get_status", lambda _run_id: newer)
    monkeypatch.setattr(serving._app, "serve_chat", lambda **_kwargs: pytest.fail("must not smoke"))
    monkeypatch.setattr(
        serving, "mark_deployed", lambda *_args, **_kwargs: pytest.fail("must not finalize")
    )

    serving._finalize_registered_deployment(
        "run-1",
        spec,
        deployment,
        checkpoint_step=None,
        is_checkpoint=False,
        prev_state="done",
        verify=True,
    )


def test_mutation_ownership_loss_after_smoke_returns_without_readback_or_finalization(monkeypatch):
    spec = types.SimpleNamespace(thinking=False, train=types.SimpleNamespace(structured_outputs=""))
    deployment = {
        "state": "deploying",
        "requested_at": 1.0,
        "desired_record": {"checkpoint": "run-1", "mutation_id": "m1"},
        "target_revision": 1,
        "mutation_id": "m1",
    }
    current = {"value": types.SimpleNamespace(deployment=deployment)}
    monkeypatch.setattr(serving._app, "get_status", lambda _run_id: current["value"])

    def smoke(**_kwargs):
        current["value"] = types.SimpleNamespace(
            deployment={"state": "deploying", "mutation_id": "m2", "requested_at": 2.0}
        )
        return {"choices": [{"message": {"content": "4"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(serving._app, "serve_chat", smoke)
    monkeypatch.setattr(
        serving._app, "read_adapter_record", lambda _run_id: pytest.fail("must not read back")
    )
    monkeypatch.setattr(
        serving, "mark_deployed", lambda *_args, **_kwargs: pytest.fail("must not finalize")
    )

    serving._finalize_registered_deployment(
        "run-1",
        spec,
        deployment,
        checkpoint_step=None,
        is_checkpoint=False,
        prev_state="done",
        verify=True,
    )


def test_finalization_write_carries_the_local_attempt_fence(monkeypatch):
    spec = types.SimpleNamespace(thinking=False, train=types.SimpleNamespace(structured_outputs=""))
    desired = {"checkpoint": "run-1", "mutation_id": "m1"}
    deployment = {
        "state": "deploying",
        "requested_at": 1.0,
        "desired_record": desired,
        "target_revision": 1,
        "mutation_id": "m1",
    }
    status = types.SimpleNamespace(deployment=deployment)
    seen = {}
    monkeypatch.setattr(serving._app, "get_status", lambda _run_id: status)
    monkeypatch.setattr(
        serving._app,
        "read_adapter_record",
        lambda _run_id: {**desired, "registry_revision": 1},
    )
    monkeypatch.setattr(serving._app, "record_matches", lambda *_args: True)
    monkeypatch.setattr(serving, "mark_deployed", lambda *_args, **kwargs: seen.update(kwargs))

    serving._finalize_registered_deployment(
        "run-1",
        spec,
        deployment,
        checkpoint_step=None,
        is_checkpoint=False,
        prev_state="done",
        verify=False,
    )

    assert seen == {
        "expect_mutation_id": "m1",
        "expect_state": "done",
        "expect_deployment_state": "deploying",
    }


def test_stale_same_checkpoint_smoke_fence_retries_while_exact_target_is_current(monkeypatch):
    import httpx

    spec = types.SimpleNamespace(thinking=False, train=types.SimpleNamespace(structured_outputs=""))
    desired = {"checkpoint": "run-1/step-4", "mutation_id": "m4"}
    deployment = {
        "desired_record": desired,
        "target_revision": 4,
        "mutation_id": "m4",
    }
    calls = {"count": 0}

    def serve_chat(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            request = httpx.Request("POST", "https://serve.example/v1/chat/completions")
            response = httpx.Response(409, text="deployment mismatch", request=request)
            raise httpx.HTTPStatusError("deployment mismatch", request=request, response=response)
        return {"choices": [{"message": {"content": "4"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(serving._app, "serve_chat", serve_chat)
    monkeypatch.setattr(serving._app, "read_adapter_record", lambda _run_id: desired)
    monkeypatch.setattr(serving._app, "record_matches", lambda *_args: True)

    result = serving._smoke_with_retries("run-1", spec, deployment)
    assert result["verify_sample"] == "4"
    assert calls["count"] == 2


def test_smoke_retries_transport_failure(monkeypatch):
    import httpx

    spec = types.SimpleNamespace(thinking=False, train=types.SimpleNamespace(structured_outputs=""))
    deployment = {
        "desired_record": {"checkpoint": "run-1", "mutation_id": "m1"},
        "target_revision": 1,
        "mutation_id": "m1",
    }
    calls = {"count": 0}

    def serve_chat(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            request = httpx.Request("POST", "https://serve.example/v1/chat/completions")
            raise httpx.ConnectError("reset", request=request)
        return {"choices": [{"message": {"content": "4"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(serving, "READBACK_DELAY_SECONDS", 0)
    monkeypatch.setattr(serving._app, "serve_chat", serve_chat)

    result = serving._smoke_with_retries("run-1", spec, deployment)

    assert result["verify_sample"] == "4"
    assert calls["count"] == 2
