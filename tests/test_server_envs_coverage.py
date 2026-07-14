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
import json
import multiprocessing
import subprocess
import tarfile
import time
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
    # The input is not mutated in place.
    assert original == {"a": 1, "state": "deploying"}

    pub = serving._public_deployment(
        {
            "run_id": "run-1",
            "state": "ready",
            "previous_deployment": {"x": 1},
            "endpoint_name": "https://serve.example",
            "openai_base_url": "https://serve.example/v1",
            "url": "https://stale.example/v1",
            "b": 2,
        }
    )
    assert pub == {
        "run_id": "run-1",
        "checkpoint_step": None,
        "adapter_revision": None,
        "state": "ready",
        "verified_at": None,
        "openai_model": "run-1",
        "endpoint_name": "https://serve.example",
        "openai_base_url": "https://serve.example/v1",
        "b": 2,
    }

    progressed = serving._deployment_state(
        {"state": "loading", "registered_at": 1.0},
        "registered",
        registered_at=2.0,
    )
    assert progressed["state"] == "loading"
    assert progressed["registered_at"] == 1.0


def test_deployment_attempt_is_stale_branches():
    # Not in a busy state -> never stale.
    assert serving._deployment_attempt_is_stale({"state": "ready"}) is False
    # Busy with no timestamp -> treated as stale.
    assert serving._deployment_attempt_is_stale({"state": "deploying"}) is True
    # Busy with an unparseable timestamp -> stale.
    assert (
        serving._deployment_attempt_is_stale({"state": "verifying", "updated_at": "nope"}) is True
    )
    # Busy but recently updated (via injected `now`) -> not stale.
    fresh = {"state": "registering", "updated_at": 1000.0}
    assert serving._deployment_attempt_is_stale(fresh, now=1000.0 + 10) is False
    # Busy and older than the stale window -> stale.
    old = {"state": "deploying", "requested_at": 1000.0}
    assert (
        serving._deployment_attempt_is_stale(old, now=1000.0 + serving._DEPLOYMENT_STALE_SECONDS)
        is True
    )


def test_previous_ready_deployment():
    # A currently-ready deployment is its own "previous ready" (a copy).
    ready = {"state": "ready", "adapter_hf_prefix": "sft/r1/seed0/adapter"}
    got = serving._previous_ready_deployment(ready)
    assert got == ready
    assert got is not ready
    # A busy deployment falls back to a ready `previous_deployment`.
    nested = {"state": "deploying", "previous_deployment": {"state": "deployed", "b": 2}}
    assert serving._previous_ready_deployment(nested) == {"state": "deployed", "b": 2}
    # terminal states never resurrect a preserved previous deployment.
    undeployed = {"state": "undeployed", "previous_deployment": {"state": "ready"}}
    assert serving._previous_ready_deployment(undeployed) is None
    # Nothing ready anywhere -> None.
    assert serving._previous_ready_deployment({"state": "deploying"}) is None
    assert (
        serving._previous_ready_deployment(
            {"state": "deploying", "previous_deployment": {"state": "failed"}}
        )
        is None
    )


def test_chat_messages_from_payload_validation():
    assert serving._chat_messages_from_payload({}) == []
    assert serving._chat_messages_from_payload({"messages": None}) == []

    valid = [{"role": "user", "content": "hi"}]
    assert serving._chat_messages_from_payload({"messages": valid}) is valid

    with pytest.raises(HTTPException) as not_list:
        serving._chat_messages_from_payload({"messages": "nope"})
    assert not_list.value.status_code == 400
    assert "messages must be a list" in not_list.value.detail

    with pytest.raises(HTTPException) as bad_item:
        serving._chat_messages_from_payload({"messages": [{"role": "user"}, "oops"]})
    assert bad_item.value.status_code == 400
    assert "messages[1]" in bad_item.value.detail


def test_validate_hf_repo_id_accepts_valid_and_rejects_malformed():
    # A well-formed id passes silently.
    serving._validate_hf_repo_id("owner/name")
    # A malformed id becomes a 400 before any download.
    with pytest.raises(HTTPException) as exc:
        serving._validate_hf_repo_id("bad//id")
    assert exc.value.status_code == 400
    assert "valid HuggingFace repo id" in exc.value.detail


def test_resolve_deploy_step_branches(monkeypatch):
    monkeypatch.setattr(serving._app, "list_checkpoints", lambda spec: [{"step": 20}, {"step": 40}])

    # No step requested -> final adapter (None), no lookup needed.
    assert serving._resolve_deploy_step("run-1", object(), None) is None
    # Matching int / integer-float / numeric-string all resolve.
    assert serving._resolve_deploy_step("run-1", object(), 20) == 20
    assert serving._resolve_deploy_step("run-1", object(), 40.0) == 40
    assert serving._resolve_deploy_step("run-1", object(), "40") == 40

    # A resolvable-but-unknown step is a 404 that lists what IS available.
    with pytest.raises(HTTPException) as not_found:
        serving._resolve_deploy_step("run-1", object(), 999)
    assert not_found.value.status_code == 404
    assert "20, 40" in not_found.value.detail

    # Bad step shapes are 400: bool, non-integer float, negative, and junk strings.
    for bad in (True, 20.5, -5, "-5", "abc", "1.5"):
        with pytest.raises(HTTPException) as exc:
            serving._resolve_deploy_step("run-1", object(), bad)
        assert exc.value.status_code == 400, bad


def test_deployment_cas_lost(monkeypatch):
    # No guard -> never lost, and get_status is not consulted.
    monkeypatch.setattr(
        serving._app,
        "get_status",
        lambda _r: pytest.fail("get_status must not be called without a guard"),
    )
    assert serving._deployment_cas_lost("run-1", None) is False

    # Guard set and live state diverged -> lost; state matches -> not lost.
    monkeypatch.setattr(
        serving._app, "get_status", lambda _r: types.SimpleNamespace(state="cancelled")
    )
    assert serving._deployment_cas_lost("run-1", "deploying") is True
    monkeypatch.setattr(
        serving._app, "get_status", lambda _r: types.SimpleNamespace(state="deploying")
    )
    assert serving._deployment_cas_lost("run-1", "deploying") is False


def test_recover_deployments_fails_stale_and_skips_fresh_and_missing(monkeypatch):
    import time

    rows = [{"run_id": "r-stale"}, {"run_id": "r-fresh"}, {"run_id": "r-missing"}]
    monkeypatch.setattr(serving.db, "all_runs", lambda: rows)

    statuses = {
        # Busy with no timestamp -> stale.
        "r-stale": types.SimpleNamespace(run_id="r-stale", deployment={"state": "deploying"}),
        # Busy but freshly updated -> not stale.
        "r-fresh": types.SimpleNamespace(
            run_id="r-fresh", deployment={"state": "deploying", "updated_at": time.time()}
        ),
    }

    def fake_get_status(run_id):
        if run_id == "r-missing":
            raise FileNotFoundError(run_id)
        return statuses[run_id]

    monkeypatch.setattr(serving._app, "get_status", fake_get_status)

    marked: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        serving, "mark_deployment_failed", lambda run_id, failed: marked.append((run_id, failed))
    )

    assert serving.recover_deployments() == 1
    assert len(marked) == 1
    run_id, failed = marked[0]
    assert run_id == "r-stale"
    assert failed["state"] == "failed"
    assert "control-plane restart" in failed["error"]


def _smoke_result(
    revision: str,
    checkpoint: str,
    content: str = "The answer is 4",
    finish_reason: str = "stop",
) -> dict:
    hf_revision = revision.rsplit(".", 1)[-1]
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "freesolo": {
            "adapter_revision": revision,
            "checkpoint": checkpoint,
            "hf_revision": hf_revision,
        },
        "_freesolo_headers": {
            "adapter_revision": revision,
            "checkpoint": checkpoint,
            "hf_revision": hf_revision,
        },
    }


def test_run_deployment_smoke_uses_fixed_fallback_before_first_request(monkeypatch):
    revision = "run-1@final." + "a" * 40
    spec = types.SimpleNamespace(
        thinking=False,
        environment=types.SimpleNamespace(id="owner/env", params={}, resolved_sha="b" * 40),
    )
    monkeypatch.setattr(
        serving,
        "load_environment",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )
    calls = []

    def fake_serve_chat(**kwargs):
        calls.append(kwargs)
        return _smoke_result(revision, "run-1")

    monkeypatch.setattr(serving._app, "serve_chat", fake_serve_chat)
    out = serving._run_deployment_smoke(
        "run-1", spec, serving_model=revision, expected_checkpoint="run-1"
    )
    assert out["verify_kind"] == "fixed_fallback"
    assert out["verify_turns"] == 1
    assert calls[0]["messages"] == [{"role": "user", "content": serving._SMOKE_PROMPT}]
    assert calls[0]["expected_checkpoint"] == "run-1"


def test_run_deployment_smoke_multi_turn_is_bounded_and_never_scores(monkeypatch):
    revision = "run-1@step-20." + "c" * 40
    checkpoint = "run-1/step-20"
    state = {"messages": [{"role": "user", "content": "start"}], "done": False}

    class Env:
        multi_turn = True

        def dataset(self):
            return [{"id": 1}]

        def new_rollout_state(self, example):
            assert example == {"id": 1}
            return state

        def record_model_turn(self, rollout, content):
            rollout["messages"].append({"role": "assistant", "content": content})

        def rollout_done(self, rollout):
            return False

        def env_reply(self, messages, rollout):
            messages.append({"role": "user", "content": "continue"})

        def reward(self, *args, **kwargs):
            pytest.fail("deployment smoke must not call reward")

        def judge(self, *args, **kwargs):
            pytest.fail("deployment smoke must not call judge")

        def score(self, *args, **kwargs):
            pytest.fail("deployment smoke must not call score")

        def grade(self, *args, **kwargs):
            pytest.fail("deployment smoke must not call grade")

        def quality(self, *args, **kwargs):
            pytest.fail("deployment smoke must not call quality")

    spec = types.SimpleNamespace(
        thinking=False,
        environment=types.SimpleNamespace(id="owner/env", params={}, resolved_sha=None),
    )
    monkeypatch.setattr(serving, "load_environment", lambda *args, **kwargs: Env())
    calls = []

    def fake_serve_chat(**kwargs):
        calls.append(kwargs)
        return _smoke_result(revision, checkpoint, content=f"turn {len(calls)}")

    monkeypatch.setattr(serving._app, "serve_chat", fake_serve_chat)
    out = serving._run_deployment_smoke(
        "run-1", spec, serving_model=revision, expected_checkpoint=checkpoint
    )
    assert out["verify_kind"] == "environment_multi_turn"
    assert out["verify_turns"] == 2
    assert len(calls) == 2


def test_run_deployment_smoke_blocked_environment_setup_times_out_within_budget(monkeypatch):
    # a hanging load_environment must fail the smoke with the stable timeout error inside the
    # budget, and no generation may ever run — never a fixed-prompt fallback after expiry.
    revision = "run-1@final." + "e" * 40
    spec = types.SimpleNamespace(
        thinking=False,
        environment=types.SimpleNamespace(id="owner/env", params={}, resolved_sha=None),
    )

    def hang(*args, **kwargs):
        time.sleep(30.0)

    monkeypatch.setattr(serving, "load_environment", hang)
    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **kwargs: pytest.fail("timed-out environment setup must not reach generation"),
    )
    started = time.monotonic()
    with pytest.raises(ServingError, match="deployment_smoke_timeout: bounded smoke exceeded"):
        serving._run_deployment_smoke(
            "run-1", spec, serving_model=revision, expected_checkpoint="run-1", budget_s=0.2
        )
    assert time.monotonic() - started < 5.0


def test_run_deployment_smoke_passes_remaining_budget_to_serve_chat(monkeypatch):
    # each generation gets the remaining budget, not the fixed 30-minute client timeout, and a
    # later turn sees strictly less than the first.
    revision = "run-1@step-20." + "f" * 40
    checkpoint = "run-1/step-20"
    state = {"messages": [{"role": "user", "content": "start"}]}

    class Env:
        multi_turn = True

        def dataset(self):
            return [{"id": 1}]

        def new_rollout_state(self, example):
            return state

        def record_model_turn(self, rollout, content):
            rollout["messages"].append({"role": "assistant", "content": content})

        def rollout_done(self, rollout):
            return False

        def env_reply(self, messages, rollout):
            messages.append({"role": "user", "content": "continue"})

    spec = types.SimpleNamespace(
        thinking=False,
        environment=types.SimpleNamespace(id="owner/env", params={}, resolved_sha=None),
    )
    monkeypatch.setattr(serving, "load_environment", lambda *args, **kwargs: Env())
    timeouts = []

    def fake_serve_chat(**kwargs):
        timeouts.append(kwargs["timeout_s"])
        time.sleep(0.05)
        return _smoke_result(revision, checkpoint)

    monkeypatch.setattr(serving._app, "serve_chat", fake_serve_chat)
    out = serving._run_deployment_smoke(
        "run-1", spec, serving_model=revision, expected_checkpoint=checkpoint, budget_s=10.0
    )
    assert out["verify_turns"] == 2
    assert len(timeouts) == 2
    assert all(t <= 10.0 for t in timeouts)
    assert timeouts[1] < timeouts[0]  # turn 2 only gets what turn 1 left over


def test_run_deployment_smoke_bounds_chat_by_wall_clock_deadline(monkeypatch):
    revision = "run-1@final." + "a" * 40
    spec = types.SimpleNamespace(
        thinking=False,
        environment=types.SimpleNamespace(id="owner/env", params={}, resolved_sha=None),
    )
    monkeypatch.setattr(
        serving,
        "load_environment",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )

    def slow_serve_chat(**kwargs):
        time.sleep(1.0)
        return _smoke_result(revision, "run-1")

    monkeypatch.setattr(serving._app, "serve_chat", slow_serve_chat)
    started = time.monotonic()
    with pytest.raises(ServingError, match="deployment_smoke_timeout: bounded smoke exceeded"):
        serving._run_deployment_smoke(
            "run-1", spec, serving_model=revision, expected_checkpoint="run-1", budget_s=0.05
        )
    assert time.monotonic() - started < 0.5


def test_run_deployment_smoke_expired_budget_fails_before_generation(monkeypatch):
    # once the deadline expires no further generation starts, and expiry never falls back to the
    # fixed prompt.
    revision = "run-1@final." + "a" * 40
    spec = types.SimpleNamespace(
        thinking=False,
        environment=types.SimpleNamespace(id="owner/env", params={}, resolved_sha=None),
    )

    class Env:
        multi_turn = False

        def dataset(self):
            time.sleep(0.05)  # consumes the whole (tiny) budget inside setup
            return [{"id": 1}]

        def prompt_messages(self, example):
            return [{"role": "user", "content": "hello"}]

    monkeypatch.setattr(serving, "load_environment", lambda *args, **kwargs: Env())
    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **kwargs: pytest.fail("expired budget must not start a generation"),
    )
    with pytest.raises(ServingError, match="deployment_smoke_timeout"):
        serving._run_deployment_smoke(
            "run-1", spec, serving_model=revision, expected_checkpoint="run-1", budget_s=0.01
        )


def test_run_deployment_smoke_env_setup_error_still_falls_back_with_budget_left(monkeypatch):
    # a pre-generation environment exception (not a timeout) keeps the fixed-prompt fallback as
    # long as budget remains.
    revision = "run-1@final." + "b" * 40
    spec = types.SimpleNamespace(
        thinking=False,
        environment=types.SimpleNamespace(id="owner/env", params={}, resolved_sha=None),
    )
    monkeypatch.setattr(
        serving,
        "load_environment",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("env repo broken")),
    )
    calls = []

    def fake_serve_chat(**kwargs):
        calls.append(kwargs)
        return _smoke_result(revision, "run-1")

    monkeypatch.setattr(serving._app, "serve_chat", fake_serve_chat)
    out = serving._run_deployment_smoke(
        "run-1", spec, serving_model=revision, expected_checkpoint="run-1", budget_s=10.0
    )
    assert out["verify_kind"] == "fixed_fallback"
    assert len(calls) == 1
    assert calls[0]["timeout_s"] <= 10.0


def test_run_deployment_smoke_does_not_fallback_after_first_request(monkeypatch):
    revision = "run-1@final." + "d" * 40

    class Env:
        multi_turn = True

        def dataset(self):
            return [{}]

        def new_rollout_state(self, example):
            return {"messages": [{"role": "user", "content": "start"}]}

        def record_model_turn(self, state, content):
            raise RuntimeError("transition broke")

        def rollout_done(self, state):
            return False

    spec = types.SimpleNamespace(
        thinking=False,
        environment=types.SimpleNamespace(id="owner/env", params={}, resolved_sha=None),
    )
    monkeypatch.setattr(serving, "load_environment", lambda *args, **kwargs: Env())
    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **kwargs: _smoke_result(revision, "run-1"),
    )
    with pytest.raises(ServingError, match="smoke_environment_failed"):
        serving._run_deployment_smoke(
            "run-1", spec, serving_model=revision, expected_checkpoint="run-1"
        )


_STRUCTURED_SMOKE_REVISION = "run-1@final." + "a" * 40


def _structured_smoke_spec(*, thinking: bool, constraint: dict | None = None):
    return types.SimpleNamespace(
        thinking=thinking,
        train=types.SimpleNamespace(
            structured_outputs="" if constraint is None else json.dumps(constraint)
        ),
        environment=types.SimpleNamespace(id="owner/env", params={}, resolved_sha="b" * 40),
    )


def _structured_smoke_response(content: str, finish_reason: str = "stop") -> dict:
    return _smoke_result(
        _STRUCTURED_SMOKE_REVISION,
        "run-1",
        content,
        finish_reason=finish_reason,
    )


def _run_structured_smoke(monkeypatch, run_id: str, spec):
    assert run_id == "run-1"
    monkeypatch.setattr(
        serving,
        "load_environment",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )
    return serving._run_deployment_smoke(
        run_id,
        spec,
        serving_model=_STRUCTURED_SMOKE_REVISION,
        expected_checkpoint=run_id,
    )


def _schema_validation_child_pids() -> set[int | None]:
    return {
        child.pid
        for child in multiprocessing.active_children()
        if child.name == serving._JSON_SCHEMA_PROCESS_NAME
    }


def test_run_deployment_smoke_success_and_empty(monkeypatch):
    spec = _structured_smoke_spec(thinking=False)

    def fake_serve_chat(**kwargs):
        assert kwargs["run_id"] == _STRUCTURED_SMOKE_REVISION
        assert kwargs["temperature"] == 0.0
        return _structured_smoke_response("The answer is 4")

    monkeypatch.setattr(serving._app, "serve_chat", fake_serve_chat)
    out = _run_structured_smoke(monkeypatch, "run-1", spec)
    assert out["verify_finish_reason"] == "stop"
    assert out["verify_sample"] == "The answer is 4"
    assert out["thinking_tag"] is False
    assert out["verify_latency_s"] >= 0.0

    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **_k: _structured_smoke_response("<think>still reasoning"),
    )
    thinking_out = _run_structured_smoke(
        monkeypatch, "run-1", _structured_smoke_spec(thinking=True)
    )
    assert thinking_out["thinking_tag"] is True
    assert thinking_out["verify_sample"] == "<think>still reasoning"

    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **_k: _structured_smoke_response("   ", "length"),
    )
    with pytest.raises(ServingError, match="no content"):
        _run_structured_smoke(monkeypatch, "run-1", spec)


@pytest.mark.parametrize(
    ("constraint", "content", "sample"),
    [
        (
            {"json": {"type": "object", "required": ["answer"]}},
            '<think>{"ignored":</think>{"answer": 4}',
            '{"answer": 4}',
        ),
        (
            {"json_object": True},
            '<think>[invalid reasoning json</think>{"answer": 4}',
            '{"answer": 4}',
        ),
        (
            {
                "json": {
                    "$defs": {"answer": {"type": "string"}},
                    "type": "object",
                    "properties": {"answer": {"$ref": "#/$defs/answer"}},
                    "required": ["answer"],
                }
            },
            '<think>2+2</think>{"answer": "4"}',
            '{"answer": "4"}',
        ),
        (
            {
                "json": {
                    "type": "object",
                    "properties": {"literal": {"type": "string"}},
                    "required": ["literal"],
                }
            },
            '<think>x</think>{"literal": "</think>"}',
            '{"literal": "</think>"}',
        ),
        ({"choice": ["4", "four"]}, "<think>2+2</think>4", "4"),
        ({"choice": ["<think>4"]}, "<think>2+2</think><think>4", "<think>4"),
        ({"regex": "[0-9]+"}, "<think>2+2</think>4", "4"),
        (
            {"regex": "answer</think>literal"},
            "<think>x</think>answer</think>literal",
            "answer</think>literal",
        ),
    ],
)
def test_thinking_structured_smoke_validates_only_answer_after_reasoning(
    monkeypatch, constraint, content, sample
):
    monkeypatch.setattr(
        serving._app, "serve_chat", lambda **_k: _structured_smoke_response(content)
    )
    out = _run_structured_smoke(
        monkeypatch, "run-1", _structured_smoke_spec(thinking=True, constraint=constraint)
    )
    assert out["verify_sample"] == sample
    assert out["verify_finish_reason"] == "stop"
    assert out["thinking_tag"] is True
    assert out["verify_latency_s"] >= 0.0


def test_structured_smoke_rejects_external_schema_ref_without_network(monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    constraint = {"json": {"$ref": f"http://127.0.0.1:{port}/schema.json"}}
    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **_k: _structured_smoke_response("<think>2+2</think>{}"),
    )

    try:
        with pytest.raises(ServingError, match="schema reference could not be resolved"):
            _run_structured_smoke(
                monkeypatch, "run-1", _structured_smoke_spec(thinking=True, constraint=constraint)
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)
    assert requests == []


def test_structured_smoke_reports_missing_local_schema_fragment_neutrally(monkeypatch):
    constraint = {
        "json": {
            "$defs": {"answer": {"type": "string"}},
            "$ref": "#/$defs/missing",
        }
    }
    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **_k: _structured_smoke_response('<think>2+2</think>{"answer": "4"}'),
    )

    with pytest.raises(ServingError, match="schema reference could not be resolved") as exc_info:
        _run_structured_smoke(
            monkeypatch, "run-1", _structured_smoke_spec(thinking=True, constraint=constraint)
        )
    assert "external retrieval" not in str(exc_info.value)


def test_direct_structured_regex_timeout_is_bounded(monkeypatch):
    answer = "a" * 10_000 + "!"
    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **_k: _structured_smoke_response(f"<think>x</think>{answer}"),
    )

    started = time.monotonic()
    with pytest.raises(ServingError, match=r"regex evaluation exceeded the 0\.05s deadline"):
        _run_structured_smoke(
            monkeypatch,
            "run-1",
            _structured_smoke_spec(thinking=True, constraint={"regex": "(a+)+$"}),
        )
    assert time.monotonic() - started < 1.0


@pytest.mark.parametrize("keyword", ["pattern", "patternProperties"])
def test_json_schema_pathological_regex_branch_times_out_before_fallback(monkeypatch, keyword):
    pathological = "(a+)+$"
    bad_value = "a" * 10_000 + "!"
    if keyword == "pattern":
        instance = bad_value
        failing_branch = {"type": "string", "pattern": pathological}
    else:
        instance = {bad_value: 1}
        failing_branch = {
            "type": "object",
            "patternProperties": {pathological: {"type": "integer"}},
        }
    schema = {"anyOf": [failing_branch, {"const": instance}]}
    answer = json.dumps(instance)
    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **_k: _structured_smoke_response(f"<think>x</think>{answer}"),
    )

    children_before = _schema_validation_child_pids()
    started = time.monotonic()
    with pytest.raises(ServingError, match="wall-clock deadline"):
        _run_structured_smoke(
            monkeypatch,
            "run-1",
            _structured_smoke_spec(thinking=True, constraint={"json": schema}),
        )
    assert time.monotonic() - started < 5.0
    assert _schema_validation_child_pids() == children_before


def test_json_schema_not_combinator_timeout_cannot_invert_to_success(monkeypatch):
    bad_value = "a" * 10_000 + "!"
    schema = {"not": {"type": "string", "pattern": "(a+)+$"}}
    answer = json.dumps(bad_value)
    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **_k: _structured_smoke_response(f"<think>x</think>{answer}"),
    )

    children_before = _schema_validation_child_pids()
    started = time.monotonic()
    with pytest.raises(ServingError, match="wall-clock deadline"):
        _run_structured_smoke(
            monkeypatch,
            "run-1",
            _structured_smoke_spec(thinking=True, constraint={"json": schema}),
        )
    assert time.monotonic() - started < 5.0
    assert _schema_validation_child_pids() == children_before


def test_json_schema_pattern_properties_unevaluated_timeout_is_killed(monkeypatch):
    bad_key = "a" * 10_000 + "!"
    instance = {bad_key: 1}
    schema = {
        "type": "object",
        "patternProperties": {"(a+)+$": {"type": "integer"}},
        "unevaluatedProperties": False,
    }
    answer = json.dumps(instance)
    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **_k: _structured_smoke_response(f"<think>x</think>{answer}"),
    )

    children_before = _schema_validation_child_pids()
    started = time.monotonic()
    with pytest.raises(ServingError, match="wall-clock deadline"):
        _run_structured_smoke(
            monkeypatch,
            "run-1",
            _structured_smoke_spec(thinking=True, constraint={"json": schema}),
        )
    assert time.monotonic() - started < 5.0
    assert _schema_validation_child_pids() == children_before


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("constraint_kind", ["json", "json_object"])
def test_structured_json_rejects_nonfinite_constants(monkeypatch, constant, constraint_kind):
    if constraint_kind == "json":
        constraint = {"json": {"type": "number"}}
        answer = constant
    else:
        constraint = {"json_object": True}
        answer = f'{{"value": {constant}}}'
    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **_k: _structured_smoke_response(f"<think>x</think>{answer}"),
    )

    with pytest.raises(ServingError, match="non-finite JSON constant"):
        _run_structured_smoke(
            monkeypatch, "run-1", _structured_smoke_spec(thinking=True, constraint=constraint)
        )


@pytest.mark.parametrize(
    ("constraint", "content", "finish_reason", "match"),
    [
        (
            {"json": {"type": "object", "required": ["answer"]}},
            "<think>x</think>{}",
            "stop",
            "violates the configured JSON schema",
        ),
        ({"json_object": True}, "<think>x</think>[]", "stop", "not a JSON object"),
        ({"json_object": True}, "<think>x</think>{", "stop", "not valid JSON"),
        ({"choice": ["4"]}, "<think>x</think>5", "stop", "is not one of"),
        ({"regex": "[0-9]+"}, "<think>x</think>four", "stop", "does not match"),
        ({"choice": ["4"]}, "4", "stop", "never closed its reasoning"),
        ({"choice": ["4"]}, "<think>2+2", "stop", "never closed its reasoning"),
        ({"choice": ["4"]}, "<think>x</think>   ", "stop", "no answer"),
        (
            {"choice": ["4"]},
            "<think>x</think><think>4",
            "stop",
            "is not one of",
        ),
        (
            {"choice": ["4"]},
            "<think>x</think>garbage</think>4",
            "stop",
            "is not one of",
        ),
        (
            {"choice": ["4"]},
            "<think>x</think>4",
            "length",
            "truncated at the maximum token length",
        ),
    ],
)
def test_thinking_structured_smoke_rejects_invalid_output(
    monkeypatch, constraint, content, finish_reason, match
):
    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **_k: _structured_smoke_response(content, finish_reason),
    )
    with pytest.raises(ServingError, match=match):
        _run_structured_smoke(
            monkeypatch, "run-1", _structured_smoke_spec(thinking=True, constraint=constraint)
        )


def test_nonthinking_structured_smoke_validates_whole_stripped_content(monkeypatch):
    content = "  <think>literal</think>4  "
    monkeypatch.setattr(
        serving._app, "serve_chat", lambda **_k: _structured_smoke_response(content)
    )
    out = _run_structured_smoke(
        monkeypatch,
        "run-1",
        _structured_smoke_spec(
            thinking=False,
            constraint={"choice": ["<think>literal</think>4"]},
        ),
    )
    assert out["verify_sample"] == "<think>literal</think>4"
