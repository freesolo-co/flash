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
from typing import ClassVar

import pytest

from flash.server import envs

pytest.importorskip("fastapi")
from fastapi import HTTPException

import flash.server.routes.serving as serving
from flash.engine.recipe import RECIPE
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
    original = {"a": 1, "state": "queued"}
    out = serving._deployment_state(original, "ready", detail="done")
    assert out["state"] == "ready"
    assert out["detail"] == "done"
    assert out["a"] == 1
    assert isinstance(out["updated_at"], float)
    # The input is not mutated in place.
    assert original == {"a": 1, "state": "queued"}

    pub = serving._public_deployment(
        {
            "state": "ready",
            "previous_deployment": {"x": 1},
            "endpoint_name": "https://serve.example",
            "openai_base_url": "https://serve.example/v1",
            "url": "https://stale.example/v1",
            "b": 2,
        }
    )
    assert pub == {
        "state": "ready",
        "endpoint_name": "https://serve.example",
        "openai_base_url": "https://serve.example/v1",
        "b": 2,
        "run_id": None,
        "checkpoint_step": None,
        "adapter_revision": None,
        "verified_at": None,
        "openai_model": None,
    }


def test_deployment_attempt_is_stale_branches():
    # Not in a busy state -> never stale.
    assert serving._deployment_attempt_is_stale({"state": "ready"}) is False
    # Busy with no timestamp -> treated as stale.
    assert serving._deployment_attempt_is_stale({"state": "queued"}) is True
    # Busy with an unparseable timestamp -> stale.
    assert (
        serving._deployment_attempt_is_stale({"state": "smoke_testing", "updated_at": "nope"})
        is True
    )
    # Busy but recently updated (via injected `now`) -> not stale.
    fresh = {"state": "reconciling", "updated_at": 1000.0}
    assert serving._deployment_attempt_is_stale(fresh, now=1000.0 + 10) is False
    # Busy and older than the stale window -> stale.
    old = {"state": "queued", "requested_at": 1000.0}
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


def test_recover_deployments_fails_busy_and_skips_missing(monkeypatch):
    import time

    rows = [{"run_id": "r-stale"}, {"run_id": "r-fresh"}, {"run_id": "r-missing"}]
    monkeypatch.setattr(serving.db, "all_runs", lambda: rows)

    statuses = {
        # busy with no timestamp
        "r-stale": types.SimpleNamespace(
            run_id="r-stale", state="done", deployment={"state": "queued"}
        ),
        # busy and freshly updated
        "r-fresh": types.SimpleNamespace(
            run_id="r-fresh",
            state="done",
            deployment={"state": "queued", "updated_at": time.time()},
        ),
    }

    def fake_get_status(run_id):
        if run_id == "r-missing":
            raise FileNotFoundError(run_id)
        return statuses[run_id]

    monkeypatch.setattr(serving._app, "get_status", fake_get_status)

    import flash.runner as runner

    marked: list[tuple[str, dict]] = []
    reported = []

    def mark_failed(run_id, failed):
        marked.append((run_id, failed))
        return types.SimpleNamespace(run_id=run_id, state="done", deployment=failed)

    monkeypatch.setenv("FLASH_DEPLOY_SYNC", "1")
    monkeypatch.setattr(serving, "mark_deployment_failed", mark_failed)
    monkeypatch.setattr(runner, "_report_status", reported.append)

    assert serving.recover_deployments() == 2
    assert [run_id for run_id, _failed in marked] == ["r-stale", "r-fresh"]
    assert all(failed["state"] == "failed" for _run_id, failed in marked)
    assert all("control-plane restart" in failed["error"] for _run_id, failed in marked)
    assert [status.run_id for status in reported] == ["r-stale", "r-fresh"]
    assert all(status.deployment["state"] == "failed" for status in reported)


def test_recover_deployments_reports_restored_ready_predecessor(monkeypatch):
    import flash.runner as runner

    previous = {
        "state": "ready",
        "endpoint_name": "https://serve.example",
        "adapter_revision": "r-stale@final." + "a" * 40,
    }
    status = types.SimpleNamespace(
        run_id="r-stale",
        state="deployed",
        deployment={"state": "queued", "previous_deployment": previous},
    )
    monkeypatch.setattr(serving.db, "all_runs", lambda: [{"run_id": "r-stale"}])
    monkeypatch.setattr(serving._app, "get_status", lambda _run_id: status)

    def mark_failed(run_id, failed):
        assert run_id == "r-stale"
        return types.SimpleNamespace(
            run_id=run_id,
            state="deployed",
            deployment={
                **previous,
                "last_deploy_error": failed["error"],
                "last_deploy_failed_at": time.time(),
            },
        )

    reported = []

    monkeypatch.setenv("FLASH_DEPLOY_SYNC", "1")
    monkeypatch.setattr(serving, "mark_deployment_failed", mark_failed)
    monkeypatch.setattr(runner, "_report_status", reported.append)

    assert serving.recover_deployments() == 1
    assert len(reported) == 1
    assert reported[0].deployment["state"] == "ready"
    assert "control-plane restart" in reported[0].deployment["last_deploy_error"]


def _smoke_spec(
    *,
    thinking: bool,
    constraint: dict | None = None,
    max_completion_tokens: int | None = None,
    max_context_tokens: int | None = None,
    algorithm: str = "grpo",
    stop_sequences: tuple[str, ...] = (),
):
    return types.SimpleNamespace(
        model="Qwen/Qwen3.5-4B",
        algorithm=algorithm,
        thinking=thinking,
        train=types.SimpleNamespace(
            max_completion_tokens=max_completion_tokens,
            max_context_tokens=max_context_tokens,
            structured_outputs="" if constraint is None else json.dumps(constraint),
            stop_sequences=stop_sequences,
        ),
    )


_SMOKE_REVISION = "run-1@final." + "a" * 40


def _smoke_response(content: str, finish_reason: str = "stop") -> dict:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "freesolo": {
            "adapter_revision": _SMOKE_REVISION,
            "checkpoint": "run-1",
            "hf_revision": "a" * 40,
        },
        "_freesolo_headers": {
            "adapter_revision": _SMOKE_REVISION,
            "checkpoint": "run-1",
            "hf_revision": "a" * 40,
        },
    }


def _run_smoke(spec, *, budget_s: float = 600.0):
    return serving._run_deployment_smoke(
        "run-1",
        spec,
        serving_model=_SMOKE_REVISION,
        expected_checkpoint="run-1",
        budget_s=budget_s,
    )


def _schema_validation_child_pids() -> set[int | None]:
    return {
        child.pid
        for child in multiprocessing.active_children()
        if child.name == serving._JSON_SCHEMA_PROCESS_NAME
    }


def test_run_deployment_smoke_uses_only_trusted_fixed_prompt(monkeypatch):
    class UntrustedEnvironment:
        def __getattribute__(self, name):
            pytest.fail(f"control-plane smoke accessed user environment field {name!r}")

    spec = _smoke_spec(thinking=False)
    spec.environment = UntrustedEnvironment()
    calls = []

    def fake_serve_chat(**kwargs):
        calls.append(kwargs)
        return _smoke_response("The answer is 4")

    monkeypatch.setattr(serving._app, "serve_chat", fake_serve_chat)
    out = _run_smoke(spec, budget_s=10.0)

    assert out["verify_kind"] == "fixed_prompt"
    assert out["verify_turns"] == 1
    assert calls[0]["messages"] == [{"role": "user", "content": serving._SMOKE_PROMPT}]
    assert calls[0]["expected_checkpoint"] == "run-1"
    assert calls[0]["timeout_s"] <= 10.0
    assert calls[0]["retry_unavailable"] is True


def test_run_deployment_smoke_uses_thinking_completion_budget(monkeypatch):
    calls = []

    def fake_serve_chat(**kwargs):
        calls.append(kwargs)
        return _smoke_response('<think>reasoning</think>{"answer":"4"}')

    monkeypatch.setattr(serving._app, "serve_chat", fake_serve_chat)
    _run_smoke(_smoke_spec(thinking=True, constraint={"json_object": True}))

    assert calls[0]["max_tokens"] == 1536


@pytest.mark.parametrize(
    "spec",
    [
        _smoke_spec(
            algorithm="sft",
            thinking=False,
            constraint={"json_object": True},
            max_completion_tokens=8192,
        ),
        _smoke_spec(algorithm="grpo", thinking=False, max_completion_tokens=8192),
        _smoke_spec(algorithm="opd", thinking=False, max_completion_tokens=8192),
    ],
)
def test_run_deployment_smoke_keeps_non_thinking_paths_at_256(monkeypatch, spec):
    """A non-thinking adapter emits its answer immediately, so 256 tokens is enough for every
    algorithm. Only the thinking path needs the run's real budget."""
    calls = []

    def fake_serve_chat(**kwargs):
        calls.append(kwargs)
        return _smoke_response("{}")

    monkeypatch.setattr(serving._app, "serve_chat", fake_serve_chat)
    _run_smoke(spec)

    assert calls[0]["max_tokens"] == 256


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        # opd with an explicit budget, no grammar: previously fell to 256 and could not deploy.
        (_smoke_spec(algorithm="opd", thinking=True, max_completion_tokens=8192), 8192),
        # grpo thinking with no grammar and no explicit budget -> the thinking recipe default.
        (_smoke_spec(algorithm="grpo", thinking=True), 1536),
    ],
)
def test_run_deployment_smoke_budgets_thinking_without_structured_outputs(
    monkeypatch, spec, expected
):
    """A thinking adapter spends its budget reasoning BEFORE emitting content, so 256 tokens buys a
    truncated <think> block and no answer -- the smoke then fails "returned no content
    (finish_reason='length')" and the deployment is rejected. That is a property of thinking, not of
    structured_outputs, so a run using stop_sequences instead of a grammar was undeployable: every
    checkpoint failed the same gate. The larger budget must follow spec.thinking alone."""
    calls = []

    def fake_serve_chat(**kwargs):
        calls.append(kwargs)
        return _smoke_response("<think>2+2 is 4</think>The answer is 4")

    monkeypatch.setattr(serving._app, "serve_chat", fake_serve_chat)
    out = _run_smoke(spec)

    assert calls[0]["max_tokens"] == expected
    # every thinking smoke splits on </think> now, grammar or not, so the sample is the answer.
    assert out["verify_sample"] == "The answer is 4"
    assert out["thinking_tag"] is True


def test_run_deployment_smoke_rejects_truncated_thinking_without_a_grammar(monkeypatch):
    """Widening the budget must not turn a hard failure into a silent pass. Serving hands the
    reasoning back in reasoning_content and flash folds it into a balanced block, so a run cut off
    mid-thought arrives with non-empty content and clears the empty-content check -- the smoke would
    certify a truncated non-answer as a working deployment. finish_reason='length' on a thinking run
    is a failure whether or not a grammar is configured."""
    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **_k: _smoke_response("<think>2 plus 2 is</think>", "length"),
    )

    with pytest.raises(ServingError, match="smoke generation was truncated at the maximum token"):
        _run_smoke(_smoke_spec(algorithm="opd", thinking=True, max_completion_tokens=8192))


def test_run_deployment_smoke_forwards_configured_stop_sequences(monkeypatch):
    """A run trained with stop_sequences terminates on its delimiter and need never emit EOS. If the
    smoke does not forward them, serving generates past the answer to max_tokens and returns
    finish_reason='length' -- and the truncation guard above then rejects a checkpoint that answered
    correctly, making every such thinking run undeployable."""
    calls = []

    def fake_serve_chat(**kwargs):
        calls.append(kwargs)
        return _smoke_response("<think>2+2 is 4</think><answer>4</answer>")

    monkeypatch.setattr(serving._app, "serve_chat", fake_serve_chat)
    _run_smoke(
        _smoke_spec(
            algorithm="grpo",
            thinking=True,
            max_completion_tokens=8192,
            stop_sequences=("</answer>",),
        )
    )


def test_run_deployment_smoke_rejects_a_stop_that_fires_while_still_reasoning(monkeypatch):
    """Forwarding stop_sequences opened a hole the truncation guard cannot cover. A run whose
    delimiter appears inside its reasoning terminates mid-thought with finish_reason='stop', not
    'length', so the truncation check passes; serving folds the partial reasoning into a balanced
    <think> block, so the empty-content check passes too. Without a post-</think> answer
    requirement on every thinking smoke, that checkpoint activates and then answers nothing on
    real requests."""
    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        # finish_reason="stop": the delimiter fired, so this is NOT caught as truncation.
        lambda **_k: _smoke_response("<think>2 plus 2 is</think>   ", "stop"),
    )

    with pytest.raises(ServingError, match="no answer after </think>"):
        _run_smoke(
            _smoke_spec(
                algorithm="grpo",
                thinking=True,
                max_completion_tokens=8192,
                stop_sequences=("</answer>",),
            )
        )


def test_sft_smoke_budget_ignores_the_rollout_only_completion_knob(monkeypatch):
    """max_completion_tokens is a rollout knob the SFT worker ignores: it trains one packed
    prompt+completion block bounded by max_context_tokens. Honouring it for the SFT smoke caps
    generation below what the run legitimately produces, so a 2048-context thinking SFT run with a
    leftover max_completion_tokens=320 trains valid long reasoning it can never deploy."""
    calls = []

    def fake_serve_chat(**kwargs):
        calls.append(kwargs)
        return _smoke_response("<think>2+2 is 4</think>The answer is 4")

    monkeypatch.setattr(serving._app, "serve_chat", fake_serve_chat)
    _run_smoke(
        _smoke_spec(
            algorithm="sft",
            thinking=True,
            max_context_tokens=2048,
            max_completion_tokens=320,
        )
    )

    # the sft thinking recipe default, not the 320 the worker never reads.
    assert calls[0]["max_tokens"] == RECIPE.sft.max_seq_len_thinking
    assert calls[0]["max_tokens"] > 320


def test_run_deployment_smoke_sends_no_stop_when_none_configured(monkeypatch):
    """An unconfigured run must not send stop=[]: an empty list is a different request than an
    absent key on some OpenAI surfaces, and the run never asked for a delimiter."""
    calls = []

    def fake_serve_chat(**kwargs):
        calls.append(kwargs)
        return _smoke_response("The answer is 4")

    monkeypatch.setattr(serving._app, "serve_chat", fake_serve_chat)
    _run_smoke(_smoke_spec(thinking=False))

    assert calls[0]["stop"] is None


def test_chat_body_carries_stop_sequences(monkeypatch):
    """The stop sequences must reach the wire body, not just the flash-side call."""
    from flash.serve import deploy as _deploy

    sent = {}

    class _Resp:
        status_code = 200
        headers: ClassVar[dict[str, str]] = {}

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "4"}, "finish_reason": "stop"}]}

    class _Client:
        def post(self, url, json=None, headers=None, timeout=None):
            sent.update(json or {})
            return _Resp()

    monkeypatch.setattr(_deploy, "serving_openai_base_url", lambda: "https://serving.example/v1")
    monkeypatch.setattr(_deploy, "_internal_key_header", dict)
    monkeypatch.setattr(_deploy, "_chat_http_client", _Client)

    _deploy.chat("run-1", [{"role": "user", "content": "hi"}], stop=["</answer>"])
    assert sent["stop"] == ["</answer>"]

    sent.clear()
    _deploy.chat("run-1", [{"role": "user", "content": "hi"}])
    assert "stop" not in sent


def test_zero_completion_budget_resolves_to_thinking_recipe_default():
    from flash.serve.preflight import resolve_effective_completion_tokens

    spec = _smoke_spec(
        thinking=True,
        constraint={"json_object": True},
        max_completion_tokens=0,
    )

    assert resolve_effective_completion_tokens(spec) == 1536


def test_thinking_sft_smoke_budget_comes_from_the_sft_recipe_not_the_rl_default():
    from flash.serve.preflight import resolve_smoke_completion_tokens

    spec = _smoke_spec(thinking=True, algorithm="sft")

    # the rl thinking default (1536) is shorter than what sft actually trains, so resolving to it
    # would truncate the smoke and reject a checkpoint that answered correctly.
    assert RECIPE.rl.max_completion_len_thinking < RECIPE.sft.max_seq_len_thinking
    assert resolve_smoke_completion_tokens(spec) == RECIPE.sft.max_seq_len_thinking


def test_sft_smoke_budget_follows_an_explicit_context_over_the_recipe_default():
    from flash.serve.preflight import resolve_smoke_completion_tokens

    # the worker bounds the packed block by max_context_tokens and only falls back to the recipe
    # when it is unset (flash/engine/worker/sft.py), so the smoke has to resolve the same way.
    # sizing an 8192-context run at the 2048 recipe default truncated the smoke and rejected a
    # checkpoint that answered correctly.
    assert RECIPE.sft.max_seq_len_thinking < 8192
    spec = _smoke_spec(thinking=True, algorithm="sft", max_context_tokens=8192)
    assert resolve_smoke_completion_tokens(spec) == 8192

    # a shorter explicit context is honoured too -- the point is that the worker's number wins,
    # not that the budget only ever grows.
    short = _smoke_spec(thinking=True, algorithm="sft", max_context_tokens=512)
    assert resolve_smoke_completion_tokens(short) == 512

    # non-thinking takes the same path.
    assert (
        resolve_smoke_completion_tokens(
            _smoke_spec(thinking=False, algorithm="sft", max_context_tokens=4096)
        )
        == 4096
    )

    # a non-positive value is not a budget, so the recipe default still applies.
    assert (
        resolve_smoke_completion_tokens(
            _smoke_spec(thinking=True, algorithm="sft", max_context_tokens=0)
        )
        == RECIPE.sft.max_seq_len_thinking
    )


def test_nonthinking_sft_smoke_budget_comes_from_the_sft_recipe_not_the_rl_default():
    from flash.serve.preflight import resolve_smoke_completion_tokens

    spec = _smoke_spec(thinking=False, algorithm="sft")

    assert resolve_smoke_completion_tokens(spec) == RECIPE.sft.max_seq_len


def test_sft_contributes_no_completion_budget_to_the_serving_context_guard():
    from flash.serve.preflight import resolve_effective_completion_tokens

    # max_context_tokens spans prompt AND completion for sft, so handing it to the guard as a
    # completion budget double-counts the prompt: a 4096-context run that fits a 4096 serving cap
    # exactly was rejected for exceeding the 4096-256 completion capacity. the guard checks the
    # context separately, so sft must contribute no completion number at all.
    assert resolve_effective_completion_tokens(_smoke_spec(thinking=True, algorithm="sft")) is None
    assert (
        resolve_effective_completion_tokens(
            _smoke_spec(thinking=True, algorithm="sft", max_context_tokens=4096)
        )
        is None
    )
    # the rollout-only knob must not resurrect a budget for sft either.
    assert (
        resolve_effective_completion_tokens(
            _smoke_spec(thinking=True, algorithm="sft", max_completion_tokens=320)
        )
        is None
    )


def test_rollout_budget_ignores_context_tokens():
    from flash.engine.recipe import RECIPE
    from flash.serve.preflight import resolve_effective_completion_tokens

    # grpo budgets the completion, not the whole rollout, so max_context_tokens must not become
    # its completion budget the way it does for sft.
    spec = _smoke_spec(thinking=True, algorithm="grpo", max_context_tokens=4096)

    assert resolve_effective_completion_tokens(spec) == RECIPE.rl.max_completion_len_thinking


def test_run_deployment_smoke_retries_recognized_cold_503(monkeypatch):
    calls = []
    sleeps = []

    def fake_serve_chat(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise serving.RetryableServingUnavailable("adapter_loading", 0.25)
        return _smoke_response("The answer is 4")

    monkeypatch.setattr(serving._app, "serve_chat", fake_serve_chat)
    monkeypatch.setattr(serving.time, "sleep", sleeps.append)

    out = _run_smoke(_smoke_spec(thinking=False), budget_s=10.0)

    assert out["verify_sample"] == "The answer is 4"
    assert sleeps == [0.25]
    assert len(calls) == 2
    assert 0 < calls[1]["timeout_s"] <= calls[0]["timeout_s"] <= 10.0


def test_run_deployment_smoke_retry_stays_inside_wall_clock_budget(monkeypatch):
    clock = [100.0]
    calls = []
    sleeps = []

    def fake_serve_chat(**kwargs):
        calls.append(kwargs)
        raise serving.RetryableServingUnavailable("engine_unavailable", 10.0)

    def fake_sleep(delay):
        sleeps.append(delay)
        clock[0] += delay

    monkeypatch.setattr(serving._app, "serve_chat", fake_serve_chat)
    monkeypatch.setattr(serving.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(serving.time, "sleep", fake_sleep)

    with pytest.raises(ServingError, match="deployment_smoke_timeout"):
        _run_smoke(_smoke_spec(thinking=False), budget_s=2.0)

    assert len(calls) == 1
    assert calls[0]["timeout_s"] == 2.0
    assert sleeps == [2.0]
    assert clock[0] == 102.0


def test_run_deployment_smoke_bounds_chat_by_wall_clock_deadline(monkeypatch):
    def slow_serve_chat(**kwargs):
        time.sleep(1.0)
        return _smoke_response("The answer is 4")

    monkeypatch.setattr(serving._app, "serve_chat", slow_serve_chat)
    started = time.monotonic()
    with pytest.raises(ServingError, match="deployment_smoke_timeout: bounded smoke exceeded"):
        _run_smoke(_smoke_spec(thinking=False), budget_s=0.05)
    assert time.monotonic() - started < 0.5


def test_run_deployment_smoke_expired_budget_fails_before_generation(monkeypatch):
    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **kwargs: pytest.fail("expired budget must not start a generation"),
    )

    with pytest.raises(ServingError, match="deployment_smoke_timeout"):
        _run_smoke(_smoke_spec(thinking=False), budget_s=0.0)


def test_run_deployment_smoke_success_and_empty(monkeypatch):
    spec = _smoke_spec(thinking=False)

    def fake_serve_chat(**kwargs):
        assert kwargs["run_id"] == _SMOKE_REVISION
        assert kwargs["messages"] == [{"role": "user", "content": serving._SMOKE_PROMPT}]
        assert kwargs["temperature"] == 0.0
        assert 0 < kwargs["timeout_s"] <= 600.0
        return _smoke_response("The answer is 4")

    monkeypatch.setattr(serving._app, "serve_chat", fake_serve_chat)
    out = _run_smoke(spec)
    assert out["verify_kind"] == "fixed_prompt"
    assert out["verify_turns"] == 1
    assert out["verify_finish_reason"] == "stop"
    assert out["verify_sample"] == "The answer is 4"
    assert out["thinking_tag"] is False
    assert out["verify_latency_s"] >= 0.0

    # a thinking adapter that stops mid-reasoning has answered nothing, so it must not activate.
    # this used to pass because the </think> requirement was gated on structured outputs; a run
    # using stop_sequences reaches exactly this shape with finish_reason="stop", which slips past
    # the truncation guard.
    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **_k: _smoke_response("<think>still reasoning"),
    )
    with pytest.raises(ServingError, match="never closed its reasoning"):
        _run_smoke(_smoke_spec(thinking=True))

    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **_k: _smoke_response("<think>2+2 is 4</think>The answer is 4"),
    )
    thinking_out = _run_smoke(_smoke_spec(thinking=True))
    assert thinking_out["thinking_tag"] is True
    assert thinking_out["verify_sample"] == "The answer is 4"

    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **_k: _smoke_response("   ", "length"),
    )
    with pytest.raises(ServingError, match="no content"):
        _run_smoke(spec)


def test_unconstrained_thinking_smoke_rejects_a_reconstructed_empty_answer(monkeypatch):
    """An answerless thinking generation must not activate a deployment.

    `_smoke_provenance` rejects an empty generation, but flash now folds the split-out
    `reasoning_content` back into a balanced block before it gets there
    (flash/serve/deploy.py::_balanced_thinking_content). Reasoning that exhausts the token budget
    therefore arrives as a NONEMPTY `<think>...</think>` wrapping an empty answer, which sails past
    the emptiness check. `_thinking_answer` asserts an answer exists for every thinking smoke;
    before it did, only the grammar-constrained path checked, so an unconstrained deployment could
    go live on a smoke that produced no answer at all.

    Both ways of arriving answerless are covered, because they are rejected by different guards: a
    budget exhaustion carries finish_reason "length" and is caught by the truncation check, while a
    run whose stop_sequence fires mid-reasoning carries "stop" and reaches `_thinking_answer`.
    """
    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **_k: _smoke_response("<think>reasoned but ran out of budget</think>", "length"),
    )

    with pytest.raises(ServingError, match="truncated at the maximum token length"):
        _run_smoke(_smoke_spec(thinking=True))

    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **_k: _smoke_response("<think>reasoned, then the stop sequence fired</think>"),
    )

    with pytest.raises(ServingError, match="no answer after"):
        _run_smoke(_smoke_spec(thinking=True))


def test_thinking_smoke_rejects_a_retained_close_tag_as_the_answer(monkeypatch):
    """A doubled close tag is not an answer, whichever way it arose.

    A compatibility backend that retains only the sampled `</think>` in `content` gives the fold no
    answer to place behind the block, so `_balanced_thinking_content` emits `<think>why</think>
    </think>`. Splitting on the first close then left `</think>` as the answer, which is nonempty and
    sailed through, activating a checkpoint that answered nothing.

    The fold cannot decide this: those bytes are identical to an adapter whose answer genuinely is
    the tag, and it also backs the public chat route where the text must survive. Rejecting at the
    deployment gate is the fail-closed direction, and neither shape answers "what is 2+2".
    """
    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **_k: _smoke_response("<think>why</think></think>"),
    )
    with pytest.raises(ServingError, match="only a close tag"):
        _run_smoke(_smoke_spec(thinking=True))


def test_thinking_smoke_accepts_a_tagless_answer_only_under_the_open_model_policy(monkeypatch):
    """A model whose template flash cannot verify must not be undeployable for omitting the tag.

    `flash.schema` warns and proceeds when the catalog reports `thinking == "unknown"`, which only
    the open-model policy produces. Such a run may answer with no `<think>` block at all, so the
    strict requirement would reject a correct answer and strand the trained adapter. Every catalog
    model keeps the strict path, since for those the missing tag really does mean the reasoning
    never closed.
    """
    monkeypatch.setattr(serving._app, "serve_chat", lambda **_k: _smoke_response("4"))

    strict = _smoke_spec(thinking=True)
    with pytest.raises(ServingError, match="never closed its reasoning"):
        _run_smoke(strict)

    unknown = _smoke_spec(thinking=True)
    unknown.model = "some-org/not-in-the-catalog"
    unknown.model_policy = "allow"
    out = _run_smoke(unknown)
    assert out["verify_sample"] == "4"
    assert out["thinking_tag"] is False

    # the relaxed branch still rejects a run that answered nothing, which is the defect the strict
    # requirement exists to catch. only the tag demand is lifted, not the answer demand.
    monkeypatch.setattr(serving._app, "serve_chat", lambda **_k: _smoke_response("   "))
    with pytest.raises(ServingError, match="no content"):
        _run_smoke(unknown)


def test_unconstrained_thinking_smoke_accepts_an_answer_after_the_block(monkeypatch):
    # the companion direction: a reconstructed block WITH an answer after it still passes, so the
    # check above rejects answerlessness rather than reconstruction.
    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **_k: _smoke_response("<think>reasoned</think>The answer is 4"),
    )

    out = _run_smoke(_smoke_spec(thinking=True))

    # the sample is the ANSWER, not the whole generation: reasoning is what the deployment must not
    # be judged on, and it is already reported through thinking_tag.
    assert out["verify_sample"] == "The answer is 4"
    assert out["thinking_tag"] is True


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
    monkeypatch.setattr(serving._app, "serve_chat", lambda **_k: _smoke_response(content))
    out = _run_smoke(_smoke_spec(thinking=True, constraint=constraint))
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
        lambda **_k: _smoke_response("<think>2+2</think>{}"),
    )

    try:
        with pytest.raises(ServingError, match="schema reference could not be resolved"):
            _run_smoke(_smoke_spec(thinking=True, constraint=constraint))
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
        lambda **_k: _smoke_response('<think>2+2</think>{"answer": "4"}'),
    )

    with pytest.raises(ServingError, match="schema reference could not be resolved") as exc_info:
        _run_smoke(_smoke_spec(thinking=True, constraint=constraint))
    assert "external retrieval" not in str(exc_info.value)


def test_direct_structured_regex_timeout_is_bounded(monkeypatch):
    answer = "a" * 10_000 + "!"
    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **_k: _smoke_response(f"<think>x</think>{answer}"),
    )

    started = time.monotonic()
    with pytest.raises(ServingError, match=r"regex evaluation exceeded the 0\.05s deadline"):
        _run_smoke(
            _smoke_spec(thinking=True, constraint={"regex": "(a+)+$"}),
        )
    assert time.monotonic() - started < 1.0


def test_json_schema_validation_respects_global_smoke_deadline(monkeypatch):
    answer = json.dumps("a" * 10_000 + "!")
    schema = {"type": "string", "pattern": "(a+)+$"}
    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **_kwargs: _smoke_response(f"<think>x</think>{answer}"),
    )

    children_before = _schema_validation_child_pids()
    started = time.monotonic()
    with pytest.raises(
        ServingError, match=r"deployment_smoke_timeout: bounded smoke exceeded 0\.2s"
    ):
        _run_smoke(
            _smoke_spec(thinking=True, constraint={"json": schema}),
            budget_s=0.2,
        )
    assert time.monotonic() - started < 1.5
    assert _schema_validation_child_pids() == children_before


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
        lambda **_k: _smoke_response(f"<think>x</think>{answer}"),
    )

    children_before = _schema_validation_child_pids()
    started = time.monotonic()
    with pytest.raises(ServingError, match="wall-clock deadline"):
        _run_smoke(
            _smoke_spec(thinking=True, constraint={"json": schema}),
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
        lambda **_k: _smoke_response(f"<think>x</think>{answer}"),
    )

    children_before = _schema_validation_child_pids()
    started = time.monotonic()
    with pytest.raises(ServingError, match="wall-clock deadline"):
        _run_smoke(
            _smoke_spec(thinking=True, constraint={"json": schema}),
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
        lambda **_k: _smoke_response(f"<think>x</think>{answer}"),
    )

    children_before = _schema_validation_child_pids()
    started = time.monotonic()
    with pytest.raises(ServingError, match="wall-clock deadline"):
        _run_smoke(
            _smoke_spec(thinking=True, constraint={"json": schema}),
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
        lambda **_k: _smoke_response(f"<think>x</think>{answer}"),
    )

    with pytest.raises(ServingError, match="non-finite JSON constant"):
        _run_smoke(_smoke_spec(thinking=True, constraint=constraint))


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
            "smoke generation was truncated at the maximum token length",
        ),
    ],
)
def test_thinking_structured_smoke_rejects_invalid_output(
    monkeypatch, constraint, content, finish_reason, match
):
    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **_k: _smoke_response(content, finish_reason),
    )
    with pytest.raises(ServingError, match=match):
        _run_smoke(_smoke_spec(thinking=True, constraint=constraint))


def test_nonthinking_structured_smoke_validates_whole_stripped_content(monkeypatch):
    content = "  <think>literal</think>4  "
    monkeypatch.setattr(serving._app, "serve_chat", lambda **_k: _smoke_response(content))
    out = _run_smoke(
        _smoke_spec(
            thinking=False,
            constraint={"choice": ["<think>literal</think>4"]},
        ),
    )
    assert out["verify_sample"] == "<think>literal</think>4"
