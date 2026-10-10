"""Coverage for control-plane env publishing helpers (`flash.server.domain.envs`) and the serving

These target error / edge branches the existing suite leaves uncovered: git-subprocess failure
translation, archive-extraction guards, publish/slug input validation, and the pure
deployment-lifecycle helpers behind the deploy/chat routes. Everything stays hermetic -- no real
git/network/GPU -- matching the direct-call + monkeypatch style of tests/test_env_publish.py and the
offline conftest.
"""

from __future__ import annotations

import base64
import copy
import io
import json
import multiprocessing
import subprocess
import tarfile
import time
import types
from typing import ClassVar

import pytest

import flash.runner.lifecycle.reporting as runner_reporting
import flash.runner.supervise.transitions as runner_transitions
from flash.server.domain.registry import envs

pytest.importorskip("fastapi")
from fastapi import HTTPException

import flash.server.routes.serving as serving
import flash.server.routes.serving_completion as serving_completion
import flash.server.routes.serving_smoke as serving_smoke
from flash.content import multimodal
from flash.engine.plan.recipe import RECIPE
from flash.serve.contract.errors import ServingError
from flash.serve.request import transport as serving_transport


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
# flash.server.domain.registry.envs
# ===========================================================================


def test_pure_url_and_redact_helpers():
    # the repository url is always credential-free.
    url = envs._repo_url("owner/repo")
    assert url == "https://github.com/owner/repo.git"

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
    # A namespaced name must be exactly three non-empty segments.
    with pytest.raises(envs.EnvPublishError, match="namespace/project/name"):
        envs.publish_slug_for_name("a/b/c/d", {"org_slug": "acme"}, "proj")
    with pytest.raises(envs.EnvPublishError, match="namespace/project/name"):
        envs.publish_slug_for_name("acme/proj/", {"org_slug": "acme"}, "proj")

    # publish_package validates argument TYPES before touching storage.
    monkeypatch.setattr(envs, "_github_publish_once", lambda **_k: None)
    with pytest.raises(envs.EnvPublishError, match="env name must be a string"):
        envs.publish_package(package_b64="x", name=123, key={}, project_slug="proj")  # type: ignore[arg-type]
    with pytest.raises(envs.EnvPublishError, match="base64 string"):
        envs.publish_package(package_b64=123, name="e", key={}, project_slug="proj")  # type: ignore[arg-type]

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
        envs._github_publish(tmp_path, name="e", key={"org_slug": "acme"}, project_slug="proj")
    assert calls["n"] == 1


def test_github_download_wrapper_uses_default_repo(monkeypatch):
    seen: dict[str, object] = {}

    def fake_once(*, repo, token, publish_root):
        seen.update(repo=repo, token=token, publish_root=publish_root)
        return b"package-bytes"

    monkeypatch.setattr(envs, "_github_download_once", fake_once)
    assert envs._github_download("ns/project/env", token="tok") == b"package-bytes"
    assert seen == {
        "repo": envs._DEFAULT_GITHUB_REPO,
        "token": "tok",
        "publish_root": "ns/project/env",
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
        "checkpoint_id": None,
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


def _png_data_uri(image_format: str = "PNG", color: str = "red") -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format=image_format)
    mime = {"PNG": "image/png", "GIF": "image/gif"}[image_format]
    return f"data:{mime};base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _image_block(url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


def _sized_png_data_uri(width: int, height: int) -> str:
    """A real PNG of exact dimensions, for the pixel and decoded-memory limits."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "red").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _oversized_source_data_uri(payload_bytes: int) -> str:
    """A data URI whose source exceeds a byte limit before any decode is attempted.

    The bytes are not a decodable image on purpose: the byte limits must reject on size
    alone, so a test that fed a real image here would not distinguish the byte guard from
    the decoder refusing the result.
    """
    blob = b"\x89PNG\r\n\x1a\n" + b"\x00" * payload_bytes
    return "data:image/png;base64," + base64.b64encode(blob).decode("ascii")


@pytest.mark.parametrize(
    ("messages", "expected_detail"),
    [
        pytest.param(
            [{"role": "user", "content": [_image_block(_png_data_uri()) for _ in range(5)]}],
            "exceeding the 4-image limit",
            id="fifth_image",
        ),
        pytest.param(
            [{"role": "assistant", "content": [_image_block(_png_data_uri())]}],
            "allowed only in user messages",
            id="assistant_image",
        ),
        pytest.param(
            [{"role": "user", "content": [_image_block(_png_data_uri("GIF"))]}],
            "unsupported MIME type",
            id="unsupported_mime",
        ),
        pytest.param(
            [{"role": "user", "content": [_image_block("https://example.com/a.png")]}],
            "remote image URLs are not supported",
            id="remote_url",
        ),
        pytest.param(
            [{"role": "user", "content": [_image_block(_sized_png_data_uri(8192, 8192))]}],
            f"exceeding the {multimodal.MAX_IMAGE_PIXELS}-pixel limit",
            id="per_image_pixels",
        ),
        pytest.param(
            [{"role": "user", "content": [_image_block(_sized_png_data_uri(9000, 8))]}],
            "image dimensions 9000x8 exceed the 8192x8192 limit",
            id="dimensions",
        ),
        pytest.param(
            [
                {
                    "role": "user",
                    "content": [
                        _image_block(
                            _oversized_source_data_uri(multimodal.MAX_IMAGE_SOURCE_BYTES + 1)
                        )
                    ],
                }
            ],
            f"image source exceeds the {multimodal.MAX_IMAGE_SOURCE_BYTES}-byte limit",
            id="per_image_source_bytes",
        ),
        pytest.param(
            # three 6 MiB sources: each is under the per-image cap, their total is not.
            [
                {
                    "role": "user",
                    "content": [
                        _image_block(_oversized_source_data_uri(6 * 1024 * 1024)) for _ in range(3)
                    ],
                }
            ],
            f"exceeding the {multimodal.MAX_TOTAL_IMAGE_SOURCE_BYTES}-byte limit",
            id="aggregate_source_bytes",
        ),
        pytest.param(
            # four 2400x2400 images: each is under the per-image pixel cap, but decoding
            # them cumulatively as RGB needs ~69 MB against a 64 MiB budget.
            [
                {
                    "role": "user",
                    "content": [_image_block(_sized_png_data_uri(2400, 2400)) for _ in range(4)],
                }
            ],
            f"example decoded images exceed the {multimodal.MAX_TOTAL_DECODED_BYTES}-byte limit",
            id="aggregate_decoded_bytes",
        ),
    ],
)
def test_chat_payload_rejects_images_training_would_refuse(messages, expected_detail):
    """serving admits exactly the images training does, so a request cannot bypass the contract.

    without this the chat route forwarded any caller content straight upstream: training enforced
    the count, role, mime, byte, and decoded-memory limits while serving enforced none of them.
    """
    with pytest.raises(HTTPException) as rejected:
        serving._chat_messages_from_payload({"messages": messages})
    assert rejected.value.status_code == 400
    assert expected_detail in rejected.value.detail


def test_chat_payload_forwards_text_only_requests_byte_for_byte():
    """a text-only request goes upstream exactly as the caller wrote it.

    the normalizer rewrites scalar ``content`` into block form; forwarding that rewritten shape
    would change the wire format of every existing text-only request. asserted on a deep copy
    rather than on list identity, since identity also holds if the messages were mutated in place.
    """
    text_only = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
    before = copy.deepcopy(text_only)
    forwarded = serving._chat_messages_from_payload({"messages": text_only})
    assert forwarded == before
    assert text_only == before


def test_chat_payload_forwards_images_in_the_shape_it_validated():
    """an admitted image request reaches the engine as canonical openai ``image_url`` blocks.

    the normalizer accepts sdk spellings the upstream openai-compatible endpoint does not, so
    forwarding the caller's own blocks would admit a request the engine then rejects with a 502.
    validating one representation and forwarding another is the bug this pins.
    """
    uri = _png_data_uri()
    aliased = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "describe"},
                {"type": "input_image", "input_image": uri},
            ],
        }
    ]
    before = copy.deepcopy(aliased)
    forwarded = serving._chat_messages_from_payload({"messages": aliased})
    assert [block["type"] for block in forwarded[0]["content"]] == ["text", "image_url"]
    assert forwarded[0]["content"][1]["image_url"]["url"] == uri
    # the caller's own object is never rewritten in place.
    assert aliased == before


def test_chat_payload_preserves_image_order_when_canonicalizing():
    """four images arrive upstream in the order the caller sent them, with their exact bytes."""
    uris = [_png_data_uri(color=color) for color in ("red", "green", "blue", "yellow")]
    messages = [{"role": "user", "content": [_image_block(uri) for uri in uris]}]
    forwarded = serving._chat_messages_from_payload({"messages": messages})
    assert [block["image_url"]["url"] for block in forwarded[0]["content"]] == uris


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

    # a permanent checkpoint id is mandatory at this boundary.
    with pytest.raises(HTTPException) as missing:
        serving._resolve_deploy_step("run-1", object(), None)
    assert missing.value.status_code == 400
    assert missing.value.detail == "checkpoint_id is required"
    # canonical final and step checkpoint ids resolve.
    assert serving._resolve_deploy_step("run-1", object(), "run-1/final") is None
    assert serving._resolve_deploy_step("run-1", object(), "run-1/step-20") == 20
    assert serving._resolve_deploy_step("run-1", object(), "run-1/step-40") == 40

    # a canonical-but-unknown step is a 404 that lists what is available.
    with pytest.raises(HTTPException) as not_found:
        serving._resolve_deploy_step("run-1", object(), "run-1/step-999")
    assert not_found.value.status_code == 404
    assert "20, 40" in not_found.value.detail

    # Bad step shapes are 400: bool, non-integer float, negative, and junk strings.
    for bad in (True, 20.5, -5, "-5", "abc", "1.5"):
        with pytest.raises(HTTPException) as exc:
            serving._resolve_deploy_step("run-1", object(), bad)
        assert exc.value.status_code == 400, bad


def test_recover_deployments_fails_busy_and_skips_missing(monkeypatch):
    import time

    rows = [
        {"run_id": "r-stale"},
        {"run_id": "r-fresh"},
        {"run_id": "r-held"},
        {"run_id": "r-missing"},
    ]
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
        "r-held": types.SimpleNamespace(
            run_id="r-held",
            state="done",
            deployment={"state": "reconciling", "updated_at": time.time()},
        ),
    }

    def fake_get_status(run_id):
        if run_id == "r-missing":
            raise FileNotFoundError(run_id)
        return statuses[run_id]

    monkeypatch.setattr(serving._app, "get_status", fake_get_status)
    marked: list[tuple[str, dict]] = []
    reported = []

    def mark_failed(run_id, failed):
        marked.append((run_id, failed))
        return types.SimpleNamespace(run_id=run_id, state="done", deployment=failed)

    monkeypatch.setenv("FLASH_DEPLOY_SYNC", "1")
    monkeypatch.setattr(runner_transitions, "mark_deployment_failed", mark_failed)
    monkeypatch.setattr(runner_reporting, "_report_status", reported.append)

    from flash.server.platform.locks import _RunLock

    held_lock = _RunLock("r-held")
    assert held_lock.acquire(blocking=False) is True
    try:
        # r-stale and r-fresh both recover: the lock is the ownership proof, so a busy record whose
        # lock this pass can take has no live lifecycle whatever its timestamp says. r-held is the
        # one that must survive -- its lock is genuinely held, so a live owner still has it.
        assert serving_completion.recover_deployments() == 2
    finally:
        held_lock.release()
    assert sorted(run_id for run_id, _failed in marked) == ["r-fresh", "r-stale"]
    assert all(failed["state"] == "failed" for _run_id, failed in marked)
    assert all("control-plane restart" in failed["error"] for _run_id, failed in marked)
    assert sorted(status.run_id for status in reported) == ["r-fresh", "r-stale"]
    assert all(status.deployment["state"] == "failed" for status in reported)


def test_recover_deployments_rechecks_busy_state_under_lock(monkeypatch):
    statuses = [
        types.SimpleNamespace(run_id="r-settled", deployment={"state": "queued"}),
        types.SimpleNamespace(
            run_id="r-settled",
            deployment={"state": "ready"},
            spec={"run_id": "r-settled", "model": "Qwen/Qwen3.5-9B", "algorithm": "sft"},
        ),
    ]
    monkeypatch.setattr(serving.db, "all_runs", lambda: [{"run_id": "r-settled"}])
    monkeypatch.setattr(serving._app, "get_status", lambda _run_id: statuses.pop(0))
    monkeypatch.setattr(
        runner_transitions,
        "mark_deployment_failed",
        lambda *_args: pytest.fail("a deployment that settled under the lock must not be failed"),
    )

    assert serving_completion.recover_deployments() == 0
    assert statuses == []


def test_recover_deployments_retires_a_ready_deployment_this_build_cannot_serve(monkeypatch):
    # a run accepted under an algorithm this build has since dropped keeps a `ready`
    # deployment record. every serving route parses the persisted spec before inference, so
    # chat raises there instead of answering -- while `/v1/deployments` still lists the record
    # as active, and only BUSY states were recovered at startup. the record therefore survived
    # every restart as a deployment that looks live and can never respond.
    rows = [{"run_id": "r-retired"}, {"run_id": "r-servable"}]
    monkeypatch.setattr(serving.db, "all_runs", lambda: rows)

    project = "11111111-1111-4111-8111-111111111111"
    statuses = {
        "r-retired": types.SimpleNamespace(
            run_id="r-retired",
            state="done",
            deployment={"state": "ready"},
            spec={
                "run_id": "r-retired",
                "model": "Qwen/Qwen3.5-9B",
                "algorithm": "opsd",
                "project": project,
            },
        ),
        # same ready state, a spec this build still parses: must be left serving, so the check
        # cannot be passing merely because it fails every ready record it sees.
        "r-servable": types.SimpleNamespace(
            run_id="r-servable",
            state="done",
            deployment={"state": "ready"},
            spec={
                "run_id": "r-servable",
                "model": "Qwen/Qwen3.5-9B",
                "algorithm": "sft",
                "project": project,
            },
        ),
    }
    monkeypatch.setattr(serving._app, "get_status", lambda run_id: statuses[run_id])
    marked: list[tuple[str, dict]] = []

    def mark_failed(run_id, failed):
        marked.append((run_id, failed))
        return types.SimpleNamespace(run_id=run_id, state="done", deployment=failed)

    monkeypatch.setenv("FLASH_DEPLOY_SYNC", "1")
    monkeypatch.setattr(runner_transitions, "mark_deployment_failed", mark_failed)
    monkeypatch.setattr(runner_reporting, "_report_status", lambda status: None)

    assert serving_completion.recover_deployments() == 1
    assert [run_id for run_id, _failed in marked] == ["r-retired"]
    for _run_id, failed in marked:
        assert failed["state"] == "failed"
        # the reason names the actual cause, not the restart the busy branch reports.
        assert "no longer supported" in failed["error"]


def test_recover_deployments_reports_restored_ready_predecessor(monkeypatch):
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
    monkeypatch.setattr(runner_transitions, "mark_deployment_failed", mark_failed)
    monkeypatch.setattr(runner_reporting, "_report_status", reported.append)

    assert serving_completion.recover_deployments() == 1
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
    model: str = "some-org/text-only-model",
):
    return types.SimpleNamespace(
        model=model,
        algorithm=algorithm,
        thinking=thinking,
        train=types.SimpleNamespace(
            max_completion_tokens=max_completion_tokens,
            max_context_tokens=max_context_tokens,
            structured_outputs="" if constraint is None else json.dumps(constraint),
            stop_sequences=stop_sequences,
        ),
    )


_SMOKE_REVISION = "run-1/final"
_MISSING_REASONING = object()


def _smoke_response(
    content: str,
    finish_reason: str = "stop",
    *,
    reasoning_content: object = _MISSING_REASONING,
    request_adapter: str | None = _SMOKE_REVISION,
) -> dict:
    message = {"content": content}
    if reasoning_content is not _MISSING_REASONING:
        message["reasoning_content"] = reasoning_content
    response = {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "freesolo": {"checkpoint_id": _SMOKE_REVISION},
        "_freesolo_headers": {"checkpoint_id": _SMOKE_REVISION},
    }
    if request_adapter is not None:
        response["_freesolo_lora_request_adapter"] = request_adapter
    return response


def _smoke_expected_colour(run_id: str = "run-1") -> str:
    """The colour this run_id must answer, derived the same way production derives it.

    hardcoding a colour here would silently re-introduce the guessable smoke: the point of the
    per-run choice is that no single colour is always right, so the test must not know one either.
    """
    expected, _messages = serving_smoke._smoke_image_challenge(run_id)
    return expected


def _run_smoke(spec, *, budget_s: float = 600.0, advertised=None, adapter_targets_images=None):
    return serving._run_deployment_smoke(
        "run-1",
        spec,
        serving_model=_SMOKE_REVISION,
        expected_checkpoint=_SMOKE_REVISION,
        org_id="org-1",
        advertised_capabilities=advertised,
        adapter_targets_images=adapter_targets_images,
        budget_s=budget_s,
    )


_ATTESTING = frozenset({serving_smoke.LORA_REQUEST_ATTESTATION_CAPABILITY})


def _schema_validation_child_pids() -> set[int | None]:
    return {
        child.pid
        for child in multiprocessing.active_children()
        if child.name == serving._JSON_SCHEMA_PROCESS_NAME
    }


def test_real_image_capable_model_with_text_adapter_uses_fixed_prompt(monkeypatch):
    class UntrustedEnvironment:
        def __getattribute__(self, name):
            pytest.fail(f"control-plane smoke accessed user environment field {name!r}")

    spec = _smoke_spec(thinking=False, model="Qwen/Qwen3.5-9B")
    spec.environment = UntrustedEnvironment()
    calls = []

    def fake_serve_chat(**kwargs):
        calls.append(kwargs)
        return _smoke_response("The answer is 4")

    monkeypatch.setattr(serving._app, "serve_chat", fake_serve_chat)
    out = _run_smoke(spec, budget_s=10.0, adapter_targets_images=False)

    assert out["verify_kind"] == "fixed_prompt"
    assert out["verify_turns"] == 1
    assert calls[0]["messages"] == [{"role": "user", "content": serving._SMOKE_PROMPT}]
    assert calls[0]["expected_checkpoint"] == _SMOKE_REVISION
    assert calls[0]["timeout_s"] <= 10.0
    assert calls[0]["retry_unavailable"] is True


def test_image_deployment_smoke_uses_valid_trusted_image_without_persisting_it(monkeypatch):
    import base64
    import io

    from PIL import Image

    expected_colour, expected_messages = serving_smoke._smoke_image_challenge("run-1")
    # EVERY variant must be a true solid square, not just the one this run_id happens to draw.
    # checking only the drawn variant leaves the others free to rot, and a variant whose pixels
    # did not match its announced colour would fail a correctly-seeing model.
    rgb_for = {"RED": (255, 0, 0), "BLUE": (0, 0, 255), "GREEN": (0, 128, 0)}
    for colour, data_uri in serving_smoke._SMOKE_IMAGE_VARIANTS:
        image = Image.open(io.BytesIO(base64.b64decode(data_uri.split(",", 1)[1], validate=True)))
        image.load()
        assert image.size == (32, 32)
        pixels = image.convert("RGB").tobytes()
        distinct = {pixels[i : i + 3] for i in range(0, len(pixels), 3)}
        assert distinct == {bytes(rgb_for[colour])}, f"{colour} variant is not a solid square"
        image.close()
    # the prompt must not name any candidate colour, or the answer could be read off the question.
    for colour, _uri in serving_smoke._SMOKE_IMAGE_VARIANTS:
        assert colour.lower() not in serving_smoke._SMOKE_IMAGE_PROMPT.lower()
    calls = []

    def fake_serve_chat(**kwargs):
        calls.append(kwargs)
        return _smoke_response(expected_colour)

    monkeypatch.setattr(serving._app, "serve_chat", fake_serve_chat)
    out = _run_smoke(
        _smoke_spec(thinking=False, model="Qwen/Qwen3.5-9B"),
        adapter_targets_images=True,
    )

    assert out["verify_kind"] == "fixed_image"
    assert out["verify_turns"] == 1
    assert out["verify_lora_request_adapter"] == _SMOKE_REVISION
    assert calls[0]["messages"] == expected_messages
    assert calls[0]["expected_checkpoint"] == _SMOKE_REVISION
    for _colour, data_uri in serving_smoke._SMOKE_IMAGE_VARIANTS:
        assert data_uri not in json.dumps(out)


def test_smoke_uses_the_capability_set_deploy_gated_on_not_a_second_healthz(monkeypatch):
    """The advertised set is handed down, so a rolling serving deploy cannot fail this open.

    ``deploy_adapter`` captures capabilities once before registration. Re-reading /healthz inside
    the smoke would let a mid-rollout replica that does not advertise the attestation -- or one
    transient failure -- accept a response that omits a header this deployment WAS promised.
    """
    from flash.serve.deployment import deploy as deploy_mod

    healthz_calls = 0

    def exploding_capabilities(**_kwargs):
        nonlocal healthz_calls
        healthz_calls += 1
        raise RuntimeError("a second /healthz must never be issued from the smoke")

    monkeypatch.setattr(deploy_mod, "_require_serving_capabilities", exploding_capabilities)
    response = _smoke_response(_smoke_expected_colour(), request_adapter=None)
    monkeypatch.setattr(serving._app, "serve_chat", lambda **_kwargs: response)

    # the handed-down set still enforces the contract strictly...
    with pytest.raises(ServingError, match="omitted LoRA request adapter attestation"):
        _run_smoke(
            _smoke_spec(thinking=False, model="Qwen/Qwen3.5-9B"),
            advertised=_ATTESTING,
            adapter_targets_images=True,
        )
    # ...and a backend that never claimed the header still degrades rather than failing.
    out = _run_smoke(
        _smoke_spec(thinking=False, model="Qwen/Qwen3.5-9B"),
        advertised=frozenset(),
        adapter_targets_images=True,
    )
    assert out["verify_kind"] == "fixed_image"
    assert healthz_calls == 0, "the smoke re-fetched capabilities instead of using deploy's set"


def test_lora_attestation_predicate_reads_only_the_handed_down_set():
    """No captured set means not advertised; it must never fall back to a live lookup."""
    assert serving_smoke._lora_attestation_advertised(_ATTESTING) is True
    assert serving_smoke._lora_attestation_advertised(frozenset()) is False
    assert serving_smoke._lora_attestation_advertised(None) is False


def test_image_deployment_smoke_rejects_missing_lora_request_attestation(monkeypatch):
    """A backend that ADVERTISES the attestation must actually send it."""
    response = _smoke_response(_smoke_expected_colour(), request_adapter=None)
    monkeypatch.setattr(serving._app, "serve_chat", lambda **_kwargs: response)
    with pytest.raises(ServingError, match="omitted LoRA request adapter attestation"):
        _run_smoke(
            _smoke_spec(thinking=False, model="Qwen/Qwen3.5-9B"),
            advertised=_ATTESTING,
            adapter_targets_images=True,
        )


def test_image_deployment_smoke_allows_missing_attestation_when_not_advertised(monkeypatch):
    """The header is emitted by the serving image, not by the run.

    Demanding it from a backend that never claimed to produce it failed every deployment
    org-wide (187 failed / 1 ready) while proving nothing about the adapter under test. This is
    the same shape of bug ``REVISION_PROVENANCE_CAPABILITY`` already had to fix once.
    """
    response = _smoke_response(_smoke_expected_colour(), request_adapter=None)
    monkeypatch.setattr(serving._app, "serve_chat", lambda **_kwargs: response)
    out = _run_smoke(
        _smoke_spec(thinking=False, model="Qwen/Qwen3.5-9B"),
        advertised=frozenset(),
        adapter_targets_images=True,
    )
    assert out["verify_kind"] == "fixed_image"


def test_image_deployment_smoke_rejects_wrong_adapter_even_when_not_advertised(monkeypatch):
    """Degrading on ABSENCE must not degrade on a WRONG identity.

    If the backend volunteers an adapter id at all, a mismatch means some other LoRA answered,
    which stays a hard failure whatever the backend advertises.
    """
    response = _smoke_response(_smoke_expected_colour(), request_adapter="run-1/step-20")
    monkeypatch.setattr(serving._app, "serve_chat", lambda **_kwargs: response)
    with pytest.raises(ServingError, match="wrong LoRA request adapter"):
        _run_smoke(
            _smoke_spec(thinking=False, model="Qwen/Qwen3.5-9B"),
            advertised=frozenset(),
            adapter_targets_images=True,
        )


def test_image_deployment_smoke_rejects_mismatched_lora_request_adapter(monkeypatch):
    response = _smoke_response(_smoke_expected_colour(), request_adapter="run-1/step-20")
    monkeypatch.setattr(serving._app, "serve_chat", lambda **_kwargs: response)

    with pytest.raises(ServingError, match="wrong LoRA request adapter"):
        _run_smoke(
            _smoke_spec(thinking=False, model="Qwen/Qwen3.5-9B"),
            adapter_targets_images=True,
        )


def test_image_deployment_smoke_still_rejects_wrong_provenance(monkeypatch):
    response = _smoke_response(_smoke_expected_colour())
    response["freesolo"]["checkpoint_id"] = "run-1/step-20"
    monkeypatch.setattr(serving._app, "serve_chat", lambda **_kwargs: response)

    with pytest.raises(ServingError, match="wrong checkpoint"):
        _run_smoke(
            _smoke_spec(thinking=False, model="Qwen/Qwen3.5-9B"),
            adapter_targets_images=True,
        )


@pytest.mark.parametrize(
    "answer",
    ["a red-and-blue checkerboard", "I cannot see an image.", "purple"],
)
def test_image_deployment_smoke_rejects_an_answer_that_is_not_the_shown_square(monkeypatch, answer):
    """The colour assertion is what makes this smoke image-dependent, so pin it.

    a deployment whose vision path is broken -- wrong processor, dropped media, placeholders that
    never expanded -- still answers *something*, and without this the suite would accept it as
    verified. the sibling test proves the prompt names no candidate colour, so the only way to
    answer is to decode the image.
    """
    monkeypatch.setattr(serving._app, "serve_chat", lambda **_kwargs: _smoke_response(answer))

    with pytest.raises(ServingError, match="did not identify the trusted"):
        _run_smoke(
            _smoke_spec(thinking=False, model="Qwen/Qwen3.5-9B"),
            adapter_targets_images=True,
        )


def test_image_smoke_colour_actually_varies_across_deployments():
    """The challenge must not collapse to one colour, or it is guessable again.

    every other test here derives the expected colour from production, so they all stay
    self-consistent if the selection is hardcoded to a single variant -- the exact bug this gate
    exists to prevent would pass unnoticed. this asserts the property directly: over many run ids
    at least two distinct colours must appear, and each must be a genuine solid square.
    """
    seen = {serving_smoke._smoke_image_challenge(f"run-{i}")[0] for i in range(60)}
    assert len(seen) >= 2, f"smoke colour never varies (always {seen}) - it is guessable"
    assert seen <= {c for c, _uri in serving_smoke._SMOKE_IMAGE_VARIANTS}

    # and the message must always carry the variant matching the announced colour, so a mismatched
    # pairing cannot make a correct model look wrong.
    by_colour = dict(serving_smoke._SMOKE_IMAGE_VARIANTS)
    for i in range(60):
        colour, messages = serving_smoke._smoke_image_challenge(f"run-{i}")
        assert messages[0]["content"][1]["image_url"]["url"] == by_colour[colour]


def test_image_deployment_smoke_rejects_the_other_trusted_colours(monkeypatch):
    """Answering a colour that is real, but not the one THIS run was shown, must fail.

    this is the check a single fixed colour could never make: it proves the smoke reads the
    deployment's own challenge rather than accepting any plausible colour word. a model guessing
    "red" because that is the obvious answer to "what colour is the square" is wrong here whenever
    the run drew blue or green.
    """
    expected = _smoke_expected_colour()
    others = [c for c, _uri in serving_smoke._SMOKE_IMAGE_VARIANTS if c != expected]
    assert others, "the challenge needs more than one colour or it is guessable"

    for wrong in others:
        response = _smoke_response(wrong)
        # bind the response per iteration: a bare closure over the loop variable would read
        # whatever the LAST iteration assigned, so every case would assert the same colour.
        monkeypatch.setattr(serving._app, "serve_chat", lambda _r=response, **_kwargs: _r)
        with pytest.raises(ServingError, match="did not identify the trusted"):
            _run_smoke(
                _smoke_spec(thinking=False, model="Qwen/Qwen3.5-9B"),
                adapter_targets_images=True,
            )


def test_unavailable_adapter_modality_defaults_to_nonblocking_text_smoke(monkeypatch):
    """Unknown modality must weaken verification instead of stranding a working deployment."""
    calls = []

    def fake_serve_chat(**kwargs):
        calls.append(kwargs)
        return _smoke_response("The answer is 4")

    monkeypatch.setattr(serving._app, "serve_chat", fake_serve_chat)

    out = _run_smoke(_smoke_spec(thinking=False, model="Qwen/Qwen3.5-9B"))

    assert out["verify_kind"] == "fixed_prompt"
    assert calls[0]["messages"] == [{"role": "user", "content": serving._SMOKE_PROMPT}]


def test_image_deployment_smoke_keeps_structured_validation_as_a_separate_call(monkeypatch):
    calls = []
    validated = []
    validate_structured_smoke = serving_smoke._validate_structured_smoke

    def fake_serve_chat(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            assert kwargs["structured_outputs"] == {}
            return _smoke_response(_smoke_expected_colour())
        assert "structured_outputs" not in kwargs
        return _smoke_response('{"answer":"4"}')

    def capture_validation(answer, constraint, **kwargs):
        validated.append((answer, constraint))
        return validate_structured_smoke(answer, constraint, **kwargs)

    monkeypatch.setattr(serving._app, "serve_chat", fake_serve_chat)
    monkeypatch.setattr(serving_smoke, "_validate_structured_smoke", capture_validation)
    out = _run_smoke(
        _smoke_spec(
            thinking=False,
            model="Qwen/Qwen3.5-9B",
            constraint={"json_object": True},
        ),
        adapter_targets_images=True,
    )

    assert out["verify_kind"] == "fixed_image"
    assert out["verify_turns"] == 2
    assert calls[0]["messages"] == serving_smoke._smoke_image_challenge("run-1")[1]
    assert calls[1]["messages"] == [{"role": "user", "content": serving._SMOKE_PROMPT}]
    assert validated == [('{"answer":"4"}', {"json_object": True})]


@pytest.mark.parametrize(
    ("request_adapter", "error", "advertised"),
    [
        # absence only fails when the backend claimed it would send it
        (None, "omitted LoRA request adapter attestation", True),
        # a WRONG identity fails either way -- some other LoRA answered
        ("run-1@final." + "b" * 40, "wrong LoRA request adapter", True),
        ("run-1@final." + "b" * 40, "wrong LoRA request adapter", False),
    ],
)
def test_image_structured_smoke_requires_attestation_on_second_request(
    monkeypatch, request_adapter, error, advertised
):
    calls = 0

    def fake_serve_chat(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _smoke_response(_smoke_expected_colour())
        return _smoke_response('{"answer":"4"}', request_adapter=request_adapter)

    monkeypatch.setattr(serving._app, "serve_chat", fake_serve_chat)

    with pytest.raises(ServingError, match=error):
        _run_smoke(
            _smoke_spec(
                thinking=False,
                model="Qwen/Qwen3.5-9B",
                constraint={"json_object": True},
            ),
            advertised=_ATTESTING if advertised else frozenset(),
            adapter_targets_images=True,
        )

    assert calls == 2


def test_run_deployment_smoke_uses_thinking_completion_budget(monkeypatch):
    calls = []

    def fake_serve_chat(**kwargs):
        calls.append(kwargs)
        return _smoke_response('<think>reasoning</think>{"answer":"4"}')

    monkeypatch.setattr(serving._app, "serve_chat", fake_serve_chat)
    _run_smoke(_smoke_spec(thinking=True, constraint={"json_object": True}))

    assert calls[0]["max_tokens"] == 1536


def test_run_deployment_smoke_does_not_cap_a_constrained_request(monkeypatch):
    """End to end: a grammar reaches serving as the adapter default, so the request keeps the run's
    explicit budget rather than the smoke ceiling. Capping it would truncate a long-but-legal
    constrained answer and reject a working adapter."""
    calls = []

    def fake_serve_chat(**kwargs):
        calls.append(kwargs)
        return _smoke_response('<think>reasoning</think>{"answer":"4"}')

    monkeypatch.setattr(serving._app, "serve_chat", fake_serve_chat)
    _run_smoke(
        _smoke_spec(
            thinking=True,
            algorithm="grpo",
            constraint={"json_object": True},
            max_completion_tokens=8192,
        )
    )

    assert calls[0]["max_tokens"] == 8192


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
        # opd without grammar must still get a thinking-sized budget, bounded by the smoke ceiling.
        (_smoke_spec(algorithm="opd", thinking=True, max_completion_tokens=8192), 2048),
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
        return _smoke_response(
            "<think>2+2 is 4</think>The answer is 4",
            reasoning_content="2+2 is 4",
        )

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
    <think> block, so the empty-content check passes too. Without a post-</think> answer requirement
    on every thinking smoke, that checkpoint activates and then answers nothing on real requests.
    """
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
        return _smoke_response(
            "<think>2+2 is 4</think>The answer is 4",
            reasoning_content="2+2 is 4",
        )

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
    from flash.serve.deployment import deploy as _deploy

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

    monkeypatch.setattr(
        serving_transport, "serving_openai_base_url", lambda: "https://serving.example/v1"
    )
    monkeypatch.setattr(serving_transport, "_internal_key_header", dict)
    monkeypatch.setattr(serving_transport, "_chat_http_client", _Client)

    _deploy.chat(
        "run-1/final",
        [{"role": "user", "content": "hi"}],
        org_id="org-1",
        stop=["</answer>"],
    )
    assert sent["stop"] == ["</answer>"]

    sent.clear()
    _deploy.chat("run-1/final", [{"role": "user", "content": "hi"}], org_id="org-1")
    assert "stop" not in sent


def test_chat_captures_lora_request_adapter_attestation_for_smoke(monkeypatch):
    from flash.serve.deployment import deploy as _deploy

    class _Resp:
        status_code = 200
        headers: ClassVar[dict[str, str]] = {
            "X-Freesolo-Checkpoint": _SMOKE_REVISION,
            "X-Freesolo-LoRA-Request-Adapter": _SMOKE_REVISION,
        }

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {"content": _smoke_expected_colour()},
                        "finish_reason": "stop",
                    }
                ]
            }

    class _Client:
        def post(self, url, json=None, headers=None, timeout=None):
            return _Resp()

    monkeypatch.setattr(
        serving_transport, "serving_openai_base_url", lambda: "https://serving.example/v1"
    )
    monkeypatch.setattr(serving_transport, "_internal_key_header", dict)
    monkeypatch.setattr(serving_transport, "_chat_http_client", _Client)

    out = _deploy.chat(
        _SMOKE_REVISION,
        [{"role": "user", "content": "hi"}],
        org_id="org-1",
        expected_checkpoint=_SMOKE_REVISION,
    )

    assert out["choices"][0]["message"]["content"] == _smoke_expected_colour()
    assert out["_freesolo_lora_request_adapter"] == _SMOKE_REVISION


def test_zero_completion_budget_resolves_to_thinking_recipe_default():
    from flash.serve.deployment.preflight import resolve_effective_completion_tokens

    spec = _smoke_spec(
        thinking=True,
        constraint={"json_object": True},
        max_completion_tokens=0,
    )

    assert resolve_effective_completion_tokens(spec) == 1536


def test_thinking_sft_smoke_budget_comes_from_the_sft_recipe_not_the_rl_default():
    from flash.serve.deployment.preflight import resolve_smoke_completion_tokens

    spec = _smoke_spec(thinking=True, algorithm="sft")

    # the rl thinking default (1536) is shorter than what sft actually trains, so resolving to it
    # would truncate the smoke and reject a checkpoint that answered correctly.
    assert RECIPE.rl.max_completion_len_thinking < RECIPE.sft.max_seq_len_thinking
    assert resolve_smoke_completion_tokens(spec) == RECIPE.sft.max_seq_len_thinking


def test_sft_smoke_budget_follows_an_explicit_context_over_the_recipe_default():
    from flash.serve.deployment.preflight import resolve_smoke_completion_tokens

    # the worker bounds the packed block by max_context_tokens and only falls back to the recipe
    # when it is unset (flash/engine/worker/entry/sft.py), so below the ceiling the smoke resolves
    # the same way: sizing a 512-context run at the 2048 recipe default would over-allocate.
    short = _smoke_spec(thinking=True, algorithm="sft", max_context_tokens=512)
    assert resolve_smoke_completion_tokens(short) == 512

    # non-thinking takes the same path.
    assert (
        resolve_smoke_completion_tokens(
            _smoke_spec(thinking=False, algorithm="sft", max_context_tokens=1024)
        )
        == 1024
    )

    # a non-positive value is not a budget, so the recipe default still applies.
    assert (
        resolve_smoke_completion_tokens(
            _smoke_spec(thinking=True, algorithm="sft", max_context_tokens=0)
        )
        == RECIPE.sft.max_seq_len_thinking
    )


def test_smoke_budget_is_capped_independently_of_the_training_context():
    """The smoke asks one fixed trivial question, so the run's training context must not size it.

    Inheriting that number spent the smoke's 600s wall clock -- which also has to cover cold-starting
    the base model and loading the adapter -- generating tokens nobody reads: a thinking sft run at
    an 8192 context asked for 8192 tokens and the deployment died on deployment_smoke_timeout. It
    also coupled the knobs backwards, since raising max_context_tokens to avoid training truncation
    made the run HARDER to deploy.
    """
    from flash.serve.deployment.preflight import (
        SMOKE_COMPLETION_TOKEN_CEILING,
        resolve_smoke_completion_tokens,
    )

    assert SMOKE_COMPLETION_TOKEN_CEILING < 8192
    for spec in (
        _smoke_spec(thinking=True, algorithm="sft", max_context_tokens=8192),
        _smoke_spec(thinking=True, algorithm="opd", max_completion_tokens=8192),
        _smoke_spec(thinking=True, algorithm="grpo", max_completion_tokens=8192),
    ):
        assert resolve_smoke_completion_tokens(spec) == SMOKE_COMPLETION_TOKEN_CEILING

    # the ceiling only ever lowers the request: every default-configured run still smokes at exactly
    # what it does today, so capping cannot truncate a checkpoint that passes now.
    assert RECIPE.sft.max_seq_len_thinking <= SMOKE_COMPLETION_TOKEN_CEILING
    assert RECIPE.rl.max_completion_len_thinking <= SMOKE_COMPLETION_TOKEN_CEILING
    assert RECIPE.opd.max_completion_len_thinking <= SMOKE_COMPLETION_TOKEN_CEILING


def test_a_configured_grammar_keeps_the_runs_own_budget():
    """A grammar is the adapter's serving default, so the smoke generates under it too.

    The shortest string a constraint admits can exceed the ceiling -- a long `choice`, a
    fixed-repetition `regex`, a schema with a large `minLength`. Capping there truncates the only
    legal answer, `finish_reason="length"` fails the truncation guard, and an adapter that serves
    correctly becomes undeployable. That case passes today on an explicit budget, so the ceiling
    must not reach it.
    """
    from flash.serve.deployment.preflight import (
        SMOKE_COMPLETION_TOKEN_CEILING,
        resolve_smoke_completion_tokens,
    )

    for algorithm in ("grpo", "opd"):
        spec = _smoke_spec(
            thinking=True,
            algorithm=algorithm,
            constraint={"choice": ["a" * 20000]},
            max_completion_tokens=8192,
        )
        # unconstrained the same run is capped; the grammar is the only difference.
        assert resolve_smoke_completion_tokens(spec) == SMOKE_COMPLETION_TOKEN_CEILING
        assert resolve_smoke_completion_tokens(spec, constrained=True) == 8192


def test_nonthinking_sft_smoke_budget_comes_from_the_sft_recipe_not_the_rl_default():
    from flash.serve.deployment.preflight import resolve_smoke_completion_tokens

    spec = _smoke_spec(thinking=False, algorithm="sft")

    assert resolve_smoke_completion_tokens(spec) == RECIPE.sft.max_seq_len


def test_sft_contributes_no_completion_budget_to_the_serving_context_guard():
    from flash.serve.deployment.preflight import resolve_effective_completion_tokens

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
    from flash.engine.plan.recipe import RECIPE
    from flash.serve.deployment.preflight import resolve_effective_completion_tokens

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


def test_run_deployment_smoke_retries_capacity_unavailable_within_deadline(monkeypatch):
    calls = []
    sleeps = []

    def fake_serve_chat(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise serving.RetryableServingUnavailable("serving_capacity_unavailable", 1.0)
        return _smoke_response("The answer is 4")

    monkeypatch.setattr(serving._app, "serve_chat", fake_serve_chat)
    monkeypatch.setattr(serving.time, "sleep", sleeps.append)

    out = _run_smoke(_smoke_spec(thinking=False), budget_s=3.0)

    assert out["verify_sample"] == "The answer is 4"
    assert sleeps == [1.0]
    assert len(calls) == 2
    assert all(0 < call["timeout_s"] <= 3.0 for call in calls)


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

    # a thinking adapter stopped mid-reasoning has no answer. ``finish_reason="stop"`` bypasses the
    # truncation guard, so the closing-tag requirement must apply without structured outputs.
    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **_k: _smoke_response("<think>still reasoning"),
    )
    with pytest.raises(ServingError, match="never closed its reasoning"):
        _run_smoke(_smoke_spec(thinking=True, model="Qwen/Qwen3.5-9B"))

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
    (flash/serve/deployment/deploy.py::_balanced_thinking_content). `_thinking_answer` asserts an answer exists
    for every thinking smoke; before it did, only the grammar-constrained path checked, so an
    unconstrained deployment could go live on a smoke that produced no answer at all.
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
    </think>`. The fold cannot decide this: those bytes are identical to an adapter whose answer
    genuinely is the tag, and it also backs the public chat route where the text must survive.
    """
    monkeypatch.setattr(
        serving._app,
        "serve_chat",
        lambda **_k: _smoke_response("<think>why</think></think>"),
    )
    with pytest.raises(ServingError, match="only a close tag"):
        _run_smoke(_smoke_spec(thinking=True))


def test_thinking_smoke_accepts_a_tagless_answer_only_for_an_uncataloged_model(monkeypatch):
    """A model whose template flash cannot verify must not be undeployable for omitting the tag.

    `flash.schema` warns and proceeds when the catalog reports `thinking == "unknown"`, which only
    the open-model policy produces. Such a run may answer with no `<think>` block at all, so the
    strict requirement would reject a correct answer and strand the trained adapter.
    """
    monkeypatch.setattr(serving._app, "serve_chat", lambda **_k: _smoke_response("4"))

    strict = _smoke_spec(thinking=True, model="Qwen/Qwen3.5-9B")
    with pytest.raises(ServingError, match="never closed its reasoning"):
        _run_smoke(strict)

    # a model the catalog does not vouch for cannot be held to the tag. submit rejects these, so
    # only a stale caller reaches here -- but the smoke must not invent a guarantee for it.
    unknown = _smoke_spec(thinking=True)
    unknown.model = "some-org/not-in-the-catalog"
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
    responses = iter(
        [
            _smoke_response(f"<think>vision</think>{_smoke_expected_colour()}"),
            _smoke_response(content, finish_reason),
        ]
    )
    monkeypatch.setattr(serving._app, "serve_chat", lambda **_k: next(responses))
    with pytest.raises(ServingError, match=match):
        _run_smoke(
            _smoke_spec(
                thinking=True,
                constraint=constraint,
                model="Qwen/Qwen3.5-9B",
            ),
            adapter_targets_images=True,
        )


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
