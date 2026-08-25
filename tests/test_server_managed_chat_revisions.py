from __future__ import annotations

import time

import pytest

from tests._helpers.chat_provenance import (
    managed_chat_result as _managed_chat_result,
)
from tests._helpers.chat_provenance import (
    managed_stream_headers as _managed_stream_headers,
)
from tests._helpers.managed_chat import _RawManagedChatResponse
from tests.test_server_api import SPEC, _bearer, _login, _make_run

pytest_plugins = ("tests._helpers.server_api_plugin",)


def test_chat_step_selector_prefers_current_revision_for_redeployed_step(api, monkeypatch):
    import flash.runner.lifecycle.state as runner_state
    import flash.runner.lifecycle.status as runner_status
    import flash.runner.results.verified_revisions as runner_verified_revisions
    import flash.runner.supervise.transitions as runner_transitions
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    runner_state._save_status(status)
    revisions = [f"{run_id}@step-20." + "a" * 40, f"{run_id}@step-20." + "b" * 40]
    for revision in revisions:
        runner_transitions.mark_checkpoint_deployed(
            run_id,
            {
                "state": "ready",
                "endpoint_name": "https://serve.example",
                "adapter_revision": revision,
            },
            verification_generation=runner_verified_revisions.verified_adapter_revision_generation(
                run_id
            ),
        )
    assert runner_status.get_status(run_id).deployment["adapter_revision"] == revisions[1]
    seen = {}

    def fake_chat(**kwargs):
        seen.update(kwargs)
        return _managed_chat_result(kwargs["run_id"])

    monkeypatch.setattr(app_mod, "serve_chat", fake_chat)

    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={"messages": [{"role": "user", "content": "hello"}], "step": 20},
        headers=_bearer(key),
    )

    assert response.status_code == 200, response.text
    assert seen["run_id"] == revisions[1]


def test_chat_step_selector_rejects_multiple_verified_revisions(api, monkeypatch):
    import flash.runner.results.verified_revisions as runner_verified_revisions
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = _make_run(api, key, "done")
    revisions = [f"{run_id}@step-20." + "a" * 40, f"{run_id}@step-20." + "c" * 40]
    for revision in revisions:
        generation = runner_verified_revisions.verified_adapter_revision_generation(run_id)
        assert runner_verified_revisions.add_verified_adapter_revision(
            run_id,
            revision,
            expected_generation=generation,
        )
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **kwargs: pytest.fail("an ambiguous step must not reach serving"),
    )

    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={"messages": [{"role": "user", "content": "hello"}], "step": 20},
        headers=_bearer(key),
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "multiple verified revisions at step 20" in detail
    assert "flash models deployments" in detail


def test_chat_rejects_missing_messages_before_invalid_explicit_revision(api, monkeypatch):
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = _make_run(api, key, "done")
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **_kwargs: pytest.fail("an invalid request must not reach serving"),
    )

    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={"adapter_revision": "not-an-immutable-revision"},
        headers=_bearer(key),
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "messages must be a nonempty array of objects"


def test_chat_rejects_malformed_messages_before_ambiguous_step(api, monkeypatch):
    import flash.runner.results.verified_revisions as runner_verified_revisions
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = _make_run(api, key, "done")
    revisions = [f"{run_id}@step-20." + "a" * 40, f"{run_id}@step-20." + "c" * 40]
    for revision in revisions:
        generation = runner_verified_revisions.verified_adapter_revision_generation(run_id)
        assert runner_verified_revisions.add_verified_adapter_revision(
            run_id,
            revision,
            expected_generation=generation,
        )
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **_kwargs: pytest.fail("an invalid request must not reach serving"),
    )

    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={"messages": "not-a-message-list", "step": 20},
        headers=_bearer(key),
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "messages must be a nonempty array of objects"


def test_chat_step_selector_requires_a_verified_deployment(api, monkeypatch):
    import flash.runner.results.verified_revisions as runner_verified_revisions
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = _make_run(api, key, "done")
    revisions = [f"{run_id}@step-20." + "a" * 40, f"{run_id}@final." + "b" * 40]
    for revision in revisions:
        generation = runner_verified_revisions.verified_adapter_revision_generation(run_id)
        assert runner_verified_revisions.add_verified_adapter_revision(
            run_id,
            revision,
            expected_generation=generation,
        )
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **kwargs: pytest.fail("an unverified step must not reach serving"),
    )

    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={"messages": [{"role": "user", "content": "hello"}], "step": 40},
        headers=_bearer(key),
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "deploy it first" in detail
    assert f"flash models deploy {run_id}/step-40" in detail
    assert "currently deployed steps: 20, final" in detail


def test_chat_rejects_adapter_revision_and_step_together(api, monkeypatch):
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = _make_run(api, key, "done")
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **kwargs: pytest.fail("an ambiguous target must not reach serving"),
    )

    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "adapter_revision": f"{run_id}@step-20." + "a" * 40,
            "step": 20,
        },
        headers=_bearer(key),
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "pass either adapter_revision or step, not both"


@pytest.mark.parametrize("step", ["abc", 1.5, -1, True])
def test_chat_rejects_invalid_step_selector(api, monkeypatch, step):
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = _make_run(api, key, "done")
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **kwargs: pytest.fail("an invalid step must not reach serving"),
    )

    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={"messages": [{"role": "user", "content": "hello"}], "step": step},
        headers=_bearer(key),
    )

    assert response.status_code == 400
    assert "invalid checkpoint step" in response.json()["detail"]


def test_chat_ready_record_without_ledger_membership_rejects_revision(api, monkeypatch):
    import flash.runner.lifecycle.state as runner_state
    import flash.runner.lifecycle.status as runner_status
    import flash.runner.results.verified_revisions as runner_verified_revisions
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    revision = f"{run_id}@final." + "b" * 40
    status = runner_status.get_status(run_id)
    status.state = "deployed"
    status.deployment = {
        "state": "ready",
        "endpoint_name": "https://serve.example",
        "adapter_revision": revision,
    }
    runner_state._save_status(status)

    assert "verification_generation" not in runner_status.get_status(run_id).deployment
    assert runner_verified_revisions.read_verified_adapter_revisions(run_id) == frozenset()
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **kwargs: pytest.fail("unverified revision must not reach serving"),
    )

    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "adapter_revision": revision,
        },
        headers=_bearer(key),
    )

    assert response.status_code == 409
    assert "has not passed a successful deployment smoke" in response.json()["detail"]


def test_chat_bare_alias_rejects_status_only_ready_record(api, monkeypatch):
    import flash.runner.lifecycle.state as runner_state
    import flash.runner.lifecycle.status as runner_status
    import flash.runner.results.verified_revisions as runner_verified_revisions
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = _make_run(api, key, "deployed")
    status = runner_status.get_status(run_id)
    status.deployment = {"state": "ready", "endpoint_name": "https://serve.example"}
    runner_state._save_status(status)
    assert runner_verified_revisions.read_verified_adapter_revisions(run_id) == frozenset()
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **kwargs: pytest.fail("status-only ready record must not reach serving"),
    )

    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={"messages": [{"role": "user", "content": "hello"}]},
        headers=_bearer(key),
    )

    assert response.status_code == 409
    assert "no active deployment" in response.json()["detail"]


def test_chat_bare_alias_rejects_confirmed_active_failed_record(api, monkeypatch):
    import flash.runner.lifecycle.state as runner_state
    import flash.runner.lifecycle.status as runner_status
    import flash.runner.results.verified_revisions as runner_verified_revisions
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = _make_run(api, key, "deployed")
    revision = f"{run_id}@final." + "a" * 40
    runner_verified_revisions.add_verified_adapter_revision(
        run_id,
        revision,
        expected_generation=runner_verified_revisions.verified_adapter_revision_generation(run_id),
    )
    status = runner_status.get_status(run_id)
    status.deployment = {
        "state": "failed",
        "adapter_revision": revision,
        "alias_activation_confirmed": True,
        "error": "post-activation verification failed",
    }
    runner_state._save_status(status)
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **_kwargs: pytest.fail("a failed alias must not serve bare chat"),
    )

    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={"messages": [{"role": "user", "content": "hello"}]},
        headers=_bearer(key),
    )

    assert response.status_code == 409
    assert "deployment failed" in response.json()["detail"]


def test_chat_reconciling_alias_rejects_bare_and_allows_verified_revision(api, monkeypatch):
    import flash.runner.results.verified_revisions as runner_verified_revisions
    import flash.runner.supervise.transitions as runner_transitions
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = _make_run(api, key, "done")
    previous_revision = f"{run_id}@step-10." + "b" * 40
    previous = {
        "state": "ready",
        "endpoint_name": "https://old.example",
        "adapter_revision": previous_revision,
    }
    runner_transitions.mark_deployed(
        run_id,
        previous,
        verification_generation=runner_verified_revisions.verified_adapter_revision_generation(
            run_id
        ),
    )
    runner_transitions.mark_deployment_pending(
        run_id,
        {
            "state": "reconciling",
            "requested_at": time.time(),
            "updated_at": time.time(),
            "activation_outcome_unknown": True,
            "previous_deployment": previous,
        },
    )
    served_revisions = []

    def serve_chat(**kwargs):
        served_revisions.append(kwargs["run_id"])
        return _managed_chat_result(kwargs["run_id"])

    monkeypatch.setattr(app_mod, "serve_chat", serve_chat)

    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={"messages": [{"role": "user", "content": "hello"}]},
        headers=_bearer(key),
    )

    assert response.status_code == 409
    assert "deployment is reconciling" in response.json()["detail"]
    assert served_revisions == []

    explicit = api.post(
        f"/v1/runs/{run_id}/chat",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "adapter_revision": previous_revision,
        },
        headers=_bearer(key),
    )

    assert explicit.status_code == 200, explicit.text
    assert served_revisions == [previous_revision]
    assert runner_verified_revisions.read_verified_adapter_revisions(run_id) == frozenset(
        {previous_revision}
    )


def test_chat_selects_immutable_revisions_independently(api, monkeypatch):
    import flash.runner.lifecycle.state as runner_state
    import flash.runner.lifecycle.status as runner_status
    import flash.runner.results.verified_revisions as runner_verified_revisions
    import flash.runner.supervise.transitions as runner_transitions
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    runner_state._save_status(status)
    revisions = [f"{run_id}@step-20." + "a" * 40, f"{run_id}@step-40." + "b" * 40]
    for revision in revisions:
        runner_transitions.mark_checkpoint_deployed(
            run_id,
            {
                "state": "ready",
                "endpoint_name": "https://serve.example",
                "adapter_revision": revision,
            },
            verification_generation=runner_verified_revisions.verified_adapter_revision_generation(
                run_id
            ),
        )
    assert runner_verified_revisions.read_verified_adapter_revisions(run_id) == frozenset(revisions)
    seen = []

    def fake_chat(**kwargs):
        seen.append(kwargs["run_id"])
        return _managed_chat_result(kwargs["run_id"])

    monkeypatch.setattr(app_mod, "serve_chat", fake_chat)

    for revision in revisions:
        response = api.post(
            f"/v1/runs/{run_id}/chat",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "adapter_revision": revision,
            },
            headers=_bearer(key),
        )
        assert response.status_code == 200, response.text

    assert seen == revisions


def test_chat_rejects_cross_run_immutable_revision(api, monkeypatch):
    import flash.runner.lifecycle.state as runner_state
    import flash.runner.lifecycle.status as runner_status
    import flash.runner.results.verified_revisions as runner_verified_revisions
    import flash.runner.supervise.transitions as runner_transitions
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    runner_state._save_status(status)
    revision = f"{run_id}@final." + "a" * 40
    runner_transitions.mark_deployed(
        run_id,
        {
            "state": "ready",
            "endpoint_name": "https://serve.example",
            "adapter_revision": revision,
        },
        verification_generation=runner_verified_revisions.verified_adapter_revision_generation(
            run_id
        ),
    )
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **kwargs: pytest.fail("cross-run revision must not reach serving"),
    )

    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "adapter_revision": "another-run@final." + "c" * 40,
        },
        headers=_bearer(key),
    )

    assert response.status_code == 400
    assert "belongs to run another-run" in response.json()["detail"]


def test_chat_uses_saved_thinking_flag_not_payload_override(api, monkeypatch):
    import flash.runner.lifecycle.state as runner_state
    import flash.runner.lifecycle.status as runner_status
    import flash.runner.results.verified_revisions as runner_verified_revisions
    import flash.runner.supervise.transitions as runner_transitions
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs",
        json={"spec": {**SPEC, "thinking": True}, "dry_run": True},
        headers=_bearer(key),
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    runner_state._save_status(status)
    revision = f"{run_id}@final." + "a" * 40
    runner_transitions.mark_deployed(
        run_id,
        {
            "state": "ready",
            "endpoint_name": "https://serve.example",
            "adapter_revision": revision,
        },
        verification_generation=runner_verified_revisions.verified_adapter_revision_generation(
            run_id
        ),
    )

    seen = {}

    def fake_chat(**kwargs):
        seen.update(kwargs)
        return _managed_chat_result(kwargs["run_id"])

    monkeypatch.setattr(app_mod, "serve_chat", fake_chat)

    resp = api.post(
        f"/v1/runs/{run_id}/chat",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "chat_template_kwargs": {"enable_thinking": False},
        },
        headers=_bearer(key),
    )

    assert resp.status_code == 200, resp.text
    assert seen["thinking"] is True
    assert "enable_thinking" not in seen


def test_chat_forwards_user_supplied_system_prompt(api, monkeypatch):
    import flash.runner.lifecycle.state as runner_state
    import flash.runner.lifecycle.status as runner_status
    import flash.runner.results.verified_revisions as runner_verified_revisions
    import flash.runner.supervise.transitions as runner_transitions
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    runner_state._save_status(status)
    revision = f"{run_id}@final." + "a" * 40
    runner_transitions.mark_deployed(
        run_id,
        {
            "state": "ready",
            "endpoint_name": "https://serve.example",
            "adapter_revision": revision,
        },
        verification_generation=runner_verified_revisions.verified_adapter_revision_generation(
            run_id
        ),
    )

    seen = {}

    def fake_chat(**kwargs):
        seen.update(kwargs)
        return _managed_chat_result(kwargs["run_id"])

    monkeypatch.setattr(app_mod, "serve_chat", fake_chat)

    messages = [
        {"role": "system", "content": "stay terse"},
        {"role": "user", "content": "hello"},
    ]
    resp = api.post(
        f"/v1/runs/{run_id}/chat",
        json={
            "messages": messages,
            "chat_template_kwargs": {"enable_thinking": True},
        },
        headers=_bearer(key),
    )

    assert resp.status_code == 200, resp.text
    assert seen["messages"] == messages
    assert seen["thinking"] is False
    assert "enable_thinking" not in seen


def test_chat_serves_cancelled_run_with_active_checkpoint_deployment(api, monkeypatch):
    """A run cancelled mid-RL can deploy a per-step checkpoint (stays `cancelled`, listed active by
    /v1/deployments). The chat route must SERVE that live adapter, not 409 on the cancelled state."""
    import flash.runner.lifecycle.state as runner_state
    import flash.runner.lifecycle.status as runner_status
    import flash.runner.results.verified_revisions as runner_verified_revisions
    import flash.runner.supervise.transitions as runner_transitions
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "cancelled"
    runner_state._save_status(status)
    revision = f"{run_id}@step-40." + "a" * 40
    runner_transitions.mark_checkpoint_deployed(
        run_id,
        {
            "state": "ready",
            "endpoint_name": "https://serve.example",
            "adapter_revision": revision,
        },
        verification_generation=runner_verified_revisions.verified_adapter_revision_generation(
            run_id
        ),
    )

    monkeypatch.setattr(
        app_mod,
        "serve_chat_sse",
        lambda **_kwargs: _RawManagedChatResponse(
            [
                b'data: {"choices":[{"index":0,"delta":{"content":"hi there"}}]}\n\n',
                b"data: [DONE]\n\n",
            ],
            headers=_managed_stream_headers(revision),
        ),
    )
    with api.stream(
        "POST",
        f"/v1/runs/{run_id}/chat",
        json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
        headers=_bearer(key),
    ) as resp:
        text = resp.read().decode()
    assert resp.status_code == 200, text
    assert text == (
        'data: {"choices":[{"index":0,"delta":{"content":"hi there"}}]}\n\ndata: [DONE]\n\n'
    )


def test_chat_cancelled_run_without_deployment_is_409(api):
    """A cancelled run with no active deployment still 409s, pointing the user at `flash deploy`."""
    import flash.runner.lifecycle.state as runner_state
    import flash.runner.lifecycle.status as runner_status

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "cancelled"
    runner_state._save_status(status)

    r = api.post(
        f"/v1/runs/{run_id}/chat",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers=_bearer(key),
    )
    assert r.status_code == 409
    assert "deploy a checkpoint" in r.json()["detail"]


def test_chat_rejects_undeployed_record_with_previous_ready_deployment(api, monkeypatch):
    import flash.runner.lifecycle.state as runner_state
    import flash.runner.lifecycle.status as runner_status
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    status.deployment = {
        "state": "undeployed",
        "previous_deployment": {"state": "ready", "endpoint_name": "https://old.example"},
    }
    runner_state._save_status(status)
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **kwargs: pytest.fail("undeployed aliases must never be served"),
    )

    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers=_bearer(key),
    )

    assert response.status_code == 409
    assert "no active deployment" in response.json()["detail"]


def test_chat_rejects_non_finite_sampling_params_with_400(api, monkeypatch):
    """Non-finite sampling values must return 400, including OverflowError from ``int(inf)``.

    OverflowError is ArithmeticError, not TypeError or ValueError, so the guard must include it.
    """
    import flash.runner.lifecycle.state as runner_state
    import flash.runner.lifecycle.status as runner_status
    import flash.runner.results.verified_revisions as runner_verified_revisions
    import flash.runner.supervise.transitions as runner_transitions
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    runner_state._save_status(status)
    revision = f"{run_id}@final." + "a" * 40
    runner_transitions.mark_deployed(
        run_id,
        {
            "state": "ready",
            "endpoint_name": "https://serve.example",
            "adapter_revision": revision,
        },
        verification_generation=runner_verified_revisions.verified_adapter_revision_generation(
            run_id
        ),
    )
    monkeypatch.setattr(app_mod, "serve_chat_stream", lambda **k: iter(["hi"]))

    headers = {**_bearer(key), "content-type": "application/json"}
    for body in (
        b'{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1e400}',
        b'{"messages": [{"role": "user", "content": "hi"}], "temperature": 1e400}',
    ):
        r = api.post(f"/v1/runs/{run_id}/chat", content=body, headers=headers)
        assert r.status_code == 400, (body, r.status_code, r.text)
