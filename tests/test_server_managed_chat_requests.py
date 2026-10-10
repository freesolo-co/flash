from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from flash.serve.contract.provenance import (
    CheckpointProvenance,
    decode_flash_body,
    decode_freesolo_headers,
    validate_body_provenance,
)
from flash.server.routes.serving_revisions import _authorized_chat_checkpoint
from tests._helpers.chat_provenance import managed_chat_result as _managed_chat_result
from tests._helpers.managed_chat import _deployed_chat_run
from tests.test_server_api import SPEC, _bearer, _login

pytest_plugins = ("tests._helpers.server_api_plugin",)

RUN_ID = "flash-1234567890-abcdef12"
CHECKPOINT_ID = f"{RUN_ID}/final"


def _tool_payload():
    return [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def _ready_deployment() -> dict[str, str]:
    return {"state": "ready", "checkpoint_id": CHECKPOINT_ID, "openai_model": CHECKPOINT_ID}


def test_managed_chat_authorizes_one_explicit_verified_checkpoint() -> None:
    assert (
        _authorized_chat_checkpoint(
            RUN_ID,
            _ready_deployment(),
            CHECKPOINT_ID,
            {CHECKPOINT_ID},
        )
        == CHECKPOINT_ID
    )


@pytest.mark.parametrize(
    ("checkpoint_id", "detail"),
    [
        (None, "checkpoint_id must"),
        (RUN_ID, "checkpoint_id must"),
        (RUN_ID + "@final." + "a" * 40, "checkpoint_id must"),
        ("other-run/final", "belongs to run"),
    ],
)
def test_managed_chat_rejects_missing_bare_composite_and_cross_run_targets(
    checkpoint_id: object, detail: str
) -> None:
    with pytest.raises(HTTPException, match=detail):
        _authorized_chat_checkpoint(
            RUN_ID,
            _ready_deployment(),
            checkpoint_id,
            {CHECKPOINT_ID},
        )


def test_managed_chat_rejects_mandatory_whitespace_stop_for_active_tool_grammar(api, monkeypatch):
    import flash.runner.lifecycle.state as runner_state
    import flash.runner.lifecycle.status as runner_status
    import flash.runner.results.verified_revisions as runner_verified_revisions
    import flash.runner.supervise.transitions as runner_transitions
    import flash.server.asgi.app as app_mod

    key = _login()
    spec = json.loads(json.dumps(SPEC))
    spec["train"] = {**spec["train"], "stop_sequences": [" \t\r\n"]}
    run_id = api.post(
        "/v1/runs", json={"spec": spec, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    runner_state._save_status(status)
    checkpoint_id = f"{run_id}/final"
    runner_transitions.mark_deployed(
        run_id,
        {
            "state": "ready",
            "endpoint_name": "https://serve.example",
            "checkpoint_id": checkpoint_id,
            "openai_model": checkpoint_id,
        },
        verification_generation=runner_verified_revisions.verified_checkpoint_generation(run_id),
    )
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **_kwargs: pytest.fail("mandatory whitespace stop must not reach serving"),
    )
    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={
            "checkpoint_id": f"{run_id}/final",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": _tool_payload(),
        },
        headers=_bearer(key),
    )

    assert response.status_code == 400
    assert "whitespace separators" in response.json()["detail"]


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
def test_managed_chat_rejects_active_tool_stop_marker_collision_before_forwarding(
    api, monkeypatch, stream
):
    import flash.server.asgi.app as app_mod

    key, run_id = _deployed_chat_run(api)
    monkeypatch.setattr(
        app_mod,
        "serve_chat_sse" if stream else "serve_chat",
        lambda **_kwargs: pytest.fail("colliding stop must not reach serving"),
    )
    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={
            "checkpoint_id": f"{run_id}/final",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": _tool_payload(),
            "stop": "</tool_call>",
            "stream": stream,
        },
        headers=_bearer(key),
    )

    assert response.status_code == 400
    assert "grammar markers" in response.json()["detail"]


@pytest.mark.parametrize("number", ["1.0", "1e3", "9007199254740993.0", "1e-400"])
def test_managed_raw_json_rejects_decimal_numeric_enums(api, monkeypatch, number):
    import flash.server.asgi.app as app_mod

    key, run_id = _deployed_chat_run(api)
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **_kwargs: pytest.fail("decimal numeric enum must not reach serving"),
    )
    raw = (
        '{"checkpoint_id":"' + run_id + '/final",'
        '"messages":[{"role":"user","content":"weather"}],"tools":[{"type":"function",'
        '"function":{"name":"weather","parameters":{"type":"object","properties":'
        '{"value":{"type":"number","enum":['
        + number
        + ']}},"required":["value"],"additionalProperties":false}}}]}'
    )

    response = api.post(
        f"/v1/runs/{run_id}/chat",
        content=raw,
        headers={**_bearer(key), "content-type": "application/json"},
    )

    assert response.status_code == 400
    assert "numeric enum members must be JSON integers" in response.json()["detail"]


def test_managed_chat_forwards_normalized_tools_without_parsing_output(api, monkeypatch):
    import flash.server.asgi.app as app_mod

    key, run_id = _deployed_chat_run(api)
    seen = {}

    def fake_chat(**kwargs):
        seen.update(kwargs)
        result = _managed_chat_result(kwargs["run_id"])
        result["choices"][0] = {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
        return result

    monkeypatch.setattr(app_mod, "serve_chat", fake_chat)
    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={
            "checkpoint_id": f"{run_id}/final",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": _tool_payload(),
        },
        headers=_bearer(key),
    )
    assert response.status_code == 200, response.text
    assert seen["tool_choice"] == "auto"
    assert seen["parallel_tool_calls"] is True
    assert seen["tools"][0].name == "weather"
    assert response.json()["choices"][0]["finish_reason"] == "tool_calls"


def test_managed_chat_rejects_unverified_checkpoint() -> None:
    with pytest.raises(HTTPException, match="has not passed"):
        _authorized_chat_checkpoint(
            RUN_ID,
            _ready_deployment(),
            CHECKPOINT_ID,
            set(),
        )


def test_managed_chat_accepts_any_exact_verified_ready_sibling() -> None:
    sibling = f"{RUN_ID}/step-20"
    assert (
        _authorized_chat_checkpoint(
            RUN_ID,
            _ready_deployment(),
            sibling,
            {CHECKPOINT_ID, sibling},
        )
        == sibling
    )


def test_managed_provenance_contains_only_checkpoint_identity() -> None:
    provenance = CheckpointProvenance(CHECKPOINT_ID)
    assert provenance.freesolo_body() == {"checkpoint_id": CHECKPOINT_ID}
    assert provenance.freesolo_headers() == {"X-Freesolo-Checkpoint": CHECKPOINT_ID}
    assert decode_flash_body({"checkpoint_id": CHECKPOINT_ID}) == provenance
    assert decode_freesolo_headers({"x-freesolo-checkpoint": CHECKPOINT_ID}) == provenance

    response = validate_body_provenance(
        {"flash_provenance": {"checkpoint_id": CHECKPOINT_ID}},
        provenance,
    )
    assert response["freesolo"] == {"checkpoint_id": CHECKPOINT_ID}
    assert "artifact_revision" not in str(response)
    assert "hf_revision" not in str(response)
