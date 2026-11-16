"""exact candidate-token logprobs (`logprob_token_ids`) across request, transport and response."""

from __future__ import annotations

import pytest

import flash.cli.parsing.main as cli
from flash.serve.request import transport
from flash.serve.request.openai import OpenAIRequestError, parse_chat_request
from flash.serve.runtime.errors import RuntimeConfigurationError
from flash.serve.runtime.sampling import (
    MAX_LOGPROB_TOKEN_IDS,
    normalize_token_logprobs,
    response_logprob_width,
    validate_logprob_token_ids,
    validate_sampling_relationships,
    vllm_logprob_kwargs,
)
from flash.serve.runtime.types import GenerationRequest
from flash.serving.src.io.cache_namespace import internal_generation_payload
from flash.serving.src.io.openai_request import OpenAIGenerateRequest
from flash.serving.src.io.responses import openai_generate_fields

# 36 option ids, the largest letter-logit readout (A..Z, 0..9).
OPTION_IDS = list(range(1000, 1036))


class _Logprob:
    def __init__(self, value: float, token: str) -> None:
        self.logprob = value
        self.decoded_token = token


def _payload(**updates):
    return {
        "model": "run/final",
        "messages": [{"role": "user", "content": "pick one"}],
        **updates,
    }


@pytest.mark.parametrize(
    "value",
    [[0], OPTION_IDS, list(range(MAX_LOGPROB_TOKEN_IDS)), (5, 6)],
)
def test_validate_accepts_unique_non_negative_ids_up_to_vllm_cap(value) -> None:
    assert validate_logprob_token_ids(value) == tuple(value)
    assert validate_logprob_token_ids(None) is None


@pytest.mark.parametrize(
    "value",
    [
        [],
        list(range(MAX_LOGPROB_TOKEN_IDS + 1)),
        [1, 1],
        [-1],
        [True],
        [1.0],
        ["1"],
        "12",
        5,
        {1: 2},
    ],
)
def test_validate_rejects_malformed_ids(value) -> None:
    with pytest.raises(RuntimeConfigurationError, match="logprob_token_ids must be a list"):
        validate_logprob_token_ids(value)


def test_relationships_require_logprobs_and_exclude_top_k() -> None:
    with pytest.raises(ValueError, match="requires logprobs=true"):
        validate_sampling_relationships(
            n=1, temperature=0.0, logprobs=False, top_logprobs=0, logprob_token_ids=(1,)
        )
    with pytest.raises(ValueError, match="positive top_logprobs"):
        validate_sampling_relationships(
            n=1, temperature=0.0, logprobs=True, top_logprobs=2, logprob_token_ids=(1,)
        )
    validate_sampling_relationships(
        n=1, temperature=0.0, logprobs=True, top_logprobs=0, logprob_token_ids=(1,)
    )


def test_vllm_kwargs_leave_logprobs_unset_so_max_logprobs_does_not_cap_candidates() -> None:
    assert vllm_logprob_kwargs(
        logprobs=True, top_logprobs=0, logprob_token_ids=tuple(OPTION_IDS)
    ) == {"logprobs": None, "logprob_token_ids": OPTION_IDS}
    assert vllm_logprob_kwargs(logprobs=True, top_logprobs=5, logprob_token_ids=None) == {
        "logprobs": 5
    }
    assert vllm_logprob_kwargs(logprobs=False, top_logprobs=0, logprob_token_ids=None) == {
        "logprobs": None
    }
    assert response_logprob_width(0, tuple(OPTION_IDS)) == 37
    assert response_logprob_width(4, None) == 4


def test_parse_chat_request_carries_ids_and_keeps_top_logprobs_cap() -> None:
    request = parse_chat_request(
        _payload(logprobs=True, logprob_token_ids=OPTION_IDS),
        require_model=True,
        allow_managed_selectors=False,
    )
    assert request.logprob_token_ids == tuple(OPTION_IDS)
    assert request.top_logprobs == 0
    plain = parse_chat_request(
        _payload(logprobs=True, top_logprobs=20), require_model=True, allow_managed_selectors=False
    )
    assert plain.logprob_token_ids is None
    with pytest.raises(
        OpenAIRequestError, match="top_logprobs must be an integer from 0 through 20"
    ):
        parse_chat_request(
            _payload(logprobs=True, top_logprobs=36),
            require_model=True,
            allow_managed_selectors=False,
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"logprob_token_ids": [1]}, "requires logprobs=true"),
        ({"logprobs": True, "top_logprobs": 3, "logprob_token_ids": [1]}, "positive top_logprobs"),
        ({"logprobs": True, "logprob_token_ids": [1, 1]}, "logprob_token_ids must be"),
        ({"logprobs": True, "logprob_token_ids": []}, "logprob_token_ids must be"),
    ],
)
def test_parse_chat_request_rejects_invalid_combinations(updates, message) -> None:
    with pytest.raises(OpenAIRequestError, match=message):
        parse_chat_request(_payload(**updates), require_model=True, allow_managed_selectors=False)


def test_parse_chat_request_rejects_ids_with_active_tools() -> None:
    tools = [
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
    with pytest.raises(OpenAIRequestError, match="tools cannot be combined with logprobs"):
        parse_chat_request(
            _payload(logprobs=True, logprob_token_ids=[1], tools=tools),
            require_model=True,
            allow_managed_selectors=False,
        )


def _body(**updates):
    return transport.chat_request_body(
        "run/final",
        [{"role": "user", "content": "hi"}],
        0.0,
        1,
        False,
        stream=False,
        top_p=0.95,
        n=1,
        seed=None,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        logprobs=True,
        top_logprobs=0,
        stop=None,
        chat_template_kwargs=None,
        structured_outputs=None,
        **updates,
    )


def test_transport_body_round_trips_through_the_serving_parser() -> None:
    assert "logprob_token_ids" not in _body()
    body = _body(logprob_token_ids=tuple(OPTION_IDS))
    assert body["logprob_token_ids"] == OPTION_IDS
    parsed = parse_chat_request(body, require_model=True, allow_managed_selectors=False)
    assert parsed.logprob_token_ids == tuple(OPTION_IDS)


def test_deploy_chat_helpers_forward_ids_to_transport(monkeypatch) -> None:
    from flash.serve.deployment import deploy

    seen: list[object] = []
    monkeypatch.setattr(transport, "_chat_http_client", lambda: object())
    monkeypatch.setattr(transport, "serving_openai_base_url", lambda: "https://serve.example/v1")
    monkeypatch.setattr(
        transport, "request_chat_json", lambda *_a, **kw: seen.append(kw["logprob_token_ids"])
    )
    monkeypatch.setattr(
        transport, "request_chat_sse", lambda *_a, **kw: seen.append(kw["logprob_token_ids"])
    )
    messages = [{"role": "user", "content": "hi"}]
    deploy.chat("run/final", messages, org_id="org", logprobs=True, logprob_token_ids=[4, 5])
    deploy.chat_sse("run/final", messages, org_id="org", logprobs=True, logprob_token_ids=[6])
    deploy.chat("run/final", messages, org_id="org")
    assert seen == [[4, 5], [6], None]


def test_hosted_envelope_carries_ids_and_omits_them_when_unset() -> None:
    parsed = parse_chat_request(
        _payload(logprobs=True, logprob_token_ids=OPTION_IDS),
        require_model=True,
        allow_managed_selectors=False,
    )
    fields = openai_generate_fields(parsed, "run/final")
    assert fields["logprob_token_ids"] == OPTION_IDS
    envelope = OpenAIGenerateRequest.model_validate(fields)
    assert envelope.logprob_token_ids == OPTION_IDS
    assert envelope.logprob_token_id_tuple == tuple(OPTION_IDS)
    assert internal_generation_payload(envelope)["logprob_token_ids"] == OPTION_IDS

    plain = parse_chat_request(
        _payload(logprobs=True), require_model=True, allow_managed_selectors=False
    )
    plain_envelope = OpenAIGenerateRequest.model_validate(
        openai_generate_fields(plain, "run/final")
    )
    assert "logprob_token_ids" not in openai_generate_fields(plain, "run/final")
    assert "logprob_token_ids" not in internal_generation_payload(plain_envelope)


def test_hosted_envelope_validates_ids() -> None:
    base = {"model_id": "run/final", "prompt": "hi", "logprobs": True}
    with pytest.raises(ValueError, match="logprob_token_ids must be"):
        OpenAIGenerateRequest.model_validate({**base, "logprob_token_ids": [3, 3]})
    with pytest.raises(ValueError, match="requires logprobs=true"):
        OpenAIGenerateRequest.model_validate({**base, "logprobs": False, "logprob_token_ids": [3]})


def test_generation_request_normalizes_ids_to_a_tuple() -> None:
    request = GenerationRequest(prompt="hi", logprobs=True, logprob_token_ids=[7, 8])
    assert request.logprob_token_ids == (7, 8)
    with pytest.raises(RuntimeConfigurationError, match="requires logprobs=true"):
        GenerationRequest(prompt="hi", logprob_token_ids=[7])


def test_normalization_keeps_every_candidate_and_the_sampled_token() -> None:
    # vllm returns the sampled token first and then each requested id; the sampled greedy token
    # outranks every candidate, so truncating to len(ids) would silently drop the weakest option.
    candidates = {999: _Logprob(-0.01, "The")}
    for offset, token_id in enumerate(OPTION_IDS):
        candidates[token_id] = _Logprob(-1.0 - offset, chr(ord("A") + offset % 26))
    # a padded column from a mixed vllm batch carries -inf and is dropped.
    candidates[0] = _Logprob(float("-inf"), "!")
    width = response_logprob_width(0, tuple(OPTION_IDS))
    [position] = normalize_token_logprobs([999], [candidates], top_logprobs=width)
    returned = [record["logprob"] for record in position["top_logprobs"]]
    assert len(returned) == len(OPTION_IDS) + 1
    assert returned[0] == -0.01
    assert returned[1:] == [-1.0 - offset for offset in range(len(OPTION_IDS))]


def test_normalization_when_sampled_token_is_a_candidate() -> None:
    candidates = {1000: _Logprob(-0.2, "A"), 1001: _Logprob(-1.8, "B")}
    [position] = normalize_token_logprobs(
        [1000], [candidates], top_logprobs=response_logprob_width(0, (1000, 1001))
    )
    assert position["token"] == "A"
    assert [record["token"] for record in position["top_logprobs"]] == ["A", "B"]


def test_cli_chat_parses_comma_separated_ids() -> None:
    args = cli._build_parser().parse_args(
        ["models", "chat", "flash-1/final", "-m", "hi", "--logprob-token-ids", "32,33,34"]
    )
    assert args.logprob_token_ids == [32, 33, 34]


@pytest.mark.parametrize("value", ["", "a,b", "1,1", "-1", ",".join(["1"] * 2)])
def test_cli_chat_rejects_invalid_ids(value) -> None:
    with pytest.raises(SystemExit):
        cli._build_parser().parse_args(
            ["models", "chat", "flash-1/final", "-m", "hi", "--logprob-token-ids", value]
        )
