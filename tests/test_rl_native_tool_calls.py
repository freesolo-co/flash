"""Native tool calling in multi-turn GRPO rollouts.

An environment that declares ``tool_schemas`` gets the serving contract end to end: every render
(first prompt, each generation prefix, environment-reply glue) passes the normalized declarations
as ``tools=``, each policy turn is parsed by the serving parser into an OpenAI assistant message
before ``step_episode`` sees it, and tool results ride back as ``role: tool`` messages. These tests
drive the real adapter, bridge, and child turn loop with the real Qwen3.5 chat template (a
character-level tokenizer keeps them offline), and an opt-in test repeats the byte comparison with
the real Qwen3.5-9B tokenizer.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import jinja2
import pytest

from flash.engine.worker.train.core.child.glue import (
    EnvGlueTokenizer,
    validate_structured_messages,
    validate_transcript_messages,
)
from flash.serve.request.tool_calls import (
    detached_template_messages,
    normalize_tools,
    tools_wire,
)
from flash.serve.runtime.reasoning_tools import parse_tool_output

TEMPLATE = (Path(__file__).parent / "fixtures/qwen3_5_9b_chat_template.jinja").read_text()
IM_END = "<|im_end|>"
IM_END_ID = 0xE000

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_details",
            "description": "Look up one order by id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "The order id."}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_to_human_agents",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
]
WIRE = tools_wire(normalize_tools(TOOLS))
SYSTEM = {"role": "system", "content": "You are a retail support agent."}
CALL = (
    "<tool_call>\n<function=get_order_details>\n<parameter=order_id>\n#W1\n</parameter>\n"
    "</function>\n</tool_call>"
)
ANSWER = "Your order #W1 has shipped."


class _TemplateError(Exception):
    pass


def _raise(message):
    raise _TemplateError(message)


class _CharTokenizer:
    """the real chat template; one id per character, with <|im_end|> as a single special id."""

    def __init__(self):
        environment = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
        environment.globals["raise_exception"] = _raise
        environment.filters["tojson"] = lambda value, **kwargs: json.dumps(
            value, ensure_ascii=False
        )
        self.template = environment.from_string(TEMPLATE)
        self.renders: list[dict] = []

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, **kwargs):
        assert tokenize is False
        self.renders.append(dict(kwargs))
        return self.template.render(
            messages=messages, add_generation_prompt=add_generation_prompt, **kwargs
        )

    def encode_text(self, text):
        ids = []
        for index, part in enumerate(text.split(IM_END)):
            if index:
                ids.append(IM_END_ID)
            ids.extend(ord(character) for character in part)
        return ids

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=self.encode_text(text))

    def decode(self, ids, skip_special_tokens=False):
        special = "" if skip_special_tokens else IM_END
        return "".join(special if value == IM_END_ID else chr(value) for value in ids)


def _serving_render(tokenizer, messages, tools=TOOLS):
    """what serving renders for a chat request with these messages and tools."""
    return tokenizer.apply_chat_template(
        detached_template_messages(messages),
        tools=tools_wire(normalize_tools(tools)),
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
        preserve_thinking=False,
    )


def _tool_env_class(*, schemas, steps):
    from freesolo.environments import (
        EnvironmentEpisode,
        EnvironmentMultiTurn,
        EnvironmentStepResult,
        RewardResult,
    )

    class _RetailEnv(EnvironmentMultiTurn):
        dataset: ClassVar[list] = [{"input": "Where is my order #W1?"}]
        tool_schemas = schemas

        def start_episode(self, example, prompt_text):
            return [dict(SYSTEM), {"role": "user", "content": example.input}]

        def max_episode_turns(self, example) -> int:
            return 4

        def step_episode(self, example, messages, assistant_response):
            steps.append((json.loads(json.dumps(messages)), assistant_response))
            last = messages[-1]
            if last.get("tool_calls"):
                return EnvironmentStepResult(
                    done=False,
                    messages=tuple(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": call["function"]["name"],
                            "content": json.dumps({"order_id": "#W1", "status": "shipped"}),
                        }
                        for call in last["tool_calls"]
                    ),
                )
            return EnvironmentStepResult(done=True)

        def score_episode(self, example, episode: EnvironmentEpisode) -> RewardResult:
            return RewardResult(score=1.0, threshold=1.0)

    return _RetailEnv


def _adapter(*, schemas=None, steps=None):
    from flash.envs.loading.adapter import FreesoloEnvironment

    env_class = _tool_env_class(schemas=schemas, steps=steps if steps is not None else [])
    return FreesoloEnvironment(env_class(), "retail", source=None)


def _child_env(monkeypatch, tools):
    from flash.engine.worker.train.rl.rollout.multi_turn import multi_turn_child_env

    values = multi_turn_child_env(
        {
            "max_turns": 4,
            "engine_len": 16384,
            "max_completion": 4096,
            "stop_sequences": (),
            "eos_token_ids": {IM_END_ID},
            "tools": tools,
        },
        reward_url="http://bridge.invalid",
        thinking=False,
    )
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    return values


def _drive_episode(monkeypatch, env, generations):
    """run the real child turn loop against the real bridge; returns (output, generation prompts)."""
    from flash.engine.worker.train.rl.child import multiturn as grpo_multiturn
    from flash.engine.worker.train.rl.launch.inputs import _build_grpo_prompts
    from flash.engine.worker.train.rl.rollout.multi_turn import MultiTurnBridge

    tokenizer = _CharTokenizer()
    tools = env.tools() or None
    _child_env(monkeypatch, tools)
    examples = env.dataset()
    prompts = _build_grpo_prompts(
        examples,
        [env.prompt_messages(example) for example in examples],
        False,
        None,
        tokenizer,
        None,
        100_000,
        context=SimpleNamespace(config=SimpleNamespace(thinking=False)),
        tools=tools,
    )
    bridge = MultiTurnBridge(
        env,
        [prompt["example"] for prompt in prompts],
        env_prompts=[prompt["env_prompt"] for prompt in prompts],
        max_turns=4,
        prompt_ids=[prompt["prompt_ids"] for prompt in prompts],
        tokenizer=tokenizer,
        tools=tools,
    )
    routes = bridge.routes()
    prefixes: list[list[int]] = []
    pending = list(generations)

    class _Base:
        def __init__(self):
            self.tokenizer = tokenizer
            self.processor = None
            self.rollout_config = SimpleNamespace(response_length=16384)
            self.server_manager = self

        def _get_mm_processor_kwargs(self, audio_data=None):
            return {}

        async def process_multi_modal_info(self, messages):
            return {}

        def _assert_mm_supported(self, has_media):
            pass

        async def ct_build_initial_tokens(self, messages, *, tools=None, **kwargs):
            # verl's continuous-token builder: one native render with tools=.
            text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
                preserve_thinking=False,
                **({} if tools is None else {"tools": tools}),
            )
            return tokenizer.encode_text(text)

        async def generate(self, *, request_id, prompt_ids, sampling_params, **kwargs):
            prefixes.append(list(prompt_ids))
            ids = tokenizer.encode_text(pending.pop(0) + IM_END)
            return SimpleNamespace(
                token_ids=ids, log_probs=[0.0] * len(ids), num_preempted=0, stop_reason="completed"
            )

    captured = {}

    def agent_loop_output(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    loop_class = grpo_multiturn.build_flash_grpo_multi_turn_agent_loop(
        register=lambda name: lambda cls: cls,
        agent_loop_base=_Base,
        agent_loop_output=agent_loop_output,
        bridge_post=lambda url, path, payload: routes[path](payload),
    )

    async def _go():
        instance = loop_class()
        instance.loop = asyncio.get_running_loop()
        await instance.run(
            {},
            raw_prompt=prompts[0]["prompt"],
            index=0,
            flash_rollout_identity={
                "optimizer_step": 1,
                "sample_index": 0,
                "rollout_ordinal": 0,
                "validate": False,
            },
        )

    asyncio.run(_go())
    return captured, prefixes, tokenizer, prompts[0]


def test_tool_env_rollout_steps_on_the_parsed_call_and_renders_like_serving(monkeypatch):
    steps: list = []
    env = _adapter(schemas=TOOLS, steps=steps)
    output, prefixes, tokenizer, prompt = _drive_episode(monkeypatch, env, [CALL, ANSWER])

    # the environment stepped on the structured message the serving parser recovered, while the
    # raw text stays the action argument.
    (first_messages, first_response), (second_messages, second_response) = steps
    assert first_response == CALL
    call_turn = first_messages[-1]
    assert call_turn["role"] == "assistant"
    assert call_turn["content"] == ""
    (call,) = call_turn["tool_calls"]
    assert call["type"] == "function"
    assert call["id"]
    assert call["function"]["name"] == "get_order_details"
    assert json.loads(call["function"]["arguments"]) == {"order_id": "#W1"}
    # the tool reply was accepted and reached the next step's transcript.
    assert second_messages[-2]["role"] == "tool"
    assert second_messages[-2]["tool_call_id"] == call["id"]
    assert second_messages[-1] == {"role": "assistant", "content": ANSWER}
    assert second_response == ANSWER

    # every generation prompt is byte-identical to the serving render of the same conversation.
    user = {"role": "user", "content": "Where is my order #W1?"}
    assert prompt["rendered"] == _serving_render(tokenizer, [SYSTEM, user])
    assert tokenizer.decode(prefixes[0]) == _serving_render(tokenizer, [SYSTEM, user])
    assert tokenizer.decode(prefixes[1]) == _serving_render(
        tokenizer, [SYSTEM, user, call_turn, second_messages[-2]]
    )
    assert "# Tools" in tokenizer.decode(prefixes[1])

    # only policy tokens train: both turns (each with its sampled <|im_end|>) and no glue.
    mask = output["response_mask"]
    response = output["response_ids"]
    turn_lengths = [len(tokenizer.encode_text(text + IM_END)) for text in (CALL, ANSWER)]
    assert sum(mask) == sum(turn_lengths)
    policy = [token for token, keep in zip(response, mask, strict=True) if keep]
    assert tokenizer.decode(policy) == CALL + IM_END + ANSWER + IM_END
    assert prefixes[1] == [*prefixes[0], *response[: len(prefixes[1]) - len(prefixes[0])]]
    assert output["num_turns"] == 2
    assert output["extra_fields"]["flash_episode_termination"] == "environment_terminal"


def test_malformed_tool_call_text_stays_plain_content():
    env = _adapter(schemas=TOOLS)
    state = env.new_rollout_state(env.dataset()[0])
    malformed = "<tool_call>\n<function=no_such_tool>\n</function>\n</tool_call>"
    assert env.record_model_turn(state, malformed) == {"role": "assistant", "content": malformed}
    unclosed = "<tool_call>\n<function=get_order_details>\n<parameter=order_id>\n#W1"
    assert env.record_model_turn(state, unclosed) == {"role": "assistant", "content": unclosed}
    # serving keeps the text before the call verbatim; the template trims it when rendering.
    assert env.record_model_turn(state, "Sure.\n\n" + CALL)["content"] == "Sure.\n\n"


def test_thinking_turns_parse_calls_only_after_the_reasoning_close():
    # the serving parser's thinking mode: a call inside unfinished reasoning is not executable.
    env = _adapter(schemas=TOOLS)
    env.thinking = True
    state = env.new_rollout_state(env.dataset()[0])
    reasoned = env.record_model_turn(state, "look it up\n</think>\n\n" + CALL)
    assert reasoned["content"] == "look it up\n</think>\n\n"
    assert reasoned["tool_calls"][0]["function"]["name"] == "get_order_details"
    unfinished = "still thinking " + CALL
    assert env.record_model_turn(state, unfinished) == {"role": "assistant", "content": unfinished}


def test_tool_free_env_is_unchanged(monkeypatch):
    env = _adapter()
    assert env.tools() == []
    state = env.new_rollout_state(env.dataset()[0])
    # without declarations the call text is just content, exactly as before.
    assert env.record_model_turn(state, CALL) == {"role": "assistant", "content": CALL}
    assert _child_env(monkeypatch, None)["FLASH_VERL_TOOLS"] == ""

    tokenizer = _CharTokenizer()
    reply = [{"role": "tool", "content": "{}"}]
    EnvGlueTokenizer(tokenizer, thinking=False)(reply)
    # no tools key at all, so the render is byte-identical to the pre-tools one.
    assert tokenizer.renders == [{"enable_thinking": False, "preserve_thinking": False}]
    for message in (
        {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "x"}]},
    ):
        with pytest.raises(ValueError, match="unsupported transcript metadata"):
            EnvGlueTokenizer(tokenizer, thinking=False)([message])
        with pytest.raises(ValueError, match="unsupported transcript metadata"):
            validate_structured_messages([message], source="environment reply")


def test_tool_env_transcript_accepts_only_openai_tool_fields_on_their_roles():
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_order_details", "arguments": '{"order_id": "#W1"}'},
    }
    accepted = validate_transcript_messages(
        [
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "get_order_details",
                "content": "{}",
            },
        ],
        source="environment reply",
        tools=True,
    )
    assert accepted[0] == {"role": "assistant", "content": "", "tool_calls": [call]}
    for message, match in (
        ({"role": "user", "content": "x", "tool_call_id": "call_1"}, "requires the tool role"),
        ({"role": "tool", "content": "x", "tool_calls": [call]}, "require the assistant role"),
        (
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [dict(call, function={"name": "f"})],
            },
            "exactly name and arguments",
        ),
        (
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [dict(call, function={"name": "f", "arguments": "[1]"})],
            },
            "must encode a JSON object",
        ),
        ({"role": "tool", "content": "x", "metadata": {}}, "unsupported transcript metadata"),
    ):
        with pytest.raises(ValueError, match=match):
            validate_structured_messages([message], source="environment reply", tools=True)


def test_tool_env_glue_matches_the_full_serving_render():
    tokenizer = _CharTokenizer()
    user = {"role": "user", "content": "Where is my order #W1?"}
    call_turn = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_order_details", "arguments": '{"order_id": "#W1"}'},
            }
        ],
    }
    reply = [{"role": "tool", "tool_call_id": "call_1", "content": '{"status": "shipped"}'}]
    glue = EnvGlueTokenizer(tokenizer, thinking=False, tools=WIRE)(reply)
    assert tokenizer.renders[-1]["tools"] == WIRE
    accumulated = _serving_render(tokenizer, [SYSTEM, user]) + CALL + tokenizer.decode(glue)
    assert accumulated == _serving_render(tokenizer, [SYSTEM, user, call_turn, *reply])


@pytest.mark.parametrize(
    ("schemas", "match"),
    [
        ([], "nonempty array"),
        ([{"type": "function", "function": {"name": "bad name", "parameters": {}}}], "name"),
        (
            [
                {
                    "type": "function",
                    "function": {"name": "f", "parameters": {"type": "object"}, "strict": True},
                }
            ],
            "strict",
        ),
        ([{"function": {"name": "f", "parameters": {"type": "object"}}}], "type and function"),
    ],
)
def test_malformed_tool_schemas_fail_at_load(schemas, match):
    with pytest.raises(ValueError, match=rf"environment tool_schemas: .*{match}"):
        _adapter(schemas=schemas)


def test_tool_schemas_may_be_a_method():
    from flash.envs.loading.adapter import FreesoloEnvironment

    env_class = _tool_env_class(schemas=None, steps=[])

    class _MethodEnv(env_class):
        def tool_schemas(self):
            return tuple(TOOLS)

    assert FreesoloEnvironment(_MethodEnv(), "retail", source=None).tools() == WIRE


def test_grpo_tool_resolution_refuses_shapes_that_cannot_match_serving():
    from flash.engine.worker.train.rl.launch.inputs import _resolve_grpo_tools

    env = _adapter(schemas=TOOLS)
    rows = env.dataset()

    def resolve(**overrides):
        kwargs = {"multi_turn": True, "multimodal": False, "stop_sequences": ()}
        kwargs.update(overrides)
        return _resolve_grpo_tools(env, rows, **kwargs)

    assert resolve() == WIRE
    assert (
        _resolve_grpo_tools(_adapter(), rows, multi_turn=True, multimodal=False, stop_sequences=())
        is None
    )
    with pytest.raises(RuntimeError, match="only for multi-turn"):
        resolve(multi_turn=False)
    with pytest.raises(RuntimeError, match="image-bearing"):
        resolve(multimodal=True)
    with pytest.raises(ValueError, match="stop sequences cannot overlap"):
        resolve(stop_sequences=("</tool_call>",))
    mismatched = [dict(rows[0], tools=TOOLS[:1])]
    with pytest.raises(ValueError, match="differ from the environment"):
        _resolve_grpo_tools(env, mismatched, multi_turn=True, multimodal=False, stop_sequences=())
    same = [dict(rows[0], tools=TOOLS)]
    assert (
        _resolve_grpo_tools(env, same, multi_turn=True, multimodal=False, stop_sequences=()) == WIRE
    )


_LIVE = os.environ.get("FLASH_RL_TOOLS_TEMPLATE_LIVE") == "1"
_QWEN = "Qwen/Qwen3.5-9B"
_QWEN_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


@pytest.mark.skipif(
    not _LIVE, reason="set FLASH_RL_TOOLS_TEMPLATE_LIVE=1 (needs the cached tokenizer)"
)
def test_live_qwen35_rl_tool_prompt_equals_the_serving_render():
    from transformers import AutoTokenizer

    from flash.content.stop_tokens import chat_stop_token_ids
    from flash.engine.worker.train.core.child.glue import prepare_assistant_turn
    from flash.engine.worker.train.core.rollout.generation import (
        generation_eos_from_cached_config,
    )
    from flash.engine.worker.train.rl.launch.inputs import _build_grpo_prompts

    tokenizer = AutoTokenizer.from_pretrained(_QWEN, revision=_QWEN_REVISION)
    env = _adapter(schemas=TOOLS)
    examples = env.dataset()
    tools = env.tools()
    (prompt,) = _build_grpo_prompts(
        examples,
        [env.prompt_messages(example) for example in examples],
        False,
        None,
        tokenizer,
        None,
        100_000,
        context=SimpleNamespace(config=SimpleNamespace(thinking=False)),
        tools=tools,
    )
    user = {"role": "user", "content": "Where is my order #W1?"}
    serving_prompt = tokenizer.apply_chat_template(
        [SYSTEM, user],
        tools=tools_wire(normalize_tools(TOOLS)),
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    assert prompt["rendered"] == serving_prompt
    assert prompt["prompt_ids"] == tokenizer(serving_prompt, add_special_tokens=False).input_ids

    # the policy emits the call followed by <|im_end|>, which stops generation as an eos: the same
    # halting set the launch hands the child (checkpoint eos plus any chat-turn boundary).
    eos = generation_eos_from_cached_config(_QWEN, _QWEN_REVISION, tokenizer) | frozenset(
        chat_stop_token_ids(_QWEN, tokenizer, is_chat=True) or ()
    )
    im_end = tokenizer.convert_tokens_to_ids(IM_END)
    assert im_end in eos
    generated = [*tokenizer(CALL, add_special_tokens=False).input_ids, im_end]
    turn = prepare_assistant_turn(
        tokenizer,
        generated,
        stop_reason="completed",
        max_tokens=4096,
        eos_token_ids=eos,
        stop_sequences=(),
    )
    # the call markers are ordinary text, so the decoded completion keeps them for the parser.
    assert turn["completion_text"] == CALL
    assert turn["termination"] == "eos"
    state = env.new_rollout_state(examples[0], prompt["env_prompt"])
    call_turn = env.record_model_turn(state, turn["completion_text"])
    assert call_turn["tool_calls"][0]["function"]["name"] == "get_order_details"

    reply = [{"role": "tool", "tool_call_id": call_turn["tool_calls"][0]["id"], "content": "{}"}]
    glue = EnvGlueTokenizer(tokenizer, thinking=False, tools=tools)(reply)
    from flash.engine.worker.train.core.child.glue import dedup_seam_terminator

    accumulated = [*prompt["prompt_ids"], *generated, *dedup_seam_terminator(generated, glue)]
    serving_next = tokenizer.apply_chat_template(
        detached_template_messages([SYSTEM, user, call_turn, *reply]),
        tools=tools_wire(normalize_tools(TOOLS)),
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    assert tokenizer.decode(accumulated) == serving_next
    assert accumulated == tokenizer(serving_next, add_special_tokens=False).input_ids

    # the serving parser recovers the same call from the template's own rendering of the turn.
    header = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    start = serving_next.index(header, len(serving_prompt) - len(header)) + len(header)
    rendered_turn = serving_next[start : serving_next.index(IM_END, start)]
    parsed = parse_tool_output(rendered_turn, normalize_tools(TOOLS), thinking=False)
    (recovered,) = parsed.calls
    assert recovered.name == "get_order_details"
    assert json.loads(recovered.arguments) == {"order_id": "#W1"}
