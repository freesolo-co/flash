"""cpu contracts for multi-turn OPD through verl 0.8.0."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from flash.engine.worker.opd import _gkd_loss_from_logps
from flash.engine.worker.opd_verl import _BridgePrompt, _TeacherAlignmentBridge
from flash.engine.worker.opd_verl_multiturn import (
    EnvGlueTokenizer,
    build_flash_multi_turn_agent_loop,
    prepare_assistant_turn,
    validate_glue_template,
    validate_teacher_messages,
)
from flash.engine.worker.opd_verl_plugin import (
    _flash_groupwise_reverse_kl_values,
    _full_sequence_signal_sequences,
    _turn_outputs,
    deterministic_rollout_seed,
)
from flash.engine.worker.tokenizer_align import TeacherToken


class _ChatTokenizer:
    eos_token_id = ord("|")
    pad_token_id = 0

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    ):
        assert tokenize is False
        assert enable_thinking is False
        text = "".join(
            f"{str(message['role'])[0]}:{message['content']}|" for message in messages
        )
        if add_generation_prompt:
            text += "a:"
        return text

    def __call__(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return SimpleNamespace(input_ids=[ord(character) for character in text])

    def decode(self, token_ids, *, skip_special_tokens):
        text = "".join(chr(int(token_id)) for token_id in token_ids)
        return text.replace("|", "") if skip_special_tokens else text


class _Teacher:
    def __init__(self):
        self.batches = []

    def score_many(self, items):
        self.batches.append(list(items))
        return [
            [
                TeacherToken(text=character, logprob=-0.25, start=index, end=index + 1)
                for index, character in enumerate(completion)
            ]
            for _prompt, completion in items
        ]

    def score(self, prompt, completion):
        return self.score_many([(prompt, completion)])[0]


class _BridgeEnv:
    multi_turn = True
    is_tool_env = False
    max_turns = 4

    def __init__(self):
        self.states = []
        self.step_calls = []

    def new_rollout_state(self, example):
        state = {
            "prompt": [{"role": "user", "content": example["question"]}],
            "messages": [{"role": "user", "content": example["question"]}],
            "assistant": [],
            "done": False,
            "max_episode_turns": example.get("turn_limit", 2),
        }
        self.states.append(state)
        return state

    def record_model_turn(self, state, content):
        state["assistant"].append(content)

    def env_reply(self, messages, state):
        self.step_calls.append((id(state), tuple(state["assistant"])))
        if len(state["assistant"]) >= state["max_episode_turns"]:
            state["done"] = True
            return []
        return [{"role": "tool", "content": f"obs-{len(state['assistant'])}"}]

    def rollout_done(self, state, max_turns):
        return state["done"] or len(state["assistant"]) >= max_turns


def _bridge(env=None, teacher=None, *, max_turns=4):
    env = env or _BridgeEnv()
    teacher = teacher or _Teacher()
    prompt = _BridgePrompt(
        student_messages=[{"role": "user", "content": "q"}],
        teacher_messages=[{"role": "user", "content": "q"}],
        prompt_ids=(1, 2),
        image_descriptors=(),
        package_root=None,
        example={"question": "q", "turn_limit": 2},
    )
    return _TeacherAlignmentBridge(
        prompts=[prompt],
        tokenizer=_ChatTokenizer(),
        teacher=teacher,
        thinking_prefill="",
        eos_token_ids=frozenset({ord("|")}),
        stop_sequences=(),
        mutation_callback=lambda: None,
        active_env=env,
        multi_turn=True,
        max_turns=max_turns,
    ), env, teacher


def _step_payload(session_id, ordinal, prefix, text):
    response_ids = [ord(character) for character in text] + [ord("|")]
    return {
        "session_id": session_id,
        "turn_ordinal": ordinal,
        "accepted_prefix": prefix,
        "raw_response_ids": response_ids,
        "response_ids": response_ids,
        "completion_text": text,
        "termination": "eos",
        "stop_reason": "completed",
        "max_tokens": 8,
        "truncated": False,
        "skip_reason": "",
    }


def test_unequal_turn_lengths_match_legacy_equal_turn_scalar_and_gradient():
    torch = pytest.importorskip("torch")
    student = torch.tensor(
        [
            [-0.4, -0.8, 0.0, 0.0, 0.0],
            [-0.2, -0.5, -0.7, -1.0, -1.3],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    teacher = torch.tensor(
        [
            [-0.6, -1.1, 0.0, 0.0, 0.0],
            [-0.3, -0.9, -0.9, -1.4, -1.4],
        ],
        dtype=torch.float64,
    )
    groups = torch.tensor([[0, 1, -1, -1, -1], [0, 1, 1, 2, 2]])
    response_mask = torch.tensor(
        [[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool
    )
    coefficient = 0.7

    values = _flash_groupwise_reverse_kl_values(
        student, teacher, groups, response_mask, coefficient
    )
    expanded_loss = (
        (values * response_mask).sum(dim=-1) / response_mask.sum(dim=-1)
    ).mean()
    expanded_gradient = torch.autograd.grad(
        expanded_loss, student, retain_graph=True
    )[0]

    legacy_turn_losses = [
        _gkd_loss_from_logps(
            student[0, :2],
            [([0], float(teacher[0, 0])), ([1], float(teacher[0, 1]))],
            kl_coef=coefficient,
        ),
        _gkd_loss_from_logps(
            student[1],
            [
                ([0], float(teacher[1, 0])),
                ([1, 2], float(teacher[1, 1])),
                ([3, 4], float(teacher[1, 3])),
            ],
            kl_coef=coefficient,
        ),
    ]
    legacy_loss = torch.stack(legacy_turn_losses).mean()
    legacy_gradient = torch.autograd.grad(legacy_loss, student, retain_graph=True)[0]

    flat_student = torch.cat([student[0, :2], student[1]]).detach().requires_grad_(True)
    flat_teacher = torch.cat([teacher[0, :2], teacher[1]])
    flat_groups = torch.tensor([[0, 1, 2, 3, 3, 4, 4]])
    flat_mask = torch.ones((1, 7), dtype=torch.bool)
    flat_values = _flash_groupwise_reverse_kl_values(
        flat_student.unsqueeze(0),
        flat_teacher.unsqueeze(0),
        flat_groups,
        flat_mask,
        coefficient,
    )
    token_weighted_loss = flat_values.mean()
    token_weighted_gradient = torch.autograd.grad(token_weighted_loss, flat_student)[0]

    assert float(expanded_loss.detach()) == pytest.approx(0.02765, abs=1e-12)
    assert torch.allclose(expanded_loss, legacy_loss, atol=1e-12, rtol=1e-12)
    assert torch.allclose(expanded_gradient, legacy_gradient, atol=1e-12, rtol=1e-12)
    assert not torch.allclose(expanded_loss, token_weighted_loss, atol=1e-12, rtol=1e-12)
    assert not torch.allclose(
        torch.cat([expanded_gradient[0, :2], expanded_gradient[1]]),
        token_weighted_gradient,
        atol=1e-12,
        rtol=1e-12,
    )


def test_bridge_scores_each_turn_against_its_pre_turn_snapshot_in_one_batch():
    bridge, env, teacher = _bridge()
    bridge.start_multiturn(
        index=0,
        session_id="episode",
        prompt_ids=[1, 2],
        raw_prompt=[{"role": "user", "content": "q"}],
    )
    first = _step_payload("episode", 0, [1, 2], "A")
    reply = bridge.step_multiturn(first)
    assert reply == {
        "messages": [{"role": "tool", "content": "obs-1"}],
        "terminal": False,
    }
    second_prefix = [1, 2, *first["response_ids"], 70, 71]
    second = bridge.step_multiturn(
        _step_payload("episode", 1, second_prefix, "BC")
    )
    assert second["terminal"] is True

    scored = bridge.score_multiturn("episode")["turns"]

    assert len(teacher.batches) == 1
    assert len(teacher.batches[0]) == 2
    first_prompt, first_completion = teacher.batches[0][0]
    second_prompt, second_completion = teacher.batches[0][1]
    assert first_completion == "A"
    assert second_completion == "BC"
    assert "Assistant: A" not in first_prompt
    assert "Tool: obs-1" not in first_prompt
    assert "Assistant: A" in second_prompt
    assert "Tool: obs-1" in second_prompt
    assert scored[0]["teacher_ids"][len(first["accepted_prefix"]) - 1] == 0
    assert scored[1]["teacher_ids"][len(second_prefix) - 1 : len(second_prefix) + 1] == [
        0,
        1,
    ]
    assert env.step_calls == [
        (id(env.states[0]), ("A",)),
        (id(env.states[0]), ("A", "BC")),
    ]


def test_bridge_serializes_tool_steps_and_isolates_rollout_sessions():
    bridge, env, _teacher = _bridge(max_turns=4)
    for session_id in ("left", "right"):
        bridge.start_multiturn(
            index=0,
            session_id=session_id,
            prompt_ids=[1, 2],
            raw_prompt=[{"role": "user", "content": "q"}],
        )
        bridge.step_multiturn(_step_payload(session_id, 0, [1, 2], session_id[0]))

    assert len(env.step_calls) == 2
    assert env.step_calls[0][0] != env.step_calls[1][0]
    assert [state["assistant"] for state in env.states] == [["l"], ["r"]]


def _make_loop(monkeypatch, generated_turns, step_replies, *, max_model_len=128, max_turns=4):
    tokenizer = _ChatTokenizer()
    calls = {"steps": [], "sampling": [], "closed": 0, "registry": {}}
    sessions = {"turns": []}

    class Output:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Base:
        def __init__(self):
            self.tokenizer = tokenizer
            self.rollout_config = SimpleNamespace(response_length=4)
            self.server_manager = Server()
            self.loop = asyncio.get_running_loop()

        async def apply_chat_template(self, messages):
            text = tokenizer.apply_chat_template(messages)
            return tokenizer(text, add_special_tokens=False).input_ids

    class Server:
        def __init__(self):
            self.turns = iter(generated_turns)

        async def generate(self, **kwargs):
            calls["sampling"].append(dict(kwargs["sampling_params"]))
            token_ids, stop_reason = next(self.turns)
            return SimpleNamespace(
                token_ids=list(token_ids),
                log_probs=[-0.1] * len(token_ids),
                stop_reason=stop_reason,
                num_preempted=0,
                extra_fields={},
            )

    def register(name):
        def decorate(cls):
            calls["registry"][name] = f"{cls.__module__}.{cls.__qualname__}"
            return cls

        return decorate

    def post_json(_url, _token, path, payload):
        if path == "/multiturn/start":
            return {"max_turns": max_turns}
        if path == "/multiturn/step":
            calls["steps"].append(payload)
            sessions["turns"].append(payload)
            return step_replies[len(calls["steps"]) - 1]
        if path == "/multiturn/score":
            return {
                "turns": [
                    {
                        "teacher_ids": [-1]
                        * (len(turn["accepted_prefix"]) + len(turn["response_ids"])),
                        "teacher_logprobs": [0.0]
                        * (len(turn["accepted_prefix"]) + len(turn["response_ids"])),
                    }
                    for turn in sessions["turns"]
                ]
            }
        if path == "/multiturn/close":
            calls["closed"] += 1
            return {"ok": True}
        raise AssertionError(path)

    monkeypatch.setenv("FLASH_OPD_BRIDGE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("FLASH_OPD_BRIDGE_TOKEN", "token")
    monkeypatch.setenv("FLASH_OPD_SEED", "42")
    monkeypatch.setenv("FLASH_OPD_MAX_TURNS", str(max_turns))
    monkeypatch.setenv("FLASH_OPD_MAX_MODEL_LEN", str(max_model_len))
    monkeypatch.setenv(
        "FLASH_OPD_ENV_CAPABILITIES",
        '["new_rollout_state","record_model_turn","env_reply","rollout_done"]',
    )
    monkeypatch.setenv("FLASH_OPD_STOP_SEQUENCES", "[]")
    monkeypatch.setenv("FLASH_OPD_EOS_TOKEN_IDS", f"[{ord('|')}]")
    monkeypatch.setenv("FLASH_OPD_THINKING", "0")
    loop_class = build_flash_multi_turn_agent_loop(
        register=register,
        agent_loop_base=Base,
        agent_loop_output=Output,
        post_json=post_json,
        deterministic_seed=deterministic_rollout_seed,
    )
    return loop_class, calls, tokenizer


def test_child_rollout_expands_two_turns_with_exact_order_and_seam_dedup(monkeypatch):
    loop_class, calls, tokenizer = _make_loop(
        monkeypatch,
        [([ord("A"), ord("|")], "completed"), ([ord("B"), ord("|")], "completed")],
        [
            {"messages": [{"role": "tool", "content": "obs"}], "terminal": False},
            {"messages": [], "terminal": True},
        ],
        max_turns=2,
    )

    async def run():
        loop = loop_class()
        return await loop.run(
            {"temperature": 1.0},
            raw_prompt=[{"role": "user", "content": "q"}],
            global_steps=3,
            index=7,
            session_id=1,
        )

    outputs = asyncio.run(run())
    initial = tokenizer(
        tokenizer.apply_chat_template([{"role": "user", "content": "q"}]),
        add_special_tokens=False,
    ).input_ids
    glue = tokenizer("t:obs|a:", add_special_tokens=False).input_ids

    assert len(outputs) == 2
    assert outputs[0].prompt_ids == initial
    assert outputs[0].response_ids == [ord("A"), ord("|")]
    assert outputs[0].response_mask == [1, 1]
    assert outputs[1].prompt_ids == [*initial, ord("A"), ord("|"), *glue]
    assert outputs[1].response_ids == [ord("B"), ord("|")]
    assert ord("o") in outputs[1].prompt_ids
    assert ord("o") not in outputs[0].response_ids + outputs[1].response_ids
    assert calls["sampling"][0]["seed"] != calls["sampling"][1]["seed"]
    assert "<locals>" not in calls["registry"]["flash_multi_turn"]
    assert calls["closed"] == 1


@pytest.mark.parametrize(
    ("generated", "reply", "max_model_len", "expected_turns", "expected_reason"),
    [
        (
            [([ord("A"), ord("B"), ord("C"), ord("D")], "completed")],
            [{"messages": [], "terminal": True}],
            128,
            1,
            "truncated_rollout",
        ),
        (
            [([ord("A"), ord("|")], "completed")],
            [{"messages": [], "terminal": False}],
            128,
            1,
            "",
        ),
        ([], [], len("u:q|a:") + 8, 0, None),
    ],
)
def test_child_rollout_handles_truncation_no_reply_and_context_exhaustion(
    monkeypatch,
    generated,
    reply,
    max_model_len,
    expected_turns,
    expected_reason,
):
    loop_class, calls, _tokenizer = _make_loop(
        monkeypatch,
        generated,
        reply,
        max_model_len=max_model_len,
        max_turns=2,
    )

    async def run():
        return await loop_class().run(
            {},
            raw_prompt=[{"role": "user", "content": "q"}],
            global_steps=0,
            index=0,
            session_id=0,
        )

    outputs = asyncio.run(run())
    assert len(outputs) == expected_turns
    if expected_reason is not None:
        assert calls["steps"][0]["skip_reason"] == expected_reason
    assert calls["closed"] == 1


def test_prepare_assistant_turn_rejects_cap_only_and_trims_stop_sequence():
    tokenizer = _ChatTokenizer()
    truncated = prepare_assistant_turn(
        tokenizer,
        [ord("A"), ord("B")],
        stop_reason="completed",
        max_tokens=2,
        eos_token_ids=frozenset({ord("|")}),
        stop_sequences=(),
    )
    stopped = prepare_assistant_turn(
        tokenizer,
        [ord("A"), ord("!")],
        stop_reason="completed",
        max_tokens=4,
        eos_token_ids=frozenset({ord("|")}),
        stop_sequences=("!",),
    )

    assert truncated["truncated"] is True
    assert truncated["skip_reason"] == "truncated_rollout"
    assert stopped["termination"] == "stop"
    assert stopped["response_ids"] == [ord("A")]
    assert stopped["completion_text"] == "A"


def test_variable_turn_output_normalization_and_no_signal_filtering():
    torch = pytest.importorskip("torch")
    teacher_ids = torch.tensor(
        [[-1, 0, -1], [-1, -1, -1], [-1, 2, -1]]
    ).unsqueeze(-1)

    assert _turn_outputs(["a0", "a1"]) == ["a0", "a1"]
    assert _turn_outputs("b0") == ["b0"]
    assert _full_sequence_signal_sequences(teacher_ids).tolist() == [True, False, True]


def test_multiturn_message_and_template_preflight_fail_closed():
    with pytest.raises(ValueError, match="unsupported transcript metadata"):
        validate_teacher_messages(
            [
                {
                    "role": "tool",
                    "content": "result",
                    "name": "calculator",
                    "tool_call_id": "call-1",
                }
            ],
            source="environment reply",
        )

    class BadTokenizer(_ChatTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return "template dropped the assistant probe"

    with pytest.raises(ValueError, match="assistant-content probe"):
        validate_glue_template(BadTokenizer(), thinking=False)


def test_env_glue_keeps_one_assistant_terminator_at_the_seam():
    tokenizer = _ChatTokenizer()
    glue = EnvGlueTokenizer(tokenizer, thinking=False)(
        [{"role": "tool", "content": "obs"}]
    )
    assert glue[0] == ord("|")
    assert tokenizer.decode(glue, skip_special_tokens=False) == "|t:obs|a:"


def test_lifecycle_accepts_bounded_multiturn_and_rejects_missing_contract(monkeypatch):
    from flash.envs import registry
    from flash.runner import lifecycle
    from flash.spec import EnvironmentSpec, JobSpec, TrainSpec

    class Environment:
        multi_turn = True
        is_tool_env = False
        max_turns = 2

        def dataset(self):
            return [{"input": "q"}]

        def prompt_messages(self, _record):
            return [{"role": "user", "content": "q"}]

        def new_rollout_state(self, _record):
            return {}

        def record_model_turn(self, _state, _content):
            return None

        def env_reply(self, _messages, _state):
            return []

        def rollout_done(self, _state, _max_turns):
            return True

    monkeypatch.setattr(registry, "load_environment", lambda *_args, **_kwargs: Environment())
    spec = JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="opd",
        environment=EnvironmentSpec(id="local", params={"multi_turn": True}),
        train=TrainSpec(max_examples=1),
    )
    lifecycle._preflight_opd_verl_environment(spec)

    Environment.max_turns = 0
    with pytest.raises(ValueError, match="positive bounded turn limit"):
        lifecycle._preflight_opd_verl_environment(spec)


def test_spec_rejects_structured_multiturn_but_not_plain_multiturn():
    from flash import runner
    from flash.spec import EnvironmentSpec, JobSpec, TrainSpec

    plain = JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="opd",
        environment=EnvironmentSpec(id="local", params={"multi_turn": True}),
    )
    runner._require_supported_opd_verl_spec(plain)

    structured = JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="opd",
        environment=EnvironmentSpec(id="local", params={"multi_turn": True}),
        train=TrainSpec(structured_outputs='{"choice":["4"]}'),
    )
    with pytest.raises(ValueError, match="per-turn constraint contract"):
        runner._require_supported_opd_verl_spec(structured)


def test_multimodal_multiturn_opd_remains_rejected():
    from flash.multimodal import validate_multimodal_training

    with pytest.raises(ValueError, match="only for single-turn"):
        validate_multimodal_training("Qwen/Qwen3.5-4B", "opd", multi_turn=True)
