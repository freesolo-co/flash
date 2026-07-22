"""cpu contracts for multi-turn OPD through verl 0.8.0."""

from __future__ import annotations

import asyncio
import threading
import time
from types import MethodType, SimpleNamespace

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
    _PERMANENT_TEACHER_EXIT,
    _TRANSIENT_TEACHER_EXIT,
    FlashTeacherBridgeError,
    _flash_groupwise_reverse_kl_values,
    _full_sequence_signal_sequences,
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


class _ProcessExit(BaseException):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


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
    glue = EnvGlueTokenizer(bridge.tokenizer, thinking=False)(reply["messages"])
    second_prefix = [
        1,
        2,
        *first["response_ids"],
        *glue[1:],
    ]
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


def test_bridge_rejects_every_forged_or_incomplete_environment_context():
    for case in (
        "missing_glue",
        "changed_glue",
        "extra_suffix",
        "dropped_assistant",
        "reordered_assistant",
    ):
        bridge, _env, _teacher = _bridge()
        bridge.start_multiturn(
            index=0,
            session_id=case,
            prompt_ids=[1, 2],
            raw_prompt=[{"role": "user", "content": "q"}],
        )
        first = _step_payload(case, 0, [1, 2], "AB")
        reply = bridge.step_multiturn(first)
        glue = EnvGlueTokenizer(bridge.tokenizer, thinking=False)(reply["messages"])[1:]
        valid = [1, 2, *first["response_ids"], *glue]
        if case == "missing_glue":
            forged = [1, 2, *first["response_ids"]]
        elif case == "changed_glue":
            forged = list(valid)
            forged[-1] += 1
        elif case == "extra_suffix":
            forged = [*valid, 70]
        elif case == "dropped_assistant":
            forged = [1, 2, *glue]
        else:
            forged = [1, 2, *reversed(first["response_ids"]), *glue]
        with pytest.raises(ValueError, match="exactly match the authenticated"):
            bridge.step_multiturn(_step_payload(case, 1, forged, "C"))


def test_per_example_turn_limit_executes_cap_turn_environment_transition_once():
    bridge, env, _teacher = _bridge(max_turns=4)
    bridge.prompts[0].example["turn_limit"] = 1
    start = bridge.start_multiturn(
        index=0,
        session_id="one-turn",
        prompt_ids=[1, 2],
        raw_prompt=[{"role": "user", "content": "q"}],
    )

    result = bridge.step_multiturn(_step_payload("one-turn", 0, [1, 2], "A"))

    assert start == {"max_turns": 1}
    assert result["terminal"] is True
    assert env.step_calls == [(id(env.states[0]), ("A",))]
    assert env.states[0]["assistant"] == ["A"]


def test_bridge_lifecycle_requests_are_idempotent_and_closed_sessions_cannot_resurrect():
    bridge, env, teacher = _bridge()
    kwargs = {
        "index": 0,
        "session_id": "idempotent",
        "prompt_ids": [1, 2],
        "raw_prompt": [{"role": "user", "content": "q"}],
    }

    assert bridge.start_multiturn(**kwargs) == {"max_turns": 2}
    assert bridge.start_multiturn(**kwargs) == {"max_turns": 2}
    assert len(env.states) == 1
    assert bridge.active_session_count == 1

    step = _step_payload("idempotent", 0, [1, 2], "A")
    first_step = bridge.step_multiturn(step)
    assert bridge.step_multiturn(step) == first_step
    assert env.step_calls == [(id(env.states[0]), ("A",))]
    assert env.states[0]["assistant"] == ["A"]

    first_score = bridge.score_multiturn("idempotent")
    assert bridge.score_multiturn("idempotent") == first_score
    assert len(teacher.batches) == 1
    assert len(teacher.batches[0]) == 1

    divergent = _step_payload("idempotent", 0, [1, 2], "B")
    with pytest.raises(ValueError, match="ordinal was reused"):
        bridge.step_multiturn(divergent)

    assert bridge.close_multiturn("idempotent") == {"ok": True}
    assert bridge.close_multiturn("idempotent") == {"ok": True}
    assert bridge.active_session_count == 0
    with pytest.raises(ValueError, match="already closed"):
        bridge.start_multiturn(**kwargs)


def test_periodic_session_reaper_bounds_abandoned_sessions():
    bridge, _env, _teacher = _bridge()
    bridge.session_lease_s = 0.02
    bridge.session_reap_interval_s = 0.005
    bridge.start()
    try:
        bridge.start_multiturn(
            index=0,
            session_id="abandoned",
            prompt_ids=[1, 2],
            raw_prompt=[{"role": "user", "content": "q"}],
        )
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            with bridge._sessions_lock:
                if not bridge._sessions:
                    break
            time.sleep(0.005)
        with bridge._sessions_lock:
            assert bridge._sessions == {}
            assert "abandoned" in bridge._session_tombstones
    finally:
        bridge.close()


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


def _make_loop(
    monkeypatch,
    generated_turns,
    step_replies,
    *,
    max_model_len=128,
    max_turns=4,
    phase_control=None,
    score_failure_index=None,
    process_exit=None,
):
    tokenizer = _ChatTokenizer()
    active_sessions = set()
    calls = {
        "steps": [],
        "sampling": [],
        "closed": 0,
        "registry": {},
        "active_sessions": active_sessions,
    }
    session_indexes = {}
    session_turns = {}

    def block_phase(phase):
        if phase_control is None or phase_control["phase"] != phase:
            return
        phase_control["entered"].set()
        assert phase_control["release"].wait(timeout=2.0)

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
            if phase_control is not None and phase_control["phase"] == "generation":
                phase_control["entered"].set()
                assert await asyncio.to_thread(phase_control["release"].wait, 2.0)
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
            active_sessions.add(payload["session_id"])
            session_indexes[payload["session_id"]] = payload["index"]
            session_turns[payload["session_id"]] = []
            block_phase("start")
            if phase_control is not None and phase_control["phase"] == "start_error":
                phase_control["entered"].set()
                raise FlashTeacherBridgeError(
                    "lost start response", classification="transient"
                )
            return {"max_turns": max_turns}
        if path == "/multiturn/step":
            block_phase("step")
            calls["steps"].append(payload)
            session_turns[payload["session_id"]].append(payload)
            return step_replies[len(calls["steps"]) - 1]
        if path == "/multiturn/score":
            block_phase("score")
            if session_indexes[payload["session_id"]] == score_failure_index:
                raise FlashTeacherBridgeError(
                    "teacher rejected one prompt", classification="permanent"
                )
            return {
                "turns": [
                    {
                        "teacher_ids": [-1]
                        * (len(turn["accepted_prefix"]) + len(turn["response_ids"])),
                        "teacher_logprobs": [0.0]
                        * (len(turn["accepted_prefix"]) + len(turn["response_ids"])),
                    }
                    for turn in session_turns[payload["session_id"]]
                ]
            }
        if path == "/multiturn/close":
            active_sessions.discard(payload["session_id"])
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
        permanent_teacher_exit=_PERMANENT_TEACHER_EXIT,
        transient_teacher_exit=_TRANSIENT_TEACHER_EXIT,
        process_exit=process_exit,
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
    for output in outputs:
        expected_length = len(output.prompt_ids) + len(output.response_ids)
        assert tuple(output.extra_fields["teacher_ids"].shape) == (expected_length, 1)
        assert tuple(output.extra_fields["teacher_logprobs"].shape) == (expected_length, 1)
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


@pytest.mark.parametrize("phase", ["start", "generation", "step", "score"])
def test_child_cancellation_closes_active_session_after_inflight_work(monkeypatch, phase):
    control = {
        "phase": phase,
        "entered": threading.Event(),
        "release": threading.Event(),
    }
    loop_class, calls, _tokenizer = _make_loop(
        monkeypatch,
        [([ord("A"), ord("|")], "completed")],
        [{"messages": [], "terminal": True}],
        max_turns=1,
        phase_control=control,
    )

    async def run_and_cancel():
        task = asyncio.create_task(
            loop_class().run(
                {},
                raw_prompt=[{"role": "user", "content": "q"}],
                global_steps=0,
                index=0,
                session_id=0,
            )
        )
        assert await asyncio.to_thread(control["entered"].wait, 1.0)
        task.cancel()
        control["release"].set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run_and_cancel())
    assert calls["active_sessions"] == set()
    assert calls["closed"] == 1


def test_child_lost_start_response_still_closes_session_before_transient_exit(monkeypatch):
    control = {
        "phase": "start_error",
        "entered": threading.Event(),
        "release": threading.Event(),
    }

    def process_exit(code):
        raise _ProcessExit(code)

    loop_class, calls, _tokenizer = _make_loop(
        monkeypatch,
        [],
        [],
        phase_control=control,
        process_exit=process_exit,
    )

    async def run():
        with pytest.raises(_ProcessExit) as error:
            await loop_class().run(
                {},
                raw_prompt=[{"role": "user", "content": "q"}],
                global_steps=0,
                index=0,
                session_id=0,
            )
        assert error.value.code == _TRANSIENT_TEACHER_EXIT

    asyncio.run(run())
    assert calls["active_sessions"] == set()
    assert calls["closed"] == 1


def test_one_multiturn_teacher_failure_exits_before_partial_batch_actor_update(monkeypatch):
    actor_updates = []

    def process_exit(code):
        raise _ProcessExit(code)

    loop_class, calls, _tokenizer = _make_loop(
        monkeypatch,
        [([ord("A"), ord("|")], "completed")],
        [
            {"messages": [], "terminal": True},
            {"messages": [], "terminal": True},
        ],
        max_turns=1,
        score_failure_index=1,
        process_exit=process_exit,
    )

    async def run_prompt(index):
        return await loop_class().run(
            {},
            raw_prompt=[{"role": "user", "content": "q"}],
            global_steps=0,
            index=index,
            session_id=0,
        )

    async def run_batch():
        results = await asyncio.gather(
            run_prompt(0),
            run_prompt(1),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, _ProcessExit):
                raise result
        actor_updates.append(True)

    with pytest.raises(_ProcessExit) as error:
        asyncio.run(run_batch())

    assert error.value.code == _PERMANENT_TEACHER_EXIT
    assert actor_updates == []
    assert calls["active_sessions"] == set()
    assert calls["closed"] == 2


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
        (
            [([ord("A"), ord("|")], "completed")],
            [{"messages": [], "terminal": True}],
            len("u:q|a:") + 4,
            1,
            "",
        ),
    ],
)
def test_child_rollout_handles_truncation_no_reply_and_exact_context_boundary(
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


@pytest.mark.opd_verl_backend
def test_verl_actor_postprocess_promotes_teacher_tensors_for_every_output(monkeypatch):
    import importlib.metadata

    torch = pytest.importorskip("torch")
    assert importlib.metadata.version("verl") == "0.8.0"
    assert importlib.metadata.version("ray")

    from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
    from verl.trainer import main_ppo_sync

    actor_class = main_ppo_sync.AgentLoopWorkerTQ
    worker_class = actor_class.__ray_metadata__.modified_class
    worker = object.__new__(worker_class)
    captured = {}
    teacher_calls = []

    async def compute_score(_self, _outputs, kwargs):
        assert kwargs["uid"] == "rollout"

    async def compute_teacher(_self, output, **_kwargs):
        teacher_calls.append(output)
        assert "teacher_ids" in output.extra_fields
        assert "teacher_logprobs" in output.extra_fields

    def compute_multi_modal_inputs(_self, _output, _input_ids):
        return None

    def compute_position_ids(_self, input_ids, _attention_mask, _multi_modal_inputs):
        return torch.arange(input_ids.shape[1], dtype=torch.int64).unsqueeze(0)

    async def async_kv_batch_put(**kwargs):
        captured.update(kwargs)

    worker._compute_score = MethodType(compute_score, worker)
    worker._compute_teacher_logprobs = MethodType(compute_teacher, worker)
    worker._compute_multi_modal_inputs = MethodType(compute_multi_modal_inputs, worker)
    worker._compute_position_ids = MethodType(compute_position_ids, worker)
    monkeypatch.setattr(
        main_ppo_sync,
        "tq",
        SimpleNamespace(async_kv_batch_put=async_kv_batch_put),
    )

    outputs = []
    for offset in (0, 10):
        teacher_ids = torch.tensor([[-1], [offset], [-1], [-1]], dtype=torch.int32)
        teacher_logprobs = torch.tensor(
            [[0.0], [-0.25 - offset], [0.0], [0.0]], dtype=torch.float32
        )
        outputs.append(
            AgentLoopOutput(
                prompt_ids=[1, 2],
                response_ids=[3 + offset, 4 + offset],
                response_mask=[1, 1],
                num_turns=1,
                metrics={
                    "generate_sequences": 0.0,
                    "tool_calls": 0.0,
                    "compute_score": 0.0,
                    "num_preempted": 0,
                },
                extra_fields={
                    "teacher_ids": teacher_ids,
                    "teacher_logprobs": teacher_logprobs,
                },
            )
        )

    asyncio.run(
        worker_class._agent_loop_postprocess(
            worker,
            outputs,
            False,
            uid="rollout",
            session_id=0,
            global_steps=1,
            raw_prompt=[{"role": "user", "content": "q"}],
        )
    )

    assert teacher_calls == [outputs[-1]]
    fields = captured["fields"]
    assert tuple(fields["teacher_ids"].shape) == (2, 4, 1)
    assert tuple(fields["teacher_logprobs"].shape) == (2, 4, 1)
    assert fields["teacher_ids"][0, 1, 0].item() == 0
    assert fields["teacher_ids"][1, 1, 0].item() == 10


def test_variable_turn_no_signal_filtering():
    torch = pytest.importorskip("torch")
    teacher_ids = torch.tensor(
        [[-1, 0, -1], [-1, -1, -1], [-1, 2, -1]]
    ).unsqueeze(-1)

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


def test_env_glue_probe_cannot_collide_with_environment_content():
    bridge, _env, _teacher = _bridge()
    messages = [
        {"role": "tool", "content": "result contains flash-env-glue-probe exactly once"}
    ]
    child_glue = EnvGlueTokenizer(bridge.tokenizer, thinking=False)(messages)
    assert bridge._env_glue is not None
    parent_glue = bridge._env_glue(messages)
    expected = bridge.tokenizer(
        "|t:result contains flash-env-glue-probe exactly once|a:",
        add_special_tokens=False,
    ).input_ids

    assert child_glue == parent_glue == expected


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
