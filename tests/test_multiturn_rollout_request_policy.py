"""focused tests for bounded physical rollout request policy."""

from __future__ import annotations

import sys
import threading
import types
from types import SimpleNamespace

import pytest

from flash.engine.multiturn_rollout import (
    RolloutRequestExhaustedError,
    build_rollout_func,
    resolve_rollout_request_timeout_seconds,
    rollout_async,
)
from flash.envs.base import BaseEnvironment


class _Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += float(seconds)


class _Env:
    multi_turn = True
    rollout_rewards_many = BaseEnvironment.rollout_rewards_many

    def __init__(self, turns=1):
        self.turns = turns
        self.record_calls = []

    def new_rollout_state(self, example):
        return {"prompt": [{"role": "user", "content": "hi"}], "completion": []}

    def record_model_turn(self, state, content):
        self.record_calls.append(content)
        state["completion"].append({"role": "assistant", "content": content})

    def rollout_done(self, state, max_turns):
        return len(self.record_calls) >= self.turns

    def env_reply(self, messages, state):
        reply = {"role": "user", "content": "next"}
        state["completion"].append(reply)
        return [reply]

    def reward(self, completion, example, state=None):
        return 1.0


def _render(messages, add_generation_prompt):
    return [1, 2, 3]


def _env_glue(messages):
    return [4]


def _completed(request_id, token=5, text="ok"):
    return request_id, [token], [-0.1], text


class _Tokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return "prompt"

    def __call__(self, *args, **kwargs):
        return SimpleNamespace(input_ids=[1, 2, 3])


@pytest.fixture
def _stub_vllm(monkeypatch):
    class _StructuredOutputsParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    mod = types.ModuleType("vllm")
    mod.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    sampling = types.ModuleType("vllm.sampling_params")
    sampling.RequestOutputKind = SimpleNamespace(FINAL_ONLY="final_only")
    sampling.StructuredOutputsParams = _StructuredOutputsParams
    mod.sampling_params = sampling
    monkeypatch.setitem(sys.modules, "vllm", mod)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling)


def test_managed_timeout_resolution_is_bounded():
    assert resolve_rollout_request_timeout_seconds(1024) == 600.0
    assert resolve_rollout_request_timeout_seconds(4096) == 2048.0
    assert resolve_rollout_request_timeout_seconds(8192) == 3600.0
    assert resolve_rollout_request_timeout_seconds(1_000_000) == 3600.0


@pytest.mark.usefixtures("_stub_vllm")
def test_retry_preserves_identity_ordering_and_fresh_sampling():
    clock = _Clock()
    env = _Env()
    operations = []
    submissions = []
    pending = set()
    first_step = True

    def add_request(request_id, prompt, sampling_params):
        operations.append(("submit", request_id))
        submissions.append((request_id, list(prompt["prompt_token_ids"]), sampling_params))
        pending.add(request_id)

    def abort_request(request_ids):
        for request_id in request_ids:
            operations.append(("abort", request_id))
            pending.discard(request_id)

    def step():
        nonlocal first_step
        if first_step:
            first_step = False
            clock.advance(6.0)
            return []
        clock.advance(0.1)
        pending.discard("attempt-2")
        stale = SimpleNamespace(
            request_id="attempt-1",
            finished=True,
            outputs=[SimpleNamespace(token_ids=[99], logprobs=None, text="stale")],
        )
        accepted = SimpleNamespace(
            request_id="attempt-2",
            finished=True,
            outputs=[SimpleNamespace(token_ids=[7], logprobs=None, text="accepted")],
        )
        return [stale, accepted]

    llm_engine = SimpleNamespace(
        model_config=SimpleNamespace(get_vocab_size=lambda: 1000),
        add_request=add_request,
        abort_request=abort_request,
        step=step,
        has_unfinished_requests=lambda: bool(pending),
    )
    engine = SimpleNamespace(llm_engine=llm_engine)
    trainer = SimpleNamespace(
        vllm_generation=SimpleNamespace(llm=engine),
        args=SimpleNamespace(vllm_enable_sleep_mode=False),
    )
    ids = iter(["attempt-1", "attempt-2"])
    structured = {"json": {"type": "object"}}
    rollout_func = build_rollout_func(
        active_env=env,
        tok=_Tokenizer(),
        examples_by_key={},
        max_completion=8,
        max_turns=1,
        temperature=0.7,
        top_p=1.0,
        stop=None,
        thinking=False,
        engine_max_len=100,
        structured_outputs=structured,
        request_timeout_seconds=5.0,
        request_max_attempts=2,
        monotonic=clock,
        request_id_factory=lambda: next(ids),
    )

    result = rollout_func([[{"role": "user", "content": "hi"}]], trainer)

    assert operations == [
        ("submit", "attempt-1"),
        ("abort", "attempt-1"),
        ("submit", "attempt-2"),
    ]
    assert [row[0] for row in submissions] == ["attempt-1", "attempt-2"]
    assert submissions[0][1] == submissions[1][1]
    first_params, second_params = submissions[0][2], submissions[1][2]
    assert first_params is not second_params
    assert first_params.max_tokens == second_params.max_tokens
    assert first_params.temperature == second_params.temperature
    assert first_params.top_p == second_params.top_p
    assert first_params.output_kind == second_params.output_kind == "final_only"
    assert first_params.structured_outputs is not second_params.structured_outputs
    assert first_params.structured_outputs.kwargs == second_params.structured_outputs.kwargs
    assert first_params.structured_outputs.kwargs == structured
    assert env.record_calls == ["accepted"]
    assert result["completion_ids"] == [[7]]


def test_exhaustion_aborts_each_attempt_and_raises():
    clock = _Clock()
    pending = set()
    submitted = []
    aborted = []
    ids = iter(["attempt-1", "attempt-2"])

    def submit(request_id, prefix, max_tokens, initial, images):
        submitted.append((request_id, list(prefix), max_tokens, initial))
        pending.add(request_id)

    def abort(request_ids):
        aborted.extend(request_ids)
        pending.difference_update(request_ids)

    def poll():
        clock.advance(6.0)
        return []

    with pytest.raises(RolloutRequestExhaustedError, match="exhausted 2 physical attempt"):
        rollout_async(
            examples=[{}],
            active_env=_Env(),
            render=_render,
            submit=submit,
            poll=poll,
            busy=lambda: bool(pending),
            abort=abort,
            env_glue=_env_glue,
            max_turns=1,
            per_turn_max_tokens=8,
            request_timeout_seconds=5.0,
            request_max_attempts=2,
            monotonic=clock,
            request_id_factory=lambda: next(ids),
        )

    assert aborted == ["attempt-1", "attempt-2"]
    assert submitted[0][1:] == submitted[1][1:]


def test_failure_aborts_all_active_requests():
    pending = set()
    aborted = []

    def submit(request_id, prefix, max_tokens, initial, images):
        pending.add(request_id)

    def abort(request_ids):
        aborted.extend(request_ids)
        pending.difference_update(request_ids)

    with pytest.raises(RuntimeError, match="engine step failed"):
        rollout_async(
            examples=[{}, {}],
            active_env=_Env(),
            render=_render,
            submit=submit,
            poll=lambda: (_ for _ in ()).throw(RuntimeError("engine step failed")),
            busy=lambda: bool(pending),
            abort=abort,
            env_glue=_env_glue,
            max_turns=1,
            per_turn_max_tokens=8,
        )

    assert len(aborted) == 2
    assert pending == set()


def test_submit_failure_best_effort_aborts_attempted_id():
    aborted = []

    with pytest.raises(RuntimeError, match="submit failed"):
        rollout_async(
            examples=[{}],
            active_env=_Env(),
            render=_render,
            submit=lambda *args: (_ for _ in ()).throw(RuntimeError("submit failed")),
            poll=lambda: [],
            busy=lambda: False,
            abort=lambda request_ids: aborted.extend(request_ids),
            env_glue=_env_glue,
            max_turns=1,
            per_turn_max_tokens=8,
            request_id_factory=lambda: "submit-failure-id",
        )

    assert aborted == ["submit-failure-id"]


def test_request_deadlines_do_not_create_episode_cutoff():
    clock = _Clock()
    env = _Env(turns=2)
    pending = []
    aborted = []

    def submit(request_id, prefix, max_tokens, initial, images):
        pending.append(request_id)

    def poll():
        clock.advance(4.0)
        return [_completed(pending.pop(0))]

    result = rollout_async(
        examples=[{}],
        active_env=env,
        render=_render,
        submit=submit,
        poll=poll,
        busy=lambda: bool(pending),
        abort=lambda ids: aborted.extend(ids),
        env_glue=_env_glue,
        max_turns=2,
        per_turn_max_tokens=8,
        request_timeout_seconds=5.0,
        monotonic=clock,
    )

    assert clock.value == 8.0
    assert len(result[0]["completion_ids"]) == 3
    assert aborted == []


def test_concurrent_rollouts_use_process_unique_ids():
    barrier = threading.Barrier(2)
    all_ids = [[], []]
    errors = []

    def run(index):
        pending = []

        def submit(request_id, prefix, max_tokens, initial, images):
            all_ids[index].append(request_id)
            pending.append(request_id)
            barrier.wait(timeout=2.0)

        try:
            rollout_async(
                examples=[{}],
                active_env=_Env(),
                render=_render,
                submit=submit,
                poll=lambda: [_completed(pending.pop())],
                busy=lambda: bool(pending),
                abort=lambda ids: None,
                env_glue=_env_glue,
                max_turns=1,
                per_turn_max_tokens=8,
            )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert set(all_ids[0]).isdisjoint(all_ids[1])
