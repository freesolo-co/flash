"""Engine-independent V1 conformance and the production generation/ingestion adapter."""

import asyncio
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from flash.engine.worker.train.core.child.rollout_contract import (
    REQUEST_FIELD,
    RESULT_FIELD,
    RolloutContractServer,
)
from flash.engine.worker.train.core.rollout.contract import (
    RolloutRequest,
    digest,
    token_digest,
    validate_result,
    validate_tokens,
)
from flash.engine.worker.train.core.rollout.episode import EPISODE_FIELD, validate_episode
from flash.engine.worker.train.core.rollout.identity import model_identity, policy_identity
from flash.engine.worker.train.core.rollout.lifecycle import PolicyLease
from flash.engine.worker.train.rl.child.contract import (
    ContractRolloutClient,
    validate_queue_rollouts,
)


def model_config():
    return SimpleNamespace(
        path="/models/snapshots/" + "a" * 40,
        runtime_options={"algorithm": "grpo", "rollout_run_id": "run-a"},
        tokenizer=SimpleNamespace(
            backend_tokenizer=SimpleNamespace(to_str=lambda: '{"vocab":{"a":1}}'),
            special_tokens_map={"eos_token": "</s>"},
            chat_template="{{ messages }}",
        ),
    )


def request(**changes):
    policy = policy_identity(model_identity(model_config()), 0)
    return replace(
        RolloutRequest(
            1,
            policy,
            1,
            7,
            0,
            "attempt-a",
            0,
            "request-a",
            digest([1]),
            "processed_logprobs",
        ),
        **changes,
    )


def result(req=None):
    return {
        "request": (req or request()).encode(),
        "load_ack": "server-ack",
        "termination": "stop",
        "sampling": {"temperature": 0.8, "top_p": 1.0},
        "token_digest": digest([[2, 3], [-0.2, -0.3]]),
    }


def test_strict_wire_roundtrip_and_version_separation():
    req = request()
    assert RolloutRequest.decode(req.encode()) == req
    assert req.protocol_version == 1
    assert req.policy.policy_version == 0
    for field, value in [
        ("protocol_version", 2),
        ("protocol_version", True),
        ("sample_index", True),
    ]:
        with pytest.raises(ValueError, match="rollout"):
            RolloutRequest.decode(dict(req.encode(), **{field: value}))
    with pytest.raises(ValueError, match="field set"):
        RolloutRequest.decode(dict(req.encode(), surprise="ignored?"))
    for field, value in [("base_revision", "main"), ("tokenizer_digest", "latest")]:
        wire = req.encode()
        wire["policy"][field] = value
        with pytest.raises(ValueError, match="rollout"):
            RolloutRequest.decode(wire)


@pytest.mark.parametrize(
    "logs",
    [
        None,
        [-0.2],
        [-0.2, -0.3, -0.4],
        [0.1, -0.3],
        [float("nan"), 0],
        [float("-inf"), 0],
        [True, 0],
    ],
)
def test_missing_shifted_and_invalid_logprobs_are_rejected(logs):
    with pytest.raises(ValueError, match="rollout"):
        validate_tokens([2, 3], logs)


@pytest.mark.parametrize(
    "change",
    [
        {"attempt_id": "old-attempt"},
        {"request_id": "old-request"},
        {"policy": replace(request().policy, policy_version=1)},
        {"policy": replace(request().policy, adapter_revision="different-weights")},
        {"policy": replace(request().policy, template_digest="f" * 64)},
        {"logprob_definition": "raw_logprobs"},
    ],
)
def test_result_must_match_policy_attempt_and_probability_definition(change):
    with pytest.raises(ValueError, match="does not match"):
        validate_result(request(), result(request(**change)), [2, 3], [-0.2, -0.3])


def test_result_rejects_cancellation_and_token_substitution():
    with pytest.raises(ValueError, match="cancelled"):
        validate_result(request(), dict(result(), termination="cancelled"), [], None)
    with pytest.raises(ValueError, match="digest"):
        validate_result(request(), result(), [3, 2], [-0.2, -0.3])


def test_load_ack_fences_inflight_requests_and_duplicate_retries():
    lease = PolicyLease()
    req = request()
    with pytest.raises(ValueError, match="not been acknowledged"):
        lease.admit(req)
    lease.acknowledge(req.policy)
    ack = lease.admit(req)
    with pytest.raises(ValueError, match="duplicate"):
        lease.admit(req)
    lease.invalidate()
    with pytest.raises(ValueError, match="changed during"):
        lease.finish(ack)
    with pytest.raises(ValueError, match="increase"):
        lease.acknowledge(req.policy)
    lease.acknowledge(replace(req.policy, policy_version=1, adapter_revision="run-a/policy/1"))
    with pytest.raises(ValueError, match="not been acknowledged"):
        lease.admit(replace(req, attempt_id="retry", request_id="retry"))


class Backend:
    async def collective_rpc(self, method, **kwargs):
        if self.fail_load:
            raise RuntimeError("load failed")

    async def set_global_steps(self, step):
        self.global_steps = step

    async def generate(self, prompt_ids, sampling_params, request_id, **kwargs):
        self.calls += 1
        if self.invalidate_during_generation:
            self._contract().invalidate()
        return SimpleNamespace(
            token_ids=[2, 3],
            log_probs=[-0.2, -0.3],
            stop_reason=self.stop_reason,
            extra_fields={
                "global_steps": self.global_steps,
                "finish_reason": "stop",
                "_flash_effective_sampling": {"temperature": 0.8, "max_tokens": 2},
            },
        )


class Server(RolloutContractServer, Backend):
    def __init__(self):
        self.model_config = model_config()
        self.config = SimpleNamespace(logprobs_mode="processed_logprobs")
        self.fail_load = False
        self.invalidate_during_generation = False
        self.calls = 0
        self.stop_reason = "completed"


async def load(server, step=0):
    await server.collective_rpc("update_weights_from_ipc")
    await server.set_global_steps(step)


async def generate(server, req=None):
    req = req or request()
    return await server.generate(
        prompt_ids=[1],
        sampling_params={"logprobs": True, REQUEST_FIELD: req.encode()},
        request_id=req.request_id,
    )


def test_server_only_acknowledges_successful_load_and_reports_actual_policy():
    async def scenario():
        server = Server()
        with pytest.raises(ValueError, match="successful weight loading"):
            await server.set_global_steps(0)
        with pytest.raises(ValueError, match="acknowledged"):
            await generate(server)
        server.fail_load = True
        with pytest.raises(RuntimeError, match="load failed"):
            await load(server)
        with pytest.raises(ValueError, match="acknowledged"):
            await generate(server)
        server.fail_load = False
        await load(server)
        output = await generate(server)
        validate_result(
            request(), output.extra_fields[RESULT_FIELD], output.token_ids, output.log_probs
        )
        assert server.calls == 1
        assert output.extra_fields[RESULT_FIELD]["sampling"]["max_tokens"] == 2
        with pytest.raises(ValueError, match="duplicate"):
            await generate(server)

    asyncio.run(scenario())


@pytest.mark.parametrize("cancelled", [False, True])
def test_server_update_and_cancellation_races_cannot_yield_trainable_results(cancelled):
    async def scenario():
        server = Server()
        await load(server)
        if cancelled:
            server.stop_reason = "aborted"
            output = await generate(server)
            with pytest.raises(ValueError, match="cancelled"):
                validate_result(
                    request(), output.extra_fields[RESULT_FIELD], output.token_ids, output.log_probs
                )
        else:
            server.invalidate_during_generation = True
            with pytest.raises(ValueError, match="changed during"):
                await generate(server)

    asyncio.run(scenario())


def episode_output():
    return SimpleNamespace(
        prompt_ids=[1],
        response_ids=[2, 3],
        response_logprobs=[-0.2, -0.3],
        response_mask=[1, 1],
        extra_fields={},
    )


def accepted_episode():
    async def scenario():
        server = Server()
        await load(server)
        client = ContractRolloutClient(server)
        model = model_config()
        agent = SimpleNamespace(
            config=SimpleNamespace(
                actor_rollout_ref=SimpleNamespace(model=model, rollout=server.config)
            ),
            tokenizer=model.tokenizer,
        )
        identity = {"optimizer_step": 1, "sample_index": 7, "rollout_ordinal": 0, "validate": False}
        with client.episode_contract(agent, identity):
            await client.generate(
                prompt_ids=[1], sampling_params={"logprobs": True}, request_id="req"
            )
            return client.annotate(episode_output())

    return asyncio.run(scenario())


def test_production_client_receipt_survives_float32_queue_transport_and_detects_changes():
    import struct

    output = accepted_episode()
    receipt = output.extra_fields[EPISODE_FIELD]
    logs = [struct.unpack("f", struct.pack("f", value))[0] for value in output.response_logprobs]
    validate_episode(receipt, [1], [2, 3], logs, [1, 1], run_id="run-a", optimizer_step=1)
    for prompt, response, mask in [
        ([9], [2, 3], [1, 1]),
        ([1], [3, 2], [1, 1]),
        ([1], [2, 3], [1, 0]),
    ]:
        with pytest.raises(ValueError, match="training tensors"):
            validate_episode(
                receipt, prompt, response, logs, mask, run_id="run-a", optimizer_step=1
            )
    with pytest.raises(ValueError, match="stale"):
        validate_episode(receipt, [1], [2, 3], logs, [1, 1], run_id="resumed-run", optimizer_step=1)


def test_assistant_provenance_is_independent_of_loss_eligibility():
    output = accepted_episode()
    receipt = output.extra_fields[EPISODE_FIELD]
    receipt["token_digest"] = token_digest([1], [2, 3, 9], [-0.2, -0.3, 0.0], [0, 0, 0])
    receipt["assistant_mask"] = [1, 1, 0]
    validate_episode(
        receipt, [1], [2, 3, 9], [-0.2, -0.3, 0.0], [0, 0, 0], run_id="run-a", optimizer_step=1
    )
    receipt["token_digest"] = token_digest([1], [2, 3, 9], [-0.2, -0.3, 0.0], [0, 0, 1])
    with pytest.raises(ValueError, match="environment tokens"):
        validate_episode(
            receipt, [1], [2, 3, 9], [-0.2, -0.3, 0.0], [0, 0, 1], run_id="run-a", optimizer_step=1
        )


def test_training_ingestion_rejects_duplicates_before_accepting_any_row(monkeypatch):
    from tests.test_verl_child_plugins import _modules

    class Row(list):
        def tolist(self):
            return list(self)

    output = accepted_episode()
    data = {
        "prompts": [Row([1])],
        "responses": [Row([2, 3])],
        "rollout_log_probs": [Row([-0.2, -0.3])],
        "response_mask": [Row([1, 1])],
        "extra_fields": [output.extra_fields],
        "index": [7],
        "session_id": [0],
        "global_steps": [1],
    }
    _modules(monkeypatch, {"transfer_queue": {"kv_batch_get": lambda **kwargs: data}})
    batch = SimpleNamespace(keys=["a"], partition_id="train")
    accepted = set()
    validate_queue_rollouts(
        batch,
        run_id="run-a",
        optimizer_step=1,
        accepted=accepted,
        expected_policy=request().policy,
        logprob_definition="processed_logprobs",
    )
    assert accepted == {(7, 0)}
    with pytest.raises(ValueError, match="duplicate logical"):
        validate_queue_rollouts(
            batch,
            run_id="run-a",
            optimizer_step=1,
            accepted=accepted,
            expected_policy=request().policy,
            logprob_definition="processed_logprobs",
        )
    data["index"][0] = 99
    with pytest.raises(ValueError, match="does not match its training row"):
        validate_queue_rollouts(
            batch,
            run_id="run-a",
            optimizer_step=1,
            accepted=set(),
            expected_policy=request().policy,
            logprob_definition="processed_logprobs",
        )
    data["extra_fields"][0] = {}
    with pytest.raises(ValueError, match="field set"):
        validate_queue_rollouts(
            batch,
            run_id="run-a",
            optimizer_step=1,
            accepted=set(),
            expected_policy=request().policy,
            logprob_definition="processed_logprobs",
        )


def test_turn_evidence_rejects_mixed_attempts_and_overlapping_boundaries():
    output = accepted_episode()
    receipt = output.extra_fields[EPISODE_FIELD]
    receipt["turns"].append(copy.deepcopy(receipt["turns"][0]))
    with pytest.raises(ValueError, match="overlap"):
        validate_episode(
            receipt, [1], [2, 3], [-0.2, -0.3], [1, 1], run_id="run-a", optimizer_step=1
        )


def test_tokenizer_identity_excludes_transient_padding_but_tracks_vocab_and_template():
    model = model_config()
    original = model_identity(model)
    model.tokenizer.backend_tokenizer.to_str = lambda: '{"padding":{"length":8},"vocab":{"a":1}}'
    assert model_identity(model) == original
    model.tokenizer.backend_tokenizer.to_str = lambda: '{"vocab":{"a":2}}'
    assert model_identity(model) != original
    model.tokenizer.chat_template = "different template"
    assert model_identity(model)[3] != original[3]


def test_concurrent_episode_scopes_do_not_mix_attempts_or_turns():
    async def scenario():
        server = Server()
        await load(server)
        client = ContractRolloutClient(server)
        agent = SimpleNamespace(
            config=SimpleNamespace(
                actor_rollout_ref=SimpleNamespace(model=server.model_config, rollout=server.config)
            ),
            tokenizer=server.model_config.tokenizer,
        )

        async def episode(index):
            identity = {
                "optimizer_step": 1,
                "sample_index": index,
                "rollout_ordinal": 0,
                "validate": False,
            }
            with client.episode_contract(agent, identity):
                await asyncio.sleep(0)
                await client.generate(
                    prompt_ids=[1], sampling_params={"logprobs": True}, request_id=f"req-{index}"
                )
                return client.annotate(episode_output()).extra_fields[EPISODE_FIELD]

        receipts = await asyncio.gather(episode(7), episode(8))
        requests = [
            RolloutRequest.decode(item["turns"][0]["result"]["request"]) for item in receipts
        ]
        assert [item.sample_index for item in requests] == [7, 8]
        assert requests[0].attempt_id != requests[1].attempt_id

    asyncio.run(scenario())


def test_multiturn_receipt_keeps_untrainable_assistant_turns_and_environment_gaps():
    async def scenario():
        server = Server()
        await load(server)
        client = ContractRolloutClient(server)
        agent = SimpleNamespace(
            config=SimpleNamespace(
                actor_rollout_ref=SimpleNamespace(model=server.model_config, rollout=server.config)
            ),
            tokenizer=server.model_config.tokenizer,
        )
        identity = {"optimizer_step": 1, "sample_index": 7, "rollout_ordinal": 0, "validate": False}
        with client.episode_contract(agent, identity):
            await client.generate(
                prompt_ids=[1], sampling_params={"logprobs": True}, request_id="first"
            )
            await client.generate(
                prompt_ids=[1, 2, 3, 9], sampling_params={"logprobs": True}, request_id="second"
            )
            output = SimpleNamespace(
                prompt_ids=[1],
                response_ids=[2, 3, 9, 2, 3],
                response_logprobs=[-0.2, -0.3, 0.0, -0.2, -0.3],
                response_mask=[1, 1, 0, 0, 0],
                extra_fields={
                    "flash_all_turn_spans": [(0, 2), (3, 5)],
                    "flash_episode_termination": "unusable_turn",
                },
            )
            receipt = client.annotate(output).extra_fields[EPISODE_FIELD]
            assert receipt["assistant_mask"] == [1, 1, 0, 1, 1]
            assert receipt["termination"] == "unusable_turn"
            assert len(receipt["turns"]) == 2
            receipt["turns"][1]["result"]["request"]["attempt_id"] = "superseded"
            with pytest.raises(ValueError, match="mixes policies or attempts"):
                validate_episode(
                    receipt,
                    output.prompt_ids,
                    output.response_ids,
                    output.response_logprobs,
                    output.response_mask,
                    run_id="run-a",
                    optimizer_step=1,
                )

    asyncio.run(scenario())


def test_server_rejects_unknown_protocol_and_ambiguous_greedy_probabilities_before_generation():
    async def scenario():
        server = Server()
        await load(server)
        params = {"logprobs": True, REQUEST_FIELD: dict(request().encode(), protocol_version=2)}
        with pytest.raises(ValueError, match="protocol version"):
            await server.generate(prompt_ids=[1], sampling_params=params, request_id="request-a")
        params[REQUEST_FIELD] = request().encode()
        params["temperature"] = 0.0
        with pytest.raises(ValueError, match="stochastic sampling"):
            await server.generate(prompt_ids=[1], sampling_params=params, request_id="request-a")
        assert server.calls == 0

    asyncio.run(scenario())


def test_caller_cancellation_cannot_reuse_an_admitted_request():
    class WaitingBackend(Backend):
        async def generate(self, **kwargs):
            self.started.set()
            await asyncio.Event().wait()

    class Waiting(RolloutContractServer, WaitingBackend):
        pass

    async def scenario():
        server = Waiting()
        server.model_config = model_config()
        server.config = SimpleNamespace(logprobs_mode="processed_logprobs")
        server.fail_load = False
        server.started = asyncio.Event()
        await load(server)
        task = asyncio.create_task(generate(server))
        await server.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(ValueError, match="duplicate"):
            await generate(server)

    asyncio.run(scenario())
