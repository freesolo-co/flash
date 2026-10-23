"""Exercise the real pinned upstream transport and tensor serialization on CPU."""

import asyncio
import importlib.util
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("verl") is None, reason="requires pinned upstream Verl environment"
)


def test_request_identity_survives_upstream_routing_and_tensor_ingestion(monkeypatch):
    import torch
    import transfer_queue as tq
    from verl.experimental.agent_loop.agent_loop import AgentLoopOutput

    from flash.engine.worker.train.core.rollout.episode import EPISODE_FIELD
    from flash.engine.worker.train.rl.child.contract import (
        ContractRolloutClient,
        validate_queue_rollouts,
    )
    from flash.engine.worker.train.rl.child.transport import RolloutContractTransport
    from tests.test_versioned_rollout_contract import Server, episode_output, load, request

    server = Server()
    routed_ids = []

    async def generate(**kwargs):
        routed_ids.append(kwargs["request_id"])
        return await server.generate(**kwargs)

    class Transport(RolloutContractTransport):
        async def _acquire_server(self, request_id, **extra):
            return "server", SimpleNamespace(generate=SimpleNamespace(remote=generate))

        def _release_server(self, server_id, request_id=None):
            pass

    config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            model=server.model_config,
            rollout=SimpleNamespace(
                name="flash_vllm", full_determinism=False, logprobs_mode="processed_logprobs"
            ),
        )
    )
    client = ContractRolloutClient(Transport(config))
    agent = SimpleNamespace(config=config, tokenizer=server.model_config.tokenizer)
    identity = {"optimizer_step": 1, "sample_index": 7, "rollout_ordinal": 0, "validate": False}

    async def scenario():
        await load(server)
        with client.episode_contract(agent, identity):
            await client.generate(
                prompt_ids=[1], sampling_params={"logprobs": True}, request_id="caller-id"
            )
            output = AgentLoopOutput(**vars(episode_output()), reward_score=1.0, metrics={})
            return client.annotate(output)

    output = asyncio.run(scenario())
    assert routed_ids == ["caller-id"]
    fields = output.as_dict()
    assert fields["rollout_log_probs"].dtype == torch.float32
    assert EPISODE_FIELD in fields["extra_fields"]
    from verl.utils.tensordict_utils import list_of_dict_to_tensordict

    fields.update(index=7, session_id=0, global_steps=1)
    queued = list_of_dict_to_tensordict([fields])
    monkeypatch.setattr(tq, "kv_batch_get", lambda **kwargs: queued)
    validate_queue_rollouts(
        SimpleNamespace(keys=["a"], partition_id="train"),
        run_id="run-a",
        optimizer_step=1,
        accepted=set(),
        expected_policy=request().policy,
        logprob_definition="processed_logprobs",
    )


@pytest.mark.parametrize("cancelled", [False, True])
def test_effective_sampling_receipt_includes_omitted_defaults(cancelled):
    import msgspec

    from flash.engine.worker.train.core.child.rollout_contract import RolloutContractServer
    from tests.test_versioned_rollout_contract import model_config

    class Sampling(msgspec.Struct, omit_defaults=True):
        temperature: float = 1.0
        top_p: float = 1.0
        max_tokens: int = 8

    class Backend:
        def _postprocess_output(self, output, final_res, sampling_params):
            return output

    class Server(RolloutContractServer, Backend):
        pass

    server = Server()
    server.model_config = model_config()
    output = server._postprocess_output(
        SimpleNamespace(extra_fields={}, stop_reason="aborted" if cancelled else "completed"),
        SimpleNamespace(prompt_token_ids=None if cancelled else [1]),
        Sampling(),
    )
    assert ("_flash_engine_prompt_digest" in output.extra_fields) is not cancelled
    assert output.extra_fields["_flash_effective_sampling"] == {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 8,
    }


@pytest.mark.parametrize("engine_prompt", [[1, 8, 8, 2], [1, 8, 2], [1, 8, 8, 8, 2], None])
def test_multimodal_receipt_checks_final_engine_expansion(engine_prompt):
    from dataclasses import replace

    import msgspec
    from verl.utils.tokenizer import normalize_token_ids
    from verl.workers.rollout.utils import qwen2_5_vl_dedup_image_tokens

    from flash.engine.worker.train.core.child.rollout_contract import (
        REQUEST_FIELD,
        RESULT_FIELD,
        RolloutContractServer,
    )
    from flash.engine.worker.train.core.rollout.contract import digest, validate_result
    from tests.test_versioned_rollout_contract import Backend, Server, load, request

    class Sampling(msgspec.Struct):
        temperature: float = 0.8

    class Qwen2VLImageProcessor:
        pass

    class EngineBackend(Backend):
        async def generate(self, **kwargs):
            # Run the pinned upstream's real normalization and deduplication.
            normalized = normalize_token_ids(kwargs["prompt_ids"])
            submitted = qwen2_5_vl_dedup_image_tokens(normalized, self.model_config.processor)
            assert submitted == [1, 8, 2]
            output = await super().generate(**kwargs)
            output.extra_fields.pop("_flash_engine_prompt_digest")
            # GPU expansion is represented by RequestOutput's final conditioning tokens.
            final_res = SimpleNamespace(prompt_token_ids=engine_prompt)
            return self._postprocess_output(output, final_res, Sampling())

        def _postprocess_output(self, output, final_res, sampling_params):
            return output

    class EngineServer(RolloutContractServer, EngineBackend):
        __init__ = Server.__init__

    async def scenario():
        server = EngineServer()
        server.model_config.processor = SimpleNamespace(
            image_processor=Qwen2VLImageProcessor(), image_token_id=8, video_token_id=9
        )
        await load(server)
        prompt = [1, 8, 8, 2]
        req = replace(request(), prompt_digest=digest(prompt))
        output = await server.generate(
            prompt_ids=prompt,
            sampling_params={"logprobs": True, REQUEST_FIELD: req.encode()},
            request_id=req.request_id,
        )
        validate_result(req, output.extra_fields[RESULT_FIELD], [2, 3], [-0.2, -0.3])

    if engine_prompt == [1, 8, 8, 2]:
        asyncio.run(scenario())
    else:
        with pytest.raises(ValueError, match="conditioning tokens"):
            asyncio.run(scenario())


def test_trainer_validates_before_native_batch_padding(monkeypatch):
    from flash.engine.worker.train.core.child.trainer import FlashSyncTrainer
    from flash.engine.worker.train.rl.child import trainer as recipe
    from tests.test_versioned_rollout_contract import model_config

    events = []
    model = model_config()
    trainer = recipe.FlashGrpoTrainer.__new__(recipe.FlashGrpoTrainer)
    trainer.global_steps = 1
    trainer.tokenizer = model.tokenizer
    trainer.processor = None
    trainer.runtime = SimpleNamespace(rollout_run_id="run-a")
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            model=model,
            rollout=SimpleNamespace(logprobs_mode="processed_logprobs"),
        )
    )

    def validate(batch, **kwargs):
        assert kwargs["expected_policy"].policy_version == 0
        assert kwargs["optimizer_step"] == 1
        events.append("validated")

    def balance(self, batch, metrics, **kwargs):
        events.append("padded")
        return batch

    monkeypatch.setattr(recipe, "validate_queue_rollouts", validate)
    monkeypatch.setattr(FlashSyncTrainer, "_balance_batch", balance)
    batch = object()
    assert trainer._balance_batch(batch, {}) is batch
    assert events == ["validated", "padded"]

    def reject(batch, **kwargs):
        raise ValueError("stale policy")

    monkeypatch.setattr(recipe, "validate_queue_rollouts", reject)
    with pytest.raises(ValueError, match="stale policy"):
        trainer._balance_batch(batch, {})
    assert events == ["validated", "padded"]
