"""CPU integration contracts; run in an environment with the exact upstream VCS pin installed."""

import importlib
import importlib.util
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("verl") is None, reason="requires the pinned upstream Verl environment"
)


def _config(algorithm):
    from flash.engine.worker.train.core.child.launch import configure
    from flash.engine.worker.train.core.child.options import RuntimeOptions

    if algorithm == "sft":
        from tests.test_sft_train import _cfg, build_sft_overrides

        overrides = build_sft_overrides(_cfg(ulysses_sp_size=1))
    elif algorithm == "grpo":
        from flash.engine.worker.train.entry.rl_train import build_verl_overrides
        from tests.test_rl_train import _overrides_cfg

        overrides = build_verl_overrides(_overrides_cfg())
    else:
        from flash.engine.worker.train.opd.orchestration.overrides import build_opd_overrides
        from tests.test_opd_train import _config as opd_config

        overrides = build_opd_overrides(opd_config())
    return configure(RuntimeOptions(algorithm, 10, "ready"), overrides)


@pytest.mark.parametrize("algorithm", ["sft", "grpo", "opd"])
def test_production_overrides_validate_with_upstream(algorithm):
    from verl.utils.config import omega_conf_to_dataclass, validate_config

    from flash.engine.worker.train.core.child.configuration import FlashEngineConfig

    config = _config(algorithm)
    if algorithm == "sft":
        engine = omega_conf_to_dataclass(config.engine)
    else:
        validate_config(config, use_reference_policy=False, use_critic=False)
        actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
        engine = actor.engine
        assert config.trainer.use_v1
        assert config.transfer_queue.enable
        assert config.transfer_queue.backend.SimpleStorage.num_data_storage_units == 1
        assert config.actor_rollout_ref.rollout.name == "flash_vllm"
        assert config.actor_rollout_ref.rollout.engine_kwargs.vllm.worker_cls == (
            "flash.engine.worker.train.core.child.vllm_worker.FlashGPUWorker"
        )
    assert isinstance(engine, FlashEngineConfig)
    assert engine.strategy == "flash_fsdp2"
    assert engine.native_config().strategy == "fsdp2"


@pytest.mark.parametrize("algorithm", ["sft", "rl", "opd"])
def test_packaged_trainers_import_against_real_upstream(algorithm):
    importlib.import_module(f"flash.engine.worker.train.{algorithm}.child.trainer")
    if algorithm != "sft":
        importlib.import_module(f"flash.engine.worker.train.{algorithm}.child.rollout")


def test_opd_uses_only_student_pool_and_preserves_distillation_config():
    from verl.trainer.ppo.utils import Role

    from flash.engine.worker.train.opd.child.trainer import FlashPPOTrainer

    trainer = FlashPPOTrainer.__new__(FlashPPOTrainer)
    config = trainer.config = _config("opd")
    trainer._init_resource_pool_mgr()
    assert trainer.config is config
    assert config.distillation.enabled
    assert Role.TeacherModel not in trainer.mapping
    assert set(trainer.resource_pool_manager.resource_pool_spec) == {"global_pool"}
    assert trainer.get_teacher_client() is None


@pytest.mark.parametrize("nested", [False, True])
def test_grpo_turn_credit_roundtrips_upstream_dataproto(monkeypatch, nested):
    import torch
    from tensordict import NonTensorStack, TensorDict
    from transfer_queue import KVBatchMeta

    from flash.engine.worker.train.rl.child import trainer as recipe

    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    advantages = torch.ones(2, 3)
    if nested:
        mask = torch.nested.nested_tensor([mask[0, :2], mask[1]], layout=torch.jagged)
        advantages = torch.nested.nested_tensor(
            [advantages[0, :2], advantages[1]], layout=torch.jagged
        )
    fields = TensorDict(
        {
            "response_mask": mask,
            "advantages": advantages,
            "uid": NonTensorStack("same", "same"),
            "extra_fields": NonTensorStack(
                {"flash_turn_spans": [(0, 2)], "flash_turn_rewards": [1.0]},
                {"flash_turn_spans": [(0, 2), (2, 3)], "flash_turn_rewards": [3.0, 4.0]},
            ),
        },
        batch_size=2,
    )
    writes = []
    monkeypatch.setattr(recipe.tq, "kv_batch_get", lambda **kwargs: fields)
    monkeypatch.setattr(recipe.tq, "kv_batch_put", lambda **kwargs: writes.append(kwargs))
    batch = KVBatchMeta(keys=["a", "b"], tags=[{}, {}], partition_id="train")
    recipe.apply_queue_per_turn_credit(batch)
    actual = writes[0]["fields"]["advantages"]
    if nested:
        actual = actual.to_padded_tensor(0)
    torch.testing.assert_close(actual, torch.tensor([[-1.0, -1.0, 0], [1.0, 1.0, 0]]))
    assert writes[0]["keys"] == batch.keys


def test_grpo_rejects_mask_emptied_after_advantage_calculation(monkeypatch):
    import torch

    from flash.engine.worker.train.rl.child import trainer as recipe

    monkeypatch.setattr(
        recipe.tq, "kv_batch_get", lambda **kwargs: {"response_mask": torch.zeros(2, 3)}
    )
    trainer = recipe.FlashGrpoTrainer.__new__(recipe.FlashGrpoTrainer)
    with pytest.raises(RuntimeError, match="no trainable response tokens"):
        trainer._update_actor(SimpleNamespace(keys=["a", "b"], partition_id="train"), {})


@pytest.mark.parametrize("failed", [False, True])
def test_native_replay_returns_whole_groups_or_fails_before_consuming(monkeypatch, failed):
    from omegaconf import OmegaConf

    from flash.engine.worker.train.opd.child import replay as recipe

    config = _config("opd")
    buffer = recipe.BoundedReplayBuffer(
        SimpleNamespace(),
        trainer_mode="sync",
        trainer_config=config.trainer.v1.sync,
        max_off_policy_threshold=None,
        max_off_policy_strategy="drop",
        sampler_kwargs=OmegaConf.create({}),
    )
    rows = {
        "prompt": {
            "is_prompt": True,
            "global_steps": 1,
            "status": "failure" if failed else "finished",
        }
    }
    rows.update({f"prompt_{i}_0": {"status": "success", "global_steps": 1} for i in range(2)})
    import transfer_queue as tq

    clears = []
    monkeypatch.setattr(tq, "kv_list", lambda: {"train": rows})
    monkeypatch.setattr(tq, "kv_clear", lambda **kwargs: clears.append(kwargs))
    monkeypatch.setattr(recipe, "_actor_probe_state", lambda trainer: "unknown")
    if failed:
        with pytest.raises(RuntimeError, match="failed prompt"):
            buffer.sample(1, "train", 1)
        assert not clears
    else:
        batch, _ = buffer.sample(1, "train", 1)
        assert set(batch.keys) == {"prompt_0_0", "prompt_1_0"}
        assert clears == [{"partition_id": "train", "keys": ["prompt"]}]


@pytest.mark.parametrize("failure", [None, "queue", "init", "fit"])
@pytest.mark.parametrize("tracking_fails", [False, True])
def test_task_runner_closes_resources_without_masking_startup_failure(
    monkeypatch, failure, tracking_fails
):
    import transfer_queue as tq

    from flash.engine.worker.train.core.child import trainer as recipe

    events = []

    class Trainer:
        def __init__(self, config):
            pass

        def init(self):
            if failure == "init":
                raise ValueError("init failed")
            self._dump_executor = object()
            events.append("initialized")

        def get_llm_client(self):
            return None

        get_teacher_client = get_llm_client
        get_reward_handles = get_llm_client

        def fit(self, manager):
            def finish(exit_code):
                events.append(("finish", exit_code))
                if tracking_fails:
                    raise OSError("tracking unavailable")

            self.logger = SimpleNamespace(finish=finish)
            if failure == "fit":
                raise ValueError("fit failed")
            events.append("trained")

        def _shutdown_dump_executor(self):
            events.append("drained")

    original_import = recipe.importlib.import_module

    def import_module(name):
        if name == "flash.engine.worker.train.opd.child.trainer":
            return SimpleNamespace(FlashPPOTrainer=Trainer)
        if name == "flash.engine.worker.train.opd.child.rollout":
            return SimpleNamespace(
                FlashOpdManager=SimpleNamespace(create=lambda **kwargs: events.append("manager"))
            )
        return original_import(name)

    config = _config("opd")

    def init_queue(*args):
        events.append("queue")
        if failure == "queue":
            raise RuntimeError("queue initialization timed out")

    monkeypatch.setattr(recipe.importlib, "import_module", import_module)
    monkeypatch.setattr(recipe, "_init_transfer_queue", init_queue)
    monkeypatch.setattr(recipe, "publish_ready", lambda options: events.append("ready"))
    monkeypatch.setattr(tq, "close", lambda: events.append("closed"))
    if failure == "queue":
        with pytest.raises(RuntimeError, match="queue initialization timed out"):
            recipe.FlashTaskRunner().run(config)
        assert events == ["queue"]
        return
    if failure:
        with pytest.raises(ValueError, match=f"{failure} failed"):
            recipe.FlashTaskRunner().run(config)
    else:
        recipe.FlashTaskRunner().run(config)
    assert events[-1] == "closed"
    if failure == "init":
        assert events == ["queue", "closed"]
    else:
        assert events[:4] == ["queue", "initialized", "manager", "ready"]
        assert ("finish", int(failure is not None)) in events
        assert events[-2] == "drained"


@pytest.mark.parametrize("mask_truncated", [False, True])
@pytest.mark.parametrize("finish_reason", ["length", "stop"])
def test_grpo_agent_carries_identity_and_masks_only_truncation(
    monkeypatch, finish_reason, mask_truncated
):
    import asyncio

    from verl.experimental.agent_loop.agent_loop import AgentLoopOutput

    from flash.engine.worker.train.core.child.observations import ObservedRolloutClient
    from flash.engine.worker.train.rl.child import rollout as recipe

    events = []
    output = AgentLoopOutput(
        prompt_ids=[1],
        response_ids=[2, 3],
        response_mask=[1, 1],
        extra_fields={"finish_reason": finish_reason, "global_steps": 1},
        metrics={},
    )

    class BaseAgent:
        async def run(self, params, **kwargs):
            assert kwargs["extra_info"]["flash_rollout_identity"]["rollout_ordinal"] == 2
            events.append("generated")
            return await self.observations.generate()

    async def generate():
        return output

    class Agent(recipe.FlashGrpoPolicy, BaseAgent):
        pass

    agent = Agent.__new__(Agent)
    agent.config = _config("grpo")
    from flash.core.spec import TrainSpec
    from flash.engine.worker.train.rl.launch.config import grpo_mask_truncated_completions

    agent.config.actor_rollout_ref.model.runtime_options.mask_truncated_completions = (
        grpo_mask_truncated_completions(TrainSpec(mask_truncated_completions=mask_truncated))
    )

    class IdentityOnlyClient(ObservedRolloutClient):
        def episode_contract(self, agent, identity):
            return self.episode()

    agent.observations = IdentityOnlyClient(SimpleNamespace(generate=generate))
    agent.tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: "completion")
    monkeypatch.setenv("FLASH_VERL_REWARD_URL", "http://unused")
    monkeypatch.setattr(recipe, "post_json", lambda *args, **kwargs: events.append("registered"))

    def score(*args):
        events.append("scored")
        return 1.0

    monkeypatch.setattr(recipe, "compute_score", score)
    actual = asyncio.run(agent.run({}, global_steps=1, index=3, session_id=2))
    assert events == ["generated", "scored"]
    assert actual.response_mask == (
        [0, 0] if mask_truncated and finish_reason == "length" else [1, 1]
    )
    assert actual.reward_score == 1.0
    assert actual.extra_fields["min_global_steps"] == actual.extra_fields["max_global_steps"] == 1


def test_opd_teacher_facts_survive_native_agent_serialization(monkeypatch):
    import asyncio

    import torch
    from verl.experimental.agent_loop.agent_loop import AgentLoopOutput

    from flash.engine.worker.train.opd.child.teacher import compute_flash_teacher_logprobs

    ids = torch.tensor([[-1], [0], [1]], dtype=torch.int32)
    logs = torch.tensor([[0.0], [-0.4], [-0.6]])
    calls = []

    async def score(**kwargs):
        calls.append(kwargs)
        return ids, logs

    agent = SimpleNamespace(
        distillation_enabled=True,
        teacher_key="index",
        teacher_server_manager=SimpleNamespace(compute_teacher_logprobs_single=score),
    )
    output = AgentLoopOutput(
        prompt_ids=[1], response_ids=[2, 3], response_mask=[1, 1], reward_score=0.0, metrics={}
    )
    monkeypatch.delenv("FLASH_OPD_STRUCTURED_OUTPUTS", raising=False)
    asyncio.run(compute_flash_teacher_logprobs(agent, output, [1], [2, 3], False, {"index": 7}))
    asyncio.run(compute_flash_teacher_logprobs(agent, output, [1], [2, 3], False, {"index": 7}))
    assert len(calls) == 1
    assert calls[0]["routing_key"] == 7
    fields = output.as_dict()
    torch.testing.assert_close(fields["teacher_ids"], ids)
    torch.testing.assert_close(fields["teacher_logprobs"], logs)
    assert fields["rm_scores"].tolist() == [0.0, 0.0]


def test_grpo_manager_registers_whole_tensordict_batch_before_native_dispatch(monkeypatch):
    from tensordict import NonTensorData, NonTensorStack, TensorDict
    from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ

    from flash.engine.worker.train.rl.child import rollout as recipe
    from flash.engine.worker.train.rl.rollout.identity import RolloutIdentityLedger

    ledger = RolloutIdentityLedger(2, 2)
    registered = []

    def register(url, path, payload, **kwargs):
        ledger.register(payload["identities"])
        registered.extend(payload["identities"])

    def dispatch(self, prompts):
        assert len(registered) == 4
        for identity in registered:
            ledger.record(identity, identity["sample_index"])
        return "dispatched"

    config = _config("grpo")
    config.actor_rollout_ref.rollout.n = 2
    manager = recipe.FlashGrpoManager.__new__(recipe.FlashGrpoManager)
    manager.config = config
    prompts = TensorDict(
        {"index": NonTensorStack(10, 11), "global_steps": NonTensorData(17)}, batch_size=2
    )
    monkeypatch.setenv("FLASH_VERL_REWARD_URL", "http://unused")
    monkeypatch.setattr(recipe, "post_json", register)
    monkeypatch.setattr(AgentLoopManagerTQ, "generate_sequences", dispatch)
    assert manager.generate_sequences(prompts) == "dispatched"
    assert [(item["sample_index"], item["rollout_ordinal"]) for item in registered] == [
        (10, 0),
        (10, 1),
        (11, 0),
        (11, 1),
    ]
    ledger.seal(17)
    ledger.finalize({17})


@pytest.mark.parametrize("algorithm", ["grpo", "opd"])
def test_rollout_server_decodes_extended_model_config(monkeypatch, algorithm):
    from flash.engine.worker.train.core.child.configuration import FlashModelConfig
    from tests.test_verl_child_plugins import _import_recipe, _modules

    config = _config(algorithm)
    # Exercise native structured decoding without downloading weights or starting a GPU server.
    monkeypatch.setattr(FlashModelConfig, "__post_init__", lambda self: None)
    _modules(
        monkeypatch,
        {
            "verl.workers.rollout.vllm_rollout.utils": {"VLLM_LORA_INT_ID": 123},
            "verl.workers.rollout.vllm_rollout.vllm_async_server": {
                "vLLMHttpServer": object,
                "vLLMReplica": object,
            },
        },
    )
    recipe = _import_recipe(monkeypatch, "flash.engine.worker.train.core.child.rollout")
    decoded = recipe.FlashVLLMServer()._init_model_config(config.actor_rollout_ref.model)
    assert isinstance(decoded, FlashModelConfig)
    assert decoded.runtime_options["algorithm"] == algorithm
    assert decoded.path == config.actor_rollout_ref.model.path


def test_custom_rollout_replica_and_native_weight_sender_share_actor_name(monkeypatch):
    from tests.test_verl_child_plugins import _import_recipe, _modules

    # Skip CUDA package discovery; the sender's actual actor lookup needs no GPU operations.
    version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata, "version", lambda name: "0.19.1" if name == "vllm" else version(name)
    )
    _modules(monkeypatch, {"verl.third_party.vllm": {"VLLM_SLEEP_LEVEL": 1}})
    from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter

    _modules(
        monkeypatch,
        {
            "verl.workers.rollout.vllm_rollout.utils": {"VLLM_LORA_INT_ID": 123},
            "verl.workers.rollout.vllm_rollout.vllm_async_server": {
                "vLLMHttpServer": object,
                "vLLMReplica": object,
            },
        },
    )
    recipe = _import_recipe(monkeypatch, "flash.engine.worker.train.core.child.rollout")
    replica = recipe.FlashVLLMReplica.__new__(recipe.FlashVLLMReplica)
    sender = ServerAdapter.__new__(ServerAdapter)
    replica.config = sender.config = _config("grpo").actor_rollout_ref.rollout
    sender._has_server = True
    sender.server_handle = None
    sender._pd_role = None
    sender.replica_rank, sender.node_rank = 2, 1
    expected_name = f"{replica._get_server_name_prefix()}server_2_1"
    handle = object()
    actors = {expected_name: handle}
    monkeypatch.setattr("ray.get_actor", actors.__getitem__)
    assert sender._ensure_server_handle()
    assert sender.server_handle is handle


@pytest.mark.parametrize("compact", [False, True])
def test_native_adapter_export_accepts_full_and_compact_state(tmp_path, compact):
    import json

    import torch
    from safetensors.torch import load_file

    from flash.engine.worker.verl.adapter_merger import AdapterMerger

    (tmp_path / "lora_train_meta.json").write_text(
        json.dumps({"r": 2, "lora_alpha": 8, "task_type": "CAUSAL_LM"})
    )
    merger = AdapterMerger.__new__(AdapterMerger)
    merger.config = SimpleNamespace(local_dir=str(tmp_path), target_dir=str(tmp_path / "export"))
    prefix = "base_model.model.layers.0.q_proj"
    state = {
        f"{prefix}.lora_A.default.weight": torch.ones(2, 4),
        f"{prefix}.lora_B.default.weight": torch.full((4, 2), 3.0),
    }
    if not compact:
        state[f"{prefix}.base_layer.weight"] = torch.ones(4, 4)
    merger.save_hf_model_and_tokenizer(state)
    output = tmp_path / "export" / "lora_adapter"
    tensors = load_file(str(output / "adapter_model.safetensors"))
    assert set(tensors) == {f"{prefix}.lora_A.weight", f"{prefix}.lora_B.weight"}
    assert torch.equal(tensors[f"{prefix}.lora_B.weight"], torch.full((4, 2), 3.0))
    config = json.loads((output / "adapter_config.json").read_text())
    assert (config["r"], config["lora_alpha"], config["target_modules"]) == (2, 8, ["q_proj"])
    assert list((tmp_path / "export").iterdir()) == [output]
    with pytest.raises(RuntimeError, match="no LoRA"):
        merger.save_hf_model_and_tokenizer({})


@pytest.mark.parametrize("multimodal", [False, True])
def test_multiturn_prompt_uses_native_continuous_token_api(multimodal):
    import asyncio

    import numpy as np
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase

    from flash.engine.worker.train.core.child.glue import prepare_episode_prompt

    calls = []
    images = [object()] if multimodal else None

    class Builder:
        def supports_multimodal(self):
            return True

        def build_initial_tokens(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return [11, 12, 13]

    class Loop(AgentLoopBase):
        async def run(self, *args, **kwargs):
            raise NotImplementedError

        async def process_multi_modal_info(self, messages):
            return {"images": images} if images else {}

    async def prepare():
        loop = Loop.__new__(Loop)
        loop.loop = asyncio.get_running_loop()
        loop.processor = (
            SimpleNamespace(
                image_processor=lambda **kwargs: {
                    "pixel_values": np.zeros((1, 3), dtype=np.float32),
                    "image_grid_thw": np.ones((1, 3), dtype=np.int64),
                }
            )
            if multimodal
            else None
        )
        loop.continuous_token_builder = Builder()
        loop.mm_processor_kwargs = {}
        loop.rollout_config = SimpleNamespace(prompt_length=10)
        prompt = [{"role": "user", "content": "Name this color."}]
        result = await prepare_episode_prompt(loop, prompt)
        assert result.prompt_ids == [11, 12, 13]
        assert calls == [
            (
                prompt,
                {
                    "tools": None,
                    "images": images,
                    "videos": None,
                    "audios": None,
                    "mm_inputs_out": None,
                },
            )
        ]
        if multimodal:
            loop.processor = None
            with pytest.raises(ValueError, match="Multimodal inputs require"):
                await prepare_episode_prompt(loop, prompt)

    asyncio.run(prepare())


@pytest.mark.parametrize("lengths", [(89, 89, 89, 89), (17, 23, 19), (4,), (4, 4), (1, 1, 1, 1)])
@pytest.mark.parametrize("relabel", [False, True])
def test_queue_positions_preserve_axes_through_native_microbatch_selection(lengths, relabel):
    import torch
    from tensordict import TensorDict
    from transfer_queue.storage.managers.simple_storage_manager import AsyncSimpleStorageManager
    from verl.utils.tensordict_utils import index_select_tensor_dict, maybe_fix_3d_position_ids

    from flash.engine.worker.train.core.child.batch import normalize_position_ids

    rows = [torch.arange(4 * n).reshape(4, n) + i * 1000 for i, n in enumerate(lengths)]
    data = TensorDict(
        {
            "input_ids": torch.nested.as_nested_tensor(
                [torch.arange(n) for n in lengths], layout=torch.jagged
            ),
            "position_ids": AsyncSimpleStorageManager._pack_field_values(rows),
        },
        batch_size=len(rows),
    )
    if relabel:
        maybe_fix_3d_position_ids(data)
    normalize_position_ids(data)
    normalize_position_ids(data)  # repeated infer/ref/train admission preserves the same positions
    selected = index_select_tensor_dict(data, list(reversed(range(len(rows)))))
    for actual, expected in zip(selected["position_ids"].unbind(), reversed(rows), strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("method", ["infer_batch", "train_mini_batch"])
def test_flash_worker_decodes_positions_before_upstream_execution(monkeypatch, method):
    import torch
    from tensordict import TensorDict
    from transfer_queue.storage.managers.simple_storage_manager import AsyncSimpleStorageManager
    from verl.workers.engine_workers import TrainingWorker

    from flash.engine.worker.train.core.child.worker import (
        FlashActorRolloutWorker,
        FlashTrainingWorker,
    )

    expected = [torch.arange(28).reshape(4, 7), torch.arange(28, 56).reshape(4, 7)]
    data = TensorDict(
        {
            "input_ids": torch.nested.as_nested_tensor([torch.arange(7)] * 2, layout=torch.jagged),
            "position_ids": AsyncSimpleStorageManager._pack_field_values(expected),
        },
        batch_size=2,
    )

    def execute(self, actual):
        for row, reference in zip(actual["position_ids"].unbind(), expected, strict=True):
            torch.testing.assert_close(row, reference)
        return "native result"

    monkeypatch.setattr(TrainingWorker, method, execute)
    assert FlashActorRolloutWorker.actor_worker_cls is FlashTrainingWorker
    assert FlashActorRolloutWorker.ref_worker_cls is FlashTrainingWorker
    assert (
        getattr(FlashTrainingWorker.__new__(FlashTrainingWorker), method)(data) == "native result"
    )


def test_opd_multiturn_bridge_scores_skip_native_reward_recomputation(monkeypatch):
    import asyncio

    from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, AgentLoopWorker

    from flash.engine.worker.train.opd.child import rollout

    outputs = [
        AgentLoopOutput(
            prompt_ids=[1] * size,
            response_ids=[2],
            response_mask=[1],
            metrics={},
            extra_fields={"teacher_ids": [size], "teacher_logprobs": [-0.3]},
        )
        for size in (89, 111)
    ]

    async def run(self, sampling_params, **kwargs):
        return outputs

    monkeypatch.setattr(rollout._MultiTurn, "run", run)
    agent = rollout.FlashMultiTurnAgentLoop.__new__(rollout.FlashMultiTurnAgentLoop)

    async def verify():
        result = await agent.run({})
        # A configured reward service must not be invoked or reconstruct multimodal positions.
        worker = AgentLoopWorker.__new__(AgentLoopWorker)
        worker.reward_loop_worker_handles = [object()]
        await worker._compute_score(result, {})
        assert result is outputs
        for output in result:
            assert output.reward_score == 0.0
            assert output.extra_fields["teacher_logprobs"] == [-0.3]
            assert output.extra_fields["reward_extra_info"] == {}

    asyncio.run(verify())
