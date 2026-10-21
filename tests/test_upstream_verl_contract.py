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
        assert config.actor_rollout_ref.rollout.name == "flash_vllm"
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
