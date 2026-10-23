"""CPU contracts against the pinned, unmodified Verl; no CUDA or Ray cluster needed."""

import asyncio
import importlib.util
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("verl") is None, reason="requires pinned upstream Verl"
)


def _config():
    from flash.engine.worker.train.core.child.launch import configure
    from flash.engine.worker.train.core.child.options import RuntimeOptions
    from flash.engine.worker.train.rl.launch.pool import RolloutPool
    from flash.engine.worker.train.rl.launch.verl_config import build_verl_overrides
    from tests.test_rl_train import _overrides_cfg

    pool = RolloutPool(2).encode()
    options = RuntimeOptions("grpo", 2, "ready", rollout_pool=pool)
    overrides = build_verl_overrides(_overrides_cfg(n_gpus=1, rollout_pool=pool))
    return configure(options, overrides), options


def test_upstream_composes_and_admits_separate_sync_resources():
    from verl.trainer.ppo.utils import Role
    from verl.utils.config import validate_config

    from flash.engine.worker.train.rl.child.pool import FlashSeparateGrpoTrainer

    config, options = _config()
    # Ray receives nested OmegaConf mappings, even after the outer dict() call.
    assert type(options).decode(dict(config.actor_rollout_ref.model.runtime_options)) == options
    validate_config(config, use_reference_policy=False, use_critic=False)
    assert config.trainer.v1.trainer_mode == "sync"
    trainer = FlashSeparateGrpoTrainer.__new__(FlashSeparateGrpoTrainer)
    trainer.config, trainer.runtime = config, options
    trainer._init_resource_pool_mgr()
    assert set(trainer.role_worker_mapping) == {Role.Actor}
    assert Role.ActorRollout not in trainer.mapping
    assert Role.ActorRolloutRef not in trainer.mapping
    assert trainer.resource_pool_manager.resource_pool_spec == {"global_pool": [1]}
    assert trainer._get_n_gpus_for_throughput() == 3


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("actor_rollout_ref.hybrid_engine", True),
        ("actor_rollout_ref.rollout.load_format", "dummy"),
        ("actor_rollout_ref.rollout.n_gpus_per_node", 1),
        ("actor_rollout_ref.rollout.checkpoint_engine.backend", "naive"),
        ("ray_kwargs.ray_init.num_gpus", 1),
        ("actor_rollout_ref.model.lora.merge", True),
    ],
)
def test_pool_rejects_conflicting_hydra_config(key, value):
    from omegaconf import OmegaConf

    from flash.engine.worker.train.rl.launch.pool import validate_pool_config

    config, options = _config()
    OmegaConf.update(config, key, value, force_add=True)
    with pytest.raises(ValueError, match="separate rollout"):
        validate_pool_config(config, options)


@pytest.mark.parametrize("fail", [False, True])
def test_adapter_sender_exports_only_adapter_and_offloads_on_failure(fail):
    from flash.engine.worker.train.rl.child.pool_weights import PoolActorWorker

    worker = PoolActorWorker.__new__(PoolActorWorker)
    weights = [("layer.lora_A.weight", object())]
    engine = Mock(is_param_offload_enabled=True)
    engine.get_per_tensor_param.return_value = (weights, {"r": 8})
    worker.actor = SimpleNamespace(engine=engine)
    send = AsyncMock(side_effect=RuntimeError("transfer failed") if fail else None)
    worker.checkpoint_engine = SimpleNamespace(send_weights=send)
    if fail:
        with pytest.raises(RuntimeError, match="transfer failed"):
            asyncio.run(worker.update_weights(global_steps=7, mode="nccl"))
    else:
        asyncio.run(worker.update_weights(global_steps=7, mode="nccl"))
    engine.get_per_tensor_param.assert_called_once_with(layered_summon=True, base_sync_done=True)
    send.assert_awaited_once_with(weights, global_steps=7)
    engine.to.assert_called_once_with("cpu", model=True, optimizer=False, grad=False)


def test_receiver_requires_metadata_and_preserves_version_and_adapter_mode():
    from flash.engine.worker.train.rl.child.pool_weights import PoolCheckpointWorker

    worker = PoolCheckpointWorker.__new__(PoolCheckpointWorker)
    with pytest.raises(RuntimeError, match="metadata"):
        asyncio.run(worker.update_weights(global_steps=7))
    config = {"r": 8, "lora_alpha": 16, "target_modules": ["q_proj"]}
    worker.set_adapter_config(config)
    config["r"] = 99
    weights = object()
    worker.checkpoint_engine = SimpleNamespace(receive_weights=Mock(return_value=weights))
    worker.server_adapter = SimpleNamespace(update_weights=AsyncMock())
    asyncio.run(worker.update_weights(global_steps=7))
    worker.checkpoint_engine.receive_weights.assert_called_once_with(global_steps=7)
    worker.server_adapter.update_weights.assert_awaited_once_with(
        weights,
        global_steps=7,
        peft_config={"r": 8, "lora_alpha": 16, "target_modules": ["q_proj"]},
        base_sync_done=True,
    )


def test_receiver_registers_server_adapter_in_a_fresh_process():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from verl.checkpoint_engine.base import CheckpointEngineWorker\n"
                "from verl.workers.rollout.base import _ROLLOUT_REGISTRY\n"
                "from flash.engine.worker.train.rl.child.pool_weights import PoolCheckpointWorker\n"
                "def initialize(self):\n"
                "    assert ('flash_vllm', 'async') in _ROLLOUT_REGISTRY\n"
                "CheckpointEngineWorker.__init__ = initialize\n"
                "PoolCheckpointWorker()\n"
            ),
        ],
        check=True,
        timeout=60,
    )


def test_initial_publication_follows_restore_and_reaches_all_replicas(monkeypatch):
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    from flash.engine.worker.train.rl.child import pool

    events = []
    trainer = pool.FlashSeparateGrpoTrainer.__new__(pool.FlashSeparateGrpoTrainer)
    trainer.config, trainer.runtime = _config()
    trainer.global_steps = 7
    actor = SimpleNamespace(get_adapter_config=lambda: [{"r": 8}])
    receivers = [
        SimpleNamespace(set_adapter_config=SimpleNamespace(remote=Mock())) for _ in range(2)
    ]
    replicas = [SimpleNamespace(workers=[worker]) for worker in receivers]
    manager = SimpleNamespace(get_replicas=lambda: replicas)

    def restore(self):
        events.append("restore")
        self.actor_rollout_wg = actor
        self.llm_server_manager = SimpleNamespace(global_load_balancer="empty")

    def create(**kwargs):
        events.append("replicas")
        return manager

    checkpoint = SimpleNamespace(update_weights=lambda step: events.append(("publish", step)))
    monkeypatch.setattr(PPOTrainer, "_setup", restore)
    monkeypatch.setattr(pool.PoolServerManager, "create", create)
    monkeypatch.setattr(pool, "CheckpointEngineManager", Mock(return_value=checkpoint))
    monkeypatch.setattr(pool.ray, "kill", Mock())
    monkeypatch.setattr(pool.ray, "get", lambda refs: refs)
    trainer.init()
    assert events == ["restore", "replicas", ("publish", 7)]
    for receiver in receivers:
        receiver.set_adapter_config.remote.assert_called_once_with({"r": 8})
    assert pool.CheckpointEngineManager.call_args.kwargs["config"].backend == "nccl"
    assert pool.CheckpointEngineManager.call_args.kwargs["replicas"] == replicas


def test_sync_publication_failure_stops_step_and_dedicated_replicas_do_not_sleep():
    from flash.engine.worker.train.rl.child.pool import FlashSeparateGrpoTrainer

    trainer = FlashSeparateGrpoTrainer.__new__(FlashSeparateGrpoTrainer)
    trainer.global_steps, trainer.timing_raw, trainer.curr_step_profile = 3, {}, False
    trainer.checkpoint_manager = Mock()
    trainer.on_sample_end()
    trainer.checkpoint_manager.sleep_replicas.assert_not_called()
    trainer.checkpoint_manager.update_weights.side_effect = RuntimeError("replica unavailable")
    with pytest.raises(RuntimeError, match="replica unavailable"):
        trainer.on_step_end()
    trainer.checkpoint_manager.update_weights.assert_called_once_with(3)
