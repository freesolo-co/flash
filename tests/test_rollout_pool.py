"""Admission and resource accounting for the opt-in rollout pool pilot."""

import pytest

from flash.engine.worker.train.core.child.options import RuntimeOptions
from flash.engine.worker.train.rl.launch.pool import ROLLOUT_POOL_ENV, RolloutPool
from flash.engine.worker.train.rl.launch.verl_config import build_verl_overrides
from tests.test_rl_train import _overrides_cfg


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        {"gpus": True},
        {"gpus": 0},
        {"gpus": 8},
        {"gpus": "2"},
        {"gpus": 2, "tensor_parallel": 3},
        {"gpus": 2, "tensor_parallel": False},
        {"gpus": 2, "unknown": 1},
        {"gpus": 2, "gpu_memory_utilization": float("nan")},
        {"gpus": 2, "gpu_memory_utilization": 0.95},
        {"gpus": 2, "gpu_memory_utilization": True},
    ],
)
def test_pool_rejects_invalid_admission(value):
    with pytest.raises(ValueError, match="rollout pool"):
        RolloutPool.decode(value)


def test_pool_is_opt_in_and_preserves_training_capacity():
    assert RolloutPool.from_env({}) is None
    pool = RolloutPool.from_env({ROLLOUT_POOL_ENV: '{"gpus": 4, "tensor_parallel": 2}'})
    assert pool.replicas == 2
    assert pool.trainer_gpus(8) == 4
    assert RolloutPool.decode(pool.encode()) == pool
    with pytest.raises(ValueError, match="at least one"):
        pool.trainer_gpus(4)


@pytest.mark.parametrize(
    "changes", [{"algorithm": "opd"}, {"multimodal": True}, {"algorithm": "sft"}]
)
def test_runtime_rejects_unsupported_pool_models(changes):
    with pytest.raises(ValueError, match="dense text GRPO"):
        RuntimeOptions.decode(
            {
                "algorithm": "grpo",
                "total_steps": 2,
                "readiness_path": "ready",
                "rollout_pool": {"gpus": 1},
                **changes,
            }
        )


def test_pool_has_independent_tp_memory_and_total_ray_accounting():
    config = _overrides_cfg(n_gpus=2, rollout_pool=RolloutPool(4, 1, 0.75).encode())
    overrides = build_verl_overrides(config)
    assert "trainer.n_gpus_per_node=2" in overrides
    assert "+ray_kwargs.ray_init.num_gpus=6" in overrides
    assert "actor_rollout_ref.rollout.n_gpus_per_node=4" in overrides
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=1" in overrides
    assert "actor_rollout_ref.rollout.gpu_memory_utilization=0.75" in overrides
    assert "actor_rollout_ref.rollout.checkpoint_engine.backend=nccl" in overrides
    assert "actor_rollout_ref.hybrid_engine=false" in overrides
    keys = [item.split("=", 1)[0].lstrip("+") for item in overrides]
    assert len(keys) == len(set(keys))


def test_default_rollout_remains_colocated():
    overrides = build_verl_overrides(_overrides_cfg(n_gpus=2))
    assert "+ray_kwargs.ray_init.num_gpus=2" in overrides
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=2" in overrides
    assert not any("hybrid_engine" in item or "backend=nccl" in item for item in overrides)


@pytest.mark.parametrize(("allocated", "rollout", "expected_training"), [(4, 2, 2), (4, 1, 2)])
def test_resolved_resume_width_excludes_rollout_gpus(
    monkeypatch, worker_context, allocated, rollout, expected_training
):
    from tests.test_rl_train import _capability_env, _capability_resolve

    monkeypatch.setenv(ROLLOUT_POOL_ENV, f'{{"gpus":{rollout}}}')
    inputs = _capability_resolve(
        monkeypatch,
        _capability_env(),
        gpu_count=allocated,
        model="Qwen/Qwen3.5-9B",
        worker_context=worker_context,
    )
    # 4 prompts x 8 generations divide over 2 ranks, not 3. The resume loader
    # and trainer both consume this resolved width.
    assert inputs["dp_cards"] == expected_training
    assert inputs["rollout_pool"]["gpus"] == rollout


def test_pool_timings_survive_metric_stream_and_parent_parser():
    from flash.engine.worker.train.rl.child.metrics import _metric_line
    from flash.engine.worker.verl.child_io import parse_verl_step_metrics

    line = _metric_line(
        {
            "step": 7,
            "data": {
                "timing_s/gen": 10.0,
                "timing_s/update_actor": 2.0,
                "timing_s/update_weights": 0.4,
                "rollout_pool/gpus": 2,
                "rollout_pool/replicas": 2,
                "rollout_pool/training_gpus": 1,
                "rollout_pool/published_step": 7,
            },
        }
    )
    assert parse_verl_step_metrics(line) == {
        "step": 7,
        "rollout_seconds": 10.0,
        "actor_update_seconds": 2.0,
        "publication_seconds": 0.4,
        "rollout_gpus": 2,
        "rollout_replicas": 2,
        "training_gpus": 1,
        "published_step": 7,
    }
