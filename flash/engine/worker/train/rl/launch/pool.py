"""Operator-only, fixed-capacity pilot of separate GRPO rollout GPUs.

The provider still rents and accounts for one allocation. This partitions that
allocation; it does not advertise cloud autoscaling or cross-job sharing.
"""

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

ROLLOUT_POOL_ENV = "FLASH_GRPO_ROLLOUT_POOL"


@dataclass(frozen=True)
class RolloutPool:
    gpus: int
    tensor_parallel: int = 1
    gpu_memory_utilization: float = 0.8

    def __post_init__(self):
        for name in ("gpus", "tensor_parallel"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 7:
                raise ValueError(f"rollout pool {name} must be an integer in 1..7")
        if self.gpus % self.tensor_parallel:
            raise ValueError("rollout pool gpus must be divisible by tensor_parallel")
        value = self.gpu_memory_utilization
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 0.9:
            raise ValueError("rollout pool gpu_memory_utilization must be in (0, 0.9]")

    @classmethod
    def decode(cls, value):
        if not isinstance(value, Mapping):
            raise ValueError("rollout pool must be an object")
        try:
            return cls(**value)
        except TypeError as exc:
            raise ValueError(f"invalid rollout pool: {exc}") from exc

    @classmethod
    def from_env(cls, environ):
        raw = environ.get(ROLLOUT_POOL_ENV, "")
        return cls.decode(json.loads(raw)) if raw else None

    def trainer_gpus(self, allocated_gpus):
        if type(allocated_gpus) is not int or not self.gpus < allocated_gpus <= 8:
            raise ValueError("separate rollout requires at least one training GPU within gpu.count")
        return allocated_gpus - self.gpus

    def encode(self):
        return asdict(self)

    @property
    def replicas(self):
        return self.gpus // self.tensor_parallel


def pool_overrides(pool):
    """Native Verl knobs; sync training remains selected by Flash's entrypoint."""
    return [
        "actor_rollout_ref.hybrid_engine=false",
        "actor_rollout_ref.rollout.nnodes=1",
        f"actor_rollout_ref.rollout.n_gpus_per_node={pool.gpus}",
        "actor_rollout_ref.rollout.checkpoint_engine.backend=nccl",
        # NCCL uses two staging buffers per participant in addition to vLLM's budget.
        "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=64",
    ]


def validate_pool_config(config, options):
    """Reject mismatched launch data before Ray reserves any GPUs."""
    if options.rollout_pool is None:
        if not config.actor_rollout_ref.hybrid_engine:
            raise ValueError("separate rollout requires Flash rollout_pool runtime options")
        return
    pool = RolloutPool.decode(options.rollout_pool)
    pool.trainer_gpus(options.dp_cards + pool.gpus)
    rollout = config.actor_rollout_ref.rollout
    model = config.actor_rollout_ref.model
    expected = {
        "trainer.nnodes": (config.trainer.nnodes, 1),
        "trainer.n_gpus_per_node": (config.trainer.n_gpus_per_node, options.dp_cards),
        "rollout.nnodes": (rollout.nnodes, 1),
        "rollout.n_gpus_per_node": (rollout.n_gpus_per_node, pool.gpus),
        "rollout.tensor_model_parallel_size": (
            rollout.tensor_model_parallel_size,
            pool.tensor_parallel,
        ),
        "rollout.data_parallel_size": (rollout.data_parallel_size, 1),
        "rollout.pipeline_model_parallel_size": (rollout.pipeline_model_parallel_size, 1),
        "rollout.gpu_memory_utilization": (
            rollout.gpu_memory_utilization,
            pool.gpu_memory_utilization,
        ),
        "hybrid_engine": (config.actor_rollout_ref.hybrid_engine, False),
        "rollout.checkpoint_engine.backend": (rollout.checkpoint_engine.backend, "nccl"),
        "rollout.free_cache_engine": (rollout.free_cache_engine, False),
        "rollout.load_format": (rollout.load_format, "safetensors"),
        "rollout.disaggregation.enabled": (rollout.disaggregation.enabled, False),
        "ray.num_gpus": (config.ray_kwargs.ray_init.num_gpus, options.dp_cards + pool.gpus),
    }
    for name, (actual, wanted) in expected.items():
        if actual != wanted:
            raise ValueError(f"separate rollout requires {name}={wanted}; got {actual}")
    if (
        model.lora_rank <= 0
        or model.lora.get("merge", False)
        or model.get("target_parameters")
        or options.algorithm != "grpo"
        or options.multimodal
    ):
        raise ValueError("separate rollout pilot requires dense text GRPO with unmerged LoRA")
