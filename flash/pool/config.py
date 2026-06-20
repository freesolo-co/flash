"""Runtime + provisioning config for the rollout pool, read from env (FLASH_POOL_*).

Two configs:
  * :class:`RouterConfig` — knobs for the running router (retries, default replica count, health
    probe cadence). Read by :func:`flash.pool.router.create_pool_app`.
  * :class:`PoolPlan` — what fleet to pre-rent (base model -> GPU class + count), read by
    :func:`flash.pool.provision.provision_pool`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


@dataclass
class RouterConfig:
    # Retries across DISTINCT backends when a generation fails (failover). 0 = no failover.
    max_retries: int = 2
    # Default warm replicas per adapter when an adapter is registered without an explicit count.
    default_replicas: int = 1
    # Seconds between background health probes; 0 disables the loop (tests, single-shot use).
    health_interval: float = 15.0

    @classmethod
    def from_env(cls) -> RouterConfig:
        return cls(
            max_retries=_int("FLASH_POOL_MAX_RETRIES", 2),
            default_replicas=_int("FLASH_POOL_DEFAULT_REPLICAS", 1),
            health_interval=_float("FLASH_POOL_HEALTH_INTERVAL", 15.0),
        )


@dataclass
class PoolMember:
    """One slice of the desired fleet: ``count`` GPUs of ``gpu`` class each serving ``base_model``."""

    base_model: str
    gpu: str  # managed GPU class label (see `slm gpus`), e.g. "RTX5090"
    count: int = 1
    max_loras: int = 8


@dataclass
class PoolPlan:
    members: list[PoolMember] = field(default_factory=list)

    @classmethod
    def from_toml(cls, path: str) -> PoolPlan:
        import tomllib

        with open(path, "rb") as f:
            data = tomllib.load(f)
        members = [
            PoolMember(
                base_model=m["base_model"],
                gpu=m["gpu"],
                count=int(m.get("count", 1)),
                max_loras=int(m.get("max_loras", 8)),
            )
            for m in data.get("pool", [])
        ]
        return cls(members=members)
