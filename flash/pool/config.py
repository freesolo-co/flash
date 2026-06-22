"""Runtime + provisioning config for the rollout pool, read from env (FLASH_POOL_*).

Two configs:
  * :class:`RouterConfig` — knobs for the running router (retries, default replica count, health
    probe cadence). Read by :func:`flash.pool.router.create_pool_app`.
  * :class:`PoolPlan` — what fleet to pre-rent (base model -> GPU class + count), read by
    :func:`flash.pool.provision.provision_pool`.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from flash.spec import strict_int as _spec_strict_int


def _int(name: str, default: int, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        val = int(raw.strip())
    except ValueError:
        return default
    # An out-of-range env value (e.g. a negative retry/replica count that would make the retry loop
    # skip every attempt) is treated like an unparseable one: fall back to the safe default.
    if minimum is not None and val < minimum:
        return default
    return val


def _float(name: str, default: float, minimum: float | None = None) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        val = float(raw.strip())
    except ValueError:
        return default
    # nan/inf parse fine but slip past the `minimum` guard (nan < x is always False) and break
    # downstream loops/sizing — treat a non-finite env value like an unparseable one.
    if not math.isfinite(val):
        return default
    if minimum is not None and val < minimum:
        return default
    return val


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
            max_retries=_int("FLASH_POOL_MAX_RETRIES", 2, minimum=0),
            default_replicas=_int("FLASH_POOL_DEFAULT_REPLICAS", 1, minimum=1),
            health_interval=_float("FLASH_POOL_HEALTH_INTERVAL", 15.0, minimum=0.0),
        )


@dataclass
class PoolMember:
    """One slice of the desired fleet: ``count`` GPUs of ``gpu`` class each serving ``base_model``."""

    base_model: str
    gpu: str  # managed GPU class label (see `flash gpus`), e.g. "RTX 5090" or "A100 PCIe"
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
                count=_strict_int(
                    m.get("count", 1), field="count", base_model=m.get("base_model"), minimum=1
                ),
                max_loras=_strict_int(
                    m.get("max_loras", 8), field="max_loras", base_model=m.get("base_model"), minimum=1
                ),
            )
            for m in data.get("pool", [])
        ]
        return cls(members=members)


def _strict_int(
    value: object, *, field: str, base_model: object = None, minimum: int | None = None
) -> int:
    """Coerce a TOML scalar to an int WITHOUT silently truncating (pool-config wording).

    Thin wrapper over the shared strict-int contract in flash.spec so the no-truncation /
    no-bool / whole-float-only rule lives in exactly one place; this just renders the
    pool-specific ``pool <field> [for base_model ...]`` message. ``minimum`` enforces a lower
    bound (e.g. 1 for count/max_loras, so count=0/-1 fails instead of yielding an empty fleet).
    """
    where = f" for base_model {base_model!r}" if base_model is not None else ""
    return _spec_strict_int(value, name=f"pool {field}{where}", minimum=minimum)
