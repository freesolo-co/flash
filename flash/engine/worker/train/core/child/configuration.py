"""Flash configuration types selected through upstream Hydra and engine registries."""

from dataclasses import dataclass, field, fields

from verl.workers.config import FSDPActorConfig, HFModelConfig
from verl.workers.config.engine import EngineConfig, FSDPEngineConfig


@dataclass
class FlashModelConfig(HFModelConfig):
    runtime_options: dict = field(default_factory=dict)

    def __post_init__(self):
        from flash.engine.worker.train.core.child import registration  # noqa: F401

        if self.runtime_options.get("algorithm") == "opd":
            from flash.engine.worker.train.opd.child import loss  # noqa: F401
        super().__post_init__()


@dataclass
class FlashEngineConfig(FSDPEngineConfig):
    strategy: str = "flash_fsdp2"

    def __post_init__(self):
        EngineConfig.__post_init__(self)
        if self.strategy not in {"flash_fsdp", "flash_fsdp2"}:
            raise ValueError(f"unsupported Flash engine: {self.strategy}")
        # importing the implementation registers a distinct backend, without replacing fsdp.
        from flash.engine.worker.train.core.child import engine  # noqa: F401

    def native_config(self):
        values = {item.name: getattr(self, item.name) for item in fields(FSDPEngineConfig)}
        values["strategy"] = self.strategy.removeprefix("flash_")
        return FSDPEngineConfig(**values)


@dataclass
class FlashActorConfig(FSDPActorConfig):
    def __post_init__(self):
        super().__post_init__()
        # upstream synchronizes actor.strategy into the engine; retain our registered selection.
        object.__setattr__(self.engine, "strategy", "flash_" + self.strategy)
