"""validated, serializable options for an explicitly constructed trainer runtime."""

from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class RuntimeOptions:
    algorithm: str
    total_steps: int
    readiness_path: str
    seed: int = 42
    save_at_steps: tuple[int, ...] = ()
    checkpoint_published_dir: str = ""
    reentrant_gradient_checkpointing: bool = False
    reentrant_checkpointing: bool = False
    lora_language_prefix: str = ""
    multimodal: bool = False
    gdn_model_type: str | None = None
    wandb: bool = False
    dp_cards: int = 1
    per_turn_credit: bool = False
    stop_sequences: tuple[str, ...] = ()
    image_pad_token_id: int | None = None
    structured_outputs: dict | None = None
    kl_ref_adapter: bool = False
    multi_turn: bool = False
    no_signal_attempts: int = 1

    @classmethod
    def decode(cls, value):
        unknown = set(value) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"unknown trainer runtime options: {sorted(unknown)}")
        options = cls(**value)
        if options.algorithm not in {"sft", "grpo", "opd"}:
            raise ValueError("unknown training algorithm")
        if type(options.total_steps) is not int or options.total_steps <= 0:
            raise ValueError("total_steps must be a positive integer")
        if options.no_signal_attempts <= 0 or options.dp_cards <= 0:
            raise ValueError("attempt and device counts must be positive")
        if any(
            type(step) is not int or not 0 < step <= options.total_steps
            for step in options.save_at_steps
        ):
            raise ValueError("checkpoint steps must be within the training horizon")
        return options

    def encode(self):
        return asdict(self)

    def should_save(self, step):
        return not self.save_at_steps or step in self.save_at_steps or step == self.total_steps
