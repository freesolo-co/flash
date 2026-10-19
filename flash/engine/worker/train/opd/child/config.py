"""remote distillation configuration without a local teacher GPU pool."""

from dataclasses import dataclass

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import DistillationConfig, DistillationLossConfig


@dataclass
class FlashRemoteDistillationConfig(DistillationConfig):
    # verl's BaseConfig.__setattr__ rejects any assignment to a field that is not listed here
    # (base_config.py:33-38), so __post_init__ below can only normalize distillation_loss if the
    # field is declared mutable. verl's own DistillationConfig declares teacher_models for exactly
    # the same reason; it never reassigns distillation_loss, so it did not need to.
    _mutable_fields = DistillationConfig._mutable_fields | {"distillation_loss"}

    bridge_url: str = ""
    bridge_token: str = ""
    kl_penalty_coef: float = 1.0

    def __post_init__(self):
        if not self.enabled:
            return
        self.distillation_loss = omega_conf_to_dataclass(
            self.distillation_loss, dataclass_type=DistillationLossConfig
        )
        if self.distillation_loss.loss_mode != "flash_groupwise_reverse_kl":
            raise ValueError("flash remote distillation requires flash_groupwise_reverse_kl")
        if not self.bridge_url or not self.bridge_token:
            raise ValueError("flash remote distillation bridge configuration is missing")
        if self.kl_penalty_coef <= 0:
            raise ValueError("flash remote distillation kl_penalty_coef must be positive")
        self.teacher_models = {}
