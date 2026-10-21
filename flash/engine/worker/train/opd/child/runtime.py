"""OPD-specific orchestration over the common copied child runtime."""

from __future__ import annotations

if __name__ == "flash_opd_runtime":
    import flash_verl_runtime as runtime
else:
    from flash.engine.worker.train.core.child import runtime


def install(config: dict) -> None:
    marker_file = str(config["marker_file"])
    language_prefix = str(config.get("lora_language_prefix") or "")
    if language_prefix:
        runtime.install_deferred_text_lora_targeting(language_prefix, marker_file)
    model_type = config.get("gdn_model_type")
    if model_type:
        runtime.install_deferred_gdn(str(model_type), marker_file)
    runtime.install_deferred_lora_rollout_guard(marker_file)
    if config.get("wandb"):
        runtime.install_wandb_link_reporting()


def install_checkpoint_schedule(trainer_type, config: dict) -> None:
    """Filter the synchronous PPO trainer after its teacher extensions are installed.

    OPD does not use SFT's CheckpointHandler. Importing its trainer during the
    earlier runtime setup would freeze the unpatched teacher methods into Ray actors.
    """
    required = frozenset(int(step) for step in config.get("save_at_steps", ()))
    if not required:
        return
    final_step = int(config["total_steps"])

    def patch() -> None:
        original = trainer_type._save_checkpoint
        if getattr(original, "_flash_exact_save_patched", False):
            return

        def save_checkpoint(self):
            if int(self.global_steps) not in required and int(self.global_steps) != final_step:
                return None
            return original(self)

        save_checkpoint._flash_exact_save_patched = True
        trainer_type._save_checkpoint = save_checkpoint

    runtime.install_required("opd-core", str(config["marker_file"]), patch)
