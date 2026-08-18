"""OPD-specific orchestration over the common copied child runtime."""

from __future__ import annotations

if __name__ == "flash_opd_runtime":
    import flash_verl_runtime as runtime
else:
    from flash.engine.worker.train.core.child import runtime


def install(config: dict) -> None:
    marker_file = str(config["marker_file"])
    save_at_steps = tuple(config.get("save_at_steps", ()))
    if save_at_steps:
        runtime.install_required(
            "opd-core",
            marker_file,
            runtime.install_checkpoint_handler_filter,
            save_at_steps,
            int(config["total_steps"]),
        )
    language_prefix = str(config.get("lora_language_prefix") or "")
    if language_prefix:
        runtime.install_deferred_text_lora_targeting(language_prefix, marker_file)
    model_type = config.get("gdn_model_type")
    if model_type:
        runtime.install_deferred_gdn(str(model_type), marker_file)
    runtime.install_deferred_lora_rollout_guard(marker_file)
    if config.get("wandb"):
        runtime.install_wandb_link_reporting()
