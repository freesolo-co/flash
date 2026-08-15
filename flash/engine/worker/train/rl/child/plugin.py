"""child-side GRPO plugin loaded by verl's external-module hook."""

from __future__ import annotations

if __name__ == "flash_grpo_plugin":
    import flash_grpo_patches as patches
    import flash_verl_runtime as runtime
else:
    from flash.engine.worker.train.core.child import runtime
    from flash.engine.worker.train.rl.child import patches

PLUGIN_LOADED_EXTERNALLY = __name__ == "flash_grpo_plugin"


def _install_multi_turn() -> None:
    if __name__ == "flash_grpo_plugin":
        from flash_grpo_multiturn import build_flash_grpo_multi_turn_agent_loop
    else:
        from flash.engine.worker.train.rl.child.multiturn import (
            build_flash_grpo_multi_turn_agent_loop,
        )
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopBase,
        AgentLoopOutput,
        register,
    )

    build_flash_grpo_multi_turn_agent_loop(
        register=register,
        agent_loop_base=AgentLoopBase,
        agent_loop_output=AgentLoopOutput,
    )


def install() -> None:
    config = runtime.load_plugin_config_file("FLASH_GRPO_PLUGIN_CONFIG_PATH")
    marker_file = str(config["marker_file"])
    installers = (
        (
            "rank-device-assert",
            int(config.get("dp_cards", 1)) > 1,
            "verl.single_controller.base.worker",
            patches.install_rank_device_assert,
            (int(config.get("dp_cards", 1)),),
            {},
        ),
        (
            "reentrant-checkpointing",
            bool(config.get("reentrant_checkpointing")),
            "verl.workers.engine.fsdp.transformer_impl",
            patches.install_reentrant_checkpointing,
            (),
            {"multimodal": bool(config.get("multimodal"))},
        ),
        (
            "entropy-quantile",
            config.get("entropy_quantile") is not None and float(config["entropy_quantile"]) < 1.0,
            "verl.workers.utils.losses",
            patches.install_entropy_quantile,
            (float(config.get("entropy_quantile") or 1.0),),
            {},
        ),
        (
            "per-turn-credit",
            bool(config.get("per_turn_credit")),
            "verl.trainer.ppo.ray_trainer",
            patches.install_per_turn_credit,
            (),
            {},
        ),
        (
            "stop-sequences",
            bool(config.get("stop_sequences")),
            "verl.experimental.agent_loop.agent_loop",
            patches.install_stop_sequences,
            (tuple(config.get("stop_sequences", ())),),
            {},
        ),
        (
            "image-pad-ban",
            config.get("image_pad_token_id") is not None,
            "verl.experimental.agent_loop.agent_loop",
            patches.install_image_pad_ban,
            (config.get("image_pad_token_id"),),
            {},
        ),
        (
            "structured-outputs",
            bool(config.get("structured_outputs")),
            "verl.experimental.agent_loop.agent_loop",
            patches.install_structured_outputs,
            (config.get("structured_outputs"),),
            {},
        ),
        (
            "exact-save-steps",
            bool(config.get("save_at_steps")),
            "verl.trainer.ppo.ray_trainer",
            patches.install_exact_save_steps,
            (tuple(config.get("save_at_steps", ())), int(config["total_steps"])),
            {},
        ),
        (
            "kl-ref-adapter",
            bool(config.get("kl_ref_adapter")),
            "verl.workers.engine.fsdp.transformer_impl",
            patches.install_kl_ref_adapter,
            (),
            {},
        ),
    )
    for name, enabled, target, installer, args, kwargs in installers:
        if enabled:
            runtime.install_deferred_required(
                name,
                marker_file,
                target,
                installer,
                *args,
                **kwargs,
            )
    if config.get("multi_turn"):
        runtime.install_deferred_required(
            "multi-turn-loop",
            marker_file,
            "verl.experimental.agent_loop.agent_loop",
            _install_multi_turn,
        )
    runtime.install_deferred_lora_rollout_guard(marker_file)
    model_type = config.get("gdn_model_type")
    if model_type:
        runtime.install_deferred_gdn(str(model_type), marker_file)
    if config.get("wandb"):
        runtime.install_wandb_link_reporting()


if PLUGIN_LOADED_EXTERNALLY:
    install()
