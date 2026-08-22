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


def required_patch_specs(config: dict) -> tuple[tuple, ...]:
    """describe every fail-closed GRPO patch without importing its target modules."""
    entropy_quantile = config.get("entropy_quantile")
    model_type = config.get("gdn_model_type")
    return (
        (
            "rank-device-assert",
            int(config.get("dp_cards", 1)) > 1,
            "required",
            "verl.single_controller.base.worker",
            patches.install_rank_device_assert,
            (int(config.get("dp_cards", 1)),),
            {},
        ),
        (
            "nonempty-response-mask",
            True,
            "required",
            "verl.trainer.ppo.rollout_corr_helper",
            patches.install_nonempty_response_mask,
            (),
            {},
        ),
        (
            "exact-rollout-identity",
            True,
            "required",
            "verl.experimental.agent_loop.agent_loop",
            patches.install_exact_rollout_identity,
            (),
            {},
        ),
        (
            "reentrant-checkpointing",
            bool(config.get("reentrant_checkpointing")),
            "required",
            "verl.workers.engine.fsdp.transformer_impl",
            patches.install_reentrant_checkpointing,
            (),
            {"multimodal": bool(config.get("multimodal"))},
        ),
        (
            runtime.TEXT_LORA_TARGET_SHIM,
            bool(config.get("lora_language_prefix")),
            "required",
            "verl.workers.engine.fsdp.transformer_impl",
            runtime.install_text_lora_targeting,
            (str(config.get("lora_language_prefix") or ""),),
            {},
        ),
        (
            "entropy-quantile",
            entropy_quantile is not None and float(entropy_quantile) < 1.0,
            "required",
            "verl.workers.utils.losses",
            patches.install_entropy_quantile,
            (float(entropy_quantile) if entropy_quantile is not None else 1.0,),
            {},
        ),
        (
            "per-turn-credit",
            bool(config.get("per_turn_credit")),
            "required",
            "verl.trainer.ppo.ray_trainer",
            patches.install_per_turn_credit,
            (),
            {},
        ),
        (
            "stop-sequences",
            bool(config.get("stop_sequences")),
            "required",
            "verl.experimental.agent_loop.agent_loop",
            patches.install_stop_sequences,
            (tuple(config.get("stop_sequences", ())),),
            {},
        ),
        (
            "image-pad-ban",
            config.get("image_pad_token_id") is not None,
            "required",
            "verl.experimental.agent_loop.agent_loop",
            patches.install_image_pad_ban,
            (config.get("image_pad_token_id"),),
            {},
        ),
        (
            "structured-outputs",
            bool(config.get("structured_outputs")),
            "required",
            "verl.experimental.agent_loop.agent_loop",
            patches.install_structured_outputs,
            (config.get("structured_outputs"),),
            {},
        ),
        (
            "exact-save-steps",
            bool(config.get("save_at_steps")),
            "required",
            "verl.trainer.ppo.ray_trainer",
            patches.install_exact_save_steps,
            (tuple(config.get("save_at_steps", ())), int(config["total_steps"])),
            {},
        ),
        (
            "kl-ref-adapter",
            bool(config.get("kl_ref_adapter")),
            "required",
            "verl.workers.engine.fsdp.transformer_impl",
            patches.install_kl_ref_adapter,
            (),
            {},
        ),
        (
            "multi-turn-loop",
            bool(config.get("multi_turn")),
            "required",
            "verl.experimental.agent_loop.agent_loop",
            _install_multi_turn,
            (),
            {},
        ),
        (runtime.LORA_ROLLOUT_GUARD_SHIM, True, "lora", None, None, (), {}),
        (
            "gdn-varlen",
            bool(model_type),
            "gdn",
            None,
            None,
            (str(model_type),),
            {},
        ),
    )


def required_patch_names(config: dict) -> list[str]:
    return [name for name, enabled, *_rest in required_patch_specs(config) if enabled]


def install() -> None:
    config = runtime.load_plugin_config_file("FLASH_GRPO_PLUGIN_CONFIG_PATH")
    marker_file = str(config["marker_file"])
    for name, enabled, kind, target, installer, args, kwargs in required_patch_specs(config):
        if not enabled:
            continue
        if kind == "required":
            runtime.install_deferred_required(
                name,
                marker_file,
                target,
                installer,
                *args,
                **kwargs,
            )
        elif kind == "lora":
            runtime.install_deferred_lora_rollout_guard(marker_file)
        elif kind == "gdn":
            runtime.install_deferred_gdn(*args, marker_file)
        else:
            raise AssertionError(f"unknown GRPO patch kind: {kind}")
    if config.get("wandb"):
        runtime.install_wandb_link_reporting()


if PLUGIN_LOADED_EXTERNALLY:
    install()
