"""explicit FSDP engine selected by the model configuration in every training rank."""

import copy
from dataclasses import is_dataclass, replace

from verl.workers.engine import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

from flash.engine.worker.train.core.child.model import (
    configure_gdn,
    install_vision_input_grads,
    resolve_text_lora_target_modules,
)
from flash.engine.worker.train.core.child.options import RuntimeOptions
from flash.engine.worker.train.core.child.startup import initialize_process


@EngineRegistry.register(model_type="language_model", backend=["flash_fsdp", "flash_fsdp2"])
class FlashFSDPEngine(FSDPEngineWithLMHead):
    def __init__(self, *args, **kwargs):
        initialize_process()
        if kwargs["model_config"].runtime_options.get("algorithm") == "opd":
            from flash.engine.worker.train.opd.child import loss  # noqa: F401
        kwargs["engine_config"] = kwargs["engine_config"].native_config()
        super().__init__(*args, **kwargs)
        self.runtime = RuntimeOptions.decode(self.model_config.runtime_options)

    def _build_module(self):
        from flash.engine.worker.train.core.child.flashinfer_cache import activate_current

        activate_current()
        module = super()._build_module()
        options = self.runtime
        if options.reentrant_checkpointing:
            module.enable_input_require_grads()
            module.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": True}
            )
            if options.multimodal:
                install_vision_input_grads(module)
        if options.gdn_model_type:
            configure_gdn(module, options.gdn_model_type)
        return module

    def _build_lora_module(self, module):
        prefix = self.runtime.lora_language_prefix
        targets = None
        engine = self
        if prefix and self.model_config.lora_adapter_path is None:
            if (
                self.model_config.target_modules != "all-linear"
                or self.model_config.exclude_modules is not None
            ):
                raise RuntimeError(
                    "exact language LoRA targeting requires all-linear and no exclusion regex"
                )
            if not is_dataclass(self.model_config):
                raise RuntimeError(
                    "exact language LoRA targeting requires Verl's dataclass model config"
                )
            targets = resolve_text_lora_target_modules(module, prefix)
            engine = copy.copy(self)
            engine.model_config = replace(self.model_config, target_modules=targets)
        result = super(FlashFSDPEngine, engine)._build_lora_module(module)
        if targets is not None and set(result.base_model.targeted_module_names) != set(targets):
            raise RuntimeError("PEFT did not attach the complete exact language LoRA target set")
        if self.runtime.kl_ref_adapter:
            from flash.engine.worker.train.core.child.reference import snapshot

            result = snapshot(result)
        return result

    def disable_adapter(self):
        if not self.runtime.kl_ref_adapter:
            return super().disable_adapter()
        from flash.engine.worker.train.core.child.reference import use_ref_adapter

        return use_ref_adapter(self.module)

    def _build_lr_scheduler(self, optimizer):
        if self.runtime.algorithm != "sft":
            return super()._build_lr_scheduler(optimizer)
        from transformers import get_linear_schedule_with_warmup

        config = self.optimizer_config
        warmup = config.lr_warmup_steps
        if warmup <= 0:
            warmup = int(config.lr_warmup_steps_ratio * config.total_training_steps)
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup, num_training_steps=config.total_training_steps
        )

    def _build_optimizer(self, module):
        optimizer = super()._build_optimizer(module)
        if self.runtime.algorithm == "opd":
            import os

            from flash.engine.worker.train.opd.child.bridge import (
                _wrap_optimizer_with_mutation_notice,
            )

            _wrap_optimizer_with_mutation_notice(
                optimizer, os.environ["FLASH_OPD_BRIDGE_URL"], os.environ["FLASH_OPD_BRIDGE_TOKEN"]
            )
        return optimizer
