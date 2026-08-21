"""child-side SFT runtime patches loaded by verl's external-module hook."""

from __future__ import annotations

from importlib import import_module

if __name__ == "flash_sft_plugin":
    import flash_verl_runtime as runtime
else:
    from flash.engine.worker.train.core.child import runtime

PLUGIN_LOADED_EXTERNALLY = __name__ == "flash_sft_plugin"


def _install_exact_dataloaders() -> None:
    from torch.utils.data.distributed import DistributedSampler
    from torchdata.stateful_dataloader import StatefulDataLoader

    sampler_init = DistributedSampler.__init__
    if not getattr(sampler_init, "_flash_exact_sampler", False):

        def exact_sampler_init(self, *args, **kwargs):
            kwargs["shuffle"] = False
            return sampler_init(self, *args, **kwargs)

        exact_sampler_init._flash_exact_sampler = True
        DistributedSampler.__init__ = exact_sampler_init

    loader_init = StatefulDataLoader.__init__
    if not getattr(loader_init, "_flash_exact_loader", False):

        def exact_loader_init(self, *args, **kwargs):
            kwargs["drop_last"] = False
            return loader_init(self, *args, **kwargs)

        exact_loader_init._flash_exact_loader = True
        StatefulDataLoader.__init__ = exact_loader_init


def _install_seeded_dataloader(seed: int) -> None:
    import random

    import numpy as np
    import torch
    from verl.trainer import sft_trainer

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    original = sft_trainer.SFTTrainer._build_dataloader
    if getattr(original, "_flash_seeded_dataloader", False):
        return

    def seeded_build_dataloader(self):
        result = original(self)
        self.train_sampler.seed = seed
        val_sampler = getattr(self, "val_sampler", None)
        if val_sampler is not None:
            val_sampler.seed = seed
        return result

    seeded_build_dataloader._flash_seeded_dataloader = True
    sft_trainer.SFTTrainer._build_dataloader = seeded_build_dataloader


def _install_linear_scheduler() -> None:
    from transformers import get_linear_schedule_with_warmup
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    def build_lr_scheduler(self, optimizer):
        config = self.optimizer_config
        warmup_steps = config.lr_warmup_steps
        if warmup_steps <= 0:
            warmup_steps = int(config.lr_warmup_steps_ratio * config.total_training_steps)
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=config.total_training_steps,
        )

    FSDPEngine._build_lr_scheduler = build_lr_scheduler


def _install_reentrant_checkpointing(*, multimodal: bool) -> None:
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    original = FSDPEngine._build_module
    if getattr(original, "_flash_reentrant_sft", False):
        return

    def build_reentrant_module(self):
        module = original(self)
        module.enable_input_require_grads()
        module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        if multimodal:
            runtime.install_vision_input_grads(module)
        return module

    build_reentrant_module._flash_reentrant_sft = True
    FSDPEngine._build_module = build_reentrant_module


def _install_loraplus(ratio: float, ready_marker: str) -> None:
    if ratio <= 1:
        return
    from peft.optimizers import create_loraplus_optimizer
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    def build_loraplus_optimizer(self, module):
        config = self.optimizer_config
        optimizer_cls = getattr(import_module(config.optimizer_impl), config.optimizer)
        optimizer_kwargs = {"lr": config.lr, "weight_decay": config.weight_decay}
        if "adam" in config.optimizer.lower() or "ademamix" in config.optimizer.lower():
            optimizer_kwargs["betas"] = config.betas
        if config.override_optimizer_config:
            optimizer_kwargs.update(config.override_optimizer_config)
        try:
            optimizer = create_loraplus_optimizer(
                model=module,
                optimizer_cls=optimizer_cls,
                optimizer_kwargs=optimizer_kwargs,
                loraplus_lr_ratio=ratio,
                loraplus_weight_decay=config.weight_decay,
            )
        except TypeError:
            optimizer = create_loraplus_optimizer(
                model=module,
                optimizer_cls=optimizer_cls,
                lr=config.lr,
                loraplus_lr_ratio=ratio,
                loraplus_weight_decay=config.weight_decay,
                **{key: value for key, value in optimizer_kwargs.items() if key != "lr"},
            )
        print(
            f"{ready_marker} ratio={ratio:g} optimizer={optimizer_cls.__name__}",
            flush=True,
        )
        return optimizer

    FSDPEngine._build_optimizer = build_loraplus_optimizer


def _install_core(config: dict) -> None:
    seed = int(config["seed"])
    _install_exact_dataloaders()
    _install_seeded_dataloader(seed)
    _install_linear_scheduler()
    save_at_steps = tuple(config.get("save_at_steps", ()))
    if save_at_steps:
        runtime.install_checkpoint_handler_filter(save_at_steps, int(config["total_steps"]))
    if config.get("reentrant_gradient_checkpointing"):
        _install_reentrant_checkpointing(multimodal=bool(config.get("multimodal")))
    _install_loraplus(
        float(config.get("loraplus_ratio", 1.0)),
        str(config["loraplus_ready_marker"]),
    )


def install() -> None:
    config = runtime.load_plugin_config("FLASH_SFT_PLUGIN_CONFIG")
    marker_file = str(config["marker_file"])
    runtime.install_required("sft-core", marker_file, _install_core, config)
    language_prefix = str(config.get("lora_language_prefix") or "")
    if language_prefix:
        runtime.install_deferred_text_lora_targeting(language_prefix, marker_file)
    model_type = config.get("gdn_model_type")
    if model_type:
        runtime.install_deferred_gdn(str(model_type), marker_file)
        runtime.install_deferred_flash_qla(str(model_type), marker_file)
    if config.get("wandb"):
        runtime.install_wandb_link_reporting()


if PLUGIN_LOADED_EXTERNALLY:
    install()
