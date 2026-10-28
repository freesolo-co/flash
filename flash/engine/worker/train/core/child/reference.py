"""a frozen adapter owned by the actor engine for warm-start KL reference."""

from contextlib import contextmanager

import torch.nn as nn
from peft.tuners.tuners_utils import BaseTunerLayer

adapter_name = "flash_kl_ref"
_KL_REF_ADAPTER_MARKER = "[flash-verl] kl reference anchored to the warm-start adapter"


def snapshot(module):
    if adapter_name in getattr(module, "peft_config", {}):
        return module
    source_adapter = module.active_adapter
    module.add_adapter(adapter_name, module.peft_config[source_adapter])
    demoted = 0
    for container in module.modules():
        if isinstance(container, nn.ModuleDict) and adapter_name in container:
            leaf = container[adapter_name]
            source = container[source_adapter]
            for attr, _ in list(leaf.named_parameters(recurse=False)):
                # Non-source FSDP ranks still hold meta weights here. Clone the source
                # directly so its device/dtype (including meta) survive until FSDP
                # materializes and broadcasts buffers, including non-persistent ones.
                frozen = source.get_parameter(attr).detach().clone()
                delattr(leaf, attr)
                leaf.register_buffer(attr, frozen, persistent=False)
                demoted += 1
    if not demoted:
        raise RuntimeError("flash kl reference snapshot found no adapter weights to freeze")
    print(_KL_REF_ADAPTER_MARKER + " " + repr(demoted), flush=True)
    return module


@contextmanager
def use_ref_adapter(module):
    inner = getattr(module, "_fsdp_wrapped_module", module)
    if adapter_name not in getattr(inner, "peft_config", {}):
        raise RuntimeError("flash kl reference adapter missing")
    layers = [value for value in module.modules() if isinstance(value, BaseTunerLayer)]
    if not layers:
        raise RuntimeError("flash kl reference: no lora layers on the actor module")
    saved = [layer._active_adapter for layer in layers]
    for layer in layers:
        layer._active_adapter = [adapter_name]
    try:
        yield
    finally:
        for layer, previous in zip(layers, saved, strict=True):
            layer._active_adapter = previous
