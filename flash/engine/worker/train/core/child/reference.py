"""a frozen adapter owned by the actor engine for warm-start KL reference."""

from contextlib import contextmanager

import torch.nn as nn
from peft.tuners.tuners_utils import BaseTunerLayer

adapter_name = "flash_kl_ref"
_KL_REF_ADAPTER_MARKER = "[flash-verl] kl reference anchored to the warm-start adapter"


def snapshot(module):
    if adapter_name in getattr(module, "peft_config", {}):
        return module
    module.add_adapter(adapter_name, module.peft_config[module.active_adapter])
    demoted = 0
    for container in module.modules():
        if isinstance(container, nn.ModuleDict) and adapter_name in container:
            leaf = container[adapter_name]
            for attr, _value in list(leaf.named_parameters(recurse=False)):
                # Nonzero FSDP ranks can still have meta adapter parameters here.
                # Clone the source's device/dtype as well as its values; copying
                # a meta source into PEFT's newly allocated CPU adapter fails.
                # Pinned Verl materializes these nonpersistent buffers and
                # broadcasts rank zero's frozen values during FSDP initialization.
                frozen = container["default"].get_parameter(attr).detach().clone()
                delattr(leaf, attr)
                leaf.register_buffer(attr, frozen, persistent=False)
                demoted += 1
    if not demoted:
        raise RuntimeError("flash kl reference snapshot found no adapter weights to freeze")
    return module


def validate_snapshot(module):
    """Do not start training if FSDP left the frozen reference unmaterialized."""
    buffers = [
        (name, value) for name, value in module.named_buffers() if f".{adapter_name}." in name
    ]
    if not buffers:
        raise RuntimeError("flash kl reference snapshot has no frozen buffers")
    pending = [name for name, value in buffers if value.is_meta]
    if pending:
        raise RuntimeError(f"flash kl reference not materialized after FSDP: {pending}")
    print(_KL_REF_ADAPTER_MARKER + " " + repr(len(buffers)), flush=True)


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
