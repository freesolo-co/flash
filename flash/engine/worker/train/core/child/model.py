"""model-instance customization through supported model and PyTorch hook APIs."""

_NON_LANGUAGE_LORA_SEGMENTS = frozenset(
    {
        "mtp",
        "multi_modal_projector",
        "patch_embed",
        "visual",
        "vision",
        "vision_encoder",
        "vision_model",
        "vision_tower",
    }
)


def _gdn_seq_idx(position_ids, cu_seq_lens):
    import torch

    lengths = cu_seq_lens.diff()
    return (
        torch.repeat_interleave(
            torch.arange(lengths.numel(), device=position_ids.device, dtype=torch.int32),
            lengths.to(torch.int64),
        )
        .unsqueeze(0)
        .contiguous()
    )


def _text_lora_module_types():
    import torch
    from transformers.pytorch_utils import Conv1D

    return (torch.nn.Linear, Conv1D)


def resolve_text_lora_target_modules(
    module, language_prefix: str, *, module_types=None
) -> list[str]:
    """Resolve exact runtime paths for every supported language LoRA module."""
    named_modules = list(module.named_modules())
    language_root = _text_lora_language_root(named_modules, language_prefix)
    supported_types = module_types or _text_lora_module_types()
    candidates = {
        name
        for name, child in named_modules
        if name.startswith(f"{language_root}.") and isinstance(child, supported_types)
    }
    targets = sorted(
        name
        for name in candidates
        if not (set(name.lower().split(".")) & _NON_LANGUAGE_LORA_SEGMENTS)
    )
    if not targets:
        raise RuntimeError(
            "flash text LoRA targeting found no supported language modules under runtime root "
            f"{language_root!r}"
        )

    matched = {
        name
        for name, child in named_modules
        if isinstance(child, supported_types)
        and any(name == target or name.endswith(f".{target}") for target in targets)
    }
    if matched != set(targets):
        extras = sorted(matched - set(targets))
        raise RuntimeError(
            "flash text LoRA targets are not unambiguous under PEFT suffix matching; "
            f"unexpected matches={extras[:4]}"
        )
    return targets


def install_vision_input_grads(module) -> None:
    """force the vision tower's first activation to require grad under reentrant checkpointing.

    a reentrant checkpointed block produces no gradient at all unless at least one of its inputs
    requires grad. ``enable_input_require_grads()`` only covers the text embedding, so pixel values
    stay grad-free and every ``visual.blocks.*`` submodule receives exactly zero. the LM half still
    trains and ``visual.merger`` still moves (it is downstream of the grad-carrying text hidden
    states), so the total adapter delta looks healthy and inference still answers image questions --
    the frozen tower is only visible per lora pair.
    """
    import torch

    get_base = getattr(module, "get_base_model", None)
    if callable(get_base):
        base = get_base()
    else:
        peft_base = getattr(module, "base_model", None)
        base = getattr(peft_base, "model", module)

    def require_output_grad(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) and output else output
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
            tensor.requires_grad_(True)

    for path, submodule in base.named_modules():
        if path.endswith("visual.patch_embed"):
            submodule.register_forward_hook(require_output_grad)
            print(f"[flash-verl] vision input gradients enabled at {path}", flush=True)
            return
    print(
        "[flash-verl] no visual.patch_embed found; vision input gradients not installed", flush=True
    )


def configure_gdn(module, model_type: str) -> None:
    from transformers.modeling_flash_attention_utils import (
        _is_packed_sequence,
        prepare_fa_kwargs_from_position_ids,
    )

    def packed_boundaries(_module, args, kwargs):
        kwargs = dict(kwargs)
        if kwargs.get("cu_seq_lens_q") is not None or kwargs.get("seq_idx") is not None:
            return args, kwargs
        positions = kwargs.get("position_ids")
        if positions is not None and positions.ndim == 3:
            positions = positions[0]
        if (
            positions is not None
            and positions.ndim == 2
            and _is_packed_sequence(positions, positions.shape[0])
        ):
            (cu_q, cu_k), (max_q, max_k) = prepare_fa_kwargs_from_position_ids(positions)
            kwargs.update(
                cu_seq_lens_q=cu_q,
                cu_seq_lens_k=cu_k,
                max_length_q=max_q,
                max_length_k=max_k,
                seq_idx=_gdn_seq_idx(positions, cu_q),
            )
        return args, kwargs

    found = False
    for layer in module.modules():
        if f"transformers.models.{model_type}." not in type(layer).__module__:
            continue
        if type(layer).__name__.endswith("TextModel"):
            layer.register_forward_pre_hook(packed_boundaries, with_kwargs=True)
            found = True
    if not found:
        raise RuntimeError(f"no supported {model_type} text model found for packed boundaries")


def _text_lora_language_root(named_modules, language_prefix: str) -> str:
    prefix = language_prefix.strip(".")
    if not prefix:
        raise RuntimeError("flash text LoRA targeting requires a non-empty language module prefix")
    roots = {
        name for name, _module in named_modules if name == prefix or name.endswith(f".{prefix}")
    }
    if len(roots) != 1:
        raise RuntimeError(
            "flash text LoRA targeting requires exactly one runtime language root for "
            f"{prefix!r}; found {sorted(roots)}"
        )
    return next(iter(roots))
