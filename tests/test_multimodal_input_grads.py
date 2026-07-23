from __future__ import annotations

import pytest

from flash.engine.worker.perf.memory import (
    enable_multimodal_input_require_grads,
    make_multimodal_input_require_grads_callback,
)

torch = pytest.importorskip("torch")
checkpoint = pytest.importorskip("torch.utils.checkpoint").checkpoint
nn = torch.nn


class _VisionBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Module()
        self.visual.patch_embed = nn.Linear(4, 4)
        self.visual.patch_embed.requires_grad_(False)


class _PeftLikeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wrapped = _VisionBackbone()

    def get_base_model(self):
        return self.wrapped


def _checkpointed_loss(model, block, merger):
    hidden = model.get_base_model().visual.patch_embed(torch.randn(2, 4))
    hidden = checkpoint(block, hidden, use_reentrant=True)
    return merger(hidden).sum()


def test_vision_patch_hook_restores_reentrant_checkpoint_gradients():
    model = _PeftLikeModel()
    block = nn.Linear(4, 4)
    merger = nn.Linear(4, 1)

    with pytest.warns(UserWarning, match="None of the inputs have requires_grad=True"):
        _checkpointed_loss(model, block, merger).backward()
    assert block.weight.grad is None
    assert merger.weight.grad is not None

    merger.zero_grad(set_to_none=True)
    handle = enable_multimodal_input_require_grads(model)
    assert handle is not None

    _checkpointed_loss(model, block, merger).backward()
    assert block.weight.grad is not None
    assert merger.weight.grad is not None

    replacement_handle = enable_multimodal_input_require_grads(model)
    assert replacement_handle is not None
    assert replacement_handle is not handle
    assert len(model.get_base_model().visual.patch_embed._forward_hooks) == 1


def test_vision_patch_hook_is_noop_without_visual_patch_embed():
    assert enable_multimodal_input_require_grads(nn.Linear(2, 2)) is None


def test_multimodal_callback_installs_hook_on_trainer_model():
    pytest.importorskip("transformers")
    model = _PeftLikeModel()
    control = object()

    callback = make_multimodal_input_require_grads_callback()

    assert callback.on_train_begin(None, None, control, model=model) is control
    output = model.get_base_model().visual.patch_embed(torch.randn(1, 4))
    assert output.requires_grad
