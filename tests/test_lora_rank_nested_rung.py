"""regression for the nested peft parameter-wrapper rung.

drop-in target: tests/test_export.py (or tests/test_lora_rank_preflight.py).

why this is needed: tests/test_export.py:1273-1298 already covers a flattened rank-8192 fused tensor,
but only at the OUTER `mlp.experts` rung. peft nests ParamWrapper when two target_parameters share
one owner module, so a real adapter carries TWO rungs:

    mlp.experts              -> down_proj
    mlp.experts.base_layer   -> gate_up_proj

`...mlp.experts.base_layer` contains `mlp.experts` but does not END with it, so a suffix-only
predicate misses it and compares the tensor against the scalar rank. verified against the real
Qwen3.6-35B-A3B adapter: 80 of 160 expert tensors rejected, all of them the base_layer rung.
"""

import pytest

from flash.adapters.lora_rank import lora_tensor_rank_disagrees, strict_declared_lora_ranks

_MODEL = "Qwen/Qwen3.6-35B-A3B"
_RANK = 32
_EXPERTS = 256
_STACKED = _RANK * _EXPERTS  # 8192


def _config():
    return {
        "r": _RANK,
        "lora_alpha": 64,
        "base_model_name_or_path": _MODEL,
        "target_modules": "all-linear",
        "target_parameters": ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
    }


@pytest.mark.parametrize(
    ("module", "shapes"),
    [
        # outer rung: down_proj. already covered by the existing test.
        ("model.layers.0.mlp.experts", ((_STACKED, 512), (2048, _STACKED))),
        # nested rung: gate_up_proj. THIS is the regression - it was rejected.
        ("model.layers.0.mlp.experts.base_layer", ((_STACKED, 2048), (1024, _STACKED))),
    ],
)
def test_both_fused_expert_rungs_agree_with_the_stacked_rank(module, shapes):
    """both peft wrapper rungs must be recognised as stacked, not just the outer one."""
    declared = strict_declared_lora_ranks(_config())
    a_shape, b_shape = shapes
    assert not lora_tensor_rank_disagrees(f"{module}.lora_A.weight", a_shape, declared)
    assert not lora_tensor_rank_disagrees(f"{module}.lora_B.weight", b_shape, declared)


def test_nested_rung_still_rejects_a_genuinely_wrong_rank():
    """the fix must not turn the nested rung into an unconditional pass.

    without this, widening the predicate could accept any shape on base_layer and the gate would
    silently stop protecting that rung.
    """
    declared = strict_declared_lora_ranks(_config())
    module = "model.layers.0.mlp.experts.base_layer"
    # scalar rank instead of the stacked width, and a wrong stacked multiple
    assert lora_tensor_rank_disagrees(f"{module}.lora_A.weight", (_RANK, 2048), declared)
    assert lora_tensor_rank_disagrees(f"{module}.lora_A.weight", (_STACKED + _RANK, 2048), declared)
    assert lora_tensor_rank_disagrees(f"{module}.lora_B.weight", (1024, _RANK), declared)


def test_ordinary_dense_modules_are_unaffected():
    """dense tensors keep the plain scalar-rank contract."""
    declared = strict_declared_lora_ranks(_config())
    dense = "model.layers.0.self_attn.q_proj"
    assert not lora_tensor_rank_disagrees(f"{dense}.lora_A.weight", (_RANK, 2048), declared)
    assert lora_tensor_rank_disagrees(f"{dense}.lora_A.weight", (_STACKED, 2048), declared)


def test_lookalike_modules_are_not_treated_as_stacked():
    """guard against a fix that matches by substring instead of by rung structure."""
    declared = strict_declared_lora_ranks(_config())
    for lookalike in (
        "model.layers.0.mlp.experts_other",  # shares a prefix, different module
        "model.layers.0.mlp.expertsbase_layer",  # no separator
        "base_layer",  # nested marker with no owner
    ):
        assert lora_tensor_rank_disagrees(f"{lookalike}.lora_A.weight", (_STACKED, 2048), declared)
