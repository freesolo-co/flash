"""Offline tests for the Qwen3.5/3.6 GRPO vLLM weight-sync name remap.

The colocated vLLM rollout engine loads a Qwen3.5/3.6 checkpoint as
``Qwen3_5ForConditionalGeneration`` whose LM params live under ``language_model.*``, while TRL's
trainer (an ``AutoModelForCausalLM``) pushes ``model.*`` / ``lm_head.*`` names at the first
``sync_weights()``. ``flash.engine.worker.lora._remap_vl_sync_weights`` bridges the two. These
tests exercise the pure remap and its gating without a GPU / vLLM.
"""

from __future__ import annotations


def test_remap_prefixes_text_only_model_names():
    """TEXT-ONLY trainer (dense Qwen3.5 -> Qwen3_5ForCausalLM, params ``model.*``): bare ``model.``
    names get the ``language_model.`` prefix so they land under ``language_model.model.*``. ``lm_head.``
    is now passed through UNTOUCHED — the engine's ``hf_to_vllm_mapper`` maps ``lm_head.`` ->
    ``language_model.lm_head.`` itself, so the end result at vLLM is identical."""
    from flash.engine.worker.lora import _remap_vl_sync_weights

    src = [
        ("model.layers.0.self_attn.q_proj.weight", "T0"),
        ("model.embed_tokens.weight", "T1"),
        ("model.norm.weight", "T2"),
        ("lm_head.weight", "T3"),
    ]
    out = dict(_remap_vl_sync_weights(iter(src)))
    assert out == {
        "language_model.model.layers.0.self_attn.q_proj.weight": "T0",
        "language_model.model.embed_tokens.weight": "T1",
        "language_model.model.norm.weight": "T2",
        "lm_head.weight": "T3",  # passthrough: the engine mapper handles lm_head.
    }


def test_remap_multimodal_names_pass_through_for_engine_mapper():
    """MULTIMODAL trainer (the Qwen3.6-35B-A3B MoE resolves to the full
    ``Qwen3_5MoeForConditionalGeneration``, params ``model.language_model.*`` / ``model.visual.*`` /
    ``lm_head.*``): these MUST pass through untouched. The engine's own ``hf_to_vllm_mapper`` maps
    them (``model.language_model.`` -> ``language_model.model.`` etc.), so the sync is byte-identical
    to the proven initial on-disk load. Regression for the fused-MoE-expert
    ``KeyError: experts.w13_weight`` (prepending ``language_model.`` here broke the mapper match)."""
    from flash.engine.worker.lora import _remap_vl_sync_weights

    src = [
        ("model.language_model.layers.0.mlp.experts.gate_up_proj", "E0"),  # the crashing fused expert
        ("model.language_model.layers.0.mlp.experts.down_proj", "E1"),
        ("model.language_model.layers.0.self_attn.qkv_proj.weight", "D0"),
        ("model.visual.blocks.0.attn.qkv.weight", "V0"),
        ("lm_head.weight", "L0"),
    ]
    out = list(_remap_vl_sync_weights(iter(src)))
    assert out == src  # every name untouched


def test_remap_leaves_non_lm_names_untouched():
    from flash.engine.worker.lora import _remap_vl_sync_weights

    # Already-namespaced VL names and vision-tower names must pass through verbatim — only the
    # trainer's bare ``model.``/``lm_head.`` keys get prefixed.
    src = [
        ("language_model.model.layers.0.mlp.gate_proj.weight", "A"),
        ("visual.blocks.0.attn.qkv.weight", "B"),
        ("multi_modal_projector.linear.weight", "C"),
    ]
    out = list(_remap_vl_sync_weights(iter(src)))
    assert out == src


def test_remap_strips_peft_base_model_wrapper():
    from flash.engine.worker.lora import _remap_vl_sync_weights

    # A continued-adapter (warm-start) sync can surface base-model names through the peft wrapper
    # as ``base_model.model.<...>``; strip the wrapper, then apply the language_model. prefix.
    src = [
        ("base_model.model.model.layers.1.mlp.up_proj.weight", "X"),
        ("base_model.model.lm_head.weight", "Y"),
    ]
    out = dict(_remap_vl_sync_weights(iter(src)))
    assert out == {
        "language_model.model.layers.1.mlp.up_proj.weight": "X",
        "lm_head.weight": "Y",  # passthrough after wrapper strip: the engine mapper handles lm_head.
    }


def test_patch_vllm_lm_weight_sync_noop_for_non_vl(monkeypatch):
    import flash.engine.worker.lora as lora

    # Pure-text architecture (is_vl_checkpoint False) must not install the remap — the trainer and
    # vLLM agree on ``model.*`` names there, so remapping would break a healthy sync.
    monkeypatch.setattr(lora, "is_vl_checkpoint", lambda model_id: False)
    assert lora.patch_vllm_lm_weight_sync("Qwen/Qwen3-4B-Instruct-2507") is False


def test_patch_vllm_lm_weight_sync_gating_remaps_only_when_active(monkeypatch):
    """The patched load_weights remaps iff _LM_SYNC_REMAP_ON['on'] is set (so the INITIAL
    checkpoint load — flag off — is untouched and only train-time syncs are remapped). We fake the
    vLLM module + model class to exercise the real patch installer without vLLM/CUDA."""
    import sys
    import types

    import flash.engine.worker.lora as lora

    monkeypatch.setattr(lora, "is_vl_checkpoint", lambda model_id: True)

    seen: list[list[tuple[str, str]]] = []

    class FakeVLModel:
        def load_weights(self, weights, *a, **k):
            seen.append(list(weights))

    fake_mod = types.ModuleType("vllm.model_executor.models.qwen3_5")
    fake_mod.Qwen3_5ForConditionalGeneration = FakeVLModel
    # The MoE module is absent here; the installer must tolerate a missing class and patch what
    # it can.
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.qwen3_5", fake_mod)
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.models.qwen3_5_moe", types.ModuleType("x")
    )

    # Flag starts OFF (initial-load semantics): names pass through unchanged.
    lora._LM_SYNC_REMAP_ON["on"] = False
    try:
        assert lora.patch_vllm_lm_weight_sync("Qwen/Qwen3.5-0.8B") is True
        inst = FakeVLModel()
        inst.load_weights(iter([("model.layers.0.q.weight", "T")]))
        assert seen[-1] == [("model.layers.0.q.weight", "T")]

        # Flag ON (train-time sync): the same trainer names get the language_model. prefix.
        lora._LM_SYNC_REMAP_ON["on"] = True
        inst.load_weights(iter([("model.layers.0.q.weight", "T")]))
        assert seen[-1] == [("language_model.model.layers.0.q.weight", "T")]
    finally:
        lora._LM_SYNC_REMAP_ON["on"] = False


def test_patch_vllm_lm_weight_sync_idempotent(monkeypatch):
    import sys
    import types

    import flash.engine.worker.lora as lora

    monkeypatch.setattr(lora, "is_vl_checkpoint", lambda model_id: True)

    class FakeVLModel:
        def load_weights(self, weights, *a, **k):
            return None

    fake_mod = types.ModuleType("vllm.model_executor.models.qwen3_5")
    fake_mod.Qwen3_5ForConditionalGeneration = FakeVLModel
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.qwen3_5", fake_mod)
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.models.qwen3_5_moe", types.ModuleType("x")
    )

    assert lora.patch_vllm_lm_weight_sync("Qwen/Qwen3.5-9B") is True
    first = FakeVLModel.load_weights
    # A second call must NOT double-wrap (the _flash_sync_patched sentinel guards it).
    assert lora.patch_vllm_lm_weight_sync("Qwen/Qwen3.5-9B") is False
    assert FakeVLModel.load_weights is first


def test_patch_vllm_lm_weight_sync_warns_on_missing_required_module(monkeypatch, capsys):
    """If the REQUIRED dense vLLM module can't be imported, the installer must NOT silently no-op
    (the run would otherwise crash later at the first sync_weights() with a far less actionable
    error). It logs a loud warning and returns False; the OPTIONAL MoE module stays quiet."""
    import sys

    import flash.engine.worker.lora as lora

    monkeypatch.setattr(lora, "is_vl_checkpoint", lambda model_id: True)
    # None in sys.modules makes importlib.import_module raise deterministically (whether or not a
    # real vLLM is installed in the test env).
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.qwen3_5", None)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.qwen3_5_moe", None)

    assert lora.patch_vllm_lm_weight_sync("Qwen/Qwen3.5-0.8B") is False
    out = capsys.readouterr().out
    assert "patch_vllm_lm_weight_sync" in out
    assert "vllm.model_executor.models.qwen3_5" in out
    # The optional MoE module's absence must NOT produce its own warning.
    assert "qwen3_5_moe" not in out


def test_patch_vllm_lm_weight_sync_warns_when_required_class_renamed(monkeypatch, capsys):
    """The required module imports but no longer has the expected class (vLLM renamed it) -> warn,
    don't silently no-op."""
    import sys
    import types

    import flash.engine.worker.lora as lora

    monkeypatch.setattr(lora, "is_vl_checkpoint", lambda model_id: True)
    # Module present but missing Qwen3_5ForConditionalGeneration.
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.models.qwen3_5",
        types.ModuleType("vllm.model_executor.models.qwen3_5"),
    )
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.models.qwen3_5_moe", types.ModuleType("x")
    )

    assert lora.patch_vllm_lm_weight_sync("Qwen/Qwen3.5-9B") is False
    out = capsys.readouterr().out
    assert "Qwen3_5ForConditionalGeneration" in out
