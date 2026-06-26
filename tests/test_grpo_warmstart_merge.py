"""Warm-start GRPO = merge SFT into the base + train a FRESH LoRA, then recombine for deploy.

The worker no longer hands GRPOTrainer a live (CPU-resident) PeftModel for ``init_from_adapter``
runs — that pre-loaded object stalled colocate-vLLM trainer construction at 0% GPU for ~25 min.
Instead ``_init_adapter_model`` MERGES the SFT adapter into a disk copy of the base and returns a
model-PATH string + a fresh ``LoraConfig`` (the proven from-base shape). Because the SFT is baked
into the base, the fresh GRPO LoRA's rank/alpha are INDEPENDENT of the SFT run. The deployable
artifact is reassembled at finalize (``combine_warmstart_into_adapter``) so serving still loads the
catalog base + the run's single adapter.

These tests run on CPU (no GPU/network). The numeric ones need torch/peft/transformers and skip
cleanly where those aren't installed (the offline CI lane).
"""

from __future__ import annotations

import json
import os

import pytest

from flash.spec import JobSpec


def _spec(*, init_from_adapter: str = "", lora_rank: int = 32, lora_alpha: int = 64) -> JobSpec:
    return JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3-4B-Instruct-2507",
            "algorithm": "grpo",
            "environment": {"id": "github:owner/repo@main:env/environment.py"},
            "train": {
                "seeds": [0],
                "steps": 10,
                "init_from_adapter": init_from_adapter,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
            },
        }
    )


def _tiny_cfg():
    from transformers import LlamaConfig

    return LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
    )


def _randomize_lora(peft_model, torch) -> None:
    """Make every LoRA A/B non-zero so the adapter carries a REAL delta (lora_B inits to 0)."""
    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.normal_(0.0, 0.1)


# ---------------------------------------------------------------------------
# From-base + rank/alpha decoupling: ALWAYS a (path_str, fresh LoraConfig) pair.
# ---------------------------------------------------------------------------
def test_init_adapter_from_base_returns_string_and_grpo_rank_alpha(monkeypatch) -> None:
    pytest.importorskip("peft")
    from peft import LoraConfig

    import flash.engine.worker as w

    # avoid the network AutoConfig probe in make_lora's exclude-modules path
    monkeypatch.setattr(w, "lora_exclude_modules", lambda model_id: None)
    # GRPO chooses its OWN rank/alpha (16/8 here), independent of any SFT lineage.
    monkeypatch.setattr(w, "JOB_SPEC", _spec(lora_rank=16, lora_alpha=8))

    model, peft_cfg = w._init_adapter_model("some/base-model")
    assert model == "some/base-model"  # a STRING, not a live PeftModel
    assert isinstance(peft_cfg, LoraConfig)
    assert peft_cfg.r == 16
    assert peft_cfg.lora_alpha == 8


# ---------------------------------------------------------------------------
# VL guard: warm-start merge is not yet safe for natively-multimodal checkpoints — fail FAST
# (before importing peft / loading the base), not hang.
# ---------------------------------------------------------------------------
def test_init_adapter_warmstart_vl_checkpoint_raises(monkeypatch, tmp_path) -> None:
    import flash.engine.worker as w
    import flash.engine.worker.adapter as adapter

    monkeypatch.setattr(w, "JOB_SPEC", _spec(init_from_adapter="o/r:sft/run/seed0"))
    adir = tmp_path / "adapter"
    adir.mkdir()
    monkeypatch.setattr(adapter, "_download_adapter", lambda prefix: str(adir))
    monkeypatch.setattr(adapter, "remap_vl_adapter_dir", lambda a, m: 0)
    monkeypatch.setattr(adapter, "is_vl_checkpoint", lambda m: True)

    with pytest.raises(RuntimeError, match="multimodal"):
        w._init_adapter_model("Qwen/Qwen3.5-4B")


def test_init_adapter_warmstart_vl_by_adapter_evidence_raises(monkeypatch, tmp_path) -> None:
    # Even if the is_vl_checkpoint config probe says text-only (flaky network), an adapter that
    # carried '.language_model.' keys (remap returned > 0) is VL evidence -> still guarded.
    import flash.engine.worker as w
    import flash.engine.worker.adapter as adapter

    monkeypatch.setattr(w, "JOB_SPEC", _spec(init_from_adapter="o/r:sft/run/seed0"))
    adir = tmp_path / "adapter"
    adir.mkdir()
    monkeypatch.setattr(adapter, "_download_adapter", lambda prefix: str(adir))
    monkeypatch.setattr(adapter, "remap_vl_adapter_dir", lambda a, m: 7)  # stripped 7 VL keys
    monkeypatch.setattr(adapter, "is_vl_checkpoint", lambda m: False)

    with pytest.raises(RuntimeError, match="multimodal"):
        w._init_adapter_model("Qwen/Qwen3.5-4B")


# ---------------------------------------------------------------------------
# Warm-start merge-to-string (real tiny models): returns a model PATH whose base has the SFT
# baked in, plus the GRPO job's (decoupled) rank/alpha.
# ---------------------------------------------------------------------------
def test_init_adapter_warmstart_merges_sft_and_returns_string_path(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    import flash.engine.worker as w
    import flash.engine.worker.adapter as adapter

    torch.manual_seed(0)
    base_model = AutoModelForCausalLM.from_config(_tiny_cfg()).eval()
    w_dir = tmp_path / "base"
    base_model.save_pretrained(w_dir)

    # SFT adapter (rank 4) on the base, with a real non-zero delta.
    sft = get_peft_model(
        AutoModelForCausalLM.from_pretrained(w_dir),
        LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    _randomize_lora(sft, torch)
    sft_dir = tmp_path / "sft"
    sft.save_pretrained(sft_dir)
    base_plus_sft = sft.merge_and_unload().to(torch.bfloat16).eval()  # the merged-base reference

    # GRPO job: rank 8 / alpha 16 — DIFFERENT from the SFT's 4/8 (decoupling). Warm-start from sft_dir.
    monkeypatch.setattr(
        w, "JOB_SPEC", _spec(init_from_adapter="o/r:sft/run/seed0", lora_rank=8, lora_alpha=16)
    )
    monkeypatch.setattr(adapter, "_download_adapter", lambda prefix: str(sft_dir))
    monkeypatch.setattr(adapter, "optimal_attn_impl", lambda: None)  # plain SDPA on CPU
    # The tiny LlamaConfig base has no tokenizer; the real model_id always does. _save_merged_base
    # copies model_id's tokenizer into the merged dir (vLLM colocate needs it) — stub it here.
    import transformers

    class _StubTok:
        def save_pretrained(self, d):
            with open(os.path.join(d, "tokenizer_config.json"), "w") as f:
                f.write("{}")

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _StubTok()
    )

    model_path, peft_cfg = w._init_adapter_model(str(w_dir))

    # (1) returns a STRING path (proven from-base shape), not a live PeftModel
    assert isinstance(model_path, str)
    assert os.path.isdir(model_path)
    assert os.path.isfile(os.path.join(model_path, "config.json"))
    # (2) fresh LoRA at the GRPO job's rank/alpha, independent of the SFT adapter (4/8)
    assert isinstance(peft_cfg, LoraConfig)
    assert peft_cfg.r == 8
    assert peft_cfg.lora_alpha == 16

    # (3) the merged base on disk has the SFT baked in: its logits match base+SFT, and differ
    # clearly from the original base (the SFT was not silently dropped).
    ids = torch.randint(0, 64, (1, 5))
    merged = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).eval()
    base_only = AutoModelForCausalLM.from_pretrained(w_dir, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        lm = merged(ids).logits.float()
        l_sft = base_plus_sft(ids).logits.float()
        l_base = base_only(ids).logits.float()
    assert torch.allclose(lm, l_sft, atol=2e-2, rtol=2e-2)  # merged == base + SFT
    assert (lm - l_base).abs().max() > 1e-2  # and clearly != base alone


# ---------------------------------------------------------------------------
# Deploy correctness: cat(SFT, GRPO) on the ORIGINAL base reproduces the trained model
# (base + SFT-merged + GRPO), even when the two LoRAs have DIFFERENT ranks.
# ---------------------------------------------------------------------------
def test_cat_combined_adapter_reproduces_trained_warmstart_model(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    base = AutoModelForCausalLM.from_config(_tiny_cfg()).eval()
    w_dir = tmp_path / "base"
    base.save_pretrained(w_dir)

    # SFT adapter (rank 4) on the original base.
    sft = get_peft_model(
        AutoModelForCausalLM.from_pretrained(w_dir),
        LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    _randomize_lora(sft, torch)
    sft_dir = tmp_path / "sft"
    sft.save_pretrained(sft_dir)

    # base' = base + SFT (what warm-start GRPO trains on), then a GRPO LoRA (rank 8) on top.
    base_prime = sft.merge_and_unload().eval()
    grpo = get_peft_model(
        base_prime,
        LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    _randomize_lora(grpo, torch)
    grpo_dir = tmp_path / "adapter"
    grpo.save_pretrained(grpo_dir)

    ids = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        trained_logits = grpo.eval()(ids).logits  # base + SFT(merged) + GRPO = the trained model

    # Recombine on the ORIGINAL base via the SAME cat PEFT applies inside the worker helper.
    combo = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(w_dir), str(sft_dir), adapter_name="sft"
    )
    combo.load_adapter(str(grpo_dir), adapter_name="grpo")
    combo.add_weighted_adapter(
        ["sft", "grpo"], [1.0, 1.0], adapter_name="default", combination_type="cat"
    )
    assert combo.peft_config["default"].r == 12  # 4 + 8 (cat sums ranks)
    combo.set_adapter("default")
    with torch.no_grad():
        combined_logits = combo.eval()(ids).logits

    # The single combined adapter on the original base == the trained warm-start model.
    assert torch.allclose(combined_logits, trained_logits, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# combine_warmstart_into_adapter wiring: rewrites the GRPO adapter dir in place to the combined
# "default" adapter (at the dir ROOT, where deploy/serve reads it); no-op for a from-base run.
# ---------------------------------------------------------------------------
def test_combine_warmstart_into_adapter_writes_default_at_root(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    import flash.engine.worker as w
    import flash.engine.worker.adapter as adapter

    torch.manual_seed(0)
    base = AutoModelForCausalLM.from_config(_tiny_cfg()).eval()
    w_dir = tmp_path / "base"
    base.save_pretrained(w_dir)
    sft = get_peft_model(
        AutoModelForCausalLM.from_pretrained(w_dir),
        LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    _randomize_lora(sft, torch)
    sft_dir = tmp_path / "sft"
    sft.save_pretrained(sft_dir)
    grpo = get_peft_model(
        sft.merge_and_unload(),
        LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    _randomize_lora(grpo, torch)
    grpo_dir = tmp_path / "adapter"
    grpo.save_pretrained(grpo_dir)

    # warm-start run: SFT resolves to sft_dir
    monkeypatch.setattr(w, "JOB_SPEC", _spec(init_from_adapter="o/r:sft/run/seed0"))
    monkeypatch.setattr(adapter, "_download_adapter", lambda prefix: str(sft_dir))

    assert w.combine_warmstart_into_adapter(str(w_dir), str(grpo_dir)) is True
    acfg = json.loads((grpo_dir / "adapter_config.json").read_text())
    assert acfg["r"] == 12  # cat: rank = 4 + 8
    assert acfg["lora_alpha"] == 12  # cat sets alpha == rank
    assert (grpo_dir / "adapter_model.safetensors").is_file()

    # from-base run -> no recombine
    monkeypatch.setattr(w, "JOB_SPEC", _spec(init_from_adapter=""))
    assert w.combine_warmstart_into_adapter(str(w_dir), str(grpo_dir)) is False


def test_combine_warmstart_exported_from_worker_package() -> None:
    import flash.engine.worker as w

    assert hasattr(w, "combine_warmstart_into_adapter")
    assert "combine_warmstart_into_adapter" in w.__all__
