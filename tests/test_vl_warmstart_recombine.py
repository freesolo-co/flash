"""Offline tests for the VL warm-start SFT⊕GRPO adapter recombine.

Bug: the VL merge-into-base warm-start (#296) merges the SFT into the base and trains a FRESH LoRA
on the merged weights, then saves ONLY that GRPO LoRA. Deployed on the ORIGINAL catalog base it
drops the SFT entirely (served output collapses to ~base). ``recombine_lora_adapters`` stacks the
two LoRAs into one rank-(r_sft+r_grpo) adapter whose delta reproduces ``SFT_delta + GRPO_delta`` on
the unmodified base — the exact model GRPO trained.

These exercise the pure tensor math (exactness for equal AND differing scales, incl. rsLoRA), the
config the recombined adapter writes, and the loud guards — all without a GPU / transformers / peft.
"""

from __future__ import annotations

import json
import math
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import load_file, save_file

from flash.catalog import serving_lora_rank_cap
from flash.engine.worker.lora import (
    recombine_lora_adapters,
    validate_recombined_lora_rank,
)

# Realistic Qwen3.5 VL adapter key stems. The SFT adapter is trained against the FULL multimodal
# model, so its LM modules live under ``language_model.`` (MODULES). The fresh GRPO LoRA is saved by
# the text-only AutoModelForCausalLM trainer, so the SAME modules have no infix (TEXT_MODULES). The
# recombine must line these equivalent modules up for math, then emit the serving-compatible
# language_model form.
MODULES = [
    "base_model.model.model.language_model.layers.0.self_attn.q_proj",
    "base_model.model.model.language_model.layers.0.self_attn.v_proj",
    "base_model.model.model.language_model.layers.5.mlp.gate_proj",
]
TEXT_MODULES = [m.replace(".language_model.", ".") for m in MODULES]


def _write_adapter(
    adir: str,
    *,
    modules,
    r,
    alpha,
    in_f=8,
    out_f=6,
    use_rslora=False,
    seed=0,
    use_dora=False,
    modules_to_save=None,
    dtype=torch.float32,
):
    os.makedirs(adir, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for m in modules:
        # Real PEFT adapters embed the adapter name in the saved key (``...lora_A.default.weight``),
        # not the bare ``...lora_A.weight`` form — write the representative keys.
        sd[f"{m}.lora_A.default.weight"] = torch.randn(r, in_f, generator=g, dtype=dtype) * 0.1
        sd[f"{m}.lora_B.default.weight"] = torch.randn(out_f, r, generator=g, dtype=dtype) * 0.1
    save_file(sd, os.path.join(adir, "adapter_model.safetensors"), metadata={"format": "pt"})
    cfg = {
        "peft_type": "LORA",
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "use_rslora": use_rslora,
        "use_dora": use_dora,
        "target_modules": ["q_proj", "v_proj", "gate_proj"],
        "base_model_name_or_path": "Qwen/Qwen3.5-2B",
    }
    if modules_to_save:
        cfg["modules_to_save"] = modules_to_save
    with open(os.path.join(adir, "adapter_config.json"), "w") as f:
        json.dump(cfg, f)
    return sd, cfg


def _scale(cfg):
    r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    return alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r


def _read_cfg(adir):
    with open(os.path.join(adir, "adapter_config.json")) as f:
        return json.load(f)


def _delta(adir, module):
    cfg = _read_cfg(adir)
    sd = load_file(os.path.join(adir, "adapter_model.safetensors"))
    A = sd[f"{module}.lora_A.default.weight"].to(torch.float64)
    B = sd[f"{module}.lora_B.default.weight"].to(torch.float64)
    return _scale(cfg) * (B @ A)


@pytest.mark.parametrize(
    ("r_sft", "a_sft", "rs_sft", "r_grpo", "a_grpo", "rs_grpo"),
    [
        (4, 8, False, 4, 8, False),  # equal scale 2.0 (the common managed-flash case)
        (4, 8, False, 2, 8, False),  # differing scales 2.0 vs 4.0, differing ranks
        (8, 16, True, 4, 8, False),  # rsLoRA on one side
    ],
)
def test_recombine_reproduces_sum_of_deltas(
    tmp_path, r_sft, a_sft, rs_sft, r_grpo, a_grpo, rs_grpo
):
    sft = str(tmp_path / "sft")
    grpo = str(tmp_path / "grpo")
    out = str(tmp_path / "out")
    # Production forms: SFT keys carry the `.language_model.` infix; the GRPO LoRA is text-only.
    _write_adapter(sft, modules=MODULES, r=r_sft, alpha=a_sft, use_rslora=rs_sft, seed=1)
    _write_adapter(grpo, modules=TEXT_MODULES, r=r_grpo, alpha=a_grpo, use_rslora=rs_grpo, seed=2)

    rank = recombine_lora_adapters(sft, grpo, out)
    assert rank == r_sft + r_grpo

    out_cfg = _read_cfg(out)
    # Scales are baked into B, so the recombined adapter carries unit scale (alpha == r).
    assert out_cfg["r"] == r_sft + r_grpo
    assert out_cfg["lora_alpha"] == r_sft + r_grpo
    assert out_cfg["use_rslora"] is False

    # Output is emitted in the VL SFT key namespace; serving expects the language_model wrapper.
    out_sd = load_file(os.path.join(out, "adapter_model.safetensors"))
    assert all(".language_model." in k for k in out_sd), (
        "recombined keys must target language_model"
    )
    assert not any(k.startswith("base_model.model.model.layers.") for k in out_sd)

    # The recombined delta must EXACTLY equal SFT_delta + GRPO_delta on the original base. GRPO's
    # text-only module maps back to the language_model module the SFT and output adapters use.
    for m_sft, m_text in zip(MODULES, TEXT_MODULES, strict=True):
        want = _delta(sft, m_sft) + _delta(grpo, m_text)
        got = _delta(out, m_sft)
        assert torch.allclose(got, want, atol=1e-6, rtol=1e-5), f"{m_sft}: recombine delta mismatch"


def test_recombine_normalizes_language_model_infix(tmp_path):
    # Regression: the default Qwen3.5 VL warm-start saves an infixed SFT adapter and a text-only
    # GRPO LoRA. The recombine must treat the equivalent LM modules as the SAME target (not raise
    # "DIFFERENT modules") and emit language_model keys — otherwise the artifact does not load under
    # the serving wrapper.
    sft = str(tmp_path / "sft")
    grpo = str(tmp_path / "grpo")
    out = str(tmp_path / "out")
    _write_adapter(sft, modules=MODULES, r=4, alpha=8, seed=1)  # infixed (full VL model)
    _write_adapter(
        grpo, modules=TEXT_MODULES, r=4, alpha=8, seed=2
    )  # text-only (AutoModelForCausalLM)

    rank = recombine_lora_adapters(sft, grpo, out)
    assert rank == 8

    out_sd = load_file(os.path.join(out, "adapter_model.safetensors"))
    out_modules = {k.rsplit(".lora_", 1)[0] for k in out_sd}
    assert out_modules == set(MODULES), "recombined modules must use the serving language_model set"
    assert all(".language_model." in k for k in out_sd)
    assert not any(k.startswith("base_model.model.model.layers.") for k in out_sd)


def test_recombined_rank_preflight_allows_sum_at_serving_cap(tmp_path):
    sft = str(tmp_path / "sft")
    _write_adapter(sft, modules=MODULES, r=16, alpha=32, seed=1)
    max_rank = serving_lora_rank_cap("Qwen/Qwen3.5-4B")

    assert validate_recombined_lora_rank(sft, 16, max_rank=max_rank) == (16, 16, max_rank)


def test_recombined_rank_preflight_allows_model_specific_rank64_cap(tmp_path):
    sft = str(tmp_path / "sft")
    _write_adapter(sft, modules=MODULES, r=32, alpha=64, seed=1)
    max_rank = serving_lora_rank_cap("Qwen/Qwen3.5-2B")

    assert max_rank == 64
    assert validate_recombined_lora_rank(sft, 32, max_rank=max_rank) == (32, 32, 64)


def test_recombined_rank_preflight_does_not_apply_rank32_fallback_without_serving_cap(tmp_path):
    sft = str(tmp_path / "sft")
    _write_adapter(sft, modules=MODULES, r=32, alpha=64, seed=1)

    assert serving_lora_rank_cap("unknown/model") is None
    assert validate_recombined_lora_rank(sft, 32, max_rank=None) == (32, 32, 64)


def test_recombined_rank_preflight_rejects_undeployable_sum(tmp_path):
    sft = str(tmp_path / "sft")
    _write_adapter(sft, modules=MODULES, r=24, alpha=48, seed=1)
    max_rank = serving_lora_rank_cap("Qwen/Qwen3.5-4B")

    with pytest.raises(ValueError, match=r"set GRPO train\.lora_rank <= 8"):
        validate_recombined_lora_rank(sft, 16, max_rank=max_rank)


def test_recombine_rejects_rank_above_serving_cap(tmp_path):
    sft = str(tmp_path / "sft")
    grpo = str(tmp_path / "grpo")
    out = str(tmp_path / "out")
    _write_adapter(sft, modules=MODULES, r=24, alpha=48, seed=1)
    _write_adapter(grpo, modules=MODULES, r=16, alpha=32, seed=2)
    _set_base(sft, "Qwen/Qwen3.5-4B")

    with pytest.raises(ValueError, match=r"rank-stacked SFT\+GRPO adapter would be rank 40"):
        recombine_lora_adapters(sft, grpo, out)


def test_recombine_allows_rank64_for_model_with_serving_cap64(tmp_path):
    sft = str(tmp_path / "sft")
    grpo = str(tmp_path / "grpo")
    out = str(tmp_path / "out")
    _write_adapter(sft, modules=MODULES, r=32, alpha=64, seed=1)
    _write_adapter(grpo, modules=TEXT_MODULES, r=32, alpha=64, seed=2)

    assert recombine_lora_adapters(sft, grpo, out) == 64
    assert _read_cfg(out)["r"] == 64


def test_init_adapter_model_preflights_vl_recombined_rank_before_model_load(tmp_path, monkeypatch):
    import flash.engine.worker.adapter as worker_adapter

    sft = str(tmp_path / "sft")
    _write_adapter(sft, modules=MODULES, r=24, alpha=48, seed=1)
    peft = types.ModuleType("peft")
    peft.PeftModel = object
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setattr(worker_adapter, "_download_adapter", lambda _prefix: sft)
    monkeypatch.setattr(worker_adapter, "adapter_is_vl_warmstart", lambda _adir, _model_id: True)
    monkeypatch.setattr(worker_adapter, "optimal_attn_impl", lambda: None)
    monkeypatch.setattr(
        W,
        "JOB_SPEC",
        types.SimpleNamespace(
            train=types.SimpleNamespace(
                init_from_adapter="owner/runs:sft/sft-run/seed0",
                lora_rank=16,
            )
        ),
        raising=False,
    )

    with pytest.raises(ValueError, match=r"SFT rank 24 \+ GRPO rank 16"):
        worker_adapter._init_adapter_model("Qwen/Qwen3.5-4B")


def test_init_adapter_model_uses_model_specific_serving_cap(tmp_path, monkeypatch):
    import flash.engine.worker.adapter as worker_adapter

    sft = str(tmp_path / "sft")
    merged = str(tmp_path / "merged")
    _write_adapter(sft, modules=MODULES, r=32, alpha=64, seed=1)
    monkeypatch.setattr(worker_adapter, "_download_adapter", lambda _prefix: sft)
    monkeypatch.setattr(worker_adapter, "adapter_is_vl_warmstart", lambda _adir, _model_id: True)
    monkeypatch.setattr(worker_adapter, "optimal_attn_impl", lambda: None)

    def fake_merge(_adir, _model_id, _attn_kw):
        W._VL_WARMSTART_SFT_DIR = _adir
        W._VL_WARMSTART_MODEL_ID = _model_id
        return merged

    monkeypatch.setattr(
        worker_adapter,
        "_merge_vl_warmstart_adapter",
        fake_merge,
    )
    monkeypatch.setattr(
        worker_adapter, "make_lora", lambda model_id, **_kwargs: {"model_id": model_id}
    )
    monkeypatch.setattr(W, "_VL_WARMSTART_MODEL_ID", None, raising=False)
    monkeypatch.setattr(
        W,
        "JOB_SPEC",
        types.SimpleNamespace(
            train=types.SimpleNamespace(
                init_from_adapter="owner/runs:sft/sft-run/seed0",
                lora_rank=32,
            )
        ),
        raising=False,
    )

    assert worker_adapter._init_adapter_model("Qwen/Qwen3.5-2B") == (
        merged,
        {"model_id": merged},
    )
    assert W._VL_WARMSTART_MODEL_ID == "Qwen/Qwen3.5-2B"


def test_init_adapter_model_clears_stale_vl_warmstart_state(monkeypatch):
    import flash.engine.worker.adapter as worker_adapter

    monkeypatch.setattr(
        worker_adapter, "make_lora", lambda model_id, **_kwargs: {"model_id": model_id}
    )
    monkeypatch.setattr(W, "_VL_WARMSTART_SFT_DIR", "/tmp/stale-sft", raising=False)
    monkeypatch.setattr(W, "_VL_WARMSTART_MODEL_ID", "Qwen/Qwen3.5-2B", raising=False)
    monkeypatch.setattr(
        W,
        "JOB_SPEC",
        types.SimpleNamespace(train=types.SimpleNamespace(init_from_adapter="")),
        raising=False,
    )

    assert worker_adapter._init_adapter_model("Qwen/Qwen3.5-4B") == (
        "Qwen/Qwen3.5-4B",
        {"model_id": "Qwen/Qwen3.5-4B"},
    )
    assert W._VL_WARMSTART_SFT_DIR is None
    assert W._VL_WARMSTART_MODEL_ID is None


def test_recombine_rejects_mismatched_target_modules(tmp_path):
    sft = str(tmp_path / "sft")
    grpo = str(tmp_path / "grpo")
    out = str(tmp_path / "out")
    _write_adapter(sft, modules=MODULES, r=4, alpha=8, seed=1)
    _write_adapter(
        grpo,
        modules=[*MODULES[:-1], "base_model.model.model.language_model.layers.9.mlp.up_proj"],
        r=4,
        alpha=8,
        seed=2,
    )
    with pytest.raises(ValueError, match=r"DIFFERENT modules"):
        recombine_lora_adapters(sft, grpo, out)


def test_recombine_rejects_dora(tmp_path):
    sft = str(tmp_path / "sft")
    grpo = str(tmp_path / "grpo")
    out = str(tmp_path / "out")
    _write_adapter(sft, modules=MODULES, r=4, alpha=8, seed=1, use_dora=True)
    _write_adapter(grpo, modules=MODULES, r=4, alpha=8, seed=2)
    with pytest.raises(ValueError, match=r"DoRA"):
        recombine_lora_adapters(sft, grpo, out)


def test_recombine_rejects_modules_to_save(tmp_path):
    sft = str(tmp_path / "sft")
    grpo = str(tmp_path / "grpo")
    out = str(tmp_path / "out")
    _write_adapter(sft, modules=MODULES, r=4, alpha=8, seed=1)
    _write_adapter(grpo, modules=MODULES, r=4, alpha=8, seed=2, modules_to_save=["lm_head"])
    with pytest.raises(ValueError, match=r"modules_to_save"):
        recombine_lora_adapters(sft, grpo, out)


def _patch_cfg(adir: str, **extra) -> None:
    """Merge ``extra`` into an already-written adapter_config.json (to inject unsupported fields)."""
    p = os.path.join(adir, "adapter_config.json")
    with open(p) as f:
        cfg = json.load(f)
    cfg.update(extra)
    with open(p, "w") as f:
        json.dump(cfg, f)


def test_recombined_rank_preflight_rejects_per_module_rank_pattern(tmp_path):
    sft = str(tmp_path / "sft")
    _write_adapter(sft, modules=MODULES, r=4, alpha=8, seed=1)
    _patch_cfg(sft, rank_pattern={"q_proj": 8})

    with pytest.raises(ValueError, match=r"rank_pattern"):
        validate_recombined_lora_rank(sft, 4, max_rank=32)


def test_recombine_rejects_non_lora_peft_type(tmp_path):
    sft = str(tmp_path / "sft")
    grpo = str(tmp_path / "grpo")
    out = str(tmp_path / "out")
    _write_adapter(sft, modules=MODULES, r=4, alpha=8, seed=1)
    _write_adapter(grpo, modules=MODULES, r=4, alpha=8, seed=2)
    # AdaLoRA (or any non-plain peft_type) has different scale/tensor semantics the cat-recombine
    # math doesn't model — it must fail loudly up front, not silently mis-deploy.
    _patch_cfg(grpo, peft_type="ADALORA")
    with pytest.raises(ValueError, match=r"peft_type"):
        recombine_lora_adapters(sft, grpo, out)


def test_recombine_rejects_per_module_rank_alpha_patterns(tmp_path):
    sft = str(tmp_path / "sft")
    grpo = str(tmp_path / "grpo")
    out = str(tmp_path / "out")
    # A non-empty rank_pattern/alpha_pattern breaks the uniform-(r, alpha) assumption and would be
    # silently dropped by out_cfg — reject both rather than emit an incorrectly-scaled adapter.
    for bad in ({"rank_pattern": {"q_proj": 8}}, {"alpha_pattern": {"q_proj": 16}}):
        _write_adapter(sft, modules=MODULES, r=4, alpha=8, seed=1)
        _write_adapter(grpo, modules=MODULES, r=4, alpha=8, seed=2)
        _patch_cfg(sft, **bad)
        with pytest.raises(ValueError, match=r"rank_pattern|alpha_pattern"):
            recombine_lora_adapters(sft, grpo, out)


def test_recombine_missing_safetensors_raises(tmp_path):
    sft = str(tmp_path / "sft")
    grpo = str(tmp_path / "grpo")
    out = str(tmp_path / "out")
    _write_adapter(sft, modules=MODULES, r=4, alpha=8, seed=1)
    os.makedirs(grpo)
    with open(os.path.join(grpo, "adapter_config.json"), "w") as f:
        json.dump({"r": 4, "lora_alpha": 8}, f)
    with pytest.raises(ValueError, match=r"no adapter_model\.safetensors"):
        recombine_lora_adapters(sft, grpo, out)


def test_recombine_missing_adapter_config_raises(tmp_path):
    sft = str(tmp_path / "sft")
    grpo = str(tmp_path / "grpo")
    out = str(tmp_path / "out")
    _write_adapter(sft, modules=MODULES, r=4, alpha=8, seed=1)
    _write_adapter(grpo, modules=TEXT_MODULES, r=4, alpha=8, seed=2)
    os.remove(os.path.join(grpo, "adapter_config.json"))  # weights present, config gone
    with pytest.raises(ValueError, match=r"no adapter_config\.json"):
        recombine_lora_adapters(sft, grpo, out)


def test_recombine_rejects_unpaired_lora_tensors(tmp_path):
    sft = str(tmp_path / "sft")
    grpo = str(tmp_path / "grpo")
    out = str(tmp_path / "out")
    # Both adapters carry a lora_A with NO paired lora_B (same key set, so the module-set check
    # passes) — recombine must name the missing B key rather than throw a bare KeyError.
    for adir in (sft, grpo):
        _write_adapter(adir, modules=TEXT_MODULES, r=4, alpha=8, seed=1)
        st = os.path.join(adir, "adapter_model.safetensors")
        sd = {k: v for k, v in load_file(st).items() if ".lora_B." not in k}
        save_file(sd, st, metadata={"format": "pt"})
    with pytest.raises(ValueError, match=r"no matching lora_B"):
        recombine_lora_adapters(sft, grpo, out)


def test_recombine_rejects_lora_b_without_paired_lora_a(tmp_path):
    sft = str(tmp_path / "sft")
    grpo = str(tmp_path / "grpo")
    out = str(tmp_path / "out")
    # Symmetric to the above: a lora_B with NO paired lora_A (same key set on both, so the module-set
    # check passes). The recombine loop iterates only lora_A keys, so an orphan B would be SILENTLY
    # dropped — recombine must instead fail loudly and name the unpaired key.
    for adir in (sft, grpo):
        _write_adapter(adir, modules=TEXT_MODULES, r=4, alpha=8, seed=1)
        st = os.path.join(adir, "adapter_model.safetensors")
        sd = {k: v for k, v in load_file(st).items() if ".lora_A." not in k}
        save_file(sd, st, metadata={"format": "pt"})
    with pytest.raises(ValueError, match=r"no matching lora_A"):
        recombine_lora_adapters(sft, grpo, out)


def _set_base(adir, value):
    cfg = _read_cfg(adir)
    if value is None:
        cfg.pop("base_model_name_or_path", None)
    else:
        cfg["base_model_name_or_path"] = value
    with open(os.path.join(adir, "adapter_config.json"), "w") as f:
        json.dump(cfg, f)


def test_recombine_names_catalog_base_from_sft_not_grpo_temp(tmp_path):
    # The GRPO adapter trained on the merged base names the ephemeral /tmp/flash_sft_merged_* dir as
    # its base. The recombined config must name the real catalog base (from the SFT config), never the
    # GRPO temp path.
    sft, grpo, out = str(tmp_path / "sft"), str(tmp_path / "grpo"), str(tmp_path / "out")
    _write_adapter(sft, modules=TEXT_MODULES, r=4, alpha=8, seed=1)  # base = Qwen/Qwen3.5-2B
    _write_adapter(grpo, modules=TEXT_MODULES, r=4, alpha=8, seed=2)
    _set_base(grpo, "/tmp/flash_sft_merged_xyz")
    recombine_lora_adapters(sft, grpo, out)
    assert _read_cfg(out)["base_model_name_or_path"] == "Qwen/Qwen3.5-2B"


def test_recombine_drops_stale_temp_base_when_sft_lacks_base(tmp_path):
    # If the SFT config carries no base to override with (external/legacy adapter), recombine must DROP
    # the field rather than inherit the GRPO adapter's now-deleted /tmp merged-base path.
    sft, grpo, out = str(tmp_path / "sft"), str(tmp_path / "grpo"), str(tmp_path / "out")
    _write_adapter(sft, modules=TEXT_MODULES, r=4, alpha=8, seed=1)
    _write_adapter(grpo, modules=TEXT_MODULES, r=4, alpha=8, seed=2)
    _set_base(sft, None)
    _set_base(grpo, "/tmp/flash_sft_merged_xyz")
    recombine_lora_adapters(sft, grpo, out)
    assert not _read_cfg(out).get("base_model_name_or_path")  # dropped, not a dangling temp path


def test_recombine_preserves_higher_dtype_of_either_adapter(tmp_path):
    # A higher-precision GRPO B must not be downcast to the SFT's dtype; output A and B stay consistent.
    sft, grpo, out = str(tmp_path / "sft"), str(tmp_path / "grpo"), str(tmp_path / "out")
    _write_adapter(sft, modules=TEXT_MODULES, r=4, alpha=8, seed=1, dtype=torch.float16)
    _write_adapter(grpo, modules=TEXT_MODULES, r=4, alpha=8, seed=2, dtype=torch.float32)
    recombine_lora_adapters(sft, grpo, out)
    sd = load_file(os.path.join(out, "adapter_model.safetensors"))
    b = next(v for k, v in sd.items() if ".lora_B." in k)
    a = next(v for k, v in sd.items() if ".lora_A." in k)
    assert b.dtype == torch.float32  # promoted, not downcast to fp16
    assert a.dtype == torch.float32  # cat auto-promotes; A/B stay consistent


# --- orchestrator gating: recombined_warmstart_adapter_dir(src) ---------------------------------
# Gated on the `_VL_WARMSTART_SFT_DIR` marker that _init_adapter_model sets ONLY on the VL merge path.
from flash.engine.worker._pkg import W


def test_orchestrator_noop_when_not_vl_warmstart(tmp_path, monkeypatch):
    # Marker unset: fresh-LoRA or non-VL continued-adapter run (saved adapter already deployable).
    grpo = str(tmp_path / "grpo")
    _write_adapter(grpo, modules=MODULES, r=4, alpha=8, seed=2)
    monkeypatch.setattr(W, "_VL_WARMSTART_SFT_DIR", None, raising=False)
    assert W.recombined_warmstart_adapter_dir(grpo) is None


def test_orchestrator_recombines_for_vl_warmstart(tmp_path, monkeypatch):
    # Marker set (VL merge happened): stack the SFT back into the GRPO-only saved adapter.
    sft = str(tmp_path / "sft")
    grpo = str(tmp_path / "grpo")
    # Production forms: SFT infixed (full VL model), GRPO text-only (AutoModelForCausalLM trainer).
    _write_adapter(sft, modules=MODULES, r=4, alpha=8, seed=1)
    _write_adapter(grpo, modules=TEXT_MODULES, r=4, alpha=8, seed=2)
    # an aux file should be carried into the deployable dir; trainer state must NOT be.
    with open(os.path.join(grpo, "special_tokens_map.json"), "w") as f:
        f.write("{}")
    with open(os.path.join(grpo, "optimizer.pt"), "w") as f:
        f.write("x")
    monkeypatch.setattr(W, "_VL_WARMSTART_SFT_DIR", sft, raising=False)

    out = W.recombined_warmstart_adapter_dir(grpo)
    assert out is not None
    assert out != grpo
    assert _read_cfg(out)["r"] == 8  # r_sft + r_grpo
    assert os.path.isfile(os.path.join(out, "special_tokens_map.json"))  # aux carried over
    assert not os.path.exists(os.path.join(out, "optimizer.pt"))  # trainer state skipped
    # exactness: recombined delta == SFT_delta + GRPO_delta. Output uses the serving language_model
    # namespace, so the text-only GRPO module maps back to the SFT/output module.
    for m_sft, m_text in zip(MODULES, TEXT_MODULES, strict=True):
        assert torch.allclose(
            _delta(out, m_sft), _delta(sft, m_sft) + _delta(grpo, m_text), atol=1e-6, rtol=1e-5
        )


def test_orchestrator_recombine_uses_recorded_model_cap(tmp_path, monkeypatch):
    # Regression: init preflight uses the selected model's serving cap, so finalize must use the
    # same cap instead of falling back to rank 32 when the SFT adapter lacks base_model_name_or_path.
    sft = str(tmp_path / "sft")
    grpo = str(tmp_path / "grpo")
    _write_adapter(sft, modules=MODULES, r=32, alpha=64, seed=1)
    _write_adapter(grpo, modules=TEXT_MODULES, r=32, alpha=64, seed=2)
    _set_base(sft, None)
    monkeypatch.setattr(W, "_VL_WARMSTART_SFT_DIR", sft, raising=False)
    monkeypatch.setattr(W, "_VL_WARMSTART_MODEL_ID", "Qwen/Qwen3.5-2B", raising=False)

    out = W.recombined_warmstart_adapter_dir(grpo)

    assert out is not None
    assert _read_cfg(out)["r"] == 64


def test_orchestrator_cleans_temp_dir_when_recombine_fails(tmp_path, monkeypatch):
    # If the recombine raises (malformed adapter / config guard), the freshly-created
    # flash_recomb_adapter_* temp dir must be removed — the per-step publish path catches and
    # continues, so a leak would accumulate under /tmp across repeated failures.
    import tempfile as _tempfile

    import flash.engine.worker.adapter as A

    sft = tmp_path / "sft"
    sft.mkdir()  # marker dir must exist; contents irrelevant (recombine is stubbed to raise)
    grpo = str(tmp_path / "grpo")
    os.makedirs(grpo)
    monkeypatch.setattr(W, "_VL_WARMSTART_SFT_DIR", str(sft), raising=False)

    created: list[str] = []
    real_mkdtemp = _tempfile.mkdtemp

    def tracking_mkdtemp(*a, **k):
        d = real_mkdtemp(*a, dir=str(tmp_path), **k)
        created.append(d)
        return d

    monkeypatch.setattr(_tempfile, "mkdtemp", tracking_mkdtemp)

    def boom(*a, **k):
        raise ValueError("recombine: simulated failure")

    monkeypatch.setattr(A, "recombine_lora_adapters", boom)

    with pytest.raises(ValueError, match="simulated failure"):
        W.recombined_warmstart_adapter_dir(grpo)

    assert created, "mkdtemp should have been called"
    for d in created:
        assert not os.path.exists(d), f"leaked temp dir {d}"


def test_orchestrator_raises_when_recorded_sft_dir_missing(tmp_path, monkeypatch):
    # The VL merge baked the SFT into the (ephemeral) training base, so the saved GRPO adapter is
    # SFT-less. If the recorded SFT dir is gone at finalize we MUST fail loud, not ship it broken.
    grpo = str(tmp_path / "grpo")
    _write_adapter(grpo, modules=MODULES, r=4, alpha=8, seed=2)
    monkeypatch.setattr(W, "_VL_WARMSTART_SFT_DIR", str(tmp_path / "gone"), raising=False)
    with pytest.raises(RuntimeError, match=r"SFT-less"):
        W.recombined_warmstart_adapter_dir(grpo)
