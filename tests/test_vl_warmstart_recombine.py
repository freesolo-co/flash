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

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import load_file, save_file

from flash.engine.worker.lora import recombine_lora_adapters

# Realistic Qwen3.5 VL adapter key stems (FULL multimodal model -> LM under language_model.).
MODULES = [
    "base_model.model.model.language_model.layers.0.self_attn.q_proj",
    "base_model.model.model.language_model.layers.0.self_attn.v_proj",
    "base_model.model.model.language_model.layers.5.mlp.gate_proj",
]


def _write_adapter(adir: str, *, modules, r, alpha, in_f=8, out_f=6, use_rslora=False, seed=0,
                   use_dora=False, modules_to_save=None, dtype=torch.float32):
    os.makedirs(adir, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for m in modules:
        sd[f"{m}.lora_A.weight"] = torch.randn(r, in_f, generator=g, dtype=dtype) * 0.1
        sd[f"{m}.lora_B.weight"] = torch.randn(out_f, r, generator=g, dtype=dtype) * 0.1
    save_file(sd, os.path.join(adir, "adapter_model.safetensors"), metadata={"format": "pt"})
    cfg = {
        "peft_type": "LORA", "r": r, "lora_alpha": alpha, "lora_dropout": 0.0,
        "use_rslora": use_rslora, "use_dora": use_dora,
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
    A = sd[f"{module}.lora_A.weight"].to(torch.float64)
    B = sd[f"{module}.lora_B.weight"].to(torch.float64)
    return _scale(cfg) * (B @ A)


@pytest.mark.parametrize(
    ("r_sft", "a_sft", "rs_sft", "r_grpo", "a_grpo", "rs_grpo"),
    [
        (4, 8, False, 4, 8, False),    # equal scale 2.0 (the common managed-flash case)
        (4, 8, False, 2, 8, False),    # differing scales 2.0 vs 4.0, differing ranks
        (8, 16, True, 4, 8, False),    # rsLoRA on one side
    ],
)
def test_recombine_reproduces_sum_of_deltas(tmp_path, r_sft, a_sft, rs_sft, r_grpo, a_grpo, rs_grpo):
    sft = str(tmp_path / "sft")
    grpo = str(tmp_path / "grpo")
    out = str(tmp_path / "out")
    _write_adapter(sft, modules=MODULES, r=r_sft, alpha=a_sft, use_rslora=rs_sft, seed=1)
    _write_adapter(grpo, modules=MODULES, r=r_grpo, alpha=a_grpo, use_rslora=rs_grpo, seed=2)

    rank = recombine_lora_adapters(sft, grpo, out)
    assert rank == r_sft + r_grpo

    out_cfg = _read_cfg(out)
    # Scales are baked into B, so the recombined adapter carries unit scale (alpha == r).
    assert out_cfg["r"] == r_sft + r_grpo
    assert out_cfg["lora_alpha"] == r_sft + r_grpo
    assert out_cfg["use_rslora"] is False

    # The recombined delta must EXACTLY equal SFT_delta + GRPO_delta on the original base.
    for m in MODULES:
        want = _delta(sft, m) + _delta(grpo, m)
        got = _delta(out, m)
        assert torch.allclose(got, want, atol=1e-6, rtol=1e-5), f"{m}: recombine delta mismatch"


def test_recombine_rejects_mismatched_target_modules(tmp_path):
    sft = str(tmp_path / "sft")
    grpo = str(tmp_path / "grpo")
    out = str(tmp_path / "out")
    _write_adapter(sft, modules=MODULES, r=4, alpha=8, seed=1)
    _write_adapter(grpo, modules=[*MODULES[:-1], "base_model.model.model.language_model.layers.9.mlp.up_proj"],
                   r=4, alpha=8, seed=2)
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


# --- orchestrator gating: recombined_warmstart_adapter_dir(src) ---------------------------------
# Gated on the `_VL_WARMSTART_SFT_DIR` marker that _init_adapter_model sets ONLY on the VL merge path.
import flash.engine.worker as W


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
    _write_adapter(sft, modules=MODULES, r=4, alpha=8, seed=1)
    _write_adapter(grpo, modules=MODULES, r=4, alpha=8, seed=2)
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
    # exactness: recombined delta == SFT_delta + GRPO_delta
    for m in MODULES:
        assert torch.allclose(_delta(out, m), _delta(sft, m) + _delta(grpo, m), atol=1e-6, rtol=1e-5)


def test_orchestrator_raises_when_recorded_sft_dir_missing(tmp_path, monkeypatch):
    # The VL merge baked the SFT into the (ephemeral) training base, so the saved GRPO adapter is
    # SFT-less. If the recorded SFT dir is gone at finalize we MUST fail loud, not ship it broken.
    grpo = str(tmp_path / "grpo")
    _write_adapter(grpo, modules=MODULES, r=4, alpha=8, seed=2)
    monkeypatch.setattr(W, "_VL_WARMSTART_SFT_DIR", str(tmp_path / "gone"), raising=False)
    with pytest.raises(RuntimeError, match=r"SFT-less"):
        W.recombined_warmstart_adapter_dir(grpo)
