"""CPU tests for the pool trainer's optimization config + model->GPU plan.

The kernel application itself (Liger/Chalk/QLoRA/FLA-drop/8-bit AdamW) runs on the GPU inside
``build_lora_policy_update`` (covered live); here we pin that the SAME optimization SET as
origin/dev is selected per model, and that every catalog model maps to a GPU — including 35B, which
the pool can train (QLoRA trainer fits an ordinary card) where dev's colocate needs an H200.
"""

from __future__ import annotations

from flash.engine.pool_policy import _params_b, pool_gpu_plan, resolve_opt_config


def test_params_b_parsing():
    assert _params_b("Qwen/Qwen3.5-0.8B") == 0.8
    assert _params_b("Qwen/Qwen3.5-4B") == 4.0
    assert _params_b("Qwen/Qwen3.6-35B-A3B") == 35.0


def test_qwen35_gets_full_dev_stack():
    # Qwen3.5/3.6 bf16 tier: Liger + Chalk + FLA-drop + 8-bit AdamW, language-only LoRA, no QLoRA.
    oc = resolve_opt_config("Qwen/Qwen3.5-4B")
    assert oc.liger
    assert oc.chalk
    assert oc.drop_fla
    assert oc.optim_8bit
    assert oc.qlora is False
    assert oc.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def test_catalog_4bit_tiers_enable_qlora():
    # 9B and 35B-A3B are 4bit-qlora in the catalog -> QLoRA on by default.
    assert resolve_opt_config("Qwen/Qwen3.5-9B").qlora is True
    assert resolve_opt_config("Qwen/Qwen3.6-35B-A3B").qlora is True
    # a bf16 model -> QLoRA off
    assert resolve_opt_config("Qwen/Qwen3.5-2B").qlora is False


def test_non_qwen35_skips_fla_drop():
    # FLA-drop is a Qwen3.5/3.6-on-Hopper fix; MiniCPM (Llama-ish) must not trigger it.
    assert resolve_opt_config("openbmb/MiniCPM5-1B").drop_fla is False


def test_overrides_respected():
    oc = resolve_opt_config("Qwen/Qwen3.5-4B", liger=False, qlora=True, optim_8bit=False, target_modules="all-linear")
    assert oc.liger is False
    assert oc.qlora is True
    assert oc.optim_8bit is False
    assert oc.target_modules == "all-linear"


def test_every_model_maps_to_a_gpu():
    from flash.catalog import MODELS

    for model_id in MODELS:
        plan = pool_gpu_plan(model_id)
        assert plan["trainer_gpu"], model_id
        assert plan["inference_gpu"], model_id
        assert plan["trainer_vram_gb"] > 0


def test_35b_trainer_fits_ordinary_card_via_qlora():
    # The disaggregation + QLoRA win: dev's colocate needs ~103 GB (H200) for 35B; the pool's QLoRA
    # trainer (no colocated vLLM) must fit well under that.
    plan = pool_gpu_plan("Qwen/Qwen3.6-35B-A3B")
    assert plan["qlora"] is True
    assert plan["trainer_vram_gb"] < 48  # fits a 48 GB card (vs 103 GB colocate)
    # and a QLoRA 35B trainer is far cheaper than an H200
    assert plan["trainer_usd_hr"] < 2.0
