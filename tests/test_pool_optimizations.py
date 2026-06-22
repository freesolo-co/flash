"""CPU tests for the pool trainer's optimization config + model->GPU plan.

The kernel application itself (Chalk/FLA-drop/8-bit AdamW) runs on the GPU inside
``build_lora_policy_update`` (covered live); here we pin that the SAME optimization SET as
origin/dev is selected per model (chalk standalone, Liger faded per flash#66; QLoRA removed in #74),
and that every catalog model maps to a GPU — including 35B, which the pool's LoRA-only (no colocated
vLLM) trainer can land on a cheaper card than dev's H200 colocate.
"""

from __future__ import annotations

from flash.engine.pool_policy import _params_b, pool_gpu_plan, resolve_opt_config


def test_params_b_parsing():
    assert _params_b("Qwen/Qwen3.5-0.8B") == 0.8
    assert _params_b("Qwen/Qwen3.5-4B") == 4.0
    assert _params_b("Qwen/Qwen3.6-35B-A3B") == 35.0


def test_qwen35_gets_full_dev_stack():
    # Qwen3.5/3.6 bf16 tier: chalk standalone, Liger faded (flash#66) + FLA-drop + 8-bit AdamW,
    # language-only LoRA, no QLoRA.
    oc = resolve_opt_config("Qwen/Qwen3.5-4B")
    assert oc.liger is False
    assert oc.chalk
    assert oc.drop_fla
    assert oc.optim_8bit
    assert oc.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def test_catalog_is_all_bf16_so_qlora_off_everywhere():
    # dev removed QLoRA (#74): the catalog is all-bf16 now, so QLoRA is off for every model.
    assert pool_gpu_plan("Qwen/Qwen3.5-9B")["qlora"] is False
    # a bf16 model -> QLoRA off
    assert pool_gpu_plan("Qwen/Qwen3.5-2B")["qlora"] is False


def test_non_qwen35_skips_fla_drop():
    # FLA-drop is a Qwen3.5/3.6-on-Hopper fix; MiniCPM (Llama-ish) must not trigger it.
    assert resolve_opt_config("openbmb/MiniCPM5-1B").drop_fla is False


def test_overrides_respected():
    oc = resolve_opt_config("Qwen/Qwen3.5-4B", liger=True, optim_8bit=False, target_modules="all-linear")
    assert oc.liger is True
    assert oc.optim_8bit is False
    assert oc.target_modules == "all-linear"


def test_every_model_maps_to_a_gpu():
    from flash.catalog import MODELS

    for model_id in MODELS:
        plan = pool_gpu_plan(model_id)
        assert plan["trainer_gpu"], model_id
        assert plan["inference_gpu"], model_id
        assert plan["trainer_vram_gb"] > 0


def test_35b_trainer_is_bf16_no_qlora():
    # QLoRA was removed (#74) — GRPO merges the LoRA into the 4-bit base and collapses the importance
    # ratio (no learning). The 35B trainer is bf16 now; the disaggregation win is that the LoRA-only
    # trainer still avoids the colocated vLLM, so it lands on a cheaper card than dev's H200 colocate.
    plan = pool_gpu_plan("Qwen/Qwen3.6-35B-A3B")
    assert plan["qlora"] is False
    assert plan["trainer_usd_hr"] < 5.0  # still far cheaper than an H200 colocate
