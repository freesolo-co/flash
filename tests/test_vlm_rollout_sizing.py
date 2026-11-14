"""Vision-language GRPO rollout sizing on one colocated GPU, and fail-fast on a dead engine.

Regression for an 80 GB H100 Qwen3.5-9B-Base GRPO run (4096 context, 16 sequences, rank 32) that
the planner admitted and the worker sized to gpu_memory_utilization=0.3939: vllm died with
"No available memory for the cache blocks" and the worker heartbeated at step 0 for 20 minutes.
"""

from __future__ import annotations

import pytest

from flash.core.catalog import MODELS, supports_image_training
from flash.engine.plan import kv_sizing
from flash.engine.plan.kv_sizing import (
    _VLM_ENGINE_RESERVE_GB,
    colocate_kv_util,
    grpo_kv_floor_gb,
    vision_engine_reserve_gb,
)
from flash.engine.worker.verl import diagnostics

VLM = "Qwen/Qwen3.5-9B-Base"
H100_GB = 80
# the failed run: weights plus vllm's non-kv runtime already exceeded this budget.
FAILED_UTIL = 0.3939


def _vlm_util(total_gb, *, tp=1, sleep=True, reserve=None):
    info = MODELS[VLM]
    return colocate_kv_util(
        info.params_b,
        4096,
        total_gb,
        sleep_mode=sleep,
        num_generations=8,
        model_info=info,
        tensor_parallel=tp,
        lora_rank=32,
        rollout_cache_min_gb=getattr(info, "rollout_cache_min_gb", 0.0),
        vision_reserve_gb=vision_engine_reserve_gb(VLM) if reserve is None else reserve,
    )


def test_reserve_applies_to_vision_models_only():
    assert supports_image_training(VLM)
    assert vision_engine_reserve_gb(VLM) == _VLM_ENGINE_RESERVE_GB
    text_only = next(m for m in MODELS if not supports_image_training(m))
    assert vision_engine_reserve_gb(text_only) == 0.0


def test_h100_vlm_engine_has_a_kv_pool_beyond_the_failed_budget():
    weights_gb = MODELS[VLM].params_b * 2.0
    util = _vlm_util(H100_GB)
    engine_gb = util * H100_GB
    # the failed budget left vllm no kv: its non-kv runtime is >= 31.5 - weights. the new engine
    # must hold that measured runtime plus the 8 GB minimum kv pool.
    measured_runtime_gb = FAILED_UTIL * H100_GB - weights_gb
    assert engine_gb - weights_gb >= measured_runtime_gb + kv_sizing._KV_CAP
    assert util <= kv_sizing._VLM_SINGLE_GPU_UTIL_CAP
    # the trainer's resident bf16 copy and its headroom still fit beside the engine.
    assert H100_GB - engine_gb >= weights_gb + kv_sizing._TRAINER_HEADROOM_GB


@pytest.mark.parametrize("total_gb", [141, 180])
def test_larger_cards_need_no_lift(total_gb):
    util = _vlm_util(total_gb)
    assert util <= kv_sizing._colocate_util_cap(MODELS[VLM].params_b * 2.0, total_gb)


def test_text_models_keep_the_previous_budget():
    info = MODELS[VLM]
    args = (info.params_b, 4096, H100_GB)
    kwargs = {
        "sleep_mode": True,
        "num_generations": 8,
        "model_info": info,
        "lora_rank": 32,
        "rollout_cache_min_gb": getattr(info, "rollout_cache_min_gb", 0.0),
    }
    # without a vision reserve the cap and budget are exactly the pre-change computation.
    assert colocate_kv_util(*args, **kwargs, vision_reserve_gb=0.0) == colocate_kv_util(
        *args, **kwargs
    )
    assert colocate_kv_util(*args, **kwargs) <= 0.45


def test_lift_is_single_gpu_sleep_mode_only():
    weights_gb = MODELS[VLM].params_b * 2.0
    fixed = weights_gb + 14.0
    common = {"vision_reserve_gb": 14.0}
    lifted = kv_sizing._engine_util_cap(weights_gb, fixed, 80, single_rank_sleep=True, **common)
    resident = kv_sizing._engine_util_cap(weights_gb, fixed, 80, single_rank_sleep=False, **common)
    assert lifted > 0.45
    assert resident == kv_sizing._colocate_util_cap(weights_gb, 80)


def test_planner_floor_and_worker_budget_agree():
    info = MODELS[VLM]
    reserve = vision_engine_reserve_gb(VLM)
    floor = grpo_kv_floor_gb(
        info.params_b,
        4096,
        8,
        model_info=info,
        lora_rank=32,
        rollout_cache_min_gb=getattr(info, "rollout_cache_min_gb", 0.0),
        vision_reserve_gb=reserve,
    )
    weights_gb = info.params_b * 2.0
    for card in (floor, floor + 10, H100_GB):
        if card < floor:
            continue
        engine_gb = _vlm_util(card) * card
        # every card the planner admits gives the worker room for weights, reserve and a kv pool.
        assert engine_gb - weights_gb - reserve > 0
    # without the reserve the planner would admit a smaller card than the engine can use.
    smaller = grpo_kv_floor_gb(
        info.params_b,
        4096,
        8,
        model_info=info,
        lora_rank=32,
        rollout_cache_min_gb=getattr(info, "rollout_cache_min_gb", 0.0),
    )
    assert smaller <= floor


def _watchdog():
    tail = diagnostics.ChildOutputTail()
    watchdog = diagnostics.VerlChildSilenceWatchdog(tail, baseline_step=0)
    torn = []
    watchdog.bind(child_alive=lambda: True, teardown=lambda: torn.append(True))
    return tail, watchdog, torn


@pytest.mark.parametrize(
    "line",
    [
        (
            "(EngineCore pid=3258) ERROR [core.py:1108] ValueError: No available memory for the cache "
            "blocks. Try increasing `gpu_memory_utilization` when initializing the engine."
        ),
        "RuntimeError: Engine core initialization failed. See root cause above.",
    ],
)
def test_dead_rollout_engine_ends_the_run_before_the_first_step(line):
    tail, watchdog, torn = _watchdog()
    tail.record(line)
    watchdog.check()  # never armed: no step has run
    assert torn == [True]
    with pytest.raises(RuntimeError, match="rollout_engine_failed"):
        watchdog.raise_if_failed()


def test_ordinary_teardown_noise_does_not_fail_the_run():
    tail, watchdog, torn = _watchdog()
    tail.record("vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue.")
    watchdog.check()
    assert torn == []
    watchdog.raise_if_failed()
