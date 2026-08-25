from __future__ import annotations

from flash.serving.src.store import settings as cfg
from flash.serving.src.store.settings import Settings


def test_settings_holds_only_runtime_wiring() -> None:
    # settings is runtime wiring only, with no tuning knobs.
    s = Settings(_env_file=None)
    allowed = {
        "hf_api_key",
        "internal_key",
        "deployment_sha",
        "deployment_id",
        "backend_url",
        "supabase_url",
        "supabase_service_role_key",
    }
    assert set(type(s).model_fields) == allowed


def test_hf_token_is_canonical() -> None:
    assert Settings(_env_file=None, HF_TOKEN="hf_secret").hf_api_key == "hf_secret"


def test_backend_url_has_no_production_default() -> None:
    assert Settings(_env_file=None).backend_url == ""


def test_internal_key_reads_canonical_env() -> None:
    assert Settings(_env_file=None, FREESOLO_INTERNAL_KEY="k1").internal_key == "k1"


def test_deployment_identity_is_optional_and_reads_canonical_env() -> None:
    settings = Settings(_env_file=None)
    assert settings.deployment_sha == ""
    assert settings.deployment_id == ""

    settings = Settings(
        _env_file=None,
        FREESOLO_DEPLOYMENT_SHA="abc123",
        FREESOLO_DEPLOYMENT_ID="456-2",
    )
    assert settings.deployment_sha == "abc123"
    assert settings.deployment_id == "456-2"


def test_vllm_engine_kwargs_is_minimal_and_always_on() -> None:
    # Only the proven, always-on engine config; everything else uses vLLM defaults.
    assert cfg.vllm_engine_kwargs() == {
        "enable_prefix_caching": True,
        "disable_log_stats": True,
    }


def test_proven_values_are_hardcoded_constants() -> None:
    assert cfg.ENABLE_PREFIX_CACHING is True
    assert cfg.DISABLE_LOG_STATS is True
    # max_loras (hot adapters/batch) and max_lora_rank both linearly size the pre-allocated LoRA
    # buffers (the dominant serving VRAM cost). The global default stays small for L4 models; larger
    # bases can override after real-GPU validation.
    assert cfg.MAX_LORAS == 16
    assert cfg.MAX_LORA_RANK == 32
    assert cfg.PROMPT_TOKEN_CACHE_SIZE == 2048
    assert cfg.PRELOAD_CACHED_LORAS is True
    assert cfg.RELOAD_INTERVAL_SECONDS == 30.0
    # FP8 is baked in everywhere (memory-first): online fp8 weight quant + fp8 KV cache. These are
    # module constants (like DTYPE), NOT Settings env knobs — see test_deleted_knobs_are_gone.
    assert cfg.QUANTIZATION == "fp8"
    assert cfg.KV_CACHE_DTYPE == "fp8"


def test_deleted_knobs_are_gone_from_settings() -> None:
    # The neutral/losing canary toggles were removed entirely (no opt-in, no env knob).
    s = Settings(_env_file=None)
    for gone in (
        "enforce_eager",
        "kv_cache_dtype",
        "speculative_config_json",
        "performance_mode",
        "max_num_seqs",
        "max_num_batched_tokens",
        "enable_chunked_prefill",
        "async_scheduling",
        "scheduler_delay_factor",
        "stream_interval",
        "specialize_active_lora",
        "preemption_mode",
        "quantization",
        "load_format",
    ):
        assert not hasattr(s, gone), f"{gone} should have been deleted"
    assert not hasattr(s, "vllm_scheduler_kwargs")
