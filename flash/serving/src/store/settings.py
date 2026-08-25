from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

HOSTING_CACHE_DIR = Path("/vol/hosting-cache")
ADAPTER_CACHE_DIR = HOSTING_CACHE_DIR / "adapters"

# ── Hardcoded serving config (no knobs) ───────────────────────────────────────────────────────
# An optimization either exists (baked in here) or it doesn't — there is nothing to tune at deploy
# time, so these are plain constants, not env vars. Every value is the proven/safe choice; the
# losing or neutral canary toggles were deleted (vLLM's defaults are used everywhere else). Change
# a value here in code if it ever needs to move.
TRUST_REMOTE_CODE = True
DTYPE = "auto"
# FP8 everywhere, baked in (memory-first; see README "Quantization"). For models without
# `serve_model_id`, vLLM ONLINE-quantizes the normal bf16 HF checkpoint to FP8 (E4M3) at load time.
# Larger validated tiers can instead set model_config.engine.serve_model_id to load a pre-quantized
# FP8 checkpoint directly, avoiding the bf16 load transient. The old per-base GPTQ-Int8 repos were
# removed because they 404'd; the current serve_model_id entries point only at verified FP8 repos.
# Native FP8 tensor cores exist on compute capability >= 8.9 (L4/L40S/H100); A100 works through
# vLLM's Marlin weight-only FP8 fallback. A per-model "quantization" override can set this to None
# for a deliberate bf16 fallback; pre-quantized checkpoints also pass None so vLLM auto-detects the
# checkpoint's quantization.
QUANTIZATION = "fp8"
# FP8 KV cache (E4M3): 1 byte/element vs 2 for bf16 -> ~50% KV-cache VRAM, i.e. ~2x more cacheable
# tokens (longer context / more concurrent sequences on the same card). It is just a storage dtype,
# orthogonal to weight quant and SAFE with multi-LoRA AND the fused MoE, so it is on for every base,
# including pre-quantized checkpoints and any explicit bf16 fallback. Uncalibrated dynamic e4m3
# retains ~97-98% accuracy; we do NOT turn on calculate_kv_scales (warmup-estimated scales corrupt
# the Qwen3 GDN-hybrid's recurrent state). A serving canary previously measured fp8 KV at +8-10%
# throughput under load.
KV_CACHE_DTYPE = "fp8"
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.90
# Fallback 8192-token serving context for models without an engine override. Catalog entries may
# raise or lower it after per-model real-GPU validation.
MAX_MODEL_LEN = 8192
# vLLM PRE-ALLOCATES the GPU LoRA buffers at engine init, sized max_loras x max_lora_rank (NOT by the
# number of adapters actually loaded), so BOTH are linear VRAM levers and together the single biggest
# serving cost — larger than the (now-FP8) base weights. max_loras is how many DISTINCT adapters serve
# in one batch and stay GPU-resident; beyond it vLLM LRU-evicts to the CPU pool (max_cpu_loras) and
# copies back on a miss (~2-15 ms added TTFT, model-size dependent). The global default keeps the small
# L4 models cheap; larger bases can override this in model_config after real-GPU validation. 256 adapters
# can still be DEPLOYED per base (max_cpu_loras); only the configured max_loras are hot at once.
MAX_LORAS = 16
# vLLM also pre-allocates LoRA buffers at max_lora_rank (NOT the adapter's real rank). Flash trains at
# rank 32 by default (recipe.py LoRAConfig.rank=32), so the global fallback stays 32. Curated small
# L4 tiers override to rank 64; larger pre-quantized tiers keep rank 32. TRADEOFF: an adapter trained
# above the effective rank is REJECTED at load (vLLM raises; POST /adapters fails loudly and rolls back).
MAX_LORA_RANK = 32
MAX_CPU_LORAS = 256
PROMPT_TOKEN_CACHE_SIZE = 2048
PRELOAD_CACHED_LORAS = True
# Prefix caching: measured ~4.7x throughput / ~10x lower TTFT for shared system prompts, zero
# overhead on misses. CUDA graphs stay on by default, with per-model eager fallbacks when boot
# canaries show that graph memory leaves too little KV-cache headroom.
ENABLE_PREFIX_CACHING = True
DISABLE_LOG_STATS = True
RELOAD_INTERVAL_SECONDS = 30.0
ADAPTER_TABLE = "hosted_lora_adapters"


def vllm_engine_kwargs() -> dict[str, Any]:
    """The only always-on engine config passed to vLLM; everything else uses vLLM defaults."""
    return {
        "enable_prefix_caching": ENABLE_PREFIX_CACHING,
        "disable_log_stats": DISABLE_LOG_STATS,
    }


class Settings(BaseSettings):
    """Runtime credentials, endpoints, and deployment identity; never tuning knobs."""

    hf_api_key: str | None = Field(default=None, validation_alias="HF_TOKEN")
    # The single shared platform internal key. Serving uses it for every trusted-infra purpose:
    # authenticating adapter (un)registration on POST/DELETE /adapters, bypassing external chat auth
    # for trusted server-to-server callers, and authenticating serving's own calls back to the
    # backend (usage metering + /api/serving/authorize). There is no separate serving-only key.
    internal_key: str | None = Field(
        default=None,
        validation_alias="FREESOLO_INTERNAL_KEY",
    )
    deployment_sha: str = Field(
        default="",
        validation_alias="FREESOLO_DEPLOYMENT_SHA",
    )
    deployment_id: str = Field(
        default="",
        validation_alias="FREESOLO_DEPLOYMENT_ID",
    )
    # The platform backend (FastAPI) base URL. Durable outbox delivery settles usage through
    # {backend_url}/api/billing/serving-usage/durable and external chat authorization uses
    # {backend_url}/api/serving/authorize, both authenticated with ``internal_key``. Hosted serving
    # requires complete backend, Supabase, deployment, and credential wiring; no default URL exists.
    backend_url: str = Field(
        default="",
        validation_alias="PLATFORM_BACKEND_URL",
    )
    supabase_url: str | None = Field(
        default=None,
        validation_alias="SUPABASE_URL",
    )
    supabase_service_role_key: str | None = Field(
        default=None,
        validation_alias="SUPABASE_SERVICE_ROLE_KEY",
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def has_supabase(self) -> bool:
        return bool(self.supabase_url and self.supabase_service_role_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
