"""Progressively-enriched system prompts for the LLM cost estimator.

The experiment's independent variable. Version 1 asks Claude to price a run with no
Flash-specific knowledge; each later version appends one more *layer* of framework
fact, in the order a human would need them to reach the right number:

    v1  base task only (general world knowledge)
    v2  + GPU price/VRAM catalog
    v3  + how Flash picks a GPU (cheapest validated class that fits)
    v4  + the per-step compute/throughput model + recipe token counts
    v5  + GRPO rollout split, MoE active-params, QLoRA
    v6  + cold-start overhead and the exact end-to-end cost formula

Every layer is rendered from the *live* framework data (the GPU registry, the model
catalog, the frozen recipe, and the analytical constants) so the prompts can never
drift from the code they describe. ``experiment.py`` measures how the estimator's
error against the analytical ground truth shrinks as the version climbs.
"""

from __future__ import annotations

from flash.catalog import list_models
from flash.engine.recipe import RECIPE
from flash.providers.base import GPU_CLASSES

from . import analytical as A
from .config import RunConfig
from .hardware import GPU_COMPUTE_TFLOPS

NUM_VERSIONS = 6

VERSION_LABELS: dict[int, str] = {
    1: "v1 base",
    2: "v2 +pricing",
    3: "v3 +GPU pick",
    4: "v4 +timing",
    5: "v5 +GRPO/MoE",
    6: "v6 +formula",
}

BASE_SYSTEM = """\
You are a cost estimator for Flash, a managed LoRA post-training service that runs
SFT and GRPO fine-tuning jobs on rented single-GPU instances. Given one training run,
estimate the total USD cost to run it end to end (the instance is billed for its whole
wall-clock lifetime, including cold-start setup, not just training).

Reply with ONE JSON object and nothing else:
{"cost_usd": <number>, "gpu": "<GPU class you assumed>", "reasoning": "<= 60 words"}
"""


def _pricing_layer() -> str:
    rows = []
    for g in GPU_CLASSES:
        val = "validated" if g.validated else "unvalidated"
        rows.append(f"  {g.name:<18} {g.vram_gb:>3} GB  ${g.hourly_usd:>5.2f}/hr  ({val})")
    return (
        "## GPU hardware & pricing\n"
        "Flash rents one GPU per run across RunPod Flash and Vast.ai. Managed classes "
        "(static fallback $/hr; the live market is similar):\n" + "\n".join(rows)
    )


def _selection_layer() -> str:
    rows = [
        f"  {m.id:<22} {m.params:<28} min_vram={m.min_vram_gb} GB  quant={m.quant}"
        for m in list_models()
    ]
    return (
        "## How Flash picks the GPU\n"
        "A run lands on the CHEAPEST validated GPU class whose VRAM fits the run "
        f"(sized with {A.vram_headroom():.0%} headroom). Most jobs fit consumer/prosumer "
        "cards (RTX A5000/4090/5090), NOT big datacenter GPUs. SFT holds one weight copy; "
        "GRPO colocates a second (vLLM rollout) copy + KV cache, so GRPO floors at "
        "~28 GB (24 GB for <=1B models) and routes to a bigger card than the same model's "
        "SFT. Curated models and sizes:\n" + "\n".join(rows)
    )


def _timing_layer() -> str:
    tf = "\n".join(
        f"  {name:<18} {flops:>5.0f} TFLOPS"
        for name, flops in sorted(GPU_COMPUTE_TFLOPS.items(), key=lambda kv: kv[1])
    )
    return (
        "## Per-step compute / throughput\n"
        "Wall clock = cold-start setup + steps x seconds-per-step. Per-step compute "
        "(FLOPs) ~= multiplier x active_params x tokens_processed, and\n"
        "  seconds_per_step = FLOPs / (gpu_peak_bf16_FLOPs x MFU)\n"
        f"SFT: multiplier = {A.SFT_FLOPS_PER_TOKEN_PER_PARAM:.0f} (fwd+bwd), "
        f"MFU_train = {A.MFU_TRAIN}; tokens/step = effective_batch x seq_len = "
        f"{RECIPE.sft.effective_batch} x {RECIPE.sft.max_seq_len} = "
        f"{RECIPE.sft.effective_batch * RECIPE.sft.max_seq_len}.\n"
        "GPU peak bf16 throughput:\n" + tf
    )


def _grpo_moe_layer() -> str:
    return (
        "## GRPO rollout cost, MoE, and QLoRA\n"
        f"A GRPO step has two phases. (1) Rollout: vLLM generates "
        f"prompts_per_step x group_size = {RECIPE.rl.prompts_per_step} x "
        f"{RECIPE.rl.group_size} = {RECIPE.rl.prompts_per_step * RECIPE.rl.group_size} "
        f"completions of ~{RECIPE.rl.max_completion_len} tokens; cost = "
        f"{A.GRPO_GEN_FLOPS_PER_TOKEN_PER_PARAM:.0f} x active_params x gen_tokens / "
        f"(peak x MFU_decode={A.MFU_DECODE}). (2) Update: policy fwd+bwd + frozen-ref "
        f"fwd over those completions; cost = {A.GRPO_UPDATE_FLOPS_PER_TOKEN_PER_PARAM:.0f} "
        f"x active_params x gen_tokens / (peak x MFU_train={A.MFU_TRAIN}). (3) Reward: every "
        "completion is scored through the verifiers environment; cost = completions x "
        "reward_seconds_per_completion, which depends on the env (exact-match ~0.01s; a "
        "sandboxed code-exec / LLM-judge reward ~3s and can DOMINATE the step). "
        "seconds_per_step = rollout + reward + update, so GRPO is several times costlier "
        "per step than SFT.\n"
        "MoE: compute scales with ACTIVE params per token (e.g. a 35B-A3B model costs "
        "like 3B), not total. QLoRA (4-bit) shrinks VRAM so a cheaper card fits, but "
        "compute FLOPs still scale with the full param count."
    )


def _formula_layer() -> str:
    return (
        "## Cold-start overhead and the exact cost formula\n"
        f"Cold start (billed): worker boot {A.WORKER_BOOT_S:.0f}s + deps "
        f"{A.DEPS_INSTALL_S:.0f}s + model download (total_params x 2 GB / "
        f"{A.DOWNLOAD_RATE_GBPS} GB/s) + {A.VLLM_INIT_S:.0f}s vLLM init for GRPO.\n"
        f"Total wall clock is capped at {A.DEFAULT_WALL_CAP_S / 3600:.0f}h.\n"
        "Putting it together:\n"
        "  setup_s   = boot + deps + download(+ vllm_init if grpo)\n"
        "  train_s   = steps x seconds_per_step  (clamped so setup+train <= cap)\n"
        "  cost_usd  = (setup_s + train_s) / 3600 x gpu_hourly_usd\n"
        "Use the cheapest fitting validated GPU's $/hr."
    )


# Ordered framework layers appended cumulatively for v2..v6.
_LAYERS = (
    _pricing_layer,
    _selection_layer,
    _timing_layer,
    _grpo_moe_layer,
    _formula_layer,
)


def prompt_for_version(version: int) -> str:
    """System prompt for version ``v`` in 1..NUM_VERSIONS (v1 = base, v6 = full)."""
    if not 1 <= version <= NUM_VERSIONS:
        raise ValueError(f"version must be in 1..{NUM_VERSIONS}, got {version}")
    parts = [BASE_SYSTEM, *(layer() for layer in _LAYERS[: version - 1])]
    return "\n\n".join(parts)


def render_run(config: RunConfig) -> str:
    """The user message describing the run to price (knob values, recipe-resolved)."""
    n = config.normalized()
    lines = [
        "Estimate the cost of this Flash run:",
        f"  model      : {n.model_id}",
        f"  method     : {n.method.upper()}",
        f"  steps      : {n.steps}",
        f"  seq_len    : {n.seq_len}",
        f"  lora_rank  : {n.lora_rank}",
        f"  thinking   : {n.thinking}",
    ]
    if n.is_grpo:
        lines.insert(5, f"  completion : {n.completion_len} tokens")
        lines.insert(6, f"  prompts/step x group : {n.batch_size} x {n.group_size}")
    else:
        lines.insert(5, f"  effective_batch : {n.batch_size}")
    if n.setup_repeats > 1:
        # Multi-seed: each seed is its own job that re-pays the cold start, so the bill
        # covers this many setups -- tell the model to price the repeated cold start.
        lines.append(f"  cold starts: {n.setup_repeats} (multi-seed; setup billed per seed)")
    if n.gpu:
        lines.append(f"  gpu (pinned): {n.gpu}")
    if n.environment:
        lines.append(f"  environment: {n.environment}")
    return "\n".join(lines)
