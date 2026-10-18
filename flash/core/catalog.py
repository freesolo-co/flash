"""Curated model catalog for one-consumer-GPU LoRA jobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import ceil
from typing import Any

ALGORITHMS = ("sft", "grpo", "opd")

# algorithms whose training step samples on-policy student completions, unlike fixed-dataset sft.
# import ``samples_on_policy`` instead of hand-rolling algorithm tuples at each call site.
_ON_POLICY_ALGORITHMS = frozenset({"grpo", "opd"})
_IMAGE_TRAINING_MODELS = frozenset(
    {
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.8-27B",
        "Qwen/Qwen3.6-35B-A3B",
    }
)


def supports_image_training(model: str | ModelInfo | None) -> bool:
    """Return whether a curated model supports image-bearing SFT, GRPO, and OPD."""
    model_id = model.id if isinstance(model, ModelInfo) else model
    return bool(model_id and model_id in _IMAGE_TRAINING_MODELS)


def samples_on_policy(algorithm: str) -> bool:
    """True when a training step samples on-policy student completions.

    Tolerant of the ``rl`` phase-name alias for grpo (``JobSpec.phase``) so callers that pass a
    phase rather than an algorithm resolve identically."""
    algo = (algorithm or "").strip().lower()
    if algo == "rl":  # phase-name alias for grpo
        algo = "grpo"
    return algo in _ON_POLICY_ALGORITHMS


def optimizer_batch_key(algorithm: str) -> str:
    """The ``[train]`` key holding the optimizer batch for this algorithm.

    The two names are different quantities and the schema rejects the wrong one, so a reader that
    guesses finds nothing and silently falls back to the recipe default. Sizing, ranking and the
    quote must agree on the name or they price and allocate the same job differently.

    Unknown and empty algorithms resolve to the sft name, matching how the sizing paths treat
    anything that is not recognisably rl.
    """
    algo = (algorithm or "").strip().lower()
    return "prompts_per_step" if algo in ("grpo", "rl", "opd") else "batch_size"


def normalize_algorithm(value: str) -> str:
    """Canonical (lowercased, validated) algorithm name."""
    if not value:
        value = "sft"
    elif not isinstance(value, str):
        # A truthy non-string (e.g. a JSON number/bool/array) would AttributeError on .lower(), which
        # escapes the callers' ValueError/ConfigError guards -> uncaught 500. Raise ValueError instead.
        raise ValueError(f"algorithm must be a string, got {type(value).__name__}")
    value = value.lower()
    if value not in ALGORITHMS:
        raise ValueError(f"unsupported algorithm: {value}; known: {', '.join(ALGORITHMS)}")
    return value


DEFAULT_GPU = "RTX 5090"

# Over-estimating is memory-safe (larger VRAM estimate, smaller cap); fallback = largest catalog vocab.
_DEFAULT_VOCAB_SIZE = 248_320


@dataclass(frozen=True)
class ServingCapacity:
    gpu: str
    max_loras: int
    max_cpu_loras: int
    max_lora_rank: int
    max_model_len: int
    serve_model_id: str = ""
    max_num_seqs: int = 0
    max_num_batched_tokens: int = 0
    tensor_parallel_size: int = 0
    gpu_memory_utilization: float = 0.0
    image_limit: int = 0


@dataclass(frozen=True)
class ModelInfo:
    id: str
    display_name: str
    params: str
    algos: tuple[str, ...]
    min_vram_gb: int
    # numeric total parameters in billions, read DIRECTLY by the cost and VRAM equations rather
    # than parsed from the ``params`` display string. sizes the FULL checkpoint for vram, disk, and
    # download. REQUIRED: ``test_every_catalog_entry_sets_params_b`` asserts every entry sets it
    # > 0, so a new entry can never silently fall back to a parsed string again.
    params_b: float
    # module prefix containing every language layer on the loaded conditional-generation model.
    # required by text-only lora targeting; kept out of the public catalog rows below.
    lora_language_prefix: str = ""
    # runner-managed immutable training revision for the one model that requires an exact hub pin.
    # this is internal preparation policy, not a public catalog field.
    managed_revision: str = ""
    quant: str = "bf16"
    recommended_gpu: str = DEFAULT_GPU
    # 0 => GRPO uses min_vram_gb like SFT; set when colocated vLLM rollout needs a bigger card.
    grpo_min_vram_gb: int = 0
    # 0 => non-grpo sizing uses the param-based estimate; set when a model must not down-route to the cheapest card.
    sft_min_vram_gb: int = 0
    # resident-only guard: vllm wake/reload hangs after allocator fragmentation. reject configs that
    # miss the resident peak and pin free_cache_engine=false in the verl launcher.
    sleep_unsupported: bool = False
    notes: str = ""
    # 0 = platform default (64 GB) suffices. Runner raises gpu.disk_gb to at least this.
    min_disk_gb: int = 0
    # Deployment capacity of the external freesolo multi-LoRA serving app. This is separate from
    # Flash's training GPU recommendation above; serving uses Modal/vLLM and sizes hot LoRA buffers
    # by max_loras x max_lora_rank at engine init.
    serving: ServingCapacity | None = None
    # "none" / "hybrid" (Qwen3-style) / "always" (can't disable). Every entry is curated, so the
    # capability is always known -- a new entry must state it rather than leave it to be guessed.
    thinking: str = "none"
    vocab_size: int = _DEFAULT_VOCAB_SIZE
    # Parameters ACTIVE per token in billions — only meaningful for an MoE, where a token routes
    # through a small subset of experts. The cost estimator's per-token FLOPs/step-time term reads
    # this (a token exercises only the active params), while VRAM/disk/download keep using the total
    # ``params_b``. 0.0 (the dense default) means "same as params_b" — every token hits every param.
    active_params_b: float = 0.0
    # decoder geometry for engine.vram.sft_gc_off_peak_gb. 0/0 means unknown and keeps gradient
    # checkpointing on if the runtime HF config also lacks usable values.
    num_layers: int = 0
    hidden_size: int = 0
    # vllm cache geometry. zero values mean the catalog has no architecture-aware sizing data.
    num_attention_layers: int = 0
    num_linear_attention_layers: int = 0
    num_key_value_heads: int = 0
    # QUERY heads, from the checkpoint's own config. NOT derivable as hidden_size // head_dim: these
    # models decouple head_dim from that ratio (3.5-9b is hidden 4096 / head_dim 256 and has 16
    # heads). vllm tensor parallelism requires this to divide the rented card count -- grpo
    # and opd hand that count to the rollout engine as tensor_model_parallel_size -- so a derived
    # value silently caps runs on a number that is not the constraint. no longer a sequence-parallel
    # constraint: all three algorithms pin ulysses off and shard by data.
    num_attention_heads: int = 0
    head_dim: int = 0
    linear_num_key_heads: int = 0
    linear_num_value_heads: int = 0
    linear_key_head_dim: int = 0
    linear_value_head_dim: int = 0
    linear_conv_kernel_dim: int = 0
    # fp8 attention-token equivalent of one recurrent-state page, rounded to vllm's 16-token block.
    mamba_block_size: int = 0
    # grouped (in_features, out_features, count) for peft all-linear on the full loaded model.
    lora_target_shapes: tuple[tuple[int, int, int], ...] = ()
    # fused routed experts stacked on the lora rank axis for target_parameters. zero means no
    # catalog-backed stacked allowance; ordinary and unknown target parameters carry exactly r.
    lora_expert_count: int = 0

    @property
    def is_moe(self) -> bool:
        """Return whether each token routes through only a subset of experts.

        GRPO uses this to select reentrant checkpointing; MoE routing breaks the non-reentrant
        metadata-equality assertion on recompute.
        """
        return 0.0 < self.active_params_b < self.params_b

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if not data["mamba_block_size"]:
            del data["mamba_block_size"]
        for key in (
            "num_attention_layers",
            "num_linear_attention_layers",
            "num_key_value_heads",
            "num_attention_heads",
            "head_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
            "lora_language_prefix",
            "managed_revision",
            "lora_target_shapes",
            "lora_expert_count",
        ):
            data.pop(key, None)
        serving = data["serving"]
        if serving is None:
            del data["serving"]
        else:
            for key in (
                "serve_model_id",
                "max_num_seqs",
                "max_num_batched_tokens",
                "tensor_parallel_size",
                "gpu_memory_utilization",
                "image_limit",
            ):
                if serving.get(key) in ("", 0, 0.0, None):
                    serving.pop(key, None)
        return data


DEFAULT_MODEL = "Qwen/Qwen3.5-9B"

# the checkpoint each public catalog model loads for customer-owned serving. hosted activation is
# a separate allowlist in flash.serving.src.engine.model_config.
SERVING_MODEL_REPOS: dict[str, str] = {
    "Qwen/Qwen3.5-9B": "Freesolo-Co/Qwen3.5-9B-FP8",
    "Qwen/Qwen3.8-27B": "Qwen/Qwen3.8-27B-FP8",
    "Qwen/Qwen3.6-35B-A3B": "Qwen/Qwen3.6-35B-A3B",
}

MODELS: dict[str, ModelInfo] = {
    "Qwen/Qwen3.5-9B": ModelInfo(
        id="Qwen/Qwen3.5-9B",
        display_name="Qwen3.5 9B",
        params="9.7B",
        params_b=9.7,
        lora_language_prefix="model.language_model",
        vocab_size=248_320,
        num_layers=32,
        hidden_size=4096,
        num_attention_layers=8,
        num_linear_attention_layers=24,
        num_key_value_heads=4,
        num_attention_heads=16,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        lora_target_shapes=(
            (1152, 1152, 27),
            (1152, 3456, 27),
            (1152, 4304, 27),
            (4096, 32, 48),
            (4096, 1024, 16),
            (4096, 4096, 56),
            (4096, 8192, 32),
            (4096, 12288, 64),
            (4304, 1152, 27),
            (4608, 4096, 1),
            (4608, 4608, 1),
            (12288, 4096, 32),
        ),
        algos=ALGORITHMS,
        min_vram_gb=48,
        min_disk_gb=120,
        # NOT QLoRA: a peft bnb merge during GRPO rollout diverges the rollout's precision from the
        # trainer's, and the importance ratio those two log-probs form collapses to 0.
        grpo_min_vram_gb=80,
        quant="bf16",
        recommended_gpu="A100 PCIe",
        serving=ServingCapacity(
            gpu="B200",
            serve_model_id=SERVING_MODEL_REPOS["Qwen/Qwen3.5-9B"],
            max_loras=16,
            max_cpu_loras=16,
            max_lora_rank=128,
            max_model_len=32768,
            max_num_seqs=8,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            image_limit=4,
        ),
        thinking="hybrid",
        notes="bf16 LoRA. ~19 GB of weights; SFT fits a 48 GB card, while colocated GRPO "
        "(two bf16 copies + KV + the 248k-vocab fp32 logits) needs an 80 GB-class card "
        "(grpo_min_vram_gb floor).",
    ),
    "Qwen/Qwen3.8-27B": ModelInfo(
        id="Qwen/Qwen3.8-27B",
        display_name="Qwen3.8 27B",
        params="27.781427952B dense (multimodal VL, hybrid GDN)",
        params_b=27.781427952,
        lora_language_prefix="model.language_model",
        managed_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        num_layers=64,
        hidden_size=5120,
        vocab_size=248_320,
        num_attention_layers=16,
        num_linear_attention_layers=48,
        num_key_value_heads=4,
        num_attention_heads=24,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        lora_target_shapes=(
            (1152, 1152, 27),
            (1152, 3456, 27),
            (1152, 4304, 27),
            (4304, 1152, 27),
            (4608, 4608, 1),
            (4608, 5120, 1),
            (5120, 48, 96),
            (5120, 1024, 32),
            (5120, 6144, 48),
            (5120, 10240, 48),
            (5120, 12288, 16),
            (5120, 17408, 128),
            (6144, 5120, 64),
            (17408, 5120, 64),
        ),
        algos=("sft", "grpo", "opd"),
        min_vram_gb=80,
        grpo_min_vram_gb=142,  # colocated GRPO (two ~54GB copies) needs B200; triggers resident-peak sizing ~150GB
        sft_min_vram_gb=80,
        quant="bf16",
        recommended_gpu="A100 PCIe",
        min_disk_gb=160,
        serving=ServingCapacity(
            gpu="B200",
            serve_model_id=SERVING_MODEL_REPOS["Qwen/Qwen3.8-27B"],
            max_loras=16,
            max_cpu_loras=16,
            max_lora_rank=64,
            max_model_len=32768,
            max_num_seqs=8,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            image_limit=4,
        ),
        thinking="hybrid",
        notes="Dense 27.781427952B multimodal VL checkpoint with image-capable bf16 LoRA training. "
        "SFT fits the 80GB A100 (~55.6GB weights); colocated GRPO needs the B200 (trainer + vLLM "
        "rollout = two copies). Hosted serving runs the official FP8 checkpoint on the B200.",
    ),
    "Qwen/Qwen3.6-35B-A3B": ModelInfo(
        id="Qwen/Qwen3.6-35B-A3B",
        display_name="Qwen3.6 35B-A3B (MoE)",
        params="35B total / ~3B active (MoE)",
        # 35.0 not 35.95: the marketing figure tips the SFT equation over the B200 budget (see test_sft_equation_covers_honest_peak_across_seq_boundary).
        params_b=35.0,
        lora_language_prefix="model.language_model",
        active_params_b=3.0,
        # Geometry for the SFT GC-off activation estimate (config.json text_config): 40 decoder
        # layers x 2048 hidden (hybrid GatedDeltaNet + full-attention, 256 experts / 8 active).
        num_layers=40,
        hidden_size=2048,
        num_attention_layers=10,
        num_linear_attention_layers=30,
        num_key_value_heads=2,
        num_attention_heads=16,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        # vllm 0.19.1 derives the 1,097,728-byte gdn state page as 1072 fp8 attention
        # tokens after the 16-token backend alignment.
        mamba_block_size=1072,
        # peft targets ordinary linears plus both fused routed-expert tensors on every layer.
        # each routed tensor has 256 expert slices, all using the adapter's uniform serving rank.
        lora_target_shapes=(
            (512, 2048, 40),
            (512, 2048, 10_240),
            (1152, 1152, 27),
            (1152, 3456, 27),
            (1152, 4304, 27),
            (2048, 1, 40),
            (2048, 32, 60),
            (2048, 512, 100),
            (2048, 1024, 10_240),
            (2048, 4096, 30),
            (2048, 8192, 40),
            (4096, 2048, 40),
            (4304, 1152, 27),
            (4608, 2048, 1),
            (4608, 4608, 1),
        ),
        lora_expert_count=256,
        vocab_size=248_320,
        algos=ALGORITHMS,
        min_vram_gb=141,
        # Floor to 100 GB so SFT lands on H200, not the thin-margin consumer Blackwell or 80 GB H100.
        sft_min_vram_gb=100,
        # Floor also engages grpo_seq_escalation_gb: long (>16k) rollouts are rejected at parse time instead of OOMing in vLLM.
        grpo_min_vram_gb=180,
        # vLLM sleep mode HANGS the 35B colocate rollout (wake/reload stalls — live-confirmed, every
        # attempt). So GRPO is RESIDENT-ONLY: model_required_vram_gb sizes on the resident peak and a
        # config too long to fit resident is REJECTED at parse time (not routed to the hanging sleep).
        sleep_unsupported=True,
        quant="bf16",
        recommended_gpu="H200",
        serving=ServingCapacity(
            gpu="B200",
            serve_model_id=SERVING_MODEL_REPOS["Qwen/Qwen3.6-35B-A3B"],
            # bf16 is the validated full-expert lora path. six hot rank-64 adapters plus 32k fit
            # with cuda graphs; the b200 measured 49.38 GiB of kv (3,678,952 tokens) for this shape.
            max_loras=6,
            max_cpu_loras=6,
            max_lora_rank=64,
            max_model_len=32768,
            max_num_seqs=8,
            max_num_batched_tokens=4096,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            image_limit=4,
        ),
        thinking="hybrid",
        min_disk_gb=200,
        notes="MoE (35B total / ~3B active), bf16 LoRA. SFT runs on the 141 GB H200 (the ~70 GB "
        "weights dominate; active-3B compute keeps activations/KV tiny, so context is ~unbounded by "
        "VRAM); colocated GRPO needs the 180 GB B200 (trainer + vLLM rollout = two 70 GB copies).",
    ),
}


def list_models() -> list[ModelInfo]:
    return sorted(MODELS.values(), key=lambda m: (m.min_vram_gb, m.id))


def get_model(model_id: str) -> ModelInfo:
    try:
        return MODELS[model_id]
    except KeyError as exc:
        allowed = ", ".join(MODELS)
        raise ValueError(
            f"unsupported model {model_id!r}; choose one of: {allowed} — or, to train another "
            f"model, fork Flash and add a ModelInfo entry for it to flash/core/catalog.py "
            f"(see SELF_HOSTING.md)"
        ) from exc


def _model_info_for_serving(model: str | ModelInfo | None) -> ModelInfo | None:
    if isinstance(model, ModelInfo):
        return model
    if isinstance(model, str) and model.strip():
        return MODELS.get(model.strip())
    return None


def serving_lora_rank_cap(model: str | ModelInfo | None) -> int | None:
    """Return the model-specific serving LoRA rank cap.

    A model with no serving entry returns None.
    """
    info = _model_info_for_serving(model)
    if info is None or info.serving is None:
        return None
    return int(info.serving.max_lora_rank)


def lora_expert_count(model: str | ModelInfo | None) -> int | None:
    """Return the exact fused-expert stack size for LoRA target parameters, when cataloged."""
    info = _model_info_for_serving(model)
    if info is None or info.lora_expert_count <= 0:
        return None
    return int(info.lora_expert_count)


def serving_context_cap(model: str | ModelInfo | None) -> int | None:
    """Return the model's served ``max_model_len``, or None without a local serving entry.

    A LoRA trained longer than it is served learns positions inference never uses, so the control
    plane caps training context to this (see
    ``flash.adapters.lora_rank.preflight_train_context_within_serving``). Mirrors ``serving_lora_rank_cap``:
    no serving entry returns None rather than a global fallback.
    """
    info = _model_info_for_serving(model)
    if info is None or info.serving is None:
        return None
    return int(info.serving.max_model_len)


def vocab_size_for(model_id: str) -> int:
    """Curated vocab_size for a model, or the safe (largest-catalog) default for an id the
    catalog does not list -- only a stale caller can produce one, since submit rejects them."""
    info = MODELS.get(model_id)
    return info.vocab_size if info is not None else _DEFAULT_VOCAB_SIZE


def resolve_vocab_size(model_id: str, revision: str = "") -> int:
    """vocab_size for a model, revision-aware when a commit is pinned (mirrors resolve_params_b).

    catalog vocab by default; for a pinned revision, the hf config vocab validated against the
    catalog (fail-closed, same as the vram path).
    """
    info = MODELS.get(model_id)
    if revision:
        from flash.engine.plan.vram import _validated_revision_geometry, fetch_hf_model_geometry

        if info is not None:
            _params_b, vocab = _validated_revision_geometry(model_id, revision, info)
            return int(vocab or info.vocab_size)
        _p, vocab, _h, _l, _heads = fetch_hf_model_geometry(model_id, revision, strict=True)
        if vocab:
            return int(vocab)
    return vocab_size_for(model_id)


def _with_merge_disk_floor(info: ModelInfo) -> ModelInfo:
    """Floor the container disk at what publishing a checkpoint actually holds at once.

    `min_disk_gb` sizes the RunPod container disk (`template.containerDiskInGb`), which is where the
    verl workdir and every export live. Publishing is concurrent with training, so at the moment the
    merger runs that one filesystem holds THREE full bf16 model copies, not one:

    * the checkpoint being published. verl saves `self.model.state_dict()`, the whole model rather
      than the lora delta, and the publisher hardlinks it into `_staging`, so verl's own
      `max_ckpt_to_keep=1` prune frees nothing while the merge still references it.
    * the next checkpoint, which training writes on its own thread while that merge runs.
    * the merger's output, a full model materialized beside its input.

    At `2 * params + 64` a 35B model was granted 200 GB and died with 24.98 GB free after both of
    its steps had trained -- the failure lands at publish, so it reads like a training fault and is
    not one. The base weight cache is NOT in this budget: it lives on the shared weight-cache volume.
    """
    return replace(info, min_disk_gb=max(info.min_disk_gb, ceil(info.params_b * 2) * 3 + 64))


def resolve_model(model_id: str, algorithm: str, model_revision: str = "") -> ModelInfo:
    """Resolve a curated model, validated for ``algorithm``; anything uncataloged is rejected.

    Resolution is card-independent: a curated entry states its own VRAM/disk requirements, so there
    is nothing to size against a GPU class or count here. (The uncataloged path used to synthesize a
    ModelInfo and fit-check it against the allocated shape, which is why this once took ``gpu`` and
    ``gpu_count``.) Whether the run actually fits the cards it is offered is the allocator's call.
    """
    algo = normalize_algorithm(algorithm)
    if model_id not in MODELS:
        return get_model(model_id)  # raises with the fork-a-catalog-entry instruction
    info = validate_model_for_algorithm(model_id, algo)
    if model_revision:
        from flash.engine.plan.vram import _validated_revision_geometry

        params_b, vocab_size = _validated_revision_geometry(model_id, model_revision, info)
        info = replace(
            info,
            params_b=params_b,
            params=f"{params_b:.1f}B",
            vocab_size=vocab_size,
        )
    return _with_merge_disk_floor(info)


def validate_model_for_algorithm(model_id: str, algorithm: str) -> ModelInfo:
    info = get_model(model_id)
    algo = normalize_algorithm(algorithm)
    # Each algorithm gates on its own capability entry (sft/grpo/opd), so a model must
    # explicitly opt into opd via its algos tuple — same contract as sft/grpo.
    if algo not in info.algos:
        allowed = ", ".join(info.algos)
        raise ValueError(f"{model_id} supports {allowed}, not {algo}")
    return info


def public_model_rows() -> list[dict[str, Any]]:
    return [_with_merge_disk_floor(model).to_dict() for model in list_models()]
