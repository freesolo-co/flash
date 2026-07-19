# Architecture-aware VRAM sizing validation

## Invariant

For curated catalog models, Flash sizes vLLM cache memory from the model's real attention and recurrent-state geometry, and sizes LoRA memory from every PEFT `all-linear` target shape. Existing measured VRAM floors, headroom, and the live-calibrated 35B resident boundary remain in force.

## Validation matrix

Compare three values for each configuration:

1. the architecture-aware requirement from this branch
2. the requirement from `origin/dev@8fdfc858`, including measured floors
3. the observed smallest fitting card or largest observed OOM card

The matrix covers:

- dense GQA: MiniCPM5-1B
- hybrid GatedDeltaNet and VL: Qwen3.5 0.8B, 2B, 4B, and 9B
- high-rank LoRA: Qwen3.5-2B at rank 128
- MoE: Qwen3.6-35B-A3B, counting the full loaded model's adapted modules rather than active experts per token
- short and long context, fp8 and bf16 KV, and group-size pressure

For every row, record the required VRAM, selected GPU tier, whether the run initialized, and whether training completed without OOM. Known-good rows must remain admitted on the same card or a cheaper validated card. Known-OOM rows must remain routed above the failing card or rejected.

## End-to-end confirmation

After CPU matrix review, run one representative Qwen3.5-4B GRPO job on the GPU tier selected by the new estimator. Use the same model, rank, context, group size, data, and worker path as the corresponding observed boundary. Confirm vLLM initialization and at least one optimizer update complete without OOM. No paid GPU run is part of this branch task.

## Metrics and direction

Primary metric: correct and safe tier or fit decisions across the matrix.

Safety gate: zero known-OOM configurations become accepted, and the representative end-to-end run has zero OOMs.

Improvement direction: fewer false rejections and less over-routing than `origin/dev@8fdfc858`, while preserving measured floors and all known fit and OOM boundaries.
