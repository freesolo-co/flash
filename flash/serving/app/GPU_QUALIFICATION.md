# Hosted GPU qualification, 2026-10-30

This change sizes the five hosted Qwen engines with their existing checkpoints,
precision, 32,768-token context, eight-sequence scheduler, CUDA graphs, vision
support, and maximum LoRA rank/slot allocations. Training hardware and customer-owned
`flash serve` profiles are separate and unchanged. DeepSeek retains four B200s and
its current 128-input capacity at the owner's request; this is not a claim that
four GPUs are its minimum for lower concurrency.

| Hosted model | Previous GPU | Selected GPU | Model + preallocated LoRA load | GPU dollars/hour, before → after |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B, BF16 | L40S | L4 | 2.11 GiB | 1.95 → 0.80 |
| Qwen3.5-4B, BF16 | H100 | L4 | 9.70 GiB | 3.95 → 0.80 |
| Qwen3.5-9B, FP8 | H100 | L40S | 24.94 GiB | 3.95 → 1.95 |
| Qwen3.8-27B, FP8 | B200 | A100 80GB | 45.53 GiB | 6.25 → 2.50 |
| Qwen3.6-35B-A3B, BF16 | B200 | H200 | 108.46 GiB | 6.25 → 4.54 |
| DeepSeek V4.1 Flash 1017 | 4 × B200 | 4 × B200 | Unchanged | 25.00 → 25.00 |

Prices are the workspace's Modal GPU rates captured on 2026-10-30, excluding CPU,
RAM, storage, regional premiums and network. They describe equal billed GPU time,
not equal throughput or cost per token. See [Modal pricing](https://modal.com/pricing).
Autoscaling, including the 2,100-second Qwen idle window and active spare replica,
is unchanged, so idle-time costs still exist.

## Smaller alternatives

- **0.8B / 4B:** L4 is the lowest-cost available 24 GB tier that passes. T4's 16 GB
  tier lacks native BF16 support required by this runtime; changing precision is
  outside this qualification. A10 has the same nominal 24 GB and costs more.
- **9B:** the 24.94 GiB model/LoRA allocation alone exceeds a 24 GB L4/A10 before
  KV cache and graphs. L40S passes. An A100-40GB request was upgraded by Modal to
  an actual A100-SXM4-80GB (79.27 GiB visible); that run returned garbled arithmetic
  output and is rejected, despite completing without an exception. It does not
  establish a working 40 GB configuration. L40S is also cheaper than A100 40GB.
- **27B:** L40S failed with CUDA OOM while allocating LoRA buffers (44.35 GiB in
  use, 62.75 MiB free, failed 10 MiB allocation). Smaller 24/40 GB cards cannot
  contain the observed 45.53 GiB load plus KV and graphs. A100 80GB is the least
  expensive passing 80 GB option; H100 also passes the initial generation probe.
- **35B:** the observed 108.46 GiB model/LoRA load alone exceeds 80 GB and 96 GB
  options. H200 fits with six rank-64 slots; reducing slots or changing precision
  would change the supported serving shape. Smaller cards were excluded by this
  measured lower bound rather than charged for a predictable OOM run.

These are practical floors for the current single-GPU Qwen serving configuration,
not proofs across other quantizations, tensor-parallel layouts, or future kernels.

## Live checks

The isolated Modal runs use the hosted image and `_LoraEngineImpl._load`, persistent
weight/compiler cache, actual device identification, and unchanged engine overrides.
They submit eight arithmetic requests, eight distinct 32,000-token prompts, load
all configured LoRA identities, then submit eight distinct 32,000-token adapter
requests. Captured text is checked for the expected numeric answer, not merely a
nonempty response. The vision tiers additionally identify a red image with LoRA
active. No production deployment is performed by this qualification.

Small models use a real rank-64 SFT adapter from the pinned
`DavidBShan/flash-pr1520-validation-20261026` dataset revision
`d8cd13a007101c2bd5d39ca5b9c53097eb02650b`. Larger models use synthetic zero-valued,
maximum-rank attention LoRA; the 35B expert check additionally exercises all experts
in one layer at rank 64. These checks verify loading, dispatch, kernels and memory,
not fine-tuned quality or every possible adapter/vision workload. All-slot identity
loading reuses the same adapter weights under distinct IDs. The runtime allocates
its configured full-model LoRA buffers regardless of the synthetic adapter's keys.

This is a serving compatibility and memory qualification, not a sustained throughput
benchmark or a guarantee of unchanged latency. Review latency after deployment
before increasing traffic; the current admission limits remain unchanged.
