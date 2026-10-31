# Hosted GPU qualification, 2026-10-30

## Qwen 9B context follow-up, 2026-10-31

Qwen3.5-9B now supports 65,536 tokens. An original Tau evaluation had reached
33,410 input tokens and exceeded the previous 32,768-token limit. The hosted
engine, public catalog and customer serving profile now agree on the larger limit.
L40S placement, FP8 weights/KV cache, CUDA graphs, eight-sequence scheduling and
16 rank-128 LoRA slots remain unchanged.

Two isolated runs used the hosted image and engine with the real rank-32 Tau
adapter from `flash-1793406137-62169ab1/step-16`. Both base and adapted generation
returned the expected arithmetic answer with exactly 33,410 and 64,000 input
tokens. A 65,537-token input was rejected with the correct context-limit error.
The first cold boot took 291 seconds; the second, with compiler cache, took 90
seconds. Model and preallocated LoRA buffers occupied 24.94 GiB.

Eight concurrently submitted, distinct 64,000-token adapter prompts all returned
the expected answer without OOM. Their distinguishing prefix prevents shared-prefix
reuse; completion latency ranged from 8.36 to 65.07 seconds. The earlier
shared-prefix batch also passed. These are short-output compatibility and memory
checks, not sustained throughput measurements or a replay of the failed Tau case.
This follow-up loads one real adapter; it does not repeat the earlier all-slot,
maximum-rank adapter qualification. No production deployment was performed.

- [Cold-start and shared-prefix checks](https://modal.com/apps/freesolo/main/ap-mIlun7NT9ODdRd9nmmjNX4).
- [Distinct long-prompt concurrency checks](https://modal.com/apps/freesolo/main/ap-0NA2ta2jcfvyw0PYdwPP5T).

The remaining sections record the original October 30 qualification at 32,768 tokens.

## Original sizing

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
  expensive passing 80 GB option; H100 also passes the final generation, adapter and image checks.
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
maximum-rank attention LoRA; the 35B expert check additionally exercises all 256 experts
in one layer at rank 64, using the same fused parameter layout as Flash exports. These checks verify loading, dispatch, kernels and memory,
not fine-tuned quality or every possible adapter/vision workload. All-slot identity
loading reuses the same adapter weights under distinct IDs. The runtime allocates
its configured full-model LoRA buffers regardless of the synthetic adapter's keys.

This is a serving compatibility and memory qualification, not a sustained throughput
benchmark or a guarantee of unchanged latency. Review latency after deployment
before increasing traffic; the current admission limits remain unchanged.

## Results and latency limits

All five selected tiers pass the final checks with correct arithmetic outputs,
all configured adapter identities loaded, and correct red-image responses on the
three vision tiers. The fused-expert H200 run also passes. Runtime KV estimates at
32,768 tokens are 69.97 sequences for 0.8B/L4, 12.84 for 4B/L4, 20.03 for 9B/L40S,
17.93 for 27B/A100 80GB, and 33.95 for 35B/H200, all above the configured eight.
These are cache-capacity estimates, not measured throughput.

**27B trades latency for the lowest hourly rate.** In the final smoke runs, all eight
distinct 32k prompts completed in 109.15 seconds on A100 80GB versus 20.9 seconds on
H100; with adapters, 127.54 versus 31.5 seconds. These are single-run batch completion
times, including incidental kernel warmup, with short outputs. They do not establish
steady-state tokens/second, but the difference is substantial. A100 is selected for
the requested lowest hourly cost among passing GPUs; busy long-context traffic may
cost more per request than on H100 despite the lower hourly rate. H100 remains a
verified alternative if latency or active-work cost becomes the priority.

Offline validation in the locked server/serving environment passed: 12,074 tests,
162 skips (normal plus isolated wall-clock test runs). Ruff lint/format, file and
function size, and source layout checks passed.

## Evidence runs

- [Initial candidate and H100 screening](https://modal.com/apps/freesolo/main/ap-Pz7F3fIde5bFXfUFdULbJ1).
- [27B L40S OOM and A100 80GB screening](https://modal.com/apps/freesolo/main/ap-7IAvnrhQI1d9spJFQEndlW).
- [Rejected 9B A100 alternative](https://modal.com/apps/freesolo/main/ap-IQulSlhj2Y5zcCJPmcmAqu).
- [Final 0.8B/L4, 9B/L40S and 35B/H200 checks](https://modal.com/apps/freesolo/main/ap-p7JClnWG52T7tvYy9tUqsd).
- [Final 4B/L4 checks](https://modal.com/apps/freesolo/main/ap-HmaNiXvgbDrngHfZz5t32v).
- [Final 27B/A100 80GB checks](https://modal.com/apps/freesolo/main/ap-1wmYDWMnxWMN5y1g90WFH1).
- [27B/H100 comparison](https://modal.com/apps/freesolo/main/ap-pwuREqhd8pWiE7HuxnKoZh).
- [35B/H200 fused-expert checks](https://modal.com/apps/freesolo/main/ap-vxBJ9GLnXe5rmPuPxZMeHT).

The first screening harness recorded completion as `ok`; its captured text was also
reviewed for correctness. The 9B/A100 run's `ok=true` is therefore **not a pass**.
Final outputs are checked against arithmetic and image-color answers; the harness
was tightened to assert those answers directly as well. The first synthetic expert
probe used per-expert keys instead of Flash's fused format and was rejected during
loading; the linked fused-format rerun is the qualifying result.
