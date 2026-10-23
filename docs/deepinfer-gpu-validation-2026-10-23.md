# DeepInfer four-B200 validation — 2026-10-23

The native serving bridge passed a real four-B200 smoke and performance run.
This is not a complete customer rollout qualification. The separate backend
catalog/accounting path and multi-replica load balancing still require validation.

## Runtime and setup

The previous Codex session `01a161c3-4b8a-7d63-8adb-97e677d237f4` and the local
`deepinfer-b200-benchmark/campaign-ratio54-20261023` artifacts were reviewed.
The successful setup used four B200s, 16 CPUs, 256 GiB host memory, the compiled
DeepInfer image, and single-node Engram transport. This validation reused the
existing checkpoint volume read-only after checking all 83 file metadata records
against the pinned model revision, including all 48 safetensors shards.

All four loaded workers reported DeepSeek's `deepinfer.engine.llm_engine.LLMEngine`,
commit `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`, CUDA 13.2, PyTorch 2.13.0+cu132,
and native binary SHA-256
`dea9f42a984ac3334d29e8842252a0b6f16986ae963716c3622d99be6b16c851`.
The checkpoint is `deepseek-ai/DeepSeek-V4.1-Flash-1017` at
`1bae0e1088993b9b82de08f59de5be5a34e42170`.

This run used the released FP4 routed experts and native mixed precision,
EP=4/DP=4, CPU-backed Engram tables, 512-token prefill chunks, eight sequences
per rank, a 32,768-token context budget, and no speculative decoding.
Adaptations are the short-context candidate-bound fix, local Engram transport,
and inherited B200/cross-compilation build settings; see the runtime evidence.

## Warm short-prompt measurements

Streaming HTTP was measured **inside the GPU sandbox**, excluding public-network,
Modal proxy, Flash router, authorization, and billing latency. Prompts had unique
random prefixes (47–56 input tokens, median 51) and requested 256 output tokens at temperature zero, seed 42.
Cache-hit counts are unavailable; no cache-hit rate is claimed.
The first trial at each concurrency is treated as warmup. Counts below are small
samples, not a sustained traffic or tail-latency qualification.

| Concurrent requests | Measured requests | Aggregate output tokens/s | Median TTFT | Median per-request decode tokens/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2 | 94.9 | 0.652 s | 124.7 |
| 4 | 4 | 355.4 | 0.372 s | 101.8 |
| 8 | 8 | 805.1 | 0.368 s | 117.6 |
| 16 | 32 | 1,460.8 | 0.405 s | 107.0 |
| 24 | 48 | 1,597.2 | 0.423 s | 75.9 |
| 32 | 64 | 1,357.5 | 0.524 s | 47.1 |

Aggregate throughput is emitted completion tokens divided by the whole batch
wall time, including prefill and draining. Per-request decode rate is
`(completion_tokens - 1) / (response_end - first_content_or_reasoning_token)`.
Native completion counters include EOS. These two rates have different denominators.

All 283 short-prompt requests across warmup and measured batches completed without
errors. At 32 concurrent requests the measured throughput fell below 24, supporting
the initial scale-out target of 24 for this workload; this is not an optimum for
all request distributions.

## Startup, context, and correctness

- GPU allocation in the successful sandbox took approximately 55 seconds; preceding
  function/sandbox requests in dev remained queued and were cancelled.
- All four workers became ready about 12.4 minutes after sandbox creation returned,
  including first-use native kernel compilation. This reused cached model weights;
  downloading 510 GB of weights into an empty volume is not included.
- First-use request shapes incurred additional compilation: one short request had
  18.2 s TTFT; the first thinking request had 33.5 s TTFT. The initial 8-, 16-,
  24-, and 32-request batches took approximately 27, 33, 38, and 24 seconds.
- Correct answers were returned for `17 × 23`, France's capital, and the average-speed
  reasoning problem. Generated Fibonacci code was coherent; its explanation was
  intentionally cut off by the 256-token output limit. This is a smoke check,
  not a model accuracy benchmark or full-model numerical parity proof.
- Six synthetic retrieval prompts returned the correct marker. Input lengths were
  approximately 1,842, 7,220, and 21,556 tokens. Second-run TTFT was respectively
  3.29 s, 12.14 s, and 33.21 s; these short answers measure prefill behavior.
- Client disconnects during decoding and a long-prompt request left concurrent
  arithmetic requests correct, with zero active requests afterward. Exact cancellation
  placement within a native kernel was not instrumented.
- A separate four-rank synthetic transport probe checked exact FP8 embedding bytes
  and packed scales with values varying by owner, row, and column/block. Invalid
  indices were zeroed, and CUDA-graph replay returned the correct values after
  changing indices. All four ranks passed. Copy
  `scripts/qualify_deepinfer_engram.py` into an existing four-GPU container and run
  it with `/opt/deepinfer-venv/bin/python`; the script does not allocate GPUs. This checks the mapped
  host-memory path used here; it does not qualify every table shape or the unused
  GPU-storage path.
- During the throughput run, peak sampled memory was 155,158 MiB / 151.52 GiB per GPU, 84.62% of the
  reported 183,359 MiB. Sampling was every five seconds and may miss brief peaks.

## Request-driven scaling and cleanup

A private Modal web function using the same serving entrypoint was capped at one
four-B200 replica, with zero minimum/spare containers and `scaledown_window=600`.
It reused the read-only checkpoint and a filesystem snapshot containing compiled
kernels (`im-01M5ND1YKKQCRSW1YHHMTXP3AV`). That warmed validation snapshot is not
the default image pinned by the PR.

The function started at zero, allocated one replica on an authenticated request,
reported all four native workers ready, and returned `391` for `17 × 23` through
the private endpoint. Health readiness took 354.47 seconds including scheduling
and startup; the subsequent buffered chat request took about 0.96 seconds in
Modal's request log. This is a separate endpoint smoke, not the throughput table.

After requests ceased, **automatic scaling returned the pool to zero**. The last
HTTP response completed at 22:24:58 UTC; zero runners were observed at 22:37:38 UTC,
759.99 seconds later, using 30-second polling. The transport probe used container
exec during the idle period. This experiment therefore establishes eventual
scale-to-zero, not a strict 600-second resource-release guarantee. The configured
idle window remained 600 seconds; actual release also involves platform scheduling
and container teardown. No manual stop preceded the observed zero.

The container inventory independently confirmed the GPU container was gone. All
seven temporary/development validation apps were then explicitly stopped, with
zero owned containers or active sandboxes. The temporary proxy token and two
scratch volumes were removed; the original checkpoint volume was unchanged.
Unrelated serving and training apps were untouched. Multi-replica scale-out and
the configured four-replica ceiling were not exercised on real GPUs.

## Historical comparison

The earlier campaign reported 11,976 aggregate output tokens/s at 95% prefix reuse
and 11,801 at 90%, with 1,627 successful requests per case. Those used roughly
88,560-token inputs, 1,494-token outputs, expanded FP8 target experts, speculative
decoding, and much larger concurrency. They do not measure this PR configuration.
Selected historical values are preserved separately to prevent conflating the runs.

## Evidence

- [Loaded GPU workers](validation/deepinfer-2026-10-23-loaded-ranks.json)
- [Image identity and exact inherited build changes](validation/deepinfer-2026-10-23-runtime.json)
- [Transport probe](validation/deepinfer-2026-10-23-engram.json)
- [Quality responses](validation/deepinfer-2026-10-23-quality.json)
- [Initial throughput trials](validation/deepinfer-2026-10-23-throughput.json)
- [Extended trials, longer prompts, and cancellation](validation/deepinfer-2026-10-23-extended.json)
- [Autoscaling observations](validation/deepinfer-2026-10-23-autoscale.json)
- [Final resource shutdown](validation/deepinfer-2026-10-23-shutdown.json)
- [Historical comparison only](validation/deepinfer-2026-10-23-prior-results.json)

Raw responses, telemetry, benchmark clients, and operational logs were retained
locally under `/tmp/flash-deepinfer-validation`. Committed metrics omit generated
long responses; no credentials are included.
