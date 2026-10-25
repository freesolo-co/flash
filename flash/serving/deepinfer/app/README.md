# DeepSeek V4.1 Flash 1017 on Modal

This backend serves `deepseek-ai/DeepSeek-V4.1-Flash-1017` through native
DeepInfer, with two B200s cooperating inside each replica. The existing Flash
CPU router retains Freesolo API-key authorization, organization attribution,
OpenAI response formatting, and durable usage settlement. Qwen/LoRA engines
keep their separate vLLM images and capacity.

## Code layout

The parent `deepinfer/` package groups code by responsibility:

| Package | Responsibility |
| --- | --- |
| `app/` | Private HTTP endpoint, hosted router client, event rendering, and Modal launcher |
| `contract/` | Pinned configuration, validated chat requests, and shared errors |
| `runtime/` | Async request bridge, GPU worker processes, prefix cache, and loaded-runtime checks |
| `native/` | Prefix-cache native bindings and Engram GPU transport |

The hosted CPU router imports `app.client` and `contract` modules. The private
HTTP server starts `runtime.bridge.Runtime`, which launches `runtime.worker`
processes. Workers load native GPU modules after selecting their CUDA device;
package initializers do not load GPU dependencies. Image preparation applies
`native.patches`, which checks the scheduler and sequence sources and adds the
prefix-cache bindings. The native FP4 expert sources are left untouched.

## Capacity and price

| Setting | Value |
| --- | --- |
| GPU allocation | `B200:2` per replica, EP=2 / DP=2 |
| Main replica pool | 1–6; two B200s minimum, one two-B200 buffer replica; at most 12 B200s total |
| Dev and other environments | 0–4 replicas, no buffer; at most 8 B200s; scale to zero when idle |
| Admission | 16 requests per replica; scale-out target 12 |
| Native scheduler | 8 sequences per rank, 4,096-token prefill chunks; any-rank prefill check every 8 decode steps |
| Expert execution | Native packed FP4 weights with FP8 activations and the released scales |
| Token delivery | First output immediately; then 16 tokens or 50 ms of engine progress |
| Idle scale-down | 600 seconds for excess replicas; main retains one replica, dev retains none |
| Host allocation | 16 CPUs / 512 GiB RAM per replica |
| Initial context limit | 32,768 tokens including output and 64 reserved tokens |
| Startup / generation budgets | 45 minutes / 10 minutes |
| Input / output / cached input | $0.15 / $0.60 / $0.003 per million tokens |

Modal measures active HTTP inputs and scales whole two-GPU replicas. Saturated
replicas have a bounded native admission queue; excess traffic queues in Modal
or receives a retryable capacity error. Each GPU group exits if one rank dies or
stalls. Closing a request aborts its native sequence and waits for acknowledgment
before releasing the slot; it does not terminate the other requests in that group.

Only Modal's explicitly selected `main` environment keeps one **replica** warm:
its two B200s remain provisioned even with no traffic. `dev`, qualification
environments, and unspecified environments have no warm minimum. The 600-second
idle window applies to capacity above the environment's floor. Main also requests
one spare two-B200 replica while the pool is active. The six-replica main cap
includes both active and buffer capacity; dev has no spare buffer. Modal's [concurrency target](https://modal.com/docs/guide/concurrent-inputs)
allows each replica to accept up to 16 requests while additional replicas start;
it is not a guarantee that cold capacity will be ready for a sudden burst.

Prices are captured as exact per-token decimals with version
`2026-10-23.deepseek-1017.1`. Completion usage includes native reasoning and EOS
tokens. GPU prefix-cache hits report the actual retained token count with
`cached_tokens_reported=True`; misses report zero. CPU prompt-token-cache hits do
not count as cached GPU input and receive no cached-input billing discount.

The GPU cache keeps at most four prefixes and 32,768 tokens of retained KV allocation per
rank. Entries expire after 600 seconds of inactivity and are evicted according to the retention policy
or under KV-memory pressure. The router derives each cache namespace from verified
organization attribution. Prefix matching checks exact token IDs; continuations
use native copy-on-write state. A stable conversation prefix is preferred, with
full-prompt reuse for single-message requests. Traffic and cache warmth determine
the reuse ratio; 92% is not a guaranteed service setting.

Prompt tokenization reuses the same bounded cache helper as the existing hosted
and packaged engines, with 128 entries per DeepInfer replica and keys that include
model revision, tenant, messages, and thinking mode. Misses run off the event loop.
Delivery batching always flushes terminal tokens; the time limit is checked at
engine-step boundaries and does not bound a stalled GPU collective.

Cache affinity and retained checkpoints can leave unequal prefill queues across
ranks. The native any-rank prefill interval is eight decode steps instead of the
upstream default of 100. Prompts shorter than 130 tokens skip prefix preparation.

## Cache policy

Live prefix-aware routing is always enabled across the two DP ranks within a
replica. Workers advertise exact namespace/prefix digests, expiring after two
seconds. Placement balances queued prefill/decode work against saved prefill.
Workers verify token equality. This does not route between Modal replicas.
Qwen engines have one GPU and one DP rank each, so this rank-placement policy has
no additional target inside those engines; Modal still places their replicas.

The fixed retention policy scores entries by decaying
reuse count times reusable prefix length divided by the full allocated prompt
size. Frequency has a 60-second half-life; TTL expiry is unchanged.
The same policy applies at entry/token
capacity and under GPU block pressure.

A fixed 1 GiB per rank bounds raw host KV/state tensors. GPU capacity evictions
spill complete native blocks and SWA/recurrent state into a process-local CPU LRU.
Only exact tenant/prefix matches restore them. Promotion drops GPU victims without
another spill to keep the host transfer within its budget. Expired entries are
discarded. Allocation refusal falls back to prefill. Restored exact prefixes are
reported as cached input; tokenization-cache hits are still not discounted.

The fixed retention policy scores entries by decaying
reuse count times reusable prefix length divided by the full allocated prompt
size. Frequency has a 60-second half-life; TTL expiry is unchanged.
The same policy applies at entry/token
capacity and under GPU block pressure.

## Deployment and qualification

### Autoscaling qualification (2026-10-25)

The autoscaling campaign completed 6,808 private HTTP generation requests with
zero errors, including cold/JIT warmup, concurrency 4–28, and live scale-out to
two replicas. Each response requested 256 output tokens; every fourth request
used a longer unique prompt. Representative measurements:

| Target / concurrent requests | Requests | Output tokens/s | p95 first-token latency | Serving replicas |
| --- | --- | --- | --- | --- |
| 12 / 10 | 360 | 764 | 0.956 s | 1 |
| 8 / 10 | 2,304 | 817 | 0.542 s | 2 |
| 12 / 28 | 3,559 | 1,260 | 8.115 s | 2 |

Keep the target at 12 and the admission limit at 16: target 8 improved tail latency
but added a second active replica for approximately 7% more throughput in this
workload. Newly scaled replicas first delivered tokens after 329–380 seconds;
the 600-second idle window avoids immediately discarding that startup investment.
Main's buffer provides additional headroom while replicas start.

These synthetic runs exclude the public router and accounting path. Comparison
windows differ in length and hardware placement, and retain startup/JIT effects;
they establish a practical baseline, not a universal optimum. Tuning scale-out
was capped at two replicas. The final six-replica main cap is configuration-tested,
not a six-replica saturation result. See [PR #1496](https://github.com/freesolo-co/flash/pull/1496)
for the final environment and idle checks.

### Native runtime qualification

The two-B200 native FP4 configuration above completed 271 generation requests,
including 24 quality cases (23 correct), a 30,303-token input, and concurrency up
to 16, with zero request errors. It used 512 GiB of host RAM for mapped Engram
storage and loading.

A separate capacity campaign measured 15,261 aggregate output tokens/s over ten
one-minute load windows and 168.7 requests/s for 32-token responses. Those runs
used 256 sequences per rank and 90% GPU memory utilization. The million-token
and 350K-output endurance tests used two sequences per rank and 92% utilization.
Neither expanded profile is the default above. The follow-up throughput campaign
had zero errors in 75,790 measured requests; the initial campaign had three HTTP
read errors, and an auxiliary long-context quality batch exceeded admission and
received HTTP 429. These are private localhost engine measurements, excluding
public router and billing qualification.

### Historical four-GPU comparisons

The native bridge passed a **four-B200 smoke and performance run**; see the
[measured results and limits in PR #1484](https://github.com/freesolo-co/flash/pull/1484).
The earlier four-GPU FP8 bridge passed 780 measured requests with identical complete
outputs across prefill and delivery settings. At 24 concurrent requests and 128
output tokens each, increasing prefill from 512 to 4,096 improved the 5,068-input-token
case from 948 to 1,607 output tokens/s and reduced median first-token latency from
1.199 to 0.577 seconds. The 26-input-token case remained near 2,680 output tokens/s.
These are short in-container HTTP comparisons, excluding startup and router/network
overhead. Batched delivery reduced 131 internal events to about 20 per response.
GPU checks also covered cached/uncached output parity, tenant isolation, eviction,
and cancellation during prefill and decoding while another request completed.
Peak sampled physical GPU memory was 88.93%. All validation GPUs were released.

A subsequent same-four-B200 FP8 comparison isolated the cold-request scheduling
fix. With 24 streams, 5,068 input tokens and 256 output tokens per request, cold
aggregate throughput increased from 624 to 1,300 output tokens/s and p95 TTFT fell
from 6.735 to 2.158 seconds. Median per-stream decode speed fell from 101.4 to 81.7
tokens/s as prefills interrupted decoding sooner. Short cold requests improved
from 1,817 to 2,514 aggregate tokens/s, with p95 TTFT falling from 3.083 to 0.229
seconds. All 792 measured requests and 48 mixed-traffic requests succeeded, with
identical complete outputs between versions for each tested workload. Native
traces showed at most eight consecutive decode steps with waiting prefills,
versus 100 before. The mixed test reduced new-request p95 TTFT from 10.300 to
2.605 seconds; established streams slowed approximately 6%.

This fix is not a universal throughput improvement: the cached long-prompt
24-stream comparison included a 2.24-second native-step stall in its first repeat.
Keeping all three repeats gives 2,512 to 1,924 aggregate tokens/s and 0.346 to
2.378 seconds p95 TTFT. Its cause remains unisolated; the following two repeats
reached 2,372 and 2,459 tokens/s. These localhost measurements exclude model
startup and public-router/network overhead, and retain all first-use effects.
Further cached-tail and sustained mixed-traffic qualification remains necessary.

The earlier FP4 configuration peaked at approximately 1,597 aggregate output
tokens/s on short prompts; that is historical evidence, not a measurement of the
new FP8/cache/batching configuration. Separate warm benchmarks exceeded 12,000
output tokens/s on both long-context and short-chat workloads at approximately
92% input-token reuse. Those experiments used retained prefix states, 128
sequences per rank, speculative decoding, and FP8 execution of losslessly
expanded FP4 expert values. The deployable bridge uses the smaller admission/context configuration in the
capacity table above and no speculative decoding. Its new bounded prefix cache
and native hit counters do not by themselves establish the experimental throughput.
The larger scheduler and speculative path still require service qualification. A 92% hit rate also depends on the
request traffic and cache warmth.

### Hosted deployment

This is not complete customer rollout qualification. CPU tests cover the HTTP/native
bridge, router authorization, pricing and settlement, thinking/non-thinking
responses, native errors, and cancellation. Keep customer traffic disabled until
the remaining service and accounting checks below are completed.

Deploy the private backend in the intended Modal environment:

```sh
modal deploy --env dev -m flash.serving.deepinfer.app.modal_app
# Production: keep one two-B200 replica warm.
modal deploy --env main -m flash.serving.deepinfer.app.modal_app
```

The `Deploy DeepInfer pools` workflow deploys this private app on relevant pushes
to `main` and `dev`, explicitly selecting the matching Modal environment. It
does not configure the hosted router credentials.

To measure the private endpoint with environment-scoped Proxy Tokens in
`MODAL_KEY` and `MODAL_SECRET`, run:

```sh
uv run python scripts/benchmark_deepinfer_autoscaling.py --env dev \
  --concurrency 10 --seconds 600 --output /tmp/deepinfer-load.json
```

The benchmark records first-token latency, completed output throughput, errors,
replica IDs, and five-second Modal pool samples. It sends a mixture of short and
long unique prompts with 256-token responses, without changing deployment
settings. Run comparisons on a quiet pool; retain cold starts and errors in the
results, and observe idle scale-down without sending HTTP health probes.

The GPU endpoint requires [Modal Proxy Tokens](https://modal.com/docs/guide/webhook-proxy-auth),
so unauthenticated requests are rejected before provisioning GPUs. The GPU backend exposes only `/health` and `/internal/generate`; OpenAI routes,
stream formatting, usage responses, and model listing use the existing Flash
hosted router. Run chat qualification through that router too, so probes exercise
the same authorization, response, and accounting path as customer requests.

After qualification, supply these connection credentials when deploying the
existing serving router, following its development wiring instructions:

- `DEEPINFER_PROXY_KEY` and `DEEPINFER_PROXY_SECRET`: an environment-scoped Modal
  Proxy Token pair, **not** ordinary Modal API credentials.

The router resolves the fixed `freesolo-deepseek-serving` / `serve` endpoint through
Modal in its own environment. Only the two authentication secrets are configured
externally; partial credentials fail closed. Without them, the native
model is absent from the router's advertised base models. No vLLM class is
created for DeepSeek, and LoRA registration is unsupported.

The bridge follows Modal's same-origin 303 result redirects during long cold
starts without submitting a second generation. It rejects cross-origin redirects
before forwarding proxy credentials. Direct qualification clients must also
follow result redirects (for example, `curl -L`). See
[Modal request timeouts](https://modal.com/docs/guide/webhook-timeouts).

**Control-plane dependency:** the separately deployed Freesolo backend must also
recognize this exact model and the same prices in `src/billing/token_pricing.py`.
Its `/api/serving/authorize` uses that catalog to authorize base-model requests.
This Flash PR alone does not change that repository or enable customer access.
Keep the router connection unset until that catalog and settlement path are ready.

Qualification must exercise cold boot and coherent output against the official
encoder, native FP8/FP4/Engram parity, thinking and streaming, cancellation during
prefill and decoding alongside another request, rank failure, load above the
12-input target, the six-replica main / four-replica dev caps, and idle scale-down to one replica in
`main` and zero in `dev`.
Verify durable accounting through the real backend at the configured rates.

## Pinned runtime

- [Model and encoder](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017):
  `1bae0e1088993b9b82de08f59de5be5a34e42170`.
- [DeepInfer](https://github.com/deepseek-ai/DeepInfer):
  `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`.
- Compiled CUDA 13.2 / Python 3.12 image: `im-01M5GXCE5NCMK4TBY608BEPZ1Y`,
  a workspace-local immutable image from the four-B200 benchmark. It contains
  compiled DeepInfer under `/opt/DeepInfer` and Python under `/opt/deepinfer-venv`.
  Other workspaces must build the pinned upstream runtime and replace this image
  identity; the image is not a portable public registry artifact.

Image preparation checks the upstream Git revision and exact source hashes before
applying the short-context indexer repair and a native operation for retaining a
completed prefix checkpoint. It rebuilds the native extension. Expert loading uses
the released FP4 path without expanding weights to FP8. The historical precision
benchmark restores its own pinned serving commit, including the old FP8 helpers. The local Engram
transport substitutes mapped shared host RAM/CUDA IPC for RDMA, preserving native
FP8 embedding values and packed scales. Cached prefixes come from real requests;
no benchmark prompts or response cache are substituted. Native model weights are downloaded at the pinned revision
into the persistent `freesolo-deepseek-1017-weights` volume.

The inherited build also selects `sm_100a` in FlashMLA and permits CUDA
cross-compilation in the DeepSelect/FlashMLA platform probes. The runtime's
`PATH` and `VIRTUAL_ENV` match the earlier successful four-B200 campaign.
Each initialized GPU worker checks its engine source and native extension paths
against the pinned source tree and reports the native binary SHA-256, commit,
GPU model, native FP4 execution mode, zero converted expert tensors, and PyTorch/CUDA versions in the private `/health` response's
`loaded_ranks` field. Both ranks must initialize before health succeeds.
These loaded-worker records are distinct from inspecting the image on CPU.

The initial API supports text-only system/user/assistant messages, one completion,
temperature, top-p, seed, max tokens, and `chat_template_kwargs.enable_thinking`.
It uses the checkpoint's `encoding/encoding.py`, because the model does not ship
a Jinja chat template. Tool calls, images, raw prompts, structured outputs,
logprobs, penalties, and custom stop sequences are rejected explicitly. The
model's advertised million-token context is not enabled in this initial capacity
configuration.
