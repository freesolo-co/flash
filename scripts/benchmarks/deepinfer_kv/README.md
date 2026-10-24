# DeepInfer KV experiments

Seven opt-in experiments against the same pinned native DeepInfer configuration.
These scripts do not enable a serving feature or change customer traffic. The PRs
are benchmark prototypes, with promotion conditional on the recorded correctness
and performance results. They compare each candidate against its own baseline;
they do not compare DeepSeek output throughput against Flash's Qwen backend.

## Baseline and scope

Reviewed Flash main `3c751157` and DeepSeek serving branch
`b6a4019ae7a8c1cd4d6967ba5ae8e22cbd0e58ad` (PR #1484), with upstream DeepInfer
`9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`.
Flash main already enables vLLM prefix caching and FP8 KV. Native DeepInfer already
has packed FP4 main/index KV, FP8 sliding-window KV, compression, shared layers,
block allocation and copy-on-write forks. None of those existing mechanisms is
claimed as a new optimization here. Upstream README explicitly leaves a KV cache
system, disaggregated serving, and replica load balancing to the serving layer.
The reviewed Flash paths do not implement the seven policies/services below.
Evidence: [Flash prefix-cache defaults](https://github.com/freesolo-co/flash/blob/3c751157/flash/serving/src/store/settings.py),
[DeepInfer feature/integration list](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/README.md),
and the serving branch
[bridge scope](https://github.com/freesolo-co/flash/blob/b6a4019ae7a8c1cd4d6967ba5ae8e22cbd0e58ad/flash/serving/deepinfer/README.md).
This is a repository integration finding, not a claim that vLLM lacks optional
connectors or that no low-level DeepInfer primitives exist.

| Experiment | A | B | Scope limit |
| --- | --- | --- | --- |
| Routing | Least-active rank | Cache and queue-work score | Four ranks within one EP group; static verified cache directory, decode cost heuristic 32 prefill-token equivalents per output token |
| Tiering | Recompute after GPU eviction | Restore complete CPU snapshot | RAM tier only; no SSD policy or asynchronous overlap |
| Sharing | Cold destination | Source snapshot through HTTPS relay | Independent replicas; RAM relay survival probe, no disk durability or RDMA |
| Retention | LRU | Frequency, then recency | Equal-sized/equal-cost prefixes; two slots per rank |
| P/D split | Two colocated groups | One prefill group, one decode group | Equal eight GPUs; receiver replays four prompt tokens; sequential burst, no pipelined steady state |
| Non-prefix reuse | Recompute changed context | Recompute boundary, reuse later blocks | Approximate prototype, not full CacheBlend; changed-fact quality probe |
| Compression | Native FP4 snapshot | FP3 main-KV host encoding | Lossy host/storage codec, not SnapKV eviction; no live GPU-memory savings |

## Measurement

Each paired arm uses identical token IDs, temperature 0, seed 1234 and at most
128 output tokens. Contexts are 2K, 8K and 16K tokens of public runtime source with
four embedded retrieval facts and four questions per prefix. Three paired repeats
alternate A/B order. Routing includes uniform and skewed requests; retention has
hot/scan and uniform-scan traces. These are synthetic controlled tests, not a
customer trace or a production capacity qualification.

The service has four B200 ranks, EP=4, native speculation=3, batch invariance,
variable-length MTP enabled, YOCO, FP8 execution of expanded expert weights, CPU Engram, 8 GiB KV per rank, a 0.96 PyTorch allocator fraction,
4096-token prefill chunks and a 24-sequence/rank cap. This is an experimental
accelerated baseline, not the deployed bridge's default configuration. Both arms
use the same immutable model revision
`1bae0e1088993b9b82de08f59de5be5a34e42170` and runtime binary identity.

Durations cover native command execution; they exclude client polling, public
HTTP authorization, scheduling between commands and model cold start. Transfer
experiments include snapshot serialization, network transfer and restoration.
Report total request-path time alongside output tokens/s: a shorter or changed
answer is not automatically a speedup. Raw token outputs, exact-match flags and
retrieval scores remain in the evidence. Exact token equality is a strong check,
not a complete quality benchmark; a passed retrieval probe is not broad quality
parity. The upstream fully deterministic recipe disables variable-length MTP; this
accelerated baseline keeps it enabled. Fixed seeds and batch invariance therefore
do not guarantee exact outputs across all schedules. Cached versus uncached paths
may differ even before these modifications;
retained versus restored snapshots provide an additional control.

Three repeats provide an exploratory median and observed range, not a reliable
confidence interval. Compilation/startup outliers must remain visible in raw
qualification data; any later steady-state rerun must be labeled separately.

## Reproduction

Requires a Modal workspace with the pinned private image and weight volume from
PR #1484. `control.py` reserves worst-case sandbox costs before provisioning and
limits allocations to two four-B200 groups. The campaign budget is $500, with a
$420 GPU/host reservation cap. CPU relay/egress have the remaining reserve.
Run from this directory with Python 3.12 and `modal==1.6.2` installed:

```sh
export KV_AB_WORKDIR=/tmp/flash-deepinfer-kv-ab
python control.py launch primary2 10800
python stage.py primary2
# Wait for /results/ready.json on each replica before issuing commands.
python control.py exec primary2 python /opt/DeepInfer/driver.py
python control.py exec primary2 python /opt/DeepInfer/routing.py
# Companion PRs add their respective drivers. For the sharing/P-D PR stack:
# python control.py launch peer2 10800
# python relay_launch.py
# python stage.py primary2  # only when starting a fresh service, not a second copy
# python stage.py peer2
# python sharing.py "$KV_AB_WORKDIR"
# python disaggregation.py "$KV_AB_WORKDIR"
python control.py stop primary2
# python control.py stop peer2
# Terminate the relay ID recorded in $KV_AB_WORKDIR/relay.json as well.
```

Do not run two command drivers concurrently against the same replica. The native
adapter requires all ranks to participate in matching collectives. State is copied
only at quiescent barriers and accepted only with matching model, binary and cache
layout identities. Snapshots contain raw GPU cache/state slots and prefix metadata;
they are not complete scheduler checkpoints and omit speculative CPU draft state.
The P/D prototype therefore replays four prompt tokens on the receiver to initialize
its speculative state. There is no concurrent eviction, tenancy, cancellation or
failure-recovery qualification in these prototypes.

`runtime/` holds the exact earlier benchmark build inputs as text fixtures; they
are copied to `.py` in the sandbox before compilation. The additional binding
exposes a sequence's allocated block table for full-state snapshots. Model weights
and credentials are never checked in. The HTTPS relay is authenticated, RAM-only
and scoped to this campaign; source code and generated synthetic prompts are its
only data.

To inspect or regenerate a result summary, decompress its raw JSON and run:

```sh
gzip -dc results/routing/raw.json.gz > /tmp/routing-trials.json
python summarize.py routing /tmp/routing-trials.json
```

The fixed GPU cache pool remains allocated in both arms. CPU tiering recovers
logical cache blocks after eviction; these experiments do not demonstrate a
reduction in total model/device allocation. `results/common/*-sampled-memory.json`
reports sampled device memory, including model and cache, rather than exact peaks
or per-policy savings.
