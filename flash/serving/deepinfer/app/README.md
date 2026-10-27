# DeepSeek V4.1 Flash 1017 on Modal

This backend serves `deepseek-ai/DeepSeek-V4.1-Flash-1017` through the pinned
DeepInfer runtime. Four B200s form one EP=4 / CP=2 / DP=2 group. The existing
Flash CPU router handles API keys, organization attribution, observed-usage
RPM/TPM admission, OpenAI streaming, and durable usage settlement. Qwen/LoRA engines
retain their separate vLLM images and capacity.

## Capacity

| Setting | Value |
| --- | --- |
| GPU allocation | Four B200s on one physical machine per container (`B200:4`) |
| Main | Two warm containers, maximum four (16 B200s total), zero buffer containers |
| Main rolling deployment | Temporarily allow five containers (20 B200s); restore four after warm handoff |
| Dev and other environments | Zero warm containers, maximum one, zero buffers |
| Admission per container | Maximum 128 requests; main planning capacity 100 concurrent streams |
| Main scale-up | Pool-wide reserve of 64 streams: third group at 136 active requests, fourth at 236 |
| Sampling | Replica heartbeats and a single CPU controller every five seconds |
| Native scheduler | Two driver ranks, 64 sequences per rank, 16,384-token prefill chunks |
| Context | 1,048,576 combined input/output tokens, with 64 reserved for native execution |
| Output | Up to the remaining context budget; 350,000-token completions fit with shorter input |
| Placement | US compute, US East routing |
| Host allocation | 32 CPUs / 512 GiB RAM |
| Startup / request budget | 45 minutes / two hours |
| Failure bounds | Ten-minute native-worker watchdog; 60-second HTTP stream idle timeout |
| Idle scale-down | 600 seconds; main retains its minimum, dev scales to zero |

Production starts additional four-GPU groups before reaching the existing group's
hard admission limit, up to four groups. Dev and other environments remain capped
at one group and scale to zero. Admission above ready capacity may receive a
retryable capacity error. After scale-to-zero, an authenticated request triggers
startup and receives 503 until the server is ready. Main rolling deployments reserve
one temporary replacement slot even at the normal maximum. With zero buffer containers, a
sudden burst can arrive before additional groups are ready; the scaling target
does not guarantee latency during cold scale-out.
Main requests `min(4, max(2, floor((active_requests + 64) / 100) + 1))` containers.
The reserve is shared across the pool, not multiplied by the number of containers.
At 136 requests it requests the third four-GPU group; at 236 it requests the fourth.
At the four-group cap it cannot preserve that reserve once demand reaches 336.
The 100-stream capacity is an operator planning
assumption, not a benchmark-derived latency guarantee or the hard admission limit.
Aggregate output tokens/sec divided by a target per-stream rate gives only a
rough concurrency budget for the same saturated workload; prompt length, cache
reuse, prefill and uneven routing also affect achievable concurrency and latency.

Each main container reports its complete generation HTTP lifetime, including
tokenization and streaming cleanup, to an environment-scoped Modal Dict from a
background thread. Health probes are excluded. A lightweight scheduled CPU
function reads the currently registered Server containers and their reports,
then changes only the GPU pool's minimum container count. Sampling adds roughly
five to ten seconds before requesting capacity, plus control-plane latency.
GPU allocation and model startup still take additional time. A reserve of 64
requests cannot guarantee enough time for an arbitrary burst or provider shortage.

The controller permits scale-down only after 600 seconds of continuously fresh,
lower demand. Missing/stale reports and controller interruptions reset that
quiet period; known demand can still increase capacity. Reports older than 20
seconds are stale. Departed containers do not contribute demand. Modal's native
target of 100 remains a fallback if the controller is unavailable and respects
the controller's minimum. Native scale-down also retains its 600-second window.
An operator's maximum of zero disables the pool and is never undone by the
controller. Dev has no controller or heartbeat publisher: it keeps target eight,
zero warm containers, and its single-container cap. The controller uses the
container-list API in the pinned Modal SDK; lookup failures fail the tick rather
than treating missing telemetry as idle capacity.

The public router sends requests to the private Modal Server URL. Modal selects
a container using its default routing; we do not send an affinity header or
provide a cross-container prefix directory. Each stream stays with its selected
container. Within that container, the two native driver ranks are selected by
pending work plus uncached prompt length, with active count as a tie-breaker.
Prefix-cache advertisements are tenant-scoped hints and workers verify exact
matches. Prefix caches are not shared between containers.
A failed worker retires its entire group;
ordinary request cancellation aborts only that sequence and releases its slot
after acknowledgment.

Large fresh suffixes use a 420ms grouping window before allowing an any-rank
prefill attempt. This is not a mandatory sleep: prefill can start earlier when
both driver ranks have eligible work. Suffixes below 8,192 uncached tokens allow
an any-rank attempt immediately, subject to decode protection and capacity.
While either rank has decoding work, the scheduler protects eight decode
iterations between prefill chunks. Those iterations are separate from the
concurrent-request autoscaling target and need not each emit exactly one token.
Collective decisions keep both drivers in the same scheduling mode. The window
starts at request admission, rather than restarting for each chunk, and is not
a TTFT bound. Startup exercises prefill shapes, representative decode batches,
and prefix capture/reuse before readiness.

Tokenization uses the checkpoint's official `encoding/encoding.py` and immutable
tokenizer revision. Native generation renders and validates its own input;
admission does not tokenize requests. The one-item batch tokenizer call releases Python's lock;
offloading the single-sequence binding to a thread alone does not. Atomic database
admission still reads current ownership and RPM/TPM limits on every request.
Ambiguous admissions are never retried or treated as allowed.

## Cache and accounting

GPU prefix checkpoints are tenant scoped and require exact token IDs. Each driver
retains at most 64 entries and 2,097,152 tokens of allocated prefix state. Entries
expire after 600 seconds of inactivity or are evicted under memory pressure.
One-off prefixes skip checkpoint creation. Each driver remembers up to 4,096
eligible tenant-scoped prefix hashes and their observation timestamps, with a
600-second lifetime. A repeated observation within that history makes a prefix
eligible for a checkpoint; subsequent requests can reuse it once ready and while
resident. Evicting an observation only forgets its reuse history and never rejects
a request. This table holds no prompt tokens or GPU state and is separate from
the 64-entry GPU cache and API rate limits. Retention favors frequently reused
prefixes relative to their allocated size. CPU spill is disabled because its
current snapshots do not capture every context-parallel worker's state.

Workers advertise prefix digests to the in-container router, which balances
saved prefill against pending work. Workers recheck namespace and token equality
before reuse. This routing does not cross Modal containers. Cache warmth depends
on actual traffic; a 92% hit fraction is a workload assumption, not a service
setting or guaranteed discount.

The CPU prompt-token cache holds at most 128 entries, keyed by tenant, model
revision, messages and thinking mode. Generation also reuses
tokenized conversation prefixes at verified added-token boundaries, preserving
the exact full-encoding IDs. That tenant-scoped cache holds at most 128 entries
and 64 MiB of packed token IDs per tokenizer, with a 600-second idle lifetime.
The shared Qwen text tokenizer uses the same bounded prefix cache; unsupported
tokenizer configurations retain full encoding. Neither CPU cache counts as GPU-cached input.
Only native reused-prefix counts receive cached-input pricing. Terminal usage
includes actual completion, reasoning and EOS tokens; token delivery flushes the
first token immediately and then batches up to 16 tokens or 50ms of engine
progress, always flushing terminal tokens.

The backend catalog captures exact prices in each usage record: $0.15 input,
$0.60 output and $0.003 cached input per million tokens, price version
`2026-10-23.deepseek-1017.1`. Settlement uses actual counts and the captured price,
with durable outbox delivery and idempotent ledger writes. The shared CPU router
settles up to eight claimed records concurrently; each starts immediately within
its lease. Shutdown cancels deliveries before releasing their leases, and a
permanent delivery error refuses new chargeable work. This applies to DeepSeek
and Qwen and does not change token prices or ledger idempotency. A backend response
explicitly reporting a lost settlement lease leaves the database-owned event
for reclamation instead of permanently quarantining it; identity conflicts still
fail closed. The backend's legacy
`gpu_seconds` field stores request generation wall time as an allocation weight;
it is not physical GPU-seconds and does not determine token charges.

Qualification results, failed comparisons and limitations are maintained in
[PR #1508](https://github.com/freesolo-co/flash/pull/1508). Benchmark scripts and
result files are intentionally not retained in this codebase. Throughput and
latency must be assessed together with input lengths, output lengths, arrival
rate, actual cache reuse and per-stream measurements.

## Deployment

```sh
modal deploy --strategy rolling --env dev -m flash.serving.deepinfer.app.modal_app

export MODAL_ENVIRONMENT=main
export DEEPINFER_ROLLOUT_ID="$(python -c 'import uuid; print(uuid.uuid4().hex)')"
python -m flash.serving.deepinfer.app.rollout --begin
modal deploy --strategy rolling --env main -m flash.serving.deepinfer.app.modal_app
python -m flash.serving.deepinfer.app.rollout
```

The `Deploy DeepInfer pools` workflow selects the matching Modal environment on
relevant pushes to `dev` and `main`. Only explicitly selected `main` gets a warm
minimum. The private `Engine` Modal Server requires proxy authentication and
exposes `/health` and `/internal/generate`.

Main explicitly uses Modal's rolling strategy. Uvicorn does not open the serving
port until every rank loads and the complete prefill, decode, and cache warmup
finishes. A failed startup never opens that port. Existing replicas continue
serving while replacements start. Each rollout has a unique identity in its
health responses and load reports. The deployment waits for at least two warmed
replacements and for every registered replica to report the current rollout.
The scheduled controller then removes the temporary fifth slot, even if CI has
disconnected. A pre-deploy marker prevents the previous controller from removing
that slot early. A timeout fails verification without stopping the serving pool;
the fifth slot stays available until the replacement handoff succeeds.
Serialize manual deployments with CI; do not overlap rollout commands.

The availability objective is 99.99% (259.2 seconds unavailable per 30 days), with
zero failed requests caused by planned redeployment as the rollout acceptance
criterion. Two warm replicas remove the single-replica dependency, but do not
establish independent provider failure domains or prove this SLO. Measure valid
public generation requests end to end, including complete streaming responses;
track deployment windows separately and include backend errors, broken streams,
and timeouts. Health checks alone are not availability evidence. GPU allocation,
the shared routing region, and upstream platform failures remain dependencies.

Modal Servers allow at most 3,600 seconds of in-flight connection draining, which
this app already uses. The generation deadline is 7,200 seconds. Consequently a
stream with more than one hour remaining when its replica retires can still be
interrupted: native rolling deployment cannot guarantee preservation of every
two-hour stream. That stronger guarantee requires independently deployed versions
with application-controlled routing and retirement after all old requests finish.
See [Modal deployment strategies](https://modal.com/docs/guide/managing-deployments)
and [Server lifecycle limits](https://modal.com/docs/guide/servers).

The hosted router discovers `Engine` in the same Modal environment. Its
`DEEPINFER_PROXY_KEY` and `DEEPINFER_PROXY_SECRET` must both be configured as
runtime secrets. Connection credentials are not included in the image or health
response. The separately deployed Freesolo backend must recognize the exact model
and price version in its billing catalog. Qualification through the hosted router
must cover authorization, exact limits, cached pricing, streaming, cancellation
and durable settlement as well as native GPU performance.

## Pinned runtime

- Model/encoder: `1bae0e1088993b9b82de08f59de5be5a34e42170`.
- DeepInfer: `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`.
- Base compiled CUDA 13.2 / Python 3.12 image with B200 kernel artifacts:
  `im-22oaWkg4YF8JC0J9nM5RjE`.
- Weights volume: `freesolo-deepseek-1017-weights`.

The base image is immutable and local to this Modal workspace. Other workspaces
must build the pinned runtime and replace its identity. The prepared image already
contains the verified native extension, short-context indexer repair,
prefix-checkpoint bindings, transport patch and pinned HTTP dependencies.
Deployment copies current application source onto that image without reapplying
patches or recompiling the extension. It installs local Engram transport in spawned
context-parallel workers and disables expandable CUDA allocations for IPC. Modal
guarantees that the four GPUs requested with `B200:4`
are on one physical machine ([GPU placement](https://modal.com/docs/guide/gpu#specifying-gpu-count)).
All native driver and context-parallel processes stay within that container;
autoscaling adds independent four-GPU replicas rather than extending a collective
across machines. `native/transport.py` therefore omits redundant hostname and
physical-domain discovery. It supplies the same positive placeholder when RDMA
bandwidth is absent, preserves detected RDMA bandwidth, and still requires positive
NVLink bandwidth. This patch applies to every image; it does not add a new runtime
speedup to the already-local benchmark configuration.

The finalized image was rebuilt from the previous compiler seed
`im-afZLHKewsK8Ldc8aZIVd5Q`, with current source paths and the native patches baked
in once. It preserves all 1,280 verified compiler files from `/root/.dj/cache`,
`/root/.tilelang/cache` and `/root/.triton/cache`, and the identical native binary.
The original compiler archive SHA-256 remains
`72608d25a9897cb402294b4874347a43f315c84e133cc6e602a2fe76735f0c9d`;
`/opt/deepinfer-compiled-kernel-manifest.json` records individual file hashes,
compiler/library versions and the source runtime identity. These are compiled
kernel artifacts, not GPU memory snapshots or cached prompts. Each container gets
its own writable cache overlay; there is no shared runtime cache volume.

Before shortening prefill warmup, the runtime verifies the trusted manifest,
every compiler artifact, native binary identity on both drivers, compiler/library
versions, native/runtime Python sources, and the model and execution geometry.
The eight-turn scheduler in `runtime/prefill_scheduler.py` was qualified with
exhaustive warmup, answer/cache checks and a matched four-GPU replay. The finalized
manifest records current source filenames and exact hashes directly; there are
no source-hash overrides or filename aliases. Its canonical SHA-256 is
`150ee7657553951cdc9d29979c40a68aaf727bfd68210b845006c7d2d5812856`.
Old source filenames are absent from the rebuilt seed. Native binary, compiler
artifacts and execution geometry are unchanged. Future runtime-source changes
must rebuild and qualify the saved image to retain shorter warmup; simply editing
the expected manifest hash is not qualification.
A qualified image exercises representative prefill shapes, all existing decode
batches, and prefix capture/reuse before readiness. Missing, modified or
incompatible artifacts retain the exhaustive 16-token-step prefill warmup.
`/health` reports `compiled_kernel_cache_verified`. Runtime, compiler or GPU
changes require a fresh qualification; a compiler cache miss may still compile.

The finalized image passed a fresh four-B200 startup check on 2026-10-26:
both drivers verified the compiler cache, readiness took 140 seconds from server
launch, and all eight generation checks passed. Repeated-prefix requests reused
2,570 of 2,581 prompt tokens; the same prompt under another tenant reused zero.
This was startup and correctness qualification, not a new throughput benchmark.
The optional DeepGEMM MLOPart localization helper remained unavailable, as in the
previous qualification logs; native generation and cache checks completed.
Temporary build and GPU apps were stopped after validation.

Native FP4 expert weights, released scales and FP8 Engram values remain intact.
Fast kernels use native multi-token prediction with variable-length drafts; this
configuration does not promise byte-identical outputs to batch-invariant kernels.
Each driver reports source revision, native binary hash, GPU model, execution mode
and library versions in `/health`. Both driver ranks must initialize successfully.

## API scope and code layout

The API supports text-only system/user/assistant messages, one completion,
temperature, top-p, seed, maximum tokens, and
`chat_template_kwargs.enable_thinking`. Tool calls, images, raw prompts,
structured outputs, logprobs, penalties and custom stop sequences are rejected.
The combined context limit includes generated output; a million-token prompt
cannot also request 350K output tokens.

`app/` contains HTTP transport and Modal deployment; `contract/` owns identity,
validation and encoding; `runtime/` owns driver processes, scheduling and caches;
`native/` owns source-checked compatibility patches and local Engram transport.
GPU dependencies are imported only after the worker selects its CUDA device.
