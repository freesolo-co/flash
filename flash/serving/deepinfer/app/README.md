# DeepSeek V4.1 Flash 1017 on Modal

This backend serves `deepseek-ai/DeepSeek-V4.1-Flash-1017` through the pinned
DeepInfer runtime. Four B200s form one EP=4 / CP=2 / DP=2 group. The existing
Flash CPU router handles API keys, organization attribution, exact token
reservations, OpenAI streaming, and durable usage settlement. Qwen/LoRA engines
retain their separate vLLM images and capacity.

## Capacity

| Setting | Value |
| --- | --- |
| GPU allocation | Four B200s in one container |
| Main | One warm container, maximum three (12 B200s total), zero buffer containers |
| Dev and other environments | Zero warm containers, maximum one, zero buffers |
| Admission per container | Maximum 128 requests; target concurrency 16 |
| Scale-up | One second of sustained demand above the target |
| Native scheduler | Two driver ranks, 64 sequences per rank, 16,384-token prefill chunks |
| Context | 1,048,576 combined input/output tokens, with 64 reserved for native execution |
| Output | Up to the remaining context budget; 350,000-token completions fit with shorter input |
| Placement | US compute, US East routing |
| Host allocation | 32 CPUs / 512 GiB RAM |
| Startup / request budget | 45 minutes / two hours |
| Failure bounds | Ten-minute native-worker watchdog; 60-second HTTP stream idle timeout |
| Idle scale-down | 600 seconds; main retains its minimum, dev scales to zero |

Production starts additional four-GPU groups before reaching the existing group's
hard admission limit, up to three groups. Dev and other environments remain capped
at one group and scale to zero. Admission above ready capacity may receive a
retryable capacity error. After scale-to-zero, an authenticated request triggers
startup and receives 503 until the server is ready. At the maximum pool size there
is no spare GPU group for a rolling replacement. With zero buffer containers, a
sudden burst can arrive before additional groups are ready; the scaling target
does not guarantee latency during cold scale-out.
The low target intentionally trades earlier GPU allocation for startup headroom;
it is not the group's maximum decoding capacity.
A failed worker retires its entire group;
ordinary request cancellation aborts only that sequence and releases its slot
after acknowledgment.

Large fresh suffixes coalesce across the two driver ranks for up to 420ms before
forcing an any-rank prefill attempt. Suffixes below 8,192 uncached tokens skip
that wait. A collective maximum over pending request ages makes every driver
choose the same scheduling vote. Startup exercises prefill shapes, representative
decode batches, and prefix capture/reuse before readiness.

Tokenization uses the checkpoint's official `encoding/encoding.py` and immutable
tokenizer revision. CPU admission and native generation share the renderer and
context validation. The one-item batch tokenizer call releases Python's lock;
offloading the single-sequence binding to a thread alone does not. Atomic database
admission still reads current ownership and RPM/TPM limits on every request.
Ambiguous admissions are never retried or treated as allowed.

## Cache and accounting

GPU prefix checkpoints are tenant scoped and require exact token IDs. Each driver
retains at most 64 entries and 2,097,152 tokens of allocated prefix state. Entries
expire after 600 seconds of inactivity or are evicted under memory pressure.
One-off prefixes skip checkpoint creation; a bounded hash-only observation table
admits them on reuse. Retention favors frequently reused prefixes relative to
their allocated size. CPU spill is disabled because its current snapshots do not
capture every context-parallel worker's state.

Workers advertise prefix digests to the in-container router, which balances
saved prefill against pending work. Workers recheck namespace and token equality
before reuse. This routing does not cross Modal containers. Cache warmth depends
on actual traffic; a 92% hit fraction is a workload assumption, not a service
setting or guaranteed discount.

The CPU prompt-token cache holds at most 128 entries, keyed by tenant, model
revision, messages and thinking mode. CPU admission and generation also reuse
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
and Qwen and does not change token prices or ledger idempotency. The backend's legacy
`gpu_seconds` field stores request generation wall time as an allocation weight;
it is not physical GPU-seconds and does not determine token charges.

Qualification results, failed comparisons and limitations are maintained in
[PR #1508](https://github.com/freesolo-co/flash/pull/1508). Benchmark scripts and
result files are intentionally not retained in this codebase. Throughput and
latency must be assessed together with input lengths, output lengths, arrival
rate, actual cache reuse and per-stream measurements.

## Deployment

```sh
modal deploy --env dev -m flash.serving.deepinfer.app.modal_app
modal deploy --env main -m flash.serving.deepinfer.app.modal_app
```

The `Deploy DeepInfer pools` workflow selects the matching Modal environment on
relevant pushes to `dev` and `main`. Only explicitly selected `main` gets a warm
minimum. The private `Engine` Modal Server requires proxy authentication and
exposes `/health`, `/internal/token-budget` and `/internal/generate`.

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
  `im-afZLHKewsK8Ldc8aZIVd5Q`.
- Weights volume: `freesolo-deepseek-1017-weights`.

The base image is immutable and local to this Modal workspace. Other workspaces
must build the pinned runtime and replace its identity. Image preparation checks
the Git revision and exact source hashes, applies the short-context indexer and
prefix-checkpoint bindings, and rebuilds the native extension. It installs local
Engram transport in spawned context-parallel workers and disables expandable CUDA
allocations for IPC. A missing RDMA bandwidth is accepted only after checking all
participants share one host and every resulting physical domain is NVLink-only.

The compiler-cache layer extends `im-01M5GXCE5NCMK4TBY608BEPZ1Y` with 1,280 files
from `/root/.dj/cache`, `/root/.tilelang/cache` and `/root/.triton/cache`.
The source archive SHA-256 is
`72608d25a9897cb402294b4874347a43f315c84e133cc6e602a2fe76735f0c9d`;
`/opt/deepinfer-compiled-kernel-manifest.json` records individual file hashes,
compiler/library versions and the source runtime identity. These are compiled
kernel artifacts, not GPU memory snapshots or cached prompts. Each container gets
its own writable cache overlay; there is no shared runtime cache volume.

Before shortening prefill warmup, the runtime verifies the trusted manifest,
every compiler artifact, native binary identity on both drivers, compiler/library
versions, native/runtime Python sources, and the model and execution geometry.
A qualified image exercises representative prefill shapes, all existing decode
batches, and prefix capture/reuse before readiness. Missing, modified or
incompatible artifacts retain the exhaustive 16-token-step prefill warmup.
`/health` reports `compiled_kernel_cache_verified`. Runtime, compiler or GPU
changes require a fresh qualification; a compiler cache miss may still compile.

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
