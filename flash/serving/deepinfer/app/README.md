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
| Main | One warm container, maximum one, zero buffer containers |
| Dev and other environments | Zero warm containers, maximum one, zero buffers |
| Admission | Maximum 128 requests; target concurrency 96 |
| Native scheduler | Two driver ranks, 64 sequences per rank, 16,384-token prefill chunks |
| Context | 1,048,576 combined input/output tokens, with 64 reserved for native execution |
| Output | Up to the remaining context budget; 350,000-token completions fit with shorter input |
| Placement | US compute, US East routing |
| Host allocation | 32 CPUs / 512 GiB RAM |
| Startup / request budget | 45 minutes / two hours |
| Failure bounds | Ten-minute native-worker watchdog; 60-second HTTP stream idle timeout |
| Idle scale-down | 600 seconds; main retains its minimum, dev scales to zero |

Each environment is capped at one four-GPU group. Admission above ready capacity
may receive a retryable capacity error. A warm minimum does not make a new
container's startup instantaneous. A failed worker retires its entire group;
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
retains at most 16 entries and 2,097,152 tokens of allocated prefix state. Entries
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
revision, messages and thinking mode. Its hits do not count as GPU-cached input.
Only native reused-prefix counts receive cached-input pricing. Terminal usage
includes actual completion, reasoning and EOS tokens; token delivery flushes the
first token immediately and then batches up to 16 tokens or 50ms of engine
progress, always flushing terminal tokens.

The backend catalog captures exact prices in each usage record: $0.15 input,
$0.60 output and $0.003 cached input per million tokens, price version
`2026-10-23.deepseek-1017.1`. Settlement uses actual counts and the captured price,
with durable outbox delivery and idempotent ledger writes. The backend's legacy
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
- Base compiled CUDA 13.2 / Python 3.12 image: `im-01M5GXCE5NCMK4TBY608BEPZ1Y`.
- Weights volume: `freesolo-deepseek-1017-weights`.

The base image is immutable and local to this Modal workspace. Other workspaces
must build the pinned runtime and replace its identity. Image preparation checks
the Git revision and exact source hashes, applies the short-context indexer and
prefix-checkpoint bindings, and rebuilds the native extension. It installs local
Engram transport in spawned context-parallel workers and disables expandable CUDA
allocations for IPC. A missing RDMA bandwidth is accepted only after checking all
participants share one host and every resulting physical domain is NVLink-only.

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
