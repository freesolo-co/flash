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
| Main | One warm container, maximum four (16 B200s total), zero buffer containers |
| Main rolling deployment | Temporarily allow five containers (20 B200s); return to four only after old streams finish |
| Dev and other environments | Zero warm containers, maximum one, zero buffers |
| Admission per container | Maximum 128 requests; main planning capacity 100 concurrent streams |
| Main scale-up | Pool-wide reserve of 64 streams: second group at 36 active requests, third at 136, fourth at 236 |
| Sampling | Authenticated health checks and one CPU controller every five seconds |
| Native scheduler | Two driver ranks, 64 sequences per rank, 16,384-token prefill chunks |
| Context | 1,048,576 combined input/output tokens, with 64 reserved for native execution |
| Output | Up to the remaining context budget; 350,000-token completions fit with shorter input |
| Placement | US compute, US East routing |
| Host allocation | 32 CPUs / 512 GiB RAM |
| Startup / generation budget | 45 minutes / one hour |
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
Main requests `min(4, max(1, floor((active_requests + 64) / 100) + 1))` containers.
The reserve is shared across the pool, not multiplied by the number of containers.
At 36 requests it requests the second four-GPU group; at 136 the third; at 236 the fourth.
At the four-group cap it cannot preserve that reserve once demand reaches 336.
The 100-stream capacity is an operator planning
assumption, not a benchmark-derived latency guarantee or the hard admission limit.
Aggregate output tokens/sec divided by a target per-stream rate gives only a
rough concurrency budget for the same saturated workload; prompt length, cache
reuse, prefill and uneven routing also affect achievable concurrency and latency.

Production uses an immutable Modal App per GPU container, each fixed at one
container. A CPU controller owns the shared capacity ledger and ready endpoint
directory in `freesolo-deepseek-fleet-state`. It polls complete generation HTTP
counts, including tokenization and streaming cleanup. Health probes do not count.
Every warming, active, draining, and stopping replica consumes a reserved slot.
Provisioning reserves its slot before calling Modal; termination does not free the
slot until Modal confirms the container is gone. Normal operation permits four
slots, with one extra only while replacing a previous version. Existing replicas
stay up when replacement provisioning or warmup fails.

An established active route survives missing health probes for up to 20 seconds
after its last successful, non-draining probe. Readiness timestamps persist in the
ledger across controller restarts; failed probes never extend that grace period.
Explicit drain acknowledgments and draining/stopping phases exclude the replica
immediately. A replica that has never passed readiness is never routed. This grace
only preserves routing: stale telemetry cannot authorize scale-down or retirement
of a predecessor during rollout.

Scale-down requires 600 seconds of fresh lower demand. Missing health data and
controller interruptions reset that period. Scale-down fences admissions and
waits for active requests to reach zero, just like a rollout. Each GPU App has
native autoscaling disabled and a fixed minimum/maximum of one: Modal cannot
independently scale down a busy replica after its one-hour grace period. The CPU
controller is required for scale-out; a controller outage leaves the current
replicas serving but prevents new capacity. Dev retains native autoscaling with
zero warm containers, target eight, and a single-container cap.

The public router rotates its first choice across ready replica URLs. On a
connection failure before sending the request, or an explicit application
rejection carrying `X-DeepInfer-Retryable: not-started`, it prefers another ready
replica. With only one replica it retries that replica. Attempts are bounded to
three within a ten-second retry window, with backoff and the original overall
deadline. Authorization, generation identity, tenant cache namespace, and usage
accounting remain outside this retry loop. Generic 429/503 responses, ambiguous
read/write failures, and partially delivered streams are never replayed. This
avoids duplicate model work and billing; it cannot resume a failed generation.
Fallback is to another ready four-B200 container in this fleet, not another
provider. Requests are not pinned to a replica across turns; prefix caches are
not shared. Each accepted stream stays with its selected container.
Within that container, the two native driver ranks are selected by
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
python -m flash.serving.deepinfer.app.fleet_deploy
```

Production also requires `DEEPINFER_PROXY_KEY` and `DEEPINFER_PROXY_SECRET` for
the CPU controller's authenticated health and drain calls. These are injected as
Modal secrets, never baked into GPU images. The public router uses the same
credentials. Deploy the updated public router before enabling fleet routing.

The `Deploy DeepInfer pools` workflow uses the versioned fleet on `main` and the
native single-container pool on `dev`. Production builds the application image
once and stores its immutable image ID with the desired version. Future scale-out
uses that exact image. Never run `modal deploy` or `modal app rollover` on an
individual production replica: doing so reintroduces the native drain deadline.

The controller creates a separately named replacement, waits for every model rank
and all warmup to finish, then publishes its authenticated URL. It replaces old
replicas one at a time so a four-container pool retains serving capacity. It
removes the predecessor from routing, atomically fences new admissions, and keeps
its App deployed until the drain endpoint confirms zero complete HTTP lifetimes.
Only then does it stop the old App. Generation is capped at one hour, separately
from startup. Preparation and cleanup also count toward draining; no retirement
timer kills that work. Stale-directory requests
get an explicit safe rejection and can retry on a ready replacement. Restarted
replicas restore their drain gate before opening the serving port. Missing drain
telemetry never authorizes termination. Application and hardware failures can
still interrupt streams.

CI waits up to two hours for warmup and retirement and fails if the transition
is incomplete. It never kills old work to meet that deadline; the controller
continues after CI disconnects. A saturated multi-replica rollout can take longer
because each predecessor can retain hour-long generations. Serialize manual deployments
with CI. An atomic controller lock without a lease timer prevents old/new controller code
from mutating the ledger concurrently. If a hard kill strands the lock, verify
that no previous controller invocation or provisioning call remains running
before removing `controller_lock` from the fleet Dict; never delete the lock just
because it is old. A missing capacity ledger blocks provisioning; recover the
existing App inventory instead of treating missing state as an empty fleet. This
deliberately favors preserving work and capacity over
automatic recovery from an uncertain mutation.

### One-time legacy migration

The existing monolithic production pool predates admission fencing. A normal
fleet deploy refuses to overwrite it. Migration is staged so existing streams
remain on the unchanged legacy deployment throughout warmup and router cutover:

1. Finish any existing legacy rollout and verify its maximum is four. Deploy the
   updated public router, which still discovers the legacy URL while no fleet
   directory exists. Record the preceding router container IDs; wait for all of
   them to finish and exit. Check the deployed router release and complete real
   streaming requests through it. Disable or update every other direct legacy
   client, and prevent rollback to a router version that ignores fleet routing.
2. Run `python -m flash.serving.deepinfer.app.fleet_migration --reserve-legacy`.
   This reserves all four possible legacy slots, regardless of current load.
   Run `python -m flash.serving.deepinfer.app.fleet_deploy --stage-migration` with
   the production environment, credentials, and a unique rollout ID. It warms one
   replacement in the authorized fifth slot and publishes its route. Until then,
   the updated router continues using the unchanged legacy deployment.
3. Verify new public requests reach the replacement. The directory cutover and
   retirement of all old router containers must leave **no source of new legacy
   admissions**. Confirm old requests have finished; do not redeploy the legacy
   pool to add a drain endpoint. When ingress is disabled, run:

   ```sh
   python -m flash.serving.deepinfer.app.fleet_migration --retire-legacy --confirm-legacy-ingress-disabled
   ```

   This explicit operator assertion is necessary because the old code cannot
   enforce an admission fence. The command also requires fresh zero-request
   telemetry from every registered legacy replica and a ready fleet route. It
   marks the legacy pool for retirement; the controller stops it and keeps its
   four slots reserved until all its containers disappear. Missing or busy
   telemetry refuses retirement. An unconfirmed migration stays at one new
   replica plus the reserved legacy capacity and requires operator completion.
4. Verify the fleet ledger has only current active replicas, the legacy App has
   stopped, and real public streaming requests complete. Normal production
   deployments now use `fleet_deploy` without migration flags.

Do not claim the old system has retroactive drain protection: it retains its
original lifecycle behavior until migration finishes. First-fleet migration must
be completed before the production workflow can report a normal rollout success.

### Availability objective

The objective is 99.99% (259.2 seconds unavailable per 30 days), with zero failed
requests caused by planned redeployment as the acceptance criterion. A minimum
of one ready replica cannot mask losing that sole replica. Two or more ready
replicas permit pre-admission failover, but share Modal and regional dependencies;
this is not independent-provider failover or proof of the SLO. Measure complete
public requests and streams, alert on error-budget burn, and exercise rollout,
failed warmup, lost telemetry, and replica failure under load. Health checks alone
do not establish request availability. GPU allocation and platform failures remain
dependencies. See [Modal Server lifecycle](https://modal.com/docs/guide/servers)
and [deployment strategies](https://modal.com/docs/guide/managing-deployments).

The separately deployed Freesolo backend must recognize the exact model and price
version in its billing catalog. Qualification through the hosted router must cover
authorization, exact limits, cached pricing, cancellation, and durable settlement
as well as native GPU performance.

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
