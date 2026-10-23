# Quality and economics benchmarks

The operator suite runs a seed × cold/warm matrix for short SFT, single-turn GRPO,
long multi-turn agents, and concurrent customer adapters. It retains input locks,
raw events, driver logs, source, and a replayable JSON report. It is separate from
the customer CLI and requires the `server` extra for Pydantic.

```bash
uv sync --extra server --extra serving --dev
uv run python -m flash.benchmarks validate examples/benchmarks/quality-economics.json
uv run python -m flash.benchmarks run examples/benchmarks/quality-economics.json \
  /tmp/flash-benchmark-smoke --smoke
uv run python -m flash.benchmarks summarize /tmp/flash-benchmark-smoke
```

`--smoke` executes **synthetic fixtures**, with no credentials, GPUs, or external
requests. It tests the whole matrix and artifact pipeline; its numbers are not
performance measurements. It always sets `production_claims_eligible=false`.
The offline tests run this path in normal CI.

## Measured GPU comparisons

See the [2026-10-23 H100/B200 comparison](benchmarks/2026-10-23-gpu-comparison/README.md)
for eight real cold/warm SFT/GRPO runs, checkpoint quality curves, provider-reported
costs, startup and component timings, and per-customer serving observations.
No checkpoint reached the unchanged 65% target. The report retains an H100 serving
stall and missing agent measurements, with exact revisions, sealed training
comparisons, raw evidence, independent regrading and offline replay.

## Measured GPU pilot

See the [2026-10-22 live GPU results](benchmarks/2026-10-22-gpu-pilot/README.md)
for real H100 SFT/GRPO cold/warm runs, checkpoint quality, mixed-adapter latency,
and a 13-turn tool-use microbenchmark. The report retains negative quality results
and clearly separates this direct-provider pilot from hosted API qualification.

## Running a real suite

Compare two sealed runs with identical case names, seeds, targets, model, held-out
data, environment, evaluator, commands, timeouts and required phases. Declare the configuration or code/image changes
being tested explicitly; undeclared differences are rejected:

```bash
uv run python -m flash.benchmarks compare /tmp/h100-results /tmp/b200-results \
  --allow-change configuration.gpu --allow-change revisions.image
```

The command verifies both artifact seals and pairs trials by case, seed and cache
condition. Ratios are candidate / baseline. It reports target attainment for all
pairs, with time ratios only when both valid trials reached the target. Fixed-horizon
elapsed time and last-observed quality changes are separate measurements. Asynchronous
checkpoint evaluations can finish out of training-step order; the last observation
is not necessarily the final training checkpoint. Incomplete actual
costs never produce a cost ratio, and target costs retain their receipt bounds.
The output is descriptive; a handful of repetitions is not a statistical superiority
test. Explicitly allowed differences can still confound an experiment and must be
explained in its report.

Copy the example plan and replace its placeholder project/environment identities,
revision locks, held-out dataset, adapter digests, and operator commands. The single
example prompt illustrates the input shape; supply at least `target.min_samples`
held-out episodes for a valid measurement. The default three seeds are 17, 29, 43.
Commit the driver, collectors, evaluators, and configuration in a clean checkout.
Run without `--smoke`, with a **new** result directory outside the checkout:

```bash
export BENCHMARK_CUSTOMER_A_KEY=...
export BENCHMARK_CUSTOMER_B_KEY=...
uv run python -m flash.benchmarks run my-suite.json /tmp/flash-benchmark-001 --source .
```

Live runs allocate training/serving resources through the configured account.
The native driver is `python -m flash.benchmarks.managed`; it uses Flash's existing
API client. Each training case submits a new seeded run, evaluates published step
checkpoints and the final adapter, and keeps the quality curve. It continues to the
configured horizon after reaching the target, allowing comparison of both first
target attainment and final quality. Wall time starts before driver launch and
includes preparation, evaluation, cleanup, and receipt collection. A checkpoint's
target time is when held-out evaluation **finishes**, not when training wrote it.

Each seed runs cold first and warm second. The warm request references the cold
trial's retained directory. The operator's preparation hook must define the cache
scope (model files, compiled kernels, environment images, serving replicas, adapter
residency), establish the requested condition, and record evidence. Request order
alone is never proof that a run was cold or warm. Warm state is an infrastructure
cache condition; the native driver still trains a fresh adapter with the same seed.

The concurrent case uses separate credential-bearing clients, launches each
customer's request stream concurrently, and requires distinct customer labels,
credentials, and pinned adapter identities. Existing adapter deployments are owned
by preparation/cleanup hooks; the driver does not silently delete them. Its result
must show actual overlapping successful requests across customers and adapters.

## Environment and infrastructure integration

Public run polling does not expose complete internal spans or provider invoices.
The suite therefore requires explicit collection hooks rather than estimating
allocation/startup from poll intervals or calling the customer quote “actual cost.”
This PR supplies the native orchestration and evidence/reporting contract; operators
must connect their environment grader and infrastructure telemetry/receipt source.
It does not add automatic production-wide tracing or a Modal invoice API client.

Set `configuration.hooks` to an importable module in the retained source checkout:

| Hook | Contract |
|---|---|
| `prepare(context)` | Establish cache condition and fixtures; validate the configured model/environment before paid work. Record resource IDs before allocation. |
| `evaluate(context, targets)` | Optional for SFT/single-turn/concurrent cases, required for agents. Return `(quality_value, held_out_sample_count)`. Each target is `(ApiClient, checkpoint_id, customer_label)`. Emit request evidence for per-customer latency. |
| `cleanup(context)` | Release resources owned by preparation, preserving only the cache scope intentionally shared with the next trial. Must be idempotent after partial preparation. |
| `collect(context)` | Retain worker/allocator timing evidence and receipts, emit provenance and money events, return the names of ledgers fully reconciled for this trial. Runs after cleanup. |

`context` carries `request`, `case`, `config`, `directory`, `recorder`, `run_id`, and
`owned_deployments`. Its recorder provides `now()`, `emit(event)`, and a timed
`span(phase, source=..., attempt=...)` context manager. The default evaluator does
exact match on the last answer of a held-out conversation. Long multi-turn agents
must use their actual environment/tool execution and a task-specific held-out grader;
the driver refuses to silently substitute the generic text evaluator.

For existing command-line tooling, the bundled `flash.benchmarks.command_hooks`
module implements the same hook interface. Configure `prepare_command`,
`cleanup_command`, and `collect_command` as argv arrays, plus `evaluate_command`
for agents/custom grading. Commands run without a shell, inherit credentials via
environment variables, and receive one absolute `context.json` path. This file
contains the request, run ID, trial directory, targets, and a `result_path` to write.
Every invocation has a separate retained directory with stdout/stderr.

Command results are JSON objects:

```json
{"events": [{"kind": "cache", "observed": "unknown", "evidence": "provider did not attest cache state"}]}
```

The collector additionally returns `complete_ledgers`, for example
`["infrastructure_actual", "customer_charge"]`. A custom evaluator returns `value`
and `samples` and should include its request/rollout/environment events. Commands
cannot emit the final completion event. An unknown cache state keeps the trial
invalid, even if the driver command exits successfully.

## Evidence contract

All numeric durations are finite, nonnegative **seconds relative to the request's
monotonic origin**. Local subprocesses share that origin. Remote processes do not:
collectors must translate their timestamps to the trial clock using measured clock
offsets and retain the original clock evidence. Do not place a worker's unadjusted
monotonic timestamps in the journal. Record every attempt, including failed attempts.

| Event | Required evidence |
|---|---|
| `span` | `phase`, `start_s`, `end_s`, `source`, optional retry `attempt` |
| `quality` | `at_s`, `metric`, `value`, `samples`, exact `checkpoint`; `split` must be `held_out` |
| `request` | `start_s`, `end_s`, `customer`, `adapter`, `success`, optional measured `ttft_s` |
| `money` | `ledger`, `amount_usd`, `start_s`, `end_s`, `receipt`, `line_item`, `resource` |
| `provenance` | Observed `revisions`, authored `configuration_digest`, full `resolved_configuration`, and `evidence` locator |
| `cache` | `observed` (`cold`, `warm`, or `unknown`) and `evidence` |
| `complete` | Final `at_s`, `outcome`, and `complete_ledgers`; exactly one, last in journal |

Supported phases are allocation, startup, rollout, environment/reward execution
(`environment_reward`), training, weight publication (`weight_publication`),
checkpointing, and evaluation. Set `required_phases` per case to those that apply;
SFT need not invent rollout/reward time. A measured zero-length span means measured
zero; an absent phase stays `null`. The native driver's evaluation timer is automatic;
worker/provider phases come from the collector. Do not label the entire training
run as optimizer time or the deployment API call as weight publication time.

`revisions` locks full 40-character model/code commits, the OCI image's SHA-256
digest, SHA-256 digests of environment/data/evaluator content, and adapter content
digests for concurrent cases. The authored configuration hash is canonical JSON
over `{parameters, seed, cache}`. `parameters` contains the complete authored case
configuration. `resolved_configuration` retains the same three-key structure, with
the actual runtime parameters including server-resolved defaults, selected hardware,
and training/evaluation settings. Its seed/cache must match the trial; its separate
digest is calculated from the retained contents and reported in the summary. This
keeps the request and the configuration actually executed distinct. A collector must
independently attest the runtime identities against the lock; copying requested
values is not verification. Put resolved worker manifests,
environment packages, held-out data, evaluation outputs, and vendor receipts under
the trial directory so the suite seals them too. Never put credential values in
configuration, events, or logs; only environment-variable names belong in the plan.

## Reading the report

`summary.json` contains each trial plus separate cold/warm groups. A first target
crossing requires the named held-out metric, the configured direction, and at least
the requested sample count. Training reward/loss is not a quality target. Missed
targets retain `null` target times/costs and remain in the attainment denominator.
Group medians explicitly describe **only reached targets**; also inspect attainment
fraction, individual seeds, and the complete quality curves. The suite does not
claim statistical significance from three seeds or interpolate unobserved quality.

For each phase, busy time is the union of its spans, including nested spans and
retries. Phase busy times may overlap and must not be added as end-to-end time.
Exclusive phase time + cross-phase overlap + unattributed time equals trial wall
time. Missing phases remain visible in `measurement_gaps`; no instrumentation
means unknown, not zero. These are wall-clock attribution measures, not summed
GPU utilization or a causal critical-path analysis.

The three independent USD ledgers are:

- `infrastructure_actual`: actual provider receipt/usage-line amounts, including
  GPU, CPU, storage, environment execution and other rented resources as applicable.
- `infrastructure_estimate`: explicitly estimated infrastructure costs.
- `customer_charge`: customer invoice/charge amounts, separately reconciled.

`known_usd` preserves partial evidence. `total_usd` and target-cost bounds are
`null` until the collector explicitly declares that ledger complete. Declaring an
empty ledger complete means a measured $0, so do that only when reconciled. Receipt
line items must be unique per ledger; parallel distinct resources are additive.
Do not copy Flash's `cost_usd` or a quote into `infrastructure_actual`.

Actual receipts often span the target crossing. The report keeps lower/upper cost
bounds and a separately named uniform-interval allocation. Phase cost attribution
also assumes uniform accrual within each receipt interval and splits simultaneous
phases equally; it reconciles to known receipt totals but is **an attribution
estimate**, not an actual vendor charge per phase. Fixed point charges remain
unattributed. Inspect exclusive time and attributed cost together when prioritizing
architectural changes; overlapping work is not automatically removable wall time.

Request results retain per-customer/per-adapter p50/p95/p99 latency, success/failure
counts, failed-request latencies, and TTFT only when measured. Quantiles use nearest
rank on successful requests. The nonstreaming default evaluator cannot measure
TTFT; streaming evaluators can emit it. A quick failed request cannot improve the
success latency distribution.

`valid` requires successful execution, held-out observations, matching runtime
provenance, and attested cache state (plus observed concurrency for that workload).
`accounting_complete` additionally requires the configured phases and reconciled
actual-cost/customer-charge ledgers. `production_claims_eligible` is false for
synthetic, invalid, incomplete-accounting, or partial-workload suites; eligibility
is an evidence completeness check, not a claim about production representativeness.

## Retention and failure behavior

The suite retains the clean harness Git revision and `source.tar`, normalized suite,
per-trial effective configuration and digest, logs, raw JSONL events, resource IDs,
execution result, and any files supplied by hooks. `checksums.json` covers the exact
artifact set. `summarize` verifies it before replaying the report; additions,
deletions, edits, symlinks, duplicate receipts, malformed events, duplicate completion,
and events beyond completion are rejected. Hashes detect changes, not a malicious
driver or replacement of both files and checksums. Keep results in durable storage.

Exit codes: 0 for a valid matrix (including explicit smoke), 1 for invalid trials,
2 for plan/artifact errors, 130 for interruption. Invalid trials are retained and
later cells still run. A deadline sends SIGTERM, allows `cleanup_timeout_s` (180s
by default), then kills the local process group. The managed driver cancels its
active training run and undeploys its own checkpoints; hooks release their resources.
If cleanup fails or is killed, inspect retained `resources.json` and `cleanup.json`
and confirm remote teardown. Killing a local process is not proof of remote cleanup.
If a submit response is lost before the server returns a run ID, `submission.json`
stays pending and cleanup is marked unconfirmed; reconcile that submission with the
control plane before launching another paid suite.
An interrupted suite retains partial evidence without a final artifact seal.
