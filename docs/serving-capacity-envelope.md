# Hosted serving capacity envelope

The hosted serving checklist carries the item *"establish a measured capacity envelope for each
model, including requests per second, tokens per second, concurrency, TTFT, latency, and error
rate."* Isolated measurements and canaries existed, but nothing produced the complete per-model
envelope, so there was no defensible answer to "how much load can each hosted model take."

This document is the envelope's contract and its results table. The harness that produces it lives
in `flash/serving/bench/`, driven by `scripts/bench_hosted_capacity.py`.

Scope: freesolo-owned hosted serving (`flash/serving/`). Not customer-owned `flash/serve`, not
training viability.

## Relationship to `flash/serving/loadtest/`

Two harnesses exist because they measure two different boundaries, and neither substitutes for the
other.

`flash/serving/loadtest/` drives the **public HTTP front door** with `httpx` against `/healthz` and
`/v1/chat/completions`. It measures what a client experiences through the router, including dispatch
and queuing. Its own documentation states that it cannot prove a replica scaled out, cannot prove an
engine was cold, and cannot separate a dispatch deadline from the underlying resource constraint.

`flash/serving/bench/` boots each model's engine **on that model's own production tier** and drives
it in-process, so the number it produces is one engine's capacity rather than the front door's
behaviour. Two consequences follow from that difference:

- It invalidates any request that reports `cached_tokens > 0` as `ERROR_CACHE_CONTAMINATED`, and any
  request whose cached-token count is unreported as `ERROR_CACHE_UNVERIFIED`. Prefix-cache reuse is
  not a fast success; it is a measurement that did not happen. The loadtest harness counts cached
  tokens but does not invalidate on them, which is correct for its purpose and wrong for this one.
- It derives a capacity curve — throughput ceiling, knee concurrency, saturation concurrency — which
  requires sweeping concurrency against a fixed engine. The loadtest harness derives no such curve.

Use the loadtest harness to ask whether the deployed service is holding up. Use this one to ask how
much a single model engine can take.

## What is measured

Three models, each on its own production tier. No benchmark-chosen card: an envelope measured on a
card the model is not served on is not that model's envelope.

| model | tier | `max_num_seqs` | `max_model_len` |
| --- | --- | --- | --- |
| `Qwen/Qwen3.5-9B` | L40S | 8 | 32768 |
| `Qwen/Qwen3.8-27B` | H100 | 8 | 32768 |
| `Qwen/Qwen3.6-35B-A3B` | H200 (bf16) | 8 | 32768 |

The engine args are production's, resolved through the same `engine_args_for()` call the real
service uses; `test_bench_engine_args_match_production_engine_args_exactly` asserts the two dicts
are equal for all three models.

Three request shapes, preregistered:

| bucket | input tokens | max output | depth floor |
| --- | --- | --- | --- |
| `short_interactive` | 512 | 128 | 300 requests |
| `medium_generation` | 8192 | 256 | 120 requests |
| `near_32k` | 31744 | 512 | 20 requests |

Six concurrency points per bucket: **1, 2, 4, 8, 12, 16** — powers of two up to the engine cap,
then past it, so the knee is bracketed rather than assumed. Decoding is fixed at
`temperature=0.0`, `top_p=1.0`, `n=1`, thinking on.

## The six metrics

Per cell (model × bucket × concurrency):

- **requests/sec** — `successful_rps`, successes that *finished inside the measurement window*
  over that window's duration.
- **tokens/sec** — `output_tokens_per_second`, engine-reported completion tokens from those same
  in-window successes, over the window. Engine-authoritative, never counted from the text.
- **concurrency** — the closed-loop level held for the cell; the envelope reports the ceiling, the
  knee, and the saturation point derived across the sweep.
- **TTFT** — `ttft_seconds`, p50/p95/p99, in-window requests only.
- **latency** — `latency_seconds`, p50/p95/p99, in-window requests only, flagged
  `p99_descriptive_only` under 400 samples.

Both rate denominators are the **steady-state window**, excluding the drain. A cell that closes its
window with requests still in flight finishes them at falling concurrency, on a progressively idler
engine; counting that tail in either the numerator or the denominator would misreport the load the
cell was actually holding. `drain_seconds` is published alongside, so the excluded tail is visible
rather than merely omitted.
- **error rate** — observed rate plus a Wilson one-sided 95% upper bound.

A request whose terminal event was DELIVERED and whose stream then failed to close cleanly is a
success, not an error: the completion was served, and charging the cleanup to the request would
inflate the error rate under exactly the slow teardown the delivered-final clock exists to keep out
of latency. Those faults are published separately as `cleanup_faults` and `cleanup_breakdown`, so an
unhealthy stream or container stays visible instead of being absorbed into a clean cell.

## Results

### Qwen/Qwen3.5-9B on L40S

Measured 2026-08-31, invocation `495dad60099e`, one container, one block. **2,614 requests
attempted, 2,614 succeeded, 0 failed** across all 18 cells (1,731 `short_interactive`, 726
`medium_generation`, 157 `near_32k`).

Engine identity is asserted, not assumed. Every cell records `replica_ids`, and all 18 name the
same single container (`41b37856...`), so the whole sweep measured one engine rather than a
re-boot per cell. Each of the three bucket artifacts independently records the engine it booted:
KV pool `760 blocks x 1056 tokens` (fp8), GDN prefill backend `triton` resolved by vLLM itself,
model `Freesolo-Co/Qwen3.5-9B-FP8` @ `878d83ed`, tokenizer `Qwen/Qwen3.5-9B` @ `c2022362`, driver
`580.95.05` read from NVML. All three agree.

`triton` is the correct GDN path on sm89; the `warning_once` fallback that gate 5 guards against
is a Blackwell failure mode and does not apply to this card.

The 1056-token block size is not a tuning choice: vLLM raises the attention block size to keep
the attention page at least as large as the mamba page, which is what sets the pool arithmetic.

#### `short_interactive` (512 in / 128 out)

| concurrency | attempted | ok | failed | rps | output tok/s | TTFT p50 | TTFT p95 | latency p50 | latency p95 | latency p99 | error rate | error bound (95% upper) | bound resolved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 202 | 202 | 0 | 0.479 | 61.3 | 0.062s | 0.063s | 2.080s | 2.082s | 2.083s | 0.000 | 0.013 | no |
| 2 | 301 | 301 | 0 | 0.926 | 118.6 | 0.124s | 0.131s | 2.152s | 2.156s | 2.157s | 0.000 | 0.009 | yes |
| 4 | 301 | 301 | 0 | 1.785 | 228.5 | 0.157s | 0.160s | 2.236s | 2.239s | 2.526s | 0.000 | 0.009 | yes |
| 8 | 305 | 305 | 0 | 3.184 | 407.6 | 0.232s | 0.264s | 2.478s | 2.485s | 2.527s | 0.000 | 0.009 | yes |
| 12 | 309 | 309 | 0 | 3.209 | 410.8 | 0.336s | 2.558s | 2.622s | 4.858s | 4.928s | 0.000 | 0.009 | yes |
| 16 | 313 | 313 | 0 | 3.210 | 410.9 | 2.615s | 2.662s | 4.913s | 4.928s | 5.058s | 0.000 | 0.009 | yes |

Curve: ceiling **410.9 tok/s** at concurrency 16, knee at **8**, saturation at **12**.

#### `medium_generation` (8192 in / 256 out)

| concurrency | attempted | ok | failed | rps | output tok/s | TTFT p50 | TTFT p95 | latency p50 | latency p95 | latency p99 | error rate | error bound (95% upper) | bound resolved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 89 | 89 | 0 | 0.210 | 53.6 | 0.634s | 0.651s | 4.738s | 4.944s | 5.087s | 0.000 | 0.030 | no |
| 2 | 121 | 121 | 0 | 0.372 | 95.3 | 0.844s | 0.903s | 5.363s | 5.397s | 5.406s | 0.000 | 0.022 | no |
| 4 | 123 | 123 | 0 | 0.604 | 154.7 | 0.989s | 1.040s | 6.558s | 6.647s | 7.415s | 0.000 | 0.022 | no |
| 8 | 127 | 127 | 0 | 0.868 | 222.3 | 1.009s | 1.383s | 8.915s | 9.179s | 12.136s | 0.000 | 0.021 | no |
| 12 | 131 | 131 | 0 | 0.871 | 223.0 | 3.507s | 7.328s | 11.714s | 15.547s | 18.583s | 0.000 | 0.020 | no |
| 16 | 135 | 135 | 0 | 0.866 | 221.8 | 9.807s | 10.229s | 17.913s | 18.208s | 21.184s | 0.000 | 0.020 | no |

Curve: ceiling **223.0 tok/s** at concurrency 12, knee at **8**, saturation at **12**.

#### `near_32k` (31744 in / 512 out)

| concurrency | attempted | ok | failed | rps | output tok/s | TTFT p50 | TTFT p95 | latency p50 | latency p95 | latency p99 | error rate | error bound (95% upper) | bound resolved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 20 | 20 | 0 | 0.087 | 44.6 | 2.954s | 2.989s | 11.477s | 11.513s | 11.531s | 0.000 | 0.119 | no |
| 2 | 21 | 21 | 0 | 0.138 | 70.4 | 3.339s | 3.736s | 14.332s | 14.484s | 15.973s | 0.000 | 0.114 | no |
| 4 | 23 | 23 | 0 | 0.187 | 94.7 | 3.664s | 8.796s | 19.835s | 24.830s | 26.980s | 0.000 | 0.105 | no |
| 8 | 27 | 27 | 0 | 0.199 | 101.7 | 3.787s | 20.605s | 30.946s | 47.729s | 50.006s | 0.000 | 0.091 | no |
| 12 | 31 | 31 | 0 | 0.199 | 101.7 | 18.812s | 39.299s | 46.336s | 66.835s | 69.110s | 0.000 | 0.080 | no |
| 16 | 35 | 35 | 0 | 0.195 | 99.7 | 34.547s | 51.735s | 63.256s | 80.238s | 82.647s | 0.000 | 0.072 | no |

Curve: ceiling **101.7 tok/s** at concurrency 8, knee at **8**, saturation at **8**.

#### Reading the 9B curve

The **knee is at concurrency 8 in all three buckets** -- exactly the engine's `max_num_seqs = 8`.
That is the number to operate at. Past it the added wait moves into TTFT rather than buying
throughput: from c=8 to c=16 `short_interactive` gains 0.8% throughput (407.6 to 410.9 tok/s) while TTFT p50
grows 11x (0.232s to 2.615s). Those requests are waiting for an
admission slot, not for a busier GPU.

The curve's `saturation` marker sits one grid point later (12 for `short_interactive` and
`medium_generation`, 8 for `near_32k`) because it marks where throughput stops improving at all,
while the knee marks where it stops improving *proportionally*. The knee is the operating point;
the gap between them is latency bought at no throughput gain.

**Operating point: concurrency 8.** Beyond it you buy latency, not throughput.

Every cell carries `p99_descriptive_only`, because no cell reaches the 400 samples a p99 needs;
the p99 column describes the sample and is not a guarantee. The error bound resolves only where
`bound resolved` says yes: `medium_generation` and `near_32k` ran 20-135 attempts against the
~268 a 1% Wilson bound requires, so they publish an unresolved bound rather than a false one.

### Cost

| lane | GPU | reserved | settled | GPU-seconds |
| --- | --- | ---: | ---: | ---: |
| canary 9B | L40S | $4.89 | $0.2019 | 372 |
| sweep 9B (3 buckets) | L40S | $37.15 | $2.4620 | 4543 |
| canary 27B | H100 | $9.91 | $1.5613 | - |

Settled cost is ~6% of the reservation. That gap is the design working as intended: a
reservation is a spending authorization priced at every bucket's `max_seconds`, every request at
`REQUEST_TIMEOUT_SECONDS`, and the boot at the full `STARTUP_TIMEOUT_SECONDS` a stuck boot may
bill, so it must be wrong in the direction that refuses a run rather than the direction that
overspends.

One operational note for whoever runs this next: the ledger holds back a **submission stop at
80% of the ceiling** for delayed charges and teardown, so a reservation must clear the *stop*,
not the ceiling. Pass `--ceiling-usd` at roughly 1.25x the reservation or the lane is refused
before it allocates.

### Remaining models

**Not yet measured.** 27B/H100 is sweeping; 35B/H200 follows. Their tables land here in the
same shape, and each tier passes its own canary -- real card identity, immutable model,
tokenizer and processor revisions, 32768 configured context, a GDN backend named by vLLM's own
resolver, finite non-empty output with a terminal finish reason, and confirmed teardown --
before its sweep is allowed to spend.

## What the envelope will not claim

These are gates in the harness, not aspirations.

1. **Dev is not live.** Production `/healthz` reports `deployment_sha c1093be8` (`origin/main`,
   2026-08-23) with six models, naming the 27B `Qwen/Qwen3.6-27B`; dev declares three and calls it
   `Qwen/Qwen3.8-27B`. This measures dev's shape, not what customers hit today.
2. **No headroom claim from absent rejections.** Dev has no capacity-rejection contract — no
   429/503, no `Retry-After`; `errors.py` maps only FunctionTimeoutError→504 and ModalError→502.
   Absence of rejections under overload is therefore not evidence of headroom.
3. **Per-container, not fleet.** `max_containers=1` measures one engine. Fleet capacity is that
   number times replicas *only if* scale-out is independently demonstrated, which this does not do.
4. **A prefix-cache hit is an error, not a fast success.** Any request reporting `cached_tokens > 0`
   is invalidated (`ERROR_CACHE_CONTAMINATED`) — its TTFT and throughput describe a cache hit
   rather than capacity. An *unreported* cached-token count is also an error
   (`ERROR_CACHE_UNVERIFIED`), because `_num_cached_tokens` returns 0 when the attribute is absent,
   so a build that stopped reporting would make the whole campaign read as clean while being
   entirely unverified.
5. **No capacity number for a kernel that would not ship.** The GDN prefill backend is asserted from
   vLLM's own resolver, not from boot success: on Blackwell the failure mode is `warning_once` then
   `return backend, "triton"` with no raise, so an unrepaired boot serves and bills the fast-card
   rate while running the slow kernel.
6. **An abandoned request is a failure, not an absence.** Cancelling in-flight tasks at cell end
   would remove them from the numerator *and* the denominator, so an overloaded cell would report a
   cleaner error rate than it earned. Issued work is drained under its own timeout and every record
   counted; anything still pending is recorded as `ERROR_TIMEOUT`.

## Sample depth: two questions, not one

A zero-failure cell needs **268** clean attempts before its Wilson bound clears 1% (z=1.645). That
is minutes of wall time at 512/128 and hours at 31744/512.

So depth is a property of the request shape, carried on the bucket, and two questions stay separate:

- `error_bound_resolved` — did we look long enough for the bound to mean anything?
- `degraded` — did the cell actually fail?

Collapsing these into one boolean makes a clean 20-request cell indistinguishable from a 5%-error
300-request cell, and `summarize_curve` then reports the first shallow cell as the saturation point,
erasing the envelope it exists to publish. A shallow clean sweep still publishes its throughput,
knee, and saturation; what stays unresolved is the error-rate bound, reported separately as
`bound_resolved_points`.

## Running it

Allocation-free, $0:

```
uv run pytest tests/serving/test_bench_hosted_capacity.py
uv run python scripts/check_file_size.py && uv run python scripts/check_function_size.py
uv run ruff check flash/serving/bench scripts/bench_hosted_capacity.py
```

Paid, only under explicit authorization — a per-model canary asserting GPU identity, exact
model/tokenizer/processor provenance, and 32768 configured context runs before any sweep, and
teardown is confirmed after each model:

```
modal run scripts/bench_hosted_capacity.py --base-model Qwen/Qwen3.5-9B --mode canary --ceiling-usd 20
modal run scripts/bench_hosted_capacity.py --base-model Qwen/Qwen3.5-9B --mode sweep --bucket short_interactive --ceiling-usd 59
```

`--ceiling-usd` is required and has no default: both commands allocate a GPU, and a spend ceiling
that a caller can forget to pass is not a ceiling. The entrypoint reserves the lane's worst-case
GPU-seconds against it and raises `BudgetExceeded` BEFORE allocating, so a ceiling below the lane's
own cost refuses the run rather than discovering the overrun partway through. The values above are
per-invocation ceilings for a 9B L40S lane; a larger model or a wider bucket selection needs a
correspondingly larger one. They are stated above what each lane reserves AND above its
submission stop, because a documented ceiling a lane cannot clear is a command that cannot run.

The canary reserves
`2700s boot + 900s method-timeout headroom + 300s probe + 5 x 1001.887s warmups + 120s scaledown = 9029s`
($4.89 at the recorded L40S rate) — ONE boot, because `_run_canary` makes a single `certify.remote()`
call that probes and warms the same container. Each warmup is priced at its request timeout PLUS the
fit that precedes it, and that fit is funded at the bound the watchdog actually terminates on
(bound + grace), not the nominal bound. The single-bucket `short_interactive` sweep reserves 26738s
($14.49): the canary term once per separately bootable call, one boot per call, two bounded probes,
`6 x 420` windows, `6 x 930` drains, 579s of funded prompt fitting, and one scaledown tail plus one
headroom grant per call. A full three-bucket sweep reserves 68544s ($37.15 on L40S) and needs a
ceiling above $47.

Every figure here is what the estimators in `scripts/bench_hosted_capacity.py` actually compute, not
a separately maintained transcription: retuning a bound moves the reservation the ceiling is checked
against, so a stale number in this runbook would authorize a run the code then refuses. The two
`--ceiling-usd` values above are asserted against the estimators by
`test_documented_ceilings_exceed_what_each_lane_reserves`.

The probe is bounded at `PROBE_TIMEOUT_SECONDS` (300s) rather than inheriting the class method
timeout, and reserved once per canary plus once per bucket. It only reads NVML, asks vLLM's resolver
which GDN prefill backend it chose, and loads the served config, so anything near that bound is a
stall — but on the class timeout a stall would have billed hours against a lane whose estimate
assigned the probe nothing at all. `_probe_in_container_within_bound` runs inside the already-loaded
method, so the 300s covers probe work rather than the cold model load, and it ENDS THE PROCESS on
timeout rather than raising: timing out a future does not stop the worker thread, and a thread stuck
in an uninterruptible C call would otherwise keep billing through the container's scaledown window.

A sweep makes `len(buckets) + 1` separately bootable calls: the canary's single `certify` call plus
one per bucket. There is no second canary boot to reserve, because there is no longer a gap between
a probe call and a warmup call for a replacement container to land in.

Prompt fitting is in the reservation AND in the method timeout, but deliberately not in any
`max_seconds`. It runs before a cell's clock starts, so tokenization cannot compete with the
measured window for CPU — but it runs on the rented GPU container, so it bills and the method clock
runs through it. At 31k input it is the single largest term outside the windows themselves: 4050s
across the `near_32k` grid against 579s for `short_interactive`. Funding it without bounding it left
`run_bucket` able to exceed its own `timeout` mid-grid, and because that timeout fires before the
call returns, the bucket's artifact would never be written — losing every cell already paid for.

That trailing `replacement` term is the non-obvious one. `max_containers=1` caps how many replicas
run at once; it does NOT pin successive `.remote()` calls to the container the previous bucket
booted. A bucket can therefore land on a cold replacement and pay another boot plus its warmups,
which bills whether or not the reservation admitted it.

A lane also has to clear its submission stop, not merely its ceiling: `reserve()` refuses at 80% of
the ceiling so settlement lag and teardown stay funded. A lane consequently needs a ceiling around
`1.25x` its own reservation -- on B200, $19.59 for the canary and $58.03 for the single-bucket
`short_interactive` sweep -- and the `--ceiling-usd 20` and `--ceiling-usd 59` above clear those
thresholds. These scale with the catalog's tier: the same canary reserves $6.12 on L40S and $14.23
on H200, so a re-tiered catalog moves every figure here. A full three-bucket sweep is the number to
watch: it reserves **$148.74 per model on B200**, against $46.44 on L40S.

The boot dominates cost (~960s of ~1000s per cell in a prior campaign), so one boot runs a whole
bucket's concurrency grid rather than one cell. `budget.py` reserves before allocation and raises
`BudgetExceeded` rather than overspending; an unrecorded GPU tier raises `UnknownGpuRate` rather
than being priced at a neighbour's rate.
