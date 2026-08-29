# Hosted serving capacity envelope

The hosted serving checklist carries the item *"establish a measured capacity envelope for each
model, including requests per second, tokens per second, concurrency, TTFT, latency, and error
rate."* Isolated measurements and canaries existed, but nothing produced the complete per-model
envelope, so there was no defensible answer to "how much load can each hosted model take."

This document is the envelope's contract and its results table. The harness that produces it lives
in `flash/serving/bench/`, driven by `scripts/bench_hosted_capacity.py`.

Scope: freesolo-owned hosted serving (`flash/serving/`). Not customer-owned `flash/serve`, not
training viability.

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

- **requests/sec** — `successful_rps`, successes over full wall time.
- **tokens/sec** — `output_tokens_per_second`, engine-reported completion tokens over full wall
  time. Engine-authoritative, never counted from the text.
- **concurrency** — the closed-loop level held for the cell; the envelope reports the ceiling, the
  knee, and the saturation point derived across the sweep.
- **TTFT** — `ttft_seconds`, p50/p95/p99.
- **latency** — `latency_seconds`, p50/p95/p99, flagged `p99_descriptive_only` under 400 samples.
- **error rate** — observed rate plus a Wilson one-sided 95% upper bound.

## Results

**Not yet measured.** The harness is complete and validated allocation-free; no GPU has been
allocated, because doing so requires a separate explicit authorization (Modal source-upload
permission and a finite numeric budget) that has not been given. The quote is ~$8 for a full pass
and ~$20 including canaries and one failed boot:

| model | GPU-s | cost |
| --- | --- | --- |
| 9B / L40S | 2280 | $1.24 |
| 27B / H100 | 2560 | $2.81 |
| 35B / H200 | 3170 | $4.00 |

`test_the_planned_sweep_costs_what_the_plan_quoted` pins those numbers against the pricing code, so
the quote presented for authorization is reproducible from the code that would spend.

When the sweep runs, the tables land here — one per model, one row per bucket and concurrency
point, plus the derived curve (`throughput_ceiling_tokens_per_second`, `knee_concurrency`,
`saturation_concurrency`).

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
modal run scripts/bench_hosted_capacity.py --base-model Qwen/Qwen3.5-9B --mode canary
modal run scripts/bench_hosted_capacity.py --base-model Qwen/Qwen3.5-9B --mode sweep --bucket short_interactive
```

The boot dominates cost (~960s of ~1000s per cell in a prior campaign), so one boot runs a whole
bucket's concurrency grid rather than one cell. `budget.py` reserves before allocation and raises
`BudgetExceeded` rather than overspending; an unrecorded GPU tier raises `UnknownGpuRate` rather
than being priced at a neighbour's rate.
