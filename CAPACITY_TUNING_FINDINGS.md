# Hosted serving capacity tuning

Source: Flash `dev` @ `94210a323f9beaa713241e305f178b364848446d`. The offline analysis below is
allocation-free; the B200/B300 measurements are recorded separately in `B200_TUNING_LOG.md`.

> **`dev` moved during this work.** It is now `b3d5714f` (+3 commits: #1333 activated hosted 27B on
> H100, #1344 header forwarding, #1345 uncapped autoscaling + buffer container). Two of those land
> directly on this file's subject and are reconciled in "What `dev` did instead" below. The 9B engine
> block is **byte-identical** across `94210a32` and `b3d5714f`, so the paid 9B measurement remains
> valid against current `dev`.

Scope: the Freesolo-owned hosted plane under `flash/serving`. Customer-owned `flash/serve` is
untouched.

## The defect

`max_inputs` (Modal) counts logical **requests**. `max_num_seqs` (vLLM) counts child **sequences**.
`dev` derived one from the other as `max_inputs = max(8, min(64, max_num_seqs * 2))`, so both active
models admitted **16 requests into an engine with 8 decode slots**.

OpenAI `n` is accepted 1..4 (`flash/serve/runtime/sampling.py:15`) and is passed straight to
`SamplingParams`, so one admitted request can fan out to four sequences. Worst case on `dev` was
therefore **64 sequences against a cap of 8**.

Surplus requests do not trigger scale-out; they queue _inside_ the container.

## Evidence

Measured twice, independently, from `internal.serving_token_usage`:

| slice                        | finding                                                                |
| ---------------------------- | ---------------------------------------------------------------------- |
| 2026-08-23, 122,572 reqs     | 9B flat ~43.8 tok/s to fleet-conc 15, ~20 by 30-33 (2.19x)             |
| 2026-08-27, 71,453 reqs (9B) | flat 42.0-44.6 to conc 15, 20.5 at conc 30 (2.13x)                     |
| 4B per-container, 2026-08-23 | conc 8: 22.4 tok/s/req, 179.6 container tok/s; conc 16: 11.3 and 181.0 |

The 4B pair is the cleanest: per-request speed halves (1.98x) while container throughput stays
**flat (1.01x)**. Both predictions of "cap-bound, not GPU-bound" confirmed to within 1%.

**What the 35B per-replica slice does and does not show.** A slice of the 899 rows carrying
`engine_replica_id` (10 replicas, last 30d) shows per-replica concurrency topping out at **exactly
8, with no bucket at 9 or above**: ~80-84 tok/s at conc 1-4, one step down to ~68-70 at conc 5-8,
then nothing. The curve is gentle saturation, not the 9B's cliff, consistent with the 35B being
KV-bound.

The measure is not tautological, which was worth checking before relying on it. The sweep-line uses
`inference_time_seconds`, whose timer starts _before_ `engine.generate` (`generation.py:140`), so the
interval spans vLLM's waiting queue rather than decode alone. A request queued behind a full engine
still counts, so it measures **offered load**, not served occupancy, and a bucket at 9+ was reachable.
`engine_replica_id` is a per-container UUID minted at engine load (`lora_engine.py:57-75`), so these
are genuinely per-container.

**But read the null result correctly.** Nothing in the engine caps in-container occupancy at 8 — the
only locks are per-adapter (`lora_engine.py:78-81`), not a global admission limiter. So the honest
reading is that offered load on this tier **never reached** the point where the surplus admission
slots would have been used, not that the surplus was observed filling with queue. On the 35B the
`max_inputs` 16 -> 8 change is therefore **preventive**: it removes headroom that measurement shows
was never exercised, on the tier where exercising it is most expensive. The _corrective_ evidence —
queueing actually degrading service — is the 9B and 4B data, not this.

Caveats recorded: the 9B numbers are fleet concurrency, not per-container (0 of 71,453 rows carry
`engine_replica_id`); the collapse is not monotonic past conc 33 at small n; the 9B rows fall in a
2026-07-28..29 busy period.

## Per-sequence cost is per-model

Both surviving models are **hybrid GatedDeltaNet**, so each sequence slot holds a recurrent state
that is **constant in context length** — the "real context is tiny" discount does not apply to it.
Shapes confirmed against vLLM `Qwen3NextGatedDeltaNet.get_state_shape`: temporal state
`(num_v_heads, head_k_dim, head_v_dim)` fp32, conv `conv_dim=8192`, `state_len=3`.

|                           | 9B (24 of 32 linear)  | 35B-A3B (30 of 40 linear) |
| ------------------------- | --------------------- | ------------------------- |
| recurrent + conv state    | 49.1 MiB/seq          | 61.4 MiB/seq              |
| logits share              | ~0.95 MiB/seq         | ~0.95 MiB/seq             |
| measured p99 total tokens | **407** (1.2% of 32k) | **8,712** (26.6% of 32k)  |
| KV at p99                 | 6.4 MiB/seq           | **85.1 MiB/seq**          |
| dominant term             | **recurrent state**   | **KV**                    |

Sizing the cap from the logits buffer alone understates the cost by ~50x. And the dominant term
**differs per model**: the 9B is state-bound, the 35B is KV-bound at its p99. The 35B's p99
completion is 8,000 against a max of 8,192, which looks like a generation cap being hit rather than
a demand tail — worth owning separately.

Method cross-check: independently derived 35B full-attention KV of **10.0 KiB/token**, applied to
the canary's documented **679,701-token** pool, gives **20.7** full-32k sequences, matching
`model_config.py`'s own "about 20x concurrency at 32k".

## The change

- `max_inputs` becomes an explicit per-model catalog value instead of a derived one, because no
  single ratio can be right at both `n=1` and `n=4`.
- `_engine_concurrency` **fails closed** when `max_inputs > max_num_seqs`.
- **9B**: `max_num_seqs` 8 -> 16, `max_inputs` 16. Costs ~0.38 GiB of state (0.89% of the 43.2 GiB
  budget), traded out of the KV pool: ~25k tokens, about 62 requests of headroom at its 407-token
  p99. Also widens vLLM's CUDA-graph ladder 5 -> 7 rungs (10 -> 14 captures with LoRA
  specialization).
- **35B-A3B**: `max_num_seqs` stays 8; `max_inputs` 16 -> 8. This is the tier with the documented
  profiling OOM, and the arithmetic does not model its all-expert activation spike.

Consequence stated in code: a model whose `max_inputs` drops also scales out earlier, and the 35B
boot is ~17 min. That is the intended trade — queueing behind a saturated engine is unbounded,
whereas the boot is paid once.

## What `dev` did instead, and why the sweep is what settles it

`dev` @ `b3d5714f` keeps the derived form and reads it as a **deliberate** buffer, not an oversight.
`_engine_concurrency` is now documented as capping `max_inputs` "near the engine's capacity with a
small boot buffer (2x, so a cold-booting replacement doesn't reject bursts)". The resulting overshoot
is uniform:

| model     | `max_num_seqs` | dev `max_inputs` | children at `n=1` | children at `n=4` |
| --------- | -------------- | ---------------- | ----------------- | ----------------- |
| 9B        | 16             | 32               | 32 (2.0x cap)     | 128 (8.0x cap)    |
| 27B (new) | 8              | 16               | 16 (2.0x cap)     | 64 (8.0x cap)     |
| 35B-A3B   | 8              | 16               | 16 (2.0x cap)     | 64 (8.0x cap)     |

So this is a genuine design disagreement, not a bug I found and `dev` missed:

- **My uncommitted change** sets `max_inputs == max_num_seqs` and fails closed above it. It treats
  in-container queueing as the thing to eliminate.
- **`dev`** accepts 2x queueing on purpose, to buy burst absorption while a replacement container
  cold-boots (420s L40S / 900s H100 / 1010s H200 — a real cost).

Both are defensible from the desk. The disagreement is empirical: **does the 2x overshoot cost more
in per-request latency than the cold boot it avoids?** That is precisely what the `n=1` above-cap and
`n=4` cells measure, and it is why the sweep runs cells at `concurrency = 2x cap` rather than
stopping at the cap.

One thing is worth flagging regardless of which side the data favors. The global `MAX_INPUTS = 64`
that bounds the derivation is justified in-code by a sweep on **A10G with Qwen2.5-1.5B** — a
different card and a model several times smaller than the smallest hosted tier, with no GDN
recurrent state at all. Whatever that sweep showed about saturation does not transfer to a 9B or 35B
hybrid on Blackwell. The number may still be right; its stated evidence does not reach.

**Nothing here is published until measured.** The offline sections above are analysis, and analysis
is what motivated the measurement, not a substitute for it.

## Validation

- `tests/serving/` suite: green (exit 0).
- Focused: `test_per_model_gpu.py`, `test_model_config.py`, `test_usage_reporter.py` green.
- `ruff format --check`, `ruff check`, `check_file_size.py`, `check_function_size.py`: all pass.
- Sabotage-verified twice: setting `max_inputs: 24` raises at import with the exact message;
  replacing the guard with `if False:` produces "DID NOT RAISE ValueError". Both restored and
  re-verified.
- Proved by introspection that `max_inputs` cannot reach vLLM: `engine_args_for` emits a fixed
  13-key allowlist, and neither `_required_immutable_args` nor `_build_specific_args` mentions it.
  The other two consumers (`lora_engine.py`, `boot.py`) read specific keys via `.get()`.

## Limits — read before shipping

- **Not measured on a GPU.** The recurrent-state and CUDA-graph figures are derived from published
  config shapes and vLLM's documented ladder, not read from the allocator. A real cold boot should
  confirm the 9B at `max_num_seqs=16` before this ships.
- `dev` has **no capacity-503 path** (`errors.py` maps only 504/502), so an absence of 503s under
  load is not evidence of headroom.
- These caps bound in-container queueing. They are not an availability or latency SLA.
- A B200 extrapolation would not transfer: without the CUTLASS repair, GDN prefill can fall back to
  Triton, so it may measure a different kernel than the L40S/H200 tiers run.
