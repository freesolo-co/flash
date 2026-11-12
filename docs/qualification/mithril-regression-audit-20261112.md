# Mithril migration regression audit — 2026-11-12

Merged main at `19a746eb9` into the migration branch. The table maps historical
Spot fixes to the implementation and regression coverage retained for Mithril.

| Historical fix | Mithril behavior and coverage |
| --- | --- |
| `e5e3faa32`: recovered-run billing | Accepted fixed Spot quotes survive missing/stale provider handles and eight-GPU recovery. `test_spot_billing.py`, `test_charge_pricing.py`. |
| `c6f0e0b45`: ownership and persisted recovery | Orphan sweeps use original run IDs; persisted specs/retry state remain compatible. `test_mithril_spot.py`, `test_persisted_spec_envelope.py`, `test_resume_on_retry.py`. |
| `82adaa593`: artifact initialization | Control-plane credentials validate private storage before worker SAS writes; Azure dependencies install before capsule bootstrap. `test_mithril_api.py`, `test_mithril_spot.py`. |
| `985518ba1`: price-cap recovery | Spot preemption requeues the same shape without spending infrastructure retries, then resumes durable checkpoints. `test_retry_decision.py`, `test_mithril_spot.py`. |
| `adf83cc95`, `9dc1c8beb`, `5b996a7f1`: custom sessions | Attempt-specific pinned TLS, reconnect only to a new attempt, checkpoint-discovery retry, accumulated session time, and first-session waits bounded by the run deadline. `test_custom_spot.py`, custom client/session/worker/checkpoint suites. |
| `815c703b`: transient worker failures | Ordinary transient faults retain bounded infrastructure retry behavior; they are not mistaken for preemptions. `test_retry_decision.py`, `test_mithril_spot.py`. |
| `50ef675f7`: full quotas and refused creates | Exact-shape project quotas include other jobs. Exhausted/unreadable quotas queue before bidding. Explicit refusal plus confirmed exhausted quota requeues only after no bid is found. Timeouts/5xx remain unreconciled; existing bids take precedence. `test_mithril_capacity.py`, `test_mithril_api.py`. |
| `268ab8d01`: exact-dollar boundaries | Separate fixed billing rates ($1/$1/$4), entry thresholds ($1.01/$1.01/$4.01), and approved running caps ($2.01/$2.51/$5.01). Single- and eight-GPU bid serialization tested for every GPU class. Paid harness overrides admission only. |
| `abfa2a40a`: SIGTERM preemption | Mithril's actual poll adapter classifies signal interruption as preemption. SFT, GRPO and custom jobs with zero retries requeue without spending infrastructure retries. Other worker faults and non-Spot classification retain their behavior. |
| PR shutdown finding | Owned terminated bids remain discoverable until all instances stop. Handle-less recovery blocks on stopping or unavailable instance state. `test_mithril_api.py`. |

The shared greedy checkpoint scheduler, cancellation, asynchronous upload durability,
and runtime capsule tests also pass. No tracked legacy provider files, imports,
configuration variables, CLI image installation, or name references remain.

Validation: 769 tests passed in the combined provider, allocator, billing,
checkpoint/recovery, custom-session/image and capsule suites. After extending
bid-cap and smoke-override coverage, 87 API/smoke tests passed (including five
additional cases). Repository-wide Ruff lint/format and file/function-size and
source-layout gates passed.

This is regression coverage, not a guarantee against all provider failures.
Project quota responses do not expose every shape or shared organization limit;
those limits remain server-enforced. Unknown launch outcomes deliberately fail
closed instead of risking duplicate rentals. Admission still uses the last auction
price, not the dynamic winning-price field. No new paid test was run for this audit;
the earlier two-VM H200 SFT checkpoint/resume qualification is recorded in
`mithril-spot-20261111.json`. Multi-GPU, GRPO and custom-session paid qualification
remain unperformed.
