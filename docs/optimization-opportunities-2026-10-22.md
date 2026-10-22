# Optimization opportunities from release requalification

These are proposed improvements, not measured speedups or completed changes.
The full paid matrix is pending production readiness; retain the earlier
2026-10-21 findings until new measurements justify revising them.

## Detect cross-service incompatibility before allocating GPUs

The hosted generation worked while its backend settlement endpoint returned
404. Validate required backend capabilities before issuing a billable canary
or admitting production traffic. A small versioned capability response would
be more stable than treating public OpenAPI discovery as a permanent runtime
contract. This can avoid wasted cold starts and terminal usage quarantine when
services deploy out of order. Keep the real settlement canary afterward: a
capability declaration cannot prove that delivery works.

## Expose release identity across the stack

The hosted router reports a source SHA and deployment ID, but the backend
health response only reports `ok`, and the stale control plane reported an
older package version. Give each deployed service a source revision and image
identity, then compare them with the expected release contract during rollout.
Measure time from image publication to live identity convergence and alert
when the host's two-minute polling process stops converging.

## Make quarantine diagnosable and recoverable

`accounting_stalled` was accurate but required a separate read-only database
inspection to reveal `permanent_http_404`. Add bounded, sanitized reason counts
to operator diagnostics without returning customer payloads or credentials.
Provide an audited recovery operation for delivery quarantines caused by a
repaired infrastructure fault. It should preserve immutable usage and prices,
retain the original quarantine history, and prove idempotency under replay.
Never convert quarantined usage to success solely to pass promotion.

## Separate rollout convergence from inference deadlines

Rolling containers and cold GPU startup need distinct budgets. Keep stable
release identity, stream completion and durable settlement as separate proofs
with explicit elapsed times. Support Modal's result redirect without sending
the generation POST again. PR #1458 already implements the redirect and
rollout-window fixes; avoid duplicating that implementation.

## Measure all advertised model tiers with bounded cost

The current promotion canary exercises only the first advertised model. Use
separate bounded probes for 9B, 27B and 35B when qualifying a release, recording
cold start, first token and total generation time separately. Keep small
deterministic contract requests for JSON schemas, tools and images; use a
separate, explicitly budgeted workload for throughput or model-quality claims.

## Tie cleanup to a resource manifest

Track each audit's Modal applications, sandboxes, volumes and published
adapters from creation through teardown. Persist receipts even after a
sandbox exits, and verify deletion independently. This reduces idle GPU spend
and makes interrupted runs recoverable without touching shared production
resources. Compare actual invoice data with duration-based estimates when
invoice access is available.
