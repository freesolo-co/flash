# Release requalification: production rollout blockers

Status at 2026-10-22 10:43 UTC: **not qualified**. The full paid rerun remains
pending because the deployed stack has not converged on the published release.

## What reached production

Flash PR #1454 reached `main` through #1455 at 08:40 UTC. Release `1.3.10`
corresponds to `3def2edc9ddbf94cc422591b1ce26a1aa8186433`. The published wheel
matches all 533 tracked Python source files. The serving image's source label
also matches that commit:

```text
ghcr.io/freesolo-co/freesolo-flash-serve@sha256:8785b5501fe9367b5a880aeb7144985e353ddae191144a7dda5134ff262773a8
```

The hosted router reports this release, but its deployment ID is
`39750647773-1-rollback`, following a failed promotion. An HTTP 200 from
`/healthz` alone does not establish a successful rollout.

## Reproduced failures

| Boundary | Observation | Consequence |
| --- | --- | --- |
| Retry workflow | Run `39749452400`, attempt 2, succeeded with deployment, readiness and canary steps skipped | A green workflow did not prove generation or settlement. PR #1458 addresses this; duplicate #1459 was closed. |
| Rolling identity | Run `39750647773` failed `health_release_not_stable` | The existing convergence deadline rejected the rollout. PR #1458 extends the deadline while retaining the stable identity window. |
| Hosted generation | Independent 9B canary returned HTTP 200 and passed stream validation | Generation itself worked; the warm request completed in 3.48 seconds. |
| Durable accounting | The same probe exhausted its 180-second accounting deadline | Verdict was `accounting_stalled`; durable settlement did not pass. |
| Backend deployment | Live OpenAPI lacks `POST /api/billing/serving-usage/durable` | Each of three canary usage records was quarantined with `permanent_http_404`. |
| Control-plane deployment | `https://flash.freesolo.co/v1/health` reports `1.2.111.post1` | The control plane is older than published `1.3.10`. |

The independent probe used the corrected validator from PR #1458 at
`fe5a645178fd68be4f7ec24747cbc57af6b527f8` in a separate process. It did not
replace production modules or configuration. Credentials remained inside the
existing router's configured environment. A unique correlation ID isolated its
settlement evidence from earlier canaries.

Read-only accounting inspection found three quarantined deliveries, no pending
deliveries and no expired delivery leases. The original workflow canary and both
audit generations all failed at the backend HTTP delivery boundary. No usage,
billing or ledger records were changed or deleted.

## Cause established, host failure still unknown

The backend route is present in source `f3cbff45`, whose image publication
workflow `39723579377` succeeded, and is unchanged in backend `main` at
`1ce67f23`. It is absent from the live API. Both that backend and the stale Flash
control plane run on the shared deployment host according to the repository's
deployment configuration. This establishes a deployment mismatch; it does not
establish whether the host's poller is paused, unable to pull, pinned to an old
image, or failing to restart containers.

Read-only SSH diagnostics could not connect to the configured host. The local
Azure CLI's cached account exists, but its refresh token has expired. Host state
has therefore not been inspected. No production restart or deployment was
attempted.

## Required recovery and rerun

1. Diagnose the deployment host and restore the intended backend and Flash
   control-plane versions. Confirm the durable settlement endpoint exists.
2. Recover the quarantined canary deliveries through an audited recovery path.
   They are terminal records: deployment alone does not automatically requeue
   them. Preserve captured prices, immutable generation facts and idempotency.
3. Land the promotion-validator fixes and run a fresh authenticated generation
   with durable settlement. Require an actual successful canary step and the
   matching live deployment identity.
4. Run the prepared multi-model training, multimodal inference, structured
   output, tool calling, adapter deployment, resume and cleanup matrix. Report
   actual outcomes and measured resource durations separately from this
   readiness investigation.

The monitor checks every five minutes. Readiness now includes the live backend
route and control-plane version alongside the release SHA, published package,
image workflows, hosted health and genuine promotion evidence.

## Validation already completed

The independently reproduced retry-workflow fix passed 127 targeted tests;
the full Python 3.12 non-wallclock suite passed 10,618 tests with 181 skipped;
the serial wallclock suite passed 25 with 2 skipped. Ruff and repository size
and layout checks passed. Those results belong to the closed duplicate fix
branch; they are not results from the pending full production requalification.
