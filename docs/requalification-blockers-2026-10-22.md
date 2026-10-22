# Release requalification: production rollout blockers

Status at 2026-10-22 17:49 UTC: **not qualified**. The stale backend and control
plane have been repaired, and a fresh canary has settled. Historical delivery
quarantines and release publication still block the full paid rerun. The earlier
investigation below records observations at 10:43 UTC; see the recovery update.

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

## Follow-up fixes prepared

PR #1458 merged into `dev` on October 22 at 16:53 UTC. This follow-up adds a
read-only backend contract preflight before any live serving replacement, and
rechecks it when the promotion CLI runs independently. A missing authorization
or durable-settlement route rejects the attempt before GPU generation. The
real stream and durable settlement proofs remain mandatory. Release versions
advance together to 1.3.12.

[Freesolo PR #792](https://github.com/freesolo-co/freesolo/pull/792) provides an explicit, organization-scoped retry
for delivery quarantines with preserved usage/prices and an audit record.
Applying it retries no records automatically.

## Host recovery at 17:49 UTC

After the operator corrected an SSH source-IP rule, host inspection established
the deployment cause: production and dev both had `PAUSE` files dated August 27.
The timers remained enabled but exited without deploying. The backend container
dated from August and Flash ran an older pinned image; rebooting restarted those
same containers. The host's updater script also differed from current `main`.

The authorized production repair saved the previous deployment files and image
references, backed up Flash's persistent state, and deployed the published
backend and Flash images using the production compose file from `main`.
Flash initially refused startup because the old host configuration lacked Modal
credentials. The existing Freesolo workspace credentials were installed in its
protected environment file, and startup then recognized the Modal provider.
No secret values are included in this report.

The updater script now matches backend `main`. Production's pause was archived
and its timer completed a successful reconciliation; dev remains paused. No
running agent jobs were present before reconciliation. Flash's state volume
was preserved. Public checks confirm backend health, authorization and durable
settlement routes, plus Flash control-plane version `1.3.10`.

The repaired backend image digest is
`sha256:3c982c004ef28b2ffabb342001738663675e8583b7367e1357bfa17dc66e2f6e`.
Flash's image digest after the first main update was
`sha256:33b44ea436fc3664747ef793b88106ac3ce03f44555de38c08755178491d1939`.
Main advanced during repair to `1819aaea51c10427579d8dba20a466c4b5876e50`,
including PR #1458, with serving deployment `39824576816-1`.

The fresh audit correlation `fse2e-ready-f85a3780242449deb51d5786e6419a8f`
has positive `canary_settled` evidence from the accounting RPC. At 17:46 UTC
the global snapshot contained two delivered records and the three earlier
quarantines, with no pending deliveries or expired leases. The router hosting
the probe process retired before its final verdict was retrieved, so this is
durable settlement evidence, not a passing complete promotion verdict. Those
older quarantines still exceed the gate's global stall threshold. No historical
usage records were reset, deleted or manually marked delivered.

A second probe attempted from the VM refused configuration before generation
because the backend uses a legacy Supabase service-role key while the serving
gate requires `sb_secret_` format. It did not produce another canary request.

The latest package publication also failed: package source changed on main
without increasing version `1.3.10`. PR #1463 supplies the missing `1.3.11` bump on dev; this PR advances
its additional package changes to `1.3.12`. The full paid matrix remains pending.

This PR passed 10,649 offline Python 3.12 tests with 181 skipped. The companion
backend branch passed 1,246 non-live tests with 23 skipped, including 13 real
PostgreSQL recovery tests covering permissions, immutable facts, concurrency,
transaction rollback and billing replay without a duplicate debit.

At 17:50 UTC, the restored production timer picked up the subsequent main
release automatically: the public Flash control plane now reports `1.3.11`.
This verifies that the repair restored ongoing reconciliation as well as the
one-time image replacement. The timer exited successfully on consecutive runs.
