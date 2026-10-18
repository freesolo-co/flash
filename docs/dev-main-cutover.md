# Releasing the permanent-checkpoint cutover

This runbook covers the breaking changes bundled in Flash [#1392](https://github.com/freesolo-co/flash/pull/1392).
It is a release checklist, not an automated deployment fence. Do not treat a green Flash test run
or a successful application rollback as proof that the platform database is compatible.

## Coordinated changes

- Flash [#1313](https://github.com/freesolo-co/flash/pull/1313): exact `/final` and `/step-N`
  identities, immutable checkpoint bindings, and customer deployment manifest v2.
- Freesolo [#761](https://github.com/freesolo-co/freesolo/pull/761): matching storage migration,
  authorization, inventory, and billing contracts.
- Tests [#18](https://github.com/freesolo-co/tests/pull/18): cross-repository checkpoint contracts.

These PRs require a maintenance-window cutover. Mixed old and new operation is unsupported.
Record the exact Flash, Freesolo, and contract-test SHAs selected for release. Re-run the contract
tests against those revisions rather than assuming the original PR pins cover subsequent changes.

## Before merging release branches

Assign a release owner, database migration owner, and rollback owner. Record the maintenance window
and a tested database backup/restore procedure. Inventory active checkpoints and customer deployments,
including each owner, intended `/final` or `/step-N` identity, artifact binding, and redeployment plan.
Do not guess a checkpoint mapping from an ambiguous legacy alias.

Resolve these blockers and record the evidence in the release PR:

- Flash `dev` must contain current `main` ancestry. An ancestry-only repair must land with **Create
  a merge commit**; squashing or rebasing drops the parent relationship and leaves the release conflict.
- The streaming canary must carry an existing internal billing organization, allow a full GPU cold
  start, and restore the previous release using that release's serving entrypoint.
- The development adapter-schema readiness check and coordinated migration must succeed. Do not
  bypass the migration's active-adapter or undrained-usage checks to force it through.
- Thinking-enabled hosted generation currently returns HTTP 503 when durable accounting is enabled:
  the runtime does not provide the exact reasoning-token facts required by settlement. Either deliver
  and validate that accounting capability, or explicitly defer affected product flows and notify their
  users before release. Keep the accounting guard until exact facts exist; do not estimate them from
  rendered text or disable billing to make requests pass. Inventory thinking-enabled adapters too,
  because their stored mode cannot be changed by a request override.
- Decide whether to include the all-B200 placement change. [#1376](https://github.com/freesolo-co/flash/pull/1376)
  records a 9B cost increase and unequal load-test coverage. Retaining it requires a documented capacity
  and cost decision, including workspace limits and representative per-tier load results. Deferring it
  requires a separate placement change and validation; this runbook does not change GPU placement.

## Maintenance sequence

1. Coordinate both repositories' automatic production deployments so merging one cannot independently
   deploy it against the other repository's old schema or application. Freeze identity writers and
   pause new deploy, chat, sample, evaluation, teardown, and accounting work through operational controls.
2. Drain in-flight inference and accounting. Export the intended checkpoint inventory, stop alias-era
   resources as required by the migration, and take the database backup. Verify the drain before proceeding.
3. Apply the Freesolo permanent-checkpoint migration. If it refuses active adapters, pending usage,
   ownership mismatches, or ambiguous identities, resolve those records through the documented inventory
   and drain process; do not remove the guards.
4. Regenerate Freesolo's `frontend/lib/database/types.ts` with `bun run types` against the migrated
   schema and land the generated update. Run the matching platform and cross-repository checks.
5. Deploy the selected Flash and Freesolo release heads under the maintenance controls. Explicitly bind
   and redeploy every intended `/final` or `/step-N` checkpoint; recreate customer deployments with
   manifest v2. Legacy manifests and bare run aliases are not accepted by the new contract.
6. Validate chat (buffered and streaming), sample, evaluation, usage settlement, restart hydration,
   undeploy, and redeploy for exact checkpoints. Verify sibling checkpoints remain isolated. Exercise
   each shipped model tier: the automatic promotion canary probes only one base model.
7. Confirm usage is attributed to the intended organization and durably settled. Record the release
   SHAs, migration result, redeployed inventory, and smoke evidence before reopening traffic and restoring
   automatic deployment triggers.

## Rollback boundary

The Modal workflow can restore application code. It cannot reverse the checkpoint database migration,
recreate stopped alias-era resources, or make old code understand new checkpoint identities.
Before migration, a failed application promotion can use the normal code rollback. After migration,
keep maintenance controls in place and follow the coordinated restore plan: restore a mutually
compatible database and both application revisions, reconcile post-backup usage, and restore the
matching deployment inventory before reopening traffic. Restoring only Flash is not a complete rollback.
