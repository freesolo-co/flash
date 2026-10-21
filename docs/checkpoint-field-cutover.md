# Checkpoint field cutover

The hosted contract uses `model_id` consistently in registration, fingerprint inputs, catalog
records, raw generation requests/responses, and Modal dispatch. Registration still validates that
the value identifies an exact checkpoint; inference also permits catalog base models. OpenAI `model` and
public checkpoint provenance are unchanged. The legacy `adapter_id` payload field is rejected.
Low-level vLLM adapter handles are a separate runtime concept and retain their existing name.

Deploy the CLI and hosted serving from the same release. The new CLI requires the
`serving_contract_v2` health capability before registering checkpoints. Old clients cannot
register against the new closed registration schema; old raw `/generate` callers must send `model_id`.

The Freesolo backend already uses `checkpoint_id` and treats fingerprints as opaque values, so
it does not require another field rename. Freesolo workers that install a pinned Flash package
must also move to the release containing this contract before registrations resume; an older
package pin will still emit `adapter_id`. Both artifact hashes remain in use. Changing the hash
input key changes every checkpoint fingerprint, even when all binding values are unchanged.

## Existing permanent checkpoint bindings

Use a maintenance window; this is not a rolling, mixed-version upgrade.

1. Back up the database and stop new training registrations and inference traffic. Finish requests
   and deliver all usage events. Resolve quarantined/disputed events before proceeding.
2. Disable hosted checkpoints through the existing lifecycle API, then stop old serving routers,
   engines and background workers. A database lock cannot stop an old binary from writing later.
3. From the new Flash checkout, run the migration's dry run against the intended database using
   an environment-provided `FLASH_CHECKPOINT_DATABASE_URL` and an administrative connection.
   Install psycopg in the operator environment if needed; it is not a serving dependency.

   ```sh
   python -m scripts.migrate_checkpoint_fingerprints
   ```

   This verifies every binding against its stored fingerprint. It rejects unknown or corrupted
   encodings. An empty table or an already migrated table requires no updates.
4. With the same stopped services and database backup, apply:

   ```sh
   python -m scripts.migrate_checkpoint_fingerprints --apply
   ```

   The transaction locks bindings and the usage outbox, requires disabled bindings and no
   undelivered events, and recomputes only fingerprints whose existing value matches the old
   encoding. It clears their endpoint/generation and advances their lifecycle timestamp so they
   must be loaded again. The immutable-update trigger is disabled only inside this locked
   transaction and is restored before commit; errors roll back all changes.
5. Deploy the new serving release and update CLI/raw-generation clients. Register/load the same
   immutable checkpoints, run readiness and sampling checks, then resume traffic and training.

Historical delivered usage and billing evidence retain their original fingerprints. Do not rewrite
accounting history or retry old unfinished requests under new identities. The migration intentionally
refuses an undrained outbox. Previously settled billing retries use the existing idempotency path.

Do not roll back only the binary after migrating: old and new hashes differ. Restore a coordinated
pre-cutover backup before resuming old services, or complete the new deployment.

## Databases without permanent bindings

If Freesolo PR #783's permanent-checkpoint migration has not run yet, follow its existing legacy
schema cutover first. Register checkpoints only with the new Flash release. A newly created empty
binding table needs no fingerprint migration; the tool will report zero changes.
