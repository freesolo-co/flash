# Managed environments in Azure Blob Storage

Managed `namespace/project/name` environments can use a private Azure Blob registry.
Explicit references to other GitHub repositories continue using GitHub. The default
is `FLASH_ENV_STORE=github`, so installing this release alone does not change storage.

## Provisioned resources

- Subscription: Microsoft Azure Sponsorship (`783eb7be-ef62-4357-8008-80f7b1763e20`).
- Resource group: `freesolo`; region: East US.
- Storage account: `freesoloenvs`; Standard ZRS, Hot tier.
- Production container: `environments`; development container: `environments-dev`.
- Endpoint: `https://freesoloenvs.blob.core.windows.net`.
- Anonymous blob access and shared-key authentication are disabled. HTTPS and TLS 1.2
  are required. Blob versioning and 14-day blob/container soft deletion are enabled.
- The existing `backend` VM managed identity and the operator account have Storage
  Blob Data Contributor on this dedicated account. No account key is needed.

## Configuration

Install Flash's `server` extra (includes Azure Identity and Blob SDKs). On the Flash
control plane:

```sh
FLASH_ENV_STORE=azure
FLASH_ENV_BLOB_ACCOUNT_URL=https://freesoloenvs.blob.core.windows.net
FLASH_ENV_BLOB_CONTAINER=environments
```

Use `environments-dev` for development. `DefaultAzureCredential` uses the VM's
managed identity in Azure or `az login` locally. The container must be reachable
from the control plane. Worker credentials and worker images do not change:
workers still consume the checksum-verified package staged into the existing HF
artifact repository, and never receive Azure storage credentials.

The companion Freesolo frontend and platform backend changes are required. Configure
both with `FLASH_ENV_STORE=azure`, their existing `FREESOLO_INTERNAL_KEY`, and an
explicit `FLASH_API_URL` pointing at the corresponding Flash control plane. The web
browser proxies file previews through Flash; the platform provisioner calls the
internal create-if-absent seed endpoint. Existing database hub coordinates remain
compatible and are interpreted as managed slugs; no schema migration is required.

## Data layout and guarantees

- `archives/<slug>/<sha256>.tar.gz`: deterministic flat environment archive.
- `revisions/<slug>/<revision>.json`: immutable mapping to the full archive hash.
- `heads/<slug>.json`: current published revision, committed after both objects exist.

New revision IDs are the first 40 hex characters of the archive SHA-256 to preserve
the existing worker-spec wire format. They are opaque environment revisions, not
Git commits. The full SHA-256 is stored and verified on every archive read, and
an existing immutable object with different bytes is rejected. Migration additionally
writes an alias using the source Git commit SHA.

Publishing compares the head's ETag and returns 409 if another writer changed it.
Seeding and migration create heads only if absent and never replace a user's package.
Deleting removes the head, retaining immutable archives/revisions for pinned jobs.
There is deliberately no automatic archive lifecycle deletion while jobs can reference
old revisions. Monitor stored bytes and add reference-aware cleanup separately.

Azure errors never fall back silently to GitHub: doing so could resurrect deleted
packages or run stale code. Missing records fail closed. Transient storage errors
are retryable; extraction limits, traversal/link checks, token scanning, namespace
authorization, and worker hash verification remain enforced.

## Backfill and cutover

1. Deploy both compatible releases while leaving `FLASH_ENV_STORE=github`.
2. Pause environment publishing/deletion and example provisioning for the final
   snapshot. Finish staging queued jobs before switching; jobs with already staged
   packages continue independently. Existing Git revision pins other than the final
   imported commit must already have staged packages, or their revision aliases
   must be imported separately before recovery.
3. Fetch a clean checkout of the exact environment-hub commit to import. Set the
   account URL/container above in the migration shell. Run:

   ```sh
   uv run python scripts/migrate_environment_registry.py /path/to/environment-hub
   uv run python scripts/migrate_environment_registry.py /path/to/environment-hub --apply
   ```

   Every imported head is downloaded and checksum-verified. Any failures produce a
   nonzero exit; do not switch while failures remain. Backfills never overwrite heads,
   so prepare an empty target for the final snapshot rather than assuming a second
   import will update packages modified since the first snapshot.
4. Switch the Flash control plane, frontend and platform provisioner together.
   Resume writes after verifying list, push, pull, file preview, example creation,
   run staging, and delete with a disposable environment.
5. Keep GitHub and old staged artifacts during the rollback window. Before any new
   Azure publish, rollback is a coordinated configuration reversal. After Azure
   writes begin, switching back requires exporting those packages/deletions and
   preserving their revision mappings; blindly flipping the flag loses updates.

Storage provisioning does not itself switch the live services. No GitHub data is
deleted by this migration.
