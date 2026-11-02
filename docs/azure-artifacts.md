# Flash artifact storage on Azure

Flash-owned source snapshots, staged environments, heartbeats, logs, metrics,
adapters, and optimizer checkpoints use private Azure Blob Storage. There is no
Hugging Face read fallback or dual-write mode. Upstream base models/tokenizers and
explicit `flash models export` destinations still use Hugging Face.

The serialized `hf_repo`, `hf_prefix`, and `repo_type="dataset"` fields remain as
legacy logical identifiers so existing run preparation digests and checkpoint
bindings are not rewritten. They no longer select Hugging Face storage.

## Storage and credentials

Provision a private container with anonymous access disabled. Configure
`FLASH_ARTIFACT_BLOB_ACCOUNT_URL` and `FLASH_ARTIFACT_BLOB_CONTAINER` on the
controller. `DefaultAzureCredential` uses its managed identity or operator Azure
login. The identity needs Blob Data Contributor and permission to generate a user
delegation key. An account connection string in
`FLASH_ARTIFACT_BLOB_CONNECTION_STRING` is an alternative.

Use separate `artifacts-dev` and `artifacts` containers. These are separate from
the managed environment registry containers. Keep their names through Infisical
loading. The Freesolo companion PR supplies that Compose wiring.

Training and CPU uploader sandboxes receive a container SAS for read/create/write/
list, valid for 27 hours (the sandbox lifetime is at most 24 hours). They never
receive the controller's account key or Azure identity. The capability covers the
deployment's artifact container, like the previous shared artifact write token;
it is not a per-run authorization boundary. Environment subprocesses must retain
the existing secret isolation. Do not share these containers with unrelated data.

Hosted serving engines need separately provisioned **read-only** access. The
hosted deployments use separate dev/prod service principals with `Storage Blob
Data Reader` scoped to their artifact container, configured through `AZURE_TENANT_ID`,
`AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, and the artifact account/container variables.
They do not receive the controller identity or an account key. Azure SDK access tokens
refresh automatically; rotate the client secret before its expiry. A read-only
`FLASH_ARTIFACT_BLOB_SAS_URL` is also supported; rotate it before expiry and redeploy
engines. For customer-owned serving, set the same variable locally; the launcher
passes Azure and optional upstream `HF_TOKEN` credentials separately through its
existing private descriptor transport. Upstream tokens never authorize Azure. Custom sessions use the controller’s
`FREESOLO_INTERNAL_KEY` to derive a per-attempt token; only that token reaches the
worker (self-hosted custom training must configure the key too).

Files are streamed into immutable SHA-256 objects. A complete immutable manifest
is written before an ETag-conditional head update. Competing publications merge
against the winning head. Downloads verify size and SHA-256 before replacing a
local file. Pruning removes files from the current head and retains immutable
revision data; physical reclamation of unreferenced objects is intentionally not
part of this cutover. Do not attach an age-only Blob lifecycle deletion policy:
it can delete objects still referenced by pinned checkpoints.

## Three-day migration

The migration tool is the only Hugging Face reader for Flash-owned storage. It is
an operator tool, never imported by the runtime. It freezes a rolling **72-hour**
UTC window when the plan is created. It selects files by their authoritative
last-commit timestamps at pinned source revisions. A checkpoint is selected as a
whole by its newest file, so an upload crossing the cutoff does not lose shards.
Older checkpoints are excluded even when their repository has a recent heartbeat.
Unknown timestamps fail the plan rather than widening the window.

1. Drain active training runs and pause submissions on the old release. Back up
   the private run records and checkpoint/serving registry. Export the registry's
   immutable artifact pins into `serving-references.json`, an array of
   `{"repo":"Freesolo-Co/flashrun-...","revision":"<40-hex>","path":"sft/<run>/checkpoints/step-1/adapter"}`. Use `[]` only
   after confirming no registry pins need preservation. Nonterminal run-state source,
   environment, warm-start, and custom-resume pins are collected automatically.
2. Create a plan for each controller/container. Keep the plan outside Git:

   ```sh
   PYTHONPATH=. python scripts/migrate_recent_hf_artifacts.py plan /secure/recent-artifacts.json \
     --run-state-directory /backup/.flash/runs \
     --references /secure/serving-references.json
   ```

3. Review its UTC cutoff/end, repositories, exact revisions, paths, sizes, and
   source hashes. A required pin outside the window is an explicit planning error;
   retire that binding or run before proceeding. Do not silently broaden the
   three-day window. Stop source writes while planning and copying; otherwise
   create a fresh plan after the final drain.
4. With the intended Azure destination configured, apply the reviewed inventory:

   ```sh
   PYTHONPATH=. python scripts/migrate_recent_hf_artifacts.py apply /secure/recent-artifacts.json
   ```

   Every file is checked against its pinned Git/LFS identity, streamed to Azure,
   downloaded again, and verified before its manifest and original-revision alias
   are published. The operation can be rerun. It never overwrites an existing
   Azure head and never deletes source Hub data. Inspect the `.verified.json`
   report before cutover. Reusing a destination with an existing head preserves
   that head; reconcile deliberately if it predates the reviewed inventory.
5. Deploy the matching Flash controller, worker source, and serving changes, plus
   the Freesolo companion. Check the selected run's logs, checkpoint listing,
   adapter download/export, a paid SFT save/resume, custom restore, and serving
   generation in dev before promoting. The Modal image extension installs the
   Azure SDK even when a qualified base image predates this release.

Archived worker source from before this cutover is retained only as data. The new
bootstrap refuses to execute its retired HF backend. Start a new run from a migrated
checkpoint instead of restarting an old attempt.

Keep the old release and source data for rollback. After Azure-only writes begin,
rollback requires reconciling those writes; switching a configuration flag is not
a rollback. Historical artifacts outside the window remain on Hugging Face and
are unavailable through the new Flash runtime.

The migration plan, copy, GPU qualification, and production deployment are
separate operational steps. Adding this code does not perform those steps.

Customer-owned `flash serve deploy` accepts only image digests in
`_AZURE_ARTIFACT_IMAGE_DIGESTS`. The Azure hydration and generation canary passed
on a Modal L40S on 2026-11-02 for
`ghcr.io/freesolo-co/freesolo-flash-serve@sha256:9dc811c3458f979ce2e020509ffbbe3e51364f551234df4c3139ef8e497192dc`.
Model-specific image qualifications still apply independently. Older certified
digests contain the retired Hub hydrator and must never receive Azure credentials.
