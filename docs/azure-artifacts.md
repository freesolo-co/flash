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

## Kernel-cache builds

Kernel-cache CI uses the separate private `build-artifacts` container. Its dedicated
identity has Blob Data Contributor on that container and Blob Delegator on the
account. GitHub uses `AZURE_BUILD_CLIENT_ID` and `AZURE_BUILD_CLIENT_SECRET`, with
the existing tenant setting. It cannot access dev or production training artifacts.
The GPU sandbox receives only a bounded SAS; the CI identity deletes its disposable
source/cache namespace afterward. Hugging Face is no longer used for build transport.

## Three-day migration

The one-time importer was retired after the final inventory was published and
verified. The completed window is **2026-10-30 04:33:53.792845 UTC through
2026-11-02 04:33:53.792845 UTC**. Files were selected by authoritative last-commit
timestamps at pinned revisions; whole checkpoints were selected by their newest
file, so the cutoff did not split checkpoint shards.

The import verified 11,159 distinct repository/content objects (601,802,167,791
bytes), then published and read back 629 revision snapshots covering 29,321 file
entries. Original Hub revision aliases preserve existing checkpoint bindings.
673 staged objects outside the final referenced inventory were removed only after
checking every published manifest. Source Hugging Face data was not deleted.

The private inventory and per-object verification receipts remain with the migration
operator. The importer can be recovered from the pre-cleanup Git revision for an
explicitly reviewed recovery; it is no longer shipped with Flash. See
[the migration evidence](azure-migration-evidence-20261102.json).

Deployment is a separate step: keep submissions paused while switching the matching
Flash controller, serving release, and Freesolo configuration. Verify production
checkpoint downloads and generation before reopening submissions. Dev SFT and
custom save/resume, exact checkpoint/optimizer restoration, downloads, and hosted
serving already passed before the production drain.

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
on all four shipping Modal profiles on 2026-11-02 for
`ghcr.io/freesolo-co/freesolo-flash-serve@sha256:9dc811c3458f979ce2e020509ffbbe3e51364f551234df4c3139ef8e497192dc`.
Model-specific image qualifications still apply independently. Older certified
digests contain the retired Hub hydrator and must never receive Azure credentials.

The live matrix covers Qwen3.5-9B (L40S), Qwen3.5-9B-Base (H100),
Qwen3.8-27B (H100), and Qwen3.6-35B-A3B (H200). The 27B and 35B tests
use fresh zero LoRA fixtures; the 35B fixture includes every fused expert layer.
Each test hydrated private Azure artifacts, generated a response, and confirmed
removal of its app, secrets, and volume. See
[the qualification evidence](azure-serving-qualification-20261102.json).
