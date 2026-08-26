# The Flash serving contract

`flash models deploy`, `chat`, `evaluate`, and `undeploy` use one public identity for every run-backed adapter:

- `<run_id>/final`
- `<run_id>/step-N`

The grammar is exact. Bare run ids, omitted checkpoints, padded steps, whitespace, extra path segments, and old composite identities are rejected. A run id is grouping and authorization metadata only.

## Permanent checkpoint binding

`POST /adapters` registers one exact checkpoint:

```json
{
  "adapter_id": "run-abc/step-10",
  "checkpoint": "run-abc/step-10",
  "run_id": "run-abc",
  "checkpoint_step": 10,
  "repo_id": "acme/artifacts",
  "repo_type": "dataset",
  "subfolder": "sft/run-abc/adapter",
  "artifact_revision": "<private-40-hex-source-commit>",
  "artifact_digest": "<private-64-hex-content-digest>",
  "artifact_fingerprint": "<private-64-hex-binding-fingerprint>",
  "base_model": "Qwen/Qwen3.5-9B",
  "lora_rank": 16,
  "org_id": "org-id",
  "thinking": false,
  "structured_outputs": null
}
```

The registration schema is closed. Lifecycle fields are server-owned. The first registration binds the org-scoped checkpoint to its private source location, content identity, base model, LoRA shape, and serving defaults. An identical retry is idempotent. Any changed immutable fact returns `409`, and undeploy never frees the identity for reuse.

Private artifact revision, digest, repository, and subfolder exist only for deterministic retrieval and cache integrity. They are not model identities and do not appear in public chat provenance, evaluation identities, analytics, or billing payloads.

## Authentication and health

Customer-owned deployments use `Authorization: Bearer <FLASH_SERVING_KEY>`. Freesolo-hosted serving uses `X-Freesolo-Internal-Key` for internal control operations and Freesolo API-key authorization for caller traffic. Credentials are compared without exposing them and are removed from cross-origin redirects.

`GET /healthz` does not load a model. Dynamic hosted serving advertises permanent checkpoint identity support. Customer-owned deployments receive an immutable manifest at boot and expose no dynamic registration surface.

## Readiness and routing

Registration returns `200` or `202`. `GET /adapters/{checkpoint_id}` reports lifecycle for that exact binding. A checkpoint becomes routable only after its exact adapter load succeeds and the generation-fenced durable row is promoted to `ready`.

Routing is direct. `POST /v1/chat/completions` requires the exact checkpoint id as `model`. There is no alias lookup, activation operation, current checkpoint, implicit final, first checkpoint, or sole-checkpoint fallback. Tenant authorization is checked against the checkpoint binding's owning organization before dispatch.

The packaged, hosted, and managed OpenAI entry points retain the shared strict sampling grammar:

- `n` is an exact integer from 1 through 4, and `n > 1` requires `temperature > 0`.
- `seed` is null or a signed 64-bit integer, excluding `-1`.
- frequency and presence penalties are finite values from -2 through 2.
- `logprobs` is a strict boolean and `top_logprobs` is an integer from 0 through 20.
- thinking-enabled adapters reject logprobs.

## Public provenance

Successful run-backed responses identify only the permanent checkpoint:

```json
{
  "freesolo": {
    "checkpoint_id": "run-abc/step-10"
  }
}
```

Hosted responses also send `X-Freesolo-Checkpoint`. Customer-owned responses expose `checkpoint_id` in `flash_provenance`. Stream and buffered responses must attest to the same checkpoint. Private source commits and digests are never returned.

## Undeploy and cleanup

`DELETE /adapters/{checkpoint_id}` disables exactly one checkpoint:

```json
{
  "ok": true,
  "checkpoint_id": "run-abc/step-10",
  "run_id": "run-abc",
  "disabled_checkpoints": ["run-abc/step-10"]
}
```

Sibling checkpoints remain independently ready and callable. Run-wide cleanup is a separately named internal administrative operation that first enumerates the run's exact checkpoint bindings.

## Customer-owned deployments

Manifest schema v2 keys adapters by checkpoint id. It keeps the private source commit, exact file table, and aggregate digest for hydration and cache verification, but publishes only checkpoint identities through `/v1/models` and chat provenance. Manifest and deployment identity v1 are rejected rather than translated.

Modal and persistent RunPod deployments retain their existing provider topology, lifecycle fencing, direct authenticated HTTPS endpoint, capacity validation, and teardown behavior.

## Cross-repository schema dependency

Flash expects the coordinated Freesolo migration to provide:

- `flash.hosted_lora_adapters` with explicit checkpoint and private binding columns;
- `public.bind_hosted_checkpoint(...)` for atomic insert-or-identical-read semantics;
- lifecycle-only mutable updates protected by immutable-field database enforcement;
- accounting RPCs accepting `checkpoint_id` and private `artifact_fingerprint` without legacy identity fields.

The old and new binaries are not compatible. Deploy them only through the documented maintenance-window cutover.
