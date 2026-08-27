# The Flash serving contract

`flash models deploy`, `chat`, `evaluate`, and `undeploy` use an HTTP serving backend and one public
identity for every run-backed adapter. `flash/serve/` owns the client, shared runtime, and
customer-owned provider deployment. Freesolo's hosted backend (`flash/serving/`) is another
implementation of this contract. A deployment serves one exact base model and its compatible LoRA
adapters. A different base model requires a separate deployment. Use
`flash serve deploy --provider modal` to provision one in your own Modal account.

Customer-owned Modal is live-qualified for `Qwen/Qwen3.5-9B`, `Qwen/Qwen3.8-27B`, and
`Qwen/Qwen3.6-35B-A3B`. The 27B and 35B-A3B qualifications are bound to the exact certified serving
image digest; another digest remains available for offline planning but cannot allocate Modal.
Hosted Qwen3.8 27B is active on H100. Its customer-owned engine runs on `H100!` and serves the pinned
`Qwen/Qwen3.8-27B-FP8` checkpoint while preserving `Qwen/Qwen3.8-27B` as the distinct logical base
and tokenizer provenance. The 35B-A3B engine serves BF16 weights on one H200 with FP8 KV cache, a
32K context, eight sequences, a 4096 batched-token cap, and six rank-64 LoRA slots.

Customer-owned `flash serve` is Modal-only and requires the explicit provider argument. Historical
customer-serving RunPod deployment identities are unsupported, with no migration, status, undeploy,
or teardown shim. Managed RunPod training remains supported and unchanged. Retired Qwen3.6 27B is
never translated into Qwen3.8 27B.

Run-backed adapters use only:

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

Function tools are implemented and offline-validated only for the exact qualified Qwen3.5 packaged
and hosted profile. This path is not live-qualified pending exact model testing. The parser identity
is `qwen3_coder`; Qwen3.6 and unqualified engines reject tools after authorization and checkpoint
resolution, before generation. Declarations must be closed function objects with unique nonempty
names. Parameters use a bounded root-object JSON Schema profile with `properties`, `required`,
`additionalProperties: false`, recursive scalar, object, and array types, descriptions, enums, and
array `items`. Numeric enum members must be JSON integers; decimal and exponent numeric enum lexemes
are rejected because Flash cannot preserve them exactly across every ingress and template boundary.
Generated and historical JSON numeric literals support at most 1024 significand digits; oversized
generated candidates remain exact text and oversized history is rejected. A historical integral
value's expanded native template result is also capped at 1024 digits. Historical nonintegral
exponent values are supported only when they convert to a finite, non-underflowing native template
number. These local serving-contract bounds do not depend on Python's process-wide integer
conversion limit. Unsupported keywords and `strict: true` are rejected.

`tool_choice` defaults to `auto` and accepts only `auto` or `none`. `parallel_tool_calls` defaults to
`true` and accepts only exact `true`; either control requires `tools`. Tools cannot be combined with
thinking, logprobs, structured outputs, a non-text `response_format`, or image messages. Historical
assistant calls require unique IDs, function type, valid names, and JSON-string object arguments.
Each immediately following tool-result turn must resolve every declared call exactly once before a
non-tool turn. External messages remain unchanged; only a detached template copy converts argument
strings to objects.

In `auto` mode, complete schema-valid Qwen3 Coder XML candidates become OpenAI `tool_calls` with
independent call indexes and `finish_reason: "tool_calls"`. Parsed responses contain no raw Qwen
tags. Ordinary text and malformed or incomplete candidates remain exact assistant text with the
native finish reason. Candidate regions are buffered in raw SSE because argument deltas cannot be
retracted safely. Native prompt, completion, cached, and reasoning token accounting remains
authoritative and is never recomputed from serialized arguments. Flash parses calls but does not
execute or resubmit them.

The decoded convenience `chat_stream` iterator remains text-only and single-choice. It rejects
`n != 1`, logprobs, and tools before opening transport; use buffered JSON or raw SSE for those
features. Raw `/generate` and `/adapters/{id}/generate` remain tool-free. Buffered hosted responses
and raw hosted SSE are authoritative; managed serving forwards them and never reparses tool calls.

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

Modal deployments retain their existing provider topology, lifecycle fencing, direct authenticated HTTPS endpoint, capacity validation, and teardown behavior.

## Cross-repository schema dependency

Flash expects the coordinated Freesolo migration to provide:

- `flash.hosted_lora_adapters` with explicit checkpoint and private binding columns;
- `public.bind_hosted_checkpoint(...)` for atomic insert-or-identical-read semantics;
- lifecycle-only mutable updates protected by immutable-field database enforcement;
- accounting RPCs accepting `checkpoint_id` and private `artifact_fingerprint` without legacy identity fields.

The old and new binaries are not compatible. Deploy them only through the documented maintenance-window cutover.
