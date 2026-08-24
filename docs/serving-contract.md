# The Flash serving contract

`flash models deploy`, `chat`, and `undeploy` use an HTTP serving backend. `flash/serve/` owns the
client, shared runtime, and customer-owned provider deployment. Freesolo's hosted backend
(`flash/serving/`) is another implementation of this contract.
A deployment serves one base model and its LoRA adapters. Use `flash serve deploy` to provision one
in your own Modal or RunPod account, or implement these endpoints and run the conformance suite
below. The maintained customer-owned profile supports only `Qwen/Qwen3.5-9B`; Qwen3.6 27B is
retired and Qwen3.8 27B is available only through the separately managed hosted-serving catalog.

## Identity model

A revision is immutable:

- `<run_id>@step-<N>.<40-hex-hf-revision>`
- `<run_id>@final.<40-hex-hf-revision>`
  It permanently identifies one repository, repository type, subfolder, base model, checkpoint,
  thinking mode, organization, structured-output policy, and provenance tuple.
  The bare `<run_id>` is a mutable alias. Registration loads a revision; activation atomically moves
  the alias. Chat uses the alias and must report the exact revision that answered.

## Authentication and health

Every protected request carries a credential the backend compares with `hmac.compare_digest`,
rejecting a mismatch with `401` or `403`. The client removes it on cross-origin redirects and keeps
it on same-origin redirects.
Which header carries it depends on who owns the deployment, because the two are trusted differently:

- Customer-owned deployments (`flash serve deploy`) use `Authorization: Bearer <FLASH_SERVING_KEY>`.
  The credential is scoped to that one deployment. `FREESOLO_INTERNAL_KEY` is a control-plane key
  and must never be sent to a customer-owned endpoint, so these backends do not read it at all.
- The Freesolo-hosted backend uses `X-Freesolo-Internal-Key`, which is meaningful only inside
  Freesolo's own infrastructure.

Run the conformance suite with the same credential the backend was deployed with.

The suite checks the dynamic-registration contract on this page, so it applies to the
Freesolo-hosted backend and custom backends that implement this registration surface. It does NOT
apply to a `flash serve deploy` deployment: that
app receives its adapters in an immutable manifest and registers them at boot, so it serves
`/healthz`, `/v1/models`, and `/v1/chat/completions` and has no `/adapters` surface at all. Pointing
the suite at one fails every registration, activation, and alias test against a perfectly healthy
deployment. Verify a customer-owned deployment through those three endpoints instead.
A custom dynamic-registration backend may intentionally be keyless; advertise that through
`requires_key: false`.
`GET /healthz` must answer without loading the model:

```json
{
  "ok": true,
  "base_models": ["Qwen/Qwen3.5-9B"],
  "requires_key": true,
  "capabilities": ["immutable_adapter_revisions", "alias_compare_and_swap"]
}
```

`capabilities` must be a JSON array of strings. Required capabilities are:

- `immutable_adapter_revisions`: one revision id always names the same artifact.
- `alias_compare_and_swap`: activation atomically rejects stale expectations.
  `revision_provenance` is preferred. Advertise
  `thinking_structured_outputs_deferred_v1` only if a grammar is deferred until after thinking.
  `requires_key: false` declares an open backend. If the field is true or absent, `flash serve status`
  probes an unknown adapter id with the configured key. An accepted key returns `404`.

## Registration and readiness

`POST /adapters` registers a revision:

```json
{
  "adapter_id": "run-abc@step-10.<40-hex>",
  "repo_id": "acme/artifacts",
  "repo_type": "dataset",
  "subfolder": "sft/run-abc/adapter",
  "base_model": "Qwen/Qwen3.5-9B",
  "checkpoint": "run-abc/step-10",
  "org_id": "org-id",
  "thinking": false,
  "structured_outputs": { "regex": "[ab]+" },
  "metadata": {
    "record_type": "revision",
    "run_id": "run-abc",
    "checkpoint_step": 10,
    "hf_revision": "<40-hex>"
  }
}
```

Registration is a closed schema: an unknown top-level field is rejected with `422`. The backend
owns `status` and `metadata.record_type`, so a client must omit them rather than send its own --
sending `status` is itself an unknown field and fails the whole request. Metadata must agree with
the revision id, and `hf_revision` must be a full commit sha.
Return `200` or `202`. An identical retry must succeed, including while the first load is pending.
Different identity under the same revision returns `409` or `422`.
`GET /adapters/{id}` returns a bare record or `{"adapter": {...}}`. Echo every identity field. The
client checks identity before lifecycle on every poll.
Lifecycle states are:

- `registered`: accepted and loading.
- `ready`: loaded and usable.
- `failed`: terminal; put the reason in `metadata.failure`.
  A disabled status is terminal. Do not report `ready` until the adapter has loaded successfully.
  Readiness uses a model-scaled 10 to 15 minute budget. Each request is capped at 60 seconds, transport
  errors and 5xx responses are retried, and `Retry-After` is clamped to two seconds. Loading must
  outlive the request that queued it.
  `structured_outputs` registered here is the revision's default, and is part of its immutable
  identity. Support `json`, `regex`, `choice`, and `json_object`, or reject unsupported forms
  during registration. A chat request may still override the default for its own call; see
  "Chat and provenance".

## Activation

`POST /adapters/{revision_id}/activate` receives:

```json
{ "expected_adapter_revision": "run-abc@step-5.<40-hex>" }
```

`null` means the alias must not exist. Compare and write atomically; return `409` when the expectation
is stale. On success return:

```json
{
  "adapter_id": "run-abc",
  "target_adapter_revision": "run-abc@step-10.<40-hex>",
  "previous_adapter_revision": null,
  "checkpoint": "run-abc/step-10",
  "updated_at": "2026-08-12T00:00:00Z"
}
```

The alias record must contain `metadata.alias_of` for response-loss reconciliation.

## Chat and provenance

`POST /v1/chat/completions` is OpenAI-compatible. `model` is the run alias. Unknown, disabled, or
unactivated aliases return `404` or `503`.
The registered grammar is the default for every call to that revision. A request may override it
per call with `structured_outputs`, or with the OpenAI-standard `response_format` accepted at this
endpoint only; `{}` (equivalently `response_format: {"type": "text"}`) means explicitly
unconstrained for that call and does not change what is registered. Requests may also include
`stop` and `chat_template_kwargs.enable_thinking`.
Every successful response reports the exact revision, checkpoint, and Hugging Face commit in:

- body object `freesolo` with `adapter_revision`, `checkpoint`, and `hf_revision`;
- headers `X-Freesolo-Adapter-Revision`, `X-Freesolo-Checkpoint`, and
  `X-Freesolo-HF-Revision`.
  Streaming uses `text/event-stream` with `data:` JSON frames containing
  `choices[0].delta.content`, followed by `data: [DONE]`. Preserve stop handling and provenance headers.
  A cold adapter may return `503` with a top-level retryable `adapter_unavailable` /
  `adapter_loading` error envelope and a `Retry-After` header.
  The shared runtime supports bounded multimodal preparation. An image-capable profile declares
  `image_limit`, loads a processor, and accepts image-bearing requests up to that limit; a text
  profile declares no `image_limit` and returns `400` for them rather than answering without seeing
  the image.

## Undeploy

`DELETE /adapters/{run_id}` disables the alias and all indexed revisions:

```json
{
  "run_id": "run-abc",
  "disabled_aliases": ["run-abc"],
  "disabled_revisions": ["run-abc@step-10.<40-hex>"]
}
```

An unknown run returns `404`. After success, records are disabled or absent and chat no longer
succeeds. The packaged serving app performs cold undeploy without starting the GPU, deletes only
`adapters/<digest>` cache paths, and returns a retryable failure if cache cleanup must be retried.
Warm residents are unloaded later using the exact revision incarnation.

## Conformance

Run against a dynamic-registration backend you own, because the suite registers, activates, chats,
and deletes real state. A `flash serve deploy` deployment is not a valid target: it has no
`/adapters` surface (see "Authentication and health" above).

```bash
uv run pytest tests/serving_conformance \
  --serving-url "$FREESOLO_SERVING_URL" \
  --conformance-repo acme/artifacts \
  --conformance-subfolder sft/run-abc/adapter \
  --conformance-base-model Qwen/Qwen3.5-9B \
  --conformance-hf-revision 8f2c1b0e5d4a39c7b6e2f014a8d35c9b7e10426f
```

Without `--serving-url`, live tests skip and offline oracle guards still run. Once a URL is supplied,
all adapter arguments and a real 40-character commit sha are mandatory. The default readiness
budget matches the shipped model-scaled client budget.
