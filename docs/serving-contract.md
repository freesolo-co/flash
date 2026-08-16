# The Flash serving contract

`flash models deploy`, `flash models chat`, and `flash models undeploy` talk to a multi-LoRA
serving backend over HTTP. Freesolo operates one at `serve.freesolo.co`; `flash/serve/` is the
client for it, and that client is in this repository even though the hosted backend is not.

This document is the interface between them. Anything that implements it works with the shipped
CLI unchanged - point `FREESOLO_SERVING_URL` at it and every deploy command works. The hosted
backend is one implementation of this contract, not the definition of it.

Two ways to get a backend:

- **`flash serve setup`** generates a Modal app from the backend template included in this
  repository and deploys it to your own Modal account. Start here; the rest of this document is why
  it does what it does.
- **Write your own** against this document, then prove it with the conformance suite at the
  bottom.

One generated app and URL serve exactly one base model. `GET /healthz` advertises that model, and
registration for any other base model is rejected before GPU allocation. Multiple base models
therefore require distinct generated apps and URLs, or an external gateway that routes each model
to its app. Freesolo's hosted multi-model backend remains external to this repository; the included
template is the self-hosted single-model implementation.

## Vocabulary

A **revision** is an immutable adapter id: `<run_id>@step-<N>.<hf_revision>` (or
`<run_id>@final.<hf_revision>`). It pins one artifact - a specific commit of a specific HuggingFace
repo subfolder - and it pins it forever.

An **alias** is the bare `<run_id>`. It is mutable, and it names whichever revision is currently
live for that run. Users chat with the alias; the backend resolves it to a revision.

That split is the whole design. A deploy registers a new revision, waits for it to load, and then
flips the alias to point at it. Traffic moves in one atomic step, and the previous revision stays
loaded and addressable the entire time.

## Authentication

Every request carries `X-Freesolo-Internal-Key`. On a standalone plane this is the plane's root
credential, so treat it as the key that controls your whole deployment, not a per-service token.

The client scopes it to one origin. `flash/serve/deploy.py` installs a request hook that deletes
the header from any request whose origin differs from the configured `FREESOLO_SERVING_URL`, on
every redirect hop. httpx strips only `Authorization` and `Cookie` across an origin change, so
without that hook a single 302 from your serving host would forward the plane credential to
whatever origin the redirect named.

Two consequences for a backend you operate: a same-origin redirect keeps the key (Modal's
async-result polls rely on this), and a cross-origin redirect silently arrives unauthenticated, so
do not build one into a request path and expect it to authenticate.

Compare the key in constant time (`hmac.compare_digest`). Reject with 401 when it does not match.

## Capabilities

`GET /healthz` returns a JSON object with a `capabilities` list. The client reads it before every
deploy and refuses to proceed unless the required strings are present.

It must be a **JSON array of strings**. An object mapping capability names to booleans carries the
same information but is rejected: the client requires a list and raises
`serving_contract_unsupported` on anything else, so every deploy fails before it starts.

| capability                                | status    | what it promises                                               |
| ----------------------------------------- | --------- | -------------------------------------------------------------- |
| `immutable_adapter_revisions`             | required  | one revision id maps to exactly one artifact, forever          |
| `alias_compare_and_swap`                  | required  | the alias flip is atomic and rejects a stale expectation       |
| `revision_provenance`                     | preferred | records echo back `run_id` / `checkpoint_step` / `hf_revision` |
| `thinking_structured_outputs_deferred_v1` | optional  | structured-output grammar applies only after `</think>`        |

The two required ones are safety-critical and the client hard-fails without them. Do not advertise
a capability you have not implemented - they are the only thing the client can check, so an
optimistic string is worse than an absent one.

`revision_provenance` is only consulted on the rare recovery path where registration returned 5xx
and the client re-reads the record to decide whether its registration actually landed. Its absence
is a warning, not an error.

`thinking_structured_outputs_deferred_v1` becomes **required** for a specific run: one deployed
with both thinking and structured outputs enabled. Applying a JSON grammar to a thinking model from
the first token constrains the reasoning block too, and the output is invalid. Withhold this string
unless the constraint is genuinely deferred until after the `</think>` marker; the client will then
refuse that combination rather than serve broken output.

## Endpoints

### `GET /healthz`

```json
{
  "ok": true,
  "requires_key": true,
  "capabilities": ["immutable_adapter_revisions", "alias_compare_and_swap"]
}
```

Must answer before the model is loaded. The client calls it during deploy preflight, and a cold
container that blocks health on a multi-minute weight load reads as an unreachable backend.

`requires_key` is optional and reports whether the backend authenticates writes at all - not
whether the caller's own key is correct. `flash serve status` follows up with an authenticated
`GET /adapters/{unknown-id}` and treats `401` or `403` as a misconfigured key. It skips that probe
only on an explicit `false`: because the field is optional, omitting it means "no claim", and
treating a missing field as "open" would report `ready` against a protected backend whose next
deploy 401s. Health itself may stay unauthenticated (the generated app leaves it open so a cold
backend is still diagnosable), which is exactly why status cannot conclude anything about the key
from this endpoint alone.

Either rejection code is accepted, so a backend that answers `403` on a bad key is diagnosed
correctly rather than reported as ready. `404` for the unknown id is the expected answer from an
accepted key.

### `POST /adapters`

Registers a revision. The client sends:

```json
{
  "adapter_id": "run-abc@step-10.<40-hex>",
  "repo_id": "acme/artifacts",
  "base_model": "Qwen/Qwen3.5-4B",
  "subfolder": "sft/run-abc/adapter",
  "repo_type": "dataset",
  "checkpoint": "run-abc/step-10",
  "metadata": {
    "record_type": "revision",
    "run_id": "run-abc",
    "checkpoint_step": 10,
    "hf_revision": "<40-hex>"
  },
  "thinking": false
}
```

`org_id` and `structured_outputs` appear when set. Accept `200` or `202`; anything else is treated
as a failure.

`structured_outputs` is the run's declared output grammar, and it is **the record's, not the
request's**: it is decided at registration, stored, and applied to every completion. A chat request
must never be able to loosen or replace it.

Its value is a JSON object of vLLM `StructuredOutputsParams` kwargs. Flash sends exactly four
constraint forms (`flash/content/structured_outputs.py:CONSTRAINT_KEYS`), one per registration:

| key           | value                                | meaning                                      |
| ------------- | ------------------------------------ | -------------------------------------------- |
| `json`        | a JSON Schema object (or its string) | output must validate against the schema      |
| `regex`       | a regex string                       | output must match the pattern                |
| `choice`      | a list of strings                    | output must be exactly one of them           |
| `json_object` | `true`                               | output must be some syntactically valid JSON |

Other `StructuredOutputsParams` kwargs (such as `whitespace_pattern`) may accompany the constraint
key. A backend that recognizes only one form silently serves unconstrained text for the other three:
the conformance suite exercises `regex`, so passing conformance does **not** prove the other forms
work. Implement all four, or reject a registration carrying a form you do not support at
registration time rather than at first chat.

Five rules:

- **Do not reject unknown top-level fields.** The client adds keys over time.
- **Do not require a `status` field.** The client deliberately does not send one, because the
  backend owns the record's status. A schema that demands it 422s every deploy.
- **Re-registration of identical content must succeed**, because the client retries after an
  ambiguous 5xx. Re-registration of _different_ content under the same `adapter_id` must be
  `409`. That conflict is what `immutable_adapter_revisions` means. "Content" includes the
  provenance metadata, not just the artifact location: the client cross-checks `run_id`,
  `checkpoint_step`, and `hf_revision` on its recovery readback, so a comparison that ignores them
  accepts a changed record as an unchanged retry.
- **`metadata` must agree with the `adapter_id`.** The id already encodes `run_id`,
  `checkpoint_step`, and `hf_revision`, so a backend should parse it and reject a payload whose
  metadata contradicts it with `422`, rather than storing both. `hf_revision` in particular must be
  a full 40-character commit sha: it is the value the backend downloads, so a mutable ref like
  `main` lets one immutable id serve different weights over time while every content check still
  sees the same string. A disagreeing `run_id` is worse than cosmetic - it files the revision under
  a different run's alias, so undeploying that run reports success while the revision keeps serving.
- **`metadata.record_type` belongs to the backend, not the caller.** This endpoint registers
  revisions, so a backend should write `"revision"` itself rather than trusting the field. A record
  stored as an alias loads but can never be activated, and undeploy will not evict or clean it up.

Registration may return before the adapter has loaded. `202` plus a `registered` lifecycle state is
the expected shape for a backend that loads in the background.

### `GET /adapters/{id}`

Returns the authoritative record, for a revision or an alias. Either the record itself or
`{"adapter": {...}}` - the client accepts both.

```json
{
  "adapter": {
    "adapter_id": "run-abc@step-10.<40-hex>",
    "repo_id": "acme/artifacts",
    "repo_type": "dataset",
    "subfolder": "sft/run-abc/adapter",
    "base_model": "Qwen/Qwen3.5-4B",
    "checkpoint": "run-abc/step-10",
    "thinking": false,
    "status": "active",
    "metadata": {
      "record_type": "revision",
      "lifecycle_state": "ready",
      "...": "..."
    }
  }
}
```

`404` means no such record, and the client treats it as "not registered yet" while polling.

**Echo the registration fields back unchanged.** After an ambiguous 5xx the client re-reads the
record and compares `adapter_id`, `repo_id`, `repo_type`, `subfolder`, `base_model`, `checkpoint`,
`thinking`, `org_id`, and `structured_outputs` field by field. A record that drops or renames one
looks like a _different_ artifact under the same revision id, which the client reports as an
immutability violation and refuses to deploy.

While a record is still loading, set `Retry-After` on the response. The client prefers it over its
own exponential backoff, but **clamps it to 2 seconds** (`READBACK_MAX_DELAY_SECONDS`), and clamps
its own backoff to the same ceiling. So `Retry-After: 30` buys you a poll every 2 seconds, not every
30: the header can only ever slow the client down to that ceiling, never past it. Size your
readiness path for a 2-second poll for the whole load, and do not rely on the header to shed load.

### Lifecycle

`metadata.lifecycle_state` (or top-level `lifecycle_state`) drives the client's readiness wait:

```
registered ──> ready
           └─> failed
```

- `registered` - accepted, not yet loaded. The client keeps polling.
- `ready` - loaded and serving. The client proceeds to activation.
- `failed` - the load failed. Put the reason in `metadata.failure`; the client surfaces it verbatim.
  `status: "disabled"` is treated the same way.

**Fail at registration, not at first chat.** A backend that reports `ready` optimistically and only
discovers the adapter is unloadable when a user chats turns a clear deploy-time error into a
mystery 500 long after the deploy claimed success. Validate by actually loading the adapter.

The client gives up after its readiness budget and reports the last state it saw, leaving the
previous alias serving. A revision that never reaches `ready` never takes traffic.

That budget is **5 minutes** (`REVISION_READY_BUDGET_SECONDS`), and on a scale-to-zero backend it
is easy to exceed without anything being wrong: the first deploy after an idle period pays a cold
start, and for a GDN-hybrid model that is a multi-minute `torch.compile` before the engine can
load anything. Two things follow. Return `Retry-After` while loading, so the wait is paced by your
estimate rather than the client's backoff. And do the load somewhere that outlives the request:
if the work is a fire-and-forget task on a web container, an autoscaler can take the container
away mid-load and the record sits at `registered` forever, which the client can only report as a
timeout.

### `POST /adapters/{revision_id}/activate`

The compare-and-swap. Body:

```json
{ "expected_adapter_revision": "run-abc@step-5.<40-hex>" }
```

`null` means "the alias does not exist yet". Commit only if the alias currently targets exactly
`expected_adapter_revision`; otherwise `409`. Accepting a stale expectation lets two concurrent
deploys of one run silently overwrite each other, which is precisely what
`alias_compare_and_swap` promises cannot happen.

The check and the write must be atomic. A read-then-write with a gap between them is not a
compare-and-swap, no matter what `/healthz` says.

**What the generated Modal app actually guarantees.** One CPU-only lifecycle coordinator is pinned
to one container and serialized to one input. Registration, settlement, activation, and removal all
pass through it, so the expectation check and alias write cannot overlap another durable lifecycle
mutation. `modal.Dict` remains the durable record store across cold starts; the API and GPU engine
read it but never write lifecycle state.

A custom backend may use any primitive that provides the same atomic compare-and-swap guarantee,
such as a database row lock or Redis transaction. The endpoint contract above is unchanged.

On success, return all five fields - the client validates every one and fails the deploy on any
mismatch:

```json
{
  "adapter_id": "run-abc",
  "target_adapter_revision": "run-abc@step-10.<40-hex>",
  "previous_adapter_revision": null,
  "checkpoint": "run-abc/step-10",
  "updated_at": "2026-08-12T00:00:00Z"
}
```

`updated_at` must be a non-empty string.

The alias record itself carries `metadata.alias_of` pointing at the live revision. The client reads
it directly to reconcile an activation whose response was lost in flight, so it must be present and
correct even when the response never arrived.

### `DELETE /adapters/{id}`

Disables the alias and every revision under it. Returns:

```json
{
  "run_id": "run-abc",
  "disabled_aliases": ["run-abc"],
  "disabled_revisions": ["run-abc@step-10.<40-hex>"]
}
```

`run_id` must match what was requested, and both lists must be lists of strings. `404` for an
unknown run is normal - the client maps it to "nothing to undeploy", not an error.

The response is a report, not the state change itself. The client performs no read-back, so a
backend that returns these lists without disabling anything makes `flash models undeploy` report
success over a run that keeps serving and keeps billing. After this call the alias and every
revision it named must read back `disabled` (or be gone), and chat through the alias must stop
succeeding.

Disabling is a state change on the record, not necessarily an engine eviction. The contract is that
the alias stops resolving; reclaiming GPU memory is your implementation's business.

The generated Modal app disables the alias and revisions before deleting their digest-scoped cache
directories from the volume. `DELETE` never starts or calls the GPU engine, so an undeploy remains
GPU-free when the app is cold. A cache cleanup error leaves every record disabled and returns a
retryable failure; repeating the same `DELETE` retries cleanup without reviving the run.

### `POST /v1/chat/completions`

OpenAI-compatible. The `model` field carries the **alias** (`run-abc`), and the backend resolves it
to whichever revision the alias currently targets. Requests also arrive with
`chat_template_kwargs: {"enable_thinking": bool}` and optionally `stop`.

Return a normal OpenAI chat completion carrying provenance in **both** a body object and three
response headers. These were documented as optional, and they are not: a managed plane runs a
deployment smoke against every deploy, and `_smoke_provenance`
(`flash/server/routes/serving_smoke.py`) rejects the completion unless all of it is present and
exact. A backend that returns an otherwise valid completion without them fails every deploy at the
smoke step.

A `freesolo` object alongside the OpenAI fields:

```json
{
  "choices": [
    { "index": 0, "message": { "role": "assistant", "content": "..." } }
  ],
  "freesolo": {
    "adapter_revision": "run-abc@step-10.<40-hex>",
    "checkpoint": "run-abc/step-10",
    "hf_revision": "<40-hex>"
  }
}
```

and the same three values as headers: `X-Freesolo-Adapter-Revision`, `X-Freesolo-Checkpoint`, and
`X-Freesolo-HF-Revision`.

All three must describe the revision that actually generated the response - the one the alias
resolved to, not the alias itself. The smoke compares the headers as an exact set, so sending a
subset, or extra `X-Freesolo-*` provenance headers, fails it. `hf_revision` is the segment after
the final `.` in the revision id.

Standalone use of `flash models chat` does not check any of this, so a backend that omits it still
answers chat. It is deployment through a managed plane that requires it.

An unknown or not-yet-activated model is `404` or `503`.

#### Streaming

`flash models chat` and `flash env eval` both send `"stream": true` on every request, so a backend
that implements only the one-shot JSON above satisfies the deployment smoke and still produces no
usable CLI chat. Streaming is required for those two commands, not optional.

The client branches on **response `Content-Type`**, which is what makes a partial implementation
safe to grow into: answer `application/json` and it reads the whole body and takes the one-shot
`choices[0].message.content` path above, ignoring `stream` entirely. Answer anything else and it
decodes the body as server-sent events.

The SSE shape it decodes, line by line:

```
data: {"choices": [{"delta": {"content": "Hel"}}]}

data: {"choices": [{"delta": {"content": "lo"}}]}

data: [DONE]
```

- Only lines beginning with `data:` are read; anything else (including SSE `event:` and comment
  lines) is skipped, so blank-line framing and keep-alives are fine.
- Each payload is a JSON object whose `choices[]` carry `delta`. Text comes from `delta.content`.
- `data: [DONE]` ends the stream. Without it the client stops at end-of-body, so the sentinel is
  what distinguishes a finished answer from a truncated one.
- A thinking model may put its reasoning on `delta.reasoning_content` instead; the client re-wraps
  that into a `<think>` block so the streamed text matches the one-shot string. A backend that
  streams reasoning inline inside `content` also works.

Provenance headers are sent on the streaming response exactly as on the one-shot one - they are
headers, so they arrive before the first chunk either way.

#### The cold-start envelope

A scale-to-zero backend will be asked for a completion while the container is still booting. Deploy
smoke recognizes one specific 503 body and retries within its deadline instead of failing the
deploy:

```json
{
  "error": {
    "type": "adapter_unavailable",
    "code": "adapter_loading",
    "retryable": true,
    "requested_model": "run-abc",
    "adapter_revision": "run-abc",
    "retry_after_seconds": 2
  }
}
```

Every field is checked; `requested_model` and `adapter_revision` both echo the requested `model`.
Send `Retry-After` as a header too - the client prefers it.

This envelope must sit at the **top level** of the response body. A framework that wraps a raised
error as `{"detail": {...}}` does not match, and the retry silently never happens.

## Conformance suite

`tests/serving_conformance/` runs this document against a live backend. It registers a revision,
waits for `ready`, activates it, chats through the alias, checks that a stale compare-and-swap is
rejected and that a mutated re-registration conflicts, then undeploys.

Four of its checks go past the happy path, because those are the ones a plausible-looking backend
fails: it re-registers under one revision id with **each** identity-bearing field changed in turn
(not just `subfolder`), activates two revisions of one run **concurrently** and requires exactly
one winner, **reads back** the records after undeploy rather than trusting the response, and drives
a `structured_outputs` registration through to a real completion and matches the output against the
grammar.

The adapter it registers is real and must exist - point it at one your backend can load:

```bash
uv run pytest tests/serving_conformance \
  --serving-url "$FREESOLO_SERVING_URL" \
  --conformance-repo acme/artifacts \
  --conformance-subfolder sft/run-abc/adapter \
  --conformance-base-model Qwen/Qwen3.5-4B \
  --conformance-hf-revision 8f2c1b0e5d4a39c7b6e2f014a8d35c9b7e10426f
```

Without `--serving-url` the whole suite skips, so it costs nothing in the offline run. With
`--serving-url` but no adapter, it FAILS rather than skipping: at that point only `/healthz` and the
404 checks can run, and a green exit there would prove none of registration, readiness, activation,
chat, or teardown.

`--conformance-hf-revision` is required and must be a real 40-character commit sha. It is both the
commit the backend downloads and the suffix of the immutable revision id, so a branch name or a
placeholder would pass the id grammar and then fail minutes later, on the GPU, as an unresolvable
revision rather than as the missing argument it is.

`--conformance-ready-timeout` defaults to 300 seconds because that is the budget `flash models
deploy` itself enforces (`REVISION_READY_BUDGET_SECONDS`). A backend that only reaches `ready` after
longer than this is one the shipped client cannot drive, so the suite would otherwise pass a target
that fails in real use. Raise it deliberately if a first cold start has to pull weights over a slow
link, but know that a pass above 300s no longer means "works with flash unchanged".

It writes: registering, activating, and deleting adapters are all real state changes on the target.
Run it against a backend you own.

`tests/test_serving_backend_contract.py` asserts the same contract against the app that
`flash serve setup` generates, with the GPU stubbed out, so the offline suite covers the shape and
the conformance suite covers a real deployment.
