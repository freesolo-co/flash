# Freesolo LoRA Serving Modal App

Multi-LoRA serving on Modal with **demand-driven GPU containers per base model**.

- `LoraEngine` is a Modal class **parametrized by `base_model`**. Modal runs separate GPU
  containers per distinct base model, each with a vLLM engine that holds all adapters for that
  model. Many adapters of the same base model share its GPU capacity.
- `router` is a CPU web endpoint that tracks adapter to base-model routing and dispatches each
  request to the correct `LoraEngine`. It keeps one CPU container warm so it can receive the
  request that starts a scaled-to-zero GPU engine.

Adding a LoRA calls `register.remote.aio(...)` on that base model's engine, so adapter deployment
starts the matching GPU container when none is running. Inference similarly starts the engine
through its remote generation method. Uncataloged base models are rejected instead of falling
onto the default L4 tier.

GPU engines use `MIN_CONTAINERS = 0` and scale down after up to 35 idle minutes. A normal deploy does
not boot any GPU model; registration or inference starts the matching engine on demand. The product
chat route allows 30 minutes and gives
first-party serving a 1,700-second request budget, leaving frontend headroom. Adapter undeploy
returns after durable routing state is disabled, while
best-effort gpu eviction continues as response background work. The `start_all` entrypoint (`uv run
modal run flash/serving/app/modal_app.py`) remains an explicit manual diagnostic that boots engines and blocks until
each reports healthy. Pass `--base-model ...` to check one model.

The routing layer (`src/router.py`) carries no `modal`/`vllm` imports and is exhaustively
unit-tested in `tests/serving/test_router.py` (multi-base-model dispatch, shared-GPU multi-LoRA,
register-routing, OpenAI shape, auth, 404s) — run with `uv run pytest tests/serving/`.

Operator load tests use the isolated `flash.serving.loadtest` package, documented in
[`docs/hosted-inference-load-testing.md`](../../../docs/hosted-inference-load-testing.md). It is
never imported by production serving. It drives only `GET /healthz` and `POST /v1/chat/completions`,
manages no Modal lifecycle, and cannot by itself prove replica scale-out, cold state, GPU identity,
the cause of a capacity rejection, or an availability SLA.

Persistence is only for hydration and recovery. On startup each base model's engine loads
_its_ ready adapters from the `hosted_lora_adapters` Supabase table; `POST /adapters`
registers a new adapter on its base model's engine and tracks it in the router.

## Deployment environments

Production and development use the same `APP_NAME` and `HF_CACHE_VOLUME_NAME`; Modal environment
isolation namespaces both. `SERVING_DEPLOYMENT_MODE` identifies the deployment as `production` or
`development` and defaults to `production`. The app validates that development mode targets the actual
Modal environment `dev` and that production mode does not. The mode does not change autoscaling or
engine behavior.

Both environments keep `MIN_CONTAINERS = 0`, so every GPU engine scales to zero and starts on demand.
GPU autoscaling is capped at `MAX_CONTAINERS = 4` per model, including its active buffer.
`base_model` is a `modal.parameter()`, so each model has its own container pool. With three models,
the configured limits total 12 GPU containers per environment. Requests queue when a model's four
containers are busy; an idle model's capacity cannot be borrowed by another model.

The CPU router remains warm and uncapped. `BUFFER_CONTAINERS = 1` applies to the router and every
engine, but only while active, so GPU engines still scale to zero when idle. Idle windows are set by
`SCALEDOWN_WINDOW_SECONDS_BY_GPU`; the active B200 tiers use 2100 seconds. The router uses
`MAX_INPUTS = 64` with `TARGET_INPUTS = 48`; engine concurrency follows each model's `max_num_seqs`.

### Production

Production uses the default Modal environment, production Supabase, the production platform backend,
and the custom domain `https://serve.freesolo.co`. The production workflow remains
`.github/workflows/deploy-modal.yml` and deploys without an environment flag:

```bash
# from the repo root: modal's CLI puts the working directory on sys.path, and modal_app.py
# imports `flash.serving.src...`, so deploying from anywhere else fails at import.
uv sync --extra serving
set -a && source .env && set +a
export FREESOLO_DEPLOYMENT_ID="manual-production-$(git rev-parse --short=12 HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"
uv run modal deploy flash/serving/app/modal_app.py
```

Required production wiring is `HF_TOKEN`, `FREESOLO_INTERNAL_KEY`,
`PLATFORM_BACKEND_URL`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, a nonblank
`FREESOLO_DEPLOYMENT_ID`, and `SERVING_CUSTOM_DOMAIN=serve.freesolo.co`.

Optional capacity observations (queue wait, streaming time to first token, replica
concurrency, boot duration, and first-request status) persist under
`attestation_evidence.capacity` in the usage outbox and settled usage row. They do
not extend the billing RPC's strict top-level field contract. Missing observations
are omitted; checkpoint identity and token accounting remain required.

#### Promotion gate

Set the repository Actions variable `SERVING_CANARY_ORG_ID` to an existing Freesolo-owned
organization UUID with billing configured. The canary sends it as `X-Freesolo-Org-Id`; its
usage is attributed and charged to that organization, never to an invented tenant or an
arbitrary customer. Missing or malformed UUIDs fail before deployment; operators must also
verify that the organization exists and can settle usage before enabling the workflow.
The streaming probe uses the shared deadlines in `timeouts.py`: the full 2700-second engine
startup allowance plus 600 seconds for the request and 60 seconds of network headroom (3360
seconds total). The server and probe derive their limits from the same constants. The job allows
90 minutes including rollback.
Rollback resolves the serving entrypoint from the restored tree, including the pre-reorganization
`flash/serving/modal_app.py` path.

`modal deploy` replaces the live app in place, and a `/healthz` poll only reads back the identity
the deploy step just injected. That proves a router process booted; it does not prove a GPU engine
started, that a generation ran, that streaming works, or that usage settled. So after readiness the
workflow runs `python -m flash.serving.promotion.gate`, which must prove all three in order:

1. the live router reports `ok`, this exact `deployment_sha`, this run's attempt id, and at least
   one engine;
2. an authenticated `stream=true` request carrying `X-Correlation-ID: fspromo-<run>-<attempt>`
   returns SSE with at least one non-empty content delta, a terminal finish reason, terminal usage
   with non-zero completion tokens, and `data: [DONE]`;
3. this release's durable delivery loop is running rather than wedged: no expired lease, and no
   undelivered row older than the stall threshold.

Exact-head binding comes from steps 1 and 2 together. The router proves its identity, and then that
same router serves a real generation; engines receive only `HF_TOKEN` and cannot report a release
sha, and `engine_replica_id` is stripped from public bodies, so the response itself cannot carry it.

Step 3 is deliberately weaker than "the canary's own usage row settled". The only durable accounting
surface the router exposes is `serving_usage_backlog_snapshot`, which aggregates every generation in
flight and has no per-correlation read. Asserting a drained backlog there would be **fail-open** (a
burst of unrelated traffic can drain the counters to zero while the canary's own row is stuck) and
flaky besides (live traffic keeps them nonzero on a perfectly healthy release). Expired leases and
undelivered age are the parts of that snapshot that mean the same thing regardless of whose rows are
in it. Proving per-request settlement would need a read-by-correlation-id RPC, which does not exist
today; the canary still sends `X-Correlation-ID` so the row is identifiable by an operator.

If any step after the deploy fails, the workflow redeploys the sha the live app reported _before_
this deploy and re-reads `/healthz` to confirm the restored release under a distinct `-rollback`
attempt id. The job still exits non-zero: a restored predecessor means production is serving, not
that this commit shipped.

There is **no ingress fence**. Traffic is not gated during the canary window, so a small amount of
live traffic can reach a release that is about to be rolled back. Closing that window needs a
runtime admission gate and a race-free drain (a request admitted just before closure creates no
durable row before GPU dispatch, so a zero-backlog check cannot see it), which is deliberately not
part of this mechanism.

### Development

Development uses the dedicated Modal environment `dev`, development Supabase,
`PLATFORM_BACKEND_URL=https://api-dev.freesolo.co`, and the custom domain
`https://serve-dev.freesolo.co`. The generated Modal endpoint
`https://freesolo-dev--freesolo-lora-serving.modal.run` remains active alongside the custom domain and
is checked by the deployment workflow only to prove that the router belongs to the `dev` Modal
environment. Development backend, Flash, readiness checks, and user traffic use the custom domain.

The dev workflow `.github/workflows/deploy-modal-dev.yml` creates the Modal environment idempotently,
sets its web suffix to `dev`, reads the repository secret `SUPABASE_SERVICE_ROLE_KEY_DEV` and forwards it
as `SUPABASE_SERVICE_ROLE_KEY`, verifies `flash.hosted_lora_adapters`, and deploys only with `--env dev`.

The custom domain requires one-time setup before the first deployment:

1. In Modal workspace settings, add `serve-dev.freesolo.co` under Custom Domains.
2. Add the exact DNS records Modal displays to Cloudflare. The expected primary record is a DNS-only
   CNAME from `serve-dev` to `cname.modal.domains`; add any Modal-provided TXT record exactly.
3. Keep Cloudflare proxying and CNAME flattening disabled for the verification record.
4. Validate the domain in Modal and wait until it reports ready. Modal provisions and renews TLS.

For a manual development deploy, export the same development credentials and wiring first. Development
mode intentionally ignores the repository root `.env`, so production values cannot fill a missing dev
variable. It also requires the production and development Supabase refs to be present and different:

```bash
# from the repo root, for the same reason as the production block above.
uv sync --extra serving
uv run modal environment create dev  # once, if absent
uv run modal environment update dev --set-web-suffix dev
export SERVING_DEPLOYMENT_MODE=development
export SERVING_CUSTOM_DOMAIN=serve-dev.freesolo.co
export PLATFORM_BACKEND_URL=https://api-dev.freesolo.co
export SUPABASE_PROJECT_REF="replace-with-production-project-ref"
export SUPABASE_PROJECT_REF_DEV="replace-with-development-project-ref"
export SUPABASE_URL=https://${SUPABASE_PROJECT_REF_DEV}.supabase.co
export SUPABASE_SERVICE_ROLE_KEY="replace-with-development-server-key"
export FREESOLO_INTERNAL_KEY="replace-with-shared-internal-key"
export HF_TOKEN="replace-with-hugging-face-token"
export FREESOLO_DEPLOYMENT_ID="manual-development-$(git rev-parse --short=12 HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"
uv run modal deploy --env dev flash/serving/app/modal_app.py
```

Explicitly warm one development model without changing the zero floor:

```bash
uv run modal run --env dev flash/serving/app/modal_app.py --base-model Qwen/Qwen3.5-9B
```

### No autoscaling or vLLM perf knobs

There are **no env knobs for autoscaling or vLLM/engine performance**. An optimization either exists
(baked into `modal_app.py` / `src/settings.py` as a constant) or it was deleted — measured on real
GPUs, so there's nothing to tune at deploy time:

- **On, hardcoded:** FP8 weights on the dense tiers + FP8 KV cache everywhere (see Quantization
  below), prefix caching (~4.7×
  throughput / ~10× lower TTFT for shared prompts), CUDA graphs where the per-model boot canary
  leaves enough KV headroom, `disable_log_stats`, the prompt-token cache (2048), 64 concurrent router inputs and 8 per engine (autoscaling target 6), multi-LoRA defaults (**16 hot** / **rank 32** / 256 CPU) with per-model rank overrides
  (**128** for the 9B, **64** for the 27B; the 35B MoE runs rank 64 at only 6 hot slots,
  every other tier keeps 16), and `VLLM_CACHE_ROOT` on the persistent volume so vLLM's
  torch.compile cache survives scale-to-zero instead of recompiling on every cold start.
- **Deleted (neutral or losing):** speculative decoding / ngram (−13.5% on diverse output),
  bitsandbytes (−49%), `enforce_eager` toggle, the scheduler knobs (`max_num_batched_tokens`,
  `max_num_seqs`, partial-prefill, async-scheduling, scheduling-policy, gdn-backend,
  scheduler-delay — neutral globally; only real-GPU-validated per-model caps remain in
  `model_config.py`), `performance_mode`, CUDA-graph-size, `stream_interval`, `specialize_active_lora`.
- **Quantization: pre-quantized FP8 for the active dense tiers, bf16 for the 35B MoE (memory-first).**
  The active 9B uses Freesolo's compressed-tensors checkpoint with no online quantization. Qwen3.8-27B
  is also routed by the hosted catalog. Its official E4M3 checkpoint
  is pinned at revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, with `[128, 128]` weight
  blocks and the logical BF16 tokenizer/processor pinned to
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The 35B-A3B vision-language MoE is the
  **exception**: it serves the **official bf16
  base weights** on a B200 (`quantization=None`), because vLLM's fused-MoE LoRA path cannot run under
  FP8 at the shape a full all-expert flash adapter needs.

  vLLM auto-detects each checkpoint's declared quantization, so the engine passes no online
  `quantization`. `settings.KV_CACHE_DTYPE="fp8"` still halves the KV cache (E4M3, uncalibrated
  dynamic scales — `calculate_kv_scales` stays OFF: warmup scales corrupt the Qwen3 GDN-hybrid).
  ⚠ `serve_model_id` is pointed only at checkpoints **verified to exist** (a missing repo
  404-crash-loops, which is why the old GPTQ `serve_model_id` was removed); the owned `Freesolo-Co/*-FP8`
  checkpoints are VL-preserving (each VL-unified base's vision tower + MoE router gates stay in
  original precision) and already published to the operator HF org. Native FP8 tensor cores need
  compute capability ≥ 8.9; all active hosted engines run on B200.

- **The real memory lever — LoRA buffers:** vLLM PRE-ALLOCATES the GPU LoRA buffers at `max_loras ×
max_lora_rank`, regardless of how many adapters load. Both are linear levers. The qualified 9B runs
  **16 × 128** and the 35B MoE runs **6 × 64** bf16, both on B200. The hosted 27B uses **16 × 64**. Adapters trained above the effective rank are rejected at load; a base regularly serving
  more distinct adapters than its hot limit pays a swap latency from the CPU pool unless its hot
  limit equals `MAX_CPU_LORAS`.

> The base model is no longer a single app-wide setting — it comes from each adapter's
> `base_model`, and Modal provisions a GPU per distinct value. `HOSTING_BASE_MODEL` is
> unused in the multi-base-model app.

### Hosted GPU configuration

All active hosted engines use B200 with a 2100-second idle window. Each catalog entry must
explicitly declare its GPU; unsupported hardware is rejected before deployment. Training GPU
selection and customer-owned serving profiles are separate.

| Base model | Checkpoint | GPU | Hot LoRA shape | Context |
| --- | --- | --- | --- | --- |
| Qwen3.5-9B | owned FP8 (`Freesolo-Co/*-FP8`) | B200 | 16 × 128 | 32768 |
| Qwen3.8-27B | official native FP8 (`Qwen/Qwen3.8-27B-FP8`) | B200 | 16 × 64 | 32768 |
| Qwen3.6-35B-A3B (MoE, VL) | official bf16 | B200 | 6 × 64 | 32768 |

CUDA graphs, prefix caching, FP8 KV, and the persistent compilation cache remain enabled.
The 35B keeps bf16 weights because the FP8 fused-MoE kernel cannot load full-expert LoRA.
The vision encoder remains loaded so adapters' vision-tower keys bind.

The router accepts 64 concurrent inputs; each GPU engine accepts 8 with an autoscaling target
of 6. The router limit is not the number of requests packed onto one GPU.
Adapters are registered and loaded on demand, using immutable source identities and the
persistent disk cache. Hosted engines do not hydrate the database at boot, preload adapters,
or pin adapters against LRU eviction.

### Owned checkpoints

The `Freesolo-Co/*-FP8` checkpoints are pre-produced (VL-preserving FP8) and already published to the
operator HF org; serving consumes them directly via `serve_model_id`. vLLM auto-detects the
compressed-tensors quantization at load, so no serving-image change is needed to consume a checkpoint.
The one-off quantization job that produced them is not kept in this repo.

## Endpoints

- `GET /healthz` — `{ok, base_models: [...], unsupported_base_models?: [...], gpus: <count>, gpu_by_model: {base: tier}, gpu_tiers: [...], adapters: <count>}` (`gpu*` fields cover supported catalog base models; unsupported hydrated models are listed separately)
- `GET /adapters`
- `POST /adapters` with `X-Freesolo-Internal-Key` and an immutable revision registration (optional `structured_outputs` default — see below)
- `POST /generate` with `adapter_id`
- `POST /adapters/{adapter_id}/generate`
- `POST /v1/chat/completions` where `model` is the adapter id

### Checkpoint headers (all three inference endpoints)

Concurrent checkpoint deploys mutate the single serving surface for an adapter
id (a redeploy overwrites the record's checkpoint), so every inference endpoint
(`/generate`, `/adapters/{adapter_id}/generate`, `/v1/chat/completions`,
streaming included) supports a checkpoint contract:

- **`X-Freesolo-Checkpoint` (response header)** — echoes the checkpoint that
  actually served the request, using the public `RUN_ID` or `RUN_ID/step-N`
  form. The header is **omitted entirely when the adapter has no checkpoint**.
- **`X-Freesolo-Expected-Checkpoint` (request header)** — pins the request to
  an expected checkpoint. If the adapter is serving a different checkpoint
  (e.g. a parallel sweep re-deployed another step between deploy and eval), the
  request fails with **409 Conflict** instead of silently generating from the
  wrong checkpoint. The value is compared after stripping whitespace; send an
  empty value to require "no checkpoint".

Serving accepts caller chat messages, including system-role messages, from
trusted platform sampling paths. It does not persist those prompts. Prompt
rendering still forces `enable_thinking` from the adapter's stored training
default, so callers cannot flip a thinking/no-thinking adapter at sampling time.

### Structured outputs (guided decoding)

Every inference endpoint supports structured outputs (vLLM guided decoding via
`StructuredOutputsParams`) through these request forms:

- **`structured_outputs`** (all endpoints) — takes a
  canonical constraint dict with **exactly one** of `json` (a JSON schema, or a raw JSON-schema
  dict/string directly), `regex`, `choice` (list of strings), or `json_object: true`, plus
  optional `disable_any_whitespace`, `disable_additional_properties`, `whitespace_pattern`.
  Convenience aliases: `json_schema` / `schema` → `json`, `choices` → `choice`; a bare
  JSON-schema dict (e.g. `{"type": "object", ...}`) is accepted directly.
- **OpenAI `response_format`** (on `/v1/chat/completions` only) — the standard OpenAI shapes are
  accepted so an OpenAI-SDK client works unchanged: `{"type": "json_object"}`, `{"type":
"json_schema", "json_schema": {"schema": {...}}}` (or the flattened `{"type": "json_schema",
"schema": {...}}`), and `{"type": "text"}` (unconstrained). Translated to the canonical form at
  that endpoint; `structured_outputs` takes precedence when both are sent.

The spec normalizes to the canonical form at the router (a bad spec is a 422 naming the problem),
and streaming works unchanged. Structured outputs are entirely optional: omit the field to
generate freeform, unconstrained text.

**Per-adapter default.** `POST /adapters` accepts an optional `structured_outputs` field (same
forms) that is persisted with the record and applied whenever a request doesn't carry its own
spec — e.g. register a JSON-only extraction adapter once and every client gets schema-constrained
output. A per-call spec replaces the default for that call, and an explicit "off" value (`false`,
`""`, `"none"`, or `{}`) disables it for that call — so a caller can always opt back out and
generate unconstrained.

**Thinking adapters.** Serving enables vLLM's `qwen3` reasoning parser engine-wide. Prefer
chat/messages requests because serving renders the required reasoning markers automatically. For a
non-thinking adapter, the rendered prompt contains a closed empty reasoning block, so the
structured-output grammar is active from the first generated token. For a thinking adapter, the
rendered prompt leaves an open `<think>` block; generation must emit `</think>` before the grammar
activates. Raw `prompt` callers must provide the mode-specific state themselves: a non-thinking raw
prompt must already contain a closing `</think>` marker, while a thinking raw prompt must leave an
open `<think>` block for generation to close.

Internal Python code uses `checkpoint_id` for permanent checkpoint bindings and `model_id`
where a selector can also name a catalog base model. The existing HTTP/Modal JSON spelling
`adapter_id` remains a serialization alias; this naming cleanup does not change API payloads
or database columns.

`artifact_digest` verifies the artifact files. `artifact_fingerprint` identifies their immutable
binding to an organization, checkpoint, source revision, base model and serving configuration,
including the content digest. The same files can have the same digest but different binding
fingerprints. Fingerprint serialization retains its original keys so stored fingerprints remain
valid; neither hash is an authentication credential.

`POST /adapters`:

```json
{
  "adapter_id": "people-search/step-20",
  "repo_id": "Freesolo-Co/people-search-lora",
  "repo_type": "model",
  "base_model": "Qwen/Qwen3.5-9B",
  "checkpoint": "people-search/step-20",
  "org_id": "org-id",
  "private": true,
  "thinking": false,
  "run_id": "people-search",
  "checkpoint_step": 20,
  "artifact_revision": "8f2c1b0e5d4a39c7b6e2f014a8d35c9b7e10426f",
  "artifact_digest": "<private-64-hex-content-digest>",
  "artifact_fingerprint": "<private-64-hex-binding-fingerprint>",
  "lora_rank": 16
}
```

`POST /generate`:

```json
{
  "adapter_id": "people-search/step-20",
  "messages": [
    { "role": "user", "content": "Find senior search engineers in SF" }
  ],
  "max_tokens": 512,
  "temperature": 0,
  "structured_outputs": {
    "json": {
      "type": "object",
      "properties": {
        "names": { "type": "array", "items": { "type": "string" } }
      }
    }
  }
}
```
