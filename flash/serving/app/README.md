# Freesolo LoRA Serving Modal App

Multi-LoRA serving on Modal with **one warm exact-model engine class per active base model**.

- Each active base model has its own immutable `LoraEngine_<model>_<contract>` Modal class. Its
  identity covers the exact model and every decorator-affecting traffic policy value. The class's
  vLLM engine holds all adapters for that model, so many adapters share the model's GPU capacity.
- `router` is a CPU web endpoint that tracks adapter to base-model routing and dispatches each
  request to the exact model class. Each router replica has finite concurrency with `max_inputs=36`
  and `target_inputs=27`, independent of the number of engine replicas.

Adding a LoRA validates the exact immutable revision on one engine replica before the durable row is
made routable. That is not global residency. Any other replica lazy-loads and attests the same revision
before generation, using a replica-local mutable final directory while the immutable Hugging Face,
base-weight, and compile caches remain shared. Modal owns engine queueing and replica creation.
Uncataloged base models are rejected instead of falling onto another model or GPU tier.

Each CPU router and GPU engine function uses `min_containers=1` and `buffer_containers=1`, with no Flash-configured maximum.
The current two-model catalog therefore has a warm floor of 2 GPU containers
plus one requested buffer per model. Each engine replica keeps `max_inputs=8` and
`target_inputs=6`. Modal workspace and platform limits still bound actual scaling.
The product chat route allows 30 minutes and gives first-party serving a 1,700-second request budget,
leaving frontend headroom. Adapter undeploy returns after durable routing state is disabled. That
persisted fence is the cross-replica correctness boundary: new requests refresh authoritative state
before dispatch, while requests already in flight may finish. A single Modal unregister reaches only
one arbitrary replica, so physical LoRA cleanup is replica-local and opportunistic, not fleet-wide.
The `start_all` entrypoint (`uv run
modal run flash/serving/app/modal_app.py`) remains an explicit manual diagnostic that checks the warm
engines and blocks until each reports healthy. Pass `--base-model ...` to check one model.

The routing layer (`src/router.py`) carries no `modal`/`vllm` imports and is exhaustively
unit-tested in `tests/serving/test_router.py` (multi-base-model dispatch, shared-GPU multi-LoRA,
register-routing, OpenAI shape, auth, 404s) — run with `uv run pytest tests/serving/`.

Persistence is only for hydration and recovery. On startup each base model's engine loads
_its_ ready adapters from the `hosted_lora_adapters` Supabase table; `POST /adapters`
registers a new adapter on its base model's engine and tracks it in the router. Durable request usage
accounting is mandatory for inference. `/healthz` reports `accounting_ok`; an unhealthy accounting
outbox makes readiness return 503 and inference fails closed before GPU dispatch.

## Deployment environments

Production and development use the same `APP_NAME` and `HF_CACHE_VOLUME_NAME`; Modal environment
isolation namespaces both. `SERVING_DEPLOYMENT_MODE` identifies the deployment as `production` or
`development` and defaults to `production`. The app validates that development mode targets the actual
Modal environment `dev` and that production mode does not. The mode does not change autoscaling or
engine behavior.

Both environments keep one warm container and request one buffer container for every advertised
model and for each CPU router function. Flash omits `max_containers`, so Modal owns queueing and
replica scaling subject to workspace and platform limits. Each engine replica admits at most eight
inputs and targets six before scale-out. Each router replica admits at most 36 inputs and targets 27;
those router values do not change with engine fleet size. Flash does not maintain a process-local
application queue or use function statistics to authorize or deny a request. The router authors one
private 120-second absolute pre-header deadline and forwards it unchanged to the engine. Non-streaming
enqueue and result waits consume the remaining budget. The current legacy generator transport bounds
only its first remote event; after that event, normal streaming is governed by the request timeout.
The worker checks the same deadline before adapter hydration and immediately before GPU generation, so
abandoned queued work cannot start late. Modal resource exhaustion and pre-header expiry return
retryable `503 serving_capacity_unavailable` responses with `Retry-After: 1`; caller rate or quota
limits remain the only reason for a 429.

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

Explicitly check one warm development model:

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
  leaves enough KV headroom, `disable_log_stats`, the prompt-token cache (2048), per-engine
  `max_inputs=8` packing, multi-LoRA defaults (**16 hot** / **rank 32** / 256 CPU) with per-model rank overrides
  (**128** for the 9B, **64** for the 27B; the 35B MoE runs rank 64 at only 6 hot slots,
  every other tier keeps 16), and `VLLM_CACHE_ROOT` on the persistent volume so vLLM's
  torch.compile cache survives container replacement instead of recompiling on every cold start.
- **Deleted (neutral or losing):** speculative decoding / ngram (−13.5% on diverse output),
  bitsandbytes (−49%), `enforce_eager` toggle, the scheduler knobs (`max_num_batched_tokens`,
  `max_num_seqs`, partial-prefill, async-scheduling, scheduling-policy, gdn-backend,
  scheduler-delay — neutral globally; only real-GPU-validated per-model caps remain in
  `model_config.py`), `performance_mode`, CUDA-graph-size, `stream_interval`, `specialize_active_lora`.
- **Quantization: pre-quantized FP8 for the active dense tier, bf16 for the 35B MoE (memory-first).**
  The active 9B uses Freesolo's compressed-tensors checkpoint with no online quantization. Qwen3.8-27B
  is a **pending hosted candidate**, not an active hosted model. Its exact official E4M3 checkpoint
  remains pinned at revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, with `[128, 128]` weight
  blocks and the logical BF16 tokenizer/processor pinned to
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, so the canary metadata remains testable without
  advertising or routing the model. The 35B-A3B vision-language MoE is the
  **exception**: it serves the **official bf16
  base weights** on an H200 (`quantization=None`), because vLLM's fused-MoE LoRA path cannot run under
  FP8 at the shape a full all-expert flash adapter needs (see the GPU tiers † footnote below).

  vLLM auto-detects each checkpoint's declared quantization, so the engine passes no online
  `quantization`. `settings.KV_CACHE_DTYPE="fp8"` still halves the KV cache (E4M3, uncalibrated
  dynamic scales — `calculate_kv_scales` stays OFF: warmup scales corrupt the Qwen3 GDN-hybrid).
  ⚠ `serve_model_id` is pointed only at checkpoints **verified to exist** (a missing repo
  404-crash-loops, which is why the old GPTQ `serve_model_id` was removed); the owned `Freesolo-Co/*-FP8`
  checkpoints are VL-preserving (each VL-unified base's vision tower + MoE router gates stay in
  original precision) and already published to the operator HF org. Native FP8 tensor cores need
  compute capability ≥ 8.9; every dense FP8 tier runs on an L4, L40S, or H100 (all native). No serving
  tier runs on A100 now that the 35B is bf16 on H200. The pending Qwen3.8 candidate allocates no
  hosted tier until its exact canary passes.

- **The real memory lever — LoRA buffers:** vLLM PRE-ALLOCATES the GPU LoRA buffers at `max_loras ×
max_lora_rank`, regardless of how many adapters load. Both are linear levers. The qualified 9B runs
  **16 × 128** on L40S. Qwen3.8-27B's H100 LoRA capacity remains pending exact-checkpoint
  qualification. The 35B MoE runs **6 × 64** bf16 on H200. Adapters trained above the effective rank are rejected at load; a base regularly serving
  more distinct adapters than its hot limit pays a swap latency from the CPU pool unless its hot
  limit equals `MAX_CPU_LORAS`.

> The base model is no longer a single app-wide setting — it comes from each adapter's
> `base_model`, and Modal provisions a GPU per distinct value. `HOSTING_BASE_MODEL` is
> unused in the multi-base-model app.

### GPU tiers

Quantization (FP8 weights + KV for the dense tiers, bf16 for the 35B MoE) plus per-model LoRA buffer
sizing determine the GPU tier. Tiers live in `src/model_config.py`; only cataloged bases are accepted,
and a catalog entry with no explicit GPU uses `DEFAULT_GPU` (L4). Every tier needs a one-time real-GPU
cold-boot smoke test first.

| Base model                | Checkpoint                                   | GPU       | Hot LoRA shape | Context |
| ------------------------- | -------------------------------------------- | --------- | -------------- | ------- |
| Qwen3.5-9B                | owned FP8 (`Freesolo-Co/*-FP8`)              | **L40S**§ | 16 × 128       | 32768   |
| Qwen3.8-27B candidate     | official native FP8 (`Qwen/Qwen3.8-27B-FP8`) | inactive‡ | pending        | pending |
| Qwen3.6-35B-A3B (MoE, VL) | official **bf16** (`Qwen/Qwen3.6-35B-A3B`)   | **H200**† | 6 × 64         | 32768   |

§ 9B (L40S) runs **16 × 128** at 32k context with `max_num_seqs=8`, CUDA graphs
(`enforce_eager=False`), and `gpu_memory_utilization=0.90`. Its rank-128 × 16 buffer OOMed an L4 and
2×L4 in the real-GPU sweep, so it runs on the 48 GiB L40S, the cheapest Ada card that fits it at 32k.

‡ Qwen3.8-27B is not in active hosted routing, registration, base-model health, or engine deployment.
Its proposed H100 tier, hot LoRA capacity, context limit, CUDA-graph behavior, throughput, and latency
remain pending qualification on the exact immutable official FP8 checkpoint. Qwen3.6-27B measurements
do not qualify or predict the Qwen3.8 checkpoint.

† 35B-A3B runs **bf16 on an H200** (141 GiB) at **6 × 64**, the one serving path that gives a flash
adapter its full all-expert LoRA AND CUDA graphs at speed. It gets rank 64 like the 27B
tier but at only **6 hot slots** (`6 × 64`, not `16 × 64`): its fused-MoE LoRA buffer scales with
`max_loras × rank × num_experts`, so `16 × 64` (~140 GiB) is unreachable on a single GPU. Why bf16 and
not the FP8 checkpoint every other tier uses:

- **FP8 on A100** materializes the fused-MoE experts back to bf16 (~76 GiB on the 80 GiB card),
  leaving no room for CUDA-graph capture, so it is forced eager at ~4-10 tok/s.
- **FP8 on H200/B200** fails outright: the fused-MoE LoRA kernel dies with `Unsupported lhs dtype
fp8e4nv` (only A100's Marlin kernel runs this MoE's LoRA), so full-expert LoRA will not even load.
- **bf16 on H200** sidesteps the fp8 kernel: ~107 GiB weights+LoRA + ~17 GiB KV + graph capture all
  fit the 141 GiB card, every expert gets its LoRA and graphs (~129 tok/s single-stream, canaried
  2026-07-06). `gpu_memory_utilization=0.90`, `max_model_len=32768` (32k; the real-GPU canary held a
  ~679,701-token KV cache at 6 × 64, ~20× concurrency, where 8 × 64 overflowed and fit only ~19k),
  `max_num_batched_tokens=4096`, `max_num_seqs=8`, `enforce_eager=False`.

The vision encoder is loaded (no `language_model_only`) so flash adapters' vision-tower LoRA keys bind.
There is no A100 fallback and no expert-exclusion "fast profile" knob: one faithful path.

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

`POST /adapters`:

```json
{
  "adapter_id": "people-search@step-20.8f2c1b0e5d4a39c7b6e2f014a8d35c9b7e10426f",
  "repo_id": "Freesolo-Co/people-search-lora",
  "repo_type": "model",
  "base_model": "Qwen/Qwen3.5-9B",
  "checkpoint": "people-search/step-20",
  "org_id": "org-id",
  "private": true,
  "thinking": false,
  "metadata": {
    "record_type": "revision",
    "run_id": "people-search",
    "checkpoint_step": 20,
    "hf_revision": "8f2c1b0e5d4a39c7b6e2f014a8d35c9b7e10426f"
  }
}
```

`POST /generate`:

```json
{
  "adapter_id": "people-search-lora",
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
