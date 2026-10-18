# Hosted inference load testing

The `flash.serving.loadtest` package is an operator-only, public-HTTP load harness for the hosted
inference front door. It is not imported by production serving and it does not create, resize, inspect,
or destroy provider resources. Operators run it explicitly with the serving extra:

```bash
uv run --extra serving python -m flash.serving.loadtest validate scenario.json
uv run --extra serving python -m flash.serving.loadtest discover scenario.json
uv run --extra serving python -m flash.serving.loadtest run scenario.json results/fake-five-phase
uv run --extra serving python -m flash.serving.loadtest summarize results/fake-five-phase
uv run --extra serving python -m flash.serving.loadtest verify results/fake-five-phase
```

`validate` is allocation-free and performs no HTTP request. `discover` calls only `GET /healthz`.
`run` calls only `GET /healthz` and `POST /v1/chat/completions`. The credential value is read from the
environment variable named by `credential_env`; never put a credential value in a scenario file.
There are no request retries.

## Evidence boundary

The harness verifies exact deployment SHA and deployment id, required capabilities, required base
models, and immutable adapter response provenance. It checks health before and after every phase and
at the midpoint of each duration phase. Its evidence remains HTTP-only:

- it cannot prove that a live replica scaled out;
- it cannot prove that an engine was actually cold or at scale zero;
- it cannot distinguish a dispatch deadline from the underlying resource constraint;
- it cannot establish an availability SLA;
- fake and test runs cannot support production claims.

A `cold_burst` therefore records either `cold_scale_out_intent_http_unattested` or
`true_scale_zero_intent_http_unattested`. That is an honest statement of the authored intent, not an
attestation of provider state.

## Strict scenario

Scenarios are strict Pydantic v2 JSON. Unknown fields and fields that belong to another phase kind are
rejected. The authored `phases` list may contain any subset of the five supported discriminated kinds
in any order, except that a present `cold_burst` must be the first inference-producing phase. A complete
five-phase scenario uses `cold_burst`, `warm`, `sustained`, `mixed`, then `overload`.

Dynamic base-model discovery reads `base_models` from `/healthz`, then applies `include`, `exclude`, and
`require`. `require` is a hard gate.

Explicit adapter targets must name the immutable model id, base model, and checkpoint. The request
carries `X-Freesolo-Expected-Checkpoint` and the response must match `X-Freesolo-Checkpoint`.

`adapter_revision` and `hf_revision` are optional, because the hosted app does not emit them on every
deployment. Omitting one verifies checkpoint identity alone rather than passing a header check the
deployment never answered. Supplying one asserts that the deployment does emit it, so a response that
omits or changes it is a provenance mismatch, never a skipped assertion. A supplied `hf_revision` must
still be a canonical 40-character hub commit SHA.

`/healthz` is read for `ok`, deployment identity, capabilities, and `base_models`. `accounting_ok` is
optional: a deployment that does not report it is not treated as an accounting failure, while an
explicit `false` stops the run. Note that `/healthz.gpus` is the configured supported-base-model engine
count, not a live container count, so it can never be read as a replica count.

This fake example covers every phase kind:

```json
{
  "schema_version": 1,
  "name": "fake five-phase hosted inference",
  "endpoint": "https://example.invalid",
  "expected_deployment": {
    "sha": "94210a3",
    "deployment_id": "fake-deployment-1"
  },
  "credential_env": "FAKE_SERVING_TOKEN",
  "required_capabilities": ["permanent_checkpoint_identity"],
  "discovery": {
    "enabled": true,
    "include": ["model-a", "model-b", "model-c"],
    "exclude": ["model-b"],
    "require": ["model-a"]
  },
  "targets": [],
  "profiles": [
    {
      "name": "short",
      "weight": 1.0,
      "messages": [{ "role": "user", "content": "Return one short sentence." }],
      "max_tokens": 16,
      "temperature": 0.0,
      "top_p": 0.95
    }
  ],
  "client": {
    "connect_timeout_seconds": 5.0,
    "read_timeout_seconds": 30.0,
    "write_timeout_seconds": 10.0,
    "pool_timeout_seconds": 5.0,
    "max_in_flight": 8,
    "max_scheduling_lag_ms": 100.0
  },
  "seed": 17,
  "fake": true,
  "phases": [
    {
      "name": "cold",
      "kind": "cold_burst",
      "requests": 4,
      "burst_window_seconds": 0.2,
      "cold_intent": "cold_scale_out"
    },
    {
      "name": "warm",
      "kind": "warm",
      "requests": 4,
      "concurrency": 2
    },
    {
      "name": "sustained",
      "kind": "sustained",
      "duration_seconds": 2.0,
      "rate_rps": 2.0
    },
    {
      "name": "mixed",
      "kind": "mixed",
      "duration_seconds": 2.0,
      "rate_rps": 3.0
    },
    {
      "name": "overload",
      "kind": "overload",
      "expects_capacity_contract": false,
      "stages": [
        { "duration_seconds": 1.0, "rate_rps": 4.0 },
        { "duration_seconds": 1.0, "rate_rps": 8.0 }
      ]
    }
  ]
}
```

`warm` is the only closed-loop phase. Its authored request count is completed with bounded
concurrency. Every other phase is open-loop. Its dispatch offsets are computed before the phase
starts. If the client cannot admit an offset within `max_in_flight` and `max_scheduling_lag_ms`, the
harness writes a terminal `client_admission_missed` record instead of converting that offered request
into delayed closed-loop work. Overload stages use the same rule and never retry.

## Stream and artifact contract

Streaming requests ask for usage. Every SSE `data:` payload must be valid JSON, followed by exactly
one `[DONE]`; data after `[DONE]`, a missing terminal finish reason, or a missing `[DONE]` is a protocol
error. The first nonempty `content` or `reasoning_content` delta defines time to first generated token.
The first nonempty visible `content` delta is measured separately. Missing usage makes token throughput
unavailable for that request; the harness never estimates tokens from chunks.

Only HTTP 503 with error code `serving_capacity_unavailable` and `Retry-After: 1` is counted as exact
capacity rejection. Other 503 responses and 429 responses are separate classes.

An error body can echo the request back, so only a machine error code is read out of it. A code that
does not look like a slug is recorded as `unrecognized_code`, and a free-text `message` is never
persisted.

### Overload cannot be demonstrated without a capacity contract

Current Flash `dev` implements no retryable capacity contract. Its engine error mapping returns 504
for an engine timeout and 502 for other upstream failures; there is no capacity 503 and no
`Retry-After: 1`. An overload phase run against such a deployment therefore cannot produce a clean
rejection signal at all.

An overload phase declares this with `expects_capacity_contract`, which defaults to `false`. The flag
never changes how a response is classified. It only records whether an absence of capacity rejections
is an accurate description of the deployment or a failed expectation:

| observed                        | `expects_capacity_contract` | verdict                                               |
| ------------------------------- | --------------------------- | ----------------------------------------------------- |
| at least one exact capacity 503 | either                      | `capacity_rejection_observed`                         |
| none                            | `false`                     | `overload_not_demonstrated`                           |
| none                            | `true`                      | `overload_not_demonstrated_despite_declared_contract` |

**An absence of capacity rejections is never evidence of headroom.** A deployment without a capacity
contract sheds load in ways this harness cannot distinguish from ordinary failure over public HTTP.
When any overload phase declares no capacity contract, the summary carries that limitation explicitly.
No overload phase is ever summarized as a success on the strength of a quiet run.

Each run claims a new result directory and writes:

- `scenario.authored.json`, with prompt content redacted;
- `scenario.resolved.json`, with discovered targets and cold-intent labels;
- `events.jsonl`, through one event writer with one terminal row per scheduled request;
- `summary.json`, atomically;
- `complete.json`, last, with row counts, byte counts, and SHA-256 hashes.

An interrupted run closes the event stream but never writes `complete.json`. It remains inspectable and
`verify` rejects it as incomplete. Version 1 has no partial-phase resume.

Artifacts never contain credentials, authorization or request headers, generated text, raw response
bodies, or per-request prompts. Persisted response headers are limited to content type, retry-after,
and immutable provenance.

## Metrics

`summary.json` reports overall and per-phase, per-target, and per-profile values:

- scheduled, dispatched, completed, successful, and client-admission-missed counts;
- offered, dispatch, and completion rates with explicit denominator seconds and basis;
- scheduling, header, first-generated, first-visible, and total latency p50, p90, p95, p99, and max;
- usage coverage and token totals, aggregate output token rate, and decode token rate;
- peak client in-flight, HTTP status and error counts, retry count, exact capacity ratio, and 429 count;
- authored versus achieved target and profile mix;
- explicit sustained and mixed authored, scheduling, dispatch, and completion windows.

## B200 owner handoff

The harness does not allocate a B200 and does not attest provider state. The B200 deployment owner must
complete the provider-side handoff before an operator runs a non-fake scenario:

1. identify the already deployed endpoint, exact release SHA, deployment id, and intended base models;
2. confirm the B200 resource is owned, budgeted, and governed outside this harness;
3. decide whether the cold intent is `cold_scale_out` or `true_scale_zero`, and establish that state
   through the provider control plane without using the harness;
4. provide only the name of the existing credential environment variable in the scenario;
5. run `validate`, then `discover`, and confirm the resolved two-of-three or other intended model set;
6. run into a new result directory, then run `verify` before interpreting `summary.json`;
7. retain provider-side replica and hardware evidence separately if a scale-out or B200 claim is needed.

A successful HTTP load run is not provider-side proof of B200 identity, replica count, scale-to-zero,
or autoscaler behavior.
