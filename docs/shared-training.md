# Experimental shared LoRA training

`flash.training.shared` is an opt-in, operator-owned training service. Multiple
authenticated sessions keep separate LoRA weights, optimizer state, training RNG,
policy versions, and checkpoints while sharing one frozen base model and GPU
execution thread. SFT, GRPO, and OPD have token-based client recipe helpers.

This is a new backend, not a replacement for the deployed verl workers. Ordinary
`flash train` jobs still use their existing execution path. The service is not yet
connected to Flash run orchestration, hosted authentication, environment execution,
deployment artifacts, or customer billing. Keep it experimental until those
integrations and their end-to-end tests exist.

## Execution and isolation

One command executes at a time. Selecting an adapter clears gradients and selects
that adapter's optimizer and RNG. Training commands perform a complete update;
gradients never accumulate across commands. The base parameters remain frozen.
Sampling uses the same model, full-distribution temperature-1 sampling, and an
independent request seed. There is no retained KV cache across adapter switches.

The operator selects a model at an immutable Hugging Face commit, common LoRA
targets/rank/alpha, and capacity limits. Requests cannot select model code, local
paths, tenant identities, Python callbacks, or credentials. Reward and teacher
callbacks run in the client process. Every session read, mutation, and checkpoint
restore requires the same authenticated tenant and owner that created it. All
owners within a tenant can inspect that tenant's usage metadata, but not each
other's session results or checkpoints.

This process boundary isolates logical training state, not arbitrary tenant code:
never execute uploaded environments or reward code in the trainer. A device or
unexpected backend failure stops the entire pool and marks resident sessions lost.
Admission limits bound accepted work; they do not prove every model/context/adapter
configuration will fit VRAM. Operators must qualify their selected configuration.

## Fair scheduling and admission

SQLite transactions reserve adapter slots, request slots, and tokens before work
is accepted, including when several API connections submit simultaneously. Limits
apply to outstanding work (queued **and running**). A sampling request reserves
prompt length plus its maximum completion, not just the tokens ultimately emitted.
Training reserves all input tokens, including masked prompt tokens.

The scheduler selects tenants in round-robin order, then their oldest command.
Several API keys or sessions belonging to one tenant do not buy extra turns.
Request/context limits bound one tenant's individual command size. This prevents
starvation among admitted tenants while commands finish; it does not promise equal
GPU time, preemption, or a latency SLO. Sampling and checkpoint commands can have
very different costs from training commands. Closing remains admissible at quota
so clients can release resident adapters. Close follows previously accepted work;
queued non-lifecycle work can be cancelled to expedite it. Running work cannot be
cancelled through the API.

There is no cross-client physical batching (#10), adapter eviction, automatic
session expiration, separate rollout pool, or asynchronous policy lag. Clients must
close idle sessions. Admission can reject new sessions while existing ones hold
all adapter slots.

## Attribution

Each executed command has exactly one physical usage row, keyed by its command ID.
Retries using the same idempotency key return the original command rather than
performing or charging the work twice. Reusing a key for different work is rejected.
Usage includes tenant, session, operation, queue wait, execution duration, reserved
tokens, actual input/loss/output tokens where available, and execution outcome.
Adapter load/release timestamps describe residency; an interrupted lifetime is
marked uncertain rather than assigning an invented release time.

Execution time is synchronized elapsed wall time around the backend operation,
including associated CPU work; it is not a GPU-kernel profiler measurement. Startup,
observed idle gaps, and shutdown are unattributed pool overhead. The operator's
`Journal.pool_usage()` reconciles tenant execution totals plus overhead to observed
pool time. If configured, `rate_microusd_per_hour` estimates resource cost from that
time using an exact integer numerator and common denominator. Sum those rational
amounts before monetary rounding. The rate is immutable for a journal.

These are resource estimates, **not invoices or customer charges**. Provider
provisioning, unobserved time after a crash, reserved idle capacity, teacher APIs,
storage, and network charges are outside this accounting boundary. A command
interrupted before a durable receipt has unknown execution time, explicitly
reported as such; it is never reported as a measured zero-cost execution.

## Run an experimental pool

Install the `shared-training` extra on the GPU host. It is separate from
`serve-runtime`, which requires a different PyTorch version:

```sh
pip install 'freesolo-flash[shared-training]'
python -m flash.training.shared.server --config pool.json --state /durable/shared-pool
```

For source qualification, install the extra from this checkout instead of the
published package. Supply `FLASH_SHARED_CREDENTIALS` through the operator's secret
manager as a JSON object mapping random bearer tokens (at least 32 characters) to
`{"tenant":"organization-id","owner":"owner-id"}`. Tenant/owner IDs should be
stable across credential rotation. Several credentials can map to the same owner.
There is no caller-supplied identity override.

An example **small-context qualification** configuration:

```json
{
  "model": "Qwen/Qwen3.5-9B",
  "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
  "device": "cuda",
  "rank": 8,
  "alpha": 16,
  "targets": ["q_proj", "v_proj"],
  "limits": {
    "adapters": 4,
    "adapters_per_tenant": 2,
    "context_tokens": 128,
    "request_tokens": 512
  }
}
```

Use a durable local filesystem for the journal and checkpoints, one server worker
per state directory, and a TLS reverse proxy for remote clients. The default bind
is loopback only. The client refuses non-HTTPS endpoints except loopback and refuses
redirects. The filesystem lock prevents two runtimes from recovering/executing the
same journal. Changing the model/runtime configuration requires a new directory.
An operator must manage disk capacity and retention; automatic journal/checkpoint
compaction is not implemented.

## Client recipes

```python
from flash.training.shared.client import Client
from flash.training.shared.recipes import sft_step, grpo_step, opd_step

client = Client("https://operator-pool.example", token)
session = client.create("sft", key="durable-job-key", seed=7, learning_rate=1e-4)
try:
    result = sft_step(session, [(prompt_token_ids, completion_token_ids)])
    checkpoint = session.checkpoint()
finally:
    session.close()
```

Tokenize with the pool model's pinned tokenizer and the intended chat template.
`sft_step` masks prompts and averages over completion tokens. `grpo_step` samples
a group per prompt, calls `reward(prompt_ids, completion_ids)`, subtracts the group
mean without standard-deviation normalization, and uses a clipped objective with
fixed maximum-completion normalization. It rejects batches with no reward variation.
`opd_step` samples the student and calls `teacher(prompt_ids, completion_ids)` for
aligned group IDs and teacher log-probability sums. Every token in an alignment
group must carry the same group logsum; group -1 excludes unaligned tokens. The
objective is the sampled reverse-KL surrogate, averaged over aligned tokens then
sequences. The caller owns tokenizer alignment and the external teacher transport.

All sampling/training requests name the exact resident policy version; stale
requests fail before updating weights. These are text-token recipes, not full
parity with Flash's hosted environment recipes. Multimodal inputs, multi-turn
tools, structured generation, distributed training, alternate sampling settings,
the production loss/configuration matrix, and vLLM rollout conformance remain
unqualified here. Native autoregressive sampling recomputes context without a KV
cache and is intended for correctness qualification, not throughput claims.

For recoverable submission, use `session.submit(operation, payload, key=stable_key)`
and persist the returned ID before calling `session.wait(id)`. Repeat a timed-out
submission with the same key and identical payload. High-level recipe helpers
generate fresh keys; rerunning an entire helper is not a transport retry protocol.
Only one coordinator should submit updates for a session.

## Checkpoints and recovery

Checkpoint completion means adapter weights, AdamW state, RNG, version, and model
identity have been written, fsynced, atomically renamed, and durably receipted.
They are internal pool checkpoints, not deployable Hugging Face adapter exports.
A restore references an owned, completed checkpoint by source session and version;
clients cannot provide filesystem paths or pickle files.

On restart, all previous resident sessions become lost, queued work is cancelled,
and executing work becomes unknown. Ambiguous optimizer steps are never replayed.
Create a fresh session with the same algorithm, seed, and learning rate, then call
`fresh.restore(source_session_id, saved_version)`. Continue from that explicit
version. Retrying an old creation key returns its old lost/closed session; recovery
requires a new key. If no checkpoint was completed, restart training from scratch.

## Qualification

`tests/test_shared_training*.py` cover concurrent admission, tenant fairness,
ownership, idempotency, cancellation, crash recovery, attribution, real tensor
isolation, recipe calls over ASGI, and checkpoint continuation. The dedicated CI
workflow installs real CPU PyTorch/PEFT on Python 3.11 and 3.12; missing tensor
dependencies cannot silently skip that job.

An explicitly paid, bounded GPU qualification uses an owned Modal sandbox and
terminates it in `finally`:

```sh
python scripts/qualify_shared_training.py --run-paid-test --output fixture.json
python scripts/qualify_shared_training.py --run-paid-test \
  --model-id Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --output qwen.json
```

The fixture run uses an L4; the model run uses an H100. Actual-model qualification
checks updates, peer adapter/optimizer isolation, exact checkpoint continuation,
and unchanged base parameters for all three objectives. These short runs do not
establish long-context capacity, training quality, throughput, production billing,
or end-to-end managed `flash train` behavior.
