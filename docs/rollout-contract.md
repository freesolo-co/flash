# Rollout contract V1

Flash's GRPO path now checks a versioned request/result contract at generation,
episode assembly, and training ingestion. The existing reward identity ledger
still checks exact step membership and completeness. These checks are complementary:
membership alone cannot establish which weights generated a token or what its
probability means.

The standard-library contract lives in `flash/engine/worker/train/core/rollout/`.
The first production adapter is synchronous GRPO with the pinned upstream Verl
and vLLM runtime. This does not enable a remote rollout service or change OPD.

## Identity

Every generation request carries:

| Field | Meaning |
| --- | --- |
| `protocol_version` | Wire semantics; V1 rejects unknown versions and fields. |
| `policy.run_id` | Fresh execution lineage generated in the parent launch config. A restart/resume gets a new lineage. |
| `policy.base_revision` | Immutable 40-character commit from the resolved HF snapshot path. Mutable model references are rejected. |
| `policy.tokenizer_digest` | SHA256 of the serialized tokenizer backend and special-token mapping, excluding transient padding/truncation settings. |
| `policy.template_digest` | SHA256 of tokenizer/processor templates and chat-template options. |
| `policy.policy_version` | Version acknowledged by the server after weight synchronization; distinct from optimizer step and protocol version. |
| `policy.adapter_revision` | Immutable in-memory snapshot name, `<lineage>/policy/<version>`. This is a synchronized snapshot identity, not a weight-content hash or an exported adapter repository revision. |
| `optimizer_step`, `sample_index`, `rollout_ordinal` | Logical training membership, preserving the existing ledger's meaning. |
| `attempt_id` | Fresh UUID for one episode execution, shared across its turns. |
| `turn_index`, `request_id` | Ordered turn and unique generation request within the attempt. |
| `prompt_digest` | SHA256 of the exact token prefix sent for this generation. |
| `logprob_definition` | Explicit probability semantics, checked against the engine and trainer. |

In synchronous GRPO, optimizer step N consumes policy version N−1. A different
policy version is rejected; V1 has no implicit policy-lag allowance. The trainer
also checks the full policy identity, not just its numeric version.

The adapter uses upstream's client extension point to preserve request IDs through
load balancing. Upstream's default client replaces those IDs.

## Token and probability semantics

All offsets index response tokens after the prompt, with half-open `[start, end)`
turn boundaries. Each logprob is the natural log of the returned token's conditional
probability at that position, recorded during generation. Missing, shorter, longer,
nonfinite, or positive logprob vectors are rejected before episode assembly.

V1 distinguishes the pinned runtime's two probability modes:

- `raw_logprobs`: log-softmax of logits at sampler entry, before sampler-level
  temperature, penalties and filtering. Masks applied before sampler entry are
  already reflected; this is not a promise of an unconstrained model distribution.
- `processed_logprobs`: log-softmax after sampling transforms, including
  temperature, penalties and support filtering. This adapter requires stochastic
  sampling (temperature at least `1e-5`) for this mode, since the pinned sampler's
  greedy path does not return the probability of the deterministic decision.

The definitions follow the pinned [vLLM sampler](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/v1/sample/sampler.py)
and [top-k/top-p sampler](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/v1/sample/ops/topk_topp_sampler.py).
An engine with different semantics must implement an explicit conversion with the
necessary evidence or reject the request. A matching field name is insufficient.
Numerical parity across inference kernels is not established by this protocol.

Results include effective engine sampling parameters after default resolution and
context-budget clamping, including defaults that vLLM normally omits when
serializing. The existing training recipe selects the probability mode; the
contract does not silently substitute one distribution for another.

Before issuing a successful result receipt, the server also checks the final
`RequestOutput.prompt_token_ids` reported by vLLM against the request digest.
These are the engine's conditioning tokens after preprocessing, including
multimodal placeholder expansion. Intermediate image-token deduplication is
allowed when expansion restores the requested token sequence; missing evidence or
a different final sequence rejects the result. This checks token conditioning,
not the content of image/audio/video payloads.

Episode receipts distinguish `assistant_mask` (token provenance) from the existing
`response_mask` (loss eligibility). Environment tokens can never enter the loss.
All assistant turn boundaries survive, including unusable turns that receive no
loss or per-turn reward. Existing reward-bearing spans remain separate.

Generation termination is `stop`, `length`, or `cancelled`. Episode termination
additionally distinguishes environment completion, turn limit, context limit,
response limit, absence of a follow-up, and an unusable turn.

Episode assembly verifies each retained generation span against the original token
IDs, logprobs and conditioning prefix. Only prefix-preserving stop trimming is
accepted. The receipt binds the assembled prompt, response, loss mask and logprobs
with a SHA256 digest. Logprobs are canonicalized to float32 for this digest, matching
upstream's queue transport. The trainer checks the digest before batch padding or
old-policy scoring. Receipts contain compact per-turn evidence rather than copies
of the token/logprob arrays.

## Load, cancellation and retry lifecycle

The server invalidates its load acknowledgement before receiving new weights.
Only successful full weight transfer followed by upstream's cache-reset/version
notification can establish a new acknowledgement. Versions must increase within a
lineage, preventing an acknowledged snapshot name from being reassigned. A failed
load leaves generation disabled.

Admission requires an exact match to the acknowledged policy. Completion must
still have the same acknowledgement; a weight update during generation invalidates
the result. Each result carries the server-issued acknowledgement, independent of
the requested policy fields.

A request ID is admitted at most once per loaded policy, including requests that
fail or are cancelled. Cancelled output is explicitly marked and never accepted
for training. A caller cancellation that discards the response cannot make that
response trainable; this protocol does not promise that all GPU work has stopped.

V1 does not automatically retry an episode or resume partial generation. A new
execution gets a new attempt ID, and a restarted trainer gets a new lineage. The
client rejects a response for a different attempt/request. The trainer and reward
ledger reject duplicate logical rollouts, including competing attempts; they do
not silently pick a winner. Supporting retry supersession across independent
workers requires a coordinator that explicitly selects the winning attempt.

## Compatibility and validation

V1 is mandatory on the current GRPO generation and ingestion path. Legacy or
partially populated results fail closed. Full weight synchronization is supported;
delta updates, asynchronous lag, and validation episodes need separate adapters.
The existing production GRPO configuration disables validation rollouts.

The CPU conformance tests cover wire validation, stale policies, failed loads,
weight changes during generation, cancellation, duplicate requests and logical
results, concurrent episode reuse, context/token/probability alignment, and separate
provenance/loss masks. Additional tests use the exact upstream client and
`AgentLoopOutput.as_dict()` to check request routing and float32 serialization.
A GPU training smoke test is still required before deployment; CPU tests do not
exercise CUDA weight transfer or inference kernels.
