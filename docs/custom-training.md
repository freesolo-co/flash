# Build your own training algorithm

Custom mode lets a local Python script own the algorithm while a dedicated remote GPU owns
the model and optimizer. Install `freesolo-flash[custom]`, run `flash custom init algorithm.py`,
and edit the generated DPO example. DPO is implemented entirely in that script; it is not a
new built-in worker recipe.

This first version provisions a **single GPU in your own Modal account** using your Modal
profile or `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`. Modal bills the dedicated GPU, including
time spent inspecting results between calls. It does not submit a managed Freesolo TOML run,
use Freesolo credits, or appear in `flash runs`. The existing SFT/GRPO/OPD paths are unchanged.

```bash
pip install 'freesolo-flash[custom]'
modal token new  # if you have not configured Modal already
flash custom init algorithm.py
python algorithm.py --model Qwen/Qwen3-0.6B --gpu A10G --steps 6 --output my-training
```

The generated script saves a probability inspection file, per-step metrics, worker events,
and a checkpoint ZIP. Start with a small model and short context while developing the algorithm.
Use `--model Qwen/Qwen3.5-9B --gpu H100` for a larger model after checking your GPU budget.

## What the user builds

1. **Choose the data and sampling policy.** Tokenize locally supplied text through the
   session tokenizer, or sample from the current adapter. Keep the returned token IDs,
   policy log-probabilities, sampling log-probabilities, finish reason, and model version.
2. **Compute supervision.** Your script owns rewards, judges, preference pairs, baselines,
   advantage normalization, replay selection, filtering, and reference probabilities.
3. **Write the objective.** A regular PyTorch function consumes one differentiable vector of
   target-token log-probabilities per example. It may compare multiple examples, mask any
   positions, or combine objectives. Return a scalar loss and finite numeric metrics.
4. **Choose updates and evaluation.** Explicitly accumulate gradients, call `optim_step`,
   sample or score an evaluation batch, inspect metrics, and save checkpoints. No hidden
   optimizer step, scheduler, reward normalization, or epoch loop runs on the worker.

```python
from flash.custom import Datum, TrainingClient

with TrainingClient.create(
    "Qwen/Qwen3-0.6B",
    gpu="A10G",
    max_steps=20,
    max_wall_seconds=1800,
    idle_seconds=300,
) as client:
    prompt = client.encode("Question: What is 2 + 2?\nAnswer:")
    answer = client.encode(" 4.")
    batch = [Datum(prompt + answer)]

    def objective(data, logprobs):
        loss = -logprobs[0][len(prompt) - 1 :].mean()
        return loss, {"answer_nll": loss.item()}

    result = client.forward_backward_custom(batch, objective)
    print(result["metrics"])
    print(client.optim_step(learning_rate=1e-4)["metrics"])
    print(client.forward(batch, top_k=5)["outputs"])
    client.save_checkpoint("experiment-1")
    client.download_checkpoint("experiment-1", "experiment-1.zip")
```

The starter uses DPO: score chosen and rejected answers under the initial model, keep those
reference scores fixed, then optimize the log-sigmoid of the policy/reference preference
margin. The same API can express clipped PPO, REINFORCE, sequence ranking, or mixtures of
SFT and preference losses without adding algorithm names to Flash's catalog.

## Token alignment and probability semantics

For `Datum(tokens=[10, 20, 30, 40])`, the three outputs are
`log p(20 | 10)`, `log p(30 | 10,20)`, and `log p(40 | 10,20,30)`.
There is no invented BOS/EOS and no probability for the first input token. For a prompt of
length `P`, the first completion target is at output index `P-1`.

`forward_backward` requires `len(tokens)-1` explicit weights and accumulates gradients of
`-sum(weight * logprob)` across every row. Signed weights are allowed. Use zero weights for
prompt targets and divide by your chosen token/sequence/batch denominator explicitly. Calls
accumulate until `optim_step` or `zero_grad`; the engine never divides by the number of calls.

`forward` returns untempered policy log-probabilities and optional top-k alternatives at
each position. Obtain probabilities with `math.exp(logprob)`. `sample` returns both these
policy scores and the actual temperature-adjusted sampling-distribution scores. Greedy
sampling has behavior log-probability zero. Use the correct behavior scores for importance
sampling; changing temperature changes the behavior distribution. The initial sampler uses
temperature only (no top-p/top-k filtering), and returns raw completion token IDs including
EOS if generated. `finish_reason="length"` is an actual truncation; your algorithm decides
whether to train on it.

## How custom losses reach the GPU

The first GPU forward returns log-probabilities. Your loss runs on CPU tensors with autograd
enabled. Flash sends `-dLoss/dLogprob` as weights for a second GPU forward/backward. By the
chain rule, this weighted surrogate has the same parameter gradient as your original loss.
The surrogate's numeric value is **not** the custom loss; both are named separately in the
result and worker metrics. This costs an additional forward pass.

The model runs with dropout disabled for both passes, while autograd remains enabled. The
client serializes its operations and the worker rejects stale model versions. Your Python
callback is never uploaded, pickled, imported, or executed on the worker.

## Observability and state

| Operation | Result or effect |
| --- | --- |
| `encode`, `decode` | Use the model's tokenizer without installing Transformers locally |
| `sample` | Completion tokens/text, policy and sampling logprobs, finish reason |
| `forward(data, top_k=5)` | Per-target logprobs and up to 20 alternatives per position |
| `forward_backward(data)` | Explicit weighted NLL; accumulate gradients only |
| `forward_backward_custom(data, loss_fn)` | Local differentiable loss, custom metrics, remote gradients |
| `optim_step(learning_rate=..., max_grad_norm=...)` | AdamW update, gradient norm, accumulation count; clears gradients |
| `zero_grad()` | Discard accumulated gradients |
| `save_checkpoint(name)` | Immutable adapter + optimizer + RNG snapshot, weight digest and model metadata |
| `load_checkpoint(name)` | Restore a checkpoint within the same live session; advances model version |
| `download_checkpoint(name, path)` | Export the checkpoint ZIP before session shutdown |
| `status()`, `logs(after=cursor)` | Session state and bounded, cursor-addressed worker metrics/events |
| `close()` | Close the session and terminate the GPU provisioned by `create` |

Every operation receipt includes `session_id`, `sequence`, `model_version`, completed update
count, pending backward count, and configured limits. Loading a checkpoint restores the
optimizer and RNG but never rewinds the session's billed update counter or bypasses its cap.
Checkpoints require an optimizer boundary: apply or discard pending gradients first.

Keep your own dataset cursor, reference scores, reward state, and hyperparameters alongside
the checkpoint. The server cannot capture arbitrary state inside your local Python program.
Checkpoint storage is ephemeral in this first version; download before closing. The ZIP
contains a standard PEFT adapter, tokenizer, engine state, and metadata. Cross-session
optimizer resume and direct registration in hosted `flash models` are not implemented yet.

Worker logs contain operation timings and numeric metrics, not prompts or complete token
arrays. Those arrays are returned directly for your script to inspect/store. Logs retain the
latest 1000 events and report `truncated=true` when a reader has fallen behind.

## Retry and lifecycle contract

An optimizer timeout is ambiguous: the weights may already have changed. Flash retains the
exact pending request. Call `retry_pending()`; the worker returns its previous receipt without
applying the operation again. All other new operations are blocked until that outcome is
resolved. Do not replace a timed-out request with a new client and blindly repeat it.

The worker serializes execution, rejects reused sequences with different contents, and
rejects old receipts after a newer operation. A restarted worker has a new session ID, so
old operations cannot silently replay into fresh weights. A partial GPU failure poisons the
session and it cannot continue. There is no automatic replay/restart of a paid training step.

Always use a context manager. Normal completion and Python exceptions attempt GPU teardown.
An idle timeout shuts down an abandoned worker; a provider-enforced hard wall limit bounds
even a killed local process or wedged GPU call. The startup path reconciles the unique sandbox
name after ambiguous creation and attempts teardown rather than launching a duplicate GPU.

## Scope and validation

This is an experimental, text-only, single-GPU LoRA runtime for Hugging Face causal language
models on A10G or H100 GPUs. It uses the pinned Flash worker image, standard PEFT LoRA, dropout-free training and
AdamW with zero weight decay. It does not expose hidden states, full vocabulary logits,
custom model layers, arbitrary optimizers, multimodal batches, or distributed training.
Custom objectives must be differentiable functions of the returned target-token logprobs.
It favors a small inspectable API over the throughput of the optimized verl recipes.

`scripts/check_custom_gpu.py` provisions real hardware and implements DPO locally. It checks
that loss improves, gradients are nonzero, adapter weights change, sampling matches policy
scoring, checkpoint restoration reproduces scores/updates, and losing an optimizer response
then retrying applies the update only once. This is a correctness canary on synthetic examples,
not a claim about generalization or production-scale training quality.

```bash
python scripts/check_custom_gpu.py --output /tmp/custom-small.json
python scripts/check_custom_gpu.py --model Qwen/Qwen3.5-9B --gpu H100 \
  --output /tmp/custom-9b.json
```

The unit tests also compare custom-loss gradients against direct PyTorch autograd and test
HTTP authentication, malformed masks, stale versions, concurrent duplicate submissions,
partial failures, optimizer limits, and checkpoint boundary validation without network access.

### Recorded GPU runs

[Machine-readable validation results](custom-training-gpu-validation.json) record these
2026-10-29 runs, with nonzero gradients, changed adapter hashes, checkpoint export/restore,
sampling/scoring agreement, and safe retry after losing an optimizer response:

| Model | GPU | DPO updates | Initial loss | Final loss |
| --- | --- | ---: | ---: | ---: |
| Qwen3 0.6B | A10G | 6 | 0.693147 | 0.000514 |
| Qwen3.5 9B | H100 | 6 | 0.693147 | 0.000460 |

The A10G run also restored the optimizer and reproduced the first update's exact parameter
hash. The generated CLI starter ran independently on A10G, produced its local artifacts,
and exported a checkpoint. All of these GPU sessions were terminated afterward.
