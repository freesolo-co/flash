# Build your own training algorithm

Custom mode uses the same `flash train run.toml` entry point, project ownership, GPU
supervision, private artifact repository, checkpoint listing, and adapter deployment paths
as SFT and GRPO. Your Python file stays on your machine. Flash runs the model and optimizer
on the remote GPU; your file owns data, sampling, rewards, losses, and update timing.

Both the CLI and the control plane must include custom-mode support. This feature is under
review; an existing deployed server does not acquire it just by upgrading the client.

```bash
pip install 'freesolo-flash[custom]'
flash login
flash custom init algorithm.py
flash train run.toml
```

```toml
project = "YOUR-PROJECT-UUID"
model = "Qwen/Qwen3.5-9B"
algorithm = "custom"

[custom]
file = "algorithm.py"          # relative to this TOML file
max_wall_seconds = 3600
idle_seconds = 900

[train]
max_steps = 100
max_context_tokens = 2048
lora_rank = 16
save_every = 10

[gpu]
type = "H100"
```

Managed custom mode currently supports the GPU-tested Qwen3.5 9B on one Modal H100.
Users authenticate to Flash; provider and artifact credentials remain on the control plane
and worker. `flash train run.toml --cost` previews the dedicated-session cost cap.
The server charges measured session time, including time waiting for the local script,
within that cap. Model initialization is excluded from measured session time.

## The local file

Define `async def main(client)` for a nonblocking interface, or `def main(client)` for the
blocking interface. Flash selects the matching client automatically. Both have the same
operations and correctness guarantees. The generated file contains editable DPO data and
loss code, not a request to run a hidden server-side DPO recipe.

```python
from flash.custom import Datum


async def main(client):
    prompt = await client.encode("Question: What is 2 + 2?\nAnswer:")
    answer = await client.encode(" 4.")
    batch = [Datum(prompt + answer)]

    def objective(data, logprobs):
        loss = -logprobs[0][len(prompt) - 1 :].mean()
        return loss, {"answer_nll": loss.item()}

    for step in range(client.step, client.config["train"]["max_steps"]):
        result = await client.forward_backward_custom(batch, objective)
        update = await client.optim_step(
            learning_rate=1e-4,
            algorithm_state={"next_example": step + 1},
        )
        print(result["metrics"], update["metrics"])
        if update.get("checkpoint_ref"):
            print("Saved", update["checkpoint_ref"])
```

For the synchronous version, use `def main(client)` and remove `await`. No configuration
switch is needed. Independent reward calls, data preparation, and other sessions can run
concurrently with an awaited GPU operation. Operations on **one model stay serialized**;
async does not run conflicting weight updates concurrently or add multi-GPU training.
The complete two-pass custom loss holds its place in the same ordered queue.

Cancellation of an await does not cancel a submitted GPU mutation. A later operation cannot
overtake it. If an HTTP response is lost, new operations fail closed until
`retry_pending()` retrieves that operation's receipt. This avoids applying an update twice.

Keep the local process running (for example in tmux). `--background` is rejected for custom
mode because the algorithm lives locally. A local exception cancels the managed run;
completed durable checkpoints remain available. Idle and wall limits bound abandoned GPUs.
Automatic worker restart is disabled: continuing arbitrary local algorithm state requires
an explicit resume.

## Durable progress and resume

`train.save_every`, `train.save_at_steps`, the final optimizer update, and normal completion
save durable checkpoints. A successful save means the adapter and full training state have
committed together to the run's private artifact repository. The files survive GPU teardown.

Each snapshot contains:

- The PEFT adapter, tokenizer, and immutable base-model revision.
- AdamW state, CPU/CUDA RNG state, completed step, and parameter digest.
- The JSON `algorithm_state` supplied with the latest optimizer update (up to 1 MiB).

Use `algorithm_state` for your dataset cursor, frozen reference scores, reward statistics,
and other state required to continue. Flash cannot infer arbitrary state inside your Python
process. Large datasets should remain external; save stable identifiers/cursors instead.
The generated DPO example saves its initial reference scores and restores them without
recomputing the reference from the updated policy.

```bash
flash runs checkpoint RUN_ID
flash models deploy RUN_ID/step-100
```

To continue on a fresh GPU, add the saved reference and increase the **total** step horizon:

```toml
[custom]
file = "algorithm.py"
resume_from = "RUN_ID/step-100"

[train]
max_steps = 200
lora_rank = 16
```

The other model/project settings remain in the TOML. Resume verifies access to the source
run, pins its artifact revision, and restores the model, optimizer, RNG, and algorithm state
before invoking the local file. Read `client.step` and `client.restored_state`. The source
LoRA rank and alpha must match. A resumed experiment gets its own run ID and checkpoints.

Explicit `save_checkpoint(name, algorithm_state=...)` is also available at optimizer
boundaries. A step already saved automatically is immutable; use its `RUN_ID/step-N`
reference rather than saving a second version of that step. `download_checkpoint` remains
an optional ZIP export; downloading is no longer necessary to preserve managed progress.
Managed restores always start a new run through `custom.resume_from`; in-place rollback
cannot replace weights beneath an already-published run/step identity.

## Inspection and custom losses

| Operation | Result or effect |
| --- | --- |
| `encode`, `decode` | Model tokenizer, without local Transformers |
| `sample` | Tokens/text, policy and temperature-adjusted sampling logprobs, finish reason |
| `forward(data, top_k=5)` | Target-token logprobs and up to 20 alternatives per position |
| `forward_backward(data)` | Explicit weighted negative log-likelihood; accumulate gradients |
| `forward_backward_custom(data, loss_fn)` | Local differentiable loss and metrics; remote gradients |
| `optim_step(...)` | AdamW update, gradient norm, accumulation count, scheduled checkpoint |
| `zero_grad()` | Discard accumulated gradients |
| `save_checkpoint(...)` | Durable managed snapshot and run/step reference |
| `load_checkpoint(name)` | Standalone sessions only; managed runs use `custom.resume_from` |
| `status()`, `logs(after=cursor)` | Session state and bounded worker events |

For tokens `[10, 20, 30, 40]`, the outputs score targets `20, 30, 40` respectively.
For a prompt of length P, the first completion score is at index P-1. Prompt masking,
normalization, rewards, baselines, and reference policies belong to the local algorithm.
Convert a logprob to probability with `math.exp(logprob)`.

The custom loss runs locally over CPU PyTorch tensors. Flash sends its token derivatives
back for a second GPU forward/backward. By the chain rule this gives the original loss's
parameter gradient. Dropout is disabled and weights cannot change between the two passes.
The callback is never uploaded or pickled. This requires an extra forward pass; the
weighted surrogate's numeric value is logged separately from the actual custom loss.

Worker logs contain timing and numeric metrics, not prompts or full token arrays. Those
arrays are returned directly to the user's script. The session log retains 1000 events
and reports truncation; normal Flash console artifacts preserve worker output.

The current runtime is text-only, single-GPU LoRA with AdamW. It does not expose hidden
states, full-vocabulary logits, arbitrary optimizers, model layers, or distributed training.
The async API overlaps local/independent work; it is not a distributed execution engine.

## Validation and low-level experiments

The [managed GPU resume evidence](custom-managed-gpu-validation.json) records a local DPO
script training Qwen3.5 9B on an H100, publishing step 2, and shutting down its GPU. A new
managed run on a fresh H100 restored identical logprobs and algorithm state, completed step
3 with changed weights, and published a new checkpoint through the normal listing. Both
workers terminated successfully. This used the branch's control-plane lifecycle with real
Modal GPUs and private artifact storage; the hosted production API was not deployed.
`scripts/check_custom_managed_gpu.py` provides an opt-in paid reproduction against a server
running this branch.

The original [GPU evidence](custom-training-gpu-validation.json) covers local custom DPO,
nonzero gradients, changed weights, scoring/sampling agreement, optimizer restoration, and
exactly-once retries on A10G (Qwen3 0.6B) and H100 (Qwen3.5 9B). These are synthetic
correctness tests, not generalization benchmarks. CPU tests compare the two-pass custom
loss gradient to direct PyTorch autograd.

`TrainingClient.create(...)` / `AsyncTrainingClient.create(...)` remain low-level,
customer-owned Modal sandbox helpers for experiments and conformance tests. Those standalone
sandboxes do **not** use managed durable storage. Use `algorithm = "custom"` with
`flash train` for the normal Flash run/checkpoint lifecycle described above.
