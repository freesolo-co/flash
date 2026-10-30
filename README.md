# Flash

LoRA post-training for open-weight models: SFT and GRPO. You
describe a run in a TOML file, Flash allocates a GPU, trains, streams checkpoints, and
serves the resulting adapter.

For a Python loop that owns its loss, rewards, optimizer steps, and token-level inspection,
see [custom algorithm mode](docs/custom-training.md). `flash custom init algorithm.py`
writes an editable local DPO example. Set `algorithm = "custom"` and `[custom] file = "algorithm.py"`
in your training TOML, then use `flash train` with managed durable checkpoints and sync or async calls.

```bash
pip install freesolo-flash
export FREESOLO_API_KEY=fslo_...
flash login
flash train run.toml
```

The allocator picks the cheapest validated GPU class that fits the run, then supervises it
server-side: stall watchdog, bounded auto-retry resuming from the last streamed checkpoint,
and endpoint GC.

## What this repository is

Flash is the **client and control plane** for Freesolo's hosted post-training service:

| Path               | What it is                                                        |
| ------------------ | ----------------------------------------------------------------- |
| `flash/cli/`       | the `flash` CLI — no declared runtime dependencies\*              |
| `flash/server/`    | the FastAPI control plane — run submission, auth, project scoping |
| `flash/engine/`    | the GPU worker and training recipes — verl + colocated vLLM       |
| `flash/providers/` | the GPU substrate — pricing, allocation, submit/poll              |
| `flash/envs/`      | environment loading                                               |

\* `pip install freesolo-flash` pulls nothing. Commands that run an environment **locally** —
`flash env test` and `flash env eval` — do need the `freesolo` SDK, which they import at the
point of use; install it with `pip install freesolo`.

Two components stay Freesolo-operated and are **not** in this repository:

| Component                 | Where it lives                                                                                         | Self-hosted equivalent                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| Multi-tenant identity     | `api.freesolo.co`; verifies keys and owns projects/orgs                                                | `FLASH_STANDALONE=1` runs single-tenant on your own key           |
| Hosted Multi-LoRA serving | `serve.freesolo.co`, deployed from `flash/serving/`; `flash/serve/` owns the client and shared runtime | `flash serve deploy` provisions serving in your own Modal account |

So there are three ways to use Flash: against the **hosted service**, **self-hosted** against
your own GPU accounts, or as **training and provider code to read and modify**. The training
path is self-hostable end to end — see **[SELF_HOSTING.md](SELF_HOSTING.md)**.

## Using the hosted service

The active training catalog is `Qwen/Qwen3.5-9B`, `Qwen/Qwen3.8-27B`, and
`Qwen/Qwen3.6-35B-A3B`. The hosted serving catalog routes all three models on B200.
The retired dense Qwen3.6 27B identity is never translated into Qwen3.8. Customer-owned
Modal supports all three active models; 27B and 35B-A3B are bound to the certified serving image
digest. Customer-owned `flash serve` requires explicit `--provider modal`. Training and serving both use Modal.

`flash login` is not interactive. Export `FREESOLO_API_KEY` first so the key does not appear in process listings:

```bash
pip install freesolo-flash
export FREESOLO_API_KEY=fslo_...
flash login          # validates the key, stores it in ~/.flash/config.json
flash whoami         # confirm the identity behind it
```

Every run names an environment, which supplies the task data and the reward or SFT target.
Environments are published under a project, which scopes them to an organization:

```bash
flash projects create my-project                       # returns a project uuid
flash projects list                                    # look up existing uuids
flash env setup                                        # scaffold environment.py + dataset/train.jsonl
flash env push --project PROJECT_UUID --name my-env .  # returns an environment id
```

Every training TOML carries a required top-level `project`, validated against the
authenticated organization before Flash allocates anything:

```toml
project = "your-project-uuid"
model = "Qwen/Qwen3.5-9B"
algorithm = "sft"

[environment]
id = "your-org/your-project/my-env"

[train]
epochs = 1
max_examples = 1000
lora_rank = 32              # lora_alpha defaults to 2 x lora_rank; set it to override
```

```bash
flash train run.toml                  # submit, prints a run id
flash runs status RUN_ID               # follow it
flash models deploy RUN_ID/final             # serve the final checkpoint
flash models chat RUN_ID/final -m "hello"    # talk to it
```

Run management lives under `flash runs` (`list`, `status`, `log`, `cancel`, `checkpoint`)
and serving under `flash models` (`deploy`, `chat`, `deployments`, `undeploy`, `export`).
`flash models` lists supported base models — six curated Qwen checkpoints — and `flash gpus`
lists GPU classes with estimated $/hr.

Image-capable training and serving accept up to 32 images per conversation, including screenshots from successive tool turns. Compressed-byte, pixel, decoded-memory, and model context limits still apply; an image-count allowance does not override those budgets.

Intermediate RL checkpoints are deployable: list them with `flash runs checkpoint RUN_ID`,
then pass `RUN_ID/step-N` as the checkpoint id. To copy a finished adapter into your own
HuggingFace repo:

```bash
export HF_TOKEN=hf_...
flash models export --adapter-id RUN_ID/final --repository your-org/your-repo
```

To download adapter files directly, use the run owner's existing API key. No user
Hugging Face token or destination repository is needed:

```bash
curl --fail --get "https://flash.freesolo.co/v1/runs/$RUN_ID/adapter" \
  -H "Authorization: Bearer $FREESOLO_API_KEY" \
  --data-urlencode "checkpoint_id=$RUN_ID/final" \
  --output adapter.zip
```

The checkpoint is required: use `RUN_ID/final` for a finished run or an available
`RUN_ID/step-N` checkpoint. The ZIP contains normalized PEFT configuration and safetensors
weights, plus `manifest.json` (`flash-adapter-download-v1`) with run/checkpoint identity,
base-model and artifact-source revisions, and each file's byte count and SHA-256.
`X-Flash-Run-Id`, `X-Flash-Checkpoint-Id`, and `X-Flash-Adapter-Sha256` response headers
identify the archive. Verify the requested checkpoint and archive/file hashes before
loading. Artifact-storage credentials remain on the control plane, and downloads do not
create Hugging Face repositories. Downloads use private, non-cacheable responses and
are limited to 8 GiB of adapter files.

There are no built-in task environments — the environment you push defines the task.
Single-turn and bounded multi-turn environments are supported.

For single-turn GRPO rewards that intentionally score partial responses, such as draft-token
prefix agreement, set `train.mask_truncated_completions = false`. This retains generated
tokens in the loss when sampling reaches `train.max_completion_tokens`, without inventing
an EOS token. Choose the token limit and reward together; Flash does not prescribe a draft
length. Generation errors and empty batches still fail. Set the flag to `true` for rewards
requiring finished responses. By default, length-limited responses are masked, including
when `stop_sequences` is configured. Responses that actually finish on a stop string remain
trainable. This option is not supported for multi-turn environments, whose actions must
be complete and usable.

### Token probabilities in GRPO reward functions

Single-turn SDK scorers can read `response_text.rollout_context` from the string
passed to `score_response` (or each string in `score_responses`). Existing string
operations and callback signatures are unchanged. Multi-turn scorers receive the
same context in `episode.metadata["rollout_context"]`. Flash-native environment
callbacks can read `state["rollout_context"]`.

The optional `RolloutContext` contains `policy` (including `policy_version`, base
revision and tokenizer/template digests), `optimizer_step`, `sample_index`,
`rollout_ordinal`, `logprob_definition`, and an ordered tuple of `turns`. Each turn has:

- `prompt_token_ids`: the exact token prefix used for this generation.
- `token_ids` and `token_logprobs`: aligned tuples of generated token IDs and their
  natural-log probabilities, captured during rollout generation.
- `sampling`: effective engine sampling parameters, including temperature and
  truncation settings; `finish_reason`: `stop` or `length`.

These are complete engine generations, including reasoning and special/stop tokens,
before text cleanup or loss masking. They need not match the answer-only string's
tokenization. Environment replies occur in subsequent prompts, never as generated
tokens with fabricated probabilities. Multi-turn contexts also include generated
turns rejected as unusable by the environment; they are not a loss mask or a list of
only scoreable actions. Image-conditioned generations carry token prefixes, not image
payloads; a separate target scorer must also receive the original image context.

`logprob_definition` explicitly distinguishes `raw_logprobs` (model distribution)
from `processed_logprobs` (after the engine's sampling transformations). Do not treat
raw probabilities as the sampling distribution when temperature, top-p or other
processors change it. These are ordinary numbers with no gradient connection.
Only sampled-token probabilities are provided, not vocabulary distributions or
scores for arbitrary target tokens.

For example, a probabilistic speculative-decoding reward can combine draft logprobs
with a separately implemented target scorer:

```python
import math


def expected_accepted_prefix(draft_logprobs, target_logprobs):
    prefix_probability = 1.0
    expected_tokens = 0.0
    for draft, target in zip(draft_logprobs, target_logprobs, strict=True):
        prefix_probability *= math.exp(min(0.0, target - draft))
        expected_tokens += prefix_probability
    return expected_tokens


# Inside score_response(example, response_text):
context = getattr(response_text, "rollout_context", None)
if context is not None:
    turn = context.turns[0]
    # Score turn.token_ids with your target at the SAME prefixes/tokenization,
    # using the intended target distribution. Draft probabilities must describe
    # the actual proposal distribution; inspect context.logprob_definition.
```

The target-scoring call is user supplied. Greedy speculative decoding instead uses
token-prefix agreement. A policy-dependent reward is frozen during each GRPO update;
this alone does not implement an exact differentiable acceptance-rate objective.
Context is absent for replay/echo evaluation and other paths without generation
metadata. Handle that explicitly; Flash never reconstructs probabilities from text.
Converting a scorer input with `str(response_text)` drops its structured attributes.

### Calling a deployed adapter from your own app

Deploy once with `flash models deploy RUN_ID/final`, then POST chat requests with your API key:

```bash
export RUN_ID=flash-1782194170-ce1cfcff
export FREESOLO_API_KEY=fslo_...

curl -X POST "https://flash.freesolo.co/v1/runs/$RUN_ID/chat" \
  -H "Authorization: Bearer $FREESOLO_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"checkpoint_id\": \"$RUN_ID/final\", \"messages\": [{\"role\": \"user\", \"content\": \"hello\"}], \"max_tokens\": 256}"
```

The response uses the OpenAI chat-completions shape; read `choices[0].message.content`. The
checkpoint id is the permanent serving identity. If that checkpoint is not deployed yet, the
endpoint returns `409` with a hint to deploy first. Prefer this control-plane endpoint over calling the serving backend
directly: it enforces run ownership and forwards per-run serving options such as
thinking-mode parity.

### Workload estimates (SFT)

`flash train`, `--dry-run`, and `--cost` read the pinned environment's packaged dataset
without importing `environment.py`. `dataset/train.jsonl` is the canonical default;
`dataset/train.json` also works, and `[environment.params] split` or `dataset_path` can
select another packaged file.

The estimate tokenizes raw `input`/`output` fields plus the statically readable training
contract. SFT rows can instead supply `prompt_messages`, a nonempty chat-message list
that is authoritative for both profiling and training. This preserves stored multi-turn
histories and screenshot blocks; image-bearing transcripts use unpacked execution.
The row still requires `input` for dataset identity. Environment-added prompts, few-shot
examples, tool schemas, filters, and transformations are **not** executed, so real training
can retain fewer rows, truncate more often, and cost more than the estimate. If no readable dataset exists, cost, dry-run, and
submit all fail before GPU allocation rather than billing a profiling run.

## Working on the code

The test suite is CPU-only and offline by default. No GPU, no network, no credentials:

```bash
uv sync --extra server --dev
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

CI (`.github/workflows/ci.yml`) runs those on both supported interpreters (3.11 and 3.12),
plus two size gates and a serial pass for timing-sensitive tests:

```bash
uv run python scripts/check_file_size.py      # no module in flash/ over 1000 lines
uv run python scripts/check_function_size.py  # no function in flash/ over 150 lines
uv run pytest -q -m wallclock                 # asserts on real elapsed time; runs alone
```

`mypy` also runs in CI but is **advisory** — it reports existing type errors without failing
the build. Formatting is gated, so run `uv run ruff format .` before you push.

Use `flash`, `flash-cli`, or `python -m flash.cli` from a checkout.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branching model — in short, **pull requests go
into `dev`**.

### Layout

- `flash/core/catalog.py` — curated model catalog (Qwen3.5, Qwen3.8 dense, and Qwen3.6 MoE), VRAM
  sizing, and each model's `thinking` capability
- `flash/schema/`, `flash/core/spec.py` — TOML to `JobSpec`
- `flash/runner/` — server-side run supervisor (durable job handle, retries, cost guard)
- `flash/providers/` — GPU substrate behind the `base.Provider` protocol, with `allocator.py`
  picking the cheapest fitting class
- `flash/engine/` — the on-GPU worker (verl + colocated vLLM rollouts) and the shared recipe.
  SFT targets and RL rewards route through the active environment, so task-specific grading
  lives with the example, not in the engine
- `flash/envs/` — environment registry and the Freesolo SDK adapter
- `flash/serve/` — serving client, shared runtime, and customer-owned provider deployment
- `flash/serving/` — the hosted multi-LoRA Modal app behind `serve.freesolo.co`
- `flash/server/` — the FastAPI control plane (run via the separate `flash-server` command)
- `tests/` — pytest suite, CPU-only and offline

Within `flash/engine/worker/`, trainers live under `train/`, split by algorithm (`sft/`,
`rl/`) over a shared `core/`. Each carries a `child/` holding stdlib-only modules
**copied** into the verl subprocess rather than imported — flash and verl pin incompatible
torch/vllm versions, so neither can import the other.

## Self-hosting

Run your own control plane against your own GPU accounts, with no Freesolo backend involved.
**[SELF_HOSTING.md](SELF_HOSTING.md) is the full guide**; the short version:

```bash
pip install 'freesolo-flash[server]'   # the base install is client-only

export FLASH_STANDALONE=1
export FREESOLO_INTERNAL_KEY=$(openssl rand -hex 32)
export HF_TOKEN=hf_...
export FLASH_HF_NAMESPACE=your-hf-username   # a namespace your HF_TOKEN can write to
export MODAL_TOKEN_ID=...
export MODAL_TOKEN_SECRET=...

flash-server --host 0.0.0.0 --port 8080
```

Training and serving use **Modal**. Both Modal token variables are required. Training runs in GPU
Sandboxes with persisted attempt identities, checkpoint recovery, cancellation, and a shared
`flash-weights` Volume. GPU selection remains automatic within Modal's supported shapes.

`FLASH_STANDALONE=1` stops the plane calling out for project, environment, and billing
validation, and trusts `FREESOLO_INTERNAL_KEY` as a single-tenant operator credential.
External bearer tokens are rejected rather than accepted unverified. A standalone plane is
**single-tenant** — whoever holds that key can spend your GPU budget, so keep it off
untrusted networks. See [the security model](SELF_HOSTING.md#the-security-model).

`flash serve deploy` provisions serving in your own Modal account and requires explicit
`--provider modal`. Each deployment owns one exact base model and its compatible adapters; another
base model requires another deployment. Legacy provider deployment identities are unsupported.
It implements the same contract `flash/serve/` speaks.

Training selects the public worker image by its qualified digest. The historical `cu128` and
`cu128-sm*` tags can contain an incompatible trainer; they are build outputs, not the launch
contract. The selected image runs the pinned upstream Verl stack, with native kernel JIT when
no compatible baked cache is available:

```bash
docker pull ghcr.io/freesolo-co/flash-worker@sha256:34c84092f87b9ae43457620a8e5c0dc2e3dc30fdd5de5b4300c1eeb384e9a066
```

## Small Qwen3.5 models

Qwen3.5-0.8B and Qwen3.5-4B support text SFT and GRPO with pinned BF16 checkpoints.
Hosted serving uses L40S for 0.8B and exactly H100 for 4B (no H200 substitution),
with 32k context, four hot LoRA slots and rank up to 64. Image input and image training
are not enabled for these two entries. Customer-owned deployment profiles are available
for offline planning; live customer-owned allocation awaits separate qualification.

## Release channels

Two channels are published to PyPI from the _same source_, distinguished by one line in
`flash/_internal/channel.py`:

| Channel | PyPI package         | CLI         | Default plane           | Published on                           |
| ------- | -------------------- | ----------- | ----------------------- | -------------------------------------- |
| prod    | `freesolo-flash`     | `flash`     | `flash.freesolo.co`     | a version bump merged to `main`        |
| dev     | `freesolo-flash-dev` | `flash-dev` | `flash-dev.freesolo.co` | a push to `dev` with an unused version |

Each environment holds exactly **one** channel: both packages ship the same import package
with one baked `CHANNEL` line, so installing both makes the later install win for _both_
CLIs. For side-by-side prod and staging, use a virtualenv per channel (or `pipx`).

Within a commit, `[project].version` and `[tool.flash-dev].version` must match (CI enforces
this), so cutting a release bumps both. The **published** channels still differ, because dev
publishes on merge to `dev` while prod publishes only once `dev` is promoted to `main`.

Serving promotion requires stable release identity, a real streaming canary with durable billing
settlement, and no stalled accounting rows from the candidate deployment and release. Historical
quarantines remain in the global backlog for reconciliation; they do not roll back a later release
whose own accounting is healthy. Unreadable accounting evidence still fails the promotion gate.

Either CLI honours an explicit `FLASH_API_URL` or `login --api-url`; the channel only sets
the default.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security issues: [SECURITY.md](SECURITY.md) — do not
open a public issue.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

Checkpoint deployment states are `deploying`, `ready`, `undeploying`, `undeployed`,
and `failed`. Queueing, loading, and smoke verification happen within `deploying`.
Removal stays `undeploying` until confirmed, including retries after cleanup errors;
it never becomes `undeployed` merely because a removal request was sent. Existing
persisted states are translated when loaded, preserving registration and generation
fences. Deploy alongside the matching Freesolo deployment-state migration and backend.
