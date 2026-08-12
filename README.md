# Flash

LoRA post-training for open-weight models: SFT, GRPO, and on-policy distillation. You
describe a run in a TOML file, Flash allocates a GPU, trains, streams checkpoints, and
serves the resulting adapter.

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
| `flash/cli/`       | the `flash` CLI — pure standard library, no runtime dependencies  |
| `flash/server/`    | the FastAPI control plane — run submission, auth, project scoping |
| `flash/engine/`    | the GPU worker and training recipes — verl + colocated vLLM       |
| `flash/providers/` | the GPU substrate — pricing, allocation, submit/poll              |
| `flash/envs/`      | environment loading                                               |

Two components stay Freesolo-operated and are **not** in this repository:

| Component             | Where it lives                                        | Self-hosted equivalent                                    |
| --------------------- | ----------------------------------------------------- | --------------------------------------------------------- |
| Multi-tenant identity | `api.freesolo.co` — verifies keys, owns projects/orgs | `FLASH_STANDALONE=1` runs single-tenant on your own key   |
| Multi-LoRA serving    | `serve.freesolo.co` — `flash/serve/` is a thin client | adapters land in your HF repos; serve them with any stack |

So there are three ways to use Flash: against the **hosted service**, **self-hosted** against
your own GPU accounts, or as **training and provider code to read and modify**. The training
path is self-hostable end to end — see **[SELF_HOSTING.md](SELF_HOSTING.md)**.

## Using the hosted service

`flash login` is not interactive — pass the key explicitly or export `FREESOLO_API_KEY` first:

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
model = "Qwen/Qwen3.5-4B"
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
flash models deploy RUN_ID             # serve the trained adapter
flash models chat RUN_ID -m "hello"    # talk to it
```

Run management lives under `flash runs` (`list`, `status`, `log`, `cancel`, `checkpoint`)
and serving under `flash models` (`deploy`, `chat`, `deployments`, `undeploy`, `export`).
`flash models` lists supported base models — six curated Qwen checkpoints — and `flash gpus`
lists GPU classes with estimated $/hr.

Intermediate RL checkpoints are deployable: list them with `flash runs checkpoint RUN_ID`,
then pass `RUN_ID/step-N` as the adapter id. To copy a finished adapter into your own
HuggingFace repo:

```bash
flash models export --adapter-id RUN_ID --repository your-org/your-repo
```

There are no built-in task environments — the environment you push defines the task.
Single-turn and bounded multi-turn environments are supported.

### Calling a deployed adapter from your own app

Deploy once with `flash models deploy RUN_ID`, then POST chat requests with your API key:

```bash
export RUN_ID=flash-1782194170-ce1cfcff
export FREESOLO_API_KEY=fslo_...

curl -X POST "https://flash.freesolo.co/v1/runs/$RUN_ID/chat" \
  -H "Authorization: Bearer $FREESOLO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "hello"}], "max_tokens": 256}'
```

The response uses the OpenAI chat-completions shape; read `choices[0].message.content`. The
run id is the adapter id. If the run is not deployed yet, the endpoint returns `409` with a
hint to deploy first. Prefer this control-plane endpoint over calling the serving backend
directly: it enforces run ownership and forwards per-run serving options such as
thinking-mode parity.

### Workload estimates (SFT)

`flash train`, `--dry-run`, and `--cost` read the pinned environment's packaged dataset
without importing `environment.py`. `dataset/train.jsonl` is the canonical default;
`dataset/train.json` also works, and `[environment.params] split` or `dataset_path` can
select another packaged file.

The estimate tokenizes raw `input`/`output` fields plus the statically readable training
contract. Environment-added prompts, few-shot examples, tool schemas, filters, and
transformations are **not** executed, so real training can retain fewer rows, truncate more
often, and cost more than the estimate. If no readable dataset exists, cost, dry-run, and
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

> **The `flash` command can belong to something else.** The `server` and `dev` extras install
> `runpod-flash`, which declares its own `flash` console script; whichever installs last wins,
> and RunPod's exits 0 while doing nothing. Use **`flash-cli`** — the same entry point under a
> name nothing else claims — or `python -m flash.cli` from a checkout. A base
> `pip install freesolo-flash` is unaffected.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branching model — in short, **pull requests go
into `dev`**.

### Layout

- `flash/core/catalog.py` — curated model catalog (Qwen3.5 and Qwen3.6, dense and MoE), VRAM
  sizing, and each model's `thinking` capability
- `flash/schema/`, `flash/core/spec.py` — TOML to `JobSpec`
- `flash/runner/` — server-side run supervisor (durable job handle, retries, cost guard)
- `flash/providers/` — GPU substrate behind the `base.Provider` protocol, with `allocator.py`
  picking the cheapest fitting class
- `flash/engine/` — the on-GPU worker (verl + colocated vLLM rollouts) and the shared recipe.
  SFT targets and RL rewards route through the active environment, so task-specific grading
  lives with the example, not in the engine
- `flash/envs/` — environment registry and the Freesolo SDK adapter
- `flash/serve/`, `flash/server/` — serving client and the FastAPI control plane (run via the
  separate `flash-server` command)
- `tests/` — pytest suite, CPU-only and offline

Within `flash/engine/worker/`, trainers live under `train/`, split by algorithm (`sft/`,
`rl/`, `opd/`) over a shared `core/`. Each carries a `child/` holding stdlib-only modules
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
export RUNPOD_API_KEY=...                    # or LAMBDA_API_KEY, or VAST_API_KEY

flash-server --host 0.0.0.0 --port 8080
```

You need **one** of RunPod, Lambda, or Vast. Providers whose key is unset are never
considered, and the allocator only proposes classes it can actually provision. Startup fails
only when all three are missing.

`FLASH_STANDALONE=1` stops the plane calling out for project, environment, and billing
validation, and trusts `FREESOLO_INTERNAL_KEY` as a single-tenant operator credential.
External bearer tokens are rejected rather than accepted unverified. A standalone plane is
**single-tenant** — whoever holds that key can spend your GPU budget, so keep it off
untrusted networks. See [the security model](SELF_HOSTING.md#the-security-model).

The GPU worker image is public and published under an explicit CUDA tag, not `latest`:

```bash
docker pull ghcr.io/freesolo-co/flash-worker:cu128
```

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

Either CLI honours an explicit `FLASH_API_URL` or `login --api-url`; the channel only sets
the default.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security issues: [SECURITY.md](SECURITY.md) — do not
open a public issue.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
