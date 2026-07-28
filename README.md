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

The allocator picks the cheapest validated GPU class that fits the run — one dedicated
worker allocation per run, on a single GPU today — supervised server-side (stall watchdog,
bounded auto-retry resuming from the last streamed checkpoint, endpoint GC).

## What this repository is

Flash is the **client and control plane** for Freesolo's hosted post-training service.
This repository contains:

- the `flash` CLI (`flash/cli/`) — no declared runtime dependencies (commands that run
  an environment locally, such as `flash env test`, need the `freesolo` SDK),
- the FastAPI control plane (`flash/server/`) — run submission, auth, project scoping,
- the GPU worker and training recipes (`flash/engine/`) — TRL plus colocated vLLM
  rollouts,
- the GPU provider substrate (`flash/providers/`) — pricing, allocation, submit/poll,
- the environment loading machinery (`flash/envs/`).

It does **not** contain everything needed to stand up an equivalent service from scratch.
The following are Freesolo-operated and not part of this repository:

| Component               | Where it lives                                        |
| ----------------------- | ----------------------------------------------------- |
| Identity and API keys   | `api.freesolo.co` — verifies keys, owns projects/orgs |
| Multi-LoRA serving      | `serve.freesolo.co` — `flash/serve/` is a thin client |
| Managed environment hub | a private repository of published environments        |

If you are evaluating Flash, the honest summary is: **use it against the hosted service**,
or **read and modify the training/provider code**, which is self-contained and the most
reusable part of the repository. Running your own end-to-end copy of the service is
possible but requires replacing the components above — see [Self-hosting](#self-hosting).

## Using the hosted service

Install the client and authenticate with a freesolo API key. `flash login` is not
interactive — pass the key explicitly or export `FREESOLO_API_KEY` first:

```bash
pip install freesolo-flash
export FREESOLO_API_KEY=fslo_...
flash login          # validates the key and stores it in ~/.flash/config.json
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

Project ids also appear in your Freesolo dashboard. Every training TOML carries a required
top-level `project = "<uuid>"`, which Flash validates against the authenticated
organization before it allocates a run. Then describe the run and submit it:

```toml
project = "your-project-uuid"
model = "Qwen/Qwen3.5-4B"
algorithm = "sft"

[environment]
id = "your-name/my-env"

[train]
epochs = 1
max_examples = 1000
lora_rank = 32
```

```bash
flash train run.toml                  # submit, prints a run id
flash runs status RUN_ID               # follow it
flash models deploy RUN_ID             # serve the trained adapter
flash models chat RUN_ID -m "hello"    # talk to it
```

Run management lives under `flash runs` (`status`, `log`, `cancel`, `checkpoint`) and
serving under `flash models` (`deploy`, `chat`, `deployments`, `undeploy`, `export`).
`flash models` on its own lists supported base models and `flash gpus` lists GPU classes
with estimated $/hr. To copy a finished adapter into your own HuggingFace repo:

```bash
flash models export --adapter-id RUN_ID --repository your-org/your-repo
```

Intermediate RL checkpoints are deployable too — list them with
`flash runs checkpoint RUN_ID`, then pass `RUN_ID/step-N` as the adapter id.

There are no built-in task environments — the environment you push defines the task.
Single-turn and bounded multi-turn environments are supported.

## Calling a deployed adapter from your own app

Deploy once, then POST chat requests with your API key:

```bash
export FLASH_API_URL=https://flash.freesolo.co
export FREESOLO_API_KEY=fslo_...
export RUN_ID=flash-1782194170-ce1cfcff

curl -X POST "$FLASH_API_URL/v1/runs/$RUN_ID/deploy" \
  -H "Authorization: Bearer $FREESOLO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"dry_run": false}'

curl -X POST "$FLASH_API_URL/v1/runs/$RUN_ID/chat" \
  -H "Authorization: Bearer $FREESOLO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a two-sentence summary of the run."}],
    "temperature": 0.0,
    "max_tokens": 256
  }'
```

The response uses the OpenAI chat-completions shape; read `choices[0].message.content`.
The run id is the adapter id for serving. If the run is not deployed yet,
`/v1/runs/<run_id>/chat` returns `409` with a hint to deploy first.

Prefer the control-plane endpoint over calling the serving backend directly: it enforces
run ownership and forwards per-run serving options such as thinking-mode parity.

## Working on the code

The test suite is CPU-only and offline by default. No GPU, no network, no credentials:

```bash
uv sync --extra server --dev
uv run pytest -q                         # ~170 test files, offline
uv run ruff check .                      # lint
```

Those three are exactly what CI runs (`.github/workflows/ci.yml`).

To exercise the CLI from a dev checkout, invoke the module rather than the `flash` script:

```bash
uv run python -m flash.cli --help
```

The `--dev` group installs `runpod-flash`, which also declares a `flash` console script,
so `uv run flash` in this environment may launch RunPod's CLI instead of this one.
`python -m flash.cli` is unambiguous. Installed users are unaffected.

Formatting is not enforced repo-wide yet, so run `ruff format` on the files you touched
rather than the whole tree. See [CONTRIBUTING.md](CONTRIBUTING.md) for the branching
model — in short, **pull requests go into `dev`**.

### Layout

- `flash/catalog.py` — curated model catalog (Qwen3.5 and Qwen3.6, dense and MoE),
  VRAM-fit sizing, and each model's `thinking` capability
- `flash/schema/`, `flash/spec.py` — TOML to `JobSpec`
- `flash/runner/` — server-side run supervisor (durable job handle, retries, cost guard,
  endpoint GC)
- `flash/providers/` — GPU substrate (pricing, GPU classes, durable submit/poll,
  preflight) behind the `base.Provider` protocol, with `allocator.py` picking the cheapest
  fitting class
- `flash/engine/` — the on-GPU worker (TRL + colocated vLLM rollouts; distillation scores
  on-policy student samples against a remote teacher) and the shared recipe. SFT targets
  and RL rewards route through the active environment, so task-specific grading lives with
  the example, not in the engine
- `flash/envs/` — environment registry and the adapter that loads Freesolo SDK
  environments onto the worker's interface
- `flash/serve/`, `flash/server/` — serving client and the FastAPI control plane (run via
  the separate `flash-server` command)
- `tests/` — pytest suite (CPU-only, offline-by-default)

## Self-hosting

You can run your own control plane, but read this first — it is an operator deployment,
not a one-command install.

The control plane needs the `server` extra — the base install above is client-only and
`flash-server` will refuse to start without it:

```bash
pip install 'freesolo-flash[server]'
```

`flash-server` then fails fast at startup unless all of the following are present (see
`flash/providers/preflight.py` and `.env.example`). Note it reads the PROCESS
environment, so load your `.env` (`set -a && . ./.env && set +a`) rather than relying on
the file being present:

- `RUNPOD_API_KEY` — **two or more distinct** RunPod account keys, comma-separated. A
  single-account pool cannot reap or fail over across accounts, so the preflight rejects
  it.
- `LAMBDA_API_KEY` — Lambda Cloud API key.
- `HF_TOKEN` — write access to the dataset repos flash creates for artifacts. The repo is
  assigned by the control plane and scoped to the environment, so runs sharing an
  environment share a repo, each under its own prefix.
- `FREESOLO_INTERNAL_KEY` — control-plane authentication. Requests presenting this key
  authenticate as a single service identity with no network call, which is the path to use
  if you are not integrating with Freesolo identity.
- `GITHUB_TOKEN` — access to the managed environment repository.

Beyond credentials, three seams point at Freesolo services and would need replacing for a
fully independent deployment:

1. **User authentication.** Unknown bearer tokens are verified against
   `{FREESOLO_BASE_URL}/api/auth/verify` (`flash/server/auth.py`). Only the internal-key
   path works without the Freesolo backend.
2. **Environments.** Publishing and managed-slug loading target a private environment
   repository (`flash/server/envs.py`, `flash/envs/loader.py`).
3. **Serving.** `flash/serve/` is a client for the Freesolo multi-LoRA serving app; point
   it elsewhere with `FREESOLO_SERVING_URL`, but this repository does not include a
   serving backend.

The GPU worker image is public and can be pulled directly. It is published under an
explicit CUDA tag, not `latest`:

```bash
docker pull ghcr.io/freesolo-co/flash-worker:cu128
```

## Release channels

Two channels are published to PyPI from the _same source_, distinguished by one line in
`flash/_channel.py` (`CHANNEL`):

| Channel | PyPI package         | CLI         | Default plane           | Published from                                                                                         |
| ------- | -------------------- | ----------- | ----------------------- | ------------------------------------------------------------------------------------------------------ |
| prod    | `freesolo-flash`     | `flash`     | `flash.freesolo.co`     | push to `main` that bumps `[project].version` (`.github/workflows/publish.yml`)                        |
| dev     | `freesolo-flash-dev` | `flash-dev` | `flash-dev.freesolo.co` | push to `dev` whose `[tool.flash-dev].version` isn't on PyPI yet (`.github/workflows/publish-dev.yml`) |

Each environment holds exactly **one** channel: both packages ship the same import package
(`flash/`) with one baked `CHANNEL` line, so installing both into the same environment
makes the later install win for _both_ CLIs. For side-by-side prod and staging, install
each channel in its own virtualenv (or via `pipx`, which isolates per tool). The dev build
is produced by `scripts/build_dev_dist.py`, which renames the package/CLI and flips
`CHANNEL` to `dev` before `uv build`.

Within any single commit the two version fields are locked together:
`[project].version` and `[tool.flash-dev].version` must match (CI enforces this via
`.github/workflows/version-parity.yml`), so cutting a release means bumping both together.
The **published** channels can still differ, because dev publishes on merge to `dev` while
prod only publishes once `dev` is promoted to `main` — so `freesolo-flash-dev` is normally
one or more versions ahead of `freesolo-flash`.

Either CLI still honours an explicit `FLASH_API_URL` / the `login --api-url` flag; the
channel only sets the default.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security issues: [SECURITY.md](SECURITY.md) — do
not open a public issue.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
