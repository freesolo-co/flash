# Self-hosting Flash

Run your own Flash control plane against your own GPU accounts, with no Freesolo backend
involved.

This is an operator deployment, not a one-command install: you supply the GPU credentials,
you hold the auth key, and you decide which of the three providers to use. What it gets
you is the whole training path - SFT, GRPO, and on-policy distillation on verl with
colocated vLLM rollouts, the allocator, the stall watchdog, checkpoint streaming, and
endpoint GC - pointed at hardware you pay for directly.

## What you need

Two credentials and at least one GPU account:

|                         |                                                                                                                                                                               |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **One GPU provider**    | a RunPod, Lambda, **or** Vast account. One is enough.                                                                                                                         |
| `HF_TOKEN`              | a HuggingFace token with **write** access. Flash streams code, checkpoints, and adapters through HuggingFace dataset repos, so every run needs it whichever provider you use. |
| `FLASH_HF_NAMESPACE`    | the HuggingFace user or org those repos are created under - **set this to one your `HF_TOKEN` can write to.** It defaults to Freesolo's namespace, which you cannot write to. |
| `FREESOLO_INTERNAL_KEY` | the key your clients present to your plane. Generate it yourself.                                                                                                             |

Everything else is optional.

## Quickstart

```bash
pip install 'freesolo-flash[server]'   # the base install is client-only

export FLASH_STANDALONE=1
export FREESOLO_INTERNAL_KEY=$(openssl rand -hex 32)
export HF_TOKEN=hf_...
export FLASH_HF_NAMESPACE=your-hf-username   # where run artifacts are created
export RUNPOD_API_KEY=...              # or LAMBDA_API_KEY, or VAST_API_KEY

flash-server --host 0.0.0.0 --port 8080
```

Then point a client at it:

```bash
flash login --api-url http://your-plane:8080 --api-key "$FREESOLO_INTERNAL_KEY"
flash train run.toml
```

Because `--api-url` points at your own plane, `flash login` stores the key and checks it
against that plane. It does **not** send it to `api.freesolo.co` for verification: your
plane authenticates `FREESOLO_INTERNAL_KEY` itself, and that key controls the plane, so it
must not travel to a service you do not run. (If you operate your own Freesolo-compatible
auth backend, pass `--freesolo-url` and verification happens against it.)

Every run config needs a top-level `project` uuid - it groups runs and is required in
standalone mode too (see [project ids](#what-flashstandalone1-does)). Any uuid works
on a standalone plane, so you can pick one once and reuse it. `flash projects create` and
`flash projects list` are **hosted-only** - they talk to the Freesolo org directory, which
a standalone plane does not have - so generate one yourself (`python -c "import uuid;
print(uuid.uuid4())"`) and pass it explicitly to `flash env setup --project <uuid>`:

```toml
project = "11111111-1111-4111-8111-111111111111"
model = "Qwen/Qwen3.5-4B"
algorithm = "sft"

[environment]
id = "github:your-org/your-repo@main:path/to/env"

[train]
epochs = 1
max_examples = 1000
```

`flash-server` reads the **process** environment, not `.env`. If you keep credentials in a
file, load it: `set -a && . ./.env && set +a && flash-server`, or
`docker run --env-file .env ...`. Copy `.env.example` to start from a documented template.

The process environment is the only contract, so a Kubernetes Secret, systemd
`EnvironmentFile`, or any orchestrator's secret store works as-is. To pull secrets from a
secret manager at container start instead, wrap the image's entrypoint; `deploy/infisical/`
is a working example you can copy for another provider.

### The state directory

Everything the plane persists locally - the SQLite database of keys and run ownership, run
records, results, and the CLI's saved login - lives under one root, `~/.flash` by default
(`/root/.flash` in the container). Set `FLASH_DATA_DIR` to move it somewhere a rootless
container, a mounted PVC, or a `ProtectHome` systemd unit can actually write:

```bash
FLASH_DATA_DIR=/var/lib/flash
```

Everything moves together, so back up or mount that one directory. In Docker, setting
`FLASH_DATA_DIR` also means mounting your own volume at the new path - the image's `VOLUME`
declaration names the default location, and state written anywhere else lands on the
container's writable layer and is lost when the container is replaced.

Run exactly **one** instance per state directory. State is local files plus SQLite; there is
no horizontal scaling. On networked storage, where a lock can be held far longer than on a
local disk, raise `FLASH_SQLITE_BUSY_TIMEOUT_SECONDS` (default 30).

### Logs

`flash-server` logs at INFO. Provider resolution, capability revocation, reaper startup, and
degraded-configuration warnings are reported there and nowhere else, so that is the first
place to look when the plane misbehaves. `FLASH_LOG_LEVEL` turns it up (`DEBUG`) or down
(`WARNING`), and `FLASH_LOG_FORMAT=json` emits one JSON object per line for a structured log
sink.

## Environments

An environment is the Python package defining your task - dataset, rollout, reward. Point
`[environment] id` at any GitHub repository you control:

| Form                                          | Resolves to                                                  |
| --------------------------------------------- | ------------------------------------------------------------ |
| `github:owner/repo@ref:path/to/env`           | that repo at that ref. **Use this when self-hosting.**       |
| `https://github.com/owner/repo/tree/ref/path` | the same thing, in browser-URL form.                         |
| `namespace/name`                              | Freesolo's managed hub (`freesolo-co/environment-hub`) only. |

A bare `namespace/name` slug is **not** a generic shorthand - it always resolves against
Freesolo's hub, so a self-hosted plane should use the explicit `github:` form. `ref` may be
a branch, tag, or commit sha; pin a sha if you want runs to be reproducible.

Public repos work without credentials, subject to GitHub's unauthenticated rate limit. Set
`GITHUB_TOKEN` for private repos. `flash env push` publishes to the managed hub and is not
part of a self-hosted deployment.

**A local directory is not a supported environment source.** The GPU worker fetches the
environment itself, so it needs a source it can reach; an `[environment] path` is rejected
at submit. Push your environment to a GitHub repository (public or private with
`GITHUB_TOKEN`) and reference it with the `github:` form above.

## Choosing providers

Flash allocates across RunPod, Lambda, and Vast. **Configure the ones you have; the rest
are simply never considered.** There is no requirement to hold all three, and no
placeholder value to set for the ones you skip.

| Provider | Variable         | Notes                                                           |
| -------- | ---------------- | --------------------------------------------------------------- |
| RunPod   | `RUNPOD_API_KEY` | One key, or several comma-separated for multi-account failover. |
| Lambda   | `LAMBDA_API_KEY` |                                                                 |
| Vast     | `VAST_API_KEY`   |                                                                 |

The allocator ranks candidate GPU classes only on the substrates you configured, so a
plane with just `VAST_API_KEY` set allocates on Vast and never proposes a class it cannot
provision. Startup fails only when **all three** are missing.

A single RunPod key works. It logs a warning at startup because a one-account pool cannot
ride out that account's quota or credit exhaustion by moving to another - an availability
property, not a correctness one. Runs still allocate, and idle endpoints are still reaped
within the account you configured.

## What `FLASH_STANDALONE=1` does

Flash's managed deployment keeps organizations, projects, and billing in a Freesolo SaaS
backend, and the plane validates against it on every run. A self-hoster has no such
backend - without this flag the plane resolves `api.freesolo.co`, the validation call
fails, and **every run is rejected 503**. The flag is what makes self-hosting a supported
configuration rather than an unreachable one.

With it set:

- **`FREESOLO_INTERNAL_KEY` is the only credential the plane accepts.** External bearer
  tokens are rejected, not accepted unverified - there is no backend to verify them
  against, and treating "cannot verify" as "accept" would turn a self-hosted plane into an
  open one.
- **Project ids are taken as given.** They are still required and still validated for
  shape, so runs stay grouped; what is skipped is the ownership check against an org
  directory that does not exist here.
- **Environment mirror validation is skipped.** The environment package itself is still
  resolved and authorized normally.
- **Backend reporting is off** - billing precheck and charge, realized-cost
  reconciliation, checkpoint registration, and the hosted artifact GC sweep. Otherwise these would send your operator key to `api.freesolo.co`
  (and, for the GC, `serve.freesolo.co`) and log a warning per run or per startup.

Self-hosting relaxes the billing boundaries, not the catalog. Trainable models are the
curated ones on both deployments; see [adding a model](#adding-a-model).

## Adding a model

The catalog (`flash models list`) is six curated Qwen checkpoints. Anything else is rejected
at config parse time, before a GPU is rented:

```
unsupported model 'meta-llama/Llama-3.1-8B'; choose one of: ... - or, to train another
model, fork Flash and add a ModelInfo entry for it to flash/core/catalog.py
```

That is the whole workflow: fork, add an entry to `MODELS` in `flash/core/catalog.py`, and the
model is trainable. There is no config key that accepts an uncataloged id.

An entry is a `ModelInfo`. Copy the nearest existing one and correct it against the model's
`config.json`. What matters:

| Field                                                               | Why                                                                                      |
| ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `params_b`                                                          | The authoritative size. Every VRAM, disk, and cost term reads it. Must be > 0.           |
| `vocab_size`                                                        | Drives the logits term, which dominates SFT peak memory on a large vocabulary.           |
| `num_layers`, `hidden_size`, `num_key_value_heads`, `head_dim`      | The KV-cache geometry. Wrong here means a run that OOMs on the card it was sold.         |
| `num_attention_layers`, `num_linear_attention_layers`, `linear_*`   | Hybrid/linear-attention geometry, if the model has any.                                  |
| `algos`                                                             | Which of SFT/GRPO/OPD you are vouching for.                                              |
| `min_vram_gb`, `min_disk_gb`, `grpo_min_vram_gb`, `sft_min_vram_gb` | Allocation floors. The generic estimate is a starting point, not a substitute.           |
| `thinking`                                                          | `none`, `always`, or `hybrid` - whether the chat template opens a `<think>` block.       |
| `serving`                                                           | Serving capacity, if you intend to serve the adapter. Omit to train only.                |
| `active_params_b`                                                   | MoE only: the per-token active count, so FLOPs terms do not bill the full parameter set. |

`test_every_catalog_entry_sets_params_b` enforces the `params_b > 0` floor, and
`tests/test_catalog_consistency.py` checks the rest of the invariants - run it after adding
an entry.

This is more work than a config flag, and it produces a better-sized run. Flash used to have
a `model_policy = "allow"` flag that took any HuggingFace id and synthesized a `ModelInfo`
from a parameter-count lookup. It filled in four of the ~20 fields above; the KV geometry
stayed zeroed and the vocabulary sat at a default, and those zeros went into the same VRAM
equations a curated entry feeds, wearing the same type. Runs were sized against numbers
nobody had checked. Writing the entry means the numbers are real.

Gated repos (Llama among them) still require you to have accepted the licence on your HF
account, and private repos need `HF_TOKEN` to have read access.

> **RunPod endpoint concurrency is not capped by Flash**, on a self-hosted plane or a
> managed one. Flash used to carry a slot store intended to hold the account to a
> 58-endpoint ceiling, but it was claimed from a code path the live deploy replaced and so
> never enforced anything; it has been removed rather than left to imply a guarantee. If
> you expect many concurrent runs, cap them upstream of Flash or raise the worker quota on
> your RunPod account - otherwise a large enough burst hits RunPod's account limit and the
> excess deploys fail there.

### The security model

**A standalone plane is single-tenant.** It cannot distinguish organizations, so anyone
holding `FREESOLO_INTERNAL_KEY` can submit runs, read any run's status and logs, and spend
your GPU budget. Treat that key like a root password: generate it with
`openssl rand -hex 32` (or equivalent), never commit it, and rotate it if it leaks.

Rotating is safe: a standalone plane records run ownership against a fixed single-tenant
owner, not against the key's value, so runs started under the old key stay listed,
inspectable, and cancellable under the new one. The old key stops working immediately.

The same applies when you turn `FLASH_STANDALONE` on for a state directory that already
has runs in it: the single-tenant owner adopts them, so they stay listed and cancellable
rather than becoming invisible. This is safe only because standalone is single-tenant -
there is one principal, so every run in that store is already yours. Do not point a
standalone plane at the state directory of a **multi-tenant** deployment: it has no way to
tell whose runs those were, and it will treat all of them as the operator's.

Do not expose a standalone plane to untrusted callers. Put it on a private network, behind
a VPN, or behind an authenticating reverse proxy. If you need real multi-tenancy - separate
organizations, per-user keys, project ownership enforcement - you need an identity backend
serving the `/api/auth/verify` contract in `flash/server/platform/auth.py`, and you should run
without `FLASH_STANDALONE` and set `FREESOLO_BASE_URL` to point at it.

## Optional pieces

| Variable                  | Effect if unset                                                                                                                                                                                                             |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GITHUB_TOKEN`            | Environments in **private** GitHub repos cannot be fetched and `flash env push` is unavailable. Environments in **public** GitHub repos still work. Warns at startup.                                                       |
| `PARASAIL_API_KEY`        | On-policy distillation (`opd`) submissions fail before GPU allocation. Set it on the control plane; workers never receive it. `sft` and `grpo` do not use it.                                                               |
| `FLASH_CONTROL_PANEL_URL` | OPD submissions fail before GPU allocation. Set it to this control plane's worker-reachable HTTPS origin.                                                                                                                   |
| `FREESOLO_SERVING_URL`    | In standalone mode `flash deploy`/`undeploy`/`chat` **refuse to run** rather than send your plane's key to Freesolo's serving app - and refuse the same way if you point it AT that app. Training is unaffected. See below. |

## Serving

`flash/serve/` is a **client** for a multi-LoRA serving app (a Modal app that serves every
adapter on one GPU per base model, scaling to zero when idle). This repository does not
include that serving backend.

Training, checkpoint streaming, and adapter export are fully self-hostable and do not
depend on it. Your trained adapters land in your own HuggingFace repos and can be served
by any stack that loads LoRA adapters - vLLM, TGI, or your own. Point `FREESOLO_SERVING_URL`
at a compatible deployment if you want `flash deploy` and `flash chat` to work end to end.

On a standalone plane those three commands **error out** until you point it at a backend
you operate. Every serving request carries `FREESOLO_INTERNAL_KEY`, and on your plane that
key is what grants full control of it - so reaching Freesolo's serving app would hand your
plane's credential to a service you do not run. Export the adapter and serve it yourself if
you do not have a compatible backend.

Setting it to a Freesolo-hosted URL is refused exactly like leaving it unset: a value copied
from a managed `.env` is the likelier way to end up there, so both paths raise rather than
only the fallback. Any other host, including `localhost`, is yours to use.

The catalog reports a `serving.serve_model_id` per model - the pre-quantized FP8 checkpoint
Freesolo's serving app loads, and most of those repos are private. They are **informational
only**: nothing in the training path reads them, so they cannot block a run. If you stand up
your own serving backend, quantize the base model yourself (or serve it unquantized) rather
than expecting to pull those repo names.

## The worker image

The GPU worker image is public and pulls directly. It is published under an explicit CUDA
tag, not `latest`:

```bash
docker pull ghcr.io/freesolo-co/flash-worker:cu128
```

## Verifying your setup

`flash-server` runs a preflight at startup and refuses to serve if it cannot run a job.
The rule is "enough to run a job", not "everything Freesolo runs in production":
configuration that only degrades an optional capability warns instead of blocking boot.

It fails on:

- no GPU provider configured (all three keys missing) - the error names every acceptable
  way out
- `HF_TOKEN` missing
- `FREESOLO_INTERNAL_KEY` missing

It warns on a single-account RunPod pool and a missing `GITHUB_TOKEN`, then logs the
providers it resolved:

```
GPU provider(s) configured: vast
```

That line is the quickest confirmation your credentials were picked up. Then submit a
small run and watch it allocate.

## Troubleshooting

**Every run is rejected 503 with "project validation is unavailable".** `FLASH_STANDALONE`
is not set, so the plane is trying to reach a Freesolo backend. Set it to `1`.

**`401` on every request.** In standalone mode only `FREESOLO_INTERNAL_KEY` is accepted.
Confirm the client is sending that exact value (`flash login --api-key ...`), with no
trailing newline.

**Startup says no GPU provider is configured** but you set a key. The value must be
non-empty and non-whitespace. For RunPod specifically, a value of `","` parses to zero
usable accounts.

**Runs fail to allocate on a class you expected.** The allocator only proposes classes on
configured providers. Check the `GPU provider(s) configured:` startup line.
