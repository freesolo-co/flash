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

`flash env test` runs a scaffolded environment **locally**, so it needs the `freesolo` SDK in the
interpreter that loads your `environment.py`. The `[server]` extra above already includes it. The
bare `pip install freesolo-flash` does not - it is client-only and pulls nothing - so on a machine
where you author environments without the server extra, validation fails until you add it:

```bash
pip install freesolo                                     # same interpreter as the CLI
uv tool install freesolo-flash --with freesolo --force   # existing uv tool: rewrite it with the SDK
pipx inject freesolo-flash freesolo                      # existing pipx tool
```

The lower two matter if you installed the CLI with `uv tool` or `pipx`: the `flash` executable then
lives in its own venv, and a plain `pip install freesolo` lands in your shell's interpreter instead

- `python -c "import freesolo"` works while `flash env test` still fails. Use the line for the
  manager you installed with; `--force` is what makes `uv tool install` update an existing tool
  rather than no-op, and pipx has no equivalent of `--with`, so the SDK goes in via `inject`.

(`flash env eval` is not local in this sense. It evaluates a run against a **published** hub
environment and refuses a generic `github:` ref outright, which is the only kind a standalone plane
accepts - so installing the SDK does not make it usable self-hosted.)

The `server` extra pulls in `runpod-flash`, which declares its own `flash` console script.
Whichever distribution installs last wins, so on a plane host the bare `flash` command may launch
RunPod's CLI instead of this one - it exits 0 and does nothing, which is easy to miss. Use
**`flash-cli`**: the same CLI under a name nothing else claims. Check with `flash --help`; if it
does not mention `train`, use `flash-cli` (or `python -m flash.cli`) wherever this guide says
`flash`. A client-only machine that never installs the `server` extra is unaffected.

`flash-server` speaks plain HTTP. For anything but loopback, put a TLS-terminating reverse proxy
(nginx, Caddy, a cloud load balancer) in front of it and point DNS for your hostname at the proxy -
the plane itself never terminates TLS. On the same machine, skip the proxy and use
`http://127.0.0.1:8080` below.

Then point a client at it:

```bash
export FREESOLO_API_KEY="$FREESOLO_INTERNAL_KEY"
# remote: the https address of your TLS proxy, not the plane's own :8080
flash login --api-url https://your-plane.example
# same machine: loopback plaintext is fine
# flash login --api-url http://127.0.0.1:8080
flash train run.toml
```

(On a host that has the `server` extra installed, run these as `flash-cli` - see the note above.)

Because `--api-url` points at your own plane, `flash login` stores the key and checks it
against that plane. It does **not** send it to `api.freesolo.co` for verification: your
plane authenticates `FREESOLO_INTERNAL_KEY` itself, and that key controls the plane, so it
must not travel to a service you do not run. (If you operate your own Freesolo-compatible
auth backend, pass `--freesolo-url` and verification happens against it.)

`--api-url` is the address **your client** reaches the plane at. Terminate it with TLS whenever it
leaves the machine: `flash login` and every command after it send `FREESOLO_INTERNAL_KEY` as an
`Authorization: Bearer` header, and on a standalone plane that key is the whole authorization
boundary and owns every run - anyone who observes it on the wire can submit billed GPU jobs and
read or cancel your existing ones. Plaintext `http://` is for loopback only (`http://127.0.0.1:8080`
during local development); `flash login` warns when you give it a non-loopback `http://` URL.
On-policy distillation additionally needs the address a
**rented GPU worker** reaches it at, which is usually a different, public one - set
`FLASH_PUBLIC_URL` on the plane for that (see [optional pieces](#optional-pieces)). `sft` and
`grpo` do not need it.

Every run config needs a top-level `project` uuid - it groups runs and is required in
standalone mode too (see [project ids](#what-flashstandalone1-does)). Any uuid works
on a standalone plane, so you can pick one once and reuse it. `flash projects create <name>`
mints one locally for you (there is no org directory to record it in, and your plane accepts
any well-formed uuid), so pass it to `flash env setup --project <uuid>`. `flash projects list`
has nothing to enumerate and says so; keep track of the ids you use:

```toml
project = "11111111-1111-4111-8111-111111111111"
model = "Qwen/Qwen3.5-9B"
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
`EnvironmentFile`, or any orchestrator's secret store works as-is.

To pull secrets from [Infisical](https://infisical.com) at container start instead, set
`INFISICAL_CLIENT_ID` (plus its secret, project id, and path). The published images carry the
CLI, so this is a configuration change rather than a rebuild, and leaving the variable unset
keeps the `--env-file` behaviour above unchanged:

```bash
docker run -p 8080:8080 -v flash-state:/root/.flash \
  -e INFISICAL_CLIENT_ID=... -e INFISICAL_CLIENT_SECRET=... \
  -e INFISICAL_PROJECT_ID=... -e INFISICAL_PATH=/flash \
  ghcr.io/freesolo-co/freesolo-flash:main
```

Building the image yourself omits the CLI unless you ask for it
(`docker build --build-arg INSTALL_INFISICAL=true .`); with the variable set but the CLI
absent, the container refuses to start rather than booting without the secrets it was told to
fetch. The switch is unset-vs-set, not truthiness: setting `INFISICAL_CLIENT_ID` to an empty
string also refuses, since that is what a `${VAR}` with nothing behind it produces rather than
a deliberate request for the plain-environment path. See `deploy/infisical/README.md` for the
full variable list, and copy its entrypoint if you use a different secret manager.

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

| Form                                          | Resolves to                                                   |
| --------------------------------------------- | ------------------------------------------------------------- |
| `github:owner/repo@ref:path/to/env`           | that repo at that ref. **Use this when self-hosting.**        |
| `https://github.com/owner/repo/tree/ref/path` | the same thing, in browser-URL form.                          |
| `namespace/project/name`                      | Freesolo's managed hub only - **rejected when self-hosting.** |

A bare `namespace/project/name` slug is **not** a generic shorthand - it always resolves
against `freesolo-co/environment-hub`, which is private. `ref` may be a branch, tag, or
commit sha; pin a sha if you want runs to be reproducible.

**Each plane accepts only the form it can actually fetch, and the two do not overlap.**
Freesolo's managed service accepts `namespace/project/name` and nothing else: the hub is the
only repo it can vouch for, being the one `flash env push` writes to and the one whose
packages carry a validated project association. Submitting a `github:` or browser-URL ref
there is rejected at submit with a `400`.

A standalone plane (`FLASH_STANDALONE=1`) is the exact inverse: the two GitHub forms are its
only environment source, and a `namespace/project/name` slug is **rejected at submit with a
`400`**. Every slug maps onto Freesolo's private hub, so such an id names the one repository
your plane certainly cannot read - left to run it would fail on a rented GPU with a bare
GitHub `404`, after the run had already cost you money. The refusal names the repo and the
form to use instead. This applies to every spelling of the hub, including an explicit
`github:freesolo-co/environment-hub@main:...` ref and its browser-URL form.

A local directory is not a supported source either (see below). The check reads
`FLASH_STANDALONE` on the server, so it is the plane you submit to, not the CLI you submit
from, that decides.

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
- **Managed hub environment ids are refused at submit.** Every `namespace/project/name` slug resolves
  to Freesolo's private hub, which this plane cannot read, so it is rejected with a `400`
  naming the repo instead of failing later on a rented GPU. Use the `github:` form
  ([environments](#environments)).
- **Backend reporting is off** - billing precheck and charge, realized-cost
  reconciliation, checkpoint registration, and the hosted artifact GC sweep. Otherwise these would send your operator key to `api.freesolo.co`
  (and, for the GC, `serve.freesolo.co`) and log a warning per run or per startup.
- **Hosted-only CLI commands say so.** `flash projects create` mints a uuid locally instead
  of calling the org directory; `flash projects list` and `flash traces export` have no local
  store to read and refuse with the reason. Traces are recorded by the freesolo SDK into the
  hosted backend, so write the same `{"input", "output"}` JSONL rows yourself and train on
  them directly. Setting `FREESOLO_BASE_URL` to a Freesolo-compatible backend you run keeps
  all three on the normal hosted path, since they then have a directory to call.

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

| Variable               | Effect if unset                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GITHUB_TOKEN`         | Environments in **private** GitHub repos cannot be fetched and `flash env push` is unavailable. Environments in **public** GitHub repos still work. Warns at startup.                                                                                                                                                                                                                                                                                                      |
| `PARASAIL_API_KEY`     | On-policy distillation (`opd`) submissions fail before GPU allocation. Set it on the control plane; workers never receive it. `sft` and `grpo` do not use it.                                                                                                                                                                                                                                                                                                              |
| `FLASH_PUBLIC_URL`     | On-policy distillation (`opd`) submissions fail before GPU allocation. This is **this plane's own public HTTPS origin** - the address a rented GPU worker dials back to reach the teacher broker. It is not the same setting as the client's `FLASH_API_URL` / `--api-url`, which may point at a private address (`http://your-plane:8080`, a tunnel, a VPN) that no worker can resolve; set this one to an origin reachable from outside. `sft` and `grpo` do not use it. |
| `FREESOLO_SERVING_URL` | In standalone mode `flash models deploy`/`undeploy`/`chat` **refuse to run** rather than send your plane's key to Freesolo's serving app - and refuse the same way if you point it AT that app. Training is unaffected. See below.                                                                                                                                                                                                                                         |

## Serving

`flash serve deploy` provisions serving in **your own** Modal or RunPod account, running the
published worker image against one base model and one run's adapter. Training and export remain
independent of serving. Catalog serving checkpoint repositories are informational only and are never
resolved by the training path.

```bash
# the `server` extra, not the bare install: `serve deploy` resolves the adapter through
# huggingface_hub and drives modal's sdk, and `[project].dependencies` is empty by design.
pip install 'freesolo-flash[server]'
export HF_TOKEN=hf_...
export FLASH_SERVING_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')

# modal: `modal token new` writes these, or set them directly
export MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...

flash serve deploy \
  --provider modal \
  --model Qwen/Qwen3.5-9B \
  --run <run-id> \
  --deployment-id my-9b-serving \
  --image ghcr.io/freesolo-co/freesolo-flash-serve@sha256:<digest> \
  --artifact-repo <hub-repo> \
  --artifact-subfolder <path-within-repo> \
  --lora-rank 32 \
  --modal-workspace <your-workspace> \
  --modal-environment main \
  --modal-region us-east
```

The three `--modal-*` placement flags are required for `--provider modal` even though `--help`
lists them as optional: they are optional to _argparse_ because RunPod takes its own pair instead,
and `placement_for` is what requires exactly one provider's set. Omitting them exits with
`modal placement requires environment, region, workspace_name`.

For RunPod, export `RUNPOD_API_KEY` instead and swap the placement flags rather than adding to
them -- `placement_for` rejects the other provider's inputs instead of ignoring them, so keeping
`--modal-*` alongside `--provider runpod` fails before anything is created:

```bash
flash serve deploy \
  --provider runpod \
  --model Qwen/Qwen3.5-9B \
  --run <run-id> \
  --deployment-id my-9b-serving \
  --image ghcr.io/freesolo-co/freesolo-flash-serve@sha256:<digest> \
  --artifact-repo <hub-repo> \
  --artifact-subfolder <path-within-repo> \
  --lora-rank 32 \
  --runpod-account <your-account-id> \
  --runpod-data-center <data-center-id>
```

Both `--runpod-*` flags are required and must be nonempty, for the same reason the `--modal-*`
trio is: `--help` lists them as optional only because each provider takes its own set.

Provider credentials are read from the process environment for the duration of a single call and are
never written to the deployment record, logs, or command arguments — so any later resize or teardown
requires exporting them again. `--image` must be digest-qualified (`name@sha256:...`) so the
deployment is pinned to an exact immutable image. Add `--dry-run` to resolve and validate every
input, including the adapter's provenance, without provisioning anything or incurring cost.

The command prints the endpoint URL. Call it directly with the key you generated, which is scoped to
that one deployment:

```bash
curl https://<your-app>.modal.run/v1/chat/completions \
  -H "Authorization: Bearer $FLASH_SERVING_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "<run-id>", "messages": [{"role": "user", "content": "hello"}]}'
```

Do **not** point `FREESOLO_SERVING_URL` at this endpoint. Those commands authenticate with
`X-Freesolo-Internal-Key` and carry `FREESOLO_INTERNAL_KEY`, the key that controls your whole plane;
a customer-owned deployment does not read that header at all, so the request would fail `401` after
sending a plane-wide credential to a provider endpoint. The deployment serves `/healthz`,
`/v1/models`, and `/v1/chat/completions` -- it receives its adapters in an immutable manifest at
boot and has no `/adapters` surface for `flash models deploy` to drive.

The shared runtime supports bounded multimodal preparation. A profile that declares `image_limit`
loads a processor and accepts image-bearing requests up to that limit; a text profile declares none
and returns `400` for them.

`FREESOLO_SERVING_URL` belongs to the separate multi-LoRA backend behind `flash models deploy` /
`chat` / `undeploy`; standalone refuses an unset or Freesolo-hosted value there for the same reason.
See [docs/serving-contract.md](docs/serving-contract.md) for both credential rules, the normative
endpoints, and the conformance command.

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
Confirm `FREESOLO_API_KEY` contains that exact value, with no trailing newline, then run
`flash login` again.

**Startup says no GPU provider is configured** but you set a key. The value must be
non-empty and non-whitespace. For RunPod specifically, a value of `","` parses to zero
usable accounts.

**Runs fail to allocate on a class you expected.** The allocator only proposes classes on
configured providers. Check the `GPU provider(s) configured:` startup line.
