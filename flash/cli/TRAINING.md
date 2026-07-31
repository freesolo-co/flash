# TRAINING.md — how to actually improve a model with Flash

> **If you are an AI agent asked to train a model here, read this first.**
> `flash env setup` dropped this file next to your `environment.py` and `configs/`.
> It is the playbook Freesolo's own training agents follow to turn a _finished_
> run into a model that _measurably improved_. The mechanics live in the hosted
> docs (https://docs.freesolo.co); this file is the judgment that sits on top of them.

A run that reaches `done` is **not** the same as a run that worked. Submitting a run
is not a result. The whole job is to design the learning signal, read what the run
actually produced, and decide — honestly — whether the model got better.

---

## Using Flash

Flash is a **managed** training service with a thin CLI/client. You author an
environment (the task + its reward), publish it, and submit SFT, GRPO, or on-policy
**distillation** runs from a TOML config. Flash allocates the cheapest fitting managed GPU class, runs the job,
streams logs back, and serves the result. You never handle infrastructure credentials —
you authenticate once with a freesolo API key, and everything below is a `flash` CLI
command.

### Install & authenticate

```bash
pip install freesolo-flash          # installs the `flash` CLI (import name is also `flash`)
flash login --api-key fslo_...       # or: export FREESOLO_API_KEY=fslo_...  (create a key at https://freesolo.co)
flash whoami                         # confirm the identity behind your key
flash projects create "my project"     # prints the canonical project UUID
flash models list                     # supported base model ids
flash gpus                           # managed GPU classes with estimated $/hr
```

**Install the CLI and the `freesolo` SDK into the _same_ interpreter.** Your
`environment.py` imports `freesolo`, and the CLI loads that module in **its own**
interpreter — so an isolated install (`pipx`, `uv tool install`) can run `flash` fine
while `flash env test` fails with _"the 'freesolo' package is required to run Freesolo
environments"_ even though `python -c "import freesolo"` works in your shell. The
message names the package, not the interpreter that is missing it, so the obvious fix
appears to do nothing. Two ways to keep them together:

```bash
uv venv                                         # uv pip install needs an environment to install INTO
uv pip install freesolo-flash freesolo          # same venv; then use ./.venv/bin/flash
uv tool install freesolo-flash --with freesolo  # isolated tool venv, SDK injected
```

`uv venv` first is not optional: with no virtual environment active, `uv pip install`
installs nothing and exits with _"No virtual environment found; run `uv venv` to create an
environment"_. (The `uv tool install` line manages its own venv and needs no `uv venv`.)

If you use the `uv tool` form, note that `freesolo-flash` does **not** declare `freesolo`
as a dependency — it is injected, so it survives only as long as uv remembers the `--with`.
`uv tool upgrade freesolo-flash` does remember it and keeps the SDK. Re-running
`uv tool install freesolo-flash` **without** `--with freesolo` does not: it rewrites the
tool venv from the new command and the SDK disappears, so `flash env test` starts failing
right after a reinstall that reported success. Always carry the flag through a reinstall:
`uv tool install freesolo-flash --with freesolo --force`.

**`flash whoami` before you create anything.** `flash login` writes a machine-wide
credential to `~/.flash`, but `FREESOLO_API_KEY` in your environment — including one
picked up from a `.env` file you sourced — **silently overrides it**. Because both
credentials are valid, a mismatch does not surface as an auth failure; it surfaces later
as `project '<uuid>' does not belong to the authenticated organization`, or as a project
quietly created in the wrong org. Pin the key explicitly for a run rather than relying on
ambient state:

```bash
flash whoami                                    # which org/key am I actually using?
FREESOLO_API_KEY=fslo_... flash train ...       # pin this run's identity
env -u FREESOLO_API_KEY flash train ...         # or force the `flash login` credential
```

**Run visibility follows the key that submitted the run, not the org.** If you rotate API
keys, runs submitted by the old key return `unknown run_id` from the new one — same
account, same org, same user. You cannot re-serve, re-deploy, or re-evaluate those runs
from the new key, so **archive baseline results to disk** rather than planning to
regenerate them. (`init_from_adapter` is unaffected: it resolves server-side at submit.)

### The project layout (`flash env setup` created this)

```text
environment.py          # the task: how to prompt the model and how to score it
dataset/train.jsonl     # training rows, one JSON object per line: {"input": ..., "output": ...}
configs/sft.toml        # an SFT run config
configs/rl.toml         # a GRPO (RL) run config
configs/opd.toml        # an on-policy distillation run config
TRAINING.md             # this file
```

`flash env setup` scaffolds into the **current directory** and takes no positional name —
`flash env setup my-env` is an error. The name is supplied later, at `flash env push
--name`. It also bakes `--project` into every generated config, so **create the project
first**, then `cd` into an empty folder and scaffold:

```bash
mkdir my-env && cd my-env
flash env setup --project <project-uuid>
```

**Dataset rows: put per-example state under `metadata`.** A `TaskExample` exposes `input`,
`output`, `metadata`, and the untouched source row as `.record`. Extra top-level keys do
survive to the worker on `.record`, but `id`/`_id` are always **ignored and replaced** with
an auto-generated positional handle, and individual consumers may project a row onto the
canonical columns. Reading `example.metadata["board"]` (or `example.record["board"]`) is
stable; assuming an `example.board` attribute is not. Assert the shape in your own test so
the constraint is enforced rather than remembered — an environment that reads a field the
worker does not carry fails **remotely, on a paid run**, long after `flash env test` passed.

### 1. Author the environment

`environment.py` defines the task. A single-turn env subclasses
`EnvironmentSingleTurn`, turns a row into a prompt, and scores the model's response
with a `RewardResult` (see _Reward design_ below). `load_environment()` is the entry
point Flash calls:

```python
from freesolo.datasets.types import TaskExample
from freesolo.environments import EnvironmentSingleTurn, RewardResult

class MyEnv(EnvironmentSingleTurn):
    dataset = load_jsonl("dataset/train.jsonl")   # rows -> TaskExample(input=..., output=...)

    def build_prompt_messages(self, example: TaskExample, prompt_text: str):
        return [{"role": "user", "content": example.input}]

    def score_response(self, example: TaskExample, response_text: str) -> RewardResult:
        expected = str(example.output or "").strip()
        score = 1.0 if expected and expected in response_text else 0.0
        return RewardResult(score=score, threshold=1.0)

def load_environment(**kwargs) -> MyEnv:
    return MyEnv()
```

For tool use, dialogue, or games, subclass `EnvironmentMultiTurn` instead and drive the
conversation across turns. The reward is the same `RewardResult` contract either way.

### 2. Publish the environment

A managed run references a **published** environment by id — so push your folder first:

```bash
flash env push --project <project-uuid> --name my-env .       # uploads this project; prints an env id like "your-org/my-env"
flash env list                       # local env sources you can push
```

To train against an env someone else published, just set its slug as `[environment] id` —
no separate step is needed. Paste the returned id into `[environment] id` in **both** configs.
Re-push after any
edit to `environment.py` or `dataset/` so the managed run uses your change.

**Validate locally before you push** — but know what the local gate does and does not
cover:

```bash
flash env test .                     # imports environment.py, loads the dataset, runs the scorer
flash env pull your-org/my-env -o ./pulled   # -o must be a FILE for a single file, a DIR for a
                                             # whole env; a new path is created for you, but an
                                             # existing non-empty dir is refused
```

**Validate the parameters your run actually trains on.** Left alone, `flash env test` calls
`load_environment()` at its _defaults_ — potentially a different dataset file than your run
uses, so the gate can pass while the real training path is broken, or fail while the real
path is fine. Pass the run's own `[environment.params]` instead: `--split` sets the split,
and `--param KEY=VALUE` (repeatable, values parsed as TOML scalars) sets every other kwarg.
For a run configured as

```toml
[environment.params]
split = "train_sft"
difficulty = "hard"
max_turns = 4
```

validate it with the matching invocation:

```bash
flash env test . --split train_sft --param difficulty=hard --param max_turns=4
```

Values keep the types `[environment.params]` would give them — `max_turns=4` arrives as an
int, `strict=true` as a bool, and anything that does not parse as a TOML scalar as text.

`flash env eval` takes the same two flags, for the same reason: a held-out suite scored
against a differently-configured environment is not measuring your run.

**It is also a three-example smoke test, not a dataset audit.** `flash env test` runs the
contract checks against the **first 3 rows only** (`_DEFAULT_EPISODES = 3`, and it iterates
`dataset[:episode_count]`), so a malformed prompt, a scorer that raises, or a bad episode
shape anywhere past row 3 sails through a green local gate and is first exercised by the
paid worker. On a heterogeneous dataset — mixed sources, mixed difficulty, a long tail of
odd rows — loop your own scorer over **every** row before you pay for a run. The local pass
means "this env imports and the contract holds for a sample", not "every row is safe".

**A hash comparison against the published env is not a valid staleness check.** `flash env
push` injects a small `sys.path` import shim into `environment.py` at publish time, so
`environment.py` **always** hashes differently after a round-trip while `dataset/` files
pass through byte-identical. Diff the content — the reward logic, constants, and episode
contract should be identical — and hash the dataset files for the data half.

### 3. Configure the run (TOML)

```toml
model = "Qwen/Qwen3.5-4B"   # see `flash models list`
project = "PROJECT_UUID"  # required UUID from `flash projects create`
# model_revision = "main"   # optional ref resolved to an immutable hugging face commit before submit
algorithm = "sft"           # "sft" (supervised), "grpo" (RL), or "opd" (on-policy distillation)
# thinking = true           # opt-in reasoning mode, for models that support it
# seed = 42                 # reproducible per-run seed; omitted defaults to 42

[environment]
id = "your-org/my-env"      # the id printed by `flash env push`
# params = { split = "train" }    # kwargs passed to load_environment(); the table is
                                   # `params` — NOT `args`
# secrets = ["SERPAPI_API_KEY"]   # only the NAMES of env vars your environment reads;
                                   # values are pulled from your shell/.env at submit time,
                                   # never stored in the spec

[train]
epochs = 1                  # one pass over the retained train rows
max_examples = 2            # rows to train on (the starter dataset has 2)
# max_steps = 100           # positive values set the exact optimizer-update horizon
# save_at_steps = [10, 50, 100]  # requires max_steps; overrides save_every
# multi-turn GRPO defaults to one reward per rollout; "per_turn" gives turn-level credit.
# credit_assignment = "per_episode"   # "per_turn" needs per_turn_rewards metadata and is
                                      # unsupported for tool-calling envs — see below
lora_rank = 32              # lora_alpha is managed: always derived as 2 x lora_rank
# All SFT/GRPO knobs live under [train]. Do not add [sft] or [grpo] tables.

[wandb]
# project  = "my-project"   # the table allows exactly `project` and `run_name`;
# run_name = "sft-run-1"    # `name` is rejected
```

**Key placement that is easy to get wrong.** Every one of these is a real submit-time
error or a wrong-config-that-still-runs; `--dry-run` catches the loud ones for free.

| You might write              | The schema wants                               | What happens                                                                       |
| ---------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------------------- |
| `[environment] args = {...}` | `[environment] params = {...}`                 | rejected as an unknown `[environment]` key                                         |
| `[wandb] name = "..."`       | `[wandb] run_name = "..."`                     | `[wandb] unknown key(s): name`                                                     |
| `[train] thinking = true`    | top-level `thinking = true`                    | rejected under `[train]`; misplacing it trains the wrong mode                      |
| `[train] lora_alpha = 64`    | omit it                                        | rejected — alpha is managed as `2 x lora_rank`                                     |
| `[train] max_tokens = 512`   | `max_completion_tokens` / `max_context_tokens` | rejected as an unknown `[train]` key                                               |
| `[sft]` / `[grpo]` tables    | `[train]`                                      | rejected — allowed tables are `environment`, `train`, `gpu`, `wandb`, `worker_env` |

`WANDB_API_KEY` is a default runtime secret, but it is only read from the process
environment or a `.env` **next to your CWD or the config** — not from a `.env` one
directory up. Verify what the server actually recorded rather than trusting the file: the
submitted spec is echoed back, so confirm e.g. `"thinking": false` there before you spend.

**SFT requires an explicit `[train] max_examples`**, even for an uncapped run — set it to
the dataset's exact row count. Count records, not newlines: `wc -l` is off by one when the
final JSON object has no trailing newline (silently dropping that last row from training)
and over-counts if the file has blank lines. Count the way the loader reads it:

```bash
python -c "print(sum(1 for l in open('dataset/train.jsonl') if l.strip()))"
```

That line is right only when your environment serves exactly that file unmodified. The
worker never reads `dataset/train.jsonl` — it calls `env.dataset()` and slices the result
(`train[:max_examples]`). If `[environment.params]` selects a different split, or
`load_environment()` filters, dedupes, or generates rows, the file count is the wrong
number: too low and you silently train on a subset, too high and you have mispriced the
run. Count what the worker will actually get, with the **same params as the submitted
config**:

```python
from flash.envs.registry import load_environment

env = load_environment("<your-env-id>", params={...})  # the [environment.params] you submit
print(len(env.dataset()))
```

There is no "all" sentinel, so this number is duplicated from the data into the config and
**can drift**: if the dataset later grows, a stale `max_examples` silently trains on a
subset instead of erroring. Re-check it whenever the dataset or the params change.

GPU allocation and HF artifacts are **managed by default**: leave `[gpu] type` unset to
let the allocator pick the cheapest fitting validated class, while `train.hf_repo` remains
platform-managed. For controlled experiments, `[gpu] provider` restricts allocation to one
provider and `[gpu] type` pins one exact active validated GPU class. Run artifacts are stored in a private environment-scoped repo with content-addressed
Flash code snapshots. Set `seed` only at the top level; `[worker_env]` cannot override
`SEED`, `RUN_ID`, `HF_REPO`, or `FLASH_ARM`. Compose or tweak configs without editing files: `--config
extra.toml` (deep-merge) and `--set key=value` (e.g. `--set train.epochs=3`).

### 4. Submit

```bash
flash train configs/sft.toml --dry-run     # validate the config on the server — no GPU, no charge
flash train configs/sft.toml --cost        # pre-flight USD estimate, then exit
flash train configs/sft.toml               # submit and follow logs (Ctrl-C detaches)
flash train configs/sft.toml --background  # submit and return immediately
```

> **Killing `flash train` does NOT cancel the run.** Submission happens server-side; the
> command then merely _streams_ status. Interrupting it — or wrapping it in `timeout` —
> kills only the client-side stream while the run keeps billing on its GPU, and re-running
> the command creates a **second** run. Two GPUs then bill in parallel for the same
> experiment.
>
> - Never wrap `flash train` in `timeout`, and use `--background` when you do not want to
>   watch.
> - After any interrupted submit, reconcile with `flash runs list` **before** re-running.
>   The list shows only run id, state, algorithm, cost, GPU and model — not `max_examples`
>   or `[wandb] run_name` — so identify the live duplicate by running `flash runs status`
>   on each recent candidate id, and cancel the one you did not mean to keep.
> - Probe affordability with `--dry-run`: it starts no GPU job and incurs no training
>   charge. It _does_ persist a run record, so expect a `dry_run` entry to appear in
>   `flash runs list` — don't mistake it for a live duplicate during that reconcile.
> - A cancel lands in `cancelled` whether the run was still in setup or already training;
>   it is not reclassified as `failed`. Either way billing stops.

### 5. Monitor

```bash
flash runs status <run-id>            # state + accrued cost
flash runs log <run-id>               # reward/loss trend + worker console/error logs + any traceback
flash runs log <run-id> --follow      # stream a live run to completion
flash runs list                  # all your runs and their state/cost
flash runs cancel <run-id>            # stop a run
```

Live GRPO metrics update at the managed HEARTBEAT cadence, not after every optimizer step.
The terminal heartbeat carries the latest bounded metric backlog so short runs still surface data.

**A run that looks frozen usually is not.** Reporting is on a fixed multi-minute cycle, and
some stages are announced once and then do un-instrumented work — so an unchanging stage
label, a flat step counter, or an idle GPU **prove nothing on their own**. Two distinct
situations produce an identical-looking `status`:

- _Normal quiet._ The last heartbeat is recent; the worker is simply between reports.
- _Preemption._ The worker died mid-stage and stopped heartbeating. The control plane keeps
  serving the **last** heartbeat it received, so `status` still reports `running` at
  whatever stage was announced, with a stale GPU snapshot attached.

**Heartbeat _age_ is the useful signal — but it is not proof on its own, and quiet is
normal.** Uploads are deliberately throttled: during training the worker publishes at most
about once every **900 s** (15 min), so a timestamp that has not moved for several minutes
is the _expected_ steady state, not a symptom. Upload failures can stretch the gap further.
`flash runs status` says so itself once the heartbeat passes 5 minutes — _"heartbeat uploads
are throttled; quiet is not dead"_.

**The threshold is algorithm-dependent: OPD is far tighter than 15 min.** The 900 s figure is
the throttle for an ordinary stage ping. OPD's post-update ping is issued with `force=True`,
which bypasses that throttle and is instead bounded by the forced-commit floor of **60 s**
(`_HB_FORCE_MIN_INTERVAL_S`), gated on the step having advanced. So an OPD run that is still
applying updates re-publishes roughly every optimizer step, subject only to that 60 s floor —
and an OPD heartbeat stuck for several minutes already means updates have stopped, which is
exactly the window the 15 min guidance below would tell you to ignore on a billed run.

So treat age as a threshold, not a verdict, and pick the threshold for the algorithm:

- _OPD, past ~2-3 min in a stepping stage (`opd_step`):_ act. Updates should be re-publishing
  at the 60 s floor, so a multi-minute gap is already anomalous rather than expected quiet.
  (Setup and rollout stages — `opd_model_load`, `opd_filtering_prompts`, `opd_vllm_initializing`
  — are liveness-driven and legitimately quieter; the tight threshold is for stepping.)
- _Otherwise, under ~15 min:_ tells you nothing is wrong. Do not act.
- _Otherwise, well past ~15 min:_ suspicious, still not conclusive. Corroborate before deciding —
  check the provider/attempt state and follow `flash runs log <run-id> -f`, which streams
  independently of the heartbeat upload cycle. A retry that has already started will show a
  new attempt.

**Exception — startup is exempt from all of the above.** While the run is warming up
(`status` shows a `warmup` row: initializing the model, vLLM, and training kernels) a long
quiet stretch at 0% GPU is the normal case, not a stall. `status` itself calls this out and
says _"setup is not billed; do not cancel"_. It can run far longer than the panel's
"typically several minutes, sometimes 15-20 min" — 40+ minutes before the first step is
something we have measured on a real run. Do not start the ~15 min clock until you have seen
step 0 advance to step 1; before that, the only thing worth watching is `flash runs log -f`.

Never derive seconds-per-step from total elapsed time (early steps include one-time warmup
that can dominate a short run).

> **Even on a genuinely stale heartbeat, do NOT cancel.** Flash **auto-retries a preempted job from its
> last checkpoint**, and the provider's `job_preempted` notice can lag the freeze by ~10
> minutes. Cancelling races that retry and kills a run that was recovering on its own.
> Wait for either the automatic retry or a genuine terminal state. Cancel only when a run
> is confirmed dead with no retry pending, or is burning money on a definite unrecoverable
> fault. An idle GPU during model load is also not evidence of a hang — and it is not
> evidence that your GPU class is unhealthy, so do not "fix" it by pinning another one.

### 6. Deploy & chat

```bash
flash runs checkpoint <run-id>       # deployable per-step RL checkpoints
flash models deploy <run-id>            # serve the trained adapter
flash models deploy <run-id>/step-N     # serve an intermediate checkpoint
flash models chat <run-id> -m "hello"   # chat with the deployed adapter
flash models deployments                # active serving endpoints
flash models undeploy <run-id>          # tear the endpoint down
flash models export --adapter-id <run-id> --repository <you>/<repo>  # copy adapter weights to your HF repo
```

> **`flash models deploy` returns before the revision is servable.** It returns
> `state: queued`, and while the new revision reconciles, `flash models deployments` can
> still show `ready` for the **previous** one. An eval harness that polls for `ready` and
> starts immediately will hit an endpoint that rejects its requests with _"deployment is
> reconciling"_ — and if the harness records per-row errors but still prints an aggregate,
> you get a confident, entirely wrong number. This exact race turned a healthy checkpoint
> into an apparent catastrophic regression (mean 0.07 vs its true 0.68).
>
> Before evaluating: poll until `state == ready` **and** the reported step equals the step
> you are evaluating. After evaluating, apply the error-count check from "Don't let your own
> harness lie to you" below.

**Before exporting, confirm your HF token can write to the target repository.** A token that
is valid — and that works everywhere else — may resolve to a different account or org than
the `--repository` you pass, or may simply be read-only or a fine-grained token scoped to
other repos. Either way the export fails on permissions, or silently lands in the wrong
namespace.

Check **write access to the exact destination**, not identity. `whoami()` happily returns
the right account and org memberships for a token that cannot write anything, so it does not
establish the precondition:

```python
from huggingface_hub import HfApi

api = HfApi(token="<the same token the export will use>")

# 1. create the destination if it is missing. private=True matches what
#    `flash models export` does: it always creates the repo private first, so the
#    destination is never transiently public.
api.create_repo("<you>/<repo>", repo_type="model", exist_ok=True, private=True)

# 2. THEN actually write. Step 1 alone does not prove write access on a repo that
#    already exists (see below); a real commit does.
api.upload_file(
    path_or_fileobj=b"ok",
    path_in_repo=".flash-write-probe",
    repo_id="<you>/<repo>",
    repo_type="model",
    commit_message="write probe",
)
```

**Why the second step is not redundant.** On an existing repo, `create_repo(exist_ok=True)`
is not a write test. The server rejects the create, and `exist_ok` is precisely the flag that
swallows the rejection — by two different routes, neither of which checks whether you can
write:

- **409 (already exists):** swallowed unconditionally and returned as success. No permission
  is consulted at all.
- **403 (no write permission on the namespace):** swallowed too, then retried as
  `repo_info()` — a _read_. If the read succeeds, the call returns a repo URL and looks like
  it passed.

So a read-only or wrongly-scoped token sails through against any destination that already
exists, which is the normal case for a repo you export to more than once. Only the upload
exercises the permission the export actually needs. (Verified on the 1.2.0 floor and on
current hub — both swallow paths are present in every version Flash supports.)

Two more details that make the difference between a probe and a placebo:

- **Pass the same token the export will use.** Flash resolves the export token as
  `--api-key` > `HF_TOKEN` in the environment > a local `.env` / `.env.local`, and forwards
  exactly that value. `huggingface_hub` does none of that — with no `token=` it falls back to
  your ambient cached login, so the probe can pass on one credential while the export fails
  on a different one. If you are exporting with `--api-key`, or with a token that lives only
  in `.env`, pass that literal value here.
- **Ask for `private=True`.** The real export creates the repo private and only flips it
  public afterwards if you passed `--public`. A probe that omits this can leave a brand-new
  repo publicly visible until the export catches up, and it fails outright under an org
  policy that permits private repos but not public ones.

The probe leaves a `.flash-write-probe` file behind; delete it with
`api.delete_file(".flash-write-probe", "<you>/<repo>", repo_type="model")` once it passes, or
just let the export overwrite the repo contents around it.

`create_repo` and `upload_file` work on every `huggingface-hub` version Flash supports. There is also
`auth_check("<you>/<repo>", write=True)`, which checks write access without creating
anything — but the `write=` argument only exists in **hub ≥ 1.5.0** (Flash's floor is
1.2.0). On an older hub it raises `TypeError: ... unexpected keyword argument 'write'`,
and dropping the argument to silence that turns it back into a read-only check that a
token with no write access still passes. Verify your version before relying on it. It takes
`token=` too, and needs it for the same reason.

Do this _before_ the run finishes, not after. If several tokens are in play, identify them
by fingerprint rather than by printing them.

### Loading an exported adapter locally (transformers + peft)

Flash trains the Qwen3.5/3.6 family against the full multimodal module tree, so adapter
weights saved by the worker carry a `language_model.` infix in their tensor keys
(`base_model.model.model.language_model.layers.*`). During `flash models export`, adapters that
can be represented in the text-only namespace are normalized to
`base_model.model.model.layers.*`. When an export retains non-LM tensors, it keeps the
multimodal namespace instead. Do not infer the namespace solely from whether `flash models export`
was used: inspect the keys. For normalized keys, load with vanilla peft and
`AutoModelForCausalLM`:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B")  # qwen3.5 causal-lm class
model = PeftModel.from_pretrained(base, "<you>/<repo>")
```

**Before you trust any local eval, check the key namespace matches the model class you
loaded.** peft does NOT error on mismatched keys — it emits a `UserWarning` about missing
adapter keys and silently applies _nothing_, so you would benchmark the bare base model
believing it is your adapter. If the inspected keys carry the `language_model.` infix,
load the base with the multimodal class whose parameters live under that namespace instead:

```python
# infixed keys (base_model.model.model.language_model.layers.*) need the multimodal class:
from transformers import AutoModelForImageTextToText

base = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-0.8B")
# qwen3.5/3.6 dense models resolve to Qwen3_5ForConditionalGeneration
# qwen3.6-35b-a3b (moe) resolves to Qwen3_5MoeForConditionalGeneration
model = PeftModel.from_pretrained(base, "<you>/<repo>")
```

To check which form you have, read the safetensors key names (stdlib, no torch needed):
the JSON header of `adapter_model.safetensors` lists every tensor key. Rule of thumb:
`model.layers.*` keys pair with `AutoModelForCausalLM`; `model.language_model.layers.*`
keys pair with the `*ForConditionalGeneration` class. After loading, confirm the adapter
actually applied: outputs (or a probe metric) must differ from the bare base model.

The rest of this file is about doing the above _well_ — designing a reward that teaches,
and deciding honestly whether a run improved.

---

## The loop

Work in tight, attributable iterations. Each one is a hypothesis:

```
1. Reconstruct state — what's the best run so far, and what have you already tried?
2. Form a hypothesis — pick ONE lever and say WHY it will move the metric.
3. Change that ONE lever.
4. Validate with `flash train configs/sft.toml --dry-run`. This server-side preview checks
   config/schema, model+algorithm compatibility, LoRA rank, runtime-secret presence, warm-start
   source, serving context cap, and cost for free, with no GPU or charge. It does not import or run
   `environment.py`; dataset loading, episode shapes, reward/scorer, worker imports, model load, and
   GPU/training are first exercised on the worker after cold-start.
5. Submit — `flash train configs/sft.toml`.
6. Judge — read the metric trend AND a sample of real rollouts (see below).
7. Keep the best run; revert the change if it didn't beat the noise band. Repeat.
```

**Lever priority (highest impact first):** reward design → data / curriculum →
training knobs. The reward is the teacher; spend your effort there before touching
hyperparameters.

**One controlled change at a time.** Bundling changes makes the effect
unattributable. Never re-run a setting that already failed at a negligibly
different value.

---

## Before you trust a run — the checklist

A run is only evidence of improvement when **all** of these hold:

- [ ] The run reached `done` (confirmed via `flash runs status <run-id>`), not merely submitted.
- [ ] The SFT loss fell or the reward trend rose (GRPO `reward`) — **beyond the noise band**, not within it.
- [ ] You **probed the trained adapter on real inputs** (`flash models deploy` + `flash models chat`), including cases it should fail — not just the metrics.
- [ ] The score is real behavior, not empty/truncated/templated outputs, skipped rows, leakage, a swallowed exception, or a format-only win.
- [ ] If you track a clean success signal separately from the shaped reward (an explicit `RewardMetric`), _that_ moved too.

If any box is unchecked, the run is not done improving — keep training, don't declare success.

---

## Common Flash issues and mitigations

Most bad Flash runs fail in a small number of predictable ways. Check these before
spending another GPU run:

| Issue                                                        | Symptom                                                                                                                                                                                  | Mitigation                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Environment id is blank or stale                             | A blank id fails config validation. A stale published id can pass `flash train --dry-run` because dry-run does not import or run `environment.py`; the worker then uses old reward/data. | Run `flash env push --project <project-uuid> --name my-env .` after every environment/data edit and paste the returned id into every config you submit.                                                                                                                                                                                                                                                                         |
| Local-only env path in config                                | Config validation says there is no local path mode                                                                                                                                       | Publish first, then use the returned slug in `[environment] id`. `flash train` only runs published env ids, not local paths.                                                                                                                                                                                                                                                                                                    |
| Config knobs are in the wrong table                          | Validation rejects `[grpo]`, `[sft]`, or unknown `[train]` keys                                                                                                                          | Put `epochs`, `group_size`, `max_completion_tokens`, `temperature`, `max_context_tokens`, LoRA, and other training knobs under `[train]`.                                                                                                                                                                                                                                                                                       |
| GPU selection is not what you expected                       | Leaving `[gpu] type` unset may select a different fitting class as prices or capacity change                                                                                             | Set `[gpu] type` to an active validated class to hard-pin it, or leave it unset for managed cheapest-fit allocation. `train.hf_repo` and `model_policy` remain platform-managed.                                                                                                                                                                                                                                                |
| Secrets are not available on the worker                      | Reward code works locally but remote logs show missing API keys or auth failures                                                                                                         | List secret names under `[environment] secrets = [...]`, export those env vars locally before submit, or put them in local `.env` / `.env.local`. Never put secret values in `[worker_env]` or hard-code them in the config.                                                                                                                                                                                                    |
| Wrong model / thinking setting                               | Config validation fails, or chat behavior does not match the run                                                                                                                         | Config validation is authoritative for model and thinking compatibility. Thinking is a run-level choice, and `flash models chat` does not expose an override flag.                                                                                                                                                                                                                                                              |
| Thinking reward grades the wrong text                        | Rewards accidentally score hidden reasoning, or ignore reasoning you meant to inspect                                                                                                    | By default, score the answer text. In thinking mode the response object is still string-compatible, but also exposes `.completion`, `.thinking`, and `.raw` when a reward intentionally needs those fields.                                                                                                                                                                                                                     |
| All-zero or flat GRPO reward                                 | `reward` stays near 0 and outputs do not improve                                                                                                                                         | Make the reward dense: give partial credit for parse/format/execution/correctness tiers, and log a separate clean `success` metric. Do not keep rerunning an all-zero reward.                                                                                                                                                                                                                                                   |
| Reward rises but behavior is worse                           | Short, templated, malformed, or reward-hacked outputs score well                                                                                                                         | Deploy the adapter and probe real examples. Add hard validity gates before judge calls, penalize degenerate shortcuts, and judge the outcome rather than the surface string.                                                                                                                                                                                                                                                    |
| OPD makes the student worse, not better                      | The distilled adapter scores _below_ its SFT/base start even though the per-token loss fell                                                                                              | The teacher, not a knob, is the ceiling. Reverse-KL only pulls the student toward the managed GLM-5.2 teacher, so a teacher that is weak or wrong on _your_ task transfers its mistakes. Vet the teacher before you spend — the "On-policy distillation" section gives the two ways to do that with a managed key. If it can't beat your student, use GRPO or SFT instead — OPD cannot exceed a teacher that can't do the task. |
| Output is truncated                                          | Correct-looking answers cut off mid-response or JSON is incomplete                                                                                                                       | Increase `max_completion_tokens` for GRPO/OPD rollouts or `max_context_tokens` for total prompt+completion context only after seeing truncation. Oversizing them by default just burns memory/cost.                                                                                                                                                                                                                             |
| Infrastructure, CUDA, OOM, vLLM, or kernel failure           | Run errors before useful metrics, often during setup/model load                                                                                                                          | Treat this as infrastructure pressure, not proof the model is too large. Read `flash runs log <run-id>`, reduce footprint (`max_context_tokens`, `max_completion_tokens`, `group_size`) if needed, and let Flash retry/allocate another fitting GPU class.                                                                                                                                                                      |
| Run looks stuck after disconnecting                          | Terminal stopped streaming but the job may still be alive                                                                                                                                | Ctrl-C detaches. Use `flash runs log <run-id> --follow` to reattach, `flash runs log <run-id>` for the console/error output, or `flash runs cancel <run-id>` if you intentionally want to stop it.                                                                                                                                                                                                                              |
| Two runs training the same thing                             | `flash runs list` shows a live run you thought you had killed, on top of the one you just submitted                                                                                      | Killing the `flash train` client only detaches; it never cancels. Always `flash runs list` and cancel the stale run _before_ re-submitting, and never wrap `flash train` in `timeout`.                                                                                                                                                                                                                                          |
| Wrong identity spends the money                              | Push/train lands in an org you did not expect, or a run id you know is valid comes back "unknown"                                                                                        | `FREESOLO_API_KEY` in the environment silently overrides `flash login`. Run `flash whoami` first. Run visibility is scoped to the key that created the run, so archive result baselines to disk rather than relying on `flash runs list` to find them later.                                                                                                                                                                    |
| `flash env test` passes but the run trains on the wrong data | Local validation is green; the remote run loads a different split                                                                                                                        | `flash env test` loads `environment.py` with **no** `[environment.params]`, so a params-driven split selection is not exercised. Assert the split inside your own test, or default to the split you actually train on.                                                                                                                                                                                                          |
| Final checkpoint regresses                                   | Last step is worse than an earlier checkpoint                                                                                                                                            | Run `flash runs checkpoint <run-id>`, deploy a specific step with `flash models deploy <run-id>/step-N`, and compare with held-out probes before exporting or relying on the final adapter.                                                                                                                                                                                                                                     |
| Export fails before upload                                   | CLI says no HuggingFace token                                                                                                                                                            | Pass `flash models export --api-key hf_...`, or set `HF_TOKEN` in your shell, `.env`, or `.env.local`. Exports are private unless you pass `--public`.                                                                                                                                                                                                                                                                          |
| Exported adapter is a silent no-op locally                   | peft warns about missing adapter keys and local eval matches the bare base model                                                                                                         | The adapter's key namespace does not match the loaded model class. `model.layers.*` keys pair with `AutoModelForCausalLM`; `model.language_model.layers.*` keys pair with `Qwen3_5ForConditionalGeneration` / `Qwen3_5MoeForConditionalGeneration` (via `AutoModelForImageTextToText`). See "Loading an exported adapter locally".                                                                                              |
| SFT loss improves but quality does not                       | Train loss falls while held-out behavior stalls or degrades                                                                                                                              | Keep a held-out split outside training. Deploy and score that split; if quality drops, reduce epochs or improve data instead of adding more passes.                                                                                                                                                                                                                                                                             |
| Cost surprises                                               | A quick experiment uses more GPU time than intended                                                                                                                                      | Start with `--dry-run` and `--cost`, keep `epochs` and `max_examples` small for smoke tests, and scale only after reward/data wiring is proven. Setup time is reported for observability; customer cost is based on training-loop GPU time.                                                                                                                                                                                     |

---

## Judge the run, don't just finish it

- **Judge the trend, not a single number.** The proof of training is the curve:
  loss falling (SFT) or `reward` rising over steps (GRPO). Record the base/early
  value and the final value. A flat or noisy trend with no improvement is not success.
- **Read the model's outputs, not just the metrics.** A rising reward (or falling loss)
  can come from reward-hacking or a degenerate output the metric still credits — metrics
  alone never establish that the model got better. For GRPO and OPD, `flash runs log` surfaces a
  handful of full (untruncated) sample completions at the heartbeat cadence — GRPO shows
  each completion's reward, OPD its distillation loss — with the first sample-bearing update
  forced through, so you can catch skipped reasoning or a parroted prompt placeholder by
  step 1-2. These are bounded diagnostics, not every rollout or a held-out
  evaluation, so still **deploy the adapter and probe it**: `flash models deploy <run-id>` then
  `flash models chat <run-id> -m "..."` on at least a few real inputs, including ones it should
  get wrong.

  ```bash
  flash runs status <run-id>            # state + accrued cost
  flash runs log <run-id>               # metric trend + worker console/error logs (+ traceback)
  flash runs log <run-id> --follow      # stream a live run until completion
  flash models deploy <run-id>            # serve the adapter, then `flash models chat` it to read real outputs
  ```

- **Decide with the noise band — and size it for a _difference_.** Record the eval-split
  size `N` and the metric's sampling noise, then treat a gap _inside_ the band as **no
  change** — neither improvement nor regression. A within-noise gain is not a win. But use
  the right band: `1.96·√(p(1-p)/N)` is the error bar on **one** measurement, and comparing
  two runs involves two noisy measurements, so the band for their difference is wider.

  - _Two runs evaluated separately:_ `1.96·√(2p(1-p)/N)` — a factor of √2 ≈ 1.41 wider. Using
    the single-run band here is not a rounding error: simulating two identical models at
    `N = 200, p = 0.6`, a pure-noise gap clears the single-run band **~17%** of the time
    versus ~5% for the correct one. That is one in six "improvements" being nothing at all.
  - _Both evaluated on the same rows_ (the usual case, and the better one): compare **per-row**
    and use a paired method — a paired bootstrap over rows, or McNemar on the
    one-right/other-wrong counts. Pairing cancels the shared row-difficulty noise, so it
    resolves smaller real differences than either unpaired band.

  Either way, the independent unit is the **row**, not the sample. With `k` samples per row,
  extra samples do help — they average out the model's own sampling noise — but only down to
  a floor set by how much the rows themselves differ. Simulating `N = 200` rows: going from
  `k = 1` to `k = 16` shrinks the metric's sd from `0.035` to `0.009` when every row is
  equally hard, but only from `0.035` to `0.018` once rows vary in difficulty (sd `0.25`).
  The remaining half is between-row variance, and no amount of resampling the same 200 rows
  removes it. So `k` buys you a bounded improvement; **more rows** is what actually buys
  precision. To use the `k > 1` gain rather than throwing it away, resample _rows_ (a cluster
  bootstrap: draw rows with replacement, keep each drawn row's `k` samples together) instead
  of assuming the simple `√(p(1-p)/N)` forms above.

---

## Reward design (GRPO) — your highest-impact lever

The reward defines what the model learns; its quality sets the ceiling on what GRPO
can reach. Rewards are rubric / `score_response` functions in your `environment.py`.

### Make it graded and dense — avoid the all-zero cold start

If `reward` is flat at ~0.000, every rollout in the group scored the same, the
advantage is zero, and the policy gets **no gradient**. That is a reward-design bug,
not a model to keep training. Reshape the reward to credit **ordered partial
progress** so even an untrained base model earns a small nonzero score and better
attempts score strictly higher:

```text
well-formed / parses → schema- & safety-valid → executes / runs → correct / relevant
```

Gate only the **top** tiers against gaming; keep the lower tiers dense. GRPO needs
_within-group variance_ to learn — if every rollout in a group scores identically,
there is nothing to optimize.

### Separate the shaped reward from a clean success signal

A good GRPO reward is usually **shaped** — partial credit so the model always has a
gradient to climb. But a shaped score is the wrong thing to judge _final quality_ on:
it can rise from reward-hacking while the outcome you care about stays flat. Report the
shaped value as `score`, and surface the clean pass/fail as an **explicit
`RewardMetric`** so it shows up in the run's metric breakdown — a bare `threshold` is
used for grading but is _not_ logged on its own, so it gives you nothing to judge:

```python
from freesolo.environments import RewardResult, RewardMetric

def score_response(self, example, response_text) -> RewardResult:
    score = graded_score(example, response_text)         # shaped 0-1 — what GRPO optimizes
    return RewardResult(
        score=score,
        threshold=1.0,                                   # success = score >= threshold
        metrics=(RewardMetric(name="success", score=float(score >= 1.0)),),  # logged: judge on this
    )
```

`score` is what GRPO optimizes (it becomes the run's `total`). In standard (single-turn)
GRPO, each `RewardMetric` is averaged across scored completions and logged by name at
the managed heartbeat cadence, which is not guaranteed to be every optimizer step. That
is how the clean success rate becomes visible. Multi-turn scoring currently reports only
the scalar reward. Use the shaped `score` to confirm the model is learning _at all_, and
judge the run on the explicit `success` metric.

When `thinking = true`, score the final answer unless you intentionally need the
reasoning trace. Flash passes a string-compatible response object to `score_response`;
`str(response_text)` is the answer text, while `response_text.completion`,
`response_text.thinking`, and `response_text.raw` are available for rewards that
explicitly inspect the separated completion, reasoning, or original raw model output.

### Reward rules that prevent silent failure

- **Return `0.0` explicitly — never let scoring raise.** An uncaught exception in
  scoring fails the whole run. Guard every parse and lookup and return
  `RewardResult(score=0.0, error=...)` for missing evidence, a parse failure, or an
  unsafe/unsupported output.
- **Gate LLM judges behind the hard checks.** Run deterministic validity checks first
  and return `score=0.0` on any parse/schema/safety failure, so the policy can't
  reward-hack a lenient judge with malformed-but-plausible text.
- **Judge the realistic outcome, not the raw string.** Give a judge the runtime
  output, tool result, or executed-query records. For database / search / retrieval
  tasks, grade the _returned records_, not the query text — the query is only
  secondary validity evidence.
- **A small format penalty beats a hard zero for shaping.** A useful trick:
  `reward = format_coef * (correct_format - 1) + correct_answer` with `format_coef≈0.1`
  — a tiny penalty for bad formatting, full credit for a correct, well-formatted answer.
- **Anti-patterns.** Don't reward length or verbosity. Don't ship a reward that is
  always 0 or always 1 (no signal). Simpler rewards usually beat clever ones — a
  mediocre _stable_ reward beats a "perfect" reward you keep tweaking. Changing the
  reward resets progress, so keep the best checkpoint before you do.

---

## SFT conventions

Pick SFT when you already have good answers and want the model to imitate them.

- **Data quality is the ceiling.** SFT can only be as good as the answers you show it.
  A small set of high-quality examples beats a large mediocre one. Keep response format
  consistent (if you want JSON, _every_ example is JSON) and keep the prompt format the
  same as inference time.
- **Watch the loss fall — and check overfitting yourself.** Flash SFT logs **training
  loss only**; it runs no mid-training held-out eval (evaluation is deferred to the
  deploy/serving side). A falling train loss alone can be memorization, so keep an eval
  split the run never trains on, then **deploy the adapter and score it on that split**
  (`flash models deploy` + `flash models chat`). If held-out quality stalls or drops while train loss
  keeps falling, reduce `epochs` or add more data — not more passes.
- **Start `max_context_tokens` small and grow it on evidence.** Begin from the smallest
  `max_context_tokens` that plausibly fits prompt + completion, and only raise it when you see
  truncation (outputs cut off mid-thought, degraded loss). A bigger context just costs
  more.
- **For Qwen3.5 thinking multi-turn SFT, put reasoning only in the final assistant
  turn.** Qwen3.5's chat template strips literal `<think>` blocks from prior assistant
  history and pre-opens `<think>\n` in the next generation prompt. If every assistant
  turn in a gold multi-turn transcript includes `<think>...</think>`, training sees a
  different tag layout than inference and can learn doubled or misplaced thinking
  tags. Keep intermediate assistant turns as the actual code/tool/action text only;
  put `<think>...</think>` plus the final answer in the final assistant target. Flash's
  completion-only SFT masking uses the longest shared token prefix, so the template's
  pre-opened `<think>\n` is treated as prompt text instead of training the model to
  emit another opener.
- **SFT is a great warm start for GRPO.** SFT first to teach the format and a competent
  baseline, then GRPO to optimize past it. Across that lineage keep the **same base
  model**. Warm-start CONTINUES the one SFT adapter in place — GRPO/OPD keep training
  the same LoRA (VL and text-only alike), so the run trains and serves at the SFT
  adapter's rank-`r` and just has to fit the selected model's serving `max_lora_rank` (some
  serving models allow rank 128, larger serving paths cap at 64). Do **NOT** set `lora_rank`
  for a warm-start: the source adapter's rank/alpha metadata is authoritative. Flash reads the
  rank from the source adapter and uses it for cost, GPU allocation, and GRPO-sleep sizing, so
  setting `lora_rank` alongside `init_from_adapter` is rejected at submit; it also rejects a
  source adapter whose rank exceeds the serving cap.

```toml
# configs/rl.toml — warm-start GRPO from the SFT run's adapter
algorithm = "grpo"

[train]
# the sft run id (as printed by `flash runs status`); add /step-n to warm-start from a
# specific checkpoint listed by `flash runs checkpoint <run-id>`
init_from_adapter = "<sft-run-id>"
# do NOT set lora_rank for a warm-start: the source adapter's rank and alpha metadata are
# authoritative (lora_alpha is always managed as 2 x rank), and setting lora_rank alongside
# init_from_adapter is rejected
```

SFT, GRPO, and OPD all accept **epoch-driven** configs (`epochs`). For GRPO/OPD,
an epoch is one pass over the retained prompt pool after `max_examples` and prompt-budget filtering;
optimizer-step counts are derived from those epochs. A positive `[train] max_steps` replaces that
derived count with an exact update horizon for every algorithm. `[train] save_at_steps`
requires a positive `max_steps` so its horizon is authoritative even when SFT packing changes the
realized batch shape. When non-empty, exact save steps suppress periodic `save_every` checkpoints, and the
run fails if a requested exact save cannot be saved and published.

---

## On-policy distillation (`algorithm = "opd"`)

Pick distillation when a much stronger **teacher** model can grade your student's work
token-by-token. The student samples on-policy (like GRPO), a managed teacher (GLM 5.2 by default,
or another via `[train] teacher_model`) scores each of the _student's_ completions, and a dense per-token
loss teaches the student to match the teacher — far more sample-efficient than reward-based RL and
with no reward to design. It supports `epochs` like SFT/GRPO and produces a LoRA served exactly like SFT.

- **Vet the teacher on your task before you distil — this is a precondition, not a formality.**
  Reverse-KL can only pull the student _toward_ the selected teacher, so OPD's ceiling is
  roughly the teacher's own competence at your task. If the teacher is weak, frequently wrong, or
  solves the task with a strategy your environment can't reward, distillation faithfully transfers
  those flaws and **drives the student _below_ its SFT/base starting point instead of above it** — a
  low per-token loss just means it matched a bad teacher. Only run OPD when the teacher clearly
  beats your student (and your target bar) _and_ solves the task the way you want the student to. When
  the teacher is at or below your student on the task, OPD is the wrong tool — reach for GRPO
  (reward-driven, can exceed any single teacher) or SFT on curated data instead. `[train] teacher_model`
  lets you pick the teacher that best fits your task without changing anything else.

  **How to check, given the key is managed.** Flash has no command that generates teacher
  rollouts — the platform's Fireworks key is used only inside the paid OPD worker, to score the
  student's own tokens — so there is nothing to run locally with your Flash credentials alone.
  Two workable routes:

  - _Best, if you can:_ get your own access to the same model (the allow-list is Fireworks-hosted,
    and these are widely available elsewhere too), point your environment's scorer at it, and grade
    a held-out split exactly as you would a student. Read the score _and_ a sample of trajectories —
    a teacher that is right for the wrong reasons transfers the wrong reasons.
  - _Cheapest, always available:_ run a deliberately short OPD probe (a small `max_steps` on a
    slice of the data), deploy that adapter, and evaluate it against the same student baseline. A
    short run that moves the student the wrong way is a warning, not a verdict — trajectories dip
    before recovering, and a slice need not represent the eval distribution. Treat it as a signal
    to spend a little more evidence, not a lot more steps: evaluate two or three checkpoints from
    that probe on the full split before deciding. A dip that deepens across checkpoints is the
    teacher or the algorithm; a single down reading is noise until it repeats.

    **Schedule those checkpoints before you submit, or they will not exist.** OPD saves every
    **20** optimizer steps by default, so a probe short enough to be cheap (under 20 updates)
    publishes only its final step and there is nothing to compare. Name the steps explicitly:

    ```toml
    [train]
    max_steps    = 12
    save_at_steps = [4, 8, 12]   # exact steps; overrides the periodic save_every entirely
    ```

    Every entry must land within `max_steps`, and an out-of-range one is **rejected, not skipped**:
    config validation fails the submit outright, and the worker re-checks it again before training
    starts. So a stale entry left over from a longer run does not quietly cost you one checkpoint —
    it stops the run from starting.

- **Pick the teacher with `[train] teacher_model`; the key stays managed.** The teacher defaults to
  the managed **GLM 5.2** and is selectable from a fixed, managed allow-list:
  `glm-5.2` (default), `deepseek-v4-pro`, `kimi-k2.6`. Every option is
  a Fireworks-hosted model reached with the platform's own key, so there is nothing to export or
  declare — an opd run submits like any other, and a `FIREWORKS_API_KEY` in your shell is ignored.
  Arbitrary bring-your-own teacher models or keys are not supported (the allow-list is curated to
  teachers verified to echo-score the student's tokens). The key is never stored in the spec or needed
  at serving time; teacher token cost varies by model and is shown in the pre-flight estimate.
- **The student (Qwen) and the teacher have different tokenizers.** Flash
  bridges the vocabulary mismatch with **groupwise reverse-KL** (the collinear-ai _spider_ / Tinker
  method): it aligns the two tokenizations by shared decoded-text spans and applies per-span reverse
  KL using only realized-token logprobs — no vocabulary projection, so it covers every token exactly
  and works for any student tokenizer. When the tokenizers happen to agree it reduces to plain
  per-token reverse KL (Thinking Machines, _On-Policy Distillation_). Nothing to configure.
- **Works for multi-turn envs too.** Against an `EnvironmentMultiTurn`, opd rolls out each episode
  (driving `step_episode` / observations just like GRPO) and distils EVERY assistant turn against the
  teacher, each conditioned on the transcript up to that turn — the episode's total reverse-KL over
  the student's generated tokens is the sum of its per-turn reverse-KLs. Env/observation tokens are
  never distilled (they're context, not the student's output). Set `[train] max_context_tokens` to bound the
  transcript; the teacher must cover it (the allow-listed teachers' contexts far exceed the default budget).
- **Judge it like SFT.** Distillation logs a falling per-token loss; a low loss alone is not proof.
  Keep a held-out split, `flash models deploy` the adapter, and score it — confirm the student actually
  moved toward the teacher's behavior, not just its surface tokens.

```toml
# configs/opd.toml
model = "Qwen/Qwen3.5-4B"
algorithm = "opd"

[environment]
id = "your-org/my-env"

[train]
epochs = 1
max_examples = 2
lora_rank = 32
# teacher_model = "glm-5.2"                             # managed teacher to distil from; one of
#                                                       # glm-5.2 (default) | deepseek-v4-pro | kimi-k2.6
#                                                       # (key stays managed)
# kl_penalty_coef = 1.0                                 # reverse-KL scale
```

The cross-tokenizer reverse-KL is computed over shared decoded-text spans and so **cannot supervise
the zero-width stop token**. No auxiliary EOS loss is applied. `truncated_rollouts` records completions
that reached the length cap without EOS or a configured stop. Warm-starting from an SFT adapter can
still improve initial termination behavior.

A verbose teacher can also inflate the _content_ the student distils toward through long per-turn
reasoning and extra multi-turn looping, so episodes still forfeit against the length/turn budget
regardless of stop-token behavior. Because the teacher scores the
student's rollouts **conditioned on your environment's own system prompt**, you can shrink its target
distribution at the source: give the prompt used for OPD rollouts a **hard, specific reasoning
budget** — e.g. "reason in at most two or three sentences, then act; once you have started, do not
reconsider" — rather than a vague "be brief." The phrasing matters. A soft brevity request can
**backfire on a thinking teacher**, trimming the median while _inflating_ the long tail (the model
spirals when told to be brief on a problem it finds hard), which is exactly the tail that drives
runaway. Constrain the content with the prompt and monitor `truncated_rollouts` for length-cap
failures. This assumes the teacher is still strong at the task (vet it first, above).

### Reverse-KL over-sharpens — cut steps and watch entropy (every model)

Reverse-KL is **mode-seeking**: it sharpens the student's next-token distribution toward the
teacher's dominant mode, and it keeps sharpening for as long as you train. This affects **every OPD
run, at every size** — the whole Flash catalog is small by frontier standards (0.8B-9B dense plus a
3B-active MoE), so treat over-sharpening as a default risk, not a small-model edge case. The student's
per-token entropy falls as training proceeds; past the point where it has learned the task, extra
steps only over-sharpen — _lowering_ accuracy. Push it far enough and the distribution peaks so hard
that **greedy (temperature=0) decoding falls into a repetition loop** that repeats a phrase to the
length cap and never emits your answer. The loss looks healthy the whole run (reverse-KL is being
minimized _by_ the collapse), so it is invisible in the loss curve and only surfaces at serving —
where **temperature=0 is the default**, so it hits real callers, not just a sampled eval.

**Severity scales with size**: on the largest catalog models over-training mostly just leaves
accuracy on the table (a late checkpoint slightly worse than an earlier one); on the smallest it
turns into the full-blown greedy loop. But the fix is the same everywhere, and cutting steps helped
_every_ size tested. Four levers, each attacking the same over-sharpening — the first two apply to
every run, the last two matter more the smaller the model:

- **Train fewer steps (highest leverage, every size).** The student typically peaks early — often
  around ~20 optimizer steps — and every step after is pure over-sharpening that _lowers_ accuracy
  while _raising_ the loop rate. **Which knob to cut depends on whether your config sets `max_steps`.**
  A positive `max_steps` is the authoritative update count (`resolve_update_horizon()` returns it and
  ignores the derived horizon), so with one set, cutting `max_examples` or `epochs` does not shorten
  training at all — you would rerun the identical overlong horizon at the same cost. Lower or remove
  `max_steps` instead. Only when the horizon is _derived_ (no `max_steps`, or `max_steps = 0`) does
  cutting `max_examples` (or `epochs`) stop the run before the collapse. Either way,
  deploy an early **checkpoint** (`flash runs checkpoint <run>`, `flash models deploy <run>/step-N`) rather
  than the final adapter. This helped at every size tested — a 4B went 42% acc / 44% loop at full
  length -> **74% / 0% at step 20**, and even models that never looped came out equal-or-better at the
  earlier checkpoint. When in doubt, sweep a few checkpoints and pick the best, don't assume the last
  step is the best.
- **Lower the rank (more, the smaller the model).** A rank-32 adapter is a large relative perturbation
  to a small model, giving reverse-KL more capacity to over-sharpen. Dropping `lora_rank` to 16 (or 8)
  often clears the loop outright (a 4B sft->opd went 42%/44% at rank 32 -> 76%/2% at rank 16). Since
  the whole catalog is small, prefer a modest rank (16) as the default for OPD and only raise it with a
  reason.
- **Match the teacher to the student.** A _stronger_ teacher is not universally better — the harder
  it is for the student to match, the harder the collapse. On a 2B, a closer/weaker teacher can beat a
  frontier one outright; a frontier `teacher_model` only earns its keep once the student is large
  enough to track it (~9B+). Early-stopping also largely neutralizes this gap, since the teacher-driven
  over-sharpening only compounds over many steps.
- **Diagnose it at serving.** Evaluate at **temperature=0** and flag
  `finish_reason=length` completions that never emit your answer token. Compare an early checkpoint
  against the final one to watch the loop emerge over steps.

### Distilling from base with no format anchor

`opd` straight from a base model (no SFT warm-start) faithfully distils the teacher's _reasoning_ but
the student never learns your **answer format** — it terminates (`finish_reason=stop`) without ever
emitting the boxed/tagged answer, so completions score unparseable even when the reasoning is fine.
On-policy distillation reinforces the student's _own_ tokens, so if the base never produces the
format there is nothing to reinforce (and a downstream GRPO pass can't rescue it — with no
correctly-formatted rollout to reward, RL has no signal to climb). Two fixes:

- **Warm-start from an SFT adapter** (`[train] init_from_adapter`) — the SFT installs the output
  format first, then OPD refines the content. This is the reliable default for structured-answer tasks.
- **Constrain the rollouts with `[train] structured_outputs`** (guided decoding) so the answer is
  forced into your schema instead of being left to the base model's habits. This fixes the _format_
  problem only; the _loop_ problem is fixed by the levers above.

  Field order in that schema is a `thinking = false` lever, not a rescue for runaway reasoning.
  Under `thinking = true` the grammar is deliberately held until the model emits `</think>`, so it
  constrains only what comes after the reasoning phase — putting `answer` first cannot make the
  answer precede the thinking, and if reasoning never terminates the answer is never generated at
  all. For thinking runs, treat a length cap or a reasoning loop as a budget problem and reach for
  the SFT warm-start and the prompt/`max_completion_tokens` levers above.

---

## GRPO knobs that matter

Set these in `[train]`. Each is `None` by default — the worker's tuned recipe fills
in a sensible value, so only override with a reason.

| Knob                           | Convention                                                                                                                                                                                                                                                                                                                                                                                                        |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `group_size`                   | Completions sampled per prompt (default 8). More = more signal and more cost; drop to 4 to trim cost. The group needs _within-group variance_ for an advantage to exist.                                                                                                                                                                                                                                          |
| `max_completion_tokens`        | Completion budget per rollout. Size it to the expected output length; too small silently truncates good answers and poisons the reward, while too large just costs more.                                                                                                                                                                                                                                          |
| `temperature`                  | Rollout sampling temperature. Keep it near 1.0 for GRPO — too low collapses diversity (and the model can collapse within a few steps); raise it to widen exploration against uniform-reward groups.                                                                                                                                                                                                               |
| `kl_penalty_coef`              | Keeps the trained model from drifting too far from the base. Raise it to anchor against entropy collapse; lower it for more freedom to move.                                                                                                                                                                                                                                                                      |
| `thinking_length_penalty_coef` | Per-reasoning-token reward deduction — curb overthinking, but watch it doesn't push the model into terse degeneracy.                                                                                                                                                                                                                                                                                              |
| `learning_rate`                | Change it in small steps. Too high destabilizes RL and degrades output quality; if the model is collapsing, lower it.                                                                                                                                                                                                                                                                                             |
| `batch_size`                   | The effective prompts-per-step. Too small and the reward trend is pure noise; size it so the trend is readable.                                                                                                                                                                                                                                                                                                   |
| `structured_outputs`           | Guided decoding for every GRPO/OPD rollout: a JSON schema (inline table or JSON string), `regex`, or `choice`. The sampler then _cannot_ emit off-format text, so the reward measures content instead of formatting. Works with `thinking = true`: the grammar is held until the `</think>` boundary (via a reasoning-aware decoding gate), so the model reasons freely first and only its answer is constrained. |

For thinking models, `max_completion_tokens` is shared between `<think>` reasoning and the final
answer or action, so undersizing it can truncate the action and teach the model to stop reasoning;
watch `truncation_rate`, which counts completions not ending in EOS and is not strictly
`finish_reason=length` when stop sequences or multi-turn rollouts are involved.

> **On a derived horizon, `batch_size` buys optimizer steps cheaply.** `batch_size`
> overrides the tuned prompts-per-step. Total generated tokens are
> `steps x prompts_per_step x group_size x max_completion_tokens`, so when the step count
> is _derived_ from the data, lowering `batch_size` raises it and leaves the token bill
> roughly flat — the spend that buys 5 steps at the default batching buys ~80 at
> `batch_size = 4`. A single-digit-step run is weak evidence either way: usually too few
> updates to show that a setup works, but — as the `temperature` row above notes — not too
> few to destabilize one. Read a flat result at 5 steps as "not measured yet" rather than
> "no effect," and check your batching before concluding you are under-funded. When you do
> lengthen the horizon, evaluate the checkpoints along the way rather than only the last
> one; jumping straight to ~80 steps without reading the short run trades one uninformative
> run for a longer, more expensive one.
>
> Two things break the flat-cost approximation, so treat it as a hypothesis to check, not a
> rule. **`[train] max_steps` overrides the derived horizon**: with it set the step count is
> exactly what you asked for, lowering `batch_size` does not add steps at all, and you
> simply train on fewer prompts. And per-step cost is not purely token-proportional —
> GRPO reward waves and OPD teacher calls add latency per step, so multiplying steps can
> multiply that overhead into a materially larger GPU bill. Smaller batches also mean
> noisier per-step gradients. Compare concrete `--cost` estimates before and after the
> change and let those numbers, not the ratio, decide.

For pure multi-turn GRPO, Flash gives each Flash-owned vLLM generation request a managed
10-to-60-minute absolute deadline and at most two physical attempts. A timed-out request is
aborted before retry. Enforcement is cooperative between engine polls, and there is no total
episode elapsed-time cutoff. This policy does not time out OPD, TRL-native tool loops, or
environment calls.

> **The reward-hacking signature:** a smoothed reward rising while mean generated
> length collapses. Whenever any shortness or format pressure is active, verify the
> gate by scoring a few truncated or opener-only probe responses — they should score low.

---

## Multi-turn environments — two silent traps

### `credit_assignment = "per_turn"` is inert unless your environment emits `per_turn_rewards`

**First, check that per-turn credit is available to you at all.** It is supported on the
pure multi-turn rollout path — an `EnvironmentMultiTurn` that drives its own turn loop. A
multi-turn environment that exposes **tools** runs on TRL's native tool loop instead, and
there `per_turn` is not implemented: the worker raises
`credit_assignment='per_turn' is not supported for tool-calling multi-turn environments`
and the run dies at startup. Nothing catches this earlier — `flash train --dry-run` accepts
the config, because the trainer choice depends on the environment object, not the spec. If
your environment exposes tools, stay on `per_episode`; emitting `per_turn_rewards` does not
unlock it.

On the pure multi-turn path, setting `credit_assignment = "per_turn"` is only _half_ the
request. The trainer reads the
turn-level signal from `RewardResult.metadata["per_turn_rewards"]`; **when that key is
missing it silently falls back to episode-level credit.** The config validates, `flash runs
status` echoes `per_turn` back, and nothing in the CLI, the status output, or the worker log
says the setting was dropped. You get a full-price run of the default scheme wearing the
label of the one you asked for — and the natural reading of that result ("per-turn credit
doesn't help") is a wrong conclusion drawn from a setting that never applied.

Emit the list from `score_episode`, and derive the episode scalar from that same list so the
two cannot disagree:

```python
import math

# exactly one finite reward per GENERATED assistant turn - assert it here, because
# nothing downstream checks the length and both mismatch directions are bad (see below).
# count episode.turns, NOT episode.messages: messages starts out holding everything
# start_episode() returned, so any assistant few-shot demo in the prompt inflates the
# count. turns starts empty and collects only what the rollout actually generated.
assistant_turns = sum(1 for t in episode.turns if t.role == "assistant")
assert len(turn_scores) == assistant_turns, (len(turn_scores), assistant_turns)
assert all(math.isfinite(s) for s in turn_scores)

return RewardResult(
    score=episode_score,                                  # unchanged episode scalar
    metadata={"per_turn_rewards": [float(s) for s in turn_scores]},
)
```

**The length is a hard contract, and it is checked nowhere.** The extractor validates only
that the value is a non-string iterable of floats; it never compares the count to the
rollout's turns. The trainer then walks `range(len(turn_rewards))` and indexes
`spans[turn_index]` for each one, so the two mismatch directions fail differently and
neither is a safe fallback:

- **Too many rewards** — `IndexError: tuple index out of range`, mid-training, after you have
  already paid for the rollouts.
- **Too few rewards** — no error at all. The unmatched turns keep the zero they were
  initialized with, so those tokens train on **zero advantage**: that part of the episode
  silently contributes no learning signal, and the run looks healthy throughout.

The easiest way to get this wrong is the hard turn cap. The rollout loop breaks out before
the final `env_reply`, so the last assistant turn is generated and recorded **without** a
matching `step_episode` call — an environment that appends one reward per `step_episode`
comes up exactly one short on precisely the episodes that hit the cap. Count the assistant
entries in `episode.turns`, not the number of steps you took and not the assistant messages
in `episode.messages` (those include any few-shot demo the prompt carried in).

**Do not verify this by the absence of a warning** — a missing key produces no output at
all. There are three distinct silent-fallback layers on this path, two of which emit
nothing. The positive marker is the one-shot log line `[rl] multi-turn per-turn
group-relative credit is active`, but `flash runs log` returns only a bounded tail window,
so a marker printed early in training scrolls out of reach and its absence proves nothing.
The reliable check is local: feed a `RewardResult` through the extractor and confirm you get
per-turn values rather than `None`.

### Scoring must not depend on state accumulated in `step_episode`

The worker's turn loop breaks out **before** the final `env_reply` when an episode hits the
hard turn cap. An environment that accumulates state inside `step_episode` therefore scores
the second-to-last position on exactly those episodes, silently under- or over-crediting the
model — and only on the capped ones, which is a hard bias to spot in aggregate.

Make scoring a pure function of the transcript: re-derive state from the assistant turns on
every hook call. That is correct whether the episode ended by succeeding, by an env-signalled
done, or by hitting the cap, and it also means concurrent rollouts sharing one environment
instance cannot corrupt each other.

**Measure efficiency against each example's own optimum, not the turn budget.** A reward
like `1 - turns_taken / max_turns` makes perfect play unreachable: an example whose best
solution needs 5 turns caps at `1 - 5/12 = 0.58` even when played perfectly, so "perfect
across the dataset" is undefined and your gold trajectories never score 1.0. Use
`minimal_steps / turns_used` instead — a minimal solve scores exactly 1.0 and detours
degrade smoothly.

**Gate that ratio on success and clamp it.** Applied unconditionally it is an active
hazard, not just imprecise: an episode that gives up after one turn still collects
`minimal_steps / 1`, so on a five-step example quitting immediately scores **5.0** —
strictly better than solving it. GRPO optimises exactly that, and you get a run that
learns to terminate. Score failures at zero, divide only after a solve, and bound the
result to your reward range:

```python
reward = min(1.0, minimal_steps / turns_used) if solved else 0.0
```

Assert both ends in a local test: a gold trajectory scores 1.0, and a one-turn give-up
scores 0.0.

---

## Curriculum — start easy, scale up

Starting too hard produces zero learning signal; the model never succeeds, the reward
stays at 0, and there is nothing to climb. Start where the base model can _partially_
succeed, then raise difficulty as it improves. The "Goldilocks zone" — where most
rollouts score somewhere between all-fail and all-pass — is where GRPO has the most
signal.

- If nearly every prompt is solved (most groups score ~1.0): **increase difficulty** —
  harder prompts, tighter format/reward, more epochs, or more data.
- If nearly nothing is solved (most groups score ~0.0): **decrease difficulty** —
  easier or few-shot prompts, a more lenient (denser) reward, or warm-start with SFT.
- In between: good signal — keep iterating at this difficulty.

---

## Diagnose before you re-run

When the reward stalls, a chunk of outputs fail, or the checkpoint underperforms,
don't treat failures as one bucket. Read a sample of the **actual failing
generations** (raw outputs, not just scores), classify the dominant mode, and apply a
targeted fix rather than leaning on the reward gate to slowly select against it. Then
**re-measure that mode** to confirm it dropped.

| Failure mode                                        | What you see                                                                             | Targeted fix                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Repetition / looping collapse                       | the same phrase repeats until truncation                                                 | repetition or length penalty. Then split the cause before touching `temperature`: if loops appear from the start and the outputs are incoherent, sampling is too hot — lower it. If they _emerge_ over steps while the loss looks fine, this is the entropy collapse described under OPD above, and lowering `temperature` sharpens the dominant tokens further and strips the within-group diversity GRPO needs. Fix that one with fewer steps / an earlier checkpoint, a lower `lora_rank`, or a KL anchor — not with colder sampling. |
| Overthinking / verbose reasoning                    | reasoning eats the whole token budget                                                    | `thinking_length_penalty_coef`; tighten the prompt with a _hard, specific_ budget ("reason in at most N sentences, then act") — a vague "be brief" can backfire on a thinking model and lengthen the tail                                                                                                                                                                                                                                                                                                                                |
| Completion truncation                               | answers cut off mid-thought                                                              | raise `max_completion_tokens` / `max_context_tokens`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| OPD rollouts never stop (high `truncated_rollouts`) | on-policy completions run to the length cap without an EOS; raising the cap barely helps | No auxiliary EOS loss is applied. Warm-start from SFT and shrink what has to terminate: constrain the _teacher's_ reasoning at the source with a hard, specific budget in the env prompt used for OPD rollouts (the teacher scores the student conditioned on it); a vague "be brief" can backfire on a thinking teacher. First confirm the teacher itself terminates and is strong on the task; a bad teacher is distilled in, not out.                                                                                                 |
| Unparsed / over-escaped output                      | reward can't read the answer                                                             | robust parser; return `0.0` on parse fail; format gate                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Wrapper / markdown around structured output         | prose around the JSON/answer                                                             | a format gate; `stop_sequences`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Uniform-reward groups                               | every rollout in a group scores the same → no gradient                                   | shape the reward for partial credit; raise `temperature`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Too-hard prompts                                    | the base never succeeds, reward stays at 0                                               | curriculum / easier prompts; warm-start with SFT                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Judge-rewarded degenerate output                    | short, templated answers a judge still rates well                                        | a minimum-substance zero-gate ahead of the judge                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |

---

## When a run stalls

A plateau is not automatically a capability ceiling. Before you call it one:

1. **Probe with best-of-N.** Run a best-of-N / pass@k probe at a coverage temperature
   (well above greedy) on a less-fitted checkpoint.
2. **Read the result.** High best-of-N but a collapsed greedy output and low sample
   diversity is **entropy collapse**, not a ceiling — and it's fixable: anchor harder
   with `kl_penalty_coef`, lower the `learning_rate`, or widen exploration. Only if the
   probe shows no headroom is it a genuine ceiling.
3. **Change a different lever.** If there's real headroom, try a _different_ lever from
   the one that just failed — a different knob, reward shape, or data family — one
   controlled change at a time.

Actively research established GRPO/SFT techniques (exploration / entropy control, KL
scheduling, reward shaping, curriculum / difficulty filtering, rejection-sampling SFT
on high-reward rollouts) rather than guessing — and count a technique as helpful only
on a beyond-noise improvement.

---

## Scale the evidence

- **A smoke test is not proof.** A single-digit derived step count, a tiny dataset, or a handful
  of rollouts only validates the wiring. Scale `epochs`, the dataset size,
  and `group_size` to the model and the data you actually have before you trust a
  result. Don't cite budget alone as the reason for an underpowered run.
- **Use the data you have.** Deliberately assign every usable row to training or to a
  held-out eval split; if a planned holdout is so small that one example swings the
  metric by several points, enlarge it during split design rather than gating on noise.

---

## Don't let your own harness lie to you

Every item here produced a confident, _wrong_ conclusion that survived until someone went
looking. None were Flash defects — they are analysis bugs, and they are the most expensive
kind because the run itself looks fine.

- **Never join eval results to your dataset by index.** If the harness _samples_ rows
  (`random.sample`, a shuffled loader) the output order is neither a prefix nor file order.
  Zipping `results[i]` to `dataset` line `i` mismatched **78 of 150** rows and produced a
  detailed, entirely wrong root-cause analysis of where the model was failing. Carry an
  explicit row id through scoring and join on it.
- **Check the error count before you read any aggregate.** A harness that records per-row
  errors and still averages over what is left will happily print a number computed from a
  handful of rows. Discard a result file with a non-trivial error count instead of
  interpreting it.
- **Served evals are not reproducible even at temperature 0.** Re-evaluating the same
  checkpoint yields a small spread, so a "best checkpoint" chosen on a gap narrower than
  that spread is choosing noise. Establish the spread by evaluating one checkpoint two or
  three times, then require a gap larger than it. Write each result to a distinct path —
  two jobs writing the same output file silently overwrite each other.
- **Sanity-check a metric against its neighbours.** A metrics dict reading `accuracy: 0.0`
  next to `mean_shaped_score: 0.979` is worth stopping on — but it is not proof of a bug.
  If success requires the shaped score to _reach_ 1.0, then a run of consistent near misses
  produces exactly that pair, and it is real evidence (of near misses, or of reward hacking
  that farms partial credit without solving). Resolve it before acting: check a couple of
  rows individually and see whether the per-row pass/fail agrees with the per-row shaped
  score. Only a row that is marked failed while its own shaped score clears the success
  threshold proves the reader is broken. One common cause of a genuinely broken read here:
  `RewardMetric` carries its number in `.score`, while `.value` is a separate optional field
  that is often `None` — pulling `.value` yields zeros for every row.
- **Verify the split you evaluate is the split you think.** Confirm gold answers score
  1.0 through your own scorer before trusting any number derived from it, and re-check for
  overlap between train and eval rows.
- **Re-grade stored completions instead of resampling** when you fix a parser or scorer.
  That keeps old and new numbers apples-to-apples; resampling confounds the parser change
  with sampling noise. Archive the pre-fix results rather than overwriting them.
- **A judge that is itself a reasoning model breaks naive parsing.** Its `<think>` block can
  consume the token budget and truncate the verdict, and it often floats a candidate score
  mid-reasoning before settling on a different one — so a regex taking the _first_ match
  reads a discarded intermediate as the verdict and mis-grades silently. Strip the reasoning
  block, search the post-reasoning answer, take the **last** match, and give the judge
  enough tokens to finish.
- **State the noise band before you compare.** With a small eval split, differences of a few
  points are not distinguishable from noise; see the noise-band formula above, and treat an
  in-band difference as _no change_ rather than a weak win.

---

## Treat crashes as infra, not model size

> A CUDA / OOM / vLLM / kernel / infrastructure error is an **infrastructure** problem, not a
> sign the model is too big. Lower `max_context_tokens`, `max_completion_tokens`, or `group_size` to shrink
> the run's footprint and let the allocator retry onto the next fitting GPU class — do
> **not** switch to a smaller model to make a crash disappear. That silently destroys
> quality.

---

## Command reference

```bash
flash whoami                          # confirm which identity/org you are about to spend money as
flash env setup                       # scaffold environment.py, dataset/, configs/, this file
flash env test .                      # load + run the environment locally, before any GPU spend
flash env push --project <project-uuid> --name my-env .        # publish the environment; paste the returned id into [environment]
flash env pull your-org/my-env        # download a published environment into the current folder
flash env delete --project <project-uuid> your-org/my-env -y   # delete a published environment
flash train configs/sft.toml --dry-run # validate the config on the server (no GPU, no charge)
flash train configs/sft.toml --cost    # pre-flight USD estimate, then exit
flash train configs/sft.toml           # submit and follow logs (Ctrl-C detaches; --background to skip following)
flash runs status <run-id>                 # state + accrued cost
flash runs log <run-id>                    # reward/loss trend + worker console/error logs
flash runs log <run-id> --follow           # stream a live run to completion
flash runs list                       # list your runs and their state/cost
flash runs cancel <run-id>                 # stop a live run
flash runs checkpoint <run-id>            # list deployable RL checkpoints
flash models deploy <run-id>                 # serve the trained adapter
flash models deploy <run-id>/step-N          # serve a specific RL checkpoint
flash models chat <run-id> -m "probe"        # stream a reply from the deployed adapter
flash models deployments                     # list active serving deployments
flash models undeploy <run-id>               # tear down an active deployment
flash models export --adapter-id <run-id> --repository <you>/<repo>  # export final adapter
flash models export --adapter-id <run-id>/step-N --repository <you>/<repo>  # export a checkpoint
```

See the full reference at https://docs.freesolo.co.
