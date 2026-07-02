"""The TRAINING.md playbook scaffolded by `flash env setup`. Update it here, not in a copy."""

from __future__ import annotations

TRAINING_MD = r"""# TRAINING.md — how to actually improve a model with Flash

> **If you are an AI agent asked to train a model here, read this first.**
> `flash env setup` dropped this file next to your `environment.py` and `configs/`.
> It is the playbook Freesolo's own training agents follow to turn a *finished*
> run into a model that *measurably improved*. The mechanics live in the hosted
> docs (https://freesolo.co/docs); this file is the judgment that sits on top of them.

A run that reaches `done` is **not** the same as a run that worked. Submitting a run
is not a result. The whole job is to design the learning signal, read what the run
actually produced, and decide — honestly — whether the model got better.

---

## Using Flash

Flash is a **managed** training service with a thin CLI/client. You author an
environment (the task + its reward), publish it, and submit SFT or GRPO runs from a
TOML config. Flash allocates the cheapest fitting managed GPU class, runs the job,
streams logs back, and serves the result. You never handle infrastructure credentials —
you authenticate once with a freesolo API key, and everything below is a `flash` CLI
command.

### Install & authenticate

```bash
pip install freesolo-flash          # installs the `flash` CLI (import name is also `flash`)
flash login --api-key fslo_...       # or: export FREESOLO_API_KEY=fslo_...  (create a key at https://freesolo.co)
flash whoami                         # confirm the identity behind your key
flash models                         # supported base models (and which support `thinking`)
flash gpus                           # managed GPU classes with estimated $/hr
```

### The project layout (`flash env setup` created this)

```text
environment.py          # the task: how to prompt the model and how to score it
dataset/train.jsonl     # training rows, one JSON object per line: {"input": ..., "output": ...}
configs/rl.toml         # a GRPO (RL) run config
configs/sft.toml        # an SFT run config
TRAINING.md             # this file
```

### 1. Author the environment

`environment.py` defines the task. A single-turn env subclasses
`EnvironmentSingleTurn`, turns a row into a prompt, and scores the model's response
with a `RewardResult` (see *Reward design* below). `load_environment()` is the entry
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
flash env push --name my-env .       # uploads this project; prints an env id like "your-org/my-env"
flash env list                       # local env sources you can push
```

To train against an env someone else published, just set its slug as `[environment] id` —
no separate step is needed. Paste the returned id into `[environment] id` in **both** configs.
Re-push after any
edit to `environment.py` or `dataset/` so the managed run uses your change.

### 3. Configure the run (TOML)

```toml
model = "Qwen/Qwen3.5-4B"   # see `flash models`
algorithm = "grpo"          # "grpo" (RL) or "sft"
# thinking = true           # opt-in reasoning mode, for models that support it

[environment]
id = "your-org/my-env"      # the id printed by `flash env push`
# secrets = ["SERPAPI_API_KEY"]   # only the NAMES of env vars your environment reads;
                                   # values are pulled from your shell/.env at submit time,
                                   # never stored in the spec

[train]
steps = 150                 # GRPO is step-driven; SFT is epoch-driven (epochs = N)
lora_rank = 32
lora_alpha = 64
# All GRPO/SFT knobs live under [train]. Do not add [grpo] or [sft] tables.
```

GPU and HF artifacts are **fully managed** — do not pick `gpu.type` or set
`train.hf_repo`; the allocator picks the cheapest validated managed GPU class that fits,
and run artifacts are stored in a private environment-scoped repo with content-addressed
Flash code snapshots. Compose or tweak configs without editing files: `--config
extra.toml` (deep-merge) and `--set key=value` (e.g. `--set train.steps=300`).

### 4. Submit

```bash
flash train configs/rl.toml --dry-run   # validate the config locally — no GPU, no charge
flash train configs/rl.toml --cost      # pre-flight USD estimate, then exit
flash train configs/rl.toml             # submit and follow logs (Ctrl-C detaches)
flash train configs/rl.toml --background  # submit and return immediately
```

### 5. Monitor

```bash
flash status <run-id>            # state + accrued cost
flash log <run-id>               # reward/loss trend + worker console/error logs + any traceback
flash log <run-id> --follow      # stream a live run to completion
flash status <run-id>            # current run state, cost, and deployment info
flash runs                       # all your runs and their state/cost
flash cancel <run-id>            # stop a run
```

### 6. Deploy & chat

```bash
flash checkpoints <run-id>       # deployable per-step RL checkpoints
flash deploy <run-id>            # serve the trained adapter
flash deploy <run-id>/step-N     # serve an intermediate checkpoint
flash chat <run-id> -m "hello"   # chat with the deployed adapter
flash deployments                # active serving endpoints
flash undeploy <run-id>          # tear the endpoint down
flash export --adapter-id <run-id> --repository <you>/<repo>  # copy adapter weights to your HF repo
```

The rest of this file is about doing the above *well* — designing a reward that teaches,
and deciding honestly whether a run improved.

---

## The loop

Work in tight, attributable iterations. Each one is a hypothesis:

```
1. Reconstruct state — what's the best run so far, and what have you already tried?
2. Form a hypothesis — pick ONE lever and say WHY it will move the metric.
3. Change that ONE lever.
4. Validate locally — `flash train configs/rl.toml --dry-run` (catches config errors
   for free; a paid run on a broken config or an all-zero reward is wasted budget).
5. Submit — `flash train configs/rl.toml`.
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

- [ ] The run reached `done` (confirmed via `flash status <run-id>`), not merely submitted.
- [ ] The reward trend rose (GRPO `reward_mean`) or the SFT loss fell — **beyond the noise band**, not within it.
- [ ] You **probed the trained adapter on real inputs** (`flash deploy` + `flash chat`), including cases it should fail — not just the metrics.
- [ ] The score is real behavior, not empty/truncated/templated outputs, skipped rows, leakage, a swallowed exception, or a format-only win.
- [ ] If you track a clean success signal separately from the shaped reward (an explicit `RewardMetric`), *that* moved too.

If any box is unchecked, the run is not done improving — keep training, don't declare success.

---

## Common Flash issues and mitigations

Most bad Flash runs fail in a small number of predictable ways. Check these before
spending another GPU run:

| Issue | Symptom | Mitigation |
| --- | --- | --- |
| Environment id is blank or stale | `flash train --dry-run` fails, or the worker uses old reward/data | Run `flash env push --name my-env .` after every environment/data edit and paste the returned id into every config you submit. |
| Local-only env path in config | Config validation says there is no local path mode | Publish first, then use the returned slug in `[environment] id`. `flash train` only runs published env ids, not local paths. |
| Config knobs are in the wrong table | Validation rejects `[grpo]`, `[sft]`, or unknown `[train]` keys | Put `steps`, `epochs`, `group_size`, `max_tokens`, `temperature`, `max_length`, LoRA, and other training knobs under `[train]`. |
| Trying to pin managed infrastructure | `gpu.type`, `train.hf_repo`, or `model_policy` changes do not do what you expected | Treat GPU choice, model policy, and the run artifact repo as managed. Tune the model, algorithm, environment, and `[train]` knobs instead. |
| Secrets are not available on the worker | Reward code works locally but remote logs show missing API keys or auth failures | List secret names under `[environment] secrets = [...]`, export those env vars locally before submit, or put them in local `.env` / `.env.local`. Never put secret values in `[worker_env]` or hard-code them in the config. |
| Wrong model / thinking setting | Config validation fails, or chat behavior does not match the run | Use `flash models`; set `thinking = true` only for supported models. Thinking is a training-time/run-level choice and serving preserves that parity, so `flash chat` does not expose an override flag. |
| Thinking reward grades the wrong text | Rewards accidentally score hidden reasoning, or ignore reasoning you meant to inspect | By default, score the answer text. In thinking mode the response object is still string-compatible, but also exposes `.completion`, `.thinking`, and `.raw` when a reward intentionally needs those fields. |
| All-zero or flat GRPO reward | `reward_mean` stays near 0 and outputs do not improve | Make the reward dense: give partial credit for parse/format/execution/correctness tiers, and log a separate clean `success` metric. Do not keep rerunning an all-zero reward. |
| Reward rises but behavior is worse | Short, templated, malformed, or reward-hacked outputs score well | Deploy the adapter and probe real examples. Add hard validity gates before judge calls, penalize degenerate shortcuts, and judge the outcome rather than the surface string. |
| Output is truncated | Correct-looking answers cut off mid-response or JSON is incomplete | Increase `max_tokens` for GRPO rollouts or `max_length` for SFT only after seeing truncation. Oversizing them by default just burns memory/cost. |
| Infrastructure, CUDA, OOM, vLLM, or kernel failure | Run errors before useful metrics, often during setup/model load | Treat this as infrastructure pressure, not proof the model is too large. Read `flash log <run-id>`, reduce footprint (`max_length`, `max_tokens`, `group_size`) if needed, and let Flash retry/allocate another fitting GPU class. |
| Run looks stuck after disconnecting | Terminal stopped streaming but the job may still be alive | Ctrl-C detaches. Use `flash log <run-id> --follow` to reattach, `flash log <run-id>` for the console/error output, or `flash cancel <run-id>` if you intentionally want to stop it. |
| Final checkpoint regresses | Last step is worse than an earlier checkpoint | Run `flash checkpoints <run-id>`, deploy a specific step with `flash deploy <run-id>/step-N`, and compare with held-out probes before exporting or relying on the final adapter. |
| Export fails before upload | CLI says no HuggingFace token | Pass `flash export --api-key hf_...`, or set `HF_TOKEN` in your shell, `.env`, or `.env.local`. Exports are private unless you pass `--public`. |
| SFT loss improves but quality does not | Train loss falls while held-out behavior stalls or degrades | Keep a held-out split outside training. Deploy and score that split; if quality drops, reduce epochs or improve data instead of adding more passes. |
| Cost surprises | A quick experiment uses more GPU time than intended | Start with `--dry-run` and `--cost`, cap steps/epochs for smoke tests, and scale only after reward/data wiring is proven. Setup time is reported for observability; customer cost is based on training-loop GPU time. |

---

## Judge the run, don't just finish it

- **Judge the trend, not a single number.** The proof of training is the curve:
  `reward_mean` rising over steps (GRPO) or loss falling (SFT). Record the base/early
  value and the final value. A flat or noisy trend with no improvement is not success.
- **Read the model's outputs, not just the metrics.** A rising reward can come from
  reward-hacking or a degenerate output the reward still credits — metrics alone never
  establish that the model got better. Flash does not expose training-time rollouts
  through the CLI (`flash log` gives you the metric trend and the worker's console/error
  logs, not the sampled generations), so to read real outputs **deploy the adapter and
  probe it**: `flash deploy <run-id>` then `flash chat <run-id> -m "..."` on at least a
  few real inputs, including ones it should get wrong.

  ```bash
  flash status <run-id>            # state + accrued cost
  flash log <run-id>               # metric trend + worker console/error logs (+ traceback)
  flash log <run-id> --follow      # stream a live run until completion
  flash deploy <run-id>            # serve the adapter, then `flash chat` it to read real outputs
  ```

- **Decide with the noise band.** When comparing two runs or two checkpoints, record
  the eval-split size `N` and the metric's approximate sampling noise — about
  `1.96·√(p(1-p)/N)` for a rate metric `p`. Treat a difference *inside* that band as
  **no change** — neither improvement nor regression. A within-noise gain is not a win.

---

## Reward design (GRPO) — your highest-impact lever

The reward defines what the model learns; its quality sets the ceiling on what GRPO
can reach. Rewards are rubric / `score_response` functions in your `environment.py`.

### Make it graded and dense — avoid the all-zero cold start

If `reward_mean` is flat at ~0.000, every rollout in the group scored the same, the
advantage is zero, and the policy gets **no gradient**. That is a reward-design bug,
not a model to keep training. Reshape the reward to credit **ordered partial
progress** so even an untrained base model earns a small nonzero score and better
attempts score strictly higher:

```text
well-formed / parses → schema- & safety-valid → executes / runs → correct / relevant
```

Gate only the **top** tiers against gaming; keep the lower tiers dense. GRPO needs
*within-group variance* to learn — if every rollout in a group scores identically,
there is nothing to optimize.

### Separate the shaped reward from a clean success signal

A good GRPO reward is usually **shaped** — partial credit so the model always has a
gradient to climb. But a shaped score is the wrong thing to judge *final quality* on:
it can rise from reward-hacking while the outcome you care about stays flat. Report the
shaped value as `score`, and surface the clean pass/fail as an **explicit
`RewardMetric`** so it shows up in the run's metric breakdown — a bare `threshold` is
used for grading but is *not* logged on its own, so it gives you nothing to judge:

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

`score` is what GRPO optimizes (it becomes the run's `total`). Each `RewardMetric` you
attach is logged by name in the per-scorer breakdown — that is how the clean success
rate becomes visible. Use the shaped `score` to confirm the model is learning *at all*,
and judge the run on the explicit `success` metric.

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
  tasks, grade the *returned records*, not the query text — the query is only
  secondary validity evidence.
- **A small format penalty beats a hard zero for shaping.** A useful trick:
  `reward = format_coef * (correct_format - 1) + correct_answer` with `format_coef≈0.1`
  — a tiny penalty for bad formatting, full credit for a correct, well-formatted answer.
- **Anti-patterns.** Don't reward length or verbosity. Don't ship a reward that is
  always 0 or always 1 (no signal). Simpler rewards usually beat clever ones — a
  mediocre *stable* reward beats a "perfect" reward you keep tweaking. Changing the
  reward resets progress, so keep the best checkpoint before you do.

---

## SFT conventions

Pick SFT when you already have good answers and want the model to imitate them.

- **Data quality is the ceiling.** SFT can only be as good as the answers you show it.
  A small set of high-quality examples beats a large mediocre one. Keep response format
  consistent (if you want JSON, *every* example is JSON) and keep the prompt format the
  same as inference time.
- **Watch the loss fall — and check overfitting yourself.** Flash SFT logs **training
  loss only**; it runs no mid-training held-out eval (evaluation is deferred to the
  deploy/serving side). A falling train loss alone can be memorization, so keep an eval
  split the run never trains on, then **deploy the adapter and score it on that split**
  (`flash deploy` + `flash chat`). If held-out quality stalls or drops while train loss
  keeps falling, reduce `epochs` or add more data — not more passes.
- **Start `max_length` small and grow it on evidence.** Begin from the smallest
  `max_length` that plausibly fits prompt + completion, and only raise it when you see
  truncation (outputs cut off mid-thought, degraded loss). A bigger context just costs
  more.
- **SFT is a great warm start for GRPO.** SFT first to teach the format and a competent
  baseline, then GRPO to optimize past it. Across that lineage keep the **same base
  model**. For text-only continued adapters, keep the same adapter shape. For VL
  warm-starts, Flash trains a fresh GRPO LoRA and rank-stacks it with the SFT LoRA
  for deployment, so `SFT rank + GRPO rank` must stay within the selected model's
  effective serving `max_lora_rank`. That cap comes from the serving/model policy:
  some small serving models allow rank 64, while larger serving paths can cap at rank
  32. Flash preflights the rank-stacked deploy rank against that model-specific cap.

```toml
# configs/rl.toml — warm-start GRPO from the SFT run's adapter
algorithm = "grpo"

[train]
# the SFT run id (as printed by `flash status`); add /step-N to warm-start from a
# specific checkpoint listed by `flash checkpoints <run-id>`
init_from_adapter = "<sft-run-id>"
lora_rank = 16     # for VL warm-starts, SFT rank + GRPO rank must fit the effective serving cap
lora_alpha = 32
```

SFT is **epoch-driven** (`epochs`); GRPO is **step-driven** (`steps`).

---

## GRPO knobs that matter

Set these in `[train]`. Each is `None` by default — the worker's tuned recipe fills
in a sensible value, so only override with a reason.

| Knob | Convention |
| --- | --- |
| `group_size` | Completions sampled per prompt (default 8). More = more signal and more cost; drop to 4 to trim cost. The group needs *within-group variance* for an advantage to exist. |
| `max_tokens` | Completion budget per rollout. Size it to the expected output length — too small silently truncates good answers and poisons the reward; too large just costs more. |
| `temperature` | Rollout sampling temperature. Keep it near 1.0 for GRPO — too low collapses diversity (and the model can collapse within a few steps); raise it to widen exploration against uniform-reward groups. |
| `kl_penalty_coef` | Keeps the trained model from drifting too far from the base. Raise it to anchor against entropy collapse; lower it for more freedom to move. |
| `thinking_length_penalty_coef` | Per-reasoning-token reward deduction — curb overthinking, but watch it doesn't push the model into terse degeneracy. |
| `learning_rate` | Change it in small steps. Too high destabilizes RL and degrades output quality; if the model is collapsing, lower it. |
| `batch_size` | The effective prompts-per-step. Too small and the reward trend is pure noise; size it so the trend is readable. |

> **The reward-hacking signature:** a smoothed reward rising while mean generated
> length collapses. Whenever any shortness or format pressure is active, verify the
> gate by scoring a few truncated or opener-only probe responses — they should score low.

---

## Curriculum — start easy, scale up

Starting too hard produces zero learning signal; the model never succeeds, the reward
stays at 0, and there is nothing to climb. Start where the base model can *partially*
succeed, then raise difficulty as it improves. The "Goldilocks zone" — where most
rollouts score somewhere between all-fail and all-pass — is where GRPO has the most
signal.

- If nearly every prompt is solved (most groups score ~1.0): **increase difficulty** —
  harder prompts, tighter format/reward, more steps.
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

| Failure mode | What you see | Targeted fix |
| --- | --- | --- |
| Repetition / looping collapse | the same phrase repeats until truncation | repetition or length penalty; lower `temperature` |
| Overthinking / verbose reasoning | reasoning eats the whole token budget | `thinking_length_penalty_coef`; tighten the prompt |
| Max-token truncation | answers cut off mid-thought | raise `max_tokens` / `max_length` |
| Unparsed / over-escaped output | reward can't read the answer | robust parser; return `0.0` on parse fail; format gate |
| Wrapper / markdown around structured output | prose around the JSON/answer | a format gate; `stop_sequences` |
| Uniform-reward groups | every rollout in a group scores the same → no gradient | shape the reward for partial credit; raise `temperature` |
| Too-hard prompts | the base never succeeds, reward stays at 0 | curriculum / easier prompts; warm-start with SFT |
| Judge-rewarded degenerate output | short, templated answers a judge still rates well | a minimum-substance zero-gate ahead of the judge |

---

## When a run stalls

A plateau is not automatically a capability ceiling. Before you call it one:

1. **Probe with best-of-N.** Run a best-of-N / pass@k probe at a coverage temperature
   (well above greedy) on a less-fitted checkpoint.
2. **Read the result.** High best-of-N but a collapsed greedy output and low sample
   diversity is **entropy collapse**, not a ceiling — and it's fixable: anchor harder
   with `kl_penalty_coef`, lower the `learning_rate`, or widen exploration. Only if the
   probe shows no headroom is it a genuine ceiling.
3. **Change a different lever.** If there's real headroom, try a *different* lever from
   the one that just failed — a different knob, reward shape, or data family — one
   controlled change at a time.

Actively research established GRPO/SFT techniques (exploration / entropy control, KL
scheduling, reward shaping, curriculum / difficulty filtering, rejection-sampling SFT
on high-reward rollouts) rather than guessing — and count a technique as helpful only
on a beyond-noise improvement.

---

## Scale the evidence

- **A smoke test is not proof.** A single-digit `steps`, a tiny dataset, or a handful
  of rollouts only validates the wiring. Scale `steps` / `epochs`, the dataset size,
  and `group_size` to the model and the data you actually have before you trust a
  result. Don't cite budget alone as the reason for an underpowered run.
- **Use the data you have.** Deliberately assign every usable row to training or to a
  held-out eval split; if a planned holdout is so small that one example swings the
  metric by several points, enlarge it during split design rather than gating on noise.

---

## Treat crashes as infra, not model size

> A CUDA / OOM / vLLM / kernel / infrastructure error is an **infrastructure** problem, not a
> sign the model is too big. Lower `max_length`, `max_tokens`, or `group_size` to shrink
> the run's footprint and let the allocator retry onto the next fitting GPU class — do
> **not** switch to a smaller model to make a crash disappear. That silently destroys
> quality.

---

## Command reference

```bash
flash env setup                       # scaffold environment.py, dataset/, configs/, this file
flash env push --name my-env .        # publish the environment; paste the returned id into [environment]
flash env pull your-org/my-env        # download a published environment into the current folder
flash env delete your-org/my-env -y   # delete a published environment
flash train configs/rl.toml --dry-run # validate the config locally (no GPU, no charge)
flash train configs/rl.toml --cost    # pre-flight USD estimate, then exit
flash train configs/rl.toml           # submit and follow logs (Ctrl-C detaches; --background to skip following)
flash status <run-id>                 # state + accrued cost
flash log <run-id>                    # reward/loss trend + worker console/error logs
flash log <run-id> --follow           # stream a live run to completion
flash runs                            # list your runs and their state/cost
flash cancel <run-id>                 # stop a live run
flash checkpoints <run-id>            # list deployable RL checkpoints
flash deploy <run-id>                 # serve the trained adapter
flash deploy <run-id>/step-N          # serve a specific RL checkpoint
flash chat <run-id> -m "probe"        # stream a reply from the deployed adapter
flash deployments                     # list active serving deployments
flash undeploy <run-id>               # tear down an active deployment
flash export --adapter-id <run-id> --repository <you>/<repo>  # export final adapter
flash export --adapter-id <run-id>/step-N --repository <you>/<repo>  # export a checkpoint
```

See the full reference at https://freesolo.co/docs.
"""
