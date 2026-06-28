"""The TRAINING.md playbook scaffolded by `flash env setup`.

This is the single source of truth for the TRAINING.md that lands in a user's
project folder. It is written for the AI coding agent a user points at their
environment: a finished run is not a run that worked, and most of the value is in
how you design the signal, what you read, and how you decide a run is good.

Keep it flash-accurate — every command, config field, and import below exists in
this codebase (`flash <cmd>`, `[train]` fields in ``flash/spec.py``, and
``freesolo.environments``). Update it here, not in a copy.
"""

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
TOML config. Flash allocates the cheapest fitting GPU across providers, runs the job,
streams logs back, and serves the result. You never handle provider credentials or a
GPU — you authenticate once with a freesolo API key, and everything below is a `flash`
CLI command.

### Install & authenticate

```bash
pip install freesolo-flash          # installs the `flash` CLI (import name is also `flash`)
flash login --api-key fslo_...       # or: export FREESOLO_API_KEY=fslo_...  (create a key at https://freesolo.co)
flash whoami                         # confirm the identity behind your key
flash models                         # supported base models (and which support `thinking`)
flash gpus                           # managed GPU classes with live $/hr
```

### The project layout (`flash env setup` created this)

```text
environment.py          # the task: how to prompt the model and how to score it
datasets/train.jsonl    # training rows, one JSON object per line: {"input": ..., "output": ...}
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
    dataset = load_jsonl("datasets/train.jsonl")   # rows -> TaskExample(input=..., output=...)

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
edit to `environment.py` or `datasets/` so the managed run uses your change.

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
```

GPU and the HF artifact repo are **fully managed** — there is no GPU knob; the allocator
picks the cheapest class that fits, and each run gets its own artifact repo. Compose or
tweak configs without editing files: `--config extra.toml` (deep-merge) and
`--set key=value` (e.g. `--set train.steps=300`).

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
flash status <run-id> --logs     # reward/loss trend + worker console/error logs + any traceback
flash status <run-id> --follow   # stream a live run to completion
flash runs                       # all your runs and their state/cost
flash cancel <run-id>            # stop a run
```

### 6. Deploy & chat

```bash
flash checkpoints <run-id>       # deployable per-step RL checkpoints
flash deploy <run-id>            # serve the trained adapter (--step N for an intermediate checkpoint)
flash chat <run-id> -m "hello"   # chat with the deployed adapter
flash deployments                # active serving endpoints
flash undeploy <run-id>          # tear the endpoint down
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

## Judge the run, don't just finish it

- **Judge the trend, not a single number.** The proof of training is the curve:
  `reward_mean` rising over steps (GRPO) or loss falling (SFT). Record the base/early
  value and the final value. A flat or noisy trend with no improvement is not success.
- **Read the model's outputs, not just the metrics.** A rising reward can come from
  reward-hacking or a degenerate output the reward still credits — metrics alone never
  establish that the model got better. Flash does not expose training-time rollouts
  through the CLI (`--logs` gives you the metric trend and the worker's console/error
  logs, not the sampled generations), so to read real outputs **deploy the adapter and
  probe it**: `flash deploy <run-id>` then `flash chat <run-id> -m "..."` on at least a
  few real inputs, including ones it should get wrong.

  ```bash
  flash status <run-id>            # state + accrued cost
  flash status <run-id> --logs     # metric trend + worker console/error logs (+ traceback)
  flash status <run-id> --follow   # stream a live run until completion
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
  model and the same `lora_rank` / `lora_alpha`** — `init_from_adapter` loads a LoRA
  adapter specific to one base model and one adapter shape, so mixing sizes is an
  invalid shape mismatch.

```toml
# configs/rl.toml — warm-start GRPO from the SFT run's adapter
algorithm = "grpo"

[train]
# paste the full adapter_ref `flash status <sft-run-id>` prints, verbatim
# (shape: <owner>/<repo>:sft/<run-id> — the owner/repo prefix is required)
init_from_adapter = "your-org/your-repo:sft/<sft-run-id>"
lora_rank = 32     # must match the SFT run
lora_alpha = 64    # must match the SFT run
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

> A CUDA / OOM / vLLM / kernel / provider error is an **infrastructure** problem, not a
> sign the model is too big. Lower `max_length`, `max_tokens`, or `group_size` to shrink
> the run's footprint and let the allocator retry onto the next fitting GPU class — do
> **not** switch to a smaller model to make a crash disappear. That silently destroys
> quality.

---

## Command reference

```bash
flash env setup                       # scaffold environment.py, datasets/, configs/, this file
flash env push --name my-env .        # publish the environment; paste the returned id into [environment]
flash train configs/rl.toml --dry-run # validate the config locally (no GPU, no charge)
flash train configs/rl.toml --cost    # pre-flight USD estimate, then exit
flash train configs/rl.toml           # submit and follow logs (Ctrl-C detaches; --background to skip following)
flash status <run-id>                 # state + accrued cost
flash status <run-id> --logs          # reward/loss trend + worker console/error logs
flash status <run-id> --follow        # stream a live run to completion
flash runs                            # list your runs and their state/cost
flash deploy <run-id>                 # serve the trained adapter
```

See the full reference at https://freesolo.co/docs.
"""
