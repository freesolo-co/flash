# Authoring & using environments

AutoSLM is **verifiers-only**: every environment is a Prime Intellect
[`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) environment. The env defines
the task — its dataset, the prompts, and the weighted reward `Rubric`. There are no built-in
task environments and no training contract/oracle: the `verifiers` env *is* the task
definition. AutoSLM wraps it via `autoslm.envs.verifiers_adapter` so it runs unchanged on
the trainer (single-turn SFT/GRPO/eval is fully supported).

## The interface

Each environment module exposes `load_environment(**kwargs)` returning a
`verifiers.Environment` (e.g. a `SingleTurnEnv`) with a `Rubric` of weighted reward funcs:

```python
import verifiers as vf
from datasets import Dataset


def load_environment(**kwargs) -> vf.Environment:
    dataset = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": "What is 2 + 2?"}], "answer": "4"}]
    )

    def correct_answer(completion, answer, **_):
        text = completion[-1]["content"] if isinstance(completion, list) else str(completion)
        return 1.0 if str(answer) in text else 0.0

    rubric = vf.Rubric(funcs=[correct_answer], weights=[1.0])
    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric, **kwargs)
```

Rows carry `prompt` (chat messages) + `answer` (+ optional `info`). The adapter maps the
verifiers `dataset`, `system_prompt`, `parser`, and the weighted `rubric` reward funcs (sync
or async) onto AutoSLM's interface; a `RubricGroup` is flattened and a `JudgeRubric`'s judge
is supplied to reward funcs that declare it.

## Scaffold a custom environment

```bash
uv run slm env init my-env          # creates environments/my_env/my_env.py (a verifiers env)
```

The scaffold emits a real `verifiers` env. Publish it to the Prime Hub and reference it by
its slug in your config:

```toml
[environment]
id = "owner/my-env"   # the Hub slug you get after `slm env push`
```

> The managed service runs published Hub envs only — Flash ships only the `autoslm` package,
> not your project tree, so the worker never sees a local env file. Publish your scaffolded env
> to the Prime Hub with `slm env push` and reference it by `id` (see below).

## Prime Hub / `verifiers` interop (managed runs)

Install a published env, then reference it by id — this is the path managed runs must use:

```bash
uv run slm env install owner/my-env     # via the prime CLI or pip; recorded in ~/.autoslm/envs.json
uv run slm env push environments/my-env # publish your own local env to the Hub
```
```toml
[environment]
id = "owner/my-env"
```

The GPU worker auto-installs `verifiers` + the env package for that run. Multi-turn/tool
environments (`ToolEnv`/`MultiTurnEnv`) are accepted but not yet wired end-to-end for GRPO
training (the rollout loop in the worker is a roadmap item); single-turn is fully supported.
