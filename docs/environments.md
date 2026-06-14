# Authoring & using environments

AutoSLM environments define the task: the dataset, how prompts are rendered, the SFT target, and
the RL reward.

## The interface

Each environment exposes `load_environment(**kwargs)` returning an object implementing:

```python
class Environment:
    id: str
    def dataset(self, split: str) -> list[dict]: ...         # training rows ("train")
    def prompt_messages(self, example: dict) -> list[dict]:  # chat messages
    def sft_target(self, example: dict) -> str: ...          # assistant text for SFT
    def reward(self, completion: str, example: dict) -> float: ...  # RL reward
    def grade(self, completion: str, example: dict) -> bool: ...    # optional bool scorer reward can build on
```

Subclass `autoslm.envs.base.BaseEnvironment` for sensible defaults.

> **Worked example.** [`examples/gsm8k/`](../examples/gsm8k/) is the smallest
> complete implementation of this interface and the recommended template — env +
> grader + data split across three files, with a ready-to-run config for every
> algorithm. See its [README](../examples/gsm8k/README.md) and
> [algorithms.md](algorithms.md).

## Scaffold a custom environment

```bash
uv run slm env init my-env          # creates environments/my_env/my_env.py
```

Publish it to the Prime Hub with `slm env push`, then reference it by id:

```toml
[environment]
id = "you/my-env"
```

> `[environment] path = "environments/my_env"` loads a local directory and is only
> usable on the machine where the training code runs — the **managed service rejects
> it** (the server never sees your filesystem). Publish to the Hub instead.

## Built-ins

- **`gsm8k`** — GSM8K verifiable math (shared boxed/`####` grader).
- **`math`** — competition math (DeepScaleR → MATH-500), LaTeX/numeric grading.
- **`tests_pass`** — coding tasks: apply a diff in a temp copy and reward on a test command.
  ⚠️ The temp-dir runner is **not** a security sandbox; treat untrusted env/tool code as
  unsafe until container sandboxing lands (see roadmap).

## Prime Hub / `verifiers` interop

AutoSLM runs Prime Intellect `verifiers` environments unchanged (single-turn) through
`autoslm.envs.verifiers_adapter`:

```bash
uv run slm env install owner/my-env     # via the prime CLI or pip; recorded in ~/.autoslm/envs.json
```
```toml
[environment]
id = "owner/my-env"
```

The adapter maps the verifiers `dataset`, `system_prompt`, `parser`, and the
weighted `rubric` reward funcs (sync or async) onto AutoSLM's interface. The GPU worker
auto-installs `verifiers` + the env package for that run. Multi-turn/tool environments
(`ToolEnv`/`MultiTurnEnv`) are a roadmap item (they need the rollout loop in the worker).
