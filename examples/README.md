# Example environments

Self-contained example task environments. The wheel force-includes this `examples/`
tree (see `pyproject.toml`), so they ship with the installed package as well as the
source checkout. The registry **path-loads** each folder as a built-in environment id
(`autoslm/envs/registry.py`), so they are usable out of the box and double as templates
for authoring your own task.

| folder | env id | task | configs |
|---|---|---|---|
| [`gsm8k/`](gsm8k/) | `gsm8k` | grade-school math (GSM8K), verifiable numeric answer | SFT · GRPO |
| [`math/`](math/) | `math` | competition math (MATH), LaTeX/numeric grading | GRPO |

Use one from any config:

```toml
[environment]
id = "gsm8k"   # or "math"
```

```bash
slm train examples/gsm8k/gsm8k_grpo.toml
slm train examples/math/math_grpo_autoslm.toml
```

## Anatomy of an example

Each example is a small package implementing the `Environment` interface
(`docs/environments.md`), split so each concern is independently readable:

- **`env.py`** — the `Environment` subclass the registry loads (`dataset`,
  `prompt_messages`, `sft_target`, `reward`, `grade`).
- **`grading.py`** — a dependency-free grader that is the **single source of
  truth** for scoring, shared by the SFT target check and the GRPO reward/grading,
  so both training arms score a model identically.
- **`data.py`** *(gsm8k)* — dataset loading and prompt formatting. (The `math`
  example keeps its loading inside `env.py` because it supports several dataset
  shapes and JSONL offline overrides.)
- **`__init__.py`** — re-exports `load_environment` for the path loader.
- **`*.toml`** — ready-to-run configs (`slm train <path>`).

See [`gsm8k/README.md`](gsm8k/README.md) for the per-algorithm config walkthrough —
it is the recommended starting point, covering both algorithms on one task.

## Authoring your own

Copy `gsm8k/` as the template, keep the three-file split, and implement the
interface. Keep one grader so the SFT target check and GRPO reward/grading agree.
Scaffold and publish with:

```bash
slm env init my-env     # scaffold environments/my_env/
slm env push            # publish to the Prime Hub, then reference by id
```

The managed service does **not** load local `[environment] path = "..."` dirs (the
server never sees your filesystem) — publish to the Hub and reference by id
instead. See `docs/environments.md` for the full interface and `verifiers`/Prime
Hub interop.
