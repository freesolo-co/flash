# autoenv — automated post-training environment generator

A benchmark harness that measures **how well an AI agent can replicate a paper's small-LM
training result using Flash**. It is built *on top of* Flash (it imports `flash.*`
one-directionally; `flash` never imports `autoenv`, so the zero-dependency `flash` client is
untouched), and it lives in its own top-level package with its own `autoenv` console script.

## The pipeline

```
PaperCase manifest
   │
   ├─ ingest   resolve the dataset refs to canonical {input, output} rows
   ├─ gate     is this replicable on Flash?  (5 checks → GateReport)
   ├─ drive    scaffold a Flash env + config and train      (agent backend)
   ├─ eval     run the trained adapter over the held-out split, score it
   └─ score    improvement-normalized achievement vs the paper  (0..1)
```

### Difficulty modes
- **easy** (built first): the reward is *supplied* and frozen into the env — the agent only
  builds the env wiring + config and trains. Isolates "can it train" from "can it design
  rewards."
- **hard** (later): the agent designs the reward itself.

### Scoring
`achievement = clamp((agent − base) / (paper − base), 0, 1)` — credits the delta the agent's
training produced over the *measured* untrained base, relative to the paper's delta. Robust to
training a different (Flash-supported) base model than the paper used. The **headline metric is
independent of the env reward** the agent optimized (computed on `(gold, response)`), so a
reward-hacked run can't inflate the score.

## What's implemented (milestones M0–M1, offline + CI-tested)
- `manifest.py` — `PaperCase` schema + TOML loader/validator.
- `ingest/sources.py` — dataset resolution (local / http / `hf:` / `org/name`).
- `gate/` — the five eligibility checks, each reusing a real Flash function
  (`catalog`, `cost.spec.estimate_for_spec`, `engine.vram.check_fit`, `schema`).
- `drive/` — workspace scaffolding via Flash's own `flash env setup`, the `ScriptedBaseline`
  agent (deterministic control), and the dry-run/cost-gated orchestrator.
- `eval/` + `score/` — the pure, dependency-free pieces (metric registry, deterministic split
  + leakage guard, normalization) are complete and tested. Generation (`deploy` + `chat`) is
  the live M2 piece.

## Reuse, don't reinvent
Every stage leans on existing Flash surfaces — see `tests/test_autoenv_flash_contract.py` for
the exact list it depends on (a contract test fails CI if Flash renames one).

## Try it (offline)

```bash
uv run autoenv gate  autoenv/cases/arithmetic_smoke_sft.toml          # eligibility report
uv run autoenv run   autoenv/cases/arithmetic_smoke_sft.toml --offline # gate + dry-run drive
uv run pytest tests/test_autoenv_*.py
```

A real training run (`autoenv run … --real`) plus `eval`/`score`/`report` need Flash
credentials + a GPU and are the subject of milestone M2.

## Roadmap
- **M2** — first full real run: `Qwen3.5-0.8B`, easy-mode SFT, `--real` submit → deploy+chat
  eval → improvement-normalized score. Implements `eval/generate.py`, `eval/score.py`, the
  `score`/`report` CLI.
- **M3** — hybrid paper intake: LLM drafts a `PaperCase`, human approves into `cases/`.
- **M4** — hard mode, the `ClaudeAgent` backend, GRPO cases, intermediate-checkpoint eval.
