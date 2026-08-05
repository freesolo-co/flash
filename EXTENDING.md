# Extending Flash — design

Status: **design proposal**. Nothing in this document is implemented beyond what the
Layout section of `README.md` already describes. It defines the direction: Flash as a
barebones managed post-training core that users extend by prompting their own coding
agents, rather than a batteries-included framework.

## The model

The reference point is the pi coding agent: a tiny fixed core (4 tools, a sub-1,000-token
system prompt), extended not by vendor feature releases but by users prompting an agent
that reads pi's own extension docs and builds what is missing. The vendor owns the
irreducible harness; everything else is userland the agent can write, verify, and rewrite.

Translated to a managed post-training service:

| pi                              | Flash                                                             |
| ------------------------------- | ----------------------------------------------------------------- |
| 4 built-in tools                | Managed core: allocation, supervision/retry, checkpoint streaming, serving, billing |
| Extensions built by the agent   | User-authored trainers and environments, pushed and loaded by pinned id |
| Extension docs the agent reads  | `flash/cli/TRAINING.md` today; this file and `.claude/skills/` next |
| Agent edits pi itself           | Agent edits the *user's* trainer/environment package — never the control plane |
| Verify locally in seconds       | CPU-only conformance tests → local dry-run → capped managed smoke run |

Flash is closer to this model than it looks. Environments already are it: users write
`environment.py` (dataset, prompts, rewards, grading), `flash env push` publishes it, and
the worker loads it by id through `flash/envs/loader.py` with sha pinning, bounded
caching, and safe archive extraction. Arbitrary user code already runs on managed GPUs;
the trust decision is made. `TRAINING.md` (1,293 lines, shipped in-package, written into
user projects by `flash env setup`, surfaced via `flash train --doc`) is already
documentation an agent reads to do the work. The CPU-only offline test suite (~3,900
tests) is the free, fast verification loop that pi gets from running locally.

What is *not* pi-shaped is the algorithm set. `("sft", "grpo", "opd")` is hardcoded
across ~25 files — the `ALGORITHMS` tuple and per-model algorithm lists in
`flash/catalog.py`, 14 per-algorithm branches in `flash/engine/vram.py`, the dispatch
dict in `flash/engine/worker/__init__.py`, and per-algorithm handling in
`flash/schema/__init__.py`, `flash/spec.py`, `flash/runner/__init__.py`,
`flash/cost/spec.py`, `flash/cost/types.py`, `flash/serve/preflight.py`,
`flash/multimodal.py`, `flash/lora_rank.py`, `flash/cli/env_setup.py`, and
`flash/server/repo_cleanup.py`. There is no trainer contract and no registry. Adding an
algorithm today is open-heart surgery on billing-, allocation-, and retry-critical code.

## Phase 1 — the trainer contract

Collapse the sprawl into one registration surface, in the house style of the
`Provider` protocol (`flash/providers/base.py`):

```python
@runtime_checkable
class Trainer(Protocol):
    name: str                # "sft" | "grpo" | "opd" | user-defined
    artifact_phase: str      # HF artifact path prefix (checkpoints, repo cleanup)
    on_policy: bool          # replaces _ON_POLICY_ALGORITHMS membership

    def train_schema(self) -> dict: ...
        # jsonschema fragment for the [train] keys this trainer accepts

    def vram_profile(self, spec: JobSpec, model: ModelInfo) -> VramProfile: ...
        # feeds the allocator; unknown trainers may return a declared profile
        # validated at dry-run, or fall back to a pessimistic default

    def cost_facts(self, spec: JobSpec) -> CostFacts: ...
        # pre-flight cost estimation inputs

    def supports(self, model: ModelInfo) -> bool: ...
        # replaces per-model algos= lists in the catalog

    def run(self, ctx: WorkerContext) -> RunMetrics: ...
        # the on-GPU entrypoint; ctx carries model/tokenizer/environment/vLLM
        # handles, heartbeat, and the artifact upload channel
```

The built-in trainers move onto this contract first. If `run_sft`, `run_rl`, and
`run_opd` cannot live on it, user agents cannot either — dogfooding is the honesty test.
Supervision does not change: the stall watchdog, retry budget, checkpoint streaming, and
the DONE-file protocol are process-level and algorithm-agnostic already.

## Phase 2 — `flash trainer push`

User trainers ship exactly like environments:

- `flash trainer setup` scaffolds a trainer package plus its offline conformance tests
  (mirroring `flash env setup`).
- `flash trainer push` publishes it; runs reference `trainer = "owner/name"` in the TOML;
  the worker loads it by pinned id, reusing `flash/envs/loader.py` resolution/caching and
  `flash/envs/archive.py` extraction limits.
- User `[train]` keys live under a namespaced table (e.g. `[train.custom]`) so
  `introduced_in` schema-drift detection keeps working for core keys.

Managed guarantees for unknown algorithms come from three mechanisms: conservative-ceiling
VRAM allocation (declared profile or pessimistic default — OOM escalation in
`flash/runner/lifecycle.py` already absorbs under-estimates), algorithm-agnostic
process supervision, and post-hoc calibration of cost facts from measured runs.

Rejected alternatives: forking Flash (users would be forking a control plane and billing
system they don't operate) and built-in-only PRs (gates every extension on vendor review —
the opposite of the model).

## Phase 3 — shed the core

Once the contract holds, move optional weight out of core:

- **OPD** (~2,900 lines, hand-rolled loop, no TRL trainer) becomes the first reference
  extension — proof the contract expresses a real, hard algorithm.
- Multimodal branches and structured-outputs plumbing become trainer-owned concerns.
- The irreducible core that remains: spec/schema, allocation + VRAM ceilings, run
  supervision/retry/billing, checkpoint streaming, environment and trainer loading,
  serving, and the zero-dependency client CLI.

## The agent-facing layer

For a user's coding agent to build a trainer unaided, it needs:

- **This file**, extended with the frozen `WorkerContext` surface and one worked example
  (e.g. DPO) once Phase 1 lands. Discipline: the contract plus the context surface must
  fit in ~5–8k tokens, so an agent holds the entire extension API in context at once.
- **`.claude/skills/add-trainer/`** (and later `add-provider`, `add-teacher`) next to the
  existing `verify` skill.
- **`flash trainer test`**: the verification ladder, cheapest first —
  1. offline conformance: contract checks and `run()` against a fake `WorkerContext`,
     CPU-only, seconds, free;
  2. local dry-run: smallest catalog model, a few steps, via the `[gpu]` extra;
  3. managed smoke run: capped steps, capped cost, before any full run.

  An agent-built trainer should reach high confidence at tiers 1–2 without spending a
  dollar. This ladder is the substitute for pi's verify-locally-in-seconds loop, and the
  design stands or falls on it: a full GPU run costs real money and hours, so it must be
  the *last* check, never the first.

## Known risks

- **Support burden.** User trainers failing mid-run on rented GPUs will look like Flash
  failures. Run logs must attribute errors to userland vs the managed core explicitly.
- **Worker image pins.** User trainers inherit the baked stack (`trl`, `vllm`, `torch`
  pins in `flash/providers/_worker.py`); the documented contract surface is what is
  frozen, not the transitive stack. Per-job pip extension exists (`chalk_extra_pip()`)
  but widens the failure surface.
- **Cost-estimation gaming.** Declared VRAM profiles influence GPU selection; ceiling
  defaults and post-hoc measurement bound the exposure.
- **Release channels.** Trainer packages are user artifacts, not channel artifacts — they
  must not fork with the prod/dev package split.
