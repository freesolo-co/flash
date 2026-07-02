# On-policy distillation (`algorithm = "opd"`)

Flash's third training algorithm, alongside `sft` and `grpo`. It trains a **student**
model (e.g. a Qwen catalog model) to imitate a much stronger **teacher** — a
Fireworks-hosted GLM model (`accounts/fireworks/models/glm-5p2`) reached over the
OpenAI-compatible API — on the student's *own* on-policy generations.

Unlike SFT (imitate a fixed gold answer) or GRPO (optimise a scalar reward), distillation
gives a **dense, per-token** learning signal from the teacher, which is far more
sample-efficient than reward-based RL and does not need a hand-designed reward.

## Why on-policy

Off-policy sequence distillation (train on text the *teacher* generated) suffers from
exposure bias: the student is never corrected on the mistakes *it* actually makes at
inference time. On-policy distillation closes that loop — the **student samples**, and the
**teacher grades those samples token-by-token**, so the gradient is concentrated exactly
where the student diverges from the teacher (Agarwal et al. 2023, *GKD*; Thinking Machines,
*On-Policy Distillation*, 2025).

Per step:

1. Sample a batch of prompts from the run's environment (`env.dataset()` / `env.prompt_messages()`).
2. The **student** generates completions on-policy (HF `model.generate` with the live LoRA).
3. The **teacher** scores each completion (`echo=true, logprobs=k` — returns per-token
   `token_logprobs`, `top_logprobs`, and character `text_offset`) — *without* paying for
   generation (`max_tokens=0`).
4. The teacher signal is aligned onto the student's tokens (see below) and a per-token
   distillation loss is backpropagated through the student LoRA.
5. The adapter is checkpointed to HF exactly like SFT; it deploys/serves through the
   identical path (the serving side never calls the teacher).

The student stays resident; there is **no local frozen reference model** (GRPO's extra
forward) because the teacher's per-token distribution arrives from the API. That makes a
opd step's compute `generate + policy fwd/bwd` (~6 FLOPs/token/param), lighter than
GRPO's `generate + policy + reference` (~8).

## The cross-tokenizer problem

The teacher (GLM) and the student (Qwen) have **different tokenizers and vocabularies**. A
teacher token like `" capital"` has no fixed identity in the student's vocab, and the two
segment the same string at different byte boundaries. So the teacher's per-token
distribution cannot be compared to the student's directly. Flash ships **three** strategies,
selectable with `[train] tokenizer_alignment` (default `align`), each a different answer to
this problem — pick by how divergent the two tokenizations are and how much signal you want:

### 1. `seqkd` — sequence-level KD by re-tokenization (tokenizer-free)

Sidesteps the mismatch entirely. The teacher *generates* a target completion (chat
completion; its inline reasoning is stripped to the final answer). The student is trained
with plain completion-only cross-entropy on that text **re-tokenized in the student's own
vocabulary**. No logits, no alignment — the most robust fallback, and a strong baseline.
It is *off-policy* (teacher-generated targets), so it trades the on-policy correction for
maximum robustness.

### 2. `align` — sparse top-k forward-KL over a character-span alignment (default)

The true on-policy path with a dense per-token teacher signal. The student samples; the
teacher echo-scores that exact string. Because both tokenizers expose **character offsets**
(`text_offset` from Fireworks; `return_offsets_mapping` from the student tokenizer), teacher
and student token boundaries are aligned by their character spans. At each student
completion position the teacher's **top-k** next-token candidates (surface strings) are
projected onto the student vocabulary (student-tokenize each candidate, take its first
token id) to form a sparse target distribution `p_T`, and we minimise the forward-KL
cross-entropy `−Σ p_T(v)·log p_S(v)` at that position. Mode-covering, low-variance, and
genuinely on-policy. This is the recommended default for Qwen students distilling from GLM.

### 3. `uld` — Universal Logit Distillation (sorted-distribution / optimal transport)

The most robust to *severe* tokenizer divergence. It never maps token identities at all:
at each aligned position it compares the **sorted** top-k probability vectors of teacher and
student (a truncated Wasserstein-1 / L1 between the sorted distributions), following Boizard
et al. 2024, *Universal Logit Distillation*. Vocabulary-agnostic — it only needs the teacher
top-k and a position alignment — at the cost of a coarser (distribution-shape-only) signal
than `align`.

| strategy | on-policy | needs teacher logprobs | needs token-identity map | signal density | robustness to divergence |
|----------|-----------|------------------------|--------------------------|----------------|--------------------------|
| `seqkd`  | no        | no (generate only)     | no                       | sequence CE    | highest                  |
| `align`  | yes       | yes (top-k + offsets)  | yes (surface→student)    | dense per-token| medium                   |
| `uld`    | yes       | yes (top-k + offsets)  | no (sorted only)         | per-token shape| high                     |

All three run against the same teacher client and the same on-policy sampling loop; only the
per-token loss differs (`flash/engine/worker/tokenizer_align.py`).

## Configuration

```toml
# configs/opd.toml  (scaffolded by `flash env setup`)
model = "Qwen/Qwen3.5-4B"
algorithm = "opd"

[environment]
id = "your-org/my-env"
secrets = ["FIREWORKS_API_KEY"]     # the teacher key; value read from your shell/.env at submit

[train]
steps = 100                          # opd is step-driven (like GRPO)
lora_rank = 32
# teacher_model = "accounts/fireworks/models/glm-5p2"   # default
# tokenizer_alignment = "align"      # align | seqkd | uld
# teacher_top_logprobs = 5           # k for align/uld (Fireworks echo caps at 5)
# temperature = 1.0                  # student on-policy sampling temperature
# max_tokens = 512                   # student completion budget
# group_size = 1                     # student samples per prompt
```

`FIREWORKS_API_KEY` is a training-time runtime secret (never stored in the spec, never
needed at serving time). It rides the same out-of-band channel as `WANDB_API_KEY`.

## Cost

A opd run is billed on training-loop GPU time (like every algorithm) **plus** the
external Fireworks teacher token spend (`flash train --cost` shows both lines). Teacher
scoring uses `max_tokens=0` echo, so it is priced on *input* tokens only
(`prompt + completion` per scored sample).
