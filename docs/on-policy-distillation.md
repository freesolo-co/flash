# On-policy distillation (`algorithm = "opd"`)

Flash's third training algorithm, alongside `sft` and `grpo`. It trains a **student** model (a Qwen /
MiniCPM / Kimi catalog model) to imitate a much stronger **teacher** — a Fireworks-hosted GLM model
(`accounts/fireworks/models/glm-5p2`) reached over the OpenAI-compatible API — on the student's *own*
on-policy generations.

Unlike SFT (imitate a fixed gold answer) or GRPO (optimise a scalar reward), distillation gives a
**dense, per-token** learning signal from the teacher, which is far more sample-efficient than
reward-based RL and does not need a hand-designed reward.

## Why on-policy

Off-policy sequence distillation (train on text the *teacher* generated) suffers from exposure bias:
the student is never corrected on the mistakes *it* actually makes at inference time. On-policy
distillation closes that loop — the **student samples**, and the **teacher grades those samples
token-by-token**, so the gradient is concentrated exactly where the student diverges from the teacher
(Agarwal et al. 2023, *GKD — Generalized Knowledge Distillation*; Thinking Machines, *On-Policy
Distillation*, 2025).

Per step:

1. Sample a batch of prompts from the run's environment (`env.dataset()` / `env.prompt_messages()`).
2. The **student** generates completions on-policy (HF `model.generate` with the live LoRA).
3. The **teacher** echo-scores each completion (`echo=true, logprobs=1, max_tokens=0` — returns the
   realized per-token `token_logprobs` and character `text_offset`) — *without* paying for generation.
4. The teacher signal is aligned onto the student's tokens (see below) and a groupwise reverse-KL loss
   is backpropagated through the student LoRA.
5. The adapter is checkpointed to HF exactly like SFT; it deploys/serves through the identical path
   (the serving side never calls the teacher).

The student stays resident; there is **no local frozen reference model** (GRPO's extra forward)
because the teacher's per-token signal arrives from the API. That makes an opd step's compute
`generate + policy fwd/bwd` (~6 FLOPs/token/param), lighter than GRPO's `generate + policy + reference`
(~8).

## The cross-tokenizer problem

The teacher (GLM) and the student (Qwen / MiniCPM / Kimi) have **different tokenizers and
vocabularies**. They segment the same completion string at different byte boundaries, and token id *N*
means something different in each vocab — so the teacher's per-token distribution cannot be compared
to the student's directly, and there is no shared vocabulary to project onto.

Flash bridges this with **groupwise reverse-KL** — the realignment used by
[`collinear-ai/spider`](https://github.com/collinear-ai/spider) and the Tinker-cookbook on-policy
distillation reference (`_build_alignment_groups` / `_compute_groupwise_reverse_kl`). Instead of
comparing per-position *distributions*, it compares the **probability of matching text spans**:

1. **Character-offset anchoring.** Both sides expose character offsets into the same completion string
   — the student via the fast tokenizer's `return_offsets_mapping`, the teacher via Fireworks'
   `text_offset` from echo scoring. The loss uses the student's *sampled* token ids (not a
   re-tokenization of the decoded text), so it trains the true on-policy tokens.
2. **Shared-span alignment.** A group boundary is any character position that begins a token in
   **both** tokenizers (the coarsest common refinement of the two tokenizations). Between consecutive
   shared boundaries, the student tokens and teacher tokens covering that span form one group. Where
   the tokenizers disagree locally, the span just **grows until they next agree**, so *every* student
   token is covered — no masking, and student-only spans are merged into their neighbor rather than
   dropped.
3. **Per-span reverse KL.** For each span, sum the teacher's realized-token logprobs
   (`log P_teacher(span)`) and the student's (`log P_student(span)`). The per-token advantage is the
   negative reverse KL, `−(log P_student(span) − log P_teacher(span)) / |span|`, spread across the
   student tokens in the span, and applied as an importance-sampling / REINFORCE policy-gradient
   surrogate (`loss = Σ reverse_kl · log π_student`). This needs **only each side's logprob of the
   text it actually produced** — no top-k candidates, no vocabulary projection, no shared tokenizer.

Because it only ever uses realized-token logprobs over matching text, it is exact across arbitrary
tokenizer mismatch and works for any student (Qwen3.5, MiniCPM, Kimi, …) against the GLM teacher.
**When the two tokenizers happen to agree token-for-token**, every span is a single student token
against a single teacher token, and this collapses to exactly the standard per-token reverse-KL
on-policy distillation of Agarwal / Thinking Machines.

The whole alignment is pure/offline (`flash/engine/worker/tokenizer_align.py`); the loss lives in
`flash/engine/worker/opd.py` (`gkd_loss`).

## Two-stage pipelines (SFT → opd)

opd honors `[train] init_from_adapter`, so it can **warm-start from an SFT adapter** (continuing that
LoRA, the same mechanism GRPO uses) rather than starting from base — making `SFT → opd` a genuine
continuation. Point the opd run's `init_from_adapter` at the SFT run's checkpoint ref
(`<run_id>` or `<run_id>/step-N`).

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
# kl_penalty_coef = 1.0              # reverse-KL scale (1.0 = plain reverse KL)
# temperature = 1.0                  # student on-policy sampling temperature
# max_tokens = 512                   # student completion budget
# group_size = 1                     # student samples per prompt
# init_from_adapter = "<sft_run_id>" # warm-start from an SFT adapter (SFT -> opd)
```

`FIREWORKS_API_KEY` is a training-time runtime secret (never stored in the spec, never needed at
serving time). It rides the same out-of-band channel as `WANDB_API_KEY`, and is auto-declared a
required secret for opd runs.

## Cost

An opd run is billed on training-loop GPU time (like every algorithm) **plus** the external Fireworks
teacher token spend (`flash train --cost` shows both lines). Teacher scoring uses `max_tokens=0` echo,
so it is priced on *input* tokens only (`prompt + completion` per scored sample) — the teacher never
generates.
