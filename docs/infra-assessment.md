# Infra assessment: vime / slime, and why the engine is TRL + vLLM

*June 2026. Written as part of the competitive-upgrade work; no further testing was
performed per the maintainer's confirmation.*

## Question

Would adopting **vime** (the June-2026 release connecting slime's Megatron-based
post-train stack to vLLM rollouts via vLLM's new native RL APIs) meaningfully improve
AutoSLM's underlying infrastructure?

## Answer: no — for AutoSLM's niche (1× consumer GPU, LoRA, SLMs), it is the wrong layer

Evidence gathered:

1. **No stable LoRA.** slime targets full-parameter RL scaling; LoRA support is an open
   issue (THUDM/slime #1202). The Megatron-Bridge LoRA path it depends on has had
   serious correctness bugs (dense-adapter TP-shard silent loss; MoE expert-adapter
   shard crashes) and zero TP>1 PEFT checkpoint test coverage. AutoSLM is LoRA-only by
   design — this alone disqualifies it today.
2. **Datacenter hardware assumptions.** slime/vime ship Docker images and CI for
   H100/H200/B200; Megatron + SGLang/vLLM colocation guidance assumes ≥80 GB cards.
   Nothing in its design serves a 24–32 GB RTX 4090/5090 with a 4B policy.
3. **Operational weight.** Megatron-LM + patched bridges + Ray is exactly the
   infrastructure-management burden AutoSLM exists to hide behind a TOML.

## What we adopted instead (and measured)

The same modernization vime represents — current vLLM with its RL/colocate APIs — is
consumed in AutoSLM through **TRL 1.x + vLLM 0.19** (both among the most actively
maintained options: vLLM releases ~weekly, TRL monthly minors). That upgrade delivered,
measured on real workers (`bench/results/phase1..3/`):

- Qwen3.5 / MiniCPM5 model support (transformers-5 architectures);
- GRPO per-step wall 46.6 s → ~15.5 s steady-state on the identical recipe;
- `DistillationTrainer` → the new `opd` algorithm (validated: GRPO-level lift at ~1/3
  the wall-clock on a 0.6B student);
- DPO support.

## Engine bake-off (TRL vs Unsloth)

Descoped by decision rule: the challenger had to cover the model matrix, and Unsloth's
GRPO path requires disabling vLLM fast inference for the Qwen3.5/hybrid family — a
regression on the centerpiece models. With the tuned TRL recipe within ~1.4× of the
hosted-API per-step reference, a second engine integration wasn't justified.

## Revisit when

- slime/vime lands first-class, tested LoRA (then re-evaluate for MULTI-GPU AutoSLM
  tiers, not single-card);
- vLLM's native weight-sync APIs become relevant if AutoSLM ever splits rollout and
  trainer across two cheap GPUs (disaggregated mode);
- Unsloth gains vLLM-backed GRPO for hybrid architectures (re-run the bake-off rule).
