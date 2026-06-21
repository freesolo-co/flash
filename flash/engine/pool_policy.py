"""The GPU policy step for pool-mode GRPO (advantage-weighted LoRA update).

This is the ONLY part of the distributed loop that runs on the trainer GPU — rollout and reward
already happened on the pool. It builds a PEFT LoRA model once and, per step, minimizes the
group-relative policy-gradient surrogate

    L = - mean_i  A_i * (1/|y_i|) * sum_t log pi_theta(y_i,t | x_i, y_i,<t)

over the sampled completions ``y_i`` with GRPO group-normalized advantages ``A_i`` (no value head;
KL/clip omitted — the advantages are already group-baselined). After the step it writes the updated
adapter and returns its directory so the loop can sync it to the pool.

torch/transformers/peft are imported lazily so this module (and ``flash.engine.pool_trainer``) import
without the GPU stack; the policy builder only runs on a worker.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from flash.pool.client import Experience

PolicyUpdate = Callable[[Experience, list[list[float]]], str | None]

# Language-only LoRA targets — for VL bases (Qwen3.5/3.6 are *ForConditionalGeneration) so the
# saved adapter has no vision-tower LoRA that a vLLM rollout server would reject.
_LANG_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _params_b(model_id: str) -> float:
    """Billions of parameters parsed from the model id (e.g. ...-0.8B / -4B / -35B-A3B -> 35)."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*B", model_id)
    return float(m.group(1)) if m else 7.0


def _is_qwen35(model_id: str) -> bool:
    s = model_id.lower()
    return "qwen3.5" in s or "qwen3.6" in s or "a3b" in s


@dataclass
class OptConfig:
    """Which of dev's training optimizations are on for this run (same set + defaults as
    origin/dev's worker: Liger, Chalk, FLA-drop on Hopper for Qwen3.5/3.6, QLoRA per the catalog
    quant tier, and the 8-bit paged AdamW optimizer)."""

    liger: bool
    chalk: bool
    drop_fla: bool
    qlora: bool
    optim_8bit: bool
    target_modules: object


def resolve_opt_config(
    model_id: str,
    *,
    liger: bool | None = None,
    chalk: bool | None = None,
    drop_fla: bool | None = None,
    qlora: bool | None = None,
    optim_8bit: bool | None = None,
    target_modules: object | None = None,
) -> OptConfig:
    """Resolve the optimization set for ``model_id`` with dev's defaults (each overridable). QLoRA
    defaults to ON for catalog 4-bit tiers (9B / 35B-A3B); FLA-drop defaults ON for Qwen3.5/3.6."""
    from flash.engine.worker.lora import model_quant

    quant = model_quant(model_id)
    qlora_default = ("4bit" in quant) or ("qlora" in quant)
    return OptConfig(
        liger=True if liger is None else liger,
        chalk=True if chalk is None else chalk,
        drop_fla=_is_qwen35(model_id) if drop_fla is None else drop_fla,
        qlora=qlora_default if qlora is None else qlora,
        optim_8bit=True if optim_8bit is None else optim_8bit,
        target_modules=target_modules or (_LANG_TARGETS if _is_qwen35(model_id) else "all-linear"),
    )


def pool_gpu_plan(model_id: str, *, opt: OptConfig | None = None) -> dict:
    """Pick the cheapest GPU class for the pool's TRAINER (LoRA only, NO colocated vLLM) and for the
    shared INFERENCE (vLLM rollout) — the disaggregation is exactly why the pool runs 35B where dev's
    colocate can't: a QLoRA 4-bit 35B trainer needs ~20 GB (fits an ordinary card), and only the
    shared rollout needs the big card."""
    from flash.providers.base import GPU_INFO

    opt = opt or resolve_opt_config(model_id)
    p = _params_b(model_id)
    bytes_per_param = 0.6 if opt.qlora else 2.0  # 4-bit nf4 (~0.5) + overhead, else bf16
    # trainer: base + LoRA + 8-bit optimizer state (small) + activations under grad checkpointing.
    trainer_gb = p * bytes_per_param * 1.15 + 5.0
    # inference (vLLM): the served base + KV cache, and vLLM only uses ~0.85 of VRAM, so size up;
    # an MoE (A3B) keeps ALL experts resident (expert-parallel) so it needs the full param weight.
    infer_gb = (p * bytes_per_param + 6.0) / 0.85

    def cheapest_fit(min_gb: float) -> tuple[str | None, float]:
        fits = [(g.hourly_usd, g.name) for g in GPU_INFO.values() if g.vram_gb >= min_gb]
        if not fits:
            biggest = max(GPU_INFO.values(), key=lambda g: g.vram_gb)
            return biggest.name, biggest.hourly_usd
        usd, name = min(fits)
        return name, usd

    t_gpu, t_usd = cheapest_fit(trainer_gb)
    i_gpu, i_usd = cheapest_fit(infer_gb)
    return {
        "model": model_id,
        "params_b": p,
        "qlora": opt.qlora,
        "trainer_gpu": t_gpu,
        "trainer_vram_gb": round(trainer_gb),
        "trainer_usd_hr": t_usd,
        "inference_gpu": i_gpu,
        "inference_vram_gb": round(infer_gb),
        "inference_usd_hr": i_usd,
    }


def _apply_liger(model, model_id: str) -> None:
    """Apply Liger fused kernels (RMSNorm/RoPE/SwiGLU) to the live model instance (dev uses TRL's
    use_liger_kernel; here we patch directly). Best-effort — degrades to eager if Liger is absent."""
    try:
        from liger_kernel.transformers import _apply_liger_kernel_to_instance

        base = getattr(getattr(model, "base_model", model), "model", model)  # unwrap PEFT
        _apply_liger_kernel_to_instance(model=base)
        print("[pool_policy] liger kernels applied", flush=True)
    except Exception as e:
        print(f"[pool_policy] liger skipped: {e}", flush=True)


def _build_optimizer(params, lr: float, use_8bit: bool):
    """8-bit paged AdamW (dev's fleet default via fused_optim_name/loraplus_optimizer_cls), fp32
    AdamW fallback."""
    import torch

    if use_8bit:
        try:
            from flash.engine.worker.perf import fused_optim_name, loraplus_optimizer_cls

            cls, kw = loraplus_optimizer_cls(fused_optim_name())  # -> bnb.optim.PagedAdamW8bit
            optim = cls(params, lr=lr, **kw)
            print(f"[pool_policy] optimizer = {cls.__name__}", flush=True)
            return optim
        except Exception as e:
            print(f"[pool_policy] 8-bit optimizer unavailable ({e}); fp32 AdamW", flush=True)
    return torch.optim.AdamW(params, lr=lr)


def build_lora_policy_update(
    model_id: str,
    *,
    out_dir: str,
    lora_rank: int = 32,
    lora_alpha: int | None = None,
    lr: float = 1e-5,
    max_len: int = 2048,
    device: str | None = None,
    render_prompt: Callable[[object, object], str] | None = None,
    target_modules: object | None = None,
    opt_config: OptConfig | None = None,
    grad_checkpointing: bool = True,
) -> tuple[PolicyUpdate, Callable[[], str]]:
    """Return ``(policy_update, current_uri)``. ``policy_update(exp, advantages)`` runs one optimizer
    step and writes the adapter to ``out_dir/step_<n>``, returning that path; ``current_uri()`` gives
    the latest adapter dir (the initial, untrained adapter before any step).

    Applies the SAME optimization stack as origin/dev's worker (resolved by :func:`resolve_opt_config`,
    overridable via ``opt_config``): FLA-drop on Hopper for Qwen3.5/3.6, QLoRA 4-bit base (catalog
    4-bit tiers — what lets a 35B trainer fit one card), Liger kernels, Chalk kernels, and the 8-bit
    paged AdamW optimizer. ``target_modules`` defaults to language-only for VL bases (so a vLLM
    rollout server loads the adapter cleanly)."""
    import torch

    oc = opt_config or resolve_opt_config(model_id, target_modules=target_modules)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    on_gpu = dev == "cuda" or (isinstance(dev, str) and dev.startswith("cuda"))

    # (1) FLA-drop: Qwen3.5/3.6 GDN crashes under Triton>=3.4 fla on Hopper — uninstall fla so the
    # native delta path runs (dev's _drop_fla_on_hopper, GPU-only; no-op elsewhere).
    if oc.drop_fla and on_gpu:
        try:
            from flash.engine.worker.perf import _drop_fla_on_hopper

            _drop_fla_on_hopper()
        except Exception as e:  # never abort training on the guard
            print(f"[pool_policy] drop_fla skipped: {e}", flush=True)

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # (2) QLoRA: load the frozen base in 4-bit NF4 (the COST/feasibility lever — 35B ~70GB bf16 ->
    # ~18-20GB, so the trainer fits an ordinary card; only the shared rollout needs the big GPU).
    load_kwargs = {"trust_remote_code": True}
    if oc.qlora and on_gpu:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16 if on_gpu else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    if not (oc.qlora and on_gpu):
        model = model.to(dev)
    if grad_checkpointing:
        model.gradient_checkpointing_enable()
    if oc.qlora and on_gpu:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=grad_checkpointing)

    peft_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha or 2 * lora_rank,
        target_modules=oc.target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)

    # (3) Liger kernels (fused RMSNorm/RoPE/SwiGLU) — dev sets TRL use_liger_kernel; here we apply
    # them directly to the instance. (4) Chalk gap-filling kernels on top (dev's install_chalk_kernels).
    if oc.liger and on_gpu:
        _apply_liger(model, model_id)
    if oc.chalk and on_gpu:
        try:
            from flash.engine.chalk_kernels import install_chalk_kernels

            rep = install_chalk_kernels(model)
            if rep:
                print(f"[pool_policy] chalk kernels: {rep}", flush=True)
        except Exception as e:
            print(f"[pool_policy] chalk skipped: {e}", flush=True)

    model.train()

    # (5) 8-bit paged AdamW (bitsandbytes) — dev's fleet default (fused_optim_name -> paged_adamw_8bit),
    # ~-75% optimizer-state memory with no convergence penalty. fp32 AdamW fallback if bnb absent.
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = _build_optimizer(trainable, lr, oc.optim_8bit and on_gpu)

    os.makedirs(out_dir, exist_ok=True)
    init_dir = os.path.join(out_dir, "step_0")
    model.save_pretrained(init_dir)
    tok.save_pretrained(init_dir)
    state = {"uri": init_dir}

    def _render(prompt) -> str:
        if isinstance(prompt, str):
            return prompt
        if render_prompt is not None:
            return render_prompt(tok, prompt)
        return tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

    def policy_update(exp: Experience, advantages: list[list[float]]) -> str:
        opt.zero_grad()
        total = _accumulate_loss(model, tok, dev, exp.prompts, exp.completions, advantages, _render, max_len)
        if total is not None:
            total.backward()
            opt.step()
            policy_update.last_loss = float(total.item())
        else:
            # No usable loss this step (e.g. empty/degenerate groups): record None — the "no update"
            # sentinel — rather than 0.0, which would read as a real zero-loss step in telemetry.
            policy_update.last_loss = None
        step_dir = os.path.join(out_dir, f"step_{exp.step + 1}")
        model.save_pretrained(step_dir)
        state["uri"] = step_dir
        return step_dir

    def logprob(prompt, completion: str) -> float:
        """Mean log pi(completion | prompt) under the CURRENT policy (eval probe — for telemetry /
        verifying the policy actually shifts toward high-reward completions)."""
        import torch

        ptext = _render(prompt)
        p_ids = tok(ptext, add_special_tokens=False)["input_ids"]
        c_ids = tok(completion, add_special_tokens=False)["input_ids"]
        if not c_ids:
            return 0.0
        ids = (p_ids + c_ids)[:max_len]
        n_c = min(len(c_ids), max(0, len(ids) - len(p_ids)))
        if n_c == 0:
            return 0.0
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                input_ids = torch.tensor([ids], device=dev)
                logits = model(input_ids).logits[0]
                lp = torch.log_softmax(logits[:-1], dim=-1)
                tok_lp = lp.gather(-1, input_ids[0, 1:].unsqueeze(-1)).squeeze(-1)
                return float(tok_lp[-n_c:].mean().item())
        finally:
            if was_training:
                model.train()

    policy_update.last_loss = None  # most recent advantage-weighted loss (for progress/telemetry)
    policy_update.logprob = logprob  # probe: mean logprob of a completion under the current policy
    return policy_update, (lambda: state["uri"])


def _accumulate_loss(model, tok, dev, prompts, completions, advantages, render, max_len):
    """Advantage-weighted NLL over all (prompt, completion) pairs in the batch. Returns a scalar
    tensor (or None if the batch has no usable completion tokens)."""
    import torch

    losses = []
    for prompt, group, adv_group in zip(prompts, completions, advantages, strict=True):
        ptext = render(prompt)
        p_ids = tok(ptext, add_special_tokens=False)["input_ids"]
        for completion, adv in zip(group, adv_group, strict=True):
            if adv == 0.0 or not completion:
                continue  # zero advantage contributes no gradient
            c_ids = tok(completion, add_special_tokens=False)["input_ids"]
            if not c_ids:
                continue
            ids = (p_ids + c_ids)[:max_len]
            n_c = min(len(c_ids), max(0, len(ids) - len(p_ids)))
            if n_c == 0:
                continue
            input_ids = torch.tensor([ids], device=dev)
            logits = model(input_ids).logits[0]  # [T, V]
            # logprob of each completion token from the preceding position
            logprobs = torch.log_softmax(logits[:-1], dim=-1)
            tgt = input_ids[0, 1:]
            tok_lp = logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [T-1]
            comp_lp = tok_lp[-n_c:].mean()  # mean log pi over the completion tokens
            losses.append(-adv * comp_lp)
    if not losses:
        return None
    return torch.stack(losses).mean()


def prompts_from_env(active_env, *, limit: int | None = None) -> list:
    """Pull prompt items from a verifiers env's dataset (best-effort across env shapes)."""
    ds = getattr(active_env, "dataset", None) or getattr(active_env, "train_dataset", None)
    if ds is None:
        return []
    prompts: list = []
    for row in ds:
        if isinstance(row, dict):
            prompts.append(row.get("prompt") or row.get("question") or row.get("text") or row)
        else:
            prompts.append(row)
        if limit is not None and len(prompts) >= limit:
            break
    return prompts


def batched(seq: Sequence, size: int):
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])
