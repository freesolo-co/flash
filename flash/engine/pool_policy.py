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
from collections.abc import Callable, Sequence

from flash.pool.client import Experience

PolicyUpdate = Callable[[Experience, list[list[float]]], str | None]


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
) -> tuple[PolicyUpdate, Callable[[], str]]:
    """Return ``(policy_update, current_uri)``. ``policy_update(exp, advantages)`` runs one optimizer
    step and writes the adapter to ``out_dir/step_<n>``, returning that path; ``current_uri()`` gives
    the latest adapter dir (the initial, untrained adapter before any step)."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32, trust_remote_code=True
    ).to(dev)
    peft_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha or 2 * lora_rank,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.train()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)

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
        step_dir = os.path.join(out_dir, f"step_{exp.step + 1}")
        model.save_pretrained(step_dir)
        state["uri"] = step_dir
        return step_dir

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
