"""Token-level SFT, clipped GRPO, and aligned sampled reverse-KL objectives."""

from __future__ import annotations

from flash.training.shared.contracts import ConflictError


def objective(logprobs, rows, algorithm, *, clip=0.2, normalizer=4096):
    import torch

    terms = []
    supervised = 0
    for logs, row in zip(logprobs, rows, strict=True):
        mask = torch.tensor(row["mask"], device=logs.device, dtype=torch.bool)
        supervised += int(mask.sum())
        if algorithm == "sft":
            terms.append(-logs[mask].sum())
        elif algorithm == "grpo":
            old = torch.tensor(row["old_logprobs"], device=logs.device, dtype=logs.dtype)
            advantage = torch.tensor(row["advantages"], device=logs.device, dtype=logs.dtype)
            delta = logs[mask] - old[mask]
            if bool((delta.abs() > 20).any()):
                raise ConflictError(
                    "rollout log probabilities are incompatible with current policy"
                )
            ratio = delta.exp()
            unclipped = ratio * advantage[mask]
            clipped = ratio.clamp(1 - clip, 1 + clip) * advantage[mask]
            terms.append(-torch.minimum(unclipped, clipped).sum() / normalizer)
        elif algorithm == "opd":
            groups = torch.tensor(row["teacher_groups"], device=logs.device)
            teacher = torch.tensor(row["teacher_logprobs"], device=logs.device, dtype=logs.dtype)
            selected = mask & (groups >= 0)
            group_terms = []
            for group in groups[selected].unique():
                positions = selected & (groups == group)
                student_sum = logs[positions].sum()
                coefficient = (student_sum.detach() - teacher[positions][0]) / positions.sum()
                group_terms.append(coefficient * student_sum)
            terms.append(torch.stack(group_terms).sum() / selected.sum())
        else:
            raise ValueError("unsupported training objective")
    loss = torch.stack(terms).sum()
    return loss / (supervised if algorithm == "sft" else len(rows))
