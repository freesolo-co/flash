"""Client-side recipe steps; rewards and teacher credentials stay outside the GPU pool.

These explicit token contracts intentionally do not execute environment Python in
the shared process. The caller owns dataset iteration and checkpoint cadence.
"""

from __future__ import annotations

import math


def sft_step(session, examples):
    """Train one token-weighted update from (prompt_ids, completion_ids) examples."""
    rows = []
    for prompt, completion in examples:
        if not prompt or not completion:
            raise ValueError("SFT requires nonempty prompt and completion")
        rows.append(
            {
                "tokens": list(prompt) + list(completion),
                "mask": [False] * (len(prompt) - 1) + [True] * len(completion),
            }
        )
    return session.call("train", {"version": session.version, "rows": rows})


def grpo_step(session, prompts, reward, *, group_size=4, max_tokens=128, seed=0):
    """Group-mean advantages, no standard-deviation normalization, fixed-length loss."""
    if type(group_size) is not int or group_size < 2:
        raise ValueError("GRPO requires at least two samples per prompt")
    rows = []
    for prompt_index, prompt in enumerate(prompts):
        samples = [
            session.sample(
                prompt, max_tokens=max_tokens, seed=seed + prompt_index * group_size + index
            )
            for index in range(group_size)
        ]
        rewards = [float(reward(prompt, sample["tokens"])) for sample in samples]
        if not all(math.isfinite(value) for value in rewards):
            raise ValueError("rewards must be finite")
        mean = sum(rewards) / len(rewards)
        for sample, value in zip(samples, rewards, strict=True):
            length = len(sample["tokens"])
            prefix = len(prompt) - 1
            rows.append(
                {
                    "tokens": list(prompt) + sample["tokens"],
                    "mask": [False] * prefix + [True] * length,
                    "old_logprobs": [0.0] * prefix + sample["logprobs"],
                    "advantages": [0.0] * prefix + [value - mean] * length,
                }
            )
    if not any(any(row["advantages"]) for row in rows):
        raise ValueError("GRPO batch has no reward variation")
    return session.call(
        "train", {"version": session.version, "rows": rows, "normalizer": max_tokens}
    )


def opd_step(session, prompts, teacher, *, max_tokens=128, seed=0):
    """Sample the student, then obtain aligned teacher group ids and group logsums.

    teacher(prompt_ids, completion_ids) returns (group_ids, logprobs), one entry per
    completion token. Group -1 excludes unaligned tokens. Every token in a group
    contains the same teacher log-probability sum for that aligned group.
    """
    rows = []
    for index, prompt in enumerate(prompts):
        sample = session.sample(prompt, max_tokens=max_tokens, seed=seed + index)
        groups, logs = teacher(prompt, sample["tokens"])
        if len(groups) != len(sample["tokens"]) or len(logs) != len(groups):
            raise ValueError("teacher metadata must align with student completion")
        prefix = len(prompt) - 1
        rows.append(
            {
                "tokens": list(prompt) + sample["tokens"],
                "mask": [False] * prefix + [True] * len(groups),
                "teacher_groups": [-1] * prefix + list(groups),
                "teacher_logprobs": [0.0] * prefix + list(logs),
            }
        )
    return session.call("train", {"version": session.version, "rows": rows})
