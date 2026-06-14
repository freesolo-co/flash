"""GSM8K data loading + prompt formatting + fixed seeded eval subset.

Moved out of ``autoslm.engine`` into this example so the core package carries no
task-specific data wiring. Shared by both training arms here so the data,
ordering, and eval set are identical. The only dependency back into the package
is the generic, task-agnostic ``eval_subset_indices`` helper.
"""

from __future__ import annotations

from autoslm.engine.data import eval_subset_indices

from .grading import extract_gold_answer

# System / instruction prompt used identically on both arms.
SYSTEM_PROMPT = (
    "You are a careful math assistant. Solve the grade-school math problem "
    "step by step, then give the final numeric answer on its own line in the "
    "form \\boxed{ANSWER}."
)


def build_prompt_messages(question: str) -> list[dict]:
    """Chat-format the question. Tokenized via the model's own chat template
    on both arms (identical checkpoint => identical template)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]


def build_target_text(solution: str) -> str:
    """Build the SFT target completion from a GSM8K gold solution.

    Converts the trailing '#### N' into reasoning + '\\boxed{N}' so the SFT
    target matches the format the eval/reward expects.
    """
    ans = extract_gold_answer(solution)
    # Strip the '#### N' tail from the reasoning, keep the chain-of-thought.
    body = solution.split("####")[0].strip()
    return f"{body}\nThe final answer is \\boxed{{{ans}}}."


def load_gsm8k(split: str, dataset: str = "openai/gsm8k", dataset_config: str = "main"):
    """Load GSM8K split as a list of {question, solution, gold} dicts."""
    from datasets import load_dataset

    ds = load_dataset(dataset, dataset_config, split=split)
    return [
        {
            "question": ex["question"],
            "solution": ex["answer"],
            "gold": extract_gold_answer(ex["answer"]),
        }
        for ex in ds
    ]


def fixed_eval_subset(
    num_examples: int,
    subset_seed: int,
    dataset: str = "openai/gsm8k",
    dataset_config: str = "main",
    split: str = "test",
) -> list[dict]:
    """Deterministic eval subset: same indices on both arms for a given seed."""
    full = load_gsm8k(split, dataset, dataset_config)
    return [full[i] for i in eval_subset_indices(num_examples, subset_seed, len(full))]
