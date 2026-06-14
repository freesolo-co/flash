"""GSM8K data loading + prompt formatting.

Moved out of ``autoslm.engine`` into this example so the core package carries no
task-specific data wiring. Shared by both training arms here so the data and
ordering are identical.
"""

from __future__ import annotations

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
    target matches the format the grader/reward expects.
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
