"""Public-code contexts with explicit retrieval targets, fixed token IDs and seeds."""

# ruff: noqa: RUF001 -- exact checkpoint control tokens
import hashlib
import json
from pathlib import Path


def build(tokenizer, prefix_tokens=8192):
    source = Path("/opt/DeepInfer/deepinfer/worker/cache_engine.py").read_text()
    corpus = tokenizer.encode(source, add_special_tokens=False)
    cases = []
    for group in range(4):
        secret = ["ORCHID-7429", "MAPLE-3158", "CEDAR-9064", "LOTUS-5821"][group]
        header = tokenizer.encode(
            f"<｜begin▁of▁sentence｜><｜User｜>Project {group}. Reference material follows.\n",
            add_special_tokens=False,
        )
        fact = tokenizer.encode(
            f"\nThe deployment verification code for project {group} is {secret}.\n",
            add_special_tokens=False,
        )
        filler = corpus * (prefix_tokens // len(corpus) + 2)
        cut = prefix_tokens // 3
        prefix = header + filler[:cut] + fact
        prefix += filler[: prefix_tokens - len(prefix)]
        assert len(prefix) == prefix_tokens
        questions = [
            f"What is the deployment verification code for project {group}? Return only the code.",
            "Explain the purpose of the KV cache in three short sentences.",
            "Give two concrete tests for restoring the cached state after eviction.",
            "Explain why sliding-window state must be saved alongside attention keys and values.",
        ]
        for suffix, question in enumerate(questions):
            tail = tokenizer.encode(
                "\nEnd reference.\n" + question + "<｜Assistant｜></think>",
                add_special_tokens=False,
            )
            tokens = prefix + tail
            cases.append(
                {
                    "case": f"p{group}-q{suffix}",
                    "group": group,
                    "key": f"p{group}",
                    "tokens": tokens,
                    "cut": prefix_tokens,
                    "output": 128,
                    "expected": secret if suffix == 0 else None,
                }
            )
    return cases


def main():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("/weights/model", trust_remote_code=True)
    root = Path("/results")
    for length in [2048, 8192, 16384]:
        cases = build(tokenizer, length)
        payload = json.dumps(cases)
        (root / f"workload-{length}.json").write_text(payload)
        print("WORKLOAD", length, hashlib.sha256(payload.encode()).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
