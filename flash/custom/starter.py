"""An editable, local DPO algorithm. Flash only performs the requested GPU primitives.

Install: pip install 'freesolo-flash[custom]'
Run: python algorithm.py --gpu A10G --steps 6 --output ./my-training
Use your Modal profile or MODAL_TOKEN_ID / MODAL_TOKEN_SECRET for GPU provisioning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flash.custom import Datum, TrainingClient


def build_pairs(client):
    # replace these with your preference dataset, or call client.sample and score the responses.
    preferences = [
        ("Question: What is 2 + 2?\nAnswer:", " 4.", " 9."),
        ("Question: What is the capital of France?\nAnswer:", " Paris.", " Berlin."),
    ]
    rows, prompt_lengths = [], []
    for prompt_text, chosen_text, rejected_text in preferences:
        prompt = client.encode(prompt_text)
        for answer in (chosen_text, rejected_text):
            rows.append(Datum(prompt + client.encode(answer)))
            prompt_lengths.append(len(prompt))
    return rows, prompt_lengths


def make_dpo_loss(reference, prompt_lengths, *, beta=0.1):
    # reference probabilities stay frozen at the initial adapter. chosen/rejected rows are paired.
    reference_scores = [
        sum(probs[prompt_length - 1 :])
        for probs, prompt_length in zip(reference, prompt_lengths, strict=True)
    ]

    def loss_fn(data, logprobs):
        import torch
        import torch.nn.functional as functional

        # position i predicts tokens[i+1]; the first completion target is prompt_length-1.
        scores = [
            probs[prompt_length - 1 :].sum()
            for probs, prompt_length in zip(logprobs, prompt_lengths, strict=True)
        ]
        margins = torch.stack(
            [
                scores[index]
                - scores[index + 1]
                - (reference_scores[index] - reference_scores[index + 1])
                for index in range(0, len(data), 2)
            ]
        )
        loss = -functional.logsigmoid(beta * margins).mean()
        return loss, {"preference_margin": margins.mean().item()}

    return loss_fn


def train(client, *, steps, output):
    output.mkdir(parents=True, exist_ok=True)
    data, prompt_lengths = build_pairs(client)
    scored = client.forward(data, top_k=5)
    # keep inspectable token-level evidence locally, alongside the algorithm's own metrics.
    (output / "initial-probabilities.json").write_text(
        json.dumps(
            {
                "data": [row.to_dict() for row in data],
                "forward": scored,
            },
            indent=2,
        )
    )
    reference = [row["logprobs"] for row in scored["outputs"]]
    objective = make_dpo_loss(reference, prompt_lengths)
    with (output / "metrics.jsonl").open("x") as metrics_file:
        for step in range(steps):
            backward = client.forward_backward_custom(data, objective)
            update = client.optim_step(learning_rate=3e-4, max_grad_norm=1.0)
            metrics = {"step": step + 1, **backward["metrics"], **update["metrics"]}
            metrics_file.write(json.dumps(metrics) + "\n")
            metrics_file.flush()
            print(json.dumps(metrics), flush=True)
    client.save_checkpoint("final")
    client.download_checkpoint("final", output / "final.zip")
    (output / "worker-events.json").write_text(json.dumps(client.logs(), indent=2))
    prompt = data[0].tokens[: prompt_lengths[0]]
    print(json.dumps(client.sample(prompt, max_tokens=16, temperature=0.0), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--gpu", default="A10G")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("my-training"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output directory to preserve previous results")
    with TrainingClient.create(
        args.model,
        gpu=args.gpu,
        max_steps=args.steps,
        max_wall_seconds=1800,
        idle_seconds=300,
        max_context_tokens=512,
    ) as client:
        train(client, steps=args.steps, output=args.output)


if __name__ == "__main__":
    main()
