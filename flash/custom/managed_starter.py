"""Local DPO. Select this file with algorithm = "custom" and [custom] file in run.toml.

Run: flash train run.toml
The local function can be async def main(client) or def main(client).
"""

import asyncio
import json
from pathlib import Path

from flash.custom import Datum


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


async def main(client):
    restored = client.restored_state
    if restored:
        data = [Datum(tokens) for tokens in restored["tokens"]]
        prompt_lengths = restored["prompt_lengths"]
        reference = restored["reference"]
    else:
        if client.step:
            raise ValueError("DPO resume requires the original frozen reference in algorithm_state")
        pairs = [
            ("Question: What is 2 + 2?\nAnswer:", " 4.", " 9."),
            ("Question: What is the capital of France?\nAnswer:", " Paris.", " Berlin."),
        ]
        data, prompt_lengths = [], []
        for text, chosen, rejected in pairs:
            prompt = await client.encode(text)
            for answer in (chosen, rejected):
                data.append(Datum(prompt + await client.encode(answer)))
                prompt_lengths.append(len(prompt))
        scored = await client.forward(data, top_k=5)
        reference = [row["logprobs"] for row in scored["outputs"]]
        await asyncio.to_thread(
            Path("initial-probabilities.json").write_text, json.dumps(scored, indent=2)
        )
    objective = make_dpo_loss(reference, prompt_lengths)
    for step in range(client.step, client.config["train"]["max_steps"]):
        backward = await client.forward_backward_custom(data, objective)
        update = await client.optim_step(
            learning_rate=3e-4,
            algorithm_state={
                "tokens": [row.tokens for row in data],
                "prompt_lengths": prompt_lengths,
                "reference": reference,
                "next_step": step + 1,
            },
        )
        print(
            json.dumps({"step": step + 1, **backward["metrics"], **update["metrics"]}), flush=True
        )
        if update.get("checkpoint_ref"):
            print("Saved", update["checkpoint_ref"], flush=True)
    print(
        json.dumps(
            await client.sample(data[0].tokens[: prompt_lengths[0]], max_tokens=16, temperature=0),
            indent=2,
        )
    )
