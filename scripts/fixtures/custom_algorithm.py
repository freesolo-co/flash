"""DPO conformance controller used by the recurring managed training smoke."""

import asyncio
import json
from pathlib import Path

from flash.custom import Datum
from flash.custom.starter import make_dpo_loss


async def main(client):
    if client.restored_state:
        state = client.restored_state
        expected = json.loads(await asyncio.to_thread(Path("expected.json").read_text))
        assert state == expected["algorithm_state"]
        assert client.step == 2
        data = [Datum(tokens) for tokens in state["tokens"]]
        scored = await client.forward(data)
        assert [row["logprobs"] for row in scored["outputs"]] == expected["logprobs"]
        print("RESTORE_EXACT: logits and algorithm state", flush=True)
    else:
        prompt = await client.encode("Question: What is 2 + 2?\nAnswer:")
        chosen = await client.encode(" 4.")
        rejected = await client.encode(" 9.")
        data = [Datum(prompt + chosen), Datum(prompt + rejected)]
        scored = await client.forward(data, top_k=5)
        state = {
            "tokens": [row.tokens for row in data],
            "prompt_lengths": [len(prompt), len(prompt)],
            "reference": [row["logprobs"] for row in scored["outputs"]],
            "next_step": 0,
        }
    objective = make_dpo_loss(state["reference"], state["prompt_lengths"])
    for step in range(client.step, client.config["train"]["max_steps"]):
        result = await client.forward_backward_custom(data, objective)
        state = {**state, "next_step": step + 1}
        update = await client.optim_step(learning_rate=3e-4, algorithm_state=state)
        assert update["metrics"]["grad_norm"] > 0
        print(json.dumps({"step": step + 1, **result["metrics"], **update["metrics"]}), flush=True)
    scored = await client.forward(data)
    if client.step == 2:
        await asyncio.to_thread(
            Path("expected.json").write_text,
            json.dumps(
                {
                    "logprobs": [row["logprobs"] for row in scored["outputs"]],
                    "algorithm_state": state,
                }
            ),
        )
    await client.download_checkpoint(f"step-{client.step}", f"step-{client.step}.zip")
