"""Real-model shared adapter qualification, invoked inside the owned GPU sandbox."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import time


def main():
    import torch
    from peft import get_peft_model_state_dict
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )

    from flash.training.shared.backend import PeftBackend

    model_id, revision = (
        os.environ["FLASH_SHARED_MODEL_ID"],
        os.environ["FLASH_SHARED_MODEL_REVISION"],
    )
    config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    loader = (
        AutoModelForImageTextToText
        if getattr(config, "text_config", None) is not None
        else AutoModelForCausalLM
    )
    started = time.monotonic()
    model = loader.from_pretrained(
        model_id, revision=revision, dtype=torch.bfloat16, trust_remote_code=False
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    prompt = tokenizer.encode("Question: what is 2 + 2? Answer:", add_special_tokens=False)
    target = tokenizer.encode(" 4.", add_special_tokens=False)
    result = {
        "model": model_id,
        "revision": revision,
        "gpu": torch.cuda.get_device_name(),
        "load_seconds": time.monotonic() - started,
        "algorithms": {},
    }
    with tempfile.TemporaryDirectory() as directory:
        backend = PeftBackend(
            model, root=directory, rank=8, alpha=16, base_identity=f"{model_id}@{revision}"
        )

        def weights(name):
            return {
                k: v.detach().clone()
                for k, v in get_peft_model_state_dict(backend.model, adapter_name=name).items()
            }

        frozen = [(p, p._version) for p in model.parameters() if not p.requires_grad]
        for algorithm in ("sft", "grpo", "opd"):
            one, two, restored = (algorithm + suffix for suffix in ("_one", "_two", "_restored"))
            session_config = {"algorithm": algorithm, "seed": 42, "learning_rate": 1e-4}
            backend.create(one, session_config)
            backend.create(two, {**session_config, "seed": 43})
            sampled = backend.sample(
                one, {"version": 0, "prompt": prompt, "max_tokens": 4, "seed": 17}
            )
            response = target if algorithm == "sft" else sampled["tokens"]
            row = {
                "tokens": prompt + response,
                "mask": [False] * (len(prompt) - 1) + [True] * len(response),
            }
            prefix = len(prompt) - 1
            if algorithm == "grpo":
                row.update(
                    old_logprobs=[0.0] * prefix + sampled["logprobs"],
                    advantages=[0.0] * prefix + [1.0] * len(response),
                )
            if algorithm == "opd":
                row.update(
                    teacher_groups=[-1] * prefix + list(range(len(response))),
                    teacher_logprobs=[0.0] * prefix + [v - 0.25 for v in sampled["logprobs"]],
                )
            before_one, before_two = weights(one), weights(two)
            first = backend.train(one, {"version": 0, "rows": [row]})
            after_one = weights(one)
            assert any(not torch.equal(before_one[k], after_one[k]) for k in after_one)
            assert all(torch.equal(before_two[k], weights(two)[k]) for k in before_two)
            optimizer = copy.deepcopy(backend.adapters[one]["optimizer"].state_dict())
            backend.train(two, {"version": 0, "rows": [row]})
            assert all(torch.equal(after_one[k], weights(one)[k]) for k in after_one)
            for key, state in optimizer["state"].items():
                for field, value in state.items():
                    other = backend.adapters[one]["optimizer"].state_dict()["state"][key][field]
                    assert (
                        torch.equal(value, other)
                        if isinstance(value, torch.Tensor)
                        else value == other
                    )
            saved = backend.checkpoint(one, {})
            second = backend.train(one, {"version": 1, "rows": [row]})
            final = weights(one)
            backend.create(restored, session_config)
            backend.restore(restored, {"checkpoint": saved["checkpoint"]})
            continuation = backend.train(restored, {"version": 1, "rows": [row]})
            assert continuation == second
            assert all(torch.equal(final[k], weights(restored)[k]) for k in final)
            assert all(p.grad is None and p._version == version for p, version in frozen)
            result["algorithms"][algorithm] = {
                "first_loss": first["loss"],
                "second_loss": second["loss"],
                "adapter_changed": True,
                "peer_unchanged": True,
                "optimizer_isolated": True,
                "resume_exact": True,
                "base_unchanged": True,
            }
            for name in (one, two, restored):
                backend.close(name, {})
    result["wall_seconds"] = time.monotonic() - started
    print("SHARED_MODEL_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
