"""Real-GPU conformance for the public custom training client and a user-authored DPO loss.

Run explicitly: python scripts/check_custom_gpu.py --output /tmp/custom-gpu.json
Allocates one dedicated Modal GPU and always attempts to terminate it.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import zipfile
from pathlib import Path

import torch

from flash.client.http import ApiError, RequestTimeoutError
from flash.custom import Datum, TrainingClient


def dpo_loss(data, logprobs, reference, prompt_lengths, beta=0.1):
    import torch.nn.functional as functional

    scores = [
        values[length - 1 :].sum() for values, length in zip(logprobs, prompt_lengths, strict=True)
    ]
    reference_scores = [
        sum(values[length - 1 :]) for values, length in zip(reference, prompt_lengths, strict=True)
    ]
    margins = torch.stack(
        [
            (scores[index] - scores[index + 1])
            - (reference_scores[index] - reference_scores[index + 1])
            for index in range(0, len(data), 2)
        ]
    )
    loss = -functional.logsigmoid(beta * margins).mean()
    return loss, {"preference_margin": margins.mean().item()}


def exercise(client, *, steps):
    prompt = client.encode("Question: What is 2 + 2?\nAnswer:")
    chosen = client.encode(" 4.")
    rejected = client.encode(" 9.")
    data = [Datum(prompt + chosen), Datum(prompt + rejected)]
    lengths = [len(prompt), len(prompt)]
    baseline = client.save_checkpoint("baseline")["checkpoint"]
    scored = client.forward(data, top_k=5)
    reference = [row["logprobs"] for row in scored["outputs"]]
    assert len(reference[0]) == len(data[0].tokens) - 1
    assert all(value <= 0 for row in reference for value in row)
    assert len(scored["outputs"][0]["top_logprobs"][0]) == 5
    sampled = client.sample(prompt, max_tokens=8, num_samples=2, temperature=0.8)
    for sequence in sampled["sequences"]:
        rescored = client.forward([Datum(prompt + sequence["tokens"])])["outputs"][0]["logprobs"]
        assert torch.allclose(
            torch.tensor(rescored[len(prompt) - 1 :]), torch.tensor(sequence["logprobs"]), atol=1e-5
        )
        assert len(sequence["sampling_logprobs"]) == len(sequence["tokens"])
    history = []

    def loss_fn(rows, probs):
        return dpo_loss(rows, probs, reference, lengths)

    first_step = None
    for index in range(steps):
        result = client.forward_backward_custom(data, loss_fn)
        update = client.optim_step(learning_rate=3e-4)
        assert update["metrics"]["grad_norm"] > 0
        history.append({**result["metrics"], **update["metrics"]})
        if index == 0:
            first_step = client.save_checkpoint("first-step")["checkpoint"]
    after_probs = [torch.tensor(row["logprobs"]) for row in client.forward(data)["outputs"]]
    final_loss, _ = loss_fn(data, after_probs)
    assert final_loss.item() < history[0]["custom_loss"], (
        "DPO did not improve the preference objective"
    )
    trained = client.save_checkpoint("trained")["checkpoint"]
    assert trained["parameter_sha256"] != baseline["parameter_sha256"]
    with tempfile.TemporaryDirectory() as directory:
        archive = client.download_checkpoint("trained", Path(directory) / "trained.zip")
        with zipfile.ZipFile(archive) as checkpoint:
            assert "state.pt" in checkpoint.namelist()
            assert "adapter/adapter_config.json" in checkpoint.namelist()
    before_rejection = client.status()
    try:
        client.forward_backward([Datum(prompt + chosen, [1.0])])
    except ApiError as error:
        if error.status != 400:
            raise AssertionError("invalid mask should be rejected with HTTP 400") from error
    else:
        raise AssertionError("misaligned weights accepted")
    assert client.status() == before_rejection
    client.load_checkpoint("baseline")
    restored = client.forward(data)
    assert [row["logprobs"] for row in restored["outputs"]] == reference
    client.forward_backward_custom(data, loss_fn)
    real_request = client._http._request

    def lost_response(method, path, body=None, **kwargs):
        result = real_request(method, path, body, **kwargs)
        if body and body.get("operation") == "optim_step":
            raise RequestTimeoutError("simulated lost response AFTER real GPU update")
        return result

    before_step = client.status()["step"]
    client._http._request = lost_response
    try:
        client.optim_step(learning_rate=3e-4)
    except RequestTimeoutError:
        pass
    else:
        raise AssertionError("response-loss simulation did not fire")
    finally:
        client._http._request = real_request
    pending = client.pending_request
    assert pending is not None
    retry = client.retry_pending()
    assert retry["step"] == before_step + 1
    duplicate = real_request("POST", "/v1/custom/operation", pending)
    assert duplicate == retry
    assert client.status()["step"] == before_step + 1
    replayed = client.save_checkpoint("replayed-first-step")["checkpoint"]
    assert replayed["parameter_sha256"] == first_step["parameter_sha256"]
    return {
        "model": client.status()["model"],
        "model_revision": client.status().get("model_revision"),
        "history": history,
        "initial_dpo_loss": history[0]["custom_loss"],
        "final_dpo_loss": final_loss.item(),
        "baseline_parameter_sha256": baseline["parameter_sha256"],
        "trained_parameter_sha256": trained["parameter_sha256"],
        "replayed_parameter_sha256": replayed["parameter_sha256"],
        "checks": [
            "sampling_policy_logprob_parity",
            "top_k_logprobs",
            "custom_dpo_improves",
            "nonzero_gradients",
            "adapter_weights_changed",
            "checkpoint_download",
            "checkpoint_restore_exact",
            "optimizer_restore_reproduces_first_update",
            "misaligned_mask_rejected_before_mutation",
            "lost_optimizer_response_retry_applies_once",
            "duplicate_receipt_is_identical",
        ],
        "events": client.logs()["events"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--gpu", default="A10G")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    with TrainingClient.create(
        args.model,
        gpu=args.gpu,
        max_steps=args.steps + 2,
        max_wall_seconds=1800,
        idle_seconds=300,
        max_context_tokens=256,
    ) as client:
        print(f"GPU session ready: {client.sandbox_id}", flush=True)
        report = exercise(client, steps=args.steps)
        report["gpu"] = args.gpu
        report["sandbox_id"] = client.sandbox_id
    report["wall_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in {"events", "history"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
