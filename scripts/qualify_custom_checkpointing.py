"""Paid H100 regression for custom target snapshots and checkpointed gradients.

python scripts/qualify_custom_checkpointing.py --run-paid-test --output evidence.json

Uses the qualified worker image plus this checkout's source. The isolated sandbox is
terminated on exit; no deployment or hosted adapter is changed.
"""

import argparse
import hashlib
import json
from pathlib import Path

MARKER = "CUSTOM_CHECKPOINTING_EVIDENCE="


class InProcessHTTP:
    def __init__(self, session):
        self.session = session

    def _request(self, method, path, body=None, **kwargs):
        return self.session.status() if method == "GET" else self.session.execute(body)


def check_snapshot_gradients(backend, tokens, targets):
    import torch
    from transformers.modeling_layers import GradientCheckpointingLayer

    from flash.custom import Datum, TrainingClient
    from flash.custom.session import TrainingSession

    selected, _ = backend._logprobs(tokens, targets)
    expected_loss = -selected.square().mean()
    expected_loss.backward()
    expected = [p.grad.detach().float().cpu().clone() for p in backend.parameters]
    expected_scores = selected.detach().cpu()
    expected_value = expected_loss.item()
    del selected, expected_loss
    backend.zero_grad()

    # Use the real client/session protocol, with an in-process transport to isolate GPU math.
    from unittest.mock import patch

    session = TrainingSession(backend)
    calls = []
    layer = next(m for m in backend.model.modules() if isinstance(m, GradientCheckpointingLayer))
    hook = layer.register_forward_pre_hook(lambda *args: calls.append(True))
    try:
        with patch("flash.custom.client.ApiClient", return_value=InProcessHTTP(session)):
            client = TrainingClient("http://localhost", api_key="qualification")
            data = [Datum(list(tokens), target_tokens=[list(row) for row in targets])]

            def loss_fn(rows, logprobs):
                rows[0].tokens[1] = 0
                rows[0].target_tokens[0][0] = 0
                rows[0].target_tokens[1] = [0] * len(targets[1])
                return -logprobs[0].square().mean(), {}

            result = client.forward_backward_custom(data, loss_fn)
    finally:
        hook.remove()
    actual = [p.grad.detach().float().cpu() for p in backend.parameters]
    squared_error = sum(
        (a - b).square().sum().item() for a, b in zip(actual, expected, strict=True)
    )
    squared_norm = sum(b.square().sum().item() for b in expected)
    relative_error = (squared_error / max(squared_norm, 1e-30)) ** 0.5
    score_error = (
        (torch.tensor(result["outputs"][0]["logprobs"]) - expected_scores).abs().max().item()
    )
    assert abs(result["metrics"]["custom_loss"] - expected_value) < 1e-5
    assert score_error < 1e-5, score_error
    assert relative_error < 0.01, relative_error  # BF16 recomputation tolerance
    assert len(calls) >= 3, "expected scoring, backward forward, and checkpoint recomputation"
    assert not any(m.training for m in backend.model.modules())
    backend.zero_grad()
    return {
        "gradient_relative_l2_error": relative_error,
        "logprob_max_error": score_error,
        "first_layer_forward_calls": len(calls),
        "mutated_callback_targets": True,
    }


def check_context_ceiling(backend, tokens):
    import torch

    from flash.custom import Datum

    tokens = (tokens * (4096 // len(tokens) + 1))[:4096]
    targets = [list(range(20)) for _ in range(4095)]
    weights = [[1 / (4095 * 20)] * 20 for _ in range(4095)]
    row = Datum(tokens, weights, targets)
    torch.cuda.reset_peak_memory_stats()
    scored = backend.forward([row])["outputs"][0]["logprobs"]
    trained = backend.forward_backward([row])["outputs"][0]["logprobs"]
    error = (torch.tensor(scored) - torch.tensor(trained)).abs().max().item()
    assert error < 1e-5, error
    step = backend.optim_step(learning_rate=1e-5, max_grad_norm=1.0)
    return {
        "tokens": 4096,
        "target_shape": [4095, 20],
        "logprob_max_error": error,
        "optimizer_grad_norm": step["metrics"]["grad_norm"],
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def check_functional_dropout(backend, tokens, targets):
    from flash.custom import Datum
    from flash.custom.backend import TorchBackend

    modules = [m for m in backend.model.modules() if hasattr(m, "attention_dropout")]
    assert modules
    for module in modules:
        module.attention_dropout = 0.5
    config = backend.model.config
    getattr(config, "text_config", config).attention_dropout = 0.5
    guarded = TorchBackend(backend.model, backend.tokenizer, checkpoint_dir="/tmp/dropout-check")
    assert not guarded.checkpointing
    row = Datum(tokens, [[1 / len(tokens)] * len(ids) for ids in targets], targets)
    scored = guarded.forward([row])
    trained = guarded.forward_backward([row])
    assert scored["outputs"] == trained["outputs"]
    assert not any(m.training for m in guarded.model.modules())
    guarded.close()
    return {
        "attention_dropout": 0.5,
        "attention_modules": len(modules),
        "checkpointing": False,
        "logprobs_equal": True,
        "remained_in_eval": True,
    }


def qualify():
    import torch
    import transformers

    from flash.core.catalog import get_model
    from flash.custom import Datum
    from flash.custom.backend import load_backend

    model_id = "Qwen/Qwen3.5-0.8B"
    backend = load_backend(
        model_id,
        revision=get_model(model_id).managed_revision,
        checkpoint_dir="/tmp/custom-checkpoints",
        max_tokens=4096,
    )
    assert backend.checkpointing
    tokens = backend.tokenizer.encode("Explain why the sky is blue in simple terms. " * 8)[:65]
    top = backend.forward([Datum(tokens)], top_k=3)["outputs"][0]["top_logprobs"]
    targets = [[item["token"] for item in row] for row in top]
    evidence = {
        "model": model_id,
        "revision": backend.info()["model_revision"],
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    evidence["snapshot_and_checkpointed_gradients"] = check_snapshot_gradients(
        backend, tokens, targets
    )
    evidence["context_ceiling"] = check_context_ceiling(backend, tokens)
    evidence["functional_dropout_fallback"] = check_functional_dropout(backend, tokens, targets)
    backend.close()
    print(MARKER + json.dumps(evidence), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--run-paid-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        qualify()
        return
    if not args.run_paid_test or args.output is None:
        parser.error("pass --run-paid-test and --output to provision an isolated H100")
    import modal

    from flash.providers._lifecycle.net.worker import worker_image_for_gpu

    root = Path(__file__).resolve().parents[1]
    image_ref = worker_image_for_gpu("H100")
    image = (
        modal.Image.from_registry(image_ref)
        .add_local_dir(root / "flash", "/source/flash", copy=True, ignore=["**/__pycache__/**"])
        .add_local_file(__file__, "/source/qualify.py", copy=True)
    )
    evidence = None
    with modal.enable_output(), modal.App("flash-custom-checkpointing-qualification").run() as app:
        sandbox = modal.Sandbox.create(
            "python",
            "-u",
            "/source/qualify.py",
            "--worker",
            app=app,
            image=image,
            gpu="H100",
            cpu=8,
            memory=32768,
            timeout=1200,
            workdir="/source",
            env={"PYTHONPATH": "/source", "HF_HUB_DISABLE_PROGRESS_BARS": "1"},
        )
        print(f"Sandbox: {sandbox.object_id}", flush=True)
        try:
            for line in sandbox.stdout:
                print(line, end="", flush=True)
                if line.startswith(MARKER):
                    evidence = json.loads(line[len(MARKER) :])
            sandbox.wait()
            print(sandbox.stderr.read(), flush=True)
            if sandbox.returncode != 0 or evidence is None:
                raise RuntimeError(f"GPU qualification failed: exit {sandbox.returncode}")
        finally:
            sandbox.terminate()
    evidence["worker_image"] = image_ref
    evidence["source_sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "flash/custom/backend.py",
            "flash/custom/client.py",
            "scripts/qualify_custom_checkpointing.py",
        )
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
