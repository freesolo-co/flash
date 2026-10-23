"""Operator entry point for one explicitly configured shared training pool."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from flash.training.shared.contracts import Limits, fields, integer


def build_app(config, credentials, root):
    from transformers import AutoConfig

    from flash.training.shared.http import create_app, token_authentication
    from flash.training.shared.journal import Journal
    from flash.training.shared.runtime import Runtime

    fields(
        config,
        ("model", "revision"),
        ("rank", "alpha", "targets", "limits", "device", "rate_microusd_per_hour"),
    )
    if not isinstance(config["model"], str) or not re.fullmatch(
        r"[\w.-]+/[\w.-]+", config["model"]
    ):
        raise ValueError("pool model must be a Hugging Face repository id")
    if not isinstance(config["revision"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", config["revision"]
    ):
        raise ValueError("pool requires an immutable model revision")
    rank = integer(config.get("rank", 8), "rank", 1, 128)
    alpha = integer(config.get("alpha", 16), "alpha", 1, 256)
    targets = config.get("targets", ["q_proj", "v_proj"])
    if (
        not isinstance(targets, list)
        or not targets
        or any(
            t not in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
            for t in targets
        )
    ):
        raise ValueError("unsupported LoRA target modules")
    if config.get("device", "cuda") not in ("cpu", "cuda"):
        raise ValueError("pool device must be cpu or cuda")
    authenticate = token_authentication(credentials)
    limits = Limits(**config.get("limits", {}))
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    identity = root / "model.json"
    rendered = json.dumps(config, sort_keys=True)
    try:
        with identity.open("x") as handle:
            handle.write(rendered)
    except FileExistsError:
        if identity.read_text() != rendered:
            raise ValueError(
                "pool directory belongs to a different model/runtime configuration"
            ) from None
    model_config = AutoConfig.from_pretrained(
        config["model"], revision=config["revision"], trust_remote_code=False
    )
    text_config = model_config.get_text_config()
    if limits.context_tokens > text_config.max_position_embeddings:
        raise ValueError("pool context limit exceeds model configuration")
    journal = Journal(
        root / "pool.sqlite",
        limits=limits,
        vocab_size=text_config.vocab_size,
        rate_microusd_per_hour=config.get("rate_microusd_per_hour"),
    )

    def load():
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

        from flash.training.shared.backend import PeftBackend

        device = config.get("device", "cuda")
        loader = (
            AutoModelForImageTextToText
            if getattr(model_config, "text_config", None) is not None
            else AutoModelForCausalLM
        )
        model = loader.from_pretrained(
            config["model"],
            revision=config["revision"],
            trust_remote_code=False,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        ).to(device)
        return PeftBackend(
            model,
            root=root / "checkpoints",
            rank=rank,
            alpha=alpha,
            targets=targets,
            base_identity=f"{config['model']}@{config['revision']}",
        )

    runtime = Runtime(journal, load, lock_path=root / "owner.lock")
    app = create_app(runtime, authenticate)
    app.state.shared_runtime = runtime
    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8090, type=int)
    args = parser.parse_args()
    credentials = json.loads(os.environ["FLASH_SHARED_CREDENTIALS"])
    app = build_app(json.loads(args.config.read_text()), credentials, args.state)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
