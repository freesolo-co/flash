"""Resident PEFT adapters with independent optimizers and training RNG state.

Only the pool's execution thread may call this object. No user Python, model code,
environment secrets, or filesystem paths enter this runtime through requests.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

from flash.training.shared.contracts import ConflictError
from flash.training.shared.losses import objective


class PeftBackend:
    def __init__(
        self, base, *, root, rank=8, alpha=16, targets=("q_proj", "v_proj"), base_identity="fixture"
    ):
        self.model = base
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.rank, self.alpha, self.targets = rank, alpha, tuple(targets)
        self.base_identity = base_identity
        self.adapters = {}
        self.wrapped = False
        self.device = next(base.parameters()).device
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("shared trainer supports CPU qualification or one CUDA device")
        for parameter in base.parameters():
            parameter.requires_grad_(False)

    @property
    def vocab_size(self):
        return self.model.config.get_text_config().vocab_size

    def synchronize(self):
        if self.device.type == "cuda":
            import torch

            torch.cuda.synchronize(self.device)

    @contextmanager
    def _random(self, seed=None, state=None):
        import torch

        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            if state is None:
                torch.manual_seed(seed)
            else:
                torch.set_rng_state(state["cpu"].cpu())
                if devices:
                    torch.cuda.set_rng_state(state["cuda"].cpu(), self.device)
            yield

    def _rng(self):
        import torch

        return {
            "cpu": torch.get_rng_state().clone(),
            "cuda": torch.cuda.get_rng_state(self.device).clone()
            if self.device.type == "cuda"
            else None,
        }

    def create(self, session, config):
        import torch
        from peft import LoraConfig, get_peft_model

        if session in self.adapters:
            raise ConflictError("adapter already exists")
        with self._random(seed=config["seed"]):
            lora = LoraConfig(
                r=self.rank,
                lora_alpha=self.alpha,
                target_modules=list(self.targets),
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            if self.wrapped:
                self.model.add_adapter(session, lora)
            else:
                self.model = get_peft_model(self.model, lora, adapter_name=session)
                self.wrapped = True
            self.model.set_adapter(session)
            parameters = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0.0)
            self.adapters[session] = {
                "config": config,
                "optimizer": optimizer,
                "version": 0,
                "rng": self._rng(),
            }
        return {"version": 0}

    def _select(self, session, version=None):
        state = self.adapters[session]
        if version is not None and state["version"] != version:
            raise ConflictError("requested policy version is not resident")
        self.model.set_adapter(session)
        self.model.zero_grad(set_to_none=True)
        return state

    def _logprobs(self, tokens):
        import torch

        data = torch.tensor(tokens, device=self.device, dtype=torch.long).unsqueeze(0)
        output = self.model(input_ids=data[:, :-1], use_cache=False)
        return output.logits[0].float().log_softmax(-1).gather(-1, data[0, 1:, None]).squeeze(-1)

    def train(self, session, payload):
        import torch

        state = self._select(session, payload["version"])
        with self._random(state=state["rng"]):
            self.model.train()
            logs = [self._logprobs(row["tokens"]) for row in payload["rows"]]
            loss = objective(
                logs,
                payload["rows"],
                state["config"]["algorithm"],
                clip=payload.get("clip", 0.2),
                normalizer=payload.get("normalizer", 4096),
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite loss; shared runtime must be replaced")
            loss.backward()
            parameters = [p for p in self.model.parameters() if p.requires_grad]
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            state["optimizer"].step()
            state["optimizer"].zero_grad(set_to_none=True)
            state["version"] += 1
            state["rng"] = self._rng()
        return {
            "version": state["version"],
            "loss": float(loss.detach()),
            "grad_norm": float(norm),
            "input_tokens": sum(len(row["tokens"]) - 1 for row in payload["rows"]),
            "loss_tokens": sum(
                sum(
                    selected and group >= 0
                    for selected, group in zip(
                        row["mask"], row.get("teacher_groups", [0] * len(row["mask"])), strict=True
                    )
                )
                for row in payload["rows"]
            ),
        }

    def sample(self, session, payload):
        import torch

        state = self._select(session, payload["version"])
        tokens = list(payload["prompt"])
        logprobs = []
        self.model.eval()
        with self._random(seed=payload["seed"]), torch.no_grad():
            # explicit full-distribution sampling keeps training and returned logprobs aligned.
            # no retained KV cache can survive an adapter switch.
            for _ in range(payload["max_tokens"]):
                data = torch.tensor([tokens], device=self.device, dtype=torch.long)
                logs = (
                    self.model(input_ids=data, use_cache=False)
                    .logits[0, -1]
                    .float()
                    .log_softmax(-1)
                )
                token = int(torch.multinomial(logs.exp(), 1))
                logprobs.append(float(logs[token]))
                tokens.append(token)
                eos = self.model.config.get_text_config().eos_token_id
                if token in (eos if isinstance(eos, list) else [eos]):
                    break
        return {
            "version": state["version"],
            "tokens": tokens[len(payload["prompt"]) :],
            "logprobs": logprobs,
            "input_tokens": len(payload["prompt"]),
            "output_tokens": len(logprobs),
        }

    def checkpoint(self, session, payload):
        import torch
        from peft import get_peft_model_state_dict

        state = self._select(session)
        path = self.root / session
        path.mkdir(exist_ok=True)
        target = path / f"{state['version']}.pt"
        temporary = target.with_suffix(".pending")
        content = {
            "adapter": get_peft_model_state_dict(self.model, adapter_name=session),
            "optimizer": state["optimizer"].state_dict(),
            "rng": state["rng"],
            "version": state["version"],
            "config": state["config"],
            "model_config": self.model.config.to_dict(),
            "rank": self.rank,
            "alpha": self.alpha,
            "targets": self.targets,
            "base_identity": self.base_identity,
        }
        with temporary.open("wb") as handle:
            torch.save(content, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return {
            "version": state["version"],
            "checkpoint": f"{session}/{state['version']}",
            "bytes": target.stat().st_size,
        }

    def restore(self, session, payload):
        import torch
        from peft import set_peft_model_state_dict

        state = self._select(session)
        if state["version"] != 0:
            raise ConflictError("restore requires a fresh adapter")
        checkpoint = payload["checkpoint"]
        content = torch.load(
            self.root / f"{checkpoint}.pt", map_location=self.device, weights_only=True
        )
        expected = (
            state["config"],
            self.rank,
            self.alpha,
            self.targets,
            json.dumps(self.model.config.to_dict(), sort_keys=True),
        )
        actual = (
            content["config"],
            content["rank"],
            content["alpha"],
            tuple(content["targets"]),
            json.dumps(content["model_config"], sort_keys=True),
        )
        if actual != expected or content.get("base_identity") != self.base_identity:
            raise ValueError("checkpoint runtime or adapter configuration differs")
        set_peft_model_state_dict(self.model, content["adapter"], adapter_name=session)
        state["optimizer"].load_state_dict(content["optimizer"])
        state["rng"], state["version"] = content["rng"], content["version"]
        return {"version": state["version"]}

    def close(self, session, payload):
        del self.adapters[session]
        self.model.delete_adapter(session)
        return {"closed": True}

    def execute(self, command):
        operation = command["operation"]
        return getattr(self, operation)(command["session"], command["payload"])
