"""Reference single-GPU LoRA engine for explicit custom-loop operations.

No server-side user code: the client supplies only tokens and dLoss/dLogprob weights.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from flash.custom.validation import validate_operation


class TorchBackend:
    def __init__(self, model, tokenizer, *, checkpoint_dir, max_tokens=2048, max_batch=16):
        import torch

        self.model = model
        self.tokenizer = tokenizer
        # eval disables dropout while retaining autograd: two-pass custom losses must see the same function.
        self.model.eval()
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        if not self.parameters:
            raise ValueError("model has no trainable parameters")
        self.optimizer = torch.optim.AdamW(self.parameters, lr=1e-4, weight_decay=0.0)
        self.pending_backward = 0
        self.algorithm_state = {}
        self.checkpoints = None
        self.vocab_size = model.get_output_embeddings().weight.shape[0]
        self.max_tokens = max_tokens
        self.max_batch = max_batch
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def info(self):
        return {
            "model": str(self.model.config._name_or_path),
            "model_revision": getattr(self.model.config, "_commit_hash", None),
            "vocab_size": self.vocab_size,
            "max_context_tokens": self.max_tokens,
            "max_batch": self.max_batch,
            "pending_backward": self.pending_backward,
        }

    def validate(self, operation, arguments):
        return validate_operation(self, operation, arguments)

    def execute(self, operation, arguments):
        return getattr(self, operation)(**arguments)

    def _tensor(self, tokens):
        import torch

        return torch.tensor([tokens], device=self.parameters[0].device, dtype=torch.long)

    def _logprobs(self, tokens, target_tokens=None):
        """Logprobs of the targets: the next token ([T]) or a [T, K] target matrix."""
        import torch

        ids = self._tensor(tokens)
        logits = self.model(input_ids=ids[:, :-1], use_cache=False).logits[0].float()
        if target_tokens is None:
            targets = ids[0, 1:, None]
        else:
            targets = torch.tensor(target_tokens, device=logits.device, dtype=torch.long)
        # gather targets and one normalizer per position; never copy a vocabulary row each.
        selected = logits.gather(-1, targets) - torch.logsumexp(logits, dim=-1, keepdim=True)
        return (selected.squeeze(-1) if target_tokens is None else selected), logits

    def forward(self, data, top_k=0):
        import torch

        outputs = []
        with torch.no_grad():
            for row in data:
                logprobs, logits = self._logprobs(row.tokens, row.target_tokens)
                output = {"logprobs": logprobs.cpu().tolist()}
                if top_k:
                    values, indices = logits.topk(top_k, dim=-1)
                    values -= torch.logsumexp(logits, dim=-1, keepdim=True)
                    output["top_logprobs"] = [
                        [{"token": t, "logprob": p} for t, p in zip(ids, probs, strict=True)]
                        for ids, probs in zip(indices.tolist(), values.tolist(), strict=True)
                    ]
                outputs.append(output)
        return {"outputs": outputs}

    def forward_backward(self, data, metrics=None):
        import torch

        loss_total = 0.0
        outputs = []
        for row in data:
            logprobs, _ = self._logprobs(row.tokens, row.target_tokens)
            weights = torch.tensor(row.weights, device=logprobs.device, dtype=logprobs.dtype)
            loss = -(weights * logprobs).sum()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss; session cannot continue")
            loss.backward()
            loss_total += loss.detach().item()
            outputs.append({"logprobs": logprobs.detach().cpu().tolist()})
        self.pending_backward += 1
        return {"outputs": outputs, "metrics": {**(metrics or {}), "surrogate_loss": loss_total}}

    def optim_step(self, learning_rate, max_grad_norm, algorithm_state=None):
        import torch

        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        norm = torch.nn.utils.clip_grad_norm_(
            self.parameters, max_grad_norm, error_if_nonfinite=True
        )
        self.optimizer.step()
        if algorithm_state is not None:
            self.algorithm_state = algorithm_state
        self.optimizer.zero_grad(set_to_none=True)
        accumulated = self.pending_backward
        self.pending_backward = 0
        return {
            "metrics": {
                "grad_norm": norm.item(),
                "learning_rate": learning_rate,
                "accumulated_backward_calls": accumulated,
            }
        }

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.pending_backward = 0
        return {}

    def sample(self, prompt, max_tokens, temperature, num_samples):
        import torch
        from transformers import GenerationConfig

        sequences = []
        with torch.no_grad():
            for _ in range(num_samples):
                config = GenerationConfig(
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature else 1.0,
                    top_k=0,
                    top_p=1.0,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                generated = self.model.generate(
                    input_ids=self._tensor(prompt),
                    generation_config=config,
                    attention_mask=torch.ones_like(self._tensor(prompt)),
                )
                tokens = generated.sequences[0].tolist()
                completion = tokens[len(prompt) :]
                policy, _ = self._logprobs(tokens)
                behavior = [
                    torch.log_softmax(scores[0].float(), dim=-1)[token].item()
                    if temperature
                    else 0.0
                    for scores, token in zip(generated.scores, completion, strict=True)
                ]
                sequences.append(
                    {
                        "tokens": completion,
                        "text": self.tokenizer.decode(completion, skip_special_tokens=False),
                        "logprobs": policy[len(prompt) - 1 :].tolist(),
                        "sampling_logprobs": behavior,
                        "finish_reason": "stop"
                        if completion[-1] == self.tokenizer.eos_token_id
                        else "length",
                    }
                )
        return {"sequences": sequences}

    def encode(self, text):
        return {"tokens": self.tokenizer.encode(text, add_special_tokens=False)}

    def decode(self, tokens):
        return {"text": self.tokenizer.decode(tokens, skip_special_tokens=False)}

    def save_checkpoint(self, name, metadata=None, algorithm_state=None):
        import torch

        # checkpoints contain only trusted engine state; no arbitrary client serialization is loaded.
        destination = self.checkpoint_dir / name
        stage = Path(tempfile.mkdtemp(prefix=".stage-", dir=self.checkpoint_dir))
        try:
            self.model.save_pretrained(stage / "adapter")
            self.tokenizer.save_pretrained(stage / "adapter")
            trainable = {
                n: p.detach().cpu() for n, p in self.model.named_parameters() if p.requires_grad
            }
            digest = hashlib.sha256()
            for key, value in sorted(trainable.items()):
                digest.update(key.encode())
                digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
            metadata = {
                **self.info(),
                **(metadata or {}),
                "parameter_sha256": digest.hexdigest(),
                "name": name,
            }
            torch.save(
                {
                    "parameters": trainable,
                    "optimizer": self.optimizer.state_dict(),
                    "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                },
                stage / "state.pt",
            )
            (stage / "algorithm.json").write_text(
                json.dumps(
                    self.algorithm_state if algorithm_state is None else algorithm_state,
                    allow_nan=False,
                )
            )
            (stage / "metadata.json").write_text(json.dumps(metadata))
            stage.rename(destination)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        publication = self.checkpoints.publish(destination, metadata) if self.checkpoints else {}
        return {"checkpoint": {**metadata, **publication}}

    def load_checkpoint(self, name):
        return self.restore_checkpoint(self.checkpoint_dir / name)

    def restore_checkpoint(self, source):
        import torch

        source = Path(source)
        metadata = json.loads((source / "metadata.json").read_text())
        if any(metadata.get(key) != self.info().get(key) for key in ("model", "model_revision")):
            raise ValueError("checkpoint base model or immutable revision differs")
        state = torch.load(source / "state.pt", map_location="cpu", weights_only=True)
        trainable = {key: p for key, p in self.model.named_parameters() if p.requires_grad}
        if set(trainable) != set(state["parameters"]) or any(
            p.shape != state["parameters"][key].shape for key, p in trainable.items()
        ):
            raise ValueError("checkpoint adapter topology differs")
        with torch.no_grad():
            for key, parameter in self.model.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(state["parameters"][key])
        self.optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        self.algorithm_state = (
            json.loads((source / "algorithm.json").read_text())
            if (source / "algorithm.json").exists()
            else {}
        )
        return {"checkpoint": metadata, "algorithm_state": self.algorithm_state}

    def close(self):
        self.zero_grad()
        return {}


def load_backend(
    model_id,
    *,
    revision=None,
    lora_rank=16,
    lora_alpha=None,
    seed=42,
    checkpoint_dir,
    max_tokens=2048,
):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    # pin tokenizer and weights to the same immutable revision, even if the author used a branch.
    revision = getattr(config, "_commit_hash", None) or revision
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    model_class = AutoModelForCausalLM
    targeting = {}
    if model_id == "Qwen/Qwen3.5-9B":
        from transformers import AutoModelForImageTextToText

        from flash.adapters.targets import resolve_lora_targeting

        model_class = AutoModelForImageTextToText
        surface = resolve_lora_targeting(model_id, algorithm="custom", multimodal=False)
        targeting = {
            "exclude_modules": surface.exclude_modules,
            "target_parameters": surface.target_parameters,
        }
    model = model_class.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha or 2 * lora_rank,
            lora_dropout=0,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            revision=revision,
            **targeting,
        ),
    )
    model.config._commit_hash = revision
    return TorchBackend(model, tokenizer, checkpoint_dir=checkpoint_dir, max_tokens=max_tokens)
