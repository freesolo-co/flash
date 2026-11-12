"""GPU-resident LoRA engine for explicit custom-loop operations.

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
    def __init__(
        self, model, tokenizer, *, checkpoint_dir, max_tokens=2048, max_batch=16, processor=None
    ):
        import torch

        self.model = model
        self.tokenizer = tokenizer
        # an image-capable model's processor; None keeps the session text-only.
        self.processor = processor
        # eval disables dropout while retaining autograd: two-pass custom losses must see the same function.
        self.model.eval()
        self.checkpointing = _enable_checkpointing(model)
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
            "images": self.processor is not None,
        }

    def validate(self, operation, arguments):
        return validate_operation(self, operation, arguments)

    def execute(self, operation, arguments):
        _ensure_cuda_headroom()
        return getattr(self, operation)(**arguments)

    def _tensor(self, tokens):
        import torch

        return torch.tensor(
            [tokens], device=self.model.get_input_embeddings().weight.device, dtype=torch.long
        )

    def decode_images(self, images):
        """Decode data URIs under the training image limits; validation calls this."""
        if self.processor is None:
            raise ValueError("this custom session's model does not accept images")
        from flash.content.image_descriptors import descriptor
        from flash.content.multimodal import MAX_IMAGE_DESCRIPTOR_BYTES, decode_image_descriptors

        descriptors = [
            descriptor("data_uri", uri, max_descriptor_bytes=MAX_IMAGE_DESCRIPTOR_BYTES)
            for uri in images
        ]
        return decode_image_descriptors(descriptors, None)

    def vision_inputs(self, tokens, images):
        """Processor image inputs for tokens, checked against their placeholder runs.

        Runs during validation, before any GPU work: the image-pad tokens in tokens[:-1] (the
        positions the model reads) must be exactly the runs the processor sizes these images
        to, or the model would silently misplace image features.
        """
        decoded = self.decode_images(images)
        try:
            out = self.processor.image_processor(images=decoded, return_tensors="pt")
        finally:
            for image in decoded:
                image.close()
        grid = out["image_grid_thw"]
        merge = int(getattr(self.processor.image_processor, "merge_size", 2))
        expected = int((grid.prod(-1) // (merge * merge)).sum())
        pad = self._image_pad_id()
        found = sum(1 for token in tokens[:-1] if token == pad)
        if found != expected or tokens[-1] == pad:
            raise ValueError(
                f"tokens hold {found} image placeholder tokens but these {len(images)} images "
                f"need {expected}; build image rows with encode(text, images=...)"
            )
        return {"pixel_values": out["pixel_values"], "image_grid_thw": grid}

    def _image_pad_id(self):
        from flash.content.multimodal import IMAGE_PAD_TOKEN

        return self.tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)

    def _vision_kwargs(self, ids, vision):
        """Model keyword inputs for one row's images; ids are the tokens the model reads."""
        if vision is None:
            return {}
        device = ids.device
        dtype = self.model.get_input_embeddings().weight.dtype
        return {
            "pixel_values": vision["pixel_values"].to(device=device, dtype=dtype),
            "image_grid_thw": vision["image_grid_thw"].to(device),
            # multimodal rope needs the image positions: 1 on image tokens, 0 on text.
            "mm_token_type_ids": (ids == self._image_pad_id()).int(),
        }

    def _logprobs(self, tokens, target_tokens=None, vision=None):
        """Logprobs of the targets: the next token ([T]) or a [T, K] target matrix."""
        import torch

        ids = self._tensor(tokens)
        inputs = ids[:, :-1]
        logits = (
            self.model(input_ids=inputs, use_cache=False, **self._vision_kwargs(inputs, vision))
            .logits[0]
            .float()
        )
        if target_tokens is None:
            targets = ids[0, 1:, None].to(logits.device)
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
                logprobs, logits = self._logprobs(row.tokens, row.target_tokens, row.vision)
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
        # transformers recomputes checkpointed layers only in train mode; the model has no
        # train-only behavior when checkpointing is enabled, so this is the eval function.
        self.model.train(self.checkpointing)
        try:
            for row in data:
                logprobs, _ = self._logprobs(row.tokens, row.target_tokens, row.vision)
                weights = torch.tensor(row.weights, device=logprobs.device, dtype=logprobs.dtype)
                loss = -(weights * logprobs).sum()
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite loss; session cannot continue")
                loss.backward()
                loss_total += loss.detach().item()
                outputs.append({"logprobs": logprobs.detach().cpu().tolist()})
        finally:
            self.model.eval()
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

    def sample(self, prompt, max_tokens, temperature, num_samples, vision=None):
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
                prompt_ids = self._tensor(prompt)
                generated = self.model.generate(
                    input_ids=prompt_ids,
                    generation_config=config,
                    attention_mask=torch.ones_like(prompt_ids),
                    **self._vision_kwargs(prompt_ids, vision),
                )
                tokens = generated.sequences[0].tolist()
                completion = tokens[len(prompt) :]
                policy, _ = self._logprobs(tokens, vision=vision)
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

    def encode(self, text, images=None):
        if images is None:
            return {"tokens": self.tokenizer.encode(text, add_special_tokens=False)}
        # the processor expands each <|image_pad|> into the run these images need.
        try:
            out = self.processor(text=[text], images=images, add_special_tokens=False)
        finally:
            for image in images:
                image.close()
        ids = out["input_ids"][0]
        return {"tokens": [int(t) for t in (ids.tolist() if hasattr(ids, "tolist") else ids)]}

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
            from flash.custom.model_parallel import restore_cuda_rng

            restore_cuda_rng(state["cuda_rng"])
        self.algorithm_state = (
            json.loads((source / "algorithm.json").read_text())
            if (source / "algorithm.json").exists()
            else {}
        )
        return {"checkpoint": metadata, "algorithm_state": self.algorithm_state}

    def close(self):
        self.zero_grad()
        return {}


# cuBLAS handle/workspace creation and Triton kernels allocate outside PyTorch's caching
# allocator, so they need memory the driver can still hand out.
CUDA_HEADROOM_BYTES = 6 * 1024**3


def _ensure_cuda_headroom(min_free_bytes=CUDA_HEADROOM_BYTES):
    """Return PyTorch's unused cached blocks to the driver when free GPU memory runs low.

    A session's operations see many sequence lengths, so cached blocks rarely fit the next
    request and the caching allocator keeps reserving: a resumed Qwen3.5-9B DPO session held
    ~78 GiB reserved with only ~19 GiB allocated. The first backward then needed memory outside
    that cache (cuBLAS handle creation, Triton kernels) and failed with CUBLAS_STATUS_ALLOC_FAILED
    / "Triton Error [CUDA]: out of memory" at <1 GiB free. Releasing the cache before an operation
    restores that headroom; it changes no tensors. Best effort, a no-op without CUDA.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        if all(
            torch.cuda.mem_get_info(device)[0] >= min_free_bytes
            for device in range(torch.cuda.device_count())
        ):
            return False
        torch.cuda.empty_cache()
        return True
    except Exception:
        return False


def _enable_checkpointing(model):
    """Checkpoint catalog models without active dropout or stateful normalization.

    Without checkpointing a 4,096-token row keeps every layer's activations, which exceeds
    one GPU for larger models. transformers only recomputes in train mode. Absence of
    nn.Dropout is not sufficient: Qwen attention uses functional dropout. The supported
    catalog architectures have no other train-only text behavior.
    """
    import torch

    if not getattr(model, "supports_gradient_checkpointing", False):
        return False
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    if getattr(text_config, "attention_dropout", None) != 0:
        return False
    for module in model.modules():
        # Inspect the live attention setting too: it is copied from config on construction.
        if getattr(module, "attention_dropout", 0) != 0:
            return False
        if isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.p > 0:
            return False
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            return False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return True


def load_backend(
    model_id,
    *,
    revision=None,
    lora_rank=16,
    lora_alpha=None,
    seed=42,
    checkpoint_dir,
    max_tokens=2048,
    init_adapter=None,
    gpu_count=1,
):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    # pin tokenizer and weights to the same immutable revision, even if the author used a branch.
    revision = getattr(config, "_commit_hash", None) or revision
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    from flash.core.catalog import supports_image_training

    model_class = AutoModelForCausalLM
    targeting = {}
    processor = None
    # Image-trained checkpoints keep the multimodal module layout their adapters target;
    # custom training adapts only the cataloged language modules of that layout.
    if supports_image_training(model_id):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from flash.adapters.targets import resolve_lora_targeting

        model_class = AutoModelForImageTextToText
        # the vision tower stays frozen (language-module adapters only), but image rows run
        # through it, so the session can accept images; same pinned revision as the weights.
        processor = AutoProcessor.from_pretrained(
            model_id, revision=revision, trust_remote_code=False
        )
        surface = resolve_lora_targeting(model_id, algorithm="custom", multimodal=False)
        targeting = {
            "exclude_modules": surface.exclude_modules,
            "target_parameters": surface.target_parameters,
        }
    if init_adapter and _uses_language_model_layout(init_adapter):
        # SFT and GRPO adapt Qwen3.5 text modules under model.language_model even for text-only
        # runs; load that same layout so the source's module paths and exclusions resolve.
        from transformers import AutoModelForImageTextToText

        model_class = AutoModelForImageTextToText
    from flash.custom.model_parallel import load_placement, validate_placement

    placement = load_placement(model_id, lora_rank, max_tokens, gpu_count)
    model = model_class.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        **placement,
    )
    if gpu_count == 1:
        model = model.to("cuda")
    if init_adapter:
        model = _warm_start(model, init_adapter, lora_rank, lora_alpha)
    else:
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
    validate_placement(model, gpu_count)
    model.config._commit_hash = revision
    return TorchBackend(
        model,
        tokenizer,
        checkpoint_dir=checkpoint_dir,
        max_tokens=max_tokens,
        processor=processor,
    )


def _uses_language_model_layout(adapter_dir):
    """Whether a source adapter's tensors address the multimodal class's language_model."""
    from safetensors import safe_open

    with safe_open(str(Path(adapter_dir) / "adapter_model.safetensors"), "np") as weights:
        names = weights.keys()  # a safetensors handle, not a dict
    return any(".language_model." in name for name in names)


def _warm_start(model, adapter_dir, lora_rank, lora_alpha):
    """Continue a verified source LoRA: its exact modules and weights, with fresh optimizer state.

    The source config keeps its target modules; dropout is disabled like every custom adapter so
    gradient checkpointing stays exact. Every source tensor must land on a trainable parameter:
    peft only warns about unmatched keys, which would silently restart from zero.
    """
    from peft import LoraConfig, get_peft_model, load_peft_weights, set_peft_model_state_dict

    config = LoraConfig.from_pretrained(adapter_dir)
    if (config.r, config.lora_alpha) != (lora_rank, lora_alpha or 2 * lora_rank):
        raise ValueError("warm-start adapter rank or alpha differs from the prepared run")
    config.lora_dropout = 0
    config.inference_mode = False
    model = get_peft_model(model, config)
    weights = load_peft_weights(adapter_dir, device="cuda")
    result = set_peft_model_state_dict(model, weights)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if result.unexpected_keys or len(weights) != len(trainable):
        raise ValueError(
            "warm-start adapter weights do not match the model's LoRA parameters "
            f"({len(weights)} source tensors, {len(trainable)} trainable, "
            f"{len(result.unexpected_keys)} unmatched)"
        )
    return model
