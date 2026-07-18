"""Optional Chalk selected-token log-probabilities for TRL GRPO."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChalkGrpoRuntime:
    selective_log_softmax: Callable[..., Any]


def _import_selective_log_softmax() -> Callable[..., Any]:
    module = importlib.import_module("chalk.ops.grpo")
    return module.selective_log_softmax


def probe_chalk_grpo(
    *,
    importer: Callable[[], Callable[..., Any]] = _import_selective_log_softmax,
    torch_module=None,
) -> ChalkGrpoRuntime | None:
    """Return a validated Chalk runtime, or None for the unchanged TRL fallback."""
    try:
        torch = torch_module or importlib.import_module("torch")
        selective_log_softmax = importer()
        use_cuda = bool(torch.cuda.is_available())
        device = torch.device("cuda") if use_cuda else torch.device("cpu")
        dtype = torch.bfloat16 if use_cuda else torch.float32
        devices = [torch.cuda.current_device()] if use_cuda else []
        with torch.random.fork_rng(devices=devices):
            hidden = torch.randn(2, 3, 8, device=device, dtype=dtype, requires_grad=True)
            weight = torch.randn(17, 8, device=device, dtype=dtype, requires_grad=True)
            targets = torch.tensor([[1, 5, 9], [2, 7, 16]], device=device, dtype=torch.long)
            actual = selective_log_softmax(hidden, weight, targets, temperature=0.7)
            expected = torch.log_softmax((hidden @ weight.t()).float() / 0.7, dim=-1)
            expected = expected.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            tolerance = 2e-2 if use_cuda else 1e-5
            if actual.shape != targets.shape or not torch.isfinite(actual).all():
                return None
            if not torch.allclose(actual.float(), expected, rtol=tolerance, atol=tolerance):
                return None
            actual.sum().backward()
            if hidden.grad is None or weight.grad is None:
                return None
            if not torch.isfinite(hidden.grad).all() or not torch.isfinite(weight.grad).all():
                return None
            if use_cuda:
                torch.cuda.synchronize()
        return ChalkGrpoRuntime(selective_log_softmax=selective_log_softmax)
    except Exception:
        return None


def _chunked_entropy(
    hidden,
    weight,
    bias,
    temperature: float,
    *,
    seq_chunk_size: int = 2048,
    vocab_chunk_size: int = 32768,
):
    """Compute exact categorical entropy without allocating a token-by-vocabulary tensor."""
    import torch

    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    entropies = torch.empty(flat_hidden.shape[0], device=hidden.device, dtype=torch.float32)
    inv_temperature = 1.0 / float(temperature)
    with torch.no_grad():
        for seq_start in range(0, flat_hidden.shape[0], seq_chunk_size):
            seq_end = min(seq_start + seq_chunk_size, flat_hidden.shape[0])
            hidden_chunk = flat_hidden[seq_start:seq_end]
            rows = hidden_chunk.shape[0]
            running_max = torch.full(
                (rows,), float("-inf"), device=hidden.device, dtype=torch.float32
            )
            running_sum = torch.zeros_like(running_max)
            running_weighted_logits = torch.zeros_like(running_max)
            for vocab_start in range(0, weight.shape[0], vocab_chunk_size):
                vocab_end = min(vocab_start + vocab_chunk_size, weight.shape[0])
                logits = (hidden_chunk @ weight[vocab_start:vocab_end].to(hidden.dtype).t()).float()
                if bias is not None:
                    logits = logits + bias[vocab_start:vocab_end].float()
                logits = logits * inv_temperature
                next_max = torch.maximum(running_max, logits.amax(dim=-1))
                old_scale = torch.exp(running_max - next_max)
                chunk_exp = torch.exp(logits - next_max.unsqueeze(-1))
                running_sum = running_sum * old_scale + chunk_exp.sum(dim=-1)
                running_weighted_logits = (
                    running_weighted_logits * old_scale + (chunk_exp * logits).sum(dim=-1)
                )
                running_max = next_max
            log_z = running_max + torch.log(running_sum)
            entropies[seq_start:seq_end] = log_z - running_weighted_logits / running_sum
    return entropies.reshape(hidden.shape[:-1])


def _core_model(model):
    if getattr(model, "peft_config", None) is not None:
        return model.base_model.model
    return model


def _accepted_model_inputs(module, inputs: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(module.forward).parameters.values()
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters)
    names = {p.name for p in parameters}
    if accepts_kwargs:
        return inputs
    return {name: value for name, value in inputs.items() if name in names}


def _output_projection(model):
    head = model.get_output_embeddings()
    if head is None or not hasattr(head, "weight"):
        raise RuntimeError("Chalk GRPO needs a tensor-backed output projection")
    if getattr(head, "disable_adapters", False) is False and getattr(head, "active_adapters", []):
        raise RuntimeError("Chalk GRPO does not bypass an adapter-updated output projection")
    return head.weight, getattr(head, "bias", None)


def _maybe_gather_projection(trainer, weight, bias):
    plugin = getattr(getattr(trainer.accelerator, "state", None), "deepspeed_plugin", None)
    if plugin is None or getattr(plugin, "zero_stage", None) != 3:
        return nullcontext()
    try:
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

        params = [weight] if bias is None else [weight, bias]
        if all(p.ds_status == ZeroParamStatus.AVAILABLE for p in params):
            return nullcontext()
        import deepspeed

        return deepspeed.zero.GatheredParameters(params, modifier_rank=None)
    except (AttributeError, ImportError) as exc:
        raise RuntimeError("unable to gather the output projection for Chalk GRPO") from exc


def make_chalk_grpo_trainer(base_trainer, runtime: ChalkGrpoRuntime):
    """Build a narrow trainer subclass that changes only final-projection log-probabilities."""
    from trl.models.utils import _ForwardRedirection

    class ChalkGRPOTrainer(base_trainer):
        _flash_chalk_runtime = runtime

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._flash_forward_redirection = _ForwardRedirection()
            self._flash_chalk_failed = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if return_outputs:
                raise ValueError("The GRPOTrainer does not support returning outputs")
            if self._flash_chalk_failed:
                return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
            unwrapped_model = self.accelerator.unwrap_model(model)
            return self._flash_forward_redirection(
                model, unwrapped_model, self._compute_loss, unwrapped_model, inputs
            )

        def _get_per_token_logps_and_entropies(
            self,
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            batch_size=None,
            compute_entropy=False,
            pixel_values=None,
            image_grid_thw=None,
            num_images=None,
            pixel_attention_mask=None,
            image_sizes=None,
            token_type_ids=None,
            mm_token_type_ids=None,
            image_position_ids=None,
        ):
            if self._flash_chalk_failed:
                return super()._get_per_token_logps_and_entropies(
                    model,
                    input_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    compute_entropy,
                    pixel_values,
                    image_grid_thw,
                    num_images,
                    pixel_attention_mask,
                    image_sizes,
                    token_type_ids,
                    mm_token_type_ids,
                    image_position_ids,
                )
            try:
                return self._flash_selected_logps_and_entropies(
                    model,
                    input_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    compute_entropy,
                    pixel_values,
                    image_grid_thw,
                    num_images,
                    pixel_attention_mask,
                    image_sizes,
                    token_type_ids,
                    mm_token_type_ids,
                    image_position_ids,
                )
            except Exception as exc:
                self._flash_chalk_failed = True
                print(
                    f"[rl][warn] Chalk GRPO selected-token path failed; using TRL logits fallback: {exc!r}",
                    flush=True,
                )
                return super()._get_per_token_logps_and_entropies(
                    model,
                    input_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    compute_entropy,
                    pixel_values,
                    image_grid_thw,
                    num_images,
                    pixel_attention_mask,
                    image_sizes,
                    token_type_ids,
                    mm_token_type_ids,
                    image_position_ids,
                )

        def _flash_selected_logps_and_entropies(
            self,
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            batch_size,
            compute_entropy,
            pixel_values,
            image_grid_thw,
            num_images,
            pixel_attention_mask,
            image_sizes,
            token_type_ids,
            mm_token_type_ids,
            image_position_ids,
        ):
            import torch

            batch_size = batch_size or input_ids.size(0)
            all_logps = []
            all_entropies = []
            unwrapped_model = self.accelerator.unwrap_model(model)
            core_model = _core_model(unwrapped_model)
            backbone = core_model.base_model
            weight, bias = _output_projection(core_model)
            for start in range(0, input_ids.size(0), batch_size):
                end = min(start + batch_size, input_ids.size(0))
                model_inputs = {
                    "input_ids": input_ids[start:end],
                    "attention_mask": attention_mask[start:end],
                    "use_cache": False,
                }
                if pixel_values is not None:
                    if image_grid_thw is not None and num_images is not None:
                        rows_per_image = image_grid_thw.prod(dim=-1)
                        rows_per_sample = torch.split(rows_per_image, num_images)
                        rows_per_sample = torch.stack([rows.sum() for rows in rows_per_sample])
                        cumulative_rows = torch.cat(
                            [torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)]
                        )
                        row_start = int(cumulative_rows[start].item())
                        row_end = int(cumulative_rows[end].item())
                        image_counts = torch.tensor([0, *list(num_images)]).cumsum(0)
                        image_start = int(image_counts[start].item())
                        image_end = int(image_counts[end].item())
                        model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                        model_inputs["image_grid_thw"] = image_grid_thw[image_start:image_end]
                    elif image_position_ids is not None and num_images is not None:
                        image_counts = torch.tensor([0, *list(num_images)]).cumsum(0)
                        image_start = int(image_counts[start].item())
                        image_end = int(image_counts[end].item())
                        model_inputs["pixel_values"] = pixel_values[image_start:image_end]
                        model_inputs["image_position_ids"] = image_position_ids[image_start:image_end]
                    else:
                        model_inputs["pixel_values"] = pixel_values[start:end]
                for name, value in (
                    ("pixel_attention_mask", pixel_attention_mask),
                    ("image_sizes", image_sizes),
                    ("token_type_ids", token_type_ids),
                    ("mm_token_type_ids", mm_token_type_ids),
                ):
                    if value is not None:
                        model_inputs[name] = value[start:end]
                hidden = backbone(**_accepted_model_inputs(backbone, model_inputs)).last_hidden_state
                hidden = hidden[:, :-1, :][:, -logits_to_keep:, :]
                targets = input_ids[start:end, -logits_to_keep:]
                with _maybe_gather_projection(self, weight, bias):
                    logps = self._flash_chalk_runtime.selective_log_softmax(
                        hidden,
                        weight,
                        targets,
                        bias=bias,
                        temperature=self.temperature,
                    )
                    all_logps.append(logps)
                    if compute_entropy:
                        all_entropies.append(
                            _chunked_entropy(hidden, weight, bias, self.temperature)
                        )
            logps = torch.cat(all_logps, dim=0)
            entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
            return logps, entropies

    ChalkGRPOTrainer.__name__ = "ChalkGRPOTrainer"
    ChalkGRPOTrainer.__qualname__ = "ChalkGRPOTrainer"
    return ChalkGRPOTrainer


def resolve_chalk_grpo_trainer(base_trainer):
    runtime = probe_chalk_grpo()
    if runtime is None:
        return base_trainer, False
    return make_chalk_grpo_trainer(base_trainer, runtime), True
