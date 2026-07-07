"""Colocated vLLM rollout helper for OPD student generation.

OPD owns its teacher scoring and GKD loss loop, so it cannot reuse TRL's GRPOTrainer wrapper directly.
This module keeps the vLLM surface small: build one resident LLM, save the current PEFT adapter to a
versioned temp dir after each optimizer step, and generate with a matching LoRARequest.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OpdVllmOutput:
    """One vLLM completion, normalized to the fields OPD needs."""

    token_ids: list[int]
    text: str
    finish_reason: str | None = None
    stop_reason: object = None

    @property
    def terminated(self) -> bool:
        """True when vLLM stopped on EOS or a configured stop string, not the max-token cap."""
        reason = (self.finish_reason or "").lower()
        return bool(self.stop_reason is not None or reason in {"stop", "eos"})


def opd_lora_rank(model, default: int = 32) -> int:
    """Best-effort PEFT LoRA rank for vLLM's max_lora_rank."""
    cfgs = getattr(model, "peft_config", None) or {}
    cfg_iter = cfgs.values() if isinstance(cfgs, dict) else (cfgs,)
    for cfg in cfg_iter:
        rank = getattr(cfg, "r", None)
        if isinstance(rank, dict):
            vals = [int(v) for v in rank.values() if isinstance(v, int) and v > 0]
            if vals:
                return max(vals)
        if isinstance(rank, int) and rank > 0:
            return rank
    try:
        return max(1, int(default))
    except (TypeError, ValueError):
        return 32


def opd_vllm_kwargs(model_id: str, knobs: Any, seq_cap: int) -> dict[str, Any]:
    """Direct vLLM LLM(...) kwargs mirroring the GRPO colocate rollout tuning."""
    kwargs: dict[str, Any] = {
        "gpu_memory_utilization": 0.10,
        "kv_cache_dtype": None,
        "max_num_batched_tokens": None,
        "attention_backend": None,
        "mm_encoder_attn_backend": None,
        "enforce_eager": None,
    }
    try:
        import torch

        cc = torch.cuda.get_device_capability()
        card_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        cc, card_gb = (0, 0), 0.0

    fp8_kv = bool(cc >= (8, 9))
    kwargs["kv_cache_dtype"] = "fp8" if fp8_kv else None
    kwargs["max_num_batched_tokens"] = max(8192, int(seq_cap)) if card_gb >= 140 else None

    if card_gb > 0:
        try:
            from flash.catalog import MODELS
            from flash.engine.vram import colocate_kv_util, resolve_params_b

            info = MODELS.get(model_id)
            active_b = float(getattr(info, "active_params_b", 0.0) or 0.0) if info else 0.0
            kwargs["gpu_memory_utilization"] = colocate_kv_util(
                resolve_params_b(model_id),
                int(seq_cap),
                card_gb,
                False,
                num_generations=max(1, int(knobs.group_size)),
                active_params_b=active_b,
                fp8_kv=fp8_kv,
            )
        except Exception as exc:
            print(f"[opd] vLLM memory-util sizing failed; using 0.10: {exc}")

    from flash.engine.worker.gpu_setup import (
        force_vit_sdpa_on_blackwell,
        force_vllm_backend_for_sm120,
    )

    attention_backend = force_vllm_backend_for_sm120()
    if attention_backend:
        kwargs["attention_backend"] = attention_backend
    if cc and cc[0] in (10, 12):
        force_vit_sdpa_on_blackwell()
        kwargs["mm_encoder_attn_backend"] = "TORCH_SDPA"

    try:
        import vllm as _vllm_mod

        ver_base = _vllm_mod.__version__.split("+")[0]
        vllm_ver = tuple(int(x) for x in ver_base.split(".")[:3])
        if vllm_ver > (0, 19, 0) and cc != (10, 0):
            kwargs["enforce_eager"] = True
            print(
                f"[opd][warn] enforce_eager=True on the vLLM rollout (cc={cc[0]}.{cc[1]} -> "
                "prevent 0.19.1 aot_compile/slot-mapping crash; B200/sm100 keeps CUDA graphs)"
            )
    except Exception:
        pass
    return kwargs


@dataclass
class OpdVllmRolloutEngine:
    """Resident vLLM engine plus a versioned OPD LoRA adapter request."""

    model_source: str
    max_model_len: int
    temperature: float
    top_p: float
    stop_sequences: tuple[str, ...] = ()
    lora_rank: int = 32
    gpu_memory_utilization: float = 0.10
    kv_cache_dtype: str | None = None
    max_num_batched_tokens: int | None = None
    attention_backend: str | None = None
    mm_encoder_attn_backend: str | None = None
    enforce_eager: bool | None = None
    seed: int | None = None
    adapter_root: str | None = None
    _version: int = 0
    _lora_int_id: int | None = None
    _lora_request: object | None = None
    _sync_dirs: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        self._SamplingParams = SamplingParams
        self._LoRARequest = LoRARequest
        kwargs: dict[str, Any] = {
            "model": self.model_source,
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "max_model_len": max(1, int(self.max_model_len)),
            "gpu_memory_utilization": float(self.gpu_memory_utilization),
            "enable_lora": True,
            "max_lora_rank": max(1, int(self.lora_rank)),
            # The next adapter is loaded before the previous one is dropped on some vLLM paths.
            "max_loras": 2,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
        }
        if self.kv_cache_dtype:
            kwargs["kv_cache_dtype"] = self.kv_cache_dtype
        if self.max_num_batched_tokens:
            kwargs["max_num_batched_tokens"] = int(self.max_num_batched_tokens)
        if self.attention_backend:
            kwargs["attention_backend"] = self.attention_backend
        if self.mm_encoder_attn_backend:
            kwargs["mm_encoder_attn_backend"] = self.mm_encoder_attn_backend
        if self.enforce_eager is not None:
            kwargs["enforce_eager"] = bool(self.enforce_eager)
        if self.seed is not None:
            kwargs["seed"] = int(self.seed)
        self.llm = LLM(**kwargs)
        if self.adapter_root is None:
            self.adapter_root = tempfile.mkdtemp(prefix="flash_opd_vllm_lora_")
        else:
            os.makedirs(self.adapter_root, exist_ok=True)

    @property
    def sync_count(self) -> int:
        return self._version

    def sync_from_model(self, model) -> None:
        """Save the current PEFT adapter and make future generations use that exact version."""
        old_lora_id = self._lora_int_id
        self._version += 1
        adapter_dir = os.path.join(self.adapter_root, f"adapter-{self._version:06d}")
        os.makedirs(adapter_dir, exist_ok=True)
        model.save_pretrained(adapter_dir)
        self._sync_dirs.append(adapter_dir)
        self._lora_int_id = self._version
        self._lora_request = self._LoRARequest(
            f"opd-step-{self._version}", self._lora_int_id, adapter_dir
        )
        if old_lora_id is not None:
            self._remove_lora(old_lora_id)

    def _remove_lora(self, lora_id: int) -> None:
        """Best-effort dynamic-LoRA cache cleanup across vLLM API variants."""
        for obj in (getattr(self, "llm", None), getattr(getattr(self, "llm", None), "llm_engine", None)):
            remover = getattr(obj, "remove_lora", None)
            if callable(remover):
                try:
                    remover(lora_id)
                    return
                except Exception as exc:
                    print(f"[opd] vLLM remove_lora({lora_id}) failed; continuing: {exc}")
                    return

    def _sampling_params(self, max_tokens: int):
        kwargs = {
            "max_tokens": max(1, int(max_tokens)),
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "stop": list(self.stop_sequences) if self.stop_sequences else None,
        }
        # Keep stop strings in the returned text when supported so OPD can trim ids/text in one place,
        # matching the shared OPD stop-trimming path. Older vLLM builds ignore unsupported kwargs by
        # raising.
        if self.stop_sequences:
            kwargs["include_stop_str_in_output"] = True
        try:
            return self._SamplingParams(**kwargs)
        except TypeError:
            kwargs.pop("include_stop_str_in_output", None)
            return self._SamplingParams(**kwargs)

    def generate(self, prompt_ids_batch: list[list[int]], *, max_tokens: int) -> list[OpdVllmOutput]:
        if not prompt_ids_batch:
            return []
        if self._lora_request is None:
            raise RuntimeError("opd vLLM rollout used before sync_from_model()")
        prompts = [{"prompt_token_ids": [int(t) for t in ids]} for ids in prompt_ids_batch]
        outputs = self.llm.generate(
            prompts,
            sampling_params=self._sampling_params(max_tokens),
            lora_request=self._lora_request,
            use_tqdm=False,
        )
        return [_normalize_output(out) for out in outputs]

    def generate_one(self, prompt_ids: list[int], *, max_tokens: int) -> OpdVllmOutput:
        return self.generate([prompt_ids], max_tokens=max_tokens)[0]

    def close(self) -> None:
        shutdown = getattr(getattr(self, "llm", None), "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception as exc:
                print(f"[opd] vLLM shutdown failed; continuing: {exc}")
        if self.adapter_root:
            shutil.rmtree(self.adapter_root, ignore_errors=True)


def _normalize_output(out) -> OpdVllmOutput:
    comp = out.outputs[0]
    return OpdVllmOutput(
        token_ids=[int(t) for t in getattr(comp, "token_ids", ())],
        text=str(getattr(comp, "text", "") or ""),
        finish_reason=getattr(comp, "finish_reason", None),
        stop_reason=getattr(comp, "stop_reason", None),
    )
