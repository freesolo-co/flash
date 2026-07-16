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
    # Per-token grammar-forced mask (parallel to token_ids): True where guided decoding left exactly
    # one legal token, so the student had no real choice. Empty when logprobs were unavailable.
    forced: tuple[bool, ...] = ()

    @property
    def terminated(self) -> bool:
        """True when vLLM stopped on EOS or a configured stop string, not the max-token cap."""
        reason = (self.finish_reason or "").lower()
        return bool(self.stop_reason is not None or reason in {"stop", "eos"})


def opd_lora_rank(model, default: int = 32) -> int:
    """Best-effort maximum PEFT LoRA rank for vLLM's max_lora_rank."""
    cfgs = getattr(model, "peft_config", None) or {}
    cfg_iter = cfgs.values() if isinstance(cfgs, dict) else (cfgs,)
    ranks: list[int] = []
    for cfg in cfg_iter:
        rank = getattr(cfg, "r", None)
        if isinstance(rank, int) and not isinstance(rank, bool) and rank > 0:
            ranks.append(rank)
        elif isinstance(rank, dict):
            # Some PEFT configs express per-module ranks as a dict-valued `r`; take the max.
            ranks.extend(
                int(value)
                for value in rank.values()
                if isinstance(value, int) and not isinstance(value, bool) and value > 0
            )
        pattern = getattr(cfg, "rank_pattern", None)
        if isinstance(pattern, dict):
            ranks.extend(
                int(value)
                for value in pattern.values()
                if isinstance(value, int) and not isinstance(value, bool) and value > 0
            )
    if ranks:
        return max(ranks)
    try:
        return max(1, int(default))
    except (TypeError, ValueError):
        return 32


def _startup_oom_error(
    *,
    free_gb: float,
    total_gb: float,
    requested_util: float,
    margin_gb: float,
    rollout_batch_size: int,
) -> BaseException:
    requested_gb = float(requested_util) * float(total_gb)
    msg = (
        "Free memory on device cuda:0 "
        f"({free_gb:.2f}/{total_gb:.2f} GiB) on startup is less than desired GPU "
        f"memory utilization ({requested_util:.6f} -> {requested_gb:.2f} GiB) "
        f"with {margin_gb:.2f} GiB startup margin after reducing OPD rollout_batch_size "
        f"to {rollout_batch_size}. Retry on a larger GPU."
    )
    try:
        import torch

        return torch.cuda.OutOfMemoryError(msg)
    except Exception:
        return RuntimeError(msg)


def _decode_only_compilation_config() -> dict[str, Any]:
    return {
        "mode": 0,  # CompilationMode.NONE: no torch.compile/AOT.
        "cudagraph_mode": "FULL_DECODE_ONLY",
    }


def opd_vllm_kwargs(model_id: str, knobs: Any, seq_cap: int) -> dict[str, Any]:
    """Direct vLLM LLM(...) kwargs mirroring the GRPO colocate rollout tuning."""
    kwargs: dict[str, Any] = {
        "gpu_memory_utilization": 0.10,
        "kv_cache_dtype": None,
        "max_num_seqs": None,
        "max_num_batched_tokens": None,
        "rollout_batch_size": None,
        "attention_backend": None,
        "mm_encoder_attn_backend": None,
        "enforce_eager": None,
        "compilation_config": None,
    }
    free_gb = 0.0
    startup_oom: BaseException | None = None
    try:
        import torch

        cc = torch.cuda.get_device_capability()
        card_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024**3)
            if total_bytes:
                card_gb = total_bytes / (1024**3)
        except Exception:
            pass
    except Exception:
        cc, card_gb = (0, 0), 0.0

    fp8_kv = bool(cc >= (8, 9))
    kwargs["kv_cache_dtype"] = "fp8" if fp8_kv else None
    if card_gb >= 140:
        kwargs["max_num_batched_tokens"] = max(8192, int(seq_cap))

    if card_gb > 0:
        try:
            from flash.catalog import MODELS
            from flash.engine.vram import (
                colocate_kv_util,
                opd_rollout_concurrency,
                resolve_params_b,
            )

            info = MODELS.get(model_id)
            params_b = resolve_params_b(model_id)
            active_b = float(getattr(info, "active_params_b", 0.0) or 0.0) if info else 0.0
            target_concurrency = opd_rollout_concurrency(
                getattr(knobs, "prompts_per_step", 1),
                getattr(knobs, "group_size", 1),
            )
            rollout_concurrency = target_concurrency

            def _util_for(num_generations: int) -> float:
                return colocate_kv_util(
                    params_b,
                    int(seq_cap),
                    card_gb,
                    False,
                    num_generations=num_generations,
                    active_params_b=active_b,
                    fp8_kv=fp8_kv,
                )

            util = _util_for(rollout_concurrency)
            if free_gb > 0:
                # vLLM rejects startup unless its whole executor budget is free right now. OPD builds
                # the rollout engine after the student + warm-start adapter are resident, so choose the
                # largest rollout chunk that fits the observed residual memory instead of reserving for
                # every prompt in the optimizer step.
                startup_margin_gb = max(1.0, min(4.0, card_gb * 0.03))
                max_startup_util = max(0.0, (free_gb - startup_margin_gb) / max(1.0, card_gb))
                while rollout_concurrency > 1 and util > max_startup_util:
                    rollout_concurrency -= 1
                    util = _util_for(rollout_concurrency)
                if rollout_concurrency < target_concurrency:
                    print(
                        "[opd] reduced vLLM rollout batch "
                        f"{target_concurrency}->{rollout_concurrency} to fit startup memory "
                        f"(free={free_gb:.1f} GiB, request={util * card_gb:.1f} GiB, "
                        f"margin={startup_margin_gb:.1f} GiB)"
                    )
                if util > max_startup_util:
                    startup_oom = _startup_oom_error(
                        free_gb=free_gb,
                        total_gb=card_gb,
                        requested_util=util,
                        margin_gb=startup_margin_gb,
                        rollout_batch_size=rollout_concurrency,
                    )
            kwargs["gpu_memory_utilization"] = util
            kwargs["max_num_seqs"] = rollout_concurrency
            kwargs["rollout_batch_size"] = rollout_concurrency
        except Exception as exc:
            print(f"[opd] vLLM memory-util sizing failed; using 0.10: {exc}")
    if card_gb < 140:
        from flash.catalog import opd_mamba_batched_token_floor

        kwargs["max_num_batched_tokens"] = opd_mamba_batched_token_floor(
            model_id, seq_cap, kwargs["max_num_seqs"]
        )
    if startup_oom is not None:
        raise startup_oom

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

    # Blackwell OPD startup is more reliable with the V1 engine core in-process: the old B200 failure
    # reached a resident SyncMPClient child process and then died/hung before the first rollout. This
    # does NOT disable CUDA graphs; it only avoids the fragile parent/child EngineCore startup seam.
    blackwell_inproc_v1 = cc and cc[0] in (10, 12)
    if blackwell_inproc_v1:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        print(
            f"[opd][warn] Blackwell/sm{cc[0]}{cc[1]}: VLLM_ENABLE_V1_MULTIPROCESSING=0 "
            "for OPD rollout (avoid V1 SyncMPClient EngineCore startup stall)"
        )

    try:
        import vllm as _vllm_mod

        ver_base = _vllm_mod.__version__.split("+")[0]
        vllm_ver = tuple(int(x) for x in ver_base.split(".")[:3])
        # vLLM's default 0.19.x graph path malfunctioned on Blackwell OPD startup, but full eager mode
        # throws away the decode CUDA graph speed path. Use a narrower graph profile instead: disable
        # torch.compile/AOT while keeping decode-only CUDA graph capture. That avoids the fragile
        # compile/slot-mapping path but still exercises CUDA graphs on B200/RTX 5090.
        if vllm_ver >= (0, 19, 0) and blackwell_inproc_v1:
            kwargs["enforce_eager"] = False
            kwargs["compilation_config"] = _decode_only_compilation_config()
            print(
                f"[opd] cc={cc[0]}.{cc[1]}: using decode-only vLLM CUDA graphs for OPD rollout "
                "(torch.compile disabled, V1 EngineCore in-process)"
            )
        elif vllm_ver >= (0, 19, 0) and cc not in {(8, 0), (9, 0)}:
            kwargs["enforce_eager"] = True
            print(
                f"[opd][warn] enforce_eager=True on the vLLM rollout (cc={cc[0]}.{cc[1]} -> "
                "prevent 0.19.x aot_compile/slot-mapping crash on this unvalidated GPU family)"
            )
        elif vllm_ver >= (0, 19, 0):
            print(f"[opd] cc={cc[0]}.{cc[1]}: keeping vLLM CUDA graphs for OPD rollout speed")
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
    model_revision: str = field(default="", kw_only=True)
    stop_sequences: tuple[str, ...] = ()
    # the exact model/tokenizer halt set used for generation and termination classification.
    eos_token_ids: tuple[int, ...] = ()
    # StructuredOutputsParams kwargs (parsed [train] structured_outputs); None = unconstrained.
    structured_outputs: dict[str, Any] | None = None
    # vLLM EngineArgs.reasoning_parser (e.g. "deepseek_r1") when thinking + a constraint are both on:
    # defers the guided grammar until </think> so the student reasons freely first. None = off.
    reasoning_parser: str | None = None
    lora_rank: int = 32
    gpu_memory_utilization: float = 0.10
    kv_cache_dtype: str | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    rollout_batch_size: int | None = None
    attention_backend: str | None = None
    mm_encoder_attn_backend: str | None = None
    enforce_eager: bool | None = None
    compilation_config: dict[str, Any] | None = None
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
        # Import fails loudly at engine build (not first generate) when a constraint is configured
        # against a vLLM without structured-outputs support — never silently roll unconstrained.
        self._StructuredOutputsParams = None
        if self.structured_outputs:
            from vllm.sampling_params import StructuredOutputsParams

            self._StructuredOutputsParams = StructuredOutputsParams
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
        if self.model_revision:
            kwargs["revision"] = self.model_revision
        if self.reasoning_parser:
            # Gate the structured-outputs grammar on </think>: with a constraint AND thinking on, vLLM
            # otherwise binds the schema from token 0 and forbids the <think> reasoning phase. Only set
            # alongside a constraint (the caller's reasoning_parser_for guarantees this).
            kwargs["reasoning_parser"] = self.reasoning_parser
        if self.kv_cache_dtype:
            kwargs["kv_cache_dtype"] = self.kv_cache_dtype
        if self.max_num_seqs:
            kwargs["max_num_seqs"] = int(self.max_num_seqs)
        if self.max_num_batched_tokens:
            kwargs["max_num_batched_tokens"] = int(self.max_num_batched_tokens)
        if self.attention_backend:
            kwargs["attention_backend"] = self.attention_backend
        if self.mm_encoder_attn_backend:
            kwargs["mm_encoder_attn_backend"] = self.mm_encoder_attn_backend
        if self.enforce_eager is not None:
            kwargs["enforce_eager"] = bool(self.enforce_eager)
        if self.compilation_config:
            kwargs["compilation_config"] = dict(self.compilation_config)
        if self.seed is not None:
            kwargs["seed"] = int(self.seed)
        try:
            self.llm = LLM(**kwargs)
        except RuntimeError as exc:
            startup_oom = self._enginecore_startup_oom(exc)
            if startup_oom is not None:
                raise startup_oom from exc
            raise
        if self.adapter_root is None:
            self.adapter_root = tempfile.mkdtemp(prefix="flash_opd_vllm_lora_")
        else:
            os.makedirs(self.adapter_root, exist_ok=True)

    def _enginecore_startup_oom(self, exc: RuntimeError) -> BaseException | None:
        """Recast vLLM's parent EngineCore wrapper as OOM when the hidden root is memory preflight."""
        msg = str(exc).lower()
        if (
            "enginecore initialization failed" not in msg
            and "engine core initialization failed" not in msg
        ):
            return None
        try:
            import torch

            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024**3)
            total_gb = total_bytes / (1024**3)
            requested_gb = float(self.gpu_memory_utilization) * total_gb
            if free_gb < requested_gb:
                return torch.cuda.OutOfMemoryError(
                    "vLLM EngineCore startup OOM: "
                    f"free={free_gb:.2f} GiB < requested={requested_gb:.2f} GiB "
                    f"(gpu_memory_utilization={float(self.gpu_memory_utilization):.3f})"
                )
        except Exception:
            return None
        return None

    @property
    def sync_count(self) -> int:
        return self._version

    def sync_from_model(self, model) -> None:
        """Save the current PEFT adapter and make future generations use that exact version."""
        old_lora_id = self._lora_int_id
        old_adapter_dir = self._sync_dirs[-1] if self._sync_dirs else None
        self._version += 1
        adapter_dir = os.path.join(self.adapter_root, f"adapter-{self._version:06d}")
        os.makedirs(adapter_dir, exist_ok=True)
        model.save_pretrained(adapter_dir)
        self._sync_dirs.append(adapter_dir)
        self._lora_int_id = self._version
        self._lora_request = self._LoRARequest(
            f"opd-step-{self._version}", self._lora_int_id, adapter_dir
        )
        if old_adapter_dir and old_lora_id is not None and self._remove_lora(old_lora_id):
            shutil.rmtree(old_adapter_dir, ignore_errors=True)
            self._sync_dirs = [d for d in self._sync_dirs if d != old_adapter_dir]

    def _remove_lora(self, lora_id: int) -> bool:
        """Best-effort dynamic-LoRA cache cleanup across vLLM API variants."""
        for obj in (
            getattr(self, "llm", None),
            getattr(getattr(self, "llm", None), "llm_engine", None),
        ):
            remover = getattr(obj, "remove_lora", None)
            if callable(remover):
                try:
                    remover(lora_id)
                    return True
                except Exception as exc:
                    print(f"[opd] vLLM remove_lora({lora_id}) failed; continuing: {exc}")
                    return False
        return False

    def _sampling_params(self, max_tokens: int, *, seed: int | None = None):
        kwargs = {
            "max_tokens": max(1, int(max_tokens)),
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "stop": list(self.stop_sequences) if self.stop_sequences else None,
        }
        if seed is not None:
            kwargs["seed"] = int(seed)
        if self.eos_token_ids:
            kwargs["stop_token_ids"] = list(self.eos_token_ids)
        # Keep stop strings in the returned text when supported so OPD can trim ids/text in one place,
        # matching the shared OPD stop-trimming path. Older vLLM builds ignore unsupported kwargs by
        # raising.
        if self.stop_sequences:
            kwargs["include_stop_str_in_output"] = True
        if self._StructuredOutputsParams is not None:
            kwargs["structured_outputs"] = self._StructuredOutputsParams(**self.structured_outputs)
            # Request 2 logprobs so a grammar-forced position (only one legal token) is detectable:
            # its top-2 dict carries a single finite logprob, the other padded with -inf. Those
            # positions are masked out of the OPD reverse-KL, where the unconstrained teacher would
            # otherwise inject spurious signal the student had no choice about (_forced_from_logprobs).
            kwargs["logprobs"] = 2
        try:
            return self._SamplingParams(**kwargs)
        except TypeError:
            # Retry only for the cosmetic stop-echo kwarg. structured_outputs stays in the retry:
            # a configured constraint the sampler can't apply must fail the run, not silently
            # train on unconstrained text the reward believes is schema-bound.
            if "include_stop_str_in_output" not in kwargs:
                raise
            kwargs.pop("include_stop_str_in_output", None)
            return self._SamplingParams(**kwargs)

    def generate(
        self,
        prompt_ids_batch: list[list[int]],
        *,
        max_tokens: int,
        request_seeds: list[int] | None = None,
    ) -> list[OpdVllmOutput]:
        if not prompt_ids_batch:
            return []
        if request_seeds is not None and len(request_seeds) != len(prompt_ids_batch):
            raise ValueError("opd rollout request seed count must match prompt count")
        if self._lora_request is None:
            raise RuntimeError("opd vLLM rollout used before sync_from_model()")
        limit = max(1, int(self.rollout_batch_size or len(prompt_ids_batch)))
        out: list[OpdVllmOutput] = []
        for start in range(0, len(prompt_ids_batch), limit):
            prompts = [
                {"prompt_token_ids": [int(t) for t in ids]}
                for ids in prompt_ids_batch[start : start + limit]
            ]
            seeds = request_seeds[start : start + limit] if request_seeds is not None else None
            # vLLM's structured-output processor stamps per-request backend state onto the
            # StructuredOutputsParams instance, so a single shared constrained params corrupts
            # every sequence after the first (same reason multiturn_rollout builds one per
            # request). Hand vLLM a fresh params per prompt when constrained; unconstrained runs
            # have no per-request state and share one cheaply.
            sampling_params = (
                [
                    self._sampling_params(
                        max_tokens, seed=seeds[index] if seeds is not None else None
                    )
                    for index in range(len(prompts))
                ]
                if self._StructuredOutputsParams is not None or seeds is not None
                else self._sampling_params(max_tokens)
            )
            outputs = self.llm.generate(
                prompts,
                sampling_params=sampling_params,
                lora_request=self._lora_request,
                use_tqdm=False,
            )
            out.extend(_normalize_output(item) for item in outputs)
        return out

    def close(self) -> None:
        shutdown = getattr(getattr(self, "llm", None), "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception as exc:
                print(f"[opd] vLLM shutdown failed; continuing: {exc}")
        if self.adapter_root:
            shutil.rmtree(self.adapter_root, ignore_errors=True)


def _forced_from_logprobs(lps, n_tokens: int) -> tuple[bool, ...]:
    """Per-token grammar-forced mask derived from vLLM logprobs.

    A guided-decoding position is *forced* when exactly one token was grammatically legal: the
    backend sets every other logit to -inf, so the single legal token gets logprob 0.0. With
    ``logprobs>=2`` requested, vLLM's top-k is ``torch.topk``-based and returns a FIXED-size dict,
    padding the surplus slot(s) with -inf entries -- so dict *length* does not distinguish forced
    from free. Counting the finite (non -inf) logprobs does: exactly one finite entry == forced.
    A row with ZERO finite entries (an empty/all -inf row -- a wiring anomaly, never a real forced
    position, whose chosen token always carries a finite logprob) is treated as free, so a genuine
    free choice is never silently dropped from the loss. Returns () when logprobs are unavailable
    (unconstrained rollouts request none) -> the OPD loss runs unmasked, exactly as before.
    """
    if lps is None:
        return ()
    forced: list[bool] = []
    for i in range(n_tokens):
        # vLLM emits one logprob row per generated token, in order. If it ever returns fewer rows
        # than tokens (a wiring anomaly -- logprobs>=2 is always requested when constrained), mask
        # the prefix we can see and leave the unverifiable tail UNMASKED, rather than dropping the
        # whole sample's mask and silently re-admitting the forced-position teacher signal.
        if i >= len(lps):
            forced.append(False)
            continue
        legal = sum(
            1
            for lp in lps[i].values()
            if (val := getattr(lp, "logprob", lp)) is not None and val > float("-inf")
        )
        # Exactly one finite entry == grammar-forced. Zero finite entries is a wiring anomaly
        # (empty/all -inf row), NOT proof of forcing, so treat it as free -- otherwise a genuine
        # free choice gets silently dropped from the loss (parity with the missing-row branch).
        forced.append(legal == 1)
    return tuple(forced)


def _normalize_output(out) -> OpdVllmOutput:
    comp = out.outputs[0]
    token_ids = [int(t) for t in getattr(comp, "token_ids", ())]
    return OpdVllmOutput(
        token_ids=token_ids,
        text=str(getattr(comp, "text", "") or ""),
        finish_reason=getattr(comp, "finish_reason", None),
        stop_reason=getattr(comp, "stop_reason", None),
        forced=_forced_from_logprobs(getattr(comp, "logprobs", None), len(token_ids)),
    )
