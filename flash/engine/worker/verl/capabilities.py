"""Ask the verl child what it can do, and turn the answers into launch decisions.

One subprocess pays the torch/verl import once and answers every independent capability question
at the same time; everything below either runs that probe or reads its result to decide how the
child gets launched -- which interpreter, which attention backend, whether the rollout can sleep,
whether GDN boundary resets are available.

Split out of `flash.engine.worker.train.entry.backend_common` to keep that module under the file-size limit.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
from collections.abc import Sequence

from flash.engine.support.verl_policy import FsdpGeneration

# ---------------------------------------------------------------------------
# batched child-interpreter capability probe
# answer independent questions in one child to pay one torch/verl import. child probes may import only
# packages installed in that venv; parent-only hub reads are interpolated. each answer fails closed
# independently and flushes before the live CUDA GDN smoke, which must run last.
# ---------------------------------------------------------------------------
_CAPABILITY_PROBE = """
import json


def emit(key, value):
    print("FLASH_VERL_CAPS=" + json.dumps({key: value}), flush=True)


try:
    from verl.workers.config.rollout import RolloutConfig

    emit("rollout_fields", sorted(RolloutConfig.__dataclass_fields__))
except Exception:
    emit("rollout_fields", None)

try:
    import torch

    emit(
        "capability",
        list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    )
except Exception:
    emit("capability", None)

try:
    import flashinfer  # noqa: F401

    emit("flashinfer", True)
except Exception:
    emit("flashinfer", False)

try:
    import wandb  # noqa: F401

    emit("wandb", True)
except Exception:
    emit("wandb", False)

# gdn is asked only when the parent resolved a model_type to ask about (a gdn hybrid). the module
# name is a parent-resolved literal: it comes from the checkpoint config, which needs a hub/cache
# read the child must not repeat.
if %(gdn_module)r:
    try:
        import importlib
        import inspect

        from transformers.utils.import_utils import (
            is_causal_conv1d_available,
            is_flash_linear_attention_available,
        )

        # the reason the LAST failing check produced, so a clean negative can name it. the two
        # package flags cannot express the source check: it fails with both installed, and an
        # operator reading "fla=True causal_conv1d=True" alongside "install fla + causal_conv1d"
        # is being sent to fix something that is not broken.
        why = ""
        ok = is_flash_linear_attention_available() and is_causal_conv1d_available()
        if not ok:
            why = (
                "fla="
                + str(is_flash_linear_attention_available())
                + " causal_conv1d="
                + str(is_causal_conv1d_available())
                + " (install the missing package into the verl interpreter)"
            )
        if ok:
            import causal_conv1d  # noqa: F401  -- fail a built-but-broken ABI here, not at model load

            mod = importlib.import_module(%(gdn_module)r)
            gdn = next(
                (
                    c
                    for n, c in vars(mod).items()
                    if isinstance(c, type) and n.endswith("GatedDeltaNet")
                ),
                None,
            )
            # read THROUGH decorators. transformers wraps this forward in
            # @force_accelerate_hooks("conv1d"), which returns a plain closure and sets no
            # __wrapped__, so both getsource() and inspect.unwrap() return the hook wrapper --
            # ~1.1k chars of accelerate plumbing that mentions neither name. That reads as "this
            # transformers build cannot reset gdn state", which is a FALSE NEGATIVE: every released
            # 5.x reaches the values as kwargs.get("cu_seq_lens_q") / kwargs.get("seq_idx") in the
            # real body, so no version bump can fix it and the raise sends operators to reinstall
            # packages that are already present. The undecorated function is reachable through the
            # wrapper's closure cells; walk them until the source answers the question.
            src = ""
            if gdn is not None:
                fn = gdn.forward
                seen = set()
                while True:
                    src = inspect.getsource(fn)
                    if ("cu_seq_lens_q" in src) and ("seq_idx" in src):
                        break
                    nxt = None
                    for cell in getattr(fn, "__closure__", None) or ():
                        try:
                            val = cell.cell_contents
                        except ValueError:
                            continue
                        if inspect.isfunction(val) and id(val) not in seen:
                            seen.add(id(val))
                            nxt = val
                            break
                    if nxt is None:
                        break
                    fn = nxt
            ok = ("cu_seq_lens_q" in src) and ("seq_idx" in src)
            if not ok:
                # both packages are present, so this is the transformers version, not the env.
                why = (
                    "fla and causal_conv1d are both installed, but "
                    + (
                        "no *GatedDeltaNet class in " + %(gdn_module)r
                        if gdn is None
                        else (
                            "its forward() takes neither cu_seq_lens_q nor seq_idx"
                            if ("cu_seq_lens_q" not in src) and ("seq_idx" not in src)
                            else "its forward() is missing "
                            + ("cu_seq_lens_q" if "cu_seq_lens_q" not in src else "seq_idx")
                        )
                    )
                    + " -- this transformers build cannot reset gdn state at packed boundaries; "
                    + "installing packages will not fix it"
                )
        if ok:
            import torch

            # a causal_conv1d compiled without this arch imports fine and raises at the first
            # forward, so smoke it now: the answer decides whether we pack, and being wrong means
            # training contaminated.
            if torch.cuda.is_available():
                from causal_conv1d import causal_conv1d_fn

                causal_conv1d_fn(
                    torch.zeros(1, 4, 8, device="cuda", dtype=torch.bfloat16),
                    torch.zeros(4, 3, device="cuda", dtype=torch.bfloat16),
                )
                torch.cuda.synchronize()
        if not ok:
            # say WHY on a clean negative too: which of the three checks failed is the difference
            # between "install fla" and "rebuild causal_conv1d for this arch".
            print(
                "[verl] gdn boundary resets unavailable in the child: " + why,
                flush=True,
            )
        emit("gdn_boundary_resets", bool(ok))
    except Exception as e:
        # a bare False here is what made this invisible. the caller raises on a hybrid that cannot
        # reset, and an operator reading only that has no way back to which check fell over --
        # a missing wheel, a broken ABI, or a conv kernel compiled without this arch all look
        # identical from the parent. printed to STDOUT, which probe_verl_capabilities forwards
        # on the '[verl] ' prefix -- stderr would be dropped, it is only reported when the
        # child answers nothing at all.
        import traceback

        print(
            "[verl] gdn boundary resets unavailable in the child: "
            + " | ".join(traceback.format_exc(limit=3).strip().splitlines()),
            flush=True,
        )
        emit("gdn_boundary_resets", False)
"""

# every answer's fail-closed default, used when the child dies, times out, or prints nothing
# readable. `None` means "could not answer" for the two callers that must distinguish that from a
# negative answer: rollout_fields (unknown verl -> do not claim a fork field is missing) and
# capability (unknown card -> leave the library default alone rather than guessing a workaround).
_CAPABILITIES_UNAVAILABLE: dict = {
    "rollout_fields": None,
    "capability": None,
    "flashinfer": False,
    "wandb": False,
    "gdn_boundary_resets": None,
}

# the batched timeout is the sum of the former 300 + 600 + 120 + 120 + 120 second budgets, so the
# final GDN smoke retains its cold-import allowance. reaching the sum indicates a wedged child.
_CAPABILITY_PROBE_TIMEOUT_S = 300 + 600 + 120 + 120 + 120


def probe_verl_capabilities(python_bin: str, gdn_module: str = "") -> dict:
    """ask the VERL CHILD every independent capability question in ONE subprocess.

    ``gdn_module`` is parent-resolved; ``""`` skips the live CUDA GDN smoke. every key has a fail-closed
    default, and incrementally flushed answers survive a later child timeout.
    """
    probe = _CAPABILITY_PROBE % {"gdn_module": gdn_module}
    caps = dict(_CAPABILITIES_UNAVAILABLE)
    timed_out = False
    try:
        done = subprocess.run(
            [python_bin, "-c", probe],
            capture_output=True,
            text=True,
            timeout=_CAPABILITY_PROBE_TIMEOUT_S,
        )
        stdout, stderr, returncode = done.stdout, done.stderr, done.returncode
    except subprocess.TimeoutExpired as e:
        # the killed child still flushed every answer it reached, and TimeoutExpired carries that
        # output. a slow gdn smoke must not retract the cheap answers that already landed -- an
        # empty rollout_fields aborts grpo, which is a worse outcome than training padded.
        timed_out = True
        stdout = (
            e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        )
        stderr = (
            e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        )
        returncode = None
        print(
            f"[verl] capability probe timed out after {_CAPABILITY_PROBE_TIMEOUT_S}s; "
            "keeping the answers it flushed, failing the rest closed"
        )
    except (OSError, subprocess.SubprocessError) as e:
        print(f"[verl] capability probe failed ({e}); treating every capability as unavailable")
        return caps
    answered_any = False
    for line in (stdout or "").splitlines():
        if not line.startswith("FLASH_VERL_CAPS="):
            continue
        with contextlib.suppress(Exception):
            answered = json.loads(line.split("=", 1)[1])
            if isinstance(answered, dict):
                # merge rather than replace: a child from a future/older flash that answers a key
                # this build does not know must not add it, and one it never reached must keep its
                # fail-closed default rather than vanishing from the dict.
                caps.update({k: v for k, v in answered.items() if k in caps})
                answered_any = True
    # the child's own diagnostics, forwarded BEFORE the early return. the gdn question prints why it
    # answered no, and it is answered LAST -- so every real run answers something earlier, and a
    # bare `return caps` here would discard the one line the gate's raise tells operators to read.
    # matched on the prefix rather than forwarded wholesale: the child's stdout also carries verl's
    # own import chatter, which is not worth a parent log line.
    for line in (stdout or "").splitlines():
        if line.startswith("[verl] "):
            print(line, flush=True)
    if answered_any:
        return caps
    if not timed_out:
        print(
            "[verl] capability probe returned no readable answer "
            f"(exit {returncode}); treating every capability as unavailable"
        )
    if stderr:
        print(f"[verl] capability probe stderr: {stderr.strip()[:800]}")
    return caps


def verl_declares_rollout_field(caps: dict, field: str) -> bool:
    """report whether the probed verl declares `field` on RolloutConfig.

    the fork adds rollout fields stock verl does not have. hydra composes an unknown key happily and
    only fails later in omega_conf_to_dataclass, so callers ask here before emitting a fork-only
    override rather than letting the run abort at dataclass conversion. an unanswerable probe reads
    as "not declared", which is what the subprocess version returned when the child could not be
    reached: the caller raises with the cause instead of composing a key that would abort in
    dataclass conversion on a paid gpu.
    """
    fields = caps.get("rollout_fields")
    return bool(fields) and field in fields


def gdn_probe_module(model_id: str, revision: str = "") -> str:
    """the ``transformers.models.*`` modeling module the gdn question must inspect.

    resolved by the PARENT and handed to the child as a literal, because it comes from the
    checkpoint config -- a hub/cache read the child should not repeat. pair every call with
    ``gdn_reset_arch_from_caps`` below so the shim is rendered for the SAME arch the child cleared.
    """
    from flash.engine.worker.model.packing import gdn_model_type

    model_type = gdn_model_type(model_id, revision=revision)
    return f"transformers.models.{model_type}.modeling_{model_type}"


def strict_gdn_probe_module(model_id: str, revision: str = "") -> str:
    """``gdn_probe_module`` that RAISES rather than guessing the arch, for a packed run.

    ``gdn_model_type`` answers ``qwen3_5`` both when the config says so and when the config could
    not be read at all. Those are the same string and a different meaning. For a packed run the
    difference is silent corruption: ``Qwen/Qwen3.6-35B-A3B`` is ``qwen3_5_moe``, so the fallback
    names the DENSE module -- the child clears it, the shim patches it, the MoE layers it was
    supposed to protect stay unpatched, and packed examples bleed state while the log reports
    ``gdn packed-boundary resets active``. A failed read must stop the run instead.
    """
    from transformers import AutoConfig

    from flash.engine.worker.io.hf import model_revision_kwargs

    try:
        cfg = AutoConfig.from_pretrained(
            model_id, trust_remote_code=True, **model_revision_kwargs(revision)
        )
        model_type = getattr(cfg, "model_type", None)
    except Exception as e:
        raise RuntimeError(
            f"packed gdn run could not read the model config for {model_id!r}, so the modeling "
            "module to patch is unknown. refusing to guess: the fallback arch would patch a "
            "different module than the model uses and train across packed example boundaries "
            "while reporting resets as active."
        ) from e
    if not model_type:
        raise RuntimeError(
            f"packed gdn run read the config for {model_id!r} but it declares no model_type, so "
            "the modeling module to patch is unknown. refusing to guess."
        )
    return f"transformers.models.{model_type}.modeling_{model_type}"


def gdn_reset_arch_from_caps(caps: dict, gdn_module: str) -> str | None:
    """the architecture to patch when the VERL CHILD can honor packed GDN boundary resets, else None.

    transformers silently discards ``cu_seqlens`` without fla and ``seq_idx`` without causal_conv1d,
    contaminating packed boundaries. probe the child interpreter because it owns the model stack.

    return the verified ``model_type`` rather than a bool so the shim cannot patch a different module;
    ``qwen3_5`` and ``qwen3_5_moe`` are distinct. child probes must not import parent-only ``flash``.
    """
    if not caps.get("gdn_boundary_resets"):
        return None
    # "transformers.models.<arch>.modeling_<arch>" -> "<arch>", read from the string the CHILD was
    # handed rather than re-resolved here: a second resolve could disagree with what was cleared.
    return gdn_module.split(".")[2]


def require_gdn_boundary_resets(caps: dict, gdn_module: str) -> str | None:
    """The arch to patch for a gdn hybrid, raising if the child cannot honor boundary resets.

    On grpo/opd there is no safe fallback, so this raises rather than launching. verl's padded path
    (``use_remove_padding=False``) is boundary-correct but cannot complete a step: flash sets
    ``use_fused_kernels=True`` unconditionally and verl's fsdp engine has no guard for that pair, so
    the padded+fused branch returns a dense ``[bsz, response_len]`` where siblings re-nest via
    ``cu_seqlens``. That fails at EVERY batch size, measured dead at ``padding.py:144`` on four
    matched real-gpu arms, and the error names neither gdn nor this decision -- so letting it launch
    spends the whole rental on an untraceable shape error. Dropping ``use_fused_kernels`` is NOT the
    alternative: ``engine/vram.py`` sizes assuming no dense ``[b, s, vocab]`` logits tensor exists
    (~130 GB at 32k on a 248k vocab), trading this crash for an OOM. Give the child the kernels.

    sft reaches this too, and a packed sft run needs resets for the same reason: its control-plane
    gate is device-independent by construction (see ``gdn_packing_contract_available``), so it can
    prove the kernels are installed but never that
    the conv kernel runs on this card. an ``exact-unpacked`` sft run stays safe without resets --
    ``examples_per_update`` is 1, so there are no packed neighbours to contaminate -- which is why
    sft asks this only for a packed profile.
    """
    if not gdn_module:
        return None
    arch = gdn_reset_arch_from_caps(caps, gdn_module)
    if arch is None:
        raise RuntimeError(
            "gdn hybrid without child-side boundary resets: the padded fallback "
            "(use_remove_padding=False) is incompatible with the use_fused_kernels=True this "
            "recipe sets, and cannot complete a training step on verl's fsdp engine. see the "
            "'[verl] gdn boundary resets unavailable' line above for why the child could not "
            "honor resets -- installing fla + causal_conv1d in the verl interpreter is the fix, "
            "not disabling fused kernels."
        )
    return arch


def resolve_verl_loggers(caps: dict) -> list[str]:
    """verl's ``trainer.logger`` list, gated on the interpreter that actually logs.

    verl logs from its own interpreter, so wandb must be importable THERE, not in flash's env. The
    prebuilt worker image's /opt/verl-venv does not install wandb, and a preset FLASH_VERL_PYTHON is
    returned by resolve_verl_python untouched (flash does not own that interpreter), so a run with
    WANDB_API_KEY set would otherwise ask verl for a logger it cannot import and die at logger init
    on a paid GPU.
    """
    if not os.environ.get("WANDB_API_KEY"):
        return ["console"]
    if not caps.get("wandb"):
        print(
            "[verl] WANDB_API_KEY set but wandb is unavailable in the verl interpreter; "
            "using console logger only"
        )
        return ["console"]
    return ["console", "wandb"]


def verl_device_capability(caps: dict) -> tuple[int, int] | None:
    """device 0's ``(major, minor)`` cuda capability as the VERL interpreter sees it, or None.

    probed in verl's interpreter rather than flash's because verl owns the rollout engine and pins
    its own torch/vllm stack, and both callers below are deciding what that stack can do. None means
    the probe could not answer -- no cuda, no torch, a hung import -- and every caller must read it
    as "leave the default alone" rather than guessing a workaround onto an unknown card.
    """
    cc = caps.get("capability")
    if not cc:
        return None
    try:
        return (int(cc[0]), int(cc[1]))
    except (TypeError, ValueError, IndexError):
        return None


# verl ships two fused linear-CE backends and dispatches on `fused_kernel_options.impl_backend`
# (models/transformers/monkey_patch.py). all three flash trainers used to pin `torch`, which is
# verl's own default (config/model/hf_model.yaml:67) -- a restated default rather than a choice.
#
# MEASURED, 4096x2560 hidden into a 248320 vocab, bf16, fwd+bwd, paired and alternating, 10 reps
# after warmup, verl's own entry points on both arms:
#
#     card               sm    triton vs torch   peak VRAM        triton err vs fp32
#     A100 SXM4 80GB     80    3.28x SLOWER      2.75x lower      lower on all 4 quantities
#     A40                86    1.43x SLOWER      2.75x lower      lower on all 4 quantities
#     H100 80GB HBM3     90    2.25x FASTER      2.72x lower      lower on all 4 quantities
#
# so the choice is card-dependent and a blanket flip would slow every sm80/sm86 run. the split is
# sm90: the triton kernel's TMA path is gated on `get_device_capability()[0] >= 9`
# (verl/utils/kernel/kernels.py:48), and below that it takes a general mainloop that loses to the
# torch chunked loop. the H100 win is stable across shapes (1.9x at 1024 tokens, 2.15x at 2048,
# 2.25x at 4096 and 8192).
#
# ACCURACY is not the deciding axis but is worth recording, because "faster" would not be worth a
# worse gradient: measured against an fp32 log-softmax arbiter, triton is closer on EVERY card and
# every quantity -- log_probs 3e-6 vs 7e-4, grad_hidden 6.5e-3 vs 1.8e-2, grad_weight 2.8e-3 vs
# 1.8e-2. the two backends differ from each other by ~1.9e-2 on gradients, and that gap is mostly
# the torch backend's own error, not triton's.
_TRITON_FUSED_CE_MIN_CAPABILITY = (9, 0)


def fused_ce_backend(caps: dict) -> str:
    """`triton` where it is measurably faster (sm90+), `torch` everywhere else.

    Takes the capability from the out-of-process probe (`verl_device_capability`) rather than
    calling `torch.cuda.get_device_capability` here, for two reasons:

    * this runs in the long-lived PARENT, which does not otherwise touch cuda. initializing a
      context here to answer one question retains it for the process lifetime, on the same devices
      `torchrun` is about to own -- unbudgeted VRAM against a reserve sized without it, which is
      exactly the kind of overhead that OOMs a job sized near a card's limit.
    * the question is about verl's kernels, and verl pins its own torch. the parent's torch is the
      wrong interpreter to ask.

    An unanswerable probe (no cuda, no torch, hung import) yields None and falls back to `torch`,
    which is the current behaviour on every card.
    """
    cc = verl_device_capability(caps)
    if cc is None:
        return "torch"
    return "triton" if cc >= _TRITON_FUSED_CE_MIN_CAPABILITY else "torch"


def resolve_rollout_enforce_eager(cc: tuple[int, int] | None) -> bool:
    """whether this GPU must run the rollout eagerly instead of capturing cuda graphs.

    One knob is enough and cannot fight verl's: vllm 0.19.1 resolves ``enforce_eager`` LAST, forcing
    ``compilation_config.mode=NONE`` and ``cudagraph_mode=NONE`` regardless of what was requested
    (``config/vllm.py:847-853`` and ``:1024-1029``). An unanswerable probe (``cc is None``) leaves
    verl's default in place rather than guessing eager onto an unknown card.
    """
    if cc is None:
        return False
    # sm86 is the only measured graph-capture failure. everything else -- sm80/sm89/sm90 and
    # blackwell -- captures. an arch nobody has measured keeps verl's default rather than being
    # forced eager on the strength of one other Ampere minor version's defect.
    if cc == (8, 6):
        print(
            "[verl] sm86: enforce_eager=True for the rollout (vllm 0.19.1 graph capture degenerates "
            "on this arch: completions repeat to the token cap without emitting EOS)"
        )
        return True
    return False


def rollout_sleep_unsupported(model_id: str) -> bool:
    """whether vLLM's sleep/wake cycle is non-functional for this model.

    a few models HANG on wake/reload rather than erroring (live-confirmed, every attempt), so the
    catalog flags them and the rollout engine must stay RESIDENT instead. shared by grpo and opd:
    verl runs the same ``vllm_async_server`` sleep path for both, so the flag has to mean the same
    thing on both drivers or the one that ignores it wedges on the flagged model.
    """
    from flash.core.catalog import MODELS

    info = MODELS.get(model_id)
    return bool(info is not None and getattr(info, "sleep_unsupported", False))


def rollout_fp8_kv(cc_ok: bool, gdn_hybrid: bool, model_id: str) -> bool:
    """whether the rollout engine may cache KV in fp8. shared by grpo and opd.

    ``cc_ok`` (cc>=8.9) is the hardware floor. the gdn exclusion on top of it is narrower than it
    looks: the crash it avoids is vllm's ``init_fp8_kv_scales``, which zeroes each cache tensor and
    hits ``'list' object has no attribute 'zero_'`` on a hybrid cache. that function is reachable
    from exactly ONE place -- ``gpu_worker.wake_up`` (``v1/worker/gpu_worker.py:198-203``) -- and
    verl only ever wakes an engine it put to sleep: both training-path callers are gated on
    ``rollout.free_cache_engine`` (``engine_workers.py:706,741`` -> ``vllm_rollout.py:161``). a model
    the catalog flags ``sleep_unsupported`` runs with that flag pinned false by
    ``rollout_resident_overrides``, so it never sleeps, never wakes, and never reaches the crash.

    the resident engine still gets fp8 for its ordinary attention layers: the recurrent state is a
    separate ``MambaSpec`` carrying its own dtypes, so fp8 never applies to it. vram sizing already
    prices this exact combination -- ``mamba_block_size=1072`` is the fp8-derived attention-token
    page (the 1,097,728-byte gdn state over a 1024-byte fp8 attention token, rounded to vllm's
    16-token alignment) and ``kv_sizing`` applies it only when ``fp8_kv`` is set.

    this also closes a planner/worker disagreement rather than merely freeing memory: the grpo
    resident wall in ``vram.py`` sizes a ``sleep_unsupported`` model with ``fp8_kv=True`` already,
    so a worker that sent bf16 needed ~6-7 GB MORE than the planner had reserved for it (measured
    at 4096x8 and 8192x4). the two halves now assume the same cache dtype.
    """
    if not cc_ok:
        return False
    if not gdn_hybrid:
        return True
    # gdn + sleeping engine is the combination that crashes on wake; gdn + resident is not.
    return rollout_sleep_unsupported(model_id)


_MIN_ROLLOUT_MAX_NUM_SEQS = 16


def rollout_max_num_seqs(rollout_batch: int) -> int:
    """concurrent sequence slots to provision the rollout engine for.

    verl leaves ``rollout.max_num_seqs`` at its yaml default of 1024 (rollout.yaml:79) and passes
    it straight to the vllm engine (vllm_async_server.py:261). vllm sizes two up-front, fixed
    allocations from that number, neither of which shrinks to the batch actually submitted:

    * cuda-graph capture. ``_set_cudagraph_sizes`` (vllm ``config/vllm.py``, 0.19.1 -- the version
      ``Dockerfile.worker`` pins) derives a capture ladder from
      ``max_cudagraph_capture_size = min(max_num_seqs * 2, 512, max_num_batched_tokens)``, taking
      ``[1,2,4]``, then ``range(8, min(size+1, 256), 8)``, then ``range(256, size+1, 16)`` -- each
      rung clamped by that ceiling. verl's ``max_num_batched_tokens`` default is 8192
      (``rollout.yaml:73``), so it does not bind. at 1024 the ceiling is 512 and the ladder is 51
      sizes; a lora run captures each size twice, since the dispatcher takes
      ``product(cudagraph_capture_sizes, lora_cases)`` with ``lora_cases == [0, 1]`` when
      ``cudagraph_specialize_lora`` is on (``v1/cudagraph_dispatcher.py:110-193``, default True) --
      exactly the ``0/102`` seen in the failing console. the ladder saturates at 512, so 1024, 512
      and 256 are indistinguishable; only dropping below 256 shortens it, and 32 yields a ceiling
      of 64, 11 sizes, and 22 graphs. note the prose comment above that function omits the per-rung
      clamp and predicts 34 sizes for any small value; the implementation is authoritative.
    * recurrent state. a gdn/mamba hybrid reserves one state block per decode slot, a dense
      allocation that -- unlike paged kv -- cannot be shared, paged out, or grown on demand.

    both are paid before the first rollout token, inside the ``gpu_memory_utilization`` budget. a
    grpo/opd step submits exactly ``prompts_per_step * group_size`` sequences, so a run with 32
    concurrent generations was provisioning for 1024 and dying during graph capture with ~80% of
    the card free -- on 234, 358, and 460 GB alike, since the cost is fixed rather than
    proportional to the card.

    the floor keeps small runs on a usable capture ladder rather than collapsing it to one size.
    """
    if rollout_batch <= 0:
        raise ValueError("rollout_batch must be positive")
    return max(_MIN_ROLLOUT_MAX_NUM_SEQS, int(rollout_batch))


def rollout_resident_overrides(sleep_unsupported: bool) -> list[str]:
    """verl overrides that pin the rollout engine resident, or ``[]`` to keep verl's default.

    disable both ``free_cache_engine`` and ``enable_sleep_mode`` or flagged models hang on wake
    (``vllm_async_server.py:626,648`` gates sleep on both). VRAM sizing already uses the resident
    peak. prefixes differ: the yaml key takes a bare override, while ``enable_sleep_mode`` is
    dataclass-only (``workers/config/rollout.py:277``) and so requires ``+``.
    """
    if not sleep_unsupported:
        return []
    return [
        "actor_rollout_ref.rollout.free_cache_engine=false",
        "+actor_rollout_ref.rollout.enable_sleep_mode=false",
    ]


def rollout_layered_summon_overrides(target_parameters: Sequence[str] | None) -> list[str]:
    """gather the fsdp weight sync one submodule at a time for fused-expert lora models.

    the rollout weight sync collects lora params through ``collect_lora_params``
    (verl ``utils/fsdp_utils.py:673``). its default branch opens
    ``FSDP.summon_full_params(module)`` over the WHOLE model, so every layer's expert stack is
    unsharded and resident simultaneously. ``layered_summon`` takes
    ``layered_summon_lora_params`` instead, which summons one submodule at a time and calls
    ``empty_cache()`` between layers, so the peak is a single layer's slab.

    that difference only matters when a target tensor is large, and a non-empty
    ``target_parameters`` is exactly the fused-moe case. there ``mlp.experts.gate_up_proj`` is a
    fused bf16 slab of ``lora_expert_count x 2048 x 1024 x 2`` bytes -- on Qwen3.6-35B-A3B
    (``lora_expert_count=256``) that is 1,073,741,824 bytes, i.e. EXACTLY 1024 MiB, which is
    precisely the contiguous request that OOM'd at step 0 on a 180 GB B200 across every card count
    and context tested. dense models keep verl's default: their per-tensor gathers are small and
    the layered walk is slower.

    verl gates the layered path on the base weights already living in vllm
    (``fsdp_utils.py:684``: it raises unless ``base_sync_done``, which
    ``engine_workers.py:614`` derives as ``"dummy" not in load_format``). the rl and opd drivers
    both already pin ``load_format=safetensors``, so that precondition holds wherever this
    applies. ``layered_summon`` is a rollout dataclass field
    (``workers/config/rollout.py:261``) that is also declared in ``rollout.yaml:98``, so it takes
    a bare override.
    """
    if not target_parameters:
        return []
    return ["actor_rollout_ref.rollout.layered_summon=true"]


def actor_fsdp_strategy_overrides(fsdp_generation: FsdpGeneration) -> list[str]:
    """render one explicit bare actor strategy override from the resolved private policy."""
    strategy = "fsdp2" if fsdp_generation == 2 else "fsdp"
    return [f"actor_rollout_ref.actor.strategy={strategy}"]


def rollout_mm_processor_cache_overrides() -> list[str]:
    """disable vllm's split multimodal processor cache for every verl rollout.

    verl sleep resets the engine-side receiver without resynchronizing the frontend sender, so a
    repeated image can become a hash-only request for an item the receiver no longer holds. zero is
    vllm's disable value for both halves and is inert when a rollout has no multimodal items.
    """
    return ["+actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0"]


def trainer_dtype_overrides() -> list[str]:
    """verl overrides that store the frozen base in bf16 instead of verl's fp32 default.

    fsdp.yaml passes its fp32 default to ``from_pretrained`` (``transformer_impl.py:234``). flash
    freezes the base and computes under bf16 mixed precision, while PEFT keeps trainable adapter
    weights fp32 (``peft/mapping_func.py:35``, not overridden by verl), so fp32 base residency has
    no master-weight benefit. this is specific to LoRA, not full tuning.

    set only ``actor`` because ``ref_in_actor`` aliases the reference worker to it
    (``ray_trainer.py:897``), rather than creating a second resident copy.
    """
    return ["actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16"]


def resolve_blackwell_attention_backends(
    caps: dict, cc: tuple[int, int] | None
) -> tuple[str | None, str | None]:
    """the rollout ``(attention_backend, mm_encoder_attn_backend)`` this GPU needs, or ``(None, None)``.

    on Blackwell, use FLASHINFER or TRITON_ATTN for decoder attention; TORCH_SDPA is not registered
    there in vllm 0.19.1 (``vllm/platforms/cuda.py:99-111``). pin ViT attention to TORCH_SDPA
    because no published cutlass version satisfies both CUTE APIs; ``cuda.py:344-373`` picks the
    default ViT backend and honors an explicit override. non-Blackwell or unknown caps keep defaults.
    """
    if cc is None:
        return (None, None)
    major = cc[0]
    if major not in (10, 12):
        return (None, None)
    has_flashinfer = bool(caps.get("flashinfer"))
    decoder = "FLASHINFER" if has_flashinfer else "TRITON_ATTN"
    if not has_flashinfer:
        print(
            f"[verl] sm{major}0 (Blackwell): flashinfer is not importable in the verl interpreter "
            "-> attention_backend=TRITON_ATTN (PTX-independent registered decoder backend)"
        )
    else:
        print(
            f"[verl] sm{major}0 (Blackwell): attention_backend=FLASHINFER "
            "(flash-attn PTX is unreliable on consumer Blackwell -> empty-rollout failures)"
        )
    print(
        f"[verl] sm{major}0 (Blackwell): mm_encoder_attn_backend=TORCH_SDPA "
        "(vllm 0.19.1 ViT CUTE flash-attn is unimportable vs every nvidia-cutlass-dsl)"
    )
    return (decoder, "TORCH_SDPA")
