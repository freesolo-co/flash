"""Datums-parity GRPO knobs + init-from-adapter wiring (CPU-only, no GPU/network).

The SDK ships the GRPO recipe knobs (group_size/temperature/advantage_clip/
kl_penalty_coef/thinking_length_penalty_coef) plus the optimizer/batching knobs
(learning_rate/batch_size/max_length/save_every) in the job spec's ``[train]`` table
(TrainSpec) — NOT ``[environment.params]``, which is forwarded verbatim to the Freesolo
env's ``load_environment`` — and an optional ``train.init_from_adapter``; these tests
cover the pure plumbing the worker uses to honor them (the GPU trainer wiring itself is
exercised by the live smokes).
"""

from __future__ import annotations

import pytest

from flash.schema import ConfigError, spec_from_dict
from flash.spec import JobSpec


class _Tok:
    """Whitespace tokenizer stub: one token per word."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def test_think_token_count_counts_the_think_span() -> None:
    import flash.engine.worker as w

    tok = _Tok()
    assert w.think_token_count("<think>a b c</think>the answer", tok) == 3
    assert w.think_token_count("no reasoning here", tok) == 0
    # an unclosed block (budget exhausted) counts everything after <think>
    assert w.think_token_count("pre <think>a b c d", tok) == 4
    assert w.think_token_count(None, tok) == 0
    assert w.think_token_count("<think></think>x", tok) == 0
    # prompt-opened hybrid thinking: the chat template appended <think> to the PROMPT, so the
    # completion starts mid-reasoning with only the closing </think>. The reasoning is everything
    # before that close (without this the penalty no-ops on the common enable_thinking=true path).
    assert w.think_token_count('a b c d</think>{"x": 1}', tok) == 4
    assert w.think_token_count("</think>just the answer", tok) == 0
    # case 3: prompt-opened thinking that NEVER closes (ran out of max_tokens) — no tags at all. With
    # prompt_opened_thinking the WHOLE completion is unterminated reasoning and is counted, so the
    # longest rambles can't dodge the penalty; without the flag a tag-less completion is plain text (0).
    assert w.think_token_count("rambling on and on forever", tok, prompt_opened_thinking=True) == 5
    assert w.think_token_count("rambling on and on forever", tok) == 0
    # the flag does NOT change a completion that already carries a tag (cases 1/2 still win).
    assert w.think_token_count("a b c</think>ans", tok, prompt_opened_thinking=True) == 3
    assert w.think_token_count("<think>a b</think>ans", tok, prompt_opened_thinking=True) == 2
    assert w.think_token_count("", tok, prompt_opened_thinking=True) == 0
    # Case 1 vs 2 is decided by tag ORDER, not presence: a prompt-opened completion that CLOSES its
    # reasoning and then echoes a literal <think> in the answer must count the span up to the FIRST
    # </think> (the reasoning), NOT anchor on the echoed opener (which would count "echo here" = 2).
    assert w.think_token_count("a b c d</think>answer with <think> echo here", tok) == 4
    # a self-tagged block followed by an echoed opener still counts only the first real span.
    assert w.think_token_count("<think>a b c</think>tail <think> echo", tok) == 3
    # prompt-opened + NEVER closed + an echoed <think>: count the WHOLE completion (it's all
    # unterminated reasoning), not just the text after the echoed opener.
    assert w.think_token_count("reason 42 <think> more", tok, prompt_opened_thinking=True) == 4
    # the same echoed completion WITHOUT the prompt-open signal: the model opened <think> itself
    # (unclosed) -> count after that opener (case: model-opened unclosed).
    assert w.think_token_count("reason 42 <think> more", tok) == 1
    # prompt-opened + an echoed <think> BEFORE the first </think>: the prompt pre-opened reasoning, so
    # the span is the WHOLE pre-opened reasoning from the start through the first close
    # ("reason 42 <think> more" = 4) -- NOT just the sliver after the echoed opener (" more" = 1).
    assert (
        w.think_token_count("reason 42 <think> more </think> ans", tok, prompt_opened_thinking=True)
        == 4
    )
    # the same string WITHOUT the prompt-open signal: the model opened AND closed its own <think>, so
    # only the span between the model's tags counts (" more" = 1) -- case 1.
    assert w.think_token_count("reason 42 <think> more </think> ans", tok) == 1


def test_prompt_opens_thinking_detects_preopened_tag() -> None:
    import flash.engine.worker as w

    # A hybrid template pre-opens <think> at the end of the generation prompt (no closing tag).
    assert w.prompt_opens_thinking("<|im_start|>assistant\n<think>\n") is True
    # An uncurated/non-thinking template appends no <think> -> a tagless completion is a real answer.
    assert w.prompt_opens_thinking("<|im_start|>assistant\n") is False
    assert w.prompt_opens_thinking("") is False
    assert w.prompt_opens_thinking(None) is False
    # A prompt that opened AND closed a <think> (e.g. a few-shot exemplar) is NOT pre-opened.
    assert w.prompt_opens_thinking("...<think>example</think>...<|im_start|>assistant\n") is False
    # If the LAST think is left open (after an earlier closed one), it IS pre-opened.
    assert w.prompt_opens_thinking("<think>ex</think>q<|im_start|>assistant\n<think>\n") is True
    # FALSE-POSITIVE guard: a user/system message that merely CONTAINS an unclosed literal <think>
    # must NOT count as pre-opened when the generation suffix didn't actually prefill thinking (the
    # detection anchors on the trailing <think> suffix, not a scan of the whole prompt).
    assert w.prompt_opens_thinking("user asked <think> about x<|im_start|>assistant\n") is False


def test_graded_text_hides_tagless_prompt_opened_reasoning(monkeypatch) -> None:
    import flash.engine.worker as w

    monkeypatch.setattr(w, "THINKING", True)
    # Tagless completion under a prompt-opened <think>: the generation never closed reasoning, so the
    # env must grade NOTHING (scores 0) — not the raw ramble (which a raw-text fallback could reward).
    assert w.graded_text("rambling forever no answer", prompt_opened_thinking=True) == ""
    # Without the prompt-opened signal (e.g. an uncurated template that didn't pre-open), the same
    # tagless text is a normal answer and is graded as-is.
    assert w.graded_text("the answer is 42", prompt_opened_thinking=False) == "the answer is 42"
    # A normally-tagged thinking completion is unaffected: strip to the post-</think> answer.
    assert (
        w.graded_text("reasoning...</think>\\boxed{5}", prompt_opened_thinking=True) == "\\boxed{5}"
    )
    # Echoed <think> while still unterminated (no </think>): the WHOLE thing is reasoning -> hidden,
    # NOT just the text before the echoed opener (which a raw-text fallback could otherwise reward).
    assert w.graded_text("reason 42 <think> still going", prompt_opened_thinking=True) == ""
    # THINKING off: no stripping at all, even with the flag set.
    monkeypatch.setattr(w, "THINKING", False)
    assert w.graded_text("rambling forever", prompt_opened_thinking=True) == "rambling forever"


def test_grpo_overrides_reads_train_knobs(monkeypatch) -> None:
    import flash.engine.worker as w

    knobs = {
        "group_size": 4,
        "temperature": 0.7,
        "max_tokens": 256,
        "advantage_clip": 1.5,
        "kl_penalty_coef": 0.02,
        "thinking_length_penalty_coef": 0.001,
    }
    # GRPO knobs live in [train]/TrainSpec, NOT [environment.params].
    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "grpo",
            "environment": {"id": "github:owner/repo@main:env/environment.py"},
            "train": {**knobs},
        }
    )
    monkeypatch.setattr(w, "JOB_SPEC", spec)
    assert w.grpo_overrides() == knobs
    # A leftover grpo_config in environment.params must NOT be read by the worker.
    poisoned = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "grpo",
            "environment": {
                "id": "github:owner/repo@main:env/environment.py",
                "params": {"grpo_config": knobs},
            },
            "train": {},
        }
    )
    monkeypatch.setattr(w, "JOB_SPEC", poisoned)
    assert w.grpo_overrides() == {}
    # only the knobs actually set are returned (a partial set omits the rest)
    monkeypatch.setattr(
        w,
        "JOB_SPEC",
        JobSpec.from_dict(
            {
                "model": "Qwen/Qwen3.5-0.8B",
                "algorithm": "grpo",
                "environment": {"id": "github:owner/repo@main:env/environment.py"},
                "train": {"group_size": 2},
            }
        ),
    )
    assert w.grpo_overrides() == {"group_size": 2}
    # no [train] knobs -> empty (recipe defaults apply downstream)
    monkeypatch.setattr(
        w,
        "JOB_SPEC",
        JobSpec.from_dict({"model": "Qwen/Qwen3.5-0.8B", "algorithm": "grpo"}),
    )
    assert w.grpo_overrides() == {}
    monkeypatch.setattr(w, "JOB_SPEC", None)
    assert w.grpo_overrides() == {}


def test_train_grpo_knobs_parse_and_roundtrip() -> None:
    from flash.schema import spec_from_dict

    raw = {
        "model": "Qwen/Qwen3.5-0.8B",
        "algorithm": "grpo",
        "model_policy": "allow",
        "environment": {"id": "github:owner/repo@main:env/environment.py"},
        "gpu": {"type": "cheapest"},
        "train": {
            "steps": 10,
            "hf_repo": "owner/runs",
            "group_size": 4,
            "temperature": 0.7,
            "max_tokens": 256,
            "kl_penalty_coef": 0.02,
            "advantage_clip": 1.5,
            "thinking_length_penalty_coef": 0.001,
        },
    }
    spec = spec_from_dict(raw, run_id="grpo-x")
    assert spec.train.group_size == 4
    assert spec.train.temperature == 0.7
    assert spec.train.max_tokens == 256
    assert spec.train.kl_penalty_coef == 0.02
    assert spec.train.advantage_clip == 1.5
    assert spec.train.thinking_length_penalty_coef == 0.001
    # survives the JSON round-trip the worker reconstructs from
    rt = JobSpec.from_dict(spec.to_dict()).train
    assert rt.group_size == 4
    assert rt.thinking_length_penalty_coef == 0.001
    # GRPO knobs are NOT in environment.params (that goes verbatim to load_environment)
    assert spec.environment.params == {}


def test_opt_int_float_reject_bools() -> None:
    """A JSON boolean must NOT silently coerce to a numeric train knob: bool is an int
    subclass in Python, so ``int(True)`` would become 1. JobSpec.from_dict (via
    _opt_int/_opt_float) rejects it, matching schema._opt_num."""
    import pytest

    from flash.spec import _opt_float, _opt_int

    for bad in (True, False):
        with pytest.raises(TypeError):
            _opt_int(bad)
        with pytest.raises(TypeError):
            _opt_float(bad)

    # Genuine numbers (and None) still parse.
    assert _opt_int(None) is None
    assert _opt_int(4) == 4
    assert _opt_float(None) is None
    assert _opt_float(0.7) == 0.7

    # A bool train knob propagates through JobSpec.from_dict as an error, not a 0/1 coercion.
    with pytest.raises(TypeError):
        JobSpec.from_dict(
            {
                "model": "Qwen/Qwen3.5-0.8B",
                "algorithm": "grpo",
                "environment": {"id": "github:owner/repo@main:env/environment.py"},
                "train": {"steps": 10, "group_size": True},
            }
        )


def _spec_raw(ref: str) -> dict:
    return {
        "model": "Qwen/Qwen3.5-0.8B",
        "algorithm": "grpo",
        "model_policy": "allow",
        "environment": {"id": "github:owner/repo@main:env/environment.py"},
        "gpu": {"type": "cheapest"},
        "train": {"steps": 10, "hf_repo": "owner/runs", "init_from_adapter": ref},
    }


def test_init_from_adapter_parses_and_roundtrips() -> None:
    """The canonical short ref (`<run_id>`) is accepted and survives the JSON round-trip."""
    spec = spec_from_dict(_spec_raw("run-x"), run_id="grpo-x")
    assert spec.train.init_from_adapter == "run-x"
    assert JobSpec.from_dict(spec.to_dict()).train.init_from_adapter == "run-x"
    # absent -> empty string (train fresh from base)
    raw = _spec_raw("run-x")
    raw["train"].pop("init_from_adapter")
    assert spec_from_dict(raw, run_id="grpo-y").train.init_from_adapter == ""


def test_init_from_adapter_accepts_checkpoint_step_ref() -> None:
    """`<run_id>/step-N` (exactly what `flash checkpoints` lists) warm-starts from that saved
    step instead of the run-level adapter — e.g. continuing GRPO from the best checkpoint."""
    spec = spec_from_dict(_spec_raw("run-x/step-40"), run_id="grpo-x")
    assert spec.train.init_from_adapter == "run-x/step-40"
    assert JobSpec.from_dict(spec.to_dict()).train.init_from_adapter == "run-x/step-40"


@pytest.mark.parametrize(
    "bad_ref",
    [
        "Freesolo-Co/flashrun-run-x:sft/run-x",  # legacy long storage form: no longer accepted
        "Freesolo-Co/flashrun-run-x:rl/run-x/checkpoints/step-40",
        "run-x/step-",  # no step number
        "run-x/step-4/adapter",  # trailing path
        "run-x/checkpoints/step-4",  # storage-layout spelling
        "run-x/../other",  # traversal
        "run x/step-4",  # bad run id
    ],
)
def test_init_from_adapter_rejects_non_short_refs(bad_ref: str) -> None:
    with pytest.raises(ConfigError, match="run_id"):
        spec_from_dict(_spec_raw(bad_ref), run_id="grpo-x")


@pytest.mark.parametrize(
    "bad_ref",
    [
        123,
        False,
        ["run-x/step-4"],
    ],
)
def test_init_from_adapter_rejects_non_string_value(bad_ref: object) -> None:
    raw = {
        "model": "Qwen/Qwen3.5-0.8B",
        "algorithm": "grpo",
        "environment": {"id": "github:owner/repo@main:env/environment.py"},
        "gpu": {"type": "cheapest"},
        "train": {
            "steps": 10,
            "init_from_adapter": bad_ref,
        },
    }
    with pytest.raises(ConfigError, match=r"train\.init_from_adapter must be a string"):
        spec_from_dict(raw, run_id="grpo-x")


def test_hf_repo_is_managed_not_user_set() -> None:
    # [train] hf_repo is the platform-managed per-run HF artifact repo: the control plane assigns
    # it server-side at submit (see runner.submit_job). It is NOT required and a user-supplied
    # value is IGNORED — verified through both schema (server) and JobSpec.from_dict (worker).
    raw = {
        "model": "Qwen/Qwen3.5-0.8B",
        "algorithm": "grpo",
        "environment": {"id": "github:owner/repo@main:env/environment.py"},
        "train": {"steps": 10},
    }
    # absent -> fine (no longer required); left blank for the control plane to assign
    spec = spec_from_dict(raw, run_id="hf-x")
    assert spec.train.hf_repo == ""
    assert JobSpec.from_dict(spec.to_dict()).train.hf_repo == ""
    # user-supplied -> ignored (the control plane overrides it at submit)
    spec2 = spec_from_dict(
        {**raw, "train": {"steps": 10, "hf_repo": "someone-else/their-repo"}},
        run_id="hf-y",
    )
    assert spec2.train.hf_repo == ""


def test_optimizer_and_batching_knobs_roundtrip() -> None:
    # The SDK's SftConfig/GrpoConfig optimizer/batching knobs must survive schema
    # (server validation) AND the worker's JobSpec.from_dict, or the worker would silently
    # train with recipe defaults while W&B reports the user's values.
    raw = {
        "model": "Qwen/Qwen3.5-0.8B",
        "algorithm": "grpo",
        "model_policy": "allow",
        "environment": {"id": "github:owner/repo@main:env/environment.py"},
        "gpu": {"type": "cheapest"},
        "train": {
            "hf_repo": "owner/runs",
            "learning_rate": 3e-5,
            "batch_size": 16,
            "max_length": 2048,
            "save_every": 5,
            "max_tokens": 512,
            "stop_sequences": ["</answer>", "\n\n"],
        },
    }
    spec = spec_from_dict(raw, run_id="grpo-z")
    for s in (spec, JobSpec.from_dict(spec.to_dict())):  # server parse + worker re-parse
        assert s.train.learning_rate == 3e-5
        assert s.train.batch_size == 16
        assert s.train.max_length == 2048
        assert s.train.save_every == 5
        assert s.train.max_tokens == 512
        assert s.train.stop_sequences == ("</answer>", "\n\n")
    # omitted optimizer knobs stay None so the worker applies its recipe defaults
    bare = spec_from_dict({**raw, "train": {"hf_repo": "owner/runs"}}, run_id="grpo-w")
    assert bare.train.learning_rate is None
    assert bare.train.batch_size is None
    assert bare.train.stop_sequences == ()
    # a bare-string stop_sequences is ONE stop, never split into characters
    one = spec_from_dict(
        {**raw, "train": {"hf_repo": "owner/runs", "stop_sequences": "</answer>"}},
        run_id="grpo-s",
    )
    assert one.train.stop_sequences == ("</answer>",)
    assert JobSpec.from_dict(one.to_dict()).train.stop_sequences == ("</answer>",)
    # an empty string means "no stop configured" -> (), not ("",); empty list entries drop
    empty = spec_from_dict(
        {**raw, "train": {"hf_repo": "owner/runs", "stop_sequences": ""}},
        run_id="grpo-e",
    )
    assert empty.train.stop_sequences == ()
    dropped = spec_from_dict(
        {**raw, "train": {"hf_repo": "owner/runs", "stop_sequences": ["x", ""]}},
        run_id="grpo-d",
    )
    assert dropped.train.stop_sequences == ("x",)


def test_rl_per_device_logits_budget_cap(monkeypatch) -> None:
    """The per-device completion micro-batch caps to the fp32-logits budget: a short completion
    keeps the base (8), a long one (4096 tok x ~152k vocab x 4 B ~ 2.5 GB/unit) caps to fit the
    6 GB budget, pushing the rest into grad-accum. (Offline: the colocate VRAM cap is GPU-only.)"""
    import flash.engine.worker as w

    monkeypatch.setattr(w, "THINKING", False, raising=False)
    _no_cuda(monkeypatch)  # assert the offline contract deterministically, even on a GPU host
    # short completion: budget non-binding -> base default 8
    assert w.rl_per_device_comps(512, vocab=152_000, use_vllm=True) == 8
    # long completion: 6e9 / (4096*152000*4) ~ 2.4 -> capped to 2 (budget fixed at 6 GB, managed)
    assert w.rl_per_device_comps(4096, vocab=152_000, use_vllm=True) == 2


def test_rl_per_device_fused_logits_lifts_budget(monkeypatch) -> None:
    """When the fused GRPO loss is on (use_liger_kernel, unconditional on the GRPO path) the fp32
    [pd, completion, vocab] logits are NEVER materialized, so the 6 GB logits budget should NOT bind
    — it models a tensor that doesn't exist. ``fused_logits=True`` drops the budget term so the
    ceiling / activation cap binds instead. The biggest beneficiary is a long multi-turn transcript,
    where the unfused cap collapses to 1."""
    import flash.engine.worker as w

    monkeypatch.setattr(w, "THINKING", False, raising=False)
    _no_cuda(monkeypatch)  # offline: result = min(default 8, logits_cap)
    # A 4096-tok completion at 248k vocab: unfused budget ~= 6e9/(4096*248320*4) ~ 1.4 -> binds at 1.
    assert w.rl_per_device_comps(4096, vocab=248_320, use_vllm=True) == 1
    # Fused: the 6 GB term is dropped -> the offline default (8) binds, NOT the phantom-logits cap.
    assert w.rl_per_device_comps(4096, vocab=248_320, use_vllm=True, fused_logits=True) == 8
    # Default is unfused (the safety net stays in force for the non-liger fallback path).
    assert w.rl_per_device_comps(4096, vocab=248_320, use_vllm=True, fused_logits=False) == 1


def _no_cuda(monkeypatch) -> None:
    """Force the offline path: a `torch` whose CUDA is unavailable, so rl_per_device_comps falls
    back to its historical default regardless of whether the test host has a GPU."""
    import sys
    import types

    t = types.ModuleType("torch")
    t.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", t)


def _fake_cuda(monkeypatch, vram_gb: float) -> None:
    """Install a fake `torch` exposing a live CUDA card of `vram_gb` so the colocate VRAM cap
    in rl_per_device_comps engages off-GPU."""
    import sys
    import types

    t = types.ModuleType("torch")

    class _Props:
        total_memory = int(vram_gb * 1024**3)

    t.cuda = types.SimpleNamespace(
        is_available=lambda: True, get_device_properties=lambda _i: _Props()
    )
    monkeypatch.setitem(sys.modules, "torch", t)


def test_rl_per_device_grows_into_vram_on_short_seq(monkeypatch) -> None:
    """On a VRAM-constrained card (24 GB) with a SHORT sequence, the per-device completion
    micro-batch grows from the old seq-independent colocate cap (~5 for 0.8B) to the MEASURED-safe
    plateau entry (8) -> +~12% trainer throughput (RunPod RTX 4090: pd8 fits at 19 GB). Holding
    the effective batch constant (grad-accum) makes this a pure speed/VRAM knob."""
    import flash.engine.worker as w

    monkeypatch.setattr(w, "THINKING", False, raising=False)
    _fake_cuda(monkeypatch, 23.99)  # RTX 4090 / 3090-class 24 GiB
    # 0.8B, seq 1024: old cap = 24/(7.5*0.894/1.41) = 5; calibrated to grow to the measured-safe 8.
    assert w.rl_per_device_comps(128, vocab=248_320, use_vllm=True, params_b=0.8, seq_len=1024) == 8
    # An even shorter sequence does NOT grow further than the proven-safe value (floor clamps it).
    assert w.rl_per_device_comps(128, vocab=248_320, use_vllm=True, params_b=0.8, seq_len=512) == 8


def test_rl_per_device_grows_to_plateau_ceiling_on_roomy_card(monkeypatch) -> None:
    """On a roomy card (A100 80 GB) the micro-batch grows to the plateau ceiling (16) — MEASURED
    flat-vs-default there — and never beyond (pd>=24 regresses)."""
    import flash.engine.worker as w

    monkeypatch.setattr(w, "THINKING", False, raising=False)
    _fake_cuda(monkeypatch, 79.3)
    assert (
        w.rl_per_device_comps(128, vocab=248_320, use_vllm=True, params_b=0.8, seq_len=1024) == 16
    )


def test_rl_per_device_moe_active_params_aware(monkeypatch) -> None:
    """MoE (A3B): the per-device completion micro-batch's activation/VRAM cap must size on the
    ACTIVE backbone (~3B for the 35B-A3B), not the 35B resident total. Without it, sqrt(35) crushes
    vram_cap BELOW the dense default ceiling (8), throttling the A3B below dense models despite cheap
    active compute + ~100 GB free VRAM during the sleep-offloaded backward (the 2-8% GPU-util bug).
    On a B200-class card at the GRPO default sequence: total-width -> 5, active-width -> 8 (~1.6x),
    a pure speed/VRAM knob (grad-accum holds the effective batch, so reward is identical)."""
    import flash.engine.worker as w

    monkeypatch.setattr(w, "THINKING", False, raising=False)
    _fake_cuda(monkeypatch, 178.0)  # B200-class ~180 GB
    # 35B total width throttles the A3B to 5 (below the dense default ceiling of 8) ...
    assert (
        w.rl_per_device_comps(384, vocab=248_320, use_vllm=True, params_b=35.0, seq_len=2368) == 5
    )
    # ... active 3B width lifts it to the dense-validated ceiling (8). ~1.6x bigger GEMMs / step.
    assert (
        w.rl_per_device_comps(
            384, vocab=248_320, use_vllm=True, params_b=35.0, active_params_b=3.0, seq_len=2368
        )
        == 8
    )
    # Dense (no active_params_b) is byte-unaffected: falls back to params_b.
    assert (
        w.rl_per_device_comps(
            384, vocab=248_320, use_vllm=True, params_b=35.0, active_params_b=None, seq_len=2368
        )
        == 5
    )


def test_rl_per_device_thinking_not_grown_by_short_seq(monkeypatch) -> None:
    """THINKING runs are excluded from the short-seq growth path: thinking emits long completions
    whose activation/logprob cost the prompt-only seq_len gate cannot see, so the conservative
    thinking default (2) must hold even on a roomy card + short seq where a non-thinking run would
    grow to the plateau ceiling (16)."""
    import flash.engine.worker as w

    monkeypatch.setattr(w, "THINKING", True, raising=False)
    _fake_cuda(monkeypatch, 79.3)  # same roomy card that grows a non-thinking run to 16
    assert w.rl_per_device_comps(128, vocab=248_320, use_vllm=True, params_b=0.8, seq_len=1024) == 2
    # An even shorter seq does not lift it either — the default ceiling, not _RL_PER_DEVICE_MAX.
    assert w.rl_per_device_comps(128, vocab=248_320, use_vllm=True, params_b=0.8, seq_len=512) == 2


def test_rl_per_device_no_change_at_or_above_ref_seq(monkeypatch) -> None:
    """At/above the calibration seq (2048) the value is byte-for-byte the historical one — growth
    is GATED to short sequences. Holds in BOTH directions: a constrained card stays at its old (low)
    cap, and a roomy card does NOT grow above the old default (the unvalidated long-seq region)."""
    import flash.engine.worker as w

    monkeypatch.setattr(w, "THINKING", False, raising=False)
    # Constrained 24 GB: old colocate cap = 24/(7.5*0.894/1.41) = 5 at any seq -> still 5.
    _fake_cuda(monkeypatch, 23.99)
    assert w.rl_per_device_comps(256, vocab=248_320, use_vllm=True, params_b=0.8, seq_len=2048) == 5
    assert w.rl_per_device_comps(256, vocab=248_320, use_vllm=True, params_b=0.8, seq_len=8192) == 5
    # Roomy 80 GB: at seq 1024 it would grow to 16, but at seq>=ref the gate holds it at the old
    # default 8 (NO unvalidated long-seq growth — the regression is in tokens-in-flight = pd x seq).
    _fake_cuda(monkeypatch, 79.3)
    assert (
        w.rl_per_device_comps(128, vocab=248_320, use_vllm=True, params_b=0.8, seq_len=1024) == 16
    )
    assert w.rl_per_device_comps(128, vocab=248_320, use_vllm=True, params_b=0.8, seq_len=2048) == 8
    assert w.rl_per_device_comps(128, vocab=248_320, use_vllm=True, params_b=0.8, seq_len=8192) == 8


def test_rl_per_device_interpolates_between_floor_and_ref(monkeypatch) -> None:
    """A short seq strictly between the floor and the reference grows the cap by an interior
    (non-clamped) seq_scale, not just the clamp endpoints."""
    import flash.engine.worker as w

    monkeypatch.setattr(w, "THINKING", False, raising=False)
    _fake_cuda(monkeypatch, 23.99)  # 24 GiB
    # seq 1536: seq_scale = 1536/2048 = 0.75 -> 24/(7.5*0.894/1.41*0.75) = 6.7 -> 6 (interior point).
    assert w.rl_per_device_comps(128, vocab=248_320, use_vllm=True, params_b=0.8, seq_len=1536) == 6


def test_rl_per_device_logits_budget_bounds_growth(monkeypatch) -> None:
    """Even with abundant VRAM, the fp32-logits budget stays a hard ceiling on the micro-batch
    (the Liger-fallback safety net): a long completion caps it regardless of free VRAM."""
    import flash.engine.worker as w

    monkeypatch.setattr(w, "THINKING", False, raising=False)
    _fake_cuda(monkeypatch, 80)  # roomy A100 -> VRAM cap is large
    # completion 4096 @ 248k vocab: 6e9/(4096*248320*4) ~ 1.4 -> logits budget binds at 1.
    assert (
        w.rl_per_device_comps(4096, vocab=248_320, use_vllm=True, params_b=0.8, seq_len=1024) == 1
    )


def test_rl_per_device_never_exceeds_hard_max(monkeypatch) -> None:
    """The hard ceiling (16) holds even on an enormous card with a tiny completion (the throughput
    plateau tops out there; pd>=24 was measured to regress)."""
    import flash.engine.worker as w

    monkeypatch.setattr(w, "THINKING", False, raising=False)
    _fake_cuda(monkeypatch, 640)  # absurd VRAM
    assert w.rl_per_device_comps(64, vocab=152_000, use_vllm=True, params_b=0.5, seq_len=512) == 16


def test_optimizer_knob_validation_rejects_bad_values() -> None:
    # schema is the server's 400 layer: nonsensical/malformed knobs must raise
    # ConfigError at parse time, not TypeError (500) or a silently-misbehaving worker.
    from flash.schema import ConfigError

    base = {
        "model": "Qwen/Qwen3.5-0.8B",
        "algorithm": "grpo",
        "model_policy": "allow",
        "environment": {"id": "github:owner/repo@main:env/environment.py"},
        "gpu": {"type": "cheapest"},
    }
    bad_cases = [
        {"batch_size": 0},  # must be >= 1
        {"batch_size": -4},
        {"max_length": 0},
        {"save_every": 0},
        {"group_size": 0},
        {"learning_rate": 0},  # must be > 0
        {"learning_rate": -1e-5},
        {"temperature": -0.1},  # must be >= 0
        {"kl_penalty_coef": -1},
        {"batch_size": 1.5},  # non-integer
        {"batch_size": "16"},  # wrong type (string)
        {"learning_rate": [1]},  # wrong type (list) -> 400, not a 500 TypeError
        {"stop_sequences": {"a": 1}},  # dict not allowed
        {"stop_sequences": [1, 2]},  # non-string entries
        {"learning_rate": float("nan")},  # non-finite -> 400, not a silent NaN to the optimizer
        {"learning_rate": float("inf")},
        {"temperature": float("inf")},
        {"batch_size": float("inf")},  # int knob: must 400, not OverflowError(500) from int(inf)
        {"max_tokens": float("nan")},
    ]
    for bad in bad_cases:
        with pytest.raises(ConfigError):
            spec_from_dict({**base, "train": {**bad}}, run_id="bad")


def test_steps_and_epochs_reject_non_integer_at_parse() -> None:
    # steps/epochs must run through _train_int like every other integer knob: a
    # non-integer (e.g. steps=1.5) must 400 at parse time, not slip through to the
    # worker and crash int("1.5") AFTER a paid GPU is provisioned.
    from flash.schema import ConfigError

    base = {
        "model": "Qwen/Qwen3.5-0.8B",
        "algorithm": "grpo",
        "model_policy": "allow",
        "environment": {"id": "github:owner/repo@main:env/environment.py"},
        "gpu": {"type": "cheapest"},
    }
    for bad in ({"steps": 1.5}, {"epochs": 2.5}, {"steps": 0}, {"epochs": -1}):
        with pytest.raises(ConfigError):
            spec_from_dict({**base, "train": {"hf_repo": "o/r", **bad}}, run_id="bad")

    # Genuine integers still parse and round-trip.
    spec = spec_from_dict(
        {**base, "train": {"hf_repo": "o/r", "steps": 10, "epochs": 3}},
        run_id="ok",
    )
    assert spec.train.steps == 10
    assert spec.train.epochs == 3


def test_build_grpo_prompt_dataset_keeps_columns_arrow_safe() -> None:
    # Reproduces the ifeval-lite crash: a VALID env whose per-row metadata.param is an int for some
    # rows and a str for others. Embedding the rich record in the dataset makes PyArrow infer one
    # column type across all rows and crash; build_grpo_prompt_dataset stores a stable int index
    # instead, and reward_fn maps it back to the original record.
    import flash.engine.worker as w

    prompts = [
        {"prompt": "p0", "example": {"id": "a", "metadata": {"param": 8}}},  # int param
        {"prompt": "p1", "example": {"id": "b", "metadata": {"param": "gentle"}}},  # str param
        {"prompt": "p2", "example": {"id": "c", "metadata": {"param": 12}}},
    ]
    rows, examples = w.build_grpo_prompt_dataset(prompts)

    # Columns are trivially typed: the TRL-required prompt + an int index. The rich record is gone.
    assert rows == [
        {"prompt": "p0", "example_idx": 0},
        {"prompt": "p1", "example_idx": 1},
        {"prompt": "p2", "example_idx": 2},
    ]
    # The parallel lookup preserves each original record EXACTLY (no JSON/type coercion).
    assert examples == [p["example"] for p in prompts]
    # reward_fn maps the batch's example_idx column back to the original objects (heterogeneous
    # params survive: 8 -> int, 'gentle' -> str), and a shuffled/repeated batch still resolves.
    batch_idx = [2, 0, 1, 0]
    mapped = [examples[int(i)] for i in batch_idx]
    assert [e["metadata"]["param"] for e in mapped] == [12, 8, "gentle", 8]


def test_grpo_masks_truncated_completions_by_default() -> None:
    """GRPO drops truncated (non-EOS) completions from the loss by default.

    TRL's GRPOConfig default is False (it WOULD train on truncated rollouts); a truncated
    completion is not a real sample from the policy's distribution over finished sequences, so
    including it biases the policy gradient and — on long-completion / multi-turn envs that
    frequently hit the budget — can degrade the model below its SFT start.
    """
    import flash.engine.worker as w

    # No stop_sequences (the common case) -> masking ON.
    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "grpo",
            "environment": {"id": "owner/env"},
            "train": {},
        }
    )
    assert w.grpo_mask_truncated_completions(spec.train) is True
    # Defensive: a None train spec (no JOB_SPEC) still resolves to the safe default (ON).
    assert w.grpo_mask_truncated_completions(None) is True


def test_grpo_truncation_masking_off_when_stop_sequences_set() -> None:
    """With stop_sequences, vLLM strips the stop string so a normally-terminated completion does
    NOT end in EOS — TRL's "last token != EOS" truncation check would then flag (and mask) every
    completion, so the run would learn nothing. Gate the flag OFF in that case."""
    import flash.engine.worker as w

    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "grpo",
            "environment": {"id": "owner/env"},
            "train": {"stop_sequences": ["</answer>"]},
        }
    )
    assert spec.train.stop_sequences == ("</answer>",)
    assert w.grpo_mask_truncated_completions(spec.train) is False


def test_run_rl_wires_mask_truncated_completions_to_the_gating_helper() -> None:
    """run_rl's grpo_kwargs literal must pin mask_truncated_completions to the
    grpo_mask_truncated_completions(...) helper. The helper's True/False logic is covered above, but
    nothing else guarantees run_rl actually USES it: drop that dict entry and the helper tests still
    pass while GRPO silently reverts to TRL's footgun default (False). Assert the wiring on run_rl's
    AST (not a source substring) so it survives reformatting/quote changes — mirrors
    tests/test_flash_worker.py's _train_body AST checks."""
    import ast
    import inspect

    import flash.engine.worker as w

    tree = ast.parse(inspect.getsource(w.run_rl))

    # Accept plain OR annotated assignment (`grpo_kwargs = {...}` / `grpo_kwargs: dict = {...}`) so
    # the test stays focused on the wiring invariant, not the exact assignment node type.
    def _grpo_dict(node):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            return None
        if isinstance(value, ast.Dict) and any(
            isinstance(t, ast.Name) and t.id == "grpo_kwargs" for t in targets
        ):
            return value
        return None

    grpo_dict = next((d for node in ast.walk(tree) if (d := _grpo_dict(node)) is not None), None)
    assert grpo_dict is not None, "run_rl no longer builds a grpo_kwargs dict literal"

    value = next(
        (
            v
            for k, v in zip(grpo_dict.keys, grpo_dict.values, strict=True)
            if isinstance(k, ast.Constant) and k.value == "mask_truncated_completions"
        ),
        None,
    )
    assert value is not None, (
        "run_rl's grpo_kwargs no longer pins mask_truncated_completions -> GRPO would silently "
        "revert to TRL's footgun default (False) and train on truncated rollouts"
    )
    # Must be wired to the helper (so stop_sequences gating applies), not a bare True/False literal.
    # Accept a bare name OR a qualified call (`grpo_mask_truncated_completions(...)` /
    # `worker.grpo_mask_truncated_completions(...)`) — only the called name is the invariant.
    func = value.func if isinstance(value, ast.Call) else None
    called = (
        func.id
        if isinstance(func, ast.Name)
        else func.attr
        if isinstance(func, ast.Attribute)
        else None
    )
    assert called == "grpo_mask_truncated_completions", (
        "mask_truncated_completions must be wired to grpo_mask_truncated_completions(...) so the "
        "stop_sequences gating is honored"
    )


def test_run_rl_threads_prompt_opened_thinking_to_grading_and_penalty() -> None:
    """run_rl must forward the computed _prompt_opens_thinking flag to thinking helpers.

    The leaf-helper tests above cover the flag's True/False logic, but nothing else guarantees
    run_rl actually PASSES it: drop a keyword and the helper tests stay green while production
    silently reverts to the pre-fix no-op (the thinking-length penalty and extracted scorer fields
    are wrong, and a tag-less reasoning ramble is graded as the answer — PR #281). Assert on run_rl's
    AST so it survives reformatting — mirrors test_run_rl_wires_mask_truncated_completions_* above.
    """
    import ast
    import inspect

    import flash.engine.worker as w

    tree = ast.parse(inspect.getsource(w.run_rl))

    def _forwards_flag(call: ast.Call, fname: str) -> bool:
        # The called name, whether bare (think_token_count(...)) or qualified (_w.graded_text(...)).
        func = call.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else None
        )
        if name != fname:
            return False
        # ...passing prompt_opened_thinking=_prompt_opens_thinking (the computed flag, not a literal).
        return any(
            kw.arg == "prompt_opened_thinking"
            and isinstance(kw.value, ast.Name)
            and kw.value.id == "_prompt_opens_thinking"
            for kw in call.keywords
        )

    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    for fname in ("graded_text", "thinking_text", "think_token_count"):
        assert any(_forwards_flag(c, fname) for c in calls), (
            f"run_rl must call {fname}(..., prompt_opened_thinking=_prompt_opens_thinking); without "
            "it the prompt-opened-<think> fix silently no-ops (the length penalty and <think> strip "
            "do nothing on the common enable_thinking=true path — PR #281)"
        )


def test_run_rl_fused_logits_keeps_cap_when_vllm_importance_sampling_runs() -> None:
    """Codex MtcPF: TRL's colocated-vLLM path runs a SEPARATE old_per_token_logps forward for its
    importance-sampling (TIS) correction whenever vllm_importance_sampling_correction is on — TRL's
    default, which run_rl only tunes (mode/clip) and never disables — even at mu==1. That unfused
    [pd, completion, vocab] forward still materializes full logits, so the 6 GB cap must stay; dropping
    it (fused_logits=True) sizes per_device too large and OOMs the IS forward on long completions. So
    the fused_logits gate must ALSO require the vLLM IS forward to be off (``_vllm_is_logprob_forward``),
    resolved BEFORE the sizer and from the correction's effective state + use_vllm. AST-based so it
    survives reformatting (mirrors the resolved-mu wiring test above)."""
    import ast
    import inspect

    import flash.engine.worker as w

    src = inspect.getsource(w.run_rl)
    tree = ast.parse(src)

    def _is_sizer(call: ast.Call) -> bool:
        f = call.func
        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
        return name == "rl_per_device_comps"

    sizer = next((n for n in ast.walk(tree) if isinstance(n, ast.Call) and _is_sizer(n)), None)
    assert sizer is not None, "run_rl no longer calls rl_per_device_comps"
    fused_kw = next((k for k in sizer.keywords if k.arg == "fused_logits"), None)
    assert fused_kw is not None, "rl_per_device_comps call no longer passes fused_logits"

    names = {n.id for n in ast.walk(fused_kw.value) if isinstance(n, ast.Name)}
    assert "_vllm_is_logprob_forward" in names, (
        "fused_logits must keep the 6 GB cap when TRL's vLLM importance-sampling correction runs an "
        "old_per_token_logps forward (on by default even at mu==1) — gate on _vllm_is_logprob_forward"
    )

    gate_line = min(
        (
            t.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for t in node.targets
            if isinstance(t, ast.Name) and t.id == "_vllm_is_logprob_forward"
        ),
        default=None,
    )
    assert gate_line is not None, "run_rl no longer assigns _vllm_is_logprob_forward"
    assert gate_line < sizer.lineno, (
        "_vllm_is_logprob_forward must be resolved BEFORE the rl_per_device_comps call"
    )
    # The gate must reflect BOTH the vLLM rollout path and the correction's effective state — the only
    # vLLM path runs the forward, and the field name is what TRL keys the correction off of.
    assert "use_vllm" in {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}, (
        "the TIS gate must consider use_vllm (only the vLLM rollout path runs the IS forward)"
    )
    assert "vllm_importance_sampling_correction" in src, (
        "run_rl must detect TRL's vllm_importance_sampling_correction state to gate the logits cap"
    )


def test_trl_grpoconfig_truncation_default_is_the_footgun_we_override(tmp_path) -> None:
    """Document the TRL contract the recipe depends on: the field exists and TRL defaults it OFF,
    so the explicit True in run_rl is load-bearing. A TRL rename would silently drop our override,
    so fail loudly here if the field disappears."""
    import dataclasses as dc

    GRPOConfig = pytest.importorskip("trl").GRPOConfig
    fields = {f.name: f for f in dc.fields(GRPOConfig)}
    assert "mask_truncated_completions" in fields, (
        "TRL renamed/removed mask_truncated_completions — update the GRPO recipe in run_rl"
    )
    assert fields["mask_truncated_completions"].default is False
    # The flag composes with the recipe's loss_type without GRPOConfig rejecting the combo
    # (use_cpu/bf16=False so the config validates on a CPU-only CI host). tmp_path gives a unique,
    # writable output_dir (no /tmp collisions between parallel test runs).
    cfg = GRPOConfig(
        output_dir=str(tmp_path),
        loss_type="dr_grpo",
        mask_truncated_completions=True,
        report_to=[],
        bf16=False,
        use_cpu=True,
    )
    assert cfg.mask_truncated_completions is True


def test_build_grpo_prompt_dataset_survives_dataset_from_list() -> None:
    # The actual failure point: Dataset.from_list over the rich records raises ArrowInvalid on the
    # mixed-type column, while the index-based rows construct cleanly.
    Dataset = pytest.importorskip("datasets").Dataset
    ArrowInvalid = pytest.importorskip("pyarrow.lib").ArrowInvalid

    import flash.engine.worker as w

    prompts = [
        {"prompt": "p0", "example": {"metadata": {"param": 8}}},
        {"prompt": "p1", "example": {"metadata": {"param": "gentle"}}},
    ]
    # Old shape (rich record embedded) crashes exactly as observed in run_rl.
    with pytest.raises(ArrowInvalid):
        Dataset.from_list(prompts)

    # The fix's rows build a valid dataset and round-trip the index column.
    rows, examples = w.build_grpo_prompt_dataset(prompts)
    ds = Dataset.from_list(rows)
    assert ds.column_names == ["prompt", "example_idx"]
    assert list(ds["example_idx"]) == [0, 1]
    assert [examples[i]["metadata"]["param"] for i in ds["example_idx"]] == [8, "gentle"]
