"""Datums-parity GRPO knobs + init-from-adapter wiring (CPU-only, no GPU/network).

The SDK ships the GRPO recipe knobs (group_size/temperature/
kl_penalty_coef/thinking_length_penalty_coef) plus the optimizer/batching knobs
(learning_rate/prompts_per_step/max_context_tokens/save_every) in the job spec's ``[train]`` table
(TrainSpec) — NOT ``[environment.params]``, which is forwarded verbatim to the Freesolo
env's ``load_environment`` — and an optional ``train.init_from_adapter``; these tests
cover the pure plumbing the worker uses to honor them (the GPU trainer wiring itself is
exercised by the live smokes).
"""

from __future__ import annotations

import re

import pytest

import flash.engine.worker.model.decoding as worker_decoding
import flash.engine.worker.runtime.state as worker_state
import flash.engine.worker.train.rl.launch.config as worker_grpo_config
from flash.core.grpo import SUPPORTED_GRPO_GROUP_SIZES
from flash.core.spec import JobSpec
from flash.schema import ConfigError, spec_from_dict


class _Tok:
    """Whitespace tokenizer stub: one token per word."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def test_think_token_count_counts_the_think_span() -> None:

    tok = _Tok()
    assert worker_decoding.think_token_count("<think>a b c</think>the answer", tok) == 3
    assert worker_decoding.think_token_count("no reasoning here", tok) == 0
    # an unclosed block (budget exhausted) counts everything after <think>
    assert worker_decoding.think_token_count("pre <think>a b c d", tok) == 4
    assert worker_decoding.think_token_count(None, tok) == 0
    assert worker_decoding.think_token_count("<think></think>x", tok) == 0
    # prompt-opened hybrid thinking: the chat template appended <think> to the PROMPT, so the
    # completion starts mid-reasoning with only the closing </think>. The reasoning is everything
    # before that close (without this the penalty no-ops on the common enable_thinking=true path).
    assert worker_decoding.think_token_count('a b c d</think>{"x": 1}', tok) == 4
    assert worker_decoding.think_token_count("</think>just the answer", tok) == 0
    # case 3: prompt-opened thinking that NEVER closes (ran out of max_completion_tokens) — no tags at all. With
    # prompt_opened_thinking the WHOLE completion is unterminated reasoning and is counted, so the
    # longest rambles can't dodge the penalty; without the flag a tag-less completion is plain text (0).
    assert (
        worker_decoding.think_token_count(
            "rambling on and on forever", tok, prompt_opened_thinking=True
        )
        == 5
    )
    assert worker_decoding.think_token_count("rambling on and on forever", tok) == 0
    # the flag does NOT change a completion that already carries a tag (cases 1/2 still win).
    assert (
        worker_decoding.think_token_count("a b c</think>ans", tok, prompt_opened_thinking=True) == 3
    )
    assert (
        worker_decoding.think_token_count("<think>a b</think>ans", tok, prompt_opened_thinking=True)
        == 2
    )
    assert worker_decoding.think_token_count("", tok, prompt_opened_thinking=True) == 0
    # Case 1 vs 2 is decided by tag ORDER, not presence: a prompt-opened completion that CLOSES its
    # reasoning and then echoes a literal <think> in the answer must count the span up to the FIRST
    # </think> (the reasoning), NOT anchor on the echoed opener (which would count "echo here" = 2).
    assert (
        worker_decoding.think_token_count("a b c d</think>answer with <think> echo here", tok) == 4
    )
    # a self-tagged block followed by an echoed opener still counts only the first real span.
    assert worker_decoding.think_token_count("<think>a b c</think>tail <think> echo", tok) == 3
    # prompt-opened + NEVER closed + an echoed <think>: count the WHOLE completion (it's all
    # unterminated reasoning), not just the text after the echoed opener.
    assert (
        worker_decoding.think_token_count(
            "reason 42 <think> more", tok, prompt_opened_thinking=True
        )
        == 4
    )
    # the same echoed completion WITHOUT the prompt-open signal: the model opened <think> itself
    # (unclosed) -> count after that opener (case: model-opened unclosed).
    assert worker_decoding.think_token_count("reason 42 <think> more", tok) == 1
    # prompt-opened + an echoed <think> BEFORE the first </think>: the prompt pre-opened reasoning, so
    # the span is the WHOLE pre-opened reasoning from the start through the first close
    # ("reason 42 <think> more" = 4) -- NOT just the sliver after the echoed opener (" more" = 1).
    assert (
        worker_decoding.think_token_count(
            "reason 42 <think> more </think> ans", tok, prompt_opened_thinking=True
        )
        == 4
    )
    # the same string WITHOUT the prompt-open signal: the model opened AND closed its own <think>, so
    # only the span between the model's tags counts (" more" = 1) -- case 1.
    assert worker_decoding.think_token_count("reason 42 <think> more </think> ans", tok) == 1


def test_prompt_opens_thinking_detects_preopened_tag() -> None:

    # A hybrid template pre-opens <think> at the end of the generation prompt (no closing tag).
    assert worker_decoding.prompt_opens_thinking("<|im_start|>assistant\n<think>\n") is True
    # An uncurated/non-thinking template appends no <think> -> a tagless completion is a real answer.
    assert worker_decoding.prompt_opens_thinking("<|im_start|>assistant\n") is False
    assert worker_decoding.prompt_opens_thinking("") is False
    assert worker_decoding.prompt_opens_thinking(None) is False
    # A prompt that opened AND closed a <think> (e.g. a few-shot exemplar) is NOT pre-opened.
    assert (
        worker_decoding.prompt_opens_thinking("...<think>example</think>...<|im_start|>assistant\n")
        is False
    )
    # If the LAST think is left open (after an earlier closed one), it IS pre-opened.
    assert (
        worker_decoding.prompt_opens_thinking("<think>ex</think>q<|im_start|>assistant\n<think>\n")
        is True
    )
    # FALSE-POSITIVE guard: a user/system message that merely CONTAINS an unclosed literal <think>
    # must NOT count as pre-opened when the generation suffix didn't actually prefill thinking (the
    # detection anchors on the trailing <think> suffix, not a scan of the whole prompt).
    assert (
        worker_decoding.prompt_opens_thinking("user asked <think> about x<|im_start|>assistant\n")
        is False
    )


def test_graded_text_hides_tagless_prompt_opened_reasoning(monkeypatch) -> None:

    monkeypatch.setattr(worker_state, "THINKING", True)
    # Tagless completion under a prompt-opened <think>: the generation never closed reasoning, so the
    # env must grade NOTHING (scores 0) — not the raw ramble (which a raw-text fallback could reward).
    assert (
        worker_decoding.graded_text("rambling forever no answer", prompt_opened_thinking=True) == ""
    )
    # Without the prompt-opened signal (e.g. an uncurated template that didn't pre-open), the same
    # tagless text is a normal answer and is graded as-is.
    assert (
        worker_decoding.graded_text("the answer is 42", prompt_opened_thinking=False)
        == "the answer is 42"
    )
    # A normally-tagged thinking completion is unaffected: strip to the post-</think> answer.
    assert (
        worker_decoding.graded_text("reasoning...</think>\\boxed{5}", prompt_opened_thinking=True)
        == "\\boxed{5}"
    )
    # Echoed <think> while still unterminated (no </think>): the WHOLE thing is reasoning -> hidden,
    # NOT just the text before the echoed opener (which a raw-text fallback could otherwise reward).
    assert (
        worker_decoding.graded_text("reason 42 <think> still going", prompt_opened_thinking=True)
        == ""
    )
    # THINKING off: no stripping at all, even with the flag set.
    monkeypatch.setattr(worker_state, "THINKING", False)
    assert (
        worker_decoding.graded_text("rambling forever", prompt_opened_thinking=True)
        == "rambling forever"
    )


def test_grpo_overrides_reads_train_knobs(monkeypatch) -> None:

    train_knobs = {
        "group_size": 4,
        "temperature": 0.7,
        "max_completion_tokens": 256,
        "kl_penalty_coef": 0.02,
        "entropy_quantile": 0.2,
        "thinking_length_penalty_coef": 0.001,
    }
    grpo_knobs = {
        **train_knobs,
        "max_tokens": train_knobs["max_completion_tokens"],
    }
    del grpo_knobs["max_completion_tokens"]
    # GRPO knobs live in [train]/TrainSpec, NOT [environment.params].
    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "grpo",
            "environment": {"id": "github:owner/repo@main:env/environment.py"},
            "train": {**train_knobs},
        }
    )
    monkeypatch.setattr(worker_state, "JOB_SPEC", spec)
    assert worker_grpo_config.grpo_overrides() == grpo_knobs
    # A leftover grpo_config in environment.params must NOT be read by the worker.
    poisoned = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "grpo",
            "environment": {
                "id": "github:owner/repo@main:env/environment.py",
                "params": {"grpo_config": grpo_knobs},
            },
            "train": {},
        }
    )
    monkeypatch.setattr(worker_state, "JOB_SPEC", poisoned)
    assert worker_grpo_config.grpo_overrides() == {}
    # only the knobs actually set are returned (a partial set omits the rest)
    monkeypatch.setattr(
        worker_state,
        "JOB_SPEC",
        JobSpec.from_dict(
            {
                "model": "Qwen/Qwen3.5-9B",
                "algorithm": "grpo",
                "environment": {"id": "github:owner/repo@main:env/environment.py"},
                "train": {"group_size": 2},
            }
        ),
    )
    assert worker_grpo_config.grpo_overrides() == {"group_size": 2}
    # no [train] knobs -> empty (recipe defaults apply downstream)
    monkeypatch.setattr(
        worker_state,
        "JOB_SPEC",
        JobSpec.from_dict({"model": "Qwen/Qwen3.5-9B", "algorithm": "grpo"}),
    )
    assert worker_grpo_config.grpo_overrides() == {}
    monkeypatch.setattr(worker_state, "JOB_SPEC", None)
    assert worker_grpo_config.grpo_overrides() == {}


def test_train_grpo_knobs_parse_and_roundtrip() -> None:
    from flash.schema import spec_from_dict

    raw = {
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": "grpo",
        "environment": {"id": "github:owner/repo@main:env/environment.py"},
        "gpu": {},
        "train": {
            "epochs": 1,
            "max_examples": 10,
            "group_size": 4,
            "temperature": 0.7,
            "max_completion_tokens": 256,
            "kl_penalty_coef": 0.02,
            "entropy_quantile": 0.2,
            "thinking_length_penalty_coef": 0.001,
        },
    }
    spec = spec_from_dict(raw, run_id="grpo-x")
    assert spec.train.group_size == 4
    assert spec.train.temperature == 0.7
    assert spec.train.max_completion_tokens == 256
    assert spec.train.kl_penalty_coef == 0.02
    assert spec.train.entropy_quantile == 0.2
    assert spec.train.thinking_length_penalty_coef == 0.001
    # survives the JSON round-trip the worker reconstructs from
    rt = JobSpec.from_dict(spec.to_dict()).train
    assert rt.group_size == 4
    assert rt.entropy_quantile == 0.2
    assert rt.thinking_length_penalty_coef == 0.001
    # GRPO knobs are NOT in environment.params (that goes verbatim to load_environment)
    assert spec.environment.params == {}


def test_entropy_knobs_parse_from_toml_roundtrip_and_override(tmp_path, monkeypatch) -> None:
    from flash.schema import spec_and_train_keys_from_file

    config = tmp_path / "grpo.toml"
    config.write_text(
        "\n".join(
            [
                'model = "Qwen/Qwen3.5-9B"',
                'algorithm = "grpo"',
                "",
                "[environment]",
                'id = "owner/project/env"',
                "",
                "[train]",
                "entropy_quantile = 0.2",
            ]
        )
    )

    spec = spec_and_train_keys_from_file(str(config), run_id="entropy")[0]
    assert spec.train.entropy_quantile == 0.2

    roundtripped = JobSpec.from_dict(spec.to_dict())
    assert roundtripped.train.entropy_quantile == 0.2

    monkeypatch.setattr(worker_state, "JOB_SPEC", roundtripped)
    assert worker_grpo_config.grpo_overrides() == {"entropy_quantile": 0.2}


def test_opt_int_float_reject_bools() -> None:
    """A JSON boolean must NOT silently coerce to a numeric train knob: bool is an int
    subclass in Python, so ``int(True)`` would become 1. JobSpec.from_dict (via
    opt_int/opt_float) rejects it, matching schema._opt_num."""
    import pytest

    from flash.core.spec_persistence import opt_float, opt_int

    for bad in (True, False):
        with pytest.raises(TypeError):
            opt_int(bad)
        with pytest.raises(TypeError):
            opt_float(bad)

    # Genuine numbers (and None) still parse.
    assert opt_int(None) is None
    assert opt_int(4) == 4
    assert opt_float(None) is None
    assert opt_float(0.7) == 0.7

    # A bool train knob propagates through JobSpec.from_dict as an error, not a 0/1 coercion.
    with pytest.raises(TypeError):
        JobSpec.from_dict(
            {
                "model": "Qwen/Qwen3.5-9B",
                "algorithm": "grpo",
                "environment": {"id": "github:owner/repo@main:env/environment.py"},
                "train": {"epochs": 1, "max_examples": 10, "group_size": True},
            }
        )


def _spec_raw(ref: str) -> dict:
    return {
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": "grpo",
        "environment": {"id": "github:owner/repo@main:env/environment.py"},
        "gpu": {},
        "train": {"epochs": 1, "max_examples": 10, "init_from_adapter": ref},
    }


def test_init_from_adapter_parses_and_roundtrips() -> None:
    """The canonical final checkpoint survives the JSON round-trip."""
    spec = spec_from_dict(_spec_raw("run-x/final"), run_id="grpo-x")
    assert spec.train.init_from_adapter == "run-x/final"
    assert JobSpec.from_dict(spec.to_dict()).train.init_from_adapter == "run-x/final"
    # absent -> empty string (train fresh from base)
    raw = _spec_raw("run-x/final")
    raw["train"].pop("init_from_adapter")
    assert spec_from_dict(raw, run_id="grpo-y").train.init_from_adapter == ""


def test_init_from_adapter_accepts_checkpoint_step_ref() -> None:
    """`<run_id>/step-N` (exactly what `flash runs checkpoint` lists) warm-starts from that saved
    step instead of the run-level adapter — e.g. continuing GRPO from the best checkpoint."""
    spec = spec_from_dict(_spec_raw("run-x/step-40"), run_id="grpo-x")
    assert spec.train.init_from_adapter == "run-x/step-40"
    assert JobSpec.from_dict(spec.to_dict()).train.init_from_adapter == "run-x/step-40"


@pytest.mark.parametrize(
    "bad_ref",
    [
        "Freesolo-Co/flashrun-run-x:sft/run-x",  # legacy long storage form: no longer accepted
        "Freesolo-Co/flashrun-run-x:rl/run-x/checkpoints/step-40",
        "run-x/run-x/step-40",  # doubled run_id: only ONE run_id segment, then optional /step-N
        "run-x/step-40/step-1",  # more than one /step-N segment
        "run-x/step-",  # no step number
        "run-x/step-1111111111111111111",  # too many digits to be a bounded step
        "run-x/step-4/adapter",  # trailing path
        "run-x/checkpoints/step-4",  # storage-layout spelling
        "run-x/../other",  # traversal
        "run x/step-4",  # bad run id
    ],
)
def test_init_from_adapter_rejects_non_short_refs(bad_ref: str) -> None:
    with pytest.raises(ConfigError, match="run_id"):
        spec_from_dict(_spec_raw(bad_ref), run_id="grpo-x")


def test_init_from_adapter_accepts_only_the_two_canonical_shapes() -> None:
    """Pin warm starts to permanent final or saved-step checkpoint IDs."""
    for good in ("run-x/final", "run-x/step-40"):
        assert spec_from_dict(_spec_raw(good), run_id="grpo-x").train.init_from_adapter == good
    # Anything with a second path segment other than /final or a single /step-N is rejected.
    for bad in (
        "run-x",
        "run-x/run-x/step-40",
        "run-x/step-40/step-1",
        "Freesolo-Co/repo:sft/run-x",
    ):
        with pytest.raises(ConfigError, match="run_id"):
            spec_from_dict(_spec_raw(bad), run_id="grpo-x")


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
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": "grpo",
        "environment": {"id": "github:owner/repo@main:env/environment.py"},
        "gpu": {},
        "train": {
            "epochs": 1,
            "max_examples": 10,
            "init_from_adapter": bad_ref,
        },
    }
    with pytest.raises(ConfigError, match=r"train\.init_from_adapter must be a string"):
        spec_from_dict(raw, run_id="grpo-x")


def test_hf_repo_is_managed_not_user_set() -> None:
    # [train] hf_repo is the platform-managed per-run HF artifact repo: the control plane assigns
    # it server-side at submit (see runner_submit.submit_job). It is not a user config key -- the schema
    # rejects a user-supplied value outright rather than silently ignoring it.
    raw = {
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": "grpo",
        "environment": {"id": "github:owner/repo@main:env/environment.py"},
        "train": {"epochs": 1, "max_examples": 10},
    }
    # absent -> fine (not a user key); left blank for the control plane to assign
    spec = spec_from_dict(raw, run_id="hf-x")
    assert spec.train.hf_repo == ""
    assert JobSpec.from_dict(spec.to_dict()).train.hf_repo == ""
    # user-supplied -> rejected as an unknown [train] key (managed, not user-authorable)
    with pytest.raises(ConfigError, match="hf_repo"):
        spec_from_dict(
            {
                **raw,
                "train": {"epochs": 1, "max_examples": 10, "hf_repo": "someone-else/their-repo"},
            },
            run_id="hf-y",
        )


def test_optimizer_and_batching_knobs_roundtrip() -> None:
    # The SDK's SftConfig/GrpoConfig optimizer/batching knobs must survive schema
    # (server validation) AND the worker's JobSpec.from_dict, or the worker would silently
    # train with recipe defaults while W&B reports the user's values.
    raw = {
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": "grpo",
        "environment": {"id": "github:owner/repo@main:env/environment.py"},
        "gpu": {},
        "train": {
            "learning_rate": 3e-5,
            "prompts_per_step": 16,
            "max_context_tokens": 2048,
            "save_every": 5,
            "max_completion_tokens": 512,
            "stop_sequences": ["</answer>", "\n\n"],
        },
    }
    spec = spec_from_dict(raw, run_id="grpo-z")
    for s in (spec, JobSpec.from_dict(spec.to_dict())):  # server parse + worker re-parse
        assert s.train.learning_rate == 3e-5
        assert s.train.prompts_per_step == 16
        assert s.train.max_context_tokens == 2048
        assert s.train.save_every == 5
        assert s.train.max_completion_tokens == 512
        assert s.train.stop_sequences == ("</answer>", "\n\n")
    # omitted optimizer knobs stay None so the worker applies its recipe defaults
    bare = spec_from_dict({**raw, "train": {"max_examples": 8}}, run_id="grpo-w")
    assert bare.train.learning_rate is None
    assert bare.train.prompts_per_step is None
    assert bare.train.stop_sequences == ()
    # a bare-string stop_sequences is ONE stop, never split into characters
    one = spec_from_dict(
        {**raw, "train": {"max_examples": 8, "stop_sequences": "</answer>"}},
        run_id="grpo-s",
    )
    assert one.train.stop_sequences == ("</answer>",)
    assert JobSpec.from_dict(one.to_dict()).train.stop_sequences == ("</answer>",)
    # an empty string means "no stop configured" -> (), not ("",); empty list entries drop
    empty = spec_from_dict(
        {**raw, "train": {"max_examples": 8, "stop_sequences": ""}},
        run_id="grpo-e",
    )
    assert empty.train.stop_sequences == ()
    dropped = spec_from_dict(
        {**raw, "train": {"max_examples": 8, "stop_sequences": ["x", ""]}},
        run_id="grpo-d",
    )
    assert dropped.train.stop_sequences == ("x",)


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


def test_optimizer_knob_validation_rejects_bad_values() -> None:
    # schema is the server's 400 layer: nonsensical/malformed knobs must raise
    # ConfigError at parse time, not TypeError (500) or a silently-misbehaving worker.
    from flash.schema import ConfigError

    base = {
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": "grpo",
        "environment": {"id": "github:owner/repo@main:env/environment.py"},
        "gpu": {},
    }
    bad_cases = [
        # the batching knob for a rollout algorithm is prompts_per_step; spelling it batch_size
        # here would raise for being sft-only and would pass even with range checking removed.
        {"prompts_per_step": 0},  # must be >= 1
        {"prompts_per_step": -4},
        {"max_context_tokens": 0},
        {"save_every": 0},
        {"group_size": 0},
        {"learning_rate": 0},  # must be > 0
        {"learning_rate": -1e-5},
        {"temperature": -0.1},  # must be >= 0
        {"kl_penalty_coef": -1},
        {"entropy_quantile": -0.01},
        {"entropy_quantile": 1.01},
        {"prompts_per_step": 1.5},  # non-integer
        {"prompts_per_step": "16"},  # wrong type (string)
        {"learning_rate": [1]},  # wrong type (list) -> 400, not a 500 TypeError
        {"stop_sequences": {"a": 1}},  # dict not allowed
        {"stop_sequences": [1, 2]},  # non-string entries
        {"learning_rate": float("nan")},  # non-finite -> 400, not a silent NaN to the optimizer
        {"learning_rate": float("inf")},
        {"temperature": float("inf")},
        {"prompts_per_step": float("inf")},  # int knob: 400, not OverflowError(500) from int(inf)
        {"max_completion_tokens": float("nan")},
    ]
    for bad in bad_cases:
        with pytest.raises(ConfigError):
            spec_from_dict({**base, "train": {**bad}}, run_id="bad")


def test_grpo_rejects_single_generation_group_before_paid_worker() -> None:
    """group-relative advantages need two generations per prompt; reject group_size=1 at parse
    time rather than on the paid GPU worker."""
    from flash.schema import ConfigError

    base = {
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": "grpo",
        "environment": {"id": "github:owner/repo@main:env/environment.py"},
    }

    # derived from the contract, not spelled out: widening `SUPPORTED_GRPO_GROUP_SIZES` must not
    # leave this asserting wording that only held for the old tuple.
    allowed = ", ".join(str(value) for value in SUPPORTED_GRPO_GROUP_SIZES)
    with pytest.raises(
        ConfigError, match=rf"group_size.*{re.escape(f'one of {{{allowed}}}')}.*GRPO"
    ):
        spec_from_dict(
            {**base, "train": {"epochs": 1, "max_examples": 8, "group_size": 1}}, run_id="bad"
        )

    spec = spec_from_dict(
        {**base, "train": {"epochs": 1, "max_examples": 8, "group_size": 2}}, run_id="ok"
    )
    assert spec.train.group_size == 2


def test_opd_allows_single_generation_group() -> None:
    """OPD distils individual completions, so group_size=1 remains a valid smoke/test setting."""
    spec = spec_from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "opd",
            "environment": {"id": "github:owner/repo@main:env/environment.py"},
            "train": {"epochs": 1, "max_examples": 8, "group_size": 1},
        },
        run_id="opd-ok",
    )
    assert spec.train.group_size == 1


def test_epochs_reject_non_integer_at_parse() -> None:
    # epochs must run through _train_int like every other integer knob: a
    # non-integer (e.g. epochs=1.5) must 400 at parse time, not slip through to the
    # worker and crash int("1.5") AFTER a paid GPU is provisioned.
    from flash.schema import ConfigError

    base = {
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": "sft",
        "environment": {"id": "github:owner/repo@main:env/environment.py"},
        "gpu": {},
    }
    for bad in ({"epochs": 2.5}, {"epochs": -1}):
        with pytest.raises(ConfigError):
            spec_from_dict({**base, "train": {"max_examples": 8, **bad}}, run_id="bad")

    # Genuine integers still parse and round-trip.
    spec = spec_from_dict(
        {**base, "train": {"max_examples": 8, "epochs": 3}},
        run_id="ok",
    )
    assert spec.train.epochs == 3


def test_build_grpo_prompt_dataset_keeps_columns_arrow_safe() -> None:
    # Reproduces the ifeval-lite crash: a VALID env whose per-row metadata.param is an int for some
    # rows and a str for others. Embedding the rich record in the dataset makes PyArrow infer one
    # column type across all rows and crash; build_grpo_prompt_dataset stores a stable int index
    # instead, and reward_fn maps it back to the original record.

    prompts = [
        {"prompt": "p0", "example": {"id": "a", "metadata": {"param": 8}}},  # int param
        {"prompt": "p1", "example": {"id": "b", "metadata": {"param": "gentle"}}},  # str param
        {"prompt": "p2", "example": {"id": "c", "metadata": {"param": 12}}},
    ]
    rows, examples = worker_grpo_config.build_grpo_prompt_dataset(prompts)

    # Columns are trivially typed: the prompt + an int index. The rich record is gone.
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

    verl's rollout default would train on truncated rollouts; a truncated
    completion is not a real sample from the policy's distribution over finished sequences, so
    including it biases the policy gradient and — on long-completion / multi-turn envs that
    frequently hit the budget — can degrade the model below its SFT start.
    """

    # No stop_sequences (the common case) -> masking ON.
    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "grpo",
            "environment": {"id": "owner/project/env"},
            "train": {},
        }
    )
    assert worker_grpo_config.grpo_mask_truncated_completions(spec.train) is True
    # Defensive: a None train spec (no JOB_SPEC) still resolves to the safe default (ON).
    assert worker_grpo_config.grpo_mask_truncated_completions(None) is True


def test_grpo_truncation_masking_off_when_stop_sequences_set() -> None:
    """With stop_sequences, vLLM strips the stop string so a normally-terminated completion does
    NOT end in EOS -- the "last token != EOS" truncation check would then flag (and mask) every
    completion, so the run would learn nothing. Gate the flag OFF in that case."""

    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "grpo",
            "environment": {"id": "owner/project/env"},
            "train": {"stop_sequences": ["</answer>"]},
        }
    )
    assert spec.train.stop_sequences == ("</answer>",)
    assert worker_grpo_config.grpo_mask_truncated_completions(spec.train) is False


def test_build_grpo_prompt_dataset_survives_dataset_from_list() -> None:
    # The actual failure point: Dataset.from_list over the rich records raises ArrowInvalid on the
    # mixed-type column, while the index-based rows construct cleanly.
    Dataset = pytest.importorskip("datasets").Dataset
    ArrowInvalid = pytest.importorskip("pyarrow.lib").ArrowInvalid

    prompts = [
        {"prompt": "p0", "example": {"metadata": {"param": 8}}},
        {"prompt": "p1", "example": {"metadata": {"param": "gentle"}}},
    ]
    # Old shape (rich record embedded) crashes exactly as observed in run_rl.
    with pytest.raises(ArrowInvalid):
        Dataset.from_list(prompts)

    # The fix's rows build a valid dataset and round-trip the index column.
    rows, examples = worker_grpo_config.build_grpo_prompt_dataset(prompts)
    ds = Dataset.from_list(rows)
    assert ds.column_names == ["prompt", "example_idx"]
    assert list(ds["example_idx"]) == [0, 1]
    assert [examples[i]["metadata"]["param"] for i in ds["example_idx"]] == [8, "gentle"]


def test_grpo_rejects_prompt_budget_at_parse_time_before_provisioning() -> None:
    """max_context_tokens <= the resolved completion budget leaves no prompt room, and the grpo
    worker only discovers that in `_resolve_sequence_lengths` ("engine length leaves no room for
    the completion") after the GPU is provisioned and billed. Reject it at spec-parse time, with
    the SAME completion resolver the worker uses so the two can never disagree."""
    from flash.engine.plan.recipe import RECIPE

    def _spec(train_extra, thinking=False):
        return spec_from_dict(
            {
                "model": "Qwen/Qwen3.5-9B",
                "algorithm": "grpo",
                "thinking": thinking,
                "environment": {"id": "github:owner/repo@main:env/environment.py"},
                "train": {"epochs": 1, "max_examples": 5, **train_extra},
            },
            run_id="grpo-budget",
        )

    # room left after an explicit max_completion_tokens -> ok.
    _spec({"max_context_tokens": 2048, "max_completion_tokens": 512})
    # exactly one prompt token is the boundary the worker accepts (prompt_budget > 0).
    _spec({"max_context_tokens": 513, "max_completion_tokens": 512})
    # no prompt budget at all -> the worker would raise, so reject at parse.
    with pytest.raises(ConfigError, match="prompt budget"):
        _spec({"max_context_tokens": 512, "max_completion_tokens": 512})
    with pytest.raises(ConfigError, match="prompt budget"):
        _spec({"max_context_tokens": 400, "max_completion_tokens": 512})
    # max_completion_tokens omitted -> the RL recipe default, per thinking, exactly as the worker
    # resolves it; NOT the opd recipe's.
    _spec({"max_context_tokens": RECIPE.rl.max_completion_len + 1})
    with pytest.raises(ConfigError, match="prompt budget"):
        _spec({"max_context_tokens": RECIPE.rl.max_completion_len})
    _spec({"max_context_tokens": RECIPE.rl.max_completion_len_thinking + 1}, thinking=True)
    with pytest.raises(ConfigError, match="prompt budget"):
        _spec({"max_context_tokens": RECIPE.rl.max_completion_len_thinking}, thinking=True)
    # an omitted context is the recipe-sized run, which always fits -> nothing to check.
    _spec({})


def test_grpo_prompt_budget_guard_matches_the_worker_resolver() -> None:
    """The submit guard must never be stricter than the worker: both resolve the completion length
    through `grpo_completion_len`, and the worker additionally clamps the context down to the model
    architecture, which can only shrink the prompt budget it checks."""
    import inspect

    from flash import schema
    from flash.engine.plan.recipe import RECIPE
    from flash.engine.plan.vram import grpo_completion_len
    from flash.engine.worker.train.rl.launch import inputs

    assert grpo_completion_len(None, False) == RECIPE.rl.max_completion_len
    assert grpo_completion_len(None, True) == RECIPE.rl.max_completion_len_thinking
    assert grpo_completion_len(4096, True) == 4096
    # a zero/unset knob falls back to the recipe on both sides.
    assert grpo_completion_len(0, False) == RECIPE.rl.max_completion_len

    # both enforcement sites must read the shared helper, not re-derive the recipe inline.
    assert "grpo_completion_len" in inspect.getsource(inputs._resolve_sequence_lengths)
    assert "grpo_completion_len" in inspect.getsource(schema._validate_on_policy_prompt_budget)
