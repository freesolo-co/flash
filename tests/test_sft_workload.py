from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

import pytest

from flash.core.spec import EnvironmentSpec, JobSpec, TrainSpec
from flash.engine.profiling.sft_workload import prepare_sft_workload, sft_tokens_for_updates
from flash.engine.profiling.workload_profile import (
    sft_profile_input_digest,
    unpacked_batch_warning,
)


class FakeTokenizer:
    eos_token = "|"
    eos_token_id = 2
    pad_token = None
    pad_token_id = 0
    all_special_ids: ClassVar[list[int]] = [0, 1, 2]

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
        **_kwargs,
    ):
        assert not tokenize
        assert isinstance(enable_thinking, bool)
        text = "".join(str(message.get("content") or "") for message in messages)
        return text + (">" if add_generation_prompt else "")

    def __call__(self, texts, *, truncation=False, max_length=None):
        if isinstance(texts, str):
            texts = [texts]
        ids = [
            [self.eos_token_id if char == self.eos_token else 3 + ord(char) % 89 for char in text]
            for text in texts
        ]
        if not truncation:
            # the cap must not apply when the caller did not ask for it: the untruncated encode
            # is how a truncated row reports its real size instead of the cap.
            return {"input_ids": ids}
        assert max_length is not None, "truncation=True requires an explicit max_length"
        return {"input_ids": [row[:max_length] for row in ids]}


class FakeEnvironment:
    multi_turn = False
    package_root = None

    def __init__(self):
        self._rows = [
            {"prompt": "one", "answer": "alpha"},
            {"prompt": "two", "answer": "beta"},
            {"prompt": "three", "answer": ""},
            {"prompt": "four", "answer": "outside-prefix"},
        ]

    def dataset(self):
        return list(self._rows)

    def prompt_messages(self, row):
        return [{"role": "user", "content": row["prompt"]}]

    def sft_completion(self, row):
        return [{"role": "assistant", "content": row["answer"]}]


def _spec() -> JobSpec:
    spec = JobSpec(
        model="test/model",
        model_revision="a" * 40,
        algorithm="sft",
        environment=EnvironmentSpec(
            id="team/example",
            resolved_sha="b" * 40,
        ),
        train=TrainSpec(
            epochs=2,
            batch_size=2,
            max_context_tokens=32,
            max_examples=3,
        ),
        seed=7,
    )
    digest = sft_profile_input_digest(
        spec,
        tokenizer_revision=spec.model_revision,
        producer_version="1.2.3",
    )
    return replace(
        spec,
        workload_profile_input_digest=digest,
        workload_profile_producer_version="1.2.3",
    )


def _prepare(spec: JobSpec, *, packed: bool = True):
    return prepare_sft_workload(
        spec,
        FakeEnvironment(),
        tokenizer_loader=lambda _model, _revision: FakeTokenizer(),
        producer_version="1.2.3",
        packing_support=lambda _model, _revision: ("pure-attention", packed),
    )


def test_exact_sft_workload_packs_retained_prefix_rows_deterministically() -> None:
    first = _prepare(_spec())
    second = _prepare(_spec())

    assert first.rows == second.rows
    assert first.profile == second.profile
    assert first.profile.source_examples == 4
    assert first.profile.selected_examples == 3
    assert first.profile.retained_examples == 2
    assert first.profile.dropped_examples == 1
    assert first.profile.packing_mode == "packed"
    assert first.profile.architecture_mode == "pure-attention"
    assert first.profile.packed_blocks == 1
    assert first.coerced_singleturn_targets == 0
    assert len(first.rows) == 2
    assert all(len(row["loss_mask"]) == len(row["input_ids"]) for row in first.rows)
    # exactly the columns the parquet schema declares. a row carrying an extra key would be dropped
    # on the way to verl, so a profile measured from it would describe work that never ran.
    assert all(
        set(row) == {"input_ids", "loss_mask", "images", "multimodal_inputs"} for row in first.rows
    )
    assert first.profile.supervised_tokens_per_epoch == sum(
        sum(row["loss_mask"]) for row in first.rows
    )
    assert first.profile.authoritative_steps == 2
    assert first.profile.authoritative_real_tokens == 2 * first.profile.real_tokens_per_epoch
    assert first.profile.authoritative_compute_tokens == first.profile.authoritative_real_tokens


def test_exact_unpacked_mode_trains_one_example_per_update() -> None:
    prepared = _prepare(_spec(), packed=False)
    packed = _prepare(_spec(), packed=True)

    assert prepared.profile.packing_mode == "exact-unpacked"
    assert prepared.profile.examples_per_update == 1
    assert prepared.profile.packed_blocks == 2
    assert prepared.profile.derived_steps == 4
    # the mode selects how examples are grouped into an update, not a different token layout: same
    # rows, same tokens, more updates. isolation here is bought by giving each example its own
    # update rather than by boundary metadata, because verl packs unconditionally and its fsdp
    # engine never sends the gdn reset kwargs -- a batch of two would let the second example train
    # on the first's carried state. that longer horizon is what the quote must price.
    assert prepared.rows == packed.rows
    assert prepared.profile.real_tokens_per_epoch == packed.profile.real_tokens_per_epoch
    assert prepared.profile.derived_steps > packed.profile.derived_steps


def test_unpacked_run_warns_that_the_configured_batch_size_is_ignored(capsys) -> None:
    """An unpacked prepare announces the one-example-per-update override on stderr."""
    _prepare(_spec(), packed=False)
    err = capsys.readouterr().err

    assert "sequence packing is OFF" in err
    assert "the configured batch_size 2 no longer groups examples into an update" in err
    assert "learning_rate" in err


def test_packed_run_does_not_warn_about_the_batch_size(capsys) -> None:
    packed = _prepare(_spec(), packed=True)
    err = capsys.readouterr().err

    assert packed.profile.examples_per_update == 2
    assert "sequence packing is OFF" not in err


@pytest.mark.parametrize(
    ("architecture_mode", "expected"),
    [
        ("multimodal", "multimodal"),
        ("gdn-hybrid", "linear-attention recurrence"),
        ("unsupported", "no boundary-safe packing path"),
    ],
)
def test_unpacked_warning_names_the_reason_the_packing_decision_froze(
    architecture_mode: str, expected: str
) -> None:
    """The reason comes from the architecture label the packing decision recorded on the profile."""
    message = unpacked_batch_warning(
        packing_mode="exact-unpacked",
        architecture_mode=architecture_mode,
        examples_per_update=1,
        configured_batch_size=32,
    )

    assert message is not None
    assert expected in message


def test_unpacked_warning_names_the_recipe_default_when_batch_size_was_omitted() -> None:
    """An omitted batch_size is the recipe's 32, which is the batch packing discarded: the cli
    reads it straight off the spec, so the default has to resolve here or the number goes missing.
    """
    from flash.engine.plan.recipe import RECIPE

    message = unpacked_batch_warning(
        packing_mode="exact-unpacked",
        architecture_mode="multimodal",
        examples_per_update=1,
        configured_batch_size=None,
    )

    assert message is not None
    # an omitted batch_size is reported as the default, not as something the user configured.
    assert f"the default batch_size {RECIPE.sft.effective_batch}" in message


def test_unpacked_warning_is_silent_when_the_authored_batch_was_already_one() -> None:
    """Nothing was overridden, so there is nothing to warn about."""
    assert (
        unpacked_batch_warning(
            packing_mode="exact-unpacked",
            architecture_mode="multimodal",
            examples_per_update=1,
            configured_batch_size=1,
        )
        is None
    )


@pytest.mark.parametrize("authored", [True, False])
def test_worker_unpacked_warning_names_the_batch_source_truthfully(capsys, authored: bool) -> None:
    """The worker path resolves batch_size to the recipe default before it warns, so handing the
    resolved number to the helper made an omitted knob read as one the user configured -- the
    opposite of what the cli says about the same run.
    """
    from flash.engine.plan.recipe import RECIPE

    spec = _spec()
    spec = _rebuild_digest(
        replace(spec, train=replace(spec.train, batch_size=2 if authored else None))
    )

    _prepare(spec, packed=False)

    warning = capsys.readouterr().err
    expected = (
        "the configured batch_size 2"
        if authored
        else f"the default batch_size {RECIPE.sft.effective_batch}"
    )
    assert expected in warning


def _rebuild_digest(spec: JobSpec) -> JobSpec:
    digest = sft_profile_input_digest(
        spec,
        tokenizer_revision=spec.model_revision,
        producer_version="1.2.3",
    )
    return replace(
        spec,
        workload_profile_input_digest=digest,
        workload_profile_producer_version="1.2.3",
    )


def _spec_with_max_steps(max_steps: int) -> JobSpec:
    return _rebuild_digest(replace(_spec(), train=replace(_spec().train, max_steps=max_steps)))


def _uneven_spec(max_steps: int | None = None) -> JobSpec:
    """Three retained rows at batch 2, so each epoch ends on a half-full batch."""
    base = _spec()
    return _rebuild_digest(
        replace(base, train=replace(base.train, max_examples=4, max_steps=max_steps))
    )


def test_max_steps_replaces_the_derived_packed_horizon() -> None:
    prepared = _prepare(_spec_with_max_steps(5))

    assert prepared.profile.derived_steps == 2
    assert prepared.profile.authoritative_steps == 5


def test_derived_horizon_counts_one_update_per_block_per_epoch() -> None:
    """Blocks are already batches, so the horizon must not divide by the batch a second time."""
    prepared = _prepare(_uneven_spec())

    assert prepared.profile.retained_examples == 3
    assert prepared.profile.examples_per_update == 2
    assert prepared.profile.packed_blocks == 2
    assert prepared.profile.derived_steps == 4


def test_partial_horizon_tokens_count_the_exact_batches_verl_consumes() -> None:
    """A truncated horizon must sum the rows it actually reaches, not scale an epoch total."""
    prepared = _prepare(_uneven_spec(max_steps=1))
    first_batch = sum(len(row["input_ids"]) for row in prepared.rows[:2])

    assert prepared.profile.authoritative_steps == 1
    assert prepared.profile.authoritative_real_tokens == first_batch
    assert prepared.profile.authoritative_real_tokens < prepared.profile.real_tokens_per_epoch


def test_horizon_wraps_through_the_same_fixed_row_order() -> None:
    """Update 3 revisits batch 1, so its tokens repeat that batch rather than averaging the epoch."""
    prepared = _prepare(_uneven_spec(max_steps=3))
    epoch_tokens = prepared.profile.real_tokens_per_epoch
    first_batch = sum(len(row["input_ids"]) for row in prepared.rows[:2])

    assert prepared.profile.authoritative_real_tokens == epoch_tokens + first_batch


def test_trailing_partial_batch_is_kept_not_dropped() -> None:
    """verl runs with drop_last disabled, so the odd last row must still be billed."""
    prepared = _prepare(_uneven_spec(max_steps=2))
    epoch_tokens = sum(len(row["input_ids"]) for row in prepared.rows)

    assert prepared.profile.authoritative_real_tokens == epoch_tokens
    assert prepared.profile.real_tokens_per_epoch == epoch_tokens


def test_zero_updates_consume_no_tokens() -> None:
    prepared = _prepare(_spec())

    assert (
        sft_tokens_for_updates(
            prepared.rows,
            examples_per_update=2,
            updates=0,
            field="input_ids",
        )
        == 0
    )
    with pytest.raises(ValueError, match="updates must be"):
        sft_tokens_for_updates(
            prepared.rows,
            examples_per_update=2,
            updates=-1,
            field="input_ids",
        )


def test_probe_failure_fails_the_profile_instead_of_freezing_a_wrong_label(monkeypatch) -> None:
    """A transient config-fetch failure must not mint an ``unsupported`` architecture label.

    The label is frozen into the profile and compared byte-for-byte by the training worker. If a
    hub blip could answer "unsupported" here, a later re-derivation that reached the config would
    say "gdn-hybrid" and every training run built on that profile would die with a false
    "sft workload changed after the quote was frozen" -- with no takeover path, because the profile
    itself stays ``done``.
    """
    from flash.engine.profiling import sft_workload

    def _boom(model_id, revision=""):
        raise OSError("hub read timed out")

    monkeypatch.setattr(sft_workload, "probe_is_pure_attention", _boom)

    with pytest.raises(RuntimeError, match="could not resolve the model config"):
        prepare_sft_workload(
            _spec(),
            FakeEnvironment(),
            tokenizer_loader=lambda _model, _revision: FakeTokenizer(),
            producer_version="1.2.3",
        )


def test_gdn_probe_failure_also_fails_closed(monkeypatch) -> None:
    """The second probe carries the same risk: False-on-error would freeze ``unsupported``."""
    from flash.engine.profiling import sft_workload

    monkeypatch.setattr(
        sft_workload, "probe_is_pure_attention", lambda model_id, revision="": False
    )

    def _boom(model_id, revision=""):
        raise OSError("hub read timed out")

    monkeypatch.setattr(sft_workload, "probe_is_gdn_hybrid", _boom)

    with pytest.raises(RuntimeError, match="could not resolve the model config"):
        prepare_sft_workload(
            _spec(),
            FakeEnvironment(),
            tokenizer_loader=lambda _model, _revision: FakeTokenizer(),
            producer_version="1.2.3",
        )


class LongRowEnvironment(FakeEnvironment):
    """One row far past any cap under test, one comfortably inside it.

    FakeTokenizer emits one id per character, so an answer of N characters is N tokens. The long
    row is what gives the truncation assertions the ability to fail: a fixture where nothing
    exceeds the cap would report zero truncated rows no matter what the measurement did.
    """

    def __init__(self):
        super().__init__()
        self._rows = [
            {"prompt": "short", "answer": "alpha"},
            {"prompt": "long", "answer": "z" * 400},
        ]


def _measured(max_context_tokens: int):
    spec = replace(_spec(), train=replace(_spec().train, max_context_tokens=max_context_tokens))
    spec = replace(
        spec,
        workload_profile_input_digest=sft_profile_input_digest(
            spec,
            tokenizer_revision=spec.model_revision,
            producer_version="1.2.3",
        ),
    )
    return prepare_sft_workload(
        spec,
        LongRowEnvironment(),
        tokenizer_loader=lambda _model, _revision: FakeTokenizer(),
        producer_version="1.2.3",
        packing_support=lambda _model, _revision: ("pure-attention", True),
    )


def test_binding_cap_reports_the_true_length_not_the_censored_one(capsys) -> None:
    """The profile must say how long the rows really are, and the warning must name that number.

    ``realized_max_length`` is measured after the slice, so it saturates at the cap exactly when the
    cap binds -- the one case where the number matters. Asserting it equals the cap while
    ``untruncated_max_length`` runs past it is what distinguishes a real measurement from reading
    the setting back.
    """
    prepared = _measured(64)

    assert prepared.profile.max_length == 64
    assert prepared.profile.realized_max_length == 64
    assert prepared.profile.untruncated_max_length > 64
    assert prepared.profile.truncated_examples == 1

    warning = capsys.readouterr().err
    assert "warning: [train] max_context_tokens 64 truncated 1 of 2 sft rows" in warning
    # the actionable half: the number to set. a warning that only says "some rows truncated"
    # cannot be acted on without a second run.
    assert str(prepared.profile.untruncated_max_length) in warning


def test_a_cap_that_does_not_bind_reports_no_truncation_and_stays_quiet(capsys) -> None:
    """The paired control. Without it the assertions above pass for a measurement wired to a constant."""
    prepared = _measured(4096)

    assert prepared.profile.truncated_examples == 0
    assert prepared.profile.untruncated_max_length == prepared.profile.realized_max_length
    assert prepared.profile.untruncated_max_length < 4096

    assert "max_context_tokens" not in capsys.readouterr().err


class ThinkingTokenizer(FakeTokenizer):
    """A tokenizer whose chat template reproduces Qwen3.5's ``<think>`` placement rule.

    Transcribed from ``Qwen/Qwen3.5-0.8B``'s own template rather than paraphrased, because the two
    details that make the warning necessary are both easy to get wrong from memory:

    * reasoning survives only on assistant turns AFTER the last non-tool user message
      (``loop.index0 > ns.last_query_index``), not merely on the last turn;
    * a trailing assistant turn ALWAYS opens a ``<think>`` block, empty when that turn authored no
      reasoning. That empty block is why survival is counted as non-empty spans;
    * ``reasoning_content`` is read in PREFERENCE to an inline span. A fake that only ever splits
      ``content`` would tear an answer apart at a ``<think>`` tag the answer merely quotes, and
      then disagree with the real template about which text is this turn's reasoning.

    ``tests/test_sft_workload_live.py`` pins this fake against the real tokenizer.
    """

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
        **_kwargs,
    ):
        assert not tokenize
        last_query = max(
            (
                index
                for index, message in enumerate(messages)
                if message.get("role") == "user"
                and not str(message.get("content") or "").startswith("<tool_response>")
            ),
            default=-1,
        )
        parts = []
        for index, message in enumerate(messages):
            raw = message.get("content")
            if isinstance(raw, list):
                content = "".join(
                    block.get("text") or ""
                    for block in raw
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            else:
                content = str(raw or "")
            reasoning = ""
            supplied = message.get("reasoning_content")
            # the split is ASSISTANT-only in the real template: a literal <think> span in a system
            # or user message is passed through verbatim, which is what lets prompt text contribute
            # a rendered span that is never supervised.
            if (
                message.get("role") == "assistant"
                and isinstance(supplied, str)
                and supplied.strip()
            ):
                # the field wins over an inline span, and `content` stays whole: any <think> the
                # answer quotes is answer text, not this turn's reasoning.
                reasoning = supplied.strip()
            elif message.get("role") == "assistant" and "</think>" in content:
                reasoning = content.split("</think>")[0].split("<think>")[-1].strip()
                content = content.split("</think>")[-1].lstrip("\n")
            # the assistant HEADER and the closing <|im_end|> are both part of the reasoning
            # layout, not decoration: the template only opens a <think> block straight after the
            # header, and <|im_end|> is what bounds one turn's reasoning from the next turn's. a
            # fake that drops either renders a shape no structural parser can read -- without the
            # header there is no anchor to find, and without the terminator one turn's block runs
            # into the following turn's closer.
            if message.get("role") == "assistant" and index > last_query:
                parts.append(
                    f"<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content}<|im_end|>"
                )
            elif message.get("role") == "assistant":
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            else:
                parts.append(content)
        text = "".join(parts)
        return text + ("<|im_start|>assistant\n<think>\n" if add_generation_prompt else "")


class ThinkingEnvironment(FakeEnvironment):
    multi_turn = True

    def __init__(self, completion, prompt="board"):
        super().__init__()
        self._rows = [{"prompt": prompt, "answer": "ignored"}]
        self._completion = completion

    def sft_completion(self, row):
        return [dict(message) for message in self._completion]


def _thinking_prepared(completion, prompt="board"):
    spec = replace(_spec(), train=replace(_spec().train, max_context_tokens=512, max_examples=0))
    spec = replace(
        spec,
        thinking=True,
        workload_profile_input_digest=sft_profile_input_digest(
            spec,
            tokenizer_revision=spec.model_revision,
            producer_version="1.2.3",
        ),
    )
    return prepare_sft_workload(
        spec,
        ThinkingEnvironment(completion, prompt=prompt),
        tokenizer_loader=lambda _model, _revision: ThinkingTokenizer(),
        producer_version="1.2.3",
        packing_support=lambda _model, _revision: ("pure-attention", True),
    )


_MULTITURN_TARGET = [
    {"role": "assistant", "content": "<think>first</think>a1"},
    {"role": "user", "content": "next"},
    {"role": "assistant", "content": "<think>second</think>a2"},
    {"role": "user", "content": "next"},
    {"role": "assistant", "content": "<think>third</think>a3"},
]


def test_a_multiturn_thinking_target_warns_that_the_template_ate_its_reasoning(capsys) -> None:
    """The defect: 3 authored reasoning blocks, 1 trained on, and nothing said so.

    A green ``flash env test`` and a correct-looking dataset both survive this, because the stored
    messages were never wrong -- only the render is. The warning has to name how much was lost,
    since "some reasoning was dropped" cannot be acted on.
    """
    prepared = _thinking_prepared(_MULTITURN_TARGET)

    assert prepared.authored_reasoning_turns == 3
    assert prepared.rendered_reasoning_spans == 1

    warning = capsys.readouterr().err
    assert "dropped 2 of 3 authored reasoning blocks" in warning
    assert "33%" in warning
    # the actionable half: without the restructuring instruction the user's only reading is
    # "turn thinking off", which discards the reasoning the dataset was built to teach.
    assert "K single-turn rows" in warning


def test_a_final_position_thinking_target_keeps_its_reasoning_and_stays_quiet(capsys) -> None:
    """The paired control. Without it the assertions above pass for a warning wired to always fire."""
    prepared = _thinking_prepared([{"role": "assistant", "content": "<think>only</think>a"}])

    assert prepared.authored_reasoning_turns == 1
    assert prepared.rendered_reasoning_spans == 1
    assert "authored reasoning blocks" not in capsys.readouterr().err


def test_reasoning_stripped_to_nothing_is_not_read_as_one_surviving_block(capsys) -> None:
    """The trap that makes naive ``count("<think>")`` wrong, and it fires on the WORST input.

    Reasoning on every turn but the last renders one EMPTY ``<think>`` block. Counting raw opening
    tags scores that as one survivor, so the transcript that lost ALL of its reasoning is the one
    that would report the smallest loss.
    """
    prepared = _thinking_prepared(
        [
            {"role": "assistant", "content": "<think>first</think>a1"},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": "a2"},
        ]
    )

    assert prepared.authored_reasoning_turns == 1
    assert prepared.rendered_reasoning_spans == 0

    warning = capsys.readouterr().err
    assert "dropped 1 of 1 authored reasoning blocks" in warning
    assert "0%" in warning


def test_a_dataset_with_no_authored_reasoning_stays_quiet(capsys) -> None:
    """The other control: the always-injected empty block must not be read as lost reasoning.

    Every thinking render carries one empty ``<think>`` block, so a rule keyed on tags rather than
    content would warn about dropped reasoning for a dataset that authored none.
    """
    prepared = _thinking_prepared([{"role": "assistant", "content": "plain answer"}])

    assert prepared.authored_reasoning_turns == 0
    assert prepared.rendered_reasoning_spans == 0
    assert "authored reasoning blocks" not in capsys.readouterr().err


def test_reasoning_carried_in_reasoning_content_counts_as_authored(capsys) -> None:
    """The template reads ``reasoning_content`` ahead of an inline span, so the source count must too.

    Counting only literal ``<think>`` in ``content`` would score these rows as reasoning-free and
    report no loss for a transcript that is losing all of it.
    """
    prepared = _thinking_prepared(
        [
            {"role": "assistant", "content": "a1", "reasoning_content": "first"},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": "a2", "reasoning_content": "second"},
        ]
    )

    assert prepared.authored_reasoning_turns == 2
    assert "dropped" in capsys.readouterr().err


def test_adjacent_empty_think_blocks_do_not_merge_into_one_survivor(capsys) -> None:
    """Two consecutive trailing assistant turns that authored nothing render two EMPTY blocks.

    A span pattern whose body may cross a delimiter lets the required non-space character be the
    ``<`` of the first closing tag, swallowing both blocks as one match. That reads as a surviving
    span on a transcript where nothing survived, which suppresses the warning on the worst input.
    """
    prepared = _thinking_prepared(
        [
            {"role": "assistant", "content": "<think>first</think>a1"},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": "a2"},
            {"role": "assistant", "content": "a3"},
        ]
    )

    assert prepared.authored_reasoning_turns == 1
    assert prepared.rendered_reasoning_spans == 0
    assert "dropped 1 of 1 authored reasoning blocks" in capsys.readouterr().err


def test_consecutive_reasoned_turns_are_counted_individually(capsys) -> None:
    """The paired control for the pattern above: real adjacent spans must still count separately.

    A pattern tightened until it stops merging empty blocks can also stop matching the second of two
    real ones, which would invent reasoning loss for a transcript that lost none.
    """
    prepared = _thinking_prepared(
        [
            {"role": "assistant", "content": "<think>first</think>a1"},
            {"role": "assistant", "content": "<think>second</think>a2"},
        ]
    )

    assert prepared.authored_reasoning_turns == 2
    assert prepared.rendered_reasoning_spans == 2
    assert "authored reasoning blocks" not in capsys.readouterr().err


def test_a_think_span_in_the_prompt_does_not_offset_reasoning_lost_from_the_target(capsys) -> None:
    """Only the supervised span counts, so prompt text cannot pay for a target's lost reasoning.

    An environment that documents the format by showing a literal ``<think>...</think>`` in its
    system prompt renders a real span that is never trained on. Counting the full render would let
    it cancel a dropped target block and silence the warning.
    """

    class PromptThinkEnvironment(ThinkingEnvironment):
        def prompt_messages(self, row):
            return [
                {"role": "system", "content": "answer as <think>reasoning</think>answer"},
                {"role": "user", "content": row["prompt"]},
            ]

    spec = replace(_spec(), train=replace(_spec().train, max_context_tokens=512, max_examples=0))
    spec = replace(
        spec,
        thinking=True,
        workload_profile_input_digest=sft_profile_input_digest(
            spec,
            tokenizer_revision=spec.model_revision,
            producer_version="1.2.3",
        ),
    )
    prepared = prepare_sft_workload(
        spec,
        PromptThinkEnvironment(
            [
                {"role": "assistant", "content": "<think>first</think>a1"},
                {"role": "user", "content": "next"},
                {"role": "assistant", "content": "a2"},
            ]
        ),
        tokenizer_loader=lambda _model, _revision: ThinkingTokenizer(),
        producer_version="1.2.3",
        packing_support=lambda _model, _revision: ("pure-attention", True),
    )

    assert prepared.authored_reasoning_turns == 1
    # the prompt's own span is excluded, so the target's loss is still visible
    assert prepared.rendered_reasoning_spans == 0
    assert "dropped 1 of 1 authored reasoning blocks" in capsys.readouterr().err


def test_reasoning_in_a_dropped_row_is_not_reported_against_the_retained_rows(capsys) -> None:
    """A row whose completion is truncated away is not trained on, so its reasoning is not lost to
    the template -- the row is simply gone, and the existing drop warning covers it.

    Counting it here would report a reasoning loss "across N rows" that the retained rows did not
    incur, and could warn about a run whose every trained row keeps its reasoning.
    """

    class MixedEnvironment(ThinkingEnvironment):
        def __init__(self):
            super().__init__([])
            self._rows = [{"prompt": "x" * 400, "answer": ""}, {"prompt": "ok", "answer": ""}]

        def sft_completion(self, row):
            if row["prompt"] == "ok":
                return [{"role": "assistant", "content": "<think>kept</think>a"}]
            return [
                {"role": "assistant", "content": "<think>lost</think>a1"},
                {"role": "user", "content": "next"},
                {"role": "assistant", "content": "a2"},
            ]

    spec = replace(_spec(), train=replace(_spec().train, max_context_tokens=64, max_examples=0))
    spec = replace(
        spec,
        thinking=True,
        workload_profile_input_digest=sft_profile_input_digest(
            spec,
            tokenizer_revision=spec.model_revision,
            producer_version="1.2.3",
        ),
    )
    prepared = prepare_sft_workload(
        spec,
        MixedEnvironment(),
        tokenizer_loader=lambda _model, _revision: ThinkingTokenizer(),
        producer_version="1.2.3",
        packing_support=lambda _model, _revision: ("pure-attention", True),
    )

    # the long row lost its whole completion to the cap and was dropped
    assert prepared.profile.dropped_examples == 1
    assert prepared.profile.retained_examples == 1
    # so only the retained row's reasoning is accounted, and it kept all of it
    assert prepared.authored_reasoning_turns == 1
    assert prepared.rendered_reasoning_spans == 1
    assert "authored reasoning blocks" not in capsys.readouterr().err


def test_block_form_assistant_content_counts_as_authored_reasoning(capsys) -> None:
    """Content blocks are a supported target shape, and a shape missed here silences the warning.

    ``reasoned_assistant_turns`` reading only string ``content`` would score a block-form multi-turn
    target as authoring nothing, so a row losing all its reasoning would report none.
    """
    prepared = _thinking_prepared(
        [
            {"role": "assistant", "content": [{"type": "text", "text": "<think>first</think>a1"}]},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": [{"type": "text", "text": "<think>second</think>a2"}]},
        ]
    )

    assert prepared.authored_reasoning_turns == 2
    assert "dropped 1 of 2 authored reasoning blocks" in capsys.readouterr().err


def test_reasoning_the_cap_kept_is_not_reported_as_dropped(capsys) -> None:
    """Truncation is judged per span, not per row: the cap usually cuts only the answer tail.

    A single final assistant turn whose ``<think>`` block sits entirely inside the retained tokens
    loses nothing, however much of its answer the cap removes. Discarding every span on a truncated
    row would warn that the template dropped reasoning it actually kept.
    """
    completion = [{"role": "assistant", "content": "<think>kept</think>" + "tail " * 80}]
    spec = replace(_spec(), train=replace(_spec().train, max_context_tokens=64, max_examples=0))
    spec = replace(
        spec,
        thinking=True,
        workload_profile_input_digest=sft_profile_input_digest(
            spec,
            tokenizer_revision=spec.model_revision,
            producer_version="1.2.3",
        ),
    )
    prepared = prepare_sft_workload(
        spec,
        ThinkingEnvironment(completion),
        tokenizer_loader=lambda _model, _revision: ThinkingTokenizer(),
        producer_version="1.2.3",
        packing_support=lambda _model, _revision: ("pure-attention", True),
    )

    # the row IS truncated, but only past the reasoning block
    assert prepared.profile.truncated_examples == 1
    assert prepared.authored_reasoning_turns == 1
    assert prepared.rendered_reasoning_spans == 1
    assert "authored reasoning blocks" not in capsys.readouterr().err


def test_an_answer_quoting_think_is_not_credited_when_reasoning_content_is_stripped(
    capsys,
) -> None:
    """``reasoning_content`` is the reasoning, so ``content`` is answer text and stays intact.

    Splitting the answer on ``</think>`` as well would delete a span the full render still has, so
    the baseline would come up short and credit that quoted span as a survivor -- offsetting an
    earlier block the template really dropped and silencing the warning.
    """
    prepared = _thinking_prepared(
        [
            {"role": "assistant", "content": "<think>lost</think>a1"},
            {"role": "user", "content": "next"},
            {
                "role": "assistant",
                "content": "answer shows <think>example</think> format",
                "reasoning_content": "real reasoning",
            },
        ]
    )

    assert prepared.authored_reasoning_turns == 2
    # only the final turn's reasoning survives; the quoted span cancels rather than counting
    assert prepared.rendered_reasoning_spans == 1
    assert "dropped 1 of 2 authored reasoning blocks" in capsys.readouterr().err


def test_an_answer_quoting_think_tags_cannot_mask_turns_the_template_dropped(capsys) -> None:
    """Survivors are probed per turn, so quoted tags cannot pay for dropped reasoning.

    A ``<think>`` tag inside a supervised answer is a real non-empty span, indistinguishable in the
    rendered text from reasoning. Summing spans across the render lets an answer that quotes the
    format cover several stripped turns and silence the warning entirely. Re-rendering with one
    turn's reasoning removed is immune: the quote is identical in both renders.
    """
    prepared = _thinking_prepared(
        [
            {"role": "assistant", "content": "<think>first</think>a1"},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": "<think>second</think>a2"},
            {"role": "user", "content": "next"},
            {
                "role": "assistant",
                "content": "quotes <think>x</think> and <think>y</think> here",
                "reasoning_content": "real reasoning",
            },
        ]
    )

    assert prepared.authored_reasoning_turns == 3
    # two quoted spans would otherwise bring the count to 3 and report no loss at all
    assert prepared.rendered_reasoning_spans == 1
    assert "dropped 2 of 3 authored reasoning blocks" in capsys.readouterr().err


def test_a_truncated_row_does_not_claim_reasoning_the_cap_removed(capsys) -> None:
    """A block past ``max_context_tokens`` never reaches the loss, so it is not a survivor.

    Reporting it would understate the real reasoning loss while quoting an exact survival
    percentage.
    """
    long_reasoning = "<think>" + "survivor " * 30 + "</think>a2"
    completion = [
        {"role": "assistant", "content": "<think>lost</think>a1"},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": long_reasoning},
    ]
    spec = replace(_spec(), train=replace(_spec().train, max_context_tokens=64, max_examples=0))
    spec = replace(
        spec,
        thinking=True,
        workload_profile_input_digest=sft_profile_input_digest(
            spec,
            tokenizer_revision=spec.model_revision,
            producer_version="1.2.3",
        ),
    )
    prepared = prepare_sft_workload(
        spec,
        ThinkingEnvironment(completion),
        tokenizer_loader=lambda _model, _revision: ThinkingTokenizer(),
        producer_version="1.2.3",
        packing_support=lambda _model, _revision: ("pure-attention", True),
    )

    # the row IS retained and IS truncated: it is trained on, with its reasoning cut away
    assert prepared.profile.retained_examples == 1
    assert prepared.profile.truncated_examples == 1
    assert prepared.authored_reasoning_turns == 2
    # the template kept one block and stripped the other; the cap then cut the one it kept, so
    # nothing reaches the loss -- but the two causes stay separate because the remedies differ
    assert prepared.rendered_reasoning_spans == 1
    assert prepared.truncated_reasoning_spans == 1
    err = capsys.readouterr().err
    assert "the chat template dropped 1 of 2 authored reasoning blocks" in err
    assert "max_context_tokens cut 1 rendered reasoning block" in err
    assert "0 of 2 authored reasoning blocks reach the loss" in err


def test_a_reasoned_assistant_turn_in_the_prompt_does_not_cancel_a_surviving_target_block(
    capsys,
) -> None:
    """A prompt span the FULL render strips must not be subtracted from the full render's count.

    A prompt ending in a reasoned assistant turn is trailing while the prompt is rendered alone, so
    that render keeps its reasoning -- but the completion adds a later user turn, which moves the
    template's boundary past it and strips it from the full render. Subtracting the prompt render's
    count would remove a span the full render never had, cancelling the target's own surviving block
    and warning about a row that lost nothing.
    """

    class PriorTurnEnvironment(ThinkingEnvironment):
        def prompt_messages(self, row):
            return [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": "<think>promptreason</think>a1"},
            ]

    spec = replace(_spec(), train=replace(_spec().train, max_context_tokens=512, max_examples=0))
    spec = replace(
        spec,
        thinking=True,
        workload_profile_input_digest=sft_profile_input_digest(
            spec,
            tokenizer_revision=spec.model_revision,
            producer_version="1.2.3",
        ),
    )
    prepared = prepare_sft_workload(
        spec,
        PriorTurnEnvironment(
            [
                {"role": "user", "content": "next"},
                {"role": "assistant", "content": "<think>targetreason</think>a2"},
            ]
        ),
        tokenizer_loader=lambda _model, _revision: ThinkingTokenizer(),
        producer_version="1.2.3",
        packing_support=lambda _model, _revision: ("pure-attention", True),
    )

    # the one authored target turn is in final position, so the template keeps it: nothing is lost
    assert prepared.authored_reasoning_turns == 1
    assert prepared.rendered_reasoning_spans == 1
    assert "authored reasoning blocks" not in capsys.readouterr().err


def test_an_earlier_surviving_block_is_not_reported_lost_when_a_later_one_is_truncated(
    capsys,
) -> None:
    """Each surviving span is measured against the cap at ITS OWN end offset.

    Two consecutive trailing assistant turns both keep their reasoning, and the cap falls between
    the two blocks. Measuring one span's position in a render that does not contain it -- as any
    scheme comparing offsets from two DIFFERENT renders must -- reads the earlier, fully retained
    block as ending where the later one does, and reports it truncated. The row would then claim
    zero survivors and warn that the template dropped every block, while the first block is in fact
    supervised.
    """
    completion = [
        {"role": "assistant", "content": "<think>" + "early " * 6 + "</think>a1"},
        {"role": "assistant", "content": "<think>" + "late " * 40 + "</think>a2"},
    ]
    # the fake tokenizer is one token per character: the first block closes at 53 and the second at
    # 273, so this cap falls between them and retains exactly one of the two.
    spec = replace(_spec(), train=replace(_spec().train, max_context_tokens=100, max_examples=0))
    spec = replace(
        spec,
        thinking=True,
        workload_profile_input_digest=sft_profile_input_digest(
            spec,
            tokenizer_revision=spec.model_revision,
            producer_version="1.2.3",
        ),
    )
    prepared = prepare_sft_workload(
        spec,
        ThinkingEnvironment(completion),
        tokenizer_loader=lambda _model, _revision: ThinkingTokenizer(),
        producer_version="1.2.3",
        packing_support=lambda _model, _revision: ("pure-attention", True),
    )

    assert prepared.authored_reasoning_turns == 2
    # the template kept BOTH (they are consecutive trailing turns); the cap then cut only the
    # second, so exactly one block reaches the loss and the remedy named is the cap, not splitting
    assert prepared.rendered_reasoning_spans == 2
    assert prepared.truncated_reasoning_spans == 1
    err = capsys.readouterr().err
    assert "max_context_tokens cut 1 rendered reasoning block" in err
    assert "1 of 2 authored reasoning blocks reach the loss" in err
    assert "the chat template dropped" not in err


def test_a_think_tag_quoted_in_the_prompt_does_not_swallow_the_turns_that_follow(capsys) -> None:
    """A bare ``<think>`` in the PROMPT is content, and must not consume the blocks after it.

    A user asking what the tag means renders a literal unmatched opener into the prompt. Tracking
    delimiters by depth leaves that opener permanently unclosed, so the template's own closer
    completes nothing and every later block reads as stripped -- a total-loss warning for a row the
    template kept whole. The reasoning layout is what identifies a block, not the raw tags.
    """
    completion = [{"role": "assistant", "content": "<think>real reasoning</think>an answer"}]
    prepared = _thinking_prepared(completion, prompt="what does the <think> tag mean?")

    assert prepared.authored_reasoning_turns == 1
    assert prepared.rendered_reasoning_spans == 1
    assert prepared.truncated_reasoning_spans == 0
    assert "reasoning" not in capsys.readouterr().err


@pytest.mark.parametrize(
    ("label", "quoted"),
    [
        # inline, as prose about the tag
        ("same line", "the </think> tag closes it "),
        # quoting the LAYOUT reproduces the template's own delimiter shape, `\n</think>\n\n`, which
        # is exactly what a structural end-marker matches. a rule that stops at the FIRST one ends
        # here instead of at the template's closer, and only a newline-delimited quote can prove it
        # does not: a same-line quote never produces that sequence.
        ("own line", "the layout is:\n</think>\n\nlike that, "),
    ],
)
def test_a_closing_tag_quoted_inside_reasoning_does_not_end_the_block_early(
    capsys, label: str, quoted: str
) -> None:
    """The block ends at the closer the TEMPLATE emits, not at one the reasoning happens to quote.

    Reasoning that discusses the delimiter renders an unmatched ``</think>`` inside its own body.
    Ending the span there puts its end before the real one, so a cap falling between the two calls
    a cut block retained and overstates what reaches the loss.

    The cap has to sit BETWEEN the two candidate ends to discriminate them. Measured on this fake
    (one token per character), an early end lands at 60 and the template's real closer at 132/141,
    so 100 is truncated under the correct rule and retained under an early one. A cap below both --
    40, as this test first used -- reports truncation either way and cannot fail.
    """
    completion = [
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": quoted + "rest " * 12,
        }
    ]
    spec = replace(_spec(), train=replace(_spec().train, max_context_tokens=100, max_examples=0))
    spec = replace(
        spec,
        thinking=True,
        workload_profile_input_digest=sft_profile_input_digest(
            spec,
            tokenizer_revision=spec.model_revision,
            producer_version="1.2.3",
        ),
    )
    prepared = prepare_sft_workload(
        spec,
        ThinkingEnvironment(completion),
        tokenizer_loader=lambda _model, _revision: ThinkingTokenizer(),
        producer_version="1.2.3",
        packing_support=lambda _model, _revision: ("pure-attention", True),
    )

    # the cap cuts the block, and measuring to the real closer is what sees that
    assert prepared.authored_reasoning_turns == 1
    assert prepared.truncated_reasoning_spans == 1
    assert "max_context_tokens cut 1 rendered reasoning" in capsys.readouterr().err


def test_reasoning_containing_a_balanced_tag_is_measured_to_its_real_end(capsys) -> None:
    """A span's end is the OUTER closer, even when the reasoning quotes a balanced pair inside it.

    A turn reasoning about the tag format renders `<think>start <think>x</think> rest</think>`.
    Reading the span as ending at the INNER closer would place its end well before the real one, so
    a cap falling between the two would call a cut block retained and overstate what reaches the
    loss. Here the cap sits past the inner closer but before the outer one.
    """
    completion = [
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "start <think>x</think> " + "rest " * 12,
        }
    ]
    spec = replace(_spec(), train=replace(_spec().train, max_context_tokens=40, max_examples=0))
    spec = replace(
        spec,
        thinking=True,
        workload_profile_input_digest=sft_profile_input_digest(
            spec,
            tokenizer_revision=spec.model_revision,
            producer_version="1.2.3",
        ),
    )
    prepared = prepare_sft_workload(
        spec,
        ThinkingEnvironment(completion),
        tokenizer_loader=lambda _model, _revision: ThinkingTokenizer(),
        producer_version="1.2.3",
        packing_support=lambda _model, _revision: ("pure-attention", True),
    )

    assert prepared.authored_reasoning_turns == 1
    assert prepared.rendered_reasoning_spans == 1
    # measured to the outer closer, so the cap is correctly seen to cut it
    assert prepared.truncated_reasoning_spans == 1
    assert "max_context_tokens cut 1 rendered reasoning block" in capsys.readouterr().err


def test_reasoning_sampled_without_an_opening_tag_still_counts_as_authored(capsys) -> None:
    """``reasoned</think>answer`` is reasoning: the PROMPT supplied the opening tag.

    ``flash/serve/thinking.py`` recognises the same shape. Requiring a balanced pair would leave
    such a turn out of the authored denominator entirely, so an early turn whose reasoning the
    template strips would be lost with nothing reporting it.
    """
    completion = [
        {"role": "assistant", "content": "early reasoning</think>a1"},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "<think>late</think>a2"},
    ]
    prepared = _thinking_prepared(completion)

    # both turns authored reasoning, and the opener-less one is the turn the template strips
    assert prepared.authored_reasoning_turns == 2
    assert prepared.rendered_reasoning_spans == 1
    assert prepared.truncated_reasoning_spans == 0
    assert "the chat template dropped 1 of 2 authored reasoning blocks" in capsys.readouterr().err


def test_the_templates_own_empty_think_block_is_not_counted_as_authored(capsys) -> None:
    """A turn carrying only ``<think>\\n\\n</think>`` authored nothing, so nothing can be lost.

    The template stamps that empty block onto qualifying trailing assistant turns, so it comes back
    in any transcript captured from a previous render. Counting it as authored marks a block the
    real render does not have; the span counts then disagree and the row is reported as having lost
    ALL of its reasoning -- a total-loss warning for a dataset that lost none.
    """
    completion = [{"role": "assistant", "content": "<think>\n\n</think>just an answer"}]
    prepared = _thinking_prepared(completion)

    assert prepared.authored_reasoning_turns == 0
    assert prepared.rendered_reasoning_spans == 0
    assert prepared.truncated_reasoning_spans == 0
    assert "reasoning" not in capsys.readouterr().err


@pytest.mark.parametrize(
    ("label", "content"),
    [
        # an extra opener before the real block: the template takes the LAST <think> before the
        # first </think>, so stamping the first one puts the marker outside the kept span
        ("extra opener", "<think>outer <think>real reasoning</think>a"),
        # the template concatenates text blocks before splitting, so a delimiter can straddle a
        # block boundary and exists in neither block on its own
        (
            "split delimiter",
            [{"type": "text", "text": "<thi"}, {"type": "text", "text": "nk>r</think>a"}],
        ),
    ],
)
def test_reasoning_the_template_keeps_is_marked_wherever_its_delimiters_sit(
    label: str, content, capsys
) -> None:
    """The marker must land inside the block the template keeps, not merely near a ``<think>``.

    If it lands outside, the marker never reaches the render, and the row reports a drop for
    reasoning that in fact reached the loss -- a false warning telling the user to restructure a
    dataset that was fine.
    """
    prepared = _thinking_prepared([{"role": "assistant", "content": content}])

    assert prepared.authored_reasoning_turns == 1
    assert prepared.rendered_reasoning_spans == 1, label
    assert "authored reasoning blocks" not in capsys.readouterr().err


def test_a_quoting_answer_on_a_dropped_turn_is_not_credited_as_a_survivor(capsys) -> None:
    """A turn whose reasoning the template dropped stays dropped, however its answer is written.

    The early turn supplies reasoning through ``reasoning_content`` and its ANSWER quotes the
    ``<think>`` format. The template strips that turn's reasoning but renders the quote verbatim, so
    any scheme that asks "did perturbing this turn change the span count?" sees the quote move and
    credits the turn. Survival is an identity question: the quote is not this turn's reasoning and
    can never stand in for it.
    """
    completion = [
        {
            "role": "assistant",
            "content": "write it as <think>like this</think> ok",
            "reasoning_content": "early reasoning",
        },
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "<think>late</think>a2"},
    ]
    spec = replace(_spec(), train=replace(_spec().train, max_context_tokens=512, max_examples=0))
    spec = replace(
        spec,
        thinking=True,
        workload_profile_input_digest=sft_profile_input_digest(
            spec,
            tokenizer_revision=spec.model_revision,
            producer_version="1.2.3",
        ),
    )
    prepared = prepare_sft_workload(
        spec,
        ThinkingEnvironment(completion),
        tokenizer_loader=lambda _model, _revision: ThinkingTokenizer(),
        producer_version="1.2.3",
        packing_support=lambda _model, _revision: ("pure-attention", True),
    )

    assert prepared.authored_reasoning_turns == 2
    # only the final turn's reasoning reaches the loss; the quoted tags are answer text
    assert prepared.rendered_reasoning_spans == 1
    assert "dropped 1 of 2 authored reasoning blocks" in capsys.readouterr().err


def test_an_image_rows_visual_tokens_are_charged_against_the_reasoning_cap() -> None:
    """On an image row the cap is spent on visual tokens before any text reaches it.

    Training truncates the ids the PROCESSOR produced, in which each image has expanded into many
    visual tokens; the text render contains none of them. Measuring a reasoning block against the
    raw cap therefore calls it retained when the expansion has already pushed it past the end of
    the supervised span, overstating how much reasoning reaches the loss.
    """
    import base64
    import io as _io

    from PIL import Image

    buffer = _io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buffer, format="PNG")
    image_uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    class ImageEnvironment(FakeEnvironment):
        multi_turn = True

        def __init__(self):
            super().__init__()
            self._rows = [{"prompt": "describe", "answer": "ignored", "image": image_uri}]

        def sft_completion(self, row):
            return [{"role": "assistant", "content": "<think>" + "why " * 4 + "</think>a"}]

    class ExpandingProcessor:
        """A processor whose single image expands into ``visual`` extra ids, as a real one does."""

        # the row's reasoning block closes 40 characters into the text render, so a cap of 200
        # clears it easily on text alone; these visual tokens are what actually spend the budget.
        visual = 180

        def __init__(self):
            # the multimodal path renders text through the PROCESSOR's tokenizer, so the two must
            # be the same object the reasoning check measures with
            self.tokenizer = ThinkingTokenizer()

        def apply_chat_template(
            self, messages, *, tokenize, return_dict, return_tensors, enable_thinking, **_kwargs
        ):
            text = "".join(
                block.get("text") or ""
                for message in messages
                for block in (
                    message["content"]
                    if isinstance(message.get("content"), list)
                    else [{"type": "text", "text": str(message.get("content") or "")}]
                )
                if isinstance(block, dict)
            )
            ids = [3 + ord(char) % 89 for char in text] + [7] * self.visual
            return {"input_ids": [ids], "attention_mask": [[1] * len(ids)]}

    spec = replace(
        _spec(),
        # image-bearing rows are refused outright for a model the catalog says cannot train on them
        model="Qwen/Qwen3.5-4B",
        train=replace(_spec().train, max_context_tokens=200, max_examples=0),
    )
    spec = replace(
        spec,
        thinking=True,
        workload_profile_input_digest=sft_profile_input_digest(
            spec,
            tokenizer_revision=spec.model_revision,
            producer_version="1.2.3",
        ),
    )
    prepared = prepare_sft_workload(
        spec,
        ImageEnvironment(),
        tokenizer_loader=lambda _model, _revision: ThinkingTokenizer(),
        processor_loader=lambda _model, _revision: ExpandingProcessor(),
        producer_version="1.2.3",
        packing_support=lambda _model, _revision: ("pure-attention", True),
    )

    # the block closes 40 characters into the text render, well inside the 200-token cap, but the
    # image's 180 visual tokens are charged first and leave it past the end of the supervised span.
    # the template rendered it, so the loss is attributed to the CAP, not to the transcript shape
    assert prepared.authored_reasoning_turns == 1
    assert prepared.rendered_reasoning_spans == 1
    assert prepared.truncated_reasoning_spans == 1
