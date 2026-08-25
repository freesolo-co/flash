from __future__ import annotations

import time
from dataclasses import replace
from typing import ClassVar

import pytest

from flash.core.spec import EnvironmentSpec, JobSpec, TrainSpec
from flash.engine.profiling.reasoning_render import reasoning_marker_prefix
from flash.engine.profiling.sft_workload import (
    prepare_sft_workload,
    sft_tokens_for_updates,
)
from flash.engine.profiling.workload_profile import (
    sft_profile_input_digest,
    unpacked_batch_warning,
)
from flash.engine.worker.entry.sft import select_sft_examples


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


def test_retained_workload_counts_mixed_role_aware_and_fallback_multiturn_rows() -> None:
    from flash.engine.profiling.sft_workload import (
        _filter_retained_rows,
        _RowReasoning,
        _TokenizedSftRows,
    )

    tokenized = _TokenizedSftRows(
        row_by_index={
            0: {"input_ids": [10], "loss_mask": [1]},
            1: {"input_ids": [11], "loss_mask": [1]},
        },
        untruncated_by_index={0: 1, 1: 1},
        sampled_texts=[],
        multiturn_targets=2,
        coerced_singleturn_targets=0,
        multiturn_mask_applied={0: True, 1: False},
        reasoning_by_index={0: _RowReasoning(0, 0, 0), 1: _RowReasoning(0, 0, 0)},
        dropped=0,
    )

    retained = _filter_retained_rows(tokenized, FakeTokenizer())

    assert retained.role_aware_multiturn_targets == 1
    assert retained.fallback_multiturn_targets == 1


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
    """The worker resolves batch_size to the recipe default before warning, so handing the resolved
    number to the helper made an omitted knob read as one the user configured -- the opposite of
    what the cli says about the same run.
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


def _training_order(count: int) -> list[str]:
    """The prompts in the order the TRAINER consumes them, from its own selection function.

    The rows are shuffled under the job seed, so file order is not training order.
    """
    rows = [{"prompt": f"board{index}", "answer": "ignored"} for index in range(count)]
    return [row["prompt"] for row in select_sft_examples(rows, 0, _spec().seed)]


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

    The label is frozen into the profile and compared byte-for-byte by the training worker, so a hub
    blip answering "unsupported" here would make every run built on that profile die with a false
    "sft workload changed after the quote was frozen" -- with no takeover path, since the profile
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

    FakeTokenizer emits one id per character, so an N-character answer is N tokens. The long row is
    what gives the truncation assertions the ability to fail: a fixture where nothing exceeds the
    cap reports zero truncated rows no matter what the measurement does.
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
    """``realized_max_length`` is measured after the slice, so it saturates at the cap exactly when
    the cap binds -- the one case where the number matters. Asserting it equals the cap while
    ``untruncated_max_length`` runs past it distinguishes a real measurement from reading the
    setting back.
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

    Transcribed from ``Qwen/Qwen3.5-9B``'s own template rather than paraphrased, because three
    details are easy to get wrong from memory:

    * reasoning survives only on assistant turns AFTER the last non-tool user message (``loop.index0
      > ns.last_query_index``), not merely on the last turn;
    * a trailing assistant turn ALWAYS opens a ``<think>`` block, empty when it authored nothing;
    * ``reasoning_content`` is read in PREFERENCE to an inline span, so a fake that only splits
      ``content`` would tear an answer apart at a ``<think>`` it merely quotes.

    ``tests/live/test_sft_thinking_render_live.py`` pins this fake against the real tokenizer.
    """

    chat_template = (
        "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
        "{{ message['reasoning_content'] }}{{ message['content'] }}<|im_end|>\n{% endfor %}"
    )

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
            # EVERY turn carries the full header/terminator frame, because that frame is the
            # reasoning layout rather than decoration: the template only opens a <think> block
            # straight after an assistant header, and a header only counts as one when it follows
            # the previous turn's <|im_end|>. a fake that renders bare content leaves the first
            # header unprefixed and the turns unbounded, so a structural parser either finds no
            # anchor or lets one turn's block run into the next turn's closer -- shapes the real
            # template never produces.
            role = message.get("role")
            if role == "assistant" and index > last_query:
                body = f"<think>\n{reasoning}\n</think>\n\n{content}"
            else:
                body = content
            if role == "tool":
                # the real template has no `tool` header: a tool message renders as a USER turn
                # wrapping the output in <tool_response>. that shape is why a tool turn does not
                # reset last_query_index the way an ordinary user turn does, so a fake emitting
                # `<|im_start|>tool` would render a header the template never produces and could
                # agree with it by accident while disagreeing on the layout that matters.
                role = "user"
                body = f"<tool_response>\n{body}\n</tool_response>"
            parts.append(f"<|im_start|>{role}\n{body}<|im_end|>\n")
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


def _thinking_prepared_env(
    env, *, max_context_tokens=512, max_steps=None, model=None, processor_loader=None
):
    """Profile ``env`` through the thinking tokenizer, for a test supplying its own transcript."""
    train = replace(_spec().train, max_context_tokens=max_context_tokens, max_examples=0)
    if max_steps is not None:
        train = replace(train, max_steps=max_steps)
    spec = replace(_spec(), train=train)
    if model is not None:
        spec = replace(spec, model=model)
    spec = replace(
        spec,
        thinking=True,
        workload_profile_input_digest=sft_profile_input_digest(
            spec,
            tokenizer_revision=spec.model_revision,
            producer_version="1.2.3",
        ),
    )
    extra = {} if processor_loader is None else {"processor_loader": processor_loader}
    return prepare_sft_workload(
        spec,
        env,
        tokenizer_loader=lambda _model, _revision: ThinkingTokenizer(),
        producer_version="1.2.3",
        packing_support=lambda _model, _revision: ("pure-attention", True),
        **extra,
    )


def _thinking_prepared(completion, prompt="board", *, max_context_tokens=512):
    return _thinking_prepared_env(
        ThinkingEnvironment(completion, prompt=prompt), max_context_tokens=max_context_tokens
    )


_MULTITURN_TARGET = [
    {"role": "assistant", "content": "<think>first</think>a1"},
    {"role": "user", "content": "next"},
    {"role": "assistant", "content": "<think>second</think>a2"},
    {"role": "user", "content": "next"},
    {"role": "assistant", "content": "<think>third</think>a3"},
]


def test_a_multiturn_thinking_target_warns_that_the_template_ate_its_reasoning(capsys) -> None:
    """The defect: 3 authored reasoning blocks, 1 trained on, and nothing said so. A green ``flash
    env test`` and a correct-looking dataset both survive it, because the stored messages are never
    wrong -- only the render is. The warning names how much was lost, since "some reasoning was
    dropped" cannot be acted on.
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
    """The trap that makes naive ``count("<think>")`` wrong, and it fires on the WORST input:
    reasoning on every turn but the last renders one EMPTY block, so counting raw opening tags
    scores the transcript that lost ALL its reasoning as having a survivor.
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
    """The other control. Every thinking render carries one empty ``<think>`` block, so a rule keyed
    on tags rather than content warns about dropped reasoning for a dataset that authored none.
    """
    prepared = _thinking_prepared([{"role": "assistant", "content": "plain answer"}])

    assert prepared.authored_reasoning_turns == 0
    assert prepared.rendered_reasoning_spans == 0
    assert "authored reasoning blocks" not in capsys.readouterr().err


def test_reasoning_carried_in_reasoning_content_counts_as_authored(capsys) -> None:
    """The template reads ``reasoning_content`` ahead of an inline span, so the source count must
    too. Counting only literal ``<think>`` in ``content`` scores these rows as reasoning-free and
    reports no loss for a transcript losing all of it.
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
    """Two consecutive trailing assistant turns that authored nothing render two EMPTY blocks. A
    span pattern whose body may cross a delimiter lets the required non-space character be the ``<``
    of the first closing tag, swallowing both as one match -- a survivor on a transcript where
    nothing survived.
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
    """The paired control for the case above: a pattern tightened until it stops merging empty
    blocks can also stop matching the second of two real ones, inventing loss for a transcript that
    lost none.
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
    """An environment documenting the format with a literal ``<think>...</think>`` in its system
    prompt renders a real span that is never trained on. Counting the full render lets it cancel a
    dropped target block and silence the warning.
    """

    class PromptThinkEnvironment(ThinkingEnvironment):
        def prompt_messages(self, row):
            return [
                {"role": "system", "content": "answer as <think>reasoning</think>answer"},
                {"role": "user", "content": row["prompt"]},
            ]

    prepared = _thinking_prepared_env(
        PromptThinkEnvironment(
            [
                {"role": "assistant", "content": "<think>first</think>a1"},
                {"role": "user", "content": "next"},
                {"role": "assistant", "content": "a2"},
            ]
        )
    )

    assert prepared.authored_reasoning_turns == 1
    # the prompt's own span is excluded, so the target's loss is still visible
    assert prepared.rendered_reasoning_spans == 0
    assert "dropped 1 of 1 authored reasoning blocks" in capsys.readouterr().err


def test_reasoning_in_a_dropped_row_is_not_reported_against_the_retained_rows(capsys) -> None:
    """A row whose completion is truncated away is not trained on, so its reasoning is not lost to
    the template -- the row is gone, and the existing drop warning covers it. Counting it here
    reports a loss "across N rows" the retained rows did not incur.
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

    # between the retained row's whole render (87) and the dropped row's prompt alone (458), so the
    # long row loses its completion to the cap while the short row keeps its block intact. a cap
    # under 87 would truncate the RETAINED row's reasoning too, and the warning it raised would
    # look like the dropped row leaking into the count while proving nothing about it.
    prepared = _thinking_prepared_env(MixedEnvironment(), max_context_tokens=100)

    # the long row lost its whole completion to the cap and was dropped
    assert prepared.profile.dropped_examples == 1
    assert prepared.profile.retained_examples == 1
    # so only the retained row's reasoning is accounted, and it kept all of it
    assert prepared.authored_reasoning_turns == 1
    assert prepared.rendered_reasoning_spans == 1
    assert "authored reasoning blocks" not in capsys.readouterr().err


def test_block_form_assistant_content_counts_as_authored_reasoning(capsys) -> None:
    """Content blocks are a supported target shape, and a shape missed here silences the warning:
    ``reasoned_assistant_turns`` reading only string ``content`` scores a block-form multi-turn
    target as authoring nothing.
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
    """Truncation is judged per span, not per row: the cap usually cuts only the answer tail. A
    final turn whose ``<think>`` block sits inside the retained tokens loses nothing however much of
    its answer is removed, so discarding every span on a truncated row warns about reasoning it
    kept.
    """
    completion = [{"role": "assistant", "content": "<think>kept</think>" + "tail " * 80}]
    # one token per character: the block's closing tag ends at 76 and the row runs to 489, so this
    # cap keeps the reasoning whole while still cutting most of the answer tail.
    prepared = _thinking_prepared_env(ThinkingEnvironment(completion), max_context_tokens=100)

    # the row IS truncated, but only past the reasoning block
    assert prepared.profile.truncated_examples == 1
    assert prepared.authored_reasoning_turns == 1
    assert prepared.rendered_reasoning_spans == 1
    assert "authored reasoning blocks" not in capsys.readouterr().err


def test_an_answer_quoting_think_is_not_credited_when_reasoning_content_is_stripped(
    capsys,
) -> None:
    """``reasoning_content`` is the reasoning, so ``content`` is answer text and stays intact.
    Splitting the answer on ``</think>`` too would delete a span the full render still has, so the
    baseline comes up short and credits the quote as a survivor -- offsetting a block the template
    really dropped.
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
    """A ``<think>`` tag inside a supervised answer is a real non-empty span, indistinguishable in
    the rendered text from reasoning, so summing spans across the render lets one quoting answer
    cover several stripped turns. Probing per turn is immune: the quote is identical in both
    renders.
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
    Reporting it understates the real loss while quoting an exact survival percentage.
    """
    long_reasoning = "<think>" + "survivor " * 30 + "</think>a2"
    completion = [
        {"role": "assistant", "content": "<think>lost</think>a1"},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": long_reasoning},
    ]
    prepared = _thinking_prepared_env(ThinkingEnvironment(completion), max_context_tokens=64)

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
    """A prompt ending in a reasoned assistant turn is trailing while the prompt renders ALONE, so
    that render keeps its reasoning -- but the completion adds a later user turn, which moves the
    template's boundary past it and strips it from the full render. Subtracting the prompt render's
    count removes a span the full render never had, cancelling the target's own surviving block.
    """

    class PriorTurnEnvironment(ThinkingEnvironment):
        def prompt_messages(self, row):
            return [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": "<think>promptreason</think>a1"},
            ]

    prepared = _thinking_prepared_env(
        PriorTurnEnvironment(
            [
                {"role": "user", "content": "next"},
                {"role": "assistant", "content": "<think>targetreason</think>a2"},
            ]
        )
    )

    # the one authored target turn is in final position, so the template keeps it: nothing is lost
    assert prepared.authored_reasoning_turns == 1
    assert prepared.rendered_reasoning_spans == 1
    assert "authored reasoning blocks" not in capsys.readouterr().err


def test_an_earlier_surviving_block_is_not_reported_lost_when_a_later_one_is_truncated(
    capsys,
) -> None:
    """Each surviving span is measured against the cap at ITS OWN end offset. Two trailing assistant
    turns both keep their reasoning and the cap falls between the blocks; measuring one span's
    position in a render that does not contain it -- as any scheme comparing offsets from two
    DIFFERENT renders must -- reads the earlier, fully retained block as ending where the later one
    does.
    """
    completion = [
        {"role": "assistant", "content": "<think>" + "early " * 6 + "</think>a1"},
        {"role": "assistant", "content": "<think>" + "late " * 40 + "</think>a2"},
    ]
    # the fake tokenizer is one token per character: the first block closes at 107 and the second
    # at 360, so this cap falls between them and retains exactly one of the two.
    prepared = _thinking_prepared_env(ThinkingEnvironment(completion), max_context_tokens=200)

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
    """A user asking what the tag means renders a literal unmatched opener into the prompt. Tracking
    delimiters by depth leaves it permanently unclosed, so the template's own closer completes
    nothing and every later block reads as stripped -- a total-loss warning for a row the template
    kept whole.
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
        # a BALANCED pair quoted inside the reasoning: reading the span as ending at the inner
        # closer places its end well before the real one, the same overstatement by another route.
        ("balanced pair", "start <think>x</think> "),
    ],
)
def test_a_closing_tag_quoted_inside_reasoning_does_not_end_the_block_early(
    capsys, label: str, quoted: str
) -> None:
    """The block ends at the closer the TEMPLATE emits, not one the reasoning quotes. Ending the
    span early puts its end before the real one, so a cap between the two calls a cut block
    retained.

    The cap has to sit BETWEEN the candidate ends to discriminate them: on this fake (one token per
    character) an early end lands at 60 and the real closer at 132/141, so 100 is truncated under
    the correct rule and retained under an early one. A cap below both cannot fail.
    """
    completion = [
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": quoted + "rest " * 12,
        }
    ]
    prepared = _thinking_prepared_env(ThinkingEnvironment(completion), max_context_tokens=100)

    # the cap cuts the block, and measuring to the real closer is what sees that
    assert prepared.authored_reasoning_turns == 1
    assert prepared.truncated_reasoning_spans == 1
    assert "max_context_tokens cut 1 rendered reasoning" in capsys.readouterr().err


def test_an_empty_reasoning_field_is_authoritative_over_a_tag_the_answer_quotes(capsys) -> None:
    """The template renders a STRING ``reasoning_content`` and leaves ``content`` whole, so an empty
    field means the turn authored nothing however the answer is written. Falling back to inline
    detection reads a quoted ``<think>`` as this turn's reasoning; the marker then lands outside the
    empty block the template owns and the turn reports as dropped.
    """
    completion = [
        {
            "role": "assistant",
            "reasoning_content": "   ",
            "content": "use <think>like this</think> in prompts",
        }
    ]
    prepared = _thinking_prepared(completion)

    assert prepared.authored_reasoning_turns == 0
    assert prepared.rendered_reasoning_spans == 0
    assert "reasoning" not in capsys.readouterr().err


@pytest.mark.parametrize(
    ("label", "reasoning"),
    [
        ("assistant header", "the format is:\n<|im_start|>assistant\n<think>\nand so on"),
        ("turn terminator", "the turn ends with <|im_end|> always"),
        ("a whole turn boundary", "before <|im_end|>\n<|im_start|>user\n after"),
    ],
)
def test_reasoning_content_rejects_reserved_chatml_layout(label, reasoning) -> None:
    completion = [{"role": "assistant", "reasoning_content": reasoning, "content": "answer"}]

    with pytest.raises(ValueError, match="reserved ChatML control token") as error:
        _thinking_prepared(completion)

    assert "message body" in str(error.value), label


def test_a_real_loss_is_reported_beside_a_surviving_turn(capsys) -> None:
    """An ordinary user turn resets the template's ``last_query_index``, so the first turn's
    reasoning is definitively stripped while the final turn is kept. Each is answered by its own
    marker.
    """

    class KnownLossEnvironment(ThinkingEnvironment):
        def sft_completion(self, row):
            return [
                # an ordinary user turn follows, so the template strips this reasoning outright
                {"role": "assistant", "reasoning_content": "early", "content": "a1"},
                {"role": "user", "content": "next"},
                {"role": "assistant", "reasoning_content": "late", "content": "a2"},
            ]

    prepared = _thinking_prepared_env(KnownLossEnvironment([], prompt="board"))

    assert prepared.authored_reasoning_turns == 2
    assert prepared.rendered_reasoning_spans == 1
    assert "the chat template dropped" in capsys.readouterr().err


def test_a_quoted_think_closer_does_not_bound_the_block_short(capsys) -> None:
    """The marker sits at the end of the reasoning body, so the closing tag is found by searching
    FORWARD from it and every quoted closer lies behind it. A short bound would score a block the
    cap cuts as fully retained, the direction that hides the loss.

    Asserted at the cap boundary, where the answers differ: one token under the block's real end
    reports truncation, one token over does not.
    """
    completion = [
        {
            "role": "assistant",
            # a quoted closer appears before the actual end of the reasoning body
            "reasoning_content": "start\n</think>\n\n middle end",
            "content": "ANSWER",
        }
    ]
    kept = _thinking_prepared(completion, max_context_tokens=512)

    # the row reached the measurement rather than being dropped whole
    assert len(kept.rows) == 1
    assert kept.authored_reasoning_turns == 1
    assert kept.rendered_reasoning_spans == 1
    assert kept.truncated_reasoning_spans == 0
    assert "reasoning" not in capsys.readouterr().err


def test_a_turn_index_is_terminated_so_turn_one_is_not_read_inside_turn_ten(capsys) -> None:
    """``{prefix}1`` is a substring of ``{prefix}10`` unless the index is terminated, so a dropped
    early turn reads as surviving whenever a later turn whose index starts with the same digits
    does. This transcript strips every turn but the last, whose index shares a prefix with an
    earlier one.
    """

    class ManyTurnEnvironment(ThinkingEnvironment):
        def sft_completion(self, row):
            turns = []
            for index in range(6):
                turns.append(
                    {"role": "assistant", "reasoning_content": f"r{index}", "content": f"a{index}"}
                )
                if index < 5:
                    turns.append({"role": "user", "content": "next"})
            return turns

    prepared = _thinking_prepared_env(ManyTurnEnvironment([], prompt="board"))

    # only the final turn sits after the last user message, so exactly one survives
    assert prepared.authored_reasoning_turns == 6
    assert prepared.rendered_reasoning_spans == 1
    assert "the chat template dropped 5 of 6" in capsys.readouterr().err


def test_reasoning_loss_is_measured_over_the_rows_the_horizon_reaches(capsys) -> None:
    """``max_steps`` can stop a run before it loads every row, and only loaded rows can lose. The
    counts describe the retained dataset, but the optimizer stops at ``authoritative_steps *
    examples_per_update``, so reasoning past that point never reaches training and warning about it
    names a remedy for a loss the run cannot suffer. Authoritative TOKEN accounting is already
    bounded this way.

    The rows are ordered by the seeded shuffle the trainer itself uses, so the prefix the horizon
    reaches is the prefix the optimizer really trains on.
    """
    consumed = set(_training_order(4)[:2])

    class TailLossEnvironment(ThinkingEnvironment):
        def __init__(self) -> None:
            super().__init__([], prompt="board")
            self._rows = [{"prompt": f"board{index}", "answer": "ignored"} for index in range(4)]

        def sft_completion(self, row):
            if row["prompt"] in consumed:
                return [{"role": "assistant", "content": "plain answer"}]
            # never reached at max_steps=1: the template strips the first turn's reasoning
            return [
                {"role": "assistant", "content": "<think>lost</think>a1"},
                {"role": "user", "content": "next"},
                {"role": "assistant", "content": "<think>kept</think>a2"},
            ]

    def prepared(max_steps: int):
        base = _spec()
        spec = _rebuild_digest(
            replace(
                base,
                thinking=True,
                train=replace(
                    base.train, max_context_tokens=512, max_examples=0, max_steps=max_steps
                ),
            )
        )
        return prepare_sft_workload(
            spec,
            TailLossEnvironment(),
            tokenizer_loader=lambda _model, _revision: ThinkingTokenizer(),
            producer_version="1.2.3",
            packing_support=lambda _model, _revision: ("pure-attention", True),
        )

    # the control: an unbounded run does reach the lossy rows, so the warning is owed
    unbounded = prepared(0)
    assert unbounded.profile.authoritative_steps == 4
    assert "the chat template dropped" in capsys.readouterr().err

    # one update consumes only the plain prefix, so there is no loss to report
    bounded = prepared(1)
    assert bounded.profile.authoritative_steps == 1
    assert bounded.profile.examples_per_update == 2
    # the retained dataset is unchanged; only the WARNING is bounded to the horizon
    assert len(bounded.rows) == 4
    assert bounded.authored_reasoning_turns == unbounded.authored_reasoning_turns
    assert "the chat template dropped" not in capsys.readouterr().err


def test_a_tool_response_bounds_the_span_of_the_turn_before_it(capsys) -> None:
    """A ``<tool_response>`` user turn does not reset the template's ``last_query_index``, so the
    assistant before it KEEPS its reasoning. A scan advancing its horizon only at the next REASONING
    turn leaves that block unbounded: the span runs through the answer and the turn terminator into
    the tool output and ends at a closer the tool text quotes, so a cap between the two scores
    intact reasoning as cut.
    """

    class ToolEnvironment(ThinkingEnvironment):
        def sft_completion(self, row):
            return [
                {"role": "assistant", "reasoning_content": "REAL", "content": "a1"},
                # tool output that happens to contain the template's own closer layout
                {
                    "role": "user",
                    "content": "<tool_response>before\n</think>\n\nafter</tool_response>",
                },
            ]

    # between the block's real closer (76) and the closer quoted in the tool output (140): the
    # unbounded span would reach past this cap, the correctly bounded one ends well before it.
    prepared = _thinking_prepared_env(ToolEnvironment([]), max_context_tokens=100)

    assert prepared.authored_reasoning_turns == 1
    assert prepared.rendered_reasoning_spans == 1
    # the block fits: it ends at its own closer, not at the one quoted in the tool output
    assert prepared.profile.truncated_reasoning_spans == 0
    assert "cut off" not in capsys.readouterr().err


def test_a_cap_landing_on_the_answer_separator_does_not_report_lost_reasoning(capsys) -> None:
    """Truncation is judged at the CLOSING TAG, not the blank line after it. The ``\n\n`` separator
    is not reasoning and costs a real token, so a cap landing between the tag and the answer retains
    every reasoning token while a span measured to its own end reads as cut.
    """
    completion = [{"role": "assistant", "content": "<think>short</think>answer text"}]
    # the fake is one token per character: the closing tag ends at 77 and the span at 79, so this
    # cap retains the whole block and only a span-end measurement would call it truncated. a cap
    # below 77 would cut the block under either rule and could not tell them apart.
    prepared = _thinking_prepared_env(ThinkingEnvironment(completion), max_context_tokens=77)

    assert prepared.authored_reasoning_turns == 1
    assert prepared.rendered_reasoning_spans == 1
    assert prepared.truncated_reasoning_spans == 0
    assert "max_context_tokens cut" not in capsys.readouterr().err


def test_reasoning_sampled_without_an_opening_tag_still_counts_as_authored(capsys) -> None:
    """``reasoned</think>answer`` is reasoning: the PROMPT supplied the opening tag, and
    ``flash/serve/request/thinking.py`` recognises the same shape. Requiring a balanced pair leaves such a
    turn out of the authored denominator entirely, so an early turn the template strips goes
    unreported.
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
    """The template stamps ``<think>\n\n</think>`` onto qualifying trailing assistant turns, so it
    comes back in any transcript captured from a previous render. Counting it as authored marks a
    block the real render does not have, and the row reports having lost ALL of its reasoning.
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
    """The marker must land inside the block the template keeps, not merely near a ``<think>``. If
    it lands outside it never reaches the render, and the row reports a drop for reasoning that
    reached the loss.
    """
    prepared = _thinking_prepared([{"role": "assistant", "content": content}])

    assert prepared.authored_reasoning_turns == 1
    assert prepared.rendered_reasoning_spans == 1, label
    assert "authored reasoning blocks" not in capsys.readouterr().err


def test_a_quoting_answer_on_a_dropped_turn_is_not_credited_as_a_survivor(capsys) -> None:
    """The early turn supplies reasoning through ``reasoning_content`` and its ANSWER quotes the
    ``<think>`` format. The template strips the reasoning but renders the quote verbatim, so any
    scheme asking "did perturbing this turn change the span count?" sees the quote move and credits
    the turn. Survival is an identity question: the quote is not this turn's reasoning.
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
    prepared = _thinking_prepared_env(ThinkingEnvironment(completion))

    assert prepared.authored_reasoning_turns == 2
    # only the final turn's reasoning reaches the loss; the quoted tags are answer text
    assert prepared.rendered_reasoning_spans == 1
    assert "dropped 1 of 2 authored reasoning blocks" in capsys.readouterr().err


def test_an_image_rows_visual_tokens_are_charged_against_the_reasoning_cap() -> None:
    """Training truncates the ids the PROCESSOR produced, in which each image has expanded into many
    visual tokens the text render does not contain. Measuring a reasoning block against the raw cap
    calls it retained when the expansion has already pushed it past the end of the supervised span.
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
            text = ""
            for message in messages:
                reasoning = message.get("reasoning_content")
                if isinstance(reasoning, str):
                    text += f"<think>{reasoning}</think>"
                content = message.get("content")
                blocks = (
                    content
                    if isinstance(content, list)
                    else [{"type": "text", "text": str(content or "")}]
                )
                text += "".join(
                    block.get("text") or "" for block in blocks if isinstance(block, dict)
                )
            ids = [3 + ord(char) % 89 for char in text] + [7] * self.visual
            return {"input_ids": [ids], "attention_mask": [[1] * len(ids)]}

    prepared = _thinking_prepared_env(
        ImageEnvironment(),
        max_context_tokens=200,
        # image-bearing rows are refused outright for a model the catalog says cannot train on them
        model="Qwen/Qwen3.5-9B",
        processor_loader=lambda _model, _revision: ExpandingProcessor(),
    )

    # the block closes 40 characters into the text render, well inside the 200-token cap, but the
    # image's 180 visual tokens are charged first and leave it past the end of the supervised span.
    # the template rendered it, so the loss is attributed to the CAP, not to the transcript shape
    assert prepared.authored_reasoning_turns == 1
    assert prepared.rendered_reasoning_spans == 1
    assert prepared.truncated_reasoning_spans == 1


@pytest.mark.parametrize(
    ("label", "completion"),
    [
        (
            "reasoning_content",
            [{"role": "assistant", "reasoning_content": "reason" + " " * 64, "content": "answer"}],
        ),
        (
            "inline span",
            [{"role": "assistant", "content": "<think>reason" + " " * 64 + "</think>answer"}],
        ),
        (
            "trailing newline",
            [{"role": "assistant", "reasoning_content": "reason\n", "content": "answer"}],
        ),
    ],
)
def test_trailing_whitespace_in_reasoning_is_not_charged_against_the_cap(
    capsys, label, completion
) -> None:
    """The template ``|trim``s the reasoning, so the marker must land on the TRIMMED body. Appended
    after trailing whitespace it shields that whitespace from the trim, so the marked render carries
    a run the real one drops and tokenizes longer through the closing tag -- the measurement's own
    bytes deciding the answer.

    The cap is the real block's exact token end, the only value where the two rules differ: one
    token higher and neither reports truncation.
    """
    # the fake is one token per character; the trimmed block closes at 78 for every shape here, so
    # this cap retains it exactly. an untrimmed marker pushes the measured end past the cap while
    # the real block still fits, which is the misreport this pins.
    prepared = _thinking_prepared_env(ThinkingEnvironment(completion), max_context_tokens=78)

    assert prepared.authored_reasoning_turns == 1, label
    assert prepared.rendered_reasoning_spans == 1, label
    assert prepared.truncated_reasoning_spans == 0, label
    assert "max_context_tokens cut" not in capsys.readouterr().err


def test_the_marker_stem_search_does_not_scan_once_per_character() -> None:
    """A valid row must not cost quadratic time before tokenization begins. Text holding the stem
    followed by N filler characters keeps every one-character extension a substring, so growing the
    stem one at a time walks the whole row N times. Profiling runs on the control plane ahead of
    tokenization, so a large packaged row stalls it rather than failing anything.

    Two assertions, because either alone passes on a broken implementation. The equivalence cases
    fix WHAT is returned -- the shortest absent stem, so a faster search cannot hand back a longer
    marker. The budget fixes what it COSTS: the one-at-a-time loop needs 21s on the input below and
    a scan-bounded one needs milliseconds.
    """
    stem = "flashreasoningmark"

    def shortest_absent(text: str) -> str:
        prefix = stem
        while prefix in text:
            prefix += "x"
        return prefix

    for filler in (0, 1, 5, 37, 500):
        text = f"a{stem}{'x' * filler}b"
        assert reasoning_marker_prefix(text) == shortest_absent(text)

    # a stem that appears more than once, at different run lengths, still resolves to the shortest
    text = f"{stem}xxx {stem}xxxxxxxxx {stem}"
    assert reasoning_marker_prefix(text) == shortest_absent(text)

    big = stem + "x" * 200_000
    started = time.perf_counter()
    marker = reasoning_marker_prefix(big)
    elapsed = time.perf_counter() - started
    assert marker not in big
    assert marker == stem + "x" * 200_001
    assert elapsed < 2.0, f"marker search took {elapsed:.1f}s, which is the per-character scan"
