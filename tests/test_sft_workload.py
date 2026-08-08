from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

import pytest

from flash.engine.sft_workload import prepare_sft_workload, sft_tokens_for_updates
from flash.spec import EnvironmentSpec, JobSpec, TrainSpec
from flash.workload_profile import sft_profile_input_digest


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
    from flash.engine import sft_workload

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
    from flash.engine import sft_workload

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
