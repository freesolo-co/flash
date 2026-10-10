"""offline tests: an `eos` turn must END at an eos token, not merely contain one.

this validator is the fail-closed boundary for child-reported termination. a set-intersection test
accepts a backend that kept emitting after its own declared terminal boundary, and the parent then
records and teacher-scores that trailing span -- corrupting both the transcript and the loss targets.
"""

from __future__ import annotations

import pytest

from flash.engine.worker.train.opd.multiturn.validation import validated_multiturn_response

_EOS = 2


class _Tokenizer:
    """decodes ids to text, dropping the eos id when special tokens are skipped."""

    def decode(self, ids, skip_special_tokens=False):
        kept = [i for i in ids if not (skip_special_tokens and i == _EOS)]
        return "".join(f"t{i}" for i in kept)


def _payload(raw_ids, **overrides):
    tokenizer = _Tokenizer()
    payload = {
        "raw_response_ids": list(raw_ids),
        "response_ids": list(raw_ids),
        "completion_text": tokenizer.decode(raw_ids, skip_special_tokens=True),
        "termination": "eos",
    }
    payload.update(overrides)
    return payload


def _validate(payload):
    return validated_multiturn_response(
        payload,
        tokenizer=_Tokenizer(),
        eos_token_ids={_EOS},
        stop_sequences=[],
    )


def test_eos_as_the_final_token_is_accepted():
    raw, response, text, skip = _validate(_payload([7, 8, _EOS]))
    assert raw == [7, 8, _EOS]
    assert response == [7, 8, _EOS]
    assert text == "t7t8"
    assert skip == ""


def test_tokens_sampled_after_eos_are_rejected():
    """the exact defect: eos occurs in the span, so the intersection test passed it through."""
    with pytest.raises(ValueError, match="must end the sampled ids"):
        _validate(_payload([7, _EOS, 9]))


def test_an_eos_turn_with_no_sampled_ids_is_rejected():
    """an empty span cannot have ended at eos; indexing [-1] must not decide it."""
    with pytest.raises(ValueError, match="must end the sampled ids"):
        _validate(_payload([]))


def test_an_eos_turn_without_any_eos_token_is_still_rejected():
    with pytest.raises(ValueError, match="must end the sampled ids"):
        _validate(_payload([7, 8, 9]))


def test_an_eos_turn_must_still_preserve_the_sampled_span():
    """the boundary check must not displace the span-preservation check behind it."""
    payload = _payload([7, 8, _EOS], response_ids=[7, 8])
    with pytest.raises(ValueError, match="must preserve the sampled span"):
        _validate(payload)
