"""Adversarial and quoted values retain the public configuration contract."""

import pytest

from flash.schema import ConfigError, _apply_override, normalize_env_name_segment, spec_from_dict
from flash.schema.fields import _section_int


@pytest.mark.parametrize("value", [10**400, -(10**400)], ids=["huge-positive", "huge-negative"])
def test_huge_bounded_integer_is_a_configuration_error(value):
    with pytest.raises(ConfigError, match="between 1 and 8"):
        _section_int({"count": value}, "gpu", "count", minimum=1, maximum=8)


def test_integer_validation_does_not_round_through_float():
    value = 2**53 + 1
    assert _section_int({"steps": value}, "train", "steps", minimum=1) == value


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ('environment.pip=["requests>=2,<3", "numpy"]', ["requests>=2,<3", "numpy"]),
        ('train.stop_sequences=["END,STOP", "true", "42"]', ["END,STOP", "true", "42"]),
        ("train.stop_sequences=['literal']", ["literal"]),
        ("train.stop_sequences=[STOP1,STOP2]", ["STOP1", "STOP2"]),
    ],
)
def test_list_override_preserves_quoted_values(override, expected):
    raw = {}
    _apply_override(raw, override)
    section, field = override.split("=", 1)[0].split(".")
    assert raw[section][field] == expected


@pytest.mark.parametrize("value", [".My Env", "_My Env", "...__My Env", "--My Env", "._-My Env"])
def test_normalized_environment_name_starts_with_alphanumeric(value):
    assert normalize_env_name_segment(value) == "my-env"


@pytest.mark.parametrize("algorithm", ["sft", "grpo", "opd"])
@pytest.mark.parametrize("field", ["lora_rank", "max_context_tokens", "max_completion_tokens"])
def test_spec_rejects_dimensions_that_overflow_sizing(algorithm, field):
    with pytest.raises(ConfigError):
        spec_from_dict(
            {
                "model": "Qwen/Qwen3.5-9B",
                "algorithm": algorithm,
                "environment": {"id": "freesolo/math-agent/gsm8k"},
                "train": {field: 10**400},
            }
        )
