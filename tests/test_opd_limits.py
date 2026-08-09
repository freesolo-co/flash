from __future__ import annotations

import pytest

from flash.core.spec import EnvironmentSpec
from flash.teacher.limits import configured_opd_turn_limit, opd_teacher_request_multiplier


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        (EnvironmentSpec(params={"multi_turn": False}), (False, None)),
        (EnvironmentSpec(params={"multi_turn": True, "max_turns": 24}), (True, 24)),
        (EnvironmentSpec(params={"max_turns": 4}), (True, 8)),
        (EnvironmentSpec(params={"max_turns": 100}), (True, 64)),
        (EnvironmentSpec(id="owner/external"), (True, None)),
        (EnvironmentSpec(), (False, None)),
    ],
)
def test_configured_opd_turn_limit_is_conservative_for_unknown_external_environments(
    environment,
    expected,
):
    assert configured_opd_turn_limit(environment) == expected


def test_unknown_external_environment_uses_the_enforced_turn_ceiling():
    multi_turn, max_turns = configured_opd_turn_limit(EnvironmentSpec(id="owner/external"))

    assert (
        opd_teacher_request_multiplier(
            multi_turn=multi_turn,
            max_turns=max_turns,
        )
        == 64 * 3
    )
