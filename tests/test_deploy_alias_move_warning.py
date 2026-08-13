"""Deploying a second checkpoint of a run must say it is moving the shared alias.

`deploy` registers every checkpoint under the bare run id, so deploying `RUN/step-50` after
`RUN/step-100` repoints the one alias rather than standing up a second endpoint. Everyone
chatting bare `RUN` changes model with no signal, which is what made a redeploy read as a
serving regression instead of the deploy that caused it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import flash.cli.commands as commands
import flash.cli.commands.deploy as deploy_module
from flash.cli.commands.deploy import _alias_move_warning, cmd_deploy
from flash.client import ApiError, ClientError


class _Client:
    """A client whose deployment read records how it was asked."""

    def __init__(self, current=None, *, raises=None) -> None:
        self._current = current
        self._raises = raises
        self.deploy_calls: list[tuple] = []
        self.read_calls: list[tuple] = []
        self.read_timeouts: list[float | None] = []

    def deployed_checkpoint(self, run_id, timeout=None):
        self.read_calls.append(run_id)
        self.read_timeouts.append(timeout)
        if self._raises is not None:
            raise self._raises
        return self._current

    def deploy(self, run_id, dry_run=False):
        self.deploy_calls.append((run_id, dry_run))
        return {"state": "queued", "run_id": run_id}


def _args(run_id: str, *, dry_run: bool = False):
    return SimpleNamespace(run_id=run_id, dry_run=dry_run, wait=None)


def _ready(step):
    return {"run_id": "flash-1", "state": "ready", "checkpoint_step": step}


def test_warns_that_deploying_another_step_moves_the_shared_alias() -> None:
    """The warning must name both checkpoints: which one is lost and which one takes over."""
    client = _Client(_ready(100))

    warning = _alias_move_warning(client, "flash-1", 50)

    assert warning is not None
    assert "step-100" in warning
    assert "step-50" in warning
    assert "flash-1" in warning


def test_warning_reads_the_deployment_without_the_step_filter() -> None:
    """The displaced record is exactly the one `deployment_for`'s step filter would hide."""
    client = _Client(_ready(100))

    _alias_move_warning(client, "flash-1", 50)

    assert client.read_calls == ["flash-1"]


def test_redeploying_the_same_checkpoint_displaces_nothing() -> None:
    """Re-deploying the live checkpoint moves the alias onto what it already serves."""
    assert _alias_move_warning(_Client(_ready(100)), "flash-1", 100) is None


def test_final_adapter_is_named_final_not_step_none() -> None:
    """`checkpoint_step: None` is the final adapter; a user never typed `step-None`."""
    warning = _alias_move_warning(_Client(_ready(None)), "flash-1", 50)

    assert warning is not None
    assert "serves final" in warning
    assert "None" not in warning


def test_deploying_final_over_a_checkpoint_warns_in_the_other_direction() -> None:
    """The final adapter replacing step-100 moves the alias just as much."""
    warning = _alias_move_warning(_Client(_ready(100)), "flash-1", None)

    assert warning is not None
    assert "serves step-100" in warning
    assert "deploying final" in warning


def test_a_run_with_no_deployment_is_not_a_replacement() -> None:
    """A first deploy displaces nothing, so it must stay silent."""
    assert _alias_move_warning(_Client(None), "flash-1", 50) is None


@pytest.mark.parametrize("state", ["queued", "failed", "reconciling", "revocation_failed"])
def test_only_a_servable_revision_can_be_displaced(state: str) -> None:
    """Nothing is serving off the alias yet, so moving it costs nobody their model."""
    current = {"run_id": "flash-1", "state": state, "checkpoint_step": 100}

    assert _alias_move_warning(_Client(current), "flash-1", 50) is None


def _unsettled(step):
    """A `reconciling` record whose activation outcome was never recorded.

    Its `checkpoint_step` is the INCOMING attempt. The plane resolves what the alias really holds
    from `adapter_alias_target` / `previous_deployment`, neither of which reaches the client.
    """
    return {
        "run_id": "flash-1",
        "state": "reconciling",
        "checkpoint_step": step,
        "activation_outcome_unknown": True,
    }


def test_an_unsettled_activation_may_already_hold_the_alias() -> None:
    """`reconciling` + `activation_outcome_unknown` is the case the alias is MOST likely to move.

    The plane permits replacing exactly that record instead of rejecting it as busy, and resolves
    the authoritative target through `_activation_predecessor` when it does. Reading the state
    alone as "not serving" silenced the warning precisely where it is needed.
    """
    warning = _alias_move_warning(_Client(_unsettled(100)), "flash-1", 50)

    assert warning is not None
    assert "activation never settled" in warning
    assert "step-50" in warning


def test_an_unsettled_activation_does_not_name_a_checkpoint_it_cannot_know() -> None:
    """`checkpoint_step` there is the incoming attempt, so naming it would name the WRONG one.

    `tests/test_server_api.py` covers an attempted final revision while the alias still serves
    step-20: the record's own step describes neither what is live nor what will be displaced, and
    the predecessor that does is stripped by `public_deployment`. Saying "serves step-N" from this
    field is a confident wrong answer, which is worse than declining to name one.
    """
    warning = _alias_move_warning(_Client(_unsettled(100)), "flash-1", 50)

    assert warning is not None
    assert "step-100" not in warning
    assert "serves step-" not in warning
    assert "cannot be determined" in warning


def test_an_unsettled_activation_warns_even_on_the_same_step() -> None:
    """Equality with an untrustworthy step proves nothing, so it cannot buy silence.

    A settled record suppresses the warning on a same-step redeploy because the step is then a
    fact. Here it is the incoming attempt, so the alias may still be on something else entirely.
    """
    warning = _alias_move_warning(_Client(_unsettled(50)), "flash-1", 50)

    assert warning is not None
    assert "activation never settled" in warning


def test_a_settled_ready_record_is_stated_as_fact() -> None:
    """A confirmed `ready` revision is hedged by nothing: it IS what the id serves."""
    warning = _alias_move_warning(_Client(_ready(100)), "flash-1", 50)

    assert warning is not None
    assert "currently serves step-100" in warning
    assert "may currently serve" not in warning


def test_a_numeric_string_step_still_compares_as_a_number() -> None:
    """A plane that JSON-encodes the step as a string must not read as a different checkpoint."""
    current = {"run_id": "flash-1", "state": "ready", "checkpoint_step": "50"}

    assert _alias_move_warning(_Client(current), "flash-1", 50) is None


@pytest.mark.parametrize("failure", [ApiError(500, "boom"), ClientError("unreachable")])
def test_an_unreadable_plane_produces_no_warning_instead_of_an_error(failure) -> None:
    """The read is advisory: a plane that cannot answer must not fail the deploy it decorates."""
    assert _alias_move_warning(_Client(raises=failure), "flash-1", 50) is None


def test_the_advisory_read_is_bounded_well_under_the_client_default() -> None:
    """An advisory line must not hold a real deploy for the client's full 60s default.

    Unbounded, a stalled status GET delays every deploy behind a warning nobody asked for.
    """
    client = _Client(_ready(100))

    _alias_move_warning(client, "flash-1", 50)

    assert client.read_timeouts == [deploy_module._ALIAS_WARNING_READ_SECONDS]
    assert deploy_module._ALIAS_WARNING_READ_SECONDS < 60.0


@pytest.mark.parametrize(
    "bad_step",
    [
        "abc",
        "",
        [],
        {},
        object(),
        # `json.loads` accepts the non-standard `NaN`/`Infinity` literals, so a single 2xx body
        # reaches int() with each of these. they raise ValueError and OverflowError respectively,
        # and a guard that catches only the obvious exceptions still crashes the deploy.
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_an_unreadable_checkpoint_step_cannot_fail_the_deploy(bad_step) -> None:
    """A 2xx carrying a step this client cannot parse is an unreadable record, not a crash.

    A proxy or older plane can answer with a nonnumeric `checkpoint_step`. Converting it outside
    the guarded read let the advisory warning traceback out of the command it decorates.
    """
    current = {"run_id": "flash-1", "state": "ready", "checkpoint_step": bad_step}

    assert _alias_move_warning(_Client(current), "flash-1", 50) is None


def test_a_non_finite_step_is_reachable_from_a_real_response_body() -> None:
    """Anchor the case above in the decoder the client actually uses, not an invented value.

    `float("inf")` in a parametrize list only proves the guard catches a value a test made up.
    The client decodes with `json.loads`, which accepts the non-standard `Infinity` and `NaN`
    literals, so a control plane emitting either really does hand this code a non-finite step.
    """
    decoded = json.loads(b'{"state": "ready", "checkpoint_step": Infinity}')

    assert decoded["checkpoint_step"] == float("inf")
    with pytest.raises(OverflowError):
        int(decoded["checkpoint_step"])
    assert _alias_move_warning(_Client({**decoded, "run_id": "flash-1"}), "flash-1", 50) is None


def test_a_numeric_string_step_is_still_compared_as_a_number() -> None:
    """Normalizing must not turn `"100"` into a spurious warning against step 100."""
    current = {"run_id": "flash-1", "state": "ready", "checkpoint_step": "100"}

    assert _alias_move_warning(_Client(current), "flash-1", 100) is None
    warning = _alias_move_warning(_Client(current), "flash-1", 50)
    assert warning is not None
    assert "serves step-100" in warning


def test_a_malformed_step_does_not_stop_the_deploy_itself(monkeypatch, capsys) -> None:
    """End to end: the deploy still goes through, with no traceback and no warning."""
    client = _Client({"run_id": "flash-1", "state": "ready", "checkpoint_step": "abc"})
    monkeypatch.setattr(commands, "client_from_config", lambda: client)
    monkeypatch.setattr(commands.render, "styled", lambda: False)

    assert cmd_deploy(_args("flash-1/step-50")) == 0

    assert "currently serves" not in capsys.readouterr().err
    assert client.deploy_calls == [("flash-1/step-50", False)]


def test_deployed_state_counts_as_servable() -> None:
    """`deployed` serves just like `ready`; missing it would silence a real replacement."""
    current = {"run_id": "flash-1", "state": "deployed", "checkpoint_step": 100}

    assert _alias_move_warning(_Client(current), "flash-1", 50) is not None


def test_deploy_warns_before_it_moves_the_alias(monkeypatch, capsys) -> None:
    """After the POST the alias has already moved; the warning has to precede it."""
    client = _Client(_ready(100))
    monkeypatch.setattr(commands, "client_from_config", lambda: client)
    monkeypatch.setattr(commands.render, "styled", lambda: False)

    assert cmd_deploy(_args("flash-1/step-50")) == 0

    err = capsys.readouterr().err
    assert "warning:" in err
    assert "step-100" in err
    # the read happened, and the deploy still went through: this warns, it does not block.
    assert client.read_calls == ["flash-1"]
    assert client.deploy_calls == [("flash-1/step-50", False)]


def test_deploy_points_at_the_explicit_checkpoint_selector(monkeypatch, capsys) -> None:
    """Naming the fix matters: `RUN/step-N` compares checkpoints without fighting over the alias."""
    monkeypatch.setattr(commands, "client_from_config", lambda: _Client(_ready(100)))
    monkeypatch.setattr(commands.render, "styled", lambda: False)

    cmd_deploy(_args("flash-1/step-50"))

    assert "models chat flash-1/step-N" in capsys.readouterr().err


def test_a_dry_run_registers_nothing_so_it_warns_about_nothing(monkeypatch, capsys) -> None:
    """`--dry-run` never touches the alias, so a replacement warning there is false."""
    client = _Client(_ready(100))
    monkeypatch.setattr(commands, "client_from_config", lambda: client)
    monkeypatch.setattr(commands.render, "styled", lambda: False)

    cmd_deploy(_args("flash-1/step-50", dry_run=True))

    assert "currently serves" not in capsys.readouterr().err
    assert client.read_calls == []


def test_first_deploy_of_a_run_says_nothing_about_aliases(monkeypatch, capsys) -> None:
    """A run with nothing deployed must not gain a warning about a checkpoint that does not exist."""
    monkeypatch.setattr(commands, "client_from_config", lambda: _Client(None))
    monkeypatch.setattr(commands.render, "styled", lambda: False)

    assert cmd_deploy(_args("flash-1/step-50")) == 0

    assert "currently serves" not in capsys.readouterr().err
