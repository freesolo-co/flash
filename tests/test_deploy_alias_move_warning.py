"""Deploying a second checkpoint of a run must say it is moving the shared alias.

`deploy` registers every checkpoint under the bare run id, so deploying `RUN/step-50` after
`RUN/step-100` repoints the one alias rather than standing up a second endpoint. Everyone
chatting bare `RUN` changes model with no signal, which is what made a redeploy read as a
serving regression instead of the deploy that caused it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import flash.cli.commands as commands
from flash.cli.commands.deploy import _alias_move_warning, cmd_deploy
from flash.client import ApiError, ClientError


class _Client:
    """A client whose deployment read records how it was asked."""

    def __init__(self, current=None, *, raises=None) -> None:
        self._current = current
        self._raises = raises
        self.deploy_calls: list[tuple] = []
        self.read_calls: list[tuple] = []

    def deployed_checkpoint(self, run_id, timeout=None):
        self.read_calls.append(run_id)
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


@pytest.mark.parametrize("failure", [ApiError(500, "boom"), ClientError("unreachable")])
def test_an_unreadable_plane_produces_no_warning_instead_of_an_error(failure) -> None:
    """The read is advisory: a plane that cannot answer must not fail the deploy it decorates."""
    assert _alias_move_warning(_Client(raises=failure), "flash-1", 50) is None


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
