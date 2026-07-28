"""A configured [wandb] block with no discoverable key must not train silently unlogged.

``WANDB_API_KEY`` is an optional runtime secret: discovery looks at the process env and the
``.env``/``.env.local`` files beside the cwd and the config, and an absent optional secret is not an
error. So a key one directory up (or a run launched from another cwd) trains to completion with
logging off, which is only discoverable after the GPU spend, when the curve is gone.
"""

from __future__ import annotations

from flash.cli.commands import _warn_if_wandb_requested_without_key
from flash.spec import JobSpec, WandbSpec


def _spec(**wandb_fields) -> JobSpec:
    return JobSpec(wandb=WandbSpec(**wandb_fields))


def test_warns_when_wandb_configured_without_key(capsys):
    _warn_if_wandb_requested_without_key(_spec(project="my-project"), {})

    err = capsys.readouterr().err
    assert "WANDB_API_KEY" in err
    assert "DISABLED" in err


def test_warns_when_only_run_name_is_set(capsys):
    # a run_name alone still means the user expects to find this run in w&b
    _warn_if_wandb_requested_without_key(_spec(run_name="experiment-7"), None)

    assert "WANDB_API_KEY" in capsys.readouterr().err


def test_silent_when_key_is_discovered(capsys):
    _warn_if_wandb_requested_without_key(
        _spec(project="my-project"), {"WANDB_API_KEY": "wandb-key"}
    )

    assert capsys.readouterr().err == ""


def test_silent_when_wandb_not_configured(capsys):
    # no [wandb] block means the user never asked for logging, so there is nothing to warn about
    _warn_if_wandb_requested_without_key(_spec(), None)

    assert capsys.readouterr().err == ""
