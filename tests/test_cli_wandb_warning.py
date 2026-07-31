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
    _warn_if_wandb_requested_without_key(_spec(project="my-project"), {}, dry_run=False)

    err = capsys.readouterr().err
    assert "WANDB_API_KEY" in err
    assert "DISABLED" in err


def test_warns_when_only_run_name_is_set(capsys):
    # a run_name alone still means the user expects to find this run in w&b
    _warn_if_wandb_requested_without_key(_spec(run_name="experiment-7"), None, dry_run=False)

    assert "WANDB_API_KEY" in capsys.readouterr().err


def test_silent_when_key_is_discovered(capsys):
    _warn_if_wandb_requested_without_key(
        _spec(project="my-project"), {"WANDB_API_KEY": "wandb-key"}, dry_run=False
    )

    assert capsys.readouterr().err == ""


def test_silent_when_wandb_not_configured(capsys):
    # no [wandb] block means the user never asked for logging, so there is nothing to warn about
    _warn_if_wandb_requested_without_key(_spec(), None, dry_run=False)

    assert capsys.readouterr().err == ""


def test_warns_when_key_is_whitespace_only(capsys):
    # the server's _runtime_secrets() strips and drops the value, so a whitespace-only key reaches
    # the worker as no key at all -- the silent-no-logging failure this warning exists to prevent.
    _warn_if_wandb_requested_without_key(
        _spec(project="my-project"), {"WANDB_API_KEY": "   "}, dry_run=False
    )

    assert "WANDB_API_KEY" in capsys.readouterr().err


def test_dry_run_does_not_claim_the_run_will_train(capsys):
    # a dry-run allocates no gpu and trains nothing; saying "this run will train" contradicts the
    # dry-run notice printed moments later and can read as though a paid run started.
    _warn_if_wandb_requested_without_key(_spec(project="my-project"), None, dry_run=True)

    err = capsys.readouterr().err
    assert "DISABLED" in err
    assert "this run will train" not in err
    assert "a run submitted with this config will train" in err


def test_supported_algorithms_still_get_the_export_remedy(capsys):
    """Every shipped algorithm reaches W&B through its trainer, so a key genuinely fixes its case."""
    from flash.catalog import ALGORITHMS

    for algorithm in ALGORITHMS:
        spec = JobSpec(algorithm=algorithm, wandb=WandbSpec(project="my-project"))

        _warn_if_wandb_requested_without_key(spec, None, dry_run=False)

        err = capsys.readouterr().err
        assert "Export WANDB_API_KEY" in err, algorithm
        assert "not supported" not in err, algorithm


def test_every_shipped_worker_actually_reaches_wandb():
    """Tell users to export a key only while every worker can really use one.

    The remedy has to stay truthful: if an algorithm ships whose worker makes no
    wandb_report_to()/wandb_run_info() call, "Export WANDB_API_KEY" would clear the warning while
    the paid run still logs nowhere. Assert against the worker modules themselves rather than a
    hand-maintained list, which is how the previous opsd entry outlived the algorithm it named.
    """
    import inspect
    from pathlib import Path

    from flash.catalog import ALGORITHMS
    from flash.engine import worker as worker_pkg

    worker_dir = Path(inspect.getfile(worker_pkg)).parent
    without_wandb = set()
    for algorithm in ALGORITHMS:
        # JobSpec.phase is the algorithm -> worker-module mapping (grpo runs the rl worker)
        module = worker_dir / f"{JobSpec(algorithm=algorithm).phase}.py"
        source = module.read_text()
        if "wandb_report_to" not in source and "wandb_run_info" not in source:
            without_wandb.add(algorithm)

    assert without_wandb == set()
