"""[wandb] project / run_name config -> W&B env the worker honors. No network.

`slm train cfg.toml --set wandb.project=… --set wandb.run_name=…` (or a `[wandb]`
table) lets a run land in its own W&B project / run name instead of the hardcoded
`flash` / `flash-<phase>-<run_id>-seed<N>` defaults.
"""

from __future__ import annotations

import pytest

from flash.schema import ConfigError, spec_from_dict, spec_from_file


def _base(**extra: object) -> dict:
    cfg = {
        "model": "openbmb/MiniCPM5-1B",
        "algorithm": "sft",
        "environment": {"id": "owner/some-env"},
        "train": {"seeds": [0], "hf_repo": "me/repo"},
    }
    cfg.update(extra)
    return cfg


def test_wandb_section_folds_into_worker_env():
    spec = spec_from_dict(_base(wandb={"project": "my-proj", "run_name": "my-run"}))
    assert spec.worker_env["WANDB_PROJECT"] == "my-proj"
    assert spec.worker_env["WANDB_NAME"] == "my-run"


def test_no_wandb_section_leaves_worker_env_clean():
    spec = spec_from_dict(_base())
    assert "WANDB_PROJECT" not in spec.worker_env
    assert "WANDB_NAME" not in spec.worker_env


def test_partial_wandb_section():
    spec = spec_from_dict(_base(wandb={"project": "only-proj"}))
    assert spec.worker_env["WANDB_PROJECT"] == "only-proj"
    assert "WANDB_NAME" not in spec.worker_env


def test_explicit_worker_env_wins_over_wandb_section():
    # [worker_env] is the lower-level escape hatch; [wandb] must not clobber it.
    spec = spec_from_dict(
        _base(worker_env={"WANDB_PROJECT": "explicit"}, wandb={"project": "from-wandb"})
    )
    assert spec.worker_env["WANDB_PROJECT"] == "explicit"


def test_wandb_values_are_trimmed():
    spec = spec_from_dict(_base(wandb={"project": "  p  ", "run_name": "  r  "}))
    assert spec.worker_env["WANDB_PROJECT"] == "p"
    assert spec.worker_env["WANDB_NAME"] == "r"


@pytest.mark.parametrize(
    "bad",
    [
        {"project": 123},
        {"project": ""},
        {"run_name": "   "},
        {"unknown": "x"},
    ],
)
def test_invalid_wandb_section_rejected(bad):
    with pytest.raises(ConfigError, match="wandb"):
        spec_from_dict(_base(wandb=bad))


def test_wandb_section_must_be_a_table():
    with pytest.raises(ConfigError, match=r"\[wandb\] must be a table"):
        spec_from_dict(_base(wandb="my-project"))


def test_set_override_reaches_wandb(tmp_path):
    # The CLI surface: `--set wandb.project=… --set wandb.run_name=…`.
    cfg = tmp_path / "run.toml"
    cfg.write_text(
        'model = "openbmb/MiniCPM5-1B"\n'
        'algorithm = "sft"\n'
        "[environment]\n"
        'id = "owner/some-env"\n'
        "[train]\n"
        "seeds = [0]\n"
        'hf_repo = "me/repo"\n'
    )
    spec = spec_from_file(str(cfg), overrides=["wandb.project=cli-proj", "wandb.run_name=cli-run"])
    assert spec.worker_env["WANDB_PROJECT"] == "cli-proj"
    assert spec.worker_env["WANDB_NAME"] == "cli-run"


def test_worker_run_name_honors_override(monkeypatch):
    from flash.engine import worker

    monkeypatch.setenv("WANDB_NAME", "custom-name")
    assert worker.wandb_run_name() == "custom-name"
    monkeypatch.setenv("WANDB_NAME", "   ")  # blank -> fall back
    assert worker.wandb_run_name().startswith("flash-")
    monkeypatch.delenv("WANDB_NAME", raising=False)
    assert worker.wandb_run_name().startswith("flash-")
