"""[wandb] project / run_name config -> typed JobSpec.wandb the worker honors. No network.

`slm train cfg.toml --set wandb.project=… --set wandb.run_name=…` (or a `[wandb]` table) lets a
run land in its own W&B project / run name instead of the hardcoded `flash` /
`flash-<phase>-<run_id>-seed<N>` defaults. The naming is first-class spec config (round-tripped in
the job-spec JSON the worker reads), NOT environment variables; the WANDB_API_KEY secret stays env.
"""

from __future__ import annotations

import os

import pytest

from flash.schema import ConfigError, spec_from_dict, spec_from_file
from flash.spec import JobSpec, WandbSpec


def _base(**extra: object) -> dict:
    cfg = {
        "model": "openbmb/MiniCPM5-1B",
        "algorithm": "sft",
        "environment": {"id": "owner/some-env"},
        "train": {"seeds": [0], "hf_repo": "me/repo"},
    }
    cfg.update(extra)
    return cfg


def test_wandb_section_parses_into_spec():
    spec = spec_from_dict(_base(wandb={"project": "my-proj", "run_name": "my-run"}))
    assert spec.wandb.project == "my-proj"
    assert spec.wandb.run_name == "my-run"


def test_no_wandb_section_is_empty():
    spec = spec_from_dict(_base())
    assert spec.wandb.project is None
    assert spec.wandb.run_name is None


def test_partial_wandb_section():
    spec = spec_from_dict(_base(wandb={"project": "only-proj"}))
    assert spec.wandb.project == "only-proj"
    assert spec.wandb.run_name is None


def test_wandb_config_does_not_touch_worker_env():
    # [wandb] is its own typed field now, NOT folded into the worker_env env-var bag.
    spec = spec_from_dict(_base(wandb={"project": "p", "run_name": "r"}))
    assert "WANDB_PROJECT" not in spec.worker_env
    assert "WANDB_NAME" not in spec.worker_env


def test_wandb_values_are_trimmed():
    spec = spec_from_dict(_base(wandb={"project": "  p  ", "run_name": "  r  "}))
    assert spec.wandb.project == "p"
    assert spec.wandb.run_name == "r"


def test_wandb_round_trips_through_json():
    # The worker reads JOB_SPEC via JSON, so [wandb] must survive to_dict/from_dict.
    spec = spec_from_dict(_base(wandb={"project": "rt-proj", "run_name": "rt-run"}))
    rt = JobSpec.from_dict(spec.to_dict())
    assert rt.wandb.project == "rt-proj"
    assert rt.wandb.run_name == "rt-run"


def test_from_dict_tolerates_malformed_wandb_payload():
    # A malformed/older persisted spec can set wandb to a non-dict (e.g. a bare string); from_dict
    # must not crash the worker with AttributeError — it coerces to default naming instead.
    assert JobSpec.from_dict({"wandb": "oops"}).wandb == WandbSpec()
    assert JobSpec.from_dict({"wandb": None}).wandb == WandbSpec()
    assert JobSpec.from_dict({"wandb": ["a", "b"]}).wandb == WandbSpec()
    # Non-string / blank leaves in a dict payload are coerced+trimmed (blank -> default None).
    rt = JobSpec.from_dict({"wandb": {"project": 123, "run_name": "  "}})
    assert rt.wandb.project == "123"
    assert rt.wandb.run_name is None


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


def _toml(tmp_path) -> str:
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
    return str(cfg)


def test_set_override_reaches_wandb(tmp_path):
    # The CLI surface: `--set wandb.project=… --set wandb.run_name=…`.
    spec = spec_from_file(
        _toml(tmp_path), overrides=["wandb.project=cli-proj", "wandb.run_name=cli-run"]
    )
    assert spec.wandb.project == "cli-proj"
    assert spec.wandb.run_name == "cli-run"


@pytest.mark.parametrize("label", ["123", "1e-4", "true", "false"])
def test_set_override_preserves_numeric_looking_wandb_label(tmp_path, label):
    # A numeric- or bool-looking --set wandb.* value is the string label the user intends; it must
    # not be coerced to int/float/bool (which the [wandb] validator rejects).
    spec = spec_from_file(_toml(tmp_path), overrides=[f"wandb.run_name={label}"])
    assert spec.wandb.run_name == label


def test_worker_run_name_uses_spec_wandb(monkeypatch):
    # Primary path: the worker reads the run name from the typed spec (no env var).
    from flash.engine import worker

    monkeypatch.delenv("WANDB_NAME", raising=False)
    monkeypatch.setattr(worker, "JOB_SPEC", JobSpec(wandb=WandbSpec(run_name="from-spec")))
    assert worker.wandb_run_name() == "from-spec"


def test_worker_run_name_is_toml_only_ignores_env(monkeypatch):
    # WANDB_NAME is fully gone: even if the env var is set, the run name comes only from the
    # [wandb] config (here unset -> the stable default). No env-var override path.
    from flash.engine import worker

    monkeypatch.setattr(worker, "JOB_SPEC", None)
    monkeypatch.setenv("WANDB_NAME", "should-be-ignored")
    assert worker.wandb_run_name().startswith("flash-")


def test_report_to_inits_wandb_from_spec_without_env(monkeypatch):
    # wandb_report_to initializes the run from the typed [wandb] config via the SDK and sets NO
    # WANDB_PROJECT env var — the env var is fully gone.
    import importlib.util
    import sys
    import types

    from flash.engine import worker

    calls: dict = {}
    fake = types.ModuleType("wandb")
    fake.run = None

    def _init(**kw):
        calls.update(kw)
        fake.run = object()

    fake.init = _init
    monkeypatch.setitem(sys.modules, "wandb", fake)
    _orig_find = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: (object() if name == "wandb" else _orig_find(name, *a, **k)),
    )
    monkeypatch.setenv("WANDB_API_KEY", "k")
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.setattr(
        worker, "JOB_SPEC", JobSpec(wandb=WandbSpec(project="my-proj", run_name="my-run"))
    )

    assert worker.wandb_report_to() == ["wandb"]
    assert calls["project"] == "my-proj"  # project from the spec, via wandb.init
    assert calls["name"] == "my-run"  # run name from the spec
    assert "WANDB_PROJECT" not in os.environ  # the env var is fully gone


def test_report_to_is_best_effort_when_wandb_init_fails(monkeypatch):
    # W&B logging is optional: a broken wandb install / init failure (auth, network) must NOT
    # abort training — wandb_report_to falls back to [] instead of propagating the exception.
    import importlib.util
    import sys
    import types

    from flash.engine import worker

    fake = types.ModuleType("wandb")
    fake.run = None

    def _boom(**kw):
        raise RuntimeError("wandb backend unreachable")

    fake.init = _boom
    monkeypatch.setitem(sys.modules, "wandb", fake)
    _orig_find = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: (object() if name == "wandb" else _orig_find(name, *a, **k)),
    )
    monkeypatch.setenv("WANDB_API_KEY", "k")
    monkeypatch.setattr(worker, "JOB_SPEC", JobSpec(wandb=WandbSpec(project="p", run_name="r")))

    assert worker.wandb_report_to() == []  # degrades to no W&B logging, no crash
