"""Cross-folder contract: a training-run config the backend (../backend) accepts must
map to a valid flash JobSpec and round-trip.

The platform backend accepts a training run via RunCreateRequest
(backend/src/schemas/runs.py), whose freeform `config` dict carries the training
spec the SDK/agent authored (model / algorithm / [environment] id / [train] params).
That same config shape is what flash parses into a JobSpec
(flash/flash/schema.py spec_from_dict -> flash/flash/spec.py JobSpec). If the
backend accepts a config flash rejects (or the two drift on field names), a run the
backend admits never trains. This test pins that seam.

Both packages import in the flash venv (the backend schemas are pure pydantic), so
we add ../backend to sys.path the way the e2e tests import across packages. If the
backend isn't importable, its half is skipped.

Run: cd flash && .venv/bin/python -m pytest \
        tests/test_backend_jobspec_contract.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[2] / "backend"
if _BACKEND_ROOT.is_dir() and str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from flash.schema import ConfigError, spec_from_dict
from flash.spec import JobSpec


def _representative_config() -> dict:
    """A representative training-run config the backend's RunCreateRequest.config
    carries — model, algorithm, environment id, and train params — matching what the
    SDK/agent author into flash.toml."""
    return {
        "model": "Qwen/Qwen3.5-4B",
        "algorithm": "grpo",
        "environment": {"id": "github:owner/repo@main:my-env/environment.py", "params": {"split": "train"}},
        "train": {
            "hf_repo": "owner/my-runs",
            "steps": 120,
            "lora_rank": 32,
            "lora_alpha": 64,
            "learning_rate": 1e-5,
            "group_size": 8,
            "temperature": 0.9,
            "max_tokens": 512,
        },
        "gpu": {"type": "RTX 5090"},
    }


def test_backend_run_config_parses_into_valid_jobspec() -> None:
    """A representative backend training config maps to a JobSpec with model,
    algorithm, environment id, and train params intact."""
    spec = spec_from_dict(_representative_config(), run_id="flash-test-1")
    assert isinstance(spec, JobSpec)
    assert spec.model == "Qwen/Qwen3.5-4B"
    assert spec.algorithm == "grpo"
    assert spec.environment.id == "github:owner/repo@main:my-env/environment.py"
    assert spec.environment.params == {"split": "train"}
    # hf_repo is platform-managed: a user/backend-supplied value is ignored (left blank for the
    # control plane to assign per run at submit), so it must NOT survive parsing.
    assert spec.train.hf_repo == ""
    assert spec.train.steps == 120
    assert spec.train.group_size == 8
    assert spec.run_id == "flash-test-1"


def test_jobspec_round_trips_through_dict() -> None:
    """to_dict()/from_dict() must be lossless so the backend's stored config (a JobSpec
    serialized to a dict) reconstructs the same JobSpec on the worker."""
    spec = spec_from_dict(_representative_config(), run_id="flash-test-2")
    again = JobSpec.from_dict(spec.to_dict())
    assert again == spec


def test_backend_config_through_runcreaterequest_then_jobspec() -> None:
    """The end-to-end seam: validate the representative config through the backend's
    actual RunCreateRequest (so a backend-side schema tightening is caught), then feed
    the validated config into flash's parser. A config the backend accepts must
    produce a JobSpec.
    """
    try:
        from src.schemas.runs import RunCreateRequest
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"backend schema not importable: {exc}")

    req = RunCreateRequest(name="my run", config=_representative_config())
    assert req.config is not None
    spec = spec_from_dict(req.config, run_id="flash-test-3")
    assert spec.environment.id == "github:owner/repo@main:my-env/environment.py"
    assert spec.algorithm == "grpo"


def test_sft_backend_config_maps_to_jobspec() -> None:
    """The other algorithm the backend/agent emit: an SFT config (epochs-driven) must
    also map to a valid JobSpec."""
    config = {
        "model": "Qwen/Qwen3.5-4B",
        "algorithm": "sft",
        "environment": {"id": "github:owner/repo@main:my-env/environment.py"},
        "train": {"hf_repo": "owner/my-runs", "epochs": 3, "max_examples": 64, "batch_size": 8},
        "gpu": {"type": "RTX 5090"},
    }
    spec = spec_from_dict(config, run_id="flash-sft-1")
    assert spec.algorithm == "sft"
    assert spec.phase == "sft"
    assert spec.train.epochs == 3
    assert spec.train.max_examples == 64
    assert spec.train.batch_size == 8


def test_backend_config_missing_required_fields_is_rejected_consistently() -> None:
    """The contract also covers the negative: a config the agent could author wrong
    (no [environment] id) must be rejected by flash with a ConfigError (-> HTTP 400 on
    the control plane), not silently accepted. This keeps the backend/agent and flash
    agreeing on what a *complete* job spec is. (hf_repo is platform-managed, not a
    required user field — see test_backend_run_config_parses_into_valid_jobspec.)
    """
    no_env = {
        "model": "Qwen/Qwen3.5-4B",
        "algorithm": "grpo",
        "train": {"steps": 10},
        "gpu": {"type": "RTX 5090"},
    }
    with pytest.raises(ConfigError):
        spec_from_dict(no_env)


def test_stale_local_env_path_is_rejected_at_both_layers() -> None:
    """The migration seam: local environment `path` is gone — only published Hub `id`s
    run. A stale config carrying [environment] path must be rejected by both the
    config parser AND JobSpec.from_dict, so neither the backend's stored config nor a
    re-hydrated spec can sneak a local path onto the worker.
    """
    stale = {
        "model": "Qwen/Qwen3.5-4B",
        "algorithm": "grpo",
        "environment": {"path": "environments/local_env.py"},
        "train": {"hf_repo": "owner/my-runs", "steps": 10},
        "gpu": {"type": "RTX 5090"},
    }
    with pytest.raises(ConfigError):
        spec_from_dict(stale)
    with pytest.raises(ValueError):  # noqa: PT011 — contract test: any rejection of a stale spec
        JobSpec.from_dict(stale)
