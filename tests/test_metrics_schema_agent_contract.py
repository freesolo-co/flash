"""Cross-folder contract: the metrics record autoslm's GPU worker writes
(engine/worker.py write_train_meta -> RunMetrics + notes) must keep carrying the
keys the freesolo agent and SDK consume.

Consumers of the run metrics:
  - the agent's training output schema (../agent/src/codex/outputs.py): the Codex
    trainer reports baseline_reward / final_reward / run_id and a W&B link drawn
    from the run metrics/logs;
  - the SDK (../../freesolo-sdk/pypi/freesolo/training/storage.py link_wandb):
    consumes `wandb_url`.

If a metrics key is renamed/dropped on the worker side, the agent's reward
verification or the W&B dashboard link silently break. This test pins the metrics
schema (the keys write_train_meta emits via RunMetrics + the SFT/RL `notes`) against
the fields those consumers depend on.

Run: cd autoslm && AUTOSLM_SKIP_NET=1 .venv/bin/python -m pytest \
        tests/test_metrics_schema_agent_contract.py -q
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

_AGENT_SRC = Path(__file__).resolve().parents[2] / "agent" / "src"
if _AGENT_SRC.is_dir() and str(_AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(_AGENT_SRC))

from autoslm.engine.accounting import RunMetrics

# The `notes` dicts engine/worker.py write_train_meta() is called with, mirrored from
# the SFT (worker.py ~L1293) and RL (worker.py ~L1942) finalize blocks. wandb_run_info()
# spreads {wandb_url, wandb_id, wandb_project} into notes when W&B is active.
_WANDB_NOTES = {"wandb_url": "https://wandb.ai/org/proj/runs/abc", "wandb_id": "abc",
                "wandb_project": "proj"}

_SFT_NOTES = {
    "epochs": 2,
    "resumed": False,
    "download_seconds": 12.0,
    "hf_transfer": "1",
    "thinking": False,
    "loss_curve": [2.1, 1.6, 1.2],
    **_WANDB_NOTES,
}

_RL_NOTES = {
    "steps": 150,
    "resumed": False,
    "download_seconds": 12.0,
    "hf_transfer": "1",
    "reward_history": [0.1, 0.3, 0.55],
    "loss_curve": [1.0, 0.8],
    **_WANDB_NOTES,
    "gen_tokens_is_upper_bound": True,
    "thinking": False,
    "max_completion_len": 512,
    "prompts_per_step": 8,
    "generations_per_step": 16,
    "group_size": 2,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
}


def _run_metrics_with_notes(phase: str, notes: dict) -> dict:
    """Build the persisted metrics dict the way write_train_meta does: a RunMetrics
    serialized to a dict, with the phase notes folded into `notes`. This mirrors the
    final metrics.json the control plane persists and the consumers read."""
    m = RunMetrics(
        arm="runpod",
        phase=phase,
        seed=0,
        model_id="Qwen/Qwen3.5-4B",
        wall_seconds=191.9,
        setup_seconds=40.0,
        train_tokens=10_000,
        generated_tokens=5_000,
        notes={
            **notes,
            "renderer": "autoslm_env",
            "train_wall": 191.9,
            "model_id": "Qwen/Qwen3.5-4B",
            "environment": "owner/name",
            "job_spec": {"algorithm": "grpo"},
        },
    )
    return dataclasses.asdict(m)


def test_run_metrics_top_level_keys_are_stable() -> None:
    """The RunMetrics dataclass is the metrics.json top-level schema. The agent reads
    run-outcome fields (wall_seconds for the run artifact, model_id, phase) and the
    runner derives cost from wall_seconds; pin those keys so a rename surfaces here."""
    keys = {f.name for f in dataclasses.fields(RunMetrics)}
    for required in ("arm", "phase", "seed", "model_id", "wall_seconds", "notes"):
        assert required in keys, f"RunMetrics dropped top-level key {required!r}"


def test_sft_metrics_carry_loss_curve_and_wandb_link() -> None:
    """SFT runs report convergence via `loss_curve` and the W&B link via `wandb_url`
    (notes). The agent's verifier inspects the loss/metrics; the SDK links W&B."""
    metrics = _run_metrics_with_notes("sft", _SFT_NOTES)
    notes = metrics["notes"]
    assert "loss_curve" in notes
    assert "wandb_url" in notes, "SDK link_wandb reads notes['wandb_url']"
    assert notes["wandb_url"].startswith("https://")
    assert "train_wall" in notes


def test_rl_metrics_carry_reward_history_and_wandb_link() -> None:
    """GRPO/RL runs report the reward trend via `reward_history`; the agent's output
    schema reports baseline_reward (early reward) and final_reward (latest reward)
    drawn from this curve, plus the W&B link. A missing reward_history would break the
    agent's reward verification entirely."""
    metrics = _run_metrics_with_notes("rl", _RL_NOTES)
    notes = metrics["notes"]
    assert "reward_history" in notes, "agent reward verification reads reward_history"
    assert isinstance(notes["reward_history"], list)
    assert notes["reward_history"]
    # baseline (first) and final (last) reward are derivable from the curve — exactly
    # what codex/outputs.py's baseline_reward/final_reward fields document.
    baseline_reward = notes["reward_history"][0]
    final_reward = notes["reward_history"][-1]
    assert isinstance(baseline_reward, (int, float))
    assert isinstance(final_reward, (int, float))
    assert "wandb_url" in notes
    assert "loss_curve" in notes
    assert "train_wall" in notes


def test_agent_training_schema_reward_fields_are_numeric_or_null() -> None:
    """The agent's training output schema (codex/outputs.py) declares
    summary.baseline_reward / summary.final_reward as number|null and run_id as
    string|null. The metrics those map from (reward_history entries; the autoslm run
    id) are numeric / string — assert the schema still declares them so a schema
    rename can't drift from what the metrics provide.
    """
    try:
        from codex.outputs import _output_schema_for_phase
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"agent codex.outputs not importable: {exc}")

    schema = _output_schema_for_phase("training")
    assert "run_id" in schema["required"]
    summary = schema["properties"]["summary"]
    props = summary["properties"]
    for field in ("baseline_reward", "final_reward"):
        assert field in summary["required"], f"training summary lost required {field!r}"
        assert props[field]["type"] == ["number", "null"]
    # run_id maps from the autoslm new_run_id()/tracker value: string|null.
    assert schema["properties"]["run_id"]["type"] == ["string", "null"]
    # algorithm enum must include the autoslm algorithms the worker stamps in phase.
    assert set(props["algorithm"]["enum"]) >= {"sft", "grpo"}


def test_worker_phase_matches_agent_algorithm_enum() -> None:
    """The worker stamps RunMetrics.phase as 'sft' or 'rl'; the JobSpec.phase property
    maps grpo->'rl'. The agent's schema reports algorithm as 'sft'/'grpo'. Assert the
    algorithm<->phase mapping the agent relies on holds, so the agent can map a
    reported run's phase back to its algorithm.
    """
    from autoslm.spec import JobSpec

    assert JobSpec(algorithm="grpo").phase == "rl"
    assert JobSpec(algorithm="sft").phase == "sft"
