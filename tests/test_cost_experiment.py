"""Cost estimator: the prompt-convergence harness + report rendering.

The unit tests drive the harness with a deterministic, network-free stub estimator;
the one live test is opt-in (``AUTOSLM_LIVE=1`` + ``ANTHROPIC_API_KEY``).
"""

from __future__ import annotations

import itertools
import json
import os

import pytest

from flash.cost import RunConfig, estimate_cost
from flash.cost.experiment import (
    ExperimentResult,
    _gpu_match,
    decaying_stub_factory,
    default_grid,
    diverse_grid,
    run_experiment,
)
from flash.cost.llm import LLMEstimate
from flash.cost.report import ascii_chart, markdown_report, save_json, save_png


def test_default_grid_shape():
    grid = default_grid()
    assert len(grid) == 24  # 4 models x 2 methods x 3 step counts
    assert all(isinstance(c, RunConfig) for c in grid)
    assert {c.method for c in grid} == {"sft", "grpo"}


def test_diverse_grid_varies_every_axis_and_pins_hold():
    grid = diverse_grid()
    assert len(grid) >= 6
    gpus, methods, envs, settings = set(), set(), set(), set()
    for cfg in grid:
        e = estimate_cost(cfg)
        assert e.gpu == cfg.gpu, f"{cfg.label}: pin {cfg.gpu} escalated to {e.gpu}"
        gpus.add(cfg.gpu)
        methods.add(cfg.method)
        envs.add(cfg.environment)
        n = cfg.normalized()
        settings.add((n.seq_len, n.lora_rank, n.group_size, n.completion_len, n.thinking))
    assert len(gpus) >= 5  # various GPUs
    assert methods == {"sft", "grpo"}  # various algorithms
    assert len(envs) >= 6  # various environments
    assert len(settings) >= 6  # various settings


def test_diverse_grid_runs_through_the_harness():
    result = run_experiment(
        grid=diverse_grid(), estimator_factory=decaying_stub_factory(), max_workers=4
    )
    assert result.grid_size == len(diverse_grid())
    assert result.summaries[-1].mape == pytest.approx(0.0)


def test_stub_sweep_mape_decreases_monotonically():
    result = run_experiment(estimator_factory=decaying_stub_factory(), max_workers=4)
    mapes = [s.mape for s in result.summaries]
    assert all(m is not None for m in mapes)
    # Deterministic stub: error shrinks with version, so MAPE is non-increasing.
    assert all(a >= b - 1e-9 for a, b in itertools.pairwise(mapes))
    assert result.summaries[0].mape > result.summaries[-1].mape
    assert result.summaries[-1].mape == pytest.approx(0.0)  # v6 stub == ground truth


def test_every_cell_ok_with_stub():
    result = run_experiment(estimator_factory=decaying_stub_factory(), max_workers=4)
    for s in result.summaries:
        assert s.n_ok == s.n == result.grid_size


def test_result_is_json_serializable():
    result = run_experiment(
        versions=(1, 6), estimator_factory=decaying_stub_factory(), max_workers=4
    )
    json.dumps(result.to_dict())  # must not raise


def test_failing_estimator_degrades_gracefully():
    class Broken:
        def __init__(self, version):
            self.version = version

        def estimate(self, config):
            return LLMEstimate(self.version, None, None, "", "boom", ok=False, error="boom")

    result = run_experiment(
        versions=(1,),
        estimator_factory=lambda v: Broken(v),
        grid=default_grid()[:3],
        max_workers=2,
    )
    s = result.summaries[0]
    # No crash, just no usable estimates.
    assert s.n_ok == 0
    assert s.mape is None


def test_from_dict_reconstructs_and_resummarizes():
    result = run_experiment(estimator_factory=decaying_stub_factory(), max_workers=4)
    rebuilt = ExperimentResult.from_dict(result.to_dict())
    assert rebuilt.grid_size == result.grid_size
    assert len(rebuilt.cells) == len(result.cells)
    # Summaries are recomputed and carry the per-method split.
    for s in rebuilt.summaries:
        assert s.mape_sft is not None
        assert s.mape_grpo is not None
    assert rebuilt.summary(rebuilt.versions[-1]).mape == pytest.approx(0.0)


def test_gpu_match_tolerates_size_annotation_but_requires_the_full_class():
    # The estimate may append a VRAM annotation to the SAME class.
    assert _gpu_match("A100 PCIe 80GB", "A100 PCIe")
    assert _gpu_match("RTX 5090", "RTX 5090")
    assert not _gpu_match("RTX 4090", "A100 PCIe")
    # A generic family pick must NOT count as a hit for a more specific managed class
    # (different pricing/availability) -- this is what the exact-match fix enforces.
    assert not _gpu_match("H100", "H100 NVL")
    assert not _gpu_match("A100", "A100 PCIe")
    # ...but naming at least the full truth class still matches.
    assert _gpu_match("H100 NVL 94GB", "H100 NVL")


def test_median_ape_averages_both_middle_cells_for_even_n():
    # With an even number of ok cells, median_ape must be a TRUE median (mean of the two
    # middle values), not the upper-middle element -- so it doesn't jump with ordering.
    from flash.cost.experiment import Cell, _summarize

    apes = [10.0, 20.0, 30.0, 40.0]  # true median = (20+30)/2 = 25
    cells = tuple(
        Cell(
            version=1,
            label=f"c{i}",
            model_id="m",
            method="sft",
            steps=1,
            truth_usd=1.0,
            truth_gpu="RTX 5090",
            est_usd=1.0,
            est_gpu="RTX 5090",
            abs_pct_err=a,
            ok=True,
        )
        for i, a in enumerate(apes)
    )
    s = _summarize(1, cells, grid_size=len(cells))
    assert s.median_ape == pytest.approx(25.0)


def test_reports_render(tmp_path):
    result = run_experiment(estimator_factory=decaying_stub_factory(), max_workers=4)

    md = markdown_report(result)
    assert "Accuracy by prompt version" in md
    assert "error reduction" in md

    chart = ascii_chart(result)
    assert chart.count("\n") >= len(result.summaries)  # a line per version

    jp = save_json(result, tmp_path / "results.json")
    assert json.loads(jp.read_text())["grid_size"] == result.grid_size


def test_save_png(tmp_path):
    pytest.importorskip("matplotlib")
    result = run_experiment(
        versions=(1, 6), estimator_factory=decaying_stub_factory(), max_workers=4
    )
    p = save_png(result, tmp_path / "mape.png")
    assert p.exists()
    assert p.stat().st_size > 0


@pytest.mark.live
def test_live_sweep_converges():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("no ANTHROPIC_API_KEY")
    from flash.cost.experiment import live_estimator_factory

    grid = [RunConfig("Qwen/Qwen3.5-4B", "grpo", 150)]
    result = run_experiment(
        grid=grid,
        versions=(1, 6),
        estimator_factory=live_estimator_factory(effort="low"),
        max_workers=2,
    )
    assert isinstance(result, ExperimentResult)
    v1, v6 = result.summary(1), result.summary(6)
    assert v6.n_ok >= 1  # the full-knowledge prompt must at least produce a number
