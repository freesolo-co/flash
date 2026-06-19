"""The prompt-convergence experiment.

Run an estimator over a grid of training configs at each prompt version (1..6) and
measure how close it lands to the analytical ground truth. The headline metric is
MAPE (mean absolute percentage error) per version; the hypothesis is that MAPE falls
monotonically as the prompt gains Flash framework knowledge.

The estimator is injected via ``estimator_factory(version) -> estimator`` so the same
harness drives the live Claude estimator (the real test runs) and a deterministic stub
(the unit tests) without touching the network.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from typing import Any, Protocol

from .analytical import estimate_cost
from .config import RunConfig
from .estimate import CostEstimate
from .llm import LLMEstimate
from .prompts import NUM_VERSIONS, VERSION_LABELS


class Estimator(Protocol):
    def estimate(self, config: RunConfig) -> LLMEstimate: ...


# (version) -> an estimator for that prompt version.
EstimatorFactory = Callable[[int], Estimator]
GroundTruthFn = Callable[[RunConfig], CostEstimate]


@dataclass(frozen=True)
class Cell:
    """One (version, config) evaluation."""

    version: int
    label: str  # config display label
    model_id: str
    method: str
    steps: int
    truth_usd: float
    truth_gpu: str
    est_usd: float | None
    est_gpu: str | None
    abs_pct_err: float | None  # |est - truth| / truth * 100
    ok: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CELL_FIELDS = tuple(f.name for f in fields(Cell))


@dataclass(frozen=True)
class VersionSummary:
    version: int
    label: str
    n: int
    n_ok: int
    mape: float | None  # mean abs pct err over ok cells
    median_ape: float | None
    mean_signed_pct: float | None  # mean (est-truth)/truth*100 -> bias direction
    gpu_accuracy: float | None  # fraction of ok cells whose GPU matches ground truth
    mape_sft: float | None = None  # MAPE over SFT cells only
    mape_grpo: float | None = None  # MAPE over GRPO cells only

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentResult:
    versions: tuple[int, ...]
    grid_size: int
    cells: tuple[Cell, ...]
    summaries: tuple[VersionSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "versions": list(self.versions),
            "grid_size": self.grid_size,
            "summaries": [s.to_dict() for s in self.summaries],
            "cells": [c.to_dict() for c in self.cells],
        }

    def summary(self, version: int) -> VersionSummary:
        return next(s for s in self.summaries if s.version == version)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentResult:
        """Rebuild from a ``to_dict`` payload, re-deriving summaries from the cells.

        Summaries are recomputed (not read back) so a saved run picks up newer summary
        fields -- the report/graph can be regenerated from raw cells without re-querying.
        """
        cells = tuple(Cell(**{k: c[k] for k in _CELL_FIELDS if k in c}) for c in d["cells"])
        versions = tuple(d["versions"])
        grid_size = int(d["grid_size"])
        summaries = tuple(
            _summarize(v, [c for c in cells if c.version == v], grid_size) for v in versions
        )
        return cls(versions=versions, grid_size=grid_size, cells=cells, summaries=summaries)


def default_grid() -> list[RunConfig]:
    """A deliberately broad grid: 4 models x {SFT, GRPO} x 3 step counts = 24 runs.

    Spans a >100x cost range and exercises the corners the framework knowledge is
    meant to capture: a QLoRA 9B (cheaper card than its size implies), GRPO's colocate
    floor routing to a bigger GPU than SFT, and long GRPO runs that hit the wall cap.
    """
    models = [
        "Qwen/Qwen3.5-0.8B",
        "openbmb/MiniCPM5-1B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B",
    ]
    methods = ["sft", "grpo"]
    steps = [100, 500, 1000]
    return [
        RunConfig(m, meth, s, label=f"{m.split('/')[-1]} {meth} x{s}")
        for m in models
        for meth in methods
        for s in steps
    ]


def diverse_grid() -> list[RunConfig]:
    """A hand-picked spread that varies every axis at once: GPU class, algorithm,
    knob settings (context, LoRA rank, group size, completion budget, thinking, batch),
    and verifiers environment. Each pinned GPU is validated and fits its run, so the
    estimate prices on exactly that card. Used for the diverse-run test sweep."""
    return [
        RunConfig(
            "Qwen/Qwen3.5-0.8B",
            "sft",
            300,
            gpu="RTX 4090",
            seq_len=2048,
            lora_rank=16,
            environment="primeintellect/gsm8k",
            label="0.8B sft / RTX4090 / seq2048 r16",
        ),
        RunConfig(
            "Qwen/Qwen3.5-9B",
            "grpo",
            200,
            gpu="A100 PCIe",
            thinking=True,
            group_size=16,
            environment="primeintellect/math",
            label="9B grpo / A100 / thinking g16",
        ),
        RunConfig(
            "openbmb/MiniCPM5-1B",
            "sft",
            800,
            gpu="RTX A5000",
            seq_len=2048,
            batch_size=16,
            environment="acme/sentiment-sft",
            label="1B sft / A5000 / seq2048 b16",
        ),
        RunConfig(
            "Qwen/Qwen3.5-4B",
            "grpo",
            150,
            gpu="RTX Pro 6000 WK",
            group_size=8,
            completion_len=512,
            environment="primeintellect/code-contests",
            label="4B grpo / Pro6000WK / comp512",
        ),
        RunConfig(
            "Qwen/Qwen3.5-2B",
            "grpo",
            500,
            gpu="RTX 5090",
            lora_rank=64,
            environment="acme/tool-use",
            label="2B grpo / RTX5090 / r64",
        ),
        RunConfig(
            "Qwen/Qwen3.5-9B",
            "sft",
            400,
            gpu="H100 NVL",
            seq_len=4096,
            environment="acme/summarization-sft",
            label="9B sft / H100NVL / seq4096",
        ),
        RunConfig(
            "Qwen/Qwen3.5-2B",
            "sft",
            1000,
            gpu="RTX 3090",
            thinking=True,
            environment="acme/reasoning-sft",
            label="2B sft / RTX3090 / thinking",
        ),
        RunConfig(
            "Qwen/Qwen3.5-0.8B",
            "grpo",
            1000,
            gpu="RTX A5000",
            group_size=4,
            completion_len=256,
            environment="primeintellect/aime",
            label="0.8B grpo / A5000 / g4 comp256",
        ),
    ]


def _make_cell(
    version: int, config: RunConfig, estimator: Estimator, ground_truth_fn: GroundTruthFn
) -> Cell:
    truth = ground_truth_fn(config)
    est = estimator.estimate(config)
    ape: float | None = None
    if est.ok and est.cost_usd is not None and truth.total_usd > 0:
        ape = abs(est.cost_usd - truth.total_usd) / truth.total_usd * 100.0
    return Cell(
        version=version,
        label=config.display(),
        model_id=config.model_id,
        method=config.method,
        steps=config.steps,
        truth_usd=round(truth.total_usd, 4),
        truth_gpu=truth.gpu,
        est_usd=est.cost_usd,
        est_gpu=est.gpu,
        abs_pct_err=ape,
        ok=est.ok and ape is not None,
        error=est.error,
    )


def _summarize(version: int, cells: Sequence[Cell], grid_size: int) -> VersionSummary:
    ok = [c for c in cells if c.ok and c.abs_pct_err is not None]
    apes = sorted(c.abs_pct_err for c in ok)  # type: ignore[misc]
    signed = [
        (c.est_usd - c.truth_usd) / c.truth_usd * 100.0  # type: ignore[operator]
        for c in ok
        if c.truth_usd > 0
    ]
    gpu_hits = [c for c in ok if c.est_gpu and _gpu_match(c.est_gpu, c.truth_gpu)]
    n = len(apes)

    def _method_mape(method: str) -> float | None:
        vals = [c.abs_pct_err for c in ok if c.method == method]
        return (sum(vals) / len(vals)) if vals else None  # type: ignore[arg-type]

    return VersionSummary(
        version=version,
        label=VERSION_LABELS.get(version, f"v{version}"),
        n=grid_size,
        n_ok=n,
        mape=(sum(apes) / n) if n else None,
        median_ape=(statistics.median(apes)) if n else None,
        mean_signed_pct=(sum(signed) / len(signed)) if signed else None,
        gpu_accuracy=(len(gpu_hits) / n) if n else None,
        mape_sft=_method_mape("sft"),
        mape_grpo=_method_mape("grpo"),
    )


def _gpu_match(est_gpu: str, truth_gpu: str) -> bool:
    """GPU-name match against the (canonical) ground-truth class.

    The truth is always a concrete managed class (the allocator never returns a bare
    family), so the estimate must name AT LEAST that full class: the truth's normalized
    name must be contained in the estimate's. This tolerates the LLM appending a size
    annotation ('A100 PCIe 80GB' for 'A100 PCIe') but does NOT count a generic family
    pick as a hit for a more specific class -- 'H100' must not match 'H100 NVL', nor
    'A100' match 'A100 PCIe', since those are distinct classes with different pricing.
    """
    e = "".join(ch for ch in est_gpu.lower() if ch.isalnum())
    t = "".join(ch for ch in truth_gpu.lower() if ch.isalnum())
    return bool(t) and t in e


def run_experiment(
    grid: Sequence[RunConfig] | None = None,
    *,
    versions: Iterable[int] = range(1, NUM_VERSIONS + 1),
    estimator_factory: EstimatorFactory,
    ground_truth_fn: GroundTruthFn = estimate_cost,
    max_workers: int = 8,
    progress: Callable[[str], None] | None = None,
) -> ExperimentResult:
    """Evaluate ``estimator_factory(v)`` over ``grid`` for each version in ``versions``."""
    grid = list(grid) if grid is not None else default_grid()
    versions = tuple(versions)
    cells: list[Cell] = []
    for v in versions:
        estimator = estimator_factory(v)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = [pool.submit(_make_cell, v, cfg, estimator, ground_truth_fn) for cfg in grid]
            vcells = [f.result() for f in futs]
        cells.extend(vcells)
        if progress is not None:
            s = _summarize(v, vcells, len(grid))
            mape = f"{s.mape:.1f}%" if s.mape is not None else "n/a"
            progress(f"  {s.label:<14} MAPE={mape:>7}  ({s.n_ok}/{s.n} ok)")
    summaries = tuple(
        _summarize(v, [c for c in cells if c.version == v], len(grid)) for v in versions
    )
    return ExperimentResult(
        versions=versions, grid_size=len(grid), cells=tuple(cells), summaries=summaries
    )


def live_estimator_factory(*, effort: str = "medium", **kw: Any) -> EstimatorFactory:
    """Default factory: a live Claude estimator per version (imports ``anthropic``)."""
    from .llm import LLMCostEstimator

    def factory(version: int) -> Estimator:
        return LLMCostEstimator(version, effort=effort, **kw)

    return factory


class _DecayingStub:
    """No-API estimator: ground truth perturbed by deterministic noise that shrinks
    with version. Used by ``--offline`` demos and the experiment unit tests, so the
    harness is exercisable end to end without the network."""

    def __init__(self, version: int, ground_truth_fn: GroundTruthFn, max_err_pct: float):
        self.version = version
        self._gt = ground_truth_fn
        self._max = max_err_pct

    def estimate(self, config: RunConfig) -> LLMEstimate:
        import hashlib

        truth = self._gt(config)
        span = max(1, NUM_VERSIONS - 1)
        magnitude = self._max * (NUM_VERSIONS - self.version) / span  # v1 -> max, vN -> 0
        seed = f"{config.display()}|{self.version}".encode()
        frac = int(hashlib.md5(seed).hexdigest()[:8], 16) / 0xFFFFFFFF  # stable [0,1)
        signed = (frac * 2 - 1) * magnitude / 100.0
        # No rounding: at the top version (magnitude 0) the stub equals ground truth
        # exactly, so the harness's v_max MAPE is provably 0.
        cost = max(0.0, truth.total_usd * (1 + signed))
        return LLMEstimate(self.version, cost, truth.gpu, "stub", "", ok=True)


def decaying_stub_factory(
    *, ground_truth_fn: GroundTruthFn = estimate_cost, max_err_pct: float = 60.0
) -> EstimatorFactory:
    """A deterministic, network-free estimator factory whose error decays with version."""

    def factory(version: int) -> Estimator:
        return _DecayingStub(version, ground_truth_fn, max_err_pct)

    return factory
