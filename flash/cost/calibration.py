"""Calibrate + verify the cost equation against MEASURED real-run cost.

The estimator is **fully equation-based**: ``cost = wall-clock hours x market $/hr``,
with every term a real physical/economic quantity (FLOPs, MFU, cold-start seconds,
reward concurrency, the spot/queue rate the provider bills). There is **no output
multiplier** -- nothing scales the dollar figure to hit a target. Accuracy comes only
from getting the inputs right.

This module:

* ``verify_accuracy`` -- grade the raw equation against the measured RunPod/Vast runs in
  ``cost_estimator_results/real_runs/measured_runs.json`` (mean/median APE, aggregate
  bias, fraction within tolerance). The honest scorecard -- not forced to any value.
* ``fit_constants`` -- re-derive the *physical* constants (realized per-class $/hr, MFU,
  reward concurrency) from those measured runs, so the values hardcoded in ``analytical``
  / ``hardware`` are calibrated and regenerable -- the same way you'd measure MFU
  empirically, NOT a dimensionless fudge on the result.
* ``environment_cost_sweep`` -- price a GRPO run across reward-latency tiers + average.

Real per-run cost has irreducible variance: measured wall = provider queue + container
boot + model download + train, and the scheduling/cold-start overhead swings by minutes
in ways no config predicts. So the achievable accuracy is unbiased aggregate + a typical
run within ~30-45% median APE, reported honestly with a band -- not a fake-precise point.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from .analytical import estimate_cost
from .config import RunConfig
from .estimate import CostEstimate
from .hardware import gpu_tflops
from .rewards import reward_seconds_per_completion

# The measured-run dataset the equation is calibrated + graded against. In a source
# checkout it lives at the repo root (next to its generator and the accuracy report); in an
# installed wheel that path is absent and it is force-included under the package as
# ``flash/cost/_data/measured_runs.json`` (see pyproject ``force-include``). Prefer the
# source-root copy so a refreshed dataset is never shadowed by a stale packaged one, and
# fall back to the packaged copy so this still resolves in an installed wheel.
_HERE = Path(__file__).resolve().parent
_PACKAGED_RUNS = _HERE / "_data" / "measured_runs.json"
_SOURCE_RUNS = _HERE.parents[1] / "cost_estimator_results" / "real_runs" / "measured_runs.json"
_MEASURED_RUNS = _SOURCE_RUNS if _SOURCE_RUNS.exists() else _PACKAGED_RUNS


def _load_runs(path: Path | str | None = None) -> list[dict]:
    p = Path(path) if path is not None else _MEASURED_RUNS
    return json.loads(p.read_text())["runs"]


def _config_of(run: dict) -> RunConfig:
    """Rebuild the RunConfig a measured run priced (pinned to the GPU it actually used).

    The measured ``gpu`` is a record of fact -- the card the run *did* run on -- so it must
    be honored even when that class isn't in the default validated pool (e.g. a Vast
    ``A100 SXM`` / ``RTX 6000 Ada`` run). ``pick_gpu`` keeps a pin only if it survives its
    provisionability gate (per ``provider``), its validation gate, AND the VRAM-fit gate, so
    pass the run's own ``provider`` and ``allow_unvalidated=True`` to clear the first two.
    The VRAM-fit gate is cleared at grading time by ``estimate_cost(..., pin_must_fit=False)``
    in ``verify_accuracy`` -- the recorded card is proof the run fit, so it must not be dropped
    for a cheaper/larger one when the offline VRAM heuristic over-estimates (that would re-price
    the bill on a *different* GPU, corrupting its accuracy/bias). Forward ``max_wall_seconds``
    when the dataset carries it so a capped run is graded against the same wall cap the runner
    applied (the current dataset records none, so this is a no-op until a measured row includes
    one -- a cap is never invented here).
    """
    return RunConfig(
        run["model"],
        run["method"],
        run["steps"],
        seq_len=run.get("seq_len"),
        completion_len=run.get("completion_len"),
        batch_size=run.get("batch_size"),
        group_size=run.get("group_size"),
        environment=run.get("environment"),
        setup_repeats=run.get("setup_repeats", 1),
        gpu=run["gpu"],  # price on the card the run actually ran on (a record of fact)
        provider=run.get("provider", "auto"),  # restrict the pool to the substrate it ran on
        allow_unvalidated=True,  # honor the recorded class even if it's outside the validated pool
        max_wall_seconds=run.get("max_wall_seconds"),  # grade against the run's own cap if recorded
    )


# A run whose measured wall is below this almost never executed its configured step count
# (smoke tests get cancelled / step-capped early), so its measured cost is for a *different*
# (shorter) run than the config the equation prices -- an invalid comparison, not an
# estimator error (tell: predicted train time alone exceeds the whole measured wall). We
# report those separately rather than letting them pollute the real-run accuracy.
REAL_RUN_MIN_WALL_S = 500.0

# Effective billed rate (cost / wall) below which a row's billing is implausible: no managed
# class bills anywhere near this. Such a row is a degenerate run -- the job hung/idled and was
# force-killed with near-zero billed GPU-seconds, so its huge wall is not training time and
# its tiny cost is not the price of the configured work (e.g. a 49.5h / $0.04 SFT row bills
# $0.0008/hr -- 300x below the cheapest class's ~$0.24/hr floor). It clears the wall-length
# filter above (its wall is absurdly LONG, not short) but is just as invalid a comparison, so
# exclude it from the real-run accuracy the same way.
REAL_RUN_MIN_RATE_USD_PER_HR = 0.10


def _ran_its_work(run: dict) -> bool:
    """Whether a measured row actually executed its configured run (valid to grade against).

    Excludes both the short smoke tests (wall below the floor) and the degenerate
    hung/idle rows whose billed rate is implausibly low (cost/wall far under any class's
    real $/hr) -- in both cases the measured cost prices a *different* run than the config.
    """
    wall = run.get("wall_seconds", 0.0)
    if wall < REAL_RUN_MIN_WALL_S:
        return False
    rate = run.get("cost_usd", 0.0) / (wall / 3600.0) if wall else 0.0
    return rate >= REAL_RUN_MIN_RATE_USD_PER_HR


def verify_accuracy(path: Path | str | None = None) -> dict:
    """Grade the raw first-principles equation against measured cost. No output factor.

    Returns per-group: n, mean MAPE %, median APE %, aggregate bias (Σ estimate / Σ
    measured), and the fraction within 33% / 50%. Groups: all / sft / grpo, plus the
    ``real_*`` subsets (rows that actually ran their configured work, per ``_ran_its_work``:
    wall above the floor AND a plausible billed rate) -- the meaningful accuracy for pricing
    a real training run.
    """
    runs = _load_runs(path)
    real = [r for r in runs if _ran_its_work(r)]
    out: dict[str, dict] = {}
    groups = {
        "all": runs,
        "sft": [r for r in runs if r["method"] == "sft"],
        "grpo": [r for r in runs if r["method"] == "grpo"],
        "real": real,
        "real_sft": [r for r in real if r["method"] == "sft"],
        "real_grpo": [r for r in real if r["method"] == "grpo"],
    }
    for name, sub in groups.items():
        if not sub:
            continue
        apes, sum_est, sum_meas = [], 0.0, 0.0
        for r in sub:
            # ``pin_must_fit=False``: the run's recorded GPU is the card it DEMONSTRABLY ran on,
            # so honor it even when the offline VRAM heuristic over-estimates and would drop it
            # for a cheaper/larger class -- otherwise a measured bill (e.g. a real RTX-5090 GRPO
            # row) is graded against a DIFFERENT GPU's price, corrupting the accuracy/bias here.
            est = estimate_cost(_config_of(r), pin_must_fit=False).total_usd
            meas = r["cost_usd"]
            apes.append(100 * abs(est - meas) / meas)
            sum_est += est
            sum_meas += meas
        apes.sort()
        out[name] = {
            "n": len(sub),
            "mean_mape_pct": sum(apes) / len(apes),
            "median_ape_pct": st.median(apes),
            "agg_bias": sum_est / sum_meas,  # 1.0 = unbiased in aggregate (not forced)
            "within_33pct": sum(a <= 33 for a in apes) / len(apes),
            "within_50pct": sum(a <= 50 for a in apes) / len(apes),
        }
    return out


def fit_constants(path: Path | str | None = None) -> dict:
    """Re-derive the equation's physical constants from the measured runs.

    Returns ``{"realized_hourly_usd": {gpu: median $/hr}, "n_runs": int}`` -- the market
    realized $/hr per GPU class (median of the providers' billed rate, derived as measured
    cost / measured wall). This is an *input* the hardcoded constants should track -- a price
    calibration, not a correction applied to the output. Use it to refresh
    ``hardware.REALIZED_HOURLY_USD`` as new runs land. (MFU/reward-concurrency are calibrated
    separately against the same dataset, not returned here.)
    """
    runs = _load_runs(path)
    by_gpu_rate: dict[str, list[float]] = {}
    n_used = 0
    for r in runs:
        # Effective billed rate = measured cost / measured wall. This is the rate the
        # equation must multiply its predicted wall by to recover the billed cost (it
        # already folds in whatever the plane actually charged vs the provider's quote).
        wall = r.get("wall_seconds", 0.0)
        if wall <= 0:
            continue
        rate = r["cost_usd"] / (wall / 3600.0)
        # The billed RATE is a property of the GPU class, independent of whether the run
        # finished its configured steps -- so short smoke tests still measure it and are kept.
        # Only the degenerate hung/idle rows (rate below any class's real floor) are dropped:
        # their tiny cost over a huge idle wall isn't a real per-hour rate and would drag the
        # class median toward zero.
        if rate < REAL_RUN_MIN_RATE_USD_PER_HR:
            continue
        by_gpu_rate.setdefault(r["gpu"], []).append(rate)
        n_used += 1
    realized = {g: round(st.median(v), 3) for g, v in sorted(by_gpu_rate.items())}
    return {"realized_hourly_usd": realized, "n_runs": n_used}


# --------------------------------------------------------------------------- #
# Environment cost sweep: how a GRPO run's price moves with the reward env.
# --------------------------------------------------------------------------- #
SWEEP_ENVIRONMENTS: tuple[str, ...] = (
    "openai/gsm8k",  # trivial  (exact-match / numeric)
    "stanford/sentiment-classify",  # light    (parse + classify)
    "primeintellect/web-search",  # medium   (tool/search rollout)
    "freesolo/support-ticket",  # heavy    (LLM-as-judge)
    "swe/code-exec-contest",  # heavy    (sandboxed code execution)
    "acme/custom-unlisted-env",  # default  (unknown slug)
)


def environment_cost_sweep(
    *,
    models: tuple[str, ...] = ("openbmb/MiniCPM5-1B", "Qwen/Qwen3.5-4B"),
    steps: int = 100,
    environments: tuple[str, ...] = SWEEP_ENVIRONMENTS,
) -> list[dict]:
    """Price a GRPO run across environments (reward-latency tiers) and average.

    Environment enters cost only through the GRPO reward grader (rewards.py); the rest of
    the run is identical. One row per (model, env) with the resolved reward seconds and the
    cost, plus a per-model average and a grand average. The raw equation cost -- no factor.
    """
    rows: list[dict] = []
    grand: list[float] = []
    for model in models:
        per_model: list[float] = []
        for env in environments:
            est: CostEstimate = estimate_cost(RunConfig(model, "grpo", steps, environment=env))
            rows.append(
                {
                    "model": model,
                    "environment": env,
                    "reward_s_per_completion": reward_seconds_per_completion(env),
                    "gpu": est.gpu,
                    "usd": est.total_usd,
                    "capped": est.wall_capped,
                }
            )
            per_model.append(est.total_usd)
            grand.append(est.total_usd)
        rows.append(
            {
                "model": model,
                "environment": "AVERAGE (across envs)",
                "reward_s_per_completion": None,
                "gpu": "",
                "usd": sum(per_model) / len(per_model),
                "capped": False,
            }
        )
    rows.append(
        {
            "model": "ALL MODELS",
            "environment": "GRAND AVERAGE",
            "reward_s_per_completion": None,
            "gpu": "",
            "usd": sum(grand) / len(grand),
            "capped": False,
        }
    )
    return rows


# Keep the GPU tables importable for callers that want raw throughput facts.
__all__ = [
    "environment_cost_sweep",
    "fit_constants",
    "gpu_tflops",
    "verify_accuracy",
]
