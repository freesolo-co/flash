"""Calibrate + verify the cost equation against MEASURED real-run cost.

The estimator is **fully equation-based**: ``cost = wall-clock hours x market $/hr``, every
term a real physical/economic quantity. There is **no output multiplier**. Accuracy comes
only from getting the inputs right. This module:

* ``verify_accuracy`` -- grade the raw equation against the measured runs in
  ``_data/measured_runs.json`` (mean/median APE, aggregate bias, $ profit/loss). Not forced.
* ``fit_constants`` -- re-derive the realized per-class market $/hr (median measured cost /
  wall) from those runs, so the rates hardcoded in ``facts`` are regenerable from billing.
* ``environment_cost_sweep`` -- price a GRPO run across reward-latency tiers + average.

Real per-run cost has irreducible variance (queue + boot + download swing by minutes), so the
achievable accuracy is an unbiased aggregate + a typical run within ~30-45% median APE.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from .analytical import estimate_cost
from .facts import gpu_tflops, reward_seconds_per_completion
from .types import CostEstimate, RunConfig

# The measured-run dataset the equation is calibrated + graded against (ships in the package).
_MEASURED_RUNS = Path(__file__).resolve().parent / "_data" / "measured_runs.json"


def _load_runs(path: Path | str | None = None) -> list[dict]:
    p = Path(path) if path is not None else _MEASURED_RUNS
    return json.loads(p.read_text())["runs"]


def _config_of(run: dict) -> RunConfig:
    """Rebuild the RunConfig a measured run priced, pinned to the GPU it actually used.

    The recorded ``gpu``/``provider`` are facts, so pass ``allow_unvalidated=True`` to clear
    the provisionability/validation gates; the VRAM-fit gate is cleared at grading time by
    ``estimate_cost(..., pin_must_fit=False)``. Forwards ``max_wall_seconds`` if recorded.
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
        gpu=run["gpu"],
        provider=run.get("provider", "auto"),
        allow_unvalidated=True,
        max_wall_seconds=run.get("max_wall_seconds"),
    )


# A run whose measured wall is below this almost never executed its configured steps (smoke
# tests get cancelled / step-capped early), so its cost prices a different, shorter run than
# the config the equation prices -- an invalid comparison, reported separately.
REAL_RUN_MIN_WALL_S = 500.0

# Effective billed rate (cost / wall) below which a row is degenerate: the job hung/idled and
# was force-killed with near-zero billed GPU-seconds, so its huge wall isn't training time
# (e.g. a 49.5h / $0.04 row bills $0.0008/hr, 300x below the cheapest class). Also excluded.
REAL_RUN_MIN_RATE_USD_PER_HR = 0.10


def _ran_its_work(run: dict) -> bool:
    """Whether a row actually executed its configured run (valid to grade against): wall above
    the floor AND a plausible billed rate. Excludes smoke tests and hung/idle rows."""
    wall = run.get("wall_seconds", 0.0)
    if wall < REAL_RUN_MIN_WALL_S:
        return False
    rate = run.get("cost_usd", 0.0) / (wall / 3600.0) if wall else 0.0
    return rate >= REAL_RUN_MIN_RATE_USD_PER_HR


def verify_accuracy(path: Path | str | None = None) -> dict:
    """Grade the raw first-principles equation against measured cost. No output factor.

    Returns per-group: n, mean MAPE %, median APE %, aggregate bias (Sigma est / Sigma
    measured), fraction within 33% / 50%, and the $ profit/loss (treating the estimate as the
    quote and measured as cost). Groups: all / sft / grpo, plus the ``real_*`` subsets (rows
    that actually ran their work) -- the meaningful accuracy for pricing a real run.
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
        apes, diffs, sum_est, sum_meas = [], [], 0.0, 0.0
        for r in sub:
            # pin_must_fit=False: grade on the card the run demonstrably ran on (see _config_of).
            est = estimate_cost(_config_of(r), pin_must_fit=False).total_usd
            meas = r["cost_usd"]
            apes.append(100 * abs(est - meas) / meas)
            diffs.append(est - meas)  # + = quoted ABOVE cost (gain), - = BELOW (loss)
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
            "sum_measured_usd": sum_meas,
            "sum_estimated_usd": sum_est,
            "net_usd": sum_est - sum_meas,  # >0 = gain (over-quote), <0 = loss (under-quote)
            "net_pct": 100 * (sum_est - sum_meas) / sum_meas,
            "mean_diff_usd": sum(diffs) / len(diffs),
        }
    return out


def fit_constants(path: Path | str | None = None) -> dict:
    """Re-derive the realized per-class market $/hr from the measured runs.

    Returns ``{"realized_hourly_usd": {gpu: median $/hr}, "n_runs": int}`` -- median of the
    billed rate (measured cost / wall) per class, the *input* ``facts.REALIZED_HOURLY_USD``
    should track. Degenerate hung/idle rows (rate below any class's floor) are dropped.
    """
    runs = _load_runs(path)
    by_gpu_rate: dict[str, list[float]] = {}
    n_used = 0
    for r in runs:
        wall = r.get("wall_seconds", 0.0)
        if wall <= 0:
            continue
        rate = r["cost_usd"] / (wall / 3600.0)
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
    "openai/gsm8k",  # trivial
    "stanford/sentiment-classify",  # light
    "primeintellect/web-search",  # medium
    "freesolo/support-ticket",  # heavy (LLM-as-judge)
    "swe/code-exec-contest",  # heavy (sandboxed code execution)
    "acme/custom-unlisted-env",  # default (unknown slug)
)


def environment_cost_sweep(
    *,
    models: tuple[str, ...] = ("openbmb/MiniCPM5-1B", "Qwen/Qwen3.5-4B"),
    steps: int = 100,
    environments: tuple[str, ...] = SWEEP_ENVIRONMENTS,
) -> list[dict]:
    """Price a GRPO run across environments (reward-latency tiers) and average.

    Environment enters cost only through the reward grader; the rest of the run is identical.
    One row per (model, env), plus a per-model average and a grand average. Raw equation cost.
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


__all__ = [
    "environment_cost_sweep",
    "fit_constants",
    "gpu_tflops",
    "verify_accuracy",
]
