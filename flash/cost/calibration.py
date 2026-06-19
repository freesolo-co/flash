"""Break-even calibration: center the analytical equation on real measured cost.

The first-principles :func:`flash.cost.estimate_cost` model is the experiment's
*reference*. Graded against MEASURED control-plane cost over the real runs in
``cost_estimator_results/real_runs/``, it runs systematically HIGH -- the sum of its
quotes is ~1.4x the sum of what those runs were actually billed:

    Σ analytical (static $/hr)  /  Σ measured cost  ≈  1.42      (SFT 1.29x, GRPO 1.90x)

Two things drive the over-estimate, neither a bug in the physics:

* the registry's *static fallback* ``$/hr`` sits above the spot/queue rate the runs
  were actually billed at (a pricing-source gap, not a wall-clock gap), and
* fixed cold-start overhead over-weights the short, cheap runs.

For *pricing* we don't want to minimise per-run error -- we want a quote that, summed
over a representative workload, equals what the runs actually cost, so the platform
**breaks even**. That is a centering objective, and the unbiased estimator for it is
the ratio of the two totals:

    break-even factor (per method)  =  Σ measured cost  /  Σ analytical (static) cost

computed here from the committed ``real_runs/summary_real.json``. We calibrate
*per method* because GRPO (extra vLLM rollout + reward + vLLM-init cold start) is
over-estimated almost twice as hard as SFT; a single global factor would leave GRPO
over-priced and SFT under-priced even though the grand total balanced. Per-method
factors make each method break even on its own, so the aggregate does too.

:func:`estimate_cost` is left untouched -- so the experiment reference and every
committed artifact stay put -- and :func:`breakeven_estimate` wraps it, scaling
``total_usd`` and recording the multiplier on the returned :class:`CostEstimate`.
:func:`breakeven_factor_from_real_runs` recomputes the factors from the committed data
so the hardcoded constants below are regenerable (and pinned by
``tests/test_cost_calibration.py``); as new environments are measured and folded into
the manifest, re-run it and update the constants.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .analytical import estimate_cost
from .config import RunConfig
from .estimate import CostEstimate
from .rewards import reward_seconds_per_completion

# --- calibrated constants (Σ measured / Σ analytical-static over the committed runs) ---
# Regenerate with ``breakeven_factor_from_real_runs()``; pinned by the test-suite so a
# data refresh that moves them by >1% fails loudly rather than drifting silently.
BREAKEVEN_FACTORS: dict[str, float] = {
    "sft": 0.776,
    "grpo": 0.528,
}
BREAKEVEN_FACTOR_GLOBAL = 0.703  # method-agnostic fallback (whole-portfolio ratio)

# Number of measured real runs the factors were calibrated on (provenance for notes).
CALIBRATION_N = 13

# Path to the committed measured-run summary the factors are derived from. In a source
# checkout it lives at the repo root (next to its generator and the docs/report that
# regenerate it) -- that is the single source of truth; in an installed wheel that path
# is absent and it is force-included under the package as ``flash/cost/_data/summary_real.json``
# (see pyproject ``force-include``). Prefer the source-root copy whenever it exists so a
# refreshed summary is never shadowed by a stale packaged copy, and fall back to the
# packaged copy so this still resolves in an installed wheel.
_HERE = Path(__file__).resolve().parent
_PACKAGED_SUMMARY = _HERE / "_data" / "summary_real.json"
_SOURCE_SUMMARY = _HERE.parents[1] / "cost_estimator_results" / "real_runs" / "summary_real.json"
_REAL_RUNS_SUMMARY = _SOURCE_SUMMARY if _SOURCE_SUMMARY.exists() else _PACKAGED_SUMMARY


def breakeven_factor(method: str) -> float:
    """Per-method break-even multiplier (falls back to the global factor)."""
    return BREAKEVEN_FACTORS.get(method, BREAKEVEN_FACTOR_GLOBAL)


def breakeven_estimate(config: RunConfig, **kwargs) -> CostEstimate:
    """:func:`estimate_cost`, calibrated to break even against measured real-run cost.

    Identical breakdown to the analytical model, with ``total_usd`` scaled by the
    per-method break-even factor and ``calibration_factor`` recording the multiplier.
    This is the figure to *quote*; the raw analytical reference is
    ``total_usd / calibration_factor`` (and ``estimate_cost`` directly).
    """
    raw = estimate_cost(config, **kwargs)
    factor = breakeven_factor(raw.method)
    note = (
        f"break-even calibrated x{factor:.3f} to measured cost over {CALIBRATION_N} "
        f"real runs (raw first-principles analytical: ${raw.total_usd:.2f})"
    )
    return replace(
        raw,
        total_usd=raw.total_usd * factor,
        calibration_factor=factor,
        notes=(*raw.notes, note),
    )


# --------------------------------------------------------------------------- #
# Recomputation + verification from the committed measured runs.
# --------------------------------------------------------------------------- #
def _method_of(label: str) -> str:
    """Method for a measured-run label (``...-grpo-...`` vs SFT)."""
    return "grpo" if "grpo" in label.lower() else "sft"


def _load_measured(path: Path | str | None = None) -> list[dict]:
    import json

    p = Path(path) if path is not None else _REAL_RUNS_SUMMARY
    return json.loads(p.read_text())["measured"]


def breakeven_factor_from_real_runs(path: Path | str | None = None) -> dict[str, float]:
    """Recompute the break-even factors from the committed measured-run summary.

    Returns ``{"sft": .., "grpo": .., "global": ..}`` = Σ measured / Σ analytical-static
    over each subset. This is the source of truth the hardcoded :data:`BREAKEVEN_FACTORS`
    are pinned to; run it after new runs land to refresh the constants.
    """
    rows = _load_measured(path)
    out: dict[str, float] = {}
    for method in ("sft", "grpo"):
        sub = [r for r in rows if _method_of(r["label"]) == method]
        meas = sum(r["cost_usd"] for r in sub)
        ana = sum(r["analytical_static_usd"] for r in sub)
        out[method] = meas / ana if ana else float("nan")
    meas = sum(r["cost_usd"] for r in rows)
    ana = sum(r["analytical_static_usd"] for r in rows)
    out["global"] = meas / ana if ana else float("nan")
    return out


def verify_centering(path: Path | str | None = None) -> dict[str, dict[str, float]]:
    """Prove the calibration breaks even: Σ calibrated quote vs Σ measured, per method.

    Applies the hardcoded per-method factor to each committed run's analytical-static
    quote and reports, per method and overall, ``sum_measured``, ``sum_calibrated`` and
    their ratio (1.0 == break-even). The ratio is exactly 1.0 only if the constants
    equal :func:`breakeven_factor_from_real_runs`; small residuals are the rounding of
    the published constants.
    """
    rows = _load_measured(path)
    groups: dict[str, list[dict]] = {"sft": [], "grpo": []}
    for r in rows:
        groups[_method_of(r["label"])].append(r)
    groups["all"] = rows  # quoted with each row's own per-method factor

    out: dict[str, dict[str, float]] = {}
    for name, sub in groups.items():
        meas = sum(r["cost_usd"] for r in sub)
        cal = sum(
            r["analytical_static_usd"] * breakeven_factor(_method_of(r["label"])) for r in sub
        )
        out[name] = {
            "n": float(len(sub)),
            "sum_measured": meas,
            "sum_calibrated": cal,
            "ratio": cal / meas if meas else float("nan"),
        }
    return out


# --------------------------------------------------------------------------- #
# Environment cost sweep: how a GRPO run's price moves with the reward env.
# --------------------------------------------------------------------------- #
# Representative verifiers-env slugs, one per reward-latency tier (rewards.py), so the
# sweep spans the full range of grader cost from a free exact-match to an LLM-judge.
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

    Environment only enters cost through the GRPO reward grader (rewards.py); the rest
    of the run is identical. Returns one row per (model, env) with the resolved reward
    seconds/completion, the raw analytical cost and the break-even quote, plus one
    ``average`` row per model and a final grand-average row. This is the "test across
    various environments -> average cost" surface.
    """
    rows: list[dict] = []
    grand: list[float] = []
    for model in models:
        per_model: list[float] = []
        for env in environments:
            cfg = RunConfig(model, "grpo", steps, environment=env)
            cal = breakeven_estimate(cfg)
            raw = cal.total_usd / cal.calibration_factor
            rows.append(
                {
                    "model": model,
                    "environment": env,
                    "reward_s_per_completion": reward_seconds_per_completion(env),
                    "gpu": cal.gpu,
                    "raw_usd": raw,
                    "breakeven_usd": cal.total_usd,
                    # True where the (serial) reward model pushed wall clock past the 24h
                    # cap -- a known over-estimate the env-testing loop should retire.
                    "capped": cal.wall_capped,
                }
            )
            per_model.append(cal.total_usd)
            grand.append(cal.total_usd)
        rows.append(
            {
                "model": model,
                "environment": "AVERAGE (across envs)",
                "reward_s_per_completion": None,
                "gpu": "",
                "raw_usd": None,
                "breakeven_usd": sum(per_model) / len(per_model),
            }
        )
    rows.append(
        {
            "model": "ALL MODELS",
            "environment": "GRAND AVERAGE",
            "reward_s_per_completion": None,
            "gpu": "",
            "raw_usd": None,
            "breakeven_usd": sum(grand) / len(grand),
        }
    )
    return rows
