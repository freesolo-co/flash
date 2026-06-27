"""Benchmark a trained run: deploy → infer over the eval split → score vs the paper.

This is the eval stage's orchestrator. It deploys the trained adapter, generates responses for
a deterministic subsample of the paper's held-out eval split, scores them with the **independent
paper-faithful metric** (``autoenv.eval.metrics`` on ``(gold, response)`` — deliberately NOT the
env reward the agent optimized, so a reward-hacked run can't inflate the headline), and produces
the improvement-normalized ``BenchResult``.

The untrained base is currently taken from the paper's reported baseline (``base_reported``),
not independently served — Flash serving is adapter-scoped, so there is no bare-base endpoint —
and that is recorded in ``diagnostics["base_source"]``.
"""

from __future__ import annotations

from dataclasses import dataclass

from autoenv.eval.generate import generate_for_rows
from autoenv.eval.metrics import aggregate, score_one
from autoenv.eval.split import leakage_check, subsample
from autoenv.manifest import PaperCase
from autoenv.report.record import BenchResult
from autoenv.score.normalize import score


@dataclass(frozen=True)
class EvalConfig:
    max_eval: int = 40
    max_tokens: int = 512
    temperature: float = 0.0
    eval_seed: int = 0
    deploy: bool = True
    undeploy_after: bool = True


def evaluate_run(
    case: PaperCase,
    run_id: str,
    *,
    client,
    eval_rows: list[dict],
    train_rows: list[dict] | None = None,
    config: EvalConfig | None = None,
    model_id: str | None = None,
    log=print,
) -> BenchResult:
    """Deploy ``run_id``, score it on the paper's eval split, and return a ``BenchResult``.

    ``model_id`` is the catalog model actually trained — pass the gate/drive-resolved id so a
    nearest-size *substitution* is reported faithfully; defaults to the manifest's ``flash_model``.
    """
    cfg = config or EvalConfig()
    metric = case.metric
    flash_model = model_id or case.flash_model

    subset = subsample(eval_rows, cfg.max_eval, seed=cfg.eval_seed)
    leak = leakage_check(train_rows or [], subset)
    base_source = (
        "paper_reported (base_reported); not independently served"
        if metric.base_reported is not None
        else "none (manifest has no base_reported)"
    )
    diagnostics: dict = {
        "eval_n": len(subset),
        "base_source": base_source,
        "leakage": leak.leaked,
        "leakage_overlap": leak.overlap_count,
    }
    if leak.leaked:
        # A leaked eval set can't measure generalization — return without a (misleading) score.
        diagnostics["invalid_reason"] = "eval split overlaps the trained rows"
        return BenchResult(
            case_id=case.id,
            state="invalid",
            flash_model=flash_model,
            run_id=run_id,
            paper_metric=metric.reported,
            diagnostics=diagnostics,
        )

    if cfg.deploy:
        log(f"deploying adapter {run_id} (serving warmup can take a few minutes)...")
        client.deploy(run_id)
    try:
        log(f"generating {len(subset)} responses at temperature={cfg.temperature}...")
        responses = generate_for_rows(
            client,
            run_id,
            subset,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            on_progress=lambda i, n, _t: log(f"  {i}/{n}") if i % 10 == 0 or i == n else None,
        )
    finally:
        if cfg.deploy and cfg.undeploy_after:
            try:
                client.undeploy(run_id)
            except Exception as exc:
                log(f"warning: undeploy failed: {exc}")

    golds = [str(r.get("output", "")) for r in subset]
    pairs = list(zip(golds, responses, strict=False))
    agent_metric = aggregate(metric.name, pairs)

    # A few transcripts for inspection (and a reward-hack sniff: empty/degenerate outputs).
    diagnostics["samples"] = [
        {
            "input": str(subset[i].get("input", ""))[:160],
            "gold": golds[i][-40:],
            "response": responses[i][:200],
        }
        for i in range(min(3, len(subset)))
    ]
    diagnostics["empty_responses"] = sum(1 for r in responses if not r.strip())
    diagnostics["per_example_metric"] = (
        round(sum(score_one(metric.name, g, r) for g, r in pairs) / len(pairs), 4) if pairs else 0.0
    )

    if metric.base_reported is None:
        sc = None
        achievement = None
        ratio_val = None
        noise = None
        diagnostics["note"] = "no base_reported in manifest; only the raw agent metric is reported"
    else:
        sc = score(
            agent_metric=agent_metric,
            base_metric=metric.base_reported,
            paper_metric=metric.reported,
            higher_is_better=metric.higher_is_better,
            eval_n=len(subset),
        )
        achievement = sc.achievement
        ratio_val = sc.ratio
        noise = sc.noise_band
        if sc.note:
            diagnostics["score_note"] = sc.note

    return BenchResult(
        case_id=case.id,
        state="scored",
        flash_model=flash_model,
        run_id=run_id,
        env_id=None,
        paper_metric=metric.reported,
        base_metric=metric.base_reported,
        agent_metric=round(agent_metric, 4),
        achievement=achievement,
        ratio=ratio_val,
        noise_band=noise,
        diagnostics=diagnostics,
    )
