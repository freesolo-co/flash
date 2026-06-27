"""The five eligibility checks that decide whether a paper is replicable on Flash.

Each check cites the Flash function it reuses. A case is eligible iff every non-skipped rule
passes. The dataset-availability probe can be skipped (``probe_dataset=False``) to keep the
gate fully offline in CI; a skipped rule is surfaced, not silently dropped.
"""

from __future__ import annotations

from autoenv.gate.model_match import resolve_flash_model
from autoenv.ingest.sources import can_fetch
from autoenv.manifest import PaperCase
from autoenv.report.record import GateReport, RuleResult
from flash.catalog import ALGORITHMS
from flash.cost.spec import estimate_for_spec
from flash.engine.vram import check_fit, resolve_params_b
from flash.schema import spec_from_dict

# Keywords that suggest an inherently multimodal (non-text) task Flash can't train.
_VLM_HINTS = (
    "image",
    "vision",
    "visual",
    "vqa",
    "multimodal",
    "screenshot",
    "ocr",
    "diagram",
    "chart",
    "photo",
    "pixel",
    "video",
    "audio",
    "speech",
)


# Mirrors flash.cost.spec.DEFAULT_UNCOUNTED_SFT_EXAMPLES: the row count to assume when an SFT
# run's dataset can't be counted. Pinning it for the preflight keeps the estimate deterministic
# AND offline (it skips the env-loading count path, which would otherwise hit the network).
_PREFLIGHT_SFT_EXAMPLES = 1000


def _provisional_spec_dict(case: PaperCase, model_id: str) -> dict:
    """A representative JobSpec dict for costing the run the agent will submit.

    Uses the case's caps so the estimate reflects the bounded smoke run. The env id is a
    placeholder (nothing is published at gate time) — a valid slug so the schema accepts it,
    never loaded. SFT pins ``max_examples`` so costing stays offline and deterministic.
    """
    train: dict = {"lora_rank": 32, "lora_alpha": 64, "seeds": [0]}
    if case.algorithm == "grpo":
        train["steps"] = case.max_train_steps or 150
    else:
        train["epochs"] = 1
        if case.max_train_steps:
            train["max_steps"] = case.max_train_steps
        train["max_examples"] = case.max_train_examples or _PREFLIGHT_SFT_EXAMPLES
    return {
        "model": model_id,
        "algorithm": case.algorithm,
        "environment": {"id": f"autoenv/{case.id}"},
        "train": train,
    }


def _check_dataset(case: PaperCase, probe: bool) -> RuleResult:
    if not probe:
        return RuleResult(
            "public_dataset",
            passed=True,
            status="skipped",
            detail="dataset probe skipped (offline); refs not verified",
        )
    train_ok, train_detail = can_fetch(
        case.resolved_train(),
        split=case.dataset.train_split,
        input_field=case.dataset.input_field,
        output_field=case.dataset.output_field,
    )
    eval_ok, eval_detail = can_fetch(
        case.resolved_eval(),
        split=case.dataset.eval_split,
        input_field=case.dataset.input_field,
        output_field=case.dataset.output_field,
    )
    ok = train_ok and eval_ok
    detail = f"train: {train_detail}; eval: {eval_detail}"
    return RuleResult("public_dataset", passed=ok, status="ok" if ok else "fail", detail=detail)


def _check_text_only(case: PaperCase) -> RuleResult:
    haystack = f"{case.goal} {case.notes}".lower()
    hits = [k for k in _VLM_HINTS if k in haystack]
    if hits:
        return RuleResult(
            "text_only",
            passed=False,
            status="fail",
            detail=f"task looks multimodal (Flash trains text-only); hints: {', '.join(hits)}",
        )
    return RuleResult(
        "text_only", passed=True, detail="no multimodal hints; Flash trains text-only"
    )


def _check_metric(case: PaperCase) -> RuleResult:
    from autoenv.eval.metrics import METRICS

    ok = case.metric.name in METRICS
    detail = (
        f"metric {case.metric.name!r} is in the registry"
        if ok
        else f"metric {case.metric.name!r} unknown; known: {', '.join(sorted(METRICS))}"
    )
    return RuleResult("automatable_metric", passed=ok, status="ok" if ok else "fail", detail=detail)


def gate_case(case: PaperCase, *, probe_dataset: bool = True) -> GateReport:
    """Run the five checks and return a ``GateReport``."""
    rules: list[RuleResult] = []

    # (b) Model supported — resolve to a catalog model (or nearest-size substitute).
    substituted = False
    try:
        match = resolve_flash_model(case.flash_model, case.base_model_paper, case.algorithm)
        substituted = match.substituted
        model_id = match.model_id
        rules.append(RuleResult("model_supported", passed=True, detail=match.detail))
    except ValueError as exc:
        model_id = case.flash_model
        rules.append(RuleResult("model_supported", passed=False, status="fail", detail=str(exc)))

    # (d-i) Algorithm supported by Flash.
    algo_ok = case.algorithm in ALGORITHMS
    rules.append(
        RuleResult(
            "algorithm_supported",
            passed=algo_ok,
            status="ok" if algo_ok else "fail",
            detail=f"algorithm {case.algorithm!r}" + ("" if algo_ok else f" not in {ALGORITHMS}"),
        )
    )

    # (c) Text-only / not a VLM task.
    rules.append(_check_text_only(case))

    # (a) Dataset public.
    rules.append(_check_dataset(case, probe_dataset))

    # (e) Metric automatable.
    rules.append(_check_metric(case))

    # (d-ii) Feasibility + cost preflight (only meaningful once the model resolved).
    estimated_usd: float | None = None
    if any(r.name == "model_supported" and r.passed for r in rules) and algo_ok:
        # Pass params_b explicitly: check_fit reads HF metadata (network) for the size, but a
        # catalog model's size is known locally via resolve_params_b — keeps the gate offline.
        fit = check_fit(model_id, case.algorithm, "RTX 5090", params_b=resolve_params_b(model_id))
        if fit.verdict == "too_big":
            rules.append(RuleResult("vram_fit", passed=False, status="fail", detail=fit.describe()))
        else:
            rules.append(RuleResult("vram_fit", passed=True, detail=fit.describe()))

        try:
            spec = spec_from_dict(_provisional_spec_dict(case, model_id))
            estimate = estimate_for_spec(spec)
            estimated_usd = estimate.total_usd
            cost_ok = estimate.total_usd <= case.max_usd
            rules.append(
                RuleResult(
                    "cost_within_budget",
                    passed=cost_ok,
                    status="ok" if cost_ok else "fail",
                    detail=f"preflight ${estimate.total_usd:.2f} vs budget ${case.max_usd:.2f}",
                )
            )
        except Exception as exc:
            rules.append(
                RuleResult(
                    "cost_within_budget",
                    passed=False,
                    status="fail",
                    detail=f"could not estimate cost: {exc}",
                )
            )

    eligible = all(r.passed for r in rules if r.status != "skipped")
    return GateReport(
        case_id=case.id,
        eligible=eligible,
        flash_model=model_id,
        rules=rules,
        model_substituted=substituted,
        estimated_usd=estimated_usd,
    )
