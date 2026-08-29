"""Allocation-free tests for the hosted per-model capacity benchmark.

Two jobs. First, prove the benchmark cannot touch production: not the app name, not the catalog, not
the router, not the billing path. Second, prove the metric arithmetic is right against hand-computed
values, so a number in the published report is trustworthy without re-deriving it.

Everything here runs with no GPU and no network.
"""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import inspect
import re
import tomllib
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from flash.serving.bench.budget import (
    SUBMISSION_STOP_FRACTION,
    BudgetExceeded,
    BudgetLedger,
    UnknownGpuRate,
    rate_for_gpu,
    usd_for_gpu_seconds,
)
from flash.serving.bench.catalog import (
    BENCH_MODELS,
    bench_base_models,
    bench_catalog_summary,
    bench_engine_overrides_for,
    bench_gpu_for,
    immutable_serving_revisions,
    tokenizer_model_for,
)
from flash.serving.bench.driver import (
    REQUEST_TIMEOUT_SECONDS,
    _absorb_event,
    _build_prompt_pool,
    _drain,
    _prompt_issuer,
    _StreamOutcome,
    _validate,
    base_model_record,
    run_cell,
)
from flash.serving.bench.metrics import (
    ERROR_CACHE_CONTAMINATED,
    ERROR_CACHE_UNVERIFIED,
    ERROR_PROMPT_LENGTH,
    ERROR_TIMEOUT,
    CellResult,
    RequestRecord,
    percentile,
    reduce_cell,
    summarize_curve,
    wilson_upper_bound,
)
from flash.serving.bench.probe import gpu_matches, probe_gdn_backend, probe_gpu
from flash.serving.bench.workload import (
    BUCKETS,
    BUCKETS_BY_NAME,
    ENABLE_THINKING,
    PROMPT_TOKEN_TOLERANCE,
    TEMPERATURE,
    build_prompt_text,
    concurrency_grid,
    corpus_seed,
    request_uid,
    reseed_prompt,
    workload_checksum,
)
from flash.serving.src.engine.model_config import SERVING_MODELS, gpu_for

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_APP = REPO_ROOT / "scripts" / "bench_hosted_capacity.py"


# ── Isolation from production ─────────────────────────────────────────────────────────────────


def test_bench_app_name_differs_from_production() -> None:
    """The benchmark app must never deploy over the live production serving app."""
    from flash.serving.app.modal_app import APP_NAME as PRODUCTION_APP

    source = BENCH_APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant)
    }
    assert names["APP_NAME"] != PRODUCTION_APP
    assert names["CACHE_VOLUME_NAME"] != "freesolo-lora-serving-hf-cache"


def test_bench_app_does_not_import_router_or_billing() -> None:
    """No router, no usage outbox, no Supabase: a benchmark must not write production billing."""
    source = BENCH_APP.read_text(encoding="utf-8")
    for forbidden in (
        "build_serving_app",
        "AdapterRouter",
        "usage_outbox",
        "DurableUsageOutbox",
        "SUPABASE_SERVICE_ROLE_KEY",
        "FREESOLO_INTERNAL_KEY",
        "asgi_app",
        "custom_domains",
    ):
        assert forbidden not in source, f"benchmark app must not reference {forbidden}"


def test_bench_app_pins_one_container_per_tier() -> None:
    """Autoscaling would silently add a second card and inflate a per-card envelope."""
    source = BENCH_APP.read_text(encoding="utf-8")
    assert "max_containers=1" in source
    assert "min_containers=0" in source
    assert "max_inputs=1" in source


def test_bench_app_builds_one_class_per_production_tier() -> None:
    """``@app.cls(gpu=...)`` fixes the GPU at decoration time, so three tiers need three classes.

    Importing the module would pull in ``modal`` and the whole serving stack, so the structure is
    asserted from the source: the factory must be called once per DISTINCT tier in the catalog.
    """
    source = BENCH_APP.read_text(encoding="utf-8")
    assert "_build_bench_engine" in source
    assert "ENGINE_BY_GPU" in source
    # No single hardcoded GPU: the tier comes from the catalog, per model.
    assert "GPU_SPEC" not in source
    tiers = {bench_gpu_for(model) for model in BENCH_MODELS}
    assert len(tiers) >= 2, "the catalog spans multiple tiers; one class cannot serve them"


def test_bench_class_names_are_modal_safe_and_distinct() -> None:
    """Modal rejects a ``<locals>`` qualname and needs a unique global name per class."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_bench_names", BENCH_APP)
    assert spec is not None
    # Read the helper without executing the module (which would import modal).
    tree = ast.parse(BENCH_APP.read_text(encoding="utf-8"))
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_bench_class_name"
    )
    namespace: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<bench>", "exec"), namespace)
    name_for = namespace["_bench_class_name"]
    names = {name_for(gpu) for gpu in {bench_gpu_for(m) for m in BENCH_MODELS}}
    assert len(names) == len({bench_gpu_for(m) for m in BENCH_MODELS})
    for name in names:
        assert name.isidentifier(), f"{name!r} is not a valid Python identifier"
    # Modal spells a non-preemptible pin with "!", which is not identifier-safe.
    assert name_for("H100!").isidentifier()


def test_bench_app_avoids_postponed_annotations() -> None:
    """``from __future__ import annotations`` is a DEPLOY blocker for a Modal class.

    It makes every annotation a string, and Modal's class-parameter validator resolves
    ``base_model: str`` through the real type object. Under postponed evaluation it receives the
    string ``"str"`` and raises ``AttributeError: 'str' object has no attribute '__name__'`` at
    decoration time. This failed on the real SDK before the first GPU was allocated; the production
    app omits the import for the same reason, so the constraint is pinned here rather than
    rediscovered by a failed deploy.
    """
    tree = ast.parse(BENCH_APP.read_text(encoding="utf-8"))
    future_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "__future__"
        for alias in node.names
    }
    assert "annotations" not in future_imports


def test_bench_catalog_measures_every_model_on_its_production_tier() -> None:
    """No GPU override: an envelope on a card the model is not served on is not its envelope."""
    assert set(bench_base_models()) == set(BENCH_MODELS)
    assert set(BENCH_MODELS) == {entry["base_model"] for entry in SERVING_MODELS}
    for model in BENCH_MODELS:
        assert bench_gpu_for(model) == gpu_for(model)


def test_bench_catalog_rejects_a_model_outside_the_hosted_catalog() -> None:
    """A typo must fail here rather than boot a card for a model nothing serves."""
    with pytest.raises(ValueError, match="unsupported benchmark base model"):
        bench_gpu_for("Qwen/Qwen3.5-9B-not-a-real-model")
    with pytest.raises(ValueError, match="unsupported benchmark base model"):
        bench_engine_overrides_for("Qwen/Qwen3.5-9B-not-a-real-model")


def test_bench_catalog_preserves_production_engine_shape() -> None:
    """Only the GPU changes, so a measured number is attributable to the card.

    ``max_loras``/``max_lora_rank`` in particular must survive: vLLM pre-allocates those buffers, so
    dropping them would free VRAM the real engine does not have.
    """
    for model in BENCH_MODELS:
        production = next(e for e in SERVING_MODELS if e["base_model"] == model)["engine"]
        bench = bench_engine_overrides_for(model)
        for key in ("max_loras", "max_lora_rank", "max_model_len", "max_num_seqs"):
            assert bench[key] == production[key], f"{model}.{key} drifted from production"


def test_bench_overrides_cannot_mutate_the_production_catalog() -> None:
    """The catalog hands back a copy; a caller editing it must not reshape production's engine."""
    overrides = bench_engine_overrides_for("Qwen/Qwen3.5-9B")
    overrides["max_num_seqs"] = 9999
    assert bench_engine_overrides_for("Qwen/Qwen3.5-9B")["max_num_seqs"] != 9999


def test_35b_stays_bf16_in_the_benchmark() -> None:
    """The bf16 override is the deployable path; FP8 fused-MoE LoRA does not compile."""
    overrides = bench_engine_overrides_for("Qwen/Qwen3.6-35B-A3B")
    assert overrides["quantization"] is None
    assert overrides["serve_model_id"] == "Qwen/Qwen3.6-35B-A3B"


def test_27b_is_in_the_hosted_catalog() -> None:
    """27B moved into SERVING_MODELS on dev, so it is measured as an ordinary hosted model."""
    assert "Qwen/Qwen3.8-27B" in bench_base_models()
    assert bench_gpu_for("Qwen/Qwen3.8-27B") == gpu_for("Qwen/Qwen3.8-27B")


def test_catalog_summary_reports_each_model_tier_and_engine_shape() -> None:
    rows = {row["base_model"]: row for row in bench_catalog_summary()}
    assert set(rows) == set(BENCH_MODELS)
    for model, row in rows.items():
        assert row["gpu"] == gpu_for(model)
        assert row["max_model_len"] == 32768
        assert row["max_num_seqs"] >= 1


def test_base_model_record_registers_no_adapter() -> None:
    """Base-only: ``serve_base_model`` routes to base weights with ``lora_request=None``."""
    record = base_model_record("Qwen/Qwen3.5-9B")
    assert record["serve_base_model"] is True
    assert record["org_id"] is None
    assert record.get("checkpoint") is None


def test_base_record_validates_against_the_real_schema() -> None:
    """The forwarded record must satisfy production's own AdapterRecord."""
    from flash.serving.src.io.schemas import AdapterRecord

    parsed = AdapterRecord.model_validate(base_model_record("Qwen/Qwen3.5-9B"))
    assert parsed.serve_base_model is True
    assert parsed.base_model == "Qwen/Qwen3.5-9B"


# ── Workload contract ────────────────────────────────────────────────────────────────────────


def test_buckets_are_the_three_preregistered_shapes() -> None:
    assert [b.name for b in BUCKETS] == [
        "short_interactive",
        "medium_generation",
        "near_32k",
    ]
    assert [b.target_input_tokens for b in BUCKETS] == [512, 8192, 31744]


def test_near_32k_bucket_fits_the_configured_context() -> None:
    """Input + output must stay inside max_model_len or every request would fail on overflow."""
    near = BUCKETS[-1]
    for model in BENCH_MODELS:
        assert near.total_token_ceiling <= bench_engine_overrides_for(model)["max_model_len"]


def test_prompts_are_unique_from_their_first_characters() -> None:
    """A shared prefix would be served from vLLM's prefix cache and measure a cache hit."""
    a = build_prompt_text(request_uid("short_interactive", 4, 0, 0), 64)
    b = build_prompt_text(request_uid("short_interactive", 4, 0, 1), 64)
    assert a != b
    assert a[:24] != b[:24], "prompts must diverge immediately, not only at the tail"


def test_prompt_text_is_deterministic() -> None:
    uid = request_uid("medium_generation", 8, 1, 3)
    assert build_prompt_text(uid, 128) == build_prompt_text(uid, 128)


def test_decoding_contract_is_fixed_and_thinking_is_on() -> None:
    assert TEMPERATURE == 0.0
    assert ENABLE_THINKING is True


def test_workload_checksum_is_stable() -> None:
    assert workload_checksum() == workload_checksum()
    assert len(workload_checksum()) == 64


@pytest.mark.parametrize(
    ("cap", "expected"),
    [
        (8, (1, 2, 4, 8, 12, 16)),
        (4, (1, 2, 4, 6, 8)),
        (1, (1, 2)),
    ],
)
def test_concurrency_grid_spans_and_exceeds_the_engine_cap(cap: int, expected: tuple) -> None:
    """The grid must pass the cap: saturation is only falsifiable with a degraded point."""
    assert concurrency_grid(cap) == expected


def test_concurrency_grid_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="max_num_seqs must be >= 1"):
        concurrency_grid(0)


# ── Metric arithmetic ────────────────────────────────────────────────────────────────────────


def _record(
    uid: str,
    *,
    ok: bool = True,
    tokens: int = 100,
    ttft: float = 0.5,
    latency: float = 2.0,
    cached: int = 0,
    error: str | None = None,
) -> RequestRecord:
    record = RequestRecord(
        uid=uid,
        base_model="Qwen/Qwen3.5-9B",
        bucket="short_interactive",
        concurrency=4,
        block=0,
        started_at=0.0,
    )
    record.first_token_at = ttft
    record.finished_at = latency
    record.prompt_tokens = 512
    record.completion_tokens = tokens
    record.cached_tokens = cached
    record.finish_reason = "stop"
    record.engine_replica_id = "replica-a"
    record.ok = ok
    record.error = error
    return record


def test_percentile_matches_hand_computed_values() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 1.0) == 4.0
    assert percentile(values, 0.5) == 2.5
    assert percentile([], 0.5) is None
    assert percentile([7.0], 0.99) == 7.0


def test_throughput_uses_engine_tokens_over_full_wall_time() -> None:
    records = [_record(f"u{i}", tokens=100) for i in range(10)]
    cell = reduce_cell(
        records,
        base_model="Qwen/Qwen3.5-9B",
        bucket="short_interactive",
        concurrency=4,
        block=0,
        wall_seconds=10.0,
    )
    assert cell.completion_tokens_total == 1000
    assert cell.output_tokens_per_second == pytest.approx(100.0)
    assert cell.successful_rps == pytest.approx(1.0)
    assert cell.attempted_rps == pytest.approx(1.0)


def test_failures_count_against_the_error_rate_and_not_throughput() -> None:
    """A failed request must never contribute tokens, and never be silently dropped."""
    records = [_record(f"ok{i}") for i in range(8)]
    records += [_record(f"bad{i}", ok=False, tokens=999, error=ERROR_TIMEOUT) for i in range(2)]
    cell = reduce_cell(
        records,
        base_model="Qwen/Qwen3.5-9B",
        bucket="short_interactive",
        concurrency=4,
        block=0,
        wall_seconds=10.0,
    )
    assert cell.attempted == 10
    assert cell.succeeded == 8
    assert cell.error_rate == pytest.approx(0.2)
    assert cell.completion_tokens_total == 800, "failed requests must not contribute tokens"
    assert cell.error_breakdown == {ERROR_TIMEOUT: 2}
    assert cell.feasible is False


def test_clean_but_tiny_sample_is_not_feasible() -> None:
    """Zero observed errors on a small sample is not evidence of a sub-1% error rate."""
    cell = reduce_cell(
        [_record(f"u{i}") for i in range(5)],
        base_model="Qwen/Qwen3.5-9B",
        bucket="short_interactive",
        concurrency=4,
        block=0,
        wall_seconds=10.0,
    )
    assert cell.error_rate == 0.0
    assert cell.error_rate_upper_bound > 0.01
    assert cell.feasible is False
    # But the sample is SHALLOW, not bad. Conflating the two would make every short cell read as a
    # failure and collapse the curve at its first point.
    assert cell.error_bound_resolved is False
    assert cell.degraded is False


def test_a_demonstrated_error_rate_is_degraded_but_a_shallow_one_is_not() -> None:
    """``degraded`` answers "did it fail"; ``error_bound_resolved`` answers "did we look long enough".

    Both were once one boolean, which meant a clean 20-request cell and a 5%-error 300-request cell
    were indistinguishable, and ``summarize_curve`` called the first shallow cell the saturation
    point -- erasing the envelope it was supposed to publish.
    """
    records = [_record(f"u{i}") for i in range(285)]
    records += [_record(f"bad{i}", ok=False, error=ERROR_TIMEOUT) for i in range(15)]
    cell = reduce_cell(
        records,
        base_model="Qwen/Qwen3.5-9B",
        bucket="short_interactive",
        concurrency=4,
        block=0,
        wall_seconds=60.0,
    )
    assert cell.error_rate == pytest.approx(0.05)
    assert cell.error_bound_resolved is True
    assert cell.feasible is False
    assert cell.degraded is True


def test_error_bound_resolves_only_at_the_sample_depth_wilson_requires() -> None:
    """A zero-failure cell needs 268 clean attempts before its bound clears 1% (z=1.645)."""
    assert wilson_upper_bound(0, 267) > 0.01
    assert wilson_upper_bound(0, 268) <= 0.01


def test_large_clean_sample_becomes_feasible() -> None:
    cell = reduce_cell(
        [_record(f"u{i}") for i in range(600)],
        base_model="Qwen/Qwen3.5-9B",
        bucket="short_interactive",
        concurrency=4,
        block=0,
        wall_seconds=60.0,
    )
    assert cell.feasible is True
    assert cell.error_bound_resolved is True
    assert cell.degraded is False
    assert cell.latency_seconds["p99_descriptive_only"] is False


def test_p99_is_flagged_descriptive_on_small_samples() -> None:
    cell = reduce_cell(
        [_record(f"u{i}") for i in range(30)],
        base_model="Qwen/Qwen3.5-9B",
        bucket="short_interactive",
        concurrency=4,
        block=0,
        wall_seconds=10.0,
    )
    assert cell.latency_seconds["p99_descriptive_only"] is True


def test_wilson_bound_is_conservative_and_monotone() -> None:
    assert wilson_upper_bound(0, 0) == 1.0
    assert wilson_upper_bound(0, 20) > wilson_upper_bound(0, 2000)
    assert wilson_upper_bound(0, 2000) < 0.01
    assert wilson_upper_bound(5, 100) > 0.05


def test_reduce_cell_rejects_zero_wall_time() -> None:
    """Zero wall time would make every rate infinite rather than raising."""
    with pytest.raises(ValueError, match="wall_seconds must be positive"):
        reduce_cell([], base_model="m", bucket="b", concurrency=1, block=0, wall_seconds=0.0)


def test_cache_contaminated_requests_are_errors_not_successes() -> None:
    """A prefix-cache hit is an invalid sample; counting it would inflate throughput."""
    records = [_record(f"u{i}") for i in range(4)]
    records.append(_record("cached", ok=False, cached=480, error=ERROR_CACHE_CONTAMINATED))
    cell = reduce_cell(
        records,
        base_model="Qwen/Qwen3.5-9B",
        bucket="short_interactive",
        concurrency=4,
        block=0,
        wall_seconds=10.0,
    )
    assert cell.error_breakdown == {ERROR_CACHE_CONTAMINATED: 1}
    assert cell.succeeded == 4


# ── Curve summarization ──────────────────────────────────────────────────────────────────────


def _cell(concurrency: int, tps: float, p95: float, feasible: bool = True) -> CellResult:
    """A deep cell: 600 attempts resolves the Wilson bound, so ``feasible`` means what it says.

    No drain, so every success landed inside the window -- which is what makes this cell a valid
    stand-in for a real one rather than one that reads as degraded for lack of steady-state work.
    """
    return CellResult(
        base_model="Qwen/Qwen3.5-9B",
        bucket="short_interactive",
        concurrency=concurrency,
        block=0,
        wall_seconds=60.0,
        attempted=600,
        succeeded=600 if feasible else 300,
        succeeded_in_window=600 if feasible else 300,
        failed=0 if feasible else 300,
        error_rate=0.0 if feasible else 0.5,
        output_tokens_per_second=tps,
        latency_seconds={"p50": p95 / 2, "p95": p95, "p99": p95},
        error_bound_resolved=True,
        feasible=feasible,
    )


def _shallow_cell(concurrency: int, tps: float, p95: float) -> CellResult:
    """A clean but shallow cell: no failures observed, bound unresolved. Usable, not degraded."""
    return CellResult(
        base_model="Qwen/Qwen3.5-9B",
        bucket="near_32k",
        concurrency=concurrency,
        block=0,
        wall_seconds=60.0,
        attempted=20,
        succeeded=20,
        succeeded_in_window=20,
        failed=0,
        error_rate=0.0,
        output_tokens_per_second=tps,
        latency_seconds={"p50": p95 / 2, "p95": p95, "p99": p95},
        error_bound_resolved=False,
        feasible=False,
    )


def test_curve_finds_ceiling_and_smallest_knee() -> None:
    """The knee is the CHEAPEST concurrency buying ~all the throughput, not the fastest point."""
    cells = [
        _cell(1, 100.0, 1.0),
        _cell(2, 180.0, 1.1),
        _cell(4, 380.0, 1.3),
        _cell(8, 400.0, 2.0),
    ]
    curve = summarize_curve(cells)
    assert curve["throughput_ceiling_tokens_per_second"] == 400.0
    assert curve["knee_concurrency"] == 4
    assert curve["feasible_points"] == 4


def test_curve_detects_saturation_from_flat_throughput_and_rising_latency() -> None:
    cells = [
        _cell(1, 100.0, 1.0),
        _cell(2, 200.0, 1.1),
        _cell(4, 205.0, 2.0),
    ]
    curve = summarize_curve(cells)
    assert curve["saturation_concurrency"] == 4


def test_curve_treats_a_degraded_point_as_saturation() -> None:
    cells = [_cell(1, 100.0, 1.0), _cell(2, 200.0, 1.1), _cell(4, 150.0, 5.0, feasible=False)]
    curve = summarize_curve(cells)
    assert curve["saturation_concurrency"] == 4
    assert curve["feasible_points"] == 2
    assert curve["usable_points"] == 2


def test_curve_with_only_degraded_points_reports_nothing() -> None:
    curve = summarize_curve([_cell(1, 10.0, 1.0, feasible=False)])
    assert curve["throughput_ceiling_tokens_per_second"] is None
    assert curve["knee_concurrency"] is None
    assert curve["measured_points"] == 1


def test_a_shallow_clean_sweep_still_publishes_its_envelope() -> None:
    """near_32k cannot afford 268 attempts per point, and must not therefore report nothing.

    Every point here is clean but under-sampled, so ``feasible_points`` is 0. The curve is still
    real: the ceiling, knee, and saturation come from the throughput and latency actually measured.
    What stays unresolved is the error-rate BOUND, reported separately.
    """
    cells = [
        _shallow_cell(1, 60.0, 8.0),
        _shallow_cell(2, 110.0, 9.0),
        _shallow_cell(4, 190.0, 11.0),
        _shallow_cell(8, 188.0, 26.0),
    ]
    curve = summarize_curve(cells)
    assert curve["throughput_ceiling_tokens_per_second"] == pytest.approx(190.0)
    assert curve["knee_concurrency"] == 4
    assert curve["saturation_concurrency"] == 8
    assert curve["feasible_points"] == 0
    assert curve["bound_resolved_points"] == 0
    assert curve["usable_points"] == 4


# ── Provenance ───────────────────────────────────────────────────────────────────────────────


def test_gpu_match_accepts_decorated_names_and_rejects_the_wrong_card() -> None:
    assert gpu_matches({"gpu": {"name": "NVIDIA L40S"}}, "L40S") is True
    assert gpu_matches({"gpu": {"name": "NVIDIA H100 80GB HBM3"}}, "H100") is True
    assert gpu_matches({"gpu": {"name": "NVIDIA H200"}}, "H200") is True
    assert gpu_matches({"gpu": {"name": "NVIDIA H200"}}, "H100") is False
    assert gpu_matches({}, "H200") is False


def test_gpu_match_is_exact_token_not_substring() -> None:
    """``"L4" in "NVIDIA L40S"`` is True, so a substring match would accept the wrong card.

    Dev's three tiers do not collide today, which is exactly why this has to be asserted rather
    than left to chance: an L4 tier added later would silently pass on an L40S card.
    """
    assert gpu_matches({"gpu": {"name": "NVIDIA L40S"}}, "L4") is False
    assert gpu_matches({"gpu": {"name": "NVIDIA H100 80GB HBM3"}}, "H1") is False


def test_gpu_match_ignores_modal_non_preemptible_pin() -> None:
    """Modal spells a pinned tier ``"H100!"``; the card reports itself without the marker."""
    assert gpu_matches({"gpu": {"name": "NVIDIA H100 80GB HBM3"}}, "H100!") is True


def test_record_json_carries_no_generated_text() -> None:
    """Evidence files are published; model output must never be in them."""
    payload = _record("u0").to_json()
    assert "text" not in payload
    assert "content" not in payload
    assert payload["latency_seconds"] == pytest.approx(2.0)
    assert payload["ttft_seconds"] == pytest.approx(0.5)


# ── Budget ───────────────────────────────────────────────────────────────────────────────────


def test_gpu_seconds_price_matches_the_rate_of_its_own_tier() -> None:
    assert usd_for_gpu_seconds(3600, "L40S") == pytest.approx(1.9512)
    assert usd_for_gpu_seconds(3600, "H100") == pytest.approx(3.9492)
    assert usd_for_gpu_seconds(3600, "H200") == pytest.approx(4.5396)
    assert usd_for_gpu_seconds(0, "H200") == 0.0


def test_the_planned_sweep_costs_what_the_plan_quoted() -> None:
    """The quote presented for authorization must be reproducible from the code that spends."""
    assert usd_for_gpu_seconds(2280, "L40S") == pytest.approx(1.24, abs=0.01)
    assert usd_for_gpu_seconds(2560, "H100") == pytest.approx(2.81, abs=0.01)
    assert usd_for_gpu_seconds(3170, "H200") == pytest.approx(4.00, abs=0.01)


def test_a_non_preemptible_pin_prices_as_the_same_card() -> None:
    assert rate_for_gpu("H100!") == rate_for_gpu("H100")


def test_an_unrecorded_tier_fails_closed_instead_of_guessing() -> None:
    """Pricing an unknown card at a neighbour's rate is how a sweep overspends without tripping."""
    with pytest.raises(UnknownGpuRate):
        rate_for_gpu("B300")
    with pytest.raises(UnknownGpuRate):
        usd_for_gpu_seconds(3600, "A100")


def test_a_ledger_has_no_default_ceiling() -> None:
    """The authorized amount comes from the user; an invented default is not authorization."""
    with pytest.raises(TypeError):
        BudgetLedger()
    with pytest.raises(ValueError, match="explicitly authorized"):
        BudgetLedger(ceiling_usd=0.0)


def test_ledger_refuses_a_reservation_past_the_ceiling() -> None:
    ledger = BudgetLedger(ceiling_usd=25.0)
    ledger.reserve("lane-a", 3600 * 4, "H200")  # $18.16, under the $20 stop
    with pytest.raises(BudgetExceeded, match="ceiling"):
        ledger.reserve("lane-b", 3600 * 4, "H200")
    assert ledger.committed_usd <= 25.0


def test_reserve_refuses_in_the_band_between_the_stop_and_the_ceiling() -> None:
    """``reserve`` is gated on the SUBMISSION STOP, not merely on the ceiling.

    Gating only the ceiling made the stop advisory: a lane projecting between the stop and the
    ceiling was admitted, consuming the very reserve held back for delayed charges and teardown, and
    the reserve survived only where a caller remembered to ask ``can_submit`` first. Every caller of
    ``reserve`` is starting a new lane, so the stop is the correct bar for all of them.

    On a $25 ceiling the stop is $20. After $18.16 is held, a $4.54 H200 hour projects to $22.70 --
    UNDER the ceiling and OVER the stop, which is exactly the band the ceiling check cannot see.
    """
    ledger = BudgetLedger(ceiling_usd=25.0)
    ledger.reserve("lane-a", 3600 * 4, "H200")  # $18.16
    assert ledger.can_submit(3600, "H200") is False
    with pytest.raises(BudgetExceeded, match="submission stop"):
        ledger.reserve("lane-b", 3600, "H200")
    # The band is real: the refused lane would have stayed under the ceiling.
    assert ledger.committed_usd + usd_for_gpu_seconds(3600, "H200") < ledger.ceiling_usd
    assert ledger.committed_usd <= ledger.submission_stop_usd


def test_submission_stops_before_the_reserve_is_consumed() -> None:
    """New lanes stop at the submission threshold so teardown and lag stay funded.

    On a $20 ceiling the stop is $16. Three H200-hours is $13.62, so a fourth would cross the stop
    while the ceiling still holds ~$6 for delayed charges and teardown.
    """
    ledger = BudgetLedger(ceiling_usd=20.0)
    ledger.reserve("spent", 3600 * 3, "H200")
    assert ledger.can_submit(3600, "H200") is False
    assert ledger.remaining_usd > 0
    # The stop is not the ceiling: reserve capacity survives for settlement lag.
    assert ledger.committed_usd < 20.0


def test_settling_replaces_the_reservation_with_measured_cost() -> None:
    ledger = BudgetLedger(ceiling_usd=20.0)
    entry = ledger.reserve("lane", 3600.0, "H200")
    assert ledger.committed_usd == pytest.approx(4.5396)
    ledger.settle(entry, 1800.0)
    assert ledger.committed_usd == pytest.approx(2.2698)
    assert ledger.remaining_usd > 20.0 - 3


def test_a_mixed_tier_sweep_settles_each_lane_at_its_own_rate() -> None:
    """One rate applied to every lane would misprice the whole campaign in a single direction."""
    ledger = BudgetLedger(ceiling_usd=20.0)
    cheap = ledger.reserve("9b", 3600.0, "L40S")
    dear = ledger.reserve("35b", 3600.0, "H200")
    ledger.settle(cheap, 3600.0)
    ledger.settle(dear, 3600.0)
    assert cheap.settled_usd == pytest.approx(1.9512)
    assert dear.settled_usd == pytest.approx(4.5396)


def test_ledger_serializes_its_rate_snapshot() -> None:
    payload = BudgetLedger(ceiling_usd=20.0).to_json()
    assert payload["ceiling_usd"] == 20.0
    assert payload["usd_per_gpu_hour"]["H200"] == pytest.approx(4.5396)
    assert payload["usd_per_gpu_hour"]["L40S"] == pytest.approx(1.9512)
    assert payload["rate_snapshot_date"]


# ── Stream validation: the four gates a request must clear to count as a success ───────────────


def _outcome(**overrides: object) -> _StreamOutcome:
    """A stream that would validate clean, so each test perturbs exactly one field."""
    outcome = _StreamOutcome()
    outcome.saw_final = True
    outcome.finish_reason = "stop"
    outcome.prompt_tokens = 512
    outcome.completion_tokens = 128
    outcome.cached_tokens = 0
    outcome.cached_tokens_reported = True
    outcome.first_token_at = 0.5
    for key, value in overrides.items():
        setattr(outcome, key, value)
    return outcome


def test_a_clean_stream_validates() -> None:
    record = _record("u0", ok=False)
    record.ok = False
    _validate(_outcome(), record)
    assert record.ok is True
    assert record.error is None


def test_an_unreported_cached_token_count_is_not_a_cache_miss() -> None:
    """``_num_cached_tokens`` returns 0 when the attribute is absent, so absence looks like zero.

    A build that stopped reporting would make every request read as perfectly uncached, and the
    whole campaign would publish as clean while being entirely unverified.
    """
    record = _record("u0")
    record.ok = False
    _validate(_outcome(cached_tokens_reported=False, cached_tokens=0), record)
    assert record.ok is False
    assert record.error == ERROR_CACHE_UNVERIFIED


def test_a_reported_zero_is_a_real_cache_miss_and_passes() -> None:
    record = _record("u0")
    record.ok = False
    _validate(_outcome(cached_tokens_reported=True, cached_tokens=0), record)
    assert record.ok is True


def test_a_prefix_cache_hit_is_an_error_not_a_fast_success() -> None:
    record = _record("u0")
    record.ok = False
    _validate(_outcome(cached_tokens=64), record)
    assert record.ok is False
    assert record.error == ERROR_CACHE_CONTAMINATED


# ── Drain: an abandoned request is a failure, not an absence ──────────────────────────────────


def test_the_drain_counts_unfinished_requests_instead_of_dropping_them() -> None:
    """Cancelling in-flight tasks removes them from the numerator AND the denominator.

    ``task.cancel()`` raises at the await point, the suppress swallows it, and the append never
    runs -- so an overloaded cell reports a CLEANER error rate than it earned. That is the one
    direction a capacity claim must never err in.
    """
    # `run_cell` delegates the drain to `_drain`, and calls it inside its `finally`.
    assert "_drain(" in inspect.getsource(run_cell), "run_cell no longer drains"
    drain = inspect.getsource(_drain)
    assert "timeout=REQUEST_TIMEOUT_SECONDS" in drain, "the drain must be bounded, not unbounded"
    assert "still_pending" in drain
    assert "ERROR_TIMEOUT" in drain, "a cancelled request must be RECORDED as a timeout"
    # The record is appended after the cancel, outside the suppress that swallows CancelledError.
    cancel_at = drain.index("task.cancel()")
    append_at = drain.index("drained.append", cancel_at)
    assert append_at > cancel_at


def test_the_drain_bound_is_a_requests_own_timeout() -> None:
    """Each issued request already carries this bound, so the drain terminates rather than hangs."""
    assert REQUEST_TIMEOUT_SECONDS > 0
    assert asyncio.iscoroutinefunction(run_cell)


# ── GDN backend: assert vLLM's decision, never boot success ───────────────────────────────────


def test_gdn_probe_records_unknown_rather_than_assuming_the_fast_path() -> None:
    """vLLM is not installed here, so the probe must report why -- not default to a backend.

    On Blackwell the failure mode is ``warning_once`` then ``return backend, "triton"``: no raise.
    An unrepaired boot succeeds, serves, and bills the fast-card rate on the slow kernel, so a
    probe that inferred the backend from boot success would publish a number that would not ship.
    """
    result = probe_gdn_backend("Qwen/Qwen3.6-35B-A3B")
    assert result["base_model"] == "Qwen/Qwen3.6-35B-A3B"
    assert result["resolved"] is None
    assert result.get("reason"), "an unresolved probe must say why"


def test_gdn_probe_invokes_the_resolver_rather_than_checking_it_exists() -> None:
    """Presence of ``_resolve_gdn_prefill_backend`` says nothing about which backend it picks."""
    source = inspect.getsource(probe_gdn_backend)
    assert "resolver()" in source, "the resolver must be CALLED, not merely found"
    call_at = source.index("resolver()")
    present_at = source.index('result["resolver_present"]')
    assert call_at > present_at


# ── Sample depth is a property of the request shape ───────────────────────────────────────────


def test_each_bucket_carries_its_own_depth_floor() -> None:
    """268 attempts is minutes at 512/128 and hours at 31744/512, so depth cannot be one constant."""
    short = BUCKETS_BY_NAME["short_interactive"]
    near = BUCKETS_BY_NAME["near_32k"]
    assert short.min_requests >= 268, "a short turn can afford the depth the bound needs"
    assert near.min_requests < short.min_requests, "a near-32k turn cannot, and must not pretend to"
    for bucket in BUCKETS:
        assert bucket.min_seconds > 0
        assert bucket.max_seconds >= bucket.min_seconds


def test_run_cell_takes_its_floors_from_the_bucket() -> None:
    """A caller-supplied default would silently re-flatten depth across all three shapes."""
    params = inspect.signature(run_cell).parameters
    for name in ("min_seconds", "min_requests", "max_seconds"):
        assert params[name].default is None, f"{name} must default to the bucket's own value"


# ── The engine the benchmark boots is production's engine ─────────────────────────────────────


def test_bench_overrides_build_a_real_engine_arg_set_for_every_model() -> None:
    """Resolve ``engine_args_for`` offline for all three models, before anything is billed.

    A prior campaign skipped exactly this and paid a 75s B200 boot to discover that ``cfg`` is the
    SETTINGS module, not ``model_config`` -- an ``AttributeError`` that only surfaces once the card
    is already running. The conftest's vLLM stub makes the same resolution free.
    """
    from flash.serving.src.engine.boot import engine_args_for
    from flash.serving.src.store import settings as cfg

    for model in BENCH_MODELS:
        args = engine_args_for(model, bench_engine_overrides_for(model), cfg)
        assert args["max_model_len"] == 32768, model
        assert args["enable_lora"] is True, model
        # vLLM pre-allocates the LoRA buffers at init, so dropping these would free VRAM the real
        # engine does not have and inflate the envelope.
        assert args["max_loras"] >= 1, model
        assert args["max_lora_rank"] >= 1, model
        assert args["model"], model


def test_bench_engine_args_match_production_engine_args_exactly() -> None:
    """Same inputs, same engine. An envelope measured on a different engine is not an envelope."""
    from flash.serving.src.engine.boot import engine_args_for
    from flash.serving.src.engine.model_config import engine_overrides_for
    from flash.serving.src.store import settings as cfg

    for model in BENCH_MODELS:
        bench = engine_args_for(model, bench_engine_overrides_for(model), cfg)
        production = engine_args_for(model, engine_overrides_for(model), cfg)
        assert bench == production, f"{model} boots a different engine than production"


def test_a_drained_request_records_a_timeout_instead_of_raising() -> None:
    """The drain fallback must CONSTRUCT. `started_at` is required and has no default, so omitting
    it raised TypeError here, aborting the bucket and discarding every record of an expensive
    sweep -- the exact loss the drain exists to prevent."""
    drain = inspect.getsource(_drain)
    assert "started_at=" in drain, "drain builds RequestRecord without its required started_at"

    required = {
        name
        for name, param in inspect.signature(RequestRecord).parameters.items()
        if param.default is inspect.Parameter.empty
    }
    call = drain[drain.index("RequestRecord(") : drain.index(")", drain.index("ERROR_TIMEOUT"))]
    for name in required:
        assert f"{name}=" in call, f"drain record omits required field {name!r}"

    # And it actually builds, rather than merely mentioning the right words.
    record = RequestRecord(
        uid="drain-1",
        base_model="Qwen/Qwen3.5-9B",
        bucket="short_interactive",
        concurrency=4,
        block=0,
        started_at=1.5,
        error=ERROR_TIMEOUT,
    )
    assert record.ok is False
    assert record.error == ERROR_TIMEOUT


def test_prompts_are_fitted_before_the_clock_starts() -> None:
    """Fitting is repeated synchronous tokenization. On the event loop it blocks other in-flight
    streams and inflates their TTFT, which at 31k input exceeds the effect being measured."""
    # `run_cell` builds the pool through `_prompt_issuer`, which fits every prompt eagerly and
    # returns a callable that only pops. The ordering assertion is against run_cell; the "no fitting
    # after the clock starts" assertion has to cover both, since the pop lives in the helper.
    source = inspect.getsource(run_cell)
    pool_at = source.index("_prompt_issuer(")
    origin_at = source.index("origin = time.monotonic()")
    assert pool_at < origin_at, "prompt pool must be built before the measured window opens"
    assert "fit_prompt_to_tokens" not in source[origin_at:], "fitting leaked into the measured loop"

    issuer = inspect.getsource(_prompt_issuer)
    handout = issuer[issuer.index("def _next_prompt") :]
    assert "fit_prompt_to_tokens" not in handout, "fitting leaked into the per-request hand-out"


def _fake_pool(concurrency: int, block: int = 0) -> tuple[list[Any], list[tuple[str, str | None]]]:
    """Build a pool with the fitter stubbed; returns the pool and the (uid, corpus) calls made."""
    calls: list[tuple[str, str | None]] = []

    def fake_fit(
        _tokenizer: object, uid: str, target: int, *, corpus: str | None = None, **_kw: object
    ) -> tuple[list[dict[str, str]], int]:
        calls.append((uid, corpus))
        # Mirror the real prompt's shape: a per-request header line, then the corpus-seeded body.
        return [{"role": "user", "content": f"{uid}\nbody-{corpus}"}], target

    bucket = BUCKETS_BY_NAME["short_interactive"]
    with mock.patch("flash.serving.bench.driver.fit_prompt_to_tokens", fake_fit):
        pool = _build_prompt_pool(
            object(),
            bucket,
            concurrency=concurrency,
            block=block,
            min_requests=bucket.min_requests,
        )
    return pool, calls


def test_the_prompt_pool_covers_the_cells_request_floor() -> None:
    """A pool shorter than the floor would wrap more often, and every wrap must be reseeded."""
    bucket = BUCKETS_BY_NAME["short_interactive"]
    pool, calls = _fake_pool(concurrency=8)
    assert len(pool) >= bucket.min_requests
    uids = [uid for uid, _, _ in pool]
    # Compare the sorted list to its deduplicated form rather than only counting: a length check
    # alone reads clean when a uid is MISSING and another is duplicated, which is the exact shape a
    # broken index would produce.
    assert sorted(uids) == sorted({*uids}), "pool contains duplicate prompts"
    assert len(calls) == len(pool)


def test_the_filler_corpus_is_held_constant_across_concurrency_points() -> None:
    """A curve must vary offered load and NOTHING else.

    Seeding the prompt body from `request_uid` gave every concurrency point a different corpus, so
    two points on one curve differed in load AND in which words were sent. The body is now seeded
    from `corpus_seed`, which carries no concurrency.
    """
    _, low = _fake_pool(concurrency=2)
    _, high = _fake_pool(concurrency=16)

    low_corpora = [corpus for _, corpus in low]
    high_corpora = [corpus for _, corpus in high]
    shared = min(len(low_corpora), len(high_corpora))
    assert low_corpora[:shared] == high_corpora[:shared], (
        "the filler corpus changed with concurrency, so the curve varies more than load"
    )
    # The uids still carry it, because a request needs an id unique across the whole cell.
    assert [uid for uid, _ in low][:shared] != [uid for uid, _ in high][:shared]
    for _, corpus in low:
        assert corpus is not None
        assert "-c" not in corpus, f"corpus seed leaked a concurrency point: {corpus!r}"


def test_a_cell_that_outruns_its_pool_reseeds_instead_of_reusing_a_prompt() -> None:
    """Re-sending a pooled prompt is served from prefix cache, which `_validate` marks
    ERROR_CACHE_CONTAMINATED -- so the FASTER a cell ran, the more error rate it would earn."""
    pool, _ = _fake_pool(concurrency=4)
    first_uid, first_messages, _ = pool[0]

    wrapped_uid = request_uid("short_interactive", 4, 0, len(pool))
    reseeded = reseed_prompt(first_messages, wrapped_uid)

    original = first_messages[0]["content"]
    rewritten = reseeded[0]["content"]
    assert rewritten != original, "reseeding produced the identical prompt"
    assert rewritten[0] != original[0], "prompts must diverge at character ZERO, not later"
    # The body is reused verbatim: reseeding must not re-tokenize inside the measured window.
    assert rewritten.split("\n", 1)[1] == original.split("\n", 1)[1]
    assert first_uid not in rewritten, "the reseeded prompt still carries the original request id"

    # And the driver actually takes this path rather than merely being able to.
    issuer = inspect.getsource(_prompt_issuer)
    nxt = issuer[issuer.index("def _next_prompt") :]
    assert "reseed_prompt(" in nxt, "the wrap path re-sends a pooled prompt instead of reseeding it"
    assert "_prompt_issuer(" in inspect.getsource(run_cell), "run_cell bypasses the issuer"


def test_no_ttfa_is_published_because_the_raw_stream_cannot_measure_it() -> None:
    """`_stream_generate` emits type/text/usage and never `reasoning_content`, so the old TTFA
    condition was true for every delta and TTFA collapsed onto TTFT. Publishing TTFT twice under
    two names is worse than publishing one number."""
    generation = (REPO_ROOT / "flash" / "serving" / "src" / "engine" / "generation.py").read_text()
    assert '"reasoning_content"' not in generation

    record = _record("u0")
    assert not hasattr(record, "first_answer_at")
    assert not hasattr(record, "ttfa")
    assert "ttfa" not in record.to_json()

    cell = reduce_cell(
        [_record("u1")],
        base_model="Qwen/Qwen3.5-9B",
        bucket="short_interactive",
        concurrency=1,
        block=0,
        wall_seconds=10.0,
    )
    assert "ttfa_seconds" not in cell.to_json()


def test_gpu_identity_never_creates_a_cuda_context_in_this_process() -> None:
    """A parent-process CUDA context stole the headroom EngineCore needs for FlashInfer's first
    decode workspace and OOM-killed the 35B engine on its first request. The probe runs before every
    warmup and every sweep, so a torch-based probe would reproduce that outage."""
    # Parsed, not grepped: the docstring names `torch.cuda` to explain why it is avoided, and a
    # substring check would match that prose and pass regardless of what the code does.
    tree = ast.parse(inspect.getsource(probe_gpu).strip())
    called = {
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    cuda_calls = sorted(name for name in called if name.startswith("torch.cuda"))
    assert not cuda_calls, f"probe_gpu initializes CUDA in the parent process via {cuda_calls}"
    assert any(name.startswith("pynvml.nvml") for name in called), "identity is not read from NVML"
    assert "pynvml.nvmlDeviceGetHandleByIndex" in called

    post_mortem = (
        REPO_ROOT / "flash" / "serving" / "src" / "engine" / "lora_engine.py"
    ).read_text()
    assert "extra CUDA context" in post_mortem, "the outage this guards is no longer documented"


def test_the_workload_checksum_covers_sample_depth() -> None:
    """min_requests decides whether the error bound resolves at all. Two campaigns differing only
    in depth are different measurements and must not share a checksum."""
    baseline = workload_checksum()
    bucket = BUCKETS_BY_NAME["short_interactive"]

    for field_name, value in (
        ("min_requests", bucket.min_requests + 1),
        ("min_seconds", bucket.min_seconds + 1.0),
        ("max_seconds", bucket.max_seconds + 1.0),
    ):
        shifted = replace(bucket, **{field_name: value})
        others = [b for b in BUCKETS if b.name != bucket.name]
        with mock.patch("flash.serving.bench.workload.BUCKETS", [shifted, *others]):
            assert workload_checksum() != baseline, f"{field_name} does not enter the checksum"


def test_the_paid_entrypoint_reserves_budget_before_it_allocates_a_gpu() -> None:
    """Every path in `main` allocates. A ceiling that is advertised but never consulted stops
    nothing, and a typo in --mode would launch an unbudgeted sweep."""
    source = BENCH_APP.read_text()
    body = source[source.index("def main(") :]

    assert "ceiling_usd" in body.split(")")[0], "main takes no ceiling"
    reserve_at = body.index("ledger.reserve(")
    for remote_call in ("_engine_for(base_model)(", ".remote("):
        assert body.index(remote_call) > reserve_at, f"{remote_call} allocates before reserving"

    mode_at = body.index("if mode not in MODES")
    assert mode_at < reserve_at, "an unknown mode reaches allocation"
    assert "ceiling_usd <= 0" in body, "a zero/absent ceiling is not rejected"


def test_the_canary_refuses_a_sweep_when_warmup_generation_failed() -> None:
    """`run_request` turns exceptions, timeouts and malformed streams into ok=False records rather
    than raising, so `warmup.remote` returns normally against a wholly broken path."""
    source = BENCH_APP.read_text()
    tree = ast.parse(source)
    nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
    }
    # The check lives in one shared predicate so the canary gate and the cold-container path cannot
    # drift; assert the call rather than the inlined body, and assert the predicate does the reading.
    called = {
        child.func.id
        for child in ast.walk(nodes["_run_canary"])
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
    }
    assert "_require_healthy_warmup" in called, "the canary no longer checks its warmup records"

    predicate = ast.get_source_segment(source, nodes["_require_healthy_warmup"]) or ""
    assert '"warmups"' in predicate, "the gate reads a key the warmup payload does not emit"
    assert 'get("ok")' in predicate
    assert "raise RuntimeError" in predicate

    warmup = source[source.index("async def _run_warmup(") :]
    emitted = warmup[: warmup.index("\nasync def ")]
    assert '"warmups"' in emitted, "warmup payload key drifted from the gate that reads it"


def test_results_carry_the_engine_shape_that_produced_them() -> None:
    """Without the resolved catalog row, a capacity number cannot be tied to the checkpoint,
    context limit or sequence cap behind it, and catalog drift silently reinterprets old results."""
    source = BENCH_APP.read_text()
    assert source.count("bench_catalog_summary") >= 2, "provenance helper is still test-only"

    bucket_payload = source[source.index("async def _run_bucket(") :]
    bucket_payload = bucket_payload[: bucket_payload.index("\ndef _write_artifact")]
    assert '"engine_catalog"' in bucket_payload, "per-bucket payload omits the engine shape"

    main_body = source[source.index("def main(") :]
    assert main_body.count('"engine_catalog"') >= 2, "canary and summary must both carry it"


def test_pynvml_is_actually_guaranteed_by_the_images_declared_dependencies() -> None:
    """The probe imports `pynvml` without flash declaring a bound, so prove the chain that supplies it.

    `test_image_extras_cover_every_directly_imported_third_party_package` allowlists `pynvml` as
    transitive. An allowlist entry is a weakened guard: if a vllm bump ever drops flashinfer-python,
    or marks its nvidia-ml-py dependency conditional, the import vanishes from the image and
    `probe_gpu` degrades to `{"available": False}` -- turning the GPU identity gate into a silent
    no-op on a paid card. This test fails the moment that chain stops holding.
    """
    lock = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
    blocks = {}
    for block in lock.split("[[package]]"):
        name = re.search(r'^name = "([^"]+)"', block, re.M)
        version = re.search(r'^version = "([^"]+)"', block, re.M)
        if name and version:
            blocks[(name.group(1), version.group(1))] = block

    pinned = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    vllm_spec = next(s for s in pinned["serve-runtime"] if s.startswith("vllm"))
    vllm_version = vllm_spec.split("==")[1].strip()

    vllm = blocks[("vllm", vllm_version)]
    deps = vllm.split("dependencies = [")[1].split("\n]")[0]
    flashinfer_dep = next(
        (ln.strip() for ln in deps.splitlines() if '"flashinfer-python"' in ln), None
    )
    assert flashinfer_dep is not None, (
        f"vllm {vllm_version} no longer depends on flashinfer-python; pynvml is no longer "
        "guaranteed and probe_gpu would degrade to unavailable"
    )
    # A `marker = ...` qualifier would make the dependency conditional; version/source do not.
    assert "marker" not in flashinfer_dep, (
        f"vllm's flashinfer-python dependency became conditional: {flashinfer_dep}"
    )

    flashinfer = [b for (n, _), b in blocks.items() if n == "flashinfer-python"]
    assert flashinfer, "flashinfer-python absent from uv.lock"
    for block in flashinfer:
        line = next(
            (ln.strip() for ln in block.splitlines() if "nvidia-ml-py" in ln),
            None,
        )
        assert line is not None, "flashinfer-python no longer depends on nvidia-ml-py"
        # A marker would make the dependency conditional, so the image could resolve without it.
        assert "marker" not in line, f"nvidia-ml-py became conditional: {line}"


def test_the_gpu_identity_gate_refuses_to_pass_when_the_probe_cannot_read_a_device() -> None:
    """An unreadable device must fail the canary, never be waved through as 'close enough'.

    Guards the pairing between `probe_gpu`'s failure shape and `gpu_matches`: the probe returns
    `{"available": False, "reason": ...}` with NO `gpu` key on every error path, and the caller
    raises on a False result. If `gpu_matches` ever defaulted a missing name to a pass, an
    unidentified card would be billed and its numbers attributed to a tier nobody confirmed.
    """
    for probe in ({}, {"available": False, "reason": "pynvml unavailable: No module"}, {"gpu": {}}):
        assert gpu_matches(probe, "H200") is False, f"unreadable probe passed the gate: {probe}"
    assert gpu_matches({"gpu": {"name": "NVIDIA H200 141GB HBM3e"}}, "H200") is True
    # Substring match would make this true; token match must reject it.
    assert gpu_matches({"gpu": {"name": "NVIDIA L40S"}}, "L4") is False


# ── round-2: the measured window, its bound, and what is counted into it ──────────────────────


def test_the_cell_wait_is_bounded_so_max_seconds_can_actually_bind() -> None:
    """An unbounded `asyncio.wait` returns only when a request COMPLETES.

    A cell whose requests all stall -- the exact shape of an overloaded engine, which is the
    interesting end of the curve -- would then sit until every request hit its individual 900s
    timeout, ignoring the bucket's `max_seconds` entirely. At three buckets times six concurrency
    points that is the difference between a bounded sweep and one that outruns its budget.
    """
    source = inspect.getsource(run_cell)
    loop = source[source.index("in_flight = {_spawn()") : source.index("finally:")]
    wait_call = loop[loop.index("await asyncio.wait(") :]
    assert "timeout=" in wait_call[: wait_call.index(")")], (
        "the steady-state wait is unbounded, so max_seconds cannot bind"
    )
    assert "remaining" in loop, "the wait bound is not the cell's remaining time"
    assert "max_seconds - (" in loop, "the wait bound must derive from max_seconds, not a constant"


class _SlowEngine:
    """Fake engine whose requests take `delay` seconds and stream one valid, uncached response."""

    def __init__(self, delay: float, prompt_tokens: int) -> None:
        self.delay = delay
        self.prompt_tokens = prompt_tokens

    async def _stream_generate(
        self, _payload: Any, _forwarded: Any, _lora: Any, _generation_id: Any
    ) -> Any:
        await asyncio.sleep(self.delay)
        usage = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": 4,
            "cached_tokens": 0,
            "cached_tokens_reported": True,
        }
        yield {"type": "delta", "text": "a", **usage}
        yield {"type": "choice_finished", "finish_reason": "stop", **usage}
        yield {"type": "final", **usage}


async def _run_draining_cell() -> Any:
    """One cell whose `max_seconds` expires while every request is still in flight.

    The window must therefore close at ~`max_seconds` and the drain must run AFTER it, so
    `wall_seconds + drain_seconds` is the full wall time and `wall_seconds` alone is the window.
    """
    bucket = replace(
        BUCKETS_BY_NAME["short_interactive"],
        min_seconds=0.05,
        min_requests=1,
        max_seconds=0.1,
    )
    pool, _ = _fake_pool(concurrency=2)
    engine = _SlowEngine(delay=0.45, prompt_tokens=bucket.target_input_tokens)
    with mock.patch("flash.serving.bench.driver._build_prompt_pool", return_value=pool):
        return await run_cell(engine, object(), "m", bucket, concurrency=2, block=0)


def test_throughput_excludes_the_drain_tail_but_still_counts_its_records() -> None:
    """The drain runs at FALLING concurrency: the last request finishes alone on an idle engine.

    Dividing steady-state work by steady-state-plus-idle time understates every rate, and the bias
    grows with concurrency -- which would bend the curve down at the high end and manufacture a knee
    the engine does not have. The records still belong in the denominator of the ERROR rate.

    Driven behaviorally rather than by reading the source: a source-text assertion still passes
    when `window_seconds` is reassigned to the full wall clock further down.
    """
    result, records = asyncio.run(_run_draining_cell())

    # Every request outlives the window, so the entire cell is drain.
    assert result.wall_seconds < 0.4, (
        f"the measured window ({result.wall_seconds:.3f}s) swallowed the drain tail"
    )
    assert result.drain_seconds > 0.2, "the drain was not accounted for separately"
    # Drained records still count as attempts -- dropping them would flatter the error rate.
    assert len(records) == 2
    assert result.attempted == 2


def test_rates_divide_by_the_window_while_the_error_rate_divides_by_every_attempt() -> None:
    """Arithmetic, not structure: a drained failure must lower nothing but the success rate."""
    records = [
        RequestRecord(
            uid=f"r{i}",
            base_model="m",
            bucket="b",
            concurrency=4,
            block=0,
            started_at=0.0,
            finished_at=1.0,
            first_token_at=0.1,
            prompt_tokens=100,
            completion_tokens=50,
            cached_tokens=0,
            cached_tokens_reported=True,
            finish_reason="stop",
            ok=True,
        )
        for i in range(8)
    ]
    records.append(
        RequestRecord(
            uid="drained",
            base_model="m",
            bucket="b",
            concurrency=4,
            block=0,
            started_at=9.0,
            error=ERROR_TIMEOUT,
        )
    )
    cell = reduce_cell(
        records,
        base_model="m",
        bucket="b",
        concurrency=4,
        block=0,
        wall_seconds=10.0,
        drain_seconds=40.0,
    )
    # 8 successes over the 10s WINDOW, not over the 50s that includes the drain.
    assert cell.successful_rps == pytest.approx(0.8)
    assert cell.output_tokens_per_second == pytest.approx(40.0)
    assert cell.drain_seconds == pytest.approx(40.0)
    # The drained request is an attempt and a failure.
    assert cell.attempted == 9
    assert cell.failed == 1
    assert cell.error_rate == pytest.approx(1 / 9)


def test_a_prompt_the_engine_sized_differently_is_rejected_not_averaged_in() -> None:
    """A bucket claims a specific input size. A request the engine sized differently belongs to a
    different bucket, so counting it would silently widen the shape the numbers describe."""
    record = RequestRecord(
        uid="u",
        base_model="m",
        bucket="near_32k",
        concurrency=1,
        block=0,
        started_at=0.0,
        expected_prompt_tokens=31744,
    )
    outcome = _StreamOutcome()
    outcome.saw_final = True
    outcome.finish_reason = "stop"
    outcome.prompt_tokens = 512
    outcome.completion_tokens = 64
    outcome.cached_tokens = 0
    outcome.cached_tokens_reported = True
    outcome.first_token_at = 0.1
    _validate(outcome, record)
    assert record.ok is False
    assert record.error == ERROR_PROMPT_LENGTH

    # Inside the fitter's own tolerance it passes: the check must not be stricter than the fit.
    ok_record = replace(record, error=None, error_detail=None, ok=False)
    outcome.prompt_tokens = 31744 + PROMPT_TOKEN_TOLERANCE
    _validate(outcome, ok_record)
    assert ok_record.ok is True

    # And the driver actually supplies the expectation rather than discarding it.
    source = inspect.getsource(run_cell)
    assert "expected_prompt_tokens=exact" in source, "the fitted length is discarded at spawn"


def test_the_reservation_prices_a_sweep_from_its_buckets_own_bounds() -> None:
    """A flat estimate under-reserves a wide sweep, so the ceiling would be checked against a number
    unrelated to what the run can actually spend. The reservation must be an UPPER bound."""
    source = BENCH_APP.read_text(encoding="utf-8")
    assert "_sweep_gpu_seconds_estimate" in source
    assert "ESTIMATED_SWEEP_GPU_SECONDS" not in source, "the flat sweep estimate is still in use"
    fn = source[source.index("def _sweep_gpu_seconds_estimate") :]
    fn = fn[: fn.index("\n\n\n")]
    assert "max_seconds" in fn, "cells must be priced at their bucket's worst case"
    assert "concurrency_grid" in fn, "the grid width must come from the engine, not a constant"


def test_an_unknown_bucket_is_rejected_before_a_gpu_is_rented() -> None:
    """`--bucket` was only indexed on the remote side, so a typo paid for a full cold boot and
    warmup before failing with no measurement. Everything checkable without a card is checked here.
    """
    source = BENCH_APP.read_text(encoding="utf-8")
    main = source[source.index("def main(") :]
    guard = main.index("unknown --bucket")
    # The guard must precede every remote call, i.e. every place a GPU can be allocated.
    for remote in (".remote(", "_run_canary(", "_run_bucket("):
        first = main.find(remote)
        if first != -1:
            assert guard < first, f"the bucket guard runs after {remote}, which allocates a GPU"


def test_the_documented_paid_commands_pass_the_required_ceiling() -> None:
    """`--ceiling-usd` is required and has no default, so a documented command missing it fails at
    the point a reader has already decided to spend money."""
    doc = (REPO_ROOT / "docs" / "serving-capacity-envelope.md").read_text(encoding="utf-8")
    commands = [
        line for line in doc.splitlines() if line.startswith("modal run scripts/bench_hosted")
    ]
    assert commands, "the doc no longer shows a paid command"
    for command in commands:
        assert "--ceiling-usd" in command, f"documented paid command omits the ceiling: {command}"


def test_the_summary_artifact_names_which_bucket_each_curve_came_from() -> None:
    """A list of curves without their bucket names is unreadable once more than one bucket runs."""
    # Parsed, not sliced: `{"bucket": payload["bucket"], ...}` contains both `]` and `],` inside
    # itself, so every string-slice boundary lands in the middle of a subscript.
    #
    # Anchored on the "buckets" key's own VALUE. A bare walk for any dict carrying both keys also
    # matches the per-bucket payload built earlier in that file, so it stayed green even when the
    # summary itself dropped the label.
    tree = ast.parse(BENCH_APP.read_text(encoding="utf-8"))
    values = [
        value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Constant) and key.value == "buckets"
    ]
    assert len(values) == 1, f"expected exactly one 'buckets' summary entry, found {len(values)}"
    element = values[0]
    assert isinstance(element, ast.ListComp), "the summary's buckets entry is not a comprehension"
    assert isinstance(element.elt, ast.Dict), "each summary entry must be a labelled dict"
    keys = {k.value for k in element.elt.keys if isinstance(k, ast.Constant)}
    assert "bucket" in keys, "the summary emits curves with no bucket label"
    assert "curve" in keys, "the summary dropped the curve alongside its label"


# ── Round-3 review findings ──────────────────────────────────────────────────────────────────


def test_drain_completions_are_counted_but_excluded_from_the_rate_numerators() -> None:
    """A request that finished during the drain must not be credited to the window.

    The window closes with `concurrency` requests still in flight; those finish afterwards, at
    falling load. Summing them into the numerator while dividing by the shorter pre-drain window
    reports throughput the engine never sustained, and the overstatement is largest exactly where
    the window is shortest relative to one request.
    """
    in_window = [_record(f"w{i}", tokens=100, latency=5.0) for i in range(2)]
    # Finished at 30s, well past the 10s window: these are drain completions.
    drained = [_record(f"d{i}", tokens=100, latency=30.0) for i in range(4)]

    result = reduce_cell(
        in_window + drained,
        base_model="Qwen/Qwen3.5-9B",
        bucket="short_interactive",
        concurrency=2,
        block=0,
        wall_seconds=10.0,
        window_seconds=10.0,
    )

    # Every record is still attempted, and every success is still counted.
    assert result.attempted == 6, "drained records were dropped from the attempt denominator"
    assert result.succeeded == 6, "drained successes were dropped from the success count"
    # But only the two in-window completions drive the rates.
    assert result.succeeded_in_window == 2
    assert result.successful_rps == pytest.approx(0.2), "drain completions inflated RPS"
    assert result.output_tokens_per_second == pytest.approx(20.0), (
        "drain completions inflated token throughput"
    )
    assert result.completion_tokens_total == 200, "token total counted post-window work"


def test_drain_latency_is_excluded_because_it_ran_at_falling_concurrency() -> None:
    """A drain completion's latency describes an idler engine than the cell was measuring."""
    records = [
        _record("in", latency=2.0, ttft=0.5),
        _record("drained", latency=40.0, ttft=9.0),
    ]
    result = reduce_cell(
        records,
        base_model="Qwen/Qwen3.5-9B",
        bucket="short_interactive",
        concurrency=1,
        block=0,
        wall_seconds=5.0,
        window_seconds=5.0,
    )
    assert result.latency_seconds["p50"] == pytest.approx(2.0), "drain latency entered the sample"
    assert result.ttft_seconds["p50"] == pytest.approx(0.5), "drain TTFT entered the sample"


def test_saturation_reports_the_first_cell_when_it_is_already_degraded() -> None:
    """`pairwise` only ever tests the SECOND element, so cell 0's degradation was invisible."""
    degraded_first = _cell(concurrency=1, tps=10.0, p95=1.0, feasible=False)
    healthy_later = _cell(concurrency=2, tps=100.0, p95=1.0)
    assert degraded_first.degraded, "fixture is wrong: the first cell must be degraded"

    summary = summarize_curve([degraded_first, healthy_later])
    assert summary["saturation_concurrency"] == 1, (
        "saturation skipped the first cell and reported a later concurrency or None"
    )


def test_ttft_comes_from_the_ready_event_not_the_first_visible_text() -> None:
    """A first token decoding to "" must not push TTFT out to a later token.

    `ready` fires immediately after vLLM's first output, so it is the end of prefill even when the
    detokenizer is still buffering bytes. Timing from first non-empty text silently overstates
    prefill latency, always in the same direction.
    """
    outcome = _StreamOutcome()
    _absorb_event({"type": "ready"}, outcome, 1.0)
    _absorb_event({"type": "delta", "text": ""}, outcome, 2.0)
    _absorb_event({"type": "delta", "text": "hi"}, outcome, 5.0)
    assert outcome.first_token_at == pytest.approx(1.0), (
        "TTFT was taken from the first visible text rather than the ready event"
    )


def test_the_sweep_estimate_reserves_the_canary_and_every_drain() -> None:
    """The ceiling is only real if the reservation covers every phase that bills.

    The canary always runs, and a drain follows EVERY cell; at 900s each those tails can exceed the
    measured time they follow. An estimate covering only the windows can accept a run that then
    bills past its own ceiling.
    """
    from flash.serving.bench.driver import REQUEST_TIMEOUT_SECONDS
    from flash.serving.bench.workload import BUCKETS_BY_NAME, concurrency_grid

    # The module imports modal at top level, so the estimator is extracted and executed on its own,
    # the same way this file already reads `_bench_class_name`.
    tree = ast.parse(BENCH_APP.read_text(encoding="utf-8"))
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_sweep_gpu_seconds_estimate"
    )
    constants = {
        node.targets[0].id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }
    # The estimator now resolves the catalog and grid from module scope rather than local imports,
    # so the lifted function needs them supplied.
    namespace: dict = dict(constants)
    namespace.update(
        bench_engine_overrides_for=bench_engine_overrides_for,
        concurrency_grid=concurrency_grid,
        REQUEST_TIMEOUT_SECONDS=REQUEST_TIMEOUT_SECONDS,
    )
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<bench>", "exec"), namespace)
    estimate_for = namespace["_sweep_gpu_seconds_estimate"]

    bucket = BUCKETS_BY_NAME["short_interactive"]
    estimate = estimate_for("Qwen/Qwen3.5-9B", [bucket])

    points = len(list(concurrency_grid(8)))
    windows = float(bucket.max_seconds) * points
    drains = REQUEST_TIMEOUT_SECONDS * points
    canary = REQUEST_TIMEOUT_SECONDS * constants["CANARY_WARMUP_REQUESTS"]
    startup = constants["STARTUP_TIMEOUT_SECONDS"]
    # `max_containers=1` caps SIMULTANEOUS replicas; it does not pin successive `.remote()` calls to
    # one container. Every bucket call can therefore land on a cold replacement and pay another boot
    # plus its warmups, which bills whether or not the reservation admits it.
    replacements = (startup + canary) * 1

    assert estimate >= windows + drains + canary, (
        "the estimate omits the canary or the per-cell drain tails"
    )
    # The boot is reserved at the ceiling Modal lets a stuck boot reach, not a typical observed one.
    assert estimate == pytest.approx(startup + canary + windows + drains + replacements)


def test_a_bucket_landing_on_a_cold_container_warms_itself_before_measuring() -> None:
    """`max_containers=1` caps simultaneous replicas; it does not pin successive calls to one.

    So a bucket can land on a container the canary never gated, and would otherwise pay compile and
    lazy-workspace costs inside its measured window.
    """
    source = BENCH_APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
    }
    # An `in source` substring check would also match the COMMENT that explains the call, so
    # deleting the call itself would leave this test green. Match the call node instead.
    called = {
        child.func.id
        for child in ast.walk(nodes["_run_bucket"])
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
    }
    assert "_ensure_warm" in called, "a replacement container measures cold"

    warm_called = {
        child.func.id
        for child in ast.walk(nodes["_ensure_warm"])
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
    }
    assert "_run_warmup" in warm_called, "the cold path does not actually warm up"
    attributes = {
        child.attr for child in ast.walk(nodes["_ensure_warm"]) if isinstance(child, ast.Attribute)
    }
    assert "_bench_warmed" in attributes, "warmth is not tracked per container"


# ── Round-4: lane bounds, warmup health, and window-scoped degradation ────────────────────────


def _bench_namespace(*names: str, **injected: Any) -> dict[str, Any]:
    """Exec named top-level defs/assignments out of the Modal script, without importing it.

    The script has a module-level ``import modal`` and builds Modal classes at import time, so it
    cannot be imported in a unit test. Lifting the exact nodes keeps these tests against the real
    source rather than a transcription of it, so editing the script breaks the test.
    """
    tree = ast.parse(BENCH_APP.read_text(encoding="utf-8"))
    wanted = set(names)

    def _defines(node: ast.stmt) -> set[str]:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return {node.name}
        if isinstance(node, ast.Assign):
            return {t.id for t in node.targets if isinstance(t, ast.Name)}
        return set()

    body = [node for node in tree.body if _defines(node) & wanted]
    missing = wanted - {name for node in body for name in _defines(node)}
    assert not missing, f"the bench script no longer defines {sorted(missing)}"
    namespace: dict[str, Any] = {"Any": Any, "uuid": uuid, **injected}
    exec(compile(ast.Module(body=body, type_ignores=[]), "<bench>", "exec"), namespace)
    return namespace


def test_bench_container_timeout_covers_the_whole_grid_it_runs() -> None:
    """A bucket runs its ENTIRE concurrency grid on one boot, so the ceiling must cover the sum.

    The failure this guards is expensive and silent-until-the-end: a timeout below the grid's own
    preregistered bounds kills ``run_bucket`` after the run has already paid for every cell, and the
    artifact is never persisted -- full cost, zero evidence. The ceiling is DERIVED from the bucket
    and grid bounds here for the same reason it is derived in the script: widening a bucket must
    raise it automatically instead of quietly reintroducing the gap.
    """
    namespace = _bench_namespace(
        "_worst_case_bucket_seconds",
        "TIMEOUT_HEADROOM_SECONDS",
        "TIMEOUT_SECONDS",
        "CANARY_WARMUP_REQUESTS",
        BENCH_MODELS=BENCH_MODELS,
        bench_engine_overrides_for=bench_engine_overrides_for,
        concurrency_grid=concurrency_grid,
        BUCKETS=BUCKETS,
        REQUEST_TIMEOUT_SECONDS=REQUEST_TIMEOUT_SECONDS,
    )
    timeout = namespace["TIMEOUT_SECONDS"]

    # Recompute the worst case independently of the script's own helper.
    points = max(
        len(list(concurrency_grid(int(bench_engine_overrides_for(model).get("max_num_seqs", 8)))))
        for model in BENCH_MODELS
    )
    widest = max(bucket.max_seconds for bucket in BUCKETS)
    # A bucket that lands on a cold replacement warms it SEQUENTIALLY before the first cell. Those
    # warmups run inside the method, so a timeout sized to the grid alone kills the call mid-warmup
    # -- and because the timeout fires before `run_bucket` returns, the artifact is never written.
    warmups = REQUEST_TIMEOUT_SECONDS * namespace["CANARY_WARMUP_REQUESTS"]
    worst = points * (widest + REQUEST_TIMEOUT_SECONDS) + warmups
    assert namespace["_worst_case_bucket_seconds"]() == worst
    assert timeout >= worst, (
        f"container timeout {timeout}s is below the {worst}s its own grid is allowed to take; "
        "the bucket would be killed after paying for every cell and persist no artifact"
    )
    assert timeout > worst, "no headroom: the timeout would fire exactly at the bound"


def test_warmup_prompts_are_unique_across_invocations() -> None:
    """Fixed warmup UIDs collide with a RETAINED prefix cache and refuse a healthy engine.

    Prompts are derived from the UID, so a fixed ``warmup-{i}`` reissues the same five prompts every
    invocation. Inside the scaledown window the container survives, the second canary hits vLLM's
    retained prefix cache, and the driver scores it ``ERROR_CACHE_CONTAMINATED`` -- correctly, since
    a cached request measures nothing. The engine was fine; the harness refused it.
    """
    issued: list[str] = []

    class _Record:
        def to_json(self) -> dict[str, Any]:
            return {"ok": True}

    async def _fake_run_request(
        engine: Any, base_model: str, messages: Any, max_tokens: int, uid: str, **kw: Any
    ) -> Any:
        issued.append(uid)
        return _Record()

    def _fake_fit(tokenizer: Any, uid: str, target: int) -> tuple[list[dict[str, str]], int]:
        return ([{"role": "user", "content": uid}], target)

    namespace = _bench_namespace("_run_warmup")
    engine = mock.Mock(tokenizer=object(), base_model="Qwen/Qwen3.5-9B")
    with (
        mock.patch("flash.serving.bench.driver.run_request", _fake_run_request),
        mock.patch("flash.serving.bench.workload.fit_prompt_to_tokens", _fake_fit),
    ):
        asyncio.run(namespace["_run_warmup"](engine, 5))
        first = list(issued)
        issued.clear()
        asyncio.run(namespace["_run_warmup"](engine, 5))

    assert len(set(first)) == 5, "warmup UIDs collide WITHIN one invocation"
    assert not set(first) & set(issued), (
        "warmup UIDs repeat ACROSS invocations, so a surviving container serves the second canary "
        "from its retained prefix cache and the run is refused as cache-contaminated"
    )


def test_a_cold_replacement_container_checks_its_own_warmup_health() -> None:
    """``run_request`` returns ``ok=False`` records instead of raising, so a warmup 'succeeds'.

    The canary gate checks its warmup records. A container Modal replaces mid-sweep warms itself
    through a different path, and trusting it because the canary passed on a DIFFERENT container
    would start paid measurement against a generation path already known to be broken.
    """
    calls: list[int] = []

    async def _fake_run_warmup(engine: Any, requests: int) -> dict[str, Any]:
        calls.append(requests)
        return {"warmups": [{"ok": True}, {"ok": False, "error": "cache_contaminated"}]}

    namespace = _bench_namespace(
        "_ensure_warm",
        "_require_healthy_warmup",
        "CANARY_WARMUP_REQUESTS",
        _run_warmup=_fake_run_warmup,
    )
    namespace["_run_warmup"] = _fake_run_warmup
    engine = mock.Mock()
    del engine._bench_warmed  # a freshly booted container has never been warmed

    with pytest.raises(RuntimeError, match="replacement-container warmup failed 1/2"):
        asyncio.run(namespace["_ensure_warm"](engine))
    assert calls, "the cold path never ran a warmup at all"


def test_healthy_warmup_predicate_rejects_an_empty_or_failed_result() -> None:
    """One predicate, so the canary gate and the cold-container path cannot drift apart."""
    namespace = _bench_namespace("_require_healthy_warmup")
    check = namespace["_require_healthy_warmup"]

    check({"warmups": [{"ok": True}]}, "canary")  # the healthy case must not raise
    with pytest.raises(RuntimeError, match="returned no records"):
        check({"warmups": []}, "canary")
    with pytest.raises(RuntimeError, match="returned no records"):
        check(None, "canary")
    with pytest.raises(RuntimeError, match="canary warmup failed 1/1"):
        check({"warmups": [{"ok": False, "error": "timeout"}]}, "canary")


def test_both_lane_estimates_reserve_a_boot_at_the_ceiling_modal_allows() -> None:
    """A reservation must price the boot Modal PERMITS, not a boot typically observed.

    ``startup_timeout=STARTUP_TIMEOUT_SECONDS`` is how long a stuck boot bills for, so reserving a
    typical 1200s boot accepts a ceiling the lane's own configuration lets it exceed -- which is the
    single thing ``BudgetLedger`` exists to prevent. Both lanes are checked together because the
    canary previously used a flat constant that the sweep-side fix did not reach.
    """
    namespace = _bench_namespace(
        "_canary_gpu_seconds_estimate",
        "_sweep_gpu_seconds_estimate",
        "CANARY_WARMUP_REQUESTS",
        "STARTUP_TIMEOUT_SECONDS",
        bench_engine_overrides_for=bench_engine_overrides_for,
        concurrency_grid=concurrency_grid,
        REQUEST_TIMEOUT_SECONDS=REQUEST_TIMEOUT_SECONDS,
    )
    startup = namespace["STARTUP_TIMEOUT_SECONDS"]
    warmups = namespace["CANARY_WARMUP_REQUESTS"]

    canary = namespace["_canary_gpu_seconds_estimate"]()
    assert canary == startup + REQUEST_TIMEOUT_SECONDS * warmups
    assert canary >= startup, "the canary reserves less than a stuck boot alone would bill"

    bucket = BUCKETS_BY_NAME["short_interactive"]
    sweep = namespace["_sweep_gpu_seconds_estimate"]("Qwen/Qwen3.5-9B", [bucket])
    points = len(list(concurrency_grid(8)))
    expected = (
        startup
        + REQUEST_TIMEOUT_SECONDS * warmups
        + bucket.max_seconds * points
        + REQUEST_TIMEOUT_SECONDS * points
        # one replacement boot + warmup per bucket call, since `max_containers=1` does not pin
        # successive `.remote()` calls to the container the previous bucket booted
        + (startup + REQUEST_TIMEOUT_SECONDS * warmups) * 1
    )
    assert sweep == expected
    assert sweep - canary >= bucket.max_seconds * points, "the sweep does not price its own cells"

    source = BENCH_APP.read_text(encoding="utf-8")
    for stale in ("ESTIMATED_CANARY_GPU_SECONDS", "ESTIMATED_BOOT_GPU_SECONDS"):
        assert stale not in source, f"{stale} is a flat constant that cannot track the bounds"


def test_documented_ceilings_exceed_what_each_lane_reserves() -> None:
    """A documented ceiling below the lane's own reservation is a command that cannot run.

    ``--ceiling-usd`` is reserved against BEFORE allocation and raises ``BudgetExceeded``, so a
    copy-pasteable command whose ceiling is under its reservation fails every time it is run. The
    doc numbers are checked against the estimators rather than pinned, so raising a reservation
    fails here instead of shipping a broken runbook.
    """
    namespace = _bench_namespace(
        "_canary_gpu_seconds_estimate",
        "_sweep_gpu_seconds_estimate",
        "CANARY_WARMUP_REQUESTS",
        "STARTUP_TIMEOUT_SECONDS",
        bench_engine_overrides_for=bench_engine_overrides_for,
        concurrency_grid=concurrency_grid,
        REQUEST_TIMEOUT_SECONDS=REQUEST_TIMEOUT_SECONDS,
    )
    doc = (REPO_ROOT / "docs" / "serving-capacity-envelope.md").read_text(encoding="utf-8")
    model = "Qwen/Qwen3.5-9B"
    gpu = bench_gpu_for(model)
    documented = {
        match.group(1): float(match.group(2))
        for match in re.finditer(r"--mode (canary|sweep) .*?--ceiling-usd (\d+(?:\.\d+)?)", doc)
    }
    assert set(documented) == {"canary", "sweep"}, "the runbook lost one of its two commands"

    reserved = {
        "canary": namespace["_canary_gpu_seconds_estimate"](),
        "sweep": namespace["_sweep_gpu_seconds_estimate"](
            model, [BUCKETS_BY_NAME["short_interactive"]]
        ),
    }
    for mode, seconds in reserved.items():
        cost = usd_for_gpu_seconds(seconds, gpu)
        # A ceiling merely ABOVE the cost is still not runnable: `reserve` refuses at the
        # submission stop, so the documented value has to clear cost / SUBMISSION_STOP_FRACTION.
        required = cost / SUBMISSION_STOP_FRACTION
        assert documented[mode] >= required, (
            f"documented --ceiling-usd {documented[mode]} for the {mode} lane is below the "
            f"${required:.2f} its ${cost:.2f} reservation needs to clear the submission stop; "
            "the command raises BudgetExceeded before allocating"
        )


def test_a_cell_with_no_in_window_successes_is_degraded() -> None:
    """Drain completions keep ``succeeded`` positive on a cell whose steady-state rate is ZERO.

    A cell so slow that nothing finishes inside its window still accumulates successes during the
    drain. Testing ``succeeded`` there reads a cell delivering no throughput as usable, and because
    its in-window latency sample is empty the saturation scan can skip past it and publish no
    saturation at all -- the load ceiling would be reported as higher than the last concurrency the
    engine actually served.
    """
    window, drain = 60.0, 900.0
    records = []
    for index in range(8):
        record = _record(f"r{index}", latency=window + 100.0 + index)
        # Every request finishes AFTER the window closed: real drain completions, not lost records.
        record.first_token_at = window + 50.0
        records.append(record)

    cell = reduce_cell(
        records,
        base_model="Qwen/Qwen3.5-9B",
        bucket="near_32k",
        concurrency=8,
        block=0,
        wall_seconds=window + drain,
        drain_seconds=drain,
        window_seconds=window,
    )

    assert cell.succeeded == 8, "the drained requests must stay in the attempt accounting"
    assert cell.succeeded_in_window == 0
    assert cell.successful_rps == 0.0
    assert cell.output_tokens_per_second == 0.0
    assert not cell.feasible, "a cell with no in-window success cannot be called feasible"
    assert cell.degraded, "zero steady-state throughput read as a usable cell"

    # And the curve must saturate AT it rather than scanning past an empty latency sample.
    healthy = _cell(1, 100.0, 2.0)
    summary = summarize_curve([healthy, replace(cell, concurrency=2)])
    assert summary["saturation_concurrency"] == 2


def test_catalog_summary_records_the_immutable_revisions_it_measures() -> None:
    """A repo NAME is a moving target; the envelope has to name the commits it measured.

    The 27B engine pins weights to a commit in the ``-FP8`` repo and its tokenizer/processor to a
    DIFFERENT commit in the base repo. A summary carrying only repo names lets two runs months apart
    report the same "model" while serving different weights, and nothing in the artifact can tell
    them apart -- which silently voids every comparison the envelope exists to support.
    """
    summary = {row["base_model"]: row for row in bench_catalog_summary()}
    assert set(summary) == set(BENCH_MODELS)
    for model, row in summary.items():
        assert row["tokenizer_model"] == tokenizer_model_for(model)
        assert row["immutable_revisions"] == immutable_serving_revisions(model)
        assert row["serve_model_id"], "the served model id is what the request actually names"

    # The 27B is the case that makes this load-bearing: its tokenizer does not come from the repo
    # its weights come from, so one repo name cannot identify the pair.
    weights = summary["Qwen/Qwen3.8-27B"]["serve_model_id"]
    tokenizer = summary["Qwen/Qwen3.8-27B"]["tokenizer_model"]
    assert weights != tokenizer, (
        "the 27B tokenizer is expected to come from the base repo, not the FP8 weights repo; "
        "if this converged, the summary no longer proves the two are pinned separately"
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_ceiling_is_refused(bad: float) -> None:
    """A non-finite ceiling passes every comparison the ledger makes, so it bounds nothing.

    ``nan <= 0`` is False and ``inf <= 0`` is False, so the positivity check admits both. Then
    ``projected > nan`` is False for every projection and every finite projection is below ``inf``,
    so ``reserve`` would approve unbounded spend while reporting a configured ceiling. Finiteness is
    therefore checked BEFORE positivity.
    """
    with pytest.raises(ValueError, match="finite"):
        BudgetLedger(ceiling_usd=bad)
    with pytest.raises(ValueError, match="finite"):
        BudgetLedger(ceiling_usd=10.0, submission_stop_usd=bad)


def test_request_uid_separates_invocations_without_moving_the_corpus() -> None:
    """A retry must not re-send byte-identical prompts, and must not change the workload either.

    Without the nonce the id depends only on grid coordinates, so a rerun at the same block reissues
    every prompt verbatim; inside Modal's 120s scaledown the container and its prefix cache survive
    and the driver scores the hits ERROR_CACHE_CONTAMINATED -- discarding a paid rerun whose engine
    was healthy. The nonce must reach the UID only: ``corpus_seed`` keys the filler BODY, so moving
    it would vary the prompt text alongside the one variable a curve exists to isolate.
    """
    first = request_uid("short_interactive", 4, 0, 7, invocation="aaaa1111")
    second = request_uid("short_interactive", 4, 0, 7, invocation="bbbb2222")
    assert first != second, "two invocations reissue the same prompt id"
    assert request_uid("short_interactive", 4, 0, 7) != first, "the nonce is not in the id"

    # Same coordinates, different invocations -> same corpus. The workload stays reproducible.
    assert corpus_seed("short_interactive", 0, 7) == corpus_seed("short_interactive", 0, 7)
    for uid in (first, second):
        assert uid.startswith("short_interactive-c4-b0-i7"), (
            "the nonce must be a SUFFIX; prefixing it would break the coordinate parsing"
        )


def test_cell_result_json_carries_the_degraded_verdict() -> None:
    """``asdict`` serializes FIELDS only, so a computed verdict is absent from every artifact.

    The report contract distinguishes ``degraded`` from ``feasible``, and a consumer that cannot read
    it has to reimplement the property against the exact source revision that produced the file just
    to learn which cell the harness classified as failed, or why a saturation scan stopped where it
    did.
    """
    degraded = _cell(concurrency=8, tps=10.0, p95=4.0, feasible=False)
    healthy = _cell(concurrency=1, tps=40.0, p95=1.0)
    assert degraded.to_json()["degraded"] is True
    assert healthy.to_json()["degraded"] is False
    # The serialized value is the property, not an independently maintained field.
    assert degraded.to_json()["degraded"] == degraded.degraded
    assert "degraded" not in {f.name for f in dataclasses.fields(CellResult)}, (
        "if `degraded` became a real field this test no longer proves the property is serialized"
    )
