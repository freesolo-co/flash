"""Allocation-free tests for the hosted per-model capacity benchmark.

Two jobs. First, prove the benchmark cannot touch production: not the app name, not the catalog, not
the router, not the billing path. Second, prove the metric arithmetic is right against hand-computed
values, so a number in the published report is trustworthy without re-deriving it.

Everything here runs with no GPU and no network.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path

import pytest

from flash.serving.bench.budget import (
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
)
from flash.serving.bench.driver import (
    REQUEST_TIMEOUT_SECONDS,
    _StreamOutcome,
    _validate,
    base_model_record,
    run_cell,
)
from flash.serving.bench.metrics import (
    ERROR_CACHE_CONTAMINATED,
    ERROR_CACHE_UNVERIFIED,
    ERROR_TIMEOUT,
    CellResult,
    RequestRecord,
    percentile,
    reduce_cell,
    summarize_curve,
    wilson_upper_bound,
)
from flash.serving.bench.probe import gpu_matches, probe_gdn_backend
from flash.serving.bench.workload import (
    BUCKETS,
    BUCKETS_BY_NAME,
    ENABLE_THINKING,
    TEMPERATURE,
    build_prompt_text,
    concurrency_grid,
    request_uid,
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
    record.first_answer_at = ttft + 0.1
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
    """A deep cell: 600 attempts resolves the Wilson bound, so ``feasible`` means what it says."""
    return CellResult(
        base_model="Qwen/Qwen3.5-9B",
        bucket="short_interactive",
        concurrency=concurrency,
        block=0,
        wall_seconds=60.0,
        attempted=600,
        succeeded=600 if feasible else 300,
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
    ledger = BudgetLedger(ceiling_usd=20.0)
    ledger.reserve("lane-a", 3600 * 4, "H200")  # $18.16
    with pytest.raises(BudgetExceeded):
        ledger.reserve("lane-b", 3600 * 4, "H200")
    assert ledger.committed_usd <= 20.0


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
    source = inspect.getsource(run_cell)
    drain = source.split("if in_flight:")[-1]
    assert "timeout=REQUEST_TIMEOUT_SECONDS" in drain, "the drain must be bounded, not unbounded"
    assert "still_pending" in drain
    assert "ERROR_TIMEOUT" in drain, "a cancelled request must be RECORDED as a timeout"
    # The record is appended after the cancel, outside the suppress that swallows CancelledError.
    cancel_at = drain.index("task.cancel()")
    append_at = drain.index("records.append", cancel_at)
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
